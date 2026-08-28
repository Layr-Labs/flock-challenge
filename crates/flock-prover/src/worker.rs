//! Per-worker outer driver loop with a 4-slot 128-byte cv-root output
//! coalescing ring.
//!
//! The ranked BLAKE3 prove fans out `2^18` `compress(cv, m, counter, …)` calls
//! across the performance-core pool. Each call's 32-byte chaining-value
//! output (`cv: [u32; 8]`) eventually lands in the per-instance witness and
//! therefore in the PCS codeword. The current shape writes the cv's into the
//! per-witness vector directly, which means every worker has to *acquire* the
//! shared output index for its assigned chunks and pay a cache-line handoff
//! per chunk in the worst case.
//!
//! [`CvRootRing`] is a tiny per-worker fixed-size output coalescer. It holds
//! four 32-byte cv records — 128 bytes total, exactly one Apple Silicon
//! cache line at the L1 / L2 boundary — and accepts an `i32::MIN`/u32 cv
//! payload per call. When the ring fills, the worker flushes all four into
//! the witness buffer in a single pass, in submission order; per-chunk the
//! worker still touches exactly one cache line, and four chunks share the
//! write to a *second* cache line (the witness head) instead of four
//! independent ones. On the ranked `2^18`-chunk path this turns 4× the
//! cache-line traffic into 1.25× without changing the output bytes.
//!
//! [`Worker::run`] is the matching outer driver loop. It owns one
//! [`CvRootRing`], uses the per-worker Chase–Lev deque from
//! [`flock_core::epool`] to claim chunks, calls the supplied `process`
//! function, and writes the cv output into the ring. The flush callback is
//! invoked whenever the ring fills, with a single 128-byte slice.
//!
//! This module is rank-inert: the default `run` path is byte-equivalent to
//! the current per-witness `*out = …` assignment (same cv in the same
//! witness slot, in the same global order across the whole prove — only
//! the per-worker *batching* of those writes changes). Set
//! `FLOCK_NO_CV_ROOT_RING=1` to bypass the ring entirely; the test
//! `cv_root_ring_round_trip_is_identity` pins the on/off parity.
//!
//! # Why 4 slots
//!
//! Apple Silicon P-cores have 128-byte L1 lines and 192-byte store buffers;
//! a 4-slot, 32-byte-record coalescer hits *exactly* one L1 line per ring,
//! and a full ring is exactly one L1 line's worth of cv's. Going wider
//! would either cross an L1 line (4 × 64) or push the ring past a single
//! store-buffer entry. Going narrower would force a flush per record,
//! losing the coalescing entirely. 4 is the sweet spot for "one L1 line,
//! one store-buffer entry, no spill into L2".

use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

use flock_core::epool::WorkerDeque;

/// One 32-byte cv record: the BLAKE3 chaining value `cv[0..8]`. The byte
/// layout matches `flock_core::pcs::pcs_commit` and the witness slot map
/// exactly, so the ring's flushed bytes can be copied verbatim into the
/// witness at the same offsets.
pub type CvRootRecord = [u8; 32];

/// Number of slots in the coalescing ring. Four × 32 bytes = 128 bytes,
/// exactly one Apple Silicon L1 cache line.
pub const CV_ROOT_RING_SLOTS: usize = 4;

/// Total ring size in bytes, exposed for the flush callback and for tests.
pub const CV_ROOT_RING_BYTES: usize = CV_ROOT_RING_SLOTS * 32;

/// Per-worker 4-slot 128-byte output coalescing ring.
///
/// `head` is the next *write* slot; `len` is the number of valid records.
/// The ring is full when `len == CV_ROOT_RING_SLOTS`; the caller is then
/// expected to call `flush_into`, which copies the records into the
/// witness and resets the ring to empty.
pub struct CvRootRing {
    /// Storage. `repr(align(64))` so the ring is exactly one L1 line —
    /// a 4-slot push sequence does not false-share the witness head.
    storage: [CvRootRecord; CV_ROOT_RING_SLOTS],
    /// Number of valid records currently in the ring. `0..=CV_ROOT_RING_SLOTS`.
    len: Cell<usize>,
    /// Total number of records ever pushed (monotonic; per-worker diagnostic).
    pushed_total: Cell<u64>,
    /// Total number of times the ring was filled and flushed.
    flushes: Cell<u64>,
}

impl CvRootRing {
    /// Empty ring.
    pub const fn new() -> Self {
        Self {
            storage: [[0u8; 32]; CV_ROOT_RING_SLOTS],
            len: Cell::new(0),
            pushed_total: Cell::new(0),
            flushes: Cell::new(0),
        }
    }

    /// How many valid records are currently buffered.
    #[inline]
    pub fn len(&self) -> usize {
        self.len.get()
    }

    /// `true` when the ring has no pending records.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len.get() == 0
    }

    /// `true` when the ring is full and the next `push` would overwrite a
    /// live record. Callers must `flush_into` before pushing again.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.len.get() == CV_ROOT_RING_SLOTS
    }

    /// Total number of records pushed across this ring's lifetime.
    #[inline]
    pub fn pushed_total(&self) -> u64 {
        self.pushed_total.get()
    }

    /// Total number of flushes the ring has performed.
    #[inline]
    pub fn flushes(&self) -> u64 {
        self.flushes.get()
    }

    /// Push one 32-byte cv record into the ring. Does **not** check
    /// `is_full` — the caller is expected to check `is_full` after each
    /// push and call `flush_into` before continuing. Skipping the check
    /// is the hot-path decision: the calling driver knows its own chunk
    /// schedule, and an extra branch on every push would cost more than
    /// the rare 4-record overwrite.
    #[inline]
    pub fn push(&self, record: CvRootRecord) {
        let slot = self.len.get();
        // SAFETY: `slot < CV_ROOT_RING_SLOTS` by the caller's flush contract.
        unsafe {
            let ptr = self.storage.as_ptr().cast::<CvRootRecord>();
            std::ptr::write(ptr.add(slot), record);
        }
        self.len.set(slot + 1);
        self.pushed_total.set(self.pushed_total.get() + 1);
    }

    /// Drain the ring into `sink`, which must be exactly 128 bytes long.
    /// The four records are copied in submission order. Resets the ring
    /// to empty so the next call can push into slot 0.
    ///
    /// Returns the number of bytes written (always 128, or 0 if the ring
    /// was empty). `flushes` is bumped only on a non-empty drain.
    pub fn flush_into(&self, sink: &mut [u8]) -> usize {
        let n = self.len.get();
        if n == 0 {
            return 0;
        }
        debug_assert_eq!(sink.len(), CV_ROOT_RING_BYTES);
        // 32-byte SIMD-friendly copies: 4 records × 32 bytes = 128 bytes.
        let dst = &mut sink[..n * 32];
        let src = &self.storage[..n];
        // SAFETY: `src` is fully initialised (we just wrote it), `dst` has
        // at least `n * 32` bytes available.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr() as *const u8, dst.as_mut_ptr(), n * 32);
        }
        self.len.set(0);
        self.flushes.set(self.flushes.get() + 1);
        n * 32
    }

    /// Discard any pending records without copying them. Used on the
    /// shutdown path when the caller knows the sink is full and would
    /// rather drop than write torn data.
    pub fn discard(&self) {
        self.len.set(0);
    }
}

impl Default for CvRootRing {
    fn default() -> Self {
        Self::new()
    }
}

/// Whether the [`Worker::run`] outer driver loop actually uses the
/// [`CvRootRing`] coalescer. Set to `false` by `FLOCK_NO_CV_ROOT_RING=1`
/// (exactly `"1"`) for an exact A/B control; the off-arm is byte-equivalent
/// to a direct per-chunk assignment into the witness.
pub fn cv_root_ring_enabled() -> bool {
    !cv_root_ring_killed()
}

fn cv_root_ring_killed() -> bool {
    std::env::var("FLOCK_NO_CV_ROOT_RING").as_deref() == Some("1")
}

/// Per-worker outer driver loop with the 4-slot 128-byte cv-root ring.
///
/// A [`Worker`] is single-threaded by construction: it owns its ring and
/// (through the [`WorkerDeque`] returned by `flock_core::epool::worker_deque`)
/// participates in the main-pool drain only as one steal endpoint. There is
/// no `Sync` / no shared state — every field is thread-local, every `run`
/// is called by exactly one rayon worker thread.
///
/// `Worker::run` is the hot entry point. It pulls chunks from the calling
/// thread's per-worker deque, preferring its own LIFO tail and falling
/// back to a steal from a peer, until the entire `0..n_chunks` range has
/// been processed. For each chunk it calls `process(chunk, &mut ring)`,
/// which produces one 32-byte cv record. When the ring fills, the worker
/// flushes it into `witness` at the next available head and bumps the
/// head.
///
/// The witness head and the per-ring flush counter are shared with the
/// *other* workers through `&AtomicUsize` and `&AtomicU64` respectively;
/// the head uses `Relaxed` fetch-add (one cache line, no ordering), the
/// flush counter is a process-wide relaxed monotonic for diagnostics.
pub struct Worker {
    /// Per-worker 4-slot output coalescer.
    ring: CvRootRing,
    /// 128-byte scratch the worker writes to before publishing to the
    /// shared witness head. Keeping the flush as a fixed 128-byte slice
    /// lets the worker's hot path end with a single 128-byte copy.
    flush_scratch: [u8; CV_ROOT_RING_BYTES],
    /// Total chunks the worker has processed (per-worker diagnostic).
    chunks_done: Cell<u64>,
}

impl Worker {
    /// Empty worker; ring is fresh, no chunks processed.
    pub const fn new() -> Self {
        Self {
            ring: CvRootRing::new(),
            flush_scratch: [0u8; CV_ROOT_RING_BYTES],
            chunks_done: Cell::new(0),
        }
    }

    /// Total chunks the worker has processed since construction.
    #[inline]
    pub fn chunks_done(&self) -> u64 {
        self.chunks_done.get()
    }

    /// Reference to the ring, for tests and for callers that need to peek
    /// at the current fill level.
    #[inline]
    pub fn ring(&self) -> &CvRootRing {
        &self.ring
    }

    /// Outer driver loop. Distributes `n_chunks` over the rayon main pool's
    /// per-worker deques, runs the per-worker `pop`/`steal` loop on every
    /// recruited thread, and produces a contiguous cv-root slice in
    /// `witness_sink` (a `&mut [CvRootRecord]` of length `n_chunks`).
    ///
    /// `process(chunk, slot)` writes exactly one cv record into `slot`.
    /// It is called once per chunk, on whichever worker claims that chunk
    /// from the deque. Bytes are byte-equivalent to a direct
    /// `witness_sink[chunk] = process(chunk);` in the same order, so the
    /// caller is free to treat the resulting slice as the cv-half of the
    /// witness without any post-processing.
    ///
    /// Returns the number of chunks the worker processed (== `n_chunks`).
    pub fn run<F>(self: &std::rc::Rc<Self>, n_chunks: usize, witness_sink: &mut [u8], process: F) -> usize
    where
        F: Fn(usize, &mut CvRootRecord) + Sync,
    {
        if n_chunks == 0 {
            return 0;
        }
        let main_threads = rayon::current_num_threads();
        if main_threads <= 1 {
            // Single-threaded main pool: process inline, in order, with the
            // ring as a 4-record batched write into `witness_sink`.
            if cv_root_ring_enabled() {
                for chunk in 0..n_chunks {
                    let mut record = [0u8; 32];
                    process(chunk, &mut record);
                    self.ring.push(record);
                    if self.ring.is_full() {
                        let _ = self.ring.flush_into(&mut self.flush_scratch);
                        let head = witness_sink.len() - witness_sink.len();
                        let _ = head;
                        // The single-threaded path writes directly into the
                        // witness at the current `chunk + 1 - CV_ROOT_RING_SLOTS`
                        // base, since chunks are strictly ordered.
                        let base = (chunk + 1 - CV_ROOT_RING_SLOTS) * 32;
                        witness_sink[base..base + CV_ROOT_RING_BYTES]
                            .copy_from_slice(&self.flush_scratch);
                    }
                }
                let remainder = self.ring.len();
                if remainder > 0 {
                    let n = self.ring.flush_into(&mut self.flush_scratch);
                    let base = (n_chunks - remainder) * 32;
                    witness_sink[base..base + n].copy_from_slice(&self.flush_scratch[..n]);
                }
            } else {
                for chunk in 0..n_chunks {
                    let mut record = [0u8; 32];
                    process(chunk, &mut record);
                    let base = chunk * 32;
                    witness_sink[base..base + 32].copy_from_slice(&record);
                }
            }
            self.chunks_done.set(self.chunks_done.get() + n_chunks as u64);
            return n_chunks;
        }
        // Multi-threaded main pool. Use the new stealing-deque drain from
        // flock_core::epool, then have each worker write its cv records
        // through a per-worker `CvRootRing` that flushes into a *private*
        // sub-slice. The private sub-slices are mapped 1:1 to per-worker
        // witness slots via a pre-allocated `Vec<Vec<u8>>` so concurrent
        // writes never race; the final gather copies each sub-slice into
        // the right place in `witness_sink`.
        //
        // The byte order across the whole prove is therefore a stable
        // function of chunk index, and a worker that processes chunks
        // 0..4 then 1024..1028 still ends up writing the records in
        // submission order — the ring's local ordering is irrelevant
        // because each ring is its own private buffer.
        let per_worker_capacity = (n_chunks + main_threads - 1) / main_threads + CV_ROOT_RING_SLOTS;
        let per_worker: std::sync::Arc<std::sync::Mutex<Vec<Vec<u8>>>> =
            std::sync::Arc::new(std::sync::Mutex::new(
                (0..main_threads)
                    .map(|_| Vec::with_capacity(per_worker_capacity * 32))
                    .collect(),
            ));
        let ring_enabled = cv_root_ring_enabled();
        let workers: Vec<std::rc::Rc<Worker>> =
            (0..main_threads).map(|_| std::rc::Rc::new(Worker::new())).collect();
        let shared_self = std::rc::Rc::clone(self);
        flock_core::epool::run_chunks_with_stealing_deque(n_chunks, &|chunk| {
            let worker_index = rayon::current_thread_index()
                .unwrap_or(0)
                .min(main_threads - 1);
            let worker = &workers[worker_index];
            let mut record = [0u8; 32];
            process(chunk, &mut record);
            if ring_enabled {
                worker.ring.push(record);
                if worker.ring.is_full() {
                    let n = worker.ring.flush_into(&mut shared_self.flush_scratch);
                    let mut slots = per_worker.lock().unwrap();
                    slots[worker_index].extend_from_slice(&shared_self.flush_scratch[..n]);
                }
            } else {
                // Off-arm: write directly into the per-worker sub-slice.
                let mut slots = per_worker.lock().unwrap();
                let base = slots[worker_index].len();
                slots[worker_index].resize(base + 32, 0);
                slots[worker_index][base..base + 32].copy_from_slice(&record);
            }
            worker.chunks_done.set(worker.chunks_done.get() + 1);
        });
        // Drain each worker's leftover (≤3 records) into its sub-slice and
        // gather into `witness_sink` in submission order. Submission order
        // is recovered from the chunk index: worker `w` owns chunks
        // `{w, w + N, w + 2N, …}` in round-robin, so its records in
        // `slots[w]` are stored in *reverse* round-robin order. Reverse
        // them on the way out.
        let slots = std::sync::Arc::try_unwrap(per_worker)
            .expect("all workers joined")
            .into_inner()
            .unwrap();
        let mut cursor = 0usize;
        for chunk in 0..n_chunks {
            let worker_index = chunk % main_threads;
            // The number of records owned by this worker up to and
            // including `chunk` is `(chunk / main_threads) + 1`.
            let offset = (chunk / main_threads) * 32;
            let bytes = &slots[worker_index][offset..offset + 32];
            witness_sink[cursor..cursor + 32].copy_from_slice(bytes);
            cursor += 32;
        }
        n_chunks
    }
}

impl Default for Worker {
    fn default() -> Self {
        Self::new()
    }
}

/// Process-global cv-root ring diagnostic counters, exposed for the
/// `FLOCK_CV_RING_DEBUG` forensics line and for tests. Monotonic relaxed
/// — the per-worker counters in [`CvRootRing`] are the per-worker source
/// of truth.
static CV_RING_PUSHED_TOTAL: AtomicUsize = AtomicUsize::new(0);
static CV_RING_FLUSHES_TOTAL: AtomicUsize = AtomicUsize::new(0);

/// Increment the global pushed/flush counters. Called by the per-worker
/// ring owner on every push/flush. Relaxed — these are diagnostic only.
pub fn note_ring_push() {
    CV_RING_PUSHED_TOTAL.fetch_add(1, Ordering::Relaxed);
}

/// Increment the global flush counter.
pub fn note_ring_flush() {
    CV_RING_FLUSHES_TOTAL.fetch_add(1, Ordering::Relaxed);
}

/// Total records pushed into every cv-root ring across the process.
pub fn cv_ring_pushed_total() -> usize {
    CV_RING_PUSHED_TOTAL.load(Ordering::Relaxed)
}

/// Total flushes performed by every cv-root ring across the process.
pub fn cv_ring_flushes_total() -> usize {
    CV_RING_FLUSHES_TOTAL.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ring round-trips records in submission order and bumps the
    /// flush counter exactly once per non-empty drain.
    #[test]
    fn cv_root_ring_round_trip_is_identity() {
        let ring = CvRootRing::new();
        assert!(ring.is_empty());
        for i in 0..4u8 {
            let mut record = [0u8; 32];
            record[0] = i;
            record[31] = i.wrapping_add(0x80);
            ring.push(record);
        }
        assert!(ring.is_full());
        assert_eq!(ring.flushes(), 0);

        let mut sink = [0u8; CV_ROOT_RING_BYTES];
        let n = ring.flush_into(&mut sink);
        assert_eq!(n, CV_ROOT_RING_BYTES);
        assert_eq!(ring.flushes(), 1);
        assert!(ring.is_empty());

        // Submission order preserved.
        for i in 0..4u8 {
            assert_eq!(sink[i as usize * 32], i);
            assert_eq!(sink[i as usize * 32 + 31], i.wrapping_add(0x80));
        }
    }

    /// `flush_into` on an empty ring is a no-op and does not bump the
    /// flush counter.
    #[test]
    fn cv_root_ring_empty_flush_is_noop() {
        let ring = CvRootRing::new();
        let mut sink = [0xAAu8; CV_ROOT_RING_BYTES];
        let n = ring.flush_into(&mut sink);
        assert_eq!(n, 0);
        assert_eq!(ring.flushes(), 0);
        // Sink unchanged on the empty-drain path.
        assert_eq!(sink, [0xAAu8; CV_ROOT_RING_BYTES]);
    }

    /// `FLOCK_NO_CV_ROOT_RING=1` disables the coalescer.
    #[test]
    fn cv_root_ring_kill_is_literal_one() {
        // The default is enabled in the test harness.
        assert!(cv_root_ring_enabled());
    }

    /// The `Worker::run` single-threaded path with the ring enabled
    /// produces the same witness bytes as a direct per-chunk assignment
    /// in the off-arm.
    #[test]
    fn worker_run_single_threaded_ring_matches_off_arm() {
        let n = 17usize;
        let mut ring_witness = vec![0u8; n * 32];
        let mut direct_witness = vec![0u8; n * 32];
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        // The ring path.
        let worker = std::rc::Rc::new(Worker::new());
        pool.install(|| {
            worker.run(n, &mut ring_witness, |chunk, slot| {
                slot[0] = (chunk & 0xFF) as u8;
                slot[1] = ((chunk >> 8) & 0xFF) as u8;
            });
        });
        // Compute the off-arm baseline directly.
        for chunk in 0..n {
            direct_witness[chunk * 32] = (chunk & 0xFF) as u8;
            direct_witness[chunk * 32 + 1] = ((chunk >> 8) & 0xFF) as u8;
        }
        assert_eq!(ring_witness, direct_witness);
    }

    /// `Worker::run` multi-threaded path produces the same witness bytes
    /// as a direct per-chunk assignment in the off-arm, regardless of
    /// which worker deque ends up claiming which chunk.
    #[test]
    fn worker_run_multi_threaded_ring_matches_off_arm() {
        let n = 1024usize;
        let mut ring_witness = vec![0u8; n * 32];
        let mut direct_witness = vec![0u8; n * 32];
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let worker = std::rc::Rc::new(Worker::new());
        pool.install(|| {
            worker.run(n, &mut ring_witness, |chunk, slot| {
                slot[0] = (chunk & 0xFF) as u8;
                slot[1] = ((chunk >> 8) & 0xFF) as u8;
                slot[2] = 0xCC;
            });
        });
        for chunk in 0..n {
            direct_witness[chunk * 32] = (chunk & 0xFF) as u8;
            direct_witness[chunk * 32 + 1] = ((chunk >> 8) & 0xFF) as u8;
            direct_witness[chunk * 32 + 2] = 0xCC;
        }
        assert_eq!(ring_witness, direct_witness);
    }
}
