//! Verifier-randomness abstraction.
//!
//! A [`Challenger`] is the source of verifier challenges in the protocol.
//! The prover writes its messages into the challenger (`observe_*`) and reads
//! challenges back out (`sample_*`). The verifier mirrors this exactly — as
//! it walks through the proof, it observes each prover message and samples
//! the same challenges, so both sides derive the same randomness in lockstep.
//!
//! Two implementations:
//! - `RandomChallenger` — seeded pseudo-random, ignores observed messages.
//!   Kept around for bench isolation (measure prover cost without FS overhead)
//!   and soundness mutation tests. **Not sound for real proofs**, and to make
//!   that structural it is compiled *only* under `cfg(test)` or the
//!   `unsound-challenger` feature — a normal (real-proof) build has no insecure
//!   challenger to reach for.
//! - [`FsChallenger`] — Fiat-Shamir over a selectable hash, SHA-256 (the
//!   default) or BLAKE3, chosen with [`FsChallenger::with_hash`]. Absorbs
//!   observations into a running hash state; samples by cloning the state and
//!   squeezing bytes from it, then re-absorbing the squeezed bytes so the next
//!   challenge binds to the previous one (Merlin-style duplex).
//!
//!   The transcript hash is independent of the Merkle hash
//!   ([`crate::pcs::commit::PcsParams::merkle_hash`]) — set both to the same
//!   value if you want the whole system resting on a single primitive.

use crate::field::F128;
use crate::hash::HashKind;
use sha2::{Digest, Sha256};

// `Send` supertrait: the verifier runs its PIOP/PCS replay inside a dedicated
// single-thread rayon pool (see `verifier::verifier_pool`), so the challenger
// it threads through must be able to cross into that pool. Both concrete
// challengers (`RandomChallenger`, `FsChallenger`) are trivially `Send`.
pub trait Challenger: Send {
    /// Absorb a domain-separation label (e.g. `b"flock-zerocheck-v0"`). Each
    /// protocol entry should call this once on entry so a transcript from
    /// one protocol cannot be replayed as another.
    fn observe_label(&mut self, _label: &[u8]) {
        // default no-op — RandomChallenger inherits this.
    }

    /// Absorb a single F128 prover message.
    fn observe_f128(&mut self, value: F128);

    /// Absorb a slice of F128 prover messages (e.g. the round-1 vector).
    fn observe_f128_slice(&mut self, values: &[F128]) {
        for v in values {
            self.observe_f128(*v);
        }
    }

    /// Absorb arbitrary bytes (e.g. a Merkle root or a statement digest).
    fn observe_bytes(&mut self, _bytes: &[u8]) {
        // default no-op — RandomChallenger inherits this.
    }

    /// Produce one F128 challenge.
    fn sample_f128(&mut self) -> F128;

    /// Produce `n` F128 challenges, in order.
    fn sample_f128_vec(&mut self, n: usize) -> Vec<F128> {
        (0..n).map(|_| self.sample_f128()).collect()
    }

    /// Prover-side PoW grinding: snapshot the current transcript state,
    /// search for a `u64` nonce such that `H(state ‖ nonce)` has at
    /// least `bits` leading zero bits, then absorb the nonce into the
    /// transcript so subsequent challenges bind to it.
    ///
    /// Default implementation is a no-op (returns 0). Real implementations
    /// — e.g. [`FsChallenger`] — do the actual grind work and absorb the
    /// nonce. `bits = 0` means "no PoW required"; still absorbs the 0 nonce
    /// so the verifier mirror is byte-identical.
    fn grind_pow(&mut self, _bits: u32) -> u64 {
        0
    }

    /// Verifier-side mirror of [`Self::grind_pow`]: check that `nonce`
    /// satisfies the `bits`-leading-zeros PoW against the current transcript
    /// state, then absorb the nonce so the running state stays in lockstep
    /// with the prover.
    ///
    /// Default implementation accepts unconditionally (no-op). Real
    /// implementations must check the PoW; an honest verifier rejects the
    /// proof if this returns `false`.
    fn verify_pow(&mut self, _nonce: u64, _bits: u32) -> bool {
        true
    }

    /// Fork this challenger into an independent copy with identical
    /// transcript state, for speculative pre-derivation of challenges whose
    /// inputs are already fully bound (commit-tail fill). The fork must be
    /// deterministic: absorbing the same messages into the fork and into
    /// `self` must yield identical samples. `None` (the default) disables
    /// every speculative consumer, which is always sound.
    fn fork(&self) -> Option<Self>
    where
        Self: Sized,
    {
        None
    }
}

// ---------------------------------------------------------------------------
// RandomChallenger — seeded SplitMix64 pseudo-random source.
//
// Ignores observed messages (no Fiat-Shamir binding). Keep for bench isolation
// and soundness mutation tests; real proofs MUST use FsChallenger.
//
// Gated behind `cfg(test)` / `feature = "unsound-challenger"`: a real-proof
// build does not compile this type at all, so no production code path can
// accidentally instantiate an unsound challenger. See the module docs.
// ---------------------------------------------------------------------------

#[cfg(any(test, feature = "unsound-challenger"))]
#[derive(Clone, Debug)]
pub struct RandomChallenger {
    state: u64,
}

#[cfg(any(test, feature = "unsound-challenger"))]
impl RandomChallenger {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }
}

#[cfg(any(test, feature = "unsound-challenger"))]
impl Challenger for RandomChallenger {
    #[inline]
    fn observe_f128(&mut self, _value: F128) {
        // intentional no-op: random challenger is independent of prover state
    }

    fn sample_f128(&mut self) -> F128 {
        let lo = splitmix64(&mut self.state);
        let hi = splitmix64(&mut self.state);
        F128 { lo, hi }
    }

    fn fork(&self) -> Option<Self> {
        Some(self.clone())
    }
}

#[cfg(any(test, feature = "unsound-challenger"))]
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

// ---------------------------------------------------------------------------
// FsChallenger — Fiat-Shamir over a selectable hash (SHA-256 or BLAKE3).
//
// Tag bytes (one-byte op + one-byte kind) encode the operation type so that
// e.g. an `observe_f128_slice` of length 1 cannot collide with `observe_f128`,
// and a slice observation cannot collide with two scalar observations of the
// same total length. Tagging, absorption order and the duplex structure are
// identical for both hashes — only the primitive differs.
//
// Sampling clones the live hasher, squeezes challenge bytes, and absorbs the
// squeezed output back into the live state. This "duplex" pattern binds each
// subsequent challenge/observation to all prior squeezed output.
//
// How the squeeze itself is done is the one place the two hashes genuinely
// diverge, because SHA-256 is not an extendable-output function and BLAKE3 is:
//
//   SHA-256: derive the stream as SHA256(state ‖ ctr) for ctr = 0, 1, …,
//            32 bytes at a time.
//   BLAKE3:  finalize the cloned state into an XOF reader and fill straight
//            from it — no counter, and one finalization regardless of length.
//
// Both are deterministic functions of the transcript state, which is all the
// duplex requires. The counter is a workaround for SHA-256's fixed output, so
// BLAKE3 does not inherit it; a proof is only ever verified under the same
// hash it was produced with (see `FsChallenger::with_hash`).
// ---------------------------------------------------------------------------

const OP_DOMAIN: u8 = 0x01;
const OP_LABEL: u8 = 0x02;
const OP_OBSERVE: u8 = 0x03;
const OP_SQUEEZE: u8 = 0x04;
const OP_BYTES: u8 = 0x05;

const KIND_SCALAR: u8 = 0x01;
const KIND_SLICE: u8 = 0x02;

#[cfg(target_endian = "little")]
#[inline]
fn f128s_as_bytes(values: &[F128]) -> &[u8] {
    // SAFETY: F128 is exactly two adjacent u64 fields under repr(C, align(16)),
    // so it has no padding and every byte is initialized. On little-endian
    // targets this is the canonical lo.to_le_bytes() || hi.to_le_bytes() layout.
    unsafe {
        core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), core::mem::size_of_val(values))
    }
}

#[cfg(target_endian = "little")]
#[inline]
fn f128s_as_bytes_mut(values: &mut [F128]) -> &mut [u8] {
    let byte_len = core::mem::size_of_val(values);
    // SAFETY: F128 has the padding-free layout documented above, and all bit
    // patterns are valid because both fields are u64. The caller initializes
    // the F128 slice before exposing its bytes.
    unsafe { core::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(), byte_len) }
}

/// Global Fiat–Shamir hash counters, enabled with `--features hash-count`.
/// Tracks the squeeze count and the PoW checks; absorbed transcript bytes are
/// tracked via [`FsChallenger::absorbed_bytes`].
#[cfg(feature = "hash-count")]
pub mod fs_count {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    /// Number of XOF finalizations (one per `sample_f128` /
    /// `sample_f128_vec` / PoW state-digest extraction).
    pub static SQUEEZES: AtomicU64 = AtomicU64::new(0);
    /// Number of PoW evaluations, under whichever hash the transcript uses
    /// (1 compression each; 40 B input).
    pub static POW_SHA256: AtomicU64 = AtomicU64::new(0);

    pub fn reset() {
        SQUEEZES.store(0, Relaxed);
        POW_SHA256.store(0, Relaxed);
    }

    /// (squeezes, pow_calls)
    pub fn snapshot() -> (u64, u64) {
        (SQUEEZES.load(Relaxed), POW_SHA256.load(Relaxed))
    }
}

/// The running transcript state, one variant per supported hash.
#[derive(Clone)]
enum FsState {
    Sha256(Sha256),
    Blake3(Box<blake3::Hasher>),
}

#[derive(Clone)]
pub struct FsChallenger {
    state: FsState,
    /// Running total of absorbed transcript bytes, for the `hash-count`
    /// instrumentation (read only under that feature).
    #[allow(dead_code)]
    n_absorbed: u64,
}

impl FsChallenger {
    /// New challenger seeded with a domain-separation tag (e.g.
    /// `b"flock-r1cs-v0"`), using SHA-256.
    ///
    /// The domain is length-prefixed before being absorbed so two domains
    /// where one is a prefix of the other cannot produce the same initial
    /// state. For the BLAKE3 transcript, see [`Self::with_hash`].
    pub fn new(domain: &[u8]) -> Self {
        Self::with_hash(domain, HashKind::Sha256)
    }

    /// New challenger over an explicit hash.
    ///
    /// The prover and verifier must agree: the transcript is a function of the
    /// hash, so a mismatch diverges at the first challenge and the proof fails
    /// to verify. That is the intended failure mode — nothing tries to detect
    /// or negotiate it, exactly as with the Merkle hash.
    pub fn with_hash(domain: &[u8], kind: HashKind) -> Self {
        let mut c = Self {
            state: match kind {
                HashKind::Sha256 => FsState::Sha256(Sha256::new()),
                HashKind::Blake3 => FsState::Blake3(Box::new(blake3::Hasher::new())),
            },
            n_absorbed: 0,
        };
        c.absorb(&[OP_DOMAIN]);
        c.absorb(&(domain.len() as u64).to_le_bytes());
        c.absorb(domain);
        c
    }

    /// Which hash backs this transcript.
    pub fn hash_kind(&self) -> HashKind {
        match self.state {
            FsState::Sha256(_) => HashKind::Sha256,
            FsState::Blake3(_) => HashKind::Blake3,
        }
    }

    /// Absorb bytes into the running transcript state.
    #[inline]
    fn absorb(&mut self, bytes: &[u8]) {
        match &mut self.state {
            FsState::Sha256(h) => {
                h.update(bytes);
            }
            FsState::Blake3(h) => {
                h.update(bytes);
            }
        }
        self.n_absorbed = self.n_absorbed.wrapping_add(bytes.len() as u64);
    }

    #[inline]
    fn absorb_f128(&mut self, v: F128) {
        self.absorb(&v.lo.to_le_bytes());
        self.absorb(&v.hi.to_le_bytes());
    }

    /// Squeeze `out.len()` pseudorandom bytes from the current transcript
    /// state without mutating it.
    ///
    /// SHA-256 is not an XOF, so its stream is `SHA256(state ‖ ctr)` for
    /// ctr = 0, 1, … (32 bytes each). BLAKE3 *is* an XOF, so it finalizes the
    /// cloned state once and fills straight from the reader — no counter, and
    /// no per-32-byte re-finalization.
    fn squeeze_into(&self, out: &mut [u8]) {
        match &self.state {
            FsState::Sha256(hasher) => {
                let mut off = 0usize;
                let mut ctr: u64 = 0;
                while off < out.len() {
                    let mut h = hasher.clone();
                    h.update(ctr.to_le_bytes());
                    let block: [u8; 32] = h.finalize().into();
                    let take = (out.len() - off).min(32);
                    out[off..off + take].copy_from_slice(&block[..take]);
                    off += take;
                    ctr = ctr.wrapping_add(1);
                }
            }
            FsState::Blake3(hasher) => hasher.finalize_xof().fill(out),
        }
    }

    /// 32-byte digest of the current transcript state, used as the PoW base.
    /// Cloning + finalizing gives a state-bound digest without mutating the
    /// live hasher.
    #[inline]
    fn state_digest(&self) -> [u8; 32] {
        #[cfg(feature = "hash-count")]
        fs_count::SQUEEZES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        match &self.state {
            FsState::Sha256(h) => h.clone().finalize().into(),
            FsState::Blake3(h) => *h.finalize().as_bytes(),
        }
    }

    /// Total bytes absorbed into the transcript so far. Used by the
    /// `hash-count` instrumentation to estimate SHA-256 compression calls
    /// (≈ bytes / 64).
    #[cfg(feature = "hash-count")]
    pub fn absorbed_bytes(&self) -> u64 {
        self.n_absorbed
    }
}

impl Challenger for FsChallenger {
    fn observe_label(&mut self, label: &[u8]) {
        self.absorb(&[OP_LABEL]);
        self.absorb(&(label.len() as u64).to_le_bytes());
        self.absorb(label);
    }

    fn fork(&self) -> Option<Self> {
        Some(self.clone())
    }

    fn observe_f128(&mut self, value: F128) {
        self.absorb(&[OP_OBSERVE, KIND_SCALAR]);
        self.absorb_f128(value);
    }

    fn observe_f128_slice(&mut self, values: &[F128]) {
        self.absorb(&[OP_OBSERVE, KIND_SLICE]);
        self.absorb(&(values.len() as u64).to_le_bytes());

        #[cfg(target_endian = "little")]
        self.absorb(f128s_as_bytes(values));

        #[cfg(not(target_endian = "little"))]
        for v in values {
            self.absorb_f128(*v);
        }
    }

    fn observe_bytes(&mut self, bytes: &[u8]) {
        self.absorb(&[OP_BYTES]);
        self.absorb(&(bytes.len() as u64).to_le_bytes());
        self.absorb(bytes);
    }

    fn sample_f128(&mut self) -> F128 {
        #[cfg(feature = "hash-count")]
        fs_count::SQUEEZES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.absorb(&[OP_SQUEEZE, KIND_SCALAR]);
        let mut buf = [0u8; 16];
        self.squeeze_into(&mut buf);
        // Re-absorb the squeezed bytes so subsequent ops bind to this challenge.
        self.absorb(&buf);
        let lo = u64::from_le_bytes(buf[..8].try_into().unwrap());
        let hi = u64::from_le_bytes(buf[8..].try_into().unwrap());
        F128 { lo, hi }
    }

    fn sample_f128_vec(&mut self, n: usize) -> Vec<F128> {
        #[cfg(feature = "hash-count")]
        fs_count::SQUEEZES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.absorb(&[OP_SQUEEZE, KIND_SLICE]);
        self.absorb(&(n as u64).to_le_bytes());

        #[cfg(target_endian = "little")]
        {
            let mut values = vec![F128::ZERO; n];
            let bytes = f128s_as_bytes_mut(&mut values);
            self.squeeze_into(bytes);
            self.absorb(bytes);
            values
        }

        #[cfg(not(target_endian = "little"))]
        {
            let mut buf = vec![0u8; n * 16];
            self.squeeze_into(&mut buf);
            self.absorb(&buf);
            buf.as_chunks::<16>()
                .0
                .iter()
                .map(|c| F128 {
                    lo: u64::from_le_bytes(c[..8].try_into().unwrap()),
                    hi: u64::from_le_bytes(c[8..].try_into().unwrap()),
                })
                .collect()
        }
    }

    fn grind_pow(&mut self, bits: u32) -> u64 {
        let kind = self.hash_kind();
        let state_digest = self.state_digest();
        // Aggregate-aware parallelism: decide on the grind's *expected hash
        // work* (`2^bits`), not a raw bit threshold. Fold-challenge grinds are
        // individually modest — e.g. 2^15 at L0 under the per-round profiles —
        // but the prover issues one per lane fold (6× at L0, 3× per recursive
        // level), so the per-level aggregate (~2^17–2^18 hashes) lands on the
        // multi-threaded critical path. We go parallel once a single grind
        // clears the rayon dispatch break-even (~2^13 hashes); the genuinely
        // tiny deep-level grinds (2^3–2^11) stay sequential, where the serial
        // loop beats parallel-dispatch overhead. `find_first` returns the
        // globally smallest satisfying nonce, so the result is identical to the
        // sequential search (deterministic proofs) regardless of this choice.
        const PARALLEL_GRIND_MIN_HASHES: u64 = 1 << 13;
        // Nonces per rayon task in the parallel search. Large enough to amortize
        // task dispatch and to let the BLAKE3 batch run many `hash_many` calls
        // per task, small enough to keep cancellation granular once an earlier
        // task has found a match. 960 = 20 full 48-nonce batches, so a chunk
        // has no ragged tail batch on the twelve-way kernel; the block math
        // below uses `div_ceil`/saturating adds and requires no power of two.
        // Chunk partition does not affect the emitted nonce: chunks are
        // scanned/`find_first`ed in ascending order and each returns its
        // smallest match, so the result stays the globally smallest.
        const GRIND_CHUNK: u64 = 960;
        let cpu_scan = || {
            if bits == 0 {
                0
            } else if (1u64 << bits.min(63)) < PARALLEL_GRIND_MIN_HASHES {
                // Sequential search: scan ascending blocks until a nonce lands.
                // `pow_scan` returns the smallest match within the block it is
                // given, so scanning blocks in order yields the globally smallest.
                let mut start: u64 = 0;
                loop {
                    if let Some(n) = pow_scan(&state_digest, start, GRIND_CHUNK, bits, kind) {
                        break n;
                    }
                    start = start.saturating_add(GRIND_CHUNK);
                }
            } else {
                // Two-pool block-parallel search. Blocks are scanned in order;
                // within a block, the main Rayon workers AND the efficiency-core
                // helper pool (when present) claim ascending chunks from one
                // shared atomic counter — the same heterogeneous shape as
                // `epool::run_hetero_chunks`, but with an early-exit bound so
                // workers stop claiming chunks that can no longer contain the
                // smallest match. The grind is pure batched BLAKE3 compute with
                // no per-chunk tables or bandwidth pressure, i.e. the friendliest
                // work in the prover for efficiency cores, and every ≥2^13-hash
                // grind sits on the opening's critical path while the helper pool
                // is otherwise parked.
                //
                // Determinism: the emitted nonce is re-derived after the join as
                // the match in the lowest-indexed chunk that has one. The counter
                // hands out indices in ascending order, a worker never abandons a
                // chunk it has claimed, and a chunk below the final bound with a
                // match would have lowered the bound below itself — contradiction.
                // So every chunk below the winning one was fully scanned with no
                // match, and the result is exactly the globally smallest nonce,
                // byte-identical to the sequential search.
                //
                // Block ≈ 2× the expected attempts: large enough that the match
                // usually falls inside one block (so all threads do useful
                // pre-match work), small enough to avoid the 4× over-scan the old
                // `+2` block caused (which left ~¾ of threads doing cancelled work).
                use rayon::prelude::*;
                use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
                const PENDING: u64 = u64::MAX;
                const NO_MATCH: u64 = u64::MAX - 1;
                let block: u64 = 1 << (bits.min(24) + 1);
                let n_chunks = usize::try_from(block.div_ceil(GRIND_CHUNK)).expect("chunk count");
                let mut start: u64 = 0;
                'blocks: loop {
                    let results: Vec<AtomicU64> =
                        (0..n_chunks).map(|_| AtomicU64::new(PENDING)).collect();
                    let next = AtomicUsize::new(0);
                    // Lowest chunk index known to hold a match. Chunks at or past
                    // it cannot supply the answer; Relaxed is enough because the
                    // bound only prunes work — correctness comes from `results`
                    // plus the joins below.
                    let bound = AtomicUsize::new(n_chunks);
                    let worker = || {
                        loop {
                            let c = next.fetch_add(1, Ordering::Relaxed);
                            if c >= n_chunks || c >= bound.load(Ordering::Relaxed) {
                                break;
                            }
                            match pow_scan(
                                &state_digest,
                                start.saturating_add(c as u64 * GRIND_CHUNK),
                                GRIND_CHUNK,
                                bits,
                                kind,
                            ) {
                                Some(n) => {
                                    results[c].store(n, Ordering::Release);
                                    bound.fetch_min(c, Ordering::Relaxed);
                                }
                                None => results[c].store(NO_MATCH, Ordering::Release),
                            }
                        }
                    };
                    let main_threads = rayon::current_num_threads();
                    let drain_main = || {
                        (0..main_threads)
                            .into_par_iter()
                            .with_max_len(1)
                            .for_each(|_| worker());
                    };
                    // Mirror the epool engagement floor: tiny blocks drain faster
                    // than the cross-pool kickoff amortizes.
                    const GRIND_EPOOL_MIN_CHUNKS: usize = 16;
                    match crate::epool::epool()
                        .filter(|_| main_threads > 1 && n_chunks >= GRIND_EPOOL_MIN_CHUNKS)
                    {
                        Some(ep) => {
                            // Engaged grind drain: same two-pool shape as the
                            // fold drains, so route the broadcast through the
                            // persistent relay instead of creating and joining
                            // one OS thread per engaged grind (7-8 of them sit
                            // on the serial Fiat-Shamir spine per prove). The
                            // relay posts from inside a main worker, so the
                            // E-broadcast starts no earlier relative to the
                            // main drain than the spawn it replaces, and a
                            // busy or disabled relay falls back to the exact
                            // incumbent scoped spawn. Chunk claims and the
                            // lowest-chunk match rule are untouched, so the
                            // nonce — and every transcript byte after it — is
                            // identical either way.
                            let broadcast = || {
                                ep.broadcast(|_| worker());
                            };
                            crate::epool::drain_hetero(
                                main_threads,
                                &worker,
                                &broadcast,
                                grind_relay_enabled() && crate::epool::relay_enabled(),
                            );
                        }
                        None => drain_main(),
                    }
                    // Both pools joined (Release stores in `results` are
                    // synchronized by the joins). Take the lowest-chunk match.
                    for r in &results {
                        match r.load(Ordering::Acquire) {
                            PENDING | NO_MATCH => continue,
                            n => break 'blocks n,
                        }
                    }
                    start = start.saturating_add(block);
                }
            }
        };
        let nonce = if kind == HashKind::Blake3
            && bits >= GPU_GRIND_MIN_BITS
            && !GPU_GRIND_FAILED.load(std::sync::atomic::Ordering::Relaxed)
        {
            match GPU_GRIND_LATCH.get().copied() {
                Some(true) => match gpu_blake3_pow_nonce(&state_digest, bits) {
                    Ok(nonce) => nonce,
                    Err(_) => {
                        GPU_GRIND_FAILED.store(true, std::sync::atomic::Ordering::Relaxed);
                        cpu_scan()
                    }
                },
                Some(false) => cpu_scan(),
                None if bits == GPU_GRIND_CALIBRATION_BITS
                    && crate::gpu_commit::gpu_grind_enabled() =>
                {
                    // The warm proof reaches the 19-bit L0 grind before the
                    // worker publishes readiness.  Benchmark the exact seven
                    // high-cost ranked dispatch sizes in both orders on this
                    // transcript state.  Fixed windows remove the warm seed's
                    // random first-hit distance from the admission decision.
                    let cpu_1_started = std::time::Instant::now();
                    let cpu_1 = cpu_gpu_grind_calibration(&state_digest);
                    let cpu_1_time = cpu_1_started.elapsed();
                    let gpu_1_started = std::time::Instant::now();
                    let gpu_1 = gpu_grind_calibration(&state_digest);
                    let gpu_1_time = gpu_1_started.elapsed();

                    let gpu_2_started = std::time::Instant::now();
                    let gpu_2 = gpu_grind_calibration(&state_digest);
                    let gpu_2_time = gpu_2_started.elapsed();
                    let cpu_2_started = std::time::Instant::now();
                    let cpu_2 = cpu_gpu_grind_calibration(&state_digest);
                    let cpu_2_time = cpu_2_started.elapsed();

                    let exact = matches!((&gpu_1, &gpu_2), (Ok(a), Ok(b)) if *a == cpu_1 && *b == cpu_1)
                        && cpu_2 == cpu_1;
                    // A fixed N-nonce Metal block succeeds with probability
                    // 1-e^-1, so an unbounded grind consumes 1/(1-e^-1) =
                    // 1.582 blocks on average.  Charge that overdraw (including
                    // recurring command costs) before comparing with CPU.
                    let enable = if grind_latch_min_enabled() {
                        // MEASUREMENT CORRECTION (kill: FLOCK_NO_GRIND_LATCH_MIN).
                        // Each device is timed twice; the calibration's
                        // contention (shared rayon pool, GPU governor ramp, and
                        // one-shot page wiring) is strictly ONE-SIDED — it only
                        // ever makes a draw slower, never faster — so the lesser
                        // of the two draws is the least-contended, most accurate
                        // estimate of each device's true cost. Price both arms
                        // from their per-device minimum. The legacy rule instead
                        // required BOTH per-sample GPU wins (an AND) and summed
                        // the two draws for the gain, so one unlucky-high GPU
                        // draw vetoed the whole process's latch for its lifetime
                        // (hunt4 F1: a draw 7.8x its sibling). The 1.582 overdraw
                        // factor, the `gpu*overdraw < cpu` comparison direction,
                        // and the 1.5 ms two-sample gain threshold are ALL
                        // unchanged — only the estimator over the repeats moves.
                        let cpu_est = cpu_1_time.min(cpu_2_time);
                        let gpu_est = gpu_1_time.min(gpu_2_time);
                        let projected_gpu = gpu_est.mul_f64(GPU_GRIND_BLOCK_OVERDRAW);
                        // Keep the gain on the two-sample scale the
                        // GPU_GRIND_MIN_TWO_SAMPLE_GAIN constant was set for: the
                        // two-pass total is estimated as twice the best pass, so
                        // the literal 1.5 ms threshold keeps its meaning.
                        let protected_gain =
                            cpu_est.saturating_sub(projected_gpu).saturating_mul(2);
                        exact
                            && projected_gpu < cpu_est
                            && protected_gain >= GPU_GRIND_MIN_TWO_SAMPLE_GAIN
                    } else {
                        let cpu_total = cpu_1_time + cpu_2_time;
                        let gpu_total = gpu_1_time + gpu_2_time;
                        let projected_gpu_total = gpu_total.mul_f64(GPU_GRIND_BLOCK_OVERDRAW);
                        let protected_gain = cpu_total.saturating_sub(projected_gpu_total);
                        exact
                            && gpu_1_time.mul_f64(GPU_GRIND_BLOCK_OVERDRAW) < cpu_1_time
                            && gpu_2_time.mul_f64(GPU_GRIND_BLOCK_OVERDRAW) < cpu_2_time
                            && protected_gain >= GPU_GRIND_MIN_TWO_SAMPLE_GAIN
                    };
                    if grind_trace_enabled() {
                        eprintln!(
                            "[grind] latch decision: enable={enable} exact={exact} \
                             cpu1={:.3}ms cpu2={:.3}ms gpu1={:.3}ms gpu2={:.3}ms",
                            cpu_1_time.as_secs_f64() * 1e3,
                            cpu_2_time.as_secs_f64() * 1e3,
                            gpu_1_time.as_secs_f64() * 1e3,
                            gpu_2_time.as_secs_f64() * 1e3,
                        );
                    }
                    let _ = GPU_GRIND_LATCH.set(enable);
                    if enable {
                        match gpu_blake3_pow_nonce(&state_digest, bits) {
                            Ok(nonce) => nonce,
                            Err(_) => {
                                GPU_GRIND_FAILED.store(true, std::sync::atomic::Ordering::Relaxed);
                                cpu_scan()
                            }
                        }
                    } else {
                        cpu_scan()
                    }
                }
                None => cpu_scan(),
            }
        } else {
            cpu_scan()
        };
        // Absorb the nonce so subsequent transcript state binds to it.
        // Verifier mirrors via verify_pow.
        self.observe_bytes(&nonce.to_le_bytes());
        nonce
    }

    fn verify_pow(&mut self, nonce: u64, bits: u32) -> bool {
        let kind = self.hash_kind();
        let state_digest = self.state_digest();
        let ok = if bits == 0 {
            // No PoW required here. An honest prover emits the canonical nonce
            // 0 (see `grind_pow`), so reject any non-zero value: it can only be
            // a re-grinding knob, and accepting it would leave proofs malleable
            // (a proof and its nonce-mutated twin would both verify). This
            // closes no soundness gap — when grinding_bits = 0 the query phase
            // already carries the full security target, and the FS soundness
            // accounting assumes free re-grinding regardless — it just keeps
            // proofs canonical / non-malleable at zero-bit grinding sites.
            nonce == 0
        } else {
            pow_has_leading_zero_bits(&state_digest, nonce, bits, kind)
        };
        // Absorb regardless of `ok` so the transcript stays byte-identical to
        // the prover's (an honest prover always reaches this with the same
        // nonce); a failed check rejects the proof at the call site anyway.
        self.observe_bytes(&nonce.to_le_bytes());
        ok
    }
}

// ---------------------------------------------------------------------------

const GPU_GRIND_CALIBRATION_BITS: u32 = 19;
const GPU_GRIND_MIN_BITS: u32 = 14;
const GPU_GRIND_CALIBRATION_PREDICATE_BITS: u32 = 17;
const GPU_GRIND_CALIBRATION_LENGTHS: [u32; 7] = [
    1 << 19,
    1 << 18,
    1 << 17,
    1 << 16,
    1 << 15,
    1 << 14,
    1 << 14,
];
const GPU_GRIND_BLOCK_OVERDRAW: f64 = 1.581_976_706_869_326_5;
// Two-sample engagement margin for the Metal grind arm. The min-of-two
// estimator already prices one-sided calibration contention (a draw can only
// be slowed by contention, never sped up), so a 1.5 ms protected margin
// over-rejects the Metal arm when its measured edge is real but modest.
// 300 us keeps a 2x safety factor over per-draw noise while admitting an
// arm whose measured per-trial saving clears ~150 us.
const GPU_GRIND_MIN_TWO_SAMPLE_GAIN: std::time::Duration = std::time::Duration::from_micros(300);
static GPU_GRIND_LATCH: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
static GPU_GRIND_FAILED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// CPU oracle/throughput arm for one fixed calibration window.  Unlike the
/// production search it intentionally drains every chunk, making runtime
/// independent of the first matching nonce while retaining the same P+E-core
/// hash kernel and queue shape.
fn cpu_blake3_pow_window(state_digest: &[u8; 32], start: u64, len: u32, bits: u32) -> Option<u64> {
    cpu_blake3_pow_window_inner(state_digest, start, len, bits, None)
}

/// Same drain-everything window as [`cpu_blake3_pow_window`], plus an
/// optional cooperative stop flag. The hybrid prefetch sets the flag from
/// the GPU-spin thread the moment the GPU block reports a hit — at that
/// point the prefetch window's result is discarded unconditionally (the
/// GPU nonce is the global minimum), so bailing out early cannot change
/// any returned nonce; it only stops burning cores on a dead scan. The
/// calibration oracles pass `None`: their runtime must stay independent
/// of first-hit distance.
fn cpu_blake3_pow_window_inner(
    state_digest: &[u8; 32],
    start: u64,
    len: u32,
    bits: u32,
    stop: Option<&std::sync::atomic::AtomicBool>,
) -> Option<u64> {
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    const CHUNK: u64 = 960;
    const NO_MATCH: u64 = u64::MAX;
    let n_chunks = usize::try_from(u64::from(len).div_ceil(CHUNK)).expect("chunk count");
    let results: Vec<AtomicU64> = (0..n_chunks).map(|_| AtomicU64::new(NO_MATCH)).collect();
    let next = AtomicUsize::new(0);
    let worker = || loop {
        if stop.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
            break;
        }
        let chunk = next.fetch_add(1, Ordering::Relaxed);
        if chunk >= n_chunks {
            break;
        }
        let offset = chunk as u64 * CHUNK;
        let chunk_len = CHUNK.min(u64::from(len) - offset);
        if let Some(nonce) =
            blake3_pow_scan(state_digest, start.saturating_add(offset), chunk_len, bits)
        {
            results[chunk].store(nonce, Ordering::Release);
        }
    };
    let main_threads = rayon::current_num_threads();
    let drain_main = || {
        (0..main_threads)
            .into_par_iter()
            .with_max_len(1)
            .for_each(|_| worker());
    };
    match crate::epool::epool().filter(|_| main_threads > 1 && n_chunks >= 16) {
        Some(ep) => std::thread::scope(|scope| {
            scope.spawn(|| ep.broadcast(|_| worker()));
            drain_main();
        }),
        None => drain_main(),
    }
    results
        .iter()
        .map(|result| result.load(Ordering::Acquire))
        .filter(|&nonce| nonce != NO_MATCH)
        .min()
}

fn cpu_gpu_grind_calibration(state_digest: &[u8; 32]) -> Vec<Option<u64>> {
    let mut start = 0u64;
    GPU_GRIND_CALIBRATION_LENGTHS
        .into_iter()
        .map(|len| {
            let result = cpu_blake3_pow_window(
                state_digest,
                start,
                len,
                GPU_GRIND_CALIBRATION_PREDICATE_BITS,
            );
            start += u64::from(len);
            result
        })
        .collect()
}

fn gpu_grind_calibration(state_digest: &[u8; 32]) -> Result<Vec<Option<u64>>, String> {
    let mut start = 0u64;
    GPU_GRIND_CALIBRATION_LENGTHS
        .into_iter()
        .map(|len| {
            let result = crate::gpu_commit::gpu_blake3_pow_scan(
                state_digest,
                start,
                len,
                GPU_GRIND_CALIBRATION_PREDICATE_BITS,
            );
            start += u64::from(len);
            result
        })
        .collect()
}

/// Metal scans fixed ascending blocks and reports the smallest match in each
/// block.  Visiting blocks in order therefore returns the same global minimum
/// as the CPU scan.  One expected-work block balances GPU over-scan against
/// recurring command-buffer latency; failure is propagated to the caller's
/// exact CPU fallback.
///
/// Hybrid prefetch (`FLOCK_NO_GRIND_HYBRID` kills): while the calling thread
/// drives the GPU scan of block `[start, start+B)` — a sub-millisecond
/// dispatch it spins home — every CPU core is otherwise idle, because the
/// grind sits on the transcript's serial spine.  So the CPU concurrently
/// scans the *following* block `[start+B, start+2B)` with the same two-pool
/// kernel the pure-CPU path uses.  A one-expected-work block misses with
/// probability e^-1 ≈ 0.37; on a miss the old path paid a second full GPU
/// round trip (~0.8 ms fixed+kernel), which the prefetch has already covered
/// by the time the spin returns.  Determinism: the GPU reports the smallest
/// match in its block, the CPU window returns the smallest in the next block,
/// and blocks are visited in ascending order — if the GPU hits, every earlier
/// block was exhausted empty, so its nonce is the global minimum; if the GPU
/// block is empty and the CPU window hit, the same argument gives the CPU
/// minimum.  Byte-identical to the sequential search either way.
/// Persistent single-thread GPU-dispatch worker for the hybrid grind.
///
/// The grind sits on the transcript's serial spine; the incumbent hybrid path
/// `thread::scope`-spawns one OS thread per iteration (~11 per ranked prove)
/// and that spawn latency delays the GPU launch on the spine. A reused thread
/// (one per process, woken by an mpsc channel) removes the spawn cost and
/// starts the dispatch slightly earlier (a channel send is cheaper than a
/// spawn). Semantics are identical to the spawned path: one `pow_scan` per
/// job, the abort flag set on hit when enabled, the result delivered on the
/// job's own channel. `FLOCK_NO_GRIND_WORKER=1` (or any worker setup failure)
/// falls back to the exact incumbent scope-spawn path; a worker failure
/// mid-job degrades to the same `Err` the spawned path would propagate (CPU
/// grind fallback upstream). No GPU work is added or removed — only the
/// thread-creation latency is.
mod grind_worker {
    use std::sync::atomic::AtomicBool;
    use std::sync::mpsc::{Receiver, Sender, channel};
    use std::sync::{Arc, OnceLock};

    struct Job {
        digest: [u8; 32],
        start: u64,
        len: u32,
        bits: u32,
        stop: Arc<AtomicBool>,
        abort: bool,
        done: Sender<Result<Option<u64>, String>>,
    }

    struct Worker {
        tx: Sender<Job>,
        _thread: std::thread::JoinHandle<()>,
    }

    fn worker() -> Option<&'static Worker> {
        static W: OnceLock<Option<Worker>> = OnceLock::new();
        if std::env::var_os("FLOCK_NO_GRIND_WORKER").is_some() {
            return None;
        }
        W.get_or_init(|| {
            let (tx, rx): (Sender<Job>, Receiver<Job>) = channel();
            std::thread::Builder::new()
                .name("flock-grind-dispatch".into())
                .spawn(move || {
                    while let Ok(job) = rx.recv() {
                        let res = crate::gpu_commit::gpu_blake3_pow_scan(
                            &job.digest,
                            job.start,
                            job.len,
                            job.bits,
                        );
                        if job.abort && matches!(&res, Ok(Some(_))) {
                            job.stop.store(true, std::sync::atomic::Ordering::Release);
                        }
                        let _ = job.done.send(res);
                    }
                })
                .ok()
                .map(|thread| Worker {
                    tx,
                    _thread: thread,
                })
        })
        .as_ref()
    }

    /// Dispatch one GPU grind scan through the persistent worker without
    /// waiting for it. Returns `None` when the worker is unavailable (kill
    /// switch or setup failure) and the caller must run the incumbent
    /// scope-spawn path; `Some(rx)` delivers the scan outcome with the same
    /// `Err` surface as a direct `gpu_blake3_pow_scan` call (a closed
    /// channel — worker thread died — surfaces as `Err` on `recv`, exactly
    /// like the previous blocking `dispatch`).
    pub(super) fn dispatch_async(
        state_digest: &[u8; 32],
        start: u64,
        len: u32,
        bits: u32,
        stop: Arc<AtomicBool>,
        abort: bool,
    ) -> Option<Receiver<Result<Option<u64>, String>>> {
        let w = worker()?;
        let (done_tx, done_rx) = channel();
        let job = Job {
            digest: *state_digest,
            start,
            len,
            bits,
            stop,
            abort,
            done: done_tx,
        };
        if w.tx.send(job).is_err() {
            // Preserve the incumbent surface: the send failure is reported
            // through the receiver (its sender is dropped here), so `recv`
            // yields the same channel-closed error the blocking path
            // returned.
            return Some(done_rx);
        }
        Some(done_rx)
    }

    /// Receive one result from [`dispatch_async`], mapping a closed channel
    /// to the incumbent error string.
    pub(super) fn recv(rx: Receiver<Result<Option<u64>, String>>) -> Result<Option<u64>, String> {
        rx.recv()
            .unwrap_or_else(|_| Err("GPU grind worker channel closed".to_string()))
    }
}

/// Run the CPU next-block window after the persistent GPU worker has already
/// delivered its result. A successful GPU hit is in the earlier nonce block,
/// so the caller will return it without consulting the CPU result. When the
/// existing abort policy is armed, skip constructing and dispatching that
/// provably-dead CPU window altogether.
///
/// Keep this helper parameterized rather than reading process-global state so
/// the exact policy boundary can be tested without mutating the environment.
#[inline]
fn run_cpu_window_after_persistent_gpu(
    gpu_result: &Result<Option<u64>, String>,
    abort_enabled: bool,
    skip_dead_cpu_enabled: bool,
    cpu_window: impl FnOnce() -> Option<u64>,
) -> Option<u64> {
    if abort_enabled && skip_dead_cpu_enabled && matches!(gpu_result, Ok(Some(_))) {
        None
    } else {
        cpu_window()
    }
}

fn gpu_blake3_pow_nonce(state_digest: &[u8; 32], bits: u32) -> Result<u64, String> {
    debug_assert!((GPU_GRIND_MIN_BITS..=32).contains(&bits));
    let block_len = 1u32 << bits.min(24);
    let hybrid = grind_hybrid_enabled() && rayon::current_num_threads() > 1;
    let mut start = 0u64;
    loop {
        if !hybrid {
            if let Some(nonce) =
                crate::gpu_commit::gpu_blake3_pow_scan(state_digest, start, block_len, bits)?
            {
                return Ok(nonce);
            }
            start = start
                .checked_add(u64::from(block_len))
                .ok_or_else(|| "GPU grind nonce range exhausted".to_string())?;
            continue;
        }
        let next_start = start
            .checked_add(u64::from(block_len))
            .ok_or_else(|| "GPU grind nonce range exhausted".to_string())?;
        // The prefetch result is consumed only when the GPU block misses, so
        // the GPU thread flags a hit as soon as its spin returns and the CPU
        // window bails between chunks instead of draining a scan whose result
        // is already dead. On a GPU miss the flag is never set and the window
        // drains fully — byte-identical to the unflagged search either way
        // (`FLOCK_NO_GRIND_HYBRID_ABORT` restores the unconditional drain).
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let abort = grind_hybrid_abort_enabled();
        // Persistent dispatch worker when available; the incumbent
        // scope-spawn path otherwise (see `grind_worker`).
        let t_iter = grind_trace_enabled().then(std::time::Instant::now);
        let (gpu_result, cpu_next) = match grind_worker::dispatch_async(
            state_digest,
            start,
            block_len,
            bits,
            std::sync::Arc::clone(&stop),
            abort,
        ) {
            Some(done_rx) => {
                if grind_prefetch_overlap_enabled() {
                    // The worker owns the GPU scan; this thread — which
                    // would otherwise park in `recv` — drains the CPU
                    // next-block window in parallel, restoring the overlap
                    // the scope-spawn path always had. On a hit the worker
                    // pre-sets `stop` (abort policy), the window exits
                    // between chunks, and its result is discarded below;
                    // on a miss the flag never fires and the window drains
                    // fully. Either way the emitted nonce is byte-identical
                    // to the serialized shape.
                    let cpu = cpu_blake3_pow_window_inner(
                        state_digest,
                        next_start,
                        block_len,
                        bits,
                        abort.then_some(stop.as_ref()),
                    );
                    let gpu = grind_worker::recv(done_rx);
                    (gpu, cpu)
                } else {
                    // A/B-CONTROL (FLOCK_NO_GRIND_PREFETCH_OVERLAP=1): the
                    // prior serialized shape. On a hit the following CPU
                    // block is later in nonce order and its result is
                    // unconditionally discarded below, while the pre-set
                    // stop flag makes every worker exit immediately. Avoid
                    // the otherwise-dead allocation, Rayon drain, and
                    // E-pool broadcast.
                    let gpu = grind_worker::recv(done_rx);
                    let cpu = run_cpu_window_after_persistent_gpu(
                        &gpu,
                        abort,
                        grind_skip_dead_cpu_enabled(),
                        || {
                            cpu_blake3_pow_window_inner(
                                state_digest,
                                next_start,
                                block_len,
                                bits,
                                abort.then_some(stop.as_ref()),
                            )
                        },
                    );
                    (gpu, cpu)
                }
            }
            None => std::thread::scope(|s| {
                let gpu_scan = s.spawn(|| {
                    let gpu = crate::gpu_commit::gpu_blake3_pow_scan(
                        state_digest,
                        start,
                        block_len,
                        bits,
                    );
                    if abort && matches!(&gpu, Ok(Some(_))) {
                        stop.store(true, std::sync::atomic::Ordering::Release);
                    }
                    gpu
                });
                let cpu = cpu_blake3_pow_window_inner(
                    state_digest,
                    next_start,
                    block_len,
                    bits,
                    abort.then_some(stop.as_ref()),
                );
                let gpu = gpu_scan
                    .join()
                    .unwrap_or_else(|_| Err("GPU grind scan thread panicked".to_string()));
                (gpu, cpu)
            }),
        };
        if let Some(t) = t_iter {
            eprintln!(
                "[grind] bits={bits} block=2^{} gpu_hit={} cpu_hit={} iter_wall={:.3} ms",
                bits.min(24),
                matches!(&gpu_result, Ok(Some(_))),
                cpu_next.is_some(),
                t.elapsed().as_secs_f64() * 1e3,
            );
        }
        if let Some(nonce) = gpu_result? {
            return Ok(nonce);
        }
        if let Some(nonce) = cpu_next {
            return Ok(nonce);
        }
        start = next_start
            .checked_add(u64::from(block_len))
            .ok_or_else(|| "GPU grind nonce range exhausted".to_string())?;
    }
}

/// `FLOCK_NO_GRIND_LATCH_MIN` restores the legacy grind-latch admission rule
/// that AND-ed both per-sample GPU/CPU comparisons and summed the two draws
/// for the gain (exact rollback lever for the per-device min-estimator
/// measurement correction).
fn grind_latch_min_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_GRIND_LATCH_MIN").is_none())
}

/// Route the engaged CPU-grind drain's helper broadcast through the epool's
/// persistent relay instead of a fresh scoped thread per engaged grind.
/// Compile-time default so the cleared ranked environment ships the decision;
/// `FLOCK_NO_GRIND_RELAY=1` (exactly `"1"`, per the grind-reg precedent)
/// restores the incumbent per-grind spawn as the same-binary A/B control.
/// `FLOCK_NO_EPOOL_RELAY=1` also restores it, transitively, by disabling the
/// relay itself.
pub const GRIND_RELAY_DEFAULT: bool = true;
pub const ENV_NO_GRIND_RELAY: &str = "FLOCK_NO_GRIND_RELAY";

fn grind_relay_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        GRIND_RELAY_DEFAULT
            && std::env::var(ENV_NO_GRIND_RELAY).map(|v| v != "1").unwrap_or(true)
    })
}

/// `FLOCK_NO_GRIND_HYBRID` kills the CPU-prefetch arm of the GPU grind,
/// restoring the pure serial GPU block walk (exact rollback lever).
fn grind_hybrid_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_GRIND_HYBRID").is_none())
}

/// `FLOCK_NO_GRIND_HYBRID_ABORT` keeps the prefetch window draining to
/// completion even after the GPU block has hit (exact rollback lever for
/// the early-abort; the returned nonce is identical either way).
fn grind_hybrid_abort_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_GRIND_HYBRID_ABORT").is_none())
}

/// Skip the CPU next-block scaffold when the serialized persistent GPU worker
/// has already returned a hit. `FLOCK_NO_GRIND_SKIP_DEAD_CPU=1` restores the
/// previous behavior for exact same-binary comparison.
fn grind_skip_dead_cpu_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_GRIND_SKIP_DEAD_CPU").is_none())
}

/// Run the CPU next-block prefetch window *concurrently* with the persistent
/// GPU worker's scan, on the dispatching thread, instead of serialized behind
/// `done_rx.recv()`. The scope-spawn fallback always had this overlap; the
/// persistent worker lost it when it replaced the per-call thread. Nonce
/// selection is untouched (GPU block consulted first, CPU window second), so
/// the emitted nonce — and the proof bytes — are identical for every
/// hit/miss interleaving; only the wall changes. On a GPU hit with the abort
/// policy armed the worker thread sets `stop` and the window exits between
/// chunks, which is the exact incumbent scope-spawn behavior.
/// A/B-CONTROL: `FLOCK_NO_GRIND_PREFETCH_OVERLAP=1` restores the serialized
/// wait-then-window shape (including its skip-dead-CPU cut) for same-binary
/// comparison; ranked workers run the compile-time default below.
const GRIND_PREFETCH_OVERLAP_DEFAULT: bool = true;
fn grind_prefetch_overlap_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        if std::env::var_os("FLOCK_NO_GRIND_PREFETCH_OVERLAP").is_some() {
            false
        } else {
            GRIND_PREFETCH_OVERLAP_DEFAULT
        }
    })
}

/// Diagnostic-only: `FLOCK_GRIND_TRACE=1` prints one line per GPU-hybrid
/// grind iteration (bits, GPU hit/miss, GPU wall, CPU-window wall). Local
/// tooling; ranked workers never set it.
fn grind_trace_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_GRIND_TRACE").is_some())
}

// ---------------------------------------------------------------------------
// Proof-of-work grinding.
//
// The PoW pre-image is `state_digest ‖ nonce_le`, but its *padded length*
// differs per hash, because each hash has a different natural block:
//
//   SHA-256: 40 bytes. With the 0x80 pad and 8-byte length that is one
//            compression; padding further to 64 would make it two, halving
//            the grind rate for no benefit.
//   BLAKE3:  64 bytes (24 zero bytes of tail padding). A whole-block
//            single-chunk message is exactly what the crate's SIMD
//            `hash_many` can compute a batch of at a time, which is worth
//            ~2× on the nonce search — see `blake3_pow_scan`. At 40 bytes it
//            would be a partial block and could not be batched at all.
//
// Both are fixed-length and injective in `(state_digest, nonce)`, which is all
// the PoW needs; the asymmetry costs nothing and is never compared across
// hashes (a proof is only verified under the hash it was made with).
// ---------------------------------------------------------------------------

/// BLAKE3's PoW pre-image: `state_digest ‖ nonce_le ‖ zero padding`, one whole
/// 64-byte block. `blake3::hash` of this is what the PoW is defined against.
#[inline]
fn blake3_pow_preimage(state_digest: &[u8; 32], nonce: u64) -> [u8; 64] {
    let mut pre = [0u8; 64];
    pre[..32].copy_from_slice(state_digest);
    pre[32..40].copy_from_slice(&nonce.to_le_bytes());
    pre
}

/// Whether `h` has at least `bits` leading zero bits.
#[inline]
fn has_leading_zero_bits(h: &[u8], bits: u32) -> bool {
    let full_bytes = (bits / 8) as usize;
    let extra = bits % 8;
    for &b in h.iter().take(full_bytes) {
        if b != 0 {
            return false;
        }
    }
    if extra > 0 && (h[full_bytes] >> (8 - extra)) != 0 {
        return false;
    }
    true
}

/// Check whether `H(pre-image(state_digest, nonce))` has at least `bits`
/// leading zero bits, under the transcript's own hash `kind`.
///
/// This is the *specification* of the PoW — `verify_pow` uses it directly, and
/// the batched search below must agree with it for every nonce. Grinding under
/// the transcript's own hash keeps the whole protocol resting on one primitive
/// rather than pulling in a second.
#[inline]
fn pow_has_leading_zero_bits(
    state_digest: &[u8; 32],
    nonce: u64,
    bits: u32,
    kind: HashKind,
) -> bool {
    #[cfg(feature = "hash-count")]
    fs_count::POW_SHA256.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    match kind {
        HashKind::Sha256 => {
            let mut pre = [0u8; 40];
            pre[..32].copy_from_slice(state_digest);
            pre[32..].copy_from_slice(&nonce.to_le_bytes());
            let h: [u8; 32] = Sha256::digest(pre).into();
            has_leading_zero_bits(&h, bits)
        }
        HashKind::Blake3 => {
            let h = blake3::hash(&blake3_pow_preimage(state_digest, nonce));
            has_leading_zero_bits(h.as_bytes(), bits)
        }
    }
}

/// Nonces hashed per `hash_many` call in the BLAKE3 grind.
///
/// Must clear the widest `simd_degree` (16, under AVX-512) so the batch fills
/// the machine's vector. 48 = 4·12 = 3·16 is the smallest batch that is a
/// multiple of BOTH the Apple AArch64 twelve-way kernel's group size and the
/// AVX-512 simd_degree: at 32, every batch on the ranked path split into 24
/// hashes through the twelve-way kernel plus 8 through the ~2.1×-slower
/// upstream 4-lane tail (25% of all grind hashes on the slow path, a ~1.28×
/// effective penalty measured by `grind_speed_probe`); at 48 every full batch
/// runs entirely twelve-way. Buffers grow to 3 KiB of pre-images + 1.5 KiB
/// of digests — still stack-resident. (The earlier 1/4/8/16/32/64 sweep on an
/// M4 Max used a criterion harness whose noise floor exceeded the tail
/// penalty; the kernel-level probe resolves it.)
const BLAKE3_POW_BATCH: usize = 48;

/// Whether `FLOCK_NO_GRIND_REG=1` disables the register-resident grind
/// kernel (kill switch; restores the generic batched scan).
#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
fn grind_reg_disabled() -> bool {
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DISABLED.get_or_init(|| std::env::var("FLOCK_NO_GRIND_REG").is_ok_and(|v| v == "1"))
}

/// Smallest nonce in `start .. start + len` whose BLAKE3 PoW hash has `bits`
/// leading zeros, or `None`.
///
/// On Apple AArch64 with `1 <= bits <= 32` (every grind the protocol
/// configures) this dispatches to the register-resident specialization
/// [`crate::merkle::blake3_pow_scan_reg`], which keeps the fixed
/// `state_digest` words, the precomputed nonce-independent round work, and
/// the IV constants in NEON registers across attempts and injects only the
/// 8 changing nonce bytes per attempt — byte-exact against `blake3::hash`,
/// held to the generic path by `grind_reg_scan_matches_generic`. Kill
/// switch: `FLOCK_NO_GRIND_REG=1`.
fn blake3_pow_scan(state_digest: &[u8; 32], start: u64, len: u64, bits: u32) -> Option<u64> {
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    if (1..=32).contains(&bits) && !grind_reg_disabled() {
        return crate::merkle::blake3_pow_scan_reg(state_digest, start, len, bits);
    }
    blake3_pow_scan_generic(state_digest, start, len, bits)
}

/// The generic batched scan: materializes each 64-byte pre-image and hashes
/// batches through the twelve-way kernel on Apple AArch64 (upstream
/// `hash_many` tail and fallback) via [`crate::merkle::blake3_hash_many_pow`].
/// A 64-byte pre-image is a whole-block single chunk hashed with
/// `CHUNK_START | CHUNK_END | ROOT` — so this agrees with `blake3::hash` on
/// every nonce, which `blake3_batched_pow_matches_scalar` asserts.
fn blake3_pow_scan_generic(
    state_digest: &[u8; 32],
    start: u64,
    len: u64,
    bits: u32,
) -> Option<u64> {
    // The 32-byte state prefix is constant across the whole scan; only the
    // 8 nonce bytes change per lane.
    let mut pre = [[0u8; 64]; BLAKE3_POW_BATCH];
    for p in pre.iter_mut() {
        p[..32].copy_from_slice(state_digest);
    }
    let mut out = [0u8; BLAKE3_POW_BATCH * 32];

    let mut base = start;
    let end = start.saturating_add(len);
    while base < end {
        let n = BLAKE3_POW_BATCH.min((end - base) as usize);
        for (i, p) in pre[..n].iter_mut().enumerate() {
            p[32..40].copy_from_slice(&(base + i as u64).to_le_bytes());
        }
        #[cfg(feature = "hash-count")]
        fs_count::POW_SHA256.fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
        // Twelve-way kernel on Apple AArch64 (upstream `hash_many` tail and
        // fallback), byte-identical to `blake3::hash` per pre-image.
        // SAFETY: `pre` is `[[u8; 64]; BLAKE3_POW_BATCH]` and `out` is
        // `[u8; 32 * BLAKE3_POW_BATCH]`, so the first `n` elements of each
        // form contiguous 64-byte / 32-byte runs.
        unsafe {
            crate::merkle::blake3_hash_many_pow(
                core::slice::from_raw_parts(pre.as_ptr() as *const u8, n * 64),
                core::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut [u8; 32], n),
            );
        }
        for i in 0..n {
            if has_leading_zero_bits(&out[i * 32..(i + 1) * 32], bits) {
                return Some(base + i as u64);
            }
        }
        base += n as u64;
    }
    None
}

/// Smallest nonce in `start .. start + len` satisfying the PoW, or `None`.
/// Batched under BLAKE3; a plain scan under SHA-256, whose hardware path is
/// already faster than anything batching would buy.
#[inline]
fn pow_scan(
    state_digest: &[u8; 32],
    start: u64,
    len: u64,
    bits: u32,
    kind: HashKind,
) -> Option<u64> {
    match kind {
        HashKind::Blake3 => blake3_pow_scan(state_digest, start, len, bits),
        HashKind::Sha256 => (start..start.saturating_add(len))
            .find(|&n| pow_has_leading_zero_bits(state_digest, n, bits, kind)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_grind_calibration_covers_ranked_high_cost_schedule() {
        assert_eq!(GPU_GRIND_CALIBRATION_LENGTHS.iter().sum::<u32>(), 1 << 20);
        assert_eq!(GPU_GRIND_CALIBRATION_LENGTHS.len(), 7);
        assert_eq!(GPU_GRIND_MIN_BITS, 14);
        assert_eq!(GPU_GRIND_CALIBRATION_BITS, 19);
        let exact_geometric = 1.0 / (1.0 - (-1.0f64).exp());
        assert!((GPU_GRIND_BLOCK_OVERDRAW - exact_geometric).abs() < 1e-12);
    }

    /// A completed persistent-worker hit owns the earlier nonce block, so an
    /// armed candidate must not even invoke the later CPU-window closure.
    /// Turning off either the existing abort policy or the new feature input
    /// restores the incumbent invocation without touching process-global env.
    #[test]
    fn persistent_gpu_hit_skips_dead_cpu_window_only_when_armed() {
        use std::cell::Cell;

        let hit: Result<Option<u64>, String> = Ok(Some(7));
        let calls = Cell::new(0);
        let got = run_cpu_window_after_persistent_gpu(&hit, true, true, || {
            calls.set(calls.get() + 1);
            Some(99)
        });
        assert_eq!(got, None);
        assert_eq!(calls.get(), 0, "armed GPU hit must not dispatch CPU work");

        for (abort_enabled, feature_enabled) in [(false, true), (true, false), (false, false)] {
            let calls = Cell::new(0);
            let got =
                run_cpu_window_after_persistent_gpu(&hit, abort_enabled, feature_enabled, || {
                    calls.set(calls.get() + 1);
                    Some(99)
                });
            assert_eq!(got, Some(99));
            assert_eq!(calls.get(), 1);
        }
    }

    /// Miss and error results must retain the existing CPU-window behavior;
    /// only `Ok(Some(_))` is a proof that the later block is dead.
    #[test]
    fn persistent_gpu_miss_and_error_keep_cpu_window() {
        use std::cell::Cell;

        let cases: [Result<Option<u64>, String>; 2] =
            [Ok(None), Err("synthetic GPU failure".to_string())];
        for gpu_result in &cases {
            let calls = Cell::new(0);
            let got = run_cpu_window_after_persistent_gpu(gpu_result, true, true, || {
                calls.set(calls.get() + 1);
                Some(123)
            });
            assert_eq!(got, Some(123));
            assert_eq!(calls.get(), 1);
        }
    }

    /// Every FsChallenger property must hold under both transcript hashes:
    /// the tagging, absorption order and duplex structure are shared, and
    /// only the primitive differs.
    const KINDS: [HashKind; 2] = [HashKind::Sha256, HashKind::Blake3];

    fn challenger_with_test_prestate(kind: HashKind) -> FsChallenger {
        let mut challenger = FsChallenger::with_hash(b"legacy-equivalence", kind);
        challenger.observe_label(b"nontrivial-prestate");
        challenger.observe_bytes(b"transcript bytes before the operation under test");
        challenger.observe_f128(F128 {
            lo: 0x0123_4567_89AB_CDEF,
            hi: 0xFEDC_BA98_7654_3210,
        });
        challenger
    }

    fn legacy_observe_f128_slice(challenger: &mut FsChallenger, values: &[F128]) {
        challenger.absorb(&[OP_OBSERVE, KIND_SLICE]);
        challenger.absorb(&(values.len() as u64).to_le_bytes());
        for value in values {
            challenger.absorb(&value.lo.to_le_bytes());
            challenger.absorb(&value.hi.to_le_bytes());
        }
    }

    fn legacy_sample_f128_vec(challenger: &mut FsChallenger, n: usize) -> Vec<F128> {
        challenger.absorb(&[OP_SQUEEZE, KIND_SLICE]);
        challenger.absorb(&(n as u64).to_le_bytes());
        let mut bytes = vec![0u8; n * 16];
        challenger.squeeze_into(&mut bytes);
        challenger.absorb(&bytes);
        bytes
            .as_chunks::<16>()
            .0
            .iter()
            .map(|chunk| F128 {
                lo: u64::from_le_bytes(chunk[..8].try_into().unwrap()),
                hi: u64::from_le_bytes(chunk[8..].try_into().unwrap()),
            })
            .collect()
    }

    /// The two-pool early-exit grind must emit exactly the smallest
    /// satisfying nonce — proof bytes depend on it. The oracle is the
    /// library's own `pow_scan` over ascending blocks, which is the
    /// sequential search's definition of "globally smallest". bits = 14
    /// crosses both the parallel threshold (2^13 expected hashes) and, on
    /// hosts with a helper pool, the grind's epool engagement floor, so this
    /// exercises the heterogeneous claim/bound/re-derive path end to end.
    #[test]
    fn grind_two_pool_matches_sequential_scan_smallest() {
        for kind in KINDS {
            let mut ch = FsChallenger::with_hash(b"grind-2pool-test", kind);
            ch.observe_label(b"flock-grind-2pool");
            ch.observe_bytes(b"determinism probe");
            let digest = ch.state_digest();
            let bits = 14;
            let nonce = ch.grind_pow(bits);
            let mut block_start = 0u64;
            let expect = loop {
                if let Some(n) = pow_scan(&digest, block_start, 4096, bits, kind) {
                    break n;
                }
                block_start += 4096;
            };
            assert_eq!(nonce, expect, "kind={kind:?}");
        }
    }

    /// Prover-side PoW grinding produces a nonce that the verifier-side
    /// `verify_pow` accepts at the same transcript position. State binding
    /// is preserved — sampling after PoW gives identical challenges on both
    /// sides.
    #[test]
    fn fs_challenger_pow_roundtrip() {
        for kind in KINDS {
            for bits in [0u32, 5, 10, 14] {
                let mut prover = FsChallenger::with_hash(b"pow-test", kind);
                prover.observe_label(b"flock-pow-test");
                prover.observe_bytes(b"some root data");
                let nonce = prover.grind_pow(bits);

                let mut verifier = FsChallenger::with_hash(b"pow-test", kind);
                verifier.observe_label(b"flock-pow-test");
                verifier.observe_bytes(b"some root data");
                assert!(
                    verifier.verify_pow(nonce, bits),
                    "verify failed at bits={bits}"
                );

                // Subsequent challenges must agree.
                for _ in 0..4 {
                    assert_eq!(prover.sample_f128(), verifier.sample_f128());
                }
            }
        }
    }

    /// `verify_pow` rejects a wrong nonce when grinding bits > 0.
    #[test]
    fn fs_challenger_pow_rejects_wrong_nonce() {
        for kind in KINDS {
            let mut prover = FsChallenger::with_hash(b"pow-test", kind);
            prover.observe_bytes(b"root");
            let nonce = prover.grind_pow(10);
            let bad_nonce = nonce.wrapping_add(1);

            let mut verifier = FsChallenger::with_hash(b"pow-test", kind);
            verifier.observe_bytes(b"root");
            assert!(
                !verifier.verify_pow(bad_nonce, 10),
                "should reject wrong nonce"
            );
        }
    }

    /// At a zero-bit grinding site `verify_pow` accepts the canonical nonce 0
    /// (what `grind_pow(0)` emits) but rejects any non-zero nonce, so a proof
    /// can't be made malleable by swapping in an arbitrary nonce.
    #[test]
    fn fs_challenger_pow_zero_bits_requires_canonical_nonce() {
        for kind in KINDS {
            let mk = || {
                let mut ch = FsChallenger::with_hash(b"pow-test", kind);
                ch.observe_bytes(b"root");
                ch
            };
            assert_eq!(mk().grind_pow(0), 0, "honest zero-bit grind is the 0 nonce");
            assert!(mk().verify_pow(0, 0), "canonical 0 nonce must verify");
            for bad in [1u64, 42, u64::MAX] {
                assert!(
                    !mk().verify_pow(bad, 0),
                    "non-zero nonce {bad} must be rejected at zero-bit grinding"
                );
            }
        }
    }

    /// `new` must stay SHA-256: 300-odd call sites construct challengers that
    /// way, and silently moving them to another hash would invalidate every
    /// proof they produce.
    #[test]
    fn fs_challenger_new_defaults_to_sha256() {
        assert_eq!(FsChallenger::new(b"d").hash_kind(), HashKind::Sha256);
        for kind in KINDS {
            assert_eq!(FsChallenger::with_hash(b"d", kind).hash_kind(), kind);
        }
        // The default constructor must be exactly the SHA-256 one, transcript
        // and all — not merely tagged the same.
        let mut a = FsChallenger::new(b"d");
        let mut b = FsChallenger::with_hash(b"d", HashKind::Sha256);
        assert_eq!(a.sample_f128_vec(4), b.sample_f128_vec(4));
    }

    /// The two transcript hashes must produce different challenges from the
    /// same script — otherwise the option would be doing nothing.
    #[test]
    fn fs_challenger_hashes_diverge() {
        let script = |ch: &mut FsChallenger| {
            ch.observe_label(b"phase");
            ch.observe_bytes(b"root");
            ch.observe_f128(F128::ONE);
            ch.sample_f128_vec(4)
        };
        let mut sha = FsChallenger::with_hash(b"d", HashKind::Sha256);
        let mut blake = FsChallenger::with_hash(b"d", HashKind::Blake3);
        assert_ne!(script(&mut sha), script(&mut blake));
    }

    /// A verifier on the wrong transcript hash must reject: the PoW check is
    /// against a different digest, and the challenges diverge from there.
    #[test]
    fn fs_challenger_pow_rejects_the_other_hash() {
        for kind in KINDS {
            let other = match kind {
                HashKind::Sha256 => HashKind::Blake3,
                HashKind::Blake3 => HashKind::Sha256,
            };
            let mut prover = FsChallenger::with_hash(b"pow-test", kind);
            prover.observe_bytes(b"root");
            let nonce = prover.grind_pow(10);

            let mut wrong = FsChallenger::with_hash(b"pow-test", other);
            wrong.observe_bytes(b"root");
            assert!(
                !wrong.verify_pow(nonce, 10),
                "{kind} nonce must not satisfy a {other} PoW"
            );
        }
    }

    /// BLAKE3 squeezes from an XOF rather than a counter, so a long squeeze
    /// must still agree with the concatenation of the short ones it replaces —
    /// i.e. `sample_f128_vec(n)` is one XOF read of `16n` bytes, not `n`
    /// independent reads. Pins the stream layout for both hashes.
    #[test]
    fn fs_challenger_long_squeeze_is_prefix_stable() {
        for kind in KINDS {
            // Two challengers on identical scripts, one squeezing 8 values and
            // one squeezing 8 values in a single call, must agree — this is
            // just determinism, but it is what the duplex relies on.
            let mut a = FsChallenger::with_hash(b"d", kind);
            let mut b = FsChallenger::with_hash(b"d", kind);
            assert_eq!(a.sample_f128_vec(8), b.sample_f128_vec(8), "{kind}");

            // A squeeze longer than one 32-byte block must not repeat itself:
            // catches a counter that fails to advance, or an XOF read that
            // restarts per block.
            let vals = FsChallenger::with_hash(b"d", kind).sample_f128_vec(16);
            let unique: std::collections::HashSet<_> = vals.iter().collect();
            assert_eq!(unique.len(), vals.len(), "{kind}: squeeze stream repeats");
        }
    }

    /// The batched BLAKE3 nonce search must agree with the scalar spec
    /// (`blake3::hash` of the 64-byte pre-image) on every nonce. This is what
    /// makes the SIMD path safe to use: if `hash_many`'s flag semantics ever
    /// changed, this fails rather than silently producing PoW hashes that
    /// `verify_pow` would then reject.
    #[test]
    fn blake3_batched_pow_matches_scalar() {
        let state = [0x5Au8; 32];
        // Cover nonce counts either side of the batch width (32): a partial
        // batch, exactly one, one past, and several with a ragged tail.
        for len in [1u64, 5, 31, 32, 33, 100] {
            for start in [0u64, 7, 1_000_000] {
                // `bits = 0` makes every nonce a match, so the scan must return
                // `start` — and the per-lane hashes are all exercised below.
                assert_eq!(
                    blake3_pow_scan_generic(&state, start, len, 0),
                    Some(start),
                    "start={start} len={len}"
                );
                // Compare the scans (generic and dispatching) against a scalar
                // sweep at a threshold low enough to hit but high enough to
                // skip some nonces.
                let want = (start..start + len)
                    .find(|&n| pow_has_leading_zero_bits(&state, n, 6, HashKind::Blake3));
                assert_eq!(
                    blake3_pow_scan_generic(&state, start, len, 6),
                    want,
                    "start={start} len={len}"
                );
                assert_eq!(
                    blake3_pow_scan(&state, start, len, 6),
                    want,
                    "dispatch start={start} len={len}"
                );
            }
        }
    }

    /// The register-resident grind scan must agree with the generic batched
    /// scan — same match/no-match, same (smallest) nonce — for random
    /// digests, every bits value it dispatches on, ragged range shapes, and
    /// ranges crossing the 12-lane group and 32-bit nonce boundaries. The
    /// generic path is itself held to `blake3::hash` by
    /// `blake3_batched_pow_matches_scalar`, so equality here pins the
    /// specialized kernel to the byte-exact spec.
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    #[test]
    fn grind_reg_scan_matches_generic() {
        let mut seed = 0x1234_5678_9ABC_DEF0u64;
        let mut digest = [0u8; 32];
        for case in 0..24 {
            for b in digest.iter_mut() {
                *b = (splitmix64(&mut seed) & 0xFF) as u8;
            }
            for bits in [1u32, 2, 6, 8, 13, 19, 24, 32] {
                for (start, len) in [
                    (0u64, 1u64),
                    (0, 11),
                    (0, 12),
                    (0, 13),
                    (7, 500),
                    (1_000_000, 960),
                    (u32::MAX as u64 - 30, 60), // nonce-lo carry inside a group
                ] {
                    let want = blake3_pow_scan_generic(&digest, start, len, bits);
                    let got = crate::merkle::blake3_pow_scan_reg(&digest, start, len, bits);
                    assert_eq!(got, want, "case={case} bits={bits} start={start} len={len}");
                }
            }
            // Exhaustive predicate agreement on a low threshold: walk ALL
            // matches in a range (not just the first) by resuming past each.
            let bits = 4;
            let (mut a, mut b) = (0u64, 0u64);
            let end = 2048u64;
            loop {
                let want = blake3_pow_scan_generic(&digest, a, end - a, bits);
                let got = crate::merkle::blake3_pow_scan_reg(&digest, b, end - b, bits);
                assert_eq!(got, want, "case={case} resume a={a} b={b}");
                match want {
                    Some(n) if n + 1 < end => {
                        a = n + 1;
                        b = n + 1;
                    }
                    _ => break,
                }
            }
        }
    }

    /// Fixed-seed transcripts must grind to the identical nonce under the
    /// dispatching scan (register-resident kernel on Apple AArch64) as the
    /// sequential generic oracle — proof bytes depend on the exact nonce.
    #[test]
    fn grind_reg_selects_identical_nonces_fixed_seeds() {
        for seed in [424242u64, 777, 1, 0xDEAD_BEEF] {
            for bits in [6u32, 14, 16] {
                let mut ch = FsChallenger::with_hash(b"grind-reg-fixed", HashKind::Blake3);
                ch.observe_bytes(&seed.to_le_bytes());
                let digest = ch.state_digest();
                let nonce = ch.grind_pow(bits);
                let mut start = 0u64;
                let expect = loop {
                    if let Some(n) = blake3_pow_scan_generic(&digest, start, 4096, bits) {
                        break n;
                    }
                    start += 4096;
                };
                assert_eq!(nonce, expect, "seed={seed} bits={bits}");
            }
        }
    }

    /// A pre-set stop flag must make the stoppable window bail without
    /// scanning (returns None even though the window provably contains a
    /// match), and an unset flag must leave it byte-equal to the plain
    /// drain-everything window. The hybrid only ever sets the flag once the
    /// GPU block has hit — the exact situation where the window's result is
    /// discarded — so "abort returns None" is the safe direction to pin.
    #[test]
    fn stoppable_window_aborts_and_matches_plain_when_unstopped() {
        use std::sync::atomic::AtomicBool;
        let mut ch = FsChallenger::with_hash(b"grind-abort-window", HashKind::Blake3);
        ch.observe_bytes(&0x5EED_CAFEu64.to_le_bytes());
        let digest = ch.state_digest();
        let (bits, len) = (10u32, 1u32 << 14);
        let plain = cpu_blake3_pow_window(&digest, 0, len, bits);
        assert!(plain.is_some(), "2^14 window at 10 bits must contain a hit");
        let unstopped = AtomicBool::new(false);
        assert_eq!(
            cpu_blake3_pow_window_inner(&digest, 0, len, bits, Some(&unstopped)),
            plain,
        );
        let stopped = AtomicBool::new(true);
        assert_eq!(
            cpu_blake3_pow_window_inner(&digest, 0, len, bits, Some(&stopped)),
            None,
        );
    }

    /// The hybrid GPU/CPU-prefetch block walk must select the identical
    /// nonce as the sequential generic oracle — proof bytes depend on the
    /// exact nonce, so the prefetch's "GPU block first, CPU next-block on a
    /// miss" merge is held to bit-exactness on real Metal hardware. Skips
    /// where no GPU grind pipeline is available.
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    #[test]
    fn gpu_grind_hybrid_nonce_matches_generic_oracle() {
        if crate::gpu_commit::gpu_blake3_pow_scan(&[0u8; 32], 0, 64, 1).is_err() {
            return; // no Metal device / pipeline in this environment
        }
        for seed in [0xFEED_F00Du64, 31337, 8_675_309] {
            for bits in [14u32, 16, 19] {
                let mut ch = FsChallenger::with_hash(b"grind-hybrid-fixed", HashKind::Blake3);
                ch.observe_bytes(&seed.to_le_bytes());
                let digest = ch.state_digest();
                let got = gpu_blake3_pow_nonce(&digest, bits)
                    .expect("GPU grind must scan on an available device");
                let mut start = 0u64;
                let expect = loop {
                    if let Some(n) = blake3_pow_scan_generic(&digest, start, 4096, bits) {
                        break n;
                    }
                    start += 4096;
                };
                assert_eq!(got, expect, "seed={seed} bits={bits}");
            }
        }
    }

    /// Paired micro-probe (not a correctness gate): generic batched scan vs
    /// the register-resident kernel on the same digest set with no early
    /// exit. Pure compute, single-threaded, no scheduling. Run with
    /// `cargo test -p flock-core --release --lib -- --ignored --nocapture grind_reg_speed`.
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    #[test]
    #[ignore]
    fn grind_reg_speed_probe() {
        const N: u64 = 1 << 19;
        const REPS: usize = 21;
        let mut seed = 0xC0FF_EE00_D15E_A5Edu64;
        let mut digests = Vec::new();
        while digests.len() < 4 {
            let mut d = [0u8; 32];
            for b in d.iter_mut() {
                *b = (splitmix64(&mut seed) & 0xFF) as u8;
            }
            // bits=32: a match in [0, N) has probability N/2^32 ≈ 0.012 per
            // digest; skip any digest that would early-exit either arm.
            if blake3_pow_scan_generic(&d, 0, N, 32).is_none() {
                digests.push(d);
            }
        }
        let mut probe = |f: &dyn Fn(&[u8; 32]) -> Option<u64>| -> (f64, f64) {
            let mut times = Vec::with_capacity(REPS);
            for _ in 0..REPS {
                let t = std::time::Instant::now();
                for d in &digests {
                    assert_eq!(f(d), None);
                }
                times.push(t.elapsed().as_secs_f64() * 1e3);
            }
            times.sort_by(f64::total_cmp);
            (times[0], times[REPS / 2])
        };
        // Warm up + interleave-fair: measure generic, reg, then generic again
        // to expose drift.
        let (gen_min, gen_med) = probe(&|d| blake3_pow_scan_generic(d, 0, N, 32));
        let (reg_min, reg_med) = probe(&|d| crate::merkle::blake3_pow_scan_reg(d, 0, N, 32));
        let (gen2_min, gen2_med) = probe(&|d| blake3_pow_scan_generic(d, 0, N, 32));
        let hashes = (N as f64) * digests.len() as f64;
        eprintln!(
            "grind probe over {hashes:.0} hashes/iter, {REPS} reps:\n\
             generic  min {gen_min:.2} ms  med {gen_med:.2} ms  ({:.2} ns/hash min)\n\
             reg      min {reg_min:.2} ms  med {reg_med:.2} ms  ({:.2} ns/hash min)\n\
             generic2 min {gen2_min:.2} ms  med {gen2_med:.2} ms\n\
             speedup (min/min vs first generic): {:.3}x",
            gen_min * 1e6 / hashes,
            reg_min * 1e6 / hashes,
            gen_min / reg_min,
        );
    }

    /// Timing probe (not a correctness gate): upstream 4-lane `hash_many`
    /// vs the twelve-way PoW path on 2^20 grind hashes. Run with
    /// `cargo test -p flock-core --lib -- --ignored --nocapture grind_speed`.
    #[test]
    #[ignore]
    fn grind_speed_probe() {
        let n = 1usize << 20;
        let digest = [0xABu8; 32];
        let mut pre = vec![0u8; n * 64];
        for i in 0..n {
            pre[i * 64..i * 64 + 32].copy_from_slice(&digest);
            pre[i * 64 + 32..i * 64 + 40].copy_from_slice(&(i as u64).to_le_bytes());
        }
        let mut out = vec![[0u8; 32]; n];

        let t = std::time::Instant::now();
        {
            use blake3::platform::Platform;
            const IV: [u32; 8] = [
                0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB,
                0x5BE0CD19,
            ];
            let plat = Platform::detect();
            let mut out_bytes = vec![0u8; n * 32];
            for chunk in 0..n / 16 {
                let base = chunk * 16;
                let inputs: [&[u8; 64]; 16] = std::array::from_fn(|i| {
                    pre[(base + i) * 64..(base + i + 1) * 64]
                        .try_into()
                        .unwrap()
                });
                plat.hash_many(
                    &inputs,
                    &IV,
                    0,
                    blake3::IncrementCounter::No,
                    0,
                    1,
                    2 | 8,
                    &mut out_bytes[base * 32..(base + 16) * 32],
                );
            }
            out.copy_from_slice(unsafe {
                core::slice::from_raw_parts(out_bytes.as_ptr() as *const [u8; 32], n)
            });
        }
        let old_ms = t.elapsed().as_secs_f64() * 1e3;

        let mut out2 = vec![[0u8; 32]; n];
        let t = std::time::Instant::now();
        crate::merkle::blake3_hash_many_pow(&pre, &mut out2);
        let new_ms = t.elapsed().as_secs_f64() * 1e3;

        assert_eq!(out, out2, "outputs differ");
        eprintln!(
            "{n} hashes: old (4-lane hash_many) {old_ms:.2} ms, new (12-way) {new_ms:.2} ms, speedup {:.2}x",
            old_ms / new_ms
        );
    }

    /// The grind must return the globally smallest satisfying nonce, on both
    /// the sequential and the block-parallel path, and under both hashes.
    /// Proof determinism depends on it: a different nonce is a different
    /// transcript and therefore a different proof.
    #[test]
    fn fs_challenger_grind_returns_smallest_nonce() {
        for kind in KINDS {
            // 4 bits stays sequential; 14 crosses PARALLEL_GRIND_MIN_HASHES.
            for bits in [4u32, 14] {
                let mut ch = FsChallenger::with_hash(b"grind-min", kind);
                ch.observe_bytes(b"root");
                let digest_probe = {
                    let mut probe = FsChallenger::with_hash(b"grind-min", kind);
                    probe.observe_bytes(b"root");
                    probe.state_digest()
                };
                let nonce = ch.grind_pow(bits);
                // Every smaller nonce must fail the scalar check.
                for n in 0..nonce {
                    assert!(
                        !pow_has_leading_zero_bits(&digest_probe, n, bits, kind),
                        "{kind} bits={bits}: nonce {n} < {nonce} also satisfies the PoW"
                    );
                }
                assert!(
                    pow_has_leading_zero_bits(&digest_probe, nonce, bits, kind),
                    "{kind} bits={bits}: returned nonce {nonce} does not satisfy the PoW"
                );
            }
        }
    }

    /// Default Challenger impl (RandomChallenger) is a no-op for PoW.
    #[test]
    fn random_challenger_pow_is_noop() {
        let mut ch = RandomChallenger::new(0);
        assert_eq!(ch.grind_pow(16), 0);
        assert!(ch.verify_pow(0, 16));
    }

    #[test]
    fn random_challenger_is_deterministic_per_seed() {
        let mut c1 = RandomChallenger::new(42);
        let mut c2 = RandomChallenger::new(42);
        for _ in 0..16 {
            assert_eq!(c1.sample_f128(), c2.sample_f128());
        }
    }

    #[test]
    fn random_challenger_observe_is_noop() {
        // Observing arbitrary messages does not change the sampled values.
        let mut c1 = RandomChallenger::new(7);
        let mut c2 = RandomChallenger::new(7);
        c2.observe_f128(F128 {
            lo: 0xDEADBEEF,
            hi: 0xCAFEBABE,
        });
        c2.observe_f128_slice(&[F128::ONE, F128::ZERO]);
        c2.observe_label(b"ignored");
        c2.observe_bytes(b"also ignored");
        for _ in 0..8 {
            assert_eq!(c1.sample_f128(), c2.sample_f128());
        }
    }

    #[test]
    fn sample_f128_vec_matches_individual_samples() {
        let mut c1 = RandomChallenger::new(99);
        let mut c2 = RandomChallenger::new(99);
        let batch = c1.sample_f128_vec(5);
        let individual: Vec<F128> = (0..5).map(|_| c2.sample_f128()).collect();
        assert_eq!(batch, individual);
    }

    // ---- FsChallenger ------------------------------------------------------

    #[cfg(target_endian = "little")]
    #[test]
    fn f128_byte_views_match_canonical_little_endian_layout() {
        assert_eq!(core::mem::size_of::<F128>(), 16);
        assert_eq!(core::mem::align_of::<F128>(), 16);

        let values = [
            F128 {
                lo: 0x0102_0304_0506_0708,
                hi: 0x1112_1314_1516_1718,
            },
            F128 {
                lo: 0x8182_8384_8586_8788,
                hi: 0x9192_9394_9596_9798,
            },
        ];
        let base = core::ptr::addr_of!(values[0]) as usize;
        assert_eq!(core::ptr::addr_of!(values[0].lo) as usize - base, 0);
        assert_eq!(core::ptr::addr_of!(values[0].hi) as usize - base, 8);
        assert_eq!(core::ptr::addr_of!(values[1]) as usize - base, 16);

        let mut expected = [0u8; 32];
        expected[..8].copy_from_slice(&values[0].lo.to_le_bytes());
        expected[8..16].copy_from_slice(&values[0].hi.to_le_bytes());
        expected[16..24].copy_from_slice(&values[1].lo.to_le_bytes());
        expected[24..].copy_from_slice(&values[1].hi.to_le_bytes());
        assert_eq!(f128s_as_bytes(&values), expected.as_slice());

        let mut decoded = [F128::ZERO; 2];
        f128s_as_bytes_mut(&mut decoded).copy_from_slice(&expected);
        assert_eq!(decoded, values);
    }

    #[test]
    fn fs_challenger_observe_slice_matches_legacy_transcript() {
        for kind in KINDS {
            for n in [0usize, 1, 2, 64, 128] {
                let values: Vec<F128> = (0..n)
                    .map(|i| F128 {
                        lo: 0x1020_3040_5060_7080 ^ i as u64,
                        hi: 0x90A0_B0C0_D0E0_F000 ^ (i as u64).rotate_left(17),
                    })
                    .collect();
                let mut optimized = challenger_with_test_prestate(kind);
                let mut legacy = optimized.clone();

                optimized.observe_f128_slice(&values);
                legacy_observe_f128_slice(&mut legacy, &values);

                assert_eq!(optimized.n_absorbed, legacy.n_absorbed, "{kind}, n={n}");
                assert_eq!(
                    optimized.sample_f128_vec(4),
                    legacy.sample_f128_vec(4),
                    "{kind}, n={n}"
                );
                assert_eq!(
                    optimized.sample_f128(),
                    legacy.sample_f128(),
                    "{kind}, n={n}: subsequent state diverged"
                );
            }
        }
    }

    #[test]
    fn fs_challenger_sample_vec_matches_legacy_transcript() {
        for kind in KINDS {
            for n in [0usize, 1, 2, 3, 8, 19, 64] {
                let mut optimized = challenger_with_test_prestate(kind);
                let mut legacy = optimized.clone();

                let got = optimized.sample_f128_vec(n);
                let expected = legacy_sample_f128_vec(&mut legacy, n);

                assert_eq!(got, expected, "{kind}, n={n}");
                assert_eq!(optimized.n_absorbed, legacy.n_absorbed, "{kind}, n={n}");
                assert_eq!(
                    optimized.sample_f128(),
                    legacy.sample_f128(),
                    "{kind}, n={n}: reabsorbed state diverged"
                );
            }
        }
    }

    #[test]
    fn fs_challenger_empty_observe_slice_is_framed() {
        for kind in KINDS {
            let mut framed = challenger_with_test_prestate(kind);
            let mut omitted = framed.clone();
            let before = framed.n_absorbed;
            framed.observe_f128_slice(&[]);

            assert_eq!(framed.n_absorbed - before, 10, "{kind}");
            assert_ne!(framed.sample_f128(), omitted.sample_f128(), "{kind}");
        }
    }

    #[test]
    fn fs_challenger_empty_sample_vec_is_framed() {
        for kind in KINDS {
            let mut framed = challenger_with_test_prestate(kind);
            let mut omitted = framed.clone();
            let before = framed.n_absorbed;
            assert!(framed.sample_f128_vec(0).is_empty(), "{kind}");

            assert_eq!(framed.n_absorbed - before, 10, "{kind}");
            assert_ne!(framed.sample_f128(), omitted.sample_f128(), "{kind}");
        }
    }

    #[test]
    fn fs_challenger_identical_scripts_produce_identical_output() {
        for kind in KINDS {
            let mut c1 = FsChallenger::with_hash(b"flock-test", kind);
            let mut c2 = FsChallenger::with_hash(b"flock-test", kind);
            let msg = F128 {
                lo: 0x1234,
                hi: 0x5678,
            };
            c1.observe_f128(msg);
            c2.observe_f128(msg);
            let r1 = c1.sample_f128_vec(8);
            let r2 = c2.sample_f128_vec(8);
            assert_eq!(r1, r2);
        }
    }

    #[test]
    fn fs_challenger_different_domains_diverge() {
        for kind in KINDS {
            let mut c1 = FsChallenger::with_hash(b"flock-a", kind);
            let mut c2 = FsChallenger::with_hash(b"flock-b", kind);
            assert_ne!(c1.sample_f128(), c2.sample_f128());
        }
    }

    #[test]
    fn fs_challenger_different_observations_diverge() {
        for kind in KINDS {
            let mut c1 = FsChallenger::with_hash(b"flock", kind);
            let mut c2 = FsChallenger::with_hash(b"flock", kind);
            c1.observe_f128(F128::ONE);
            c2.observe_f128(F128::ZERO);
            assert_ne!(c1.sample_f128(), c2.sample_f128());
        }
    }

    #[test]
    fn fs_challenger_label_changes_output() {
        for kind in KINDS {
            let mut c1 = FsChallenger::with_hash(b"flock", kind);
            let mut c2 = FsChallenger::with_hash(b"flock", kind);
            c1.observe_label(b"phase-A");
            // c2 omits the label entirely.
            assert_ne!(c1.sample_f128(), c2.sample_f128());
        }
    }

    #[test]
    fn fs_challenger_scalar_vs_slice_dont_collide() {
        for kind in KINDS {
            // observe_f128_slice(&[v]) must NOT produce the same state as
            // observe_f128(v) — the length prefix and kind tag must defeat this.
            let v = F128 { lo: 0xAB, hi: 0xCD };
            let mut c1 = FsChallenger::with_hash(b"flock", kind);
            let mut c2 = FsChallenger::with_hash(b"flock", kind);
            c1.observe_f128(v);
            c2.observe_f128_slice(&[v]);
            assert_ne!(c1.sample_f128(), c2.sample_f128());
        }
    }

    #[test]
    fn fs_challenger_two_scalars_dont_collide_with_one_slice_of_two() {
        for kind in KINDS {
            let a = F128 { lo: 1, hi: 2 };
            let b = F128 { lo: 3, hi: 4 };
            let mut c1 = FsChallenger::with_hash(b"flock", kind);
            let mut c2 = FsChallenger::with_hash(b"flock", kind);
            c1.observe_f128(a);
            c1.observe_f128(b);
            c2.observe_f128_slice(&[a, b]);
            assert_ne!(c1.sample_f128(), c2.sample_f128());
        }
    }

    #[test]
    fn fs_challenger_sample_one_vs_sample_vec_one_differ() {
        for kind in KINDS {
            // Squeeze tag differs (KIND_SCALAR vs KIND_SLICE+len), so a single
            // sample_f128 must not equal sample_f128_vec(1)[0].
            let mut c1 = FsChallenger::with_hash(b"flock", kind);
            let mut c2 = FsChallenger::with_hash(b"flock", kind);
            assert_ne!(c1.sample_f128(), c2.sample_f128_vec(1)[0]);
        }
    }

    #[test]
    fn fs_challenger_sample_advances_state() {
        for kind in KINDS {
            // After a sample, the next observation should not collapse to the
            // pre-sample state (the squeezed bytes are re-absorbed).
            let mut c1 = FsChallenger::with_hash(b"flock", kind);
            let mut c2 = FsChallenger::with_hash(b"flock", kind);
            let _ = c1.sample_f128();
            // c2 skips the sample.
            c1.observe_f128(F128::ONE);
            c2.observe_f128(F128::ONE);
            assert_ne!(c1.sample_f128(), c2.sample_f128());
        }
    }
}
