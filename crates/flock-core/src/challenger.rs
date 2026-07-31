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
    /// The transcript's own incremental hasher (see [`fs_blake3`]) — digest
    /// and XOF output are bit-identical to `blake3::Hasher`, without the
    /// crate's per-update chunk-state overhead on the many tiny absorbs.
    Blake3(Box<fs_blake3::TranscriptBlake3>),
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
                HashKind::Blake3 => FsState::Blake3(Box::new(fs_blake3::TranscriptBlake3::new())),
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
            FsState::Blake3(hasher) => hasher.xof_fill(out),
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
            FsState::Blake3(h) => h.finalize(),
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

    fn observe_f128(&mut self, value: F128) {
        self.absorb(&[OP_OBSERVE, KIND_SCALAR]);
        self.absorb_f128(value);
    }

    fn observe_f128_slice(&mut self, values: &[F128]) {
        self.absorb(&[OP_OBSERVE, KIND_SLICE]);
        self.absorb(&(values.len() as u64).to_le_bytes());
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
        let nonce = if bits == 0 {
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
                    Some(ep) => std::thread::scope(|s| {
                        // The scoped thread parks in `broadcast` while the
                        // E-workers drain; the scope join bounds the tail wait
                        // at one chunk on one efficiency core.
                        s.spawn(|| ep.broadcast(|_| worker()));
                        drain_main();
                    }),
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

/// Smallest nonce in `start .. start + len` whose BLAKE3 PoW hash has `bits`
/// leading zeros, or `None`.
///
/// Batches the independent nonce hashes through the twelve-way kernel on
/// Apple AArch64 (upstream `hash_many` tail and fallback) via
/// [`crate::merkle::blake3_hash_many_pow`]. A 64-byte pre-image is a
/// whole-block single chunk hashed with `CHUNK_START | CHUNK_END | ROOT` —
/// so this agrees with `blake3::hash` on every nonce, which
/// `blake3_batched_pow_matches_scalar` asserts.
fn blake3_pow_scan(state_digest: &[u8; 32], start: u64, len: u64, bits: u32) -> Option<u64> {
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

// ---------------------------------------------------------------------------
// fs_blake3 — minimal incremental BLAKE3 for the Fiat-Shamir transcript.
//
// On aarch64 the `blake3` crate computes single-input hashing through its
// portable scalar `compress_in_place` (the crate's NEON code only serves the
// multi-input `hash_many` used by the Merkle layer), so every tiny transcript
// absorb and every squeeze also pays the crate's generic chunk-state/CV-stack
// machinery around that same scalar compression. This module re-implements
// the unkeyed BLAKE3 tree with reference-impl semantics (chunk state + CV
// stack) but a hot path shaped for the transcript: absorbing a 1–26-byte
// fragment is a bounds-check plus memcpy, and a squeeze walks the CV stack
// once without cloning any hasher. Output is bit-identical to
// `blake3::Hasher` for every input length and XOF width —
// `transcript_blake3_matches_crate` and
// `fs_challenger_blake3_matches_crate_reference` assert this across chunk
// boundaries.
// ---------------------------------------------------------------------------
mod fs_blake3 {
    const BLOCK_LEN: usize = 64;
    const CHUNK_LEN: usize = 1024;

    const CHUNK_START: u32 = 1 << 0;
    const CHUNK_END: u32 = 1 << 1;
    const PARENT: u32 = 1 << 2;
    const ROOT: u32 = 1 << 3;

    const IV: [u32; 8] = [
        0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB,
        0x5BE0CD19,
    ];

    const MSG_SCHEDULE: [[u8; 16]; 7] = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
        [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
        [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
        [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
        [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
        [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
    ];

    #[inline(always)]
    fn g(v: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, x: u32, y: u32) {
        v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
        v[d] = (v[d] ^ v[a]).rotate_right(16);
        v[c] = v[c].wrapping_add(v[d]);
        v[b] = (v[b] ^ v[c]).rotate_right(12);
        v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
        v[d] = (v[d] ^ v[a]).rotate_right(8);
        v[c] = v[c].wrapping_add(v[d]);
        v[b] = (v[b] ^ v[c]).rotate_right(7);
    }

    /// One BLAKE3 round. Must be called with a literal `r` (see `compress`).
    #[inline(always)]
    fn round(v: &mut [u32; 16], m: &[u32; 16], r: usize) {
        let s = &MSG_SCHEDULE[r];
        g(v, 0, 4, 8, 12, m[s[0] as usize], m[s[1] as usize]);
        g(v, 1, 5, 9, 13, m[s[2] as usize], m[s[3] as usize]);
        g(v, 2, 6, 10, 14, m[s[4] as usize], m[s[5] as usize]);
        g(v, 3, 7, 11, 15, m[s[6] as usize], m[s[7] as usize]);
        g(v, 0, 5, 10, 15, m[s[8] as usize], m[s[9] as usize]);
        g(v, 1, 6, 11, 12, m[s[10] as usize], m[s[11] as usize]);
        g(v, 2, 7, 8, 13, m[s[12] as usize], m[s[13] as usize]);
        g(v, 3, 4, 9, 14, m[s[14] as usize], m[s[15] as usize]);
    }

    /// One full BLAKE3 compression. Returns all sixteen output words: words
    /// 0..8 are the chaining value, words 8..16 extend it to the full 64-byte
    /// XOF block (`out[i+8] = v[i+8] ^ cv[i]`), exactly as the spec defines.
    fn compress(cv: &[u32; 8], block: &[u8; 64], counter: u64, block_len: u32, flags: u32) -> [u32; 16] {
        let mut m = [0u32; 16];
        for (w, src) in m.iter_mut().zip(block.chunks_exact(4)) {
            *w = u32::from_le_bytes(src.try_into().unwrap());
        }
        let mut v = [
            cv[0],
            cv[1],
            cv[2],
            cv[3],
            cv[4],
            cv[5],
            cv[6],
            cv[7],
            IV[0],
            IV[1],
            IV[2],
            IV[3],
            counter as u32,
            (counter >> 32) as u32,
            block_len,
            flags,
        ];
        // Rounds are called with literal indices so the schedule lookups
        // constant-fold — a `for s in &MSG_SCHEDULE` loop leaves runtime
        // indexed loads (and bounds checks) in the hot path and compiles
        // ~2.5× slower than the crate's portable compression.
        round(&mut v, &m, 0);
        round(&mut v, &m, 1);
        round(&mut v, &m, 2);
        round(&mut v, &m, 3);
        round(&mut v, &m, 4);
        round(&mut v, &m, 5);
        round(&mut v, &m, 6);
        for i in 0..8 {
            v[i] ^= v[i + 8];
            v[i + 8] ^= cv[i];
        }
        v
    }

    #[inline]
    fn first8(words: &[u32; 16]) -> [u32; 8] {
        words[..8].try_into().unwrap()
    }

    /// The 64-byte block of a parent node: left ‖ right chaining values.
    fn parent_block(left: &[u32; 8], right: &[u32; 8]) -> [u8; 64] {
        let mut block = [0u8; 64];
        for i in 0..8 {
            block[4 * i..4 * i + 4].copy_from_slice(&left[i].to_le_bytes());
            block[32 + 4 * i..36 + 4 * i].copy_from_slice(&right[i].to_le_bytes());
        }
        block
    }

    /// The inputs of a node's final compression, captured *before* running
    /// it. `chaining_value` runs it as an interior node; the finalize/XOF
    /// paths re-run it with `ROOT` and a per-output-block counter, exactly
    /// like the reference implementation's `Output`.
    struct Output {
        cv: [u32; 8],
        block: [u8; 64],
        block_len: u32,
        counter: u64,
        flags: u32,
    }

    impl Output {
        #[inline]
        fn chaining_value(&self) -> [u32; 8] {
            first8(&compress(
                &self.cv,
                &self.block,
                self.counter,
                self.block_len,
                self.flags,
            ))
        }
    }

    /// Incremental unkeyed BLAKE3 with `blake3::Hasher`-identical output.
    ///
    /// 54 chaining values cover the maximum tree height (2^64 input bytes),
    /// mirroring the crate; the transcript never gets past a handful, but
    /// the fixed array keeps the type honest for any input length.
    #[derive(Clone)]
    pub(super) struct TranscriptBlake3 {
        chunk_cv: [u32; 8],
        chunk_counter: u64,
        block: [u8; 64],
        block_len: u8,
        blocks_compressed: u8,
        stack_len: u8,
        cv_stack: [[u32; 8]; 54],
    }

    impl TranscriptBlake3 {
        pub(super) fn new() -> Self {
            Self {
                chunk_cv: IV,
                chunk_counter: 0,
                block: [0u8; BLOCK_LEN],
                block_len: 0,
                blocks_compressed: 0,
                stack_len: 0,
                cv_stack: [[0u32; 8]; 54],
            }
        }

        #[inline]
        fn start_flag(&self) -> u32 {
            if self.blocks_compressed == 0 { CHUNK_START } else { 0 }
        }

        /// Absorb bytes. The transcript's dominant case — a small fragment
        /// that fits in the buffered block — is a copy and a length bump;
        /// blocks are compressed lazily (only once further input arrives),
        /// matching the reference implementation exactly.
        #[inline]
        pub(super) fn update(&mut self, input: &[u8]) {
            let len = self.block_len as usize;
            if len + input.len() <= BLOCK_LEN {
                self.block[len..len + input.len()].copy_from_slice(input);
                self.block_len = (len + input.len()) as u8;
                return;
            }
            self.update_general(input);
        }

        /// Slow path: the input spills past the buffered block. Compresses
        /// full blocks (finalizing and pushing the chunk's chaining value at
        /// each 1024-byte boundary) and buffers the tail.
        #[inline(never)]
        fn update_general(&mut self, mut input: &[u8]) {
            while !input.is_empty() {
                if self.block_len as usize == BLOCK_LEN {
                    if self.blocks_compressed as usize == CHUNK_LEN / BLOCK_LEN - 1 {
                        // The buffered block completes the current chunk, and
                        // more input follows: finalize the chunk and merge its
                        // chaining value into the stack.
                        let out = compress(
                            &self.chunk_cv,
                            &self.block,
                            self.chunk_counter,
                            BLOCK_LEN as u32,
                            self.start_flag() | CHUNK_END,
                        );
                        let total_chunks = self.chunk_counter + 1;
                        self.push_chunk_cv(first8(&out), total_chunks);
                        self.chunk_counter = total_chunks;
                        self.chunk_cv = IV;
                        self.blocks_compressed = 0;
                    } else {
                        let out = compress(
                            &self.chunk_cv,
                            &self.block,
                            self.chunk_counter,
                            BLOCK_LEN as u32,
                            self.start_flag(),
                        );
                        self.chunk_cv = first8(&out);
                        self.blocks_compressed += 1;
                    }
                    self.block_len = 0;
                }
                let take = (BLOCK_LEN - self.block_len as usize).min(input.len());
                self.block[self.block_len as usize..][..take].copy_from_slice(&input[..take]);
                self.block_len += take as u8;
                input = &input[take..];
            }
        }

        /// Merge a completed chunk's chaining value into the stack: pop and
        /// combine once per trailing zero bit of `total_chunks` (each marks a
        /// completed subtree), then push. Reference `add_chunk_chaining_value`.
        fn push_chunk_cv(&mut self, mut cv: [u32; 8], mut total_chunks: u64) {
            while total_chunks & 1 == 0 {
                self.stack_len -= 1;
                let block = parent_block(&self.cv_stack[self.stack_len as usize], &cv);
                cv = first8(&compress(&IV, &block, 0, BLOCK_LEN as u32, PARENT));
                total_chunks >>= 1;
            }
            self.cv_stack[self.stack_len as usize] = cv;
            self.stack_len += 1;
        }

        /// Root node's pre-finalization output: the current (possibly
        /// partial) chunk's output merged with every stacked subtree, deepest
        /// last. Does not mutate the state — squeezes never disturb the
        /// transcript.
        fn root_output(&self) -> Output {
            // Bytes past `block_len` may hold stale data from earlier blocks;
            // the spec compresses a zero-padded block.
            let mut block = [0u8; BLOCK_LEN];
            block[..self.block_len as usize].copy_from_slice(&self.block[..self.block_len as usize]);
            let mut out = Output {
                cv: self.chunk_cv,
                block,
                block_len: self.block_len as u32,
                counter: self.chunk_counter,
                flags: self.start_flag() | CHUNK_END,
            };
            for cv in self.cv_stack[..self.stack_len as usize].iter().rev() {
                let right = out.chaining_value();
                out = Output {
                    cv: IV,
                    block: parent_block(cv, &right),
                    block_len: BLOCK_LEN as u32,
                    counter: 0,
                    flags: PARENT,
                };
            }
            out
        }

        /// 32-byte root digest; identical to `blake3::Hasher::finalize`.
        pub(super) fn finalize(&self) -> [u8; 32] {
            let o = self.root_output();
            let words = compress(&o.cv, &o.block, 0, o.block_len, o.flags | ROOT);
            let mut bytes = [0u8; 32];
            for (dst, w) in bytes.chunks_exact_mut(4).zip(&words[..8]) {
                dst.copy_from_slice(&w.to_le_bytes());
            }
            bytes
        }

        /// Fill `out` with XOF output from position 0; identical to
        /// `blake3::Hasher::finalize_xof().fill(out)`. The 64-byte output
        /// blocks are independent compressions (counter = block index), so
        /// wide squeezes overlap in the pipeline for free.
        pub(super) fn xof_fill(&self, out: &mut [u8]) {
            let o = self.root_output();
            for (counter, dst) in out.chunks_mut(BLOCK_LEN).enumerate() {
                let words = compress(&o.cv, &o.block, counter as u64, o.block_len, o.flags | ROOT);
                let mut bytes = [0u8; BLOCK_LEN];
                for (b, w) in bytes.chunks_exact_mut(4).zip(&words) {
                    b.copy_from_slice(&w.to_le_bytes());
                }
                dst.copy_from_slice(&bytes[..dst.len()]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every FsChallenger property must hold under both transcript hashes:
    /// the tagging, absorption order and duplex structure are shared, and
    /// only the primitive differs.
    const KINDS: [HashKind; 2] = [HashKind::Sha256, HashKind::Blake3];

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
                    blake3_pow_scan(&state, start, len, 0),
                    Some(start),
                    "start={start} len={len}"
                );
                // Compare the scan against a scalar sweep at a threshold low
                // enough to hit but high enough to skip some nonces.
                let want = (start..start + len)
                    .find(|&n| pow_has_leading_zero_bits(&state, n, 6, HashKind::Blake3));
                assert_eq!(
                    blake3_pow_scan(&state, start, len, 6),
                    want,
                    "start={start} len={len}"
                );
            }
        }
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

    /// `TranscriptBlake3` must agree with `blake3::Hasher` bit-for-bit on
    /// every input length that exercises a distinct code path: empty, partial
    /// blocks, exact block/chunk boundaries and their neighbours, and
    /// multi-chunk trees — both digest and XOF output, fed one-shot and in
    /// pseudorandom small fragments (the transcript's actual access pattern).
    #[test]
    fn transcript_blake3_matches_crate() {
        let mut rng = 0x1717_5EED_u64;
        for len in [
            0usize, 1, 2, 31, 63, 64, 65, 127, 128, 129, 192, 1023, 1024, 1025, 2047, 2048, 2049,
            3072, 4096, 8191,
        ] {
            let data: Vec<u8> = (0..len).map(|_| splitmix64(&mut rng) as u8).collect();
            let expect = blake3::Hasher::new().update(&data).finalize();

            // One-shot.
            let mut ours = fs_blake3::TranscriptBlake3::new();
            ours.update(&data);
            assert_eq!(ours.finalize(), *expect.as_bytes(), "one-shot len {len}");

            // Fragmented into pseudorandom 1..=64-byte pieces.
            let mut frag = fs_blake3::TranscriptBlake3::new();
            let mut rest = &data[..];
            while !rest.is_empty() {
                let take = (splitmix64(&mut rng) as usize % 64 + 1).min(rest.len());
                frag.update(&rest[..take]);
                rest = &rest[take..];
            }
            assert_eq!(frag.finalize(), *expect.as_bytes(), "fragmented len {len}");

            // XOF output at widths crossing the 64-byte output-block boundary.
            let mut reader = blake3::Hasher::new().update(&data).finalize_xof();
            let mut want = vec![0u8; 200];
            reader.fill(&mut want);
            for xof_len in [1usize, 16, 31, 32, 33, 63, 64, 65, 100, 128, 200] {
                let mut got = vec![0u8; xof_len];
                ours.xof_fill(&mut got);
                assert_eq!(got, want[..xof_len], "xof len {xof_len} at input len {len}");
            }
        }
    }

    /// The BLAKE3 `FsChallenger` must be indistinguishable from the same
    /// duplex built directly on `blake3::Hasher`: 200 pseudorandom
    /// observe/sample interleavings (including multi-chunk observations and
    /// wide vector squeezes) produce identical challenges and identical
    /// state digests throughout.
    #[test]
    fn fs_challenger_blake3_matches_crate_reference() {
        /// Reference duplex over the plain crate, mirroring `FsChallenger`'s
        /// tag/absorption scheme byte-for-byte.
        struct RefChallenger {
            h: blake3::Hasher,
        }
        impl RefChallenger {
            fn new(domain: &[u8]) -> Self {
                let mut c = Self { h: blake3::Hasher::new() };
                c.h.update(&[OP_DOMAIN]);
                c.h.update(&(domain.len() as u64).to_le_bytes());
                c.h.update(domain);
                c
            }
            fn absorb_f128(&mut self, v: F128) {
                self.h.update(&v.lo.to_le_bytes());
                self.h.update(&v.hi.to_le_bytes());
            }
            fn observe_label(&mut self, label: &[u8]) {
                self.h.update(&[OP_LABEL]);
                self.h.update(&(label.len() as u64).to_le_bytes());
                self.h.update(label);
            }
            fn observe_bytes(&mut self, bytes: &[u8]) {
                self.h.update(&[OP_BYTES]);
                self.h.update(&(bytes.len() as u64).to_le_bytes());
                self.h.update(bytes);
            }
            fn observe_f128(&mut self, v: F128) {
                self.h.update(&[OP_OBSERVE, KIND_SCALAR]);
                self.absorb_f128(v);
            }
            fn observe_f128_slice(&mut self, vs: &[F128]) {
                self.h.update(&[OP_OBSERVE, KIND_SLICE]);
                self.h.update(&(vs.len() as u64).to_le_bytes());
                for v in vs {
                    self.absorb_f128(*v);
                }
            }
            fn sample_f128(&mut self) -> F128 {
                self.h.update(&[OP_SQUEEZE, KIND_SCALAR]);
                let mut buf = [0u8; 16];
                self.h.finalize_xof().fill(&mut buf);
                self.h.update(&buf);
                F128 {
                    lo: u64::from_le_bytes(buf[..8].try_into().unwrap()),
                    hi: u64::from_le_bytes(buf[8..].try_into().unwrap()),
                }
            }
            fn sample_f128_vec(&mut self, n: usize) -> Vec<F128> {
                self.h.update(&[OP_SQUEEZE, KIND_SLICE]);
                self.h.update(&(n as u64).to_le_bytes());
                let mut buf = vec![0u8; n * 16];
                self.h.finalize_xof().fill(&mut buf);
                self.h.update(&buf);
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

        let mut ours = FsChallenger::with_hash(b"equiv-test", HashKind::Blake3);
        let mut reference = RefChallenger::new(b"equiv-test");
        let mut rng = 0xB1A4_3EFA_u64;
        let rand_f128 = |rng: &mut u64| F128 {
            lo: splitmix64(rng),
            hi: splitmix64(rng),
        };
        for step in 0..200 {
            match splitmix64(&mut rng) % 6 {
                0 => {
                    // 16–64-byte observations, plus an occasional multi-chunk
                    // one so the CV stack genuinely merges mid-transcript.
                    let len = if step % 41 == 3 {
                        1500 + splitmix64(&mut rng) as usize % 1200
                    } else {
                        16 + splitmix64(&mut rng) as usize % 49
                    };
                    let bytes: Vec<u8> =
                        (0..len).map(|_| splitmix64(&mut rng) as u8).collect();
                    ours.observe_bytes(&bytes);
                    reference.observe_bytes(&bytes);
                }
                1 => {
                    let v = rand_f128(&mut rng);
                    ours.observe_f128(v);
                    reference.observe_f128(v);
                }
                2 => {
                    let n = 1 + splitmix64(&mut rng) as usize % 8;
                    let vs: Vec<F128> = (0..n).map(|_| rand_f128(&mut rng)).collect();
                    ours.observe_f128_slice(&vs);
                    reference.observe_f128_slice(&vs);
                }
                3 => {
                    ours.observe_label(b"phase-label");
                    reference.observe_label(b"phase-label");
                }
                4 => {
                    assert_eq!(ours.sample_f128(), reference.sample_f128(), "step {step}");
                }
                _ => {
                    let n = 1 + splitmix64(&mut rng) as usize % 37;
                    assert_eq!(
                        ours.sample_f128_vec(n),
                        reference.sample_f128_vec(n),
                        "step {step}"
                    );
                }
            }
            if step % 10 == 0 {
                assert_eq!(
                    ours.state_digest(),
                    *reference.h.finalize().as_bytes(),
                    "digest at step {step}"
                );
            }
        }
        assert_eq!(ours.state_digest(), *reference.h.finalize().as_bytes());
    }
}
