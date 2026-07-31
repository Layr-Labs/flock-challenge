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
        // Nonces per rayon task in the parallel search. Large enough to
        // amortize task dispatch (a 1024-nonce chunk is ~12 µs under the
        // aarch64 NEON kernel, ~86 Mh/s/core — and a whole multiple of its
        // 16-lane batch), small enough to keep cancellation granular once an
        // earlier task has found a match.
        const GRIND_CHUNK: u64 = 1 << 10;
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
            // Block-parallel search. Blocks are scanned in order and each task
            // returns the smallest match within its chunk, so the result is
            // deterministic (the globally smallest satisfying nonce).
            // Block ≈ 2× the expected attempts: large enough that the match
            // usually falls inside one block (so all threads do useful
            // pre-match work), small enough to avoid the 4× over-scan the old
            // `+2` block caused (which left ~¾ of threads doing cancelled work).
            use rayon::prelude::*;
            let block: u64 = 1 << (bits.min(24) + 1);
            let n_chunks = block.div_ceil(GRIND_CHUNK);
            let mut start: u64 = 0;
            loop {
                // `find_first` takes the earliest *chunk* that yields a match
                // and cancels the rest; within a chunk `pow_scan` returns the
                // smallest nonce. A later chunk cannot hold a smaller nonce, so
                // this is exactly the globally smallest — identical to the
                // sequential search, which is what keeps proofs deterministic.
                let found = (0..n_chunks)
                    .into_par_iter()
                    .map(|c| {
                        pow_scan(
                            &state_digest,
                            start.saturating_add(c * GRIND_CHUNK),
                            GRIND_CHUNK,
                            bits,
                            kind,
                        )
                    })
                    .find_first(|r| r.is_some())
                    .flatten();
                if let Some(n) = found {
                    break n;
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

/// Smallest nonce in `start .. start + len` whose BLAKE3 PoW hash has `bits`
/// leading zeros, or `None`.
///
/// Dispatch: on aarch64 a 16-lane NEON kernel specialized to the fixed PoW
/// message shape ([`blake3_pow_neon`]) covers every real grind — profiles top
/// out at 21 grinding bits, and the kernel handles `1..=32`. The generic
/// `hash_many` batch loop ([`blake3_pow_scan_many`]) remains as the portable
/// fallback, the `bits = 0` / `bits > 32` path, and the
/// `FLOCK_NO_GRIND_OPT=1` kill-switch target (local diagnostics / one-process
/// A-B; the ranked worker's cleared environment never sets it). Both paths
/// agree with `blake3::hash` on every nonce, so the smallest satisfying nonce
/// — and therefore the proof bytes — are identical either way.
fn blake3_pow_scan(state_digest: &[u8; 32], start: u64, len: u64, bits: u32) -> Option<u64> {
    #[cfg(target_arch = "aarch64")]
    {
        static OPT: std::sync::LazyLock<bool> =
            std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_GRIND_OPT").is_none());
        if (1..=32).contains(&bits) && *OPT {
            return blake3_pow_neon::scan(state_digest, start, len, bits);
        }
    }
    blake3_pow_scan_many(state_digest, start, len, bits)
}

/// Nonces hashed per `hash_many` call in the BLAKE3 grind.
///
/// Must clear the widest `simd_degree` (16, under AVX-512) so the batch fills
/// the machine's vector; 32 leaves headroom and keeps the buffers (2 KiB of
/// pre-images + 1 KiB of digests) stack-resident. Swept 1/4/8/16/32/64 on an
/// M4 Max: 1 is ~2.2× slower at 17 bits, everything from 4 up is within noise
/// of each other.
const BLAKE3_POW_BATCH: usize = 32;

/// Smallest nonce in `start .. start + len` whose BLAKE3 PoW hash has `bits`
/// leading zeros, or `None` — generic `hash_many` batch loop.
///
/// Batches the independent nonce hashes through the crate's SIMD compression.
/// A 64-byte pre-image is a whole-block single chunk, which `hash_many`
/// reproduces byte-for-byte given `CHUNK_START` / `CHUNK_END | ROOT` — so this
/// agrees with `blake3::hash` on every nonce, which
/// `blake3_batched_pow_matches_scalar` asserts.
fn blake3_pow_scan_many(state_digest: &[u8; 32], start: u64, len: u64, bits: u32) -> Option<u64> {
    use blake3::platform::Platform;
    // BLAKE3 constants, fixed by the spec.
    const IV: [u32; 8] = [
        0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB,
        0x5BE0CD19,
    ];
    const CHUNK_START: u8 = 1;
    const CHUNK_END: u8 = 2;
    const ROOT: u8 = 8;

    let plat = Platform::detect();
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
        let inputs: [&[u8; 64]; BLAKE3_POW_BATCH] = std::array::from_fn(|i| &pre[i]);
        plat.hash_many(
            &inputs[..n],
            &IV,
            0,
            blake3::IncrementCounter::No,
            0,
            CHUNK_START,
            CHUNK_END | ROOT,
            &mut out[..n * 32],
        );
        for i in 0..n {
            if has_leading_zero_bits(&out[i * 32..(i + 1) * 32], bits) {
                return Some(base + i as u64);
            }
        }
        base += n as u64;
    }
    None
}

// ---------------------------------------------------------------------------
// blake3_pow_neon — 16-lane (4 × 4-wide) NEON kernel for the BLAKE3 PoW grind.
//
// Why not `hash_many`? Three structural costs it cannot shed for this message
// shape, together measured ~3× off the compression roofline:
//
//   * Transposition + setup: `hash4_neon` loads and transposes 16 message
//     words × 4 lanes per call — but 56 of the 64 pre-image bytes are
//     *constant* across the whole grind (32-byte state digest + 24 zero
//     bytes), and the 8 nonce bytes are pre-known as `base + lane`.
//     Broadcasting the constants once per scan deletes the entire
//     load/transpose stage, the per-batch pre-image patching, and the
//     pointer-array setup.
//   * Latency: a single 4-wide compression exposes only 4 independent G
//     chains per half-round, so Apple's 4 NEON pipes sit roughly half idle
//     waiting on the ~14-op dependency chain inside each G. Interleaving
//     `GROUPS` independent 4-wide states (16 lanes per compression at the
//     swept optimum) multiplies the live chains and fills the pipes.
//   * Output: the PoW predicate at `bits ≤ 32` reads only digest word 0, so
//     the kernel computes one output XOR per group instead of eight, and the
//     byte-wise digest scan becomes two vector compares per 8 nonces.
//
// Determinism: the kernel computes bit-identical digest words to
// `blake3::hash` (asserted lane-by-lane by
// `blake3_neon_pow_scan_matches_scalar`), scans ascending, and reports the
// lowest passing lane — the same smallest satisfying nonce as the scalar
// spec and the `hash_many` path.
// ---------------------------------------------------------------------------
#[cfg(target_arch = "aarch64")]
mod blake3_pow_neon {
    use core::arch::aarch64::*;

    /// BLAKE3 IV, fixed by the spec.
    const IV: [u32; 8] = [
        0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB,
        0x5BE0CD19,
    ];
    /// `CHUNK_START | CHUNK_END | ROOT` — a whole-block single-chunk root
    /// message, exactly what `blake3::hash` computes for 64 bytes.
    const FLAGS: u32 = 1 | 2 | 8;
    /// Per-round message-word schedule, fixed by the spec.
    const MSG_SCHEDULE: [[usize; 16]; 7] = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
        [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
        [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
        [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
        [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
        [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
    ];

    /// Rotate each 32-bit lane right by 16: a `rev32.16` byte swap.
    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn ror16(x: uint32x4_t) -> uint32x4_t {
        vreinterpretq_u32_u16(vrev32q_u16(vreinterpretq_u16_u32(x)))
    }

    /// Rotate each 32-bit lane right by 12: `shl` + `sri`.
    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn ror12(x: uint32x4_t) -> uint32x4_t {
        vsriq_n_u32(vshlq_n_u32(x, 20), x, 12)
    }

    /// Rotate each 32-bit lane right by 8: a single `tbl` byte shuffle (the
    /// constant index vector is materialized once and hoisted).
    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn ror8(x: uint32x4_t) -> uint32x4_t {
        const IDX: [u8; 16] = [1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12];
        let idx = vld1q_u8(IDX.as_ptr());
        vreinterpretq_u32_u8(vqtbl1q_u8(vreinterpretq_u8_u32(x), idx))
    }

    /// Rotate each 32-bit lane right by 7: `shl` + `sri`.
    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn ror7(x: uint32x4_t) -> uint32x4_t {
        vsriq_n_u32(vshlq_n_u32(x, 25), x, 7)
    }

    /// The 16 message words of one 4-lane group in broadcast/lane-vector
    /// form: words 0..8 are the state digest (identical across lanes), words
    /// 8..10 the per-lane nonce halves, words 10..16 the zero tail padding.
    #[derive(Clone, Copy)]
    struct Msg {
        dig: [uint32x4_t; 8],
        n_lo: uint32x4_t,
        n_hi: uint32x4_t,
        zero: uint32x4_t,
    }

    impl Msg {
        /// Message word `j`. Every call site passes a `MSG_SCHEDULE` entry,
        /// which constant-folds after inlining — no branch survives.
        #[inline(always)]
        fn w(&self, j: usize) -> uint32x4_t {
            match j {
                0..=7 => self.dig[j],
                8 => self.n_lo,
                9 => self.n_hi,
                _ => self.zero,
            }
        }
    }

    /// The BLAKE3 quarter-round on one 4-wide state.
    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn g(
        v: &mut [uint32x4_t; 16],
        a: usize,
        b: usize,
        c: usize,
        d: usize,
        mx: uint32x4_t,
        my: uint32x4_t,
    ) {
        v[a] = vaddq_u32(vaddq_u32(v[a], v[b]), mx);
        v[d] = ror16(veorq_u32(v[d], v[a]));
        v[c] = vaddq_u32(v[c], v[d]);
        v[b] = ror12(veorq_u32(v[b], v[c]));
        v[a] = vaddq_u32(vaddq_u32(v[a], v[b]), my);
        v[d] = ror8(veorq_u32(v[d], v[a]));
        v[c] = vaddq_u32(v[c], v[d]);
        v[b] = ror7(veorq_u32(v[b], v[c]));
    }

    /// `g` without the final `b`-word update, for final-round G's whose `b`
    /// output feeds nothing (`b1` is still needed mid-chain for `a`).
    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn g_no_b(
        v: &mut [uint32x4_t; 16],
        a: usize,
        b: usize,
        c: usize,
        d: usize,
        mx: uint32x4_t,
        my: uint32x4_t,
    ) {
        v[a] = vaddq_u32(vaddq_u32(v[a], v[b]), mx);
        v[d] = ror16(veorq_u32(v[d], v[a]));
        v[c] = vaddq_u32(v[c], v[d]);
        let b1 = ror12(veorq_u32(v[b], v[c]));
        v[a] = vaddq_u32(vaddq_u32(v[a], b1), my);
        v[d] = ror8(veorq_u32(v[d], v[a]));
        v[c] = vaddq_u32(v[c], v[d]);
    }

    /// `g` truncated to just the second `a`-word update, for the one
    /// final-round diagonal whose only live output is `v0`.
    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn g_a_only(
        v: &mut [uint32x4_t; 16],
        a: usize,
        b: usize,
        c: usize,
        d: usize,
        mx: uint32x4_t,
        my: uint32x4_t,
    ) {
        v[a] = vaddq_u32(vaddq_u32(v[a], v[b]), mx);
        let d1 = ror16(veorq_u32(v[d], v[a]));
        let c1 = vaddq_u32(v[c], d1);
        let b1 = ror12(veorq_u32(v[b], c1));
        v[a] = vaddq_u32(vaddq_u32(v[a], b1), my);
    }

    /// Independent 4-wide states interleaved per compression call (the scan
    /// hashes `4 * GROUPS` nonces per iteration). One 4-wide state exposes
    /// only 4 independent G chains per half-round — not enough to hide the
    /// ~2-2.5-cycle effective latency of the add/eor/sri/tbl chain on 4 NEON
    /// pipes. Swept on an M4 Max (paired vs `hash_many`, 1 core): 2 → 1.75×,
    /// 3 → 1.81×, 4 → 1.89×, 6 → 1.94×, 8 → 1.93×. 4 is the pick: 16 lanes
    /// divide `GRIND_CHUNK` exactly (a 24-lane batch masks off 8 dead lanes
    /// every 1024-nonce chunk, refunding 6's edge), and past 4 the register
    /// file is so oversubscribed that gains drown in spill traffic.
    const GROUPS: usize = 4;
    /// Nonce lanes per scan iteration.
    const LANES: usize = 4 * GROUPS;

    /// One BLAKE3 round on `GROUPS` independent 4-wide states. Same-position
    /// G's of all groups are issued adjacently so the independent dependency
    /// chains sit together in the instruction window (the OoO core spreads
    /// them across the 4 NEON pipes).
    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn round_n(vs: &mut [[uint32x4_t; 16]; GROUPS], ms: &[Msg; GROUPS], s: &[usize; 16]) {
        // Column step.
        for k in 0..GROUPS {
            g(&mut vs[k], 0, 4, 8, 12, ms[k].w(s[0]), ms[k].w(s[1]));
        }
        for k in 0..GROUPS {
            g(&mut vs[k], 1, 5, 9, 13, ms[k].w(s[2]), ms[k].w(s[3]));
        }
        for k in 0..GROUPS {
            g(&mut vs[k], 2, 6, 10, 14, ms[k].w(s[4]), ms[k].w(s[5]));
        }
        for k in 0..GROUPS {
            g(&mut vs[k], 3, 7, 11, 15, ms[k].w(s[6]), ms[k].w(s[7]));
        }
        // Diagonal step.
        for k in 0..GROUPS {
            g(&mut vs[k], 0, 5, 10, 15, ms[k].w(s[8]), ms[k].w(s[9]));
        }
        for k in 0..GROUPS {
            g(&mut vs[k], 1, 6, 11, 12, ms[k].w(s[10]), ms[k].w(s[11]));
        }
        for k in 0..GROUPS {
            g(&mut vs[k], 2, 7, 8, 13, ms[k].w(s[12]), ms[k].w(s[13]));
        }
        for k in 0..GROUPS {
            g(&mut vs[k], 3, 4, 9, 14, ms[k].w(s[14]), ms[k].w(s[15]));
        }
    }

    /// Final round (schedule row 6), pruned to what reaches the output word
    /// `v0 ^ v8`: the diagonals writing `v1/v6/v11/v12` and `v3/v4/v9/v14`
    /// are dead, `G(0,5,10,15)` only needs its `a` output (`v0`), and
    /// `G(2,7,8,13)` everything but `b` (`v8` is its `c`). Of the column
    /// G's, col0/col2 feed the live diagonals through `a`/`c` only (their
    /// `b`/`d` outputs go to the dead diagonals), while col1/col3 must stay
    /// whole for `v5/v13` and `v7/v15`. Saves ~37% of the round.
    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn round_final_n(vs: &mut [[uint32x4_t; 16]; GROUPS], ms: &[Msg; GROUPS]) {
        let s = &MSG_SCHEDULE[6];
        // Column step.
        for k in 0..GROUPS {
            g_no_b(&mut vs[k], 0, 4, 8, 12, ms[k].w(s[0]), ms[k].w(s[1]));
        }
        for k in 0..GROUPS {
            g(&mut vs[k], 1, 5, 9, 13, ms[k].w(s[2]), ms[k].w(s[3]));
        }
        for k in 0..GROUPS {
            g_no_b(&mut vs[k], 2, 6, 10, 14, ms[k].w(s[4]), ms[k].w(s[5]));
        }
        for k in 0..GROUPS {
            g(&mut vs[k], 3, 7, 11, 15, ms[k].w(s[6]), ms[k].w(s[7]));
        }
        // Diagonal step — only the two output-reaching G's survive.
        for k in 0..GROUPS {
            g_a_only(&mut vs[k], 0, 5, 10, 15, ms[k].w(s[8]), ms[k].w(s[9]));
        }
        for k in 0..GROUPS {
            g_no_b(&mut vs[k], 2, 7, 8, 13, ms[k].w(s[12]), ms[k].w(s[13]));
        }
    }

    /// Round-0 constant prefix, computed once per scan. Under the identity
    /// schedule the whole column step reads only the constant digest words
    /// `m0..m8`, and three of the four diagonal G's read only the zero words
    /// `m10..m16` — all independent of the nonce (the initial state is the
    /// spec constant: CV = IV, counter = 0, block length 64, root flags).
    /// Per iteration only the remaining diagonal `G(0,5,10,15)` — messages
    /// `m8`/`m9`, the nonce halves — runs, cutting round 0 from 8 G's to 1.
    /// Its word set is disjoint from the three precomputed diagonals, so the
    /// split is exact.
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn round0_prefix(dig: &[uint32x4_t; 8], zero: uint32x4_t) -> [uint32x4_t; 16] {
        let mut v: [uint32x4_t; 16] = [
            vdupq_n_u32(IV[0]),
            vdupq_n_u32(IV[1]),
            vdupq_n_u32(IV[2]),
            vdupq_n_u32(IV[3]),
            vdupq_n_u32(IV[4]),
            vdupq_n_u32(IV[5]),
            vdupq_n_u32(IV[6]),
            vdupq_n_u32(IV[7]),
            vdupq_n_u32(IV[0]),
            vdupq_n_u32(IV[1]),
            vdupq_n_u32(IV[2]),
            vdupq_n_u32(IV[3]),
            vdupq_n_u32(0),
            vdupq_n_u32(0),
            vdupq_n_u32(64),
            vdupq_n_u32(FLAGS),
        ];
        // Column step (schedule row 0 is the identity: m0..m8).
        g(&mut v, 0, 4, 8, 12, dig[0], dig[1]);
        g(&mut v, 1, 5, 9, 13, dig[2], dig[3]);
        g(&mut v, 2, 6, 10, 14, dig[4], dig[5]);
        g(&mut v, 3, 7, 11, 15, dig[6], dig[7]);
        // Nonce-free diagonals (m10..m16 are the zero tail padding).
        g(&mut v, 1, 6, 11, 12, zero, zero);
        g(&mut v, 2, 7, 8, 13, zero, zero);
        g(&mut v, 3, 4, 9, 14, zero, zero);
        v
    }

    /// Compress all `GROUPS` 4-lane groups from the round-0 prefix and
    /// return digest word 0 per lane (`v0 ^ v8`; full digest word `i` would
    /// be `v[i] ^ v[i+8]`, but the `bits ≤ 32` PoW predicate reads only
    /// word 0).
    #[inline(always)]
    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn compress_word0(
        prefix: &[uint32x4_t; 16],
        ms: &[Msg; GROUPS],
    ) -> [uint32x4_t; GROUPS] {
        let mut vs = [*prefix; GROUPS];
        // Rest of round 0: the nonce-carrying diagonal.
        for k in 0..GROUPS {
            g(&mut vs[k], 0, 5, 10, 15, ms[k].n_lo, ms[k].n_hi);
        }
        round_n(&mut vs, ms, &MSG_SCHEDULE[1]);
        round_n(&mut vs, ms, &MSG_SCHEDULE[2]);
        round_n(&mut vs, ms, &MSG_SCHEDULE[3]);
        round_n(&mut vs, ms, &MSG_SCHEDULE[4]);
        round_n(&mut vs, ms, &MSG_SCHEDULE[5]);
        round_final_n(&mut vs, ms);
        std::array::from_fn(|k| veorq_u32(vs[k][0], vs[k][8]))
    }

    /// Smallest nonce in `start .. start + len` (saturating) whose BLAKE3
    /// PoW digest has at least `bits` leading zero bits. Requires
    /// `1 ≤ bits ≤ 32` — the predicate then depends only on digest word 0.
    pub(super) fn scan(state_digest: &[u8; 32], start: u64, len: u64, bits: u32) -> Option<u64> {
        debug_assert!((1..=32).contains(&bits));
        // SAFETY: NEON is baseline on aarch64.
        unsafe { scan_impl(state_digest, start, len, bits) }
    }

    #[allow(unsafe_op_in_unsafe_fn)]
    unsafe fn scan_impl(state_digest: &[u8; 32], start: u64, len: u64, bits: u32) -> Option<u64> {
        let mut dig = [vdupq_n_u32(0); 8];
        for (i, d) in dig.iter_mut().enumerate() {
            *d = vdupq_n_u32(u32::from_le_bytes(
                state_digest[4 * i..4 * i + 4].try_into().unwrap(),
            ));
        }
        let zero = vdupq_n_u32(0);
        // Leading zero bits of the digest byte stream = integer leading
        // zeros of byte-reversed word 0, so the pass test is
        // `bswap32(w0) < 2^(32 - bits)`.
        let thresh = vdupq_n_u32(if bits == 32 { 1 } else { 1u32 << (32 - bits) });
        let lane_bits = vld1q_u32([1u32, 2, 4, 8].as_ptr());
        // Nonce-independent round-0 prefix, hoisted out of the scan loop.
        let prefix = round0_prefix(&dig, zero);

        let end = start.saturating_add(len);
        let mut base = start;
        while base < end {
            let n = (end - base).min(LANES as u64) as u32;
            // Lane nonces `base + i`, split into 32-bit message words. A
            // ragged tail hashes all `LANES` lanes anyway and masks the
            // extras off — cheaper than a second code path (wrapping only
            // matters at the 2^64 boundary, where the masked lanes are never
            // inspected).
            let mut lo = [0u32; LANES];
            let mut hi = [0u32; LANES];
            for i in 0..LANES {
                let x = base.wrapping_add(i as u64);
                lo[i] = x as u32;
                hi[i] = (x >> 32) as u32;
            }
            #[cfg(feature = "hash-count")]
            super::fs_count::POW_SHA256.fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);
            let ms: [Msg; GROUPS] = std::array::from_fn(|k| Msg {
                dig,
                n_lo: vld1q_u32(lo.as_ptr().add(4 * k)),
                n_hi: vld1q_u32(hi.as_ptr().add(4 * k)),
                zero,
            });
            let h0 = compress_word0(&prefix, &ms);
            let cmp: [uint32x4_t; GROUPS] = std::array::from_fn(|k| {
                vcltq_u32(
                    vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(h0[k]))),
                    thresh,
                )
            });
            let mut any = cmp[0];
            for &c in &cmp[1..] {
                any = vorrq_u32(any, c);
            }
            if vmaxvq_u32(any) != 0 {
                let mut mask = 0u32;
                for (k, &c) in cmp.iter().enumerate() {
                    mask |= vaddvq_u32(vandq_u32(c, lane_bits)) << (4 * k);
                }
                mask &= (1u32 << n) - 1;
                if mask != 0 {
                    return Some(base + u64::from(mask.trailing_zeros()));
                }
            }
            base += u64::from(n);
        }
        None
    }
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

    /// Every FsChallenger property must hold under both transcript hashes:
    /// the tagging, absorption order and duplex structure are shared, and
    /// only the primitive differs.
    const KINDS: [HashKind; 2] = [HashKind::Sha256, HashKind::Blake3];

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

    /// The 16-lane NEON kernel must agree with the scalar spec
    /// (`blake3::hash` of the 64-byte pre-image) and with the `hash_many`
    /// path on every nonce, across ragged widths, lane-boundary starts, the
    /// `2^32` nonce-word carry, and the full `1..=32` bits range the kernel
    /// accepts. This is the determinism oracle for the fast grind path.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn blake3_neon_pow_scan_matches_scalar() {
        // Several digests so the constant-broadcast path sees varied words
        // (including all-zero, which zeroes 8 of the 10 live message words).
        let digests: [[u8; 32]; 3] = [
            [0x5Au8; 32],
            [0u8; 32],
            std::array::from_fn(|i| (i as u8).wrapping_mul(37).wrapping_add(11)),
        ];
        // Starts straddle lane
        // alignment and the 2^32 boundary where the nonce-hi word kicks in.
        // Widths straddle the 16-lane batch (15/16/17), a two-batch span
        // (31/32/33), and longer ragged shapes.
        let lens = [1u64, 3, 7, 8, 9, 15, 16, 17, 24, 25, 31, 32, 33, 100];
        let starts = [0u64, 5, 8, (1u64 << 32) - 4, u64::from(u32::MAX), 1 << 40];
        for state in &digests {
            for &len in &lens {
                for &start in &starts {
                    for bits in [1u32, 6, 8, 13, 32] {
                        let want = (start..start + len).find(|&n| {
                            pow_has_leading_zero_bits(state, n, bits, HashKind::Blake3)
                        });
                        assert_eq!(
                            blake3_pow_neon::scan(state, start, len, bits),
                            want,
                            "neon vs scalar: start={start} len={len} bits={bits}"
                        );
                        assert_eq!(
                            blake3_pow_scan_many(state, start, len, bits),
                            want,
                            "hash_many vs scalar: start={start} len={len} bits={bits}"
                        );
                    }
                }
            }
        }
        // Lane-exact digest-word check: the kernel's word-0 predicate at
        // bits = 32 must single out exactly the nonces whose full digest
        // starts with 4 zero bytes — i.e. the kernel's compression output
        // word 0 is bit-identical to `blake3::hash`. Verified implicitly
        // above; here pin one known hash directly through a 1-lane scan.
        let state = digests[2];
        for nonce in [0u64, 1, 255, 1 << 33] {
            let h = blake3::hash(&blake3_pow_preimage(&state, nonce));
            // Recover the number of leading zero bits the digest actually
            // has (capped at 32) and check the kernel's accept/reject edge.
            let w0 = u32::from_le_bytes(h.as_bytes()[..4].try_into().unwrap()).swap_bytes();
            let lz = w0.leading_zeros().min(32);
            if lz >= 1 {
                assert_eq!(blake3_pow_neon::scan(&state, nonce, 1, lz), Some(nonce));
            }
            if lz < 32 {
                assert_eq!(blake3_pow_neon::scan(&state, nonce, 1, lz + 1), None);
            }
        }
    }

    /// Paired micro-bench of the two BLAKE3 grind scans (NEON kernel vs
    /// `hash_many` batch loop), plus wall-clock `grind_pow` at the profile
    /// bit range. Ignored by default; run with
    /// `cargo test --release -p flock-core --lib -- --ignored --nocapture grind_throughput`.
    #[cfg(target_arch = "aarch64")]
    #[test]
    #[ignore]
    fn grind_throughput_bench() {
        let state = [0x37u8; 32];
        // No-match scan (bits = 32 ⇒ hit probability ~2^-32 per nonce): pure
        // throughput, no early exit. Interleave the two paths A/B/A/B to
        // cancel thermal drift.
        let len: u64 = 1 << 21;
        let mut neon_best = f64::MAX;
        let mut many_best = f64::MAX;
        for rep in 0..6 {
            let t = std::time::Instant::now();
            let r = blake3_pow_neon::scan(&state, rep * len, len, 32);
            let dt = t.elapsed().as_secs_f64();
            assert_eq!(r, None);
            neon_best = neon_best.min(dt);
            let t = std::time::Instant::now();
            let r = blake3_pow_scan_many(&state, rep * len, len, 32);
            let dt = t.elapsed().as_secs_f64();
            assert_eq!(r, None);
            many_best = many_best.min(dt);
        }
        let mh = |dt: f64| (len as f64 / dt) / 1e6;
        println!(
            "scan throughput (1 core): neon {:.1} Mh/s | hash_many {:.1} Mh/s | ratio {:.2}x",
            mh(neon_best),
            mh(many_best),
            many_best / neon_best
        );
        // Grind-level wall clock at the profile bit range (parallel rayon
        // path; nonce position varies per bits, so report implied Mh/s too).
        for bits in 14..=19u32 {
            let mut ch = FsChallenger::with_hash(b"grind-bench", HashKind::Blake3);
            ch.observe_bytes(b"bench-root");
            let t = std::time::Instant::now();
            let nonce = ch.grind_pow(bits);
            let dt = t.elapsed().as_secs_f64();
            println!(
                "grind_pow bits={bits}: {:.3} ms (nonce={nonce}, ~{:.1} Mh/s aggregate)",
                dt * 1e3,
                (nonce + 1) as f64 / dt / 1e6
            );
        }
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
}
