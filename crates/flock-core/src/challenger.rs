use crate::field::F128;
use crate::hash::HashKind;
use sha2::{Digest, Sha256};

pub trait Challenger: Send {
    fn observe_label(&mut self, _label: &[u8]) {}

    fn observe_f128(&mut self, value: F128);

    fn observe_f128_slice(&mut self, values: &[F128]) {
        for v in values {
            self.observe_f128(*v);
        }
    }

    fn observe_bytes(&mut self, _bytes: &[u8]) {}

    fn sample_f128(&mut self) -> F128;

    fn sample_f128_vec(&mut self, n: usize) -> Vec<F128> {
        (0..n).map(|_| self.sample_f128()).collect()
    }

    fn grind_pow(&mut self, _bits: u32) -> u64 {
        0
    }

    fn verify_pow(&mut self, _nonce: u64, _bits: u32) -> bool {
        true
    }
}

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
    fn observe_f128(&mut self, _value: F128) {}

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

const OP_DOMAIN: u8 = 0x01;
const OP_LABEL: u8 = 0x02;
const OP_OBSERVE: u8 = 0x03;
const OP_SQUEEZE: u8 = 0x04;
const OP_BYTES: u8 = 0x05;

const KIND_SCALAR: u8 = 0x01;
const KIND_SLICE: u8 = 0x02;

#[cfg(feature = "hash-count")]
pub mod fs_count {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    pub static SQUEEZES: AtomicU64 = AtomicU64::new(0);

    pub static POW_SHA256: AtomicU64 = AtomicU64::new(0);

    pub fn reset() {
        SQUEEZES.store(0, Relaxed);
        POW_SHA256.store(0, Relaxed);
    }

    pub fn snapshot() -> (u64, u64) {
        (SQUEEZES.load(Relaxed), POW_SHA256.load(Relaxed))
    }
}

#[derive(Clone)]
enum FsState {
    Sha256(Sha256),
    Blake3(Box<blake3::Hasher>),
}

#[derive(Clone)]
pub struct FsChallenger {
    state: FsState,

    #[allow(dead_code)]
    n_absorbed: u64,
}

impl FsChallenger {
    pub fn new(domain: &[u8]) -> Self {
        Self::with_hash(domain, HashKind::Sha256)
    }

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

    pub fn hash_kind(&self) -> HashKind {
        match self.state {
            FsState::Sha256(_) => HashKind::Sha256,
            FsState::Blake3(_) => HashKind::Blake3,
        }
    }

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

    #[inline]
    fn state_digest(&self) -> [u8; 32] {
        #[cfg(feature = "hash-count")]
        fs_count::SQUEEZES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        match &self.state {
            FsState::Sha256(h) => h.clone().finalize().into(),
            FsState::Blake3(h) => *h.finalize().as_bytes(),
        }
    }

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

        const PARALLEL_GRIND_MIN_HASHES: u64 = 1 << 11;
        const GRIND_CHUNK: u64 = 1 << 8;
        let nonce = if bits == 0 {
            0
        } else if (1u64 << bits.min(63)) < PARALLEL_GRIND_MIN_HASHES {
            let mut start: u64 = 0;
            loop {
                if let Some(n) = pow_scan(&state_digest, start, GRIND_CHUNK, bits, kind) {
                    break n;
                }
                start = start.saturating_add(GRIND_CHUNK);
            }
        } else {
            use std::sync::atomic::{AtomicU64, Ordering};
            let next_chunk = AtomicU64::new(0);
            let best_chunk = AtomicU64::new(u64::MAX);
            let best_nonce = AtomicU64::new(u64::MAX);
            rayon::broadcast(|_| {
                loop {
                    let c = next_chunk.fetch_add(1, Ordering::Relaxed);
                    if c >= best_chunk.load(Ordering::Relaxed) {
                        break;
                    }
                    let start = c.saturating_mul(GRIND_CHUNK);
                    if let Some(n) = pow_scan(&state_digest, start, GRIND_CHUNK, bits, kind) {
                        best_nonce.fetch_min(n, Ordering::Relaxed);
                        best_chunk.fetch_min(c, Ordering::Relaxed);
                        break;
                    }
                }
            });
            best_nonce.load(Ordering::Relaxed)
        };

        self.observe_bytes(&nonce.to_le_bytes());
        nonce
    }

    fn verify_pow(&mut self, nonce: u64, bits: u32) -> bool {
        let kind = self.hash_kind();
        let state_digest = self.state_digest();
        let ok = if bits == 0 {
            nonce == 0
        } else {
            pow_has_leading_zero_bits(&state_digest, nonce, bits, kind)
        };

        self.observe_bytes(&nonce.to_le_bytes());
        ok
    }
}

#[inline]
fn blake3_pow_preimage(state_digest: &[u8; 32], nonce: u64) -> [u8; 64] {
    let mut pre = [0u8; 64];
    pre[..32].copy_from_slice(state_digest);
    pre[32..40].copy_from_slice(&nonce.to_le_bytes());
    pre
}

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

const BLAKE3_POW_BATCH: usize = 32;

fn blake3_pow_scan(state_digest: &[u8; 32], start: u64, len: u64, bits: u32) -> Option<u64> {
    use blake3::platform::Platform;

    const IV: [u32; 8] = [
        0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB,
        0x5BE0CD19,
    ];
    const CHUNK_START: u8 = 1;
    const CHUNK_END: u8 = 2;
    const ROOT: u8 = 8;

    let plat = Platform::detect();

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

    const KINDS: [HashKind; 2] = [HashKind::Sha256, HashKind::Blake3];

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

                for _ in 0..4 {
                    assert_eq!(prover.sample_f128(), verifier.sample_f128());
                }
            }
        }
    }

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

    #[test]
    fn fs_challenger_new_defaults_to_sha256() {
        assert_eq!(FsChallenger::new(b"d").hash_kind(), HashKind::Sha256);
        for kind in KINDS {
            assert_eq!(FsChallenger::with_hash(b"d", kind).hash_kind(), kind);
        }

        let mut a = FsChallenger::new(b"d");
        let mut b = FsChallenger::with_hash(b"d", HashKind::Sha256);
        assert_eq!(a.sample_f128_vec(4), b.sample_f128_vec(4));
    }

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

    #[test]
    fn fs_challenger_long_squeeze_is_prefix_stable() {
        for kind in KINDS {
            let mut a = FsChallenger::with_hash(b"d", kind);
            let mut b = FsChallenger::with_hash(b"d", kind);
            assert_eq!(a.sample_f128_vec(8), b.sample_f128_vec(8), "{kind}");

            let vals = FsChallenger::with_hash(b"d", kind).sample_f128_vec(16);
            let unique: std::collections::HashSet<_> = vals.iter().collect();
            assert_eq!(unique.len(), vals.len(), "{kind}: squeeze stream repeats");
        }
    }

    #[test]
    fn blake3_batched_pow_matches_scalar() {
        let state = [0x5Au8; 32];

        for len in [1u64, 5, 31, 32, 33, 100] {
            for start in [0u64, 7, 1_000_000] {
                assert_eq!(
                    blake3_pow_scan(&state, start, len, 0),
                    Some(start),
                    "start={start} len={len}"
                );

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

    #[test]
    fn fs_challenger_grind_returns_smallest_nonce() {
        for kind in KINDS {
            for bits in [4u32, 14] {
                let mut ch = FsChallenger::with_hash(b"grind-min", kind);
                ch.observe_bytes(b"root");
                let digest_probe = {
                    let mut probe = FsChallenger::with_hash(b"grind-min", kind);
                    probe.observe_bytes(b"root");
                    probe.state_digest()
                };
                let nonce = ch.grind_pow(bits);

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

            assert_ne!(c1.sample_f128(), c2.sample_f128());
        }
    }

    #[test]
    fn fs_challenger_scalar_vs_slice_dont_collide() {
        for kind in KINDS {
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
            let mut c1 = FsChallenger::with_hash(b"flock", kind);
            let mut c2 = FsChallenger::with_hash(b"flock", kind);
            assert_ne!(c1.sample_f128(), c2.sample_f128_vec(1)[0]);
        }
    }

    #[test]
    fn fs_challenger_sample_advances_state() {
        for kind in KINDS {
            let mut c1 = FsChallenger::with_hash(b"flock", kind);
            let mut c2 = FsChallenger::with_hash(b"flock", kind);
            let _ = c1.sample_f128();

            c1.observe_f128(F128::ONE);
            c2.observe_f128(F128::ONE);
            assert_ne!(c1.sample_f128(), c2.sample_f128());
        }
    }
}
