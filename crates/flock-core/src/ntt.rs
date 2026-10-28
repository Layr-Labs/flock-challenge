use crate::field::F8;

pub mod additive_ntt_f128;
pub mod inv_table;
pub mod inv_table_deg4;
pub mod parallel_f128;
pub use additive_ntt_f128::AdditiveNttF128;
pub use inv_table::InvNttTableByteSingleGf8;
pub use inv_table_deg4::InvNttTableSToV8Gf8;
pub use parallel_f128::ParallelNttF128;

#[inline]
fn next_s(s: F8, s_at_root: F8) -> F8 {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    unsafe {
        return F8(
            crate::gf2_kernels::gf2_kern_gf8_ntt_next_s(s.0 as u64, s_at_root.0 as u64) as u8,
        );
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        s * (s + s_at_root)
    }
}

pub fn compute_twiddles(k: usize, beta: F8) -> Vec<F8> {
    if k == 0 {
        return Vec::new();
    }
    let n = 1usize << k;
    let mut twiddles = vec![F8::ZERO; n - 1];

    let mut len = 1usize << (k - 1);
    let mut layer: Vec<F8> = (0..len).map(|i| beta + F8((2 * i) as u8)).collect();
    let mut s_at_root = F8::ONE;

    let mut write_at = len;
    for i in 0..len {
        twiddles[write_at - 1 + i] = layer[i];
    }

    for _ in 1..k {
        write_at >>= 1;
        let next_s_root = next_s(layer[1] + layer[0], s_at_root);
        let new_len = write_at;
        for i in 0..new_len {
            layer[i] = next_s(layer[2 * i], s_at_root);
        }
        len = new_len;
        s_at_root = next_s_root;

        let s_inv = s_at_root.inv();
        for j in 0..len {
            twiddles[write_at - 1 + j] = s_inv * layer[j];
        }
    }

    twiddles
}

#[inline]
fn fft_butterfly(v: &mut [F8], lambda: F8) {
    let n = v.len();
    let half = n >> 1;
    for i in 0..half {
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        unsafe {
            let p = crate::gf2_kernels::gf2_kern_gf8_fft_btf_pair(
                v[i].0 as u64,
                v[half + i].0 as u64,
                lambda.0 as u64,
            );
            v[i] = F8(p as u8);
            v[half + i] = F8((p >> 32) as u8);
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            let w = v[half + i];
            v[i] += lambda * w;
            v[half + i] = w + v[i];
        }
    }
}

fn fft_rec(v: &mut [F8], tw: &[F8], idx: usize) {
    let n = v.len();
    if n == 1 {
        return;
    }
    fft_butterfly(v, tw[idx - 1]);
    let half = n >> 1;
    let (lo, hi) = v.split_at_mut(half);
    fft_rec(lo, tw, 2 * idx);
    fft_rec(hi, tw, 2 * idx + 1);
}

#[inline]
fn ifft_butterfly(v: &mut [F8], lambda: F8) {
    let n = v.len();
    let half = n >> 1;
    for i in 0..half {
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        unsafe {
            let p = crate::gf2_kernels::gf2_kern_gf8_ifft_btf_pair(
                v[i].0 as u64,
                v[half + i].0 as u64,
                lambda.0 as u64,
            );
            v[i] = F8(p as u8);
            v[half + i] = F8((p >> 32) as u8);
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            v[half + i] += v[i];
            v[i] += lambda * v[half + i];
        }
    }
}

fn ifft_rec(v: &mut [F8], tw: &[F8], idx: usize) {
    let n = v.len();
    if n == 1 {
        return;
    }
    let half = n >> 1;
    let (lo, hi) = v.split_at_mut(half);
    ifft_rec(lo, tw, 2 * idx);
    ifft_rec(hi, tw, 2 * idx + 1);
    ifft_butterfly(v, tw[idx - 1]);
}

#[derive(Clone, Debug)]
pub struct AdditiveNttGf8 {
    k: usize,
    twiddles: Vec<F8>,
}

impl AdditiveNttGf8 {
    pub fn new(k: usize, beta: F8) -> Self {
        Self {
            k,
            twiddles: compute_twiddles(k, beta),
        }
    }

    pub fn k(&self) -> usize {
        self.k
    }
    pub fn domain_size(&self) -> usize {
        1usize << self.k
    }
    pub fn twiddles(&self) -> &[F8] {
        &self.twiddles
    }

    pub fn forward(&self, v: &mut [F8]) {
        assert_eq!(
            v.len(),
            self.domain_size(),
            "forward: input length must be 2^k"
        );
        if v.len() <= 1 {
            return;
        }
        fft_rec(v, &self.twiddles, 1);
    }

    pub fn inverse(&self, v: &mut [F8]) {
        assert_eq!(
            v.len(),
            self.domain_size(),
            "inverse: input length must be 2^k"
        );
        if v.len() <= 1 {
            return;
        }
        ifft_rec(v, &self.twiddles, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
    }

    fn rand_vec(rng: &mut Rng, n: usize) -> Vec<F8> {
        (0..n).map(|_| F8((rng.next_u64() & 0xff) as u8)).collect()
    }

    #[test]
    fn twiddles_size() {
        for k in 1..=7 {
            let ntt = AdditiveNttGf8::new(k, F8::ZERO);
            assert_eq!(ntt.twiddles().len(), (1usize << k) - 1);
        }
    }

    #[test]
    fn forward_inverse_roundtrip() {
        let mut rng = Rng::new(42);
        for k in 1..=7 {
            let ntt = AdditiveNttGf8::new(k, F8::ZERO);
            for _ in 0..8 {
                let original = rand_vec(&mut rng, 1 << k);
                let mut v = original.clone();
                ntt.forward(&mut v);
                ntt.inverse(&mut v);
                assert_eq!(v, original, "roundtrip failed at k={k}");
            }
        }
    }

    #[test]
    fn inverse_forward_roundtrip() {
        let mut rng = Rng::new(43);
        for k in 1..=7 {
            let ntt = AdditiveNttGf8::new(k, F8::ZERO);
            for _ in 0..8 {
                let original = rand_vec(&mut rng, 1 << k);
                let mut v = original.clone();
                ntt.inverse(&mut v);
                ntt.forward(&mut v);
                assert_eq!(v, original, "inverse∘forward roundtrip failed at k={k}");
            }
        }
    }

    #[test]
    fn forward_is_linear() {
        let mut rng = Rng::new(44);
        for k in 1..=6 {
            let ntt = AdditiveNttGf8::new(k, F8::ZERO);
            let n = 1usize << k;
            let a = rand_vec(&mut rng, n);
            let b = rand_vec(&mut rng, n);
            let ab: Vec<F8> = a.iter().zip(&b).map(|(x, y)| *x + *y).collect();

            let mut fa = a.clone();
            ntt.forward(&mut fa);
            let mut fb = b.clone();
            ntt.forward(&mut fb);
            let mut fab = ab.clone();
            ntt.forward(&mut fab);

            for i in 0..n {
                assert_eq!(fa[i] + fb[i], fab[i], "linearity failed at k={k}, i={i}");
            }
        }
    }

    #[test]
    fn nonzero_beta_roundtrip() {
        let mut rng = Rng::new(45);
        for beta_v in [0x01u8, 0x42, 0xCA, 0xFF] {
            let beta = F8(beta_v);
            for k in 1..=6 {
                let ntt = AdditiveNttGf8::new(k, beta);
                let original = rand_vec(&mut rng, 1 << k);
                let mut v = original.clone();
                ntt.forward(&mut v);
                ntt.inverse(&mut v);
                assert_eq!(v, original, "beta={beta_v:#x}, k={k}");
            }
        }
    }
}
