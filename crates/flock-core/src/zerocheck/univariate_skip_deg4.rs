use crate::field::{F8, F128, phi8};
use crate::ntt::AdditiveNttGf8;

use super::univariate_skip::build_eq;

pub const K_SKIP: usize = 6;
pub const K_V8: usize = 8;

pub const S_SIZE: usize = 1usize << K_SKIP;
pub const V8_SIZE: usize = 1usize << K_V8;
pub const LAMBDA4_SIZE: usize = V8_SIZE - S_SIZE;

pub fn round1_deg4_naive(
    a: &[bool],
    b: &[bool],
    c: &[bool],
    d: &[bool],
    z: &[bool],
    m: usize,
    r: &[F128],
) -> (Vec<F128>, Vec<F128>) {
    assert!(K_SKIP <= m, "K_SKIP must be ≤ m");
    let n = 1usize << m;
    assert_eq!(a.len(), n);
    assert_eq!(b.len(), n);
    assert_eq!(c.len(), n);
    assert_eq!(d.len(), n);
    assert_eq!(z.len(), n);
    assert_eq!(r.len(), m);

    let n_chunks_x = 1usize << (m - K_SKIP);

    let ntt_s = AdditiveNttGf8::new(K_SKIP, F8::ZERO);
    let ntt_v8 = AdditiveNttGf8::new(K_V8, F8::ZERO);

    let eq_full = build_eq(&r[K_SKIP..]);
    debug_assert_eq!(eq_full.len(), n_chunks_x);

    let mut p_abcd = vec![F128::ZERO; LAMBDA4_SIZE];
    let mut p_z = vec![F128::ZERO; LAMBDA4_SIZE];

    let mut a_col = vec![F8::ZERO; V8_SIZE];
    let mut b_col = vec![F8::ZERO; V8_SIZE];
    let mut c_col = vec![F8::ZERO; V8_SIZE];
    let mut d_col = vec![F8::ZERO; V8_SIZE];
    let mut z_col = vec![F8::ZERO; V8_SIZE];

    for x_rest in 0..n_chunks_x {
        let base = x_rest * S_SIZE;

        for s in 0..S_SIZE {
            a_col[s] = F8(a[base + s] as u8);
            b_col[s] = F8(b[base + s] as u8);
            c_col[s] = F8(c[base + s] as u8);
            d_col[s] = F8(d[base + s] as u8);
            z_col[s] = F8(z[base + s] as u8);
        }
        for s in S_SIZE..V8_SIZE {
            a_col[s] = F8::ZERO;
            b_col[s] = F8::ZERO;
            c_col[s] = F8::ZERO;
            d_col[s] = F8::ZERO;
            z_col[s] = F8::ZERO;
        }

        for col in [&mut a_col, &mut b_col, &mut c_col, &mut d_col, &mut z_col] {
            ntt_s.inverse(&mut col[..S_SIZE]);
            ntt_v8.forward(col);
        }

        let eq_x = eq_full[x_rest];
        for i in 0..LAMBDA4_SIZE {
            let lam = S_SIZE + i;
            let abcd = a_col[lam] * b_col[lam] * c_col[lam] * d_col[lam];
            p_abcd[i] += eq_x * phi8(abcd);
            p_z[i] += eq_x * phi8(z_col[lam]);
        }
    }

    (p_abcd, p_z)
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
        fn bits(&mut self, n: usize) -> Vec<bool> {
            (0..n).map(|_| self.next_u64() & 1 == 1).collect()
        }
        fn f128(&mut self) -> F128 {
            F128 {
                lo: self.next_u64(),
                hi: self.next_u64(),
            }
        }
    }

    #[test]
    fn ntt_extend_roundtrips_on_s() {
        let ntt_s = AdditiveNttGf8::new(K_SKIP, F8::ZERO);
        let ntt_v8 = AdditiveNttGf8::new(K_V8, F8::ZERO);

        let mut rng = Rng::new(0xA17);
        let original_bits: Vec<F8> = (0..S_SIZE).map(|_| F8(rng.next_u64() as u8 & 1)).collect();

        let mut buf = vec![F8::ZERO; V8_SIZE];
        buf[..S_SIZE].copy_from_slice(&original_bits);
        ntt_s.inverse(&mut buf[..S_SIZE]);
        ntt_v8.forward(&mut buf);

        for i in 0..S_SIZE {
            assert_eq!(
                buf[i], original_bits[i],
                "NTT-extend on S mismatch at i={i}"
            );
        }
    }

    #[test]
    fn round1_deg4_output_shape() {
        let m = K_SKIP + 1;
        let mut rng = Rng::new(0xBEEF);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c = rng.bits(1 << m);
        let d = rng.bits(1 << m);
        let z = rng.bits(1 << m);
        let r: Vec<F128> = (0..m).map(|_| rng.f128()).collect();
        let (p_abcd, p_z) = round1_deg4_naive(&a, &b, &c, &d, &z, m, &r);
        assert_eq!(p_abcd.len(), LAMBDA4_SIZE);
        assert_eq!(p_z.len(), LAMBDA4_SIZE);
    }

    #[test]
    fn degree4_reduces_to_degree2_when_cd_one() {
        let m = K_SKIP + 1;
        let mut rng = Rng::new(123);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c = vec![true; 1 << m];
        let d = vec![true; 1 << m];
        let z = vec![false; 1 << m];
        let r: Vec<F128> = (0..m).map(|_| rng.f128()).collect();

        let (p_abcd, _p_z) = round1_deg4_naive(&a, &b, &c, &d, &z, m, &r);

        let ntt_s = AdditiveNttGf8::new(K_SKIP, F8::ZERO);
        let ntt_v8 = AdditiveNttGf8::new(K_V8, F8::ZERO);
        let n_chunks_x = 1usize << (m - K_SKIP);
        let eq_full = build_eq(&r[K_SKIP..]);
        let mut p_ab_ref = vec![F128::ZERO; LAMBDA4_SIZE];
        let mut a_col = vec![F8::ZERO; V8_SIZE];
        let mut b_col = vec![F8::ZERO; V8_SIZE];
        for x_rest in 0..n_chunks_x {
            let base = x_rest * S_SIZE;
            for col in [&mut a_col, &mut b_col] {
                for v in col.iter_mut() {
                    *v = F8::ZERO;
                }
            }
            for s in 0..S_SIZE {
                a_col[s] = F8(a[base + s] as u8);
                b_col[s] = F8(b[base + s] as u8);
            }
            for col in [&mut a_col, &mut b_col] {
                ntt_s.inverse(&mut col[..S_SIZE]);
                ntt_v8.forward(col);
            }
            let eq_x = eq_full[x_rest];
            for i in 0..LAMBDA4_SIZE {
                let lam = S_SIZE + i;
                p_ab_ref[i] += eq_x * phi8(a_col[lam] * b_col[lam]);
            }
        }

        for i in 0..LAMBDA4_SIZE {
            assert_eq!(
                p_abcd[i],
                p_ab_ref[i],
                "degree-4 vs degree-2 mismatch at i={i} (λ={})",
                S_SIZE + i
            );
        }
    }

    #[test]
    fn linear_only_path() {
        let m = K_SKIP + 2;
        let mut rng = Rng::new(0xC0DE);
        let a = vec![false; 1 << m];
        let b = vec![false; 1 << m];
        let c = vec![false; 1 << m];
        let d = vec![false; 1 << m];
        let z = rng.bits(1 << m);
        let r: Vec<F128> = (0..m).map(|_| rng.f128()).collect();
        let (p_abcd, p_z) = round1_deg4_naive(&a, &b, &c, &d, &z, m, &r);

        for v in &p_abcd {
            assert_eq!(*v, F128::ZERO);
        }

        let ntt_s = AdditiveNttGf8::new(K_SKIP, F8::ZERO);
        let ntt_v8 = AdditiveNttGf8::new(K_V8, F8::ZERO);
        let n_chunks_x = 1usize << (m - K_SKIP);
        let eq_full = build_eq(&r[K_SKIP..]);
        let mut p_z_ref = vec![F128::ZERO; LAMBDA4_SIZE];
        let mut z_col = vec![F8::ZERO; V8_SIZE];
        for x_rest in 0..n_chunks_x {
            let base = x_rest * S_SIZE;
            for v in z_col.iter_mut() {
                *v = F8::ZERO;
            }
            for s in 0..S_SIZE {
                z_col[s] = F8(z[base + s] as u8);
            }
            ntt_s.inverse(&mut z_col[..S_SIZE]);
            ntt_v8.forward(&mut z_col);
            let eq_x = eq_full[x_rest];
            for i in 0..LAMBDA4_SIZE {
                p_z_ref[i] += eq_x * phi8(z_col[S_SIZE + i]);
            }
        }
        for i in 0..LAMBDA4_SIZE {
            assert_eq!(p_z[i], p_z_ref[i], "p_z mismatch at i={i}");
        }
    }
}
