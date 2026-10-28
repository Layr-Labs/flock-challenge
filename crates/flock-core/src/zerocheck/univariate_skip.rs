use crate::field::{F8, F128, mul_by_x, phi8};
use crate::ntt::{AdditiveNttGf8, InvNttTableByteSingleGf8};

pub fn build_eq(r: &[F128]) -> Vec<F128> {
    let n = r.len();
    let mut t = crate::alloc_uninit_f128_vec(1usize << n);
    t[0] = F128::ONE;
    if n == 0 {
        return t;
    }
    let r0 = r[0];
    t[1] = r0;
    t[0] = F128::ONE + r0;

    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    unsafe {
        use core::arch::x86_64::*;
        let ptr = t.as_mut_ptr() as *mut __m128i;
        for (i, &r_i) in r.iter().enumerate().skip(1) {
            let half = 1usize << i;
            let vr = _mm_set_epi64x(r_i.hi as i64, r_i.lo as i64);
            let (rr0, rr1) = crate::gf2_kernels::k0_f128_pre_reduce_const_m128i(vr);
            let mut x = half;
            while x >= 2 {
                x -= 2;
                let old0 = _mm_loadu_si128(ptr.add(x));
                let old1 = _mm_loadu_si128(ptr.add(x + 1));
                let (hi0, hi1) = crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                    old0, old1, rr0, rr1, rr0, rr1,
                );
                _mm_storeu_si128(ptr.add(x + half), hi0);
                _mm_storeu_si128(ptr.add(x + 1 + half), hi1);
                _mm_storeu_si128(ptr.add(x), _mm_xor_si128(old0, hi0));
                _mm_storeu_si128(ptr.add(x + 1), _mm_xor_si128(old1, hi1));
            }
        }
        return t;
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        use core::arch::aarch64::*;
        let ptr = t.as_mut_ptr() as *mut u64;
        for (i, &r_i) in r.iter().enumerate().skip(1) {
            let half = 1usize << i;
            let vr = vcombine_u64(vcreate_u64(r_i.lo), vcreate_u64(r_i.hi));
            let (rr0, rr1) = crate::gf2_kernels::k0_f128_pre_reduce_const_neon(vr);
            let mut x = half;
            while x >= 2 {
                x -= 2;
                let old0 = vld1q_u64(ptr.add(2 * x));
                let old1 = vld1q_u64(ptr.add(2 * (x + 1)));
                let (hi0, hi1) =
                    crate::gf2_kernels::k0_f128_mul_const_vec2_neon(old0, old1, rr0, rr1, rr0, rr1);
                vst1q_u64(ptr.add(2 * (x + half)), hi0);
                vst1q_u64(ptr.add(2 * (x + 1 + half)), hi1);
                vst1q_u64(ptr.add(2 * x), veorq_u64(old0, hi0));
                vst1q_u64(ptr.add(2 * (x + 1)), veorq_u64(old1, hi1));
            }
        }
        return t;
    }

    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        all(target_arch = "aarch64", target_feature = "aes")
    )))]
    {
        for (i, &r_i) in r.iter().enumerate().skip(1) {
            for x in (0..(1usize << i)).rev() {
                let old = t[x];
                let hi_val = old * r_i;
                t[x | (1 << i)] = hi_val;
                t[x] = old + hi_val;
            }
        }
        t
    }
}

pub fn round1_naive(
    a: &[bool],
    b: &[bool],
    c: &[bool],
    m: usize,
    k_skip: usize,
    r: &[F128],
) -> (Vec<F128>, Vec<F128>) {
    assert!(k_skip <= m, "k_skip must be ≤ m");
    assert_eq!(a.len(), 1usize << m);
    assert_eq!(b.len(), 1usize << m);
    assert_eq!(c.len(), 1usize << m);
    assert_eq!(r.len(), m);

    let ell = 1usize << k_skip;
    let n_chunks_x = 1usize << (m - k_skip);

    let ntt_s = AdditiveNttGf8::new(k_skip, F8::ZERO);
    let ntt_l = AdditiveNttGf8::new(k_skip, F8(ell as u8));

    let eq_full = build_eq(&r[k_skip..]);

    let mut p_ab = vec![F128::ZERO; ell];
    let mut p_c = vec![F128::ZERO; ell];

    let mut a_col = vec![F8::ZERO; ell];
    let mut b_col = vec![F8::ZERO; ell];
    let mut c_col = vec![F8::ZERO; ell];

    for x_rest in 0..n_chunks_x {
        let base = x_rest * ell;
        for s in 0..ell {
            a_col[s] = F8(a[base + s] as u8);
            b_col[s] = F8(b[base + s] as u8);
            c_col[s] = F8(c[base + s] as u8);
        }
        ntt_s.inverse(&mut a_col);
        ntt_l.forward(&mut a_col);
        ntt_s.inverse(&mut b_col);
        ntt_l.forward(&mut b_col);
        ntt_s.inverse(&mut c_col);
        ntt_l.forward(&mut c_col);

        let eq_x = eq_full[x_rest];
        for i in 0..ell {
            let ab = a_col[i] * b_col[i];
            p_ab[i] += eq_x * phi8(ab);
            p_c[i] += eq_x * phi8(c_col[i]);
        }
    }

    (p_ab, p_c)
}

pub fn pack_bits(bits: &[bool]) -> Vec<u8> {
    let n_bytes = bits.len().div_ceil(8);
    let mut out = vec![0u8; n_bytes];
    let bytes: &[u8] =
        unsafe { core::slice::from_raw_parts(bits.as_ptr() as *const u8, bits.len()) };
    let full_bytes = bytes.len() / 8;
    for byte_idx in 0..full_bytes {
        let base = byte_idx * 8;
        let x = u64::from_le_bytes(bytes[base..base + 8].try_into().unwrap());
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        let packed = unsafe { crate::gf2_kernels::gf2_kern_pack8_bool_bytes(x) as u8 };
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let packed = (x.wrapping_mul(0x0102_0408_1020_4080) >> 56) as u8;
        out[byte_idx] = packed;
    }
    for byte_idx in full_bytes..n_bytes {
        let mut byte = 0u8;
        let base = byte_idx * 8;
        for j in 0..8 {
            let bit_idx = base + j;
            if bit_idx < bits.len() && bits[bit_idx] {
                byte |= 1u8 << j;
            }
        }
        out[byte_idx] = byte;
    }
    out
}

#[derive(Clone, Debug)]
pub struct SplitEqGhash {
    pub n_lo: usize,
    pub n_hi: usize,
    pub lo: Vec<F128>,
    pub hi: Vec<F128>,
}

impl SplitEqGhash {
    pub const MAX_N_HI: usize = 7;

    pub const MAX_N_LO: usize = 12;

    pub fn new(r: &[F128]) -> Self {
        let n = r.len();
        let n_hi = n.min(Self::MAX_N_HI).max(n.saturating_sub(Self::MAX_N_LO));
        Self::with_n_hi(r, n_hi)
    }

    pub fn with_n_hi(r: &[F128], n_hi: usize) -> Self {
        let n = r.len();
        let n_hi = n_hi.min(n);
        let n_lo = n - n_hi;
        Self {
            n_lo,
            n_hi,
            lo: build_eq(&r[..n_lo]),
            hi: build_eq(&r[n_lo..]),
        }
    }
}

pub fn ntt_extend_f128_vec_ghash(in_s: &[F128], inv_table: &InvNttTableByteSingleGf8) -> Vec<F128> {
    let ell = inv_table.ell;
    assert_eq!(in_s.len(), ell);
    assert_eq!(ell, 1usize << inv_table.k);

    let mut out = vec![F128::ZERO; ell];
    let n_chunks = inv_table.n_chunks;

    let mut gamma_pow = [F128::ZERO; 128];
    gamma_pow[0] = F128::ONE;
    for b in 1..128 {
        gamma_pow[b] = mul_by_x(gamma_pow[b - 1]);
    }

    let mut input_bits = vec![0u8; n_chunks];
    let mut out_bytes = vec![F8::ZERO; ell];

    let mut transposed_lo = vec![[0u8; 64]; n_chunks];
    let mut transposed_hi = vec![[0u8; 64]; n_chunks];
    for c in 0..n_chunks {
        let base = c * 8;
        let lo_lanes: [u64; 8] = core::array::from_fn(|j| in_s[base + j].lo);
        let hi_lanes: [u64; 8] = core::array::from_fn(|j| in_s[base + j].hi);
        crate::bits::transpose_8_u64s_to_64_bytes(&lo_lanes, &mut transposed_lo[c]);
        crate::bits::transpose_8_u64s_to_64_bytes(&hi_lanes, &mut transposed_hi[c]);
    }

    for b in 0..128 {
        if b < 64 {
            for c in 0..n_chunks {
                input_bits[c] = transposed_lo[c][b];
            }
        } else {
            let b_hi = b - 64;
            for c in 0..n_chunks {
                input_bits[c] = transposed_hi[c][b_hi];
            }
        }

        inv_table.apply(&input_bits, &mut out_bytes);

        let g_b = gamma_pow[b];
        for lambda in 0..ell {
            out[lambda] += g_b * phi8(out_bytes[lambda]);
        }
    }

    out
}

pub fn round1_extract_c(
    a: &[bool],
    b: &[bool],
    c: &[bool],
    m: usize,
    k_skip: usize,
    r: &[F128],
    inv_table: &InvNttTableByteSingleGf8,
) -> (Vec<F128>, Vec<F128>) {
    assert_eq!(a.len(), 1usize << m);
    assert_eq!(b.len(), 1usize << m);
    assert_eq!(c.len(), 1usize << m);
    let a_packed = pack_bits(a);
    let b_packed = pack_bits(b);
    let c_packed = pack_bits(c);
    round1_extract_c_packed(&a_packed, &b_packed, &c_packed, m, k_skip, r, inv_table)
}

pub fn round1_extract_c_packed(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    k_skip: usize,
    r: &[F128],
    inv_table: &InvNttTableByteSingleGf8,
) -> (Vec<F128>, Vec<F128>) {
    assert!(k_skip <= m);
    let total_bytes = (1usize << m) / 8;
    assert_eq!(a_packed.len(), total_bytes);
    assert_eq!(b_packed.len(), total_bytes);
    assert_eq!(c_packed.len(), total_bytes);
    assert_eq!(r.len(), m);
    assert_eq!(inv_table.k, k_skip);

    let ell = 1usize << k_skip;
    let n_chunks = ell / 8;

    let eq = SplitEqGhash::new(&r[k_skip..]);
    let lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;

    let mut res_ab = vec![F128::ZERO; ell];
    let mut res_c_s = vec![F128::ZERO; ell];

    let mut partial_ab = vec![F128::ZERO; ell];
    let mut partial_c = vec![F128::ZERO; ell];

    let mut a_col = vec![F8::ZERO; ell];
    let mut b_col = vec![F8::ZERO; ell];

    for x_hi in 0..hi_size {
        partial_ab.iter_mut().for_each(|p| *p = F128::ZERO);
        partial_c.iter_mut().for_each(|p| *p = F128::ZERO);

        for x_lo in 0..lo_size {
            let x_rest = (x_hi << eq.n_lo) | x_lo;
            let chunk_offset = x_rest * n_chunks;

            inv_table.apply(&a_packed[chunk_offset..chunk_offset + n_chunks], &mut a_col);
            inv_table.apply(&b_packed[chunk_offset..chunk_offset + n_chunks], &mut b_col);

            let eq_lo = eq.lo[x_lo];

            for lambda in 0..ell {
                let ab = a_col[lambda] * b_col[lambda];
                partial_ab[lambda] += eq_lo * phi8(ab);
            }

            for s in 0..ell {
                let c_bit = (c_packed[chunk_offset + s / 8] >> (s % 8)) & 1;
                if c_bit != 0 {
                    partial_c[s] += eq_lo;
                }
            }
        }

        let eq_hi = eq.hi[x_hi];
        crate::field::f128_slice::scalar_mul_add_assign(&mut res_ab, &partial_ab, eq_hi);
        crate::field::f128_slice::scalar_mul_add_assign(&mut res_c_s, &partial_c, eq_hi);
    }

    let res_c_lifted = ntt_extend_f128_vec_ghash(&res_c_s, inv_table);

    (res_ab, res_c_lifted)
}

pub fn round1_extract_c_packed_with_s_hat_v(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    k_skip: usize,
    r: &[F128],
    inv_table: &InvNttTableByteSingleGf8,
) -> (Vec<F128>, Vec<F128>, Vec<F128>) {
    assert!(k_skip <= m);
    let total_bytes = (1usize << m) / 8;
    assert_eq!(a_packed.len(), total_bytes);
    assert_eq!(b_packed.len(), total_bytes);
    assert_eq!(c_packed.len(), total_bytes);
    assert_eq!(r.len(), m);
    assert_eq!(inv_table.k, k_skip);

    let ell = 1usize << k_skip;
    let n_chunks = ell / 8;

    let eq = SplitEqGhash::new(&r[k_skip..]);
    let lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;

    let mut res_ab = vec![F128::ZERO; ell];
    let mut res_c_s_0 = vec![F128::ZERO; ell];
    let mut res_c_s_1 = vec![F128::ZERO; ell];

    let mut partial_ab = vec![F128::ZERO; ell];
    let mut partial_c_0 = vec![F128::ZERO; ell];
    let mut partial_c_1 = vec![F128::ZERO; ell];

    let mut a_col = vec![F8::ZERO; ell];
    let mut b_col = vec![F8::ZERO; ell];

    for x_hi in 0..hi_size {
        partial_ab.iter_mut().for_each(|p| *p = F128::ZERO);
        partial_c_0.iter_mut().for_each(|p| *p = F128::ZERO);
        partial_c_1.iter_mut().for_each(|p| *p = F128::ZERO);

        for x_lo in 0..lo_size {
            let x_rest = (x_hi << eq.n_lo) | x_lo;
            let chunk_offset = x_rest * n_chunks;
            let b_7 = x_rest & 1;

            inv_table.apply(&a_packed[chunk_offset..chunk_offset + n_chunks], &mut a_col);
            inv_table.apply(&b_packed[chunk_offset..chunk_offset + n_chunks], &mut b_col);

            let eq_lo = eq.lo[x_lo];

            for lambda in 0..ell {
                let ab = a_col[lambda] * b_col[lambda];
                partial_ab[lambda] += eq_lo * phi8(ab);
            }

            let target = if b_7 == 0 {
                &mut partial_c_0
            } else {
                &mut partial_c_1
            };
            for s in 0..ell {
                let c_bit = (c_packed[chunk_offset + s / 8] >> (s % 8)) & 1;
                if c_bit != 0 {
                    target[s] += eq_lo;
                }
            }
        }

        let eq_hi = eq.hi[x_hi];
        crate::field::f128_slice::scalar_mul_add_assign(&mut res_ab, &partial_ab, eq_hi);
        crate::field::f128_slice::scalar_mul_add_assign(&mut res_c_s_0, &partial_c_0, eq_hi);
        crate::field::f128_slice::scalar_mul_add_assign(&mut res_c_s_1, &partial_c_1, eq_hi);
    }

    let mut res_c_s = res_c_s_0.clone();
    crate::field::f128_slice::xor_assign(&mut res_c_s, &res_c_s_1);
    let res_c_lifted = ntt_extend_f128_vec_ghash(&res_c_s, inv_table);

    let inv_zero = (F128::ONE + r[k_skip]).inv();
    let inv_one = r[k_skip].inv();
    let mut s_hat_v_c = vec![F128::ZERO; 2 * ell];
    crate::field::f128_slice::scalar_mul_into(&mut s_hat_v_c[..ell], &res_c_s_0, inv_zero);
    crate::field::f128_slice::scalar_mul_into(&mut s_hat_v_c[ell..], &res_c_s_1, inv_one);

    (res_ab, res_c_lifted, s_hat_v_c)
}

pub fn round1_evals_on_s(
    a: &[bool],
    b: &[bool],
    c: &[bool],
    m: usize,
    k_skip: usize,
    r: &[F128],
) -> (Vec<F128>, Vec<F128>) {
    assert!(k_skip <= m);
    assert_eq!(a.len(), 1usize << m);
    assert_eq!(b.len(), 1usize << m);
    assert_eq!(c.len(), 1usize << m);
    assert_eq!(r.len(), m);

    let ell = 1usize << k_skip;
    let n_chunks_x = 1usize << (m - k_skip);
    let eq_full = build_eq(&r[k_skip..]);

    let mut p_ab = vec![F128::ZERO; ell];
    let mut p_c = vec![F128::ZERO; ell];

    for x_rest in 0..n_chunks_x {
        let base = x_rest * ell;
        let eq_x = eq_full[x_rest];
        for s in 0..ell {
            if a[base + s] && b[base + s] {
                p_ab[s] += eq_x;
            }
            if c[base + s] {
                p_c[s] += eq_x;
            }
        }
    }

    (p_ab, p_c)
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
        fn bit(&mut self) -> bool {
            (self.next_u64() & 1) != 0
        }
        fn f128(&mut self) -> F128 {
            F128 {
                lo: self.next_u64(),
                hi: self.next_u64(),
            }
        }
        fn bits(&mut self, n: usize) -> Vec<bool> {
            (0..n).map(|_| self.bit()).collect()
        }
        fn f128_vec(&mut self, n: usize) -> Vec<F128> {
            (0..n).map(|_| self.f128()).collect()
        }
    }

    #[test]
    fn build_eq_basic() {
        assert_eq!(build_eq(&[]), vec![F128::ONE]);
        let r0 = F128 {
            lo: 0xCAFEBABE,
            hi: 0x12345678,
        };
        let t = build_eq(&[r0]);
        assert_eq!(t.len(), 2);
        assert_eq!(t[0], F128::ONE + r0);
        assert_eq!(t[1], r0);
        let n = 5;
        let mut rng = Rng::new(99);
        let r = rng.f128_vec(n);
        let t = build_eq(&r);
        let sum: F128 = t.iter().copied().fold(F128::ZERO, |a, b| a + b);
        assert_eq!(sum, F128::ONE, "Σ_x eq(r, x) should be 1");
    }

    #[test]
    fn round1_output_shape() {
        let m = 8;
        let k_skip = 3;
        let ell = 1usize << k_skip;
        let mut rng = Rng::new(1);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c = rng.bits(1 << m);
        let r = rng.f128_vec(m);
        let (p_ab, p_c) = round1_naive(&a, &b, &c, m, k_skip, &r);
        assert_eq!(p_ab.len(), ell);
        assert_eq!(p_c.len(), ell);
    }

    #[test]
    fn round1_all_zero_witness_gives_zero_message() {
        let m = 7;
        let k_skip = 3;
        let mut rng = Rng::new(2);
        let r = rng.f128_vec(m);
        let zeros = vec![false; 1 << m];
        let (p_ab, p_c) = round1_naive(&zeros, &zeros, &zeros, m, k_skip, &r);
        assert!(p_ab.iter().all(|v| v.is_zero()));
        assert!(p_c.iter().all(|v| v.is_zero()));
    }

    #[test]
    fn round1_deterministic() {
        let m = 6;
        let k_skip = 3;
        let mut rng = Rng::new(3);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c = rng.bits(1 << m);
        let r = rng.f128_vec(m);
        let out1 = round1_naive(&a, &b, &c, m, k_skip, &r);
        let out2 = round1_naive(&a, &b, &c, m, k_skip, &r);
        assert_eq!(out1, out2);
    }

    #[test]
    fn round1_p_c_is_xor_linear_in_c() {
        let m = 6;
        let k_skip = 3;
        let mut rng = Rng::new(4);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c1 = rng.bits(1 << m);
        let c2 = rng.bits(1 << m);
        let c_sum: Vec<bool> = c1.iter().zip(&c2).map(|(x, y)| x ^ y).collect();
        let r = rng.f128_vec(m);

        let (ab1, pc1) = round1_naive(&a, &b, &c1, m, k_skip, &r);
        let (ab2, pc2) = round1_naive(&a, &b, &c2, m, k_skip, &r);
        let (ab_sum, pc_sum) = round1_naive(&a, &b, &c_sum, m, k_skip, &r);

        assert_eq!(ab1, ab_sum);
        assert_eq!(ab2, ab_sum);
        for i in 0..pc1.len() {
            assert_eq!(pc1[i] + pc2[i], pc_sum[i]);
        }
    }

    #[test]
    fn round1_at_s_zero_for_honest_witness() {
        let m = 8;
        let k_skip = 3;
        let mut rng = Rng::new(5);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
        let r = rng.f128_vec(m);

        let (p_ab_s, p_c_s) = round1_evals_on_s(&a, &b, &c, m, k_skip, &r);
        for s in 0..p_ab_s.len() {
            assert_eq!(
                p_ab_s[s] + p_c_s[s],
                F128::ZERO,
                "P at S should be 0 for honest witness, but failed at s={s}"
            );
        }
    }

    #[test]
    fn round1_at_s_nonzero_for_random_witness() {
        let m = 8;
        let k_skip = 3;
        let mut rng = Rng::new(6);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c = rng.bits(1 << m);
        let r = rng.f128_vec(m);

        let (p_ab_s, p_c_s) = round1_evals_on_s(&a, &b, &c, m, k_skip, &r);
        let combined: Vec<F128> = p_ab_s.iter().zip(&p_c_s).map(|(x, y)| *x + *y).collect();
        let nonzero = combined.iter().any(|v| !v.is_zero());
        assert!(nonzero, "P at S should be nonzero for a random witness");
    }

    fn make_inv_table(k_skip: usize) -> InvNttTableByteSingleGf8 {
        let ntt_s = AdditiveNttGf8::new(k_skip, F8::ZERO);
        let ntt_l = AdditiveNttGf8::new(k_skip, F8(1u8 << k_skip));
        InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l)
    }

    #[test]
    fn extract_c_output_shape() {
        let m = 7;
        let k_skip = 3;
        let mut rng = Rng::new(10);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c = rng.bits(1 << m);
        let r = rng.f128_vec(m);
        let table = make_inv_table(k_skip);
        let (ab, c_l) = round1_extract_c(&a, &b, &c, m, k_skip, &r, &table);
        assert_eq!(ab.len(), 1usize << k_skip);
        assert_eq!(c_l.len(), 1usize << k_skip);
    }

    #[test]
    fn extract_c_deterministic() {
        let m = 6;
        let k_skip = 3;
        let mut rng = Rng::new(11);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c = rng.bits(1 << m);
        let r = rng.f128_vec(m);
        let table = make_inv_table(k_skip);
        let out1 = round1_extract_c(&a, &b, &c, m, k_skip, &r, &table);
        let out2 = round1_extract_c(&a, &b, &c, m, k_skip, &r, &table);
        assert_eq!(out1, out2);
    }

    #[test]
    fn extract_c_with_s_hat_v_matches_original_wire() {
        for &(m, k_skip) in &[(4, 3), (5, 3), (6, 3), (7, 4), (8, 3), (9, 6)] {
            let mut rng = Rng::new(0xBEEF + m as u64 * 7 + k_skip as u64);
            let a = pack_bits(&rng.bits(1 << m));
            let b = pack_bits(&rng.bits(1 << m));
            let c = pack_bits(&rng.bits(1 << m));
            let r = rng.f128_vec(m);
            let table = make_inv_table(k_skip);

            let (ab_old, c_old) = round1_extract_c_packed(&a, &b, &c, m, k_skip, &r, &table);
            let (ab_new, c_new, _) =
                round1_extract_c_packed_with_s_hat_v(&a, &b, &c, m, k_skip, &r, &table);

            assert_eq!(ab_old, ab_new, "res_ab mismatch m={m} k_skip={k_skip}");
            assert_eq!(c_old, c_new, "res_c_lifted mismatch m={m} k_skip={k_skip}");
        }
    }

    #[test]
    fn extract_c_with_s_hat_v_matches_fold_1b_rows() {
        use crate::pcs::pack::pack_witness;
        use crate::pcs::ring_switch::fold_1b_rows_naive;
        const K_SKIP: usize = 6;
        for &m in &[8usize, 9, 10, 11] {
            let mut rng = Rng::new(0xC0FFEE_u64.wrapping_add(m as u64));
            let z_bits = rng.bits(1 << m);
            let a = pack_bits(&rng.bits(1 << m));
            let b = pack_bits(&rng.bits(1 << m));
            let c = pack_bits(&z_bits);
            let r = rng.f128_vec(m);
            let table = make_inv_table(K_SKIP);

            let (_, _, s_hat_v_c) =
                round1_extract_c_packed_with_s_hat_v(&a, &b, &c, m, K_SKIP, &r, &table);

            let packed_c = pack_witness(&z_bits, m);
            let suffix = &r[K_SKIP + 1..];
            let suffix_tensor = build_eq(suffix);
            let want = fold_1b_rows_naive(&packed_c, &suffix_tensor);

            assert_eq!(s_hat_v_c.len(), want.len(), "length mismatch at m={m}");
            assert_eq!(s_hat_v_c, want, "s_hat_v_c mismatch at m={m}");
        }
    }

    #[test]
    fn extract_c_matches_naive() {
        for &(m, k_skip) in &[(4, 3), (5, 3), (6, 3), (7, 4), (8, 3), (9, 6)] {
            let mut rng = Rng::new(100 + m as u64 * 10 + k_skip as u64);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            let c = rng.bits(1 << m);
            let r = rng.f128_vec(m);
            let table = make_inv_table(k_skip);

            let (naive_ab, naive_c) = round1_naive(&a, &b, &c, m, k_skip, &r);
            let (opt_ab, opt_c) = round1_extract_c(&a, &b, &c, m, k_skip, &r, &table);

            assert_eq!(naive_ab, opt_ab, "AB mismatch at m={m}, k_skip={k_skip}");
            assert_eq!(naive_c, opt_c, "C mismatch at m={m}, k_skip={k_skip}");
        }
    }

    #[test]
    fn extract_c_honest_witness_combined_matches_naive() {
        let m = 8;
        let k_skip = 3;
        let mut rng = Rng::new(200);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c: Vec<bool> = a.iter().zip(&b).map(|(x, y)| *x & *y).collect();
        let r = rng.f128_vec(m);
        let table = make_inv_table(k_skip);

        let (naive_ab, naive_c) = round1_naive(&a, &b, &c, m, k_skip, &r);
        let (opt_ab, opt_c) = round1_extract_c(&a, &b, &c, m, k_skip, &r, &table);

        for i in 0..naive_ab.len() {
            assert_eq!(naive_ab[i] + naive_c[i], opt_ab[i] + opt_c[i]);
        }
    }

    #[test]
    fn split_eq_basic() {
        let mut rng = Rng::new(300);
        let n = 6;
        let r = rng.f128_vec(n);
        let full = build_eq(&r);
        let eq = SplitEqGhash::new(&r);
        assert_eq!(eq.n_lo + eq.n_hi, n);
        for x in 0..(1 << n) {
            let x_lo = x & ((1 << eq.n_lo) - 1);
            let x_hi = x >> eq.n_lo;
            assert_eq!(eq.lo[x_lo] * eq.hi[x_hi], full[x]);
        }
    }

    #[test]
    fn ntt_extend_round_trips_naive_c_path() {
        let m = 6;
        let k_skip = 3;
        let mut rng = Rng::new(400);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c = rng.bits(1 << m);
        let r = rng.f128_vec(m);
        let table = make_inv_table(k_skip);
        let _ = (&a, &b);

        let (_, naive_c) = round1_naive(&a, &b, &c, m, k_skip, &r);

        let (_, opt_c) = round1_extract_c(&a, &b, &c, m, k_skip, &r, &table);

        assert_eq!(naive_c, opt_c);
    }

    #[test]
    fn round1_message_nonzero_for_random_witness() {
        let m = 8;
        let k_skip = 3;
        let mut rng = Rng::new(7);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c = rng.bits(1 << m);
        let r = rng.f128_vec(m);

        let (p_ab, p_c) = round1_naive(&a, &b, &c, m, k_skip, &r);
        assert!(p_ab.iter().any(|v| !v.is_zero()));
        assert!(p_c.iter().any(|v| !v.is_zero()));
    }
}
