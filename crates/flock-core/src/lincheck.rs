use crate::challenger::Challenger;
use crate::field::F128;
use crate::r1cs::SparseBinaryMatrix;
use crate::zerocheck::multilinear::lagrange_weights_naive;
use serde::{Deserialize, Serialize};
use std::sync::atomic::AtomicBool;

mod kernels;

#[cfg(target_arch = "x86_64")]
pub use kernels::partial_fold_packed_z_x86_tiled_padded;
#[cfg(target_arch = "aarch64")]
pub use kernels::{
    partial_fold_packed_z_neon_iblock_padded, partial_fold_packed_z_neon_oblock_padded,
    partial_fold_packed_z_neon_single, partial_fold_packed_z_neon_single_padded,
};

pub static FOLD_IBLOCK: AtomicBool = AtomicBool::new(false);

pub trait LincheckCircuit: Sync {
    fn n_cols(&self) -> usize;

    fn fold_alpha_batched(&self, alpha: F128, eq_inner: &[F128]) -> Vec<F128>;

    fn const_pin_col(&self) -> Option<usize> {
        None
    }
}

pub struct SparseMatrixCircuit<'a> {
    pub a_0: &'a SparseBinaryMatrix,
    pub b_0: &'a SparseBinaryMatrix,

    const_pin: Option<usize>,
}

impl<'a> SparseMatrixCircuit<'a> {
    pub fn new(a_0: &'a SparseBinaryMatrix, b_0: &'a SparseBinaryMatrix) -> Self {
        debug_assert_eq!(a_0.num_rows, b_0.num_rows);
        debug_assert_eq!(a_0.num_cols, b_0.num_cols);
        Self {
            a_0,
            b_0,
            const_pin: None,
        }
    }

    pub fn with_const_pin(mut self, const_pin: Option<usize>) -> Self {
        self.const_pin = const_pin;
        self
    }
}

impl<'a> LincheckCircuit for SparseMatrixCircuit<'a> {
    fn n_cols(&self) -> usize {
        self.a_0.num_cols
    }
    fn fold_alpha_batched(&self, alpha: F128, eq_inner: &[F128]) -> Vec<F128> {
        sparse_row_fold_alpha_batched(alpha, self.a_0, self.b_0, eq_inner)
    }
    fn const_pin_col(&self) -> Option<usize> {
        self.const_pin
    }
}

#[derive(Clone)]
pub struct CscCircuit {
    n_cols: usize,
    a_col_ptr: Vec<u32>,
    a_rows: Vec<u32>,
    b_col_ptr: Vec<u32>,
    b_rows: Vec<u32>,

    const_pin: Option<usize>,
}

fn csc_from_rows(m: &SparseBinaryMatrix) -> (Vec<u32>, Vec<u32>) {
    assert!(m.num_rows <= u32::MAX as usize);
    assert!(m.num_cols <= u32::MAX as usize);
    let mut col_ptr = vec![0u32; m.num_cols + 1];
    for row in &m.rows {
        for &c in row {
            col_ptr[c + 1] += 1;
        }
    }
    for c in 0..m.num_cols {
        col_ptr[c + 1] += col_ptr[c];
    }
    let mut next = col_ptr.clone();
    let mut rows_flat = vec![0u32; *col_ptr.last().unwrap() as usize];
    for (r, row) in m.rows.iter().enumerate() {
        for &c in row {
            rows_flat[next[c] as usize] = r as u32;
            next[c] += 1;
        }
    }
    (col_ptr, rows_flat)
}

impl std::fmt::Debug for CscCircuit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CscCircuit")
            .field("n_cols", &self.n_cols)
            .field("nnz_a", &self.a_rows.len())
            .field("nnz_b", &self.b_rows.len())
            .finish()
    }
}

impl CscCircuit {
    pub fn from_matrices(a_0: &SparseBinaryMatrix, b_0: &SparseBinaryMatrix) -> Self {
        assert_eq!(a_0.num_rows, b_0.num_rows);
        assert_eq!(a_0.num_cols, b_0.num_cols);
        let (a_col_ptr, a_rows) = csc_from_rows(a_0);
        let (b_col_ptr, b_rows) = csc_from_rows(b_0);
        Self {
            n_cols: a_0.num_cols,
            a_col_ptr,
            a_rows,
            b_col_ptr,
            b_rows,
            const_pin: None,
        }
    }

    pub fn with_const_pin(mut self, const_pin: Option<usize>) -> Self {
        self.const_pin = const_pin;
        self
    }
}

impl LincheckCircuit for CscCircuit {
    fn n_cols(&self) -> usize {
        self.n_cols
    }
    fn const_pin_col(&self) -> Option<usize> {
        self.const_pin
    }
    fn fold_alpha_batched(&self, alpha: F128, eq_inner: &[F128]) -> Vec<F128> {
        use rayon::prelude::*;
        assert_eq!(eq_inner.len(), self.n_cols);
        #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
        let (ar0, ar1) = unsafe {
            use core::arch::x86_64::*;
            let valpha = _mm_set_epi64x(alpha.hi as i64, alpha.lo as i64);
            crate::gf2_kernels::k0_f128_pre_reduce_const_m128i(valpha)
        };
        #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
        let (ar0, ar1) = unsafe {
            use core::arch::aarch64::*;
            let valpha = vcombine_u64(vcreate_u64(alpha.lo), vcreate_u64(alpha.hi));
            crate::gf2_kernels::k0_f128_pre_reduce_const_neon(valpha)
        };
        let one_col = |c: usize| {
            #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
            unsafe {
                use core::arch::x86_64::*;
                let eq_ptr = eq_inner.as_ptr() as *const __m128i;
                let a_slice =
                    &self.a_rows[self.a_col_ptr[c] as usize..self.a_col_ptr[c + 1] as usize];
                let mut sa0 = _mm_setzero_si128();
                let mut sa1 = _mm_setzero_si128();
                let mut sa2 = _mm_setzero_si128();
                let mut sa3 = _mm_setzero_si128();
                let mut ai = 0usize;
                let alen = a_slice.len();
                let ap = a_slice.as_ptr();
                while ai + 4 <= alen {
                    sa0 = _mm_xor_si128(sa0, _mm_load_si128(eq_ptr.add(*ap.add(ai) as usize)));
                    sa1 = _mm_xor_si128(sa1, _mm_load_si128(eq_ptr.add(*ap.add(ai + 1) as usize)));
                    sa2 = _mm_xor_si128(sa2, _mm_load_si128(eq_ptr.add(*ap.add(ai + 2) as usize)));
                    sa3 = _mm_xor_si128(sa3, _mm_load_si128(eq_ptr.add(*ap.add(ai + 3) as usize)));
                    ai += 4;
                }
                let mut sa = _mm_xor_si128(_mm_xor_si128(sa0, sa1), _mm_xor_si128(sa2, sa3));
                while ai < alen {
                    sa = _mm_xor_si128(sa, _mm_load_si128(eq_ptr.add(*ap.add(ai) as usize)));
                    ai += 1;
                }

                let b_slice =
                    &self.b_rows[self.b_col_ptr[c] as usize..self.b_col_ptr[c + 1] as usize];
                let mut sb0 = _mm_setzero_si128();
                let mut sb1 = _mm_setzero_si128();
                let mut sb2 = _mm_setzero_si128();
                let mut sb3 = _mm_setzero_si128();
                let mut bi = 0usize;
                let blen = b_slice.len();
                let bp = b_slice.as_ptr();
                while bi + 4 <= blen {
                    sb0 = _mm_xor_si128(sb0, _mm_load_si128(eq_ptr.add(*bp.add(bi) as usize)));
                    sb1 = _mm_xor_si128(sb1, _mm_load_si128(eq_ptr.add(*bp.add(bi + 1) as usize)));
                    sb2 = _mm_xor_si128(sb2, _mm_load_si128(eq_ptr.add(*bp.add(bi + 2) as usize)));
                    sb3 = _mm_xor_si128(sb3, _mm_load_si128(eq_ptr.add(*bp.add(bi + 3) as usize)));
                    bi += 4;
                }
                let mut sb = _mm_xor_si128(_mm_xor_si128(sb0, sb1), _mm_xor_si128(sb2, sb3));
                while bi < blen {
                    sb = _mm_xor_si128(sb, _mm_load_si128(eq_ptr.add(*bp.add(bi) as usize)));
                    bi += 1;
                }
                let prod = crate::gf2_kernels::k0_f128_mul_const_m128i(sa, ar0, ar1);
                core::mem::transmute::<__m128i, F128>(_mm_xor_si128(prod, sb))
            }
            #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
            unsafe {
                use core::arch::aarch64::*;
                let eq_ptr = eq_inner.as_ptr() as *const u64;
                let a_slice =
                    &self.a_rows[self.a_col_ptr[c] as usize..self.a_col_ptr[c + 1] as usize];
                let mut sa0 = vdupq_n_u64(0);
                let mut sa1 = vdupq_n_u64(0);
                let mut sa2 = vdupq_n_u64(0);
                let mut sa3 = vdupq_n_u64(0);
                let mut ai = 0usize;
                let alen = a_slice.len();
                let ap = a_slice.as_ptr();
                while ai + 4 <= alen {
                    sa0 = veorq_u64(sa0, vld1q_u64(eq_ptr.add(2 * (*ap.add(ai) as usize))));
                    sa1 = veorq_u64(sa1, vld1q_u64(eq_ptr.add(2 * (*ap.add(ai + 1) as usize))));
                    sa2 = veorq_u64(sa2, vld1q_u64(eq_ptr.add(2 * (*ap.add(ai + 2) as usize))));
                    sa3 = veorq_u64(sa3, vld1q_u64(eq_ptr.add(2 * (*ap.add(ai + 3) as usize))));
                    ai += 4;
                }
                let mut sa = veorq_u64(veorq_u64(sa0, sa1), veorq_u64(sa2, sa3));
                while ai < alen {
                    sa = veorq_u64(sa, vld1q_u64(eq_ptr.add(2 * (*ap.add(ai) as usize))));
                    ai += 1;
                }

                let b_slice =
                    &self.b_rows[self.b_col_ptr[c] as usize..self.b_col_ptr[c + 1] as usize];
                let mut sb0 = vdupq_n_u64(0);
                let mut sb1 = vdupq_n_u64(0);
                let mut sb2 = vdupq_n_u64(0);
                let mut sb3 = vdupq_n_u64(0);
                let mut bi = 0usize;
                let blen = b_slice.len();
                let bp = b_slice.as_ptr();
                while bi + 4 <= blen {
                    sb0 = veorq_u64(sb0, vld1q_u64(eq_ptr.add(2 * (*bp.add(bi) as usize))));
                    sb1 = veorq_u64(sb1, vld1q_u64(eq_ptr.add(2 * (*bp.add(bi + 1) as usize))));
                    sb2 = veorq_u64(sb2, vld1q_u64(eq_ptr.add(2 * (*bp.add(bi + 2) as usize))));
                    sb3 = veorq_u64(sb3, vld1q_u64(eq_ptr.add(2 * (*bp.add(bi + 3) as usize))));
                    bi += 4;
                }
                let mut sb = veorq_u64(veorq_u64(sb0, sb1), veorq_u64(sb2, sb3));
                while bi < blen {
                    sb = veorq_u64(sb, vld1q_u64(eq_ptr.add(2 * (*bp.add(bi) as usize))));
                    bi += 1;
                }
                let prod = crate::gf2_kernels::k0_f128_mul_const_neon(sa, ar0, ar1);
                let res = veorq_u64(prod, sb);
                return F128 {
                    lo: vgetq_lane_u64::<0>(res),
                    hi: vgetq_lane_u64::<1>(res),
                };
            }
            #[cfg(not(any(
                all(target_arch = "x86_64", target_feature = "pclmulqdq"),
                all(target_arch = "aarch64", target_feature = "aes")
            )))]
            {
                let mut sa = F128::ZERO;
                for &r in &self.a_rows[self.a_col_ptr[c] as usize..self.a_col_ptr[c + 1] as usize] {
                    sa += unsafe { *eq_inner.get_unchecked(r as usize) };
                }
                let mut sb = F128::ZERO;
                for &r in &self.b_rows[self.b_col_ptr[c] as usize..self.b_col_ptr[c + 1] as usize] {
                    sb += unsafe { *eq_inner.get_unchecked(r as usize) };
                }
                alpha * sa + sb
            }
        };
        if self.n_cols < SUMCHECK_PAR_THRESHOLD {
            return (0..self.n_cols).map(one_col).collect();
        }
        let mut out = crate::alloc_uninit_f128_vec(self.n_cols);
        out.par_chunks_mut(8).enumerate().for_each(|(ci, chunk)| {
            let base = ci * 8;
            for (i, slot) in chunk.iter_mut().enumerate() {
                *slot = one_col(base + i);
            }
        });
        out
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuirkyPoint {
    pub z_skip: F128,

    pub x_inner_rest: Vec<F128>,

    pub x_outer: Vec<F128>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LincheckProof {
    pub rounds: Vec<(F128, F128)>,

    pub z_partial: Vec<F128>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LincheckClaim {
    pub r_inner_skip: F128,

    pub r_inner_rest: Vec<F128>,

    pub w: F128,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifyError {
    BadVectorLength {
        which: &'static str,
        expected: usize,
        got: usize,
    },

    BadInnerRestLength {
        which: &'static str,
        expected: usize,
        got: usize,
    },

    BadOuterLength {
        which: &'static str,
        expected: usize,
        got: usize,
    },

    BadMatrixShape {
        which: &'static str,
        expected: usize,
        got_rows: usize,
        got_cols: usize,
    },

    KSkipExceedsKLog {
        k_skip: usize,
        k_log: usize,
    },

    ConsistencyFailed {
        which: &'static str,
    },
}

pub fn build_eq_table(point: &[F128]) -> Vec<F128> {
    use rayon::prelude::*;
    let d = point.len();

    let mut out = crate::alloc_uninit_f128_vec(1usize << d);
    out[0] = F128::ONE;
    if d == 0 {
        return out;
    }
    let r0 = point[0];
    out[1] = r0;
    out[0] = F128::ONE + r0;
    const CHUNK: usize = 4096;
    for (j, &r_j) in point.iter().enumerate().skip(1) {
        let len = 1usize << j;
        let (lo, hi_rest) = out.split_at_mut(len);
        let hi = &mut hi_rest[..len];

        if len < SUMCHECK_PAR_THRESHOLD {
            #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
            unsafe {
                use core::arch::x86_64::*;
                let lo_ptr = lo.as_mut_ptr() as *mut __m128i;
                let hi_ptr = hi.as_mut_ptr() as *mut __m128i;
                let vr = _mm_set_epi64x(r_j.hi as i64, r_j.lo as i64);
                let (rr0, rr1) = crate::gf2_kernels::k0_f128_pre_reduce_const_m128i(vr);
                let mut x = 0usize;
                while x + 2 <= len {
                    let old0 = _mm_loadu_si128(lo_ptr.add(x));
                    let old1 = _mm_loadu_si128(lo_ptr.add(x + 1));
                    let (h0, h1) = crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                        old0, old1, rr0, rr1, rr0, rr1,
                    );
                    _mm_storeu_si128(hi_ptr.add(x), h0);
                    _mm_storeu_si128(hi_ptr.add(x + 1), h1);
                    _mm_storeu_si128(lo_ptr.add(x), _mm_xor_si128(old0, h0));
                    _mm_storeu_si128(lo_ptr.add(x + 1), _mm_xor_si128(old1, h1));
                    x += 2;
                }
                continue;
            }
            #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
            unsafe {
                use core::arch::aarch64::*;
                let lo_ptr = lo.as_mut_ptr() as *mut u64;
                let hi_ptr = hi.as_mut_ptr() as *mut u64;
                let vr = vcombine_u64(vcreate_u64(r_j.lo), vcreate_u64(r_j.hi));
                let (rr0, rr1) = crate::gf2_kernels::k0_f128_pre_reduce_const_neon(vr);
                let mut x = 0usize;
                while x + 2 <= len {
                    let old0 = vld1q_u64(lo_ptr.add(2 * x));
                    let old1 = vld1q_u64(lo_ptr.add(2 * (x + 1)));
                    let (h0, h1) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                        old0, old1, rr0, rr1, rr0, rr1,
                    );
                    vst1q_u64(hi_ptr.add(2 * x), h0);
                    vst1q_u64(hi_ptr.add(2 * (x + 1)), h1);
                    vst1q_u64(lo_ptr.add(2 * x), veorq_u64(old0, h0));
                    vst1q_u64(lo_ptr.add(2 * (x + 1)), veorq_u64(old1, h1));
                    x += 2;
                }
                continue;
            }
            #[cfg(not(any(
                all(target_arch = "x86_64", target_feature = "pclmulqdq"),
                all(target_arch = "aarch64", target_feature = "aes")
            )))]
            for (lo_x, hi_x) in lo.iter_mut().zip(hi.iter_mut()) {
                let v = *lo_x;
                let vr = v * r_j;
                *hi_x = vr;
                *lo_x = v + vr;
            }
        } else {
            lo.par_chunks_mut(CHUNK)
                .zip(hi.par_chunks_mut(CHUNK))
                .for_each(|(lo_c, hi_c)| {
                    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
                    unsafe {
                        use core::arch::x86_64::*;
                        let n_c = lo_c.len();
                        let lo_ptr = lo_c.as_mut_ptr() as *mut __m128i;
                        let hi_ptr = hi_c.as_mut_ptr() as *mut __m128i;
                        let vr = _mm_set_epi64x(r_j.hi as i64, r_j.lo as i64);
                        let (rr0, rr1) = crate::gf2_kernels::k0_f128_pre_reduce_const_m128i(vr);
                        let mut x = 0usize;
                        while x + 2 <= n_c {
                            let old0 = _mm_loadu_si128(lo_ptr.add(x));
                            let old1 = _mm_loadu_si128(lo_ptr.add(x + 1));
                            let (h0, h1) = crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                                old0, old1, rr0, rr1, rr0, rr1,
                            );
                            _mm_storeu_si128(hi_ptr.add(x), h0);
                            _mm_storeu_si128(hi_ptr.add(x + 1), h1);
                            _mm_storeu_si128(lo_ptr.add(x), _mm_xor_si128(old0, h0));
                            _mm_storeu_si128(lo_ptr.add(x + 1), _mm_xor_si128(old1, h1));
                            x += 2;
                        }
                    }
                    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
                    unsafe {
                        use core::arch::aarch64::*;
                        let n_c = lo_c.len();
                        let lo_ptr = lo_c.as_mut_ptr() as *mut u64;
                        let hi_ptr = hi_c.as_mut_ptr() as *mut u64;
                        let vr = vcombine_u64(vcreate_u64(r_j.lo), vcreate_u64(r_j.hi));
                        let (rr0, rr1) = crate::gf2_kernels::k0_f128_pre_reduce_const_neon(vr);
                        let mut x = 0usize;
                        while x + 2 <= n_c {
                            let old0 = vld1q_u64(lo_ptr.add(2 * x));
                            let old1 = vld1q_u64(lo_ptr.add(2 * (x + 1)));
                            let (h0, h1) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                                old0, old1, rr0, rr1, rr0, rr1,
                            );
                            vst1q_u64(hi_ptr.add(2 * x), h0);
                            vst1q_u64(hi_ptr.add(2 * (x + 1)), h1);
                            vst1q_u64(lo_ptr.add(2 * x), veorq_u64(old0, h0));
                            vst1q_u64(lo_ptr.add(2 * (x + 1)), veorq_u64(old1, h1));
                            x += 2;
                        }
                        return;
                    }
                    #[cfg(not(any(
                        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
                        all(target_arch = "aarch64", target_feature = "aes")
                    )))]
                    for (lo_x, hi_x) in lo_c.iter_mut().zip(hi_c.iter_mut()) {
                        let v = *lo_x;
                        let vr = v * r_j;
                        *hi_x = vr;
                        *lo_x = v + vr;
                    }
                });
        }
    }
    out
}

const SPARSE_ROW_FOLD_PAR_THRESHOLD: usize = 1usize << 12;

pub fn sparse_row_fold(matrix: &SparseBinaryMatrix, eq_table: &[F128]) -> Vec<F128> {
    assert_eq!(
        eq_table.len(),
        matrix.num_rows,
        "eq_table length must match matrix row count"
    );
    let n_cols = matrix.num_cols;
    if matrix.rows.len() < SPARSE_ROW_FOLD_PAR_THRESHOLD {
        let mut out = vec![F128::ZERO; n_cols];
        for (row_idx, row) in matrix.rows.iter().enumerate() {
            let e = eq_table[row_idx];
            for &col in row {
                out[col] += e;
            }
        }
        out
    } else {
        use rayon::prelude::*;
        matrix
            .rows
            .par_iter()
            .enumerate()
            .fold(
                || vec![F128::ZERO; n_cols],
                |mut acc, (row_idx, row)| {
                    let e = eq_table[row_idx];
                    for &col in row {
                        acc[col] += e;
                    }
                    acc
                },
            )
            .reduce(
                || vec![F128::ZERO; n_cols],
                |mut a, b| {
                    for i in 0..n_cols {
                        a[i] += b[i];
                    }
                    a
                },
            )
    }
}

pub fn partial_fold_packed_z(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    let n_log = m - k_log;
    let k = 1usize << k_log;
    let n_outer = 1usize << n_log;
    assert_eq!(z_packed.len(), (1usize << m) / 8);
    assert_eq!(eq_outer.len(), n_outer);
    assert!(n_log >= 3, "need n_outer ≥ 8 for byte stripes");
    let n_stripes = n_outer / 8;

    let mut out = vec![F128::ZERO; k];
    for byte_idx in 0..n_stripes {
        let stripe = &z_packed[byte_idx * k..(byte_idx + 1) * k];
        for (i_inner, &byte) in stripe.iter().enumerate() {
            if byte == 0 {
                continue;
            }
            let mut bits = byte;
            while bits != 0 {
                let r = bits.trailing_zeros() as usize;
                let i_outer = 8 * byte_idx + r;
                out[i_inner] += eq_outer[i_outer];
                bits &= bits - 1;
            }
        }
    }
    out
}

pub fn partial_fold_packed_z_fast(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    let k = 1usize << k_log;
    partial_fold_packed_z_fast_padded(z_packed, m, k_log, k, eq_outer)
}

pub fn partial_fold_packed_z_fast_padded(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    use rayon::prelude::*;

    let n_log = m - k_log;
    let k = 1usize << k_log;
    let n_outer = 1usize << n_log;
    assert_eq!(z_packed.len(), (1usize << m) / 8);
    assert_eq!(eq_outer.len(), n_outer);
    assert!(n_log >= 3, "need n_outer ≥ 8 for byte stripes");
    assert!(useful_bits <= k);
    let n_stripes = n_outer / 8;

    let stripes_per_chunk = (n_stripes / 256).max(1);
    let bytes_per_chunk = stripes_per_chunk * k;

    z_packed
        .par_chunks(bytes_per_chunk)
        .enumerate()
        .fold(
            || vec![F128::ZERO; k],
            |mut acc, (chunk_idx, chunk_bytes)| {
                let stripe_start = chunk_idx * stripes_per_chunk;
                let mut table = vec![F128::ZERO; 256];
                for (rel_stripe, stripe) in chunk_bytes.chunks(k).enumerate() {
                    let byte_idx = stripe_start + rel_stripe;
                    build_sum_table(&eq_outer[8 * byte_idx..8 * byte_idx + 8], &mut table);
                    for (i_inner, &z_byte) in stripe[..useful_bits].iter().enumerate() {
                        acc[i_inner] += table[z_byte as usize];
                    }
                }
                acc
            },
        )
        .reduce(
            || vec![F128::ZERO; k],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x += *y;
                }
                a
            },
        )
}

const NEON_TILE_T: usize = 8;

pub(crate) fn partial_fold_packed_z_best(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    if n_log_ok_for_tile(m, k_log, NEON_TILE_T) {
        #[cfg(target_arch = "aarch64")]
        {
            let n_log = m - k_log;
            if n_log >= OBLOCK_MIN_N_LOG && !FOLD_IBLOCK.load(std::sync::atomic::Ordering::Relaxed)
            {
                return partial_fold_packed_z_neon_oblock_padded(
                    z_packed,
                    m,
                    k_log,
                    useful_bits,
                    eq_outer,
                );
            }
            partial_fold_packed_z_neon_iblock_padded(z_packed, m, k_log, useful_bits, eq_outer)
        }
        #[cfg(all(not(target_arch = "aarch64"), target_arch = "x86_64"))]
        {
            partial_fold_packed_z_x86_tiled_padded(z_packed, m, k_log, useful_bits, eq_outer)
        }
        #[cfg(all(not(target_arch = "aarch64"), not(target_arch = "x86_64")))]
        {
            partial_fold_packed_z_fast_padded(z_packed, m, k_log, useful_bits, eq_outer)
        }
    } else {
        partial_fold_packed_z_fast_padded(z_packed, m, k_log, useful_bits, eq_outer)
    }
}

#[cfg(target_arch = "aarch64")]
const OBLOCK_MIN_N_LOG: usize = 16;

fn n_log_ok_for_tile(m: usize, k_log: usize, tile_t: usize) -> bool {
    if k_log < 3 {
        return false;
    }
    let n_log = m - k_log;
    if n_log < 3 + (tile_t.trailing_zeros() as usize) {
        return false;
    }
    let n_stripes = 1usize << (n_log - 3);
    n_stripes.is_multiple_of(tile_t)
}

#[inline]
fn build_sum_table(eq8: &[F128], table: &mut [F128]) {
    debug_assert_eq!(eq8.len(), 8);
    debug_assert_eq!(table.len(), 256);
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe {
        use core::arch::x86_64::*;
        let eq_ptr = eq8.as_ptr() as *const __m128i;
        let t_ptr = table.as_mut_ptr() as *mut __m256i;
        let v0 = _mm_setzero_si128();
        let e0 = _mm_loadu_si128(eq_ptr);
        let e1 = _mm256_broadcastsi128_si256(_mm_loadu_si128(eq_ptr.add(1)));
        let e2 = _mm256_broadcastsi128_si256(_mm_loadu_si128(eq_ptr.add(2)));
        let e3 = _mm256_broadcastsi128_si256(_mm_loadu_si128(eq_ptr.add(3)));
        let p0 = _mm256_set_m128i(e0, v0);
        let p1 = _mm256_xor_si256(p0, e1);
        let p2 = _mm256_xor_si256(p0, e2);
        let p3 = _mm256_xor_si256(p1, e2);
        _mm256_storeu_si256(t_ptr, p0);
        _mm256_storeu_si256(t_ptr.add(1), p1);
        _mm256_storeu_si256(t_ptr.add(2), p2);
        _mm256_storeu_si256(t_ptr.add(3), p3);
        _mm256_storeu_si256(t_ptr.add(4), _mm256_xor_si256(p0, e3));
        _mm256_storeu_si256(t_ptr.add(5), _mm256_xor_si256(p1, e3));
        _mm256_storeu_si256(t_ptr.add(6), _mm256_xor_si256(p2, e3));
        _mm256_storeu_si256(t_ptr.add(7), _mm256_xor_si256(p3, e3));
        macro_rules! expand_step_avx2 {
            ($i:literal, $pairs:literal) => {{
                let ei = _mm256_broadcastsi128_si256(_mm_loadu_si128(eq_ptr.add($i)));
                let mut j = 0usize;
                while j < $pairs {
                    let s0 = _mm256_loadu_si256(t_ptr.add(j));
                    let s1 = _mm256_loadu_si256(t_ptr.add(j + 1));
                    let s2 = _mm256_loadu_si256(t_ptr.add(j + 2));
                    let s3 = _mm256_loadu_si256(t_ptr.add(j + 3));
                    _mm256_storeu_si256(t_ptr.add($pairs + j), _mm256_xor_si256(s0, ei));
                    _mm256_storeu_si256(t_ptr.add($pairs + j + 1), _mm256_xor_si256(s1, ei));
                    _mm256_storeu_si256(t_ptr.add($pairs + j + 2), _mm256_xor_si256(s2, ei));
                    _mm256_storeu_si256(t_ptr.add($pairs + j + 3), _mm256_xor_si256(s3, ei));
                    j += 4;
                }
            }};
        }
        expand_step_avx2!(4, 8);
        expand_step_avx2!(5, 16);
        expand_step_avx2!(6, 32);
        expand_step_avx2!(7, 64);
    }
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
    unsafe {
        use core::arch::x86_64::*;
        let eq_ptr = eq8.as_ptr() as *const __m128i;
        let t_ptr = table.as_mut_ptr() as *mut __m128i;
        let v0 = _mm_setzero_si128();
        let e0 = _mm_loadu_si128(eq_ptr);
        let e1 = _mm_loadu_si128(eq_ptr.add(1));
        let e2 = _mm_loadu_si128(eq_ptr.add(2));
        let e3 = _mm_loadu_si128(eq_ptr.add(3));
        let t2 = _mm_xor_si128(e0, e1);
        let t4 = e2;
        let t5 = _mm_xor_si128(e0, e2);
        let t6 = _mm_xor_si128(e1, e2);
        let t7 = _mm_xor_si128(t2, e2);
        _mm_store_si128(t_ptr, v0);
        _mm_store_si128(t_ptr.add(1), e0);
        _mm_store_si128(t_ptr.add(2), e1);
        _mm_store_si128(t_ptr.add(3), t2);
        _mm_store_si128(t_ptr.add(4), t4);
        _mm_store_si128(t_ptr.add(5), t5);
        _mm_store_si128(t_ptr.add(6), t6);
        _mm_store_si128(t_ptr.add(7), t7);
        _mm_store_si128(t_ptr.add(8), e3);
        _mm_store_si128(t_ptr.add(9), _mm_xor_si128(e0, e3));
        _mm_store_si128(t_ptr.add(10), _mm_xor_si128(e1, e3));
        _mm_store_si128(t_ptr.add(11), _mm_xor_si128(t2, e3));
        _mm_store_si128(t_ptr.add(12), _mm_xor_si128(t4, e3));
        _mm_store_si128(t_ptr.add(13), _mm_xor_si128(t5, e3));
        _mm_store_si128(t_ptr.add(14), _mm_xor_si128(t6, e3));
        _mm_store_si128(t_ptr.add(15), _mm_xor_si128(t7, e3));
        macro_rules! expand_step {
            ($i:literal, $len:literal) => {{
                let ei = _mm_loadu_si128(eq_ptr.add($i));
                let mut j = 0usize;
                while j < $len {
                    let s0 = _mm_load_si128(t_ptr.add(j));
                    let s1 = _mm_load_si128(t_ptr.add(j + 1));
                    let s2 = _mm_load_si128(t_ptr.add(j + 2));
                    let s3 = _mm_load_si128(t_ptr.add(j + 3));
                    _mm_store_si128(t_ptr.add($len + j), _mm_xor_si128(s0, ei));
                    _mm_store_si128(t_ptr.add($len + j + 1), _mm_xor_si128(s1, ei));
                    _mm_store_si128(t_ptr.add($len + j + 2), _mm_xor_si128(s2, ei));
                    _mm_store_si128(t_ptr.add($len + j + 3), _mm_xor_si128(s3, ei));
                    j += 4;
                }
            }};
        }
        expand_step!(4, 16);
        expand_step!(5, 32);
        expand_step!(6, 64);
        expand_step!(7, 128);
        return;
    }
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use core::arch::aarch64::*;
        let eq_ptr = eq8.as_ptr() as *const u8;
        let t_ptr = table.as_mut_ptr() as *mut u8;
        let v0 = vdupq_n_u8(0);
        let e0 = vld1q_u8(eq_ptr);
        let e1 = vld1q_u8(eq_ptr.add(16));
        let e2 = vld1q_u8(eq_ptr.add(32));
        let e3 = vld1q_u8(eq_ptr.add(48));
        let t2 = veorq_u8(e0, e1);
        let t4 = e2;
        let t5 = veorq_u8(e0, e2);
        let t6 = veorq_u8(e1, e2);
        let t7 = veorq_u8(t2, e2);
        vst1q_u8(t_ptr, v0);
        vst1q_u8(t_ptr.add(16), e0);
        vst1q_u8(t_ptr.add(32), e1);
        vst1q_u8(t_ptr.add(48), t2);
        vst1q_u8(t_ptr.add(64), t4);
        vst1q_u8(t_ptr.add(80), t5);
        vst1q_u8(t_ptr.add(96), t6);
        vst1q_u8(t_ptr.add(112), t7);
        vst1q_u8(t_ptr.add(128), e3);
        vst1q_u8(t_ptr.add(144), veorq_u8(e0, e3));
        vst1q_u8(t_ptr.add(160), veorq_u8(e1, e3));
        vst1q_u8(t_ptr.add(176), veorq_u8(t2, e3));
        vst1q_u8(t_ptr.add(192), veorq_u8(t4, e3));
        vst1q_u8(t_ptr.add(208), veorq_u8(t5, e3));
        vst1q_u8(t_ptr.add(224), veorq_u8(t6, e3));
        vst1q_u8(t_ptr.add(240), veorq_u8(t7, e3));
        macro_rules! expand_step_neon {
            ($i:literal, $len:literal) => {{
                let ei = vld1q_u8(eq_ptr.add($i * 16));
                let mut j = 0usize;
                while j < $len {
                    let s0 = vld1q_u8(t_ptr.add(j * 16));
                    let s1 = vld1q_u8(t_ptr.add((j + 1) * 16));
                    let s2 = vld1q_u8(t_ptr.add((j + 2) * 16));
                    let s3 = vld1q_u8(t_ptr.add((j + 3) * 16));
                    vst1q_u8(t_ptr.add(($len + j) * 16), veorq_u8(s0, ei));
                    vst1q_u8(t_ptr.add(($len + j + 1) * 16), veorq_u8(s1, ei));
                    vst1q_u8(t_ptr.add(($len + j + 2) * 16), veorq_u8(s2, ei));
                    vst1q_u8(t_ptr.add(($len + j + 3) * 16), veorq_u8(s3, ei));
                    j += 4;
                }
            }};
        }
        expand_step_neon!(4, 16);
        expand_step_neon!(5, 32);
        expand_step_neon!(6, 64);
        expand_step_neon!(7, 128);
        return;
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        table[0] = F128::ZERO;
        for i in 0..8 {
            let e = eq8[i];
            let len = 1usize << i;
            for j in 0..len {
                table[len + j] = table[j] + e;
            }
        }
    }
}

pub fn pack_z_lincheck(z_logical: &[bool], m: usize, k_log: usize) -> Vec<u8> {
    let k = 1usize << k_log;
    let n_total = 1usize << m;
    assert_eq!(z_logical.len(), n_total);
    let n_outer = n_total / k;
    assert_eq!(n_outer % 8, 0, "need n_outer ≥ 8 for byte stripes");
    let n_stripes = n_outer / 8;

    let mut z_packed: Vec<u8> = crate::alloc_uninit_vec(n_total / 8);
    for byte_idx in 0..n_stripes {
        for i_inner in 0..k {
            let mut byte = 0u8;
            for r in 0..8 {
                let i_outer = 8 * byte_idx + r;
                let logical_idx = i_inner + i_outer * k;
                if z_logical[logical_idx] {
                    byte |= 1u8 << r;
                }
            }
            z_packed[byte_idx * k + i_inner] = byte;
        }
    }
    z_packed
}

pub fn pack_z_lincheck_from_packed(
    z_packed_f128: &[crate::field::F128],
    m: usize,
    k_log: usize,
) -> Vec<u8> {
    use rayon::prelude::*;
    let k = 1usize << k_log;
    let n_total = 1usize << m;
    assert_eq!(z_packed_f128.len(), n_total / 128);
    let n_outer = n_total / k;
    assert_eq!(n_outer % 8, 0, "need n_outer ≥ 8 for byte stripes");

    let mut z_packed: Vec<u8> = crate::alloc_uninit_vec(n_total / 8);

    z_packed
        .par_chunks_mut(k)
        .enumerate()
        .for_each(|(byte_idx, chunk)| {
            for i_inner in 0..k {
                let mut byte = 0u8;
                for r in 0..8 {
                    let i_outer = 8 * byte_idx + r;
                    let logical_idx = i_inner + i_outer * k;
                    let f128_idx = logical_idx / 128;
                    let local_bit = logical_idx % 128;
                    let bit = if local_bit < 64 {
                        (z_packed_f128[f128_idx].lo >> local_bit) & 1 == 1
                    } else {
                        (z_packed_f128[f128_idx].hi >> (local_bit - 64)) & 1 == 1
                    };
                    if bit {
                        byte |= 1u8 << r;
                    }
                }
                chunk[i_inner] = byte;
            }
        });
    z_packed
}

pub fn build_quirky_eq_table(z_skip: F128, x_inner_rest: &[F128], k_skip: usize) -> Vec<F128> {
    let ell_skip = 1usize << k_skip;
    let ell_rest = 1usize << x_inner_rest.len();
    let lambda_skip = lagrange_weights_naive(k_skip, z_skip);
    let eq_rest = build_eq_table(x_inner_rest);
    let total = ell_skip * ell_rest;
    let mut out = vec![F128::ZERO; total];

    for (r_idx, &er) in eq_rest.iter().enumerate() {
        let base = r_idx * ell_skip;
        crate::field::f128_slice::scalar_mul_into(
            &mut out[base..base + ell_skip],
            &lambda_skip,
            er,
        );
    }
    debug_assert_eq!(out.len(), total);
    out
}

fn inner_product(a: &[F128], b: &[F128]) -> F128 {
    crate::field::f128_slice::inner_product(a, b)
}

const SUMCHECK_PAR_THRESHOLD: usize = 1usize << 12;

fn sparse_row_fold_alpha_batched(
    alpha: F128,
    a_0: &SparseBinaryMatrix,
    b_0: &SparseBinaryMatrix,
    eq_table: &[F128],
) -> Vec<F128> {
    use rayon::prelude::*;
    let n_cols = a_0.num_cols;
    debug_assert_eq!(b_0.num_cols, n_cols);
    debug_assert_eq!(eq_table.len(), a_0.num_rows);
    debug_assert_eq!(eq_table.len(), b_0.num_rows);

    let total_rows = a_0.num_rows + b_0.num_rows;
    if total_rows < SPARSE_ROW_FOLD_PAR_THRESHOLD {
        let mut out = vec![F128::ZERO; n_cols];
        for (r, row) in a_0.rows.iter().enumerate() {
            let e = alpha * eq_table[r];
            for &c in row {
                out[c] += e;
            }
        }
        for (r, row) in b_0.rows.iter().enumerate() {
            let e = eq_table[r];
            for &c in row {
                out[c] += e;
            }
        }
        return out;
    }

    let n_rows = a_0.num_rows;
    let p = rayon::current_num_threads().max(1);

    let chunk_rows = (n_rows.div_ceil(p * 4)).max(256);
    let n_chunks = n_rows.div_ceil(chunk_rows);

    let partials: Vec<Vec<F128>> = (0..n_chunks)
        .into_par_iter()
        .map(|ci| {
            let lo = ci * chunk_rows;
            let hi = ((ci + 1) * chunk_rows).min(n_rows);
            let mut acc = vec![F128::ZERO; n_cols];
            for r in lo..hi {
                let ea = alpha * eq_table[r];
                let eb = eq_table[r];
                for &c in &a_0.rows[r] {
                    acc[c] += ea;
                }
                for &c in &b_0.rows[r] {
                    acc[c] += eb;
                }
            }
            acc
        })
        .collect();

    let mut out = vec![F128::ZERO; n_cols];
    for acc in &partials {
        crate::field::f128_slice::xor_assign(&mut out, acc);
    }
    out
}

fn sumcheck_round_eval_par(c: &[F128], z: &[F128]) -> (F128, F128) {
    use crate::field::gf2_128::F256Unreduced;
    use rayon::prelude::*;
    let half = c.len() / 2;
    debug_assert_eq!(z.len(), c.len());
    let (clo, chi) = c.split_at(half);
    let (zlo, zhi) = z.split_at(half);
    if half < SUMCHECK_PAR_THRESHOLD {
        let mut e1 = F256Unreduced::ZERO;
        let mut einf = F256Unreduced::ZERO;
        for i in 0..half {
            e1 ^= chi[i].mul_unreduced(zhi[i]);
            einf ^= (chi[i] + clo[i]).mul_unreduced(zhi[i] + zlo[i]);
        }
        return (e1.reduce(), einf.reduce());
    }
    let (e1, einf) = (0..half)
        .into_par_iter()
        .map(|i| {
            let e1_i = chi[i].mul_unreduced(zhi[i]);
            let einf_i = (chi[i] + clo[i]).mul_unreduced(zhi[i] + zlo[i]);
            (e1_i, einf_i)
        })
        .reduce(
            || (F256Unreduced::ZERO, F256Unreduced::ZERO),
            |a, b| (a.0 ^ b.0, a.1 ^ b.1),
        );
    (e1.reduce(), einf.reduce())
}

fn sumcheck_bind_top_in_place_par(v: &mut Vec<F128>, r: F128) {
    use rayon::prelude::*;
    let half = v.len() / 2;
    if half < SUMCHECK_PAR_THRESHOLD {
        for i in 0..half {
            v[i] = v[i].mle_fold(v[i + half], r);
        }
    } else {
        let (lo, hi) = v.split_at_mut(half);
        let hi = &hi[..half];
        lo.par_iter_mut()
            .zip(hi.par_iter())
            .for_each(|(lo_i, &hi_i)| {
                *lo_i = lo_i.mle_fold(hi_i, r);
            });
    }
    v.truncate(half);
}

fn sumcheck_bind_both_and_eval_next(
    comb: &mut Vec<F128>,
    z: &mut Vec<F128>,
    r: F128,
) -> (F128, F128) {
    use rayon::prelude::*;
    let len = comb.len();
    debug_assert_eq!(z.len(), len);
    let half = len / 2;
    let half2 = half / 2;
    debug_assert!(half2 >= 1, "fused step needs a well-defined next round");

    let (c_lo, c_hi) = comb.split_at_mut(half);
    let (cq0, cq1) = c_lo.split_at_mut(half2);
    let (cq2, cq3) = c_hi.split_at(half2);
    let (z_lo, z_hi) = z.split_at_mut(half);
    let (zq0, zq1) = z_lo.split_at_mut(half2);
    let (zq2, zq3) = z_hi.split_at(half2);

    let (e1, einf) = if half2 < SUMCHECK_PAR_THRESHOLD {
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        unsafe {
            let (e1_lo, e1_hi, ei_lo, ei_hi) =
                crate::gf2_kernels::k0_f128_lincheck_bind_both_and_eval_slice(
                    cq0.as_mut_ptr() as *mut u64,
                    cq1.as_mut_ptr() as *mut u64,
                    cq2.as_ptr() as *const u64,
                    cq3.as_ptr() as *const u64,
                    zq0.as_mut_ptr() as *mut u64,
                    zq1.as_mut_ptr() as *mut u64,
                    zq2.as_ptr() as *const u64,
                    zq3.as_ptr() as *const u64,
                    half2,
                    r.lo,
                    r.hi,
                );
            (
                F128 {
                    lo: e1_lo,
                    hi: e1_hi,
                },
                F128 {
                    lo: ei_lo,
                    hi: ei_hi,
                },
            )
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            use crate::field::gf2_128::F256Unreduced;
            let mut e1 = F256Unreduced::ZERO;
            let mut einf = F256Unreduced::ZERO;
            for i in 0..half2 {
                let lo = cq0[i].mle_fold(cq2[i], r);
                let hi = cq1[i].mle_fold(cq3[i], r);
                let zlo = zq0[i].mle_fold(zq2[i], r);
                let zhi = zq1[i].mle_fold(zq3[i], r);
                cq0[i] = lo;
                cq1[i] = hi;
                zq0[i] = zlo;
                zq1[i] = zhi;
                e1 ^= hi.mul_unreduced(zhi);
                einf ^= (hi + lo).mul_unreduced(zhi + zlo);
            }
            (e1.reduce(), einf.reduce())
        }
    } else {
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        {
            const CHUNK: usize = 1024;
            cq0.par_chunks_mut(CHUNK)
                .zip(cq1.par_chunks_mut(CHUNK))
                .zip(cq2.par_chunks(CHUNK))
                .zip(cq3.par_chunks(CHUNK))
                .zip(zq0.par_chunks_mut(CHUNK))
                .zip(zq1.par_chunks_mut(CHUNK))
                .zip(zq2.par_chunks(CHUNK))
                .zip(zq3.par_chunks(CHUNK))
                .map(|(((((((c0, c1), c2), c3), z0), z1), z2), z3)| unsafe {
                    let (e1_lo, e1_hi, ei_lo, ei_hi) =
                        crate::gf2_kernels::k0_f128_lincheck_bind_both_and_eval_slice(
                            c0.as_mut_ptr() as *mut u64,
                            c1.as_mut_ptr() as *mut u64,
                            c2.as_ptr() as *const u64,
                            c3.as_ptr() as *const u64,
                            z0.as_mut_ptr() as *mut u64,
                            z1.as_mut_ptr() as *mut u64,
                            z2.as_ptr() as *const u64,
                            z3.as_ptr() as *const u64,
                            c0.len(),
                            r.lo,
                            r.hi,
                        );
                    (
                        F128 {
                            lo: e1_lo,
                            hi: e1_hi,
                        },
                        F128 {
                            lo: ei_lo,
                            hi: ei_hi,
                        },
                    )
                })
                .reduce(|| (F128::ZERO, F128::ZERO), |a, b| (a.0 + b.0, a.1 + b.1))
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            use crate::field::gf2_128::F256Unreduced;
            let (e1, einf) = cq0
                .par_iter_mut()
                .zip(cq1.par_iter_mut())
                .zip(cq2.par_iter())
                .zip(cq3.par_iter())
                .zip(zq0.par_iter_mut())
                .zip(zq1.par_iter_mut())
                .zip(zq2.par_iter())
                .zip(zq3.par_iter())
                .map(|(((((((c0, c1), c2), c3), z0), z1), z2), z3)| {
                    let lo = (*c0).mle_fold(*c2, r);
                    let hi = (*c1).mle_fold(*c3, r);
                    let zlo = (*z0).mle_fold(*z2, r);
                    let zhi = (*z1).mle_fold(*z3, r);
                    *c0 = lo;
                    *c1 = hi;
                    *z0 = zlo;
                    *z1 = zhi;
                    (hi.mul_unreduced(zhi), (hi + lo).mul_unreduced(zhi + zlo))
                })
                .reduce(
                    || (F256Unreduced::ZERO, F256Unreduced::ZERO),
                    |a, b| (a.0 ^ b.0, a.1 ^ b.1),
                );
            (e1.reduce(), einf.reduce())
        }
    };

    comb.truncate(half);
    z.truncate(half);
    (e1, einf)
}

pub fn prove<Ch: Challenger>(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    k_skip: usize,
    circuit: &dyn LincheckCircuit,
    x_ab: &QuirkyPoint,
    challenger: &mut Ch,
) -> (LincheckProof, LincheckClaim) {
    prove_padded(
        z_packed,
        m,
        k_log,
        k_skip,
        1usize << k_log,
        circuit,
        x_ab,
        challenger,
    )
}

pub fn prove_padded<Ch: Challenger>(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    k_skip: usize,
    useful_bits: usize,
    circuit: &dyn LincheckCircuit,
    x_ab: &QuirkyPoint,
    challenger: &mut Ch,
) -> (LincheckProof, LincheckClaim) {
    let (proof, claim, _) = prove_padded_inner(
        z_packed,
        m,
        k_log,
        k_skip,
        useful_bits,
        circuit,
        x_ab,
        false,
        challenger,
    );
    (proof, claim)
}

pub fn prove_padded_capture_z_vec<Ch: Challenger>(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    k_skip: usize,
    useful_bits: usize,
    circuit: &dyn LincheckCircuit,
    x_ab: &QuirkyPoint,
    challenger: &mut Ch,
) -> (LincheckProof, LincheckClaim, Vec<F128>) {
    let (proof, claim, captured) = prove_padded_inner(
        z_packed,
        m,
        k_log,
        k_skip,
        useful_bits,
        circuit,
        x_ab,
        true,
        challenger,
    );
    (
        proof,
        claim,
        captured.expect("capture=true must produce z_vec"),
    )
}

#[allow(clippy::too_many_arguments)]
fn prove_padded_inner<Ch: Challenger>(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    k_skip: usize,
    useful_bits: usize,
    circuit: &dyn LincheckCircuit,
    x_ab: &QuirkyPoint,
    capture_z_vec: bool,
    challenger: &mut Ch,
) -> (LincheckProof, LincheckClaim, Option<Vec<F128>>) {
    let k = 1usize << k_log;
    let n_log = m - k_log;
    assert!(m >= k_log);
    assert!(k_skip <= k_log, "k_skip must be ≤ k_log");
    assert!(useful_bits <= k, "useful_bits ({useful_bits}) > k ({k})");
    let inner_rest_len = k_log - k_skip;
    assert_eq!(circuit.n_cols(), k);
    assert_eq!(x_ab.x_inner_rest.len(), inner_rest_len);
    assert_eq!(x_ab.x_outer.len(), n_log);

    challenger.observe_label(b"flock-lincheck-v0");
    let trace = std::env::var("LINCHECK_TRACE").is_ok();

    let alpha = challenger.sample_f128();

    let t = if trace {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let eq_inner = build_quirky_eq_table(x_ab.z_skip, &x_ab.x_inner_rest, k_skip);
    if let Some(t) = t {
        eprintln!(
            "[lc] {:<26} {:>7.2} ms",
            "build_quirky_eq",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
    let t = if trace {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let mut comb_vec = circuit.fold_alpha_batched(alpha, &eq_inner);
    if let Some(t) = t {
        eprintln!(
            "[lc] {:<26} {:>7.2} ms",
            "fold_alpha_batched",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    if let Some(col) = circuit.const_pin_col() {
        let beta = challenger.sample_f128();
        comb_vec[col] += beta;
    }

    let t = if trace {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let eq_x_outer = build_eq_table(&x_ab.x_outer);
    let mut z_vec = partial_fold_packed_z_best(z_packed, m, k_log, useful_bits, &eq_x_outer);
    if let Some(t) = t {
        eprintln!(
            "[lc] {:<26} {:>7.2} ms",
            "partial_fold_z",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    let captured_z_vec: Option<Vec<F128>> = if capture_z_vec {
        Some(z_vec.clone())
    } else {
        None
    };
    let t_sumcheck_start = if trace {
        Some(std::time::Instant::now())
    } else {
        None
    };

    let mut rounds = Vec::with_capacity(inner_rest_len);
    let mut r_rounds = Vec::with_capacity(inner_rest_len);
    if inner_rest_len > 0 {
        let (mut e1, mut einf) = sumcheck_round_eval_par(&comb_vec, &z_vec);
        for t in 0..inner_rest_len {
            challenger.observe_f128(e1);
            challenger.observe_f128(einf);
            let r = challenger.sample_f128();
            rounds.push((e1, einf));
            r_rounds.push(r);
            if t + 1 < inner_rest_len {
                let (ne1, neinf) = sumcheck_bind_both_and_eval_next(&mut comb_vec, &mut z_vec, r);
                e1 = ne1;
                einf = neinf;
            } else {
                sumcheck_bind_top_in_place_par(&mut z_vec, r);
            }
        }
    }
    if let Some(t) = t_sumcheck_start {
        eprintln!(
            "[lc] {:<26} {:>7.2} ms",
            "sumcheck (all rounds)",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    let z_partial = z_vec;
    challenger.observe_f128_slice(&z_partial);

    let r_inner_skip = challenger.sample_f128();

    let lambda = lagrange_weights_naive(k_skip, r_inner_skip);
    let w = inner_product(&lambda, &z_partial);

    let mut r_inner_rest = r_rounds;
    r_inner_rest.reverse();

    let proof = LincheckProof { rounds, z_partial };
    let claim = LincheckClaim {
        r_inner_skip,
        r_inner_rest,
        w,
    };
    (proof, claim, captured_z_vec)
}

pub fn verify<Ch: Challenger>(
    m: usize,
    k_log: usize,
    k_skip: usize,
    circuit: &dyn LincheckCircuit,
    x_ab: &QuirkyPoint,
    v_a: F128,
    v_b: F128,
    proof: &LincheckProof,
    challenger: &mut Ch,
) -> Result<LincheckClaim, VerifyError> {
    let k = 1usize << k_log;
    let n_log = m - k_log;

    if k_skip > k_log {
        return Err(VerifyError::KSkipExceedsKLog { k_skip, k_log });
    }
    let inner_rest_len = k_log - k_skip;
    let n_skip = 1usize << k_skip;

    if x_ab.x_inner_rest.len() != inner_rest_len {
        return Err(VerifyError::BadInnerRestLength {
            which: "x_ab",
            expected: inner_rest_len,
            got: x_ab.x_inner_rest.len(),
        });
    }
    if x_ab.x_outer.len() != n_log {
        return Err(VerifyError::BadOuterLength {
            which: "x_ab",
            expected: n_log,
            got: x_ab.x_outer.len(),
        });
    }
    if circuit.n_cols() != k {
        return Err(VerifyError::BadMatrixShape {
            which: "circuit",
            expected: k,
            got_rows: k,
            got_cols: circuit.n_cols(),
        });
    }
    if proof.rounds.len() != inner_rest_len {
        return Err(VerifyError::BadVectorLength {
            which: "rounds",
            expected: inner_rest_len,
            got: proof.rounds.len(),
        });
    }
    if proof.z_partial.len() != n_skip {
        return Err(VerifyError::BadVectorLength {
            which: "z_partial",
            expected: n_skip,
            got: proof.z_partial.len(),
        });
    }

    challenger.observe_label(b"flock-lincheck-v0");

    let trace = std::env::var("VERIFY_TRACE").is_ok();
    let fmt = |s: f64| -> String {
        let ms = s * 1000.0;
        if ms < 1.0 {
            format!("{:>8.2} µs", s * 1e6)
        } else {
            format!("{:>8.2} ms", ms)
        }
    };

    let alpha = challenger.sample_f128();

    let t = std::time::Instant::now();
    let eq_inner = build_quirky_eq_table(x_ab.z_skip, &x_ab.x_inner_rest, k_skip);
    if trace {
        eprintln!(
            "        [lcv] build_quirky_eq_table (2^{k_log}): {}",
            fmt(t.elapsed().as_secs_f64())
        );
    }
    let t = std::time::Instant::now();
    let mut comb_vec = circuit.fold_alpha_batched(alpha, &eq_inner);
    if trace {
        eprintln!(
            "        [lcv] circuit.fold_alpha_batched: {}",
            fmt(t.elapsed().as_secs_f64())
        );
    }

    let t = std::time::Instant::now();

    let mut target = alpha * v_a + v_b;
    if let Some(col) = circuit.const_pin_col() {
        let beta = challenger.sample_f128();
        comb_vec[col] += beta;
        target += beta;
    }
    let mut running = target;
    let mut r_rounds = Vec::with_capacity(inner_rest_len);
    for &(e1, einf) in &proof.rounds {
        challenger.observe_f128(e1);
        challenger.observe_f128(einf);
        let r = challenger.sample_f128();

        let e0 = running + e1;
        let c1 = e0 + e1 + einf;
        running = einf * r * r + c1 * r + e0;

        sumcheck_bind_top_in_place_par(&mut comb_vec, r);
        r_rounds.push(r);
    }
    debug_assert_eq!(comb_vec.len(), n_skip);
    if trace {
        eprintln!(
            "        [lcv] sumcheck replay + comb_vec fold ({} rounds): {}",
            inner_rest_len,
            fmt(t.elapsed().as_secs_f64())
        );
    }

    challenger.observe_f128_slice(&proof.z_partial);

    let final_sum = inner_product(&comb_vec, &proof.z_partial);
    if running != final_sum {
        return Err(VerifyError::ConsistencyFailed {
            which: "sumcheck-final",
        });
    }

    let r_inner_skip = challenger.sample_f128();

    let t = std::time::Instant::now();
    let lambda = lagrange_weights_naive(k_skip, r_inner_skip);
    let w = inner_product(&lambda, &proof.z_partial);
    if trace {
        eprintln!(
            "        [lcv] final consistency + lagrange_weights_naive: {}",
            fmt(t.elapsed().as_secs_f64())
        );
    }

    let mut r_inner_rest = r_rounds;
    r_inner_rest.reverse();

    Ok(LincheckClaim {
        r_inner_skip,
        r_inner_rest,
        w,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::challenger::FsChallenger;

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
        fn f128(&mut self) -> F128 {
            F128 {
                lo: self.next_u64(),
                hi: self.next_u64(),
            }
        }
        fn f128_vec(&mut self, n: usize) -> Vec<F128> {
            (0..n).map(|_| self.f128()).collect()
        }
        fn bits(&mut self, n: usize) -> Vec<bool> {
            (0..n).map(|_| self.next_u64() & 1 == 1).collect()
        }
    }

    fn mle_eval_bool(f: &[bool], point: &[F128]) -> F128 {
        let d = point.len();
        assert_eq!(f.len(), 1 << d);
        let eq = build_eq_table(point);
        let mut acc = F128::ZERO;
        for (i, &b) in f.iter().enumerate() {
            if b {
                acc += eq[i];
            }
        }
        acc
    }

    fn random_quirky_point(m: usize, k_log: usize, k_skip: usize, rng: &mut Rng) -> QuirkyPoint {
        QuirkyPoint {
            z_skip: rng.f128(),
            x_inner_rest: rng.f128_vec(k_log - k_skip),
            x_outer: rng.f128_vec(m - k_log),
        }
    }

    fn mle_eval_bool_quirky(
        f: &[bool],
        m: usize,
        k_log: usize,
        k_skip: usize,
        point: &QuirkyPoint,
    ) -> F128 {
        let k_skip_dim = 1usize << k_skip;
        let inner_rest_len = k_log - k_skip;
        let inner_rest_dim = 1usize << inner_rest_len;
        let k = 1usize << k_log;
        let n_outer = 1usize << (m - k_log);
        assert_eq!(f.len(), 1 << m);

        let lambda = crate::zerocheck::multilinear::lagrange_weights_naive(k_skip, point.z_skip);
        let eq_rest = build_eq_table(&point.x_inner_rest);
        let eq_outer = build_eq_table(&point.x_outer);
        debug_assert_eq!(lambda.len(), k_skip_dim);
        debug_assert_eq!(eq_rest.len(), inner_rest_dim);
        debug_assert_eq!(eq_outer.len(), n_outer);

        let mut acc = F128::ZERO;
        for i in 0..(1 << m) {
            if !f[i] {
                continue;
            }
            let i_skip = i & (k_skip_dim - 1);
            let i_inner_rest = (i >> k_skip) & (inner_rest_dim - 1);
            let i_outer = i / k;
            acc += lambda[i_skip] * eq_rest[i_inner_rest] * eq_outer[i_outer];
        }
        acc
    }

    fn matrix_vector_product(m: &SparseBinaryMatrix, z: &[bool]) -> Vec<bool> {
        assert_eq!(z.len(), m.num_cols);
        m.rows
            .iter()
            .map(|row| {
                let mut acc = false;
                for &col in row {
                    acc ^= z[col];
                }
                acc
            })
            .collect()
    }

    fn apply_block_diag(m_0: &SparseBinaryMatrix, z: &[bool], k_log: usize) -> Vec<bool> {
        let k = 1usize << k_log;
        assert_eq!(m_0.num_rows, k);
        assert_eq!(m_0.num_cols, k);
        assert_eq!(z.len() % k, 0);
        let n_outer = z.len() / k;
        let mut out = vec![false; z.len()];
        for i_outer in 0..n_outer {
            let z_block = &z[i_outer * k..(i_outer + 1) * k];
            let a_block = matrix_vector_product(m_0, z_block);
            out[i_outer * k..(i_outer + 1) * k].copy_from_slice(&a_block);
        }
        out
    }

    fn random_sparse_matrix(k: usize, nnz: usize, rng: &mut Rng) -> SparseBinaryMatrix {
        let mut rows: Vec<Vec<usize>> = vec![Vec::new(); k];
        let mut seen = std::collections::HashSet::new();
        let mut count = 0;
        while count < nnz {
            let r = (rng.next_u64() as usize) % k;
            let c = (rng.next_u64() as usize) % k;
            if seen.insert((r, c)) {
                rows[r].push(c);
                count += 1;
            }
        }
        for row in &mut rows {
            row.sort();
        }
        SparseBinaryMatrix {
            num_rows: k,
            num_cols: k,
            rows,
        }
    }

    #[test]
    fn eq_table_matches_direct_formula() {
        for &d in &[1usize, 2, 3, 5, 8] {
            let mut rng = Rng::new(11 + d as u64);
            let point = rng.f128_vec(d);
            let table = build_eq_table(&point);
            assert_eq!(table.len(), 1 << d);
            for i in 0..(1 << d) {
                let mut expected = F128::ONE;
                for j in 0..d {
                    let bit = ((i >> j) & 1) as u64;

                    let factor = if bit == 0 {
                        F128::ONE + point[j]
                    } else {
                        point[j]
                    };
                    expected *= factor;
                }
                assert_eq!(table[i], expected, "mismatch at d={d}, i={i}");
            }
        }
    }

    #[test]
    fn sparse_row_fold_matches_dense() {
        let mut rng = Rng::new(22);
        let k = 16;
        let nnz = 40;
        let matrix = random_sparse_matrix(k, nnz, &mut rng);
        let eq_table: Vec<F128> = rng.f128_vec(k);

        let got = sparse_row_fold(&matrix, &eq_table);

        let mut expected = vec![F128::ZERO; k];
        for (i, row) in matrix.rows.iter().enumerate() {
            for &j in row {
                expected[j] += eq_table[i];
            }
        }
        assert_eq!(got, expected);
    }

    #[test]
    fn partial_fold_matches_direct() {
        for &(m, k_log) in &[(10usize, 3), (12, 4), (14, 5), (16, 8)] {
            let mut rng = Rng::new(33 + m as u64);
            let z = rng.bits(1 << m);
            let z_packed = pack_z_lincheck(&z, m, k_log);
            let n_log = m - k_log;
            let outer_point = rng.f128_vec(n_log);
            let eq_outer = build_eq_table(&outer_point);

            let got = partial_fold_packed_z(&z_packed, m, k_log, &eq_outer);

            let k = 1usize << k_log;
            assert_eq!(got.len(), k);
            for i_inner in 0..k {
                let mut acc = F128::ZERO;
                for i_outer in 0..(1usize << n_log) {
                    let i = i_inner + i_outer * k;
                    if z[i] {
                        acc += eq_outer[i_outer];
                    }
                }
                assert_eq!(got[i_inner], acc, "mismatch at m={m}, i_inner={i_inner}");
            }
        }
    }

    #[test]
    fn partial_fold_fast_matches_serial() {
        for &(m, k_log) in &[(10usize, 3), (12, 4), (14, 5), (16, 8), (18, 10)] {
            let mut rng = Rng::new(800 + m as u64);
            let z = rng.bits(1 << m);
            let z_packed = pack_z_lincheck(&z, m, k_log);
            let n_log = m - k_log;
            let p = rng.f128_vec(n_log);
            let eq = build_eq_table(&p);

            let serial = partial_fold_packed_z(&z_packed, m, k_log, &eq);
            let fast = partial_fold_packed_z_fast(&z_packed, m, k_log, &eq);
            assert_eq!(serial, fast, "at m={m}, k_log={k_log}");
        }
    }

    #[test]
    fn partial_fold_dispatch_handles_small_k() {
        let (m, k_log) = (8usize, 2usize);
        let mut rng = Rng::new(1234);
        let z = rng.bits(1 << m);
        let z_packed = pack_z_lincheck(&z, m, k_log);
        let eq = build_eq_table(&rng.f128_vec(m - k_log));

        let serial = partial_fold_packed_z(&z_packed, m, k_log, &eq);
        let best = partial_fold_packed_z_best(&z_packed, m, k_log, 1 << k_log, &eq);
        assert_eq!(serial, best);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn partial_fold_neon_single_matches_serial() {
        for &(m, k_log) in &[(14usize, 4), (14, 5), (16, 5), (16, 8), (18, 10)] {
            if !n_log_ok_for_tile(m, k_log, NEON_TILE_T) {
                continue;
            }
            let mut rng = Rng::new(7000 + m as u64);
            let z = rng.bits(1 << m);
            let z_packed = pack_z_lincheck(&z, m, k_log);
            let n_log = m - k_log;
            let p = rng.f128_vec(n_log);
            let eq = build_eq_table(&p);

            let serial = partial_fold_packed_z(&z_packed, m, k_log, &eq);
            let neon = partial_fold_packed_z_neon_single(&z_packed, m, k_log, &eq);
            assert_eq!(serial, neon, "at m={m}, k_log={k_log}");
            let iblock =
                partial_fold_packed_z_neon_iblock_padded(&z_packed, m, k_log, 1usize << k_log, &eq);
            assert_eq!(serial, iblock, "iblock at m={m}, k_log={k_log}");
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn partial_fold_oblock_matches_iblock() {
        let cases: &[(usize, usize, usize)] = &[
            (14, 4, 1 << 4),
            (16, 8, 1 << 8),
            (18, 10, 1 << 10),
            (20, 10, 597),
            (22, 14, 15_409),
        ];
        for &(m, k_log, useful_bits) in cases {
            assert!(
                n_log_ok_for_tile(m, k_log, NEON_TILE_T),
                "case must be tile-eligible"
            );
            let k = 1usize << k_log;
            let n_log = m - k_log;
            let n_blocks = 1usize << n_log;
            let mut rng = Rng::new(7200 + (m * 31 + k_log) as u64);
            let mut z = rng.bits(1 << m);

            for blk in 0..n_blocks {
                for j in useful_bits..k {
                    z[blk * k + j] = false;
                }
            }
            let z_packed = pack_z_lincheck(&z, m, k_log);
            let eq = build_eq_table(&rng.f128_vec(n_log));
            let want =
                partial_fold_packed_z_neon_iblock_padded(&z_packed, m, k_log, useful_bits, &eq);
            let got =
                partial_fold_packed_z_neon_oblock_padded(&z_packed, m, k_log, useful_bits, &eq);
            assert_eq!(want, got, "m={m} k_log={k_log} useful={useful_bits}");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn partial_fold_x86_tiled_matches_serial() {
        for &(m, k_log) in &[(14usize, 4), (14, 5), (16, 5), (16, 8), (18, 10)] {
            if !n_log_ok_for_tile(m, k_log, 8) {
                continue;
            }
            let mut rng = Rng::new(7100 + m as u64);
            let z = rng.bits(1 << m);
            let z_packed = pack_z_lincheck(&z, m, k_log);
            let n_log = m - k_log;
            let p = rng.f128_vec(n_log);
            let eq = build_eq_table(&p);
            let k = 1usize << k_log;
            let serial = partial_fold_packed_z(&z_packed, m, k_log, &eq);
            let tiled = partial_fold_packed_z_x86_tiled_padded(&z_packed, m, k_log, k, &eq);
            assert_eq!(serial, tiled, "at m={m}, k_log={k_log}");
        }
    }

    #[test]
    fn partial_fold_padded_matches_dense() {
        let cases: &[(usize, usize, usize)] =
            &[(17, 14, 15_409), (18, 15, 31_401), (19, 16, 42_560)];
        for &(m, k_log, useful_bits) in cases {
            let mut rng = Rng::new(0xBADD_BEEF_u64.wrapping_add((k_log * 31 + m) as u64));
            let total_bits = 1usize << m;
            let n_log = m - k_log;
            let block_size = 1usize << k_log;
            let n_blocks = 1usize << n_log;

            let mut z = rng.bits(total_bits);
            for blk in 0..n_blocks {
                for j in useful_bits..block_size {
                    z[blk * block_size + j] = false;
                }
            }
            let z_packed = pack_z_lincheck(&z, m, k_log);
            let outer_point = rng.f128_vec(n_log);
            let eq_outer = build_eq_table(&outer_point);

            let dense_fast = partial_fold_packed_z_fast(&z_packed, m, k_log, &eq_outer);
            let padded_fast =
                partial_fold_packed_z_fast_padded(&z_packed, m, k_log, useful_bits, &eq_outer);
            assert_eq!(
                dense_fast, padded_fast,
                "fast: m={m}, k_log={k_log}, useful={useful_bits}"
            );

            #[cfg(target_arch = "aarch64")]
            if n_log_ok_for_tile(m, k_log, NEON_TILE_T) {
                let dense_neon = partial_fold_packed_z_neon_single(&z_packed, m, k_log, &eq_outer);
                let padded_neon = partial_fold_packed_z_neon_single_padded(
                    &z_packed,
                    m,
                    k_log,
                    useful_bits,
                    &eq_outer,
                );
                assert_eq!(
                    dense_neon, padded_neon,
                    "neon: m={m}, k_log={k_log}, useful={useful_bits}"
                );

                let dense_iblock = partial_fold_packed_z_neon_iblock_padded(
                    &z_packed,
                    m,
                    k_log,
                    1usize << k_log,
                    &eq_outer,
                );
                let padded_iblock = partial_fold_packed_z_neon_iblock_padded(
                    &z_packed,
                    m,
                    k_log,
                    useful_bits,
                    &eq_outer,
                );
                assert_eq!(
                    dense_neon, dense_iblock,
                    "iblock dense: m={m}, k_log={k_log}, useful={useful_bits}"
                );
                assert_eq!(
                    dense_neon, padded_iblock,
                    "iblock padded: m={m}, k_log={k_log}, useful={useful_bits}"
                );
            }
        }
    }

    #[test]
    fn partial_fold_is_mle_at_outer_point() {
        let m = 14;
        let k_log = 5;
        let k = 1 << k_log;
        let mut rng = Rng::new(44);
        let z = rng.bits(1 << m);
        let z_packed = pack_z_lincheck(&z, m, k_log);
        let x_outer = rng.f128_vec(m - k_log);
        let eq_outer = build_eq_table(&x_outer);

        let z_partial = partial_fold_packed_z(&z_packed, m, k_log, &eq_outer);

        for i_inner in 0..k {
            let mut point = Vec::with_capacity(m);
            for j in 0..k_log {
                point.push(if (i_inner >> j) & 1 == 1 {
                    F128::ONE
                } else {
                    F128::ZERO
                });
            }
            point.extend_from_slice(&x_outer);
            let z_eval = mle_eval_bool(&z, &point);
            assert_eq!(z_partial[i_inner], z_eval, "i_inner={i_inner}");
        }
    }

    #[test]
    fn prove_verify_roundtrip_honest() {
        for &(m, k_log, k_skip) in &[
            (10usize, 4, 0),
            (10, 4, 2),
            (10, 4, 4),
            (12, 5, 3),
            (14, 7, 6),
            (14, 7, 0),
        ] {
            let k = 1usize << k_log;
            let mut rng = Rng::new(55 + (m * 100 + k_log * 10 + k_skip) as u64);

            let nnz_per_mat = k * 2;
            let a_0 = random_sparse_matrix(k, nnz_per_mat, &mut rng);
            let b_0 = random_sparse_matrix(k, nnz_per_mat, &mut rng);

            let z = rng.bits(1 << m);
            let a = apply_block_diag(&a_0, &z, k_log);
            let b = apply_block_diag(&b_0, &z, k_log);
            let z_packed = pack_z_lincheck(&z, m, k_log);

            let x_ab = random_quirky_point(m, k_log, k_skip, &mut rng);

            let v_a = mle_eval_bool_quirky(&a, m, k_log, k_skip, &x_ab);
            let v_b = mle_eval_bool_quirky(&b, m, k_log, k_skip, &x_ab);

            let circuit = SparseMatrixCircuit::new(&a_0, &b_0);
            let mut ch_p = FsChallenger::new(b"flock-test-v0");
            let (proof, claim_p) = prove(&z_packed, m, k_log, k_skip, &circuit, &x_ab, &mut ch_p);

            let mut ch_v = FsChallenger::new(b"flock-test-v0");
            let claim_v = verify(
                m, k_log, k_skip, &circuit, &x_ab, v_a, v_b, &proof, &mut ch_v,
            )
            .unwrap_or_else(|e| {
                panic!("verify rejected honest proof at m={m},k_log={k_log},k_skip={k_skip}: {e:?}")
            });

            assert_eq!(
                claim_p, claim_v,
                "claim mismatch at m={m}, k_log={k_log}, k_skip={k_skip}"
            );

            let pt = QuirkyPoint {
                z_skip: claim_v.r_inner_skip,
                x_inner_rest: claim_v.r_inner_rest.clone(),
                x_outer: x_ab.x_outer.clone(),
            };
            assert_eq!(
                claim_v.w,
                mle_eval_bool_quirky(&z, m, k_log, k_skip, &pt),
                "w wrong at m={m}, k_log={k_log}, k_skip={k_skip}"
            );
        }
    }

    #[test]
    fn verify_rejects_mutations() {
        let m = 12;
        let k_log = 4;
        let k_skip = 2;
        let k = 1 << k_log;
        let mut rng = Rng::new(66);
        let a_0 = random_sparse_matrix(k, k * 5, &mut rng);
        let b_0 = random_sparse_matrix(k, k * 5, &mut rng);
        let z = rng.bits(1 << m);
        let a = apply_block_diag(&a_0, &z, k_log);
        let b = apply_block_diag(&b_0, &z, k_log);
        let z_packed = pack_z_lincheck(&z, m, k_log);
        let x_ab = random_quirky_point(m, k_log, k_skip, &mut rng);
        let v_a = mle_eval_bool_quirky(&a, m, k_log, k_skip, &x_ab);
        let v_b = mle_eval_bool_quirky(&b, m, k_log, k_skip, &x_ab);

        let _seed: u64 = 0xFEEDFACE;
        let circuit = SparseMatrixCircuit::new(&a_0, &b_0);
        let mut ch_p = FsChallenger::new(b"flock-test-v0");
        let (proof, _) = prove(&z_packed, m, k_log, k_skip, &circuit, &x_ab, &mut ch_p);

        let eq_inner = build_quirky_eq_table(x_ab.z_skip, &x_ab.x_inner_rest, k_skip);
        let row_a = sparse_row_fold(&a_0, &eq_inner);
        let row_b = sparse_row_fold(&b_0, &eq_inner);
        let idx = (0..k)
            .find(|&i| row_a[i] != F128::ZERO || row_b[i] != F128::ZERO)
            .expect("no row-vector slot is nonzero in either A or B — test degenerate");

        let n_skip = 1usize << k_skip;
        let skip_idx = idx % n_skip;
        let mutations: Vec<(String, Box<dyn Fn(&LincheckProof) -> LincheckProof>)> = vec![
            (
                format!("z_partial[{skip_idx}].lo bit-flip"),
                Box::new(move |p| {
                    let mut q = p.clone();
                    q.z_partial[skip_idx].lo ^= 1;
                    q
                }),
            ),
            (
                format!("z_partial[{skip_idx}].hi bit-flip"),
                Box::new(move |p| {
                    let mut q = p.clone();
                    q.z_partial[skip_idx].hi ^= 1;
                    q
                }),
            ),
        ];
        for (label, mutate) in mutations {
            let bad = mutate(&proof);
            let mut ch = FsChallenger::new(b"flock-test-v0");
            let res = verify(m, k_log, k_skip, &circuit, &x_ab, v_a, v_b, &bad, &mut ch);
            assert!(
                matches!(res, Err(VerifyError::ConsistencyFailed { .. })),
                "verify did not reject {label}: got {res:?}"
            );
        }
    }

    #[test]
    fn verify_rejects_shape_errors() {
        let m = 10;
        let k_log = 3;
        let k_skip = 1;
        let k = 1 << k_log;
        let mut rng = Rng::new(77);
        let a_0 = random_sparse_matrix(k, k, &mut rng);
        let b_0 = random_sparse_matrix(k, k, &mut rng);
        let z = rng.bits(1 << m);
        let a = apply_block_diag(&a_0, &z, k_log);
        let b = apply_block_diag(&b_0, &z, k_log);
        let z_packed = pack_z_lincheck(&z, m, k_log);
        let x_ab = random_quirky_point(m, k_log, k_skip, &mut rng);
        let v_a = mle_eval_bool_quirky(&a, m, k_log, k_skip, &x_ab);
        let v_b = mle_eval_bool_quirky(&b, m, k_log, k_skip, &x_ab);

        let circuit = SparseMatrixCircuit::new(&a_0, &b_0);
        let mut ch_p = FsChallenger::new(b"flock-test-v0");
        let (proof, _) = prove(&z_packed, m, k_log, k_skip, &circuit, &x_ab, &mut ch_p);

        let mut bad = proof.clone();
        bad.z_partial.pop();
        let mut ch = FsChallenger::new(b"flock-test-v0");
        assert!(matches!(
            verify(m, k_log, k_skip, &circuit, &x_ab, v_a, v_b, &bad, &mut ch),
            Err(VerifyError::BadVectorLength { .. })
        ));

        let mut ch = FsChallenger::new(b"flock-test-v0");
        let bad_x_ab = QuirkyPoint {
            z_skip: x_ab.z_skip,
            x_inner_rest: x_ab.x_inner_rest[..x_ab.x_inner_rest.len() - 1].to_vec(),
            x_outer: x_ab.x_outer.clone(),
        };
        assert!(matches!(
            verify(
                m, k_log, k_skip, &circuit, &bad_x_ab, v_a, v_b, &proof, &mut ch
            ),
            Err(VerifyError::BadInnerRestLength { .. })
        ));

        let mut ch = FsChallenger::new(b"flock-test-v0");
        assert!(matches!(
            verify(
                m,
                k_log,
                k_log + 1,
                &circuit,
                &x_ab,
                v_a,
                v_b,
                &proof,
                &mut ch,
            ),
            Err(VerifyError::KSkipExceedsKLog { .. })
        ));
    }
}
