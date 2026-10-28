use std::sync::OnceLock;

use crate::field::{F8, F128, PHI_8_TABLE, mul_by_x, phi8};
use crate::ntt::InvNttTableByteSingleGf8;

use super::PaddingSpec;
use super::univariate_skip::{SplitEqGhash, ntt_extend_f128_vec_ghash, pack_bits};

mod kernels;

#[cfg(all(test, target_arch = "aarch64"))]
use kernels::aarch64::{
    bit_transpose_64bytes_neon, shift_reduce_inner_ab_fused_neon, shift_reduce_inner_ab_neon,
};
#[cfg(all(test, target_arch = "aarch64"))]
use kernels::bit_transpose_64bytes_scalar;
#[cfg(all(
    test,
    any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "gfni")
    )
))]
use kernels::shift_reduce_inner_ab_scalar;
#[cfg(all(
    test,
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
use kernels::x86_64::shift_reduce_inner_ab_x86_avx512;
#[cfg(all(test, target_arch = "x86_64", target_feature = "gfni"))]
use kernels::x86_64::shift_reduce_inner_ab_x86_sse;

pub const K_SKIP: usize = 6;
const ELL: usize = 64;
const N_CHUNKS: usize = 8;
const N_INNER: usize = 7;
const N_MEDIUM: usize = 4;

pub const SMALL_CHAL_F8: [u8; 3] = [0xF7, 0x53, 0xB5];

pub const C_S_F8: u8 = 0x1C;

pub fn c_s_f128() -> F128 {
    phi8(F8(C_S_F8))
}

pub fn small_challenges_ghash() -> [F128; 3] {
    [
        phi8(F8(SMALL_CHAL_F8[0])),
        phi8(F8(SMALL_CHAL_F8[1])),
        phi8(F8(SMALL_CHAL_F8[2])),
    ]
}

pub fn medium_challenges_ghash() -> [F128; 4] {
    let g1 = F128 {
        lo: 1u64 << 1,
        hi: 0,
    };
    let g2 = F128 {
        lo: 1u64 << 2,
        hi: 0,
    };
    let g4 = F128 {
        lo: 1u64 << 4,
        hi: 0,
    };
    let g8 = F128 {
        lo: 1u64 << 8,
        hi: 0,
    };
    [
        g1 * (F128::ONE + g1).inv(),
        g2 * (F128::ONE + g2).inv(),
        g4 * (F128::ONE + g4).inv(),
        g8 * (F128::ONE + g8).inv(),
    ]
}

pub fn c_2_small_f128() -> F128 {
    let r_2 = phi8(F8(SMALL_CHAL_F8[1]));
    let r_3 = phi8(F8(SMALL_CHAL_F8[2]));
    (F128::ONE + r_2) * (F128::ONE + r_3)
}

pub fn alpha_inv_f128() -> F128 {
    phi8(F8(0x02).inv())
}

fn compute_d_inv() -> F128 {
    let g1 = F128 {
        lo: 1u64 << 1,
        hi: 0,
    };
    let g2 = F128 {
        lo: 1u64 << 2,
        hi: 0,
    };
    let g4 = F128 {
        lo: 1u64 << 4,
        hi: 0,
    };
    let g8 = F128 {
        lo: 1u64 << 8,
        hi: 0,
    };
    ((F128::ONE + g1) * (F128::ONE + g2) * (F128::ONE + g4) * (F128::ONE + g8)).inv()
}

static D_INV_CACHE: OnceLock<F128> = OnceLock::new();
fn d_inv() -> F128 {
    *D_INV_CACHE.get_or_init(compute_d_inv)
}

const CONVERT_TABLE_SIZE: usize = 16 * 256;

static CONVERT_TABLE_CACHE: OnceLock<Vec<F128>> = OnceLock::new();
static CONVERT_C_PAIR_TABLE_CACHE: OnceLock<Vec<F128>> = OnceLock::new();

fn build_convert_table() -> Vec<F128> {
    let mut gamma_pow = [F128::ZERO; 16];
    gamma_pow[0] = F128::ONE;
    for b in 1..16 {
        gamma_pow[b] = mul_by_x(gamma_pow[b - 1]);
    }
    let mut table = vec![F128::ZERO; CONVERT_TABLE_SIZE];
    for b in 0..16 {
        let g_b = gamma_pow[b];
        for v in 0..256 {
            table[b * 256 + v] = g_b * PHI_8_TABLE[v];
        }
    }
    table
}

fn build_convert_c_pair_table() -> Vec<F128> {
    let conv = convert_table();
    let mut table = vec![F128::ZERO; 8 * 256];
    for m in 0..8 {
        let b0 = 2 * m;
        let b1 = b0 + 1;
        for w in 0..256 {
            let v0 = w & 0x55;
            let v1 = (w >> 1) & 0x55;
            table[m * 256 + w] = conv[b0 * 256 + v0] + conv[b1 * 256 + v1];
        }
    }
    table
}

fn convert_table() -> &'static [F128] {
    CONVERT_TABLE_CACHE.get_or_init(build_convert_table)
}

fn convert_c_pair_table() -> &'static [F128] {
    CONVERT_C_PAIR_TABLE_CACHE.get_or_init(build_convert_c_pair_table)
}

#[inline]
pub fn bit_transpose_64bytes(input: &[u8; 64], output: &mut [u8; 64]) {
    kernels::bit_transpose_64bytes(input, output);
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn shift_reduce_inner_ab(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; 64],
    a_col: &mut [F8],
    b_col: &mut [F8],
) {
    kernels::shift_reduce_inner_ab(
        a_packed,
        b_packed,
        inv_table,
        chunk_byte_base,
        b_med,
        out,
        a_col,
        b_col,
    );
}

pub fn round1_shift_reduce_extract_c(
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
    round1_shift_reduce_extract_c_packed(&a_packed, &b_packed, &c_packed, m, k_skip, r, inv_table)
}

#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub(crate) struct KaratsubaAcc {
    pub ll: F128,
    pub mid: F128,
    pub hh: F128,
}

impl KaratsubaAcc {
    pub const ZERO: Self = Self {
        ll: F128::ZERO,
        mid: F128::ZERO,
        hh: F128::ZERO,
    };

    #[inline(always)]
    pub fn reduce(self) -> F128 {
        #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
        unsafe {
            if (self.mid.lo | self.mid.hi | self.hh.lo | self.hh.hi) == 0 {
                return self.ll;
            }
            use core::arch::x86_64::*;
            let ll = _mm_set_epi64x(self.ll.hi as i64, self.ll.lo as i64);
            let mid = _mm_set_epi64x(self.mid.hi as i64, self.mid.lo as i64);
            let hh = _mm_set_epi64x(self.hh.hi as i64, self.hh.lo as i64);
            let res = crate::gf2_kernels::k0_f128_reduce_3lane_m128i(ll, mid, hh);
            return F128 {
                lo: _mm_extract_epi64::<0>(res) as u64,
                hi: _mm_extract_epi64::<1>(res) as u64,
            };
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
        unsafe {
            if (self.mid.lo | self.mid.hi | self.hh.lo | self.hh.hi) == 0 {
                return self.ll;
            }
            use core::arch::aarch64::*;
            let ll = vld1q_u64((&self.ll as *const F128).cast::<u64>());
            let mid = vld1q_u64((&self.mid as *const F128).cast::<u64>());
            let hh = vld1q_u64((&self.hh as *const F128).cast::<u64>());
            let res = crate::gf2_kernels::k0_f128_reduce_3lane_neon(ll, mid, hh);
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
            let cross_lo = self.mid.lo ^ self.ll.lo ^ self.hh.lo;
            let cross_hi = self.mid.hi ^ self.ll.hi ^ self.hh.hi;
            crate::field::F256Unreduced {
                r0: self.ll.lo,
                r1: self.ll.hi ^ cross_lo,
                r2: self.hh.lo ^ cross_hi,
                r3: self.hh.hi,
            }
            .reduce()
        }
    }
}

pub(crate) const URM_BATCH: usize = 8;

struct WorkerState {
    partial_ab: [KaratsubaAcc; ELL],
    partial_c: [KaratsubaAcc; ELL],
    batch_ab_bytes: [[[u8; 64]; 1 << N_MEDIUM]; URM_BATCH],
    batch_c_bytes: [[[u8; 64]; 1 << N_MEDIUM]; URM_BATCH],
    batch_n_b_med: [u8; URM_BATCH],
    batch_eq_lo: [F128; URM_BATCH],
    a_col: [F8; ELL],
    b_col: [F8; ELL],
    local_res_ab: [F128; ELL],
    local_res_c_s: [F128; ELL],
}

impl WorkerState {
    fn new() -> Self {
        Self {
            partial_ab: [KaratsubaAcc::ZERO; ELL],
            partial_c: [KaratsubaAcc::ZERO; ELL],
            batch_ab_bytes: [[[0u8; 64]; 1 << N_MEDIUM]; URM_BATCH],
            batch_c_bytes: [[[0u8; 64]; 1 << N_MEDIUM]; URM_BATCH],
            batch_n_b_med: [1 << N_MEDIUM; URM_BATCH],
            batch_eq_lo: [F128::ZERO; URM_BATCH],
            a_col: [F8::ZERO; ELL],
            b_col: [F8::ZERO; ELL],
            local_res_ab: [F128::ZERO; ELL],
            local_res_c_s: [F128::ZERO; ELL],
        }
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn process_one_x_hi(
    x_hi: usize,
    big_lo_size: usize,
    n_lo_and_inner: usize,
    within_outer_mask: usize,
    b_med_counts: &[u8],
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    eq_lo_scaled: &[F128],
    eq_hi_val: F128,
    convert: &[F128],
    state: &mut WorkerState,
) {
    state
        .partial_ab
        .iter_mut()
        .for_each(|p| *p = KaratsubaAcc::ZERO);
    state
        .partial_c
        .iter_mut()
        .for_each(|p| *p = KaratsubaAcc::ZERO);

    let n_lo = n_lo_and_inner - N_INNER;
    let mut batch_len = 0usize;

    for x_outer_lo in 0..big_lo_size {
        let x_outer = x_outer_lo | (x_hi << n_lo);
        let within_hash_outer = x_outer & within_outer_mask;
        let n_b_med = b_med_counts[within_hash_outer] as usize;
        if n_b_med == 0 {
            continue;
        }

        let chunk_byte_base = ((x_outer_lo << N_INNER) | (x_hi << n_lo_and_inner)) * N_CHUNKS;
        let eq_lo_val = eq_lo_scaled[x_outer_lo];

        for b_med in 0..n_b_med {
            shift_reduce_inner_ab(
                a_packed,
                b_packed,
                inv_table,
                chunk_byte_base,
                b_med,
                &mut state.batch_ab_bytes[batch_len][b_med],
                &mut state.a_col,
                &mut state.b_col,
            );
        }
        for b_med in 0..n_b_med {
            let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;
            let c_in: &[u8; 64] = (&c_packed[byte_base_b..byte_base_b + 64])
                .try_into()
                .expect("64 c-bytes per medium position");
            bit_transpose_64bytes(c_in, &mut state.batch_c_bytes[batch_len][b_med]);
        }
        let prev_n = state.batch_n_b_med[batch_len] as usize;
        if n_b_med < prev_n {
            for b_med in n_b_med..prev_n {
                state.batch_ab_bytes[batch_len][b_med] = [0u8; 64];
                state.batch_c_bytes[batch_len][b_med] = [0u8; 64];
            }
        }
        state.batch_n_b_med[batch_len] = n_b_med as u8;
        state.batch_eq_lo[batch_len] = eq_lo_val;
        batch_len += 1;
        if batch_len == URM_BATCH {
            kernels::accumulate_convert_batch(
                &state.batch_ab_bytes,
                &state.batch_c_bytes,
                &state.batch_n_b_med,
                &state.batch_eq_lo,
                URM_BATCH,
                convert,
                &mut state.partial_ab,
                &mut state.partial_c,
            );
            batch_len = 0;
        }
    }

    if batch_len > 0 {
        kernels::accumulate_convert_batch(
            &state.batch_ab_bytes,
            &state.batch_c_bytes,
            &state.batch_n_b_med,
            &state.batch_eq_lo,
            batch_len,
            convert,
            &mut state.partial_ab,
            &mut state.partial_c,
        );
    }

    for lane in 0..ELL {
        state.local_res_ab[lane] += eq_hi_val * state.partial_ab[lane].reduce();
        state.local_res_c_s[lane] += eq_hi_val * state.partial_c[lane].reduce();
    }
}

struct WorkerStateWithSHatV {
    partial_ab: [KaratsubaAcc; ELL],
    partial_c_0: [KaratsubaAcc; ELL],
    partial_c_1: [KaratsubaAcc; ELL],
    batch_ab_bytes: [[[u8; 64]; 1 << N_MEDIUM]; URM_BATCH],
    batch_c_bytes: [[[u8; 64]; 1 << N_MEDIUM]; URM_BATCH],
    batch_n_b_med: [u8; URM_BATCH],
    batch_eq_lo: [F128; URM_BATCH],
    a_col: [F8; ELL],
    b_col: [F8; ELL],
    local_res_ab: [F128; ELL],
    local_res_c_s_0: [F128; ELL],
    local_res_c_s_1: [F128; ELL],
}

impl WorkerStateWithSHatV {
    fn new() -> Self {
        Self {
            partial_ab: [KaratsubaAcc::ZERO; ELL],
            partial_c_0: [KaratsubaAcc::ZERO; ELL],
            partial_c_1: [KaratsubaAcc::ZERO; ELL],
            batch_ab_bytes: [[[0u8; 64]; 1 << N_MEDIUM]; URM_BATCH],
            batch_c_bytes: [[[0u8; 64]; 1 << N_MEDIUM]; URM_BATCH],
            batch_n_b_med: [1 << N_MEDIUM; URM_BATCH],
            batch_eq_lo: [F128::ZERO; URM_BATCH],
            a_col: [F8::ZERO; ELL],
            b_col: [F8::ZERO; ELL],
            local_res_ab: [F128::ZERO; ELL],
            local_res_c_s_0: [F128::ZERO; ELL],
            local_res_c_s_1: [F128::ZERO; ELL],
        }
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn process_one_x_hi_with_s_hat_v(
    x_hi: usize,
    big_lo_size: usize,
    n_lo_and_inner: usize,
    within_outer_mask: usize,
    b_med_counts: &[u8],
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    eq_lo_scaled: &[F128],
    eq_hi_val: F128,
    convert: &[F128],
    convert_c_pair: &[F128],
    state: &mut WorkerStateWithSHatV,
) {
    state
        .partial_ab
        .iter_mut()
        .for_each(|p| *p = KaratsubaAcc::ZERO);
    state
        .partial_c_0
        .iter_mut()
        .for_each(|p| *p = KaratsubaAcc::ZERO);
    state
        .partial_c_1
        .iter_mut()
        .for_each(|p| *p = KaratsubaAcc::ZERO);

    let n_lo = n_lo_and_inner - N_INNER;
    let mut batch_len = 0usize;

    for x_outer_lo in 0..big_lo_size {
        let x_outer = x_outer_lo | (x_hi << n_lo);
        let within_hash_outer = x_outer & within_outer_mask;
        let n_b_med = b_med_counts[within_hash_outer] as usize;
        if n_b_med == 0 {
            continue;
        }

        let chunk_byte_base = ((x_outer_lo << N_INNER) | (x_hi << n_lo_and_inner)) * N_CHUNKS;
        let eq_lo_val = eq_lo_scaled[x_outer_lo];

        for b_med in 0..n_b_med {
            shift_reduce_inner_ab(
                a_packed,
                b_packed,
                inv_table,
                chunk_byte_base,
                b_med,
                &mut state.batch_ab_bytes[batch_len][b_med],
                &mut state.a_col,
                &mut state.b_col,
            );
        }
        let prev_n = state.batch_n_b_med[batch_len] as usize;
        if n_b_med < prev_n {
            for b_med in n_b_med..prev_n {
                state.batch_ab_bytes[batch_len][b_med] = [0u8; 64];
            }
        }
        let n_m = n_b_med.div_ceil(2);
        let prev_m = prev_n.div_ceil(2);
        for m in 0..n_m {
            let b0 = 2 * m;
            let b1 = b0 + 1;
            let off0 = chunk_byte_base + b0 * N_CHUNKS * 8;
            let c0: &[u8; 64] = (&c_packed[off0..off0 + 64])
                .try_into()
                .expect("64 c-bytes per medium position");
            let c1: Option<&[u8; 64]> = if b1 < n_b_med {
                let off1 = chunk_byte_base + b1 * N_CHUNKS * 8;
                Some(
                    (&c_packed[off1..off1 + 64])
                        .try_into()
                        .expect("64 c-bytes per medium position"),
                )
            } else {
                None
            };
            let (head, tail) = state.batch_c_bytes[batch_len].split_at_mut(b1);
            kernels::bit_transpose_and_pack_c_pair_64bytes(c0, c1, &mut head[b0], &mut tail[0]);
        }
        if n_m < prev_m {
            for m in n_m..prev_m {
                state.batch_c_bytes[batch_len][2 * m] = [0u8; 64];
                state.batch_c_bytes[batch_len][2 * m + 1] = [0u8; 64];
            }
        }
        state.batch_n_b_med[batch_len] = n_b_med as u8;
        state.batch_eq_lo[batch_len] = eq_lo_val;
        batch_len += 1;
        if batch_len == URM_BATCH {
            kernels::accumulate_convert_batch_with_s_hat_v(
                &state.batch_ab_bytes,
                &state.batch_c_bytes,
                &state.batch_n_b_med,
                &state.batch_eq_lo,
                URM_BATCH,
                convert,
                convert_c_pair,
                &mut state.partial_ab,
                &mut state.partial_c_0,
                &mut state.partial_c_1,
            );
            batch_len = 0;
        }
    }

    if batch_len > 0 {
        kernels::accumulate_convert_batch_with_s_hat_v(
            &state.batch_ab_bytes,
            &state.batch_c_bytes,
            &state.batch_n_b_med,
            &state.batch_eq_lo,
            batch_len,
            convert,
            convert_c_pair,
            &mut state.partial_ab,
            &mut state.partial_c_0,
            &mut state.partial_c_1,
        );
    }

    for lane in 0..ELL {
        state.local_res_ab[lane] += eq_hi_val * state.partial_ab[lane].reduce();
        state.local_res_c_s_0[lane] += eq_hi_val * state.partial_c_0[lane].reduce();
        state.local_res_c_s_1[lane] += eq_hi_val * state.partial_c_1[lane].reduce();
    }
}

fn build_b_med_counts(padding: &PaddingSpec) -> (usize, Vec<u8>) {
    const STRIDE: usize = 1 << (K_SKIP + N_INNER);
    const B_MED_WINDOW: usize = 1 << (K_SKIP + 3);
    const N_B_MED_MAX: usize = 1 << N_MEDIUM;

    if padding.k_log < K_SKIP + N_INNER {
        return (0, vec![N_B_MED_MAX as u8]);
    }
    let within_outer_bits = padding.k_log - K_SKIP - N_INNER;
    let within_outer_count = 1usize << within_outer_bits;
    let within_outer_mask = within_outer_count - 1;
    let useful = padding.useful_bits_per_block;
    let counts: Vec<u8> = (0..within_outer_count)
        .map(|w| {
            let block_start = w * STRIDE;
            if block_start >= useful {
                0u8
            } else {
                let bits_left = useful - block_start;
                let processed = bits_left.div_ceil(B_MED_WINDOW);
                processed.min(N_B_MED_MAX) as u8
            }
        })
        .collect();
    (within_outer_mask, counts)
}

pub fn round1_shift_reduce_extract_c_packed(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    k_skip: usize,
    r: &[F128],
    inv_table: &InvNttTableByteSingleGf8,
) -> (Vec<F128>, Vec<F128>) {
    round1_shift_reduce_extract_c_packed_padded(
        a_packed,
        b_packed,
        c_packed,
        m,
        k_skip,
        r,
        inv_table,
        &PaddingSpec::dense(m),
    )
}

pub fn round1_shift_reduce_extract_c_packed_padded(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    k_skip: usize,
    r: &[F128],
    inv_table: &InvNttTableByteSingleGf8,
    padding: &PaddingSpec,
) -> (Vec<F128>, Vec<F128>) {
    use rayon::prelude::*;

    assert_eq!(k_skip, K_SKIP, "optimized variant is k_skip=6 only");
    assert!(
        m >= k_skip + N_INNER,
        "m must be ≥ k_skip + N_INNER ({}) for the shift_reduce optimization",
        k_skip + N_INNER
    );
    let total_bytes = (1usize << m) / 8;
    assert_eq!(a_packed.len(), total_bytes);
    assert_eq!(b_packed.len(), total_bytes);
    assert_eq!(c_packed.len(), total_bytes);
    assert_eq!(r.len(), m);
    assert_eq!(inv_table.k, k_skip);

    let eq = SplitEqGhash::new(&r[k_skip + N_INNER..]);
    let big_lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    let n_lo_and_inner = eq.n_lo + N_INNER;

    let d_inv_val = d_inv();
    let eq_lo_scaled: Vec<F128> = eq.lo.iter().map(|v| *v * d_inv_val).collect();
    let convert = convert_table();
    let eq_hi = &eq.hi;

    let (within_outer_mask, b_med_counts) = build_b_med_counts(padding);

    let (res_ab, res_c_s) = (0..hi_size)
        .into_par_iter()
        .fold(
            || Box::new(WorkerState::new()),
            |mut state, x_hi| {
                let eq_hi_val = eq_hi[x_hi];
                process_one_x_hi(
                    x_hi,
                    big_lo_size,
                    n_lo_and_inner,
                    within_outer_mask,
                    &b_med_counts,
                    a_packed,
                    b_packed,
                    c_packed,
                    inv_table,
                    &eq_lo_scaled,
                    eq_hi_val,
                    convert,
                    &mut state,
                );
                state
            },
        )
        .map(|s| (s.local_res_ab, s.local_res_c_s))
        .reduce(
            || ([F128::ZERO; ELL], [F128::ZERO; ELL]),
            |(mut ab1, mut c1), (ab2, c2)| {
                for i in 0..ELL {
                    ab1[i] += ab2[i];
                    c1[i] += c2[i];
                }
                (ab1, c1)
            },
        );

    let res_c_lifted = ntt_extend_f128_vec_ghash(&res_c_s, inv_table);
    (res_ab.to_vec(), res_c_lifted)
}

pub fn round1_shift_reduce_extract_c_packed_padded_with_s_hat_v(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    k_skip: usize,
    r: &[F128],
    inv_table: &InvNttTableByteSingleGf8,
    padding: &PaddingSpec,
) -> (Vec<F128>, Vec<F128>, Vec<F128>) {
    use rayon::prelude::*;

    assert_eq!(k_skip, K_SKIP, "optimized variant is k_skip=6 only");
    assert!(
        m >= k_skip + N_INNER,
        "m must be ≥ k_skip + N_INNER ({}) for the shift_reduce optimization",
        k_skip + N_INNER
    );
    let total_bytes = (1usize << m) / 8;
    assert_eq!(a_packed.len(), total_bytes);
    assert_eq!(b_packed.len(), total_bytes);
    assert_eq!(c_packed.len(), total_bytes);
    assert_eq!(r.len(), m);
    assert_eq!(inv_table.k, k_skip);

    let eq = SplitEqGhash::new(&r[k_skip + N_INNER..]);
    let big_lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    let n_lo_and_inner = eq.n_lo + N_INNER;

    let d_inv_val = d_inv();
    let eq_lo_scaled: Vec<F128> = eq.lo.iter().map(|v| *v * d_inv_val).collect();
    let convert = convert_table();
    let convert_c_pair = convert_c_pair_table();
    let eq_hi = &eq.hi;

    let (within_outer_mask, b_med_counts) = build_b_med_counts(padding);

    let (res_ab, res_c_s_0, res_c_s_1) = (0..hi_size)
        .into_par_iter()
        .fold(
            || Box::new(WorkerStateWithSHatV::new()),
            |mut state, x_hi| {
                let eq_hi_val = eq_hi[x_hi];
                process_one_x_hi_with_s_hat_v(
                    x_hi,
                    big_lo_size,
                    n_lo_and_inner,
                    within_outer_mask,
                    &b_med_counts,
                    a_packed,
                    b_packed,
                    c_packed,
                    inv_table,
                    &eq_lo_scaled,
                    eq_hi_val,
                    convert,
                    convert_c_pair,
                    &mut state,
                );
                state
            },
        )
        .map(|s| (s.local_res_ab, s.local_res_c_s_0, s.local_res_c_s_1))
        .reduce(
            || ([F128::ZERO; ELL], [F128::ZERO; ELL], [F128::ZERO; ELL]),
            |(mut ab1, mut c0_1, mut c1_1), (ab2, c0_2, c1_2)| {
                for i in 0..ELL {
                    ab1[i] += ab2[i];
                    c0_1[i] += c0_2[i];
                    c1_1[i] += c1_2[i];
                }
                (ab1, c0_1, c1_1)
            },
        );

    let alpha = phi8(F8(0x02));
    let mut res_c_s_combined = [F128::ZERO; ELL];
    for i in 0..ELL {
        res_c_s_combined[i] = res_c_s_0[i] + alpha * res_c_s_1[i];
    }
    let res_c_lifted = ntt_extend_f128_vec_ghash(&res_c_s_combined, inv_table);

    let c_2 = c_2_small_f128();
    let mut s_hat_v_c = vec![F128::ZERO; 2 * ELL];
    for lane in 0..ELL {
        s_hat_v_c[lane] = c_2 * res_c_s_0[lane];
        s_hat_v_c[ELL + lane] = c_2 * res_c_s_1[lane];
    }

    (res_ab.to_vec(), res_c_lifted, s_hat_v_c)
}

struct WorkerStateAbOnly {
    partial_ab: [KaratsubaAcc; ELL],
    batch_ab_bytes: [[[u8; 64]; 1 << N_MEDIUM]; URM_BATCH],
    batch_n_b_med: [u8; URM_BATCH],
    batch_eq_lo: [F128; URM_BATCH],
    a_col: [F8; ELL],
    b_col: [F8; ELL],
    local_res_ab: [F128; ELL],
}

impl WorkerStateAbOnly {
    fn new() -> Self {
        Self {
            partial_ab: [KaratsubaAcc::ZERO; ELL],
            batch_ab_bytes: [[[0u8; 64]; 1 << N_MEDIUM]; URM_BATCH],
            batch_n_b_med: [1 << N_MEDIUM; URM_BATCH],
            batch_eq_lo: [F128::ZERO; URM_BATCH],
            a_col: [F8::ZERO; ELL],
            b_col: [F8::ZERO; ELL],
            local_res_ab: [F128::ZERO; ELL],
        }
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn process_one_x_hi_ab_only(
    x_hi: usize,
    big_lo_size: usize,
    n_lo_and_inner: usize,
    within_outer_mask: usize,
    b_med_counts: &[u8],
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    eq_lo_scaled: &[F128],
    eq_lo_gfni_mats: Option<&[[[u64; 16]; 16]]>,
    eq_hi_val: F128,
    convert: &[F128],
    state: &mut WorkerStateAbOnly,
) {
    let n_lo = n_lo_and_inner - N_INNER;

    if let Some(mats_slice) = eq_lo_gfni_mats {
        let (ab_rows_slice, rest) = state.batch_ab_bytes.split_at_mut(1);
        let ab_rows = &mut ab_rows_slice[0];
        let ab_plane_bank = &mut rest[0];
        *ab_plane_bank = [[0u8; 64]; 1 << N_MEDIUM];
        let mut any_active = false;

        for x_outer_lo in 0..big_lo_size {
            let x_outer = x_outer_lo | (x_hi << n_lo);
            let within_hash_outer = x_outer & within_outer_mask;
            let n_b_med = b_med_counts[within_hash_outer] as usize;
            if n_b_med == 0 {
                continue;
            }
            any_active = true;
            let chunk_byte_base = ((x_outer_lo << N_INNER) | (x_hi << n_lo_and_inner)) * N_CHUNKS;
            for b_med in 0..n_b_med {
                shift_reduce_inner_ab(
                    a_packed,
                    b_packed,
                    inv_table,
                    chunk_byte_base,
                    b_med,
                    &mut ab_rows[b_med],
                    &mut state.a_col,
                    &mut state.b_col,
                );
            }
            kernels::convert_ab_gfni_nomul(
                ab_rows,
                n_b_med,
                &mats_slice[x_outer_lo],
                ab_plane_bank,
            );
        }

        if any_active {
            kernels::fold_ab_plane_banks(
                core::slice::from_ref(ab_plane_bank),
                core::slice::from_ref(&eq_hi_val),
                1,
                &mut state.local_res_ab,
            );
        }
        return;
    }

    state
        .partial_ab
        .iter_mut()
        .for_each(|p| *p = KaratsubaAcc::ZERO);

    let mut batch_len = 0usize;

    for x_outer_lo in 0..big_lo_size {
        let x_outer = x_outer_lo | (x_hi << n_lo);
        let within_hash_outer = x_outer & within_outer_mask;
        let n_b_med = b_med_counts[within_hash_outer] as usize;
        if n_b_med == 0 {
            continue;
        }

        let chunk_byte_base = ((x_outer_lo << N_INNER) | (x_hi << n_lo_and_inner)) * N_CHUNKS;
        let eq_lo_val = eq_lo_scaled[x_outer_lo];

        for b_med in 0..n_b_med {
            shift_reduce_inner_ab(
                a_packed,
                b_packed,
                inv_table,
                chunk_byte_base,
                b_med,
                &mut state.batch_ab_bytes[batch_len][b_med],
                &mut state.a_col,
                &mut state.b_col,
            );
        }
        state.batch_n_b_med[batch_len] = n_b_med as u8;
        state.batch_eq_lo[batch_len] = eq_lo_val;
        batch_len += 1;
        if batch_len == URM_BATCH {
            kernels::accumulate_convert_batch_ab_only(
                &state.batch_ab_bytes,
                &state.batch_n_b_med,
                &state.batch_eq_lo,
                URM_BATCH,
                convert,
                &mut state.partial_ab,
            );
            batch_len = 0;
        }
    }

    if batch_len > 0 {
        kernels::accumulate_convert_batch_ab_only(
            &state.batch_ab_bytes,
            &state.batch_n_b_med,
            &state.batch_eq_lo,
            batch_len,
            convert,
            &mut state.partial_ab,
        );
    }

    for lane in 0..ELL {
        state.local_res_ab[lane] += eq_hi_val * state.partial_ab[lane].reduce();
    }
}

pub fn round1_shift_reduce_ab_only_packed_padded(
    a_packed: &[u8],
    b_packed: &[u8],
    m: usize,
    k_skip: usize,
    r: &[F128],
    inv_table: &InvNttTableByteSingleGf8,
    padding: &PaddingSpec,
) -> Vec<F128> {
    use rayon::prelude::*;

    assert_eq!(k_skip, K_SKIP, "optimized variant is k_skip=6 only");
    assert!(m >= k_skip + N_INNER);
    let total_bytes = (1usize << m) / 8;
    assert_eq!(a_packed.len(), total_bytes);
    assert_eq!(b_packed.len(), total_bytes);
    assert_eq!(r.len(), m);
    assert_eq!(inv_table.k, k_skip);

    let eq = SplitEqGhash::new(&r[k_skip + N_INNER..]);
    let big_lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    let n_lo_and_inner = eq.n_lo + N_INNER;

    let d_inv_val = d_inv();
    let eq_lo_scaled: Vec<F128> = eq.lo.iter().map(|v| *v * d_inv_val).collect();
    let convert = convert_table();
    let (within_outer_mask, b_med_counts) = build_b_med_counts(padding);
    let eq_lo_gfni_mats: Option<Vec<[[u64; 16]; 16]>> = if kernels::has_avx512_gfni() {
        let mut mats = vec![[[0u64; 16]; 16]; big_lo_size];
        for (x_outer_lo, mat) in mats.iter_mut().enumerate() {
            kernels::build_ab_eq_fold_mats(eq_lo_scaled[x_outer_lo], convert, mat);
        }
        Some(mats)
    } else {
        None
    };
    let eq_lo_gfni_mats_ref = eq_lo_gfni_mats.as_deref();

    let res_ab = (0..hi_size)
        .into_par_iter()
        .fold(
            || Box::new(WorkerStateAbOnly::new()),
            |mut state, x_hi| {
                process_one_x_hi_ab_only(
                    x_hi,
                    big_lo_size,
                    n_lo_and_inner,
                    within_outer_mask,
                    &b_med_counts,
                    a_packed,
                    b_packed,
                    inv_table,
                    &eq_lo_scaled,
                    eq_lo_gfni_mats_ref,
                    eq.hi[x_hi],
                    convert,
                    &mut state,
                );
                state
            },
        )
        .map(|s| s.local_res_ab)
        .reduce(
            || [F128::ZERO; ELL],
            |mut ab1, ab2| {
                for i in 0..ELL {
                    ab1[i] += ab2[i];
                }
                ab1
            },
        );

    res_ab.to_vec()
}

#[cfg(test)]
fn round1_shift_reduce_extract_c_packed_serial(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    k_skip: usize,
    r: &[F128],
    inv_table: &InvNttTableByteSingleGf8,
) -> (Vec<F128>, Vec<F128>) {
    assert_eq!(k_skip, K_SKIP);
    assert!(m >= k_skip + N_INNER);
    let total_bytes = (1usize << m) / 8;
    assert_eq!(a_packed.len(), total_bytes);
    assert_eq!(b_packed.len(), total_bytes);
    assert_eq!(c_packed.len(), total_bytes);
    assert_eq!(r.len(), m);
    assert_eq!(inv_table.k, k_skip);

    let eq = SplitEqGhash::new(&r[k_skip + N_INNER..]);
    let big_lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    let n_lo_and_inner = eq.n_lo + N_INNER;

    let d_inv_val = d_inv();
    let eq_lo_scaled: Vec<F128> = eq.lo.iter().map(|v| *v * d_inv_val).collect();
    let convert = convert_table();

    let (within_outer_mask, b_med_counts) = build_b_med_counts(&PaddingSpec::dense(m));

    let mut state = WorkerState::new();
    for x_hi in 0..hi_size {
        process_one_x_hi(
            x_hi,
            big_lo_size,
            n_lo_and_inner,
            within_outer_mask,
            &b_med_counts,
            a_packed,
            b_packed,
            c_packed,
            inv_table,
            &eq_lo_scaled,
            eq.hi[x_hi],
            convert,
            &mut state,
        );
    }

    let res_c_lifted = ntt_extend_f128_vec_ghash(&state.local_res_c_s, inv_table);
    (state.local_res_ab.to_vec(), res_c_lifted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ntt::AdditiveNttGf8;
    use crate::zerocheck::univariate_skip::round1_naive;

    #[test]
    fn friendly_challenges_f2_independent() {
        let mut basis: Vec<u128> = small_challenges_ghash()
            .iter()
            .chain(medium_challenges_ghash().iter())
            .map(|f| ((f.hi as u128) << 64) | (f.lo as u128))
            .collect();
        assert_eq!(
            basis.len(),
            7,
            "expected 3 small + 4 medium friendly values"
        );

        let mut rank = 0usize;
        for col in (0..128).rev() {
            let mask = 1u128 << col;
            let pivot = (rank..basis.len()).find(|&i| basis[i] & mask != 0);
            if let Some(p) = pivot {
                basis.swap(rank, p);
                for i in 0..basis.len() {
                    if i != rank && basis[i] & mask != 0 {
                        basis[i] ^= basis[rank];
                    }
                }
                rank += 1;
            }
        }
        assert_eq!(
            rank, 7,
            "friendly challenges must be F₂-linearly independent in F₁₂₈; \
             zerocheck and Ligerito L0 soundness depend on it"
        );
    }

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

    fn build_protocol_r(m: usize, outer: &[F128]) -> Vec<F128> {
        assert_eq!(outer.len(), m - K_SKIP - N_INNER);
        let mut r = vec![F128::ZERO; m];
        for (i, &small) in small_challenges_ghash().iter().enumerate() {
            r[K_SKIP + i] = small;
        }
        for (i, &med) in medium_challenges_ghash().iter().enumerate() {
            r[K_SKIP + 3 + i] = med;
        }
        for (i, &x) in outer.iter().enumerate() {
            r[K_SKIP + N_INNER + i] = x;
        }
        r
    }

    fn make_inv_table() -> InvNttTableByteSingleGf8 {
        let ntt_s = AdditiveNttGf8::new(K_SKIP, F8::ZERO);
        let ntt_l = AdditiveNttGf8::new(K_SKIP, F8(1u8 << K_SKIP));
        InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l)
    }

    #[test]
    fn output_shape() {
        let m = 14;
        let mut rng = Rng::new(1);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c = rng.bits(1 << m);
        let outer = rng.f128_vec(m - K_SKIP - N_INNER);
        let r = build_protocol_r(m, &outer);
        let table = make_inv_table();

        let (ab, c_l) = round1_shift_reduce_extract_c(&a, &b, &c, m, K_SKIP, &r, &table);
        assert_eq!(ab.len(), ELL);
        assert_eq!(c_l.len(), ELL);
    }

    #[test]
    fn deterministic() {
        let m = 14;
        let mut rng = Rng::new(2);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let c = rng.bits(1 << m);
        let outer = rng.f128_vec(m - K_SKIP - N_INNER);
        let r = build_protocol_r(m, &outer);
        let table = make_inv_table();

        let out1 = round1_shift_reduce_extract_c(&a, &b, &c, m, K_SKIP, &r, &table);
        let out2 = round1_shift_reduce_extract_c(&a, &b, &c, m, K_SKIP, &r, &table);
        assert_eq!(out1, out2);
    }

    #[test]
    fn matches_naive_with_c_s_factor() {
        let c_s = c_s_f128();
        for &m in &[13usize, 14, 15] {
            let mut rng = Rng::new(100 + m as u64);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            let c = rng.bits(1 << m);
            let outer = rng.f128_vec(m - K_SKIP - N_INNER);
            let r = build_protocol_r(m, &outer);
            let table = make_inv_table();

            let (naive_ab, naive_c) = round1_naive(&a, &b, &c, m, K_SKIP, &r);
            let (opt_ab, opt_c) = round1_shift_reduce_extract_c(&a, &b, &c, m, K_SKIP, &r, &table);

            for i in 0..ELL {
                let lhs = naive_ab[i] + naive_c[i];
                let rhs = c_s * (opt_ab[i] + opt_c[i]);
                assert_eq!(
                    lhs, rhs,
                    "combined mismatch at m={m}, i={i}:\n  naive={lhs:?}\n  C_s·opt={rhs:?}"
                );
            }

            for i in 0..ELL {
                assert_eq!(naive_ab[i], c_s * opt_ab[i], "AB mismatch at i={i}");
                assert_eq!(naive_c[i], c_s * opt_c[i], "C mismatch at i={i}");
            }
        }
    }

    #[test]
    fn small_and_medium_challenges_sanity() {
        let med = medium_challenges_ghash();
        let powers = [1u64 << 1, 1u64 << 2, 1u64 << 4, 1u64 << 8];
        for (i, &p) in powers.iter().enumerate() {
            let g = F128 { lo: p, hi: 0 };
            assert_eq!(med[i] * (F128::ONE + g), g, "β_{i} identity");
        }

        let d_inv_val = d_inv();
        let g1 = F128 {
            lo: 1u64 << 1,
            hi: 0,
        };
        let g2 = F128 {
            lo: 1u64 << 2,
            hi: 0,
        };
        let g4 = F128 {
            lo: 1u64 << 4,
            hi: 0,
        };
        let g8 = F128 {
            lo: 1u64 << 8,
            hi: 0,
        };
        let d = (F128::ONE + g1) * (F128::ONE + g2) * (F128::ONE + g4) * (F128::ONE + g8);
        assert_eq!(d * d_inv_val, F128::ONE);
    }

    #[test]
    fn parallel_matches_serial() {
        use crate::zerocheck::univariate_skip::pack_bits;

        for &m in &[13usize, 14, 15] {
            let mut rng = Rng::new(0xCAFE_F00D + m as u64);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            let c = rng.bits(1 << m);
            let outer = rng.f128_vec(m - K_SKIP - N_INNER);
            let r = build_protocol_r(m, &outer);
            let table = make_inv_table();
            let a_p = pack_bits(&a);
            let b_p = pack_bits(&b);
            let c_p = pack_bits(&c);

            let (par_ab, par_c) =
                round1_shift_reduce_extract_c_packed(&a_p, &b_p, &c_p, m, K_SKIP, &r, &table);
            let (ser_ab, ser_c) = round1_shift_reduce_extract_c_packed_serial(
                &a_p, &b_p, &c_p, m, K_SKIP, &r, &table,
            );

            assert_eq!(par_ab, ser_ab, "parallel AB ≠ serial AB at m={m}");
            assert_eq!(par_c, ser_c, "parallel C ≠ serial C at m={m}");
        }
    }

    #[test]
    fn padded_matches_dense_with_zero_padding() {
        use crate::zerocheck::PaddingSpec;
        use crate::zerocheck::univariate_skip::pack_bits;

        let cases = [
            (14usize, 15_409usize, 0usize),
            (15, 31_401, 0),
            (16, 42_560, 0),
            (16, 42_560, 3),
        ];

        for (k_log, useful_bits, n_blocks_log) in cases {
            let m = k_log + n_blocks_log;
            assert!(m >= K_SKIP + N_INNER);

            let mut rng = Rng::new(0xBEEF_DEAD_u64.wrapping_add((k_log * 31 + m) as u64));
            let n_blocks = 1usize << n_blocks_log;
            let total_bits = 1usize << m;
            let block_size = 1usize << k_log;

            let mut a = rng.bits(total_bits);
            let mut b = rng.bits(total_bits);
            let mut c = rng.bits(total_bits);
            for blk in 0..n_blocks {
                for j in useful_bits..block_size {
                    let idx = blk * block_size + j;
                    a[idx] = false;
                    b[idx] = false;
                    c[idx] = false;
                }
            }

            let outer = rng.f128_vec(m - K_SKIP - N_INNER);
            let r = build_protocol_r(m, &outer);
            let table = make_inv_table();
            let a_p = pack_bits(&a);
            let b_p = pack_bits(&b);
            let c_p = pack_bits(&c);

            let (dense_ab, dense_c) =
                round1_shift_reduce_extract_c_packed(&a_p, &b_p, &c_p, m, K_SKIP, &r, &table);
            let padding = PaddingSpec {
                k_log,
                useful_bits_per_block: useful_bits,
            };
            let (padded_ab, padded_c) = round1_shift_reduce_extract_c_packed_padded(
                &a_p, &b_p, &c_p, m, K_SKIP, &r, &table, &padding,
            );

            assert_eq!(
                dense_ab, padded_ab,
                "AB mismatch: k_log={k_log}, useful={useful_bits}, m={m}"
            );
            assert_eq!(
                dense_c, padded_c,
                "C mismatch: k_log={k_log}, useful={useful_bits}, m={m}"
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_bit_transpose_matches_scalar() {
        let mut rng = Rng::new(0xB17_BB17);
        for _ in 0..64 {
            let mut input = [0u8; 64];
            for byte in input.iter_mut() {
                *byte = (rng.next_u64() & 0xff) as u8;
            }
            let mut out_scalar = [0u8; 64];
            let mut out_neon = [0u8; 64];
            bit_transpose_64bytes_scalar(&input, &mut out_scalar);
            unsafe { bit_transpose_64bytes_neon(&input, &mut out_neon) };
            assert_eq!(out_scalar, out_neon, "bit_transpose disagreement");
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_fused_inner_matches_scalar_inner() {
        let mut rng = Rng::new(0xF050D);
        let m = 14;
        let table = make_inv_table();
        let a_bits = rng.bits(1 << m);
        let b_bits = rng.bits(1 << m);
        let a_packed = super::super::univariate_skip::pack_bits(&a_bits);
        let b_packed = super::super::univariate_skip::pack_bits(&b_bits);

        let mut a_col = vec![F8::ZERO; ELL];
        let mut b_col = vec![F8::ZERO; ELL];

        for &(chunk_byte_base, b_med) in &[(0usize, 0usize), (64, 5), (1024, 7), (4096, 15)] {
            let needed = chunk_byte_base + b_med * N_CHUNKS * 8 + 8 * N_CHUNKS;
            if needed > a_packed.len() {
                continue;
            }
            let mut out_scalar = [0u8; 64];
            let mut out_fused = [0u8; 64];
            kernels::shift_reduce_inner_ab_scalar(
                &a_packed,
                &b_packed,
                &table,
                chunk_byte_base,
                b_med,
                &mut out_scalar,
                &mut a_col,
                &mut b_col,
            );
            shift_reduce_inner_ab_fused_neon(
                &a_packed,
                &b_packed,
                &table,
                chunk_byte_base,
                b_med,
                &mut out_fused,
            );
            assert_eq!(
                out_scalar, out_fused,
                "fused-neon disagrees with scalar at (base={chunk_byte_base}, b_med={b_med})"
            );
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "gfni"))]
    #[test]
    fn x86_gfni_sse_inner_matches_scalar_inner() {
        let mut rng = Rng::new(0xF050D);
        let m = 14;
        let table = make_inv_table();
        let a_bits = rng.bits(1 << m);
        let b_bits = rng.bits(1 << m);
        let a_packed = super::super::univariate_skip::pack_bits(&a_bits);
        let b_packed = super::super::univariate_skip::pack_bits(&b_bits);

        let mut a_col = vec![F8::ZERO; ELL];
        let mut b_col = vec![F8::ZERO; ELL];

        for &(chunk_byte_base, b_med) in &[(0usize, 0usize), (64, 5), (1024, 7), (4096, 15)] {
            let needed = chunk_byte_base + b_med * N_CHUNKS * 8 + 8 * N_CHUNKS;
            if needed > a_packed.len() {
                continue;
            }
            let mut out_scalar = [0u8; 64];
            let mut out_x86 = [0u8; 64];
            kernels::shift_reduce_inner_ab_scalar(
                &a_packed,
                &b_packed,
                &table,
                chunk_byte_base,
                b_med,
                &mut out_scalar,
                &mut a_col,
                &mut b_col,
            );
            unsafe {
                shift_reduce_inner_ab_x86_sse(
                    &a_packed,
                    &b_packed,
                    &table,
                    chunk_byte_base,
                    b_med,
                    &mut out_x86,
                    &mut a_col,
                    &mut b_col,
                );
            }
            assert_eq!(
                out_scalar, out_x86,
                "gfni disagrees with scalar at (base={chunk_byte_base}, b_med={b_med})"
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x86_gfni_avx512_inner_matches_scalar_inner() {
        if !kernels::has_avx512_gfni() {
            return;
        }
        let mut rng = Rng::new(0xA5_512);
        let m = 14;
        let table = make_inv_table();
        let a_bits = rng.bits(1 << m);
        let b_bits = rng.bits(1 << m);
        let a_packed = super::super::univariate_skip::pack_bits(&a_bits);
        let b_packed = super::super::univariate_skip::pack_bits(&b_bits);
        let mut a_col = vec![F8::ZERO; ELL];
        let mut b_col = vec![F8::ZERO; ELL];

        for &(chunk_byte_base, b_med) in &[(0usize, 0usize), (64, 5), (1024, 7), (4096, 15)] {
            let needed = chunk_byte_base + b_med * N_CHUNKS * 8 + 8 * N_CHUNKS;
            if needed > a_packed.len() {
                continue;
            }
            let mut out_scalar = [0u8; 64];
            let mut out_avx512 = [0u8; 64];
            kernels::shift_reduce_inner_ab_scalar(
                &a_packed,
                &b_packed,
                &table,
                chunk_byte_base,
                b_med,
                &mut out_scalar,
                &mut a_col,
                &mut b_col,
            );
            unsafe {
                kernels::x86_64::shift_reduce_inner_ab_x86_avx512(
                    &a_packed,
                    &b_packed,
                    &table,
                    chunk_byte_base,
                    b_med,
                    &mut out_avx512,
                );
            }
            assert_eq!(
                out_scalar, out_avx512,
                "avx512/gfni disagrees with scalar at (base={chunk_byte_base}, b_med={b_med})"
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x86_gfni_bit_plane_convert_and_fold_matches_scalar() {
        if !kernels::has_avx512_gfni() {
            return;
        }
        let mut rng = Rng::new(0x6F28_B175);
        let convert = convert_table();
        for &n_b_med in &[1usize, 4, 7, 15, 16] {
            let eq_lo_val = rng.f128();
            let weight = rng.f128();
            let mut ab_rows = [[0u8; ELL]; 1 << N_MEDIUM];
            for row in ab_rows.iter_mut().take(n_b_med) {
                for b in row.iter_mut() {
                    *b = (rng.next_u64() & 0xFF) as u8;
                }
            }
            let mut expected = [F128::ZERO; ELL];
            for b_med in 0..n_b_med {
                let base = b_med * 256;
                for l in 0..ELL {
                    expected[l] +=
                        weight * eq_lo_val * convert[base + (ab_rows[b_med][l] as usize)];
                }
            }
            let mut mats = [[0u64; 16]; 16];
            let mut banks = [[[0u8; 64]; 16]; 1];
            let mut got = [F128::ZERO; ELL];
            kernels::build_ab_eq_fold_mats(eq_lo_val, convert, &mut mats);
            kernels::convert_ab_gfni_nomul(&ab_rows, n_b_med, &mats, &mut banks[0]);
            kernels::fold_ab_plane_banks(&banks, &[weight], 1, &mut got);
            assert_eq!(
                got, expected,
                "GFNI bit-plane convert+fold mismatch at n_b_med={n_b_med}"
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn x86_gfni_c4_fold_64_rows_matches_scalar() {
        if !kernels::has_avx512_gfni() {
            return;
        }
        let mut rng = Rng::new(0xC4_6F28);
        let convert = convert_table();
        let coeffs = [rng.f128(), rng.f128(), rng.f128(), rng.f128()];
        let mut rows = [0u64; 64];
        for r in rows.iter_mut() {
            *r = rng.next_u64();
        }
        let mut bcast_mats = [[0u64; 8]; 64];
        unsafe {
            kernels::x86_64::build_gf2_c4_fold_mats_x86_gfni(
                &convert[..8 * 256],
                &coeffs,
                &mut bcast_mats,
            );
        }
        for &dead_lines in &[0u8, 0xC0u8] {
            let mut got = [F128::ZERO; 16];
            unsafe {
                kernels::x86_64::fold_64_gf2_rows_gfni_c4(&rows, &bcast_mats, &mut got, dead_lines);
            }
            for q in 0..16usize {
                let line = q / 2;
                let expected = if (dead_lines & (1u8 << line)) != 0 {
                    F128::ZERO
                } else {
                    let mut sum = F128::ZERO;
                    for a in 0..4usize {
                        let w = rows[4 * q + a];
                        let mut fw = F128::ZERO;
                        for j in 0..8usize {
                            let b = ((w >> (8 * j)) & 0xFF) as usize;
                            fw += convert[j * 256 + b];
                        }
                        sum += coeffs[a] * fw;
                    }
                    sum
                };
                assert_eq!(
                    got[q], expected,
                    "GFNI C=4 fold64 mismatch at dead_lines=0x{dead_lines:02x}, q={q}"
                );
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_inner_matches_scalar_inner() {
        let mut rng = Rng::new(0x5EED);
        let m = 14;
        let table = make_inv_table();
        let n_chunks = 1 << (K_SKIP / 8);
        let _ = n_chunks;
        let a_bits = rng.bits(1 << m);
        let b_bits = rng.bits(1 << m);
        let a_packed = super::super::univariate_skip::pack_bits(&a_bits);
        let b_packed = super::super::univariate_skip::pack_bits(&b_bits);

        let mut a_col = vec![F8::ZERO; ELL];
        let mut b_col = vec![F8::ZERO; ELL];

        for &(chunk_byte_base, b_med) in &[(0usize, 0usize), (64, 5), (1024, 7), (4096, 15)] {
            let needed = chunk_byte_base + b_med * N_CHUNKS * 8 + 8 * N_CHUNKS;
            if needed > a_packed.len() {
                continue;
            }
            let mut out_scalar = [0u8; 64];
            let mut out_neon = [0u8; 64];
            kernels::shift_reduce_inner_ab_scalar(
                &a_packed,
                &b_packed,
                &table,
                chunk_byte_base,
                b_med,
                &mut out_scalar,
                &mut a_col,
                &mut b_col,
            );
            shift_reduce_inner_ab_neon(
                &a_packed,
                &b_packed,
                &table,
                chunk_byte_base,
                b_med,
                &mut out_neon,
                &mut a_col,
                &mut b_col,
            );
            assert_eq!(
                out_scalar, out_neon,
                "scalar/neon inner disagree at (base={chunk_byte_base}, b_med={b_med})"
            );
        }
    }

    #[test]
    fn convert_table_structure() {
        let t = convert_table();
        let mut g_pow = F128::ONE;
        for b in 0..16 {
            for &v in &[0u8, 1, 0x57, 0xFF] {
                let expected = g_pow * PHI_8_TABLE[v as usize];
                assert_eq!(t[b * 256 + v as usize], expected, "b={b}, v={v}");
            }
            g_pow = mul_by_x(g_pow);
        }
    }

    #[test]
    fn fusion_matches_existing_and_scalar_oracle() {
        use crate::zerocheck::univariate_skip::round1_extract_c_packed_with_s_hat_v;

        for &m in &[13usize, 14, 15] {
            let mut rng = Rng::new(0xF00D_u64.wrapping_add(m as u64));
            let a = pack_bits(&rng.bits(1 << m));
            let b = pack_bits(&rng.bits(1 << m));
            let c = pack_bits(&rng.bits(1 << m));
            let mut r = vec![F128::ZERO; m];
            for i in 0..3 {
                r[K_SKIP + i] = phi8(F8(SMALL_CHAL_F8[i]));
            }
            let medium = crate::zerocheck::univariate_skip_optimized::medium_challenges_ghash();
            for i in 0..4 {
                r[K_SKIP + 3 + i] = medium[i];
            }
            for i in 0..K_SKIP {
                r[i] = rng.f128();
            }
            for i in (K_SKIP + N_INNER)..m {
                r[i] = rng.f128();
            }

            let inv_table = {
                let ntt_s = crate::ntt::AdditiveNttGf8::new(K_SKIP, F8::ZERO);
                let ntt_l = crate::ntt::AdditiveNttGf8::new(K_SKIP, F8(1u8 << K_SKIP));
                InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l)
            };

            let (ref_ab, ref_c) = round1_shift_reduce_extract_c_packed_padded(
                &a,
                &b,
                &c,
                m,
                K_SKIP,
                &r,
                &inv_table,
                &PaddingSpec::dense(m),
            );

            let (_, _, oracle_s_hat_v) =
                round1_extract_c_packed_with_s_hat_v(&a, &b, &c, m, K_SKIP, &r, &inv_table);

            let (got_ab, got_c, got_s_hat_v) =
                round1_shift_reduce_extract_c_packed_padded_with_s_hat_v(
                    &a,
                    &b,
                    &c,
                    m,
                    K_SKIP,
                    &r,
                    &inv_table,
                    &PaddingSpec::dense(m),
                );

            assert_eq!(got_ab, ref_ab, "res_ab mismatch at m={m}");
            assert_eq!(got_c, ref_c, "res_c_lifted mismatch at m={m}");
            assert_eq!(got_s_hat_v.len(), 2 * ELL, "s_hat_v length at m={m}");
            assert_eq!(
                got_s_hat_v, oracle_s_hat_v,
                "s_hat_v_c mismatch vs scalar oracle at m={m}"
            );
        }
    }

    #[test]
    #[ignore]
    fn bench_step51_gfni() {
        use std::time::Instant;
        let has_gfni = kernels::has_avx512_gfni();
        println!("=== Step 5.1 Benchmark (has_avx512_gfni = {has_gfni}) ===");

        let m = 24usize;
        let mut rng = Rng::new(0x5151_2026);
        let a_bits = rng.bits(1 << 20);
        let b_bits = rng.bits(1 << 20);
        let c_bits = rng.bits(1 << 20);
        let a_p20 = pack_bits(&a_bits);
        let b_p20 = pack_bits(&b_bits);
        let c_p20 = pack_bits(&c_bits);
        let a_packed: Vec<u8> = a_p20.iter().copied().cycle().take(1 << (m - 3)).collect();
        let b_packed: Vec<u8> = b_p20.iter().copied().cycle().take(1 << (m - 3)).collect();
        let c_packed: Vec<u8> = c_p20.iter().copied().cycle().take(1 << (m - 3)).collect();
        let outer = rng.f128_vec(m - K_SKIP - N_INNER);
        let r = build_protocol_r(m, &outer);
        let inv_table = make_inv_table();
        let keccak_pad = PaddingSpec {
            k_log: 16,
            useful_bits_per_block: 42_560,
        };
        let dense_pad = PaddingSpec::dense(m);

        let _ = round1_shift_reduce_ab_only_packed_padded(
            &a_packed,
            &b_packed,
            m,
            K_SKIP,
            &r,
            &inv_table,
            &keccak_pad,
        );
        let iters = 10usize;
        let t0 = Instant::now();
        let mut ab_keccak = Vec::new();
        for _ in 0..iters {
            ab_keccak = round1_shift_reduce_ab_only_packed_padded(
                &a_packed,
                &b_packed,
                m,
                K_SKIP,
                &r,
                &inv_table,
                &keccak_pad,
            );
        }
        let dt_ab_keccak = t0.elapsed().as_secs_f64() * 1000.0 / (iters as f64);

        let _ = round1_shift_reduce_ab_only_packed_padded(
            &a_packed, &b_packed, m, K_SKIP, &r, &inv_table, &dense_pad,
        );
        let t1 = Instant::now();
        let mut ab_dense = Vec::new();
        for _ in 0..iters {
            ab_dense = round1_shift_reduce_ab_only_packed_padded(
                &a_packed, &b_packed, m, K_SKIP, &r, &inv_table, &dense_pad,
            );
        }
        let dt_ab_dense = t1.elapsed().as_secs_f64() * 1000.0 / (iters as f64);

        let _ = round1_shift_reduce_extract_c_packed_padded_with_s_hat_v(
            &a_packed,
            &b_packed,
            &c_packed,
            m,
            K_SKIP,
            &r,
            &inv_table,
            &keccak_pad,
        );
        let t2 = Instant::now();
        let mut abc_keccak = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..iters {
            abc_keccak = round1_shift_reduce_extract_c_packed_padded_with_s_hat_v(
                &a_packed,
                &b_packed,
                &c_packed,
                m,
                K_SKIP,
                &r,
                &inv_table,
                &keccak_pad,
            );
        }
        let dt_abc_keccak = t2.elapsed().as_secs_f64() * 1000.0 / (iters as f64);

        let mut out_inner = [0u8; 64];
        let mut a_col = [F8::ZERO; ELL];
        let mut b_col = [F8::ZERO; ELL];
        let inner_iters = 65_536usize;
        let t3 = Instant::now();
        for i in 0..inner_iters {
            let chunk_base = (i & 255) * 512;
            kernels::shift_reduce_inner_ab(
                &a_packed,
                &b_packed,
                &inv_table,
                chunk_base,
                i & 15,
                &mut out_inner,
                &mut a_col,
                &mut b_col,
            );
            std::hint::black_box(&out_inner);
        }
        let dt_inner_ns = t3.elapsed().as_nanos() as f64 / (inner_iters as f64);

        println!(
            "STEP51_BENCH_RESULT: has_gfni={has_gfni} inner_ab_ns={dt_inner_ns:.2} round1_ab_keccak_m20_ms={dt_ab_keccak:.3} round1_ab_dense_m20_ms={dt_ab_dense:.3} round1_abc_keccak_m20_ms={dt_abc_keccak:.3} ab0={:?} abc0={:?}",
            ab_keccak[0], abc_keccak.0[0]
        );
        assert_eq!(
            ab_keccak, abc_keccak.0,
            "ab_only vs abc_with_s_hat_v mismatch"
        );
        let _ = ab_dense;

        let m_mlv = 26usize;
        let a_mlv: Vec<u8> = a_p20
            .iter()
            .copied()
            .cycle()
            .take(1 << (m_mlv - 3))
            .collect();
        let b_mlv: Vec<u8> = b_p20
            .iter()
            .copied()
            .cycle()
            .take(1 << (m_mlv - 3))
            .collect();
        let tau_0 = rng.f128();
        let rho_0 = rng.f128();
        let rho_1 = rng.f128();
        let mlv_chal = rng.f128_vec(m_mlv - K_SKIP);
        let fold_tbl = crate::zerocheck::multilinear::UniSkipFoldTable::new(K_SKIP, tau_0);
        let blake3_pad = PaddingSpec {
            k_log: 14,
            useful_bits_per_block: 15_409,
        };
        let n_quarter = 1usize << (m_mlv - K_SKIP - 2);
        let mut a_out = vec![F128::ZERO; n_quarter];
        let mut b_out = vec![F128::ZERO; n_quarter];

        let _ = crate::zerocheck::multilinear::uni_skip_round2_and_round3_lookahead_packed_padded(
            &a_mlv,
            &b_mlv,
            m_mlv,
            K_SKIP,
            &fold_tbl,
            &mlv_chal,
            &blake3_pad,
        );
        let mlv_iters = 10usize;
        let t_r23 = Instant::now();
        let mut r23_out =
            crate::zerocheck::multilinear::uni_skip_round2_and_round3_lookahead_packed_padded(
                &a_mlv,
                &b_mlv,
                m_mlv,
                K_SKIP,
                &fold_tbl,
                &mlv_chal,
                &blake3_pad,
            );
        for _ in 1..mlv_iters {
            r23_out =
                crate::zerocheck::multilinear::uni_skip_round2_and_round3_lookahead_packed_padded(
                    &a_mlv,
                    &b_mlv,
                    m_mlv,
                    K_SKIP,
                    &fold_tbl,
                    &mlv_chal,
                    &blake3_pad,
                );
        }
        let dt_r23_ms = t_r23.elapsed().as_secs_f64() * 1000.0 / (mlv_iters as f64);

        let _ =
            crate::zerocheck::multilinear::uni_skip_fold_2vars_into_and_round4_msg_packed_padded(
                &a_mlv,
                &b_mlv,
                &mut a_out,
                &mut b_out,
                m_mlv,
                K_SKIP,
                &fold_tbl,
                rho_0,
                rho_1,
                &mlv_chal[2..],
                &blake3_pad,
            );
        let t_r4 = Instant::now();
        let mut r4_out = (F128::ZERO, F128::ZERO);
        for _ in 0..mlv_iters {
            r4_out = crate::zerocheck::multilinear::uni_skip_fold_2vars_into_and_round4_msg_packed_padded(
                &a_mlv,
                &b_mlv,
                &mut a_out,
                &mut b_out,
                m_mlv,
                K_SKIP,
                &fold_tbl,
                rho_0,
                rho_1,
                &mlv_chal[2..],
                &blake3_pad,
            );
        }
        let dt_r4_ms = t_r4.elapsed().as_secs_f64() * 1000.0 / (mlv_iters as f64);

        let _ = crate::zerocheck::multilinear::uni_skip_fold_2vars_into_and_round45_lookahead_packed_padded(
            &a_mlv,
            &b_mlv,
            &mut a_out,
            &mut b_out,
            m_mlv,
            K_SKIP,
            &fold_tbl,
            rho_0,
            rho_1,
            &mlv_chal[2..],
            &blake3_pad,
        );
        let t_r45 = Instant::now();
        let mut r45_out = crate::zerocheck::multilinear::uni_skip_fold_2vars_into_and_round45_lookahead_packed_padded(
            &a_mlv,
            &b_mlv,
            &mut a_out,
            &mut b_out,
            m_mlv,
            K_SKIP,
            &fold_tbl,
            rho_0,
            rho_1,
            &mlv_chal[2..],
            &blake3_pad,
        );
        for _ in 1..mlv_iters {
            r45_out = crate::zerocheck::multilinear::uni_skip_fold_2vars_into_and_round45_lookahead_packed_padded(
                &a_mlv,
                &b_mlv,
                &mut a_out,
                &mut b_out,
                m_mlv,
                K_SKIP,
                &fold_tbl,
                rho_0,
                rho_1,
                &mlv_chal[2..],
                &blake3_pad,
            );
        }
        let dt_r45_ms = t_r45.elapsed().as_secs_f64() * 1000.0 / (mlv_iters as f64);

        println!(
            "STEP51_MLV_BENCH_RESULT: has_gfni={has_gfni} round23_m22_ms={dt_r23_ms:.3} round4_m22_ms={dt_r4_ms:.3} round45_m22_ms={dt_r45_ms:.3} r23_m1={:?} r4_m1={:?} r45_m1={:?} a_out0={:?}",
            r23_out.msg2_1, r4_out.0, r45_out.msg2_1, a_out[0]
        );
    }
}
