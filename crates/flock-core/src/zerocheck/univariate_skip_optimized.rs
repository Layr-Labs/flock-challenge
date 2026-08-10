//! Round-1 prover message — fully optimized (shift_reduce + extract_c, scalar).
//!
//! NOTE (r858): this file is restored to the promoted kernel tree (r851
//! `63b290a`, pre-r852) — the r852/r853 lo-band task-split is reverted. Board
//! verdict on the split tree after 3 fresh draws (829607e1, 43861e33,
//! 79954f4d): medians 148.042/147.945/147.763 ms, ranks 5/12/16 of the 24
//! 8/8 family runs, exact Mann-Whitney P=0.37 vs the pre-split tree — no
//! measurable benefit, and the split never drew below 147.763 while the
//! promoted pre-split tree drew 147.690/147.713/147.733/147.748. The split was
//! theory (M4 P/E-core draining) never validated on the board; this draw
//! returns the draw stream to the exact code of the current frontier
//! (54b8dbf). Content delta vs r851 = this comment only.
//!
//! Scalar Rust implementation (no NEON). Three layered optimizations on top of
//! the [`super::round1_extract_c`] scaffold:
//!
//! 1. **Geometric small-eq + shift_reduce inner** (3 inner-most rest-dims).
//!    Protocol fixes the three small challenges to
//!    `r[k_skip..k_skip+3] = φ_8([0xF7, 0x53, 0xB5])`, which makes
//!    `eq_small[K] = C_s · α^K` (geometric in α, the AES root in GHASH).
//!    The shift_reduce trick computes
//!    `Σ_K eq_small[K] · φ_8(y_K)  =  C_s · φ_8(reduce(Σ_K y_K << K))`,
//!    replacing 8 F128 mults per lane with 8 u16 XOR-shifts + one F_8
//!    reduction.
//!
//! 2. **Geometric medium-eq + convert table** (4 next rest-dims).
//!    Protocol fixes the four medium challenges to
//!    `β_i = γ^{2^{i-1}} / (1 + γ^{2^{i-1}})`, which makes
//!    `eq_med[b] = γ^b / D` for `D = ∏(1+γ^{2^{i-1}})`.
//!    Precomputed table `convert[b][v] = γ^b · φ_8(v)` (64 KB) reduces the
//!    per-lane medium-eq sum from 16 F128 mults to 16 lookups + 16 XORs.
//!
//! 3. **D⁻¹ absorbed into eq_lo.**
//!    Pre-scale `eq_lo[i] ← eq_lo[i] · D⁻¹` once before the loop; this cancels
//!    the `1/D` from the medium-eq factorization, leaving only the `C_s`
//!    factor in the relative output scaling.
//!
//! Net output relationship vs the naive / structural versions:
//!   `C_s · (res_AB[i] + res_C_lifted[i])  ==  naive_p_ab[i] + naive_p_c[i]`
//! with `C_s = φ_8(0x1C)`.
//!
//! This variant is hardcoded for `k_skip = 6` (ell=64, n_chunks=8, N_INNER=7).

use std::sync::OnceLock;

use crate::field::{F8, F128, PHI_8_TABLE, mul_by_x, phi8};
use crate::ntt::InvNttTableByteSingleGf8;

use super::PaddingSpec;
use super::univariate_skip::{SplitEqGhash, build_eq, ntt_extend_f128_vec_ghash, pack_bits};

mod kernels;

#[cfg(all(test, target_arch = "aarch64"))]
use kernels::aarch64::{
    bit_transpose_64bytes_neon, shift_reduce_inner_ab_fused_neon,
    shift_reduce_inner_ab_fused_neon_checked, shift_reduce_inner_ab_neon,
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

// ---------------------------------------------------------------------------
// Protocol constants — fixed by the optimization design.
// ---------------------------------------------------------------------------

/// Number of variables folded in round 1 for the shift_reduce variant.
pub const K_SKIP: usize = 6;
const ELL: usize = 64;
const N_CHUNKS: usize = 8;
/// Total inner-most dims absorbed by the optimization: 3 small + 4 medium.
const N_INNER: usize = 7;
const N_MEDIUM: usize = 4;

/// The three small-eq challenges (as F_8 values, then embedded via φ_8).
/// Choosing these specific values is what makes `eq_small[K] = C_s · α^K`.
///
/// **Soundness dependency.** These three constants — together with the
/// four medium constants returned by [`medium_challenges_ghash`] — must be
/// **F₂-linearly independent** in F₁₂₈. Zerocheck soundness relies on this
/// (a witness aligned with the friendly subspace would otherwise let the
/// prover cancel the URM message), and so does Ligerito's L0 list-collapse
/// argument (the SZ bound `(m−7)/|F|` for MLE collisions at `r` requires
/// the seven friendly coords to span a 7-dim F₂-subspace). Asserted by
/// `tests::friendly_challenges_f2_independent`.
pub const SMALL_CHAL_F8: [u8; 3] = [0xF7, 0x53, 0xB5];

/// `C_s` as an F_8 value. Verified empirically by the C++ project.
pub const C_S_F8: u8 = 0x1C;

/// The constant `C_s = φ_8(0x1C) ∈ F_{2^128}` — the relative scaling factor
/// between this optimized output and the naive output.
pub fn c_s_f128() -> F128 {
    phi8(F8(C_S_F8))
}

/// The three F_128 small challenges (embeddings of [`SMALL_CHAL_F8`]) — caller
/// must place these at `r[k_skip..k_skip+3]` for the naive cross-check to
/// produce a result related to the optimized output by exactly `C_s`.
pub fn small_challenges_ghash() -> [F128; 3] {
    [
        phi8(F8(SMALL_CHAL_F8[0])),
        phi8(F8(SMALL_CHAL_F8[1])),
        phi8(F8(SMALL_CHAL_F8[2])),
    ]
}

/// The four F_128 medium challenges `β_i = γ^{2^{i-1}} / (1 + γ^{2^{i-1}})`.
/// Caller must place these at `r[k_skip+3..k_skip+7]` for the naive
/// cross-check.
pub fn medium_challenges_ghash() -> [F128; 4] {
    let g1 = F128 {
        lo: 1u64 << 1,
        hi: 0,
    }; // γ^1
    let g2 = F128 {
        lo: 1u64 << 2,
        hi: 0,
    }; // γ^2
    let g4 = F128 {
        lo: 1u64 << 4,
        hi: 0,
    }; // γ^4
    let g8 = F128 {
        lo: 1u64 << 8,
        hi: 0,
    }; // γ^8
    [
        g1 * (F128::ONE + g1).inv(),
        g2 * (F128::ONE + g2).inv(),
        g4 * (F128::ONE + g4).inv(),
        g8 * (F128::ONE + g8).inv(),
    ]
}

/// `C_2 = (1+r_2)(1+r_3)` where `r_2 = φ_8(0x53)` (= `α^2/(1+α^2)`),
/// `r_3 = φ_8(0xB5)` (= `α^4/(1+α^4)`). This is the residual small-eq
/// constant after the first small friendly bit (`b_3[0]`, indexed by
/// `r[k_skip] = φ_8(α)`) has been pulled out for the s_hat_v_c bank split:
///
/// ```text
/// eq([r[k_skip+1], r[k_skip+2]], (b_3[1], b_3[2])) = C_2 · α^{2 b_3[1] + 4 b_3[2]}
/// ```
///
/// Used in [`round1_shift_reduce_extract_c_packed_padded_with_s_hat_v`] to
/// post-scale the raw bank values into canonical `s_hat_v_c` (which
/// `ring_switch::fold_1b_rows` would produce against suffix `r[k_skip+1..m]`).
pub fn c_2_small_f128() -> F128 {
    let r_2 = phi8(F8(SMALL_CHAL_F8[1]));
    let r_3 = phi8(F8(SMALL_CHAL_F8[2]));
    (F128::ONE + r_2) * (F128::ONE + r_3)
}

/// `α⁻¹` in F_128, as a subfield-embedded F_8 element. Used to strip the
/// extra `α` factor from `s_hat_v_c`'s bank 1 (the K-odd lattice's raw
/// contribution is `α · α^{2 b_3[1] + 4 b_3[2]}`; canonical wants just
/// `α^{2 b_3[1] + 4 b_3[2]}`).
pub fn alpha_inv_f128() -> F128 {
    // α in F_8 = byte 0x02 (the polynomial generator). Its inverse is α^254;
    // F8::inv computes it via the standard extended Euclidean / power table.
    phi8(F8(0x02).inv())
}

/// `D = (1+γ)(1+γ^2)(1+γ^4)(1+γ^8)`; `D⁻¹` cancels the medium-eq normalization.
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

/// `D_hi = (1+gamma^4)(1+gamma^8)`.  Direct-fold4 retains the two low
/// medium coordinates instead of collapsing all four here, so its C banks
/// absorb only the normalization belonging to the two high coordinates.
fn compute_d_hi_inv() -> F128 {
    let g4 = F128 {
        lo: 1u64 << 4,
        hi: 0,
    };
    let g8 = F128 {
        lo: 1u64 << 8,
        hi: 0,
    };
    ((F128::ONE + g4) * (F128::ONE + g8)).inv()
}

static D_HI_INV_CACHE: OnceLock<F128> = OnceLock::new();
fn d_hi_inv() -> F128 {
    *D_HI_INV_CACHE.get_or_init(compute_d_hi_inv)
}

// ---------------------------------------------------------------------------
// Convert table: γ^b · φ_8(v) for b ∈ [0, 16), v ∈ [0, 256).
// 16 × 256 × 16 bytes = 64 KB. Computed once, cached via OnceLock.
// ---------------------------------------------------------------------------

const CONVERT_TABLE_SIZE: usize = 16 * 256;

static CONVERT_TABLE_CACHE: OnceLock<Vec<F128>> = OnceLock::new();

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

fn convert_table() -> &'static [F128] {
    CONVERT_TABLE_CACHE.get_or_init(build_convert_table)
}

// ---------------------------------------------------------------------------
// Mask → field tables for the eight-bank C drain.
//
// The C banks reduce to u16 masks (see `kernels::accumulate_c_banks`), so the
// only field work left on the C side is `F128 { lo: mask } * eq_lo_scaled[x]`.
// That map is F2-linear in the mask's 16 bits, so it is a pair of 256-row
// lookups instead of a multiply:
//
//     F128 { lo: m } * eq  ==  T_lo[m & 0xff] + T_hi[m >> 8]
//     T_lo[v] = F128 { lo: v } * eq,   T_hi[v] = F128 { lo: v << 8 } * eq
//
// `eq_lo_scaled` is fixed for the whole round-1 sweep, so the tables are built
// ONCE per prove (not per kernel call) and shared read-only across every x_hi
// worker: 2^n_lo * 512 * 16 B = 8 MiB at the ranked shape, of which only the
// 8 KiB slice for the current `x_outer_lo` is hot. Building per call instead
// would cost ~4 GiB of L1 stores per prove.
//
// Each row is built from the 16 basis elements `X^b * eq` by XOR doubling —
// `X^b * eq` is `mul_by_x` applied b times, so the whole build is 15 shifts +
// 510 XORs per `x_outer_lo`, ~0.5 M stores per prove.
// ---------------------------------------------------------------------------

/// F128 entries per `x_outer_lo`: two 256-row halves (low mask byte, high).
const C_MASK_TABLE_STRIDE: usize = 512;

fn build_c_mask_tables(eq_lo_scaled: &[F128]) -> Vec<F128> {
    use rayon::prelude::*;

    // Fully overwritten below (every index of both halves is written before it
    // is read), so an uninitialized scratch buffer is sound and skips an 8 MiB
    // zeroing pass.
    let mut tables = crate::scratch::take_f128(eq_lo_scaled.len() * C_MASK_TABLE_STRIDE);
    tables
        .par_chunks_mut(C_MASK_TABLE_STRIDE)
        .zip(eq_lo_scaled.par_iter())
        .for_each(|(slot, eq)| {
            let mut basis = [F128::ZERO; 16];
            basis[0] = *eq;
            for b in 1..16 {
                basis[b] = mul_by_x(basis[b - 1]);
            }
            let (t_lo, t_hi) = slot.split_at_mut(256);
            for (half, table) in [t_lo, t_hi].into_iter().enumerate() {
                table[0] = F128::ZERO;
                for b in 0..8 {
                    let (done, rest) = table.split_at_mut(1 << b);
                    let add = basis[half * 8 + b];
                    for (out, seen) in rest[..1 << b].iter_mut().zip(done.iter()) {
                        *out = *seen + add;
                    }
                }
            }
        });
    tables
}

// ---------------------------------------------------------------------------
// Four-retained-coordinate C statistic for experimental direct-fold4.
//
// The incumbent eight banks retain the two small suffix coordinates.  The
// direct-fold4 consumer needs the next two (the low medium coordinates) as
// well.  Write b_med = q + 4h, q,h in [0,4).  For each q and small-bit bank K
// we accumulate
//
//   eq_outer / D_hi * sum_h gamma^(4h) bit_K(C[q + 4h]).
//
// Re-applying eq([beta_0,beta_1], q) = gamma^q / D_lo collapses these 32
// banks exactly to the incumbent eight-bank statistic.  Keeping this
// identity explicit gives the Fold4 experiment a strong local oracle before
// it is allowed onto the ranked proof path.
// ---------------------------------------------------------------------------

const N_C_FOLD4_GROUPS: usize = 4;
const N_C_FOLD4_BANKS: usize = N_C_FOLD4_GROUPS * N_C_BANKS;
const C_FOLD4_MASK_TABLE_STRIDE: usize = 16;
const C_FOLD4_PAIR_MASK_TABLE_STRIDE: usize = 256;

/// Build the tiny mask table for the retained-medium C drain.  A four-bit
/// index selects polynomial-basis exponents {0,4,8,12}, not {0,1,2,3}.
#[cfg_attr(not(test), allow(dead_code))]
fn build_c_fold4_mask_tables(eq_lo_hi_scaled: &[F128]) -> Vec<F128> {
    use rayon::prelude::*;

    let mut tables = crate::scratch::take_f128(eq_lo_hi_scaled.len() * C_FOLD4_MASK_TABLE_STRIDE);
    tables
        .par_chunks_mut(C_FOLD4_MASK_TABLE_STRIDE)
        .zip(eq_lo_hi_scaled.par_iter())
        .for_each(|(slot, eq)| {
            let mut basis = [F128::ZERO; N_C_FOLD4_GROUPS];
            basis[0] = *eq;
            for h in 1..N_C_FOLD4_GROUPS {
                let mut next = basis[h - 1];
                for _ in 0..N_C_FOLD4_GROUPS {
                    next = mul_by_x(next);
                }
                basis[h] = next;
            }

            slot[0] = F128::ZERO;
            for h in 0..N_C_FOLD4_GROUPS {
                let (done, rest) = slot.split_at_mut(1 << h);
                let add = basis[h];
                for (out, seen) in rest[..1 << h].iter_mut().zip(done.iter()) {
                    *out = *seen + add;
                }
            }
        });
    tables
}

/// Fuse two adjacent low-coordinate Fold4 tables.  If their four-bit masks
/// are `a` and `b`, respectively, entry `a | (b << 4)` is exactly
/// `T_even[a] + T_odd[b]`.  The paired C kernel can therefore replace two
/// field-table loads and two accumulator read/modify/writes with one of each.
/// A final unpaired low slot uses an all-zero odd table.
fn build_c_fold4_pair_mask_tables(eq_lo_hi_scaled: &[F128]) -> Vec<F128> {
    use rayon::prelude::*;

    let n_pairs = eq_lo_hi_scaled.len().div_ceil(2);
    let mut tables = crate::scratch::take_f128(n_pairs * C_FOLD4_PAIR_MASK_TABLE_STRIDE);
    tables
        .par_chunks_mut(C_FOLD4_PAIR_MASK_TABLE_STRIDE)
        .enumerate()
        .for_each(|(pair, slot)| {
            let mut singles = [[F128::ZERO; C_FOLD4_MASK_TABLE_STRIDE]; 2];
            for side in 0..2 {
                let Some(eq) = eq_lo_hi_scaled.get(2 * pair + side).copied() else {
                    continue;
                };
                let mut basis = [F128::ZERO; N_C_FOLD4_GROUPS];
                basis[0] = eq;
                for h in 1..N_C_FOLD4_GROUPS {
                    let mut next = basis[h - 1];
                    for _ in 0..N_C_FOLD4_GROUPS {
                        next = mul_by_x(next);
                    }
                    basis[h] = next;
                }
                for h in 0..N_C_FOLD4_GROUPS {
                    let (done, rest) = singles[side].split_at_mut(1 << h);
                    let add = basis[h];
                    for (out, seen) in rest[..1 << h].iter_mut().zip(done.iter()) {
                        *out = *seen + add;
                    }
                }
            }

            for b in 0..C_FOLD4_MASK_TABLE_STRIDE {
                for a in 0..C_FOLD4_MASK_TABLE_STRIDE {
                    slot[a | (b << 4)] = singles[0][a] + singles[1][b];
                }
            }
        });
    tables
}

/// Portable correctness kernel for the 32-bank direct-fold4 C statistic.
/// Architecture-specific acceleration is deliberately downstream of the
/// collapse oracle: this version is simple enough to audit instruction by
/// instruction and is also useful as a differential reference.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn accumulate_c_fold4_banks_scalar(
    c_block: &[u8; (1 << N_MEDIUM) * ELL],
    n_b_med: usize,
    mask_table: &[F128],
    partial_c: &mut [[F128; ELL]; N_C_FOLD4_BANKS],
) {
    debug_assert!(n_b_med <= 1 << N_MEDIUM);
    debug_assert_eq!(mask_table.len(), C_FOLD4_MASK_TABLE_STRIDE);

    let mut transposed = [[0u8; ELL]; 1 << N_MEDIUM];
    for b_med in 0..n_b_med {
        let row: &[u8; ELL] = c_block[b_med * ELL..(b_med + 1) * ELL]
            .try_into()
            .expect("64 c-bytes per medium position");
        bit_transpose_64bytes(row, &mut transposed[b_med]);
    }

    for lane in 0..ELL {
        let mut masks = [0u16; N_C_BANKS];
        for (b_med, row) in transposed.iter().enumerate().take(n_b_med) {
            let c = row[lane];
            for (k, mask) in masks.iter_mut().enumerate() {
                *mask |= u16::from((c >> k) & 1) << b_med;
            }
        }

        for q in 0..N_C_FOLD4_GROUPS {
            for (k, mask) in masks.iter().copied().enumerate() {
                let mut nibble = 0usize;
                for h in 0..N_C_FOLD4_GROUPS {
                    nibble |= usize::from((mask >> (q + N_C_FOLD4_GROUPS * h)) & 1) << h;
                }
                partial_c[q * N_C_BANKS + k][lane] += mask_table[nibble];
            }
        }
    }
}

/// Portable reference for one retained-medium `q` group.  Unlike the
/// 32-bank oracle above, this touches only rows `q + 4h` and maintains only
/// the eight small-bit banks that belong to that group.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn accumulate_c_fold4_q_banks_scalar(
    c_block: &[u8; (1 << N_MEDIUM) * ELL],
    n_b_med: usize,
    q: usize,
    mask_table: &[F128],
    partial_c: &mut [[F128; ELL]; N_C_BANKS],
) {
    debug_assert!(n_b_med <= 1 << N_MEDIUM);
    debug_assert!(q < N_C_FOLD4_GROUPS);
    debug_assert_eq!(mask_table.len(), C_FOLD4_MASK_TABLE_STRIDE);

    let mut transposed = [[0u8; ELL]; N_C_FOLD4_GROUPS];
    for (h, row_out) in transposed.iter_mut().enumerate() {
        let b_med = q + N_C_FOLD4_GROUPS * h;
        if b_med < n_b_med {
            let row: &[u8; ELL] = c_block[b_med * ELL..(b_med + 1) * ELL]
                .try_into()
                .expect("64 c-bytes per medium position");
            bit_transpose_64bytes(row, row_out);
        }
    }

    for lane in 0..ELL {
        for (k, bank) in partial_c.iter_mut().enumerate() {
            let mut nibble = 0usize;
            for (h, row) in transposed.iter().enumerate() {
                nibble |= usize::from((row[lane] >> k) & 1) << h;
            }
            bank[lane] += mask_table[nibble];
        }
    }
}

/// Portable differential reference for the paired q-local drain. Padding for
/// the two adjacent blocks remains independent; a missing row contributes a
/// zero nibble on only its own half of the eight-bit table index.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[allow(clippy::too_many_arguments)]
fn accumulate_c_fold4_q_pair_banks_scalar(
    c_block_even: &[u8; (1 << N_MEDIUM) * ELL],
    n_b_med_even: usize,
    c_block_odd: &[u8; (1 << N_MEDIUM) * ELL],
    n_b_med_odd: usize,
    q: usize,
    pair_mask_table: &[F128],
    partial_c: &mut [[F128; ELL]; N_C_BANKS],
) {
    debug_assert!(n_b_med_even <= 1 << N_MEDIUM);
    debug_assert!(n_b_med_odd <= 1 << N_MEDIUM);
    debug_assert!(q < N_C_FOLD4_GROUPS);
    debug_assert_eq!(pair_mask_table.len(), C_FOLD4_PAIR_MASK_TABLE_STRIDE);

    let mut transposed = [[[0u8; ELL]; N_C_FOLD4_GROUPS]; 2];
    for (side, (c_block, n_b_med)) in [(c_block_even, n_b_med_even), (c_block_odd, n_b_med_odd)]
        .into_iter()
        .enumerate()
    {
        for (h, row_out) in transposed[side].iter_mut().enumerate() {
            let b_med = q + N_C_FOLD4_GROUPS * h;
            if b_med < n_b_med {
                let row: &[u8; ELL] = c_block[b_med * ELL..(b_med + 1) * ELL]
                    .try_into()
                    .expect("64 c-bytes per medium position");
                bit_transpose_64bytes(row, row_out);
            }
        }
    }

    for lane in 0..ELL {
        for (k, bank) in partial_c.iter_mut().enumerate() {
            let mut index = 0usize;
            for (side, rows) in transposed.iter().enumerate() {
                let mut nibble = 0usize;
                for (h, row) in rows.iter().enumerate() {
                    nibble |= usize::from((row[lane] >> k) & 1) << h;
                }
                index |= nibble << (4 * side);
            }
            bank[lane] += pair_mask_table[index];
        }
    }
}

/// Re-apply the two retained medium-coordinate eq weights.  The output must
/// be byte-identical to the incumbent full-medium eight-bank accumulator.
fn collapse_c_fold4_banks(banks: &[[F128; ELL]; N_C_FOLD4_BANKS]) -> [[F128; ELL]; N_C_BANKS] {
    let medium = medium_challenges_ghash();
    let low_eq = build_eq(&medium[..2]);
    let mut out = [[F128::ZERO; ELL]; N_C_BANKS];
    for q in 0..N_C_FOLD4_GROUPS {
        for k in 0..N_C_BANKS {
            let src = &banks[q * N_C_BANKS + k];
            for lane in 0..ELL {
                out[k][lane] += low_eq[q] * src[lane];
            }
        }
    }
    out
}

#[inline]
pub fn bit_transpose_64bytes(input: &[u8; 64], output: &mut [u8; 64]) {
    kernels::bit_transpose_64bytes(input, output);
}

/// Challenge-independent AB half of the optimized round-1 kernel.
///
/// The storage has exactly the same byte length and block layout as either
/// packed input: every `(x_outer, b_med)` consumes one 64-byte A block and one
/// 64-byte B block and produces one 64-byte transformed block. Keeping this
/// in a separate scratch allocation is intentional: round 2 still needs the
/// original A and B tables after the round-1 transcript challenge is sampled.
pub struct Round1AbInner {
    storage: Vec<F128>,
}

impl Round1AbInner {
    #[inline]
    fn as_bytes(&self) -> &[u8] {
        unsafe {
            core::slice::from_raw_parts(
                self.storage.as_ptr() as *const u8,
                self.storage.len() * core::mem::size_of::<F128>(),
            )
        }
    }

    /// Resident scratch bytes retained until the challenge-weighted finish.
    pub fn len_bytes(&self) -> usize {
        self.storage.len() * core::mem::size_of::<F128>()
    }

    /// Donate the now-dead transform to a byte-oriented scratch consumer
    /// without changing the allocation's element type or deallocation layout.
    pub(crate) fn into_scratch_bytes(mut self) -> crate::scratch::ScratchBytes {
        crate::scratch::ScratchBytes::from_initialized_f128(core::mem::take(&mut self.storage))
    }
}

impl Drop for Round1AbInner {
    fn drop(&mut self) {
        crate::scratch::give_f128(core::mem::take(&mut self.storage));
    }
}

/// Kill switch for the non-temporal store flavor of the deferred AB
/// precompute output: `FLOCK_NO_ZC_AB_PRE_NT=1` restores the incumbent plain
/// cached stores as a same-binary control. Store flavor only — the bytes
/// written are identical.
pub const ENV_NO_ZC_AB_PRE_NT: &str = "FLOCK_NO_ZC_AB_PRE_NT";

fn ab_pre_nt_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os(ENV_NO_ZC_AB_PRE_NT).is_none())
}

/// Kill switch for the register-direct `stnp` drain of the AB precompute
/// rows: `FLOCK_NO_ZC_AB_PRE_NT_DIRECT=1` restores the incumbent stack-bounce
/// flavor (kernel `vst1q` into a 64-byte temporary, then an `ldp`/`stnp`
/// copy) as a same-binary control. Only meaningful while the NT drain itself
/// is on; store flavor only — the bytes written are identical.
pub const ENV_NO_ZC_AB_PRE_NT_DIRECT: &str = "FLOCK_NO_ZC_AB_PRE_NT_DIRECT";

fn ab_pre_nt_direct_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os(ENV_NO_ZC_AB_PRE_NT_DIRECT).is_none())
}

/// Kill switch for the QS3 compacted AB-precompute store. Default ON: the
/// dead tail rows `[n_b_med, 16)` of each `x_outer` chunk are left untouched
/// instead of being zero-filled, because no `ab_inner` consumer ever reads
/// them (see the drop of the tail loop in
/// [`precompute_round1_ab_inner_packed_padded_with_flavor`]). Set exactly
/// `FLOCK_NO_AB_COMPACT_STORE=1` to restore the incumbent zero-fill as a
/// same-binary A/B control; any other value (or unset) keeps the compacted
/// path. The live region is byte-identical either way, so the proof bytes are
/// unchanged — the switch exists purely for screening the memory-traffic win.
pub const ENV_NO_AB_COMPACT_STORE: &str = "FLOCK_NO_AB_COMPACT_STORE";

fn ab_compact_store_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os(ENV_NO_AB_COMPACT_STORE).as_deref() != Some(std::ffi::OsStr::new("1"))
    })
}

/// Non-temporal 64-byte store (L1 stack bounce → `stnp` pair burst), the same
/// best-effort cache-bypass idiom as the witness stripe drain. The precompute
/// output is a 512 MiB write-once surface whose consumer runs tens of
/// milliseconds later (after the commitment root), so its lines are never
/// usefully cache-resident; plain stores cost a write-allocate RFO read of
/// every line while the streamed GPU commit is saturating the same memory
/// system.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn store_nt_64(src: *const u8, dst: *mut u8) {
    unsafe {
        core::arch::asm!(
            "ldp {t0:q}, {t1:q}, [{src}]",
            "stnp {t0:q}, {t1:q}, [{dst}]",
            "ldp {t0:q}, {t1:q}, [{src}, #32]",
            "stnp {t0:q}, {t1:q}, [{dst}, #32]",
            src = in(reg) src,
            dst = in(reg) dst,
            t0 = out(vreg) _,
            t1 = out(vreg) _,
            options(nostack)
        );
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline(always)]
unsafe fn store_nt_64(src: *const u8, dst: *mut u8) {
    unsafe { core::ptr::copy_nonoverlapping(src, dst, 64) };
}

/// Precompute the challenge-independent inverse-NTT/product/shift-reduce AB
/// transform. The result can be produced before the commitment root is
/// available and consumed later by
/// [`round1_shift_reduce_extract_c_packed_padded_with_precomputed_ab`].
pub fn precompute_round1_ab_inner_packed_padded(
    a_packed: &[u8],
    b_packed: &[u8],
    m: usize,
    k_skip: usize,
    inv_table: &InvNttTableByteSingleGf8,
    padding: &PaddingSpec,
) -> Round1AbInner {
    precompute_round1_ab_inner_packed_padded_with_flavor(
        a_packed,
        b_packed,
        m,
        k_skip,
        inv_table,
        padding,
        ab_pre_nt_enabled(),
        ab_compact_store_enabled(),
    )
}

/// Store-flavor-parameterized body; the public wrapper passes the latched env
/// choices, tests compare arms byte-for-byte in one process. `nt` selects the
/// non-temporal drain vs the incumbent cached store; `compact` selects the
/// QS3 tail-skip vs the incumbent zero-fill of the dead skipped-`b_med` rows.
fn precompute_round1_ab_inner_packed_padded_with_flavor(
    a_packed: &[u8],
    b_packed: &[u8],
    m: usize,
    k_skip: usize,
    inv_table: &InvNttTableByteSingleGf8,
    padding: &PaddingSpec,
    nt: bool,
    compact: bool,
) -> Round1AbInner {
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
    assert_eq!(inv_table.k, k_skip);
    assert_eq!(total_bytes % core::mem::size_of::<F128>(), 0);

    let (within_outer_mask, b_med_counts) = build_b_med_counts(padding);
    // The BLAKE3 R1CS has two 8192-bit windows per 16384-bit block. Besides
    // the full all-ones B rows at b_med 0/1 and the single-K0 tail, two mixed
    // rows have fixed one-valued K subsets: K0..1 at first-window b_med 2 and
    // K4..7 at second-window b_med 13. Restrict runtime sniffing to these five
    // candidates; every other block enters the generic kernel directly.
    let blake3_static_layout = padding.k_log == 14 && padding.useful_bits_per_block == 15_409;
    let static_b_context = kernels::prepare_static_b_context(inv_table, blake3_static_layout);
    const OUTER_BYTES: usize = (1 << N_MEDIUM) * 64;
    debug_assert_eq!(OUTER_BYTES, (1 << N_INNER) * N_CHUNKS);

    // Reuse an A-sized resident F128 allocation from the prover scratch pool.
    // Treating it as bytes is valid under the read contract every consumer
    // honors: each LIVE byte — rows `[0, n_b_med)` of every `x_outer` chunk —
    // is written below before it is read. With the QS3 compacted store the
    // dead tail rows `[n_b_med, 16)` are left recycled/uninitialized; this is
    // sound because NO consumer ever reads them (all bound their per-`b_med`
    // reads by the same `n_b_med`, derived from the same `padding`), and round
    // two rewrites the whole donated buffer before use. The kill switch
    // `FLOCK_NO_AB_COMPACT_STORE=1` restores the historical invariant "every
    // byte written below (explicit zero fills for the skipped-b_med holes)".
    let mut storage = crate::scratch::take_f128(total_bytes / core::mem::size_of::<F128>());
    let out_bytes: &mut [u8] =
        unsafe { core::slice::from_raw_parts_mut(storage.as_mut_ptr() as *mut u8, total_bytes) };

    if zc_ab_pre_hetero_enabled() {
        // QS5 hetero drain: feed the precompute through the shared two-pool
        // chunk queue instead of a main-pool-only `par_chunks_mut`. The
        // E-side of this queue is a broadcast on the SAME helper pool that is
        // draining the deferred lincheck stripe, and rayon broadcasts on one
        // pool run in submission order — so the four E-workers pick up
        // precompute jobs at the exact moment the stripe releases them,
        // with no new synchronization. Rationale (measured 2026-08-05, this
        // hardware class): post-byte16 the commit window is BOUND by this
        // precompute arm (arm 58.2 ms ≈ window 58.3 ms; the GPU graph
        // finishes at 41–53 ms with 0.00 ms host wait), while the stripe
        // releases the E-cluster at ~52 ms (E3) or ~40 ms (E4). Every
        // E-core-second spent on the tail of THIS queue therefore shortens
        // the window one-for-one — the inverse of the regime in which the
        // hetero AB drain was measured dead (GPU-bound join, E-traffic
        // raising the dominant arm). Chunk-claim order is nondeterministic
        // but each chunk writes only its own disjoint 1 KiB, so the output
        // bytes are identical to the incumbent path.
        let n_chunks = total_bytes / OUTER_BYTES;
        let ranked_fast_policy = ranked_ab_pre_fast_policy_hoist_shape(
            m,
            k_skip,
            n_chunks,
            padding.k_log,
            padding.useful_bits_per_block,
            rayon::current_num_threads(),
            crate::epool::helper_pool().map_or(0, rayon::ThreadPool::current_num_threads),
            kernels::static_b_context_is_prepared(static_b_context),
            zc_ab_pre_fast_policy_hoist_enabled(),
        ) && kernels::fast_shift_reduce_enabled();
        if ranked_fast_policy {
            // Resolve the process-wide direct-store switch once before the
            // queue starts. The ranked/default arm is compiled without the
            // per-chunk OnceLock probe, temporary buffer, or per-row bounce
            // branches; the kill-switch arm retains the exact old behavior.
            if nt && ab_pre_nt_direct_enabled() {
                precompute_ab_hetero::<{ kernels::AB_FAST_POLICY_FORCE_FAST }, true>(
                    a_packed,
                    b_packed,
                    inv_table,
                    within_outer_mask,
                    &b_med_counts,
                    blake3_static_layout,
                    static_b_context,
                    nt,
                    compact,
                    n_chunks,
                    out_bytes,
                );
            } else {
                precompute_ab_hetero::<{ kernels::AB_FAST_POLICY_FORCE_FAST }, false>(
                    a_packed,
                    b_packed,
                    inv_table,
                    within_outer_mask,
                    &b_med_counts,
                    blake3_static_layout,
                    static_b_context,
                    nt,
                    compact,
                    n_chunks,
                    out_bytes,
                );
            }
        } else {
            precompute_ab_hetero::<{ kernels::AB_FAST_POLICY_PROCESS }, false>(
                a_packed,
                b_packed,
                inv_table,
                within_outer_mask,
                &b_med_counts,
                blake3_static_layout,
                static_b_context,
                nt,
                compact,
                n_chunks,
                out_bytes,
            );
        }
        return Round1AbInner { storage };
    }

    out_bytes
        .par_chunks_mut(OUTER_BYTES)
        .enumerate()
        .for_each_init(
            || ([F8::ZERO; ELL], [F8::ZERO; ELL]),
            |(a_col, b_col), (x_outer, out_outer)| {
                precompute_ab_one_chunk::<{ kernels::AB_FAST_POLICY_PROCESS }, false>(
                    a_packed,
                    b_packed,
                    inv_table,
                    within_outer_mask,
                    &b_med_counts,
                    blake3_static_layout,
                    static_b_context,
                    nt,
                    compact,
                    x_outer,
                    out_outer,
                    a_col,
                    b_col,
                );
            },
        );

    Round1AbInner { storage }
}

/// Use a deeper queue for the sequential block-cyclic scheduler so the ranked
/// shape exposes roughly sixty-four scheduling waves on the ten-thread worker.
#[inline]
fn ab_pre_chunks_per_job(n_chunks: usize) -> usize {
    n_chunks.div_ceil(640).max(1)
}

/// Ranked-shape selector for resolving the process-wide Horner policy once
/// before the AB queue starts. Every other shape retains the incumbent
/// per-row policy lookup so this codegen specialization cannot perturb
/// library callers or differently provisioned workers.
#[allow(clippy::too_many_arguments)]
fn ranked_ab_pre_fast_policy_hoist_shape(
    m: usize,
    k_skip: usize,
    n_chunks: usize,
    k_log: usize,
    useful_bits_per_block: usize,
    main_threads: usize,
    helper_threads: usize,
    prepared_static_b: bool,
    enabled: bool,
) -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64"))
        && enabled
        && m == 32
        && k_skip == K_SKIP
        && n_chunks == (1 << 19)
        && k_log == 14
        && useful_bits_per_block == 15_409
        && main_threads == 10
        && helper_threads == 4
        && prepared_static_b
}

/// Exact same-binary rollback for the ranked AB policy specialization.
pub const ENV_NO_ZC_AB_PRE_FAST_POLICY_HOIST: &str = "FLOCK_NO_ZC_AB_PRE_FAST_POLICY_HOIST";

fn zc_ab_pre_fast_policy_hoist_enabled() -> bool {
    std::env::var_os(ENV_NO_ZC_AB_PRE_FAST_POLICY_HOIST).as_deref()
        != Some(std::ffi::OsStr::new("1"))
}

/// Compile-time default for the QS5 hetero AB-precompute drain (the ranked
/// decision must be a constant; ranked workers run with a cleared
/// environment). A/B-CONTROL: set exactly `FLOCK_NO_ZC_AB_PRE_HETERO=1` for
/// the incumbent main-pool-only drain, same binary, byte-identical output.
pub const ZC_AB_PRE_HETERO_DEFAULT: bool = true;
pub const ENV_NO_ZC_AB_PRE_HETERO: &str = "FLOCK_NO_ZC_AB_PRE_HETERO";

fn zc_ab_pre_hetero_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        ZC_AB_PRE_HETERO_DEFAULT
            && std::env::var_os(ENV_NO_ZC_AB_PRE_HETERO).as_deref()
                != Some(std::ffi::OsStr::new("1"))
    })
}

/// Drain the ranked AB precompute through both core clusters. The const policy
/// lets the ranked arm fold away the process-wide Horner flag inside the hot
/// row dispatcher. `FORCE_DIRECT` additionally folds away the direct-store
/// OnceLock lookup and stack-bounce alternative after the caller has checked
/// both `nt` and the process-wide direct-store kill switch.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn precompute_ab_hetero<const FAST_POLICY: u8, const FORCE_DIRECT: bool>(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    within_outer_mask: usize,
    b_med_counts: &[u8],
    blake3_static_layout: bool,
    static_b_context: Option<kernels::StaticBContext>,
    nt: bool,
    compact: bool,
    n_chunks: usize,
    out_bytes: &mut [u8],
) {
    // This is a private specialization contract, but keep it checked in
    // release builds so a future caller cannot accidentally select
    // non-temporal direct stores while requesting the cached-store mode.
    if FORCE_DIRECT {
        assert!(
            nt,
            "FORCE_DIRECT requires the non-temporal AB-precompute mode"
        );
    }

    const OUTER_BYTES: usize = (1 << N_MEDIUM) * 64;
    // Process each queue-owned slab monotonically. This removes permutation
    // generation and maximizes spatial locality; queue-level heterogeneity still
    // distributes independent slabs dynamically.
    let chunks_per_job = ab_pre_chunks_per_job(n_chunks);
    let n_jobs = n_chunks.div_ceil(chunks_per_job);
    let out_base = crate::epool::SyncPtr(out_bytes.as_mut_ptr());
    crate::epool::run_hetero_chunks_stateful(
        n_jobs,
        || ([F8::ZERO; ELL], [F8::ZERO; ELL]),
        |(a_col, b_col), job| {
            let chunk_start = job * chunks_per_job;
            let chunk_end = (chunk_start + chunks_per_job).min(n_chunks);
            let slab_len = chunk_end - chunk_start;
            for offset in 0..slab_len {
                let x_outer = chunk_start + offset;
                // SAFETY: offset is within this queue job's disjoint output slab.
                let out_outer = unsafe {
                    core::slice::from_raw_parts_mut(
                        out_base.ptr().add(x_outer * OUTER_BYTES),
                        OUTER_BYTES,
                    )
                };
                precompute_ab_one_chunk::<FAST_POLICY, FORCE_DIRECT>(
                    a_packed,
                    b_packed,
                    inv_table,
                    within_outer_mask,
                    b_med_counts,
                    blake3_static_layout,
                    static_b_context,
                    nt,
                    compact,
                    x_outer,
                    out_outer,
                    a_col,
                    b_col,
                );
            }
        },
    );
}

/// One `x_outer`'s worth of the challenge-independent AB transform — the
/// exact loop body both precompute drains share, factored out so the QS5
/// hetero queue and the incumbent `par_chunks_mut` cannot diverge.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn precompute_ab_one_chunk<const FAST_POLICY: u8, const FORCE_DIRECT: bool>(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    within_outer_mask: usize,
    b_med_counts: &[u8],
    blake3_static_layout: bool,
    static_b_context: Option<kernels::StaticBContext>,
    nt: bool,
    compact: bool,
    x_outer: usize,
    out_outer: &mut [u8],
    a_col: &mut [F8; ELL],
    b_col: &mut [F8; ELL],
) {
    const OUTER_BYTES: usize = (1 << N_MEDIUM) * 64;
    let within_hash_outer = x_outer & within_outer_mask;
    let n_b_med = b_med_counts[within_hash_outer] as usize;
    let chunk_byte_base = x_outer * OUTER_BYTES;

    // NT arm: the kernel drains each 64-byte block to the big buffer with
    // `stnp` (write-once lines, consumer runs after the commit root). By
    // default the kernel stores non-temporally straight from its accumulator
    // registers; `FLOCK_NO_ZC_AB_PRE_NT_DIRECT=1` restores the incumbent
    // stack bounce (kernel `vst1q` into `tmp`, then an `ldp`/`stnp` copy —
    // six extra memory ops and a store-to-load forward per row). Control arm
    // (`FLOCK_NO_ZC_AB_PRE_NT=1`) is the incumbent cached kernel write.
    // All three flavors are byte-identical.
    let direct = if FORCE_DIRECT {
        true
    } else {
        nt && ab_pre_nt_direct_enabled()
    };
    let mut tmp = [0u8; 64];
    for b_med in 0..n_b_med {
        let dst: &mut [u8; 64] = if nt && !direct {
            &mut tmp
        } else {
            (&mut out_outer[b_med * 64..(b_med + 1) * 64])
                .try_into()
                .expect("one transformed b_med block")
        };
        shift_reduce_inner_ab::<FAST_POLICY>(
            a_packed,
            b_packed,
            inv_table,
            chunk_byte_base,
            b_med,
            dst,
            a_col,
            b_col,
            !blake3_static_layout || (within_hash_outer == 0 && b_med < 2),
            !blake3_static_layout || (within_hash_outer == 1 && b_med + 1 == n_b_med),
            if blake3_static_layout && within_hash_outer == 0 && b_med == 2 {
                0x03
            } else if blake3_static_layout && within_hash_outer == 1 && b_med + 2 == n_b_med {
                0xf0
            } else {
                0
            },
            if blake3_static_layout {
                within_hash_outer
            } else {
                usize::MAX
            },
            static_b_context,
            direct,
        );
        if nt && !direct {
            // SAFETY: `b_med < n_b_med ≤ OUTER_BYTES / 64`, so the
            // 64 destination bytes are in-bounds of `out_outer`.
            unsafe { store_nt_64(tmp.as_ptr(), out_outer.as_mut_ptr().add(b_med * 64)) };
        }
    }
    // QS3 compacted store. The tail rows `[n_b_med, 16)` of this
    // chunk are dead: every `ab_inner` consumer iterates exactly
    // `0..n_b_med` — the AB completion (`accumulate_convert_ab`),
    // the fused/split `accumulate_convert_with_s_hat_v` AB half,
    // and the Fold4 AB arm all bound their per-`b_med` reads by the
    // same `n_b_med`; C reads the packed witness, not `ab_inner`;
    // and round two overwrites the whole donated buffer
    // (`into_scratch_bytes` → compact `deltas`) before reading it.
    // So zeroing the tail is pure write traffic. At the ranked
    // BLAKE3 shape (k_log=14, useful=15_409) window 1 has
    // `n_b_med = 15`, i.e. one dead 64-byte row per window-1 chunk:
    // 2^18 chunks × 64 B = 16 MiB, ~3% of the 512 MiB precompute
    // write surface. The AB precompute runs concurrently with the
    // streamed GPU commit, which saturates the same memory system,
    // so dropping these stores pays as CONTENTION RELIEF on the
    // commit arm rather than on the precompute's own wall. The kill
    // switch restores the incumbent zero-fill for a same-binary A/B;
    // both leave the live region byte-identical, so the proof is
    // unchanged.
    if !compact {
        if nt {
            let tail = &mut out_outer[n_b_med * 64..];
            debug_assert_eq!(tail.len() % 64, 0);
            let zero = [0u8; 64];
            for i in 0..tail.len() / 64 {
                // SAFETY: chunk `i` is 64 in-bounds bytes of `tail`.
                unsafe { store_nt_64(zero.as_ptr(), tail.as_mut_ptr().add(i * 64)) };
            }
        } else {
            out_outer[n_b_med * 64..].fill(0);
        }
    }
}

// ---------------------------------------------------------------------------
// Shift_reduce inner kernel (AB only — extract_c handles C separately).
//
// For one medium-position b_med and the 8 small-positions K ∈ 0..8:
//   1. Look up NTT-extended A,B at chunk `chunk_byte_base + (b_med*8 + K)*8`.
//   2. y_K[lane] = ntt_a[lane] · ntt_b[lane]  (in F_8).
//   3. acc[lane] ^= (y_K[lane] as u16) << K   (no reduction yet).
// At the end, reduce each acc[lane] back to a u8 in F_8.
//
// Output `out[lane]` is the F_8 representative of Σ_K x^K · y_K[lane] mod p.
// ---------------------------------------------------------------------------

fn shift_reduce_inner_ab<const FAST_POLICY: u8>(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; 64],
    a_col: &mut [F8],
    b_col: &mut [F8],
    check_all_ones: bool,
    check_single_k0: bool,
    const_one_mask: u8,
    bstatic_w: usize,
    static_b_context: Option<kernels::StaticBContext>,
    nt_store: bool,
) {
    kernels::shift_reduce_inner_ab::<FAST_POLICY>(
        a_packed,
        b_packed,
        inv_table,
        chunk_byte_base,
        b_med,
        out,
        a_col,
        b_col,
        check_all_ones,
        check_single_k0,
        const_one_mask,
        bstatic_w,
        static_b_context,
        nt_store,
    );
}

// ---------------------------------------------------------------------------
// Main optimized round-1 prover message.
// ---------------------------------------------------------------------------

/// Compute the round-1 prover message via the full shift_reduce + extract_c
/// optimization, in scalar Rust.
///
/// Output relative to [`super::round1_naive`]:
///   `C_s · (res_AB[i] + res_C_lifted[i]) = naive_p_ab[i] + naive_p_c[i]`
///
/// Preconditions:
/// - `k_skip == K_SKIP` (= 6)
/// - `m >= k_skip + N_INNER` (= 13)
/// - `r.len() == m`. `r[k_skip..k_skip+7]` must hold the protocol-fixed small
///   + medium constants (see [`small_challenges_ghash`] /
///   [`medium_challenges_ghash`]) for the naive cross-check to line up. Only
///   `r[k_skip+7..m]` is used internally.
/// - `inv_table.k == k_skip`.
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

// Per-worker scratch + local accumulator. ~6 KB total, stack-allocated.
struct WorkerState {
    partial_ab: [F128; ELL],
    partial_c: [F128; ELL],
    chunk_ab_bytes: [[u8; 64]; 1 << N_MEDIUM],
    chunk_c_bytes: [[u8; 64]; 1 << N_MEDIUM],
    a_col: [F8; ELL],
    b_col: [F8; ELL],
    local_res_ab: [F128; ELL],
    local_res_c_s: [F128; ELL],
}

impl WorkerState {
    fn new() -> Self {
        Self {
            partial_ab: [F128::ZERO; ELL],
            partial_c: [F128::ZERO; ELL],
            chunk_ab_bytes: [[0u8; 64]; 1 << N_MEDIUM],
            chunk_c_bytes: [[0u8; 64]; 1 << N_MEDIUM],
            a_col: [F8::ZERO; ELL],
            b_col: [F8::ZERO; ELL],
            local_res_ab: [F128::ZERO; ELL],
            local_res_c_s: [F128::ZERO; ELL],
        }
    }
}

/// Process one outer x_hi value: middle-loop over x_outer_lo (reset `partial_ab/c`,
/// run shift_reduce_inner + bit_transpose + convert+apply), then outer fold by
/// `eq_hi_val` into `state.local_res_ab/c_s`.
///
/// Called per-x_hi by both the parallel public function and the serial test oracle.
///
/// `within_outer_mask` and `b_med_counts` together encode the per-block padding
/// pattern (see [`PaddingSpec`]). For each x_outer, `within_hash_outer =
/// x_outer & within_outer_mask` is the position of its 8192-bit window within
/// a block, and `b_med_counts[within_hash_outer]` tells the kernel how many
/// of the 16 b_med 512-bit sub-windows are worth processing — the rest fall
/// entirely in zero padding and are skipped. Pass `within_outer_mask = 0` and
/// `b_med_counts = &[1 << N_MEDIUM]` to disable skipping.
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
    state.partial_ab.iter_mut().for_each(|p| *p = F128::ZERO);
    state.partial_c.iter_mut().for_each(|p| *p = F128::ZERO);

    let n_lo = n_lo_and_inner - N_INNER;

    for x_outer_lo in 0..big_lo_size {
        let x_outer = x_outer_lo | (x_hi << n_lo);
        let within_hash_outer = x_outer & within_outer_mask;
        let n_b_med = b_med_counts[within_hash_outer] as usize;
        if n_b_med == 0 {
            continue;
        }

        let chunk_byte_base = ((x_outer_lo << N_INNER) | (x_hi << n_lo_and_inner)) * N_CHUNKS;

        let eq_lo_val = eq_lo_scaled[x_outer_lo];

        // Two paths: when n_b_med == 16 (the full case — true for every
        // x_outer_lo on the dense path, and for most of them on the padded
        // path too), use compile-time loop bounds so the SIMD XOR chain
        // unrolls. The slow path handles the rare boundary window where
        // n_b_med < 16.
        if n_b_med == (1 << N_MEDIUM) {
            for b_med in 0..(1 << N_MEDIUM) {
                shift_reduce_inner_ab::<{ kernels::AB_FAST_POLICY_PROCESS }>(
                    a_packed,
                    b_packed,
                    inv_table,
                    chunk_byte_base,
                    b_med,
                    &mut state.chunk_ab_bytes[b_med],
                    &mut state.a_col,
                    &mut state.b_col,
                    true,
                    true,
                    0,
                    usize::MAX,
                    None,
                    false,
                );
                let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;
                let c_in: &[u8; 64] = (&c_packed[byte_base_b..byte_base_b + 64])
                    .try_into()
                    .expect("64 c-bytes per medium position");
                bit_transpose_64bytes(c_in, &mut state.chunk_c_bytes[b_med]);
            }

            kernels::accumulate_convert(
                &state.chunk_ab_bytes,
                &state.chunk_c_bytes,
                1 << N_MEDIUM,
                convert,
                eq_lo_val,
                &mut state.partial_ab,
                &mut state.partial_c,
            );
        } else {
            // Partial path: n_b_med ∈ (0, 1 << N_MEDIUM). At most one
            // within_hash_outer value per [`PaddingSpec`] lands here (the
            // window straddling the useful/padding boundary), so the tighter
            // loop wins despite losing the SIMD chain unroll.
            for b_med in 0..n_b_med {
                shift_reduce_inner_ab::<{ kernels::AB_FAST_POLICY_PROCESS }>(
                    a_packed,
                    b_packed,
                    inv_table,
                    chunk_byte_base,
                    b_med,
                    &mut state.chunk_ab_bytes[b_med],
                    &mut state.a_col,
                    &mut state.b_col,
                    true,
                    true,
                    0,
                    usize::MAX,
                    None,
                    false,
                );
                let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;
                let c_in: &[u8; 64] = (&c_packed[byte_base_b..byte_base_b + 64])
                    .try_into()
                    .expect("64 c-bytes per medium position");
                bit_transpose_64bytes(c_in, &mut state.chunk_c_bytes[b_med]);
            }

            kernels::accumulate_convert(
                &state.chunk_ab_bytes,
                &state.chunk_c_bytes,
                n_b_med,
                convert,
                eq_lo_val,
                &mut state.partial_ab,
                &mut state.partial_c,
            );
        }
    }

    // Outer fold by eq_hi.
    for lane in 0..ELL {
        state.local_res_ab[lane] += eq_hi_val * state.partial_ab[lane];
        state.local_res_c_s[lane] += eq_hi_val * state.partial_c[lane];
    }
}

// ---------------------------------------------------------------------------
// Fusion: eight-bank C accumulator that produces s_hat_v_c AND the four-bank
// (quad) sufficient statistic alongside round 1.
//
// The only structural change from `process_one_x_hi` is in the C-side inner
// loop: instead of one `cf_c` accumulator collapsing all 3 small bits, all
// three are kept as routing dims — `K = b_3[0] + 2 b_3[1] + 4 b_3[2]`, i.e.
// eight single-bit banks instead of the previous even/odd (`0x55`/`0xAA`)
// split on `b_3[0]` alone. `b_3[0]` is ring-switch's packed-prefix bit `b_7`;
// `b_3[1]`, `b_3[2]` are the two PCS suffix coordinates the C claim retains.
//
// The banks are accumulated **α-free** (see `kernels::accumulate_c_banks`):
// bank `K`'s raw value over the 16 `b_med` is just the u16 mask of bit `K`,
// so the C side spends no convert-table gather at all. Re-applying `α^K` and
// summing over the parity classes of `K` reconstructs the previous two banks
// exactly — the same F_2-linearity of φ_8 the `0x55`/`0xAA` split relied on,
// one level finer — so the wire `res_c_s` and `s_hat_v_c` are unchanged.
// ---------------------------------------------------------------------------

/// Number of α-free single-bit-`K` C banks: `K = b_3[0] + 2 b_3[1] + 4 b_3[2]`.
const N_C_BANKS: usize = 8;

/// Per-worker scratch + local accumulator for the eight-bank C variant.
/// Identical to [`WorkerState`] except `partial_c` and `local_res_c_s` are
/// split per `K`.
struct WorkerStateWithSHatV {
    partial_ab: [F128; ELL],
    partial_c: [[F128; ELL]; N_C_BANKS],
    chunk_ab_bytes: [[u8; 64]; 1 << N_MEDIUM],
    a_col: [F8; ELL],
    b_col: [F8; ELL],
    local_res_ab: [F128; ELL],
    local_res_c_s: [[F128; ELL]; N_C_BANKS],
}

/// Allocate the widened Fold4 banks directly on the heap. Constructing the
/// complete 32 KiB array as a `Box::new` argument can first materialize it on
/// an epool helper's smaller stack; the boxed-slice conversion allocates and
/// initializes in heap storage before the zero-copy fixed-length conversion.
fn zero_c_fold4_banks() -> Box<[[F128; ELL]; N_C_FOLD4_BANKS]> {
    let banks = vec![[F128::ZERO; ELL]; N_C_FOLD4_BANKS].into_boxed_slice();
    match banks.try_into() {
        Ok(banks) => banks,
        Err(_) => unreachable!("fixed direct-fold4 bank count"),
    }
}

/// Allocate one retained-medium group directly on the heap.  Each C worker
/// owns only 8 KiB of accumulator state, leaving room in a performance core's
/// L1 data cache for masks, table lines, and the current witness rows.
fn zero_c_fold4_q_banks() -> Box<[[F128; ELL]; N_C_BANKS]> {
    let banks = vec![[F128::ZERO; ELL]; N_C_BANKS].into_boxed_slice();
    match banks.try_into() {
        Ok(banks) => banks,
        Err(_) => unreachable!("fixed direct-fold4 q-bank count"),
    }
}

#[inline]
fn precomputed_ab_rows(ab_inner: &[u8], byte_base: usize) -> &[[u8; 64]; 1 << N_MEDIUM] {
    let bytes = &ab_inner[byte_base..byte_base + (1 << N_MEDIUM) * 64];
    // SAFETY: both source and target have alignment one and the slice above
    // proves the complete 1024-byte extent. `[[u8; 64]; 16]` has no padding,
    // so this changes only the borrow's shape, not its byte representation.
    unsafe { &*bytes.as_ptr().cast::<[[u8; 64]; 1 << N_MEDIUM]>() }
}

impl WorkerStateWithSHatV {
    fn new() -> Self {
        Self {
            partial_ab: [F128::ZERO; ELL],
            partial_c: [[F128::ZERO; ELL]; N_C_BANKS],
            chunk_ab_bytes: [[0u8; 64]; 1 << N_MEDIUM],
            a_col: [F8::ZERO; ELL],
            b_col: [F8::ZERO; ELL],
            local_res_ab: [F128::ZERO; ELL],
            local_res_c_s: [[F128::ZERO; ELL]; N_C_BANKS],
        }
    }
}

/// Eight-bank C variant of [`process_one_x_hi`]. AB-side and witness traffic
/// unchanged; the only modification is the C-side inner loop now maintains
/// eight α-free single-bit-`K` banks instead of one convert-table sum.
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
    mask_tables: &[F128],
    state: &mut WorkerStateWithSHatV,
) {
    state.partial_ab.iter_mut().for_each(|p| *p = F128::ZERO);
    state
        .partial_c
        .iter_mut()
        .for_each(|bank| bank.iter_mut().for_each(|p| *p = F128::ZERO));

    let n_lo = n_lo_and_inner - N_INNER;

    for x_outer_lo in 0..big_lo_size {
        let x_outer = x_outer_lo | (x_hi << n_lo);
        let within_hash_outer = x_outer & within_outer_mask;
        let n_b_med = b_med_counts[within_hash_outer] as usize;
        if n_b_med == 0 {
            continue;
        }

        let chunk_byte_base = ((x_outer_lo << N_INNER) | (x_hi << n_lo_and_inner)) * N_CHUNKS;
        let eq_lo_val = eq_lo_scaled[x_outer_lo];
        let c_tables =
            &mask_tables[x_outer_lo * C_MASK_TABLE_STRIDE..(x_outer_lo + 1) * C_MASK_TABLE_STRIDE];
        // The C side reads the packed witness directly: the fused mask kernel
        // subsumes the per-`b_med` `bit_transpose_64bytes` this path used to run.
        let c_rows: &[u8; (1 << N_MEDIUM) * 64] = (&c_packed
            [chunk_byte_base..chunk_byte_base + (1 << N_MEDIUM) * 64])
            .try_into()
            .expect("sixteen 64-byte c rows per x_outer_lo");

        if n_b_med == (1 << N_MEDIUM) {
            for b_med in 0..(1 << N_MEDIUM) {
                shift_reduce_inner_ab::<{ kernels::AB_FAST_POLICY_PROCESS }>(
                    a_packed,
                    b_packed,
                    inv_table,
                    chunk_byte_base,
                    b_med,
                    &mut state.chunk_ab_bytes[b_med],
                    &mut state.a_col,
                    &mut state.b_col,
                    true,
                    true,
                    0,
                    usize::MAX,
                    None,
                    false,
                );
            }

            kernels::accumulate_convert_with_s_hat_v(
                &state.chunk_ab_bytes,
                c_rows,
                1 << N_MEDIUM,
                convert,
                eq_lo_val,
                c_tables,
                &mut state.partial_ab,
                &mut state.partial_c,
            );
        } else {
            for b_med in 0..n_b_med {
                shift_reduce_inner_ab::<{ kernels::AB_FAST_POLICY_PROCESS }>(
                    a_packed,
                    b_packed,
                    inv_table,
                    chunk_byte_base,
                    b_med,
                    &mut state.chunk_ab_bytes[b_med],
                    &mut state.a_col,
                    &mut state.b_col,
                    true,
                    true,
                    0,
                    usize::MAX,
                    None,
                    false,
                );
            }

            kernels::accumulate_convert_with_s_hat_v(
                &state.chunk_ab_bytes,
                c_rows,
                n_b_med,
                convert,
                eq_lo_val,
                c_tables,
                &mut state.partial_ab,
                &mut state.partial_c,
            );
        }
    }

    // Outer fold by eq_hi (per bank).
    for lane in 0..ELL {
        state.local_res_ab[lane] += eq_hi_val * state.partial_ab[lane];
    }
    for (res, partial) in state.local_res_c_s.iter_mut().zip(&state.partial_c) {
        for lane in 0..ELL {
            res[lane] += eq_hi_val * partial[lane];
        }
    }
}

/// Challenge-weighted half of [`process_one_x_hi_with_s_hat_v`] when the AB
/// shift-reduce blocks were produced earlier. C remains live and is handled
/// exactly as in the fused path so wire output and `s_hat_v_c` stay identical.
#[inline]
#[allow(clippy::too_many_arguments)]
fn process_one_x_hi_with_precomputed_ab(
    x_hi: usize,
    big_lo_size: usize,
    n_lo_and_inner: usize,
    within_outer_mask: usize,
    b_med_counts: &[u8],
    ab_inner: &[u8],
    c_packed: &[u8],
    eq_lo_scaled: &[F128],
    eq_hi_val: F128,
    convert: &[F128],
    mask_tables: &[F128],
    split_ab_c: bool,
    direct_ab_rows: bool,
    c_drain4: bool,
    state: &mut WorkerStateWithSHatV,
) {
    state.partial_ab.iter_mut().for_each(|p| *p = F128::ZERO);
    state
        .partial_c
        .iter_mut()
        .for_each(|bank| bank.iter_mut().for_each(|p| *p = F128::ZERO));

    let n_lo = n_lo_and_inner - N_INNER;

    if split_ab_c {
        // AB's 64 KiB convert table is exactly the cache-residency boundary on
        // the efficiency cores.  The old per-window AB→C alternation touched
        // a disjoint 8 KiB C table between every two AB uses.  Complete the AB
        // population first, then drain C in a second linear pass; the witness
        // bytes read are unchanged and characteristic-two accumulation makes
        // the reordering bit-exact.
        for x_outer_lo in 0..big_lo_size {
            let x_outer = x_outer_lo | (x_hi << n_lo);
            let within_hash_outer = x_outer & within_outer_mask;
            let n_b_med = b_med_counts[within_hash_outer] as usize;
            if n_b_med == 0 {
                continue;
            }

            let chunk_byte_base = ((x_outer_lo << N_INNER) | (x_hi << n_lo_and_inner)) * N_CHUNKS;
            let chunk_ab_bytes = if direct_ab_rows {
                precomputed_ab_rows(ab_inner, chunk_byte_base)
            } else {
                for b_med in 0..n_b_med {
                    let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;
                    state.chunk_ab_bytes[b_med]
                        .copy_from_slice(&ab_inner[byte_base_b..byte_base_b + 64]);
                }
                &state.chunk_ab_bytes
            };
            kernels::accumulate_convert_ab(
                chunk_ab_bytes,
                n_b_med,
                convert,
                eq_lo_scaled[x_outer_lo],
                &mut state.partial_ab,
            );
        }

        for x_outer_lo in 0..big_lo_size {
            let x_outer = x_outer_lo | (x_hi << n_lo);
            let within_hash_outer = x_outer & within_outer_mask;
            let n_b_med = b_med_counts[within_hash_outer] as usize;
            if n_b_med == 0 {
                continue;
            }

            let chunk_byte_base = ((x_outer_lo << N_INNER) | (x_hi << n_lo_and_inner)) * N_CHUNKS;
            let c_rows: &[u8; (1 << N_MEDIUM) * 64] = (&c_packed
                [chunk_byte_base..chunk_byte_base + (1 << N_MEDIUM) * 64])
                .try_into()
                .expect("sixteen 64-byte c rows per x_outer_lo");
            let c_tables = &mask_tables
                [x_outer_lo * C_MASK_TABLE_STRIDE..(x_outer_lo + 1) * C_MASK_TABLE_STRIDE];
            kernels::accumulate_c_banks_with_policy(
                c_rows,
                n_b_med,
                c_tables,
                &mut state.partial_c,
                c_drain4,
            );
        }
    } else {
        for x_outer_lo in 0..big_lo_size {
            let x_outer = x_outer_lo | (x_hi << n_lo);
            let within_hash_outer = x_outer & within_outer_mask;
            let n_b_med = b_med_counts[within_hash_outer] as usize;
            if n_b_med == 0 {
                continue;
            }

            let chunk_byte_base = ((x_outer_lo << N_INNER) | (x_hi << n_lo_and_inner)) * N_CHUNKS;
            let eq_lo_val = eq_lo_scaled[x_outer_lo];
            let c_tables = &mask_tables
                [x_outer_lo * C_MASK_TABLE_STRIDE..(x_outer_lo + 1) * C_MASK_TABLE_STRIDE];
            // The C side reads the packed witness directly: the fused mask kernel
            // subsumes the per-`b_med` `bit_transpose_64bytes` this path used to run.
            let c_rows: &[u8; (1 << N_MEDIUM) * 64] = (&c_packed
                [chunk_byte_base..chunk_byte_base + (1 << N_MEDIUM) * 64])
                .try_into()
                .expect("sixteen 64-byte c rows per x_outer_lo");

            let chunk_ab_bytes = if direct_ab_rows {
                precomputed_ab_rows(ab_inner, chunk_byte_base)
            } else {
                for b_med in 0..n_b_med {
                    let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;
                    state.chunk_ab_bytes[b_med]
                        .copy_from_slice(&ab_inner[byte_base_b..byte_base_b + 64]);
                }
                &state.chunk_ab_bytes
            };

            kernels::accumulate_convert_with_s_hat_v(
                chunk_ab_bytes,
                c_rows,
                n_b_med,
                convert,
                eq_lo_val,
                c_tables,
                &mut state.partial_ab,
                &mut state.partial_c,
            );
        }
    }

    for lane in 0..ELL {
        state.local_res_ab[lane] += eq_hi_val * state.partial_ab[lane];
    }
    for (res, partial) in state.local_res_c_s.iter_mut().zip(&state.partial_c) {
        for lane in 0..ELL {
            res[lane] += eq_hi_val * partial[lane];
        }
    }
}

/// AB job for the experimental direct-fold4 capture.  It remains one pass per
/// `x_hi`; splitting C into four q-local jobs must not multiply AB traffic.
#[inline]
#[allow(clippy::too_many_arguments)]
fn process_one_x_hi_with_precomputed_ab_fold4_ab(
    x_hi: usize,
    big_lo_size: usize,
    n_lo_and_inner: usize,
    within_outer_mask: usize,
    b_med_counts: &[u8],
    ab_inner: &[u8],
    eq_lo_scaled: &[F128],
    eq_hi_val: F128,
    convert: &[F128],
    timing_cpu_ns: Option<&[std::sync::atomic::AtomicU64; 3]>,
    partial_ab: &mut [F128; ELL],
) {
    partial_ab.fill(F128::ZERO);

    let n_lo = n_lo_and_inner - N_INNER;

    // Ranked AB policy: its 64 KiB table stays resident for the complete band.
    let t_ab = timing_cpu_ns.map(|_| std::time::Instant::now());
    for x_outer_lo in 0..big_lo_size {
        let x_outer = x_outer_lo | (x_hi << n_lo);
        let n_b_med = b_med_counts[x_outer & within_outer_mask] as usize;
        if n_b_med == 0 {
            continue;
        }
        let chunk_byte_base = ((x_outer_lo << N_INNER) | (x_hi << n_lo_and_inner)) * N_CHUNKS;
        kernels::accumulate_convert_ab(
            precomputed_ab_rows(ab_inner, chunk_byte_base),
            n_b_med,
            convert,
            eq_lo_scaled[x_outer_lo],
            partial_ab,
        );
    }
    if let (Some(totals), Some(start)) = (timing_cpu_ns, t_ab) {
        totals[0].fetch_add(
            start.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    let t_high = timing_cpu_ns.map(|_| std::time::Instant::now());
    for value in partial_ab {
        *value *= eq_hi_val;
    }
    if let (Some(totals), Some(start)) = (timing_cpu_ns, t_high) {
        totals[2].fetch_add(
            start.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
}

// ---------------------------------------------------------------------------
// eq_lo tensor fold for the ranked AB completion drain.
//
// The drain above pays one GHASH multiply per lane per chunk:
// `partial_ab[lane] += converted_ab[lane] * eq_lo_scaled[x_lo]`, i.e.
// 2^n_hi * 2^n_lo * ELL = 33.5 M multiplies per prove at the ranked shape
// (n_hi = 7, n_lo = 12, ELL = 64). None of them is necessary.
//
// `eq_lo` is a tensor product over its challenge bits (`build_eq` gives bit i
// of the index to `r_lo[i]`), so splitting `x_lo = w·2^s + u` factors it
// exactly:
//
//     eq_lo_scaled[w·2^s + u] == eq_top_scaled[w] * eq_bot[u]
//     eq_top_scaled = build_eq(r_lo[s..]) · D^-1,  eq_bot = build_eq(r_lo[..s])
//
// F128 multiplication is F2-bilinear, so the `eq_top[w]` factor can be pushed
// through the whole gather chain and pre-multiplied into the convert table
// once per prove (`T_w[i] = convert[i] * eq_top_scaled[w]`, 2^(n_lo-s) tables
// of 64 KiB), and the `eq_bot[u]` factor can be deferred out of the chunk loop
// into 2^s per-worker banks:
//
//     bank[u][lane]    ^= gather(T_w, chunk)[lane]        (no multiply)
//     partial_ab[lane]  = XOR_u eq_bot[u] * bank[u][lane]  (once per band)
//
// which is the same sum of the same products, re-associated — bit-identical,
// not merely equal. The multiply count drops to
// 2^(n_lo-s)·|convert| + 2^n_hi·2^s·ELL ≈ 1.2 M per prove.
//
// `s` trades table footprint against bank footprint. The default keeps
// `AB_EQ_FOLD_TABLE_BITS` bits in the table index, so the ranked shape gets 32
// tables (2 MiB total, small enough to stay resident in the efficiency
// cluster's L2 as well as the performance cluster's) whose hot window is one
// 64 KiB table per 2^s consecutive chunks — exactly the incumbent convert
// table's residency — and 128 banks (128 KiB per worker, L2-resident on both
// core types). Both pools of `run_hetero_chunks` run the same `s`; nothing in
// the mechanism is core-type specific, and the E-core arm sees the same
// 64 KiB hot table it sees today. `FLOCK_ZC_AB_EQ_FOLD_S` overrides it for
// tuning sweeps.
// ---------------------------------------------------------------------------

/// Kill switch for the `eq_lo` tensor fold in the ranked AB completion:
/// exactly `FLOCK_NO_ZC_AB_EQ_FOLD=1` restores the incumbent per-chunk
/// multiply as a same-binary control. Output is bit-identical either way.
pub const ENV_NO_ZC_AB_EQ_FOLD: &str = "FLOCK_NO_ZC_AB_EQ_FOLD";

/// Tuning seam for the fold's bank/table split `s` (see the block comment):
/// an integer in `0..=16`, clamped to `n_lo`. Unset picks the default from
/// `n_lo`. Invalid values fail loudly, like the Fold4 split seam — silently
/// accepting a typo would make component profiles incomparable.
pub const ENV_ZC_AB_EQ_FOLD_S: &str = "FLOCK_ZC_AB_EQ_FOLD_S";

/// `x_lo` bits kept in the pre-scaled table index; the rest select the bank.
const AB_EQ_FOLD_TABLE_BITS: usize = 5;

/// Fold policy for [`round1_shift_reduce_ab_packed_padded_with_precomputed`].
/// Threaded as an argument rather than read from the environment inside the
/// drain so the differential test can run both arms in one process.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AbEqFold {
    /// Incumbent: one multiply per lane per chunk.
    Off,
    /// Banked fold with `s` low `x_lo` bits; `None` takes the default for the
    /// shape.
    On(Option<usize>),
}

impl AbEqFold {
    /// Resolved bank-index width, or `None` when the fold is off.
    fn bank_bits(self, n_lo: usize) -> Option<usize> {
        match self {
            AbEqFold::Off => None,
            AbEqFold::On(explicit) => Some(
                explicit
                    .unwrap_or_else(|| n_lo.saturating_sub(AB_EQ_FOLD_TABLE_BITS))
                    .min(n_lo),
            ),
        }
    }
}

fn ab_eq_fold_from_env() -> AbEqFold {
    static POLICY: OnceLock<AbEqFold> = OnceLock::new();
    *POLICY.get_or_init(|| {
        if std::env::var_os(ENV_NO_ZC_AB_EQ_FOLD).as_deref() == Some(std::ffi::OsStr::new("1")) {
            return AbEqFold::Off;
        }
        AbEqFold::On(std::env::var_os(ENV_ZC_AB_EQ_FOLD_S).map(|value| {
            value
                .to_str()
                .and_then(|text| text.parse::<usize>().ok())
                .filter(|s| *s <= 16)
                .expect("FLOCK_ZC_AB_EQ_FOLD_S must be an integer in 0..=16")
        }))
    })
}

std::thread_local! {
    /// Per-worker `u`-banks for the folded drain (`2^s * ELL` F128). Held per
    /// thread so no band allocates: both `run_hetero_chunks` pools reuse the
    /// same buffer across every band and every prove.
    static AB_EQ_FOLD_BANKS: std::cell::RefCell<Vec<F128>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Tensor factors of the scaled lo-eq weight for a `bank_bits = s` split:
/// `(eq_bot, eq_top_scaled)` with
/// `eq.lo[(w << s) | u] * D^-1 == eq_top_scaled[w] * eq_bot[u]`.
///
/// `SplitEqGhash` builds `lo` from `r_lo` LSB-first (`build_eq` gives index
/// bit `i` to `r_lo[i]`), so the low `s` index bits are exactly the
/// `r_lo[..s]` sub-product and the rest exactly the `r_lo[s..]` one.
fn ab_eq_fold_factors(r_lo: &[F128], bank_bits: usize) -> (Vec<F128>, Vec<F128>) {
    let eq_bot = build_eq(&r_lo[..bank_bits]);
    let eq_top_scaled = build_eq(&r_lo[bank_bits..])
        .into_iter()
        .map(|v| v * d_inv())
        .collect();
    (eq_bot, eq_top_scaled)
}

/// Pre-scaled convert tables for the fold: `T_w[i] = convert[i] *
/// eq_top_scaled[w]`, built once per prove and shared read-only by every
/// worker (mirrors [`build_c_mask_tables`]).
fn build_ab_eq_fold_tables(eq_top_scaled: &[F128], convert: &[F128]) -> Vec<F128> {
    use rayon::prelude::*;

    debug_assert_eq!(convert.len(), CONVERT_TABLE_SIZE);
    // Fully overwritten below (one write per slot before any read), so an
    // uninitialized scratch buffer is sound and skips a zeroing pass.
    let mut tables = crate::scratch::take_f128(eq_top_scaled.len() * CONVERT_TABLE_SIZE);
    tables
        .par_chunks_mut(CONVERT_TABLE_SIZE)
        .zip(eq_top_scaled.par_iter())
        .for_each(|(slot, scale)| {
            for (out, base) in slot.iter_mut().zip(convert.iter()) {
                *out = *base * *scale;
            }
        });
    tables
}

/// Folded twin of [`process_one_x_hi_with_precomputed_ab_fold4_ab`]: same
/// chunk sweep and same gathers, but the per-chunk `eq_lo` multiply is gone —
/// the table carries `eq_top[w]` and the band-end fold carries `eq_bot[u]`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn process_one_x_hi_with_precomputed_ab_fold4_ab_eq_folded(
    x_hi: usize,
    big_lo_size: usize,
    n_lo_and_inner: usize,
    within_outer_mask: usize,
    b_med_counts: &[u8],
    ab_inner: &[u8],
    eq_bot: &[F128],
    eq_hi_val: F128,
    tables: &[F128],
    bank_bits: usize,
    banks: &mut [F128],
    partial_ab: &mut [F128; ELL],
) {
    debug_assert_eq!(banks.len(), eq_bot.len() * ELL);
    debug_assert_eq!(eq_bot.len(), 1usize << bank_bits);
    banks.fill(F128::ZERO);

    let n_lo = n_lo_and_inner - N_INNER;
    let bank_mask = (1usize << bank_bits) - 1;

    for x_outer_lo in 0..big_lo_size {
        let x_outer = x_outer_lo | (x_hi << n_lo);
        let n_b_med = b_med_counts[x_outer & within_outer_mask] as usize;
        if n_b_med == 0 {
            continue;
        }
        let chunk_byte_base = ((x_outer_lo << N_INNER) | (x_hi << n_lo_and_inner)) * N_CHUNKS;
        let w = x_outer_lo >> bank_bits;
        let u = x_outer_lo & bank_mask;
        let bank: &mut [F128; ELL] = (&mut banks[u * ELL..(u + 1) * ELL])
            .try_into()
            .expect("one ELL-lane bank per low x_lo index");
        kernels::accumulate_convert_ab_nomul(
            precomputed_ab_rows(ab_inner, chunk_byte_base),
            n_b_med,
            &tables[w * CONVERT_TABLE_SIZE..(w + 1) * CONVERT_TABLE_SIZE],
            bank,
        );
    }

    partial_ab.fill(F128::ZERO);
    for (bank, eq_bot_val) in banks.chunks_exact(ELL).zip(eq_bot) {
        for (out, value) in partial_ab.iter_mut().zip(bank) {
            *out += *eq_bot_val * *value;
        }
    }
    for value in partial_ab {
        *value *= eq_hi_val;
    }
}

/// Challenge-weighted AB completion for the retained-coordinate ranked path,
/// without the independent C drain.  The legacy wire AB message is unchanged;
/// callers that can derive C from another honest representation use this to
/// avoid touching the 32-bank Fold4 C accumulator.
pub(crate) fn round1_shift_reduce_ab_packed_padded_with_precomputed(
    ab_inner: &Round1AbInner,
    m: usize,
    k_skip: usize,
    r: &[F128],
    padding: &PaddingSpec,
) -> Vec<F128> {
    round1_shift_reduce_ab_packed_padded_with_precomputed_with_fold(
        ab_inner,
        m,
        k_skip,
        r,
        padding,
        ab_eq_fold_from_env(),
    )
}

/// [`round1_shift_reduce_ab_packed_padded_with_precomputed`] with the `eq_lo`
/// fold policy supplied explicitly. Every arm must return the same bytes.
fn round1_shift_reduce_ab_packed_padded_with_precomputed_with_fold(
    ab_inner: &Round1AbInner,
    m: usize,
    k_skip: usize,
    r: &[F128],
    padding: &PaddingSpec,
    fold: AbEqFold,
) -> Vec<F128> {
    assert_eq!(k_skip, K_SKIP, "optimized variant is k_skip=6 only");
    let total_bytes = (1usize << m) / 8;
    assert_eq!(ab_inner.len_bytes(), total_bytes);
    assert_eq!(r.len(), m);

    let fold4_n_hi = fold4_n_hi_from_env();
    let eq = SplitEqGhash::with_n_hi(&r[k_skip + N_INNER..], fold4_n_hi);
    let big_lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    let n_lo_and_inner = eq.n_lo + N_INNER;
    let convert = convert_table();
    let (within_outer_mask, b_med_counts) = build_b_med_counts(padding);
    let ab_inner_bytes = ab_inner.as_bytes();

    let mut partials = vec![[F128::ZERO; ELL]; hi_size];
    let partials_base = crate::epool::SyncPtr(partials.as_mut_ptr());
    match fold.bank_bits(eq.n_lo) {
        None => {
            let eq_lo_scaled: Vec<F128> = eq.lo.iter().map(|v| *v * d_inv()).collect();
            crate::epool::run_hetero_chunks(hi_size, |x_hi| {
                let mut partial = [F128::ZERO; ELL];
                process_one_x_hi_with_precomputed_ab_fold4_ab(
                    x_hi,
                    big_lo_size,
                    n_lo_and_inner,
                    within_outer_mask,
                    &b_med_counts,
                    ab_inner_bytes,
                    &eq_lo_scaled,
                    eq.hi[x_hi],
                    convert,
                    None,
                    &mut partial,
                );
                // SAFETY: each queue index owns one disjoint output slot and
                // the synchronous queue join publishes all writes before
                // reduction.
                unsafe { *partials_base.ptr().add(x_hi) = partial };
            });
        }
        Some(bank_bits) => {
            let r_lo = &r[k_skip + N_INNER..k_skip + N_INNER + eq.n_lo];
            let (eq_bot, eq_top_scaled) = ab_eq_fold_factors(r_lo, bank_bits);
            let tables = build_ab_eq_fold_tables(&eq_top_scaled, convert);
            let bank_len = eq_bot.len() * ELL;
            crate::epool::run_hetero_chunks(hi_size, |x_hi| {
                let mut partial = [F128::ZERO; ELL];
                AB_EQ_FOLD_BANKS.with(|cell| {
                    let mut banks = cell.borrow_mut();
                    if banks.len() != bank_len {
                        banks.clear();
                        banks.resize(bank_len, F128::ZERO);
                    }
                    process_one_x_hi_with_precomputed_ab_fold4_ab_eq_folded(
                        x_hi,
                        big_lo_size,
                        n_lo_and_inner,
                        within_outer_mask,
                        &b_med_counts,
                        ab_inner_bytes,
                        &eq_bot,
                        eq.hi[x_hi],
                        &tables,
                        bank_bits,
                        &mut banks,
                        &mut partial,
                    );
                });
                // SAFETY: each queue index owns one disjoint output slot and
                // the synchronous queue join publishes all writes before
                // reduction.
                unsafe { *partials_base.ptr().add(x_hi) = partial };
            });
            crate::scratch::give_f128(tables);
        }
    }

    partials
        .into_iter()
        .fold([F128::ZERO; ELL], |mut left, right| {
            for lane in 0..ELL {
                left[lane] += right[lane];
            }
            left
        })
        .to_vec()
}

/// Challenge-derived inputs shared by both round-one C variants.
///
/// Built (and, at the ranked shape, submitted to the GPU) BEFORE the round-one
/// AB completion. Nothing here depends on AB, and the GPU is idle for the
/// whole zerocheck window, so starting the C fold's GPU prefix first lets it
/// cover the AB completion as well as the CPU's own share of the fold.
pub(crate) struct Round1CPrelude {
    /// eq(r_outer, ·) — owned, or built in place in the GPU fold arm's
    /// persistent upload buffer so the launch skips its 4 MiB memcpy (see
    /// `gpu_commit::zc_fold_eq_table`; `FLOCK_NO_EQ_DIRECT=1` restores the
    /// owned build). Same bytes either way; every CPU consumer reads it
    /// through `as_slice`.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    eq_outer: crate::gpu_commit::FoldEqTable,
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    eq_outer: Vec<F128>,
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    gpu: Option<crate::gpu_commit::ZcFoldJob>,
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    submitted: std::time::Instant,
}

impl Round1CPrelude {
    /// Whether a GPU C-fold prefix was actually submitted — i.e. there is a
    /// measured GPU idle window behind it worth filling (see
    /// `gpu_commit::ENV_NO_ZC_IDLE_FILL`).
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub(crate) fn gpu_in_flight(&self) -> bool {
        self.gpu.is_some()
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    #[allow(dead_code)] // the idle-fill call site is macOS/aarch64-only
    pub(crate) fn gpu_in_flight(&self) -> bool {
        false
    }
}

/// Commit-tail fill, staging wrapper (see
/// `gpu_commit::ENV_NO_COMMIT_TAIL_FILL`): submit the round-one C fold's
/// GPU prefix now — at commit-graph completion, from a forked challenger's
/// challenge vector — and park it in the fold state for adoption by
/// [`round1_c_prelude`] at zerocheck entry. No-op off the ranked arm shape.
pub(crate) fn stage_c_prelude_for_tail_fill(
    c_lincheck: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    r: &[F128],
) -> bool {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        if !ranked_c_fold_shape(m, k_log) {
            return false;
        }
        crate::gpu_commit::stage_zerocheck_c_fold_prefix(c_lincheck, m, k_log, useful_bits, r)
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        let _ = (c_lincheck, m, k_log, useful_bits, r);
        false
    }
}

/// The one production shape the GPU C-fold arm is tuned and gated for
/// (`m = 32`, `k_log = 14`). Everything else takes the exact CPU path.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn ranked_c_fold_shape(m: usize, k_log: usize) -> bool {
    // `cfg!(test)` widens the gate so the end-to-end transcript oracle can
    // drive the arm at a small shape. Production is the ranked shape only.
    (cfg!(test) || (m == 32 && k_log == 14))
        && !crate::lincheck::FOLD_IBLOCK.load(std::sync::atomic::Ordering::Relaxed)
}

/// Build `eq_outer` and submit the GPU prefix of the C fold.
pub(crate) fn round1_c_prelude(
    c_lincheck: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    r: &[F128],
) -> Round1CPrelude {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        // Commit-tail fill adoption: an identical dispatch parked at
        // commit-graph completion (exact challenge-vector + stripe match)
        // short-circuits the build-and-launch below.
        if ranked_c_fold_shape(m, k_log) {
            if let Some((eq_outer, job, submitted)) =
                crate::gpu_commit::adopt_staged_zc_fold(c_lincheck, r)
            {
                if std::env::var_os("FLOCK_ZC_TIMING").is_some() {
                    eprintln!("[commit-tail-fill] staged C prelude adopted");
                }
                return Round1CPrelude {
                    eq_outer,
                    gpu: Some(job),
                    submitted,
                };
            }
        }
        // Off the arm's shape no launch will consume a staged table, so keep
        // the incumbent owned build there (same gate as the launch below).
        let eq_outer = if ranked_c_fold_shape(m, k_log) {
            crate::gpu_commit::zc_fold_eq_table(&r[k_log..])
        } else {
            crate::gpu_commit::FoldEqTable::Owned(crate::lincheck::build_eq_table(&r[k_log..]))
        };
        let gpu = ranked_c_fold_shape(m, k_log)
            .then(|| {
                crate::gpu_commit::launch_zerocheck_c_fold(
                    c_lincheck,
                    m,
                    k_log,
                    useful_bits,
                    eq_outer.as_slice(),
                )
            })
            .flatten();
        Round1CPrelude {
            eq_outer,
            gpu,
            submitted: std::time::Instant::now(),
        }
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        let _ = (c_lincheck, m, k_log, useful_bits);
        Round1CPrelude {
            eq_outer: crate::lincheck::build_eq_table(&r[k_log..]),
        }
    }
}

/// Fold the lincheck stripe against `eq_outer`, draining the GPU prefix when
/// one is in flight. Bit-identical to `partial_fold_packed_z_best` in every
/// case: GF(2¹²⁸) add is XOR, so splitting the stripe range between the two
/// arms and XORing the halves reproduces the whole-range result exactly.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn round1_c_inner_fold(
    prelude: Round1CPrelude,
    c_lincheck: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
) -> Vec<F128> {
    let Round1CPrelude {
        eq_outer,
        gpu,
        submitted,
    } = prelude;
    if let Some(job) = gpu {
        let claim_lo = job.claim_lo();
        let head_ms = submitted.elapsed().as_secs_f64() * 1e3;
        let t_suffix = std::time::Instant::now();
        let mut out = crate::lincheck::partial_fold_packed_z_neon_oblock_padded_suffix(
            c_lincheck,
            m,
            k_log,
            useful_bits,
            eq_outer.as_slice(),
            claim_lo,
        );
        let suffix_ms = t_suffix.elapsed().as_secs_f64() * 1e3;
        match job.finish_xor_into(&mut out, head_ms, suffix_ms) {
            Ok(()) => return out,
            Err(e) => {
                // The prefix never landed and the CPU already skipped it.
                // Redo exactly those claims here — slower, still exact.
                if crate::gpu_commit::gpu_zerocheck_debug() {
                    eprintln!("[gpu-zc] prefix failed, CPU redo: {e}");
                }
                let prefix = crate::lincheck::partial_fold_packed_z_neon_oblock_padded_range(
                    c_lincheck,
                    m,
                    k_log,
                    useful_bits,
                    eq_outer.as_slice(),
                    0,
                    claim_lo,
                );
                for (a, b) in out.iter_mut().zip(prefix) {
                    *a += b;
                }
                return out;
            }
        }
    }
    crate::lincheck::partial_fold_packed_z_best(
        c_lincheck,
        m,
        k_log,
        useful_bits,
        eq_outer.as_slice(),
    )
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
fn round1_c_inner_fold(
    prelude: Round1CPrelude,
    c_lincheck: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
) -> Vec<F128> {
    crate::lincheck::partial_fold_packed_z_best(
        c_lincheck,
        m,
        k_log,
        useful_bits,
        &prelude.eq_outer,
    )
}

/// Derive the exact legacy C round-one message and its existing RingSwitch
/// helper tensors from the lincheck stripe.  The stripe represents identity C
/// (`Cz = z`) in an outer-fold-friendly layout.  Folding it at the original
/// zerocheck `r_outer` yields a tiny length-`2^k_log` inner table; retaining
/// four inner coordinates from that table is algebraically identical to the
/// row-major 32-bank Fold4 drain.
#[allow(clippy::too_many_arguments)]
pub(crate) fn round1_c_fold4_from_lincheck_stripe(
    c_lincheck: &[u8],
    m: usize,
    k_log: usize,
    k_skip: usize,
    useful_bits: usize,
    r: &[F128],
    inv_table: &InvNttTableByteSingleGf8,
    prelude: Round1CPrelude,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<F128>) {
    assert_eq!(k_skip, K_SKIP);
    assert!(
        k_log >= k_skip + 5,
        "Fold4 needs four retained tail coordinates"
    );
    assert_eq!(r.len(), m);
    assert_eq!(c_lincheck.len(), (1usize << m) / 8);

    let c_inner = round1_c_inner_fold(prelude, c_lincheck, m, k_log, useful_bits);
    let inner_tail = &r[k_skip + 1..k_log];
    let fold4 = crate::pcs::ring_switch::s_hat_v_fold4_from_z_vec(&c_inner, inner_tail);

    // Fold only retained coordinates 2 and 3 to recover the incumbent
    // four-bank tensor (coordinates 0 and 1 remain bank selectors).
    let retained_hi_eq = build_eq(&inner_tail[2..4]);
    let n_packed = 1usize << crate::pcs::LOG_PACKING;
    let mut quad = vec![F128::ZERO; 4 * n_packed];
    for q in 0..4 {
        for e in 0..4 {
            let src = (e + 4 * q) * n_packed;
            let dst = e * n_packed;
            for packed in 0..n_packed {
                quad[dst + packed] += retained_hi_eq[q] * fold4[src + packed];
            }
        }
    }
    let s_hat_v_c = crate::pcs::ring_switch::collapse_s_hat_v_quad(&quad, &inner_tail[..2]);

    // RingSwitch leaves global bit k_skip as its 128-way prefix. Fold that
    // bit at the original C point to recover C's 64 S-domain evaluations.
    // The optimized round-one convention omits C_s; the common caller restores
    // it before placing the message on the transcript.
    let prefix = r[k_skip];
    let mut res_c_s = [F128::ZERO; ELL];
    let c_s_inv = c_s_f128().inv();
    for lane in 0..ELL {
        let naive = (F128::ONE + prefix) * s_hat_v_c[lane] + prefix * s_hat_v_c[ELL + lane];
        res_c_s[lane] = c_s_inv * naive;
    }
    let round1_c_opt = ntt_extend_f128_vec_ghash(&res_c_s, inv_table);
    (round1_c_opt, s_hat_v_c, quad, fold4)
}

/// Fold8 sibling of [`round1_c_fold4_from_lincheck_stripe`]: same single
/// stripe fold, but six inner coordinates are retained (64 banks) for the
/// direct-fold8 PCS consumer. The wire outputs (`round1_c_opt`, canonical
/// `s_hat_v_c`, `quad`) are derived by collapsing the wider statistic and are
/// bitwise identical to the fold4 variant's — the extra two retained
/// coordinates only widen the exported tensor.
#[allow(clippy::too_many_arguments)]
pub(crate) fn round1_c_fold8_from_lincheck_stripe(
    c_lincheck: &[u8],
    m: usize,
    k_log: usize,
    k_skip: usize,
    useful_bits: usize,
    r: &[F128],
    inv_table: &InvNttTableByteSingleGf8,
    prelude: Round1CPrelude,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<F128>) {
    assert_eq!(k_skip, K_SKIP);
    assert!(
        k_log >= k_skip + 7,
        "Fold8 needs six retained tail coordinates"
    );
    assert_eq!(r.len(), m);
    assert_eq!(c_lincheck.len(), (1usize << m) / 8);

    let c_inner = round1_c_inner_fold(prelude, c_lincheck, m, k_log, useful_bits);
    let inner_tail = &r[k_skip + 1..k_log];
    let fold8 = crate::pcs::ring_switch::s_hat_v_fold8_from_z_vec(&c_inner, inner_tail);

    // Fold retained coordinates 2..6 to recover the incumbent four-bank
    // tensor (coordinates 0 and 1 remain bank selectors).
    let retained_hi_eq = build_eq(&inner_tail[2..6]);
    let n_packed = 1usize << crate::pcs::LOG_PACKING;
    let mut quad = vec![F128::ZERO; 4 * n_packed];
    for q in 0..16 {
        for e in 0..4 {
            let src = (e + 4 * q) * n_packed;
            let dst = e * n_packed;
            for packed in 0..n_packed {
                quad[dst + packed] += retained_hi_eq[q] * fold8[src + packed];
            }
        }
    }
    let s_hat_v_c = crate::pcs::ring_switch::collapse_s_hat_v_quad(&quad, &inner_tail[..2]);

    // RingSwitch leaves global bit k_skip as its 128-way prefix. Fold that
    // bit at the original C point to recover C's 64 S-domain evaluations.
    // The optimized round-one convention omits C_s; the common caller restores
    // it before placing the message on the transcript.
    let prefix = r[k_skip];
    let mut res_c_s = [F128::ZERO; ELL];
    let c_s_inv = c_s_f128().inv();
    for lane in 0..ELL {
        let naive = (F128::ONE + prefix) * s_hat_v_c[lane] + prefix * s_hat_v_c[ELL + lane];
        res_c_s[lane] = c_s_inv * naive;
    }
    let round1_c_opt = ntt_extend_f128_vec_ghash(&res_c_s, inv_table);
    (round1_c_opt, s_hat_v_c, quad, fold8)
}

/// Pair-fused C job with the original 32-bank accumulator and one job per
/// `x_hi`. This isolates the benefit of halving field-table/state updates
/// from the q-local scheduling experiment below.
#[inline]
#[allow(clippy::too_many_arguments)]
fn process_one_x_hi_with_precomputed_ab_fold4_c_pair(
    x_hi: usize,
    big_lo_size: usize,
    n_lo_and_inner: usize,
    within_outer_mask: usize,
    b_med_counts: &[u8],
    c_packed: &[u8],
    eq_hi_val: F128,
    c_fold4_pair_mask_tables: &[F128],
    timing_cpu_ns: Option<&[std::sync::atomic::AtomicU64; 3]>,
    partial_c: &mut [[F128; ELL]; N_C_FOLD4_BANKS],
) {
    partial_c.iter_mut().for_each(|bank| bank.fill(F128::ZERO));

    let n_lo = n_lo_and_inner - N_INNER;
    let t_c = timing_cpu_ns.map(|_| std::time::Instant::now());
    for pair in 0..big_lo_size.div_ceil(2) {
        let x_outer_lo_even = 2 * pair;
        let x_outer_lo_odd = x_outer_lo_even + 1;
        let x_outer_even = x_outer_lo_even | (x_hi << n_lo);
        let n_b_med_even = b_med_counts[x_outer_even & within_outer_mask] as usize;
        let n_b_med_odd = if x_outer_lo_odd < big_lo_size {
            let x_outer_odd = x_outer_lo_odd | (x_hi << n_lo);
            b_med_counts[x_outer_odd & within_outer_mask] as usize
        } else {
            0
        };
        if n_b_med_even == 0 && n_b_med_odd == 0 {
            continue;
        }

        let even_byte_base = ((x_outer_lo_even << N_INNER) | (x_hi << n_lo_and_inner)) * N_CHUNKS;
        let odd_byte_base = if x_outer_lo_odd < big_lo_size {
            ((x_outer_lo_odd << N_INNER) | (x_hi << n_lo_and_inner)) * N_CHUNKS
        } else {
            even_byte_base
        };
        let c_rows_even: &[u8; (1 << N_MEDIUM) * ELL] = (&c_packed
            [even_byte_base..even_byte_base + (1 << N_MEDIUM) * ELL])
            .try_into()
            .expect("sixteen 64-byte c rows per even x_outer_lo");
        let c_rows_odd: &[u8; (1 << N_MEDIUM) * ELL] = (&c_packed
            [odd_byte_base..odd_byte_base + (1 << N_MEDIUM) * ELL])
            .try_into()
            .expect("sixteen 64-byte c rows per odd x_outer_lo");
        let table: &[F128; C_FOLD4_PAIR_MASK_TABLE_STRIDE] = (&c_fold4_pair_mask_tables
            [pair * C_FOLD4_PAIR_MASK_TABLE_STRIDE..(pair + 1) * C_FOLD4_PAIR_MASK_TABLE_STRIDE])
            .try_into()
            .expect("one 256-entry Fold4 pair table");
        kernels::accumulate_c_fold4_pair_banks(
            c_rows_even,
            n_b_med_even,
            c_rows_odd,
            n_b_med_odd,
            table,
            partial_c,
        );
    }
    if let (Some(totals), Some(start)) = (timing_cpu_ns, t_c) {
        totals[1].fetch_add(
            start.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    let t_high = timing_cpu_ns.map(|_| std::time::Instant::now());
    for bank in partial_c {
        for value in bank {
            *value *= eq_hi_val;
        }
    }
    if let (Some(totals), Some(start)) = (timing_cpu_ns, t_high) {
        totals[2].fetch_add(
            start.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
}

/// Four-block follow-on to pair fusion. Two pair-table values are combined
/// before one 32-bank accumulator update, while retaining the same monolithic
/// per-x_hi scheduling and deterministic reduction shape.
#[inline]
#[allow(clippy::too_many_arguments)]
fn process_one_x_hi_with_precomputed_ab_fold4_c_four(
    x_hi: usize,
    big_lo_size: usize,
    n_lo_and_inner: usize,
    within_outer_mask: usize,
    b_med_counts: &[u8],
    c_packed: &[u8],
    eq_hi_val: F128,
    c_fold4_pair_mask_tables: &[F128],
    timing_cpu_ns: Option<&[std::sync::atomic::AtomicU64; 3]>,
    partial_c: &mut [[F128; ELL]; N_C_FOLD4_BANKS],
) {
    assert!(big_lo_size >= 4 && big_lo_size % 4 == 0);
    partial_c.iter_mut().for_each(|bank| bank.fill(F128::ZERO));

    let n_lo = n_lo_and_inner - N_INNER;
    let t_c = timing_cpu_ns.map(|_| std::time::Instant::now());
    for quartet in 0..big_lo_size / 4 {
        let lo_base = 4 * quartet;
        let n_b_med: [usize; 4] = std::array::from_fn(|side| {
            let x_outer_lo = lo_base + side;
            let x_outer = x_outer_lo | (x_hi << n_lo);
            b_med_counts[x_outer & within_outer_mask] as usize
        });
        if n_b_med.into_iter().all(|count| count == 0) {
            continue;
        }

        let c_blocks: [&[u8; (1 << N_MEDIUM) * ELL]; 4] = std::array::from_fn(|side| {
            let x_outer_lo = lo_base + side;
            let byte_base = ((x_outer_lo << N_INNER) | (x_hi << n_lo_and_inner)) * N_CHUNKS;
            (&c_packed[byte_base..byte_base + (1 << N_MEDIUM) * ELL])
                .try_into()
                .expect("sixteen 64-byte c rows per x_outer_lo")
        });
        let pair_mask_tables: [&[F128; C_FOLD4_PAIR_MASK_TABLE_STRIDE]; 2] =
            std::array::from_fn(|pair_in_quartet| {
                let pair = 2 * quartet + pair_in_quartet;
                (&c_fold4_pair_mask_tables[pair * C_FOLD4_PAIR_MASK_TABLE_STRIDE
                    ..(pair + 1) * C_FOLD4_PAIR_MASK_TABLE_STRIDE])
                    .try_into()
                    .expect("one 256-entry Fold4 pair table")
            });
        kernels::accumulate_c_fold4_four_banks(c_blocks, n_b_med, pair_mask_tables, partial_c);
    }
    if let (Some(totals), Some(start)) = (timing_cpu_ns, t_c) {
        totals[1].fetch_add(
            start.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    let t_high = timing_cpu_ns.map(|_| std::time::Instant::now());
    for bank in partial_c {
        for value in bank {
            *value *= eq_hi_val;
        }
    }
    if let (Some(totals), Some(start)) = (timing_cpu_ns, t_high) {
        totals[2].fetch_add(
            start.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
}

/// One q-local paired C job.  Adjacent `x_outer_lo` blocks are drained
/// together through a 256-entry table, halving field-table and accumulator
/// updates.  Four independent jobs cover q=0..3, each with an 8 KiB live
/// accumulator and only the four witness rows belonging to its q group.
#[inline]
#[allow(clippy::too_many_arguments)]
fn process_one_x_hi_with_precomputed_ab_fold4_c_q_pair(
    x_hi: usize,
    q: usize,
    big_lo_size: usize,
    n_lo_and_inner: usize,
    within_outer_mask: usize,
    b_med_counts: &[u8],
    c_packed: &[u8],
    eq_hi_val: F128,
    c_fold4_pair_mask_tables: &[F128],
    timing_cpu_ns: Option<&[std::sync::atomic::AtomicU64; 3]>,
    partial_c: &mut [[F128; ELL]; N_C_BANKS],
) {
    debug_assert!(q < N_C_FOLD4_GROUPS);
    partial_c.iter_mut().for_each(|bank| bank.fill(F128::ZERO));

    let n_lo = n_lo_and_inner - N_INNER;
    let t_c = timing_cpu_ns.map(|_| std::time::Instant::now());
    for pair in 0..big_lo_size.div_ceil(2) {
        let x_outer_lo_even = 2 * pair;
        let x_outer_lo_odd = x_outer_lo_even + 1;
        let x_outer_even = x_outer_lo_even | (x_hi << n_lo);
        let n_b_med_even = b_med_counts[x_outer_even & within_outer_mask] as usize;
        let n_b_med_odd = if x_outer_lo_odd < big_lo_size {
            let x_outer_odd = x_outer_lo_odd | (x_hi << n_lo);
            b_med_counts[x_outer_odd & within_outer_mask] as usize
        } else {
            0
        };
        if n_b_med_even == 0 && n_b_med_odd == 0 {
            continue;
        }

        let even_byte_base = ((x_outer_lo_even << N_INNER) | (x_hi << n_lo_and_inner)) * N_CHUNKS;
        let odd_byte_base = if x_outer_lo_odd < big_lo_size {
            ((x_outer_lo_odd << N_INNER) | (x_hi << n_lo_and_inner)) * N_CHUNKS
        } else {
            // The paired kernel is told the odd block has zero live rows, so
            // aliasing the even in-bounds block here is semantically inert.
            even_byte_base
        };
        let c_rows_even: &[u8; (1 << N_MEDIUM) * ELL] = (&c_packed
            [even_byte_base..even_byte_base + (1 << N_MEDIUM) * ELL])
            .try_into()
            .expect("sixteen 64-byte c rows per even x_outer_lo");
        let c_rows_odd: &[u8; (1 << N_MEDIUM) * ELL] = (&c_packed
            [odd_byte_base..odd_byte_base + (1 << N_MEDIUM) * ELL])
            .try_into()
            .expect("sixteen 64-byte c rows per odd x_outer_lo");
        let table: &[F128; C_FOLD4_PAIR_MASK_TABLE_STRIDE] = (&c_fold4_pair_mask_tables
            [pair * C_FOLD4_PAIR_MASK_TABLE_STRIDE..(pair + 1) * C_FOLD4_PAIR_MASK_TABLE_STRIDE])
            .try_into()
            .expect("one 256-entry Fold4 pair table");
        kernels::accumulate_c_fold4_q_pair_banks(
            c_rows_even,
            n_b_med_even,
            c_rows_odd,
            n_b_med_odd,
            q,
            table,
            partial_c,
        );
    }
    if let (Some(totals), Some(start)) = (timing_cpu_ns, t_c) {
        totals[1].fetch_add(
            start.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    let t_high = timing_cpu_ns.map(|_| std::time::Instant::now());
    for bank in partial_c {
        for value in bank {
            *value *= eq_hi_val;
        }
    }
    if let (Some(totals), Some(start)) = (timing_cpu_ns, t_high) {
        totals[2].fetch_add(
            start.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
}

/// Build the `b_med_counts` table from a [`PaddingSpec`] for use by
/// [`process_one_x_hi`].
///
/// Returns `(within_outer_mask, b_med_counts)`:
///   - `within_outer_mask` masks `x_outer` to the bits identifying the
///     within-block window.
///   - `b_med_counts[w]` is how many of the 16 b_med 512-bit sub-windows of
///     window `w` we should process. Entries past the useful prefix are 0
///     (full skip) — kernels just `continue` past those x_outer_lo iterations.
fn build_b_med_counts(padding: &PaddingSpec) -> (usize, Vec<u8>) {
    const STRIDE: usize = 1 << (K_SKIP + N_INNER); // 8192 bits per within-window
    const B_MED_WINDOW: usize = 1 << (K_SKIP + 3); // 512 bits per b_med
    const N_B_MED_MAX: usize = 1 << N_MEDIUM;

    // For k_log < K_SKIP + N_INNER (= 13) the within-window granularity is
    // coarser than the block itself — skipping at this granularity would be
    // incorrect, so we fall back to "no skip". All hash modules use
    // k_log ∈ {14, 15, 16}.
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

/// Packed-input variant of [`round1_shift_reduce_extract_c`]. **Parallel by
/// default** via rayon — the outer x_hi loop is distributed across workers,
/// each with its own scratch + local accumulator. Reduction is a per-lane
/// F128 XOR across workers (commutative + associative).
///
/// To run single-threaded for debugging, set `RAYON_NUM_THREADS=1`.
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

/// Padding-aware variant of [`round1_shift_reduce_extract_c_packed`]. Skips
/// 512-bit b_med sub-windows that fall entirely in the zero padding of every
/// witness block per `padding`. Output is byte-identical to the dense path
/// when the padding bits are honestly zero.
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

    // Parallel fold: each worker accumulates a subset of x_hi values into its
    // own WorkerState. Reduce step combines the per-worker `local_res_*` by
    // per-lane F128 XOR.
    let (res_ab, res_c_s) = (0..hi_size)
        .into_par_iter()
        .fold(WorkerState::new, |mut state, x_hi| {
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
        })
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

/// Reduce the eight α-free C banks into everything downstream needs.
///
/// Returns `(res_c_s, s_hat_v_c, quad_c)`:
/// * `res_c_s` — the wire accumulator, `Σ_K α^K · bank_K`. Bank `K` was
///   accumulated α-free, so re-applying `α^K` reproduces exactly what the
///   incumbent single/two-bank convert-table fold accumulated.
/// * `s_hat_v_c` — canonical length-128 form, `C_2 · α^{-b_3[0]} · Σ_{K ≡
///   b_3[0] (mod 2)} α^K · bank_K`, i.e. the incumbent post-scaling applied to
///   the parity-class sums.
/// * `quad_c` — the length-512 four-bank sufficient statistic,
///   `quad_c[e·128 + 64·b_3[0] + lane] = bank_{b_3[0] + 2e}[lane]`. **α-free
///   and `C_2`-free**: both constants are `low_eq`'s, and `low_eq` is
///   re-applied downstream (`collapse_s_hat_v_quad` on intake and
///   `build_direct_fold2_table` at materialize time).
///
/// `collapse_s_hat_v_quad(quad_c, [φ₈(0x53), φ₈(0xB5)]) == s_hat_v_c` by
/// construction: `low_eq[e] = C_2 · α^{2e}`, so the collapse re-weights bank
/// `2e + b_3[0]` by `C_2 · α^{2e}` and the two forms differ only by the shared
/// `α^{-b_3[0]}`. Asserted by `quad_collapses_to_wire_s_hat_v_c`.
fn finish_c_banks(banks: &[[F128; ELL]; N_C_BANKS]) -> ([F128; ELL], Vec<F128>, Vec<F128>) {
    let alpha = phi8(F8(0x02));
    let mut alpha_pow = [F128::ONE; N_C_BANKS];
    for k in 1..N_C_BANKS {
        alpha_pow[k] = alpha_pow[k - 1] * alpha;
    }

    let mut res_c_s_0 = [F128::ZERO; ELL];
    let mut res_c_s_1 = [F128::ZERO; ELL];
    for (k, bank) in banks.iter().enumerate() {
        let weight = alpha_pow[k];
        let target = if k & 1 == 0 {
            &mut res_c_s_0
        } else {
            &mut res_c_s_1
        };
        for lane in 0..ELL {
            target[lane] += weight * bank[lane];
        }
    }

    let mut res_c_s = [F128::ZERO; ELL];
    for lane in 0..ELL {
        res_c_s[lane] = res_c_s_0[lane] + res_c_s_1[lane];
    }

    let c_2 = c_2_small_f128();
    let c_2_alpha_inv = c_2 * alpha_inv_f128();
    let mut s_hat_v_c = vec![F128::ZERO; 2 * ELL];
    for lane in 0..ELL {
        s_hat_v_c[lane] = c_2 * res_c_s_0[lane];
        s_hat_v_c[ELL + lane] = c_2_alpha_inv * res_c_s_1[lane];
    }

    let mut quad_c = vec![F128::ZERO; 4 * 2 * ELL];
    for e in 0..4 {
        for b_0 in 0..2 {
            let base = e * 2 * ELL + b_0 * ELL;
            quad_c[base..base + ELL].copy_from_slice(&banks[b_0 + 2 * e]);
        }
    }

    (res_c_s, s_hat_v_c, quad_c)
}

/// Finish the retained-medium Fold4 statistic while preserving the incumbent
/// outputs.  `fold4_c` is bank-major in the exact order consumed by the PCS:
///
/// ```text
/// low = e_small + 4*q_medium
/// fold4_c[low * 128 + b_prefix * 64 + lane]
///     = banks[q_medium][b_prefix + 2*e_small][lane].
/// ```
///
/// It is alpha-free and carries neither C_2 nor the low-medium eq weights;
/// all four retained suffix weights are re-applied by the direct-fold4
/// consumer.  Collapsing there with `build_eq(suffix[..4])` therefore gives
/// the same canonical `s_hat_v_c` returned here.
fn finish_c_fold4_banks(
    banks: &[[F128; ELL]; N_C_FOLD4_BANKS],
) -> ([F128; ELL], Vec<F128>, Vec<F128>, Vec<F128>) {
    let collapsed = collapse_c_fold4_banks(banks);
    let (res_c_s, s_hat_v_c, quad_c) = finish_c_banks(&collapsed);

    let n_packed = 2 * ELL;
    let mut fold4_c = vec![F128::ZERO; 16 * n_packed];
    for q in 0..N_C_FOLD4_GROUPS {
        for e in 0..4 {
            let low = e + 4 * q;
            for b_prefix in 0..2 {
                let dst = low * n_packed + b_prefix * ELL;
                let k = b_prefix + 2 * e;
                fold4_c[dst..dst + ELL].copy_from_slice(&banks[q * N_C_BANKS + k]);
            }
        }
    }

    (res_c_s, s_hat_v_c, quad_c, fold4_c)
}

/// Same as [`round1_shift_reduce_extract_c_packed_padded`] but **also returns
/// `s_hat_v_c`** — the length-128 vector ring-switch would otherwise produce
/// via `fold_1b_rows` for the c-claim's PCS opening at suffix `r[k_skip+1..m]`.
///
/// The wire output `(res_ab, res_c_lifted)` is byte-identical to
/// [`round1_shift_reduce_extract_c_packed_padded`] — same eq weights, same
/// `C_s` drop convention. `s_hat_v_c` is returned in **canonical form**
/// (matches `fold_1b_rows`), with the residual `C_2` and `α⁻¹` scaling
/// applied internally so the caller can feed it straight into
/// `pcs::ring_switch::prove_batched_padded_with_precomputed`.
///
/// Also returns `quad_c`, the length-512 α-free four-bank sufficient statistic
/// for the same claim — see [`finish_c_banks`]. Both are produced from the same
/// eight banks, so shipping either costs the same sweep.
pub fn round1_shift_reduce_extract_c_packed_padded_with_s_hat_v_quad(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    k_skip: usize,
    r: &[F128],
    inv_table: &InvNttTableByteSingleGf8,
    padding: &PaddingSpec,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<F128>) {
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
    let mask_tables = build_c_mask_tables(&eq_lo_scaled);
    let eq_hi = &eq.hi;

    let (within_outer_mask, b_med_counts) = build_b_med_counts(padding);

    let (res_ab, banks) = (0..hi_size)
        .into_par_iter()
        .fold(WorkerStateWithSHatV::new, |mut state, x_hi| {
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
                &mask_tables,
                &mut state,
            );
            state
        })
        .map(|s| (s.local_res_ab, s.local_res_c_s))
        .reduce(
            || ([F128::ZERO; ELL], [[F128::ZERO; ELL]; N_C_BANKS]),
            |(mut ab1, mut c1), (ab2, c2)| {
                for i in 0..ELL {
                    ab1[i] += ab2[i];
                }
                for (left, right) in c1.iter_mut().zip(c2) {
                    for i in 0..ELL {
                        left[i] += right[i];
                    }
                }
                (ab1, c1)
            },
        );

    crate::scratch::give_f128(mask_tables);
    let (res_c_s, s_hat_v_c, quad_c) = finish_c_banks(&banks);
    let res_c_lifted = ntt_extend_f128_vec_ghash(&res_c_s, inv_table);

    (res_ab.to_vec(), res_c_lifted, s_hat_v_c, quad_c)
}

/// [`round1_shift_reduce_extract_c_packed_padded_with_s_hat_v_quad`] without the
/// quad. Retained as the stable three-output entry point for the round-1
/// benchmarks.
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
    let (res_ab, res_c_lifted, s_hat_v_c, _quad_c) =
        round1_shift_reduce_extract_c_packed_padded_with_s_hat_v_quad(
            a_packed, b_packed, c_packed, m, k_skip, r, inv_table, padding,
        );
    (res_ab, res_c_lifted, s_hat_v_c)
}

/// Challenge-weighted completion of round 1 using AB blocks returned by
/// [`precompute_round1_ab_inner_packed_padded`]. This is byte-identical to
/// [`round1_shift_reduce_extract_c_packed_padded_with_s_hat_v_quad`], while
/// keeping the original A and B packed buffers available for zerocheck round 2.
pub fn round1_shift_reduce_extract_c_packed_padded_with_precomputed_ab(
    ab_inner: &Round1AbInner,
    c_packed: &[u8],
    m: usize,
    k_skip: usize,
    r: &[F128],
    inv_table: &InvNttTableByteSingleGf8,
    padding: &PaddingSpec,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<F128>) {
    assert_eq!(k_skip, K_SKIP, "optimized variant is k_skip=6 only");
    assert!(
        m >= k_skip + N_INNER,
        "m must be ≥ k_skip + N_INNER ({}) for the shift_reduce optimization",
        k_skip + N_INNER
    );
    let total_bytes = (1usize << m) / 8;
    assert_eq!(ab_inner.len_bytes(), total_bytes);
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
    let mask_tables = build_c_mask_tables(&eq_lo_scaled);
    let eq_hi = &eq.hi;
    let (within_outer_mask, b_med_counts) = build_b_med_counts(padding);
    let ab_inner_bytes = ab_inner.as_bytes();
    let split_ab_c = std::env::var_os("FLOCK_NO_ZC_SPLIT_AB_C").is_none();
    let direct_ab_rows = std::env::var_os("FLOCK_NO_ZC_DIRECT_AB_ROWS").is_none();
    let c_drain4 = std::env::var_os("FLOCK_NO_ZC_C_DRAIN4").is_none();

    // The challenge-independent AB transform finishes while the commitment is
    // still running. Its challenge-weighted completion is therefore a live
    // prover phase, and each x_hi is independent. Drain those chunks through
    // the shared P/E-core queue without changing the hot conversion kernel.
    // A fixed per-index partial keeps the nondeterministic claim order out of
    // the output; the final operation is XOR, so serial reduction is exact.
    let mut partials: Vec<([F128; ELL], [[F128; ELL]; N_C_BANKS])> =
        vec![([F128::ZERO; ELL], [[F128::ZERO; ELL]; N_C_BANKS]); hi_size];
    let partials_base = crate::epool::SyncPtr(partials.as_mut_ptr());
    crate::epool::run_hetero_chunks(hi_size, |x_hi| {
        let mut state = WorkerStateWithSHatV::new();
        process_one_x_hi_with_precomputed_ab(
            x_hi,
            big_lo_size,
            n_lo_and_inner,
            within_outer_mask,
            &b_med_counts,
            ab_inner_bytes,
            c_packed,
            &eq_lo_scaled,
            eq_hi[x_hi],
            convert,
            &mask_tables,
            split_ab_c,
            direct_ab_rows,
            c_drain4,
            &mut state,
        );
        // SAFETY: the queue hands out each x_hi exactly once, so this task is
        // the exclusive owner of partials[x_hi]. The queue's completion join
        // publishes every write before the reduction below reads the vector.
        unsafe {
            *partials_base.ptr().add(x_hi) = (state.local_res_ab, state.local_res_c_s);
        }
    });
    let (res_ab, banks) = partials.into_iter().fold(
        ([F128::ZERO; ELL], [[F128::ZERO; ELL]; N_C_BANKS]),
        |(mut ab1, mut c1), (ab2, c2)| {
            for i in 0..ELL {
                ab1[i] += ab2[i];
            }
            for (left, right) in c1.iter_mut().zip(c2) {
                for i in 0..ELL {
                    left[i] += right[i];
                }
            }
            (ab1, c1)
        },
    );

    crate::scratch::give_f128(mask_tables);
    let (res_c_s, s_hat_v_c, quad_c) = finish_c_banks(&banks);
    let res_c_lifted = ntt_extend_f128_vec_ghash(&res_c_s, inv_table);

    (res_ab.to_vec(), res_c_lifted, s_hat_v_c, quad_c)
}

/// Resolve the Fold4-only lo/hi split tuning seam. Invalid values fail loudly:
/// silently accepting a typo would make component profiles incomparable.
fn fold4_n_hi_from_env() -> usize {
    match std::env::var_os("FLOCK_EXPERIMENTAL_FOLD4_N_HI") {
        None => 7,
        Some(value) => match value.to_str() {
            Some("5") => 5,
            Some("6") => 6,
            Some("7") => 7,
            _ => panic!("FLOCK_EXPERIMENTAL_FOLD4_N_HI must be exactly 5, 6, or 7"),
        },
    }
}

/// Second-stage producer A/B seam. Pair fusion stays active on both arms;
/// exact `=1` additionally partitions each 32-bank C job into four q-local
/// 8-bank jobs. Other values retain the monolithic 128-job schedule.
fn fold4_q_partition_from_value(value: Option<&std::ffi::OsStr>) -> bool {
    value == Some(std::ffi::OsStr::new("1"))
}

fn fold4_q_partition_from_env() -> bool {
    let value = std::env::var_os("FLOCK_EXPERIMENTAL_FOLD4_Q_PARTITION");
    fold4_q_partition_from_value(value.as_deref())
}

/// Monolithic producer follow-on. By default, fuse four adjacent low slots
/// into two pair-table loads and one accumulator update. Any value for the
/// emergency kill switch restores the two-block pair-fused drain.
fn fold4_pair4_from_kill_value(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_none()
}

fn fold4_pair4_from_env() -> bool {
    let value = std::env::var_os("FLOCK_NO_OPEN_DIRECT_FOLD4_PAIR4");
    fold4_pair4_from_kill_value(value.as_deref())
}

/// Experimental direct-fold4 counterpart of
/// [`round1_shift_reduce_extract_c_packed_padded_with_precomputed_ab`].
/// Besides the incumbent four outputs it returns C's raw 16x128 retained-bank
/// tensor for the PCS direct-fold4 consumer.  Callers must gate this entry
/// behind [`crate::pcs::ranked_direct_fold4_enabled`]; it is a separate symbol
/// so the kill switch can restore the previous path instruction-for-instruction.
pub fn round1_shift_reduce_extract_c_packed_padded_with_precomputed_ab_fold4(
    ab_inner: &Round1AbInner,
    c_packed: &[u8],
    m: usize,
    k_skip: usize,
    r: &[F128],
    inv_table: &InvNttTableByteSingleGf8,
    padding: &PaddingSpec,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<F128>, Vec<F128>) {
    round1_shift_reduce_extract_c_packed_padded_with_precomputed_ab_fold4_n_hi(
        ab_inner,
        c_packed,
        m,
        k_skip,
        r,
        inv_table,
        padding,
        fold4_n_hi_from_env(),
        fold4_q_partition_from_env(),
        fold4_pair4_from_env(),
    )
}

#[allow(clippy::too_many_arguments)]
fn round1_shift_reduce_extract_c_packed_padded_with_precomputed_ab_fold4_n_hi(
    ab_inner: &Round1AbInner,
    c_packed: &[u8],
    m: usize,
    k_skip: usize,
    r: &[F128],
    inv_table: &InvNttTableByteSingleGf8,
    padding: &PaddingSpec,
    fold4_n_hi: usize,
    q_partition: bool,
    pair4: bool,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<F128>, Vec<F128>) {
    let profile =
        std::env::var_os("FLOCK_FOLD4_TIMING").as_deref() == Some(std::ffi::OsStr::new("1"));
    let t_total = std::time::Instant::now();
    assert!((5..=7).contains(&fold4_n_hi));
    assert!(
        !(q_partition && pair4),
        "FLOCK_EXPERIMENTAL_FOLD4_Q_PARTITION and default pair4 are mutually exclusive; set FLOCK_NO_OPEN_DIRECT_FOLD4_PAIR4"
    );
    assert_eq!(k_skip, K_SKIP, "optimized variant is k_skip=6 only");
    assert!(
        m >= k_skip + N_INNER,
        "m must be >= k_skip + N_INNER ({}) for the shift_reduce optimization",
        k_skip + N_INNER
    );
    let total_bytes = (1usize << m) / 8;
    assert_eq!(ab_inner.len_bytes(), total_bytes);
    assert_eq!(c_packed.len(), total_bytes);
    assert_eq!(r.len(), m);
    assert_eq!(inv_table.k, k_skip);

    // The 32-bank high fold is four times wider than the incumbent C fold.
    // Keep only seven coordinates in the high factor: ranked work still has
    // 128 independently scheduled chunks, while the final high scaling and
    // deterministic partial storage shrink 4x versus the global n_hi=9 split
    // (262,144 rather than 1,048,576 C-bank field mul-adds at the ranked
    // shape). The low table remains small because Fold4 uses only 16 entries
    // per x_outer_lo.
    let t_eq_setup = std::time::Instant::now();
    let eq = SplitEqGhash::with_n_hi(&r[k_skip + N_INNER..], fold4_n_hi);
    let big_lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    let n_lo_and_inner = eq.n_lo + N_INNER;

    // AB still folds all four medium coordinates. C retains the low two and
    // therefore absorbs only D_hi^-1; the local collapse oracle proves that
    // re-applying the omitted low eq produces the incumbent D^-1 result.
    let eq_lo_scaled: Vec<F128> = eq.lo.iter().map(|v| *v * d_inv()).collect();
    let eq_lo_hi_scaled: Vec<F128> = eq.lo.iter().map(|v| *v * d_hi_inv()).collect();
    let convert = convert_table();
    let eq_setup_ms = t_eq_setup.elapsed().as_secs_f64() * 1e3;
    let t_c_table = std::time::Instant::now();
    let c_fold4_pair_mask_tables = build_c_fold4_pair_mask_tables(&eq_lo_hi_scaled);
    let c_table_ms = t_c_table.elapsed().as_secs_f64() * 1e3;
    let eq_hi = &eq.hi;
    let (within_outer_mask, b_med_counts) = build_b_med_counts(padding);
    let ab_inner_bytes = ab_inner.as_bytes();

    let timing_cpu_ns: [std::sync::atomic::AtomicU64; 3] =
        std::array::from_fn(|_| std::sync::atomic::AtomicU64::new(0));
    let timing_cpu_ns_ref = profile.then_some(&timing_cpu_ns);
    let (res_ab, banks, queue_jobs, queue_ms, reduce_ms) = if q_partition {
        type Fold4QPartial = Box<[[F128; ELL]; N_C_BANKS]>;
        // Five interleaved jobs per x_hi keep AB single-pass while giving C
        // four q-local 8 KiB working sets. Indexed slots preserve deterministic
        // x_hi-major reduction regardless of epool completion order.
        let mut ab_partials: Vec<Option<[F128; ELL]>> =
            std::iter::repeat_with(|| None).take(hi_size).collect();
        let mut c_partials: Vec<Option<Fold4QPartial>> = std::iter::repeat_with(|| None)
            .take(hi_size * N_C_FOLD4_GROUPS)
            .collect();
        let ab_partials_base = crate::epool::SyncPtr(ab_partials.as_mut_ptr());
        let c_partials_base = crate::epool::SyncPtr(c_partials.as_mut_ptr());
        let queue_jobs = hi_size * (1 + N_C_FOLD4_GROUPS);
        let t_queue = std::time::Instant::now();
        crate::epool::run_hetero_chunks(queue_jobs, |job| {
            // Put all AB jobs first so a heterogeneous pool cannot leave a
            // lone heavy AB job as the final E-core tail. The remaining tail
            // consists entirely of uniform q-local C jobs.
            if job < hi_size {
                let x_hi = job;
                let mut partial_ab = [F128::ZERO; ELL];
                process_one_x_hi_with_precomputed_ab_fold4_ab(
                    x_hi,
                    big_lo_size,
                    n_lo_and_inner,
                    within_outer_mask,
                    &b_med_counts,
                    ab_inner_bytes,
                    &eq_lo_scaled,
                    eq_hi[x_hi],
                    convert,
                    timing_cpu_ns_ref,
                    &mut partial_ab,
                );
                // SAFETY: one AB subjob owns each x_hi slot.
                unsafe {
                    *ab_partials_base.clone().ptr().add(x_hi) = Some(partial_ab);
                }
            } else {
                let c_job = job - hi_size;
                let x_hi = c_job / N_C_FOLD4_GROUPS;
                let q = c_job % N_C_FOLD4_GROUPS;
                let mut partial_c = zero_c_fold4_q_banks();
                process_one_x_hi_with_precomputed_ab_fold4_c_q_pair(
                    x_hi,
                    q,
                    big_lo_size,
                    n_lo_and_inner,
                    within_outer_mask,
                    &b_med_counts,
                    c_packed,
                    eq_hi[x_hi],
                    &c_fold4_pair_mask_tables,
                    timing_cpu_ns_ref,
                    &mut partial_c,
                );
                let c_index = x_hi * N_C_FOLD4_GROUPS + q;
                // SAFETY: each (x_hi,q) subjob owns one C slot.
                unsafe {
                    *c_partials_base.clone().ptr().add(c_index) = Some(partial_c);
                }
            }
        });
        let queue_ms = t_queue.elapsed().as_secs_f64() * 1e3;

        let t_reduce = std::time::Instant::now();
        let res_ab = ab_partials.into_iter().map(Option::unwrap).fold(
            [F128::ZERO; ELL],
            |mut left, right| {
                for lane in 0..ELL {
                    left[lane] += right[lane];
                }
                left
            },
        );
        let mut banks = zero_c_fold4_banks();
        for (c_index, partial) in c_partials.into_iter().map(Option::unwrap).enumerate() {
            let q = c_index % N_C_FOLD4_GROUPS;
            for (k, right) in partial.iter().enumerate() {
                let left = &mut banks[q * N_C_BANKS + k];
                for lane in 0..ELL {
                    left[lane] += right[lane];
                }
            }
        }
        let reduce_ms = t_reduce.elapsed().as_secs_f64() * 1e3;
        (res_ab, banks, queue_jobs, queue_ms, reduce_ms)
    } else {
        type Fold4Partial = ([F128; ELL], Box<[[F128; ELL]; N_C_FOLD4_BANKS]>);
        let mut partials: Vec<Option<Fold4Partial>> =
            std::iter::repeat_with(|| None).take(hi_size).collect();
        let partials_base = crate::epool::SyncPtr(partials.as_mut_ptr());
        let t_queue = std::time::Instant::now();
        crate::epool::run_hetero_chunks(hi_size, |x_hi| {
            let mut partial_ab = [F128::ZERO; ELL];
            process_one_x_hi_with_precomputed_ab_fold4_ab(
                x_hi,
                big_lo_size,
                n_lo_and_inner,
                within_outer_mask,
                &b_med_counts,
                ab_inner_bytes,
                &eq_lo_scaled,
                eq_hi[x_hi],
                convert,
                timing_cpu_ns_ref,
                &mut partial_ab,
            );
            let mut partial_c = zero_c_fold4_banks();
            if pair4 && big_lo_size >= 4 {
                process_one_x_hi_with_precomputed_ab_fold4_c_four(
                    x_hi,
                    big_lo_size,
                    n_lo_and_inner,
                    within_outer_mask,
                    &b_med_counts,
                    c_packed,
                    eq_hi[x_hi],
                    &c_fold4_pair_mask_tables,
                    timing_cpu_ns_ref,
                    &mut partial_c,
                );
            } else {
                process_one_x_hi_with_precomputed_ab_fold4_c_pair(
                    x_hi,
                    big_lo_size,
                    n_lo_and_inner,
                    within_outer_mask,
                    &b_med_counts,
                    c_packed,
                    eq_hi[x_hi],
                    &c_fold4_pair_mask_tables,
                    timing_cpu_ns_ref,
                    &mut partial_c,
                );
            }
            // SAFETY: each x_hi owns one indexed slot; the epool join
            // publishes it before deterministic reduction.
            unsafe {
                *partials_base.clone().ptr().add(x_hi) = Some((partial_ab, partial_c));
            }
        });
        let queue_ms = t_queue.elapsed().as_secs_f64() * 1e3;

        let t_reduce = std::time::Instant::now();
        let (res_ab, banks) = partials.into_iter().map(Option::unwrap).fold(
            ([F128::ZERO; ELL], zero_c_fold4_banks()),
            |(mut ab1, mut c1), (ab2, c2)| {
                for lane in 0..ELL {
                    ab1[lane] += ab2[lane];
                }
                for (left, right) in c1.iter_mut().zip(c2.iter()) {
                    for lane in 0..ELL {
                        left[lane] += right[lane];
                    }
                }
                (ab1, c1)
            },
        );
        let reduce_ms = t_reduce.elapsed().as_secs_f64() * 1e3;
        (res_ab, banks, hi_size, queue_ms, reduce_ms)
    };

    crate::scratch::give_f128(c_fold4_pair_mask_tables);
    let t_finish = std::time::Instant::now();
    let (res_c_s, s_hat_v_c, quad_c, fold4_c) = finish_c_fold4_banks(&banks);
    let res_c_lifted = ntt_extend_f128_vec_ghash(&res_c_s, inv_table);
    let finish_ms = t_finish.elapsed().as_secs_f64() * 1e3;
    if profile {
        use std::sync::atomic::Ordering;
        let ns_to_ms = |ns: u64| ns as f64 / 1e6;
        eprintln!(
            "[fold4-c-timing] n_hi={} q_partition={} pair4={} hi_chunks={} queue_jobs={} lo_slots={} lo_pairs={} eq_setup={:.3}ms c_pair_table={:.3}ms queue_wall={:.3}ms ab_cpu={:.3}ms c_cpu={:.3}ms high_cpu={:.3}ms reduce={:.3}ms finish={:.3}ms total={:.3}ms",
            fold4_n_hi,
            q_partition,
            pair4 && big_lo_size >= 4,
            hi_size,
            queue_jobs,
            big_lo_size,
            big_lo_size.div_ceil(2),
            eq_setup_ms,
            c_table_ms,
            queue_ms,
            ns_to_ms(timing_cpu_ns[0].load(Ordering::Relaxed)),
            ns_to_ms(timing_cpu_ns[1].load(Ordering::Relaxed)),
            ns_to_ms(timing_cpu_ns[2].load(Ordering::Relaxed)),
            reduce_ms,
            finish_ms,
            t_total.elapsed().as_secs_f64() * 1e3,
        );
    }
    (res_ab.to_vec(), res_c_lifted, s_hat_v_c, quad_c, fold4_c)
}

/// Serial reference — same I/O as [`round1_shift_reduce_extract_c_packed`],
/// no rayon. Kept under `#[cfg(test)]` as the cross-check oracle for the
/// parallel version: future "optimizations" to the parallel path must still
/// produce identical output to this straight-line loop.
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

    /// **Soundness assumption.** Zerocheck and the Ligerito PCS opening at
    /// L0 both depend on the seven "friendly" constants — three small
    /// (`φ_8(SMALL_CHAL_F8[k])`, k ∈ 0..3) and four medium
    /// (`γ^{2^i}/(1+γ^{2^i})`, i ∈ 0..4) — being **F₂-linearly independent**
    /// in F₁₂₈.
    ///
    /// Zerocheck needs this so that the prover's URM message can't be
    /// trivially canceled by a malicious witness aligned with the friendly
    /// subspace. Ligerito's L0 list-collapse argument (which leans on the
    /// zerocheck `(r, v)` claim as an OOD-equivalent) also depends on it
    /// — see the soundness writeup. If any subset of these seven values is
    /// F₂-dependent, the SZ bound `(m−7)/|F|` for collisions between
    /// distinct candidate codewords' MLEs at `r` no longer holds, and a
    /// cheating prover could engineer their witness so two candidates'
    /// MLEs agree at the friendly point with probability 1.
    ///
    /// The check: form the 7×128 binary matrix whose rows are the bit
    /// representations of the seven constants, Gauss-eliminate over F₂,
    /// assert rank = 7.
    #[test]
    fn friendly_challenges_f2_independent() {
        // Pack each F₁₂₈ element into a u128 (lo, hi → 128 bits).
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

        // Row-reduce over F₂. For each column from MSB to LSB, find a row
        // with that bit set (a pivot), swap it into place, and XOR it into
        // every other row to clear that column. Final rank = number of
        // pivots placed.
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

    /// Build the full `r` vector with the protocol-fixed constants in the
    /// small/medium slots. Only `r[k_skip + N_INNER..]` is the actual
    /// randomness fed to the optimized URM.
    fn build_protocol_r(m: usize, outer: &[F128]) -> Vec<F128> {
        assert_eq!(outer.len(), m - K_SKIP - N_INNER);
        let mut r = vec![F128::ZERO; m];
        // r[0..K_SKIP]: not used by either function — can be anything.
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

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn ranked_ab_pre_fast_policy_hoist_selector_is_exact() {
        let selected = |m, k_skip, n_chunks, k_log, useful, main, helper, prepared, enabled| {
            ranked_ab_pre_fast_policy_hoist_shape(
                m, k_skip, n_chunks, k_log, useful, main, helper, prepared, enabled,
            )
        };
        let ranked = (32, K_SKIP, 1 << 19, 14, 15_409, 10, 4, true, true);
        assert!(selected(
            ranked.0, ranked.1, ranked.2, ranked.3, ranked.4, ranked.5, ranked.6, ranked.7,
            ranked.8,
        ));

        assert!(!selected(
            31,
            K_SKIP,
            1 << 19,
            14,
            15_409,
            10,
            4,
            true,
            true
        ));
        assert!(!selected(
            32,
            K_SKIP + 1,
            1 << 19,
            14,
            15_409,
            10,
            4,
            true,
            true
        ));
        assert!(!selected(
            32,
            K_SKIP,
            (1 << 19) - 1,
            14,
            15_409,
            10,
            4,
            true,
            true
        ));
        assert!(!selected(
            32,
            K_SKIP,
            1 << 19,
            15,
            15_409,
            10,
            4,
            true,
            true
        ));
        assert!(!selected(
            32,
            K_SKIP,
            1 << 19,
            14,
            15_408,
            10,
            4,
            true,
            true
        ));
        assert!(!selected(32, K_SKIP, 1 << 19, 14, 15_409, 9, 4, true, true));
        assert!(!selected(
            32,
            K_SKIP,
            1 << 19,
            14,
            15_409,
            10,
            3,
            true,
            true
        ));
        assert!(!selected(
            32,
            K_SKIP,
            1 << 19,
            14,
            15_409,
            10,
            4,
            false,
            true
        ));
        assert!(!selected(
            32,
            K_SKIP,
            1 << 19,
            14,
            15_409,
            10,
            4,
            true,
            false
        ));
    }

    #[test]
    fn force_direct_one_chunk_matches_cached_store() {
        use crate::zerocheck::univariate_skip::pack_bits;

        const M: usize = 13;
        const OUTER_BYTES: usize = (1 << N_MEDIUM) * 64;
        let mut rng = Rng::new(0xF0CE_D1EC7);
        let a_packed = pack_bits(&rng.bits(1 << M));
        let b_packed = pack_bits(&rng.bits(1 << M));
        let inv_table = make_inv_table();
        let (within_outer_mask, b_med_counts) = build_b_med_counts(&PaddingSpec::dense(M));

        let mut cached = [0u8; OUTER_BYTES];
        let mut cached_a_col = [F8::ZERO; ELL];
        let mut cached_b_col = [F8::ZERO; ELL];
        precompute_ab_one_chunk::<{ kernels::AB_FAST_POLICY_PROCESS }, false>(
            &a_packed,
            &b_packed,
            &inv_table,
            within_outer_mask,
            &b_med_counts,
            false,
            None,
            false,
            false,
            0,
            &mut cached,
            &mut cached_a_col,
            &mut cached_b_col,
        );

        let mut direct = [0u8; OUTER_BYTES];
        let mut direct_a_col = [F8::ZERO; ELL];
        let mut direct_b_col = [F8::ZERO; ELL];
        precompute_ab_one_chunk::<{ kernels::AB_FAST_POLICY_FORCE_FAST }, true>(
            &a_packed,
            &b_packed,
            &inv_table,
            within_outer_mask,
            &b_med_counts,
            false,
            None,
            true,
            false,
            0,
            &mut direct,
            &mut direct_a_col,
            &mut direct_b_col,
        );

        assert_eq!(direct, cached);
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

    /// **The defining cross-check**: `C_s · (opt_AB + opt_C) == naive_AB + naive_C`,
    /// element-wise on Λ. Verifies all three optimization layers compose
    /// correctly — geometric small eq, geometric medium eq, and the D⁻¹
    /// pre-scaling.
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

            // Combined: C_s · (opt_AB + opt_C) == naive_AB + naive_C
            for i in 0..ELL {
                let lhs = naive_ab[i] + naive_c[i];
                let rhs = c_s * (opt_ab[i] + opt_c[i]);
                assert_eq!(
                    lhs, rhs,
                    "combined mismatch at m={m}, i={i}:\n  naive={lhs:?}\n  C_s·opt={rhs:?}"
                );
            }

            // Stronger: the AB and C pieces match independently (the AB-only
            // shift_reduce and the C bit_transpose both drop the same C_s).
            for i in 0..ELL {
                assert_eq!(naive_ab[i], c_s * opt_ab[i], "AB mismatch at i={i}");
                assert_eq!(naive_c[i], c_s * opt_c[i], "C mismatch at i={i}");
            }
        }
    }

    #[test]
    fn small_and_medium_challenges_sanity() {
        // Reach into the constants and verify their structural identities.
        // Medium: β_i · (1 + γ^{2^{i-1}}) == γ^{2^{i-1}}.
        let med = medium_challenges_ghash();
        let powers = [1u64 << 1, 1u64 << 2, 1u64 << 4, 1u64 << 8];
        for (i, &p) in powers.iter().enumerate() {
            let g = F128 { lo: p, hi: 0 };
            assert_eq!(med[i] * (F128::ONE + g), g, "β_{i} identity");
        }

        // D · D_inv == 1.
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

        // At small m the parallel overhead dominates, but the *output* must
        // still match the serial version bit-for-bit. F128 XOR-sum reduction
        // is commutative + associative, so any thread-scheduling order yields
        // the same result.
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

    /// **Padding skip is byte-identical to the dense path.** On a witness
    /// where bits `[useful_bits, 2^k_log)` of every block are honestly zero,
    /// the padded URM must produce the exact same `(round1_ab, round1_c)`
    /// vectors as the dense URM — every chunk we skip would have contributed
    /// a literal zero to the dense sum (the convert table maps φ_8(0) = 0).
    ///
    /// Covers the three hash padding shapes:
    ///   - BLAKE3: k_log=14, useful=15409 → b_med_counts ≈ [16, 15]
    ///   - SHA-2:  k_log=15, useful=31401 → b_med_counts ≈ [16, 16, 16, 14]
    ///   - Keccak: k_log=16, useful=42560 → b_med_counts = [16, 16, 16, 16, 16, 4, 0, 0]
    ///     (this is the only shape that exercises the full-skip case.)
    #[test]
    fn padded_matches_dense_with_zero_padding() {
        use crate::zerocheck::PaddingSpec;
        use crate::zerocheck::univariate_skip::pack_bits;

        // (k_log, useful_bits, n_blocks_log) — pick n_blocks_log so
        // m = k_log + n_blocks_log is small enough to keep the test fast
        // while still exercising the kernel's parallel + boundary paths.
        let cases = [
            (14usize, 15_409usize, 0usize), // BLAKE3, m=14
            (15, 31_401, 0),                // SHA-2,  m=15
            (16, 42_560, 0),                // Keccak, m=16
            (16, 42_560, 3),                // Keccak, m=19 (multiple hashes)
        ];

        for (k_log, useful_bits, n_blocks_log) in cases {
            let m = k_log + n_blocks_log;
            assert!(m >= K_SKIP + N_INNER);

            let mut rng = Rng::new(0xBEEF_DEAD_u64.wrapping_add((k_log * 31 + m) as u64));
            let n_blocks = 1usize << n_blocks_log;
            let total_bits = 1usize << m;
            let block_size = 1usize << k_log;

            // Random witness, but force bits [useful_bits, 2^k_log) of every
            // block to zero (mirrors the hash-module witness layout).
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

    /// **Requirement C-QUAD.** The four-bank statistic the capture ships for the
    /// c-claim must collapse, under the *protocol* low point
    /// `suffix_C[..2] = [φ₈(0x53), φ₈(0xB5)]` that ring-switch builds on intake,
    /// to exactly the wire `s_hat_v_c` the incumbent path observes. Checked on
    /// both entry points and on the padded (b_med-skipping) shapes, since a
    /// quad that treated padding differently would be a different object.
    #[test]
    fn quad_collapses_to_wire_s_hat_v_c() {
        use crate::pcs::ring_switch::collapse_s_hat_v_quad;
        use crate::zerocheck::PaddingSpec;
        use crate::zerocheck::univariate_skip::pack_bits;

        // Exactly how `prove_batched_padded_with_precomputed` derives the low
        // point: `dense_suffixes[1][..2]` = `r_rest[1..3]` = the second and
        // third protocol small challenges.
        let small = small_challenges_ghash();
        let low_point = [small[1], small[2]];

        // (k_log, useful_bits, n_blocks_log); `None` = dense.
        let cases: [(usize, Option<(usize, usize)>); 5] = [
            (13, None),
            (15, None),
            (14, Some((14, 15_409))), // BLAKE3 block shape (one b_med window skipped)
            (17, Some((14, 15_409))), // …across several blocks, both eq halves live
            (15, Some((15, 31_401))), // SHA-2 block shape
        ];

        for (m, padded) in cases {
            let mut rng = Rng::new(0x9AD_C011_u64.wrapping_add(m as u64));
            let total_bits = 1usize << m;
            let mut a = rng.bits(total_bits);
            let mut b = rng.bits(total_bits);
            let mut c = rng.bits(total_bits);
            let padding = match padded {
                None => PaddingSpec::dense(m),
                Some((k_log, useful_bits)) => {
                    let block_size = 1usize << k_log;
                    for blk in 0..(total_bits / block_size) {
                        for j in useful_bits..block_size {
                            let idx = blk * block_size + j;
                            a[idx] = false;
                            b[idx] = false;
                            c[idx] = false;
                        }
                    }
                    PaddingSpec {
                        k_log,
                        useful_bits_per_block: useful_bits,
                    }
                }
            };
            let (a_p, b_p, c_p) = (pack_bits(&a), pack_bits(&b), pack_bits(&c));
            let outer = rng.f128_vec(m - K_SKIP - N_INNER);
            let r = build_protocol_r(m, &outer);
            let table = make_inv_table();

            let (fused_ab, fused_c, s_hat_v_c, quad) =
                round1_shift_reduce_extract_c_packed_padded_with_s_hat_v_quad(
                    &a_p, &b_p, &c_p, m, K_SKIP, &r, &table, &padding,
                );
            assert_eq!(quad.len(), 4 * 2 * ELL, "quad length at m={m}");
            assert_eq!(
                collapse_s_hat_v_quad(&quad, &low_point),
                s_hat_v_c,
                "C-QUAD collapse mismatch at m={m}, padded={}",
                padded.is_some()
            );

            // The precomputed-AB entry point (the one the ranked prove runs)
            // must agree on all four outputs.
            let precomputed =
                precompute_round1_ab_inner_packed_padded(&a_p, &b_p, m, K_SKIP, &table, &padding);
            let (pre_ab, pre_c, pre_s_hat_v, pre_quad) =
                round1_shift_reduce_extract_c_packed_padded_with_precomputed_ab(
                    &precomputed,
                    &c_p,
                    m,
                    K_SKIP,
                    &r,
                    &table,
                    &padding,
                );
            assert_eq!(pre_ab, fused_ab, "res_ab mismatch at m={m}");
            assert_eq!(pre_c, fused_c, "res_c_lifted mismatch at m={m}");
            assert_eq!(pre_s_hat_v, s_hat_v_c, "s_hat_v_c mismatch at m={m}");
            assert_eq!(pre_quad, quad, "quad mismatch at m={m}");
        }
    }

    #[test]
    fn ab_precompute_nt_flavor_is_byte_identical() {
        use crate::zerocheck::univariate_skip::pack_bits;
        // Same padded/dense case matrix as the precompute oracle above: the
        // NT drain (stack bounce + stnp) and the incumbent direct kernel
        // write must produce identical storage bytes, including zeroed
        // padding holes.
        let cases: [(usize, Option<(usize, usize)>); 3] = [
            (13, None),
            (14, Some((14, 15_409))), // BLAKE3 block shape
            (17, Some((14, 15_409))),
        ];
        for (m, padded) in cases {
            let mut rng = Rng::new(0xAB_57_0F14_u64 ^ (m as u64));
            let total_bits = 1usize << m;
            let mut a = rng.bits(total_bits);
            let mut b = rng.bits(total_bits);
            let padding = match padded {
                None => PaddingSpec::dense(m),
                Some((k_log, useful_bits)) => {
                    let block_size = 1usize << k_log;
                    for blk in 0..(total_bits / block_size) {
                        for j in useful_bits..block_size {
                            let idx = blk * block_size + j;
                            a[idx] = false;
                            b[idx] = false;
                        }
                    }
                    PaddingSpec {
                        k_log,
                        useful_bits_per_block: useful_bits,
                    }
                }
            };
            let (a_p, b_p) = (pack_bits(&a), pack_bits(&b));
            let table = make_inv_table();
            // Store-flavor equivalence is orthogonal to the QS3 tail skip; run
            // both flavors with `compact = false` so the dead tail is zeroed
            // deterministically and the full-buffer comparison stays valid.
            let plain = precompute_round1_ab_inner_packed_padded_with_flavor(
                &a_p, &b_p, m, K_SKIP, &table, &padding, false, false,
            );
            let nt = precompute_round1_ab_inner_packed_padded_with_flavor(
                &a_p, &b_p, m, K_SKIP, &table, &padding, true, false,
            );
            assert_eq!(
                plain.as_bytes(),
                nt.as_bytes(),
                "store flavor changed bytes at m={m}, padded={}",
                padded.is_some()
            );
        }
    }

    /// QS3 compacted store: skipping the dead skipped-`b_med` tail rows must
    /// leave every LIVE byte (rows `[0, n_b_med)` of each `x_outer` chunk)
    /// byte-identical to the incumbent zero-filled buffer. Only the tail
    /// bytes — which no consumer ever reads — may differ. Covers the dense
    /// and BLAKE3-padded shapes, both store flavors.
    #[test]
    fn ab_precompute_compact_store_preserves_live_region() {
        use crate::zerocheck::univariate_skip::pack_bits;
        const OUTER_BYTES: usize = (1 << N_MEDIUM) * 64;
        let cases: [(usize, Option<(usize, usize)>); 3] = [
            (13, None),
            (14, Some((14, 15_409))), // BLAKE3 block shape
            (17, Some((14, 15_409))),
        ];
        for (m, padded) in cases {
            let mut rng = Rng::new(0xC0_9AC7_00u64 ^ (m as u64));
            let total_bits = 1usize << m;
            let mut a = rng.bits(total_bits);
            let mut b = rng.bits(total_bits);
            let padding = match padded {
                None => PaddingSpec::dense(m),
                Some((k_log, useful_bits)) => {
                    let block_size = 1usize << k_log;
                    for blk in 0..(total_bits / block_size) {
                        for j in useful_bits..block_size {
                            let idx = blk * block_size + j;
                            a[idx] = false;
                            b[idx] = false;
                        }
                    }
                    PaddingSpec {
                        k_log,
                        useful_bits_per_block: useful_bits,
                    }
                }
            };
            let (a_p, b_p) = (pack_bits(&a), pack_bits(&b));
            let table = make_inv_table();
            let (within_outer_mask, b_med_counts) = build_b_med_counts(&padding);
            for nt in [false, true] {
                let filled = precompute_round1_ab_inner_packed_padded_with_flavor(
                    &a_p, &b_p, m, K_SKIP, &table, &padding, nt, false,
                );
                let compact = precompute_round1_ab_inner_packed_padded_with_flavor(
                    &a_p, &b_p, m, K_SKIP, &table, &padding, nt, true,
                );
                let filled = filled.as_bytes();
                let compact = compact.as_bytes();
                assert_eq!(filled.len(), compact.len());
                let n_outer = filled.len() / OUTER_BYTES;
                for x_outer in 0..n_outer {
                    let n_b_med = b_med_counts[x_outer & within_outer_mask] as usize;
                    let base = x_outer * OUTER_BYTES;
                    let live = n_b_med * 64;
                    assert_eq!(
                        filled[base..base + live],
                        compact[base..base + live],
                        "compacted store changed a live byte at m={m}, nt={nt}, \
                         x_outer={x_outer}, n_b_med={n_b_med}"
                    );
                }
            }
        }
    }

    /// Owned-kernel interleaved A/B of the store flavor at the ranked
    /// geometry (m=32, BLAKE3 padding). Ignored: ~1.5 GiB and seconds of
    /// wall; run explicitly with `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn ab_precompute_nt_bench() {
        use crate::zerocheck::univariate_skip::pack_bits;
        let m = 32usize;
        let total_bits = 1usize << m;
        let padding = PaddingSpec {
            k_log: 14,
            useful_bits_per_block: 15_409,
        };
        let mut rng = Rng::new(0xBE9C4);
        let mut a = rng.bits(total_bits);
        let mut b = rng.bits(total_bits);
        let block_size = 1usize << padding.k_log;
        for blk in 0..(total_bits / block_size) {
            for j in padding.useful_bits_per_block..block_size {
                a[blk * block_size + j] = false;
                b[blk * block_size + j] = false;
            }
        }
        let (a_p, b_p) = (pack_bits(&a), pack_bits(&b));
        drop(a);
        drop(b);
        let table = make_inv_table();
        // Warmup one of each, then interleave measured reps. `compact = true`
        // exercises the shipped QS3 path (dead tail rows never stored).
        for nt in [false, true] {
            let _ = precompute_round1_ab_inner_packed_padded_with_flavor(
                &a_p, &b_p, m, K_SKIP, &table, &padding, nt, true,
            );
        }
        for rep in 0..4 {
            for nt in [rep % 2 == 1, rep % 2 == 0] {
                let t0 = std::time::Instant::now();
                let out = precompute_round1_ab_inner_packed_padded_with_flavor(
                    &a_p, &b_p, m, K_SKIP, &table, &padding, nt, true,
                );
                let ms = t0.elapsed().as_secs_f64() * 1e3;
                println!("rep={rep} nt={nt}: {ms:.2} ms");
                drop(out);
            }
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
            // SAFETY: on aarch64.
            unsafe { bit_transpose_64bytes_neon(&input, &mut out_neon) };
            assert_eq!(out_scalar, out_neon, "bit_transpose disagreement");
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_fused_inner_matches_scalar_inner() {
        // The new register-fused NEON kernel — verify against the same scalar
        // oracle as the intermediate one.
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
            shift_reduce_inner_ab_scalar(
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
                false,
            );
            // The NT drain must be byte-identical to the cached store.
            let mut out_fused_nt = [0u8; 64];
            shift_reduce_inner_ab_fused_neon(
                &a_packed,
                &b_packed,
                &table,
                chunk_byte_base,
                b_med,
                &mut out_fused_nt,
                true,
            );
            assert_eq!(out_fused, out_fused_nt, "nt store flavor must match");
            assert_eq!(
                out_scalar, out_fused,
                "fused-neon disagrees with scalar at (base={chunk_byte_base}, b_med={b_med})"
            );
        }
    }

    /// The Horner kernel folds the bit leaving lane bit 15 back in as
    /// `x^16 mod p`. Pin that constant against the scalar field arithmetic so
    /// a wrong literal can never silently pass the vector oracle.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn horner_carry_constant_matches_field() {
        let x8 = F8(crate::field::gf2_8::gf8_reduce(1u16 << 8));
        assert_eq!(x8, F8(0x1b), "x^8 mod p");
        let x16 = x8 * x8;
        assert_eq!(
            u16::from(x16.0),
            kernels::aarch64::HORNER_CARRY_X16,
            "x^16 mod p must equal the Horner carry weight"
        );
    }

    /// The `x^4`-scaled table images must be exactly the elementwise field
    /// product of the plain images with `x^4`; that identity is what makes
    /// `x^4 · T(w)` a pure table swap.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn scaled_x4_table_images_are_x4_times_plain() {
        let table = make_inv_table();
        let len = 256 * table.ell;
        let x4 = F8(1u8 << 4);
        for (plain, scaled) in [
            (table.data_ptr(), table.scaled_x4_data_ptr()),
            (
                table.half_swapped_data_ptr(),
                table.scaled_x4_half_swapped_data_ptr(),
            ) as (*const u8, *const u8),
        ] {
            for i in 0..len {
                // SAFETY: both images are `256 * ell` bytes by construction.
                let (p, s) = unsafe { (*plain.add(i), *scaled.add(i)) };
                assert_eq!(F8(p) * x4, F8(s), "scaled table image mismatch at {i}");
            }
        }
    }

    /// Byte-exact oracle for the Horner / scaled-table kernel against both the
    /// incumbent NEON kernel it replaces and the scalar reference, over random
    /// witnesses, every 64-byte-block byte alignment, every `b_med` slot, and
    /// planted degenerate rows (all-zero, all-ones, top-bit-only) that drive
    /// the Horner carry fold on every step.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_h4_shift_reduce_is_bit_exact_with_incumbent() {
        use kernels::aarch64::shift_reduce_inner_ab_fused_neon_h4;

        let table = make_inv_table();
        let mut rng = Rng::new(0x484F_524E_4552_5F34);

        const N: usize = 8192;
        let mut a_packed = vec![0u8; N];
        let mut b_packed = vec![0u8; N];
        for i in 0..N {
            a_packed[i] = (rng.next_u64() & 0xff) as u8;
            b_packed[i] = (rng.next_u64() >> 13 & 0xff) as u8;
        }
        // Degenerate 64-byte K-blocks: zero rows, const-one rows, and rows
        // whose transform maximizes the 15-bit product degree.
        a_packed[0..64].fill(0x00);
        b_packed[0..64].fill(0xff);
        a_packed[64..128].fill(0xff);
        b_packed[64..128].fill(0x00);
        a_packed[128..192].fill(0xff);
        b_packed[128..192].fill(0xff);
        a_packed[192..256].fill(0x80);
        b_packed[192..256].fill(0x01);

        let mut a_col = vec![F8::ZERO; ELL];
        let mut b_col = vec![F8::ZERO; ELL];
        let mut cases = 0usize;

        let check =
            |chunk_byte_base: usize, b_med: usize, a_col: &mut Vec<F8>, b_col: &mut Vec<F8>| {
                let base = chunk_byte_base + b_med * N_CHUNKS * 8;
                if base + 8 * N_CHUNKS > N {
                    return false;
                }
                let mut out_scalar = [0u8; 64];
                let mut out_incumbent = [0u8; 64];
                let mut out_h4 = [0u8; 64];
                shift_reduce_inner_ab_scalar(
                    &a_packed,
                    &b_packed,
                    &table,
                    chunk_byte_base,
                    b_med,
                    &mut out_scalar,
                    a_col,
                    b_col,
                );
                shift_reduce_inner_ab_fused_neon(
                    &a_packed,
                    &b_packed,
                    &table,
                    chunk_byte_base,
                    b_med,
                    &mut out_incumbent,
                    false,
                );
                shift_reduce_inner_ab_fused_neon_h4(
                    &a_packed,
                    &b_packed,
                    &table,
                    chunk_byte_base,
                    b_med,
                    &mut out_h4,
                    false,
                );
                assert_eq!(
                    out_incumbent, out_scalar,
                    "incumbent oracle drift at (base={chunk_byte_base}, b_med={b_med})"
                );
                assert_eq!(
                    out_h4, out_incumbent,
                    "h4 kernel differs from incumbent at (base={chunk_byte_base}, b_med={b_med})"
                );
                true
            };

        // Every byte alignment of the 64-byte K-block start.
        for chunk_byte_base in 0..64usize {
            for b_med in 0..8usize {
                if check(chunk_byte_base, b_med, &mut a_col, &mut b_col) {
                    cases += 1;
                }
            }
        }
        // Every `b_med` slot across the whole buffer, at production alignment.
        for b_med in 0.. {
            if !check(0, b_med, &mut a_col, &mut b_col) {
                break;
            }
            cases += 1;
        }
        assert!(cases > 600, "oracle coverage too thin: {cases} cases");
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn static_b_context_gate_respects_layout_and_legacy_policy() {
        let table = make_inv_table();
        assert!(
            kernels::aarch64::prepare_static_b_context_with_policy(&table, false, false, false)
                .is_none(),
            "non-BLAKE3 layouts must not prepare the generated static-B plan"
        );
        assert!(
            kernels::aarch64::prepare_static_b_context_with_policy(&table, true, true, false)
                .is_none(),
            "the legacy control must retain the generic fallback"
        );
        assert!(
            matches!(
                kernels::aarch64::prepare_static_b_context_with_policy(&table, true, false, true),
                Some(kernels::aarch64::StaticBContext::LegacyPerCall)
            ),
            "the context control must retain per-call lookups and static-B"
        );
        assert!(
            matches!(
                kernels::aarch64::prepare_static_b_context_with_policy(&table, true, false, false),
                Some(kernels::aarch64::StaticBContext::Prepared { .. })
            ),
            "the checked static-B plan should be prepared for its exact layout"
        );
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn prepared_static_b_context_matches_scalar_and_guard_fallback() {
        // The generated debug kernel keeps every match arm's SIMD locals in
        // the frame; give this direct-kernel oracle more than libtest's small
        // worker stack. Release builds reuse the slots aggressively.
        std::thread::Builder::new()
            .stack_size(16 << 20)
            .spawn(|| {
                let mut rng = Rng::new(0xB57A_71C0);
                let table = make_inv_table();
                let a_packed = pack_bits(&rng.bits(1 << 14));
                let mut b_packed = pack_bits(&rng.bits(1 << 14));
                let w = 0usize;
                let b_med = 3usize;
                let blk = w * (1 << N_MEDIUM) + b_med;
                let byte_base_b = b_med * N_CHUNKS * 8;

                // Force every generated static position to its expected value while
                // retaining random bytes in the dynamic positions.
                for k in 0..N_CHUNKS {
                    let off = byte_base_b + k * N_CHUNKS;
                    let (mask, expected) = kernels::aarch64::BSTATIC_MASKS[blk][k];
                    let word = u64::from_le_bytes(b_packed[off..off + 8].try_into().unwrap());
                    let word = (word & !mask) | expected;
                    b_packed[off..off + 8].copy_from_slice(&word.to_le_bytes());
                }

                let context = kernels::aarch64::prepare_static_b_context_with_policy(
                    &table, true, false, false,
                )
                .expect("enabled static-B context");
                let legacy_context = kernels::aarch64::prepare_static_b_context_with_policy(
                    &table, true, false, true,
                )
                .expect("per-call static-B control");
                let run_scalar = |b: &[u8]| {
                    let mut out = [0u8; 64];
                    let mut a_col = [F8::ZERO; ELL];
                    let mut b_col = [F8::ZERO; ELL];
                    shift_reduce_inner_ab_scalar(
                        &a_packed, b, &table, 0, b_med, &mut out, &mut a_col, &mut b_col,
                    );
                    out
                };
                let run_context = |b: &[u8], context| {
                    let mut out = [0u8; 64];
                    shift_reduce_inner_ab_fused_neon_checked(
                        &a_packed,
                        b,
                        &table,
                        0,
                        b_med,
                        &mut out,
                        false,
                        false,
                        0,
                        w,
                        Some(context),
                        false,
                    );
                    out
                };

                assert_eq!(run_context(&b_packed, context), run_scalar(&b_packed));
                assert_eq!(
                    run_context(&b_packed, legacy_context),
                    run_scalar(&b_packed)
                );

                // Break one guarded static bit. The prepared plan must still take its
                // row-local generic fallback and remain identical to the scalar oracle.
                let guarded_k = 1usize;
                let (guard_mask, _) = kernels::aarch64::BSTATIC_MASKS[blk][guarded_k];
                assert_ne!(guard_mask, 0);
                let off = byte_base_b + guarded_k * N_CHUNKS;
                let mut word = u64::from_le_bytes(b_packed[off..off + 8].try_into().unwrap());
                word ^= guard_mask & guard_mask.wrapping_neg();
                b_packed[off..off + 8].copy_from_slice(&word.to_le_bytes());
                assert_eq!(run_context(&b_packed, context), run_scalar(&b_packed));
                assert_eq!(
                    run_context(&b_packed, legacy_context),
                    run_scalar(&b_packed)
                );
            })
            .expect("spawn static-B oracle")
            .join()
            .expect("static-B oracle thread");
    }

    /// The `x^2` nibble tables used by the low-instruction static-B rows must
    /// reproduce scalar `F8` multiplication for every byte.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn mul_x2_nibble_tables_match_field() {
        let x2 = F8(1u8 << 2);
        for b in 0..=255u8 {
            let got = F8(kernels::aarch64::MUL_X2_LO[(b & 0x0f) as usize]
                ^ kernels::aarch64::MUL_X2_HI[(b >> 4) as usize]);
            assert_eq!(got, F8(b) * x2, "x^2 nibble split wrong for {b:#04x}");
        }
    }

    /// Byte-exact oracle for the low-instruction static-B kernel. Covers every
    /// live `(window, b_med)` arm, and for each of them: raw random B (guards
    /// broken, every row takes the generic path), fully-satisfied guards
    /// (every row takes the static path), and each K's guard broken in turn
    /// (mixed static/generic rows within one block).
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn bstatic_fast_kernel_is_bit_exact_with_incumbent() {
        // Same stack headroom rationale as the guard-fallback oracle above:
        // the generated 31-arm kernel is frame-heavy in debug builds.
        std::thread::Builder::new()
            .stack_size(64 << 20)
            .spawn(|| {
                let table = make_inv_table();
                let mut rng = Rng::new(0xB57A_71C0_F457_0001);
                let context = kernels::aarch64::prepare_static_b_context_with_policy(
                    &table, true, false, false,
                )
                .expect("enabled static-B context");

                const N: usize = 4096;
                let a_packed: Vec<u8> = (0..N).map(|_| (rng.next_u64() & 0xff) as u8).collect();
                let base_b: Vec<u8> = (0..N)
                    .map(|_| (rng.next_u64() >> 17 & 0xff) as u8)
                    .collect();

                let mut a_col = vec![F8::ZERO; ELL];
                let mut b_col = vec![F8::ZERO; ELL];
                let mut arms = 0usize;

                for w in 0..2usize {
                    let n_b_med = if w == 0 { 16 } else { 15 };
                    for b_med in 0..n_b_med {
                        let blk = w * (1 << N_MEDIUM) + b_med;
                        let byte_base_b = b_med * N_CHUNKS * 8;
                        for variant in 0..10usize {
                            let mut b = base_b.clone();
                            if variant >= 1 {
                                for k in 0..N_CHUNKS {
                                    let off = byte_base_b + k * N_CHUNKS;
                                    let (mask, expected) = kernels::aarch64::BSTATIC_MASKS[blk][k];
                                    let word =
                                        u64::from_le_bytes(b[off..off + 8].try_into().unwrap());
                                    let mut word = (word & !mask) | expected;
                                    if variant >= 2 && variant - 2 == k && mask != 0 {
                                        // Flip the lowest guarded bit so this row
                                        // alone falls back to the generic path.
                                        word ^= mask & mask.wrapping_neg();
                                    }
                                    b[off..off + 8].copy_from_slice(&word.to_le_bytes());
                                }
                            }

                            let mut want = [0u8; 64];
                            shift_reduce_inner_ab_scalar(
                                &a_packed, &b, &table, 0, b_med, &mut want, &mut a_col, &mut b_col,
                            );
                            let mut slow = [0u8; 64];
                            assert!(
                                kernels::aarch64::shift_reduce_inner_ab_bstatic::<false>(
                                    &a_packed, &b, &table, 0, b_med, w, context, &mut slow, false,
                                ),
                                "arm (w={w}, b_med={b_med}) must be live"
                            );
                            let mut fast = [0u8; 64];
                            assert!(
                                kernels::aarch64::shift_reduce_inner_ab_bstatic::<true>(
                                    &a_packed, &b, &table, 0, b_med, w, context, &mut fast, true,
                                ),
                                "arm (w={w}, b_med={b_med}) must be live"
                            );
                            assert_eq!(
                                slow, want,
                                "incumbent bstatic drift (w={w}, b_med={b_med}, variant={variant})"
                            );
                            assert_eq!(
                                fast, slow,
                                "fast bstatic differs (w={w}, b_med={b_med}, variant={variant})"
                            );
                        }
                        arms += 1;
                    }
                }
                assert_eq!(arms, 31, "every generated static-B arm must be covered");
            })
            .expect("spawn bstatic oracle")
            .join()
            .expect("bstatic oracle thread");
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_mixed_const_one_inner_matches_scalar_inner() {
        let mut rng = Rng::new(0xC057_01E5);
        let m = 14;
        let table = make_inv_table();
        let a_bits = rng.bits(1 << m);
        let b_bits = rng.bits(1 << m);
        let a_packed = super::super::univariate_skip::pack_bits(&a_bits);
        let b_packed = super::super::univariate_skip::pack_bits(&b_bits);
        let prepared_context =
            kernels::aarch64::prepare_static_b_context_with_policy(&table, true, false, false)
                .expect("prepared static-b context");

        for (mask, static_a_k1, prepared) in [
            (0x03u8, false, false),
            (0x03, true, false),
            (0x03, true, true),
            (0xf0, false, false),
        ] {
            let mut a_mixed = a_packed.clone();
            if static_a_k1 {
                let a_k0 = u64::from_le_bytes(a_mixed[..N_CHUNKS].try_into().unwrap())
                    & !0xffff_fffe_0000_0000;
                a_mixed[..N_CHUNKS].copy_from_slice(&a_k0.to_le_bytes());
                a_mixed[N_CHUNKS..2 * N_CHUNKS]
                    .copy_from_slice(&0x0000_0016_0000_0080u64.to_le_bytes());
            }
            let mut b_mixed = b_packed.clone();
            for k in 0..8 {
                if mask & (1 << k) != 0 {
                    b_mixed[k * N_CHUNKS..(k + 1) * N_CHUNKS].fill(u8::MAX);
                }
            }
            let mut a_col = [F8::ZERO; ELL];
            let mut b_col = [F8::ZERO; ELL];
            let mut want = [0u8; 64];
            let mut got = [0u8; 64];
            shift_reduce_inner_ab_scalar(
                &a_mixed, &b_mixed, &table, 0, 0, &mut want, &mut a_col, &mut b_col,
            );
            shift_reduce_inner_ab_fused_neon_checked(
                &a_mixed,
                &b_mixed,
                &table,
                0,
                0,
                &mut got,
                false,
                false,
                mask,
                usize::MAX,
                prepared.then_some(prepared_context),
                false,
            );
            assert_eq!(
                got, want,
                "mixed const-one mask {mask:#04x}, static_a_k1={static_a_k1}, prepared={prepared}"
            );
        }

        // The ranked (window 0, b_med 2) A K0/K1 words have their top three
        // bytes structurally zero (2048/2048 scored blocks), so the h4
        // dispatch routes both a-side transforms through
        // `apply_word_low5_into_4_regs` (the A_LOW5_K=0x03 arm). Exercise that
        // arm with random top-3-bytes-zero a_k0/a_k1 values against the scalar
        // oracle — the arm must be bit-identical.
        for trial in 0..32 {
            let mut a_mixed = a_packed.clone();
            for k in 0..2 {
                a_mixed[k * N_CHUNKS..(k + 1) * N_CHUNKS]
                    .copy_from_slice(&(rng.next_u64() & 0x0000_00ff_ffff_ffff).to_le_bytes());
            }
            let mut b_mixed = b_packed.clone();
            for k in 0..2 {
                b_mixed[k * N_CHUNKS..(k + 1) * N_CHUNKS].fill(u8::MAX);
            }
            let mut a_col = [F8::ZERO; ELL];
            let mut b_col = [F8::ZERO; ELL];
            let mut want = [0u8; 64];
            let mut got = [0u8; 64];
            shift_reduce_inner_ab_scalar(
                &a_mixed, &b_mixed, &table, 0, 0, &mut want, &mut a_col, &mut b_col,
            );
            shift_reduce_inner_ab_fused_neon_checked(
                &a_mixed,
                &b_mixed,
                &table,
                0,
                0,
                &mut got,
                false,
                false,
                0x03,
                usize::MAX,
                Some(prepared_context),
                false,
            );
            assert_eq!(
                got, want,
                "A_LOW5_K=0x03 h4 arm (trial {trial}) must be bit-identical"
            );
        }
    }

    /// The blk-30 static-B single-K0 fast path (precomputed partial instead
    /// of the eight B table-row gathers) must be byte-identical to the
    /// generic dispatch, and the guard must fall back for any other B word.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn single_k0_static_b_matches_generic() {
        let mut rng = Rng::new(0x51C0_B30);
        let table = make_inv_table();
        let context =
            kernels::aarch64::prepare_static_b_context_with_policy(&table, true, false, false)
                .expect("prepared static-b context");
        let b_med = 14usize;
        let byte_base_b = b_med * N_CHUNKS * 8;
        let len = byte_base_b + 8 * N_CHUNKS;
        for trial in 0..64 {
            let mut a_packed: Vec<u8> = (0..len).map(|_| rng.next_u64() as u8).collect();
            // Exact B plus trial mod 4 = 0 exercises the ranked top-byte-zero
            // A transform; mod 4 = 2 forces its checked full-word fallback.
            if trial % 4 == 0 {
                a_packed[byte_base_b + 7] = 0;
            } else if trial % 4 == 2 {
                a_packed[byte_base_b + 7] |= 1;
            }
            let mut b_packed = vec![0u8; len];
            // K0 = the blk-30 static constant, K1..7 = zero; one arm perturbs
            // K0 so the guard must route to the generic single-K0 kernel.
            let mut k0 = 0x0001_ffff_ffff_ffffu64;
            if trial % 2 == 1 {
                k0 ^= 1 << (trial % 49);
            }
            b_packed[byte_base_b..byte_base_b + 8].copy_from_slice(&k0.to_le_bytes());
            let run = |ctx: Option<kernels::StaticBContext>| {
                let mut out = [0u8; 64];
                kernels::aarch64::shift_reduce_inner_ab_fused_neon_checked(
                    &a_packed, &b_packed, &table, 0, b_med, &mut out, false, true, 0, 1, ctx, false,
                );
                out
            };
            assert_eq!(
                run(Some(context)),
                run(None),
                "static-b K0 word {k0:#018x} must match the generic kernel"
            );
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "gfni"))]
    #[test]
    fn x86_gfni_sse_inner_matches_scalar_inner() {
        // The SSE/GFNI fallback must remain byte-identical to the scalar oracle.
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
            shift_reduce_inner_ab_scalar(
                &a_packed,
                &b_packed,
                &table,
                chunk_byte_base,
                b_med,
                &mut out_scalar,
                &mut a_col,
                &mut b_col,
            );
            // SAFETY: gated on gfni target feature.
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

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    #[test]
    fn x86_gfni_avx512_inner_matches_scalar_inner() {
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
            shift_reduce_inner_ab_scalar(
                &a_packed,
                &b_packed,
                &table,
                chunk_byte_base,
                b_med,
                &mut out_scalar,
                &mut a_col,
                &mut b_col,
            );
            // SAFETY: test is compiled only when all kernel features are active.
            unsafe {
                shift_reduce_inner_ab_x86_avx512(
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

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_inner_matches_scalar_inner() {
        // Pin down the NEON kernel directly: same inputs, same output bytes.
        let mut rng = Rng::new(0x5EED);
        let m = 14;
        let table = make_inv_table();
        let n_chunks = 1 << (K_SKIP / 8); // unused; just sanity
        let _ = n_chunks;
        let a_bits = rng.bits(1 << m);
        let b_bits = rng.bits(1 << m);
        let a_packed = super::super::univariate_skip::pack_bits(&a_bits);
        let b_packed = super::super::univariate_skip::pack_bits(&b_bits);

        let mut a_col = vec![F8::ZERO; ELL];
        let mut b_col = vec![F8::ZERO; ELL];

        // A few representative (chunk_byte_base, b_med) values.
        for &(chunk_byte_base, b_med) in &[(0usize, 0usize), (64, 5), (1024, 7), (4096, 15)] {
            // Guard: don't read past the witness.
            let needed = chunk_byte_base + b_med * N_CHUNKS * 8 + 8 * N_CHUNKS;
            if needed > a_packed.len() {
                continue;
            }
            let mut out_scalar = [0u8; 64];
            let mut out_neon = [0u8; 64];
            shift_reduce_inner_ab_scalar(
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
        // convert[b][v] == γ^b · φ_8(v); check at a handful of (b, v).
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

    /// The two-bank fusion variant produces `(res_ab, res_c_lifted)` that
    /// matches the existing optimized output, AND a `s_hat_v_c` that matches
    /// the scalar-oracle's canonical form.
    #[test]
    fn fusion_matches_existing_and_scalar_oracle() {
        use crate::zerocheck::univariate_skip::round1_extract_c_packed_with_s_hat_v;

        for &m in &[13usize, 14, 15] {
            let mut rng = Rng::new(0xF00D_u64.wrapping_add(m as u64));
            let a = pack_bits(&rng.bits(1 << m));
            let b = pack_bits(&rng.bits(1 << m));
            let c = pack_bits(&rng.bits(1 << m));
            let mut r = vec![F128::ZERO; m];
            // Friendly inner constants must match the optimization's
            // expectations: 3 small + 4 medium ghash.
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

            // Reference 1: existing optimized output (no s_hat_v).
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

            // Reference 2: scalar oracle (canonical s_hat_v_c).
            let (_, _, oracle_s_hat_v) =
                round1_extract_c_packed_with_s_hat_v(&a, &b, &c, m, K_SKIP, &r, &inv_table);

            // System under test.
            let (got_ab, got_c, got_s_hat_v, got_quad) =
                round1_shift_reduce_extract_c_packed_padded_with_s_hat_v_quad(
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

            // Requirement C-QUAD: what ring-switch collapses the quad into on
            // intake must be exactly the wire `s_hat_v_c`.
            assert_eq!(got_quad.len(), 4 * 2 * ELL, "quad length at m={m}");
            assert_eq!(
                crate::pcs::ring_switch::collapse_s_hat_v_quad(
                    &got_quad,
                    &r[K_SKIP + 1..K_SKIP + 3]
                ),
                got_s_hat_v,
                "quad collapse mismatch at m={m}"
            );

            // The generic 8-bank scalar oracle: independent of the α algebra
            // (it divides by the literal eq weights), so it pins the α-free
            // convention the quad is defined in.
            let oracle_quad =
                crate::zerocheck::univariate_skip::round1_extract_c_packed_quad_oracle(
                    &a, &b, &c, m, K_SKIP, &r, &inv_table,
                );
            assert_eq!(
                got_quad, oracle_quad,
                "quad mismatch vs scalar oracle at m={m}"
            );
        }
    }

    /// Splitting the challenge-independent AB transform from the later eq
    /// fold must not change any round-1 wire value or the captured C opening
    /// helper. Cover m=13 through the first dimension that reaches the
    /// heterogeneous queue's 16-chunk engagement threshold, so both unsplit
    /// and split eq-table shapes plus the two-pool schedule are exercised.
    #[test]
    fn precomputed_ab_matches_fused_at_m13_through_m17() {
        for m in 13usize..=17 {
            let mut rng = Rng::new(0xAB00_0000_u64.wrapping_add(m as u64));
            let a = pack_bits(&rng.bits(1 << m));
            let b = pack_bits(&rng.bits(1 << m));
            let c = pack_bits(&rng.bits(1 << m));
            let outer = rng.f128_vec(m - K_SKIP - N_INNER);
            let r = build_protocol_r(m, &outer);
            let inv_table = make_inv_table();
            let padding = PaddingSpec::dense(m);

            let expected = round1_shift_reduce_extract_c_packed_padded_with_s_hat_v_quad(
                &a, &b, &c, m, K_SKIP, &r, &inv_table, &padding,
            );
            let precomputed =
                precompute_round1_ab_inner_packed_padded(&a, &b, m, K_SKIP, &inv_table, &padding);
            assert_eq!(precomputed.len_bytes(), a.len());
            let got = round1_shift_reduce_extract_c_packed_padded_with_precomputed_ab(
                &precomputed,
                &c,
                m,
                K_SKIP,
                &r,
                &inv_table,
                &padding,
            );

            assert_eq!(got, expected, "split round-1 mismatch at m={m}");
        }
    }

    /// The eq_lo fold's whole premise. At the ranked split shape the weight
    /// the incumbent drain multiplies each chunk by must be exactly the
    /// product of the two tensor factors the fold pushes into the pre-scaled
    /// table and the deferred bank fold. Derived through the production
    /// helper, checked against `SplitEqGhash`'s independently built `lo`.
    #[test]
    fn ab_eq_fold_factors_reproduce_split_eq_lo() {
        // Ranked shape: m = 32 leaves 19 outer challenges, of which the
        // Fold4 default keeps n_hi = 7 in the hi half.
        const N_OUTER: usize = 32 - K_SKIP - N_INNER;
        let mut rng = Rng::new(0x5EED_F01D);
        let outer = rng.f128_vec(N_OUTER);
        let eq = SplitEqGhash::with_n_hi(&outer, fold4_n_hi_from_env());
        assert_eq!((eq.n_hi, eq.n_lo), (7, 12), "ranked fold4 lo/hi split");

        let r_lo = &outer[..eq.n_lo];
        let default_bits = AbEqFold::On(None)
            .bank_bits(eq.n_lo)
            .expect("On resolves to a bank width");
        assert_eq!(
            default_bits,
            eq.n_lo - AB_EQ_FOLD_TABLE_BITS,
            "default keeps AB_EQ_FOLD_TABLE_BITS bits in the table index"
        );
        assert_eq!(AbEqFold::Off.bank_bits(eq.n_lo), None, "kill switch");

        for bank_bits in [0, 1, default_bits, eq.n_lo] {
            let (eq_bot, eq_top_scaled) = ab_eq_fold_factors(r_lo, bank_bits);
            assert_eq!(eq_bot.len(), 1 << bank_bits, "bank count at s={bank_bits}");
            assert_eq!(
                eq_top_scaled.len(),
                1 << (eq.n_lo - bank_bits),
                "table count at s={bank_bits}"
            );
            let bank_mask = (1usize << bank_bits) - 1;
            for (x, lo) in eq.lo.iter().enumerate() {
                assert_eq!(
                    *lo * d_inv(),
                    eq_top_scaled[x >> bank_bits] * eq_bot[x & bank_mask],
                    "eq_lo tensor factorization at x={x}, s={bank_bits}"
                );
            }
        }
    }

    /// The gather is an XOR chain over convert rows, so the pre-scaled table
    /// has to keep the convert table's F₂-linearity in the gathered byte (and
    /// its zero row). That is exactly what makes the scaled gather equal the
    /// unscaled gather times `eq_top[w]`, i.e. what lets the multiply leave
    /// the kernel.
    #[test]
    fn ab_eq_fold_tables_stay_linear_in_the_gathered_byte() {
        let mut rng = Rng::new(0x7AB1_E5);
        let scales = rng.f128_vec(4);
        let convert = convert_table();
        let tables = build_ab_eq_fold_tables(&scales, convert);
        assert_eq!(tables.len(), scales.len() * CONVERT_TABLE_SIZE);

        for (w, scale) in scales.iter().enumerate() {
            let table = &tables[w * CONVERT_TABLE_SIZE..(w + 1) * CONVERT_TABLE_SIZE];
            for b_med in 0..1 << N_MEDIUM {
                let base = b_med * 256;
                assert_eq!(table[base], F128::ZERO, "T[{w}] row {b_med} zero index");
                for _ in 0..16 {
                    let left = (rng.next_u64() & 0xff) as usize;
                    let right = (rng.next_u64() & 0xff) as usize;
                    assert_eq!(
                        table[base + (left ^ right)],
                        table[base + left] + table[base + right],
                        "T[{w}] row {b_med} additive at ({left},{right})"
                    );
                    assert_eq!(
                        table[base + left],
                        convert[base + left] * *scale,
                        "T[{w}] row {b_med} scale at {left}"
                    );
                }
            }
        }
        crate::scratch::give_f128(tables);
    }

    /// End-to-end producer gate for the fold: the banked drain must return
    /// the incumbent AB message bit-for-bit at every bank/table split of the
    /// shape, including a padded block whose chunks are skipped entirely.
    #[test]
    fn ab_eq_fold_matches_incumbent_round1_ab() {
        // Match the sibling round-one gates: the padded debug precompute
        // exceeds Rayon's default worker stack on AArch64.
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .stack_size(16 << 20)
            .build()
            .unwrap();
        pool.install(ab_eq_fold_matches_incumbent_round1_ab_inner);
    }

    fn ab_eq_fold_matches_incumbent_round1_ab_inner() {
        const K_LOG: usize = 14;
        // m >= 21 is the first dimension whose Fold4 lo half is non-empty,
        // i.e. the first one where the fold has a split to get wrong.
        for m in 21usize..=23 {
            for useful in [1usize << K_LOG, 15_409, 1 << 13] {
                let padding = PaddingSpec {
                    k_log: K_LOG,
                    useful_bits_per_block: useful,
                };
                let mut rng = Rng::new(0xE9F0_1D00 ^ ((m as u64) << 8) ^ useful as u64);
                let total_bits = 1usize << m;
                let mut a = rng.bits(total_bits);
                let mut b = rng.bits(total_bits);
                if useful < (1usize << K_LOG) {
                    for block in 0..(total_bits >> K_LOG) {
                        for bit in useful..(1usize << K_LOG) {
                            let index = (block << K_LOG) + bit;
                            a[index] = false;
                            b[index] = false;
                        }
                    }
                }
                let a = pack_bits(&a);
                let b = pack_bits(&b);
                let outer = rng.f128_vec(m - K_SKIP - N_INNER);
                let r = build_protocol_r(m, &outer);
                let inv_table = make_inv_table();
                let ab_inner = precompute_round1_ab_inner_packed_padded(
                    &a, &b, m, K_SKIP, &inv_table, &padding,
                );
                let n_lo =
                    SplitEqGhash::with_n_hi(&r[K_SKIP + N_INNER..], fold4_n_hi_from_env()).n_lo;
                assert!(n_lo > 0, "m={m} must exercise a non-trivial lo half");

                let want = round1_shift_reduce_ab_packed_padded_with_precomputed_with_fold(
                    &ab_inner,
                    m,
                    K_SKIP,
                    &r,
                    &padding,
                    AbEqFold::Off,
                );
                for bank_bits in 0..=n_lo {
                    let got = round1_shift_reduce_ab_packed_padded_with_precomputed_with_fold(
                        &ab_inner,
                        m,
                        K_SKIP,
                        &r,
                        &padding,
                        AbEqFold::On(Some(bank_bits)),
                    );
                    assert_eq!(
                        got, want,
                        "folded AB mismatch at m={m}, useful={useful}, s={bank_bits}"
                    );
                }
                let got_default = round1_shift_reduce_ab_packed_padded_with_precomputed_with_fold(
                    &ab_inner,
                    m,
                    K_SKIP,
                    &r,
                    &padding,
                    AbEqFold::On(None),
                );
                assert_eq!(
                    got_default, want,
                    "default-split AB mismatch at m={m}, useful={useful}"
                );
                // Whatever the environment resolves to, the shipped entry
                // point must land on the same bytes.
                let got_entry = round1_shift_reduce_ab_packed_padded_with_precomputed(
                    &ab_inner, m, K_SKIP, &r, &padding,
                );
                assert_eq!(
                    got_entry, want,
                    "entry-point AB mismatch at m={m}, useful={useful}"
                );
            }
        }
    }

    /// End-to-end producer gate for the opt-in entry point.  It must preserve
    /// every incumbent round-1/capture output and its extra 16x128 tensor must
    /// independently collapse to the same canonical C vector.
    #[test]
    fn precomputed_ab_fold4_c_capture_matches_incumbent() {
        // The incumbent padded debug kernel exceeds Rayon's default worker
        // stack on AArch64. Keep that test-only frame expansion isolated from
        // production workers while still exercising the complete cross-path
        // oracle below.
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .stack_size(16 << 20)
            .build()
            .unwrap();
        pool.install(precomputed_ab_fold4_c_capture_matches_incumbent_inner);
    }

    fn precomputed_ab_fold4_c_capture_matches_incumbent_inner() {
        let cases: [(usize, PaddingSpec); 5] = [
            (13, PaddingSpec::dense(13)),
            (15, PaddingSpec::dense(15)),
            // Eight sampled outer coordinates make the Fold4-specific n_hi=7
            // split differ from the incumbent split, proving regrouping exact.
            (21, PaddingSpec::dense(21)),
            (
                14,
                PaddingSpec {
                    k_log: 14,
                    useful_bits_per_block: 15_409,
                },
            ),
            (
                17,
                PaddingSpec {
                    k_log: 14,
                    useful_bits_per_block: 15_409,
                },
            ),
        ];

        for (m, padding) in cases {
            let mut rng = Rng::new(0xF01D_4C00_u64.wrapping_add(m as u64));
            let total_bits = 1usize << m;
            let mut a = rng.bits(total_bits);
            let mut b = rng.bits(total_bits);
            let mut c = rng.bits(total_bits);
            if padding.useful_bits_per_block < (1usize << padding.k_log) {
                let block_size = 1usize << padding.k_log;
                for block in 0..(total_bits / block_size) {
                    for bit in padding.useful_bits_per_block..block_size {
                        let index = block * block_size + bit;
                        a[index] = false;
                        b[index] = false;
                        c[index] = false;
                    }
                }
            }

            let a = pack_bits(&a);
            let b = pack_bits(&b);
            let c = pack_bits(&c);
            let outer = rng.f128_vec(m - K_SKIP - N_INNER);
            let r = build_protocol_r(m, &outer);
            let inv_table = make_inv_table();
            let precomputed =
                precompute_round1_ab_inner_packed_padded(&a, &b, m, K_SKIP, &inv_table, &padding);

            let incumbent = round1_shift_reduce_extract_c_packed_padded_with_precomputed_ab(
                &precomputed,
                &c,
                m,
                K_SKIP,
                &r,
                &inv_table,
                &padding,
            );
            for n_hi in 5..=7 {
                for (q_partition, pair4) in [(false, false), (false, true), (true, false)] {
                    let fold4 =
                        round1_shift_reduce_extract_c_packed_padded_with_precomputed_ab_fold4_n_hi(
                            &precomputed,
                            &c,
                            m,
                            K_SKIP,
                            &r,
                            &inv_table,
                            &padding,
                            n_hi,
                            q_partition,
                            pair4,
                        );

                    assert_eq!(
                        fold4.0, incumbent.0,
                        "round1 AB mismatch at m={m}, n_hi={n_hi}, q_partition={q_partition}, pair4={pair4}"
                    );
                    assert_eq!(
                        fold4.1, incumbent.1,
                        "round1 C mismatch at m={m}, n_hi={n_hi}, q_partition={q_partition}, pair4={pair4}"
                    );
                    assert_eq!(
                        fold4.2, incumbent.2,
                        "canonical C mismatch at m={m}, n_hi={n_hi}, q_partition={q_partition}, pair4={pair4}"
                    );
                    assert_eq!(
                        fold4.3, incumbent.3,
                        "quad C mismatch at m={m}, n_hi={n_hi}, q_partition={q_partition}, pair4={pair4}"
                    );
                    assert_eq!(fold4.4.len(), 16 * 2 * ELL);

                    let small = small_challenges_ghash();
                    let medium = medium_challenges_ghash();
                    let low_eq = build_eq(&[small[1], small[2], medium[0], medium[1]]);
                    let mut collapsed = vec![F128::ZERO; 2 * ELL];
                    for low in 0..16 {
                        for packed in 0..2 * ELL {
                            collapsed[packed] += low_eq[low] * fold4.4[low * 2 * ELL + packed];
                        }
                    }
                    assert_eq!(
                        collapsed, incumbent.2,
                        "fold4 C collapse mismatch at m={m}, n_hi={n_hi}, q_partition={q_partition}, pair4={pair4}"
                    );
                }
            }
        }
    }

    /// Scalar oracle for the convert-table fold, written independently of the
    /// NEON kernels: plain `F128` adds and muls, one lane at a time. Mirrors
    /// the non-aarch64 `kernels::portable::accumulate_convert`, which is
    /// `cfg`-compiled away on this target and so cannot be called here.
    #[cfg(target_arch = "aarch64")]
    fn accumulate_convert_oracle(
        chunk_ab_bytes: &[[u8; 64]; 1 << N_MEDIUM],
        chunk_c_bytes: &[[u8; 64]; 1 << N_MEDIUM],
        n_b_med: usize,
        convert: &[F128],
        eq_lo_val: F128,
        partial_ab: &mut [F128; ELL],
        partial_c: &mut [F128; ELL],
    ) {
        for lane in 0..ELL {
            let mut cf_ab = F128::ZERO;
            let mut cf_c = F128::ZERO;
            for b_med in 0..n_b_med {
                let base = b_med * 256;
                cf_ab += convert[base + chunk_ab_bytes[b_med][lane] as usize];
                cf_c += convert[base + chunk_c_bytes[b_med][lane] as usize];
            }
            partial_ab[lane] += cf_ab * eq_lo_val;
            partial_c[lane] += cf_c * eq_lo_val;
        }
    }

    /// Eight-bank oracle, same construction as [`accumulate_convert_oracle`].
    ///
    /// Deliberately routed through the **convert table** rather than the mask
    /// shortcut the kernel uses: bank `s` sums `convert[b·256 + (c & (1<<s))]`
    /// = `α^s · Σ_b γ^b · bit_s(c_b)` and is then stripped of `α^s`. Agreement
    /// with the kernel is therefore an independent check of the collapse
    /// `Σ_b γ^b · bit_s(c_b) == F128 { lo: mask_s }`, not a restatement of it.
    fn accumulate_convert_with_s_hat_v_oracle(
        chunk_ab_bytes: &[[u8; 64]; 1 << N_MEDIUM],
        c_block: &[u8; (1 << N_MEDIUM) * 64],
        n_b_med: usize,
        convert: &[F128],
        eq_lo_val: F128,
        partial_ab: &mut [F128; ELL],
        partial_c: &mut [[F128; ELL]; N_C_BANKS],
    ) {
        // The kernel fuses the per-`b_med` transpose into its mask build; the
        // oracle keeps them separate, so this is also a transpose cross-check.
        let mut chunk_c_bytes = [[0u8; 64]; 1 << N_MEDIUM];
        for b_med in 0..n_b_med {
            let row: &[u8; 64] = c_block[b_med * 64..(b_med + 1) * 64].try_into().unwrap();
            bit_transpose_64bytes(row, &mut chunk_c_bytes[b_med]);
        }
        let alpha_inv = alpha_inv_f128();
        let mut alpha_inv_pow = [F128::ONE; N_C_BANKS];
        for s in 1..N_C_BANKS {
            alpha_inv_pow[s] = alpha_inv_pow[s - 1] * alpha_inv;
        }
        for lane in 0..ELL {
            let mut cf_ab = F128::ZERO;
            let mut cf_c = [F128::ZERO; N_C_BANKS];
            for b_med in 0..n_b_med {
                let base = b_med * 256;
                let v_c = chunk_c_bytes[b_med][lane] as usize;
                cf_ab += convert[base + chunk_ab_bytes[b_med][lane] as usize];
                for (s, acc) in cf_c.iter_mut().enumerate() {
                    *acc += convert[base + (v_c & (1 << s))];
                }
            }
            partial_ab[lane] += cf_ab * eq_lo_val;
            for (s, bank) in partial_c.iter_mut().enumerate() {
                bank[lane] += (cf_c[s] * alpha_inv_pow[s]) * eq_lo_val;
            }
        }
    }

    /// The 4-lane-wide NEON convert fold must be **bit-identical** to the
    /// one-lane-at-a-time scalar oracle, for every `n_b_med` the caller can
    /// pass (0..=16, covering both the unrolled full path and the boundary
    /// window) and starting from non-zero partials, so that the `+=`
    /// accumulation semantics are exercised rather than plain assignment.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_accumulate_convert_matches_scalar_oracle() {
        let convert = convert_table();
        let mut rng = Rng::new(0xACC_C0FFEE);

        for n_b_med in 0..=(1 << N_MEDIUM) {
            let mut chunk_ab = [[0u8; 64]; 1 << N_MEDIUM];
            let mut chunk_c = [[0u8; 64]; 1 << N_MEDIUM];
            for b_med in 0..(1 << N_MEDIUM) {
                for lane in 0..ELL {
                    chunk_ab[b_med][lane] = (rng.next_u64() & 0xff) as u8;
                    chunk_c[b_med][lane] = (rng.next_u64() & 0xff) as u8;
                }
            }
            let eq_lo_val = rng.f128();

            // Non-zero, and different per lane, so a dropped or misrouted
            // accumulation cannot coincidentally match.
            let seed_ab: [F128; ELL] = core::array::from_fn(|_| rng.f128());
            let seed_c: [F128; ELL] = core::array::from_fn(|_| rng.f128());

            let (mut got_ab, mut got_c) = (seed_ab, seed_c);
            // SAFETY: aarch64 target; arrays are the exact sizes the kernel
            // indexes, and `convert` is the full 16*256-entry table.
            unsafe {
                kernels::aarch64::accumulate_convert(
                    &chunk_ab,
                    &chunk_c,
                    n_b_med,
                    convert,
                    eq_lo_val,
                    &mut got_ab,
                    &mut got_c,
                );
            }

            let (mut want_ab, mut want_c) = (seed_ab, seed_c);
            accumulate_convert_oracle(
                &chunk_ab,
                &chunk_c,
                n_b_med,
                convert,
                eq_lo_val,
                &mut want_ab,
                &mut want_c,
            );

            assert_eq!(got_ab, want_ab, "partial_ab mismatch at n_b_med={n_b_med}");
            assert_eq!(got_c, want_c, "partial_c mismatch at n_b_med={n_b_med}");
        }
    }

    /// **The fused-transpose gate.** The kernel now reads the packed witness
    /// directly and does one cross-`b_med` transpose; the reference still goes
    /// the long way round — sixteen per-`b_med` [`bit_transpose_64bytes`] calls
    /// into scratch rows, then the mask scatter. Two genuinely different routes
    /// to the same bytes, compared for every `n_b_med` (0..=16, covering the
    /// full path and the padded boundary window) over random blocks, from
    /// non-zero partials so `+=` semantics are exercised too.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn fused_c_masks_match_composed_transpose() {
        let mut rng = Rng::new(0xC0DE_BA5E);
        for round in 0..4 {
            for n_b_med in 0..=(1 << N_MEDIUM) {
                let mut c_block = [0u8; (1 << N_MEDIUM) * 64];
                for byte in c_block.iter_mut() {
                    *byte = (rng.next_u64() & 0xff) as u8;
                }
                let mask_tables = build_c_mask_tables(&[rng.f128()]);
                let seed: [[F128; ELL]; N_C_BANKS] =
                    core::array::from_fn(|_| core::array::from_fn(|_| rng.f128()));

                let mut got = seed;
                kernels::accumulate_c_banks(&c_block, n_b_med, &mask_tables, &mut got);
                let mut want = seed;
                kernels::accumulate_c_banks_scalar(&c_block, n_b_med, &mask_tables, &mut want);

                for s in 0..N_C_BANKS {
                    assert_eq!(
                        got[s], want[s],
                        "bank {s} mismatch at n_b_med={n_b_med}, round={round}"
                    );
                }
            }
        }
    }

    /// The mask tables must reproduce `F128 { lo: m } * eq` for every 16-bit
    /// mask — this is the F2-linearity that removes the multiply from the C
    /// drain. Checked against the multiply itself over all 65_536 masks, for
    /// several `eq`, plus the two halves' defining identities.
    #[test]
    fn c_mask_tables_reproduce_masked_multiply() {
        let mut rng = Rng::new(0x7AB1E_5EED);
        let eqs: Vec<F128> = (0..4).map(|_| rng.f128()).collect();
        let tables = build_c_mask_tables(&eqs);
        for (x, eq) in eqs.iter().enumerate() {
            let slot = &tables[x * C_MASK_TABLE_STRIDE..(x + 1) * C_MASK_TABLE_STRIDE];
            let (t_lo, t_hi) = slot.split_at(256);
            for v in 0..256usize {
                assert_eq!(
                    t_lo[v],
                    F128 {
                        lo: v as u64,
                        hi: 0
                    } * *eq,
                    "t_lo[{v}]"
                );
                assert_eq!(
                    t_hi[v],
                    F128 {
                        lo: (v as u64) << 8,
                        hi: 0
                    } * *eq,
                    "t_hi[{v}]"
                );
            }
            for m in 0..=u16::MAX {
                assert_eq!(
                    t_lo[usize::from(m & 0xff)] + t_hi[usize::from(m >> 8)],
                    F128 {
                        lo: u64::from(m),
                        hi: 0
                    } * *eq,
                    "mask {m} at x={x}"
                );
            }
        }
    }

    /// The Fold4 C table keeps q=b_med mod 4 as a bank coordinate and folds
    /// only h=floor(b_med/4).  Its four index bits therefore select exponents
    /// 0,4,8,12.  Check the compact lookup against the defining field
    /// multiply for every mask.
    #[test]
    fn c_fold4_mask_tables_reproduce_strided_masked_multiply() {
        let mut rng = Rng::new(0xF01D_4004);
        let eqs: Vec<F128> = (0..8).map(|_| rng.f128()).collect();
        let tables = build_c_fold4_mask_tables(&eqs);
        for (x, eq) in eqs.iter().enumerate() {
            let slot = &tables[x * C_FOLD4_MASK_TABLE_STRIDE..(x + 1) * C_FOLD4_MASK_TABLE_STRIDE];
            for nibble in 0..16usize {
                let mut spread = 0u64;
                for h in 0..N_C_FOLD4_GROUPS {
                    spread |= (((nibble >> h) & 1) as u64) << (N_C_FOLD4_GROUPS * h);
                }
                assert_eq!(
                    slot[nibble],
                    F128 { lo: spread, hi: 0 } * *eq,
                    "strided mask {nibble:#x} at x={x}"
                );
            }
        }
        crate::scratch::give_f128(tables);
    }

    /// Every composed table entry must be the exact sum of its two source
    /// entries. An odd number of low slots additionally pins the zero-table
    /// sentinel used by the singleton tail.
    #[test]
    fn c_fold4_pair_tables_compose_all_256_indices() {
        let mut rng = Rng::new(0xF01D_2A17_0256);
        let eqs: Vec<F128> = (0..5).map(|_| rng.f128()).collect();
        let singles = build_c_fold4_mask_tables(&eqs);
        let pairs = build_c_fold4_pair_mask_tables(&eqs);
        for pair in 0..eqs.len().div_ceil(2) {
            let even = &singles[(2 * pair) * C_FOLD4_MASK_TABLE_STRIDE
                ..(2 * pair + 1) * C_FOLD4_MASK_TABLE_STRIDE];
            let odd = (2 * pair + 1 < eqs.len()).then(|| {
                &singles[(2 * pair + 1) * C_FOLD4_MASK_TABLE_STRIDE
                    ..(2 * pair + 2) * C_FOLD4_MASK_TABLE_STRIDE]
            });
            let composed = &pairs[pair * C_FOLD4_PAIR_MASK_TABLE_STRIDE
                ..(pair + 1) * C_FOLD4_PAIR_MASK_TABLE_STRIDE];
            for b in 0..C_FOLD4_MASK_TABLE_STRIDE {
                for a in 0..C_FOLD4_MASK_TABLE_STRIDE {
                    let want = even[a] + odd.map_or(F128::ZERO, |table| table[b]);
                    assert_eq!(
                        composed[a | (b << 4)],
                        want,
                        "pair={pair}, even_mask={a:#x}, odd_mask={b:#x}"
                    );
                }
            }
        }
        crate::scratch::give_f128(singles);
        crate::scratch::give_f128(pairs);
    }

    /// Differential gate for the paired q-local kernel. Both blocks range
    /// independently over every legal padding boundary, which catches any
    /// accidental sharing of `n_b_med` between the two mask halves.
    #[test]
    fn c_fold4_q_pair_kernel_matches_two_independent_drains() {
        let mut rng = Rng::new(0xF01D_0A17_1717);
        for round in 0..2 {
            let mut c_even = [0u8; (1 << N_MEDIUM) * ELL];
            let mut c_odd = [0u8; (1 << N_MEDIUM) * ELL];
            for byte in &mut c_even {
                *byte = (rng.next_u64() & 0xff) as u8;
            }
            for byte in &mut c_odd {
                *byte = (rng.next_u64() & 0xff) as u8;
            }
            let eqs = [rng.f128(), rng.f128()];
            let singles = build_c_fold4_mask_tables(&eqs);
            let pairs = build_c_fold4_pair_mask_tables(&eqs);
            let pair_table: &[F128; C_FOLD4_PAIR_MASK_TABLE_STRIDE] = pairs
                .as_slice()
                .try_into()
                .expect("one composed pair table");
            for n_even in 0..=(1 << N_MEDIUM) {
                for n_odd in 0..=(1 << N_MEDIUM) {
                    for q in 0..N_C_FOLD4_GROUPS {
                        let seed: [[F128; ELL]; N_C_BANKS] =
                            core::array::from_fn(|_| core::array::from_fn(|_| rng.f128()));
                        let mut got = seed;
                        kernels::accumulate_c_fold4_q_pair_banks(
                            &c_even, n_even, &c_odd, n_odd, q, pair_table, &mut got,
                        );
                        let mut want = seed;
                        accumulate_c_fold4_q_banks_scalar(
                            &c_even,
                            n_even,
                            q,
                            &singles[..C_FOLD4_MASK_TABLE_STRIDE],
                            &mut want,
                        );
                        accumulate_c_fold4_q_banks_scalar(
                            &c_odd,
                            n_odd,
                            q,
                            &singles[C_FOLD4_MASK_TABLE_STRIDE..],
                            &mut want,
                        );
                        assert_eq!(
                            got, want,
                            "pair drain mismatch round={round}, n_even={n_even}, n_odd={n_odd}, q={q}"
                        );
                    }
                }
            }
            crate::scratch::give_f128(singles);
            crate::scratch::give_f128(pairs);
        }
    }

    /// Four-block differential oracle. The expected value performs four
    /// independent scalar 16-entry drains; the candidate performs two
    /// pair-table loads and one accumulator RMW. Vary every block's padding
    /// count independently and cross the two pair boundaries.
    #[test]
    fn c_fold4_four_kernel_matches_four_independent_drains() {
        let mut rng = Rng::new(0xF01D_4004_1717);
        let mut c_blocks = [[0u8; (1 << N_MEDIUM) * ELL]; 4];
        for block in &mut c_blocks {
            for byte in block {
                *byte = (rng.next_u64() & 0xff) as u8;
            }
        }
        let eqs: [F128; 4] = std::array::from_fn(|_| rng.f128());
        let singles = build_c_fold4_mask_tables(&eqs);
        let pairs = build_c_fold4_pair_mask_tables(&eqs);
        let block_refs: [&[u8; (1 << N_MEDIUM) * ELL]; 4] =
            std::array::from_fn(|side| &c_blocks[side]);
        let pair_tables: [&[F128; C_FOLD4_PAIR_MASK_TABLE_STRIDE]; 2] =
            std::array::from_fn(|pair| {
                (&pairs[pair * C_FOLD4_PAIR_MASK_TABLE_STRIDE
                    ..(pair + 1) * C_FOLD4_PAIR_MASK_TABLE_STRIDE])
                    .try_into()
                    .expect("one 256-entry Fold4 pair table")
            });

        let mut cases = vec![[0usize; 4], [16usize; 4], [0, 16, 0, 16], [16, 0, 16, 0]];
        let fixed = [3usize, 7, 11, 15];
        for side in 0..4 {
            for count in 0..=16 {
                let mut case = fixed;
                case[side] = count;
                cases.push(case);
            }
        }
        for left in 0..=16 {
            for right in 0..=16 {
                cases.push([left, 5, right, 13]);
                cases.push([3, left, 11, right]);
            }
        }

        for (case_index, counts) in cases.into_iter().enumerate() {
            let seed: [[F128; ELL]; N_C_FOLD4_BANKS] =
                core::array::from_fn(|_| core::array::from_fn(|_| rng.f128()));
            let mut got = seed;
            kernels::accumulate_c_fold4_four_banks(block_refs, counts, pair_tables, &mut got);

            let mut want = seed;
            for side in 0..4 {
                accumulate_c_fold4_banks_scalar(
                    block_refs[side],
                    counts[side],
                    &singles
                        [side * C_FOLD4_MASK_TABLE_STRIDE..(side + 1) * C_FOLD4_MASK_TABLE_STRIDE],
                    &mut want,
                );
            }
            assert_eq!(
                got, want,
                "four-block drain mismatch case={case_index}, counts={counts:?}"
            );
        }

        crate::scratch::give_f128(singles);
        crate::scratch::give_f128(pairs);
    }

    #[test]
    fn fold4_q_partition_requires_exact_one() {
        use std::ffi::OsStr;

        assert!(!fold4_q_partition_from_value(None));
        assert!(fold4_q_partition_from_value(Some(OsStr::new("1"))));
        for rejected in ["", "0", "4", "true", "yes"] {
            assert!(
                !fold4_q_partition_from_value(Some(OsStr::new(rejected))),
                "non-exact opt-in {rejected:?} must retain monolithic pair fusion"
            );
        }
    }

    #[test]
    fn fold4_pair4_defaults_on_and_any_kill_value_disables() {
        use std::ffi::OsStr;

        assert!(fold4_pair4_from_kill_value(None));
        for kill_value in ["", "0", "1", "2", "4", "true", "yes"] {
            assert!(
                !fold4_pair4_from_kill_value(Some(OsStr::new(kill_value))),
                "kill-switch value {kill_value:?} must retain two-block pair fusion"
            );
        }
    }

    /// Core Direct-C Fold4 oracle.  The 32 retained-coordinate banks, once
    /// collapsed under the first two protocol medium challenges, must equal
    /// the incumbent eight-bank accumulator for every legal padding boundary.
    /// Random non-zero seeds also exercise the += contract.
    #[test]
    fn c_fold4_banks_collapse_to_incumbent_banks() {
        let mut rng = Rng::new(0xC004_C011_A95E);

        for round in 0..4 {
            for n_b_med in 0..=(1 << N_MEDIUM) {
                let mut c_block = [0u8; (1 << N_MEDIUM) * ELL];
                for byte in &mut c_block {
                    *byte = (rng.next_u64() & 0xff) as u8;
                }
                let eq_outer = rng.f128();
                let current_tables = build_c_mask_tables(&[eq_outer * d_inv()]);
                let fold4_tables = build_c_fold4_mask_tables(&[eq_outer * d_hi_inv()]);

                let seed_fold4: [[F128; ELL]; N_C_FOLD4_BANKS] =
                    core::array::from_fn(|_| core::array::from_fn(|_| rng.f128()));
                let mut got_fold4 = seed_fold4;
                accumulate_c_fold4_banks_scalar(&c_block, n_b_med, &fold4_tables, &mut got_fold4);

                let mut want = collapse_c_fold4_banks(&seed_fold4);
                kernels::accumulate_c_banks_scalar(&c_block, n_b_med, &current_tables, &mut want);
                assert_eq!(
                    collapse_c_fold4_banks(&got_fold4),
                    want,
                    "collapse mismatch at n_b_med={n_b_med}, round={round}"
                );

                crate::scratch::give_f128(current_tables);
                crate::scratch::give_f128(fold4_tables);
            }
        }
    }

    /// AArch64 fused transpose/drain gate.  The optimized kernel consumes raw
    /// packed rows, while the oracle composes sixteen ordinary bit transposes
    /// and scalar mask routing.  Exercise every padding boundary from non-zero
    /// accumulators so both coordinate mapping and += semantics are pinned.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn fused_c_fold4_masks_match_scalar() {
        let mut rng = Rng::new(0xF04D_A64C_0DE5);
        for round in 0..8 {
            for n_b_med in 0..=(1 << N_MEDIUM) {
                let mut c_block = [0u8; (1 << N_MEDIUM) * ELL];
                for byte in &mut c_block {
                    *byte = (rng.next_u64() & 0xff) as u8;
                }
                let tables = build_c_fold4_mask_tables(&[rng.f128()]);
                let seed: [[F128; ELL]; N_C_FOLD4_BANKS] =
                    core::array::from_fn(|_| core::array::from_fn(|_| rng.f128()));
                let mut got = seed;
                kernels::accumulate_c_fold4_banks(&c_block, n_b_med, &tables, &mut got);
                let mut want = seed;
                accumulate_c_fold4_banks_scalar(&c_block, n_b_med, &tables, &mut want);
                assert_eq!(
                    got, want,
                    "Fold4 fused C mismatch at n_b_med={n_b_med}, round={round}"
                );
                crate::scratch::give_f128(tables);
            }
        }
    }

    /// Exhaustive coordinate-routing oracle over every medium row and K bank,
    /// sampled across byte-chunk and bit-lane boundaries.  A one-hot witness
    /// at `(b_med,K,lane)` must land only in `[q=b_med&3][K][lane]` with table
    /// index `1 << (b_med>>2)`.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn fused_c_fold4_one_hot_routes_every_coordinate() {
        let eq = F128 {
            lo: 0x8f31_17e2_906b_42d5,
            hi: 0x6ca8_01dd_7314_b9ef,
        };
        let tables = build_c_fold4_mask_tables(&[eq]);
        for b_med in 0..(1 << N_MEDIUM) {
            let q = b_med & 3;
            let h = b_med >> 2;
            for k in 0..N_C_BANKS {
                for lane in [0usize, 7, 8, 31, 63] {
                    let mut c_block = [0u8; (1 << N_MEDIUM) * ELL];
                    c_block[b_med * ELL + k * 8 + lane / 8] = 1 << (lane & 7);
                    let mut got = [[F128::ZERO; ELL]; N_C_FOLD4_BANKS];
                    kernels::accumulate_c_fold4_banks(&c_block, 1 << N_MEDIUM, &tables, &mut got);
                    for bank in 0..N_C_FOLD4_BANKS {
                        for out_lane in 0..ELL {
                            let want = if bank == q * N_C_BANKS + k && out_lane == lane {
                                tables[1 << h]
                            } else {
                                F128::ZERO
                            };
                            assert_eq!(
                                got[bank][out_lane], want,
                                "route b_med={b_med}, k={k}, lane={lane}, bank={bank}, out_lane={out_lane}"
                            );
                        }
                    }
                }
            }
        }
        crate::scratch::give_f128(tables);
    }

    /// The exported 16x128 layout must collapse under the exact four suffix
    /// challenges used by ring-switch to the canonical C vector.  This also
    /// pins the shared producer/consumer ordering `low=e_small+4*q_medium`.
    #[test]
    fn c_fold4_layout_collapses_to_wire_s_hat_v_c() {
        let mut rng = Rng::new(0xC004_16B4_1280);
        for round in 0..16 {
            let banks: [[F128; ELL]; N_C_FOLD4_BANKS] =
                core::array::from_fn(|_| core::array::from_fn(|_| rng.f128()));
            let (res_c_s, s_hat_v_c, quad_c, fold4_c) = finish_c_fold4_banks(&banks);
            let collapsed = collapse_c_fold4_banks(&banks);
            let (want_res, want_s_hat_v, want_quad) = finish_c_banks(&collapsed);
            assert_eq!(res_c_s, want_res, "wire bank finish at round={round}");
            assert_eq!(quad_c, want_quad, "quad bank finish at round={round}");
            assert_eq!(s_hat_v_c, want_s_hat_v, "canonical finish at round={round}");
            assert_eq!(fold4_c.len(), 16 * 2 * ELL);

            let small = small_challenges_ghash();
            let medium = medium_challenges_ghash();
            let low_eq = build_eq(&[small[1], small[2], medium[0], medium[1]]);
            let mut from_fold4 = vec![F128::ZERO; 2 * ELL];
            for low in 0..16 {
                for packed in 0..2 * ELL {
                    from_fold4[packed] += low_eq[low] * fold4_c[low * 2 * ELL + packed];
                }
            }
            assert_eq!(
                from_fold4, s_hat_v_c,
                "16-bank layout collapse at round={round}"
            );
        }
    }

    /// The gather-halving kernel is only bit-exact because `convert` is
    /// **F2-linear in its index bits**: `T_b[u ^ v] == T_b[u] ^ T_b[v]`. That
    /// is what lets one paired lookup stand in for two per-`b_med` lookups.
    /// Checked exhaustively over every `(b, u, v)` — 16·256·256 triples.
    #[test]
    fn convert_table_index_linear() {
        let t = convert_table();
        for b in 0..16 {
            let block = &t[b * 256..(b + 1) * 256];
            assert_eq!(block[0], F128::ZERO, "T_{b}[0] must be zero");
            for u in 0..256usize {
                for v in 0..256usize {
                    assert_eq!(
                        block[u ^ v],
                        block[u] + block[v],
                        "index-linearity failed at b={b}, u={u}, v={v}"
                    );
                }
            }
        }
    }

    /// Same contract for the eight-bank variant — this is the kernel the prover
    /// actually runs (`prove_packed_padded_capture_s_hat_v_c` always sets
    /// `capture = true`), so it gets the same bit-exactness guard.
    #[test]
    fn accumulate_convert_with_s_hat_v_matches_scalar_oracle() {
        let convert = convert_table();
        let mut rng = Rng::new(0x5A17_C0DE);

        for n_b_med in 0..=(1 << N_MEDIUM) {
            let mut chunk_ab = [[0u8; 64]; 1 << N_MEDIUM];
            let mut c_block = [0u8; (1 << N_MEDIUM) * 64];
            for b_med in 0..(1 << N_MEDIUM) {
                for lane in 0..ELL {
                    chunk_ab[b_med][lane] = (rng.next_u64() & 0xff) as u8;
                    c_block[b_med * 64 + lane] = (rng.next_u64() & 0xff) as u8;
                }
            }
            let eq_lo_val = rng.f128();

            let seed_ab: [F128; ELL] = core::array::from_fn(|_| rng.f128());
            let seed_c: [[F128; ELL]; N_C_BANKS] =
                core::array::from_fn(|_| core::array::from_fn(|_| rng.f128()));

            // The kernel drains through the mask tables; the oracle multiplies.
            // Building the tables from a one-entry `eq_lo_scaled` is exactly
            // what the prove does per `x_outer_lo`.
            let mask_tables = build_c_mask_tables(&[eq_lo_val]);

            let (mut got_ab, mut got_c) = (seed_ab, seed_c);
            kernels::accumulate_convert_with_s_hat_v(
                &chunk_ab,
                &c_block,
                n_b_med,
                convert,
                eq_lo_val,
                &mask_tables,
                &mut got_ab,
                &mut got_c,
            );

            let (mut want_ab, mut want_c) = (seed_ab, seed_c);
            accumulate_convert_with_s_hat_v_oracle(
                &chunk_ab,
                &c_block,
                n_b_med,
                convert,
                eq_lo_val,
                &mut want_ab,
                &mut want_c,
            );

            assert_eq!(got_ab, want_ab, "partial_ab mismatch at n_b_med={n_b_med}");
            for s in 0..N_C_BANKS {
                assert_eq!(
                    got_c[s], want_c[s],
                    "partial_c bank {s} mismatch at n_b_med={n_b_med}"
                );
            }
        }
    }

    /// The ranked identity-C shortcut must reproduce every value emitted by
    /// the incumbent 32-bank C producer.  Build the alternate lincheck stripe
    /// from the same packed witness, then compare AB, the round-one C wire
    /// contribution, and all three captured RingSwitch tensors independently.
    #[test]
    fn lincheck_stripe_c_fold4_matches_incumbent() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .stack_size(16 << 20)
            .build()
            .unwrap();
        pool.install(lincheck_stripe_c_fold4_matches_incumbent_inner);
    }

    fn lincheck_stripe_c_fold4_matches_incumbent_inner() {
        const M: usize = 17;
        const K_LOG: usize = 14;
        let cases = [
            PaddingSpec {
                k_log: K_LOG,
                useful_bits_per_block: 1 << K_LOG,
            },
            PaddingSpec {
                k_log: K_LOG,
                useful_bits_per_block: 15_409,
            },
        ];

        for (case, padding) in cases.into_iter().enumerate() {
            let mut rng = Rng::new(0xC57A_1F00_u64.wrapping_add(case as u64));
            let total_bits = 1usize << M;
            let mut a = rng.bits(total_bits);
            let mut b = rng.bits(total_bits);
            let mut c = rng.bits(total_bits);
            if padding.useful_bits_per_block < (1usize << K_LOG) {
                for block in 0..(total_bits >> K_LOG) {
                    for bit in padding.useful_bits_per_block..(1usize << K_LOG) {
                        let index = (block << K_LOG) + bit;
                        a[index] = false;
                        b[index] = false;
                        c[index] = false;
                    }
                }
            }

            let a = pack_bits(&a);
            let b = pack_bits(&b);
            let c = pack_bits(&c);
            let outer = rng.f128_vec(M - K_SKIP - N_INNER);
            let r = build_protocol_r(M, &outer);
            let inv_table = make_inv_table();
            let ab_inner =
                precompute_round1_ab_inner_packed_padded(&a, &b, M, K_SKIP, &inv_table, &padding);
            let incumbent = round1_shift_reduce_extract_c_packed_padded_with_precomputed_ab_fold4(
                &ab_inner, &c, M, K_SKIP, &r, &inv_table, &padding,
            );

            let got_ab = round1_shift_reduce_ab_packed_padded_with_precomputed(
                &ab_inner, M, K_SKIP, &r, &padding,
            );
            assert_eq!(got_ab, incumbent.0, "AB mismatch in case {case}");

            // `pack_bits` uses the same little-endian polynomial-basis layout
            // as packed F128 witnesses. Materialize aligned words for the
            // public lincheck stripe converter rather than transmuting a
            // byte-aligned allocation.
            let c_words: Vec<F128> = c
                .chunks_exact(16)
                .map(|bytes| F128 {
                    lo: u64::from_le_bytes(bytes[..8].try_into().unwrap()),
                    hi: u64::from_le_bytes(bytes[8..].try_into().unwrap()),
                })
                .collect();
            let c_lincheck = crate::lincheck::pack_z_lincheck_from_packed(&c_words, M, K_LOG);
            let got_c = round1_c_fold4_from_lincheck_stripe(
                &c_lincheck,
                M,
                K_LOG,
                K_SKIP,
                padding.useful_bits_per_block,
                &r,
                &inv_table,
                round1_c_prelude(&c_lincheck, M, K_LOG, padding.useful_bits_per_block, &r),
            );
            assert_eq!(got_c.0, incumbent.1, "round-one C mismatch in case {case}");
            assert_eq!(got_c.1, incumbent.2, "canonical C mismatch in case {case}");
            assert_eq!(got_c.2, incumbent.3, "quad C mismatch in case {case}");
            assert_eq!(got_c.3, incumbent.4, "fold4 C mismatch in case {case}");

            // Fold8 sibling: identical wire outputs from the wider statistic,
            // and the 64-bank tensor must collapse to every narrower form.
            let got_c8 = round1_c_fold8_from_lincheck_stripe(
                &c_lincheck,
                M,
                K_LOG,
                K_SKIP,
                padding.useful_bits_per_block,
                &r,
                &inv_table,
                round1_c_prelude(&c_lincheck, M, K_LOG, padding.useful_bits_per_block, &r),
            );
            assert_eq!(
                got_c8.0, incumbent.1,
                "fold8 round-one C mismatch in case {case}"
            );
            assert_eq!(
                got_c8.1, incumbent.2,
                "fold8 canonical C mismatch in case {case}"
            );
            assert_eq!(
                got_c8.2, incumbent.3,
                "fold8 quad C mismatch in case {case}"
            );
            let inner_tail = &r[K_SKIP + 1..K_LOG];
            assert_eq!(
                crate::pcs::ring_switch::collapse_s_hat_v_fold8(&got_c8.3, &inner_tail[..6]),
                incumbent.2,
                "fold8 suffix[..6] collapse mismatch in case {case}"
            );
            // Candidate r494: preserve the fold8-to-fold4 equivalence check.
            // Folding retained coordinates 4 and 5 must reproduce the
            // incumbent 16-bank fold4 tensor exactly.
            let hi_eq = build_eq(&inner_tail[4..6]);
            let n_packed = 1usize << crate::pcs::LOG_PACKING;
            let mut fold4_from_8 = vec![F128::ZERO; 16 * n_packed];
            for q in 0..4 {
                for e in 0..16 {
                    let src = (e + 16 * q) * n_packed;
                    let dst = e * n_packed;
                    for packed in 0..n_packed {
                        fold4_from_8[dst + packed] += hi_eq[q] * got_c8.3[src + packed];
                    }
                }
            }
            assert_eq!(
                fold4_from_8, incumbent.4,
                "fold8 → fold4 reduction mismatch in case {case}"
            );
        }
    }
}
// r495 archive identity: preserve scalar fallback semantics while forcing a distinct candidate archive.

// Competition candidate r497-20260806T021939Z: preserve kernel semantics while forcing a distinct editable archive.

// Competition candidate r501: terminal-slot-release archive identity; preserve scalar fallback semantics.

// r505: retain the specialized skip-optimized path as an explicit benchmark candidate.

// r506: archive identity marker; no runtime effect.

// Submission archive nonce r507: semantics-neutral; forces a distinct packaged candidate.

// Competition archive nonce r511: 1785983977508655435

// Competition archive nonce r512: 1785984029943258358

// chewy archive nonce r514 20260806T024159Z

// Submission archive nonce r538: 20260806T030901Z.
// chewy-cadence: r553 source-distinct submission candidate

// chewy-cadence: r555 1785988171001930925

// chewy-cadence: r557 1785988378687326387

// chewy-cadence: r558 1785988571742870116

// chewy-cadence: r560 20260806T040229Z

// r561 archive-distinct hot-line marker: 20260806T040426Z

// r562 archive-distinct hot-line marker: 20260806T040635Z

// chewy-cadence: r578 20260806T044834Z

// chewy-cadence: r582 20260806T045819Z
