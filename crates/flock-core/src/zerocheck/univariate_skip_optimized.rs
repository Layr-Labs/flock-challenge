//! Round-1 prover message — fully optimized (shift_reduce + extract_c, scalar).
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

/// Ranked C-drain reconstruct: 16×64 byte-planes → 64 F128. Identity-C
/// lincheck worker-reduce has the same plane layout.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq"
))]
#[inline]
pub(crate) fn c_plane_bank_to_f128(bank_planes: &[u8; 1024], out: &mut [F128; 64]) {
    kernels::c_plane_bank_to_f128(bank_planes, out)
}

#[cfg(all(test, target_arch = "aarch64"))]
use kernels::aarch64::{
    bit_transpose_64bytes_neon, shift_reduce_inner_ab_fused_neon,
    shift_reduce_inner_ab_fused_neon_x2, shift_reduce_inner_ab_neon,
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
pub(crate) const N_INNER: usize = 7;
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

/// Multiplicative inverse of [`C_S_F8`] in the AES `F_8` subfield.
///
/// `0x1c * 0xff = 1` modulo `x^8 + x^4 + x^3 + x + 1`. Because [`phi8`]
/// is a field embedding, `phi8(0xff)` is exactly `phi8(0x1c)^{-1}` in
/// `F_128`. Keeping the inverse in the subfield deletes the generic 127-step
/// Fermat inversion from the ranked identity-C finish.
pub const C_S_INV_F8: u8 = 0xFF;

/// The constant `C_s = φ_8(0x1C) ∈ F_{2^128}` — the relative scaling factor
/// between this optimized output and the naive output.
pub fn c_s_f128() -> F128 {
    phi8(F8(C_S_F8))
}

/// `C_s^{-1}` obtained in the embedded `F_8` subfield.
#[inline]
pub fn c_s_inv_f128() -> F128 {
    phi8(F8(C_S_INV_F8))
}

/// Exact same-binary rollback for the ranked identity-C subfield inverse.
/// Only the literal value `1` restores the incumbent per-proof Fermat
/// inversion; all other values leave the constant-time subfield load active.
const ENV_NO_ZC_C_S_SUBFIELD_INV: &str = "FLOCK_NO_ZC_C_S_SUBFIELD_INV";

#[inline]
fn c_s_subfield_inv_disabled_value(value: Option<&std::ffi::OsStr>) -> bool {
    value == Some(std::ffi::OsStr::new("1"))
}

#[inline]
fn c_s_subfield_inv_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        !c_s_subfield_inv_disabled_value(std::env::var_os(ENV_NO_ZC_C_S_SUBFIELD_INV).as_deref())
    });
    *ON
}

/// Cold incumbent arm retained for exact same-binary rollback.
#[cold]
#[inline(never)]
fn c_s_inv_fermat_fallback() -> F128 {
    c_s_f128().inv()
}

#[inline(always)]
fn c_s_inv_for_identity_c() -> F128 {
    if c_s_subfield_inv_enabled() {
        c_s_inv_f128()
    } else {
        c_s_inv_fermat_fallback()
    }
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
pub(crate) fn d_inv() -> F128 {
    *D_INV_CACHE.get_or_init(compute_d_inv)
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
fn ab_eq_fold_gfni_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("FLOCK_NO_ZC_AB_EQ_FOLD").as_deref() != Some(std::ffi::OsStr::new("1"))
    })
}

/// Tensor factors of the scaled lo-eq weight for a `bank_bits = s` split:
/// `eq_lo_scaled[(w << s) | u] == eq_top_scaled[w] * eq_bot[u]` — `build_eq`
/// gives index bit `i` to `r_lo[i]`, so the low `s` index bits are exactly
/// the `r_lo[..s]` sub-product and the rest the `r_lo[s..]` one; the `D^-1`
/// prover scale rides on the top factor.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
fn ab_eq_fold_factors(r_lo: &[F128], bank_bits: usize) -> (Vec<F128>, Vec<F128>) {
    let eq_bot = super::univariate_skip::build_eq(&r_lo[..bank_bits]);
    let eq_top_scaled = super::univariate_skip::build_eq(&r_lo[bank_bits..])
        .into_iter()
        .map(|v| v * d_inv())
        .collect();
    (eq_bot, eq_top_scaled)
}

/// GFNI bit-matrix form of the per-`w` pre-scaled convert tables: the
/// convert banks are F2-linear (`gamma^b * phi_8(v)`, an additive embedding
/// times a constant), so scaling their eight basis entries by
/// `eq_top_scaled[w]` scales every composed entry exactly, and each scaled
/// bank IS sixteen 8x8 bit matrices. 2 KiB per `w` instead of a 64 KiB
/// table, and no table expansion at all. Matrix encoding matches
/// `VGF2P8AFFINEQB`: `out.bit[i] = parity(byte[7-i] & in)`.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
fn build_ab_eq_fold_mats(eq_top_scaled: &[F128], convert: &[F128]) -> Vec<u64> {
    build_ab_eq_fold_mats_gated(eq_top_scaled, convert, crate::serial_par_enabled())
}

/// `w`-count floor for the parallel matrix build: below it a rayon dispatch
/// costs more than the remaining rows.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
const AB_EQ_FOLD_MATS_PAR_MIN_W: usize = 8;

/// Body of [`build_ab_eq_fold_mats`]. The per-`w` row — eight basis scales
/// plus a 128×8 bit transpose into sixteen 8×8 matrices — is a pure function
/// of `eq_top_scaled[w]` writing its own disjoint 256-qword row, so `par`
/// fans the 32 ranked rows across the pool with an order-preserving indexed
/// map; the incumbent ran them on ONE core immediately ahead of round one's
/// wide region. Identical per-row body either way, so the table is
/// bit-identical; `false` is the incumbent sequential build, kept as the
/// kill-switch path and the byte-identity oracle (`FLOCK_NO_SERIAL_PAR=1`).
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
fn build_ab_eq_fold_mats_gated(eq_top_scaled: &[F128], convert: &[F128], par: bool) -> Vec<u64> {
    debug_assert_eq!(convert.len(), CONVERT_TABLE_SIZE);
    let fill_row = |scale: &F128, row: &mut [u64]| {
        for bm in 0..16 {
            let basis: [F128; 8] = std::array::from_fn(|j| convert[bm * 256 + (1 << j)] * *scale);
            for k in 0..16 {
                let mut qword = 0u64;
                for i in 0..8 {
                    let bit_index = 8 * k + i;
                    let mut row_bits = 0u8;
                    for (j, b) in basis.iter().enumerate() {
                        let bit = if bit_index < 64 {
                            (b.lo >> bit_index) & 1
                        } else {
                            (b.hi >> (bit_index - 64)) & 1
                        };
                        row_bits |= (bit as u8) << j;
                    }
                    qword |= (row_bits as u64) << (8 * (7 - i));
                }
                row[bm * 16 + k] = qword;
            }
        }
    };
    let mut mats = vec![0u64; eq_top_scaled.len() * 256];
    if par && eq_top_scaled.len() >= AB_EQ_FOLD_MATS_PAR_MIN_W {
        use rayon::prelude::*;
        mats.par_chunks_mut(256)
            .zip(eq_top_scaled.par_iter())
            .for_each(|(row, scale)| fill_row(scale, row));
    } else {
        for (row, scale) in mats.chunks_mut(256).zip(eq_top_scaled.iter()) {
            fill_row(scale, row);
        }
    }
    mats
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

pub(crate) fn convert_table() -> &'static [F128] {
    CONVERT_TABLE_CACHE.get_or_init(build_convert_table)
}

const C_MASK_TABLE_STRIDE: usize = 512;

fn build_c_mask_tables(eq_lo_scaled: &[F128]) -> Vec<F128> {
    use rayon::prelude::*;

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

#[inline]
pub fn bit_transpose_64bytes(input: &[u8; 64], output: &mut [u8; 64]) {
    kernels::bit_transpose_64bytes(input, output);
}

/// Challenge-independent AB half of the optimized round-1 kernel.
///
/// Dense storage has the same byte length and block layout as either packed
/// input: every `(x_outer, b_med)` consumes one 64-byte A block and one
/// 64-byte B block and produces one 64-byte transformed block. The ranked
/// residual representation may physically pack its 29 live rows while
/// retaining that dense logical extent. Keeping this in a separate scratch
/// allocation is intentional: round 2 still needs the original A and B tables
/// after the round-1 transcript challenge is sampled.
pub struct Round1AbInner {
    storage: Vec<F128>,
    /// Logical dense byte length. Ranked residual storage may physically omit
    /// the two structural-one rows and the dead padding row from every
    /// 2-KiB witness block, while all protocol shape checks still use the
    /// original dense length.
    dense_len_bytes: usize,
    /// Bytes at the start of `storage` that were NEVER written by the
    /// producer because round 1's GPU URM share was planned to cover those
    /// x_hi windows from the raw a/b buffers (see
    /// [`planned_round1_gpu_prefix_bytes`]). Always a multiple of the
    /// per-x_hi window byte count; 0 means fully valid.
    invalid_prefix_bytes: usize,
    /// Ranked BLAKE3 only: the producer omitted the complete K-rows on which
    /// B is identically one. Round one adds their identity-C contribution
    /// before the AB message is emitted.
    ranked_one_rows_elided: bool,
    /// Ranked BLAKE3 residual rows 2..=30 are stored contiguously: 29 cache
    /// lines per witness block instead of the dense 32-line address space.
    ranked_compact: bool,
}

const RANKED_AB_DENSE_BLOCK_BYTES: usize = 32 * ELL;
const RANKED_AB_COMPACT_BLOCK_BYTES: usize = 29 * ELL;

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

    /// Take scratch-backed storage that the caller will fill completely.
    pub fn take_uninit(total_bytes: usize) -> Self {
        assert_eq!(total_bytes % core::mem::size_of::<F128>(), 0);
        Self {
            storage: crate::scratch::take_f128(total_bytes / core::mem::size_of::<F128>()),
            dense_len_bytes: total_bytes,
            invalid_prefix_bytes: 0,
            ranked_one_rows_elided: false,
            ranked_compact: false,
        }
    }

    /// Take the exact ranked residual representation. The producer and the
    /// AB-only round-one consumer both brand this geometry independently;
    /// every other shape keeps [`Self::take_uninit`].
    pub fn take_ranked_compact_uninit(total_bytes: usize) -> Self {
        assert_eq!(total_bytes, 1usize << 29);
        assert_eq!(total_bytes % RANKED_AB_DENSE_BLOCK_BYTES, 0);
        let blocks = total_bytes / RANKED_AB_DENSE_BLOCK_BYTES;
        let stored_bytes = blocks * RANKED_AB_COMPACT_BLOCK_BYTES;
        assert_eq!(stored_bytes % core::mem::size_of::<F128>(), 0);
        Self {
            storage: crate::scratch::take_f128(stored_bytes / core::mem::size_of::<F128>()),
            dense_len_bytes: total_bytes,
            invalid_prefix_bytes: 0,
            ranked_one_rows_elided: false,
            ranked_compact: true,
        }
    }

    /// Brand the exact ranked residual representation.
    pub fn set_ranked_one_rows_elided(&mut self) {
        assert_eq!(self.invalid_prefix_bytes, 0);
        if self.ranked_compact {
            assert_eq!(self.dense_len_bytes, 1usize << 29);
            assert_eq!(self.as_bytes().len(), (1usize << 18) * RANKED_AB_COMPACT_BLOCK_BYTES);
        }
        self.ranked_one_rows_elided = true;
    }

    /// Whether `storage` is the ranked residual-AB representation.
    #[inline]
    pub fn ranked_one_rows_elided(&self) -> bool {
        self.ranked_one_rows_elided
    }

    /// Whether the ranked residual rows use the physical 29-line block
    /// layout. This is an internal producer/consumer address contract only;
    /// the proof and the round-one algebra are unchanged.
    #[inline]
    pub fn ranked_compact(&self) -> bool {
        self.ranked_compact
    }

    /// Restore dense AB when a consumer cannot add the identity-C one rows.
    /// This is a cold fallback for representation/consumer gate mismatches;
    /// the ranked identity-C path never calls it.
    pub(crate) fn restore_full_if_ranked_one_rows_elided(
        &mut self,
        a_packed: &[u8],
        b_packed: &[u8],
        inv_table: &InvNttTableByteSingleGf8,
    ) {
        if !self.ranked_one_rows_elided {
            return;
        }
        // Validate before the existing prefix repair reaches raw kernels.
        assert_eq!(a_packed.len(), self.dense_len_bytes);
        assert_eq!(b_packed.len(), self.dense_len_bytes);
        assert_eq!(inv_table.k, K_SKIP);
        if self.ranked_compact {
            let dense = crate::scratch::take_f128(
                self.dense_len_bytes / core::mem::size_of::<F128>(),
            );
            crate::scratch::give_f128(core::mem::replace(&mut self.storage, dense));
            self.ranked_compact = false;
        }
        self.invalid_prefix_bytes = self.dense_len_bytes;
        self.fill_invalid_prefix(a_packed, b_packed, inv_table);
        self.ranked_one_rows_elided = false;
    }

    /// Declare the leading `bytes` of the storage unwritten (the producer
    /// skipped them because round 1's GPU share covers those x_hi windows
    /// from raw a/b). Round 1 recomputes them on CPU if the GPU share
    /// doesn't materialize.
    pub fn set_invalid_prefix_bytes(&mut self, bytes: usize) {
        assert!(bytes <= self.dense_len_bytes);
        assert!(!self.ranked_compact || bytes == 0);
        self.invalid_prefix_bytes = bytes;
    }

    /// See [`Self::set_invalid_prefix_bytes`].
    pub fn invalid_prefix_bytes(&self) -> usize {
        self.invalid_prefix_bytes
    }

    /// Recompute the invalid prefix from the raw packed a/b buffers (dense
    /// windows, byte-identical to the streaming producer's
    /// [`precompute_round1_ab_inner_windows`]), then mark the storage fully
    /// valid. CPU fallback for when the planned GPU share fails.
    fn fill_invalid_prefix(
        &mut self,
        a_packed: &[u8],
        b_packed: &[u8],
        inv_table: &InvNttTableByteSingleGf8,
    ) {
        let n = self.invalid_prefix_bytes;
        if n == 0 {
            return;
        }
        use rayon::prelude::*;
        const OUTER_BYTES: usize = (1 << N_MEDIUM) * 64;
        assert_eq!(n % OUTER_BYTES, 0);
        let out = &mut self.as_bytes_mut()[..n];
        out.par_chunks_mut(OUTER_BYTES).enumerate().for_each_init(
            || ([F8::ZERO; ELL], [F8::ZERO; ELL]),
            |(a_col, b_col), (x_outer, out_outer)| {
                shift_reduce_windows_into_blocks(
                    a_packed,
                    b_packed,
                    inv_table,
                    x_outer * OUTER_BYTES,
                    1 << N_MEDIUM,
                    out_outer,
                    a_col,
                    b_col,
                    None,
                    0,
                );
            },
        );
        self.invalid_prefix_bytes = 0;
    }

    /// Mutable byte view for a challenge-independent witness generator. Every
    /// byte must be overwritten before the transform is consumed.
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        unsafe {
            core::slice::from_raw_parts_mut(
                self.storage.as_mut_ptr() as *mut u8,
                self.storage.len() * core::mem::size_of::<F128>(),
            )
        }
    }

    /// Logical dense extent used by protocol-shape checks. The physical
    /// ranked-compact allocation can be smaller; callers that need its actual
    /// storage iterate [`Self::as_bytes`] or [`Self::as_bytes_mut`].
    pub fn len_bytes(&self) -> usize {
        self.dense_len_bytes
    }
}

/// Exact same-binary rollback for the ranked 29-row physical layout.
pub fn ranked_ab_compact_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_R1_AB_COMPACT").is_none());
    *ON
}

/// Exact same-binary rollback for ranked complete-one-row reuse.
pub fn ranked_one_rows_reuse_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_R1_ONE_REUSE").is_none());
    *ON
}

impl Drop for Round1AbInner {
    fn drop(&mut self) {
        crate::scratch::give_f128(core::mem::take(&mut self.storage));
    }
}

/// Precompute the challenge-independent inverse-NTT/product/shift-reduce AB
/// transform.  The result can be produced before the commitment root is
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
    precompute_round1_ab_inner_packed_padded_impl(
        a_packed, b_packed, m, k_skip, inv_table, padding, true,
    )
}

/// Test oracle: [`precompute_round1_ab_inner_packed_padded`] with the static-B
/// round-1 kernel hint forced off, i.e. the incumbent kernel on every window.
/// Bit-identical to the production entry by construction; exists so
/// downstream tests can compare the two on a real witness.
#[doc(hidden)]
pub fn precompute_round1_ab_inner_packed_padded_reference(
    a_packed: &[u8],
    b_packed: &[u8],
    m: usize,
    k_skip: usize,
    inv_table: &InvNttTableByteSingleGf8,
    padding: &PaddingSpec,
) -> Round1AbInner {
    precompute_round1_ab_inner_packed_padded_impl(
        a_packed, b_packed, m, k_skip, inv_table, padding, false,
    )
}

#[allow(clippy::too_many_arguments)]
fn precompute_round1_ab_inner_packed_padded_impl(
    a_packed: &[u8],
    b_packed: &[u8],
    m: usize,
    k_skip: usize,
    inv_table: &InvNttTableByteSingleGf8,
    padding: &PaddingSpec,
    allow_bstatic: bool,
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
    const OUTER_BYTES: usize = (1 << N_MEDIUM) * 64;
    debug_assert_eq!(OUTER_BYTES, (1 << N_INNER) * N_CHUNKS);
    // Static-B plan is censused for the ranked BLAKE3 layout only; anything
    // else keeps the incumbent kernel (the plan would just miss anyway).
    let bstatic_ctx = if allow_bstatic && blake3_static_layout(padding) {
        kernels::prepare_bstatic(inv_table)
    } else {
        None
    };

    // Reuse an A-sized resident F128 allocation from the prover scratch pool.
    // Treating it as bytes is valid because every byte is written below before
    // the storage is read (including explicit zero writes for padding holes).
    let mut storage = crate::scratch::take_f128(total_bytes / core::mem::size_of::<F128>());
    let out_bytes: &mut [u8] =
        unsafe { core::slice::from_raw_parts_mut(storage.as_mut_ptr() as *mut u8, total_bytes) };

    out_bytes
        .par_chunks_mut(OUTER_BYTES)
        .enumerate()
        .for_each_init(
            || ([F8::ZERO; ELL], [F8::ZERO; ELL]),
            |(a_col, b_col), (x_outer, out_outer)| {
                let within_hash_outer = x_outer & within_outer_mask;
                let n_b_med = b_med_counts[within_hash_outer] as usize;
                let chunk_byte_base = x_outer * OUTER_BYTES;

                shift_reduce_windows_into_blocks(
                    a_packed,
                    b_packed,
                    inv_table,
                    chunk_byte_base,
                    n_b_med,
                    out_outer,
                    a_col,
                    b_col,
                    bstatic_ctx.map(|p| (within_hash_outer, p)),
                    0,
                );
                out_outer[n_b_med * 64..].fill(0);
            },
        );

    Round1AbInner {
        storage,
        dense_len_bytes: total_bytes,
        invalid_prefix_bytes: 0,
        ranked_one_rows_elided: false,
        ranked_compact: false,
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
    bstatic: kernels::BstaticHint,
    nt: u8,
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
        bstatic,
        nt,
    );
}

/// True for the ranked BLAKE3 witness layout the static-B plan was censused
/// on (`k_log = 14`, `useful_bits = 15409`).
#[inline]
fn blake3_static_layout(padding: &PaddingSpec) -> bool {
    padding.k_log == 14 && padding.useful_bits_per_block == 15_409
}

/// Run `shift_reduce_inner_ab` + C-side `bit_transpose_64bytes` for the
/// `n_b_med` medium windows of one `x_outer`, routing b_med window PAIRS
/// `(w, w + 1)` through the two-window wavefront kernel
/// ([`kernels::shift_reduce_inner_ab_x2`]) and the odd tail window (n_b_med
/// is 16 or 15 at the ranked shape: 15 = 7 pairs + 1 single) through the
/// single-window path. Every output byte lands in the same
/// `chunk_ab_bytes[b_med]` / `chunk_c_bytes[b_med]` slot as the previous
/// one-window-per-call loop, bit for bit.
#[inline]
#[allow(clippy::too_many_arguments)]
fn shift_reduce_transpose_windows(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    n_b_med: usize,
    chunk_ab_bytes: &mut [[u8; 64]; 1 << N_MEDIUM],
    chunk_c_bytes: &mut [[u8; 64]; 1 << N_MEDIUM],
    a_col: &mut [F8],
    b_col: &mut [F8],
    bstatic: kernels::BstaticHint,
) {
    let mut b_med = 0;
    while b_med + 1 < n_b_med {
        let (lo, hi) = chunk_ab_bytes.split_at_mut(b_med + 1);
        kernels::shift_reduce_inner_ab_x2(
            a_packed,
            b_packed,
            inv_table,
            chunk_byte_base,
            b_med,
            &mut lo[b_med],
            &mut hi[0],
            a_col,
            b_col,
            bstatic,
            // Fold-time staging arrays are re-read L1-hot immediately:
            // always temporal.
            0,
        );
        for w in b_med..b_med + 2 {
            let byte_base_b = chunk_byte_base + w * N_CHUNKS * 8;
            let c_in: &[u8; 64] = (&c_packed[byte_base_b..byte_base_b + 64])
                .try_into()
                .expect("64 c-bytes per medium position");
            bit_transpose_64bytes(c_in, &mut chunk_c_bytes[w]);
        }
        b_med += 2;
    }
    if b_med < n_b_med {
        shift_reduce_inner_ab(
            a_packed,
            b_packed,
            inv_table,
            chunk_byte_base,
            b_med,
            &mut chunk_ab_bytes[b_med],
            a_col,
            b_col,
            bstatic,
            0,
        );
        let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;
        let c_in: &[u8; 64] = (&c_packed[byte_base_b..byte_base_b + 64])
            .try_into()
            .expect("64 c-bytes per medium position");
        bit_transpose_64bytes(c_in, &mut chunk_c_bytes[b_med]);
    }
}

/// AB-only pair-loop twin of [`shift_reduce_transpose_windows`] for the
/// precompute paths, where the transformed blocks land in `n_b_med`
/// contiguous 64-byte slots of `out_outer` instead of `chunk_ab_bytes` (and
/// C is handled later, at eq-fold time). Window pairs go through
/// [`kernels::shift_reduce_inner_ab_x2`], the odd tail through the
/// single-window kernel — byte-identical to the sequential loop.
#[inline]
#[allow(clippy::too_many_arguments)]
fn shift_reduce_windows_into_blocks(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    n_b_med: usize,
    out_outer: &mut [u8],
    a_col: &mut [F8],
    b_col: &mut [F8],
    bstatic: kernels::BstaticHint,
    nt: u8,
) {
    let mut b_med = 0;
    while b_med + 1 < n_b_med {
        let (blk0, rest) = out_outer[b_med * 64..].split_at_mut(64);
        let out0: &mut [u8; 64] = blk0.try_into().expect("one transformed b_med block");
        let out1: &mut [u8; 64] = (&mut rest[..64])
            .try_into()
            .expect("one transformed b_med block");
        kernels::shift_reduce_inner_ab_x2(
            a_packed,
            b_packed,
            inv_table,
            chunk_byte_base,
            b_med,
            out0,
            out1,
            a_col,
            b_col,
            bstatic,
            nt,
        );
        b_med += 2;
    }
    if b_med < n_b_med {
        let dst: &mut [u8; 64] = (&mut out_outer[b_med * 64..(b_med + 1) * 64])
            .try_into()
            .expect("one transformed b_med block");
        shift_reduce_inner_ab(
            a_packed,
            b_packed,
            inv_table,
            chunk_byte_base,
            b_med,
            dst,
            a_col,
            b_col,
            bstatic,
            nt,
        );
    }
}

/// `FLOCK_NO_ABINNER_NT=1` restores temporal stores for the streaming
/// ab_inner publish (exact same-binary A/B control). Resolved once per
/// process.
pub fn abinner_nt_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_ABINNER_NT").is_none());
    *ON
}

/// Drain the producing thread's write-combining buffers so non-temporally
/// published ab_inner bytes are ordered before the rayon task's release
/// store. Callers that passed `nt_out = true` to
/// [`precompute_round1_ab_inner_windows`] MUST call this once per parallel
/// task, after its last window. No-op off x86.
#[inline]
pub fn abinner_publish_fence() {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: sfence is always available on x86_64.
    unsafe {
        core::arch::x86_64::_mm_sfence();
    }
}

/// Transform complete 8192-bit outer windows from packed A/B into the
/// challenge-independent round-one representation. This streaming seam lets
/// witness generation consume each just-written block while it is still hot.
///
/// `nt_out = true` publishes the transformed blocks with non-temporal stores:
/// `out` is a slice of the 512 MiB (ranked shape) ab_inner buffer whose next
/// reader is zerocheck round 1 — after the entire commit phase, DRAM-cold —
/// so the write-allocate RFO on every published line is pure waste. The
/// blocks are emitted strictly sequentially per producer thread (one open
/// write-combining stream). Callers passing `true` MUST issue
/// [`abinner_publish_fence`] on the producing thread before the task ends
/// (see there). Pass `false` for any destination that is re-read while hot.
pub fn precompute_round1_ab_inner_windows(
    a_packed: &[u8],
    b_packed: &[u8],
    out: &mut [u8],
    inv_table: &InvNttTableByteSingleGf8,
    nt_out: bool,
) {
    const OUTER_BYTES: usize = (1 << N_MEDIUM) * 64;
    assert_eq!(a_packed.len(), b_packed.len());
    assert_eq!(a_packed.len(), out.len());
    assert_eq!(a_packed.len() % OUTER_BYTES, 0);
    assert_eq!(inv_table.k, K_SKIP);

    // Every 64-byte block shares the base pointer's residue (offsets are
    // multiples of 64), so one classification covers the whole call: ZMM
    // streams at 64-alignment, four XMM streams at the pool's usual
    // 16-mod-64, temporal otherwise.
    let nt: u8 = if nt_out {
        match out.as_ptr() as usize % 64 {
            0 => 2,
            r if r % 16 == 0 => 1,
            _ => 0,
        }
    } else {
        0
    };

    let mut a_col = [F8::ZERO; ELL];
    let mut b_col = [F8::ZERO; ELL];
    // This seam is fed one BLAKE3 block (two 8192-bit windows) at a time by
    // the witness generator, so the window's parity is its BLAKE3 outer-window
    // index `w`. That is only a static-B plan hint — a different layout just
    // misses the plan's byte checks and runs the generic rows.
    let bstatic_ctx = kernels::prepare_bstatic(inv_table);
    for outer in 0..a_packed.len() / OUTER_BYTES {
        let base = outer * OUTER_BYTES;
        shift_reduce_windows_into_blocks(
            a_packed,
            b_packed,
            inv_table,
            base,
            1 << N_MEDIUM,
            &mut out[base..base + OUTER_BYTES],
            &mut a_col,
            &mut b_col,
            bstatic_ctx.map(|p| (outer & 1, p)),
            nt,
        );
    }
}

/// Number of 64-byte medium windows in one BLAKE3 block's round-1 transform
/// (two 8192-bit outer windows of `1 << N_MEDIUM` medium positions each).
pub const ROUND1_AB_WINDOWS_PER_BLOCK: usize = 2 * (1 << N_MEDIUM);

/// Resolved round-1 plan for [`round1_ab_inner_window`], hoisted out of the
/// per-window call so the streaming producer resolves it once per buffer.
#[derive(Clone, Copy)]
pub struct Round1AbWindowPlan {
    bstatic: Option<&'static kernels::BstaticPartials>,
    kernel: kernels::ShiftReducePlan,
    nt: u8,
    bcomplement_static: bool,
}

impl Round1AbWindowPlan {
    /// True when the caller has independently established the ranked BLAKE3
    /// static-B geometry and may use the checked-table complement leaf.
    #[inline]
    pub fn bcomplement_static_eligible(&self) -> bool {
        self.bcomplement_static
    }

    /// This plan specialised to one medium-window index: the static-B partials
    /// are dropped for every window whose plan is not live, so a producer that
    /// transforms several blocks at the same window index resolves the
    /// window's static-B eligibility once instead of once per block.
    #[inline]
    pub fn for_window(self, blk: usize) -> Self {
        Self {
            bstatic: if kernels::bstatic_window_live(blk) {
                self.bstatic
            } else {
                None
            },
            ..self
        }
    }
}

/// The inverse-NTT table images the round-1 AB kernel addresses under `plan`,
/// resolved once for a run of windows over one table rather than re-derived
/// from the table struct inside every window's applies.
#[derive(Clone, Copy)]
pub struct Round1AbTableImages(*const u8, *const u8);

/// Resolve [`Round1AbTableImages`] for `inv_table` under `plan`.
pub fn round1_ab_table_images(
    inv_table: &InvNttTableByteSingleGf8,
    plan: Round1AbWindowPlan,
) -> Round1AbTableImages {
    // The σ₈ image only exists on the architectures that build it, and only a
    // kernel that reads it asks for the pointers.
    #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
    if plan.kernel.uses_images() {
        let (base, base8) = inv_table.image_ptrs();
        return Round1AbTableImages(base, base8);
    }
    let _ = (inv_table, plan);
    Round1AbTableImages(core::ptr::null(), core::ptr::null())
}

/// Resolve the round-1 window plan for `inv_table` and the output allocation.
/// Every window destination is 64 bytes and every caller offset is a multiple
/// of 64, so the output's streaming-store mode is invariant for the complete
/// transform and need not be reclassified for every window. See
/// [`round1_ab_inner_window`].
pub fn prepare_round1_ab_window_plan(
    inv_table: &InvNttTableByteSingleGf8,
    out: &[u8],
    nt_out: bool,
) -> Round1AbWindowPlan {
    let nt = if nt_out {
        match out.as_ptr() as usize % 64 {
            0 => 2,
            r if r % 16 == 0 => 1,
            _ => 0,
        }
    } else {
        0
    };
    let kernel = kernels::prepare_shift_reduce(inv_table);
    Round1AbWindowPlan {
        bstatic: kernels::prepare_bstatic(inv_table),
        kernel,
        nt,
        bcomplement_static: prepare_round1_bcomplement_static(inv_table, kernel),
    }
}

/// Validate the algebraic table identity used by the ranked-static
/// B-complement leaf. The witness-side caller separately proves the fixed-one
/// byte geometry; this plan only admits the exact table/kernel combination.
fn prepare_round1_bcomplement_static(
    inv_table: &InvNttTableByteSingleGf8,
    kernel: kernels::ShiftReducePlan,
) -> bool {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    {
        static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
            std::env::var_os("FLOCK_NO_R1_B_COMPLEMENT_STATIC").is_none()
        });
        if !*ON
            || !kernels::shift_reduce_offsets_eligible(kernel, false, 2)
            || inv_table.k != 6
            || inv_table.ell != 64
            || inv_table.n_chunks != 8
        {
            return false;
        }
        let mut ones_image = [F8::ZERO; 64];
        inv_table.apply(&[u8::MAX; 8], &mut ones_image);
        ones_image.iter().all(|value| *value == F8::ONE)
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    )))]
    {
        let _ = (inv_table, kernel);
        false
    }
}

/// Transform ONE 64-byte medium window of one BLAKE3 block, given just that
/// window's packed a and b bytes. `blk` is the window's index within the
/// block, `0..ROUND1_AB_WINDOWS_PER_BLOCK`, ascending in packed byte order.
///
/// The whole-block entry point [`precompute_round1_ab_inner_windows`] is
/// exactly this call for every `blk` in order; each window is independent, so
/// a producer that materializes windows out of order gets the same bytes.
///
/// The output policy prepared in `plan` publishes the transformed window
/// under the same contract as [`precompute_round1_ab_inner_windows`]: when
/// non-temporal, the caller MUST issue [`abinner_publish_fence`] on the
/// producing thread before the task ends.
#[inline]
pub fn round1_ab_inner_window(
    a_window: &[u8; 64],
    b_window: &[u8; 64],
    out: &mut [u8; 64],
    blk: usize,
    inv_table: &InvNttTableByteSingleGf8,
    plan: Round1AbWindowPlan,
) {
    let imgs = round1_ab_table_images(inv_table, plan);
    // SAFETY: `imgs` was just resolved from this `inv_table` and `plan`.
    unsafe {
        round1_ab_inner_window_with_images(a_window, b_window, out, blk, inv_table, plan, imgs);
    }
}

/// [`round1_ab_inner_window`] with the table images already resolved. Same
/// bytes; the images are process-invariant, so a producer that transforms many
/// windows over one table resolves them once and passes them in.
///
/// # Safety
/// As for [`round1_ab_inner_window`], and `imgs` must be
/// [`round1_ab_table_images`] of this `inv_table` and `plan`.
#[inline]
pub unsafe fn round1_ab_inner_window_with_images(
    a_window: &[u8; 64],
    b_window: &[u8; 64],
    out: &mut [u8; 64],
    blk: usize,
    inv_table: &InvNttTableByteSingleGf8,
    plan: Round1AbWindowPlan,
    imgs: Round1AbTableImages,
) {
    kernels::shift_reduce_inner_ab_at(
        a_window,
        b_window,
        inv_table,
        0,
        blk,
        out,
        plan.bstatic,
        plan.kernel,
        plan.nt,
        (imgs.0, imgs.1),
    );
}

/// Ranked default collapses window 30's eight-K Horner to the live K0
/// product. `FLOCK_NO_R1_WIN30_K0=1` restores the generic eight-K body.
#[inline]
pub fn r1_win30_k0_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_R1_WIN30_K0").is_none())
}

/// Window 30: only the first eight bytes of A and B are live. Same bytes as
/// [`round1_ab_inner_window_with_images`] for this geometry because y_1..y_7
/// are zero.
///
/// # Safety
/// As [`round1_ab_inner_window_with_images`], with `out` 64-byte aligned for
/// the ranked nt=2 stream store. Caller sfence.
#[inline]
pub unsafe fn round1_ab_inner_window30_k0(
    a_window: &[u8; 64],
    b_window: &[u8; 64],
    out: &mut [u8; 64],
    inv_table: &InvNttTableByteSingleGf8,
    plan: Round1AbWindowPlan,
    imgs: Round1AbTableImages,
    blk: usize,
) {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    if r1_win30_k0_enabled() && plan.nt == 2 && plan.kernel.uses_images() {
        unsafe {
            kernels::x86_64::shift_reduce_inner_ab_x86_avx512_window30_k0(
                a_window,
                b_window,
                out,
                (imgs.0, imgs.1),
            );
        }
        return;
    }
    unsafe {
        round1_ab_inner_window_with_images(
            a_window, b_window, out, blk, inv_table, plan, imgs,
        );
    }
}

/// `u16` count of one window-block's pre-scaled offset block for
/// [`round1_ab_inner_window_from_offsets`].
pub const ROUND1_AB_OFF_WORDS: usize = 128;

impl Round1AbWindowPlan {
    /// True when [`round1_ab_inner_window_from_offsets`] may replace
    /// [`round1_ab_inner_window_with_images`] for window `blk` under this
    /// plan (the x86 pidx+offw body on a non-static-B window). The split
    /// form is bit-identical there.
    #[inline]
    pub fn offsets_eligible(&self, blk: usize) -> bool {
        kernels::shift_reduce_offsets_eligible(self.kernel, self.bstatic.is_some(), blk)
    }
}

/// Build one window-block's [`ROUND1_AB_OFF_WORDS`] pre-scaled `u16` offsets
/// from its packed a/b window bytes — the prologue of the pidx kernel, split
/// out so a producer can build several blocks' offsets before consuming any.
///
/// # Safety
/// `off` must be 64-byte aligned. Only meaningful when a plan's
/// [`Round1AbWindowPlan::offsets_eligible`] holds (x86 AVX-512+GFNI builds).
#[inline]
#[allow(unused_variables)]
pub unsafe fn round1_ab_window_offsets(
    a_window: &[u8; 64],
    b_window: &[u8; 64],
    off: &mut [u16; ROUND1_AB_OFF_WORDS],
) {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    // SAFETY: forwarded from this function's contract.
    unsafe {
        kernels::x86_64::shift_reduce_ab_offsets_build(
            a_window.as_ptr(),
            b_window.as_ptr(),
            off.as_mut_ptr(),
        );
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    )))]
    unreachable!("offsets form is x86 AVX-512+GFNI only; gate on offsets_eligible");
}

/// [`round1_ab_inner_window_with_images`] fed from offsets prebuilt by
/// [`round1_ab_window_offsets`]. Bit-identical bytes whenever
/// `plan.offsets_eligible(blk)` — identical table addresses, arithmetic and
/// store class; the only difference is WHEN the offset stores happen.
///
/// # Safety
/// As for [`round1_ab_inner_window_with_images`], with `off` built from this
/// window-block's a/b bytes and `plan.offsets_eligible(blk)` true.
#[inline]
#[allow(unused_variables)]
pub unsafe fn round1_ab_inner_window_from_offsets(
    off: &[u16; ROUND1_AB_OFF_WORDS],
    out: &mut [u8; 64],
    plan: Round1AbWindowPlan,
    imgs: Round1AbTableImages,
) {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    // SAFETY: forwarded from this function's contract.
    unsafe {
        kernels::x86_64::shift_reduce_inner_ab_x86_avx512_from_off(
            off.as_ptr(),
            out,
            plan.nt,
            (imgs.0, imgs.1),
        );
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    )))]
    unreachable!("offsets form is x86 AVX-512+GFNI only; gate on offsets_eligible");
}

/// Fixed-ZMM-stream-store twin of [`round1_ab_inner_window_from_offsets`] for
/// the measured ranked path. It computes identical bytes but specializes the
/// already-resolved `nt=2` destination class, so the terminal store has no
/// runtime selector. Generic/cold/static callers retain the general wrapper.
///
/// # Safety
/// As for [`round1_ab_inner_window_from_offsets`], with `plan.nt == 2` and
/// `out` 64-byte aligned. The producing thread must execute
/// [`abinner_publish_fence`] before publishing the output across threads.
#[inline]
#[allow(unused_variables)]
pub unsafe fn round1_ab_inner_window_from_offsets_nt2(
    off: &[u16; ROUND1_AB_OFF_WORDS],
    out: &mut [u8; 64],
    plan: Round1AbWindowPlan,
    imgs: Round1AbTableImages,
) {
    debug_assert_eq!(plan.nt, 2);
    debug_assert_eq!(out.as_ptr() as usize & 63, 0);
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    // SAFETY: forwarded from this function's contract.
    unsafe {
        kernels::x86_64::shift_reduce_inner_ab_x86_avx512_from_off_nt2(
            off.as_ptr(),
            out,
            (imgs.0, imgs.1),
        );
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    )))]
    unreachable!("nt2 offsets form is x86 AVX-512+GFNI only");
}

/// Ranked-static B-complement twin of
/// [`round1_ab_inner_window_from_offsets_nt2`]. The caller must have proved
/// the BLAKE3 fixed-one byte geometry and checked
/// [`Round1AbWindowPlan::bcomplement_static_eligible`] for this plan.
///
/// # Safety
/// As for [`round1_ab_inner_window_from_offsets_nt2`], plus the static-byte
/// geometry above. This wrapper does no per-window raw-B guard by design.
#[inline(always)]
#[allow(unused_variables)]
pub unsafe fn round1_ab_inner_window_from_offsets_nt2_bcomplement_static(
    off: &[u16; ROUND1_AB_OFF_WORDS],
    out: &mut [u8; 64],
    plan: Round1AbWindowPlan,
    imgs: Round1AbTableImages,
    blk: usize,
) {
    debug_assert!(plan.bcomplement_static);
    debug_assert_eq!(plan.nt, 2);
    debug_assert_eq!(out.as_ptr() as usize & 63, 0);
    debug_assert!((2..=29).contains(&blk));
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    unsafe {
        kernels::x86_64_bcomplement::shift_reduce_bcomplement_offw_nt2(
            off.as_ptr(),
            out,
            (imgs.0, imgs.1),
            blk,
        );
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    )))]
    unreachable!("ranked-static B-complement is x86 AVX-512+GFNI only");
}

#[inline(always)]
#[allow(unused_variables)]
pub unsafe fn round1_ab_inner_window_from_offsets_nt2_bcomplement_static_const<
    const BLK: usize,
>(
    off: &[u16; ROUND1_AB_OFF_WORDS],
    out: &mut [u8; 64],
    plan: Round1AbWindowPlan,
    imgs: Round1AbTableImages,
) {
    debug_assert!(plan.bcomplement_static);
    debug_assert_eq!(plan.nt, 2);
    debug_assert_eq!(out.as_ptr() as usize & 63, 0);
    debug_assert!((3..=28).contains(&BLK));
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    unsafe {
        kernels::x86_64_bcomplement::shift_reduce_bcomplement_offw_nt2_const::<BLK>(
            off.as_ptr(),
            out,
            (imgs.0, imgs.1),
        );
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    )))]
    unreachable!("ranked-static B-complement is x86 AVX-512+GFNI only");
}

/// Residual twin for the two ranked windows containing complete B=1 K-rows.
/// `keep` is `0xfc` for block 2 and `0x0f` for block 29.
#[inline(always)]
#[allow(unused_variables)]
pub unsafe fn round1_ab_inner_window_from_offsets_nt2_residual(
    off: &[u16; ROUND1_AB_OFF_WORDS],
    out: &mut [u8; 64],
    plan: Round1AbWindowPlan,
    imgs: Round1AbTableImages,
    keep: u8,
) {
    debug_assert_eq!(plan.nt, 2);
    debug_assert_eq!(out.as_ptr() as usize & 63, 0);
    debug_assert!(keep == 0xfc || keep == 0x0f);
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    unsafe {
        kernels::x86_64::shift_reduce_inner_ab_x86_avx512_from_off_nt2_residual(
            off.as_ptr(),
            out,
            (imgs.0, imgs.1),
            keep,
        );
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    )))]
    unreachable!("residual nt2 offsets form is x86 AVX-512+GFNI only");
}

/// Bytes of the leading ab_inner prefix that a challenge-independent witness
/// producer may SKIP because round 1's GPU URM share is planned to cover
/// those x_hi windows from the raw a/b buffers (the CPU fold never reads the
/// precomputed transform there). Returns 0 whenever the GPU share cannot
/// engage (no Metal, small m, kill switches), so skipping is always safe:
/// the producer marks the prefix via
/// [`Round1AbInner::set_invalid_prefix_bytes`], and round 1 recomputes it on
/// CPU if the GPU share fails to materialize.
///
/// `FLOCK_NO_AB_INNER_SKIP=1` force-disables the skip (diagnostics).
pub fn planned_round1_gpu_prefix_bytes(m: usize) -> usize {
    if m < K_SKIP + N_INNER || std::env::var_os("FLOCK_NO_AB_INNER_SKIP").is_some() {
        return 0;
    }
    let n_outer = m - K_SKIP - N_INNER;
    let n_hi = n_outer.min(SplitEqGhash::MAX_N_HI);
    // planned_g only engages at hi_size = 128; anything else → no skip.
    if (1usize << n_hi) != 128 {
        return 0;
    }
    let g = crate::gpu::urm::planned_g(1 << n_hi, m);
    g * (((1usize << m) / 8) >> n_hi)
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
            shift_reduce_transpose_windows(
                a_packed,
                b_packed,
                c_packed,
                inv_table,
                chunk_byte_base,
                1 << N_MEDIUM,
                &mut state.chunk_ab_bytes,
                &mut state.chunk_c_bytes,
                &mut state.a_col,
                &mut state.b_col,
                None,
            );

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
            shift_reduce_transpose_windows(
                a_packed,
                b_packed,
                c_packed,
                inv_table,
                chunk_byte_base,
                n_b_med,
                &mut state.chunk_ab_bytes,
                &mut state.chunk_c_bytes,
                &mut state.a_col,
                &mut state.b_col,
                None,
            );

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
// DirectC eight-bank capture.
// ---------------------------------------------------------------------------

const N_C_BANKS: usize = 8;

pub(crate) struct WorkerStateWithSHatV {
    partial_ab: [F128; ELL],
    // Boxed: the eight DirectC banks are ELL * 16 * 8 = 8 KiB each, and this
    // state travels BY VALUE through rayon's fold/reduce plumbing (which
    // materializes several copies per split level in debug builds). With the
    // arrays inline the state is ~20 KiB and overflows the fixed ~8 MiB
    // test-thread stack; heap-allocating keeps the carried state ~4 KiB —
    // smaller than the pre-DirectC two-bank layout that tested clean.
    partial_c: Box<[[F128; ELL]; N_C_BANKS]>,
    chunk_ab_bytes: [[u8; 64]; 1 << N_MEDIUM],
    chunk_c_bytes: [[u8; 64]; 1 << N_MEDIUM],
    a_col: [F8; ELL],
    b_col: [F8; ELL],
    pub(crate) local_res_ab: [F128; ELL],
    pub(crate) local_res_c_s: Box<[[F128; ELL]; N_C_BANKS]>,
}

impl WorkerStateWithSHatV {
    pub(crate) fn new() -> Self {
        Self {
            partial_ab: [F128::ZERO; ELL],
            partial_c: Box::new([[F128::ZERO; ELL]; N_C_BANKS]),
            chunk_ab_bytes: [[0u8; 64]; 1 << N_MEDIUM],
            chunk_c_bytes: [[0u8; 64]; 1 << N_MEDIUM],
            a_col: [F8::ZERO; ELL],
            b_col: [F8::ZERO; ELL],
            local_res_ab: [F128::ZERO; ELL],
            local_res_c_s: Box::new([[F128::ZERO; ELL]; N_C_BANKS]),
        }
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn process_one_x_hi_with_s_hat_v(
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
    state.partial_ab.fill(F128::ZERO);
    for bank in state.partial_c.iter_mut() {
        bank.fill(F128::ZERO);
    }
    let n_lo = n_lo_and_inner - N_INNER;
    for x_outer_lo in 0..big_lo_size {
        let x_outer = x_outer_lo | (x_hi << n_lo);
        let n_b_med = b_med_counts[x_outer & within_outer_mask] as usize;
        if n_b_med == 0 {
            continue;
        }
        let chunk_byte_base = ((x_outer_lo << N_INNER) | (x_hi << n_lo_and_inner)) * N_CHUNKS;
        shift_reduce_transpose_windows(
            a_packed,
            b_packed,
            c_packed,
            inv_table,
            chunk_byte_base,
            n_b_med,
            &mut state.chunk_ab_bytes,
            &mut state.chunk_c_bytes,
            &mut state.a_col,
            &mut state.b_col,
            None,
        );
        kernels::accumulate_convert_ab(
            &state.chunk_ab_bytes,
            n_b_med,
            convert,
            eq_lo_scaled[x_outer_lo],
            &mut state.partial_ab,
        );
        // SAFETY: `[[u8; 64]; 16]` is contiguous and has exactly 1024 bytes.
        let c_block: &[u8; 16 * 64] =
            unsafe { &*state.chunk_c_bytes.as_ptr().cast::<[u8; 16 * 64]>() };
        let c_tables =
            &mask_tables[x_outer_lo * C_MASK_TABLE_STRIDE..(x_outer_lo + 1) * C_MASK_TABLE_STRIDE];
        kernels::accumulate_c_banks(c_block, n_b_med, c_tables, &mut state.partial_c);
    }
    for lane in 0..ELL {
        state.local_res_ab[lane] += eq_hi_val * state.partial_ab[lane];
        for bank in 0..N_C_BANKS {
            state.local_res_c_s[bank][lane] += eq_hi_val * state.partial_c[bank][lane];
        }
    }
}

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
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    c_nibble_luts: &[kernels::CBankNibbleLut],
    state: &mut WorkerStateWithSHatV,
) {
    state.partial_ab.fill(F128::ZERO);
    for bank in state.partial_c.iter_mut() {
        bank.fill(F128::ZERO);
    }
    let n_lo = n_lo_and_inner - N_INNER;
    for x_outer_lo in 0..big_lo_size {
        let x_outer = x_outer_lo | (x_hi << n_lo);
        let n_b_med = b_med_counts[x_outer & within_outer_mask] as usize;
        if n_b_med == 0 {
            continue;
        }
        let chunk_byte_base = ((x_outer_lo << N_INNER) | (x_hi << n_lo_and_inner)) * N_CHUNKS;
        for b_med in 0..n_b_med {
            let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;
            state.chunk_ab_bytes[b_med].copy_from_slice(&ab_inner[byte_base_b..byte_base_b + 64]);
            let c_in: &[u8; 64] = (&c_packed[byte_base_b..byte_base_b + 64])
                .try_into()
                .expect("64 c-bytes per medium position");
            bit_transpose_64bytes(c_in, &mut state.chunk_c_bytes[b_med]);
        }
        kernels::accumulate_convert_ab(
            &state.chunk_ab_bytes,
            n_b_med,
            convert,
            eq_lo_scaled[x_outer_lo],
            &mut state.partial_ab,
        );
        // SAFETY: `[[u8; 64]; 16]` is contiguous and has exactly 1024 bytes.
        let c_block: &[u8; 16 * 64] =
            unsafe { &*state.chunk_c_bytes.as_ptr().cast::<[u8; 16 * 64]>() };
        let c_tables =
            &mask_tables[x_outer_lo * C_MASK_TABLE_STRIDE..(x_outer_lo + 1) * C_MASK_TABLE_STRIDE];
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "vpclmulqdq"
        ))]
        kernels::accumulate_c_banks_prebuilt(
            c_block,
            n_b_med,
            c_tables,
            &c_nibble_luts[x_outer_lo],
            &mut state.partial_c,
        );
        #[cfg(not(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "vpclmulqdq"
        )))]
        kernels::accumulate_c_banks(c_block, n_b_med, c_tables, &mut state.partial_c);
    }
    for lane in 0..ELL {
        state.local_res_ab[lane] += eq_hi_val * state.partial_ab[lane];
        for bank in 0..N_C_BANKS {
            state.local_res_c_s[bank][lane] += eq_hi_val * state.partial_c[bank][lane];
        }
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
pub(crate) fn build_b_med_counts(padding: &PaddingSpec) -> (usize, Vec<u8>) {
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

fn finish_c_banks(banks: &[[F128; ELL]; N_C_BANKS]) -> ([F128; ELL], Vec<F128>, Vec<F128>) {
    let alpha = phi8(F8(0x02));
    let mut alpha_pow = [F128::ONE; N_C_BANKS];
    for k in 1..N_C_BANKS {
        alpha_pow[k] = alpha_pow[k - 1] * alpha;
    }
    let mut res_c_s_0 = [F128::ZERO; ELL];
    let mut res_c_s_1 = [F128::ZERO; ELL];
    for (k, bank) in banks.iter().enumerate() {
        let target = if k & 1 == 0 {
            &mut res_c_s_0
        } else {
            &mut res_c_s_1
        };
        for lane in 0..ELL {
            target[lane] += alpha_pow[k] * bank[lane];
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
/// Cost vs the original: per chunk-lane-`b_med`, +1 `vld1q_u8` + +1 `veorq_u8`
/// (the bank-split convert lookup). bit_transpose, shift_reduce, eq folds
/// are unchanged. See module-level docs for the F_2-linearity argument that
/// makes `s_hat_v_c[(λ, 0)] + s_hat_v_c[(λ, 1)] · α == res_c_s_opt[λ]`.
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
    let (ab, c, s, _) = round1_shift_reduce_extract_c_packed_padded_with_s_hat_v_quad(
        a_packed, b_packed, c_packed, m, k_skip, r, inv_table, padding,
    );
    (ab, c, s)
}

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
    round1_with_s_hat_v_impl(
        a_packed, b_packed, c_packed, m, k_skip, r, inv_table, padding, None,
    )
}

/// [`round1_shift_reduce_extract_c_packed_padded_with_s_hat_v`] with an
/// optional forced CPU/GPU split for tests (season-1 hook shape).
/// `g_override = Some(0)` forces pure CPU; `Some(g)` forces the GPU to take
/// `x_hi ∈ [0, g)` (falling back to CPU if Metal is unavailable); `None` is
/// the production auto split (Metal availability + calibration, see
/// [`crate::gpu::urm`]).
///
/// The CPU/GPU merge is a per-lane F128 XOR of eq_hi-folded partials —
/// bit-identical to the pure-CPU rayon reduction for any split point.
#[allow(clippy::too_many_arguments)]
pub(crate) fn round1_with_s_hat_v_impl(
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    k_skip: usize,
    r: &[F128],
    inv_table: &InvNttTableByteSingleGf8,
    padding: &PaddingSpec,
    g_override: Option<usize>,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<F128>) {
    use rayon::prelude::*;
    let _ = g_override;
    assert_eq!(k_skip, K_SKIP, "optimized variant is k_skip=6 only");
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
    let mask_tables = build_c_mask_tables(&eq_lo_scaled);
    let (within_outer_mask, b_med_counts) = build_b_med_counts(padding);
    let (res_ab, banks) = (0..hi_size)
        .into_par_iter()
        .fold(WorkerStateWithSHatV::new, |mut state, x_hi| {
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
                eq.hi[x_hi],
                convert,
                &mask_tables,
                &mut state,
            );
            state
        })
        .map(|s| (s.local_res_ab, s.local_res_c_s))
        .reduce(
            || ([F128::ZERO; ELL], Box::new([[F128::ZERO; ELL]; N_C_BANKS])),
            |(mut ab1, mut c1), (ab2, c2)| {
                for i in 0..ELL {
                    ab1[i] += ab2[i];
                }
                for (left, right) in c1.iter_mut().zip(c2.iter()) {
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

/// Challenge-weighted completion of round 1 using AB blocks returned by
/// [`precompute_round1_ab_inner_packed_padded`].  This is byte-identical to
/// [`round1_shift_reduce_extract_c_packed_padded_with_s_hat_v`], while keeping
/// the original A and B packed buffers available for zerocheck round 2.
///
/// `a_packed`/`b_packed` are the SAME original buffers the AB blocks were
/// precomputed from — the CPU share never touches them here, but the GPU
/// share (x_hi ∈ [0, g)) recomputes its range from a/b/c directly with the
/// season-1 URM kernel, which is bit-identical per x_hi to the precomputed
/// path, so the XOR merge is exact for any split point.
pub fn round1_shift_reduce_extract_c_packed_padded_with_precomputed_ab(
    ab_inner: &mut Round1AbInner,
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    k_skip: usize,
    r: &[F128],
    inv_table: &InvNttTableByteSingleGf8,
    padding: &PaddingSpec,
) -> (Vec<F128>, Vec<F128>, Vec<F128>) {
    let (ab, c, s, _) = round1_shift_reduce_extract_c_packed_padded_with_precomputed_ab_quad(
        ab_inner, a_packed, b_packed, c_packed, m, k_skip, r, inv_table, padding,
    );
    (ab, c, s)
}

pub fn round1_shift_reduce_extract_c_packed_padded_with_precomputed_ab_quad(
    ab_inner: &mut Round1AbInner,
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    k_skip: usize,
    r: &[F128],
    inv_table: &InvNttTableByteSingleGf8,
    padding: &PaddingSpec,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<F128>) {
    round1_with_precomputed_ab_impl(
        ab_inner, a_packed, b_packed, c_packed, m, k_skip, r, inv_table, padding, None,
    )
}

/// [`round1_shift_reduce_extract_c_packed_padded_with_precomputed_ab`] with
/// the season-1 `g_override` test hook (see [`round1_with_s_hat_v_impl`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn round1_with_precomputed_ab_impl(
    ab_inner: &mut Round1AbInner,
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    k_skip: usize,
    r: &[F128],
    inv_table: &InvNttTableByteSingleGf8,
    padding: &PaddingSpec,
    g_override: Option<usize>,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<F128>) {
    use rayon::prelude::*;
    let _ = g_override;
    assert_eq!(k_skip, K_SKIP, "optimized variant is k_skip=6 only");
    assert!(m >= k_skip + N_INNER);
    let total_bytes = (1usize << m) / 8;
    assert_eq!(ab_inner.len_bytes(), total_bytes);
    assert_eq!(a_packed.len(), total_bytes);
    assert_eq!(b_packed.len(), total_bytes);
    assert_eq!(c_packed.len(), total_bytes);
    assert_eq!(r.len(), m);
    assert_eq!(inv_table.k, k_skip);
    ab_inner.fill_invalid_prefix(a_packed, b_packed, inv_table);
    assert!(
        !ab_inner.ranked_compact(),
        "the fused C consumer requires dense AB storage"
    );
    let eq = SplitEqGhash::new(&r[k_skip + N_INNER..]);
    let big_lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    let n_lo_and_inner = eq.n_lo + N_INNER;
    let d_inv_val = d_inv();
    let eq_lo_scaled: Vec<F128> = eq.lo.iter().map(|v| *v * d_inv_val).collect();
    let convert = convert_table();
    let mask_tables = build_c_mask_tables(&eq_lo_scaled);
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    let c_nibble_luts = kernels::build_c_bank_nibble_luts(&mask_tables);
    let (within_outer_mask, b_med_counts) = build_b_med_counts(padding);
    let ab_inner_bytes = ab_inner.as_bytes();
    let (res_ab, banks) = (0..hi_size)
        .into_par_iter()
        .fold(WorkerStateWithSHatV::new, |mut state, x_hi| {
            process_one_x_hi_with_precomputed_ab(
                x_hi,
                big_lo_size,
                n_lo_and_inner,
                within_outer_mask,
                &b_med_counts,
                ab_inner_bytes,
                c_packed,
                &eq_lo_scaled,
                eq.hi[x_hi],
                convert,
                &mask_tables,
                #[cfg(all(
                    target_arch = "x86_64",
                    target_feature = "avx512f",
                    target_feature = "vpclmulqdq"
                ))]
                &c_nibble_luts,
                &mut state,
            );
            state
        })
        .map(|s| (s.local_res_ab, s.local_res_c_s))
        .reduce(
            || ([F128::ZERO; ELL], Box::new([[F128::ZERO; ELL]; N_C_BANKS])),
            |(mut ab1, mut c1), (ab2, c2)| {
                for i in 0..ELL {
                    ab1[i] += ab2[i];
                }
                for (left, right) in c1.iter_mut().zip(c2.iter()) {
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

// ---------------------------------------------------------------------------
// DirectFold4: 32-bank (q-retained) C capture for the sixteen-bank PCS open.
//
// The incumbent DirectC capture folds all four medium coordinates into the
// mask (`X^{b_med}`, γ = X, times `D⁻¹`). The direct-fold4 opening wants the
// low two medium coordinates `q = b_med & 3` RETAINED, so the producer emits
// 32 α-free banks `bank[q][K][lane] = Σ_x eq_lo(x)·D_hi⁻¹·Σ_j X^{4j}·bit_K(c[x][4j+q][lane])`
// (only the high-medium factor `X^{4j}` and its normalisation `D_hi⁻¹ =
// ((1+γ⁴)(1+γ⁸))⁻¹` are absorbed). Collapsing under `eq_med_lo(q) = X^q·D_lo⁻¹`
// gives back the incumbent eight banks EXACTLY (field distributivity, `D =
// D_lo·D_hi`), so the round-1 message / `s_hat_v_c` / `quad_c` are bit-identical.
//
// Four-block fusion: four consecutive `x_outer_lo` windows share one drain
// call. The synthetic 16-row block for retained `q` has row `4w + j` =
// transpose(c[x_lo+w][b_med = 4j+q]) and the synthetic 512-entry mask table
// has `t_lo[byte] = T4[x_lo][byte&15] + T4[x_lo+1][byte>>4]`,
// `t_hi[byte] = T4[x_lo+2][byte&15] + T4[x_lo+3][byte>>4]` with
// `T4[x][m] = (Σ_{j∈m} X^{4j})·eq_lo[x]·D_hi⁻¹` — so the UNCHANGED eight-bank
// drain kernels (`accumulate_c_banks*`) compute the 32-bank capture at exactly
// the incumbent's per-window cost (same lookups, same RMWs, same mask ops).
// ---------------------------------------------------------------------------

/// Number of retained low-medium values (`q = b_med & 3`).
pub(crate) const N_C_Q: usize = 4;

/// `D_hi⁻¹ = ((1+γ⁴)(1+γ⁸))⁻¹` — the high-medium half of [`d_inv`].
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
pub(crate) fn d_hi_inv() -> F128 {
    *D_HI_INV_CACHE.get_or_init(compute_d_hi_inv)
}

/// `eq_med_lo(q) = eq(β₀, q₀)·eq(β₁, q₁) = X^q · D_lo⁻¹` for `q ∈ 0..4`, in the
/// little-endian order of `build_eq(&[β₀, β₁])` — the weights that collapse
/// the four retained-q bank groups back to the incumbent eight banks.
pub(crate) fn c_fold4_q_weights() -> [F128; N_C_Q] {
    let beta = medium_challenges_ghash();
    super::univariate_skip::build_eq(&beta[..2])
        .try_into()
        .expect("two-coordinate eq has four entries")
}

/// Whether the four-window fold4 producer can run for this `eq` split.
#[inline]
pub(crate) fn c_fold4_capture_shape_ok(big_lo_size: usize) -> bool {
    big_lo_size.is_multiple_of(4)
}

/// Shape predicate the zerocheck driver uses to pick the fold4 producer:
/// the outer eq split (`SplitEqGhash::new(&r[k_skip + N_INNER..])`) must
/// leave a multiple-of-four low half (true for every `m >= 22`, in
/// particular the ranked `m = 32`).
pub fn c_fold4_capture_available(m: usize, k_skip: usize) -> bool {
    if k_skip != K_SKIP || m < k_skip + N_INNER {
        return false;
    }
    let n = m - k_skip - N_INNER;
    let n_lo = n - n.min(SplitEqGhash::MAX_N_HI);
    c_fold4_capture_shape_ok(1usize << n_lo)
}

/// Synthetic 512-entry mask tables, one per group of four consecutive
/// `x_outer_lo` windows (see the module comment above). Layout per group:
/// `[t_lo[256], t_hi[256]]`, exactly the shape `accumulate_c_banks*` expect.
fn build_c_fold4_tables(eq_lo: &[F128]) -> Vec<F128> {
    use rayon::prelude::*;
    assert!(eq_lo.len().is_multiple_of(4));
    let d_hi_inv_val = d_hi_inv();
    let mut tables = crate::scratch::take_f128((eq_lo.len() / 4) * C_MASK_TABLE_STRIDE);
    tables
        .par_chunks_mut(C_MASK_TABLE_STRIDE)
        .zip(eq_lo.par_chunks_exact(4))
        .for_each(|(slot, eqs)| {
            // T4[w][m] for the four windows: subset sums of X^{4j}·(eq_lo·D_hi⁻¹).
            let mut t4 = [[F128::ZERO; 16]; 4];
            for (w, &eq) in eqs.iter().enumerate() {
                let mut gens = [F128::ZERO; 4];
                gens[0] = eq * d_hi_inv_val;
                for j in 1..4 {
                    let mut v = gens[j - 1];
                    for _ in 0..4 {
                        v = mul_by_x(v);
                    }
                    gens[j] = v;
                }
                let t = &mut t4[w];
                t[0] = F128::ZERO;
                for j in 0..4 {
                    let (done, rest) = t.split_at_mut(1 << j);
                    for (out, seen) in rest[..1 << j].iter_mut().zip(done.iter()) {
                        *out = *seen + gens[j];
                    }
                }
            }
            let (t_lo, t_hi) = slot.split_at_mut(256);
            for byte in 0..256usize {
                t_lo[byte] = t4[0][byte & 15] + t4[1][byte >> 4];
                t_hi[byte] = t4[2][byte & 15] + t4[3][byte >> 4];
            }
        });
    tables
}

/// Ranked default is the fused GFNI C drain. `FLOCK_NO_ZC_C_GFNI=1` restores
/// the incumbent bit-transpose + nibble-LUT drain in the same binary (the
/// ranked worker's env is cleared, so the shipped behaviour is ON).
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
fn c_fold4_gfni_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("FLOCK_NO_ZC_C_GFNI").as_deref() != Some(std::ffi::OsStr::new("1"))
    })
}

/// Byte-plane C bank store, `[q][bank][byte-plane][lane]`.
pub(crate) const C_PLANE_BANK_BYTES: usize = N_C_Q * N_C_BANKS * 16 * ELL;
/// One four-window C group: four contiguous 1 KiB windows of `c_packed`.
pub(crate) const C_GROUP_BYTES: usize = 4 * (1 << N_MEDIUM) * ELL;
/// GFNI matrix qwords per four-window group: two mask halves × 16 output bytes.
#[allow(dead_code)] // used only by the fused GFNI C drain
pub(crate) const C_FOLD4_MATS_PER_GROUP: usize = 32;

/// GFNI bit-matrix form of [`build_c_fold4_tables`].
///
/// The synthetic fold4 mask table is F2-linear in the sixteen mask bits by
/// construction (`t_lo[byte] = t4[0][byte & 15] + t4[1][byte >> 4]`, each
/// `t4[w]` an XOR-doubling subset sum of four generators), so it never needs
/// expanding to 512 entries at all: each 8-bit half IS sixteen 8×8 bit
/// matrices — one per output byte of the F128 — built straight from its eight
/// basis entries `t4[2h + (b >> 2)][1 << (b & 3)]`. 256 bytes per group
/// instead of 8 KiB, and `VGF2P8AFFINEQB` evaluates one matrix on 64 lanes'
/// mask bytes per instruction with no table load at all. Matrix encoding
/// (hardware-verified, same as the AB drain): `out.bit[i] = parity(byte[7-i] & in)`.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
fn build_c_fold4_gfni_mats(eq_lo: &[F128]) -> Vec<u64> {
    assert!(eq_lo.len().is_multiple_of(4));
    let d_hi_inv_val = d_hi_inv();
    let mut mats = vec![0u64; (eq_lo.len() / 4) * C_FOLD4_MATS_PER_GROUP];
    // Deliberately serial. At the ranked shape this is 1024 groups x ~700 ops
    // = 0.12 ms, which is well under the ~1.2 ms it costs to wake the pool's
    // fifteen sleeping workers at this point in the prove (measured); the
    // 8 MiB table build it replaces paid that wake-up and then some.
    for (slot, eqs) in mats
        .chunks_mut(C_FOLD4_MATS_PER_GROUP)
        .zip(eq_lo.chunks_exact(4))
    {
        build_one_group_mats(slot, eqs, d_hi_inv_val);
    }
    mats
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
#[inline]
fn build_one_group_mats(slot: &mut [u64], eqs: &[F128], d_hi_inv_val: F128) {
    // gens[w][j] = X^{4j}·(eq_lo[w]·D_hi⁻¹): window `w`'s four subset-sum
    // generators, i.e. `t4[w][1 << j]`.
    let mut gens = [[F128::ZERO; 4]; 4];
    for (w, &eq) in eqs.iter().enumerate() {
        gens[w][0] = eq * d_hi_inv_val;
        for j in 1..4 {
            let mut v = gens[w][j - 1];
            for _ in 0..4 {
                v = mul_by_x(v);
            }
            gens[w][j] = v;
        }
    }
    for h in 0..2 {
        // Mask bit `b` of half `h` is drain row `8h + b`, i.e. window
        // `2h + (b >> 2)`'s medium row `4·(b & 3) + q`.
        let basis: [F128; 8] = std::array::from_fn(|b| gens[2 * h + (b >> 2)][b & 3]);
        for k in 0..16 {
            // Byte-column `k` of the basis: `col.byte[b] = basis[b].byte[k]`.
            // Its 8x8 bit transpose has `t.byte[i].bit[b] = basis[b].bit(8k+i)`,
            // which is the matrix row VGF2P8AFFINEQB wants at byte `7-i` —
            // one byte reversal away.
            let mut col = 0u64;
            for (b, value) in basis.iter().enumerate() {
                let byte = if k < 8 {
                    (value.lo >> (8 * k)) & 0xff
                } else {
                    (value.hi >> (8 * (k - 8))) & 0xff
                };
                col |= byte << (8 * b);
            }
            slot[h * 16 + k] = transpose_bits_8x8(col).swap_bytes();
        }
    }
}

/// 8x8 bit transpose of a qword read as eight byte-rows:
/// `out.byte[i].bit[b] = in.byte[b].bit[i]`. The three masked swap rounds are
/// the scalar twin of [`bit_transpose_64bytes`]'s per-qword stage.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
#[inline]
fn transpose_bits_8x8(mut x: u64) -> u64 {
    let t = (x ^ (x >> 7)) & 0x00AA_00AA_00AA_00AA;
    x ^= t ^ (t << 7);
    let t = (x ^ (x >> 14)) & 0x0000_CCCC_0000_CCCC;
    x ^= t ^ (t << 14);
    let t = (x ^ (x >> 28)) & 0x0000_0000_F0F0_F0F0;
    x ^= t ^ (t << 28);
    x
}

/// Per-worker state for the four-window fold4 producer.
/// The fused GFNI C drain's first live group of a band overwrites every one
/// of the 32 KiB byte planes, so the band-wide zero fill, that group's plane
/// loads, and the fused route's `partial_c4` zero seed are all dead work.
/// Ranked default. `FLOCK_NO_ZC_C_PLANE_FIRST_WRITE=1` restores the incumbent
/// clear-then-accumulate form, which is the byte-identity oracle.
#[inline]
fn c_plane_first_write_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var_os("FLOCK_NO_ZC_C_PLANE_FIRST_WRITE").is_none()
    });
    *ON
}

pub(crate) struct WorkerStateFold4 {
    partial_ab: [F128; ELL],
    /// Byte-plane banks for the eq-folded GFNI AB drain (`2^s x 1 KiB`);
    /// empty until the folded arm first sizes it.
    plane_banks: Vec<u8>,
    /// Byte-plane C banks for the fused GFNI drain (`[q][bank][plane][lane]`,
    /// 32 KiB) followed by the 4 KiB group staging buffer; over-allocated by
    /// 63 bytes so both stay cache-line aligned. Empty until the fused arm
    /// first sizes it.
    plane_c: Vec<u8>,
    /// True until the first live fused-C group has overwritten every plane in
    /// the current band; that first group need not read or clear stale bytes.
    plane_c_first_write: bool,
    plane_c_off: usize,
    partial_c4: Box<[[[F128; ELL]; N_C_BANKS]; N_C_Q]>,
    chunk_ab_bytes: [[u8; 64]; 1 << N_MEDIUM],
    /// Synthetic C blocks, one per retained q: row `4w + j` holds window w's
    /// transposed row `b_med = 4j + q`.
    chunk_c4: Box<[[[u8; 64]; 16]; N_C_Q]>,
    a_col: [F8; ELL],
    b_col: [F8; ELL],
    pub(crate) local_res_ab: [F128; ELL],
    pub(crate) local_res_c4: Box<[[[F128; ELL]; N_C_BANKS]; N_C_Q]>,
}

impl WorkerStateFold4 {
    pub(crate) fn new() -> Self {
        Self {
            partial_ab: [F128::ZERO; ELL],
            plane_banks: Vec::new(),
            plane_c: Vec::new(),
            plane_c_first_write: false,
            plane_c_off: 0,
            partial_c4: Box::new([[[F128::ZERO; ELL]; N_C_BANKS]; N_C_Q]),
            chunk_ab_bytes: [[0u8; 64]; 1 << N_MEDIUM],
            chunk_c4: Box::new([[[0u8; 64]; 16]; N_C_Q]),
            a_col: [F8::ZERO; ELL],
            b_col: [F8::ZERO; ELL],
            local_res_ab: [F128::ZERO; ELL],
            local_res_c4: Box::new([[[F128::ZERO; ELL]; N_C_BANKS]; N_C_Q]),
        }
    }
}

/// Fold4 twin of [`process_one_x_hi_with_precomputed_ab`]: identical AB
/// completion per window; C drained four windows at a time into 32 banks.
#[allow(clippy::too_many_arguments)]
fn process_one_x_hi_with_precomputed_ab_fold4(
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
    fold4_tables: &[F128],
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    c_nibble_luts: &[kernels::CBankNibbleLut],
    eq_fold: Option<(&[F128], &[u64], usize)>,
    c_gfni_mats: Option<&[u64]>,
    state: &mut WorkerStateFold4,
) {
    debug_assert!(big_lo_size.is_multiple_of(4));
    let _ = (&mut state.a_col, &mut state.b_col);
    state.partial_ab.fill(F128::ZERO);
    // Eq-folded GFNI AB drain: size and zero the byte-plane banks for this
    // band (`(eq_bot, mats, bank_bits)`; `FLOCK_NO_ZC_AB_EQ_FOLD=1` keeps
    // the incumbent per-chunk multiply).
    if let Some((eq_bot, _, _)) = eq_fold {
        let plane_len = eq_bot.len() * 16 * ELL;
        if state.plane_banks.len() != plane_len {
            state.plane_banks.clear();
            state.plane_banks.resize(plane_len, 0);
        } else {
            state.plane_banks.fill(0);
        }
    }
    // Fused GFNI C drain: size and zero the 32 KiB byte-plane C banks for
    // this band; `FLOCK_NO_ZC_C_GFNI=1` keeps the incumbent transpose +
    // nibble-LUT drain (`c_gfni_mats == None`).
    let first_write = c_plane_first_write_enabled();
    state.plane_c_first_write = first_write && c_gfni_mats.is_some();
    if c_gfni_mats.is_some() {
        if state.plane_c.is_empty() {
            let bytes = C_PLANE_BANK_BYTES + C_GROUP_BYTES + 63;
            state.plane_c = if first_write {
                crate::alloc_uninit_vec::<u8>(bytes)
            } else {
                vec![0u8; bytes]
            };
            let off = state.plane_c.as_ptr().align_offset(64);
            state.plane_c_off = if off <= 63 { off } else { 0 };
        } else if !first_write {
            let off = state.plane_c_off;
            state.plane_c[off..off + C_PLANE_BANK_BYTES].fill(0);
        }
    }
    // The fused route's `partial_c4` slots are all overwritten by the band-end
    // reassembly below, so only the fallback drain needs a zero seed here; an
    // all-dead fused band is cleared at the end instead.
    if !first_write || c_gfni_mats.is_none() {
        for group in state.partial_c4.iter_mut() {
            for bank in group.iter_mut() {
                bank.fill(F128::ZERO);
            }
        }
    }
    let n_lo = n_lo_and_inner - N_INNER;
    for group in 0..big_lo_size / 4 {
        let mut any_live = false;
        let mut group_counts = [0usize; 4];
        for (w, count) in group_counts.iter_mut().enumerate() {
            let x_outer = (4 * group + w) | (x_hi << n_lo);
            *count = b_med_counts[x_outer & within_outer_mask] as usize;
            any_live |= *count != 0;
        }
        // Early staged fetch: pull this group's 4 KiB of `c_packed` in strict
        // address order BEFORE the AB completion runs, so its DRAM misses
        // retire underneath four windows of AB work instead of stalling the
        // drain at the end of the group.
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512vbmi",
            target_feature = "vpclmulqdq",
            target_feature = "gfni"
        ))]
        if any_live && c_gfni_mats.is_some() {
            let group_base = (((4 * group) << N_INNER) | (x_hi << n_lo_and_inner)) * N_CHUNKS;
            let off = state.plane_c_off + C_PLANE_BANK_BYTES;
            let stage: &mut [u8; C_GROUP_BYTES] = (&mut state.plane_c[off..off + C_GROUP_BYTES])
                .try_into()
                .expect("aligned 4 KiB group staging window");
            kernels::stage_c_group(&c_packed[group_base..group_base + C_GROUP_BYTES], stage);
        }
        for w in 0..4 {
            let x_outer_lo = 4 * group + w;
            let n_b_med = group_counts[w];
            let chunk_byte_base = ((x_outer_lo << N_INNER) | (x_hi << n_lo_and_inner)) * N_CHUNKS;
            // C rows: live rows transposed into their (q, 4w+j) slot, dead rows zeroed.
            // The fused GFNI drain reads `c_packed` directly and folds this
            // transpose into its own mask build, so it needs none of this.
            if c_gfni_mats.is_none() {
                for b_med in 0..(1 << N_MEDIUM) {
                    let q = b_med & 3;
                    let j = b_med >> 2;
                    let dst = &mut state.chunk_c4[q][4 * w + j];
                    if b_med < n_b_med {
                        let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;
                        let c_in: &[u8; 64] = (&c_packed[byte_base_b..byte_base_b + 64])
                            .try_into()
                            .expect("64 c-bytes per medium position");
                        bit_transpose_64bytes(c_in, dst);
                    } else {
                        dst.fill(0);
                    }
                }
            }
            if n_b_med == 0 {
                continue;
            }
            // AB completion: identical to the incumbent per-window path.
            for b_med in 0..n_b_med {
                let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;
                state.chunk_ab_bytes[b_med]
                    .copy_from_slice(&ab_inner[byte_base_b..byte_base_b + 64]);
            }
            #[cfg(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "vpclmulqdq",
                target_feature = "gfni"
            ))]
            if let Some((_, mats, bank_bits)) = eq_fold {
                let w_idx = x_outer_lo >> bank_bits;
                let u = x_outer_lo & ((1usize << bank_bits) - 1);
                let mats_w: &[u64; 256] = (&mats[w_idx * 256..(w_idx + 1) * 256])
                    .try_into()
                    .expect("one 16x16 qword matrix block per w");
                let bank: &mut [u8; 16 * ELL] = (&mut state.plane_banks
                    [u * 16 * ELL..(u + 1) * 16 * ELL])
                    .try_into()
                    .expect("one plane bank per low index");
                kernels::accumulate_convert_ab_nomul_gfni(
                    &state.chunk_ab_bytes,
                    n_b_med,
                    mats_w,
                    bank,
                );
            } else {
                kernels::accumulate_convert_ab(
                    &state.chunk_ab_bytes,
                    n_b_med,
                    convert,
                    eq_lo_scaled[x_outer_lo],
                    &mut state.partial_ab,
                );
            }
            #[cfg(not(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "vpclmulqdq",
                target_feature = "gfni"
            )))]
            {
                debug_assert!(eq_fold.is_none());
                kernels::accumulate_convert_ab(
                    &state.chunk_ab_bytes,
                    n_b_med,
                    convert,
                    eq_lo_scaled[x_outer_lo],
                    &mut state.partial_ab,
                );
            }
        }
        if !any_live {
            continue;
        }
        // Fused GFNI drain: one call per four-window group, straight off the
        // 4 KiB of `c_packed` this group owns. No row transposes, no mask
        // tables, no LUT probes — see the kernel.
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512vbmi",
            target_feature = "vpclmulqdq",
            target_feature = "gfni"
        ))]
        if let Some(mats) = c_gfni_mats {
            let mats_g: &[u64; C_FOLD4_MATS_PER_GROUP] = (&mats
                [group * C_FOLD4_MATS_PER_GROUP..(group + 1) * C_FOLD4_MATS_PER_GROUP])
                .try_into()
                .expect("32 matrix qwords per four-window group");
            let off = state.plane_c_off;
            let (planes, stage) = state.plane_c[off..off + C_PLANE_BANK_BYTES + C_GROUP_BYTES]
                .split_at_mut(C_PLANE_BANK_BYTES);
            let planes: &mut [u8; C_PLANE_BANK_BYTES] = planes
                .try_into()
                .expect("aligned 32 KiB plane C bank window");
            if state.plane_c_first_write {
                kernels::write_c_banks_fold4_fused_gfni(stage, &group_counts, mats_g, planes);
                state.plane_c_first_write = false;
            } else {
                kernels::accumulate_c_banks_fold4_fused_gfni(stage, &group_counts, mats_g, planes);
            }
            continue;
        }
        let c_tables =
            &fold4_tables[group * C_MASK_TABLE_STRIDE..(group + 1) * C_MASK_TABLE_STRIDE];
        for q in 0..N_C_Q {
            // SAFETY: `[[u8; 64]; 16]` is contiguous and has exactly 1024 bytes.
            let c_block: &[u8; 16 * 64] =
                unsafe { &*state.chunk_c4[q].as_ptr().cast::<[u8; 16 * 64]>() };
            #[cfg(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "vpclmulqdq"
            ))]
            kernels::accumulate_c_banks_prebuilt(
                c_block,
                1 << N_MEDIUM,
                c_tables,
                &c_nibble_luts[group],
                &mut state.partial_c4[q],
            );
            #[cfg(not(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "vpclmulqdq"
            )))]
            kernels::accumulate_c_banks(c_block, 1 << N_MEDIUM, c_tables, &mut state.partial_c4[q]);
        }
    }
    // Band-end plane reassembly for the fused GFNI C drain: each bank's
    // sixteen byte planes become its 64 F128 lanes. Pure XOR accumulation,
    // so these are the incumbent's `partial_c4` values byte-for-byte.
    if c_gfni_mats.is_some() {
        if state.plane_c_first_write {
            // A fused band with no live group never wrote the plane bank, so
            // it must not be read; every logical C contribution is zero.
            for group in state.partial_c4.iter_mut() {
                for bank in group.iter_mut() {
                    bank.fill(F128::ZERO);
                }
            }
        } else {
            let off = state.plane_c_off;
            let planes = &state.plane_c[off..off + C_PLANE_BANK_BYTES];
            for q in 0..N_C_Q {
                for bank in 0..N_C_BANKS {
                    let bank_planes: &[u8; 16 * ELL] = planes[(q * N_C_BANKS + bank) * 16 * ELL..]
                        [..16 * ELL]
                        .try_into()
                        .expect("one 16-plane bank");
                    kernels::c_plane_bank_to_f128(bank_planes, &mut state.partial_c4[q][bank]);
                }
            }
        }
    }
    // Band-end plane fold for the eq-folded arm: reassemble each bank's
    // F128 lanes and apply `eq_bot[u]` once per bank — the same sums the
    // per-chunk multiply produced, reassociated.
    if let Some((eq_bot, _, _)) = eq_fold {
        // Plane-major → F128 through the same vectorized kernel the C drain
        // uses (identical bank layout: plane k, byte `k*ELL + lane`), instead
        // of 16 scalar byte loads per lane. The eq_bot multiply stays scalar
        // here — this level compiles for every arch and the loads were the
        // bulk of the band tail.
        let mut bank_f128 = [F128::ZERO; ELL];
        for (u, eq_bot_val) in eq_bot.iter().enumerate() {
            let bank: &[u8; 16 * ELL] = state.plane_banks[u * 16 * ELL..(u + 1) * 16 * ELL]
                .try_into()
                .expect("one 16-plane bank");
            kernels::c_plane_bank_to_f128(bank, &mut bank_f128);
            for lane in 0..ELL {
                state.partial_ab[lane] += *eq_bot_val * bank_f128[lane];
            }
        }
    }
    for lane in 0..ELL {
        state.local_res_ab[lane] += eq_hi_val * state.partial_ab[lane];
        for q in 0..N_C_Q {
            for bank in 0..N_C_BANKS {
                state.local_res_c4[q][bank][lane] += eq_hi_val * state.partial_c4[q][bank][lane];
            }
        }
    }
}

/// Collapse the 32 q-retained banks to the incumbent eight (exact), then run
/// the incumbent finish; additionally emit the sixteen-bank direct-fold4
/// tensor `fold4_c[(e + 4q)·128 + b₀·64 + lane] = banks[q][b₀ + 2e][lane]`
/// (bank index `e_small + 4·q`, matching `build_eq(suffix[..4])`).
fn finish_c_banks_fold4(
    banks32: &[[[F128; ELL]; N_C_BANKS]; N_C_Q],
) -> ([F128; ELL], Vec<F128>, Vec<F128>, Vec<F128>) {
    let w = c_fold4_q_weights();
    let mut banks8 = [[F128::ZERO; ELL]; N_C_BANKS];
    for q in 0..N_C_Q {
        for k in 0..N_C_BANKS {
            for lane in 0..ELL {
                banks8[k][lane] += w[q] * banks32[q][k][lane];
            }
        }
    }
    let (res_c_s, s_hat_v_c, quad_c) = finish_c_banks(&banks8);
    let mut fold4_c = vec![F128::ZERO; 16 * 2 * ELL];
    for q in 0..N_C_Q {
        for e in 0..4 {
            for b_0 in 0..2 {
                let base = (e + 4 * q) * 2 * ELL + b_0 * ELL;
                fold4_c[base..base + ELL].copy_from_slice(&banks32[q][b_0 + 2 * e]);
            }
        }
    }
    (res_c_s, s_hat_v_c, quad_c, fold4_c)
}

/// Fold4 twin of [`round1_shift_reduce_extract_c_packed_padded_with_precomputed_ab_quad`]:
/// same `(ab, c_lifted, s_hat_v_c, quad_c)` bit-for-bit, plus the sixteen-bank
/// C tensor. Requires `c_fold4_capture_shape_ok(1 << eq.n_lo)`.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn round1_shift_reduce_extract_c_packed_padded_with_precomputed_ab_fold4(
    ab_inner: &mut Round1AbInner,
    a_packed: &[u8],
    b_packed: &[u8],
    c_packed: &[u8],
    m: usize,
    k_skip: usize,
    r: &[F128],
    inv_table: &InvNttTableByteSingleGf8,
    padding: &PaddingSpec,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<F128>, Vec<F128>) {
    use rayon::prelude::*;
    assert_eq!(k_skip, K_SKIP, "optimized variant is k_skip=6 only");
    assert!(m >= k_skip + N_INNER);
    let total_bytes = (1usize << m) / 8;
    assert_eq!(ab_inner.len_bytes(), total_bytes);
    assert_eq!(a_packed.len(), total_bytes);
    assert_eq!(b_packed.len(), total_bytes);
    assert_eq!(c_packed.len(), total_bytes);
    assert_eq!(r.len(), m);
    assert_eq!(inv_table.k, k_skip);
    ab_inner.fill_invalid_prefix(a_packed, b_packed, inv_table);
    assert!(
        !ab_inner.ranked_compact(),
        "the Fold4 C consumer requires dense AB storage"
    );
    let eq = SplitEqGhash::new(&r[k_skip + N_INNER..]);
    let big_lo_size = 1usize << eq.n_lo;
    assert!(
        c_fold4_capture_shape_ok(big_lo_size),
        "fold4 C capture needs a multiple-of-four low split"
    );
    let hi_size = 1usize << eq.n_hi;
    let n_lo_and_inner = eq.n_lo + N_INNER;
    let d_inv_val = d_inv();
    let eq_lo_scaled: Vec<F128> = eq.lo.iter().map(|v| *v * d_inv_val).collect();
    let convert = convert_table();
    // Fused GFNI C drain: 32 bit matrices per four-window group replace the
    // group's 512-entry synthetic mask table and its nibble LUT outright, so
    // neither is built at all when the fused arm is live (8 MiB + 1 MiB of
    // per-prove table material at the ranked shape).
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    ))]
    let c_gfni_state = c_fold4_gfni_enabled().then(|| build_c_fold4_gfni_mats(&eq.lo));
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    )))]
    let c_gfni_state: Option<Vec<u64>> = None;
    let fold4_tables = if c_gfni_state.is_some() {
        Vec::new()
    } else {
        build_c_fold4_tables(&eq.lo)
    };
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    let c_nibble_luts = kernels::build_c_bank_nibble_luts(&fold4_tables);
    let (within_outer_mask, b_med_counts) = build_b_med_counts(padding);
    let ab_inner_bytes = ab_inner.as_bytes();
    // Eq-folded GFNI AB drain: factor `eq_lo_scaled` into `eq_top * eq_bot`
    // and carry `eq_top[w]` inside per-`w` GFNI matrices, deleting the
    // per-chunk eq multiply and every convert-table access.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    ))]
    let eq_fold_state = (ab_eq_fold_gfni_enabled() && eq.n_lo >= 2).then(|| {
        let bank_bits = eq.n_lo.saturating_sub(5).max(1);
        let r_lo = &r[k_skip + N_INNER..k_skip + N_INNER + eq.n_lo];
        let (eq_bot, eq_top_scaled) = ab_eq_fold_factors(r_lo, bank_bits);
        let mats = build_ab_eq_fold_mats(&eq_top_scaled, convert);
        (eq_bot, mats, bank_bits)
    });
    let (res_ab, banks32) = (0..hi_size)
        .into_par_iter()
        .fold(WorkerStateFold4::new, |mut state, x_hi| {
            #[cfg(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "vpclmulqdq",
                target_feature = "gfni"
            ))]
            let eq_fold_arg = eq_fold_state
                .as_ref()
                .map(|(eq_bot, mats, bank_bits)| (eq_bot.as_slice(), mats.as_slice(), *bank_bits));
            #[cfg(not(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "vpclmulqdq",
                target_feature = "gfni"
            )))]
            let eq_fold_arg: Option<(&[F128], &[u64], usize)> = None;
            process_one_x_hi_with_precomputed_ab_fold4(
                x_hi,
                big_lo_size,
                n_lo_and_inner,
                within_outer_mask,
                &b_med_counts,
                ab_inner_bytes,
                c_packed,
                &eq_lo_scaled,
                eq.hi[x_hi],
                convert,
                &fold4_tables,
                #[cfg(all(
                    target_arch = "x86_64",
                    target_feature = "avx512f",
                    target_feature = "vpclmulqdq"
                ))]
                &c_nibble_luts,
                eq_fold_arg,
                c_gfni_state.as_deref(),
                &mut state,
            );
            state
        })
        .map(|s| (s.local_res_ab, s.local_res_c4))
        .reduce(
            || {
                (
                    [F128::ZERO; ELL],
                    Box::new([[[F128::ZERO; ELL]; N_C_BANKS]; N_C_Q]),
                )
            },
            |(mut ab1, mut c1), (ab2, c2)| {
                for i in 0..ELL {
                    ab1[i] += ab2[i];
                }
                for (lq, rq) in c1.iter_mut().zip(c2.iter()) {
                    for (left, right) in lq.iter_mut().zip(rq.iter()) {
                        for i in 0..ELL {
                            left[i] += right[i];
                        }
                    }
                }
                (ab1, c1)
            },
        );
    crate::scratch::give_f128(fold4_tables);
    let (res_c_s, s_hat_v_c, quad_c, fold4_c) = finish_c_banks_fold4(&banks32);
    let res_c_lifted = ntt_extend_f128_vec_ghash(&res_c_s, inv_table);
    (res_ab.to_vec(), res_c_lifted, s_hat_v_c, quad_c, fold4_c)
}

// ---------------------------------------------------------------------------
// Ranked identity-C via one block-major outer fold
//
// At the ranked BLAKE3 shape C is the identity (`Cz = z`), so C's round-one
// message is a statistic of the committed witness itself. Folding `z` at the
// round-one outer challenges `r[k_log..]` yields the length-2^k_log inner table
// of that same multilinear, and retaining four of its inner coordinates is an
// exact reassociation of the incumbent 32-bank row-major drain. Round one then
// splits in two:
//
//   * `round1_shift_reduce_ab_packed_padded_with_precomputed` — the AB half of
//     the fused kernel, with the C drain deleted;
//   * `round1_c_fold4_from_block_major_z` — C's message and all three
//     RingSwitch capture tensors, from one outer fold.
//
// The fold is the block-major kernel lincheck already runs on this exact
// buffer, so no new representation of the witness is built: the deleted drain
// and the added fold read the same 2^(m-3) bytes.
// ---------------------------------------------------------------------------

/// Look-ahead, in 1 KiB outer windows, for the round-1 AB packed-row
/// prefetch. One window is `2^N_INNER * N_CHUNKS = 1024` bytes of
/// `ab_inner` — the sixteen contiguous 64-byte chunks one `x_outer_lo` step
/// copies into `chunk_ab_bytes` before the GFNI accumulate consumes them.
///
/// The incumbent issues NO software prefetch here at all. The sweep
/// demand-loads a window's sixteen lines back to back at the head of the
/// window and then spends ~2 000 cycles of GFNI on them, so every window
/// opens with a burst of misses that only the L2 streamer can have covered.
/// At the ranked shape it frequently has not: sixteen worker threads each
/// drive this stream while the *concurrent* identity-C fold (the other half
/// of round one's `rayon::join`) drives eight row-strided gather streams of
/// its own through the same L2s, and the streamer's tracker is shared per
/// core between the two SMT siblings.
///
/// The ranked-shape decomposition of round one measures the exposure
/// directly (16 threads, m = 32, k_log = 14, medians):
///
///   real (AB memory + compute, C full)     22.29 ms
///   AB compute only (no `ab_inner` reads)  19.37 ms
///   AB compute only + C compute only       18.81 ms
///
/// so ~2.9 ms of the AB stream is ADDITIVE, not overlapped, while the
/// concurrent C fold's own 512 MiB costs only 0.56 ms exposed — that stream
/// already carries the grouped-gather prefetch, this one carried nothing.
///
/// Two windows puts a full extra window of work behind the hint — ~2 000
/// cycles, several times a loaded DRAM miss on this box — while keeping
/// every line covered exactly once: window `W + 2` is requested at window
/// `W`, so each line is hinted once, just earlier. One window only reaches
/// back into the same window's own GFNI burst, and four or more evicts
/// before use; both measure worse. Nothing is added to the sweep and
/// nothing moves — a prefetch has no architectural effect — so the
/// accumulated lanes are bit-identical either way.
const ZC_R1AB_PF_WINDOWS: usize = 2;

/// `FLOCK_NO_ZC_R1AB_PF=1` restores the incumbent no-prefetch round-1 AB
/// sweep (exact same-binary A/B). Resolved once per process.
fn zc_r1ab_pf_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_ZC_R1AB_PF").is_none());
    *ON
}

/// `FLOCK_NO_ZC_R1AB_PF_SPREAD=1` restores the incumbent delivery of the
/// round-1 AB packed-row prefetch: the whole window's hint block issued back
/// to back ahead of the window's copy loop. The default arm issues the same
/// hints for the same lines, one per copy step, so a hint and a demand line
/// alternate instead of sixteen hints queueing ahead of sixteen loads.
/// Exact same-binary A/B: a prefetch has no architectural effect.
fn zc_r1ab_pf_spread_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_ZC_R1AB_PF_SPREAD").is_none());
    *ON
}

/// Read the ranked AB residual rows directly from their producer's live
/// span. `FLOCK_NO_ZC_R1_AB_DIRECT=1` restores the packed-row staging copy.
/// Resolved before the parallel band fold, never in the row/plane loops.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
fn zc_r1_ab_direct_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_ZC_R1_AB_DIRECT").is_none());
    *ON
}

/// `FLOCK_NO_ZC_R1AB_FIRST_WRITE=1` restores the per-band plane-bank clear
/// and load/XOR first visit. The default arm is used only when every padding
/// window is live, which proves that `w_idx == 0` overwrites each bank before
/// any later visit or the band-end collapse can read it.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
fn zc_r1ab_first_write_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_ZC_R1AB_FIRST_WRITE").is_none());
    *ON
}

/// Keep one ranked 128 KiB plane arena resident on each Rayon worker.  The
/// official all-live first-write sweep overwrites every byte before any read,
/// so warmup can park the allocation (and its faulted pages) for the measured
/// proofs instead of repeating malloc/mmap + soft faults + free/munmap.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
fn zc_r1ab_plane_cache_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_ZC_R1AB_PLANE_CACHE").is_none());
    *ON
}

const RANKED_AB_PLANE_BYTES: usize = 128 * 16 * ELL;

std::thread_local! {
    static RANKED_AB_PLANE_CACHE: std::cell::RefCell<Option<Vec<u8>>> =
        const { std::cell::RefCell::new(None) };
}

#[inline]
fn take_ranked_ab_plane() -> Vec<u8> {
    RANKED_AB_PLANE_CACHE
        .with(|slot| slot.borrow_mut().take())
        .filter(|v| v.len() == RANKED_AB_PLANE_BYTES)
        .unwrap_or_else(|| crate::alloc_uninit_vec(RANKED_AB_PLANE_BYTES))
}

#[inline]
fn give_ranked_ab_plane(v: Vec<u8>) {
    if v.len() != RANKED_AB_PLANE_BYTES {
        return;
    }
    RANKED_AB_PLANE_CACHE.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            *slot = Some(v);
        }
    });
}

/// AB-only worker state: [`WorkerStateFold4`] without any of the C banks.
pub(crate) struct WorkerStateAbOnly {
    partial_ab: [F128; ELL],
    plane_banks: Vec<u8>,
    cached_plane_banks: bool,
    chunk_ab_bytes: [[u8; 64]; 1 << N_MEDIUM],
    pub(crate) local_res_ab: [F128; ELL],
}

impl WorkerStateAbOnly {
    pub(crate) fn new() -> Self {
        Self {
            partial_ab: [F128::ZERO; ELL],
            plane_banks: Vec::new(),
            cached_plane_banks: false,
            chunk_ab_bytes: [[0u8; 64]; 1 << N_MEDIUM],
            local_res_ab: [F128::ZERO; ELL],
        }
    }
}

impl Drop for WorkerStateAbOnly {
    #[inline]
    fn drop(&mut self) {
        if self.cached_plane_banks {
            self.cached_plane_banks = false;
            give_ranked_ab_plane(core::mem::take(&mut self.plane_banks));
        }
    }
}

/// AB half of [`process_one_x_hi_with_precomputed_ab_fold4`], instruction for
/// instruction: same window order, same kernels, same band-end plane fold.
/// Every C statement (and every `c_packed` byte) is gone.
/// `DIRECT` only changes the GFNI input loads: the caller selects it once
/// per band for the ranked residual `(first, end) = (2, 16), (0, 15)` pair.
fn process_one_x_hi_ab_only<const DIRECT: bool>(
    x_hi: usize,
    big_lo_size: usize,
    n_lo_and_inner: usize,
    within_outer_mask: usize,
    b_med_counts: &[u8],
    ab_inner: &[u8],
    eq_lo_scaled: &[F128],
    eq_hi_val: F128,
    convert: &[F128],
    eq_fold: Option<(&[F128], &[u64], usize)>,
    plane_first_write: bool,
    plane_cache: bool,
    ranked_one_rows_elided: bool,
    ranked_compact: bool,
    state: &mut WorkerStateAbOnly,
) {
    state.partial_ab.fill(F128::ZERO);
    debug_assert!(!plane_first_write || b_med_counts.iter().all(|&count| count != 0));
    debug_assert!(
        !DIRECT
            || (eq_fold.is_some()
                && ranked_one_rows_elided
                && within_outer_mask == 1
                && b_med_counts == [16, 15])
    );
    debug_assert!(!ranked_compact || ranked_one_rows_elided);
    if let Some((eq_bot, _, _)) = eq_fold {
        let plane_len = eq_bot.len() * 16 * ELL;
        if state.plane_banks.len() != plane_len {
            if state.cached_plane_banks {
                state.cached_plane_banks = false;
                give_ranked_ab_plane(core::mem::take(&mut state.plane_banks));
            }
            state.plane_banks = if plane_cache {
                state.cached_plane_banks = true;
                take_ranked_ab_plane()
            } else if plane_first_write {
                // Every byte is overwritten by the `w_idx == 0` visits below
                // before the vector is read; see the function-level guard.
                crate::alloc_uninit_vec(plane_len)
            } else {
                vec![0u8; plane_len]
            };
        } else if !plane_first_write {
            state.plane_banks.fill(0);
        }
    }
    let n_lo = n_lo_and_inner - N_INNER;
    #[inline(always)]
    fn live_row_base(x_outer: usize, first_b_med: usize, compact: bool) -> usize {
        if compact {
            let row_in_block = (x_outer & 1) * (1 << N_MEDIUM) + first_b_med;
            debug_assert!((2..31).contains(&row_in_block));
            (x_outer >> 1) * RANKED_AB_COMPACT_BLOCK_BYTES
                + (row_in_block - 2) * ELL
        } else {
            x_outer * (1 << N_MEDIUM) * ELL + first_b_med * ELL
        }
    }
    // Packed-row prefetch look-ahead, resolved once per x_hi band (never
    // inside the window loop).
    #[cfg(target_arch = "x86_64")]
    let pf_windows = if zc_r1ab_pf_enabled() {
        ZC_R1AB_PF_WINDOWS
    } else {
        0
    };
    #[cfg(target_arch = "x86_64")]
    let ab_inner_ptr = ab_inner.as_ptr();
    #[cfg(target_arch = "x86_64")]
    let pf_spread = zc_r1ab_pf_spread_enabled();
    for x_outer_lo in 0..big_lo_size {
        let x_outer = x_outer_lo | (x_hi << n_lo);
        let n_b_med = b_med_counts[x_outer & within_outer_mask] as usize;
        if n_b_med == 0 {
            continue;
        }
        let first_b_med = if ranked_one_rows_elided && (x_outer & within_outer_mask) == 0 {
            2
        } else {
            0
        };
        let live_base = live_row_base(x_outer, first_b_med, ranked_compact);
        // The window `ZC_R1AB_PF_WINDOWS` steps on — exactly the lines that
        // window's copy loop will demand, and exactly as many of them (the
        // padding skip drops the last chunk of every second window). Issued
        // BEFORE this window's own copy, so the hint sits two whole windows
        // of work ahead of the demand load it feeds; issued after the copy it
        // reaches only ~1.5 windows back and measures ~0.7 ms worse.
        // `wrapping_add` keeps the past-the-end address on the last windows
        // of the last band well defined, and that hint is simply dropped.
        // The window `ZC_R1AB_PF_WINDOWS` on, and how many of its lines the
        // padding skip leaves live.
        #[cfg(target_arch = "x86_64")]
        let (n_next, next_live_base, next_first_b_med) = if pf_windows != 0 {
            let x_next = x_outer_lo + pf_windows;
            let x_next_outer = x_next | (x_hi << n_lo);
            let n_next = b_med_counts[(x_outer + pf_windows) & within_outer_mask] as usize;
            let next_first = if ranked_one_rows_elided
                && ((x_outer + pf_windows) & within_outer_mask) == 0
            {
                2
            } else {
                0
            };
            (
                n_next,
                live_row_base(x_next_outer, next_first, ranked_compact),
                next_first,
            )
        } else {
            (0, 0, 0)
        };
        // The direct specialization issues the same hints beside its first
        // input loads in the leaf. The fallback keeps the original copy and
        // its prefetch delivery intact.
        if !DIRECT {
            // SAFETY: `_mm_prefetch` is a hint — no memory is read, no fault is
            // possible, and the address is formed by `wrapping_add` on the base
            // pointer.
            #[cfg(target_arch = "x86_64")]
            let pf_one = |b_med: usize| unsafe {
                core::arch::x86_64::_mm_prefetch(
                    ab_inner_ptr
                        .wrapping_add(next_live_base + (b_med - next_first_b_med) * ELL)
                        .cast::<i8>(),
                    core::arch::x86_64::_MM_HINT_T0,
                );
            };
            #[cfg(target_arch = "x86_64")]
            if !pf_spread {
                for b_med in next_first_b_med..n_next {
                    pf_one(b_med);
                }
            }
            for b_med in first_b_med..n_b_med {
                // Spread delivery: one hint per copy step, so each hint is
                // issued next to one demand line rather than the whole block
                // queueing ahead of the copy. Same lines, same look-ahead.
                #[cfg(target_arch = "x86_64")]
                if pf_spread && b_med >= next_first_b_med && b_med < n_next {
                    pf_one(b_med);
                }
                let byte_base_b = live_base + (b_med - first_b_med) * ELL;
                state.chunk_ab_bytes[b_med]
                    .copy_from_slice(&ab_inner[byte_base_b..byte_base_b + ELL]);
            }
            #[cfg(target_arch = "x86_64")]
            if pf_spread {
                for b_med in n_b_med.max(next_first_b_med)..n_next {
                    pf_one(b_med);
                }
            }
        }
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "vpclmulqdq",
            target_feature = "gfni"
        ))]
        if let Some((_, mats, bank_bits)) = eq_fold {
            let w_idx = x_outer_lo >> bank_bits;
            let u = x_outer_lo & ((1usize << bank_bits) - 1);
            let mats_w: &[u64; 256] = (&mats[w_idx * 256..(w_idx + 1) * 256])
                .try_into()
                .expect("one 16x16 qword matrix block per w");
            let bank: &mut [u8; 16 * ELL] = (&mut state.plane_banks
                [u * 16 * ELL..(u + 1) * 16 * ELL])
                .try_into()
                .expect("one plane bank per low index");
            if DIRECT {
                // Borrow only initialized, consumed rows. In particular, do
                // not form a full-window array reference over the two omitted
                // prefix rows or the odd window's unused final row.
                let live_rows = &ab_inner
                    [live_base..live_base + (n_b_med - first_b_med) * ELL];
                let prefetch = kernels::AbDirectPrefetch {
                    next_window: if pf_windows == 0 {
                        core::ptr::null()
                    } else {
                        ab_inner_ptr.wrapping_add(next_live_base)
                    },
                    first: next_first_b_med,
                    end: n_next,
                    spread: pf_spread,
                };
                if plane_first_write && w_idx == 0 {
                    kernels::convert_ab_nomul_gfni_direct::<true>(
                        live_rows,
                        first_b_med,
                        n_b_med,
                        mats_w,
                        bank,
                        &prefetch,
                    );
                } else {
                    kernels::convert_ab_nomul_gfni_direct::<false>(
                        live_rows,
                        first_b_med,
                        n_b_med,
                        mats_w,
                        bank,
                        &prefetch,
                    );
                }
            } else if plane_first_write && w_idx == 0 {
                if first_b_med == 2 {
                    kernels::write_convert_ab_nomul_gfni_range2(
                        &state.chunk_ab_bytes,
                        n_b_med,
                        mats_w,
                        bank,
                    );
                } else {
                    kernels::write_convert_ab_nomul_gfni(
                        &state.chunk_ab_bytes,
                        n_b_med,
                        mats_w,
                        bank,
                    );
                }
            } else {
                if first_b_med == 2 {
                    kernels::accumulate_convert_ab_nomul_gfni_range2(
                        &state.chunk_ab_bytes,
                        n_b_med,
                        mats_w,
                        bank,
                    );
                } else {
                    kernels::accumulate_convert_ab_nomul_gfni(
                        &state.chunk_ab_bytes,
                        n_b_med,
                        mats_w,
                        bank,
                    );
                }
            }
        } else {
            if first_b_med == 2 {
                state.chunk_ab_bytes[0].fill(0);
                state.chunk_ab_bytes[1].fill(0);
            }
            kernels::accumulate_convert_ab(
                &state.chunk_ab_bytes,
                n_b_med,
                convert,
                eq_lo_scaled[x_outer_lo],
                &mut state.partial_ab,
            );
        }
        #[cfg(not(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "vpclmulqdq",
            target_feature = "gfni"
        )))]
        {
            debug_assert!(eq_fold.is_none());
            if first_b_med == 2 {
                state.chunk_ab_bytes[0].fill(0);
                state.chunk_ab_bytes[1].fill(0);
            }
            kernels::accumulate_convert_ab(
                &state.chunk_ab_bytes,
                n_b_med,
                convert,
                eq_lo_scaled[x_outer_lo],
                &mut state.partial_ab,
            );
        }
    }
    if let Some((eq_bot, _, _)) = eq_fold {
        // Plane-major → F128 through the same vectorized kernel the C drain
        // uses (identical bank layout: plane k, byte `k*ELL + lane`), instead
        // of 16 scalar byte loads per lane. The eq_bot multiply then rides the
        // shared `add_scaled` leaf, which selects the architecture kernel.
        let mut bank_f128 = [F128::ZERO; ELL];
        let wide = r1_eqfold_x4_enabled();
        for (u, eq_bot_val) in eq_bot.iter().enumerate() {
            let bank: &[u8; 16 * ELL] = state.plane_banks[u * 16 * ELL..(u + 1) * 16 * ELL]
                .try_into()
                .expect("one 16-plane bank");
            kernels::c_plane_bank_to_f128(bank, &mut bank_f128);
            if wide {
                crate::field::f128_slice::add_scaled(
                    &mut state.partial_ab,
                    &bank_f128,
                    *eq_bot_val,
                );
            } else {
                for lane in 0..ELL {
                    state.partial_ab[lane] += *eq_bot_val * bank_f128[lane];
                }
            }
        }
    }
    if r1_eqfold_x4_enabled() {
        let (dst, src) = (&mut state.local_res_ab, &state.partial_ab);
        crate::field::f128_slice::add_scaled(dst, src, eq_hi_val);
    } else {
        for lane in 0..ELL {
            state.local_res_ab[lane] += eq_hi_val * state.partial_ab[lane];
        }
    }
}

/// Ranked default routes the round-one AB band folds — the plane-bank
/// accumulate and the band-end `eq_hi` fold — through the shared
/// `f128_slice::add_scaled` leaf. `FLOCK_NO_ZC_R1_EQFOLD_X4=1` restores the
/// per-lane scalar loops. Read once per process; default ON.
fn r1_eqfold_x4_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_ZC_R1_EQFOLD_X4").is_none());
    *ON
}

/// Ranked default routes the round-one identity-C bank collapses (sixty-four
/// banks to sixteen, sixteen to four) through the shared
/// `f128_slice::add_scaled` leaf. `FLOCK_NO_ZC_R1_CFOLD_X4=1` restores the
/// per-slot scalar loops. Read once per process; default ON.
fn r1_cfold_x4_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_ZC_R1_CFOLD_X4").is_none());
    *ON
}

/// Round-one AB message from the challenge-independent precompute, with no C
/// drain. Bit-identical to the AB output of
/// [`round1_shift_reduce_extract_c_packed_padded_with_precomputed_ab_fold4`];
/// the caller sources C from the lincheck stripe instead.
pub fn round1_shift_reduce_ab_packed_padded_with_precomputed(
    ab_inner: &mut Round1AbInner,
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
    assert_eq!(ab_inner.len_bytes(), total_bytes);
    assert_eq!(a_packed.len(), total_bytes);
    assert_eq!(b_packed.len(), total_bytes);
    assert_eq!(r.len(), m);
    assert_eq!(inv_table.k, k_skip);
    ab_inner.fill_invalid_prefix(a_packed, b_packed, inv_table);
    let r_outer = &r[k_skip + N_INNER..];
    let n_hi = r_outer.len().min(SplitEqGhash::MAX_N_HI);
    let n_lo = r_outer.len() - n_hi;
    let big_lo_size = 1usize << n_lo;
    let hi_size = 1usize << n_hi;
    let n_lo_and_inner = n_lo + N_INNER;
    // The ranked GFNI arm factors the low equality directly from its
    // challenges below. Build only the shared 128-entry high tensor here;
    // constructing and then scaling the full 4096-entry low tensor would be
    // dead work in that arm.
    let eq_hi = build_eq(&r_outer[n_lo..]);
    let convert = convert_table();
    let (within_outer_mask, b_med_counts) = build_b_med_counts(padding);
    let ranked_one_rows_elided = ab_inner.ranked_one_rows_elided();
    let ranked_compact = ab_inner.ranked_compact();
    if ranked_one_rows_elided {
        assert_eq!(m, 32);
        assert_eq!(padding.k_log, 14);
        assert_eq!(padding.useful_bits_per_block, 15409);
        assert_eq!(ab_inner.invalid_prefix_bytes(), 0);
    }
    if ranked_compact {
        assert!(ranked_one_rows_elided);
        assert_eq!(within_outer_mask, 1);
        assert_eq!(b_med_counts.as_slice(), [16, 15]);
        assert_eq!(
            ab_inner.as_bytes().len(),
            (1usize << 18) * RANKED_AB_COMPACT_BLOCK_BYTES
        );
    }
    let ab_inner_bytes = ab_inner.as_bytes();
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    ))]
    let eq_fold_enabled = ab_eq_fold_gfni_enabled() && n_lo >= 2;
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    )))]
    let eq_fold_enabled = false;
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    ))]
    let eq_fold_state = eq_fold_enabled.then(|| {
        let bank_bits = if n_lo == 12 {
            7
        } else {
            n_lo.saturating_sub(5).max(1)
        };
        let r_lo = &r_outer[..n_lo];
        let (eq_bot, eq_top_scaled) = ab_eq_fold_factors(r_lo, bank_bits);
        let mats = build_ab_eq_fold_mats(&eq_top_scaled, convert);
        (eq_bot, mats, bank_bits)
    });
    let eq_lo_scaled: Vec<F128> = if eq_fold_enabled {
        Vec::new()
    } else {
        let d_inv_val = d_inv();
        build_eq(&r_outer[..n_lo])
            .into_iter()
            .map(|v| v * d_inv_val)
            .collect()
    };
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    ))]
    let plane_first_write = eq_fold_state.is_some()
        && zc_r1ab_first_write_enabled()
        && b_med_counts.iter().all(|&count| count != 0);
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    )))]
    let plane_first_write = false;
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    ))]
    let plane_cache = plane_first_write
        && zc_r1ab_plane_cache_enabled()
        && m == 32
        && n_lo == 12
        && hi_size == 128
        && eq_fold_state
            .as_ref()
            .is_some_and(|(eq_bot, mats, bank_bits)| {
                eq_bot.len() == 128 && mats.len() == 32 * 256 && *bank_bits == 7
            });
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    )))]
    let plane_cache = false;
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    ))]
    let direct_ab = ranked_one_rows_elided
        && eq_fold_state.is_some()
        && within_outer_mask == 1
        && b_med_counts.as_slice() == [16, 15]
        && zc_r1_ab_direct_enabled();
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    )))]
    let direct_ab = false;
    let res_ab = (0..hi_size)
        .into_par_iter()
        .fold(WorkerStateAbOnly::new, |mut state, x_hi| {
            #[cfg(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "vpclmulqdq",
                target_feature = "gfni"
            ))]
            let eq_fold_arg = eq_fold_state
                .as_ref()
                .map(|(eq_bot, mats, bank_bits)| (eq_bot.as_slice(), mats.as_slice(), *bank_bits));
            #[cfg(not(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "vpclmulqdq",
                target_feature = "gfni"
            )))]
            let eq_fold_arg: Option<(&[F128], &[u64], usize)> = None;
            if direct_ab {
                process_one_x_hi_ab_only::<true>(
                    x_hi,
                    big_lo_size,
                    n_lo_and_inner,
                    within_outer_mask,
                    &b_med_counts,
                    ab_inner_bytes,
                    &eq_lo_scaled,
                    eq_hi[x_hi],
                    convert,
                    eq_fold_arg,
                    plane_first_write,
                    plane_cache,
                    ranked_one_rows_elided,
                    ranked_compact,
                    &mut state,
                );
            } else {
                process_one_x_hi_ab_only::<false>(
                    x_hi,
                    big_lo_size,
                    n_lo_and_inner,
                    within_outer_mask,
                    &b_med_counts,
                    ab_inner_bytes,
                    &eq_lo_scaled,
                    eq_hi[x_hi],
                    convert,
                    eq_fold_arg,
                    plane_first_write,
                    plane_cache,
                    ranked_one_rows_elided,
                    ranked_compact,
                    &mut state,
                );
            }
            state
        })
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

/// Identity-C's block-major outer fold at `r_outer`. `par` routes through
/// [`crate::lincheck::fold_block_major_one_shot`] — lincheck's shipped
/// one-shot dispatch, whose ranked `n_log == 18` arm keeps eq(r_outer, ·)
/// factored as two 2^9 tables consumed inside the (already parallel) fold
/// kernel instead of materializing the dense 2^18 table with 262,144
/// serially-chained GHASH multiplies on ONE core ahead of it. The factors
/// are exact (`eq[i] = eq_lo[i_lo] · eq_hi[i_hi]` — field multiply is
/// associative with no rounding) and the fold kernel underneath is the same
/// `partial_fold_packed_z_block_major_padded_with_tables` accumulation, so
/// `ẑ` is bit-identical — the same identity lincheck's own one-shot/kick
/// paths already rely on (`last_rho_kick_then_wait_matches_oneshot_fold`).
/// Every other geometry takes the one-shot's dense arm, which is this
/// function's `par = false` body verbatim. `false` is the incumbent
/// sequential form, kept as the kill-switch path and the byte-identity
/// oracle (`FLOCK_NO_SERIAL_PAR=1`).
pub(crate) fn identity_c_inner_fold(
    z_packed: &[F128],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    r_outer: &[F128],
    par: bool,
) -> Vec<F128> {
    debug_assert_eq!(r_outer.len(), m - k_log);
    if par {
        crate::lincheck::fold_block_major_one_shot(z_packed, m, k_log, useful_bits, r_outer)
    } else {
        let eq_outer = crate::lincheck::build_eq_table(r_outer);
        crate::lincheck::partial_fold_packed_z_block_major_padded(
            z_packed,
            m,
            k_log,
            useful_bits,
            &eq_outer,
        )
    }
}

/// Derive the exact legacy round-one C message and its RingSwitch capture
/// tensors from one block-major outer fold of the identity-C witness.
///
/// Returns `(res_c_lifted, s_hat_v_c, quad_c, fold4_c, fold8_c)`.  The
/// sixty-four-bank statistic is widened only after the block-major outer fold
/// has reduced identity C to its 16,384-element inner table; unlike the old
/// raw-row DirectFold8 experiment, this does not widen round one's GFNI plane
/// state or revisit the 512 MiB witness.
fn round1_lifted_from_fold8(
    fold8: &[F128],
    inner_tail: &[F128],
    prefix: F128,
    inv_table: &InvNttTableByteSingleGf8,
) -> Vec<F128> {
    let n_packed = 1usize << crate::pcs::LOG_PACKING;
    assert_eq!(inner_tail.len(), 7);
    assert_eq!(fold8.len(), 64 * n_packed);

    let retained_top_eq = build_eq(&inner_tail[4..6]);
    let mut fold4 = vec![F128::ZERO; 16 * n_packed];
    for high in 0..4 {
        for bank in 0..16 {
            let src = (bank + 16 * high) * n_packed;
            let dst = bank * n_packed;
            crate::field::f128_slice::add_scaled(
                &mut fold4[dst..dst + n_packed],
                &fold8[src..src + n_packed],
                retained_top_eq[high],
            );
        }
    }
    let retained_hi_eq = build_eq(&inner_tail[2..4]);
    let mut quad = vec![F128::ZERO; 4 * n_packed];
    for q in 0..4 {
        for e in 0..4 {
            let src = (e + 4 * q) * n_packed;
            let dst = e * n_packed;
            crate::field::f128_slice::add_scaled(
                &mut quad[dst..dst + n_packed],
                &fold4[src..src + n_packed],
                retained_hi_eq[q],
            );
        }
    }
    let s_hat = crate::pcs::ring_switch::collapse_s_hat_v_quad(&quad, &inner_tail[..2]);
    let c_s_inv = c_s_inv_for_identity_c();
    let mut res_s = [F128::ZERO; ELL];
    for lane in 0..ELL {
        let naive = (F128::ONE + prefix) * s_hat[lane] + prefix * s_hat[ELL + lane];
        res_s[lane] = c_s_inv * naive;
    }
    ntt_extend_f128_vec_ghash(&res_s, inv_table)
}

pub fn round1_c_fold4_from_block_major_z(
    z_packed: &[F128],
    m: usize,
    k_log: usize,
    k_skip: usize,
    useful_bits: usize,
    r: &[F128],
    inv_table: &InvNttTableByteSingleGf8,
    ranked_one_rows: bool,
) -> (
    Vec<F128>,
    Vec<F128>,
    Vec<F128>,
    Vec<F128>,
    Vec<F128>,
    Option<Vec<F128>>,
) {
    assert_eq!(k_skip, K_SKIP);
    assert!(
        k_log >= k_skip + 7,
        "Fold8 needs six retained tail coordinates"
    );
    assert_eq!(r.len(), m);
    assert_eq!(z_packed.len(), (1usize << m) / 128);
    assert_eq!(inv_table.k, k_skip);
    if ranked_one_rows {
        assert_eq!(m, 32);
        assert_eq!(k_log, 14);
        assert_eq!(useful_bits, 15409);
    }

    let inner_tail = &r[k_skip + 1..k_log];
    let n_packed = 1usize << crate::pcs::LOG_PACKING;
    let par = crate::serial_par_enabled();
    let (fold4, fold8, one_fold8) = if crate::pcs::ranked_direct_fold8_enabled() {
        // Ranked shape has one coordinate above the six Fold8 bank
        // selectors. On the parallel/GFNI path, bind it while reducing the
        // worker byte planes so the full length-2^k_log C inner table is
        // never written and immediately read back by a second Rayon pass.
        let (fold8, one_fold8) = if par && inner_tail.len() == 7 {
            if ranked_one_rows {
                let (full, one) =
                    crate::lincheck::fold_block_major_one_shot_bind_top_ranked_one_rows(
                        z_packed,
                        m,
                        k_log,
                        useful_bits,
                        &r[k_log..],
                        inner_tail[6],
                    );
                (full, Some(one))
            } else {
                (
                    crate::lincheck::fold_block_major_one_shot_bind_top(
                        z_packed,
                        m,
                        k_log,
                        useful_bits,
                        &r[k_log..],
                        inner_tail[6],
                    ),
                    None,
                )
            }
        } else {
            let c_inner = identity_c_inner_fold(z_packed, m, k_log, useful_bits, &r[k_log..], par);
            let one = ranked_one_rows.then(|| {
                let mut one_inner = vec![F128::ZERO; 1 << k_log];
                one_inner[..1152].copy_from_slice(&c_inner[..1152]);
                one_inner[15104..15360].copy_from_slice(&c_inner[15104..15360]);
                crate::pcs::ring_switch::s_hat_v_fold8_from_z_vec(&one_inner, inner_tail)
            });
            (
                crate::pcs::ring_switch::s_hat_v_fold8_from_z_vec(&c_inner, inner_tail),
                one,
            )
        };
        // Collapse retained coordinates 4 and 5 to recover Fold4 exactly.
        let retained_top_eq = build_eq(&inner_tail[4..6]);
        let mut fold4 = vec![F128::ZERO; 16 * n_packed];
        let wide = r1_cfold_x4_enabled();
        for high in 0..4 {
            for bank in 0..16 {
                let src = (bank + 16 * high) * n_packed;
                let dst = bank * n_packed;
                if wide {
                    crate::field::f128_slice::add_scaled(
                        &mut fold4[dst..dst + n_packed],
                        &fold8[src..src + n_packed],
                        retained_top_eq[high],
                    );
                } else {
                    for packed in 0..n_packed {
                        fold4[dst + packed] += retained_top_eq[high] * fold8[src + packed];
                    }
                }
            }
        }
        (fold4, fold8, one_fold8)
    } else {
        assert!(
            !ranked_one_rows,
            "one-row reuse requires ranked DirectFold8"
        );
        // Kill switch restores the incumbent sixteen-bank producer; do not
        // pay for widening and collapsing a statistic no consumer will use.
        let c_inner = identity_c_inner_fold(z_packed, m, k_log, useful_bits, &r[k_log..], par);
        (
            crate::pcs::ring_switch::s_hat_v_fold4_from_z_vec(&c_inner, inner_tail),
            Vec::new(),
            None,
        )
    };

    // Fold retained coordinates 2 and 3 to recover the incumbent four-bank
    // tensor (coordinates 0 and 1 stay bank selectors).
    let retained_hi_eq = build_eq(&inner_tail[2..4]);
    let mut quad = vec![F128::ZERO; 4 * n_packed];
    let wide_quad = r1_cfold_x4_enabled();
    for q in 0..4 {
        for e in 0..4 {
            let src = (e + 4 * q) * n_packed;
            let dst = e * n_packed;
            if wide_quad {
                crate::field::f128_slice::add_scaled(
                    &mut quad[dst..dst + n_packed],
                    &fold4[src..src + n_packed],
                    retained_hi_eq[q],
                );
            } else {
                for packed in 0..n_packed {
                    quad[dst + packed] += retained_hi_eq[q] * fold4[src + packed];
                }
            }
        }
    }
    let s_hat_v_c = crate::pcs::ring_switch::collapse_s_hat_v_quad(&quad, &inner_tail[..2]);

    // RingSwitch leaves global bit `k_skip` as its 128-way prefix; folding that
    // bit at the original C point recovers C's 64 S-domain evaluations.
    let prefix = r[k_skip];
    let mut res_c_s = [F128::ZERO; ELL];
    let c_s_inv = c_s_inv_for_identity_c();
    for lane in 0..ELL {
        let naive = (F128::ONE + prefix) * s_hat_v_c[lane] + prefix * s_hat_v_c[ELL + lane];
        res_c_s[lane] = c_s_inv * naive;
    }
    let res_c_lifted = ntt_extend_f128_vec_ghash(&res_c_s, inv_table);
    let one_ab_lifted = one_fold8
        .map(|one_fold8| round1_lifted_from_fold8(&one_fold8, inner_tail, prefix, inv_table));
    (res_c_lifted, s_hat_v_c, quad, fold4, fold8, one_ab_lifted)
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

    #[test]
    fn c_s_subfield_inverse_matches_fermat_exhaustively() {
        use std::ffi::OsStr;

        // Prove the embedding/inversion identity for every nonzero F_8
        // element, not only for the selected protocol constant.
        for value in 1u16..=u8::MAX as u16 {
            let a8 = F8(value as u8);
            let inv8 = a8.inv();
            assert_eq!(a8 * inv8, F8::ONE, "F8 inverse value={value:#04x}");
            assert_eq!(
                phi8(inv8),
                phi8(a8).inv(),
                "embedded inverse value={value:#04x}"
            );
        }

        assert_eq!(F8(C_S_F8).inv(), F8(C_S_INV_F8));
        assert_eq!(c_s_f128() * c_s_inv_f128(), F128::ONE);
        assert_eq!(c_s_inv_f128(), c_s_inv_fermat_fallback());

        assert!(c_s_subfield_inv_disabled_value(Some(OsStr::new("1"))));
        for value in [
            None,
            Some(OsStr::new("")),
            Some(OsStr::new("0")),
            Some(OsStr::new("01")),
            Some(OsStr::new("true")),
            Some(OsStr::new(" 1")),
        ] {
            assert!(!c_s_subfield_inv_disabled_value(value));
        }
    }

    /// The gated identity-C outer fold must match the incumbent
    /// dense-eq-table oracle bit-for-bit — on the ranked `n_log = 18`
    /// factorized dispatch and on a dense-arm geometry — and a corrupted
    /// witness must be caught.
    #[test]
    fn identity_c_inner_fold_par_matches_seq() {
        // (m, k_log, useful_bits): first case hits the factorized n_log=18
        // arm without a full LOG2=18 prove; second stays on the dense arm.
        let cases: &[(usize, usize, usize)] = &[(25, 7, 121), (16, 8, 241)];
        for &(m, k_log, useful_bits) in cases {
            let mut rng = Rng(0x1DC0_11EC + (m * 131 + k_log) as u64);
            let n_outer = 1usize << (m - k_log);
            let chunks_per_block = (1usize << k_log) / 128;
            let z: Vec<F128> = rng.f128_vec(n_outer * chunks_per_block);
            let r_outer: Vec<F128> = rng.f128_vec(m - k_log);
            let seq = identity_c_inner_fold(&z, m, k_log, useful_bits, &r_outer, false);
            let par = identity_c_inner_fold(&z, m, k_log, useful_bits, &r_outer, true);
            assert_eq!(par, seq, "m={m} k_log={k_log}");
            // Negative control: a corrupted useful witness word must reach
            // the folded table through both routes.
            let mut bad = z.clone();
            bad[0] += F128::ONE;
            assert_ne!(
                identity_c_inner_fold(&bad, m, k_log, useful_bits, &r_outer, true),
                seq,
                "corrupted witness went undetected m={m}"
            );
        }
    }

    /// The gated parallel GFNI matrix build must match the sequential
    /// oracle bit-for-bit, and a corrupted scale must be caught.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    ))]
    #[test]
    fn ab_eq_fold_mats_par_matches_seq() {
        let mut rng = Rng(0xAB_E0_F01D);
        let convert = convert_table();
        // Ranked width (32 rows, above the par floor) and a tiny width
        // (below the floor — par falls back and must still match).
        for &n_w in &[32usize, 2] {
            let eq_top_scaled: Vec<F128> = rng.f128_vec(n_w);
            let seq = build_ab_eq_fold_mats_gated(&eq_top_scaled, convert, false);
            let par = build_ab_eq_fold_mats_gated(&eq_top_scaled, convert, true);
            assert_eq!(par, seq, "n_w={n_w}");
            let mut bad = eq_top_scaled.clone();
            bad[0] += F128::ONE;
            assert_ne!(
                build_ab_eq_fold_mats_gated(&bad, convert, true),
                seq,
                "corrupted scale went undetected n_w={n_w}"
            );
        }
    }

    /// A first-visit bank overwrite must equal the incumbent accumulate into
    /// a zeroed bank for every supported medium-row count. Poisoning the
    /// destination catches any accidental read or partial write.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    ))]
    #[test]
    fn ab_eq_fold_first_write_matches_zeroed_accumulate() {
        let mut word = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            word ^= word << 13;
            word ^= word >> 7;
            word ^= word << 17;
            word
        };
        let mut chunk_ab_bytes = [[0u8; ELL]; 1 << N_MEDIUM];
        for row in &mut chunk_ab_bytes {
            for byte in row {
                *byte = next() as u8;
            }
        }
        let mut mats = [0u64; 256];
        for matrix in &mut mats {
            *matrix = next();
        }

        for n_b_med in 1..=1 << N_MEDIUM {
            let mut incumbent = [0u8; 16 * ELL];
            let mut first_write = [0xA5u8; 16 * ELL];
            kernels::accumulate_convert_ab_nomul_gfni(
                &chunk_ab_bytes,
                n_b_med,
                &mats,
                &mut incumbent,
            );
            kernels::write_convert_ab_nomul_gfni(&chunk_ab_bytes, n_b_med, &mats, &mut first_write);
            assert_eq!(first_write, incumbent, "n_b_med={n_b_med}");
        }
    }

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
            );
            assert_eq!(
                out_scalar, out_fused,
                "fused-neon disagrees with scalar at (base={chunk_byte_base}, b_med={b_med})"
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_fused_x2_matches_scalar_inner() {
        // The two-window wavefront kernel must produce, per window, exactly
        // the bytes of the scalar oracle (and hence of the single-window
        // fused kernel) for both windows of every pair.
        let mut rng = Rng::new(0xF050D_2);
        let m = 14;
        let table = make_inv_table();
        let a_bits = rng.bits(1 << m);
        let b_bits = rng.bits(1 << m);
        let a_packed = super::super::univariate_skip::pack_bits(&a_bits);
        let b_packed = super::super::univariate_skip::pack_bits(&b_bits);

        let mut a_col = vec![F8::ZERO; ELL];
        let mut b_col = vec![F8::ZERO; ELL];

        for &(chunk_byte_base, b_med) in &[(0usize, 0usize), (64, 4), (1024, 6), (4096, 14)] {
            let needed = chunk_byte_base + (b_med + 1) * N_CHUNKS * 8 + 8 * N_CHUNKS;
            if needed > a_packed.len() {
                continue;
            }
            let mut out_scalar_0 = [0u8; 64];
            let mut out_scalar_1 = [0u8; 64];
            let mut out_x2_0 = [0u8; 64];
            let mut out_x2_1 = [0u8; 64];
            shift_reduce_inner_ab_scalar(
                &a_packed,
                &b_packed,
                &table,
                chunk_byte_base,
                b_med,
                &mut out_scalar_0,
                &mut a_col,
                &mut b_col,
            );
            shift_reduce_inner_ab_scalar(
                &a_packed,
                &b_packed,
                &table,
                chunk_byte_base,
                b_med + 1,
                &mut out_scalar_1,
                &mut a_col,
                &mut b_col,
            );
            shift_reduce_inner_ab_fused_neon_x2(
                &a_packed,
                &b_packed,
                &table,
                chunk_byte_base,
                b_med,
                &mut out_x2_0,
                &mut out_x2_1,
            );
            assert_eq!(
                out_scalar_0, out_x2_0,
                "x2 window 0 disagrees with scalar at (base={chunk_byte_base}, b_med={b_med})"
            );
            assert_eq!(
                out_scalar_1,
                out_x2_1,
                "x2 window 1 disagrees with scalar at (base={chunk_byte_base}, b_med={})",
                b_med + 1
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
            // All three terminal-store classes must produce identical bytes;
            // the NT classes need the alignment their contract demands, which
            // a 64-aligned repr(align) wrapper provides.
            #[repr(align(64))]
            struct Aligned64([u8; 64]);
            for nt in [0u8, 1, 2] {
                let mut out_avx512 = Aligned64([0u8; 64]);
                // SAFETY: test compiles only when all kernel features are
                // active; the wrapper satisfies the nt=1/2 alignment contract.
                unsafe {
                    shift_reduce_inner_ab_x86_avx512(
                        &a_packed,
                        &b_packed,
                        &table,
                        chunk_byte_base,
                        b_med,
                        &mut out_avx512.0,
                        nt,
                    );
                    core::arch::x86_64::_mm_sfence();
                }
                assert_eq!(
                    out_scalar, out_avx512.0,
                    "avx512/gfni (nt={nt}) disagrees with scalar at (base={chunk_byte_base}, b_med={b_med})"
                );
            }
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

    /// Splitting the challenge-independent AB transform from the later eq
    /// fold must not change any round-1 wire value or the captured C opening
    /// helper.  Cover the smallest three legal dimensions so both unsplit and
    /// split eq-table shapes are exercised cheaply.
    #[test]
    fn precomputed_ab_matches_fused_at_m13_through_m15() {
        for &m in &[13usize, 14, 15] {
            let mut rng = Rng::new(0xAB00_0000_u64.wrapping_add(m as u64));
            let a = pack_bits(&rng.bits(1 << m));
            let b = pack_bits(&rng.bits(1 << m));
            let c = pack_bits(&rng.bits(1 << m));
            let outer = rng.f128_vec(m - K_SKIP - N_INNER);
            let r = build_protocol_r(m, &outer);
            let inv_table = make_inv_table();
            let padding = PaddingSpec::dense(m);

            let expected = round1_shift_reduce_extract_c_packed_padded_with_s_hat_v(
                &a, &b, &c, m, K_SKIP, &r, &inv_table, &padding,
            );
            let mut precomputed =
                precompute_round1_ab_inner_packed_padded(&a, &b, m, K_SKIP, &inv_table, &padding);
            assert_eq!(precomputed.len_bytes(), a.len());
            let got = round1_shift_reduce_extract_c_packed_padded_with_precomputed_ab(
                &mut precomputed,
                &a,
                &b,
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

    /// **GPU split acceptance (season-1 hook shape)**: a forced CPU/GPU split
    /// (`g = max(1, hi_size/2)`) through BOTH round-1 entry points must be
    /// bit-identical to the forced pure-CPU run (`g = 0`) — the merge is a
    /// per-lane XOR of eq_hi-folded partials, so any drift is a kernel or
    /// wiring bug, not noise. SKIPS (does not fail) without Metal.
    #[test]
    fn gpu_forced_split_matches_pure_cpu() {
        if !crate::gpu::metal_available() {
            eprintln!("SKIP gpu_forced_split_matches_pure_cpu: Metal unavailable");
            return;
        }
        for &m in &[14usize, 15, 16] {
            let mut rng = Rng::new(0x6B0_5117 ^ m as u64);
            let a = pack_bits(&rng.bits(1 << m));
            let b = pack_bits(&rng.bits(1 << m));
            let c = pack_bits(&rng.bits(1 << m));
            let outer = rng.f128_vec(m - K_SKIP - N_INNER);
            let r = build_protocol_r(m, &outer);
            let inv_table = make_inv_table();
            let padding = PaddingSpec::dense(m);

            let hi_size = 1usize << SplitEqGhash::new(&r[K_SKIP + N_INNER..]).n_hi;
            let g = (hi_size / 2).max(1);

            let pure_cpu =
                round1_with_s_hat_v_impl(&a, &b, &c, m, K_SKIP, &r, &inv_table, &padding, Some(0));
            let split =
                round1_with_s_hat_v_impl(&a, &b, &c, m, K_SKIP, &r, &inv_table, &padding, Some(g));
            assert_eq!(split, pure_cpu, "s_hat_v split mismatch at m={m} g={g}");

            let mut precomputed =
                precompute_round1_ab_inner_packed_padded(&a, &b, m, K_SKIP, &inv_table, &padding);
            let split_pre = round1_with_precomputed_ab_impl(
                &mut precomputed,
                &a,
                &b,
                &c,
                m,
                K_SKIP,
                &r,
                &inv_table,
                &padding,
                Some(g),
            );
            assert_eq!(
                split_pre, pure_cpu,
                "precomputed-AB split mismatch at m={m} g={g}"
            );
            assert!(
                !crate::gpu::is_disabled(),
                "GPU share failed to start/finish at m={m} — split ran CPU-only"
            );
        }
    }

    /// `fill_invalid_prefix` must reproduce the skipped prefix byte-for-byte:
    /// scribbling the prefix, marking it invalid, and filling must equal the
    /// untouched full precompute. m = 20 is the smallest hi_size = 128 shape
    /// (one x_hi window = one 1024-byte outer window).
    #[test]
    fn fill_invalid_prefix_reproduces_precompute() {
        let m = 20usize;
        let mut rng = Rng::new(0xF111_2020);
        let a = pack_bits(&rng.bits(1 << m));
        let b = pack_bits(&rng.bits(1 << m));
        let inv_table = make_inv_table();
        let padding = PaddingSpec::dense(m);

        let mut full =
            precompute_round1_ab_inner_packed_padded(&a, &b, m, K_SKIP, &inv_table, &padding);
        let mut skipped =
            precompute_round1_ab_inner_packed_padded(&a, &b, m, K_SKIP, &inv_table, &padding);

        let bytes_per_window = ((1usize << m) / 8) >> 7; // hi_size = 128
        let skip_bytes = 5 * bytes_per_window;
        skipped.as_bytes_mut()[..skip_bytes].fill(0xA5);
        skipped.set_invalid_prefix_bytes(skip_bytes);
        skipped.fill_invalid_prefix(&a, &b, &inv_table);

        assert_eq!(skipped.invalid_prefix_bytes(), 0);
        assert_eq!(
            skipped.as_bytes_mut(),
            full.as_bytes_mut(),
            "fill_invalid_prefix drifted from the standalone precompute"
        );
    }

    #[test]
    fn one_rows_fallback_restores_dense_storage() {
        const BYTES: usize = 2048;
        let mut rng = Rng::new(0x0A8D_E115);
        let a = pack_bits(&rng.bits(BYTES * 8));
        let b = pack_bits(&rng.bits(BYTES * 8));
        let inv_table = make_inv_table();
        let mut expected = vec![0u8; BYTES];
        precompute_round1_ab_inner_windows(&a, &b, &mut expected, &inv_table, false);

        let mut residual = Round1AbInner::take_uninit(BYTES);
        residual.as_bytes_mut().fill(0xA5);
        residual.set_ranked_one_rows_elided();
        residual.restore_full_if_ranked_one_rows_elided(&a, &b, &inv_table);
        assert!(!residual.ranked_one_rows_elided());
        assert_eq!(residual.invalid_prefix_bytes(), 0);
        assert_eq!(residual.as_bytes(), expected.as_slice());

        // A dense object keeps its ordinary invalid-prefix contract. The
        // residual repair must not touch it or require raw input slices.
        residual.as_bytes_mut().fill(0x5A);
        residual.set_invalid_prefix_bytes(1024);
        residual.restore_full_if_ranked_one_rows_elided(&[], &[], &inv_table);
        assert_eq!(residual.invalid_prefix_bytes(), 1024);
        assert!(residual.as_bytes().iter().all(|byte| *byte == 0x5A));
    }

    #[test]
    #[should_panic]
    fn one_rows_fallback_rejects_wrong_raw_extent() {
        let inv_table = make_inv_table();
        let mut residual = Round1AbInner::take_uninit(1024);
        residual.set_ranked_one_rows_elided();
        residual.restore_full_if_ranked_one_rows_elided(&[], &[], &inv_table);
    }

    /// Round 1 with a producer-skipped (invalid) ab_inner prefix must be
    /// bit-identical to the fully-precomputed pure-CPU run. The skipped
    /// windows are covered by the forced GPU share where Metal exists, and
    /// by the CPU `fill_invalid_prefix` fallback otherwise — both paths are
    /// exercised across machines, both must agree with the oracle.
    #[test]
    fn skipped_prefix_matches_full_precompute() {
        let m = 20usize;
        let mut rng = Rng::new(0x5C1_0BEEF);
        let a = pack_bits(&rng.bits(1 << m));
        let b = pack_bits(&rng.bits(1 << m));
        let c = pack_bits(&rng.bits(1 << m));
        let outer = rng.f128_vec(m - K_SKIP - N_INNER);
        let r = build_protocol_r(m, &outer);
        let inv_table = make_inv_table();
        let padding = PaddingSpec::dense(m);

        let mut full =
            precompute_round1_ab_inner_packed_padded(&a, &b, m, K_SKIP, &inv_table, &padding);
        let expected = round1_with_precomputed_ab_impl(
            &mut full,
            &a,
            &b,
            &c,
            m,
            K_SKIP,
            &r,
            &inv_table,
            &padding,
            Some(0),
        );

        let bytes_per_window = ((1usize << m) / 8) >> 7;
        for &skipped_w in &[1usize, 7, 128] {
            let mut pre =
                precompute_round1_ab_inner_packed_padded(&a, &b, m, K_SKIP, &inv_table, &padding);
            pre.as_bytes_mut()[..skipped_w * bytes_per_window].fill(0x5A);
            pre.set_invalid_prefix_bytes(skipped_w * bytes_per_window);
            let got = round1_with_precomputed_ab_impl(
                &mut pre,
                &a,
                &b,
                &c,
                m,
                K_SKIP,
                &r,
                &inv_table,
                &padding,
                Some(0),
            );
            assert_eq!(
                got, expected,
                "skipped-prefix round1 mismatch at skipped_w={skipped_w}"
            );
        }
    }
    #[test]
    fn quad_collapses_to_wire_s_hat_v_c() {
        use crate::pcs::ring_switch::collapse_s_hat_v_quad;
        use crate::zerocheck::univariate_skip::pack_bits;

        let small = small_challenges_ghash();
        let low_point = [small[1], small[2]];
        let cases = [
            (13usize, None),
            (14, Some((14usize, 15_409usize))),
            (17, Some((14usize, 15_409usize))),
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
                    for block in 0..(total_bits / block_size) {
                        for offset in useful_bits..block_size {
                            let index = block * block_size + offset;
                            a[index] = false;
                            b[index] = false;
                            c[index] = false;
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
            let fused = round1_shift_reduce_extract_c_packed_padded_with_s_hat_v_quad(
                &a_p, &b_p, &c_p, m, K_SKIP, &r, &table, &padding,
            );
            assert_eq!(fused.3.len(), 4 * 2 * ELL);
            assert_eq!(collapse_s_hat_v_quad(&fused.3, &low_point), fused.2);

            let mut precomputed =
                precompute_round1_ab_inner_packed_padded(&a_p, &b_p, m, K_SKIP, &table, &padding);
            let pre = round1_shift_reduce_extract_c_packed_padded_with_precomputed_ab_quad(
                &mut precomputed,
                &a_p,
                &b_p,
                &c_p,
                m,
                K_SKIP,
                &r,
                &table,
                &padding,
            );
            assert_eq!(pre, fused, "precomputed DirectC mismatch at m={m}");
        }
    }

    /// The four-window 32-bank fold4 producer must reproduce the incumbent
    /// `(ab, c_lifted, s_hat_v_c, quad_c)` BYTE-FOR-BYTE (its 32→8 collapse is
    /// exact field algebra), and its sixteen-bank tensor must collapse under
    /// `suffix[..4] = [small₁, small₂, β₀, β₁]` to the wire `s_hat_v_c`.
    /// m=22 is the smallest shape where the outer split leaves a
    /// multiple-of-four low half (`c_fold4_capture_available`); m=24 adds a
    /// non-trivial `n_hi`.
    /// **Identity-C differential oracle.** At the ranked BLAKE3 shape C is the
    /// identity, so folding the packed witness at `r[k_log..]` and retaining
    /// four inner coordinates must reproduce the 32-bank row-major drain
    /// exactly — all five round-one outputs, bit for bit, on dense and
    /// honestly-padded witnesses.
    #[test]
    fn identity_c_fold_matches_row_major_fold4_drain() {
        use crate::zerocheck::univariate_skip::pack_bits;

        for (m, useful_bits) in [(22usize, 15_409usize), (24, 15_409), (22, 1usize << 14)] {
            const K_LOG: usize = 14;
            let mut rng = Rng::new(0x1DC_5721_u64.wrapping_add((m * 100_000 + useful_bits) as u64));
            let total_bits = 1usize << m;
            let mut a = rng.bits(total_bits);
            let mut b = rng.bits(total_bits);
            let mut c = rng.bits(total_bits);
            let block_size = 1usize << K_LOG;
            for block in 0..(total_bits / block_size) {
                for offset in useful_bits..block_size {
                    let index = block * block_size + offset;
                    a[index] = false;
                    b[index] = false;
                    c[index] = false;
                }
            }
            let padding = PaddingSpec {
                k_log: K_LOG,
                useful_bits_per_block: useful_bits,
            };
            let (a_p, b_p, c_p) = (pack_bits(&a), pack_bits(&b), pack_bits(&c));
            // The packed byte buffer and the F128 word buffer are the same
            // bits in the same order; the fold wants the word view.
            let c_words: Vec<F128> = c_p
                .chunks_exact(16)
                .map(|w| F128 {
                    lo: u64::from_le_bytes(w[..8].try_into().unwrap()),
                    hi: u64::from_le_bytes(w[8..].try_into().unwrap()),
                })
                .collect();

            let outer = rng.f128_vec(m - K_SKIP - N_INNER);
            let r = build_protocol_r(m, &outer);
            let table = make_inv_table();

            let mut precomputed =
                precompute_round1_ab_inner_packed_padded(&a_p, &b_p, m, K_SKIP, &table, &padding);
            let (ab_ref, c_ref, s_hat_ref, quad_ref, fold4_ref) =
                round1_shift_reduce_extract_c_packed_padded_with_precomputed_ab_fold4(
                    &mut precomputed,
                    &a_p,
                    &b_p,
                    &c_p,
                    m,
                    K_SKIP,
                    &r,
                    &table,
                    &padding,
                );

            let mut precomputed_ab_only =
                precompute_round1_ab_inner_packed_padded(&a_p, &b_p, m, K_SKIP, &table, &padding);
            let ab_new = round1_shift_reduce_ab_packed_padded_with_precomputed(
                &mut precomputed_ab_only,
                &a_p,
                &b_p,
                m,
                K_SKIP,
                &r,
                &table,
                &padding,
            );
            let (c_new, s_hat_new, quad_new, fold4_new, fold8_new, one_ab) =
                round1_c_fold4_from_block_major_z(
                    &c_words,
                    m,
                    K_LOG,
                    K_SKIP,
                    useful_bits,
                    &r,
                    &table,
                    false,
                );
            assert!(one_ab.is_none());

            assert_eq!(
                ab_new, ab_ref,
                "AB-only mismatch at m={m} useful={useful_bits}"
            );
            assert_eq!(
                c_new, c_ref,
                "stripe C message mismatch at m={m} useful={useful_bits}"
            );
            assert_eq!(
                s_hat_new, s_hat_ref,
                "stripe s_hat_v_c mismatch at m={m} useful={useful_bits}"
            );
            assert_eq!(
                quad_new, quad_ref,
                "stripe quad mismatch at m={m} useful={useful_bits}"
            );
            assert_eq!(
                fold4_new, fold4_ref,
                "stripe fold4 tensor mismatch at m={m} useful={useful_bits}"
            );
            if crate::pcs::ranked_direct_fold8_enabled() {
                assert_eq!(
                    crate::pcs::ring_switch::collapse_s_hat_v_fold8(
                        &fold8_new,
                        &r[K_SKIP + 1..K_SKIP + 7],
                    ),
                    s_hat_new,
                    "stripe fold8 tensor mismatch at m={m} useful={useful_bits}"
                );
            } else {
                assert!(fold8_new.is_empty(), "Fold8 kill switch still widened C");
            }
        }
    }

    #[test]
    fn fold4_c_capture_matches_quad_and_collapses() {
        use crate::pcs::ring_switch::{collapse_s_hat_v_fold4, collapse_s_hat_v_quad};
        use crate::zerocheck::univariate_skip::pack_bits;

        let small = small_challenges_ghash();
        let beta = medium_challenges_ghash();
        let low_point4 = [small[1], small[2], beta[0], beta[1]];
        let low_point2 = [small[1], small[2]];
        assert!(!c_fold4_capture_available(21, K_SKIP));
        assert!(c_fold4_capture_available(22, K_SKIP));
        assert!(c_fold4_capture_available(32, K_SKIP));
        // Collapse weights are exactly eq(β₀,β₁; q) = X^q·D_lo⁻¹.
        let w = c_fold4_q_weights();
        let d_lo_inv =
            ((F128::ONE + F128 { lo: 2, hi: 0 }) * (F128::ONE + F128 { lo: 4, hi: 0 })).inv();
        for q in 0..4usize {
            assert_eq!(
                w[q],
                F128 {
                    lo: 1u64 << q,
                    hi: 0
                } * d_lo_inv,
                "q weight {q}"
            );
        }
        assert_eq!(d_hi_inv() * d_lo_inv, d_inv(), "D⁻¹ = D_lo⁻¹·D_hi⁻¹");

        let cases = [
            (22usize, None),
            (22, Some((14usize, 15_409usize))),
            (24, Some((14usize, 15_409usize))),
        ];
        for (m, padded) in cases {
            let mut rng = Rng::new(0xF01D_4C0D_u64.wrapping_add(m as u64));
            let total_bits = 1usize << m;
            let mut a = rng.bits(total_bits);
            let mut b = rng.bits(total_bits);
            let mut c = rng.bits(total_bits);
            let padding = match padded {
                None => PaddingSpec::dense(m),
                Some((k_log, useful_bits)) => {
                    let block_size = 1usize << k_log;
                    for block in 0..(total_bits / block_size) {
                        for offset in useful_bits..block_size {
                            let index = block * block_size + offset;
                            a[index] = false;
                            b[index] = false;
                            c[index] = false;
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
            let mut precomputed =
                precompute_round1_ab_inner_packed_padded(&a_p, &b_p, m, K_SKIP, &table, &padding);
            let quad_route = round1_shift_reduce_extract_c_packed_padded_with_precomputed_ab_quad(
                &mut precomputed,
                &a_p,
                &b_p,
                &c_p,
                m,
                K_SKIP,
                &r,
                &table,
                &padding,
            );
            let mut precomputed4 =
                precompute_round1_ab_inner_packed_padded(&a_p, &b_p, m, K_SKIP, &table, &padding);
            let (ab, c_l, s_hat_v_c, quad_c, fold4_c) =
                round1_shift_reduce_extract_c_packed_padded_with_precomputed_ab_fold4(
                    &mut precomputed4,
                    &a_p,
                    &b_p,
                    &c_p,
                    m,
                    K_SKIP,
                    &r,
                    &table,
                    &padding,
                );
            assert_eq!(ab, quad_route.0, "fold4 route AB mismatch at m={m}");
            assert_eq!(c_l, quad_route.1, "fold4 route C-lifted mismatch at m={m}");
            assert_eq!(
                s_hat_v_c, quad_route.2,
                "fold4 route s_hat_v_c mismatch at m={m}"
            );
            assert_eq!(quad_c, quad_route.3, "fold4 route quad_c mismatch at m={m}");
            assert_eq!(fold4_c.len(), 16 * 2 * ELL);
            assert_eq!(
                collapse_s_hat_v_fold4(&fold4_c, &low_point4),
                s_hat_v_c,
                "fold4 tensor must collapse to the wire s_hat_v_c at m={m}"
            );
            assert_eq!(collapse_s_hat_v_quad(&quad_c, &low_point2), s_hat_v_c);
            // The 16 fold4 banks collapse under (β₀,β₁) to the 4 quad banks.
            let n_packed = 2 * ELL;
            for e in 0..4 {
                for packed in 0..n_packed {
                    let mut acc = F128::ZERO;
                    for q in 0..4 {
                        acc += w[q] * fold4_c[(e + 4 * q) * n_packed + packed];
                    }
                    assert_eq!(
                        acc,
                        quad_c[e * n_packed + packed],
                        "bank collapse e={e} p={packed}"
                    );
                }
            }
        }
    }

    /// The fused GFNI DirectFold4 C drain must reproduce the incumbent
    /// `bit_transpose_64bytes` + LUT drain byte-for-byte: same synthetic mask
    /// weights, same padded row geometry, same multi-group accumulation. Both
    /// accumulators are POISON-prefilled with the identical value (every byte
    /// plane `0xA5` is exactly every F128 lane `0xA5A5..A5`), so a bank slot
    /// the kernel failed to touch would show up as a mismatch.
    #[test]
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq",
        target_feature = "gfni"
    ))]
    fn fused_gfni_c_drain_matches_transpose_drain() {
        const GROUP_BYTES: usize = 4 * (1 << N_MEDIUM) * ELL;
        const N_GROUPS: usize = 3;
        let mut rng = Rng::new(0xC0FF_EE17_4C42);
        for trial in 0..6usize {
            let mut c_all = vec![0u8; N_GROUPS * GROUP_BYTES];
            for byte in c_all.iter_mut() {
                *byte = (rng.next_u64() >> 13) as u8;
            }
            // trial 0 is the dense geometry; the rest exercise ragged padding
            // (including all-dead windows, which the driver still drains).
            let counts: Vec<[usize; 4]> = (0..N_GROUPS)
                .map(|_| {
                    std::array::from_fn(|_| {
                        if trial == 0 {
                            16
                        } else {
                            (rng.next_u64() % 17) as usize
                        }
                    })
                })
                .collect();
            let eq_lo: Vec<F128> = (0..4 * N_GROUPS).map(|_| rng.f128()).collect();
            let tables = build_c_fold4_tables(&eq_lo);
            let mats = build_c_fold4_gfni_mats(&eq_lo);

            let poison = F128 {
                lo: 0xA5A5_A5A5_A5A5_A5A5,
                hi: 0xA5A5_A5A5_A5A5_A5A5,
            };
            let mut banks = vec![[[poison; ELL]; N_C_BANKS]; N_C_Q];
            let mut planes = vec![0xA5u8; C_PLANE_BANK_BYTES];

            for g in 0..N_GROUPS {
                let group = &c_all[g * GROUP_BYTES..(g + 1) * GROUP_BYTES];
                let mut chunk_c4 = vec![[[0u8; 64]; 16]; N_C_Q];
                for w in 0..4 {
                    for b_med in 0..(1 << N_MEDIUM) {
                        let dst = &mut chunk_c4[b_med & 3][4 * w + (b_med >> 2)];
                        if b_med < counts[g][w] {
                            let base = w * (1 << N_MEDIUM) * ELL + b_med * ELL;
                            let c_in: &[u8; 64] =
                                group[base..base + 64].try_into().expect("64 c-bytes");
                            bit_transpose_64bytes(c_in, dst);
                        } else {
                            dst.fill(0);
                        }
                    }
                }
                let c_tables = &tables[g * C_MASK_TABLE_STRIDE..(g + 1) * C_MASK_TABLE_STRIDE];
                for q in 0..N_C_Q {
                    // SAFETY: `[[u8; 64]; 16]` is contiguous, exactly 1024 bytes.
                    let c_block: &[u8; 16 * 64] =
                        unsafe { &*chunk_c4[q].as_ptr().cast::<[u8; 16 * 64]>() };
                    kernels::accumulate_c_banks(c_block, 1 << N_MEDIUM, c_tables, &mut banks[q]);
                }
                let mats_g: &[u64; C_FOLD4_MATS_PER_GROUP] = mats
                    [g * C_FOLD4_MATS_PER_GROUP..(g + 1) * C_FOLD4_MATS_PER_GROUP]
                    .try_into()
                    .expect("32 matrix qwords per group");
                let planes_arr: &mut [u8; C_PLANE_BANK_BYTES] = planes
                    .as_mut_slice()
                    .try_into()
                    .expect("32 KiB plane store");
                kernels::accumulate_c_banks_fold4_fused_gfni(group, &counts[g], mats_g, planes_arr);
            }

            for q in 0..N_C_Q {
                for bank in 0..N_C_BANKS {
                    let bank_planes = &planes[(q * N_C_BANKS + bank) * 16 * ELL..][..16 * ELL];
                    for lane in 0..ELL {
                        let mut lo = 0u64;
                        let mut hi = 0u64;
                        for k in 0..8 {
                            lo |= (bank_planes[k * ELL + lane] as u64) << (8 * k);
                        }
                        for k in 8..16 {
                            hi |= (bank_planes[k * ELL + lane] as u64) << (8 * (k - 8));
                        }
                        assert_eq!(
                            F128 { lo, hi },
                            banks[q][bank][lane],
                            "trial {trial} q {q} bank {bank} lane {lane}"
                        );
                    }
                }
            }
            crate::scratch::give_f128(tables);
        }
    }
}
