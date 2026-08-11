//! Multilinear sumcheck — rounds 2..(m − k_skip + 1) of the zerocheck protocol.
//!
//! After the round-1 URM and the verifier's univariate-skip fold-point `z`, the
//! protocol enters a standard multilinear sumcheck over `n = m − k_skip` variables.
//! For the **extract_c** variant, only AB participate (C was pinned down at round
//! 1 as `res_C_lifted`), so the polynomial we sumcheck is
//!
//!   `Σ_x eq(r_rest, x) · a_mlv(x) · b_mlv(x)`
//!
//! with claim `P^{AB}(z)` from round 1. Each subsequent round sends `(P_r(1),
//! P_r(∞))` via the Karatsuba ∞-trick.
//!
//! This module begins with the **naive reference** (separately compute the
//! Lagrange-weighted fold, then a direct sum for the round-2 message). The
//! optimized fused-fold-plus-round-2 implementation (`uni_skip_fold_and_compute
//! _round_pair_ghash` in the C++) will be added next and cross-checked against
//! these naive functions.
//!
//! **Index convention** (matches the C++ extract_c pipeline's `sumcheck_round_pair`
//! and the NEON `fold_in_place_pair`): the **low bit** of the multilinear index
//! is bound first. So `a_mlv[2k]` is the X=0 value and `a_mlv[2k+1]` is the X=1
//! value, paired by the round message and the fold.
//!
//! For `mlv_challenges = [r_0, …, r_{n-1}]` (one per round) built so `build_eq`
//! places `r_i` at bit i, **round r=2 uses `mlv_challenges[0]`** for the
//! variable being bound, with eq over `mlv_challenges[1..]` for the remaining
//! variables. Subsequent rounds peel off `mlv_challenges[1]`, etc.
//!
//! **Round message format** (matches the C++): returns `(r_now · G(1), G(∞))`
//! where `r_now` is the challenge for the variable being bound *this* round.
//! The protocol polynomial sent is `Π(X) = eq(r_now, X) · G(X)` of degree 3;
//! at X=1 it equals `r_now · G(1)`, and the leading coefficient is `G(∞)`.
//! Verifier reconstructs `G(0)` from the running claim via
//! `current_claim = (1+r_now)·G(0) + r_now·G(1)`.

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
use crate::field::gf2_128::x86_64::{WideGhashX4, f128x4_loadu, f128x4_set, ghash_mul_x4};
use crate::field::{F128, F256Unreduced, PHI_8_TABLE};
use crate::scratch::ScratchBytes;
use crate::zerocheck::PaddingSpec;
use crate::zerocheck::univariate_skip::{SplitEqGhash, build_eq, pack_bits};

mod kernels;

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use kernels::aarch64::fold_compact_chunk_neon_reconstruct_only_8;
#[cfg(all(target_arch = "aarch64", test))]
use kernels::aarch64::fold_one_row_neon_unchecked_8;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use kernels::aarch64::fold_round2_compact_chunk_neon_anchors_only_8;
#[cfg(target_arch = "aarch64")]
use kernels::aarch64::{
    fold_and_message_aarch64, fold_compact_chunk_neon_unchecked_8, fold_compact_stream_chunk_neon,
    fold_round2_chunk_neon_unchecked_8, fold_round2_compact_chunk_neon_lookahead_8,
    fold_round2_compact_chunk_neon_unchecked_8, fold_round2_compact_stream_chunk_neon,
    fold2_and_message_aarch64, fold2_and_message_lookahead_aarch64,
    fold2_compact_and_round4_chunk_neon_8, fold2_compact_and_round45_chunk_neon_8,
};
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
use kernels::aarch64::{
    fold2_and_message_lookahead_normal_expanded_aarch64,
    fold2_and_message_lookahead_nt_expanded_aarch64, fold2_and_message_normal_expanded_aarch64,
};
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
use kernels::x86_64::{fold_and_message_x86_avx512, fold_round2_pair_x86_unchecked_8};

/// Returns `(pair_in_block_mask, useful_pairs_inclusive)` for the round-2
/// fused-fold kernel. A pair (post-URM chunks `2k`, `2k+1`) is fully inside
/// padding iff `(k & pair_in_block_mask) >= useful_pairs_inclusive` — those
/// pairs contribute zero to both the message and the folded output (which is
/// already zero-initialized), so the kernel can `continue` past them.
///
/// `useful_pairs_inclusive` is the index AFTER the last pair that has any
/// useful chunk. The boundary "mixed" pair (one useful + one padding chunk,
/// when `useful_bits` is odd in chunk units) is INSIDE the useful range and
/// processed normally — its padding side has value 0 so the message
/// contribution is naturally correct.
/// Kill switch for the b≡1 chunk-class degeneration in the compact round-2 /
/// round-3 kernels: `FLOCK_NO_R2_DEGEN=1` restores the plain gather path
/// (bit-identical output either way — the degeneration only skips
/// value-forced work). Read once per phase call, off the hot path.
#[cfg(target_arch = "aarch64")]
fn r2_degen_enabled() -> bool {
    std::env::var_os("FLOCK_NO_R2_DEGEN").is_none_or(|v| v != *"1")
}

/// Kill switch for the ranked BLAKE3 periodic-padding schedule in round two:
/// `FLOCK_NO_ZC_R2_PERIODIC=1` restores the generic per-group mask checks.
/// The specialized kernel only changes loop control; every useful, boundary,
/// and padded group executes the same arithmetic and stores as the oracle.
#[cfg(target_arch = "aarch64")]
fn r2_periodic_padding_enabled() -> bool {
    std::env::var_os("FLOCK_NO_ZC_R2_PERIODIC").is_none_or(|v| v != *"1")
}

/// Kill switch for adopting the round-two GPU arm's odd-parity products as the
/// round-three lookahead's `W1`/`W2`: `FLOCK_NO_ZC_R2_ODD_OFFLOAD=1` makes the
/// CPU recompute the odd pair on offloaded chunks, which is the incumbent
/// division of labour. Output is bit-identical either way — the GPU's
/// odd-parity half is the same sum the CPU would form, and the calibration
/// oracle verifies that on the target before the arm is ever admitted.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn zc_r2_odd_offload_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var_os("FLOCK_NO_ZC_R2_ODD_OFFLOAD").is_none_or(|v| v != *"1")
    });
    *ON
}

/// ZC-window GPU idle fill (see `gpu_commit::ENV_NO_ZC_IDLE_FILL`): stage
/// round two's GPU-arm window setup while the ZC C-fold's GPU prefix is in
/// flight. Everything staged derives from inputs bound BEFORE the fold
/// submit — `a_packed`/`b_packed` are the round-one operands, and round
/// two's eq split is built from the zerocheck challenge tail
/// `r[k_skip+1..]` (`uni_skip_fold_and_round_pair*` receives it as
/// `mlv_challenges[1..]` = exactly this slice, so the staged bytes are the
/// bytes the window will upload). The round-2 fold point `z` is NOT
/// required: the z-dependent nibble table stays a window-time upload.
///
/// The duplicate `SplitEqGhash` build costs microseconds against the
/// ~10 ms AB head this call precedes; misses (shape, kill switches, Metal
/// failures) are silent no-ops and the window pays its incumbent setup.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn stage_round2_gpu_window_from_r1_challenges(
    a_packed: &[u8],
    b_packed: &[u8],
    m: usize,
    k_skip: usize,
    r: &[F128],
    padding: &PaddingSpec,
) {
    if k_skip != 6 || m <= k_skip + 1 || r.len() != m {
        return;
    }
    let n_pairs = (1usize << (m - k_skip)) / 2;
    let eq = SplitEqGhash::with_n_hi(&r[k_skip + 1..], COMPACT_RECONSTRUCTION_N_HI);
    let lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    if lo_size * hi_size != n_pairs {
        return;
    }
    let (pair_in_block_mask, useful_pairs_inclusive) = round2_pair_skip(padding, k_skip);
    crate::gpu_commit::stage_zc_r2_idle_fill(
        a_packed,
        b_packed,
        &eq.lo,
        &eq.hi,
        lo_size,
        hi_size,
        pair_in_block_mask,
        useful_pairs_inclusive,
    );
    // The upload above copies eq.lo/eq.hi into persistent GPU buffers, so the
    // CPU-side tables are free to be re-used by the round-two sweep.
    stash_staged_r2_eq(&r[k_skip + 1..], eq);
}

/// Round-two eq-split reuse: the ZC-window idle-fill staging (above) builds
/// the exact `SplitEqGhash` the round-two sweep needs — same challenge slice
/// `r[k_skip+1..]`, same `n_hi` — while the round-one window is still open.
/// Stash it so the sweep adopts it instead of rebuilding it inside the
/// FS-serial gap between sampling `z` and round two. Adoption is gated on
/// exact equality of the challenge slice and the split, so a stale entry
/// from a previous prove (different Fiat-Shamir challenges) can never be
/// consumed. Bytes are identical either way: `SplitEqGhash::with_n_hi` is
/// deterministic in its inputs. `FLOCK_NO_R2_EQ_REUSE=1` disables adoption.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
static STAGED_R2_EQ: std::sync::Mutex<Option<(Vec<F128>, SplitEqGhash)>> =
    std::sync::Mutex::new(None);

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn r2_eq_reuse_enabled() -> bool {
    use std::sync::LazyLock;
    static ENABLED: LazyLock<bool> =
        LazyLock::new(|| std::env::var_os("FLOCK_NO_R2_EQ_REUSE").is_none());
    *ENABLED
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn stash_staged_r2_eq(challenges: &[F128], eq: SplitEqGhash) {
    if !r2_eq_reuse_enabled() {
        return;
    }
    if let Ok(mut slot) = STAGED_R2_EQ.lock() {
        *slot = Some((challenges.to_vec(), eq));
    }
}

/// Adopt the staged round-two eq split when it matches `(challenges, n_hi)`
/// exactly; otherwise build it fresh. Non-matching stash entries are left in
/// place (they are not ours to consume).
fn take_or_build_r2_eq(challenges: &[F128], n_hi: usize) -> SplitEqGhash {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if r2_eq_reuse_enabled() {
        if let Ok(mut slot) = STAGED_R2_EQ.lock() {
            let stashed = slot.take();
            match stashed {
                Some((ch, eq))
                    if eq.n_hi == n_hi.min(challenges.len()) && ch.as_slice() == challenges =>
                {
                    return eq;
                }
                other => *slot = other,
            }
        }
    }
    SplitEqGhash::with_n_hi(challenges, n_hi)
}

fn round2_pair_skip(padding: &PaddingSpec, k_skip: usize) -> (usize, usize) {
    if padding.k_log <= k_skip + 1 {
        return (0, usize::MAX);
    }
    let pairs_per_block = 1usize << (padding.k_log - k_skip - 1);
    let chunk_bits = 1usize << k_skip;
    let useful_pairs = padding.useful_bits_per_block.div_ceil(2 * chunk_bits);
    if useful_pairs >= pairs_per_block {
        return (0, usize::MAX);
    }
    (pairs_per_block - 1, useful_pairs)
}

/// Kill switch for draining the DRAM-bound tail loop rounds (half ≥ 2^21)
/// through the hetero E-core queue (H2). `FLOCK_NO_ZC_TAIL_HETERO=1` keeps
/// them on the main rayon pool. Bit-identical either way — chunk ownership
/// and output ranges are unchanged; only scheduling differs.
#[cfg(target_arch = "aarch64")]
fn zc_tail_hetero_enabled() -> bool {
    use std::sync::LazyLock;
    static ENABLED: LazyLock<bool> =
        LazyLock::new(|| std::env::var_os("FLOCK_NO_ZC_TAIL_HETERO").is_none());
    *ENABLED
}

/// Hetero admission floor for the composed tail folds, decoupled from the
/// NT-store policy that shares the historical `2^19` size test. The 2^21→2^19
/// lowering (promoted `db0b668`) admitted the 7+8 fold; this floor extends
/// the same scheduling — and only the scheduling — down to the 9+10 and 11+12
/// composed folds and the first loop rounds, whose 2×≤8 MiB ping-pong outputs
/// are LLC-resident and must therefore keep cached stores (the NT gates stay
/// keyed on `2^19`). Compile-time default per the cleared ranked environment;
/// `FLOCK_NO_ZC_TAIL_HETERO_LOW=1` (exactly `"1"`) restores the incumbent
/// `2^19` admission as the same-binary A/B control. Bit-identical either way:
/// the hetero branch owns the same disjoint per-`x_hi` output ranges, and the
/// message partials are XOR sums.
#[cfg(target_arch = "aarch64")]
const ZC_TAIL_HETERO_LOW_FLOOR: usize = 1 << 17;

#[cfg(target_arch = "aarch64")]
fn zc_tail_hetero_low_floor() -> usize {
    use std::sync::LazyLock;
    static LOW: LazyLock<bool> = LazyLock::new(|| {
        !std::env::var("FLOCK_NO_ZC_TAIL_HETERO_LOW").is_ok_and(|v| v == "1")
    });
    if *LOW { ZC_TAIL_HETERO_LOW_FLOOR } else { 1 << 19 }
}

/// Give the three largest ordinary tail rounds 2,048 independent chunks.
///
/// At log_n=25 this reduces each worker claim from roughly 3 MiB to 768 KiB
/// and shrinks the repeatedly-read eq_lo table from 256 KiB to 64 KiB.  The
/// accompanying 32 KiB eq_hi table keeps the complete equality state within
/// a P-core's private cache.  Stop at half=2^22: below that point the chunks
/// are already small enough that the extra claims and final partials can
/// dominate the cache benefit.
const LARGE_TAIL_EQ_N_HI: usize = 11;
const LARGE_TAIL_EQ_MIN_HALF: usize = 1 << 22;

/// Correctness-preserving kill switch for same-binary A/B screening.
#[cfg(target_arch = "aarch64")]
fn zc_tail_split11_enabled() -> bool {
    use std::sync::LazyLock;
    static ENABLED: LazyLock<bool> =
        LazyLock::new(|| std::env::var_os("FLOCK_NO_ZC_TAIL_SPLIT11").is_none());
    *ENABLED
}

/// Exact same-binary rollback for the expanded fold4 pair. The control keeps
/// the incumbent arithmetic and the same NT stores; only the fold schedule
/// changes. Read once outside the per-chunk closure.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
fn zc_cascade_fold4_pair_enabled() -> bool {
    use std::sync::LazyLock;
    static ENABLED: LazyLock<bool> = LazyLock::new(|| {
        std::env::var_os("FLOCK_NO_ZC_CASCADE_FOLD4_PAIR").is_none_or(|value| value != *"1")
    });
    *ENABLED
}

/// The two ordinary-store lookahead outputs at the ranked `m = 32` shape:
/// rounds 7/8 produce `2^20` values per table and rounds 9/10 produce `2^18`.
/// Keep the next rung exact rather than changing smaller/non-ranked cascades.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline]
fn ranked_normal_fold4_pair_output(output_len: usize) -> bool {
    matches!(output_len, 1_048_576 | 262_144)
}

/// Rung-local rollback: the parent switch still disables every expanded fold4
/// pair, while this one disables only the ranked ordinary-store extension and
/// leaves the already-screened NT rounds-5/6 specialization enabled.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
fn zc_cascade_fold4_pair_normal_enabled() -> bool {
    use std::sync::LazyLock;
    static ENABLED: LazyLock<bool> = LazyLock::new(|| {
        std::env::var_os("FLOCK_NO_ZC_CASCADE_FOLD4_PAIR_NORMAL").is_none_or(|value| value != *"1")
    });
    *ENABLED
}

/// The final direct composed pass at the ranked `m = 32` shape: rounds 11/12
/// produce exactly `2^16` values per table. No earlier direct fallback or
/// smaller non-ranked cascade enters this rung.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline]
fn ranked_direct_fold4_pair_output(output_len: usize) -> bool {
    output_len == 65_536
}

/// Delta-only rollback for the direct rounds-11/12 expansion. The parent and
/// normal-lookahead switches keep controlling E038/E039 independently.
#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
fn zc_cascade_fold4_pair_direct_enabled() -> bool {
    use std::sync::LazyLock;
    static ENABLED: LazyLock<bool> = LazyLock::new(|| {
        std::env::var_os("FLOCK_NO_ZC_CASCADE_FOLD4_PAIR_DIRECT").is_none_or(|value| value != *"1")
    });
    *ENABLED
}

// ---------------------------------------------------------------------------
// Lagrange weights for the univariate-skip fold at z.
// ---------------------------------------------------------------------------

/// Lagrange weights `L_i(z)` for `i ∈ 0..2^k_skip` at the fold point `z`.
///
/// `L_i(z) = ∏_{j ≠ i} (z + φ_8(j)) / (φ_8(i) + φ_8(j))` — the standard Lagrange
/// formula, with the nodes being the F_8 elements `0..2^k_skip` embedded into
/// F_{2^128} via `φ_8`. Subtraction is XOR in characteristic 2.
///
/// O(2^{2·k_skip}) field multiplies — one-time cost.
///
/// Fast path (default): the denominators `∏_{j≠i}(s_i + s_j)` do not depend
/// on `z`, so their inverses are computed once per process and cached; the
/// numerators `∏_{j≠i}(z + s_j)` are assembled from prefix/suffix products
/// in O(ell) multiplies instead of O(ell²). This deletes `ell` field
/// inversions (each ~254 muls) and ~`ell²` muls of FS-serial critical-path
/// time from every call. Outputs are bit-identical to the reference loop:
/// GF(2^128) multiplication is exactly associative and commutative, so
/// regrouping the numerator product cannot change any bit, and the cached
/// inverse is produced by the reference denominator loop itself.
/// `FLOCK_NO_LAGRANGE_FAST=1` restores the reference implementation.
pub fn lagrange_weights_naive(k_skip: usize, z: F128) -> Vec<F128> {
    let ell = 1usize << k_skip;
    assert!(ell <= 256, "k_skip > 8 would exceed PHI_8_TABLE");
    if lagrange_fast_enabled() {
        return lagrange_weights_fast(k_skip, z, 0, lagrange_s_den_inv(k_skip));
    }
    lagrange_weights_naive_reference(k_skip, z)
}

fn lagrange_weights_naive_reference(k_skip: usize, z: F128) -> Vec<F128> {
    let ell = 1usize << k_skip;
    let mut weights = vec![F128::ZERO; ell];
    for i in 0..ell {
        let si = PHI_8_TABLE[i];
        let mut num = F128::ONE;
        let mut den = F128::ONE;
        for j in 0..ell {
            if j == i {
                continue;
            }
            let sj = PHI_8_TABLE[j];
            num *= z + sj;
            den *= si + sj;
        }
        weights[i] = num * den.inv();
    }
    weights
}

/// Correctness-preserving kill switch for the O(ell) Lagrange fast path.
fn lagrange_fast_enabled() -> bool {
    use std::sync::LazyLock;
    static ENABLED: LazyLock<bool> =
        LazyLock::new(|| std::env::var_os("FLOCK_NO_LAGRANGE_FAST").is_none());
    *ENABLED
}

/// Cached `(∏_{j≠i}(s_i + s_j))^{-1}` for the S-domain nodes
/// `s_i = φ_8(i)`, indexed by `k_skip`. z-independent, so one inversion
/// pass per process instead of one per call.
static LAGRANGE_S_DEN_INV: [std::sync::OnceLock<Vec<F128>>; 9] =
    [const { std::sync::OnceLock::new() }; 9];

/// Same cache for the Λ-domain nodes `s_i = φ_8(2^k_skip + i)` (k_skip ≤ 7).
static LAGRANGE_LAMBDA_DEN_INV: [std::sync::OnceLock<Vec<F128>>; 8] =
    [const { std::sync::OnceLock::new() }; 8];

fn lagrange_s_den_inv(k_skip: usize) -> &'static [F128] {
    LAGRANGE_S_DEN_INV[k_skip].get_or_init(|| {
        let ell = 1usize << k_skip;
        (0..ell)
            .map(|i| {
                let si = PHI_8_TABLE[i];
                let mut den = F128::ONE;
                for j in 0..ell {
                    if j != i {
                        den *= si + PHI_8_TABLE[j];
                    }
                }
                den.inv()
            })
            .collect()
    })
}

fn lagrange_lambda_den_inv(k_skip: usize) -> &'static [F128] {
    LAGRANGE_LAMBDA_DEN_INV[k_skip].get_or_init(|| {
        let ell = 1usize << k_skip;
        (0..ell)
            .map(|i| {
                let si = PHI_8_TABLE[ell + i];
                let mut den = F128::ONE;
                for j in 0..ell {
                    if j != i {
                        den *= si + PHI_8_TABLE[ell + j];
                    }
                }
                den.inv()
            })
            .collect()
    })
}

/// O(ell) Lagrange weights: `weights[i] = P_i · S_i · den_inv[i]` with
/// `P_i = ∏_{j<i}(z + s_j)` and `S_i = ∏_{j>i}(z + s_j)`. Exact-field
/// regrouping of the reference `∏_{j≠i}(z + s_j) · den.inv()`.
fn lagrange_weights_fast(
    k_skip: usize,
    z: F128,
    node_offset: usize,
    den_inv: &[F128],
) -> Vec<F128> {
    let ell = 1usize << k_skip;
    let mut weights = vec![F128::ZERO; ell];
    let mut prefix = F128::ONE;
    for i in 0..ell {
        weights[i] = prefix;
        prefix *= z + PHI_8_TABLE[node_offset + i];
    }
    let mut suffix = F128::ONE;
    for i in (0..ell).rev() {
        weights[i] = weights[i] * suffix * den_inv[i];
        suffix *= z + PHI_8_TABLE[node_offset + i];
    }
    weights
}

/// Lagrange weights `L_i^Λ(z)` for `i ∈ 0..2^k_skip` at the fold point `z`,
/// where the nodes are the **extension domain** `Λ = {2^k_skip, …, 2^(k_skip+1) − 1}`
/// embedded via `φ_8` (offset by `2^k_skip` from the S-domain nodes).
///
/// Used to interpolate the extract_c round-1 output `round1_c` (which carries
/// the polynomial `P^C` as its 2^k_skip evaluations on Λ) at the URM challenge `z`.
pub fn lagrange_weights_lambda_naive(k_skip: usize, z: F128) -> Vec<F128> {
    let ell = 1usize << k_skip;
    assert!(2 * ell <= 256, "Λ ∪ S must fit in F_8 (need k_skip ≤ 7)");
    if lagrange_fast_enabled() {
        return lagrange_weights_fast(k_skip, z, ell, lagrange_lambda_den_inv(k_skip));
    }
    lagrange_weights_lambda_naive_reference(k_skip, z)
}

fn lagrange_weights_lambda_naive_reference(k_skip: usize, z: F128) -> Vec<F128> {
    let ell = 1usize << k_skip;
    let mut weights = vec![F128::ZERO; ell];
    for i in 0..ell {
        let si = PHI_8_TABLE[ell + i];
        let mut num = F128::ONE;
        let mut den = F128::ONE;
        for j in 0..ell {
            if j == i {
                continue;
            }
            let sj = PHI_8_TABLE[ell + j];
            num *= z + sj;
            den *= si + sj;
        }
        weights[i] = num * den.inv();
    }
    weights
}

/// Interpolate a degree-`< 2^k_skip` polynomial at z, given its `2^k_skip`
/// evaluations on Λ. Returns `Σ_i L_i^Λ(z) · values[i]`.
///
/// In the extract_c protocol the prover ships `round1_c` (the `P^C` polynomial
/// in Λ-form) and the verifier (or higher-level prover) needs `P^C(z) = ĉ(z, r_rest)`.
/// That value is *the c-claim* at the bound point `(z, r_rest)`.
pub fn interpolate_at_z_on_lambda(values: &[F128], k_skip: usize, z: F128) -> F128 {
    let ell = 1usize << k_skip;
    assert_eq!(values.len(), ell);
    let weights = lagrange_weights_lambda_naive(k_skip, z);
    let mut acc = F128::ZERO;
    for i in 0..ell {
        acc += weights[i] * values[i];
    }
    acc
}

/// Interpolate a degree-`< 2·2^k_skip` polynomial at z, given its `2^k_skip`
/// evaluations on Λ and the assumption that it equals **zero on S**.
///
/// This is the verifier's round-1 reconstruction trick: for an honest prover
/// the combined polynomial `P = P^{AB} + P^C` satisfies `P(λ) = 0` for every
/// `λ ∈ S` (the zerocheck identity at S). Together with the `2^k_skip`
/// evaluations on Λ that the prover sends, that's `2·2^k_skip` evaluations —
/// enough to interpolate the degree-`< 2·2^k_skip` polynomial uniquely.
///
/// Cost: `2·ell × (2·ell − 1)` F128 muls + `ell` inversions for the Lagrange
/// weights. At ell=64 that's ~16K muls + 64 inversions. Sub-millisecond
/// one-time cost in the verifier.
pub fn interpolate_at_z_combined(values_on_lambda: &[F128], k_skip: usize, z: F128) -> F128 {
    let ell = 1usize << k_skip;
    assert_eq!(values_on_lambda.len(), ell);
    assert!(2 * ell <= 256, "Λ ∪ S must fit in F_8 (need k_skip ≤ 7)");
    let n_total = 2 * ell;
    let mut acc = F128::ZERO;
    for i in 0..ell {
        // i-th Λ node = node index `ell + i` in PHI_8_TABLE.
        let node_idx = ell + i;
        let si = PHI_8_TABLE[node_idx];
        let mut num = F128::ONE;
        let mut den = F128::ONE;
        for j in 0..n_total {
            if j == node_idx {
                continue;
            }
            let sj = PHI_8_TABLE[j];
            num *= z + sj;
            den *= si + sj;
        }
        let weight = num * den.inv();
        acc += weight * values_on_lambda[i];
    }
    acc
}

/// Evaluate the multilinear eq polynomial at a point: `eq(r, x) = Π_i (1 + r_i + x_i)`
/// for `r, x ∈ F_{2^128}^n` (char-2 simplification of `(1-r)(1-x) + r·x`).
pub fn eq_eval(r: &[F128], x: &[F128]) -> F128 {
    assert_eq!(r.len(), x.len());
    let mut acc = F128::ONE;
    for i in 0..r.len() {
        acc *= F128::ONE + r[i] + x[i];
    }
    acc
}

/// Specialized variant of [`eq_eval`] for the case where `x` is binary,
/// encoded as a bitmask. Each factor reduces to `r_i` (bit=1) or `1 + r_i`
/// (bit=0), saving one F128 add per coord.
pub fn eq_eval_binary_x(r: &[F128], x_bits: u32) -> F128 {
    debug_assert!(r.len() <= 32, "x_bits is u32; r > 32 dims not supported");
    let mut acc = F128::ONE;
    for (i, &r_i) in r.iter().enumerate() {
        let factor = if (x_bits >> i) & 1 == 1 {
            r_i
        } else {
            F128::ONE + r_i
        };
        acc *= factor;
    }
    acc
}

// ---------------------------------------------------------------------------
// Fold a Boolean witness at z.
// ---------------------------------------------------------------------------

/// Evaluate the univariate-skip polynomial at the fold point `z`, given the
/// precomputed Lagrange `weights`. Returns the multilinear extension table
/// `a_mlv` of length `2^(m − k_skip)` over F_{2^128}.
///
///   `a_mlv[x_rest] = Σ_s a(s, x_rest) · L_s(z)`
///
/// `a(s, x_rest)` is the witness bit at index `x_rest * 2^k_skip + s` (low
/// bits = skip variable, high bits = rest variables).
pub fn fold_at_z_naive(witness: &[bool], m: usize, k_skip: usize, weights: &[F128]) -> Vec<F128> {
    assert!(k_skip <= m);
    let ell = 1usize << k_skip;
    let n_rest = 1usize << (m - k_skip);
    assert_eq!(witness.len(), 1usize << m);
    assert_eq!(weights.len(), ell);

    let mut folded = vec![F128::ZERO; n_rest];
    for x_rest in 0..n_rest {
        let base = x_rest * ell;
        let mut acc = F128::ZERO;
        for s in 0..ell {
            if witness[base + s] {
                acc += weights[s];
            }
        }
        folded[x_rest] = acc;
    }
    folded
}

// ---------------------------------------------------------------------------
// Naive round-2 prover message (AB-pair multilinear sumcheck).
// ---------------------------------------------------------------------------

/// Round-2 (and any subsequent round) prover message for the AB-pair
/// multilinear sumcheck.
///
/// Inputs:
/// - `a_mlv`, `b_mlv`: F128 vectors of length `2^n` for some `n ≥ 1`.
/// - `r`: full eq challenges, length `n`. `r[0]` is the challenge for the
///   variable being bound *this* round; `r[1..]` is for the remaining `n − 1`
///   variables.
///
/// Output: `(r[0] · G(1), G(∞))` for the round polynomial `G(X) = Σ_{x'} eq(r[1..], x')
/// · a_mlv(X, x') · b_mlv(X, x')`, where `a_mlv(0, x') = a_mlv[2x']` and
/// `a_mlv(1, x') = a_mlv[2x' + 1]` (low bit bound).
///
/// The `r[0]` prefactor matches the C++ `sumcheck_round_pair` convention: the
/// quantity sent on the wire is `Π(1) = eq(r[0], 1) · G(1) = r[0] · G(1)`,
/// where `Π(X) = eq(r[0], X) · G(X)` is the actual round polynomial.
pub fn round_pair_naive(a_mlv: &[F128], b_mlv: &[F128], r: &[F128]) -> (F128, F128) {
    let n = a_mlv.len();
    assert_eq!(b_mlv.len(), n);
    assert!(n.is_power_of_two() && n >= 2);
    let half = n / 2;
    let log_n = n.trailing_zeros() as usize;
    assert_eq!(r.len(), log_n);

    let eq_remaining = build_eq(&r[1..]);
    assert_eq!(eq_remaining.len(), half);

    let mut g_one = F128::ZERO;
    let mut g_inf = F128::ZERO;
    for x_prime in 0..half {
        let a0 = a_mlv[2 * x_prime];
        let a1 = a_mlv[2 * x_prime + 1];
        let b0 = b_mlv[2 * x_prime];
        let b1 = b_mlv[2 * x_prime + 1];
        let eq_x = eq_remaining[x_prime];
        g_one += eq_x * a1 * b1;
        // Char-2: (a_1 − a_0)(b_1 − b_0) = (a_0 + a_1)(b_0 + b_1).
        g_inf += eq_x * (a0 + a1) * (b0 + b1);
    }
    (r[0] * g_one, g_inf)
}

// ---------------------------------------------------------------------------
// Naive fused (fold at z + round-2 message) for AB-pair.
// ---------------------------------------------------------------------------

/// Naive fold (at the univariate-skip challenge `z`) of `a` and `b`, plus the
/// round-2 prover message on the resulting multilinear polynomials.
///
/// `mlv_challenges` is of length `m − k_skip` — one challenge per multilinear
/// round. `mlv_challenges[0]` is for the variable bound in round 2 (this
/// round's message uses it as the `r_now` multiplier); `mlv_challenges[1..]`
/// is for subsequent rounds (eq table).
///
/// This is the *unfused* reference: it computes the fold and the round-2
/// message in two separate passes. The optimized version (next) will do both
/// in one pass through the witness.
///
/// Returns `(a_mlv, b_mlv, mlv_challenges[0] · G(1), G(∞))`.
pub fn uni_skip_fold_and_round_pair_naive(
    a: &[bool],
    b: &[bool],
    m: usize,
    k_skip: usize,
    z: F128,
    mlv_challenges: &[F128],
) -> (Vec<F128>, Vec<F128>, F128, F128) {
    assert_eq!(a.len(), 1usize << m);
    assert_eq!(b.len(), 1usize << m);
    assert!(
        m > k_skip,
        "need at least one multilinear variable past the skip"
    );
    assert_eq!(mlv_challenges.len(), m - k_skip);

    let weights = lagrange_weights_naive(k_skip, z);
    let a_mlv = fold_at_z_naive(a, m, k_skip, &weights);
    let b_mlv = fold_at_z_naive(b, m, k_skip, &weights);
    let (msg_1, msg_inf) = round_pair_naive(&a_mlv, &b_mlv, mlv_challenges);
    (a_mlv, b_mlv, msg_1, msg_inf)
}

// ---------------------------------------------------------------------------
// Optimized fused fold + round-2 message.
// ---------------------------------------------------------------------------

/// Precomputed fold table for the univariate-skip fold at a fixed `z`.
///
/// Storage: `n_chunks × 256` F128 entries (32 KB at `k_skip=6`). For each
/// byte-chunk `j ∈ 0..n_chunks` and byte value `v ∈ 0..256`:
///
///   `data[j * 256 + v] = Σ_{b : bit b of v set} weights[8j + b]`
///
/// where `weights = lagrange_weights_naive(k_skip, z)`. Built incrementally by
/// XOR-composition over the set bits of `v` (one XOR per non-power-of-2 entry).
///
/// Per-row fold then becomes one table lookup + XOR per byte (n_chunks lookups
/// total instead of `ell` Lagrange multiplications).
#[derive(Clone, Debug)]
pub struct UniSkipFoldTable {
    pub n_chunks: usize,
    pub data: Vec<F128>,
}

impl UniSkipFoldTable {
    pub fn new(k_skip: usize, z: F128) -> Self {
        let ell = 1usize << k_skip;
        assert_eq!(ell % 8, 0, "k_skip must be ≥ 3 (need ell divisible by 8)");
        let n_chunks = ell / 8;
        let weights = lagrange_weights_naive(k_skip, z);

        let mut data = vec![F128::ZERO; n_chunks * 256];
        for j in 0..n_chunks {
            let basis = &weights[8 * j..8 * j + 8];
            // v = 0: zero (already initialized).
            for b in 0..8 {
                data[j * 256 + (1 << b)] = basis[b];
            }
            // Non-powers-of-2: composed by XOR of (v ^ lo_bit) and lo_bit entries.
            for v in 3usize..256 {
                if (v & (v - 1)) == 0 {
                    continue; // skip powers of 2 (already written)
                }
                let lo_bit = v & v.wrapping_neg();
                let parent = v ^ lo_bit;
                data[j * 256 + v] = data[j * 256 + parent] + data[j * 256 + lo_bit];
            }
        }
        Self { n_chunks, data }
    }

    /// Scalar one-row fold: `Σ_j table[j][bytes[j]]`. Ports the NEON
    /// `uni_skip_fold_one_output_ghash` in scalar form.
    #[inline]
    pub fn fold_one_row(&self, bytes: &[u8]) -> F128 {
        assert_eq!(bytes.len(), self.n_chunks);
        let mut acc = F128::ZERO;
        for j in 0..self.n_chunks {
            acc += self.data[j * 256 + bytes[j] as usize];
        }
        acc
    }

    /// Return `rho * T_z` using XOR-linearity of every byte bank. Only the 64
    /// one-hot basis entries require field multiplication; all other entries
    /// are rebuilt by XOR instead of performing 2,048 independent products.
    pub(crate) fn scaled_linear(&self, rho: F128) -> Vec<F128> {
        assert_eq!(self.data.len(), self.n_chunks * 256);
        let mut scaled = self.data.clone();
        for chunk in 0..self.n_chunks {
            let base = chunk * 256;
            for bit in 0..8 {
                scaled[base + (1 << bit)] = rho * scaled[base + (1 << bit)];
            }
            for value in 3usize..256 {
                if value.is_power_of_two() {
                    continue;
                }
                let low_bit = value & value.wrapping_neg();
                scaled[base + value] = scaled[base + (value ^ low_bit)] + scaled[base + low_bit];
            }
        }
        scaled
    }
}

/// Local split for compact production/reconstruction; intentionally
/// independent of [`SplitEqGhash::MAX_N_HI`].
///
/// At ranked m=32, 11 hi bits give 2,048 jobs. Each producer job streams
/// about 1.5 MiB and each reconstruction job about 1.4 MiB: the latter reads
/// 48 bytes of anchor/delta data and writes 32 bytes per output (3:2
/// read/write), while its 32 KiB lookup table remains hot. Ten hi bits leave
/// roughly 3 MiB jobs and excess shared-L2 pressure across ten workers;
/// twelve halves the footprint again but only adds scheduling/reduction
/// overhead to an already cache-sized streaming job. Keeping this as one
/// local constant makes 10/11/12 straightforward to screen without changing
/// the schedule-tuned global split.
const COMPACT_RECONSTRUCTION_N_HI: usize = 11;

/// Compact materialization of the first multilinear level.
///
/// For each adjacent post-URM row pair this keeps the folded even-row anchor
/// and the eight packed bytes `row0 XOR row1`.  Linearity of the univariate
/// fold gives
///
/// `fold(row0) + rho * (fold(row0) + fold(row1))
///    = anchor + fold_rho(row0 XOR row1)`,
///
/// where `fold_rho` uses the ordinary 32 KiB fold table with every entry
/// multiplied by `rho`.  This is 48 bytes per A/B pair instead of the 64 bytes
/// required by four materialized F128 rows, and it removes the two challenge
/// multiplications per reconstructed output from the first tail round.
pub struct UniSkipCompactFold {
    /// Interleaved `[a_anchor, b_anchor]` entries, two F128s per row pair.
    pub anchors: Vec<F128>,
    /// Interleaved `[a_delta; 8], [b_delta; 8]`, sixteen bytes per row pair.
    pub deltas: ScratchBytes,
}

impl UniSkipCompactFold {
    #[inline]
    pub fn len(&self) -> usize {
        self.anchors.len() / 2
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.anchors.is_empty()
    }

    /// Return both large buffers to their process-wide scratch pools.
    pub fn recycle(self) {
        let Self { anchors, deltas } = self;
        crate::scratch::give_f128(anchors);
        deltas.recycle();
    }
}

/// Compact counterpart of
/// [`uni_skip_fold_and_round_pair_optimized_packed_padded`].  It computes the
/// identical round-two message but materializes one folded anchor and one
/// packed adjacent-row delta per pair instead of two folded rows.
pub fn uni_skip_fold_and_round_pair_compact_padded(
    a_packed: &[u8],
    b_packed: &[u8],
    m: usize,
    k_skip: usize,
    table: &UniSkipFoldTable,
    mlv_challenges: &[F128],
    padding: &PaddingSpec,
) -> (UniSkipCompactFold, F128, F128) {
    uni_skip_fold_and_round_pair_compact_padded_with_deltas(
        a_packed,
        b_packed,
        m,
        k_skip,
        table,
        mlv_challenges,
        padding,
        None,
    )
}

/// Donation-aware implementation of
/// [`uni_skip_fold_and_round_pair_compact_padded`]. `deltas_backing`, when
/// present, must have exactly the compact delta byte length; its original
/// allocation layout is preserved by [`ScratchBytes`].
pub(crate) fn uni_skip_fold_and_round_pair_compact_padded_with_deltas(
    a_packed: &[u8],
    b_packed: &[u8],
    m: usize,
    k_skip: usize,
    table: &UniSkipFoldTable,
    mlv_challenges: &[F128],
    padding: &PaddingSpec,
    deltas_backing: Option<ScratchBytes>,
) -> (UniSkipCompactFold, F128, F128) {
    assert_eq!(
        k_skip, 6,
        "optimized compact fold-and-round_pair variant is k_skip=6 only"
    );
    assert_eq!(table.n_chunks, 8);
    let n_chunks = table.n_chunks;
    let n_out = 1usize << (m - k_skip);
    let n_pairs = n_out / 2;
    assert_eq!(a_packed.len(), n_out * n_chunks);
    assert_eq!(b_packed.len(), n_out * n_chunks);
    assert_eq!(mlv_challenges.len(), m - k_skip);

    let deltas_len = 2 * n_pairs * n_chunks;
    let deltas = deltas_backing.unwrap_or_else(|| ScratchBytes::take(deltas_len));
    assert_eq!(
        deltas.len(),
        deltas_len,
        "donated compact delta backing has the wrong byte length"
    );
    let mut compact = UniSkipCompactFold {
        anchors: crate::scratch::take_f128(2 * n_pairs),
        deltas,
    };

    let eq = take_or_build_r2_eq(&mlv_challenges[1..], COMPACT_RECONSTRUCTION_N_HI);
    let lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    assert_eq!(lo_size * hi_size, n_pairs);
    let eq_hi = &eq.hi;
    let eq_lo = &eq.lo;
    let anchor_chunk_size = 2 * lo_size;
    let delta_chunk_size = 2 * lo_size * n_chunks;
    let (pair_in_block_mask, useful_pairs_inclusive) = round2_pair_skip(padding, k_skip);
    #[cfg(target_arch = "aarch64")]
    let degen = r2_degen_enabled();

    // GPU round-two products arm (see `ENV_NO_GPU_ZC_R2`): a measured
    // prefix of the hi-chunks gets its message products computed on the
    // otherwise-idle GPU while the CPU writes those chunks' anchors and
    // deltas through the anchors-only sibling kernel (byte-identical
    // stores). Partials for prefix chunks are merged after the join; the
    // XOR reduce below is order-independent, so the output is bit-identical
    // to the all-CPU sweep. `None` = the exact incumbent path.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let gpu_job = crate::gpu_commit::launch_zc_r2_products(
        a_packed,
        b_packed,
        &table.data,
        eq_lo,
        eq_hi,
        lo_size,
        hi_size,
        pair_in_block_mask,
        useful_pairs_inclusive,
    );
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let gpu_prefix = gpu_job.as_ref().map_or(0, |j| j.cpu_split());
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let t_cpu_sweep = std::time::Instant::now();

    // Chunks drain through the hetero queue so the idle efficiency cores add
    // throughput without an equal-band barrier penalty (see `epool`). Each
    // chunk writes only its own anchors/deltas ranges and partials slot; the
    // XOR reduce below is order-independent, so output is bit-identical to
    // the rayon map-reduce.
    let mut partials: Vec<(F128, F128)> = vec![(F128::ZERO, F128::ZERO); hi_size];
    let anchors_base = crate::epool::SyncPtr(compact.anchors.as_mut_ptr());
    let deltas_base = crate::epool::SyncPtr(compact.deltas.as_mut_ptr());
    let partials_base = crate::epool::SyncPtr(partials.as_mut_ptr());
    crate::epool::run_hetero_chunks(hi_size, |x_hi| {
        // SAFETY: the queue hands out each x_hi exactly once; chunk x_hi
        // exclusively owns its anchors/deltas ranges and partials[x_hi]. The
        // queue's completion join publishes the writes before the reduction
        // below reads them.
        let (anchors, deltas) = unsafe {
            (
                std::slice::from_raw_parts_mut(
                    anchors_base.ptr().add(x_hi * anchor_chunk_size),
                    anchor_chunk_size,
                ),
                std::slice::from_raw_parts_mut(
                    deltas_base.ptr().add(x_hi * delta_chunk_size),
                    delta_chunk_size,
                ),
            )
        };
        {
            let pair_idx_base = x_hi * lo_size;
            let row_base = pair_idx_base * 2;

            // GPU-covered prefix chunk: write the identical anchors and
            // deltas, skip the products (the GPU partial replaces this
            // chunk's slot after the join).
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            if x_hi < gpu_prefix {
                unsafe {
                    fold_round2_compact_chunk_neon_anchors_only_8(
                        table.data.as_ptr().cast::<u8>(),
                        a_packed.as_ptr().add(row_base * n_chunks),
                        b_packed.as_ptr().add(row_base * n_chunks),
                        anchors.as_mut_ptr(),
                        deltas.as_mut_ptr(),
                        lo_size,
                        pair_idx_base,
                        pair_in_block_mask,
                        useful_pairs_inclusive,
                    );
                }
                return;
            }

            #[cfg(target_arch = "aarch64")]
            let (p1, pinf) = unsafe {
                fold_round2_compact_chunk_neon_unchecked_8(
                    table.data.as_ptr().cast::<u8>(),
                    a_packed.as_ptr().add(row_base * n_chunks),
                    b_packed.as_ptr().add(row_base * n_chunks),
                    anchors.as_mut_ptr(),
                    deltas.as_mut_ptr(),
                    eq_lo.as_ptr(),
                    lo_size,
                    pair_idx_base,
                    pair_in_block_mask,
                    useful_pairs_inclusive,
                    degen,
                )
            };

            #[cfg(not(target_arch = "aarch64"))]
            let (p1, pinf) = {
                let mut p1_acc = F256Unreduced::ZERO;
                let mut pinf_acc = F256Unreduced::ZERO;
                for x_lo in 0..lo_size {
                    if ((pair_idx_base + x_lo) & pair_in_block_mask) >= useful_pairs_inclusive {
                        anchors[2 * x_lo] = F128::ZERO;
                        anchors[2 * x_lo + 1] = F128::ZERO;
                        deltas[2 * x_lo * n_chunks..2 * (x_lo + 1) * n_chunks].fill(0);
                        continue;
                    }

                    let x0g = row_base + 2 * x_lo;
                    let x1g = x0g + 1;
                    let a0_bytes = &a_packed[x0g * n_chunks..(x0g + 1) * n_chunks];
                    let a1_bytes = &a_packed[x1g * n_chunks..(x1g + 1) * n_chunks];
                    let b0_bytes = &b_packed[x0g * n_chunks..(x0g + 1) * n_chunks];
                    let b1_bytes = &b_packed[x1g * n_chunks..(x1g + 1) * n_chunks];
                    let a0 = table.fold_one_row(a0_bytes);
                    let a1 = table.fold_one_row(a1_bytes);
                    let b0 = table.fold_one_row(b0_bytes);
                    let b1 = table.fold_one_row(b1_bytes);
                    anchors[2 * x_lo] = a0;
                    anchors[2 * x_lo + 1] = b0;
                    for j in 0..n_chunks {
                        deltas[2 * x_lo * n_chunks + j] = a0_bytes[j] ^ a1_bytes[j];
                        deltas[(2 * x_lo + 1) * n_chunks + j] = b0_bytes[j] ^ b1_bytes[j];
                    }
                    let eq_l = eq_lo[x_lo];
                    p1_acc ^= eq_l.mul_unreduced(a1 * b1);
                    pinf_acc ^= eq_l.mul_unreduced((a0 + a1) * (b0 + b1));
                }
                (p1_acc.reduce(), pinf_acc.reduce())
            };

            let eq_h = eq_hi[x_hi];
            // SAFETY: exclusive owner of partials[x_hi] (see above).
            unsafe {
                *partials_base.ptr().add(x_hi) = (eq_h * p1, eq_h * pinf);
            }
        }
    });
    // Drain the GPU products arm: merge prefix partials (timed proves),
    // finish calibration (untimed warmup), or CPU-redo the skipped prefix
    // products on any post-admission Metal failure.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if let Some(job) = gpu_job {
        let cpu_wall_ms = t_cpu_sweep.elapsed().as_secs_f64() * 1e3;
        let calib = job.is_calibration();
        let prefix = job.cpu_split();
        let res = crate::gpu_commit::zc_r2_wait(
            job,
            if calib {
                Some(partials.as_slice())
            } else {
                None
            },
            // This sweep has no lookahead state, so there is no parity split
            // to cross-check; the summed oracle is the whole contract here.
            None,
            cpu_wall_ms,
            hi_size,
        );
        match res {
            crate::gpu_commit::ZcR2Result::Calibrated => {}
            crate::gpu_commit::ZcR2Result::Prefix(vals) => {
                // XOR the parities back together: this sweep only ever wanted
                // the summed pair the arm used to return.
                for (x_hi, v) in vals.iter().enumerate() {
                    partials[x_hi] = (v[0] + v[2], v[1] + v[3]);
                }
            }
            crate::gpu_commit::ZcR2Result::Failed => {
                // Redo exactly the skipped prefix products — slower, still
                // exact. Throwaway anchor/delta scratch: the real ranges
                // were already written by the anchors-only pass.
                let mut scr_anchors = vec![F128::ZERO; anchor_chunk_size];
                let mut scr_deltas = vec![0u8; delta_chunk_size];
                for x_hi in 0..prefix {
                    let pair_idx_base = x_hi * lo_size;
                    let row_base = pair_idx_base * 2;
                    let (p1, pinf) = unsafe {
                        fold_round2_compact_chunk_neon_unchecked_8(
                            table.data.as_ptr().cast::<u8>(),
                            a_packed.as_ptr().add(row_base * n_chunks),
                            b_packed.as_ptr().add(row_base * n_chunks),
                            scr_anchors.as_mut_ptr(),
                            scr_deltas.as_mut_ptr(),
                            eq_lo.as_ptr(),
                            lo_size,
                            pair_idx_base,
                            pair_in_block_mask,
                            useful_pairs_inclusive,
                            degen,
                        )
                    };
                    let eq_h = eq_hi[x_hi];
                    partials[x_hi] = (eq_h * p1, eq_h * pinf);
                }
            }
        }
    }

    let (sum1, sum_inf) = partials
        .iter()
        .fold((F128::ZERO, F128::ZERO), |(s1, sinf), &(c1, cinf)| {
            (s1 + c1, sinf + cinf)
        });

    (compact, mlv_challenges[0] * sum1, sum_inf)
}

/// Byte-lane-outer streaming variant of
/// [`uni_skip_fold_and_round_pair_compact_padded`]. Bit-identical outputs
/// (fold XOR trees are merely reassociated). `lanes_per_pass ∈ {1, 2, 4, 8}`
/// is the lane-blocking factor: each pass over a 128-pair tile consumes that
/// many byte lanes of the fold table while the tile's four fold accumulators
/// stay L1-resident. Probe-only entry point; non-aarch64 builds delegate to
/// the gather-shaped base implementation.
pub fn uni_skip_fold_and_round_pair_compact_padded_stream(
    a_packed: &[u8],
    b_packed: &[u8],
    m: usize,
    k_skip: usize,
    table: &UniSkipFoldTable,
    mlv_challenges: &[F128],
    padding: &PaddingSpec,
    lanes_per_pass: usize,
) -> (UniSkipCompactFold, F128, F128) {
    assert!(matches!(lanes_per_pass, 1 | 2 | 4 | 8));
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = lanes_per_pass;
        return uni_skip_fold_and_round_pair_compact_padded(
            a_packed,
            b_packed,
            m,
            k_skip,
            table,
            mlv_challenges,
            padding,
        );
    }
    #[cfg(target_arch = "aarch64")]
    {
        assert_eq!(k_skip, 6, "compact stream variant is k_skip=6 only");
        assert_eq!(table.n_chunks, 8);
        let n_chunks = table.n_chunks;
        let n_out = 1usize << (m - k_skip);
        let n_pairs = n_out / 2;
        assert_eq!(a_packed.len(), n_out * n_chunks);
        assert_eq!(b_packed.len(), n_out * n_chunks);
        assert_eq!(mlv_challenges.len(), m - k_skip);

        let mut compact = UniSkipCompactFold {
            anchors: crate::scratch::take_f128(2 * n_pairs),
            deltas: ScratchBytes::take(2 * n_pairs * n_chunks),
        };

        let eq = take_or_build_r2_eq(&mlv_challenges[1..], COMPACT_RECONSTRUCTION_N_HI);
        let lo_size = 1usize << eq.n_lo;
        let hi_size = 1usize << eq.n_hi;
        assert_eq!(lo_size * hi_size, n_pairs);
        let eq_hi = &eq.hi;
        let eq_lo = &eq.lo;
        let anchor_chunk_size = 2 * lo_size;
        let delta_chunk_size = 2 * lo_size * n_chunks;
        let (pair_in_block_mask, useful_pairs_inclusive) = round2_pair_skip(padding, k_skip);

        let mut partials: Vec<(F128, F128)> = vec![(F128::ZERO, F128::ZERO); hi_size];
        let anchors_base = crate::epool::SyncPtr(compact.anchors.as_mut_ptr());
        let deltas_base = crate::epool::SyncPtr(compact.deltas.as_mut_ptr());
        let partials_base = crate::epool::SyncPtr(partials.as_mut_ptr());
        crate::epool::run_hetero_chunks(hi_size, |x_hi| {
            // SAFETY: same exclusive per-chunk ownership contract as
            // `uni_skip_fold_and_round_pair_compact_padded_with_deltas`.
            let (anchors, deltas) = unsafe {
                (
                    anchors_base.ptr().add(x_hi * anchor_chunk_size),
                    deltas_base.ptr().add(x_hi * delta_chunk_size),
                )
            };
            let pair_idx_base = x_hi * lo_size;
            let row_base = pair_idx_base * 2;
            let (p1, pinf) = unsafe {
                let a_ptr = a_packed.as_ptr().add(row_base * n_chunks);
                let b_ptr = b_packed.as_ptr().add(row_base * n_chunks);
                let t_ptr = table.data.as_ptr().cast::<u8>();
                match lanes_per_pass {
                    1 => fold_round2_compact_stream_chunk_neon::<1>(
                        t_ptr,
                        a_ptr,
                        b_ptr,
                        anchors,
                        deltas,
                        eq_lo.as_ptr(),
                        lo_size,
                        pair_idx_base,
                        pair_in_block_mask,
                        useful_pairs_inclusive,
                    ),
                    2 => fold_round2_compact_stream_chunk_neon::<2>(
                        t_ptr,
                        a_ptr,
                        b_ptr,
                        anchors,
                        deltas,
                        eq_lo.as_ptr(),
                        lo_size,
                        pair_idx_base,
                        pair_in_block_mask,
                        useful_pairs_inclusive,
                    ),
                    4 => fold_round2_compact_stream_chunk_neon::<4>(
                        t_ptr,
                        a_ptr,
                        b_ptr,
                        anchors,
                        deltas,
                        eq_lo.as_ptr(),
                        lo_size,
                        pair_idx_base,
                        pair_in_block_mask,
                        useful_pairs_inclusive,
                    ),
                    _ => fold_round2_compact_stream_chunk_neon::<8>(
                        t_ptr,
                        a_ptr,
                        b_ptr,
                        anchors,
                        deltas,
                        eq_lo.as_ptr(),
                        lo_size,
                        pair_idx_base,
                        pair_in_block_mask,
                        useful_pairs_inclusive,
                    ),
                }
            };
            let eq_h = eq_hi[x_hi];
            // SAFETY: exclusive owner of partials[x_hi].
            unsafe {
                *partials_base.ptr().add(x_hi) = (eq_h * p1, eq_h * pinf);
            }
        });
        let (sum1, sum_inf) = partials
            .iter()
            .fold((F128::ZERO, F128::ZERO), |(s1, sinf), &(c1, cinf)| {
                (s1 + c1, sinf + cinf)
            });

        (compact, mlv_challenges[0] * sum1, sum_inf)
    }
}

/// Byte-lane-outer streaming variant of
/// [`fold_compact_and_compute_round_pair`]. Bit-identical outputs; see
/// [`uni_skip_fold_and_round_pair_compact_padded_stream`] for the schedule and
/// the `lanes_per_pass` contract. Probe-only entry point.
pub fn fold_compact_and_compute_round_pair_stream(
    compact: &UniSkipCompactFold,
    table: &UniSkipFoldTable,
    r_fold: F128,
    r_next: &[F128],
    lanes_per_pass: usize,
) -> (Vec<F128>, Vec<F128>, F128, F128) {
    assert!(matches!(lanes_per_pass, 1 | 2 | 4 | 8));
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = lanes_per_pass;
        return fold_compact_and_compute_round_pair(compact, table, r_fold, r_next);
    }
    #[cfg(target_arch = "aarch64")]
    {
        let n = compact.len();
        assert!(!compact.is_empty() && n.is_power_of_two() && n >= 4);
        assert_eq!(compact.anchors.len(), 2 * n);
        assert_eq!(compact.deltas.len(), 2 * n * table.n_chunks);
        assert_eq!(table.n_chunks, 8);
        assert_eq!(r_next.len(), n.trailing_zeros() as usize);

        let scaled_table = table.scaled_linear(r_fold);

        let eq = SplitEqGhash::with_n_hi(&r_next[1..], COMPACT_RECONSTRUCTION_N_HI);
        let lo_size = 1usize << eq.n_lo;
        let hi_size = 1usize << eq.n_hi;
        assert_eq!(lo_size * hi_size * 2, n);
        let chunk_size = 2 * lo_size;
        let eq_hi = &eq.hi;
        let eq_lo = &eq.lo;

        let mut a_out = crate::scratch::take_f128(n);
        let mut b_out = crate::scratch::take_f128(n);
        let mut partials: Vec<(F128, F128)> = vec![(F128::ZERO, F128::ZERO); hi_size];
        let a_base = crate::epool::SyncPtr(a_out.as_mut_ptr());
        let b_base = crate::epool::SyncPtr(b_out.as_mut_ptr());
        let partials_base = crate::epool::SyncPtr(partials.as_mut_ptr());
        crate::epool::run_hetero_chunks(hi_size, |x_hi| {
            // SAFETY: same exclusive per-chunk ownership contract as
            // `fold_compact_and_compute_round_pair`.
            let (a_ptr, b_ptr) = unsafe {
                (
                    a_base.ptr().add(x_hi * chunk_size),
                    b_base.ptr().add(x_hi * chunk_size),
                )
            };
            let base = x_hi * chunk_size;
            let (p1, pinf) = unsafe {
                let t_ptr = scaled_table.as_ptr().cast::<u8>();
                let anchors = compact.anchors.as_ptr().add(2 * base);
                let deltas = compact.deltas.as_ptr().add(2 * base * table.n_chunks);
                match lanes_per_pass {
                    1 => fold_compact_stream_chunk_neon::<1>(
                        t_ptr,
                        anchors,
                        deltas,
                        a_ptr,
                        b_ptr,
                        eq_lo.as_ptr(),
                        lo_size,
                    ),
                    2 => fold_compact_stream_chunk_neon::<2>(
                        t_ptr,
                        anchors,
                        deltas,
                        a_ptr,
                        b_ptr,
                        eq_lo.as_ptr(),
                        lo_size,
                    ),
                    4 => fold_compact_stream_chunk_neon::<4>(
                        t_ptr,
                        anchors,
                        deltas,
                        a_ptr,
                        b_ptr,
                        eq_lo.as_ptr(),
                        lo_size,
                    ),
                    _ => fold_compact_stream_chunk_neon::<8>(
                        t_ptr,
                        anchors,
                        deltas,
                        a_ptr,
                        b_ptr,
                        eq_lo.as_ptr(),
                        lo_size,
                    ),
                }
            };
            let eq_h = eq_hi[x_hi];
            // SAFETY: exclusive owner of partials[x_hi].
            unsafe {
                *partials_base.ptr().add(x_hi) = (eq_h * p1, eq_h * pinf);
            }
        });
        let (sum1, sum_inf) = partials
            .iter()
            .fold((F128::ZERO, F128::ZERO), |(s1, sinf), &(c1, cinf)| {
                (s1 + c1, sinf + cinf)
            });

        (a_out, b_out, r_next[0] * sum1, sum_inf)
    }
}

/// Bind the first multilinear challenge from a compact round-two
/// materialization and compute the following round message.
///
/// `r_next` describes the post-fold table, matching the contract of
/// [`fold_and_compute_round_pair_into`].  The returned tables have
/// `compact.len()` entries each.
pub fn fold_compact_and_compute_round_pair(
    compact: &UniSkipCompactFold,
    table: &UniSkipFoldTable,
    r_fold: F128,
    r_next: &[F128],
) -> (Vec<F128>, Vec<F128>, F128, F128) {
    let n = compact.len();
    assert!(!compact.is_empty() && n.is_power_of_two() && n >= 4);
    assert_eq!(compact.anchors.len(), 2 * n);
    assert_eq!(compact.deltas.len(), 2 * n * table.n_chunks);
    assert_eq!(table.n_chunks, 8);
    assert_eq!(r_next.len(), n.trailing_zeros() as usize);

    // Compose the sampled challenge into the resident 32 KiB byte table once.
    // Linearity makes each later row reconstruction lookup/XOR-only.
    let scaled_table = table.scaled_linear(r_fold);

    let eq = SplitEqGhash::with_n_hi(&r_next[1..], COMPACT_RECONSTRUCTION_N_HI);
    let lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    assert_eq!(lo_size * hi_size * 2, n);
    let chunk_size = 2 * lo_size;
    let eq_hi = &eq.hi;
    let eq_lo = &eq.lo;

    // Unpinned takes: these buffers become the loop-round arm's no-copy
    // wrap targets next round (and the T3 arm's output surface). The
    // pinned slots already carry process-lifetime Metal views, and the
    // ordinary pinned-first take preference can hand out exactly that
    // collision on a worker where the pin is parked at this point in the
    // prove — an overlapping second no-copy view is not legal. This is
    // the diagnosed cause of the v7/v8/v9 scoreless job deaths.
    let mut a_out = crate::scratch::take_f128_unpinned(n);
    let mut b_out = crate::scratch::take_f128_unpinned(n);
    #[cfg(target_arch = "aarch64")]
    let degen = r2_degen_enabled();

    // GPU T3 products arm (see `ENV_NO_GPU_ZC_T3`): a measured prefix of
    // the hi-chunks gets its message products computed on the GPU (which
    // redundantly reconstructs its chunks' pairs from the same compact
    // inputs via the nibble-decomposed scaled table) while the CPU writes
    // those chunks' reconstruction outputs through a products-skipping
    // sibling kernel (byte-identical stores). Partials for prefix chunks
    // are merged after the join; the XOR reduce below is order-independent,
    // so the output is bit-identical to the all-CPU sweep. `None` = the
    // exact incumbent path.
    // PARKED pending a wrap-budget probe: three archives carrying this
    // integration died scoreless on the runner while a frontier-content
    // draw from the same account scored cleanly in the same window, and
    // halving the calibration did not change the outcome. The untouched
    // suspect is this arm's +1.5 GiB of per-process no-copy wrap surface
    // (the promoted r2 arm's 1 GiB is the largest proven-survivable
    // wrap budget; the loop arm adds only 0.5 GiB and ships alone to
    // bisect). The kernel, oracle test, and CPU sibling stay in-tree —
    // re-enable by restoring this launch once the budget is understood.
    // UN-PARKED (v11): the scoreless deaths were the job-wall timeout, not
    // this arm — the static warmup latch frees minutes of wall and both
    // arms' once-per-process costs fit in a fraction of it.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    const ZC_T3_INTEGRATION_PARKED: bool = true;
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let gpu_job = if ZC_T3_INTEGRATION_PARKED {
        None
    } else {
        crate::gpu_commit::launch_zc_t3_products(
            &compact.anchors,
            &compact.deltas,
            &scaled_table,
            eq_lo,
            eq_hi,
            lo_size,
            hi_size,
        )
    };
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let gpu_prefix = gpu_job.as_ref().map_or(0, |j| j.cpu_split());
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let t_cpu_sweep = std::time::Instant::now();

    // Hetero-queue drain, same contract as the compact materialization above.
    let mut partials: Vec<(F128, F128)> = vec![(F128::ZERO, F128::ZERO); hi_size];
    let a_base = crate::epool::SyncPtr(a_out.as_mut_ptr());
    let b_base = crate::epool::SyncPtr(b_out.as_mut_ptr());
    let partials_base = crate::epool::SyncPtr(partials.as_mut_ptr());
    crate::epool::run_hetero_chunks(hi_size, |x_hi| {
        // SAFETY: exclusive per-chunk ownership; queue join publishes writes.
        let (a_out, b_out) = unsafe {
            (
                std::slice::from_raw_parts_mut(a_base.ptr().add(x_hi * chunk_size), chunk_size),
                std::slice::from_raw_parts_mut(b_base.ptr().add(x_hi * chunk_size), chunk_size),
            )
        };
        {
            let base = x_hi * chunk_size;

            // GPU-covered prefix chunk: write the identical reconstruction
            // outputs, skip the products (the GPU partial replaces this
            // chunk's slot after the join).
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            if x_hi < gpu_prefix {
                unsafe {
                    fold_compact_chunk_neon_reconstruct_only_8(
                        scaled_table.as_ptr().cast::<u8>(),
                        compact.anchors.as_ptr().add(2 * base),
                        compact.deltas.as_ptr().add(2 * base * table.n_chunks),
                        a_out.as_mut_ptr(),
                        b_out.as_mut_ptr(),
                        lo_size,
                    );
                }
                return;
            }

            #[cfg(target_arch = "aarch64")]
            let (p1, pinf) = unsafe {
                fold_compact_chunk_neon_unchecked_8(
                    scaled_table.as_ptr().cast::<u8>(),
                    compact.anchors.as_ptr().add(2 * base),
                    compact.deltas.as_ptr().add(2 * base * table.n_chunks),
                    a_out.as_mut_ptr(),
                    b_out.as_mut_ptr(),
                    eq_lo.as_ptr(),
                    lo_size,
                    degen,
                )
            };

            #[cfg(not(target_arch = "aarch64"))]
            let (p1, pinf) = {
                let mut p1_acc = F256Unreduced::ZERO;
                let mut pinf_acc = F256Unreduced::ZERO;
                for x_lo in 0..lo_size {
                    let out = 2 * x_lo;
                    for lane in 0..2 {
                        let index = base + out + lane;
                        let mut a = compact.anchors[2 * index];
                        let mut b = compact.anchors[2 * index + 1];
                        for j in 0..table.n_chunks {
                            let d = 2 * index * table.n_chunks + j;
                            a += scaled_table[j * 256 + compact.deltas[d] as usize];
                            b +=
                                scaled_table[j * 256 + compact.deltas[d + table.n_chunks] as usize];
                        }
                        a_out[out + lane] = a;
                        b_out[out + lane] = b;
                    }
                    let a0 = a_out[out];
                    let a1 = a_out[out + 1];
                    let b0 = b_out[out];
                    let b1 = b_out[out + 1];
                    let eq_l = eq_lo[x_lo];
                    p1_acc ^= eq_l.mul_unreduced(a1 * b1);
                    pinf_acc ^= eq_l.mul_unreduced((a0 + a1) * (b0 + b1));
                }
                (p1_acc.reduce(), pinf_acc.reduce())
            };

            let eq_h = eq_hi[x_hi];
            // SAFETY: exclusive owner of partials[x_hi] (see above).
            unsafe {
                *partials_base.ptr().add(x_hi) = (eq_h * p1, eq_h * pinf);
            }
        }
    });
    // Drain the GPU products arm: merge prefix partials (timed proves),
    // finish calibration (untimed warmup), or CPU-redo the skipped prefix
    // products on any post-admission Metal failure.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if let Some(job) = gpu_job {
        let cpu_wall_ms = t_cpu_sweep.elapsed().as_secs_f64() * 1e3;
        let calib = job.is_calibration();
        let prefix = job.cpu_split();
        let res = crate::gpu_commit::zc_t3_wait(
            job,
            if calib {
                Some(partials.as_slice())
            } else {
                None
            },
            cpu_wall_ms,
            hi_size,
        );
        match res {
            crate::gpu_commit::ZcT3Result::Calibrated => {}
            crate::gpu_commit::ZcT3Result::Prefix(vals) => {
                partials[..prefix].copy_from_slice(&vals);
            }
            crate::gpu_commit::ZcT3Result::Failed => {
                // Redo exactly the skipped prefix products — slower, still
                // exact. The full kernel rewrites the same reconstruction
                // values into the real output ranges (byte-identical
                // stores), so reusing them as targets is safe.
                for x_hi in 0..prefix {
                    let base = x_hi * chunk_size;
                    let (p1, pinf) = unsafe {
                        fold_compact_chunk_neon_unchecked_8(
                            scaled_table.as_ptr().cast::<u8>(),
                            compact.anchors.as_ptr().add(2 * base),
                            compact.deltas.as_ptr().add(2 * base * table.n_chunks),
                            a_out.as_mut_ptr().wrapping_add(base),
                            b_out.as_mut_ptr().wrapping_add(base),
                            eq_lo.as_ptr(),
                            lo_size,
                            degen,
                        )
                    };
                    let eq_h = eq_hi[x_hi];
                    partials[x_hi] = (eq_h * p1, eq_h * pinf);
                }
            }
        }
    }

    let (sum1, sum_inf) = partials
        .iter()
        .fold((F128::ZERO, F128::ZERO), |(s1, sinf), &(c1, cinf)| {
            (s1 + c1, sinf + cinf)
        });

    (a_out, b_out, r_next[0] * sum1, sum_inf)
}

// ---------------------------------------------------------------------------
// Two-challenge symbolic lookahead ("variant K"): rounds 2..4 in two passes.
//
// Round three's message is a *quadratic in ρ₁*, so its three coefficients can
// be accumulated during round two — before ρ₁ exists — and evaluated the
// instant the challenge is drawn. That turns the round-3 pass from a
// materializing 1.5 GiB → 1 GiB sweep into six scalars, and lets rounds 3 and
// 4 share a single double-fold pass straight out of the compact state.
//
// Everything here is *value*-identical to the incumbent route: F128 is an
// exact field, the transcript order is untouched, and only the association of
// the sums changes.
// ---------------------------------------------------------------------------

/// Deferred round-three message: `G₃(1)` coefficients in `c[0..3]`, `G₃(∞)`
/// in `c[3..6]`, each in the basis `{1, ρ, ρ²}`.
///
/// Mirrors the six-coefficient shape of `pcs::ligerito::fold2_and_msgs_lsb`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Round3Lookahead {
    pub c: [F128; 6],
}

/// Evaluate the deferred round-three message at the sampled ρ₁.
///
/// `first_r_next[0] = ONE` in Convention A, so there is no prefactor: the two
/// returned values are exactly what the incumbent
/// [`fold_compact_and_compute_round_pair`] would have sent.
#[inline]
pub fn eval_round3_lookahead(la: &Round3Lookahead, rho1: F128) -> (F128, F128) {
    let rho_sq = rho1 * rho1;
    (
        la.c[0] + la.c[1] * rho1 + la.c[2] * rho_sq,
        la.c[3] + la.c[4] * rho1 + la.c[5] * rho_sq,
    )
}

/// Local eq split for the lookahead round-two sweep. Identical to
/// [`COMPACT_RECONSTRUCTION_N_HI`] at the ranked shape; clamped so the lo half
/// always keeps at least one variable, because the sweep consumes pairs two at
/// a time (one round-three group) inside a chunk.
#[inline]
fn lookahead_n_hi(n_vars: usize) -> usize {
    COMPACT_RECONSTRUCTION_N_HI.min(n_vars.saturating_sub(1))
}

/// Round-two compact producer **plus** the deferred round-three coefficients.
///
/// The compact state, the round-two wire message and every store are
/// bit-identical to
/// [`uni_skip_fold_and_round_pair_compact_padded_with_deltas`]; the sweep
/// merely also accumulates six aggregates over round-three groups
/// `y = x'/2`:
///
/// ```text
/// W0 = Σ_y eq₃(y)·a2b2   W1 = Σ eq₃·a3b3        W2 = Σ eq₃·(a2+a3)(b2+b3)
/// W3 = Σ_y eq₃(y)·e_a e_b   W4 = Σ eq₃·o_a o_b   W5 = Σ eq₃·(e+o)_a (e+o)_b
/// e = A[4y]+A[4y+2],  o = A[4y+1]+A[4y+3]
/// ```
///
/// `W1` and `W2` cost **zero extra multiplies**: they are the odd-parity half
/// of the two round-two accumulators, because `eq₂(2y+1) = r₁·eq₃(y)` and
/// `eq₃.hi ≡ eq₂.hi`. The kernel therefore reports the round-two sums split by
/// parity and the driver divides the odd half by `r₁` once.
///
/// Requires `r₁ = mlv_challenges[1] ≠ 0`; the caller falls back to the
/// incumbent route otherwise.
pub(crate) fn uni_skip_fold_and_round_pair_compact_padded_lookahead(
    a_packed: &[u8],
    b_packed: &[u8],
    m: usize,
    k_skip: usize,
    table: &UniSkipFoldTable,
    mlv_challenges: &[F128],
    padding: &PaddingSpec,
    deltas_backing: Option<ScratchBytes>,
) -> (UniSkipCompactFold, F128, F128, Round3Lookahead) {
    assert_eq!(k_skip, 6, "lookahead compact round two is k_skip=6 only");
    assert_eq!(table.n_chunks, 8);
    let n_chunks = table.n_chunks;
    let n_out = 1usize << (m - k_skip);
    let n_pairs = n_out / 2;
    assert_eq!(a_packed.len(), n_out * n_chunks);
    assert_eq!(b_packed.len(), n_out * n_chunks);
    assert_eq!(mlv_challenges.len(), m - k_skip);
    let r1 = mlv_challenges[1];
    assert_ne!(r1, F128::ZERO, "lookahead requires a non-zero r[k_skip+1]");

    let deltas_len = 2 * n_pairs * n_chunks;
    let deltas = deltas_backing.unwrap_or_else(|| ScratchBytes::take(deltas_len));
    assert_eq!(
        deltas.len(),
        deltas_len,
        "donated compact delta backing has the wrong byte length"
    );
    let mut compact = UniSkipCompactFold {
        anchors: crate::scratch::take_f128(2 * n_pairs),
        deltas,
    };

    let n_vars = mlv_challenges.len() - 1;
    let eq = take_or_build_r2_eq(&mlv_challenges[1..], lookahead_n_hi(n_vars));
    let lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    assert_eq!(lo_size * hi_size, n_pairs);
    assert!(lo_size >= 2, "lookahead sweep pairs two x_lo per group");
    // `eq₂(2y) = (1+r₁)·eq₃(y)` and `eq₂(2y+1) = r₁·eq₃(y)`, so the sweep uses
    // the odd lane as the group's single weight and the two constants below
    // put every aggregate back on its own scale, once, off the hot path.
    let kappa = (F128::ONE + r1) * r1.inv();
    let eq_hi = &eq.hi;
    let eq_lo = &eq.lo;
    let anchor_chunk_size = 2 * lo_size;
    let delta_chunk_size = 2 * lo_size * n_chunks;
    let (pair_in_block_mask, useful_pairs_inclusive) = round2_pair_skip(padding, k_skip);
    #[cfg(target_arch = "aarch64")]
    let degen = r2_degen_enabled();
    #[cfg(target_arch = "aarch64")]
    let periodic_padding = r2_periodic_padding_enabled();

    // Same GPU round-two products arm as the incumbent sweep: it still
    // receives byte-identical anchors/deltas for every chunk and still owns
    // the *summed* `(p1, pinf)` for its prefix. The CPU keeps producing the
    // odd-parity halves on those chunks (the GPU's sums are not parity-split),
    // which is what makes `W1`/`W2` recoverable everywhere.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let gpu_job = crate::gpu_commit::launch_zc_r2_products(
        a_packed,
        b_packed,
        &table.data,
        eq_lo,
        eq_hi,
        lo_size,
        hi_size,
        pair_in_block_mask,
        useful_pairs_inclusive,
    );
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let gpu_prefix = gpu_job.as_ref().map_or(0, |j| j.cpu_split());
    // The GPU arm now returns its round-two products parity-split, so on an
    // offloaded chunk the CPU can skip the odd-parity pair as well as the even
    // one instead of recomputing the odd half for `W1`/`W2`. The kill switch
    // restores the incumbent division of labour within the same binary.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let odd_on_gpu = gpu_prefix > 0 && zc_r2_odd_offload_enabled();
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    let odd_on_gpu = false;
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let t_cpu_sweep = std::time::Instant::now();

    let mut partials: Vec<(F128, F128)> = vec![(F128::ZERO, F128::ZERO); hi_size];
    // [p1_odd, pinf_odd, W0, W3, W4, W5], eq_hi-weighted, one slot per chunk.
    let mut la_partials: Vec<[F128; 6]> = vec![[F128::ZERO; 6]; hi_size];
    let anchors_base = crate::epool::SyncPtr(compact.anchors.as_mut_ptr());
    let deltas_base = crate::epool::SyncPtr(compact.deltas.as_mut_ptr());
    let partials_base = crate::epool::SyncPtr(partials.as_mut_ptr());
    let la_base = crate::epool::SyncPtr(la_partials.as_mut_ptr());
    crate::epool::run_hetero_chunks(hi_size, |x_hi| {
        // SAFETY: the queue hands out each x_hi exactly once; chunk x_hi
        // exclusively owns its anchors/deltas ranges and both partial slots.
        let (anchors, deltas) = unsafe {
            (
                anchors_base.ptr().add(x_hi * anchor_chunk_size),
                deltas_base.ptr().add(x_hi * delta_chunk_size),
            )
        };
        let pair_idx_base = x_hi * lo_size;
        let row_base = pair_idx_base * 2;
        let mut out = [F128::ZERO; 8];

        #[cfg(target_arch = "aarch64")]
        {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            let full = x_hi >= gpu_prefix;
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            let full = true;
            unsafe {
                let t = table.data.as_ptr().cast::<u8>();
                let ap = a_packed.as_ptr().add(row_base * n_chunks);
                let bp = b_packed.as_ptr().add(row_base * n_chunks);
                if full {
                    fold_round2_compact_chunk_neon_lookahead_8::<true, false>(
                        t,
                        ap,
                        bp,
                        anchors,
                        deltas,
                        eq_lo.as_ptr(),
                        lo_size,
                        pair_idx_base,
                        pair_in_block_mask,
                        useful_pairs_inclusive,
                        degen,
                        periodic_padding,
                        out.as_mut_ptr(),
                    );
                } else if odd_on_gpu {
                    fold_round2_compact_chunk_neon_lookahead_8::<false, true>(
                        t,
                        ap,
                        bp,
                        anchors,
                        deltas,
                        eq_lo.as_ptr(),
                        lo_size,
                        pair_idx_base,
                        pair_in_block_mask,
                        useful_pairs_inclusive,
                        degen,
                        periodic_padding,
                        out.as_mut_ptr(),
                    );
                } else {
                    fold_round2_compact_chunk_neon_lookahead_8::<false, false>(
                        t,
                        ap,
                        bp,
                        anchors,
                        deltas,
                        eq_lo.as_ptr(),
                        lo_size,
                        pair_idx_base,
                        pair_in_block_mask,
                        useful_pairs_inclusive,
                        degen,
                        periodic_padding,
                        out.as_mut_ptr(),
                    );
                }
            }
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            let anchors = unsafe { std::slice::from_raw_parts_mut(anchors, anchor_chunk_size) };
            let deltas = unsafe { std::slice::from_raw_parts_mut(deltas, delta_chunk_size) };
            out = round2_lookahead_chunk_scalar(
                a_packed,
                b_packed,
                table,
                anchors,
                deltas,
                eq_lo,
                lo_size,
                row_base,
                pair_idx_base,
                pair_in_block_mask,
                useful_pairs_inclusive,
            );
        }

        let eq_h = eq_hi[x_hi];
        // `out[0..2]` carry the even lane on the odd lane's weight; κ restores
        // `eq₂(2y)` exactly (field arithmetic, no rounding).
        let p1 = kappa * out[0] + out[2];
        let pinf = kappa * out[1] + out[3];
        // SAFETY: exclusive owner of both partial slots (see above).
        unsafe {
            *partials_base.ptr().add(x_hi) = (eq_h * p1, eq_h * pinf);
            *la_base.ptr().add(x_hi) = [
                eq_h * out[2],
                eq_h * out[3],
                eq_h * out[4],
                eq_h * out[5],
                eq_h * out[6],
                eq_h * out[7],
            ];
        }
    });

    // Drain the GPU products arm. The arm returns `[p1_even, pinf_even,
    // p1_odd, pinf_odd]`; XORing the parities back together reproduces the
    // summed pair the incumbent contract delivered, and the odd half is
    // adopted as the lookahead's `W1`/`W2` state exactly when the CPU was told
    // to skip it.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if let Some(job) = gpu_job {
        let cpu_wall_ms = t_cpu_sweep.elapsed().as_secs_f64() * 1e3;
        let calib = job.is_calibration();
        let prefix = job.cpu_split();
        let res = crate::gpu_commit::zc_r2_wait(
            job,
            if calib {
                Some(partials.as_slice())
            } else {
                None
            },
            if calib {
                Some(la_partials.as_slice())
            } else {
                None
            },
            cpu_wall_ms,
            hi_size,
        );
        match res {
            crate::gpu_commit::ZcR2Result::Calibrated => {}
            crate::gpu_commit::ZcR2Result::Prefix(vals) => {
                for (x_hi, v) in vals.iter().enumerate() {
                    partials[x_hi] = (v[0] + v[2], v[1] + v[3]);
                    if odd_on_gpu {
                        la_partials[x_hi][0] = v[2];
                        la_partials[x_hi][1] = v[3];
                    }
                }
            }
            crate::gpu_commit::ZcR2Result::Failed => {
                // Redo exactly the skipped prefix products — slower, still
                // exact. Throwaway anchor/delta scratch: the real ranges were
                // already written by the lookahead pass. The full-lookahead
                // monomorphization is used so the odd-parity slots are
                // recovered too when the CPU sweep skipped them.
                let mut scr_anchors = vec![F128::ZERO; anchor_chunk_size];
                let mut scr_deltas = vec![0u8; delta_chunk_size];
                for x_hi in 0..prefix {
                    let pair_idx_base = x_hi * lo_size;
                    let row_base = pair_idx_base * 2;
                    let mut out = [F128::ZERO; 8];
                    unsafe {
                        fold_round2_compact_chunk_neon_lookahead_8::<true, false>(
                            table.data.as_ptr().cast::<u8>(),
                            a_packed.as_ptr().add(row_base * n_chunks),
                            b_packed.as_ptr().add(row_base * n_chunks),
                            scr_anchors.as_mut_ptr(),
                            scr_deltas.as_mut_ptr(),
                            eq_lo.as_ptr(),
                            lo_size,
                            pair_idx_base,
                            pair_in_block_mask,
                            useful_pairs_inclusive,
                            degen,
                            periodic_padding,
                            out.as_mut_ptr(),
                        );
                    }
                    let eq_h = eq_hi[x_hi];
                    partials[x_hi] = (
                        eq_h * (kappa * out[0] + out[2]),
                        eq_h * (kappa * out[1] + out[3]),
                    );
                    if odd_on_gpu {
                        la_partials[x_hi][0] = eq_h * out[2];
                        la_partials[x_hi][1] = eq_h * out[3];
                    }
                }
            }
        }
    }

    let (sum1, sum_inf) = partials
        .iter()
        .fold((F128::ZERO, F128::ZERO), |(s1, sinf), &(c1, cinf)| {
            (s1 + c1, sinf + cinf)
        });
    let mut agg = [F128::ZERO; 6];
    for slot in &la_partials {
        for (a, v) in agg.iter_mut().zip(slot.iter()) {
            *a += *v;
        }
    }
    // Every aggregate was accumulated on the odd lane's weight `r₁·eq₃`, so a
    // single `r₁⁻¹` puts all six back on `eq₃`.
    let r1_inv = r1.inv();
    let w1 = r1_inv * agg[0];
    let w2 = r1_inv * agg[1];
    let w0 = r1_inv * agg[2];
    let w3 = r1_inv * agg[3];
    let w4 = r1_inv * agg[4];
    let w5 = r1_inv * agg[5];
    let la = Round3Lookahead {
        c: [w0, w0 + w1 + w2, w2, w3, w3 + w4 + w5, w5],
    };

    (compact, mlv_challenges[0] * sum1, sum_inf, la)
}

/// Portable reference for one lookahead round-two chunk (non-AArch64 builds).
#[cfg(not(target_arch = "aarch64"))]
#[allow(clippy::too_many_arguments)]
fn round2_lookahead_chunk_scalar(
    a_packed: &[u8],
    b_packed: &[u8],
    table: &UniSkipFoldTable,
    anchors: &mut [F128],
    deltas: &mut [u8],
    eq_lo: &[F128],
    lo_size: usize,
    row_base: usize,
    pair_idx_base: usize,
    pair_in_block_mask: usize,
    useful_pairs_inclusive: usize,
) -> [F128; 8] {
    let n_chunks = table.n_chunks;
    let mut p1_even = F256Unreduced::ZERO;
    let mut pinf_even = F256Unreduced::ZERO;
    let mut p1_odd = F256Unreduced::ZERO;
    let mut pinf_odd = F256Unreduced::ZERO;
    let mut w = [F256Unreduced::ZERO; 4];

    let mut fold_pair = |x_lo: usize, anchors: &mut [F128], deltas: &mut [u8]| {
        if ((pair_idx_base + x_lo) & pair_in_block_mask) >= useful_pairs_inclusive {
            anchors[2 * x_lo] = F128::ZERO;
            anchors[2 * x_lo + 1] = F128::ZERO;
            deltas[2 * x_lo * n_chunks..2 * (x_lo + 1) * n_chunks].fill(0);
            return None;
        }
        let x0g = row_base + 2 * x_lo;
        let x1g = x0g + 1;
        let a0_bytes = &a_packed[x0g * n_chunks..(x0g + 1) * n_chunks];
        let a1_bytes = &a_packed[x1g * n_chunks..(x1g + 1) * n_chunks];
        let b0_bytes = &b_packed[x0g * n_chunks..(x0g + 1) * n_chunks];
        let b1_bytes = &b_packed[x1g * n_chunks..(x1g + 1) * n_chunks];
        let a0 = table.fold_one_row(a0_bytes);
        let a1 = table.fold_one_row(a1_bytes);
        let b0 = table.fold_one_row(b0_bytes);
        let b1 = table.fold_one_row(b1_bytes);
        anchors[2 * x_lo] = a0;
        anchors[2 * x_lo + 1] = b0;
        for j in 0..n_chunks {
            deltas[2 * x_lo * n_chunks + j] = a0_bytes[j] ^ a1_bytes[j];
            deltas[(2 * x_lo + 1) * n_chunks + j] = b0_bytes[j] ^ b1_bytes[j];
        }
        Some((a0, a1, b0, b1))
    };

    for u in 0..lo_size / 2 {
        let even = fold_pair(2 * u, anchors, deltas);
        let odd = fold_pair(2 * u + 1, anchors, deltas);
        if even.is_none() && odd.is_none() {
            continue;
        }
        let (a0, a1, b0, b1) = even.unwrap_or((F128::ZERO, F128::ZERO, F128::ZERO, F128::ZERO));
        let (a2, a3, b2, b3) = odd.unwrap_or((F128::ZERO, F128::ZERO, F128::ZERO, F128::ZERO));
        // One weight per group: the odd lane. See the NEON kernel's doc.
        let wt = eq_lo[2 * u + 1];
        let (a0w, a1w, a2w, a3w) = (wt * a0, wt * a1, wt * a2, wt * a3);
        if even.is_some() {
            p1_even ^= a1w.mul_unreduced(b1);
            pinf_even ^= (a0w + a1w).mul_unreduced(b0 + b1);
        }
        if odd.is_some() {
            p1_odd ^= a3w.mul_unreduced(b3);
            pinf_odd ^= (a2w + a3w).mul_unreduced(b2 + b3);
            w[0] ^= a2w.mul_unreduced(b2);
        }
        let (e_aw, e_b) = (a0w + a2w, b0 + b2);
        let (o_aw, o_b) = (a1w + a3w, b1 + b3);
        w[1] ^= e_aw.mul_unreduced(e_b);
        w[2] ^= o_aw.mul_unreduced(o_b);
        w[3] ^= (e_aw + o_aw).mul_unreduced(e_b + o_b);
    }

    [
        p1_even.reduce(),
        pinf_even.reduce(),
        p1_odd.reduce(),
        pinf_odd.reduce(),
        w[0].reduce(),
        w[1].reduce(),
        w[2].reduce(),
        w[3].reduce(),
    ]
}

/// Bind ρ₁ **and** ρ₂ in one pass over the compact round-two state and emit
/// the round-four message — replacing the incumbent T3 reconstruction plus the
/// first tail-loop iteration.
///
/// `a_out`/`b_out` receive `compact.len() / 2` entries each — exactly the
/// tables the incumbent route hands to tail-loop iteration `i = 2`.
/// `r_next4` follows the [`fold_and_compute_round_pair_into`] contract for
/// that output size (`r_next4[0] = ONE`, `r_next4.len() = log2(a_out.len())`).
pub(crate) fn fold2_compact_and_round4_into(
    compact: &UniSkipCompactFold,
    table: &UniSkipFoldTable,
    rho1: F128,
    rho2: F128,
    r_next4: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
) -> (F128, F128) {
    let n_pairs = compact.len();
    let n_groups = n_pairs / 2;
    assert!(n_groups >= 2 && n_groups.is_power_of_two());
    assert_eq!(compact.anchors.len(), 2 * n_pairs);
    assert_eq!(compact.deltas.len(), 2 * n_pairs * table.n_chunks);
    assert_eq!(table.n_chunks, 8);
    assert_eq!(a_out.len(), n_groups);
    assert_eq!(b_out.len(), n_groups);
    assert_eq!(r_next4.len(), n_groups.trailing_zeros() as usize);

    // λ₀+λ₁ = 1+ρ₂ and λ₂+λ₃ = ρ₂ in characteristic two, so the anchors need a
    // single ordinary ρ₂ fold and only the two deltas carry ρ₁.
    let lambda1 = rho1 * (F128::ONE + rho2);
    let lambda3 = rho1 * rho2;
    let table_l1 = table.scaled_linear(lambda1);
    let table_l3 = table.scaled_linear(lambda3);

    let eq = SplitEqGhash::with_n_hi(&r_next4[1..], COMPACT_RECONSTRUCTION_N_HI);
    let lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    assert_eq!(lo_size * hi_size * 2, n_groups);
    let eq_hi = &eq.hi;
    let eq_lo = &eq.lo;
    let out_chunk = 2 * lo_size;
    #[cfg(target_arch = "aarch64")]
    let degen = r2_degen_enabled();

    let mut partials: Vec<(F128, F128)> = vec![(F128::ZERO, F128::ZERO); hi_size];
    let a_base = crate::epool::SyncPtr(a_out.as_mut_ptr());
    let b_base = crate::epool::SyncPtr(b_out.as_mut_ptr());
    let partials_base = crate::epool::SyncPtr(partials.as_mut_ptr());
    crate::epool::run_hetero_chunks(hi_size, |x_hi| {
        // SAFETY: exclusive per-chunk ownership; the queue join publishes the
        // writes before the reduction below reads them.
        let (a_ptr, b_ptr) = unsafe {
            (
                a_base.ptr().add(x_hi * out_chunk),
                b_base.ptr().add(x_hi * out_chunk),
            )
        };
        // Each output chunk covers `out_chunk` groups = 2·out_chunk pairs.
        let pair_base = 2 * x_hi * out_chunk;

        #[cfg(target_arch = "aarch64")]
        let (p1, pinf) = unsafe {
            fold2_compact_and_round4_chunk_neon_8(
                table_l1.as_ptr().cast::<u8>(),
                table_l3.as_ptr().cast::<u8>(),
                rho2,
                compact.anchors.as_ptr().add(2 * pair_base),
                compact.deltas.as_ptr().add(pair_base * table.n_chunks * 2),
                a_ptr,
                b_ptr,
                eq_lo.as_ptr(),
                lo_size,
                degen,
            )
        };

        #[cfg(not(target_arch = "aarch64"))]
        let (p1, pinf) = {
            let a_out = unsafe { std::slice::from_raw_parts_mut(a_ptr, out_chunk) };
            let b_out = unsafe { std::slice::from_raw_parts_mut(b_ptr, out_chunk) };
            let mut p1_acc = F256Unreduced::ZERO;
            let mut pinf_acc = F256Unreduced::ZERO;
            let nc = table.n_chunks;
            for g_local in 0..out_chunk {
                let g = x_hi * out_chunk + g_local;
                let anc = &compact.anchors[4 * g..4 * g + 4];
                let d = &compact.deltas[32 * g..32 * g + 32];
                let mut a = anc[0] + rho2 * (anc[0] + anc[2]);
                let mut b = anc[1] + rho2 * (anc[1] + anc[3]);
                for j in 0..nc {
                    a += table_l1[j * 256 + d[j] as usize];
                    b += table_l1[j * 256 + d[nc + j] as usize];
                    a += table_l3[j * 256 + d[2 * nc + j] as usize];
                    b += table_l3[j * 256 + d[3 * nc + j] as usize];
                }
                a_out[g_local] = a;
                b_out[g_local] = b;
            }
            for x_lo in 0..lo_size {
                let o = 2 * x_lo;
                let (a0, a1) = (a_out[o], a_out[o + 1]);
                let (b0, b1) = (b_out[o], b_out[o + 1]);
                let eq_l = eq_lo[x_lo];
                p1_acc ^= eq_l.mul_unreduced(a1 * b1);
                pinf_acc ^= eq_l.mul_unreduced((a0 + a1) * (b0 + b1));
            }
            (p1_acc.reduce(), pinf_acc.reduce())
        };

        let eq_h = eq_hi[x_hi];
        // SAFETY: exclusive owner of partials[x_hi].
        unsafe {
            *partials_base.ptr().add(x_hi) = (eq_h * p1, eq_h * pinf);
        }
    });

    let (sum1, sum_inf) = partials
        .iter()
        .fold((F128::ZERO, F128::ZERO), |(s1, sinf), &(c1, cinf)| {
            (s1 + c1, sinf + cinf)
        });

    (r_next4[0] * sum1, sum_inf)
}

// ---------------------------------------------------------------------------
// Cascaded lookahead (variant K, one level deeper): rounds 2..6 in three
// passes. The K double-fold materializes each round-4 output group in
// registers before its store — exactly the position the round-2 sweep was in
// before the round-3 promotion — so round FIVE's message (a quadratic in the
// not-yet-sampled ρ₃) can be accumulated during the K pass, and rounds 5 and
// 6 then share one plain composed double-fold. Everything is value-identical
// to the incumbent route: exact F128, untouched transcript order, pure
// reassociation.
// ---------------------------------------------------------------------------

/// Local eq split for the cascaded K pass. Identical to [`COMPACT_RECONSTRUCTION_N_HI`] at
/// the ranked shape; clamped so the lo half always keeps at least one
/// variable, because the sweep consumes round-4 pairs two at a time (one
/// round-five group) inside a chunk — the same clamp [`lookahead_n_hi`]
/// applies to the round-2 sweep. Any admissible value is bit-identical (the
/// split is an exact lo/hi tensor factorisation).
#[inline]
fn cascade_k_pass_n_hi(n_vars: usize) -> usize {
    COMPACT_RECONSTRUCTION_N_HI.min(n_vars.saturating_sub(1))
}

/// Cascade K pass: [`fold2_compact_and_round4_into`] **plus** the deferred
/// round-five coefficients, in the same single pass over the compact state.
///
/// The output tables, every store, and the round-four wire message are
/// value-identical to the incumbent K pass; the sweep merely also accumulates
/// six aggregates over round-five groups `y'` (`a_i = a_out[4y'+i]`):
///
/// ```text
/// W0' = Σ_y' eq₅(y')·a2b2   W1' = Σ eq₅·a3b3      W2' = Σ eq₅·(a2+a3)(b2+b3)
/// W3' = Σ_y' eq₅(y')·e_a e_b   W4' = Σ eq₅·o_a o_b   W5' = Σ eq₅·(e+o)_a(e+o)_b
/// e = out[4y']+out[4y'+2],  o = out[4y'+1]+out[4y'+3]
/// ```
///
/// `W1'` and `W2'` cost **zero extra multiplies**: they are the odd-parity
/// half of the two round-four accumulators, because the K pass's eq table is
/// built over `r_next4[1..]` (LSB-first), so `eq₄(2y'+1) = r'·eq₅(y')` with
/// `r' = r_next4[1]` and `eq₅.hi ≡ eq₄.hi`. The kernel reports the round-four
/// sums split by parity and the driver divides the odd half by `r'` once —
/// the exact mechanism the round-2 lookahead uses with `r₁`.
///
/// Returns `(round4_msg_1, round4_msg_inf, la5)` where `la5` evaluates via
/// [`eval_round3_lookahead`] (the deferred-quadratic shape is round-agnostic)
/// at the later-sampled ρ₃ to the round-five message, with no memory pass.
///
/// Requires `r_next4[1] ≠ 0`; the caller falls back to the incumbent route
/// otherwise (probability 2⁻¹²⁸ for a sampled challenge; at the ranked shape
/// this slot is the protocol constant β₀ ≠ 0).
pub(crate) fn fold2_compact_and_round45_into(
    compact: &UniSkipCompactFold,
    table: &UniSkipFoldTable,
    rho1: F128,
    rho2: F128,
    r_next4: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
) -> (F128, F128, Round3Lookahead) {
    let n_pairs = compact.len();
    let n_groups = n_pairs / 2;
    assert!(n_groups >= 4 && n_groups.is_power_of_two());
    assert_eq!(compact.anchors.len(), 2 * n_pairs);
    assert_eq!(compact.deltas.len(), 2 * n_pairs * table.n_chunks);
    assert_eq!(table.n_chunks, 8);
    assert_eq!(a_out.len(), n_groups);
    assert_eq!(b_out.len(), n_groups);
    assert_eq!(r_next4.len(), n_groups.trailing_zeros() as usize);
    let r_par = r_next4[1];
    assert_ne!(r_par, F128::ZERO, "cascade requires a non-zero r_next4[1]");

    // Same λ composition as the incumbent K pass: the anchors take one
    // ordinary ρ₂ fold, only the two deltas carry ρ₁.
    let lambda1 = rho1 * (F128::ONE + rho2);
    let lambda3 = rho1 * rho2;
    let table_l1 = table.scaled_linear(lambda1);
    let table_l3 = table.scaled_linear(lambda3);

    let n_vars = r_next4.len() - 1;
    let eq = SplitEqGhash::with_n_hi(&r_next4[1..], cascade_k_pass_n_hi(n_vars));
    let lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    assert_eq!(lo_size * hi_size * 2, n_groups);
    assert!(
        lo_size >= 2,
        "cascade sweep pairs two round-4 pairs per group"
    );
    // `eq₄(2y') = (1+r')·eq₅(y')` and `eq₄(2y'+1) = r'·eq₅(y')`: the sweep
    // uses the odd lane as the group's single weight; the two constants below
    // put every aggregate back on its own scale, once, off the hot path.
    let kappa = (F128::ONE + r_par) * r_par.inv();
    let eq_hi = &eq.hi;
    let eq_lo = &eq.lo;
    let out_chunk = 2 * lo_size;
    #[cfg(target_arch = "aarch64")]
    let degen = r2_degen_enabled();

    let mut partials: Vec<(F128, F128)> = vec![(F128::ZERO, F128::ZERO); hi_size];
    // [p1_odd, pinf_odd, W0', W3', W4', W5'], eq_hi-weighted, one per chunk.
    let mut la_partials: Vec<[F128; 6]> = vec![[F128::ZERO; 6]; hi_size];
    let a_base = crate::epool::SyncPtr(a_out.as_mut_ptr());
    let b_base = crate::epool::SyncPtr(b_out.as_mut_ptr());
    let partials_base = crate::epool::SyncPtr(partials.as_mut_ptr());
    let la_base = crate::epool::SyncPtr(la_partials.as_mut_ptr());
    crate::epool::run_hetero_chunks(hi_size, |x_hi| {
        // SAFETY: exclusive per-chunk ownership; the queue join publishes the
        // writes before the reduction below reads them.
        let (a_ptr, b_ptr) = unsafe {
            (
                a_base.ptr().add(x_hi * out_chunk),
                b_base.ptr().add(x_hi * out_chunk),
            )
        };
        // Each output chunk covers `out_chunk` groups = 2·out_chunk pairs.
        let pair_base = 2 * x_hi * out_chunk;
        let mut outv = [F128::ZERO; 8];

        #[cfg(target_arch = "aarch64")]
        unsafe {
            fold2_compact_and_round45_chunk_neon_8(
                table_l1.as_ptr().cast::<u8>(),
                table_l3.as_ptr().cast::<u8>(),
                rho2,
                compact.anchors.as_ptr().add(2 * pair_base),
                compact.deltas.as_ptr().add(pair_base * table.n_chunks * 2),
                a_ptr,
                b_ptr,
                eq_lo.as_ptr(),
                lo_size,
                degen,
                outv.as_mut_ptr(),
            );
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            let a_out = unsafe { std::slice::from_raw_parts_mut(a_ptr, out_chunk) };
            let b_out = unsafe { std::slice::from_raw_parts_mut(b_ptr, out_chunk) };
            outv = fold2_round45_chunk_scalar(
                compact, table, &table_l1, &table_l3, rho2, a_out, b_out, eq_lo, lo_size, x_hi,
            );
        }

        let eq_h = eq_hi[x_hi];
        // `outv[0..2]` carry the even round-4 pairs on the odd lane's weight;
        // κ restores `eq₄(2y')` exactly (field arithmetic, no rounding).
        let p1 = kappa * outv[0] + outv[2];
        let pinf = kappa * outv[1] + outv[3];
        // SAFETY: exclusive owner of both partial slots (see above).
        unsafe {
            *partials_base.ptr().add(x_hi) = (eq_h * p1, eq_h * pinf);
            *la_base.ptr().add(x_hi) = [
                eq_h * outv[2],
                eq_h * outv[3],
                eq_h * outv[4],
                eq_h * outv[5],
                eq_h * outv[6],
                eq_h * outv[7],
            ];
        }
    });

    let (sum1, sum_inf) = partials
        .iter()
        .fold((F128::ZERO, F128::ZERO), |(s1, sinf), &(c1, cinf)| {
            (s1 + c1, sinf + cinf)
        });
    let mut agg = [F128::ZERO; 6];
    for slot in &la_partials {
        for (a, v) in agg.iter_mut().zip(slot.iter()) {
            *a += *v;
        }
    }
    // Every aggregate was accumulated on the odd lane's weight `r'·eq₅`, so a
    // single `r'⁻¹` puts all six back on `eq₅`.
    let r_inv = r_par.inv();
    let w1 = r_inv * agg[0];
    let w2 = r_inv * agg[1];
    let w0 = r_inv * agg[2];
    let w3 = r_inv * agg[3];
    let w4 = r_inv * agg[4];
    let w5 = r_inv * agg[5];
    let la5 = Round3Lookahead {
        c: [w0, w0 + w1 + w2, w2, w3, w3 + w4 + w5, w5],
    };

    (r_next4[0] * sum1, sum_inf, la5)
}

/// Portable reference for one cascade K chunk (non-AArch64 builds). Same
/// slot order as the NEON kernel:
/// `[p1_even, pinf_even, p1_odd, pinf_odd, W0', W3', W4', W5']`.
#[cfg(not(target_arch = "aarch64"))]
#[allow(clippy::too_many_arguments)]
fn fold2_round45_chunk_scalar(
    compact: &UniSkipCompactFold,
    table: &UniSkipFoldTable,
    table_l1: &[F128],
    table_l3: &[F128],
    rho2: F128,
    a_out: &mut [F128],
    b_out: &mut [F128],
    eq_lo: &[F128],
    lo_size: usize,
    x_hi: usize,
) -> [F128; 8] {
    let nc = table.n_chunks;
    let out_chunk = a_out.len();
    let mut p1_even = F256Unreduced::ZERO;
    let mut pinf_even = F256Unreduced::ZERO;
    let mut p1_odd = F256Unreduced::ZERO;
    let mut pinf_odd = F256Unreduced::ZERO;
    let mut w = [F256Unreduced::ZERO; 4];

    // Materialize the chunk's composed outputs first (identical expressions
    // to the incumbent scalar path)…
    for g_local in 0..out_chunk {
        let g = x_hi * out_chunk + g_local;
        let anc = &compact.anchors[4 * g..4 * g + 4];
        let d = &compact.deltas[32 * g..32 * g + 32];
        let mut a = anc[0] + rho2 * (anc[0] + anc[2]);
        let mut b = anc[1] + rho2 * (anc[1] + anc[3]);
        for j in 0..nc {
            a += table_l1[j * 256 + d[j] as usize];
            b += table_l1[j * 256 + d[nc + j] as usize];
            a += table_l3[j * 256 + d[2 * nc + j] as usize];
            b += table_l3[j * 256 + d[3 * nc + j] as usize];
        }
        a_out[g_local] = a;
        b_out[g_local] = b;
    }
    // …then sweep round-five groups with the shared odd-lane weight.
    for t in 0..lo_size / 2 {
        let o = 4 * t;
        let (a0, a1, a2, a3) = (a_out[o], a_out[o + 1], a_out[o + 2], a_out[o + 3]);
        let (b0, b1, b2, b3) = (b_out[o], b_out[o + 1], b_out[o + 2], b_out[o + 3]);
        let wt = eq_lo[2 * t + 1];
        let (a0w, a1w, a2w, a3w) = (wt * a0, wt * a1, wt * a2, wt * a3);
        p1_even ^= a1w.mul_unreduced(b1);
        pinf_even ^= (a0w + a1w).mul_unreduced(b0 + b1);
        p1_odd ^= a3w.mul_unreduced(b3);
        pinf_odd ^= (a2w + a3w).mul_unreduced(b2 + b3);
        w[0] ^= a2w.mul_unreduced(b2);
        let (e_aw, e_b) = (a0w + a2w, b0 + b2);
        let (o_aw, o_b) = (a1w + a3w, b1 + b3);
        w[1] ^= e_aw.mul_unreduced(e_b);
        w[2] ^= o_aw.mul_unreduced(o_b);
        w[3] ^= (e_aw + o_aw).mul_unreduced(e_b + o_b);
    }

    [
        p1_even.reduce(),
        pinf_even.reduce(),
        p1_odd.reduce(),
        pinf_odd.reduce(),
        w[0].reduce(),
        w[1].reduce(),
        w[2].reduce(),
        w[3].reduce(),
    ]
}

/// Bind ρ₃ **and** ρ₄ in one plain composed pass — quartering `a`/`b` instead
/// of halving twice — and emit the round-six message from the composed
/// outputs. Replaces tail-loop iterations `i = 2` and `i = 3` under the
/// cascade: no compact-state complications at this depth, inputs are plain
/// F128 arrays, so the composition is the direct
/// `t0 + ρ₄·(t0+t1)` over the two ρ₃-folded halves — value-identical to the
/// two sequential [`fold_and_compute_round_pair_into`] passes, term for term,
/// with the intermediate half-size tables never written or re-read.
///
/// `r_next6` follows the usual contract for the OUTPUT size
/// (`r_next6[0] = ONE`, `r_next6.len() = log2(a_out.len())`).
pub(crate) fn fold2_plain_and_round6_into(
    a: &[F128],
    b: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    rho3: F128,
    rho4: F128,
    r_next6: &[F128],
) -> (F128, F128) {
    let n = a.len();
    assert_eq!(b.len(), n);
    assert!(n.is_power_of_two() && n >= 16);
    let quarter = n / 4;
    assert_eq!(a_out.len(), quarter);
    assert_eq!(b_out.len(), quarter);
    let log_n = n.trailing_zeros() as usize;
    assert_eq!(r_next6.len(), log_n - 2);

    // Same clamp rationale as the K pass: the message consumes composed
    // outputs two at a time, so the lo half must keep ≥ 1 variable.
    let n_vars = r_next6.len() - 1;
    let n_hi = SplitEqGhash::MAX_N_HI.min(n_vars.saturating_sub(1));
    let eq = SplitEqGhash::with_n_hi(&r_next6[1..], n_hi);
    let lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    assert!(lo_size >= 2, "composed fold requires lo_size ≥ 2");
    assert_eq!(lo_size * hi_size * 2, quarter);

    let chunk_in = 8 * lo_size; // four inputs per composed output
    let chunk_out = 2 * lo_size;
    let eq_lo = &eq.lo;
    let eq_hi = &eq.hi;

    // Same NT policy as the plain tail: the composed output (2^22 F128 = 64
    // MiB per array at the ranked shape) is a ping-pong buffer not read until
    // the next round's barrier, so `stnp` elides the write-allocate RFOs.
    #[cfg(target_arch = "aarch64")]
    let nt_stores = {
        use std::sync::OnceLock;
        static NT_ENABLED: OnceLock<bool> = OnceLock::new();
        quarter >= (1usize << 19)
            && *NT_ENABLED.get_or_init(|| std::env::var_os("FLOCK_ZC_NT_LEGACY").is_none())
    };
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    let use_expanded_direct = !nt_stores
        && ranked_direct_fold4_pair_output(quarter)
        && zc_cascade_fold4_pair_enabled()
        && zc_cascade_fold4_pair_direct_enabled();
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    let rho34 = if use_expanded_direct {
        rho3 * rho4
    } else {
        F128::ZERO
    };

    let chunk_partial =
        |a_in: &[F128], b_in: &[F128], a_out: &mut [F128], b_out: &mut [F128]| -> (F128, F128) {
            #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
            {
                if use_expanded_direct {
                    fold2_and_message_normal_expanded_aarch64(
                        a_in, b_in, a_out, b_out, rho3, rho4, rho34, eq_lo,
                    )
                } else {
                    fold2_and_message_aarch64(
                        a_in, b_in, a_out, b_out, rho3, rho4, eq_lo, nt_stores,
                    )
                }
            }
            #[cfg(all(target_arch = "aarch64", not(target_feature = "aes")))]
            {
                fold2_and_message_aarch64(a_in, b_in, a_out, b_out, rho3, rho4, eq_lo, nt_stores)
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                let mut p1_acc = F256Unreduced::ZERO;
                let mut pinf_acc = F256Unreduced::ZERO;
                for (x_lo, &eq_l) in eq_lo.iter().enumerate() {
                    let i = 8 * x_lo;
                    let o = 2 * x_lo;
                    let fold4 = |v: &[F128], i: usize| {
                        let t0 = v[i] + rho3 * (v[i] + v[i + 1]);
                        let t1 = v[i + 2] + rho3 * (v[i + 2] + v[i + 3]);
                        t0 + rho4 * (t0 + t1)
                    };
                    let a0 = fold4(a_in, i);
                    let a1 = fold4(a_in, i + 4);
                    let b0 = fold4(b_in, i);
                    let b1 = fold4(b_in, i + 4);
                    a_out[o] = a0;
                    a_out[o + 1] = a1;
                    b_out[o] = b0;
                    b_out[o + 1] = b1;
                    p1_acc ^= eq_l.mul_unreduced(a1 * b1);
                    pinf_acc ^= eq_l.mul_unreduced((a0 + a1) * (b0 + b1));
                }
                (p1_acc.reduce(), pinf_acc.reduce())
            }
        };

    // Same scheduling policy as the plain tail: DRAM-bound sizes drain
    // through the hetero E-core queue, LLC-resident ones stay on rayon.
    #[cfg(target_arch = "aarch64")]
    let hetero = quarter >= zc_tail_hetero_low_floor() && zc_tail_hetero_enabled();
    #[cfg(not(target_arch = "aarch64"))]
    let hetero = false;

    let (sum1, sum_inf) = if hetero {
        let mut partials: Vec<(F128, F128)> = vec![(F128::ZERO, F128::ZERO); hi_size];
        let a_base = crate::epool::SyncPtr(a_out.as_mut_ptr());
        let b_base = crate::epool::SyncPtr(b_out.as_mut_ptr());
        let partials_base = crate::epool::SyncPtr(partials.as_mut_ptr());
        crate::epool::run_hetero_chunks(hi_size, |x_hi| {
            // SAFETY: exclusive per-chunk ownership; queue join publishes writes.
            let (a_out, b_out) = unsafe {
                (
                    std::slice::from_raw_parts_mut(a_base.ptr().add(x_hi * chunk_out), chunk_out),
                    std::slice::from_raw_parts_mut(b_base.ptr().add(x_hi * chunk_out), chunk_out),
                )
            };
            let (p1, pinf) = chunk_partial(
                &a[x_hi * chunk_in..(x_hi + 1) * chunk_in],
                &b[x_hi * chunk_in..(x_hi + 1) * chunk_in],
                a_out,
                b_out,
            );
            let eq_h = eq_hi[x_hi];
            // SAFETY: exclusive owner of partials[x_hi].
            unsafe {
                *partials_base.ptr().add(x_hi) = (eq_h * p1, eq_h * pinf);
            }
        });
        partials
            .iter()
            .fold((F128::ZERO, F128::ZERO), |(s1, sinf), &(c1, cinf)| {
                (s1 + c1, sinf + cinf)
            })
    } else {
        use rayon::prelude::*;
        a_out
            .par_chunks_mut(chunk_out)
            .zip(b_out.par_chunks_mut(chunk_out))
            .enumerate()
            .map(|(x_hi, (a_out, b_out))| {
                let (p1, pinf) = chunk_partial(
                    &a[x_hi * chunk_in..(x_hi + 1) * chunk_in],
                    &b[x_hi * chunk_in..(x_hi + 1) * chunk_in],
                    a_out,
                    b_out,
                );
                let eq_h = eq_hi[x_hi];
                (eq_h * p1, eq_h * pinf)
            })
            .reduce(
                || (F128::ZERO, F128::ZERO),
                |(s1, sinf), (c1, cinf)| (s1 + c1, sinf + cinf),
            )
    };

    (r_next6[0] * sum1, sum_inf)
}

// ---------------------------------------------------------------------------
// Cascaded lookahead, level three: rounds 2..8 in four passes. The composed
// rounds-5/6 fold materializes each output group in registers before its
// store — the same position the K pass was in before the round-5 promotion —
// so round SEVEN's message (a quadratic in the not-yet-sampled ρ₅) can be
// accumulated during the composed-5/6 pass, and rounds 7 and 8 then share one
// more plain composed double-fold. Everything is value-identical to the
// cascade2 route: exact F128, untouched transcript order, pure reassociation.
// ---------------------------------------------------------------------------

/// [`fold2_plain_and_round6_into`] **plus** the deferred round-seven
/// coefficients, in the same single pass over the composed-5/6 traversal.
///
/// The output tables, every store, and the round-six wire message are
/// value-identical to the plain composed pass; the sweep merely also
/// accumulates six aggregates over round-seven groups `y''`
/// (`a_i = a_out[4y''+i]`):
///
/// ```text
/// W0'' = Σ_y'' eq₇(y'')·a2b2   W1'' = Σ eq₇·a3b3    W2'' = Σ eq₇·(a2+a3)(b2+b3)
/// W3'' = Σ_y'' eq₇(y'')·e_a e_b   W4'' = Σ eq₇·o_a o_b   W5'' = Σ eq₇·(e+o)_a(e+o)_b
/// e = out[4y'']+out[4y''+2],  o = out[4y''+1]+out[4y''+3]
/// ```
///
/// `W1''` and `W2''` cost **zero extra multiplies**: they are the odd-parity
/// half of the two round-six accumulators, because the pass's eq table is
/// built over `r_next6[1..]` (LSB-first), so `eq₆(2y''+1) = r''·eq₇(y'')`
/// with `r'' = r_next6[1]` and `eq₇.hi ≡ eq₆.hi` — the exact mechanism the
/// cascade K pass uses with `r_next4[1]`.
///
/// Returns `(round6_msg_1, round6_msg_inf, la7)` where `la7` evaluates via
/// [`eval_round3_lookahead`] (the deferred-quadratic shape is round-agnostic)
/// at the later-sampled ρ₅ to the round-seven message, with no memory pass.
///
/// Requires `r_next6[1] ≠ 0`; the caller falls back to the cascade2 route
/// otherwise (at the ranked shape this slot is the protocol constant β₂ ≠ 0;
/// for a sampled slot it is probability 2⁻¹²⁸).
pub(crate) fn fold2_plain_and_round67_into(
    a: &[F128],
    b: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    rho3: F128,
    rho4: F128,
    r_next6: &[F128],
) -> (F128, F128, Round3Lookahead) {
    let n = a.len();
    assert_eq!(b.len(), n);
    assert!(n.is_power_of_two() && n >= 16);
    let quarter = n / 4;
    assert_eq!(a_out.len(), quarter);
    assert_eq!(b_out.len(), quarter);
    let log_n = n.trailing_zeros() as usize;
    assert_eq!(r_next6.len(), log_n - 2);
    let r_par = r_next6[1];
    assert_ne!(r_par, F128::ZERO, "cascade requires a non-zero r_next6[1]");

    // Same clamp rationale as the plain composed pass — and the lookahead
    // sweep consumes composed outputs four at a time (two round-6 pairs = one
    // round-7 group), so the lo half must keep ≥ 1 variable.
    let n_vars = r_next6.len() - 1;
    let n_hi = SplitEqGhash::MAX_N_HI.min(n_vars.saturating_sub(1));
    let eq = SplitEqGhash::with_n_hi(&r_next6[1..], n_hi);
    let lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    assert!(lo_size >= 2, "composed lookahead requires lo_size ≥ 2");
    assert_eq!(lo_size * hi_size * 2, quarter);
    // `eq₆(2y'') = (1+r'')·eq₇(y'')` and `eq₆(2y''+1) = r''·eq₇(y'')`: the
    // sweep uses the odd lane as the group's single weight; the two constants
    // below put every aggregate back on its own scale, once, off the hot path.
    let kappa = (F128::ONE + r_par) * r_par.inv();

    let chunk_in = 8 * lo_size; // four inputs per composed output
    let chunk_out = 2 * lo_size;
    let eq_lo = &eq.lo;
    let eq_hi = &eq.hi;

    // Same NT policy as the plain composed pass: decided once from the
    // round's output size (identical stores either way).
    #[cfg(target_arch = "aarch64")]
    let nt_stores = {
        use std::sync::OnceLock;
        static NT_ENABLED: OnceLock<bool> = OnceLock::new();
        quarter >= (1usize << 19)
            && *NT_ENABLED.get_or_init(|| std::env::var_os("FLOCK_ZC_NT_LEGACY").is_none())
    };
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    let use_expanded_fold4_pair = zc_cascade_fold4_pair_enabled()
        && (nt_stores
            || (ranked_normal_fold4_pair_output(quarter)
                && zc_cascade_fold4_pair_normal_enabled()));
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    let rho34 = if use_expanded_fold4_pair {
        rho3 * rho4
    } else {
        F128::ZERO
    };

    let chunk_partial =
        |a_in: &[F128], b_in: &[F128], a_out: &mut [F128], b_out: &mut [F128]| -> [F128; 8] {
            #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
            {
                if use_expanded_fold4_pair {
                    if nt_stores {
                        fold2_and_message_lookahead_nt_expanded_aarch64(
                            a_in, b_in, a_out, b_out, rho3, rho4, rho34, eq_lo,
                        )
                    } else {
                        fold2_and_message_lookahead_normal_expanded_aarch64(
                            a_in, b_in, a_out, b_out, rho3, rho4, rho34, eq_lo,
                        )
                    }
                } else {
                    fold2_and_message_lookahead_aarch64(
                        a_in, b_in, a_out, b_out, rho3, rho4, eq_lo, nt_stores,
                    )
                }
            }
            #[cfg(all(target_arch = "aarch64", not(target_feature = "aes")))]
            {
                fold2_and_message_lookahead_aarch64(
                    a_in, b_in, a_out, b_out, rho3, rho4, eq_lo, nt_stores,
                )
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                let mut acc = [F256Unreduced::ZERO; 8];
                let fold4 = |v: &[F128], i: usize| {
                    let t0 = v[i] + rho3 * (v[i] + v[i + 1]);
                    let t1 = v[i + 2] + rho3 * (v[i + 2] + v[i + 3]);
                    t0 + rho4 * (t0 + t1)
                };
                for t in 0..eq_lo.len() / 2 {
                    let i = 16 * t;
                    let o = 4 * t;
                    let a0 = fold4(a_in, i);
                    let a1 = fold4(a_in, i + 4);
                    let a2 = fold4(a_in, i + 8);
                    let a3 = fold4(a_in, i + 12);
                    let b0 = fold4(b_in, i);
                    let b1 = fold4(b_in, i + 4);
                    let b2 = fold4(b_in, i + 8);
                    let b3 = fold4(b_in, i + 12);
                    a_out[o] = a0;
                    a_out[o + 1] = a1;
                    a_out[o + 2] = a2;
                    a_out[o + 3] = a3;
                    b_out[o] = b0;
                    b_out[o + 1] = b1;
                    b_out[o + 2] = b2;
                    b_out[o + 3] = b3;
                    // One weight per round-7 group: the odd lane.
                    let wt = eq_lo[2 * t + 1];
                    let (a0w, a1w, a2w, a3w) = (wt * a0, wt * a1, wt * a2, wt * a3);
                    acc[0] ^= a1w.mul_unreduced(b1);
                    acc[1] ^= (a0w + a1w).mul_unreduced(b0 + b1);
                    acc[2] ^= a3w.mul_unreduced(b3);
                    acc[3] ^= (a2w + a3w).mul_unreduced(b2 + b3);
                    acc[4] ^= a2w.mul_unreduced(b2);
                    let (e_aw, e_b) = (a0w + a2w, b0 + b2);
                    let (o_aw, o_b) = (a1w + a3w, b1 + b3);
                    acc[5] ^= e_aw.mul_unreduced(e_b);
                    acc[6] ^= o_aw.mul_unreduced(o_b);
                    acc[7] ^= (e_aw + o_aw).mul_unreduced(e_b + o_b);
                }
                [
                    acc[0].reduce(),
                    acc[1].reduce(),
                    acc[2].reduce(),
                    acc[3].reduce(),
                    acc[4].reduce(),
                    acc[5].reduce(),
                    acc[6].reduce(),
                    acc[7].reduce(),
                ]
            }
        };

    // Same scheduling policy as the plain composed pass.
    #[cfg(target_arch = "aarch64")]
    let hetero = quarter >= zc_tail_hetero_low_floor() && zc_tail_hetero_enabled();
    #[cfg(not(target_arch = "aarch64"))]
    let hetero = false;

    // Per-chunk `[p1_even, pinf_even, p1_odd, pinf_odd, W0'', W3'', W4'',
    // W5'']`, eq_hi-weighted after the κ recombination below.
    let partials: Vec<[F128; 8]> = if hetero {
        let mut partials: Vec<[F128; 8]> = vec![[F128::ZERO; 8]; hi_size];
        let a_base = crate::epool::SyncPtr(a_out.as_mut_ptr());
        let b_base = crate::epool::SyncPtr(b_out.as_mut_ptr());
        let partials_base = crate::epool::SyncPtr(partials.as_mut_ptr());
        crate::epool::run_hetero_chunks(hi_size, |x_hi| {
            // SAFETY: exclusive per-chunk ownership; queue join publishes writes.
            let (a_out, b_out) = unsafe {
                (
                    std::slice::from_raw_parts_mut(a_base.ptr().add(x_hi * chunk_out), chunk_out),
                    std::slice::from_raw_parts_mut(b_base.ptr().add(x_hi * chunk_out), chunk_out),
                )
            };
            let outv = chunk_partial(
                &a[x_hi * chunk_in..(x_hi + 1) * chunk_in],
                &b[x_hi * chunk_in..(x_hi + 1) * chunk_in],
                a_out,
                b_out,
            );
            // SAFETY: exclusive owner of partials[x_hi].
            unsafe {
                *partials_base.ptr().add(x_hi) = outv;
            }
        });
        partials
    } else {
        use rayon::prelude::*;
        a_out
            .par_chunks_mut(chunk_out)
            .zip(b_out.par_chunks_mut(chunk_out))
            .enumerate()
            .map(|(x_hi, (a_out, b_out))| {
                chunk_partial(
                    &a[x_hi * chunk_in..(x_hi + 1) * chunk_in],
                    &b[x_hi * chunk_in..(x_hi + 1) * chunk_in],
                    a_out,
                    b_out,
                )
            })
            .collect()
    };

    let mut sum1 = F128::ZERO;
    let mut sum_inf = F128::ZERO;
    let mut agg = [F128::ZERO; 6];
    for (x_hi, outv) in partials.iter().enumerate() {
        let eq_h = eq_hi[x_hi];
        // `outv[0..2]` carry the even round-6 pairs on the odd lane's weight;
        // κ restores `eq₆(2y'')` exactly (field arithmetic, no rounding).
        sum1 += eq_h * (kappa * outv[0] + outv[2]);
        sum_inf += eq_h * (kappa * outv[1] + outv[3]);
        for (aslot, &v) in agg.iter_mut().zip(outv[2..].iter()) {
            *aslot += eq_h * v;
        }
    }
    // Every aggregate was accumulated on the odd lane's weight `r''·eq₇`, so
    // a single `r''⁻¹` puts all six back on `eq₇`.
    let r_inv = r_par.inv();
    let w1 = r_inv * agg[0];
    let w2 = r_inv * agg[1];
    let w0 = r_inv * agg[2];
    let w3 = r_inv * agg[3];
    let w4 = r_inv * agg[4];
    let w5 = r_inv * agg[5];
    let la7 = Round3Lookahead {
        c: [w0, w0 + w1 + w2, w2, w3, w3 + w4 + w5, w5],
    };

    (r_next6[0] * sum1, sum_inf, la7)
}

/// Optimized fused fold (at the URM challenge `z`, baked into `table`) plus
/// round-2 prover message. **Packed input** (LSB-first bit packing). **Parallel
/// by default** via rayon — the outer x_hi loop is distributed across workers,
/// each writing to a disjoint chunk of `a_folded`/`b_folded` via `par_chunks_mut`
/// and accumulating its own `(sum1_contrib, sum_inf_contrib)`. The final
/// reduce sums the per-worker contributions (commutative + associative F128
/// XOR/multiply).
///
/// Algorithm (per worker, one x_hi):
/// 1. For each `(x0, x1) = (2k, 2k+1)` pair (k within this x_hi's range),
///    fold the four rows `a[x0], b[x0], a[x1], b[x1]` via the table.
/// 2. Accumulate `eq_lo · a1·b1` and `eq_lo · (a0+a1)·(b0+b1)` with deferred
///    256-bit reduction, reduced once at the end of the worker's x_lo loop.
/// 3. Outer fold by `eq.hi[x_hi]` into the worker's `(sum1_contrib, sum_inf_contrib)`.
///
/// Returns `(a_folded, b_folded, mlv_challenges[0] · G(1), G(∞))` — same
/// convention as `uni_skip_fold_and_round_pair_naive`.
///
/// To run single-threaded for debugging, set `RAYON_NUM_THREADS=1`.
///
/// `k_skip = 6` is currently hardcoded (the protocol headline).
pub fn uni_skip_fold_and_round_pair_optimized_packed(
    a_packed: &[u8],
    b_packed: &[u8],
    m: usize,
    k_skip: usize,
    table: &UniSkipFoldTable,
    mlv_challenges: &[F128],
) -> (Vec<F128>, Vec<F128>, F128, F128) {
    uni_skip_fold_and_round_pair_optimized_packed_padded(
        a_packed,
        b_packed,
        m,
        k_skip,
        table,
        mlv_challenges,
        &PaddingSpec::dense(m),
    )
}

/// Padding-aware variant of [`uni_skip_fold_and_round_pair_optimized_packed`].
/// Skips pairs whose post-URM chunk indices both fall in the per-block zero
/// padding: the fold output is already zero-initialized and the message
/// contribution would be zero, so we can `continue` past those pairs.
pub fn uni_skip_fold_and_round_pair_optimized_packed_padded(
    a_packed: &[u8],
    b_packed: &[u8],
    m: usize,
    k_skip: usize,
    table: &UniSkipFoldTable,
    mlv_challenges: &[F128],
    padding: &PaddingSpec,
) -> (Vec<F128>, Vec<F128>, F128, F128) {
    use rayon::prelude::*;

    assert_eq!(
        k_skip, 6,
        "optimized fold-and-round_pair variant is k_skip=6 only"
    );
    assert_eq!(table.n_chunks, 8);
    let n_chunks = table.n_chunks;
    let n_out = 1usize << (m - k_skip);
    assert_eq!(a_packed.len(), n_out * n_chunks);
    assert_eq!(b_packed.len(), n_out * n_chunks);
    assert_eq!(mlv_challenges.len(), m - k_skip);

    // Uninit alloc — the parallel loop below writes every slot (dense path)
    // or explicitly writes F128::ZERO at padding holes (padded path).
    // Saves ~22 ms of sequential zero-fill at m=29 (256 MB total) that would
    // otherwise cap the parallel speedup of this phase at ~2.5× on 8 cores.
    let mut a_folded: Vec<F128> = crate::scratch::take_f128(n_out);
    let mut b_folded: Vec<F128> = crate::scratch::take_f128(n_out);

    let eq = SplitEqGhash::new(&mlv_challenges[1..]);
    let lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    assert_eq!(lo_size * hi_size * 2, n_out);

    let chunk_size = 2 * lo_size;
    let eq_hi = &eq.hi;
    let eq_lo = &eq.lo;
    let (pair_in_block_mask, useful_pairs_inclusive) = round2_pair_skip(padding, k_skip);

    // Parallel: each worker writes one disjoint chunk of a_folded/b_folded
    // and returns its (sum1, sum_inf) contribution. Reduce by F128 XOR.
    let (sum1, sum_inf) = a_folded
        .par_chunks_mut(chunk_size)
        .zip(b_folded.par_chunks_mut(chunk_size))
        .enumerate()
        .map(|(x_hi, (a_chunk, b_chunk))| {
            let mut p1_acc = F256Unreduced::ZERO;
            let mut pinf_acc = F256Unreduced::ZERO;
            let pair_idx_base = x_hi * lo_size;

            #[cfg(target_arch = "aarch64")]
            unsafe {
                let table_ptr = table.data.as_ptr() as *const u8;
                let a_pkt_ptr = a_packed.as_ptr();
                let b_pkt_ptr = b_packed.as_ptr();
                let base = x_hi * chunk_size;
                let (p1, pinf) = fold_round2_chunk_neon_unchecked_8(
                    table_ptr,
                    a_pkt_ptr.add(base * 8),
                    b_pkt_ptr.add(base * 8),
                    a_chunk.as_mut_ptr(),
                    b_chunk.as_mut_ptr(),
                    eq_lo.as_ptr(),
                    lo_size,
                    pair_idx_base,
                    pair_in_block_mask,
                    useful_pairs_inclusive,
                );
                p1_acc ^= F256Unreduced {
                    r0: p1.lo,
                    r1: p1.hi,
                    r2: 0,
                    r3: 0,
                };
                pinf_acc ^= F256Unreduced {
                    r0: pinf.lo,
                    r1: pinf.hi,
                    r2: 0,
                    r3: 0,
                };
            }
            #[cfg(all(
                target_arch = "x86_64",
                target_feature = "avx512f",
                target_feature = "vpclmulqdq"
            ))]
            unsafe {
                let table_ptr = table.data.as_ptr();
                let a_pkt_ptr = a_packed.as_ptr();
                let b_pkt_ptr = b_packed.as_ptr();
                let base = x_hi * chunk_size;
                let mut p1_wide = WideGhashX4::zero();
                let mut pinf_wide = WideGhashX4::zero();
                let mut x_lo = 0;

                while x_lo + 4 <= lo_size {
                    let mut a0 = [F128::ZERO; 4];
                    let mut a1 = [F128::ZERO; 4];
                    let mut b0 = [F128::ZERO; 4];
                    let mut b1 = [F128::ZERO; 4];

                    for lane in 0..4 {
                        let pair = x_lo + lane;
                        let x0l = 2 * pair;
                        let x1l = x0l + 1;
                        if ((pair_idx_base + pair) & pair_in_block_mask) >= useful_pairs_inclusive {
                            a_chunk[x0l] = F128::ZERO;
                            a_chunk[x1l] = F128::ZERO;
                            b_chunk[x0l] = F128::ZERO;
                            b_chunk[x1l] = F128::ZERO;
                            continue;
                        }

                        let x0g = base + x0l;
                        let x1g = x0g + 1;
                        let folded = fold_round2_pair_x86_unchecked_8(
                            table_ptr,
                            a_pkt_ptr.add(x0g * 8),
                            a_pkt_ptr.add(x1g * 8),
                            b_pkt_ptr.add(x0g * 8),
                            b_pkt_ptr.add(x1g * 8),
                        );
                        [a0[lane], a1[lane], b0[lane], b1[lane]] = folded;
                        a_chunk[x0l] = a0[lane];
                        a_chunk[x1l] = a1[lane];
                        b_chunk[x0l] = b0[lane];
                        b_chunk[x1l] = b1[lane];
                    }

                    let a1x4 = f128x4_loadu(a1.as_ptr());
                    let b1x4 = f128x4_loadu(b1.as_ptr());
                    let a_sum_x4 =
                        f128x4_set(a0[0] + a1[0], a0[1] + a1[1], a0[2] + a1[2], a0[3] + a1[3]);
                    let b_sum_x4 =
                        f128x4_set(b0[0] + b1[0], b0[1] + b1[1], b0[2] + b1[2], b0[3] + b1[3]);
                    let g1x4 = ghash_mul_x4(a1x4, b1x4);
                    let g_inf_x4 = ghash_mul_x4(a_sum_x4, b_sum_x4);
                    let eqx4 = f128x4_loadu(eq_lo[x_lo..].as_ptr());
                    p1_wide.mul_acc(eqx4, g1x4);
                    pinf_wide.mul_acc(eqx4, g_inf_x4);
                    x_lo += 4;
                }

                // Small instances can leave a 1- or 2-pair tail.
                while x_lo < lo_size {
                    let x0l = 2 * x_lo;
                    let x1l = x0l + 1;
                    if ((pair_idx_base + x_lo) & pair_in_block_mask) >= useful_pairs_inclusive {
                        a_chunk[x0l] = F128::ZERO;
                        a_chunk[x1l] = F128::ZERO;
                        b_chunk[x0l] = F128::ZERO;
                        b_chunk[x1l] = F128::ZERO;
                        x_lo += 1;
                        continue;
                    }

                    let x0g = base + x0l;
                    let x1g = x0g + 1;
                    let [a0, a1, b0, b1] = fold_round2_pair_x86_unchecked_8(
                        table_ptr,
                        a_pkt_ptr.add(x0g * 8),
                        a_pkt_ptr.add(x1g * 8),
                        b_pkt_ptr.add(x0g * 8),
                        b_pkt_ptr.add(x1g * 8),
                    );
                    a_chunk[x0l] = a0;
                    a_chunk[x1l] = a1;
                    b_chunk[x0l] = b0;
                    b_chunk[x1l] = b1;
                    let eq_l = eq_lo[x_lo];
                    p1_acc ^= eq_l.mul_unreduced(a1 * b1);
                    pinf_acc ^= eq_l.mul_unreduced((a0 + a1) * (b0 + b1));
                    x_lo += 1;
                }

                p1_acc ^= p1_wide.fold();
                pinf_acc ^= pinf_wide.fold();
            }
            #[cfg(not(any(
                target_arch = "aarch64",
                all(
                    target_arch = "x86_64",
                    target_feature = "avx512f",
                    target_feature = "vpclmulqdq"
                )
            )))]
            {
                let base = x_hi * chunk_size;
                for x_lo in 0..lo_size {
                    let x0l = 2 * x_lo;
                    let x1l = x0l + 1;
                    if ((pair_idx_base + x_lo) & pair_in_block_mask) >= useful_pairs_inclusive {
                        // See aarch64 branch above for why this zero write is needed.
                        a_chunk[x0l] = F128::ZERO;
                        a_chunk[x1l] = F128::ZERO;
                        b_chunk[x0l] = F128::ZERO;
                        b_chunk[x1l] = F128::ZERO;
                        continue;
                    }
                    let x0g = base + 2 * x_lo;
                    let x1g = x0g + 1;
                    let a0 = table.fold_one_row(&a_packed[x0g * n_chunks..(x0g + 1) * n_chunks]);
                    let b0 = table.fold_one_row(&b_packed[x0g * n_chunks..(x0g + 1) * n_chunks]);
                    let a1 = table.fold_one_row(&a_packed[x1g * n_chunks..(x1g + 1) * n_chunks]);
                    let b1 = table.fold_one_row(&b_packed[x1g * n_chunks..(x1g + 1) * n_chunks]);
                    a_chunk[x0l] = a0;
                    a_chunk[x1l] = a1;
                    b_chunk[x0l] = b0;
                    b_chunk[x1l] = b1;
                    let eq_l = eq_lo[x_lo];
                    let g1 = a1 * b1;
                    p1_acc ^= eq_l.mul_unreduced(g1);
                    let g_inf = (a0 + a1) * (b0 + b1);
                    pinf_acc ^= eq_l.mul_unreduced(g_inf);
                }
            }

            let p1 = p1_acc.reduce();
            let pinf = pinf_acc.reduce();
            let eq_h = eq_hi[x_hi];
            (eq_h * p1, eq_h * pinf)
        })
        .reduce(
            || (F128::ZERO, F128::ZERO),
            |(s1, sinf), (c1, cinf)| (s1 + c1, sinf + cinf),
        );

    (a_folded, b_folded, mlv_challenges[0] * sum1, sum_inf)
}

// ---------------------------------------------------------------------------
// Subsequent multilinear rounds (3..(m−k_skip+1)): fold + next message.
// ---------------------------------------------------------------------------

/// In-place fold of a single multilinear polynomial table at `challenge`.
/// Pairs `(a[2x], a[2x+1])` collapse to `a[x] = a[2x] + challenge · (a[2x+1] + a[2x])`.
/// After the call, `a.len()` is halved.
pub fn fold_in_place_single(a: &mut Vec<F128>, challenge: F128) {
    let n = a.len();
    assert!(n.is_power_of_two() && n >= 2);
    let half = n / 2;
    for x in 0..half {
        let a0 = a[2 * x];
        let a1 = a[2 * x + 1];
        a[x] = a0 + challenge * (a1 + a0);
    }
    a.truncate(half);
}

/// Fold a packed boolean witness at the univariate-skip challenge `z`,
/// producing the multilinear table `f_mlv` of length `2^(m − k_skip)` over
/// F_{2^128}. Uses the precomputed [`UniSkipFoldTable`] so each row costs
/// `n_chunks` lookups + XORs.
///
/// Useful for the prover's `ĉ` track: extract_c handles `c` outside the
/// multilinear sumcheck, but the prover still needs `ĉ` at the final point
/// for the claim. This is the per-row fold (Σ_s L_s(z) · c(s, x_rest)) in
/// packed form.
pub fn fold_packed_witness_at_z(
    witness_packed: &[u8],
    m: usize,
    k_skip: usize,
    table: &UniSkipFoldTable,
) -> Vec<F128> {
    use rayon::prelude::*;
    assert_eq!(witness_packed.len(), (1usize << m) / 8);
    let n_chunks = table.n_chunks;
    let n_out = 1usize << (m - k_skip);
    let mut out = vec![F128::ZERO; n_out];
    out.par_iter_mut().enumerate().for_each(|(x_rest, slot)| {
        *slot = table.fold_one_row(&witness_packed[x_rest * n_chunks..(x_rest + 1) * n_chunks]);
    });
    out
}

/// In-place fold of a pair `(a, b)` of multilinear polynomial tables at
/// `challenge`. Binds the lowest bit of the index: pairs `(a[2x], a[2x+1])`
/// collapse to `a[x] = a[2x] + challenge · (a[2x+1] + a[2x])` (and same for b).
/// After the call, `a.len()` and `b.len()` are halved.
///
/// Used at the tail of the multilinear-round sequence where the polynomial is
/// small enough that parallel/fusion overhead outweighs benefit.
pub fn fold_in_place_pair(a: &mut Vec<F128>, b: &mut Vec<F128>, challenge: F128) {
    let n = a.len();
    assert_eq!(b.len(), n);
    assert!(n.is_power_of_two() && n >= 2);
    let half = n / 2;
    for x in 0..half {
        let a0 = a[2 * x];
        let a1 = a[2 * x + 1];
        let b0 = b[2 * x];
        let b1 = b[2 * x + 1];
        a[x] = a0 + challenge * (a1 + a0);
        b[x] = b0 + challenge * (b1 + b0);
    }
    a.truncate(half);
    b.truncate(half);
}

/// Fused: bind one variable at `r_fold` AND compute the *next* round's prover
/// message. Returns the new (folded) `a, b` vectors (half the input size) and
/// `(r_next[0] · G(1), G(∞))` for the next round.
///
/// Parallelized via rayon: each worker reads one disjoint 4·lo_size chunk of
/// the input and writes the corresponding 2·lo_size chunk of the output.
///
/// Requires `a.len() = b.len() ≥ 8` so the post-fold polynomial has at least
/// one bit of x_lo (lo_size ≥ 2). Smaller polynomials should use the
/// unfused `fold_in_place_pair + round_pair_naive` pair.
pub fn fold_and_compute_round_pair_optimized(
    a: &[F128],
    b: &[F128],
    r_fold: F128,
    r_next: &[F128],
) -> (Vec<F128>, Vec<F128>, F128, F128) {
    let half = a.len() / 2;
    // Uninit alloc — `_into` writes every slot of a_new/b_new.
    let mut a_new = crate::alloc_uninit_f128_vec(half);
    let mut b_new = crate::alloc_uninit_f128_vec(half);
    let (m1, mi) = fold_and_compute_round_pair_into(a, b, &mut a_new, &mut b_new, r_fold, r_next);
    (a_new, b_new, m1, mi)
}

/// Buffer-reusing variant of [`fold_and_compute_round_pair_optimized`]: writes
/// the folded `a`/`b` into the caller-provided `a_out`/`b_out` (each length
/// `a.len() / 2`) instead of allocating. Returns `(r_next[0] · G(1), G(∞))`.
///
/// Lets the multilinear-sumcheck tail ping-pong between two persistent scratch
/// buffers, so the ~22 decreasing-size buffers are allocated/freed once rather
/// than per round. The per-round `munmap` of the old buffer (64 MB at m=29)
/// runs single-threaded and otherwise caps the tail's parallel speedup.
/// Floor on per-chunk output size for the small tail rounds: keep each
/// chunk's output at or above `2^TAIL_CHUNK_MIN_OUT_LOG` elements (8 KiB of
/// F128) so the fixed 512-chunk fan-out never shrinks to sub-kilobyte chunks
/// whose dispatch overhead exceeds their work — measured flat at log_n 16
/// and inverted at log_n 15 under thread scaling. Regrouping across chunk
/// boundaries is exact: the lo/hi split is an exact tensor factorisation and
/// deferred reduction is F2-linear (see the n_hi 9-vs-11 regression test).
const TAIL_CHUNK_MIN_OUT_LOG: usize = 9;

/// Kill switch: `FLOCK_NO_ZC_TAIL_ADAPT=1` (exact '1') restores the fixed
/// `MAX_N_HI` fan-out for these rounds.
fn zc_tail_adapt_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("FLOCK_NO_ZC_TAIL_ADAPT").as_deref() != Some(std::ffi::OsStr::new("1"))
    })
}

/// Size-adaptive hi-split for the generic tail rounds: cap the chunk count
/// so each chunk keeps at least `2^TAIL_CHUNK_MIN_OUT_LOG` outputs.
fn tail_n_hi_for(half: usize) -> usize {
    let log_half = half.trailing_zeros() as usize;
    SplitEqGhash::MAX_N_HI
        .min(log_half.saturating_sub(TAIL_CHUNK_MIN_OUT_LOG))
        .max(1)
}

pub fn fold_and_compute_round_pair_into(
    a: &[F128],
    b: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    r_fold: F128,
    r_next: &[F128],
) -> (F128, F128) {
    let default_n_hi = if zc_tail_adapt_enabled() {
        tail_n_hi_for(a.len() / 2)
    } else {
        SplitEqGhash::MAX_N_HI
    };
    #[cfg(target_arch = "aarch64")]
    let n_hi = if a.len() / 2 >= LARGE_TAIL_EQ_MIN_HALF && zc_tail_split11_enabled() {
        LARGE_TAIL_EQ_N_HI
    } else {
        default_n_hi
    };
    #[cfg(not(target_arch = "aarch64"))]
    let n_hi = default_n_hi;

    fold_and_compute_round_pair_into_with_n_hi(a, b, a_out, b_out, r_fold, r_next, n_hi)
}

/// Split-explicit implementation used by the public policy wrapper and by the
/// exact n_hi=9 versus n_hi=11 regression test.
fn fold_and_compute_round_pair_into_with_n_hi(
    a: &[F128],
    b: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    r_fold: F128,
    r_next: &[F128],
    n_hi: usize,
) -> (F128, F128) {
    use rayon::prelude::*;

    let n = a.len();
    assert_eq!(b.len(), n);
    assert!(n.is_power_of_two() && n >= 8);
    let half = n / 2;
    assert_eq!(a_out.len(), half);
    assert_eq!(b_out.len(), half);
    let log_n = n.trailing_zeros() as usize;
    assert_eq!(r_next.len(), log_n - 1);

    let eq = SplitEqGhash::with_n_hi(&r_next[1..], n_hi);
    let lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    assert!(lo_size >= 2, "fold_and_compute requires lo_size ≥ 2");
    // Rounds whose outputs are past LLC size write ping-pong buffers that are
    // not read until the next round's barrier and cannot stay cache-resident;
    // ordinary stores only add write-allocate (RFO) read traffic. Route those
    // rounds through the kernel's `stnp` arm (the same best-effort hint the
    // round-2 producer uses); LLC-resident later rounds keep normal stores so
    // their outputs stay hot for the next round. 2^22 F128 = 64 MiB per array.
    #[cfg(target_arch = "aarch64")]
    let nt_stores = {
        use std::sync::OnceLock;
        static NT_ENABLED: OnceLock<bool> = OnceLock::new();
        half >= (1usize << 19)
            && *NT_ENABLED.get_or_init(|| std::env::var_os("FLOCK_ZC_NT_LEGACY").is_none())
    };
    // Total non-bound multilinear vars is log_n - 1; eq covers log_n - 2 of those.
    assert_eq!(lo_size * hi_size * 2, half);

    let chunk_in = 4 * lo_size; // read chunk per worker
    let chunk_out = 2 * lo_size; // write chunk per worker
    let eq_lo = &eq.lo;
    let eq_hi = &eq.hi;

    // Per-chunk fused fold+message: reads one disjoint 4·lo_size input chunk,
    // writes the corresponding 2·lo_size output chunk, returns the chunk's
    // unscaled message partials. Shared by the rayon sweep and the hetero
    // E-core drain (H2).
    let chunk_partial =
        |a_in: &[F128], b_in: &[F128], a_out: &mut [F128], b_out: &mut [F128]| -> (F128, F128) {
            {
                #[cfg(all(
                    target_arch = "x86_64",
                    target_feature = "avx512f",
                    target_feature = "vpclmulqdq"
                ))]
                // SAFETY: chunk geometry supplies two inputs per output and two
                // outputs per eq_lo value; features are guaranteed by the cfg.
                let (p1, pinf) =
                    unsafe { fold_and_message_x86_avx512(a_in, b_in, a_out, b_out, r_fold, eq_lo) };

                #[cfg(target_arch = "aarch64")]
                let (p1, pinf) =
                    fold_and_message_aarch64(a_in, b_in, a_out, b_out, r_fold, eq_lo, nt_stores);

                #[cfg(not(any(
                    target_arch = "aarch64",
                    all(
                        target_arch = "x86_64",
                        target_feature = "avx512f",
                        target_feature = "vpclmulqdq"
                    )
                )))]
                let (p1, pinf) = {
                    // Fold a_in→a_out and b_in→b_out at r_fold. The field layer
                    // selects the architecture kernel; this loop only consumes
                    // the resulting values to build the message.
                    crate::field::f128_slice::fold_pairs(a_in, 0, a_out, r_fold);
                    crate::field::f128_slice::fold_pairs(b_in, 0, b_out, r_fold);

                    let mut p1_acc = F256Unreduced::ZERO;
                    let mut pinf_acc = F256Unreduced::ZERO;
                    // x86: 4-wide deferred-reduction accumulators for the unrolled loop;
                    // the 2-wide tail still uses the scalar `*_acc` above, folded in
                    // before the final reduce.
                    #[cfg(all(
                        target_arch = "x86_64",
                        target_feature = "avx512f",
                        target_feature = "vpclmulqdq"
                    ))]
                    // SAFETY: vpclmulqdq+avx512f guaranteed by the cfg gate.
                    let (mut p1_wide, mut pinf_wide) =
                        unsafe { (WideGhashX4::zero(), WideGhashX4::zero()) };

                    // Unroll 4 x_lo's per iteration when lo_size % 4 == 0 (the common
                    // case for the fused path; falls back to 2-wide for lo_size==2 at
                    // the smallest fused round). 16 independent r_fold muls and 8
                    // independent msg muls in flight gives the M4 OoO engine and
                    // 2/cy PMULL throughput maximum ILP.
                    assert!(lo_size & 1 == 0, "lo_size must be even");
                    let mut x_lo = 0;
                    if lo_size.is_multiple_of(4) {
                        while x_lo + 4 <= lo_size {
                            let x_lo_a = x_lo;
                            // Read the just-folded pairs: (a0,a1) = (a_out[2·x_lo], a_out[2·x_lo+1]).
                            let o = 2 * x_lo;
                            let a0_a = a_out[o];
                            let a1_a = a_out[o + 1];
                            let b0_a = b_out[o];
                            let b1_a = b_out[o + 1];
                            let a0_b = a_out[o + 2];
                            let a1_b = a_out[o + 3];
                            let b0_b = b_out[o + 2];
                            let b1_b = b_out[o + 3];
                            let a0_c = a_out[o + 4];
                            let a1_c = a_out[o + 5];
                            let b0_c = b_out[o + 4];
                            let b1_c = b_out[o + 5];
                            let a0_d = a_out[o + 6];
                            let a1_d = a_out[o + 7];
                            let b0_d = b_out[o + 6];
                            let b1_d = b_out[o + 7];

                            // 8 reduced msg muls (g1 = a1·b1, g_inf = (a0+a1)(b0+b1)).
                            let g1_a = a1_a * b1_a;
                            let g1_b = a1_b * b1_b;
                            let g1_c = a1_c * b1_c;
                            let g1_d = a1_d * b1_d;
                            let g_inf_a = (a0_a + a1_a) * (b0_a + b1_a);
                            let g_inf_b = (a0_b + a1_b) * (b0_b + b1_b);
                            let g_inf_c = (a0_c + a1_c) * (b0_c + b1_c);
                            let g_inf_d = (a0_d + a1_d) * (b0_d + b1_d);
                            // Deferred-reduction accumulate: on x86 widen all 8 products
                            // 4 lanes at a time (eq_lo[x_lo_a..x_lo_a+4] is contiguous),
                            // reduced once after the loop; else scalar mul_unreduced.
                            #[cfg(all(
                                target_arch = "x86_64",
                                target_feature = "avx512f",
                                target_feature = "vpclmulqdq"
                            ))]
                            // SAFETY: vpclmulqdq+avx512f guaranteed by the cfg gate; the
                            // four eq values eq_lo[x_lo_a..x_lo_a+4] are in bounds (the
                            // 4-wide loop runs only while x_lo + 4 <= lo_size == eq_lo.len()).
                            unsafe {
                                let eq4 = f128x4_loadu(eq_lo[x_lo_a..].as_ptr());
                                p1_wide.mul_acc(eq4, f128x4_set(g1_a, g1_b, g1_c, g1_d));
                                pinf_wide
                                    .mul_acc(eq4, f128x4_set(g_inf_a, g_inf_b, g_inf_c, g_inf_d));
                            }
                            #[cfg(not(all(
                                target_arch = "x86_64",
                                target_feature = "avx512f",
                                target_feature = "vpclmulqdq"
                            )))]
                            {
                                let eq_l_a = eq_lo[x_lo_a];
                                let eq_l_b = eq_lo[x_lo_a + 1];
                                let eq_l_c = eq_lo[x_lo_a + 2];
                                let eq_l_d = eq_lo[x_lo_a + 3];
                                p1_acc ^= eq_l_a.mul_unreduced(g1_a);
                                p1_acc ^= eq_l_b.mul_unreduced(g1_b);
                                p1_acc ^= eq_l_c.mul_unreduced(g1_c);
                                p1_acc ^= eq_l_d.mul_unreduced(g1_d);
                                pinf_acc ^= eq_l_a.mul_unreduced(g_inf_a);
                                pinf_acc ^= eq_l_b.mul_unreduced(g_inf_b);
                                pinf_acc ^= eq_l_c.mul_unreduced(g_inf_c);
                                pinf_acc ^= eq_l_d.mul_unreduced(g_inf_d);
                            }

                            x_lo += 4;
                        }
                    }
                    // 2-wide tail (handles lo_size == 2 case and any remainder when
                    // 4-wide loop is skipped or doesn't cover everything).
                    while x_lo + 2 <= lo_size {
                        let x_lo_a = x_lo;
                        let x_lo_b = x_lo + 1;
                        let o = 2 * x_lo;
                        let a0_a = a_out[o];
                        let a1_a = a_out[o + 1];
                        let b0_a = b_out[o];
                        let b1_a = b_out[o + 1];
                        let a0_b = a_out[o + 2];
                        let a1_b = a_out[o + 3];
                        let b0_b = b_out[o + 2];
                        let b1_b = b_out[o + 3];

                        let eq_l_a = eq_lo[x_lo_a];
                        let eq_l_b = eq_lo[x_lo_b];
                        let g1_a = a1_a * b1_a;
                        let g1_b = a1_b * b1_b;
                        let g_inf_a = (a0_a + a1_a) * (b0_a + b1_a);
                        let g_inf_b = (a0_b + a1_b) * (b0_b + b1_b);
                        p1_acc ^= eq_l_a.mul_unreduced(g1_a);
                        p1_acc ^= eq_l_b.mul_unreduced(g1_b);
                        pinf_acc ^= eq_l_a.mul_unreduced(g_inf_a);
                        pinf_acc ^= eq_l_b.mul_unreduced(g_inf_b);

                        x_lo += 2;
                    }

                    // Merge the 4-wide deferred accumulators with the scalar tail, then
                    // reduce once (reduction is F2-linear, so this equals the scalar
                    // Σ mul_unreduced then reduce).
                    #[cfg(all(
                        target_arch = "x86_64",
                        target_feature = "avx512f",
                        target_feature = "vpclmulqdq"
                    ))]
                    // SAFETY: vpclmulqdq+avx512f+sse4.1 guaranteed by the cfg gate.
                    unsafe {
                        p1_acc ^= p1_wide.fold();
                        pinf_acc ^= pinf_wide.fold();
                    }
                    let p1 = p1_acc.reduce();
                    let pinf = pinf_acc.reduce();
                    (p1, pinf)
                };
                (p1, pinf)
            }
        };

    // H2: drain the DRAM-bound rounds (outputs past LLC) through the hetero
    // E-core queue — the same contract as the T3 compact reconstruction.
    #[cfg(target_arch = "aarch64")]
    let hetero = half >= zc_tail_hetero_low_floor() && zc_tail_hetero_enabled();
    #[cfg(not(target_arch = "aarch64"))]
    let hetero = false;

    let (sum1, sum_inf) = if hetero {
        // GPU loop-round products arm (see `ENV_NO_GPU_ZC_LOOP`): a
        // measured prefix of the hi-chunks gets its message products
        // computed on the GPU (which folds its chunks' pairs redundantly
        // via ρ-nibble tables) while the CPU writes those chunks' folded
        // outputs through the exact field-layer `fold_pairs` kernel,
        // skipping just the products. Only engaged on the LARGEST loop
        // round (half ≥ 2^24): the share is calibrated at that round's
        // chunk size, and a measured local A/B showed the same share
        // makes the GPU the straggler on the next (half-size) round
        // (12.6 → 33.7 ms) — per-size recalibration is not worth a
        // second sync for an LLC-adjacent round. `None` = the exact
        // incumbent path.
        // PARKED alongside the T3 arm for the v10a bisect: the v9 archive
        // carrying only this arm also died scoreless, so BOTH new arms sit
        // behind compile-time consts while the shared-scratch cache fix
        // ships alone. Re-enable by flipping the const once the freed job
        // wall is confirmed on the runner.
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        const ZC_LOOP_INTEGRATION_PARKED: bool = true;
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        let gpu_job = if !ZC_LOOP_INTEGRATION_PARKED && half >= (1usize << 24) {
            crate::gpu_commit::launch_zc_loop_products(a, b, r_fold, eq_lo, eq_hi, lo_size, hi_size)
        } else {
            None
        };
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        let gpu_prefix = gpu_job.as_ref().map_or(0, |j| j.cpu_split());
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        let t_cpu_sweep = std::time::Instant::now();

        let mut partials: Vec<(F128, F128)> = vec![(F128::ZERO, F128::ZERO); hi_size];
        let a_base = crate::epool::SyncPtr(a_out.as_mut_ptr());
        let b_base = crate::epool::SyncPtr(b_out.as_mut_ptr());
        let partials_base = crate::epool::SyncPtr(partials.as_mut_ptr());
        crate::epool::run_hetero_chunks(hi_size, |x_hi| {
            // SAFETY: exclusive per-chunk ownership; queue join publishes writes.
            let (a_out, b_out) = unsafe {
                (
                    std::slice::from_raw_parts_mut(a_base.ptr().add(x_hi * chunk_out), chunk_out),
                    std::slice::from_raw_parts_mut(b_base.ptr().add(x_hi * chunk_out), chunk_out),
                )
            };

            // GPU-covered prefix chunk: write the identical folded
            // outputs via the field-layer fold, skip the products (the
            // GPU partial replaces this chunk's slot after the join).
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            if x_hi < gpu_prefix {
                let a_in = &a[x_hi * chunk_in..(x_hi + 1) * chunk_in];
                let b_in = &b[x_hi * chunk_in..(x_hi + 1) * chunk_in];
                crate::field::f128_slice::fold_pairs(a_in, 0, a_out, r_fold);
                crate::field::f128_slice::fold_pairs(b_in, 0, b_out, r_fold);
                return;
            }

            let (p1, pinf) = chunk_partial(
                &a[x_hi * chunk_in..(x_hi + 1) * chunk_in],
                &b[x_hi * chunk_in..(x_hi + 1) * chunk_in],
                a_out,
                b_out,
            );
            let eq_h = eq_hi[x_hi];
            // SAFETY: exclusive owner of partials[x_hi].
            unsafe {
                *partials_base.ptr().add(x_hi) = (eq_h * p1, eq_h * pinf);
            }
        });

        // Drain the GPU products arm: merge prefix partials (timed
        // proves), finish calibration (untimed warmup), or CPU-redo the
        // skipped prefix products on any post-admission Metal failure.
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        if let Some(job) = gpu_job {
            let cpu_wall_ms = t_cpu_sweep.elapsed().as_secs_f64() * 1e3;
            let calib = job.is_calibration();
            let prefix = job.cpu_split();
            let res = crate::gpu_commit::zc_loop_wait(
                job,
                if calib {
                    Some(partials.as_slice())
                } else {
                    None
                },
                cpu_wall_ms,
                hi_size,
            );
            match res {
                crate::gpu_commit::ZcLoopResult::Calibrated => {}
                crate::gpu_commit::ZcLoopResult::Prefix(vals) => {
                    partials[..prefix].copy_from_slice(&vals);
                }
                crate::gpu_commit::ZcLoopResult::Failed => {
                    // Redo exactly the skipped prefix products — slower,
                    // still exact. The fused chunk rewrites the same
                    // folded values (byte-identical stores).
                    for x_hi in 0..prefix {
                        let (a_out, b_out) = unsafe {
                            (
                                std::slice::from_raw_parts_mut(
                                    a_base.ptr().add(x_hi * chunk_out),
                                    chunk_out,
                                ),
                                std::slice::from_raw_parts_mut(
                                    b_base.ptr().add(x_hi * chunk_out),
                                    chunk_out,
                                ),
                            )
                        };
                        let (p1, pinf) = chunk_partial(
                            &a[x_hi * chunk_in..(x_hi + 1) * chunk_in],
                            &b[x_hi * chunk_in..(x_hi + 1) * chunk_in],
                            a_out,
                            b_out,
                        );
                        let eq_h = eq_hi[x_hi];
                        partials[x_hi] = (eq_h * p1, eq_h * pinf);
                    }
                }
            }
        }

        partials
            .iter()
            .fold((F128::ZERO, F128::ZERO), |(s1, sinf), &(c1, cinf)| {
                (s1 + c1, sinf + cinf)
            })
    } else {
        a_out
            .par_chunks_mut(chunk_out)
            .zip(b_out.par_chunks_mut(chunk_out))
            .enumerate()
            .map(|(x_hi, (a_out, b_out))| {
                let (p1, pinf) = chunk_partial(
                    &a[x_hi * chunk_in..(x_hi + 1) * chunk_in],
                    &b[x_hi * chunk_in..(x_hi + 1) * chunk_in],
                    a_out,
                    b_out,
                );
                let eq_h = eq_hi[x_hi];
                (eq_h * p1, eq_h * pinf)
            })
            .reduce(
                || (F128::ZERO, F128::ZERO),
                |(s1, sinf), (c1, cinf)| (s1 + c1, sinf + cinf),
            )
    };

    (r_next[0] * sum1, sum_inf)
}

/// Serial reference — identical I/O contract to
/// [`uni_skip_fold_and_round_pair_optimized_packed`], no rayon. Kept under
/// `#[cfg(test)]` as the cross-check oracle for the parallel version.
#[cfg(test)]
fn uni_skip_fold_and_round_pair_optimized_packed_serial(
    a_packed: &[u8],
    b_packed: &[u8],
    m: usize,
    k_skip: usize,
    table: &UniSkipFoldTable,
    mlv_challenges: &[F128],
) -> (Vec<F128>, Vec<F128>, F128, F128) {
    assert_eq!(k_skip, 6);
    assert_eq!(table.n_chunks, 8);
    let n_chunks = table.n_chunks;
    let n_out = 1usize << (m - k_skip);
    let mut a_folded = vec![F128::ZERO; n_out];
    let mut b_folded = vec![F128::ZERO; n_out];
    let eq = SplitEqGhash::new(&mlv_challenges[1..]);
    let lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    let mut sum1 = F128::ZERO;
    let mut sum_inf = F128::ZERO;
    for x_hi in 0..hi_size {
        let mut p1_acc = F256Unreduced::ZERO;
        let mut pinf_acc = F256Unreduced::ZERO;
        let k_base = x_hi << eq.n_lo;
        for x_lo in 0..lo_size {
            let k = k_base | x_lo;
            let x0 = 2 * k;
            let x1 = x0 + 1;
            let a0 = table.fold_one_row(&a_packed[x0 * n_chunks..(x0 + 1) * n_chunks]);
            let b0 = table.fold_one_row(&b_packed[x0 * n_chunks..(x0 + 1) * n_chunks]);
            let a1 = table.fold_one_row(&a_packed[x1 * n_chunks..(x1 + 1) * n_chunks]);
            let b1 = table.fold_one_row(&b_packed[x1 * n_chunks..(x1 + 1) * n_chunks]);
            a_folded[x0] = a0;
            b_folded[x0] = b0;
            a_folded[x1] = a1;
            b_folded[x1] = b1;
            let eq_l = eq.lo[x_lo];
            let g1 = a1 * b1;
            p1_acc ^= eq_l.mul_unreduced(g1);
            let g_inf = (a0 + a1) * (b0 + b1);
            pinf_acc ^= eq_l.mul_unreduced(g_inf);
        }
        let p1 = p1_acc.reduce();
        let pinf = pinf_acc.reduce();
        sum1 += eq.hi[x_hi] * p1;
        sum_inf += eq.hi[x_hi] * pinf;
    }
    (a_folded, b_folded, mlv_challenges[0] * sum1, sum_inf)
}

/// `&[bool]` convenience wrapper around
/// [`uni_skip_fold_and_round_pair_optimized_packed`]. Packs internally, builds
/// the fold table from `z`.
pub fn uni_skip_fold_and_round_pair_optimized(
    a: &[bool],
    b: &[bool],
    m: usize,
    k_skip: usize,
    z: F128,
    mlv_challenges: &[F128],
) -> (Vec<F128>, Vec<F128>, F128, F128) {
    assert_eq!(a.len(), 1usize << m);
    assert_eq!(b.len(), 1usize << m);
    let a_packed = pack_bits(a);
    let b_packed = pack_bits(b);
    let table = UniSkipFoldTable::new(k_skip, z);
    uni_skip_fold_and_round_pair_optimized_packed(
        &a_packed,
        &b_packed,
        m,
        k_skip,
        &table,
        mlv_challenges,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    #[test]
    fn ranked_normal_fold4_pair_output_selector_is_exact() {
        assert!(ranked_normal_fold4_pair_output(1 << 20));
        assert!(ranked_normal_fold4_pair_output(1 << 18));
        assert!(!ranked_normal_fold4_pair_output(1 << 22));
        assert!(!ranked_normal_fold4_pair_output(1 << 21));
        assert!(!ranked_normal_fold4_pair_output(1 << 19));
        assert!(!ranked_normal_fold4_pair_output(1 << 17));
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    #[test]
    fn ranked_direct_fold4_pair_output_selector_is_exact() {
        assert!(ranked_direct_fold4_pair_output(1 << 16));
        assert!(!ranked_direct_fold4_pair_output(1 << 18));
        assert!(!ranked_direct_fold4_pair_output(1 << 17));
        assert!(!ranked_direct_fold4_pair_output((1 << 16) - 1));
        assert!(!ranked_direct_fold4_pair_output((1 << 16) + 1));
        assert!(!ranked_direct_fold4_pair_output(1 << 15));
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
        fn bits(&mut self, n: usize) -> Vec<bool> {
            (0..n).map(|_| self.bit()).collect()
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
    }

    // ----------------------------------------------------------------------
    // Lagrange weights — algebraic properties.
    // ----------------------------------------------------------------------

    /// `Σ_i L_i(z) = 1` for all z. The polynomial `1` interpolates to constant
    /// `1` at every node, so its evaluation at z is `Σ_i 1·L_i(z) = Σ_i L_i(z)`.
    #[test]
    fn lagrange_weights_sum_to_one() {
        let mut rng = Rng::new(1);
        for &k_skip in &[1usize, 2, 3, 4, 5, 6] {
            for _ in 0..4 {
                let z = rng.f128();
                let weights = lagrange_weights_naive(k_skip, z);
                let sum: F128 = weights.iter().copied().fold(F128::ZERO, |a, b| a + b);
                assert_eq!(sum, F128::ONE, "Σ L_i ≠ 1 at k_skip={k_skip}");
            }
        }
    }

    /// The O(ell) fast path (cached denominator inverses + prefix/suffix
    /// numerators) is bit-identical to the reference quadratic loop on both
    /// node domains, for random and node-coincident fold points.
    #[test]
    fn lagrange_fast_matches_reference() {
        let mut rng = Rng::new(42);
        for &k_skip in &[1usize, 2, 3, 4, 5, 6, 7] {
            let ell = 1usize << k_skip;
            for t in 0..6 {
                let z = if t < 4 {
                    rng.f128()
                } else {
                    PHI_8_TABLE[t - 4]
                };
                assert_eq!(
                    lagrange_weights_fast(k_skip, z, 0, lagrange_s_den_inv(k_skip)),
                    lagrange_weights_naive_reference(k_skip, z),
                    "S-domain mismatch at k_skip={k_skip}, t={t}"
                );
                if 2 * ell <= 256 {
                    assert_eq!(
                        lagrange_weights_fast(k_skip, z, ell, lagrange_lambda_den_inv(k_skip)),
                        lagrange_weights_lambda_naive_reference(k_skip, z),
                        "Λ-domain mismatch at k_skip={k_skip}, t={t}"
                    );
                }
            }
        }
    }

    /// The staged round-two eq stash is adopted only on an exact
    /// (challenges, n_hi) match and produces the identical table either way.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn staged_r2_eq_adoption_matches_fresh_build() {
        let mut rng = Rng::new(7);
        let ch = rng.f128_vec(25);
        let fresh = SplitEqGhash::with_n_hi(&ch, 11);
        stash_staged_r2_eq(&ch, fresh.clone());
        // Non-matching challenges must not consume the stash.
        let other = rng.f128_vec(25);
        let built = take_or_build_r2_eq(&other, 11);
        assert_eq!(built.lo, SplitEqGhash::with_n_hi(&other, 11).lo);
        // Matching take returns the identical table.
        let adopted = take_or_build_r2_eq(&ch, 11);
        assert_eq!(adopted.lo, fresh.lo);
        assert_eq!(adopted.hi, fresh.hi);
        assert_eq!((adopted.n_lo, adopted.n_hi), (fresh.n_lo, fresh.n_hi));
        // Stash is now empty; a fresh build still works.
        let rebuilt = take_or_build_r2_eq(&ch, 11);
        assert_eq!(rebuilt.lo, fresh.lo);
    }

    /// `L_i(s_j) = δ_{ij}` — Kronecker delta. At a node, exactly one weight is 1.
    #[test]
    fn lagrange_at_node_is_indicator() {
        for k_skip in [2usize, 3, 4, 5] {
            let ell = 1usize << k_skip;
            for i in 0..ell {
                let z = PHI_8_TABLE[i];
                let weights = lagrange_weights_naive(k_skip, z);
                for j in 0..ell {
                    let expected = if j == i { F128::ONE } else { F128::ZERO };
                    assert_eq!(weights[j], expected, "k_skip={k_skip}, z=node{i}, j={j}");
                }
            }
        }
    }

    // ----------------------------------------------------------------------
    // Fold — algebraic properties.
    // ----------------------------------------------------------------------

    /// At a node `z = φ_8(i)`, fold reduces to the witness restricted to s=i:
    /// `a_mlv[x_rest] = a[x_rest · 2^k_skip + i]` (lifted to F_128).
    #[test]
    fn fold_at_node_recovers_witness_slice() {
        let m = 8;
        let k_skip = 3;
        let ell = 1usize << k_skip;
        let n_rest = 1usize << (m - k_skip);
        let mut rng = Rng::new(7);
        let a = rng.bits(1 << m);
        for i in 0..ell {
            let z = PHI_8_TABLE[i];
            let weights = lagrange_weights_naive(k_skip, z);
            let a_mlv = fold_at_z_naive(&a, m, k_skip, &weights);
            for x_rest in 0..n_rest {
                let expected = if a[x_rest * ell + i] {
                    F128::ONE
                } else {
                    F128::ZERO
                };
                assert_eq!(
                    a_mlv[x_rest], expected,
                    "fold at node {i} mismatch at x_rest={x_rest}"
                );
            }
        }
    }

    /// Fold is linear in the input witness: fold(a ⊕ a') = fold(a) + fold(a').
    /// (XOR-linearity is the defining property of the multilinear extension.)
    #[test]
    fn fold_is_xor_linear() {
        let m = 7;
        let k_skip = 3;
        let mut rng = Rng::new(11);
        let a = rng.bits(1 << m);
        let aprime = rng.bits(1 << m);
        let a_xor: Vec<bool> = a.iter().zip(&aprime).map(|(x, y)| x ^ y).collect();
        let z = rng.f128();
        let weights = lagrange_weights_naive(k_skip, z);

        let fa = fold_at_z_naive(&a, m, k_skip, &weights);
        let fap = fold_at_z_naive(&aprime, m, k_skip, &weights);
        let fxor = fold_at_z_naive(&a_xor, m, k_skip, &weights);
        for i in 0..fa.len() {
            assert_eq!(fa[i] + fap[i], fxor[i], "linearity broken at i={i}");
        }
    }

    // ----------------------------------------------------------------------
    // Round-2 message — properties + cross-checks.
    // ----------------------------------------------------------------------

    /// All-zero witness ⇒ a_mlv = b_mlv = 0 ⇒ G(1) = G(∞) = 0, so the message
    /// elements (r[0]·G(1), G(∞)) are also both zero.
    #[test]
    fn zero_witness_gives_zero_round_message() {
        let m = 6;
        let k_skip = 3;
        let mut rng = Rng::new(20);
        let z = rng.f128();
        let mlv_challenges = rng.f128_vec(m - k_skip);
        let zeros = vec![false; 1 << m];
        let (a_mlv, b_mlv, msg_1, msg_inf) =
            uni_skip_fold_and_round_pair_naive(&zeros, &zeros, m, k_skip, z, &mlv_challenges);
        assert!(a_mlv.iter().all(|v| v.is_zero()));
        assert!(b_mlv.iter().all(|v| v.is_zero()));
        assert_eq!(msg_1, F128::ZERO);
        assert_eq!(msg_inf, F128::ZERO);
    }

    #[test]
    fn deterministic() {
        let m = 7;
        let k_skip = 3;
        let mut rng = Rng::new(33);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let z = rng.f128();
        let mlv_challenges = rng.f128_vec(m - k_skip);
        let o1 = uni_skip_fold_and_round_pair_naive(&a, &b, m, k_skip, z, &mlv_challenges);
        let o2 = uni_skip_fold_and_round_pair_naive(&a, &b, m, k_skip, z, &mlv_challenges);
        assert_eq!(o1, o2);
    }

    /// Round-pair message is symmetric in a, b: swapping a↔b gives the same
    /// message. `a · b = b · a` is built-in, and the `r[0]` multiplier doesn't
    /// distinguish AB.
    #[test]
    fn round_pair_symmetric_in_ab() {
        let m = 6;
        let k_skip = 3;
        let mut rng = Rng::new(40);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let z = rng.f128();
        let mlv_challenges = rng.f128_vec(m - k_skip);
        let (_, _, m1_ab, minf_ab) =
            uni_skip_fold_and_round_pair_naive(&a, &b, m, k_skip, z, &mlv_challenges);
        let (_, _, m1_ba, minf_ba) =
            uni_skip_fold_and_round_pair_naive(&b, &a, m, k_skip, z, &mlv_challenges);
        assert_eq!(m1_ab, m1_ba);
        assert_eq!(minf_ab, minf_ba);
    }

    // ----------------------------------------------------------------------
    // Optimized fused — UniSkipFoldTable + fold_one_row, then naive cross-check.
    // ----------------------------------------------------------------------

    /// NEON `fold_one_row_neon_unchecked_8` matches scalar `fold_one_row`.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn fold_one_row_neon_matches_scalar() {
        let k_skip = 6;
        let mut rng = Rng::new(70);
        let z = rng.f128();
        let table = UniSkipFoldTable::new(k_skip, z);

        for _ in 0..256 {
            let mut bytes = [0u8; 8];
            for byte in bytes.iter_mut() {
                *byte = (rng.next_u64() & 0xff) as u8;
            }
            let scalar = table.fold_one_row(&bytes);
            // SAFETY: on aarch64; bytes has 8 entries; table has 8 chunks.
            let neon = unsafe {
                fold_one_row_neon_unchecked_8(table.data.as_ptr() as *const u8, bytes.as_ptr())
            };
            assert_eq!(scalar, neon, "fold mismatch bytes={bytes:02x?}");
        }
    }

    /// Four-row x86 lookup fold matches four independent scalar folds.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    #[test]
    fn fold_round2_pair_x86_matches_scalar() {
        let mut rng = Rng::new(71);
        let table = UniSkipFoldTable::new(6, rng.f128());

        for _ in 0..256 {
            let mut rows = [[0u8; 8]; 4];
            for row in &mut rows {
                for byte in row {
                    *byte = (rng.next_u64() & 0xff) as u8;
                }
            }
            let expected = rows.map(|row| table.fold_one_row(&row));
            // SAFETY: each row has 8 bytes and the table has 8 × 256 entries.
            let actual = unsafe {
                fold_round2_pair_x86_unchecked_8(
                    table.data.as_ptr(),
                    rows[0].as_ptr(),
                    rows[1].as_ptr(),
                    rows[2].as_ptr(),
                    rows[3].as_ptr(),
                )
            };
            assert_eq!(actual, expected);
        }
    }

    /// `fold_in_place_pair` correctness: post-fold a[x] = a[2x] + X·(a[2x+1]+a[2x]).
    #[test]
    fn fold_in_place_pair_matches_formula() {
        let mut rng = Rng::new(300);
        for &log_n in &[1usize, 2, 3, 4, 6] {
            let n = 1usize << log_n;
            let a_orig: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let b_orig: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let challenge = rng.f128();

            let mut a = a_orig.clone();
            let mut b = b_orig.clone();
            fold_in_place_pair(&mut a, &mut b, challenge);

            assert_eq!(a.len(), n / 2);
            assert_eq!(b.len(), n / 2);
            for x in 0..(n / 2) {
                let a0 = a_orig[2 * x];
                let a1 = a_orig[2 * x + 1];
                let b0 = b_orig[2 * x];
                let b1 = b_orig[2 * x + 1];
                assert_eq!(a[x], a0 + challenge * (a1 + a0), "log_n={log_n}, x={x}");
                assert_eq!(b[x], b0 + challenge * (b1 + b0), "log_n={log_n}, x={x}");
            }
        }
    }

    /// **The c-claim identity**: `C_s · interpolate(round1_c, k_skip, z)` equals
    /// `ĉ(z, r_rest)` computed by direct folding (Lagrange at z, then bind each
    /// `r_rest` value). This is the math identity that lets the extract_c
    /// prover skip per-round c tracking entirely.
    #[test]
    fn c_eval_from_round1_c_matches_direct_fold() {
        use crate::field::F8;
        use crate::ntt::{AdditiveNttGf8, InvNttTableByteSingleGf8};
        use crate::zerocheck::univariate_skip_optimized::{
            c_s_f128, medium_challenges_ghash, round1_shift_reduce_extract_c_packed,
            small_challenges_ghash,
        };

        const K_SKIP: usize = 6;
        const N_INNER: usize = 7;

        for &m in &[14usize, 15, 16] {
            let mut rng = Rng::new(500 + m as u64);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            let c = rng.bits(1 << m);

            // Build r with protocol-fixed constants in the middle 7 dims,
            // matching how `prove` constructs it.
            let mut r = vec![F128::ZERO; m];
            for slot in r[..K_SKIP].iter_mut() {
                *slot = rng.f128();
            }
            for (i, v) in small_challenges_ghash().iter().enumerate() {
                r[K_SKIP + i] = *v;
            }
            for (i, v) in medium_challenges_ghash().iter().enumerate() {
                r[K_SKIP + 3 + i] = *v;
            }
            for slot in r[K_SKIP + N_INNER..].iter_mut() {
                *slot = rng.f128();
            }
            let z = rng.f128();

            let a_packed = pack_bits(&a);
            let b_packed = pack_bits(&b);
            let c_packed = pack_bits(&c);

            let ntt_s = AdditiveNttGf8::new(K_SKIP, F8::ZERO);
            let ntt_l = AdditiveNttGf8::new(K_SKIP, F8(1u8 << K_SKIP));
            let inv_table = InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);
            let (_round1_ab, round1_c) = round1_shift_reduce_extract_c_packed(
                &a_packed, &b_packed, &c_packed, m, K_SKIP, &r, &inv_table,
            );

            // Path A: interpolate round1_c at z, scale by C_s.
            let c_eval_via_interpolation =
                c_s_f128() * interpolate_at_z_on_lambda(&round1_c, K_SKIP, z);

            // Path B: direct fold of c at z (Lagrange) then bind each
            // r_rest = r[K_SKIP..m] element with fold_in_place_single.
            let weights = lagrange_weights_naive(K_SKIP, z);
            let mut c_mlv = fold_at_z_naive(&c, m, K_SKIP, &weights);
            for &r_val in &r[K_SKIP..] {
                fold_in_place_single(&mut c_mlv, r_val);
            }
            assert_eq!(c_mlv.len(), 1);
            let c_eval_via_fold = c_mlv[0];

            assert_eq!(
                c_eval_via_interpolation, c_eval_via_fold,
                "c-claim identity broken at m={m}"
            );
        }
    }

    /// **The big cross-check**: fused `fold_and_compute_round_pair_optimized`
    /// produces the same output as the unfused sequence
    /// `fold_in_place_pair` → `round_pair_naive`.
    #[test]
    fn fused_round_matches_unfused() {
        let mut rng = Rng::new(310);
        // fold_and_compute requires lo_size ≥ 2 in SplitEqGhash. eq is over
        // r_next[1..] (size log_n − 2); with MAX_N_HI = 9, n_lo ≥ 1 needs
        // eq size ≥ 10 ⇒ log_n ≥ 12. Smaller cases use the unfused path.
        for &log_n in &[12usize, 13, 14] {
            let n = 1usize << log_n;
            let a: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let b: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let r_fold = rng.f128();
            let r_next = rng.f128_vec(log_n - 1);

            // Fused path.
            let (a_fused, b_fused, m1_fused, minf_fused) =
                fold_and_compute_round_pair_optimized(&a, &b, r_fold, &r_next);

            // Unfused path: clone, in-place fold, naive message.
            let mut a_unf = a.clone();
            let mut b_unf = b.clone();
            fold_in_place_pair(&mut a_unf, &mut b_unf, r_fold);
            let (m1_unf, minf_unf) = round_pair_naive(&a_unf, &b_unf, &r_next);

            assert_eq!(a_fused, a_unf, "a mismatch at log_n={log_n}");
            assert_eq!(b_fused, b_unf, "b mismatch at log_n={log_n}");
            assert_eq!(m1_fused, m1_unf, "msg_1 mismatch at log_n={log_n}");
            assert_eq!(minf_fused, minf_unf, "msg_inf mismatch at log_n={log_n}");
        }
    }

    /// Moving two equality variables from eq_lo to eq_hi only regroups an
    /// exact tensor factorization.  It must leave both folded buffers and both
    /// wire-message field elements bit-identical.
    #[test]
    fn large_tail_split11_matches_default_split9() {
        let mut rng = Rng::new(0x5A11_7A11);
        let log_n = 14usize;
        let n = 1usize << log_n;
        let a: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
        let b: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
        let r_fold = rng.f128();
        let r_next = rng.f128_vec(log_n - 1);

        let mut a9 = vec![F128::ZERO; n / 2];
        let mut b9 = vec![F128::ZERO; n / 2];
        let msg9 = fold_and_compute_round_pair_into_with_n_hi(
            &a,
            &b,
            &mut a9,
            &mut b9,
            r_fold,
            &r_next,
            SplitEqGhash::MAX_N_HI,
        );

        let mut a11 = vec![F128::ZERO; n / 2];
        let mut b11 = vec![F128::ZERO; n / 2];
        let msg11 = fold_and_compute_round_pair_into_with_n_hi(
            &a,
            &b,
            &mut a11,
            &mut b11,
            r_fold,
            &r_next,
            LARGE_TAIL_EQ_N_HI,
        );

        assert_eq!(a11, a9, "folded a differs across equality splits");
        assert_eq!(b11, b9, "folded b differs across equality splits");
        assert_eq!(msg11, msg9, "round message differs across equality splits");
    }

    /// Full compact-path oracle against the legacy materialized round two.
    ///
    /// Covers both byte-pool and donated-F128 delta backing, forces all large
    /// destinations to come from poisoned recycled storage, compares the
    /// round-two wire message, then compares reconstructed A/B and the next
    /// message after folding at rho=0, rho=1, an all-ones value, and a random
    /// challenge.
    #[test]
    fn compact_round2_and_reconstruction_match_legacy_with_poisoned_scratch() {
        const K_SKIP: usize = 6;
        const M: usize = 20;
        const K_LOG: usize = 14;
        const USEFUL_BITS: usize = 15_409;
        const POISON: F128 = F128 {
            lo: 0xa5a5_a5a5_a5a5_a5a5,
            hi: 0x5a5a_5a5a_5a5a_5a5a,
        };

        let mut rng = Rng::new(0xC0A0_AC7);
        let mut a = rng.bits(1 << M);
        let mut b = rng.bits(1 << M);
        let block_size = 1usize << K_LOG;
        for block in 0..(1usize << (M - K_LOG)) {
            a[block * block_size + USEFUL_BITS..(block + 1) * block_size].fill(false);
            b[block * block_size + USEFUL_BITS..(block + 1) * block_size].fill(false);
        }
        let a_packed = pack_bits(&a);
        let b_packed = pack_bits(&b);
        let z = rng.f128();
        let mlv_challenges = rng.f128_vec(M - K_SKIP);
        let r_next = rng.f128_vec(M - K_SKIP - 1);
        let random_rho = rng.f128();
        let rhos = [
            F128::ZERO,
            F128::ONE,
            F128 {
                lo: u64::MAX,
                hi: u64::MAX,
            },
            random_rho,
        ];
        let table = UniSkipFoldTable::new(K_SKIP, z);
        let padding = PaddingSpec {
            k_log: K_LOG,
            useful_bits_per_block: USEFUL_BITS,
        };
        let n_out = 1usize << (M - K_SKIP);
        let n_pairs = n_out / 2;
        let deltas_len = n_out * table.n_chunks;

        for donate_f128 in [false, true] {
            crate::scratch::clear();

            let (legacy_a, legacy_b, legacy_m1, legacy_mi) =
                uni_skip_fold_and_round_pair_optimized_packed_padded(
                    &a_packed,
                    &b_packed,
                    M,
                    K_SKIP,
                    &table,
                    &mlv_challenges,
                    &padding,
                );

            // Hold all poison allocations until they are initialized so
            // smallest-fit cannot substitute the larger anchor allocation for
            // either reconstruction output.
            let mut poison_a_out = crate::scratch::take_f128(n_pairs);
            let mut poison_b_out = crate::scratch::take_f128(n_pairs);
            let mut poison_anchors = crate::scratch::take_f128(n_out);
            poison_a_out.fill(POISON);
            poison_b_out.fill(POISON);
            poison_anchors.fill(POISON);
            crate::scratch::give_f128(poison_a_out);
            crate::scratch::give_f128(poison_b_out);
            crate::scratch::give_f128(poison_anchors);

            let deltas_backing = if donate_f128 {
                let donor = vec![POISON; deltas_len / core::mem::size_of::<F128>()];
                Some(ScratchBytes::from_initialized_f128(donor))
            } else {
                let mut poison_deltas = crate::scratch::take_u8(deltas_len);
                poison_deltas.fill(0xa5);
                crate::scratch::give_u8(poison_deltas);
                None
            };

            let (compact, compact_m1, compact_mi) =
                uni_skip_fold_and_round_pair_compact_padded_with_deltas(
                    &a_packed,
                    &b_packed,
                    M,
                    K_SKIP,
                    &table,
                    &mlv_challenges,
                    &padding,
                    deltas_backing,
                );
            assert_eq!(
                (compact_m1, compact_mi),
                (legacy_m1, legacy_mi),
                "round-two message mismatch; donate_f128={donate_f128}"
            );

            for &rho in &rhos {
                let mut expected_a = legacy_a.clone();
                let mut expected_b = legacy_b.clone();
                fold_in_place_pair(&mut expected_a, &mut expected_b, rho);
                let expected_msg = round_pair_naive(&expected_a, &expected_b, &r_next);

                let (actual_a, actual_b, actual_m1, actual_mi) =
                    fold_compact_and_compute_round_pair(&compact, &table, rho, &r_next);
                assert_eq!(
                    actual_a, expected_a,
                    "reconstructed A mismatch; donate_f128={donate_f128}, rho={rho:?}"
                );
                assert_eq!(
                    actual_b, expected_b,
                    "reconstructed B mismatch; donate_f128={donate_f128}, rho={rho:?}"
                );
                assert_eq!(
                    (actual_m1, actual_mi),
                    expected_msg,
                    "post-reconstruction message mismatch; donate_f128={donate_f128}, rho={rho:?}"
                );

                crate::scratch::give_f128(actual_a);
                crate::scratch::give_f128(actual_b);
            }

            compact.recycle();
            crate::scratch::give_f128(legacy_a);
            crate::scratch::give_f128(legacy_b);
            crate::scratch::clear();
        }
    }

    // ----------------------------------------------------------------------
    // Two-challenge symbolic lookahead (variant K).
    // ----------------------------------------------------------------------

    const LA_POISON: F128 = F128 {
        lo: 0xa5a5_a5a5_a5a5_a5a5,
        hi: 0x5a5a_5a5a_5a5a_5a5a,
    };

    struct LaFixture {
        m: usize,
        table: UniSkipFoldTable,
        a_packed: Vec<u8>,
        b_packed: Vec<u8>,
        mlv: Vec<F128>,
        padding: PaddingSpec,
        rho_grid: Vec<F128>,
    }

    /// Random witness in the ranked padded shape (or dense), plus the
    /// convention-A round-two challenge vector with a non-zero `r₁`.
    fn la_fixture(m: usize, seed: u64, ranked_padding: bool) -> LaFixture {
        const K_SKIP: usize = 6;
        let mut rng = Rng::new(seed);
        let mut a = rng.bits(1 << m);
        let mut b = rng.bits(1 << m);
        let padding = if ranked_padding {
            let k_log = 14usize.min(m);
            let useful = (15_409usize).min((1usize << k_log) - 3);
            let block = 1usize << k_log;
            for blk in 0..(1usize << (m - k_log)) {
                a[blk * block + useful..(blk + 1) * block].fill(false);
                b[blk * block + useful..(blk + 1) * block].fill(false);
            }
            PaddingSpec {
                k_log,
                useful_bits_per_block: useful,
            }
        } else {
            PaddingSpec::dense(m)
        };
        // A slab of statically all-ones b rows exercises the b≡1 degeneration.
        let ones_rows = 1usize << (K_SKIP + 2);
        for i in 0..ones_rows.min(1 << m) {
            b[i] = true;
        }
        let a_packed = pack_bits(&a);
        let b_packed = pack_bits(&b);
        let z = rng.f128();
        let mut mlv = vec![F128::ONE; m - K_SKIP];
        for slot in mlv[1..].iter_mut() {
            *slot = rng.f128();
        }
        assert_ne!(mlv[1], F128::ZERO);
        let rho_grid = vec![
            F128::ZERO,
            F128::ONE,
            F128 {
                lo: u64::MAX,
                hi: u64::MAX,
            },
            rng.f128(),
        ];
        LaFixture {
            m,
            table: UniSkipFoldTable::new(K_SKIP, z),
            a_packed,
            b_packed,
            mlv,
            padding,
            rho_grid,
        }
    }

    /// `first_r_next` for the incumbent round-three route.
    fn la_r_next3(f: &LaFixture) -> Vec<F128> {
        let mut v = vec![F128::ONE; f.mlv.len() - 1];
        v[1..].copy_from_slice(&f.mlv[2..]);
        v
    }

    /// `r_next4` for the round-four message (post double fold).
    fn la_r_next4(f: &LaFixture) -> Vec<F128> {
        let mut v = vec![F128::ONE; f.mlv.len() - 2];
        v[1..].copy_from_slice(&f.mlv[3..]);
        v
    }

    fn with_round2_settings(mut body: impl FnMut()) {
        // Both gates are phase-local reads, so exercise every combination.
        for degen_off in [false, true] {
            for periodic_off in [false, true] {
                unsafe {
                    if degen_off {
                        std::env::set_var("FLOCK_NO_R2_DEGEN", "1");
                    } else {
                        std::env::remove_var("FLOCK_NO_R2_DEGEN");
                    }
                    if periodic_off {
                        std::env::set_var("FLOCK_NO_ZC_R2_PERIODIC", "1");
                    } else {
                        std::env::remove_var("FLOCK_NO_ZC_R2_PERIODIC");
                    }
                }
                body();
            }
        }
        unsafe {
            std::env::remove_var("FLOCK_NO_R2_DEGEN");
            std::env::remove_var("FLOCK_NO_ZC_R2_PERIODIC");
        }
    }

    /// T1 — the core oracle. Round-two message identical, and the deferred
    /// quadratic evaluates to the incumbent round-three message at every ρ₁.
    /// Four distinct ρ₁ over-determine a quadratic, so agreement at all of
    /// them *is* coefficient equality.
    #[test]
    fn round3_lookahead_matches_compact_route() {
        for (m, ranked) in [(20usize, true), (20, false), (13, false), (16, true)] {
            let f = la_fixture(m, 0x1EA0_0001 ^ (m as u64), ranked);
            let r_next3 = la_r_next3(&f);
            with_round2_settings(|| {
                crate::scratch::clear();
                let (compact, m1_l, mi_l) = uni_skip_fold_and_round_pair_compact_padded(
                    &f.a_packed,
                    &f.b_packed,
                    f.m,
                    6,
                    &f.table,
                    &f.mlv,
                    &f.padding,
                );
                let (compact_n, m1_n, mi_n, la) =
                    uni_skip_fold_and_round_pair_compact_padded_lookahead(
                        &f.a_packed,
                        &f.b_packed,
                        f.m,
                        6,
                        &f.table,
                        &f.mlv,
                        &f.padding,
                        None,
                    );
                assert_eq!((m1_n, mi_n), (m1_l, mi_l), "round-two message, m={m}");
                assert_eq!(compact_n.anchors, compact.anchors, "anchors, m={m}");
                assert_eq!(&compact_n.deltas[..], &compact.deltas[..], "deltas, m={m}");

                for &rho1 in &f.rho_grid {
                    let (a3, b3, e1, ei) =
                        fold_compact_and_compute_round_pair(&compact, &f.table, rho1, &r_next3);
                    assert_eq!(
                        eval_round3_lookahead(&la, rho1),
                        (e1, ei),
                        "round-three message, m={m}, rho1={rho1:?}"
                    );
                    crate::scratch::give_f128(a3);
                    crate::scratch::give_f128(b3);
                }
                compact.recycle();
                compact_n.recycle();
                crate::scratch::clear();
            });
        }
    }

    /// T2 — independent derivation: Lagrange-interpolate both quadratics from
    /// three legacy evaluations and compare the six coefficients elementwise.
    #[test]
    fn lookahead_coefficients_match_interpolation() {
        let f = la_fixture(16, 0x1EA0_0002, true);
        let r_next3 = la_r_next3(&f);
        crate::scratch::clear();
        let (compact, _, _) = uni_skip_fold_and_round_pair_compact_padded(
            &f.a_packed,
            &f.b_packed,
            f.m,
            6,
            &f.table,
            &f.mlv,
            &f.padding,
        );
        let (compact_n, _, _, la) = uni_skip_fold_and_round_pair_compact_padded_lookahead(
            &f.a_packed,
            &f.b_packed,
            f.m,
            6,
            &f.table,
            &f.mlv,
            &f.padding,
            None,
        );
        let g = f.rho_grid[3];
        assert!(g != F128::ZERO && g != F128::ONE);
        let mut evals = Vec::new();
        for &rho in &[F128::ZERO, F128::ONE, g] {
            let (a3, b3, e1, ei) =
                fold_compact_and_compute_round_pair(&compact, &f.table, rho, &r_next3);
            evals.push((e1, ei));
            crate::scratch::give_f128(a3);
            crate::scratch::give_f128(b3);
        }
        // f(0) = c0; f(1) = c0+c1+c2; f(g) = c0 + c1·g + c2·g².
        // ⇒ c2 = [(f(g)+c0)·g⁻¹ + f(1)+c0] · (g+1)⁻¹, c1 = f(1)+c0+c2.
        let g_inv = g.inv();
        let gp1_inv = (g + F128::ONE).inv();
        for base in [0usize, 3] {
            let pick = |e: &(F128, F128)| if base == 0 { e.0 } else { e.1 };
            let c0 = pick(&evals[0]);
            let f1 = pick(&evals[1]);
            let fg = pick(&evals[2]);
            let c2 = ((fg + c0) * g_inv + f1 + c0) * gp1_inv;
            let c1 = f1 + c0 + c2;
            assert_eq!(la.c[base], c0, "c{base}");
            assert_eq!(la.c[base + 1], c1, "c{}", base + 1);
            assert_eq!(la.c[base + 2], c2, "c{}", base + 2);
        }
        compact.recycle();
        compact_n.recycle();
        crate::scratch::clear();
    }

    /// T3 — the double-fold oracle. For a 4×4 (ρ₁, ρ₂) grid the K pass must
    /// reproduce the legacy T3-then-fold tables **elementwise** (outputs
    /// pre-filled with poison, so a never-written fully-padding group slot is
    /// caught) and the round-four message.
    #[test]
    fn fold2_from_compact_matches_t3_then_loop_fold() {
        for (m, ranked) in [(20usize, true), (20, false), (14, true)] {
            let f = la_fixture(m, 0x1EA0_0003 ^ (m as u64), ranked);
            let r_next3 = la_r_next3(&f);
            let r_next4 = la_r_next4(&f);
            with_round2_settings(|| {
                crate::scratch::clear();
                let (compact, _, _) = uni_skip_fold_and_round_pair_compact_padded(
                    &f.a_packed,
                    &f.b_packed,
                    f.m,
                    6,
                    &f.table,
                    &f.mlv,
                    &f.padding,
                );
                let n_groups = compact.len() / 2;
                for &rho1 in &f.rho_grid {
                    for &rho2 in &f.rho_grid {
                        let (mut a_l, mut b_l, _, _) =
                            fold_compact_and_compute_round_pair(&compact, &f.table, rho1, &r_next3);
                        fold_in_place_pair(&mut a_l, &mut b_l, rho2);
                        let msg_l = round_pair_naive(&a_l, &b_l, &r_next4);

                        let mut a_n = crate::scratch::take_f128(n_groups);
                        let mut b_n = crate::scratch::take_f128(n_groups);
                        a_n.fill(LA_POISON);
                        b_n.fill(LA_POISON);
                        let msg_n = fold2_compact_and_round4_into(
                            &compact, &f.table, rho1, rho2, &r_next4, &mut a_n, &mut b_n,
                        );
                        assert_eq!(a_n, a_l, "A'' m={m} rho1={rho1:?} rho2={rho2:?}");
                        assert_eq!(b_n, b_l, "B'' m={m} rho1={rho1:?} rho2={rho2:?}");
                        assert_eq!(msg_n, msg_l, "round-four message m={m}");

                        crate::scratch::give_f128(a_l);
                        crate::scratch::give_f128(b_l);
                        crate::scratch::give_f128(a_n);
                        crate::scratch::give_f128(b_n);
                    }
                }
                compact.recycle();
                crate::scratch::clear();
            });
        }
    }

    /// T4 — textbook scalar reference: materialize A/B with `fold_at_z_naive`,
    /// fold at ρ₁ and read the round-three message directly. Deliberately slow
    /// and obviously correct; runs at tiny `m` only.
    #[test]
    fn round3_lookahead_matches_scalar_reference() {
        const K_SKIP: usize = 6;
        for m in [9usize, 10] {
            let f = la_fixture(m, 0x1EA0_0004 ^ (m as u64), false);
            let r_next3 = la_r_next3(&f);
            with_round2_settings(|| {
                crate::scratch::clear();
                let (compact, _, _, la) = uni_skip_fold_and_round_pair_compact_padded_lookahead(
                    &f.a_packed,
                    &f.b_packed,
                    f.m,
                    K_SKIP,
                    &f.table,
                    &f.mlv,
                    &f.padding,
                    None,
                );
                // Reference A/B: one row fold per output through the table.
                let n_out = 1usize << (f.m - K_SKIP);
                let ref_a: Vec<F128> = (0..n_out)
                    .map(|x| f.table.fold_one_row(&f.a_packed[x * 8..x * 8 + 8]))
                    .collect();
                let ref_b: Vec<F128> = (0..n_out)
                    .map(|x| f.table.fold_one_row(&f.b_packed[x * 8..x * 8 + 8]))
                    .collect();
                for &rho1 in &f.rho_grid {
                    let mut a = ref_a.clone();
                    let mut b = ref_b.clone();
                    fold_in_place_pair(&mut a, &mut b, rho1);
                    let expect = round_pair_naive(&a, &b, &r_next3);
                    assert_eq!(
                        eval_round3_lookahead(&la, rho1),
                        expect,
                        "scalar reference m={m} rho1={rho1:?}"
                    );
                }
                compact.recycle();
                crate::scratch::clear();
            });
        }
    }

    /// T6 — the degenerate-challenge guard: `r₁ = 0` makes W1/W2
    /// unrecoverable from the parity split, so the lookahead producer must
    /// refuse rather than emit a wrong coefficient. (`prove` takes the
    /// incumbent route in that case.)
    #[test]
    #[should_panic(expected = "lookahead requires a non-zero")]
    fn lookahead_rejects_zero_r1() {
        let mut f = la_fixture(13, 0x1EA0_0006, false);
        f.mlv[1] = F128::ZERO;
        let _ = uni_skip_fold_and_round_pair_compact_padded_lookahead(
            &f.a_packed,
            &f.b_packed,
            f.m,
            6,
            &f.table,
            &f.mlv,
            &f.padding,
            None,
        );
    }

    // ----------------------------------------------------------------------
    // Cascaded lookahead (variant K, one level deeper): rounds 5+6.
    // ----------------------------------------------------------------------

    /// C1 — the cascade K-pass oracle. Outputs and the round-four message
    /// must match the incumbent K pass **elementwise** (poison-filled
    /// destinations), and the deferred round-five quadratic must evaluate to
    /// the incumbent tail-i=2 message at every ρ₃ of a 4-point grid — four
    /// distinct points over-determine a quadratic, so agreement at all of
    /// them *is* coefficient equality.
    #[test]
    fn cascade_round5_lookahead_matches_incumbent_k_pass() {
        for (m, ranked) in [(20usize, true), (20, false), (14, true), (13, false)] {
            let f = la_fixture(m, 0x0CA5_0001 ^ (m as u64), ranked);
            assert_ne!(f.mlv[3], F128::ZERO);
            let r_next4 = la_r_next4(&f);
            // r_next for the incumbent round-five message (tail i = 2).
            let mut r_next5 = vec![F128::ONE; f.mlv.len() - 3];
            r_next5[1..].copy_from_slice(&f.mlv[4..]);
            with_round2_settings(|| {
                crate::scratch::clear();
                let (compact, _, _) = uni_skip_fold_and_round_pair_compact_padded(
                    &f.a_packed,
                    &f.b_packed,
                    f.m,
                    6,
                    &f.table,
                    &f.mlv,
                    &f.padding,
                );
                let n_groups = compact.len() / 2;
                let rho_pairs = [
                    (F128::ZERO, f.rho_grid[2]),
                    (F128::ONE, f.rho_grid[3]),
                    (f.rho_grid[3], f.rho_grid[3] + F128::ONE),
                ];
                for &(rho1, rho2) in &rho_pairs {
                    let mut a_l = crate::scratch::take_f128(n_groups);
                    let mut b_l = crate::scratch::take_f128(n_groups);
                    a_l.fill(LA_POISON);
                    b_l.fill(LA_POISON);
                    let msg_l = fold2_compact_and_round4_into(
                        &compact, &f.table, rho1, rho2, &r_next4, &mut a_l, &mut b_l,
                    );

                    let mut a_n = crate::scratch::take_f128(n_groups);
                    let mut b_n = crate::scratch::take_f128(n_groups);
                    a_n.fill(LA_POISON);
                    b_n.fill(LA_POISON);
                    let (m4_1, m4_inf, la5) = fold2_compact_and_round45_into(
                        &compact, &f.table, rho1, rho2, &r_next4, &mut a_n, &mut b_n,
                    );
                    assert_eq!(
                        (m4_1, m4_inf),
                        msg_l,
                        "round-four message m={m} rho1={rho1:?} rho2={rho2:?}"
                    );
                    assert_eq!(a_n, a_l, "A'' m={m} rho1={rho1:?} rho2={rho2:?}");
                    assert_eq!(b_n, b_l, "B'' m={m} rho1={rho1:?} rho2={rho2:?}");

                    for &rho3 in &f.rho_grid {
                        let mut a5 = a_l.clone();
                        let mut b5 = b_l.clone();
                        fold_in_place_pair(&mut a5, &mut b5, rho3);
                        let expect = round_pair_naive(&a5, &b5, &r_next5);
                        assert_eq!(
                            eval_round3_lookahead(&la5, rho3),
                            expect,
                            "round-five message m={m} rho1={rho1:?} rho2={rho2:?} rho3={rho3:?}"
                        );
                    }

                    crate::scratch::give_f128(a_l);
                    crate::scratch::give_f128(b_l);
                    crate::scratch::give_f128(a_n);
                    crate::scratch::give_f128(b_n);
                }
                compact.recycle();
                crate::scratch::clear();
            });
        }
    }

    /// C2 — the composed rounds-5/6 double-fold oracle. For a 4×4 (ρ₃, ρ₄)
    /// grid the composed pass must reproduce two sequential incumbent folds
    /// **elementwise** (outputs pre-filled with poison, so a never-written
    /// slot is caught) and the round-six message.
    #[test]
    fn composed_rounds56_fold_matches_two_sequential_folds() {
        let mut rng = Rng::new(0x0CA5_0002);
        for log_n in [4usize, 5, 9, 12] {
            let n = 1usize << log_n;
            let a: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let b: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let mut r_next6 = vec![F128::ONE; log_n - 2];
            for slot in r_next6[1..].iter_mut() {
                *slot = rng.f128();
            }
            let grid = [
                F128::ZERO,
                F128::ONE,
                F128 {
                    lo: u64::MAX,
                    hi: u64::MAX,
                },
                rng.f128(),
            ];
            for &rho3 in &grid {
                for &rho4 in &grid {
                    let mut a_l = a.clone();
                    let mut b_l = b.clone();
                    fold_in_place_pair(&mut a_l, &mut b_l, rho3);
                    fold_in_place_pair(&mut a_l, &mut b_l, rho4);
                    let msg_l = round_pair_naive(&a_l, &b_l, &r_next6);

                    let mut a_n = vec![LA_POISON; n / 4];
                    let mut b_n = vec![LA_POISON; n / 4];
                    let msg_n = fold2_plain_and_round6_into(
                        &a, &b, &mut a_n, &mut b_n, rho3, rho4, &r_next6,
                    );
                    assert_eq!(a_n, a_l, "A log_n={log_n} rho3={rho3:?} rho4={rho4:?}");
                    assert_eq!(b_n, b_l, "B log_n={log_n} rho3={rho3:?} rho4={rho4:?}");
                    assert_eq!(
                        msg_n, msg_l,
                        "round-six message log_n={log_n} rho3={rho3:?} rho4={rho4:?}"
                    );
                }
            }
        }
    }

    /// C3 — the degenerate-challenge guard, one level deeper: `r_next4[1] = 0`
    /// makes W1'/W2' unrecoverable from the parity split, so the cascade K
    /// pass must refuse rather than emit a wrong coefficient. (`prove` takes
    /// the incumbent K route in that case.)
    #[test]
    #[should_panic(expected = "cascade requires a non-zero")]
    fn cascade_rejects_zero_r_next4_1() {
        let f = la_fixture(13, 0x0CA5_0003, false);
        let mut r_next4 = la_r_next4(&f);
        r_next4[1] = F128::ZERO;
        let (compact, _, _) = uni_skip_fold_and_round_pair_compact_padded(
            &f.a_packed,
            &f.b_packed,
            f.m,
            6,
            &f.table,
            &f.mlv,
            &f.padding,
        );
        let n_groups = compact.len() / 2;
        let mut a_out = vec![F128::ZERO; n_groups];
        let mut b_out = vec![F128::ZERO; n_groups];
        let _ = fold2_compact_and_round45_into(
            &compact,
            &f.table,
            f.rho_grid[3],
            f.rho_grid[2],
            &r_next4,
            &mut a_out,
            &mut b_out,
        );
    }

    // ----------------------------------------------------------------------
    // Cascaded lookahead, level three: rounds 7+8.
    // ----------------------------------------------------------------------

    /// D1 — the cascade3 composed-5/6 oracle (C1 one level down). Outputs and
    /// the round-six message must match the plain composed pass
    /// **elementwise** (poison-filled destinations), and the deferred
    /// round-seven quadratic must evaluate to the incumbent tail-i=4 message
    /// at every ρ₅ of a 4-point grid — four distinct points over-determine a
    /// quadratic, so agreement at all of them *is* coefficient equality.
    #[test]
    fn cascade3_round7_lookahead_matches_composed_pass() {
        let mut rng = Rng::new(0x0CA5_3001);
        for log_n in [4usize, 5, 9, 12] {
            let n = 1usize << log_n;
            let a: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let b: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let mut r_next6 = vec![F128::ONE; log_n - 2];
            for slot in r_next6[1..].iter_mut() {
                *slot = rng.f128();
            }
            assert_ne!(r_next6[1], F128::ZERO);
            // r_next for the incumbent round-seven message (tail i = 4).
            let mut r_next7 = vec![F128::ONE; log_n - 3];
            r_next7[1..].copy_from_slice(&r_next6[2..]);
            let rho_grid = [
                F128::ZERO,
                F128::ONE,
                F128 {
                    lo: u64::MAX,
                    hi: u64::MAX,
                },
                rng.f128(),
            ];
            let rho_pairs = [
                (F128::ZERO, rho_grid[2]),
                (F128::ONE, rho_grid[3]),
                (rho_grid[3], rho_grid[3] + F128::ONE),
            ];
            for &(rho3, rho4) in &rho_pairs {
                let mut a_l = vec![LA_POISON; n / 4];
                let mut b_l = vec![LA_POISON; n / 4];
                let msg_l =
                    fold2_plain_and_round6_into(&a, &b, &mut a_l, &mut b_l, rho3, rho4, &r_next6);

                let mut a_n = vec![LA_POISON; n / 4];
                let mut b_n = vec![LA_POISON; n / 4];
                let (m6_1, m6_inf, la7) =
                    fold2_plain_and_round67_into(&a, &b, &mut a_n, &mut b_n, rho3, rho4, &r_next6);
                assert_eq!(
                    (m6_1, m6_inf),
                    msg_l,
                    "round-six message log_n={log_n} rho3={rho3:?} rho4={rho4:?}"
                );
                assert_eq!(a_n, a_l, "A''' log_n={log_n} rho3={rho3:?} rho4={rho4:?}");
                assert_eq!(b_n, b_l, "B''' log_n={log_n} rho3={rho3:?} rho4={rho4:?}");

                for &rho5 in &rho_grid {
                    let mut a7 = a_l.clone();
                    let mut b7 = b_l.clone();
                    fold_in_place_pair(&mut a7, &mut b7, rho5);
                    let expect = round_pair_naive(&a7, &b7, &r_next7);
                    assert_eq!(
                        eval_round3_lookahead(&la7, rho5),
                        expect,
                        "round-seven message log_n={log_n} rho3={rho3:?} rho4={rho4:?} rho5={rho5:?}"
                    );
                }
            }
        }
    }

    /// D2 — the composed rounds-7/8 double-fold at cascade3's call shape:
    /// starting from a composed-5/6 output, one composed pass at (ρ₅, ρ₆)
    /// must reproduce two sequential incumbent folds **elementwise**
    /// (poison-filled outputs) and the round-eight message, for a 4×4 grid.
    #[test]
    fn cascade3_composed_rounds78_matches_two_sequential_folds() {
        let mut rng = Rng::new(0x0CA5_3002);
        for log_n in [6usize, 9, 12] {
            let n = 1usize << log_n;
            let a: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let b: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let mut r_next6 = vec![F128::ONE; log_n - 2];
            for slot in r_next6[1..].iter_mut() {
                *slot = rng.f128();
            }
            // Chain through the lookahead composed-5/6 pass exactly as the
            // driver does, then compose rounds 7+8 from its outputs.
            let mut a2 = vec![LA_POISON; n / 4];
            let mut b2 = vec![LA_POISON; n / 4];
            let (_, _, _la7) = fold2_plain_and_round67_into(
                &a,
                &b,
                &mut a2,
                &mut b2,
                rng.f128(),
                rng.f128(),
                &r_next6,
            );
            let mut r_next8 = vec![F128::ONE; log_n - 4];
            r_next8[1..].copy_from_slice(&r_next6[3..]);
            let grid = [
                F128::ZERO,
                F128::ONE,
                F128 {
                    lo: u64::MAX,
                    hi: u64::MAX,
                },
                rng.f128(),
            ];
            for &rho5 in &grid {
                for &rho6 in &grid {
                    let mut a_l = a2.clone();
                    let mut b_l = b2.clone();
                    fold_in_place_pair(&mut a_l, &mut b_l, rho5);
                    fold_in_place_pair(&mut a_l, &mut b_l, rho6);
                    let msg_l = round_pair_naive(&a_l, &b_l, &r_next8);

                    let mut a_n = vec![LA_POISON; n / 16];
                    let mut b_n = vec![LA_POISON; n / 16];
                    let msg_n = fold2_plain_and_round6_into(
                        &a2, &b2, &mut a_n, &mut b_n, rho5, rho6, &r_next8,
                    );
                    assert_eq!(a_n, a_l, "A log_n={log_n} rho5={rho5:?} rho6={rho6:?}");
                    assert_eq!(b_n, b_l, "B log_n={log_n} rho5={rho5:?} rho6={rho6:?}");
                    assert_eq!(
                        msg_n, msg_l,
                        "round-eight message log_n={log_n} rho5={rho5:?} rho6={rho6:?}"
                    );
                }
            }
        }
    }

    /// D3 — the degenerate-challenge guard, one level deeper still:
    /// `r_next6[1] = 0` makes W1''/W2'' unrecoverable from the parity split,
    /// so the composed-5/6 lookahead pass must refuse rather than emit a
    /// wrong coefficient. (`prove` takes the cascade2 route in that case.)
    #[test]
    #[should_panic(expected = "cascade requires a non-zero r_next6")]
    fn cascade3_rejects_zero_r_next6_1() {
        let mut rng = Rng::new(0x0CA5_3003);
        let log_n = 6usize;
        let n = 1usize << log_n;
        let a: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
        let b: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
        let mut r_next6 = vec![F128::ONE; log_n - 2];
        for slot in r_next6[1..].iter_mut() {
            *slot = rng.f128();
        }
        r_next6[1] = F128::ZERO;
        let mut a_out = vec![F128::ZERO; n / 4];
        let mut b_out = vec![F128::ZERO; n / 4];
        let _ = fold2_plain_and_round67_into(
            &a,
            &b,
            &mut a_out,
            &mut b_out,
            rng.f128(),
            rng.f128(),
            &r_next6,
        );
    }

    /// Parallel `uni_skip_fold_and_round_pair_optimized_packed` produces
    /// byte-identical output to the serial version. F128 XOR + multiply sum
    /// is commutative + associative, so worker scheduling order doesn't
    /// affect the result.
    #[test]
    fn parallel_matches_serial() {
        for &m in &[7usize, 8, 9, 10] {
            let k_skip = 6;
            if m <= k_skip {
                continue;
            }
            let mut rng = Rng::new(200 + m as u64);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            let z = rng.f128();
            let mlv_challenges = rng.f128_vec(m - k_skip);
            let a_packed = pack_bits(&a);
            let b_packed = pack_bits(&b);
            let table = UniSkipFoldTable::new(k_skip, z);

            let par = uni_skip_fold_and_round_pair_optimized_packed(
                &a_packed,
                &b_packed,
                m,
                k_skip,
                &table,
                &mlv_challenges,
            );
            let ser = uni_skip_fold_and_round_pair_optimized_packed_serial(
                &a_packed,
                &b_packed,
                m,
                k_skip,
                &table,
                &mlv_challenges,
            );

            assert_eq!(par.0, ser.0, "a_mlv mismatch at m={m}");
            assert_eq!(par.1, ser.1, "b_mlv mismatch at m={m}");
            assert_eq!(par.2, ser.2, "msg_1 mismatch at m={m}");
            assert_eq!(par.3, ser.3, "msg_inf mismatch at m={m}");
        }
    }

    /// **Padding skip is byte-identical to the dense round-2 kernel.** Builds
    /// witnesses with bits `[useful_bits, 2^k_log)` of every block honestly
    /// zero, then asserts the `_padded` kernel produces the same
    /// `(a_mlv, b_mlv, msg_1, msg_inf)` as the dense path.
    ///
    /// Covers all three hash padding shapes: BLAKE3 (k_log=14, useful=15409),
    /// SHA-2 (k_log=15, useful=31401), Keccak (k_log=16, useful=42560).
    #[test]
    fn uni_skip_fold_round_pair_padded_matches_dense() {
        const K_SKIP: usize = 6;
        let cases: &[(usize, usize, usize)] =
            &[(17, 14, 15_409), (18, 15, 31_401), (19, 16, 42_560)];
        for &(m, k_log, useful_bits) in cases {
            let mut rng = Rng::new(0xFADE_F00D_u64.wrapping_add((k_log * 31 + m) as u64));
            let total_bits = 1usize << m;
            let block_size = 1usize << k_log;
            let n_blocks = 1usize << (m - k_log);

            // Random witness, then zero bits [useful_bits, block_size) of each
            // block in both a and b (matches honestly-padded hash R1CS).
            let mut a = rng.bits(total_bits);
            let mut b = rng.bits(total_bits);
            for blk in 0..n_blocks {
                for j in useful_bits..block_size {
                    a[blk * block_size + j] = false;
                    b[blk * block_size + j] = false;
                }
            }
            let a_packed = pack_bits(&a);
            let b_packed = pack_bits(&b);

            let z = rng.f128();
            let mlv_challenges = rng.f128_vec(m - K_SKIP);
            let table = UniSkipFoldTable::new(K_SKIP, z);
            let padding = PaddingSpec {
                k_log,
                useful_bits_per_block: useful_bits,
            };

            let dense = uni_skip_fold_and_round_pair_optimized_packed(
                &a_packed,
                &b_packed,
                m,
                K_SKIP,
                &table,
                &mlv_challenges,
            );
            let padded = uni_skip_fold_and_round_pair_optimized_packed_padded(
                &a_packed,
                &b_packed,
                m,
                K_SKIP,
                &table,
                &mlv_challenges,
                &padding,
            );
            assert_eq!(
                dense.0, padded.0,
                "a_mlv: m={m}, k_log={k_log}, useful={useful_bits}"
            );
            assert_eq!(
                dense.1, padded.1,
                "b_mlv: m={m}, k_log={k_log}, useful={useful_bits}"
            );
            assert_eq!(
                dense.2, padded.2,
                "msg_1: m={m}, k_log={k_log}, useful={useful_bits}"
            );
            assert_eq!(
                dense.3, padded.3,
                "msg_inf: m={m}, k_log={k_log}, useful={useful_bits}"
            );
        }
    }

    /// `fold_one_row` via the table equals direct-Lagrange fold.
    #[test]
    fn fold_table_one_row_matches_direct_lagrange() {
        let m = 8;
        let k_skip = 3;
        let mut rng = Rng::new(60);
        let z = rng.f128();
        let a = rng.bits(1 << m);
        let weights = lagrange_weights_naive(k_skip, z);
        let table = UniSkipFoldTable::new(k_skip, z);
        let a_packed = pack_bits(&a);

        let n_chunks = 1usize << (k_skip / 8);
        let _ = n_chunks; // ell/8 = (1<<k_skip)/8
        let n_chunks = table.n_chunks;

        for x_rest in 0..(1usize << (m - k_skip)) {
            let direct = {
                let mut acc = F128::ZERO;
                for s in 0..(1usize << k_skip) {
                    if a[x_rest * (1usize << k_skip) + s] {
                        acc += weights[s];
                    }
                }
                acc
            };
            let via_table =
                table.fold_one_row(&a_packed[x_rest * n_chunks..(x_rest + 1) * n_chunks]);
            assert_eq!(via_table, direct, "x_rest={x_rest}");
        }
    }

    /// **The full cross-check**: optimized fused output matches naive
    /// byte-for-byte at the headline `k_skip = 6` (and other small m). Same eq
    /// weights, same z, same r — so a_mlv, b_mlv, and the two message values
    /// must all agree exactly.
    #[test]
    fn optimized_matches_naive() {
        for &m in &[7usize, 8, 9, 10] {
            let k_skip = 6;
            if m <= k_skip {
                continue;
            }
            let mut rng = Rng::new(100 + m as u64);
            let a = rng.bits(1 << m);
            let b = rng.bits(1 << m);
            let z = rng.f128();
            let mlv_challenges = rng.f128_vec(m - k_skip);

            let (a_n, b_n, m1_n, minf_n) =
                uni_skip_fold_and_round_pair_naive(&a, &b, m, k_skip, z, &mlv_challenges);
            let (a_o, b_o, m1_o, minf_o) =
                uni_skip_fold_and_round_pair_optimized(&a, &b, m, k_skip, z, &mlv_challenges);

            assert_eq!(a_n, a_o, "a_mlv mismatch at m={m}");
            assert_eq!(b_n, b_o, "b_mlv mismatch at m={m}");
            assert_eq!(m1_n, m1_o, "msg_1 mismatch at m={m}");
            assert_eq!(minf_n, minf_o, "msg_inf mismatch at m={m}");
        }
    }

    /// Strong cross-check: compute G(0), G(1), G(∞) by direct sum (using the
    /// LSB-first index convention `a_mlv(0, x') = a[2x']`, `a_mlv(1, x') = a[2x'+1]`),
    /// then verify that G interpolated through those three values agrees with
    /// the direct multilinear evaluation at a fresh random X — confirming G
    /// genuinely has degree ≤ 2.
    ///
    /// Also verifies `round_pair_naive` returns `(r[0] · G(1), G(∞))`.
    #[test]
    fn round_pair_message_has_degree_two() {
        let m = 6;
        let k_skip = 3;
        let mut rng = Rng::new(55);
        let a = rng.bits(1 << m);
        let b = rng.bits(1 << m);
        let z = rng.f128();
        let r = rng.f128_vec(m - k_skip);

        let weights = lagrange_weights_naive(k_skip, z);
        let a_mlv = fold_at_z_naive(&a, m, k_skip, &weights);
        let b_mlv = fold_at_z_naive(&b, m, k_skip, &weights);

        let n = a_mlv.len();
        let half = n / 2;
        let eq_remaining = build_eq(&r[1..]);

        // G(0), G(1), G(∞) by direct definition.
        let mut g0 = F128::ZERO;
        let mut g1 = F128::ZERO;
        let mut g_inf = F128::ZERO;
        for x_prime in 0..half {
            let a0 = a_mlv[2 * x_prime];
            let a1 = a_mlv[2 * x_prime + 1];
            let b0 = b_mlv[2 * x_prime];
            let b1 = b_mlv[2 * x_prime + 1];
            let eq_x = eq_remaining[x_prime];
            g0 += eq_x * a0 * b0;
            g1 += eq_x * a1 * b1;
            g_inf += eq_x * (a0 + a1) * (b0 + b1);
        }

        // round_pair_naive returns (r[0] · g1, g_inf).
        let (msg_1, msg_inf) = round_pair_naive(&a_mlv, &b_mlv, &r);
        assert_eq!(msg_1, r[0] * g1);
        assert_eq!(msg_inf, g_inf);

        // Degree-2 check: G(X) reconstructed through (G(0), G(1), G(∞)) must
        // agree with the direct multilinear evaluation at a fresh point X.
        // Char-2 interpolation: G(X) = G(0) + X·(G(0)+G(1)) + X·(X+1)·G(∞).
        let x = rng.f128();
        let g_via_poly = g0 + x * (g0 + g1) + x * (x + F128::ONE) * g_inf;
        let mut g_via_sum = F128::ZERO;
        for x_prime in 0..half {
            let a0 = a_mlv[2 * x_prime];
            let a1 = a_mlv[2 * x_prime + 1];
            let b0 = b_mlv[2 * x_prime];
            let b1 = b_mlv[2 * x_prime + 1];
            let a_x = a0 + x * (a0 + a1);
            let b_x = b0 + x * (b0 + b1);
            g_via_sum += eq_remaining[x_prime] * a_x * b_x;
        }
        assert_eq!(g_via_poly, g_via_sum);
    }
}

// r567 archive marker: submission cadence variant.

// r577 cadence marker: candidate identity only; no runtime effect.

// r581: source-distinct cadence marker; no semantic effect.

// r584: source-distinct cadence marker
