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

/// Pure selector for the Apple/AES inverse-reuse micro-candidate. Only the
/// literal value `"1"` disables it; every other value keeps the candidate on.
#[inline(always)]
fn apple_inverse_micro_selected(platform_supported: bool, kill: Option<&str>) -> bool {
    platform_supported && kill != Some("1")
}

#[cfg(all(target_os = "macos", target_arch = "aarch64", target_feature = "aes"))]
#[inline]
fn lookahead_inverse_reuse_enabled() -> bool {
    static ENABLED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        let kill = std::env::var("FLOCK_NO_ZC_LOOKAHEAD_INV_REUSE").ok();
        apple_inverse_micro_selected(true, kill.as_deref())
    });
    *ENABLED
}

#[cfg(not(all(target_os = "macos", target_arch = "aarch64", target_feature = "aes")))]
#[inline(always)]
fn lookahead_inverse_reuse_enabled() -> bool {
    false
}

/// Compute the odd/even eq rescale and, on the candidate arm, retain the same
/// inverse for the post-join lookahead normalization. The control arm returns
/// `None`, so its second inversion remains at the incumbent post-join site.
#[inline]
fn lookahead_kappa_and_retained_inv(r: F128, reuse: bool) -> (F128, Option<F128>) {
    if reuse {
        let r_inv = lookahead_point_inv(r);
        // Characteristic two: (1+r)/r = 1 + r^-1.
        (F128::ONE + r_inv, Some(r_inv))
    } else {
        ((F128::ONE + r) * lookahead_point_inv(r), None)
    }
}

/// Number of `(r, r^-1)` pairs [`lookahead_point_inv`] keeps.
const LOOKAHEAD_INV_SLOTS: usize = 4;

/// `r.inv()`. The lookahead passes invert fixed protocol points of `r`
/// (`r[k_skip + 1]` and the medium points), the same values on every prove,
/// so each slot keeps one point together with its inverse and a later call
/// with an equal point takes the kept inverse (the value `inv` returns
/// again). A different point is inverted as before; it fills a free slot,
/// and once all slots hold other points it is inverted per call.
/// `FLOCK_NO_RECOMPUTE_CACHE` (any value): inverted on every call.
#[inline]
fn lookahead_point_inv(r: F128) -> F128 {
    if !crate::recompute_cache_enabled() {
        return r.inv();
    }
    static KEPT: [std::sync::OnceLock<(F128, F128)>; LOOKAHEAD_INV_SLOTS] =
        [const { std::sync::OnceLock::new() }; LOOKAHEAD_INV_SLOTS];
    for slot in &KEPT {
        match slot.get() {
            Some(&(point, inv)) if point == r => return inv,
            Some(_) => {}
            None => {
                // Another thread may fill this slot first, with its point.
                let &(point, inv) = slot.get_or_init(|| (r, r.inv()));
                if point == r {
                    return inv;
                }
            }
        }
    }
    r.inv()
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use kernels::aarch64::fold_compact_chunk_neon_reconstruct_only_8;
#[cfg(all(target_arch = "aarch64", test))]
use kernels::aarch64::fold_one_row_neon_unchecked_8;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use kernels::aarch64::fold_round2_compact_chunk_neon_anchors_only_8;
#[cfg(target_arch = "aarch64")]
use kernels::aarch64::{
    fold_and_message_aarch64, fold_compact_chunk_neon_unchecked_8, fold_compact_stream_chunk_neon,
    fold_round2_chunk_neon_unchecked_8, fold_round2_compact_chunk_neon_horner_8,
    fold_round2_compact_chunk_neon_lookahead_8, fold_round2_compact_chunk_neon_unchecked_8,
    fold_round2_compact_stream_chunk_neon, fold2_and_message_aarch64,
    fold2_and_message_lookahead_aarch64, fold2_compact_and_round4_chunk_neon_8,
    fold2_compact_and_round45_chunk_neon_8,
};
#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
use kernels::aarch64::{
    fold2_compact_and_round45_chunk_neon_8_fused, fold2_compact_and_round45_chunk_neon_8_fused_r1,
};
#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
use kernels::aarch64::fold_round2_compact_chunk_neon_prescaled_8;
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

/// Kill switch for the peeled first group of the round-two Horner kernel:
/// `FLOCK_NO_R2_FIRST_SHIFT_SKIP=1` restores the uniform loop that shifts the
/// zero accumulators in front of every window's first group. Same outputs.
#[cfg(target_arch = "aarch64")]
fn r2_first_shift_skip_enabled() -> bool {
    std::env::var_os("FLOCK_OFF_R2_FIRST_SHIFT_SKIP").is_none_or(|v| v != *"1")
}

#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
fn r2_prescale_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_OFF_R2_PRESCALE").is_none())
}

/// Kill switch for the fused cascade K kernel
/// (`fold2_compact_and_round45_chunk_neon_8_fused`): `FLOCK_NO_ZC_K_FUSED=1`
/// restores the one-body compact kernel. Same stores and slots either way.
#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
fn cascade_k_fused_enabled() -> bool {
    std::env::var_os("FLOCK_OFF_ZC_K_FUSED").is_none_or(|v| v != *"1")
}

#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
fn zc_k_one_table_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_OFF_ZC_K_ONE_TABLE").is_none())
}

/// Kill switch for the ranked BLAKE3 periodic-padding schedule in round two:
/// `FLOCK_NO_ZC_R2_PERIODIC=1` restores the generic per-group mask checks.
/// The specialized kernel only changes loop control; every useful, boundary,
/// and padded group executes the same arithmetic and stores as the oracle.
#[cfg(target_arch = "aarch64")]
fn r2_periodic_padding_enabled() -> bool {
    std::env::var_os("FLOCK_NO_ZC_R2_PERIODIC").is_none_or(|v| v != *"1")
}

/// Kill switch for the round-two dead-pair skip: `FLOCK_NO_R2_DEADPAIR_SKIP=1`
/// restores the zero-filled padding slots and the cascade K pass's blind
/// read of them. Bit-identical either way: the skipped slots hold structural
/// zeros the K pass substitutes exactly. Read once per phase call.
#[cfg(target_arch = "aarch64")]
fn r2_dead_pair_skip_enabled() -> bool {
    std::env::var_os("FLOCK_NO_R2_DEADPAIR_SKIP").is_none_or(|v| v != *"1")
}

/// Kill switch for the tail-compacted round-two compact state
/// (`FLOCK_NO_ZC_TAIL_COMPACT`, any value): with the dead-pair skip, round two
/// lays each 128-pair block out without slots for its dead pairs (anchors 244
/// instead of 256 per block, deltas 1984 instead of 2048 bytes), so the K
/// pass reads a gap-free stream (holes inside a stream are still mostly
/// fetched by the prefetcher). Same values in the live slots, same K
/// outputs and messages. Read once per process.
pub(crate) fn zc_r2_tail_compact_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_ZC_TAIL_COMPACT").is_none())
}

/// Kill switch for the head-compacted round-two compact state
/// (`FLOCK_NO_ZC_HEAD_COMPACT`, any value): with the tail-compacted layout,
/// pairs 0..7 of every 128-pair block, whose b rows are all ones at the
/// ranked shape, keep only their a anchors and a deltas (2 + 1 lines instead
/// of 4 + 2): 48 MiB fewer round-two writes and K reads per ranked proof.
/// A head pair whose b rows are not all ones makes round two rewrite the
/// whole state in the tail-compacted layout. Same values, same messages.
/// Read once per process.
pub(crate) fn zc_r2_head_compact_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_ZC_HEAD_COMPACT").is_none())
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
/// The default prefix/suffix pass takes linear time. Both node sets are
/// translates of the same additive subspace and share one denominator.
/// `FLOCK_NO_LAGRANGE_FAST=1` restores the reference implementation.
pub fn lagrange_weights_naive(k_skip: usize, z: F128) -> Vec<F128> {
    let ell = 1usize << k_skip;
    assert!(ell <= 256, "k_skip > 8 would exceed PHI_8_TABLE");
    if lagrange_fast_enabled() {
        return lagrange_weights_fast(k_skip, z, 0);
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

/// AES-field encodings of (∏_{t=1}^{2^k-1} t)^(-1), including the empty
/// product for k=0. At node i, adding the other nodes enumerates exactly
/// these nonzero differences: j ↦ i XOR j permutes the subspace and a
/// common coset offset cancels. The φ_8 embedding preserves the product.
const LAGRANGE_DEN_INV_F8: [u8; 9] = [0x01, 0x01, 0x7b, 0x45, 0x3c, 0x42, 0x6c, 0x03, 0x01];

fn lagrange_weights_fast(k_skip: usize, z: F128, node_offset: usize) -> Vec<F128> {
    let ell = 1usize << k_skip;
    let nodes = &PHI_8_TABLE[node_offset..node_offset + ell];
    let mut weights = Vec::with_capacity(ell);
    let mut prefix = PHI_8_TABLE[LAGRANGE_DEN_INV_F8[k_skip] as usize];
    weights.push(prefix);
    for &node in &nodes[..ell - 1] {
        prefix *= z + node;
        weights.push(prefix);
    }
    // weights[i] already includes the inverse denominator and every factor
    // left of i. Multiply by each right suffix, which may itself be zero.
    let mut suffix = F128::ONE;
    for i in (1..ell).rev() {
        suffix *= z + nodes[i];
        weights[i - 1] *= suffix;
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
        return lagrange_weights_fast(k_skip, z, ell);
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
    let log_n = a_mlv.len().trailing_zeros() as usize;
    assert_eq!(r.len(), log_n);
    round_pair_naive_with_eq(a_mlv, b_mlv, r[0], &build_eq(&r[1..]))
}

/// [`round_pair_naive`] with `eq_remaining = build_eq(&r[1..])` supplied
/// (any table holding the same values) and `r_0 = r[0]`.
pub fn round_pair_naive_with_eq(
    a_mlv: &[F128],
    b_mlv: &[F128],
    r_0: F128,
    eq_remaining: &[F128],
) -> (F128, F128) {
    let n = a_mlv.len();
    assert_eq!(b_mlv.len(), n);
    assert!(n.is_power_of_two() && n >= 2);
    let half = n / 2;
    assert_eq!(eq_remaining.len(), half);

    // Reduction is F2-linear, so reduce each sum only after the loop.
    let mut g_one = F256Unreduced::ZERO;
    let mut g_inf = F256Unreduced::ZERO;
    for x_prime in 0..half {
        let a0 = a_mlv[2 * x_prime];
        let a1 = a_mlv[2 * x_prime + 1];
        let b0 = b_mlv[2 * x_prime];
        let b1 = b_mlv[2 * x_prime + 1];
        let eq_x = eq_remaining[x_prime];
        g_one ^= eq_x.mul_unreduced(a1 * b1);
        // Char-2: (a_1 − a_0)(b_1 − b_0) = (a_0 + a_1)(b_0 + b_1).
        g_inf ^= eq_x.mul_unreduced((a0 + a1) * (b0 + b1));
    }
    (r_0 * g_one.reduce(), g_inf.reduce())
}

pub fn fold_in_place_and_round_pair_with_eq(
    a: &mut Vec<F128>,
    b: &mut Vec<F128>,
    challenge: F128,
    r_0: F128,
    eq_remaining: &[F128],
) -> (F128, F128) {
    let n = a.len();
    assert_eq!(b.len(), n);
    assert!(n.is_power_of_two() && n >= 4);
    let quarter = n / 4;
    assert_eq!(eq_remaining.len(), quarter);

    let mut g_one = F256Unreduced::ZERO;
    let mut g_inf = F256Unreduced::ZERO;
    for x_prime in 0..quarter {
        let i = 4 * x_prime;
        let a0 = a[i] + challenge * (a[i + 1] + a[i]);
        let a1 = a[i + 2] + challenge * (a[i + 3] + a[i + 2]);
        let b0 = b[i] + challenge * (b[i + 1] + b[i]);
        let b1 = b[i + 2] + challenge * (b[i + 3] + b[i + 2]);
        a[2 * x_prime] = a0;
        a[2 * x_prime + 1] = a1;
        b[2 * x_prime] = b0;
        b[2 * x_prime + 1] = b1;
        let eq_x = eq_remaining[x_prime];
        g_one ^= eq_x.mul_unreduced(a1 * b1);
        g_inf ^= eq_x.mul_unreduced((a0 + a1) * (b0 + b1));
    }
    a.truncate(2 * quarter);
    b.truncate(2 * quarter);
    (r_0 * g_one.reduce(), g_inf.reduce())
}

/// The equality table of a challenge suffix one shorter: `eq(r[1..])[x] =
/// eq(r)[2x] + eq(r)[2x + 1]`, since `eq(r_0, 0) + eq(r_0, 1) = 1` exactly.
/// One XOR per entry where a fresh table takes one product per entry.
pub fn eq_drop_first(eq: &[F128]) -> Vec<F128> {
    eq.chunks_exact(2).map(|p| p[0] + p[1]).collect()
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
    ///
    /// Scaling only the 64 one-hot basis entries is exact: each bank's table
    /// is GF(2)-linear (`T[v] = T[parent] + T[lo_bit]`), and field
    /// multiplication distributes over that XOR sum. Rebuilding every other
    /// entry by XOR from the scaled basis entries therefore yields exactly
    /// `rho * T[v]` for all `v` -- the same bytes a full 2,048-product scale
    /// would write, at the cost of 64 field multiplications.
    pub(crate) fn scaled_linear(&self, rho: F128) -> Vec<F128> {
        assert_eq!(self.data.len(), self.n_chunks * 256);
        // Every bank writes its base and basis before rebuilding all other slots.
        let mut scaled = crate::alloc_uninit_f128_vec(self.data.len());
        for chunk in 0..self.n_chunks {
            let base = chunk * 256;
            scaled[base] = self.data[base];
            for bit in 0..8 {
                scaled[base + (1 << bit)] = rho * self.data[base + (1 << bit)];
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
    /// Round two skipped the structural padding pairs (pairs 121..128 of
    /// every 128-pair block): their anchor and delta slots hold stale bytes,
    /// not zeros. Only the cascade K pass
    /// ([`fold2_compact_and_round45_into`]) accepts such a state; it treats
    /// those pairs as the known zeros the dense state stores.
    pub dead_tail: bool,
    /// With `dead_tail`: the state is in the tail-compacted layout, whose
    /// 128-pair blocks have no slots for the dead pairs 122..127 (anchors,
    /// `kernels::aarch64::TC_ANCHORS_PER_BLOCK` = 244 per block) and 124..127
    /// (deltas, `TC_DELTA_BYTES_PER_BLOCK` = 1984 bytes per block), so the K
    /// pass streams 92 instead of 96 lines per block. Only the one-table
    /// fused K kernel reads it. See [`zc_r2_tail_compact_enabled`].
    pub tail_compact: bool,
    /// With `tail_compact`: the compacted blocks start this many blocks'
    /// worth of shortfall (12 anchors and 64 delta bytes per block) into the
    /// buffers, at least the block count of any producer or K chunk, so the
    /// per-block rebase of a chunk walk (`chunk - shortfall * block`) never
    /// points before the allocation. The buffers keep their dense length.
    pub tail_base_blocks: usize,
    /// With `tail_compact`: the state is head-compacted as well (pairs 0..7
    /// of every block, whose b rows are all ones at the ranked shape, keep
    /// only their a anchors and a deltas: 59 + 30 lines per block instead of
    /// 61 + 31). Their b anchor is `head_b` (the fold of the all-ones row)
    /// and their b delta zero. See [`zc_r2_head_compact_enabled`].
    pub head_compact: bool,
    /// The b anchor of every head pair of a head-compacted state.
    pub head_b: F128,
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

    /// Index in `anchors` of pair `p`'s `[a, b]` anchor slots (a live pair
    /// in the tail-compacted layout). In the head-compacted layout a head
    /// pair (`p % 128 < 8`) has only its a slot here.
    pub fn anchor_index(&self, p: usize) -> usize {
        #[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
        if self.head_compact {
            use kernels::aarch64::{HC_ANCHORS_PER_BLOCK, HC_ANCHOR_GAP, HC_ANCHOR_SHIFT, HC_HEAD_PAIRS};
            let q = p % 128;
            let inner = if q < HC_HEAD_PAIRS { q } else { 2 * q - HC_ANCHOR_SHIFT };
            return self.tail_base_blocks * HC_ANCHOR_GAP + HC_ANCHOR_SHIFT + p / 128 * HC_ANCHORS_PER_BLOCK + inner;
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
        if self.tail_compact {
            return self.tail_base_blocks * kernels::aarch64::TC_ANCHOR_GAP
                + p / 128 * kernels::aarch64::TC_ANCHORS_PER_BLOCK
                + 2 * (p % 128);
        }
        2 * p
    }

    /// Byte offset in `deltas` of pair `p`'s sixteen delta bytes (a live
    /// pair in the tail-compacted layout). In the head-compacted layout a
    /// head pair has only its eight a delta bytes here.
    pub fn delta_offset(&self, p: usize) -> usize {
        #[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
        if self.head_compact {
            use kernels::aarch64::{HC_DELTA_BYTES_PER_BLOCK, HC_DELTA_GAP, HC_DELTA_SHIFT, HC_HEAD_PAIRS};
            let q = p % 128;
            let inner = if q < HC_HEAD_PAIRS { 8 * q } else { 16 * q - HC_DELTA_SHIFT };
            return self.tail_base_blocks * HC_DELTA_GAP + HC_DELTA_SHIFT + p / 128 * HC_DELTA_BYTES_PER_BLOCK + inner;
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
        if self.tail_compact {
            return self.tail_base_blocks * kernels::aarch64::TC_DELTA_GAP
                + p / 128 * kernels::aarch64::TC_DELTA_BYTES_PER_BLOCK
                + 16 * (p % 128);
        }
        16 * p
    }

    /// Return both large buffers to their process-wide scratch pools.
    pub fn recycle(self) {
        let Self { anchors, deltas, .. } = self;
        crate::scratch::give_f128(anchors);
        deltas.recycle();
    }

    /// [`Self::recycle`] after the cascade K pass: the anchors may instead be
    /// kept, with their K view mapped, for the next prove's round two
    /// (`gpu_commit::ENV_NO_ZC_K_HANDOVER`, once per process).
    pub fn recycle_after_k(self) {
        let Self { anchors, deltas, .. } = self;
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        let anchors = {
            let bytes = zc_k_premap_bytes(anchors.len() / 2);
            match crate::gpu_commit::zc_k_handover_keep(anchors, bytes) {
                Some(anchors) => anchors,
                None => {
                    deltas.recycle();
                    return;
                }
            }
        };
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
    let deltas = deltas_backing.unwrap_or_else(|| ScratchBytes::take_full_overwrite(deltas_len));
    assert_eq!(
        deltas.len(),
        deltas_len,
        "donated compact delta backing has the wrong byte length"
    );
    let mut compact = UniSkipCompactFold {
        anchors: crate::scratch::take_f128(2 * n_pairs),
        deltas,
        dead_tail: false,
        tail_compact: false,
        tail_base_blocks: 0,
        head_compact: false,
        head_b: F128::ZERO,
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
    let deltas_base = crate::epool::SyncPtr(compact.deltas.full_overwrite_ptr());
    let partials_base = crate::epool::SyncPtr(partials.as_mut_ptr());
    crate::epool::run_hetero_chunks(hi_size, |x_hi| {
        // SAFETY: the queue hands out each x_hi exactly once; chunk x_hi
        // exclusively owns its anchors/deltas ranges and partials[x_hi]. The
        // queue's completion join publishes the writes before the reduction
        // below reads them.
        let anchors = unsafe {
            std::slice::from_raw_parts_mut(
                anchors_base.ptr().add(x_hi * anchor_chunk_size),
                anchor_chunk_size,
            )
        };
        let deltas = unsafe { deltas_base.ptr().add(x_hi * delta_chunk_size) };
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
                        deltas,
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
                    deltas,
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
                // Keep the uninitialized destination outside Rust's slice
                // model until every chunk has joined. This worker owns the
                // exact `delta_chunk_size` raw range for `x_hi`.
                let mut p1_acc = F256Unreduced::ZERO;
                let mut pinf_acc = F256Unreduced::ZERO;
                for x_lo in 0..lo_size {
                    if ((pair_idx_base + x_lo) & pair_in_block_mask) >= useful_pairs_inclusive {
                        anchors[2 * x_lo] = F128::ZERO;
                        anchors[2 * x_lo + 1] = F128::ZERO;
                        // SAFETY: this pair owns two complete n_chunks rows
                        // inside the worker's disjoint raw destination.
                        unsafe {
                            core::ptr::write_bytes(deltas.add(2 * x_lo * n_chunks), 0, 2 * n_chunks)
                        };
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
                        // SAFETY: the queue grants this worker exclusive
                        // ownership of the chunk and each indexed byte is
                        // written exactly once before publication.
                        unsafe {
                            deltas
                                .add(2 * x_lo * n_chunks + j)
                                .write(a0_bytes[j] ^ a1_bytes[j]);
                            deltas
                                .add((2 * x_lo + 1) * n_chunks + j)
                                .write(b0_bytes[j] ^ b1_bytes[j]);
                        }
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
            // (This sweep never launches R2-CS; its partials merge the same.)
            crate::gpu_commit::ZcR2Result::Prefix(vals)
            | crate::gpu_commit::ZcR2Result::PrefixCs { prefix: vals, .. } => {
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

    // Every x_hi chunk has joined, including GPU-prefix anchor/delta writers.
    // Publish full coverage before the compact object can expose byte slices.
    // SAFETY: the hetero queue and optional GPU-prefix fallback above have
    // joined; their pairwise-disjoint x_hi ranges cover exactly
    // `0..compact.deltas.len()`, including padding rows.
    unsafe { compact.deltas.mark_fully_initialized() };

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
            deltas: ScratchBytes::take_full_overwrite(2 * n_pairs * n_chunks),
            dead_tail: false,
            tail_compact: false,
            tail_base_blocks: 0,
            head_compact: false,
            head_b: F128::ZERO,
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
        let deltas_base = crate::epool::SyncPtr(compact.deltas.full_overwrite_ptr());
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
        // SAFETY: every hetero job above owns a disjoint x_hi byte range;
        // together they cover the full destination and the synchronous join
        // publishes all writes before the compact state can expose a slice.
        unsafe { compact.deltas.mark_fully_initialized() };
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
    assert!(!compact.dead_tail, "dead-pair compact state is K-pass only");
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
    assert!(!compact.dead_tail, "dead-pair compact state is K-pass only");
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

/// `(hi_size, bytes of A per chunk)` of the lookahead round-two sweep over
/// `2^m`-bit A/B buffers (k_skip = 6): the chunking
/// [`uni_skip_fold_and_round_pair_compact_padded_lookahead_for`] walks, for
/// the round-two A/B rebuild (`univariate_skip_optimized::ENV_NO_R2_AB_REBUILD`).
pub(crate) fn round2_lookahead_chunk_geometry(m: usize) -> Option<(usize, usize)> {
    const K_SKIP: usize = 6;
    if m < K_SKIP + 7 || m >= usize::BITS as usize {
        return None;
    }
    let n_vars = m - K_SKIP - 1;
    let hi_size = 1usize << lookahead_n_hi(n_vars);
    let lo_size = (1usize << n_vars) / hi_size;
    // Two rows of eight bytes per pair.
    Some((hi_size, 2 * lo_size * 8))
}

/// Round-two A/B rebuild: this worker's scratch for one chunk's rebuilt a
/// rows followed by its b rows (allocated on first use, kept for the
/// thread's life).
struct R2AbScratch {
    ptr: core::cell::Cell<*mut u8>,
    /// Bytes of each of the two halves.
    bytes: core::cell::Cell<usize>,
    /// The rebuild whose producer last filled every unit slot (0 = none).
    primed_id: core::cell::Cell<u64>,
}

impl R2AbScratch {
    const ALIGN: usize = 16384;
}

impl Drop for R2AbScratch {
    fn drop(&mut self) {
        let ptr = self.ptr.get();
        if !ptr.is_null() {
            let layout = std::alloc::Layout::from_size_align(2 * self.bytes.get(), Self::ALIGN).expect("layout");
            // SAFETY: allocated below with this layout.
            unsafe { std::alloc::dealloc(ptr, layout) };
        }
    }
}

thread_local! {
    static R2_AB_SCRATCH: R2AbScratch = const {
        R2AbScratch {
            ptr: core::cell::Cell::new(core::ptr::null_mut()),
            bytes: core::cell::Cell::new(0),
            primed_id: core::cell::Cell::new(0),
        }
    };
}

/// Rebuild chunk `x_hi`'s a and b rows (`chunk_bytes` each) through the
/// round-two A/B rebuild's producer into this thread's scratch. Returns the
/// rows' addresses, valid until this thread's next call.
fn r2_rebuild_chunk(
    rebuild: &crate::zerocheck::univariate_skip_optimized::Round2AbRebuild,
    x_hi: usize,
    chunk_bytes: usize,
) -> (*const u8, *const u8) {
    R2_AB_SCRATCH.with(|s| {
        if s.bytes.get() != chunk_bytes {
            let old = s.ptr.replace(core::ptr::null_mut());
            if !old.is_null() {
                let layout =
                    std::alloc::Layout::from_size_align(2 * s.bytes.get(), R2AbScratch::ALIGN).expect("layout");
                // SAFETY: allocated below with this layout.
                unsafe { std::alloc::dealloc(old, layout) };
            }
            let layout = std::alloc::Layout::from_size_align(2 * chunk_bytes, R2AbScratch::ALIGN).expect("layout");
            // SAFETY: non-zero size.
            let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            s.ptr.set(ptr);
            s.bytes.set(chunk_bytes);
            s.primed_id.set(0);
        }
        let a = s.ptr.get();
        // SAFETY: the allocation holds `2 * chunk_bytes` bytes.
        let b = unsafe { a.add(chunk_bytes) };
        let unit = rebuild.unit_bytes;
        let units = chunk_bytes / unit;
        let primed = s.primed_id.get() == rebuild.id;
        for j in 0..units {
            // SAFETY: slot `j` of both halves holds `unit` bytes; the
            // registering prove is running (round two holds its rebuild).
            unsafe { rebuild.produce_into(x_hi * units + j, a.add(j * unit), b.add(j * unit), primed) };
        }
        s.primed_id.set(rebuild.id);
        (a.cast_const(), b.cast_const())
    })
}

/// Any value restores chunk-wide A/B materialization before round two.
#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
fn r2_ab_quad_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_R2_AB_QUAD").is_none())
}

/// R2 C capture (`univariate_skip_optimized::ENV_NO_R2_C_CAPTURE`): the
/// round-two terms of the groups the walks left out, `(ΔG(1), ΔW0, ΔW1)`.
///
/// `rows` holds round one's C inner fold rows `R2_C_CAPTURE_ROWS` (64 lanes
/// each, instance weights `eq(r[14..32], ·)` = round two's instance weights).
/// In those groups (0..4 and 59 of every 256-row block) b folds to ONE and
/// a = C, so a row's folded a summed over the instances is
/// `a(t) = Σ_s fold(e_s)·c(t, s)`, and with `h_g = eq(ρ[2..8], g)`, `d = ρ[1]`:
/// `ΔG(1) = Σ_g h_g·((1+d)·a(4g+1) + d·a(4g+3))`, `ΔW0 = Σ_g h_g·a(4g+2)`,
/// `ΔW1 = Σ_g h_g·a(4g+3)`; every other round-two and round-three term of
/// these groups is zero (b is constant across them).
fn r2_c_capture_terms(table: &UniSkipFoldTable, rows: &[F128], mlv_challenges: &[F128]) -> (F128, F128, F128) {
    use crate::zerocheck::univariate_skip_optimized::R2_C_CAPTURE_ROWS;
    const GROUPS: [usize; 5] = [0, 1, 2, 3, 59];
    assert_eq!(rows.len(), R2_C_CAPTURE_ROWS.len() * 64);
    assert!(mlv_challenges.len() >= 8);
    let lane_fold: Vec<F128> = (0..64).map(|s| table.fold_one_row(&(1u64 << s).to_le_bytes())).collect();
    let d = mlv_challenges[1];
    let (mut g1, mut t0, mut t1) = (F128::ZERO, F128::ZERO, F128::ZERO);
    for (k, &g) in GROUPS.iter().enumerate() {
        let h = (0..6).fold(F128::ONE, |h, bit| {
            let c = mlv_challenges[2 + bit];
            h * if (g >> bit) & 1 == 1 { c } else { F128::ONE + c }
        });
        let mut a = [F128::ZERO; 3];
        for (j, a) in a.iter_mut().enumerate() {
            let idx = 3 * k + j;
            debug_assert_eq!(R2_C_CAPTURE_ROWS[idx], 4 * g + 1 + j);
            let row = &rows[64 * idx..64 * (idx + 1)];
            *a = lane_fold.iter().zip(row).fold(F128::ZERO, |acc, (l, c)| acc + *l * *c);
        }
        g1 += h * ((F128::ONE + d) * a[0] + d * a[2]);
        t0 += h * a[1];
        t1 += h * a[2];
    }
    (g1, t0, t1)
}

/// Interleave rebuilt ranked A/B quads and the existing full round-two walk.
/// The original chunk still owns all outputs and partials. Reduction is linear
/// over XOR, so XORing the eight reduced unit sums equals reducing once after
/// the original chunk. No producer call may outlive the registering scope.
///
/// # Safety
/// Inputs and outputs satisfy the full kernel's contract for `lo` pairs. The
/// producer supplies complete 8192-byte A/B units, whose constant primed rows
/// are identical between quads, and the chunk is exclusively CPU owned.
#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
#[allow(clippy::too_many_arguments)]
unsafe fn r2_rebuild_quad_walk(
    r: &crate::zerocheck::univariate_skip_optimized::Round2AbRebuild,
    x_hi: usize, table: *const u8, fold7: usize, anchors: *mut F128,
    deltas: *mut u8, eq: *const F128, prep: *const F128, lo: usize,
    pair_base: usize, tail: bool, head: *const std::sync::atomic::AtomicBool,
    xb_rho: *const F128, skip_pure: bool,
) -> [F128; 8] {
    // R2 C capture: only the window-scaled walk leaves groups out.
    assert!(!skip_pure || !xb_rho.is_null());
    let unit = r.unit_bytes;
    assert!(unit > 0 && unit.is_multiple_of(2048));
    assert!((16 * lo).is_multiple_of(unit));
    let pairs = unit / 16;
    let units = 16 * lo / unit;
    let mut out = [F128::ZERO; 8];
    for j in 0..units {
        // One scratch slot is reused. r2_rebuild_chunk marks it primed
        // only after the first complete write for this registration.
        let (a, b) = r2_rebuild_chunk(r, x_hi * units + j, unit);
        let at = j * pairs;
        let blocks = at / 128;
        let (anchor_at, delta_at) = if !head.is_null() {
            (blocks * kernels::aarch64::HC_ANCHORS_PER_BLOCK,
             blocks * kernels::aarch64::HC_DELTA_BYTES_PER_BLOCK)
        } else if tail {
            (blocks * kernels::aarch64::TC_ANCHORS_PER_BLOCK,
             blocks * kernels::aarch64::TC_DELTA_BYTES_PER_BLOCK)
        } else {
            (2 * at, 16 * at)
        };
        let mut part = [F128::ZERO; 8];
        unsafe {
            let ap = anchors.add(anchor_at);
            let dp = deltas.add(delta_at);
            let ep = eq.add(at);
            let pp = if prep.is_null() { prep } else { prep.add(at) };
            if !xb_rho.is_null() {
                // Each unit contains whole 32-group windows. Their rho
                // table is common; lambda comes from this unit's eq slice.
                kernels::aarch64::fold_round2_compact_chunk_neon_lookahead_8_xb(
                    table, fold7 as *const u8, a, b, ap, dp, ep, xb_rho,
                    pairs, true, tail, head, skip_pure, part.as_mut_ptr());
            } else if fold7 != 0 {
                kernels::aarch64::fold_round2_compact_chunk_neon_lookahead_8_pieces7(
                    table, fold7 as *const u8, a, b, ap, dp, ep, pp,
                    pairs, pair_base + at, 127, 121, true, true, true,
                    tail, head, part.as_mut_ptr());
            } else {
                fold_round2_compact_chunk_neon_lookahead_8::<true, false>(
                    table, a, b, ap, dp, ep, pp, pairs, pair_base + at,
                    127, 121, true, true, true, tail, head, part.as_mut_ptr());
            }
        }
        for lane in 0..8 { out[lane] += part[lane]; }
    }
    out
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
#[cfg_attr(not(test), allow(dead_code))]
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
    uni_skip_fold_and_round_pair_compact_padded_lookahead_for(
        a_packed,
        b_packed,
        m,
        k_skip,
        table,
        mlv_challenges,
        padding,
        deltas_backing,
        false,
        None,
    )
}

/// The K pass GPU prefix arm's premap length over a compact state of
/// `n_pairs` pairs: its dense anchors prefix in bytes, which covers every
/// compacted layout's prefix (0 = no premap).
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn zc_k_premap_bytes(n_pairs: usize) -> usize {
    if n_pairs < 16 {
        return 0;
    }
    let share = crate::gpu_commit::gpu_zc_k_share();
    if share == 0 {
        return 0;
    }
    let k_hi = 1usize << cascade_k_pass_n_hi(n_pairs.trailing_zeros() as usize - 2);
    let prefix_pairs = share.min(k_hi) * (n_pairs / k_hi);
    // The R2-CS kernel stores through this view too: with the measured
    // R2-CS share (`gpu_commit::ENV_NO_ZC_R2_CS_SHARE`) it covers that
    // share's ceiling, which lies above the K share (0 otherwise).
    let r2_pairs = crate::gpu_commit::zc_r2_cs_view_chunks(k_hi) * (n_pairs / k_hi);
    prefix_pairs.max(r2_pairs).min(n_pairs) * 32
}

/// R2-CS (`gpu_commit::ENV_NO_ZC_R2_CS`): before round one of a prove whose
/// round two runs the lookahead sweep, take that sweep's compact anchors and
/// premap the K pass arm's view over them ([`crate::gpu_commit::zc_k_premap`])
/// when R2-CS is admitted for timed launches and that view will exist. The
/// sweep then takes these anchors (`anchors_premapped`) instead of its own.
/// `None` = the sweep takes and premaps them itself.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub(crate) fn round2_anchors_premapped_early(m: usize, k_skip: usize) -> Option<Vec<F128>> {
    if k_skip != 6 || m <= k_skip || !crate::gpu_commit::zc_r2_cs_premap_early() {
        return None;
    }
    let n_pairs = (1usize << (m - k_skip)) / 2;
    let bytes = zc_k_premap_bytes(n_pairs);
    if bytes == 0 {
        return None;
    }
    // K view hand-over (`gpu_commit::ENV_NO_ZC_K_HANDOVER`): the anchors the
    // previous prove's K pass kept, with their view already mapped and now
    // published as this prove's premap.
    if let Some(anchors) = crate::gpu_commit::zc_k_handover_take(2 * n_pairs, bytes) {
        return Some(anchors);
    }
    let anchors = crate::scratch::take_f128(2 * n_pairs);
    crate::gpu_commit::zc_k_premap(&anchors, bytes);
    Some(anchors)
}

/// Pair count per cascade K-pass output chunk for a compact state of
/// `n_pairs` pairs (the split [`fold2_compact_and_round45_into`] uses).
fn cascade_k_pairs_per_chunk(n_pairs: usize) -> usize {
    let n_groups = n_pairs / 2;
    let n_vars = (n_groups.trailing_zeros() as usize).saturating_sub(1);
    let n_lo = n_vars - cascade_k_pass_n_hi(n_vars);
    4 << n_lo
}

/// [`uni_skip_fold_and_round_pair_compact_padded_lookahead`] for a caller
/// that will hand the compact state only to the cascade K pass
/// (`k_pass_only`). Then, on AArch64 at the ranked padding geometry, the
/// structural padding pairs (121..128 of every 128-pair block) get no anchor
/// or delta stores and the state is marked [`UniSkipCompactFold::dead_tail`].
/// The round-two message and lookahead are unchanged; every live slot holds
/// the same bytes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn uni_skip_fold_and_round_pair_compact_padded_lookahead_for(
    a_packed: &[u8],
    b_packed: &[u8],
    m: usize,
    k_skip: usize,
    table: &UniSkipFoldTable,
    mlv_challenges: &[F128],
    padding: &PaddingSpec,
    deltas_backing: Option<ScratchBytes>,
    k_pass_only: bool,
    anchors_premapped: Option<Vec<F128>>,
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
    let deltas = deltas_backing.unwrap_or_else(|| ScratchBytes::take_full_overwrite(deltas_len));
    assert_eq!(
        deltas.len(),
        deltas_len,
        "donated compact delta backing has the wrong byte length"
    );
    // Anchors taken (and the K view premapped) before round one, see
    // `round2_anchors_premapped_early`; else taken here.
    let (anchors, premapped) = match anchors_premapped {
        Some(a) if a.len() == 2 * n_pairs => (a, true),
        other => {
            if let Some(a) = other {
                crate::scratch::give_f128(a);
            }
            (crate::scratch::take_f128(2 * n_pairs), false)
        }
    };
    let mut compact = UniSkipCompactFold {
        anchors,
        deltas,
        dead_tail: false,
        tail_compact: false,
        tail_base_blocks: 0,
        head_compact: false,
        head_b: F128::ZERO,
    };
    // K pass GPU prefix arm (`FLOCK_NO_ZC_K_PREMAP` kills this step): map
    // its view of the anchors prefix now, while round two runs, so the K
    // command buffer does not wait for the mapping. The dense prefix covers
    // every compacted layout's prefix.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if !premapped {
        let bytes = zc_k_premap_bytes(n_pairs);
        if bytes > 0 {
            crate::gpu_commit::zc_k_premap(&compact.anchors, bytes);
        }
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    let _ = premapped;

    let n_vars = mlv_challenges.len() - 1;
    let eq = take_or_build_r2_eq(&mlv_challenges[1..], lookahead_n_hi(n_vars));
    let lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    assert_eq!(lo_size * hi_size, n_pairs);
    assert!(lo_size >= 2, "lookahead sweep pairs two x_lo per group");
    // Round-two A/B rebuild (`univariate_skip_optimized::ENV_NO_R2_AB_REBUILD`):
    // the AB arm published only chunks `..rebuild_from` of A and B, and every
    // walk below rebuilds a later chunk's rows into worker scratch before it
    // reads them (the GPU arm reads the prefix only). A prefix that is not
    // whole chunks of this sweep gets the rest published first: the record
    // path.
    let chunk_ab_bytes = 2 * lo_size * n_chunks;
    let rebuild = crate::zerocheck::univariate_skip_optimized::round2_ab_rebuild_take(a_packed, b_packed).and_then(|r| {
        if r.published.is_multiple_of(chunk_ab_bytes)
            && r.rebuild_from.is_multiple_of(chunk_ab_bytes)
            && chunk_ab_bytes.is_multiple_of(r.unit_bytes)
        {
            Some(r)
        } else {
            r.complete_globals();
            crate::zerocheck::univariate_skip_optimized::round2_ab_rebuild_latch_off();
            None
        }
    });
    // CPU walks rebuild chunks `rebuild_from..`; the GPU arm reads chunks
    // below `ab_ready` only.
    let rebuild_from = rebuild.map_or(hi_size, |r| r.rebuild_from / chunk_ab_bytes);
    #[cfg_attr(not(all(target_os = "macos", target_arch = "aarch64")), allow(unused_variables))]
    let ab_ready = rebuild.map_or(hi_size, |r| r.published / chunk_ab_bytes);
    // Chunk `x_hi`'s a and b rows: in the buffers, or (past the published
    // prefix) rebuilt into this worker's scratch, valid until its next
    // rebuild.
    let r2_rows = |x_hi: usize| -> (*const u8, *const u8) {
        match &rebuild {
            Some(r) if x_hi >= rebuild_from => r2_rebuild_chunk(r, x_hi, chunk_ab_bytes),
            // SAFETY: chunk `x_hi` lies inside both buffers.
            _ => unsafe {
                (a_packed.as_ptr().add(x_hi * chunk_ab_bytes), b_packed.as_ptr().add(x_hi * chunk_ab_bytes))
            },
        }
    };
    // `eq₂(2y) = (1+r₁)·eq₃(y)` and `eq₂(2y+1) = r₁·eq₃(y)`, so the sweep uses
    // the odd lane as the group's single weight and the two constants below
    // put every aggregate back on its own scale, once, off the hot path.
    let (kappa, retained_r1_inv) =
        lookahead_kappa_and_retained_inv(r1, lookahead_inverse_reuse_enabled());
    let eq_hi = &eq.hi;
    let eq_lo = &eq.lo;
    let anchor_chunk_size = 2 * lo_size;
    let delta_chunk_size = 2 * lo_size * n_chunks;
    let (pair_in_block_mask, useful_pairs_inclusive) = round2_pair_skip(padding, k_skip);
    #[cfg(target_arch = "aarch64")]
    let degen = r2_degen_enabled();
    #[cfg(target_arch = "aarch64")]
    let periodic_padding = r2_periodic_padding_enabled();
    // Dead-pair skip. The K pass substitutes zeros for pairs 121..128 of
    // each 128-pair block, which equals the dense state only because the
    // fold table's index-0 entries are zero (`scaled_linear` copies them).
    // Its block loop needs 128-pair-aligned producer and K chunks. The delta
    // backing must already be initialized memory (the donated round-one
    // buffer), so the skipped bytes stay valid and the completion mark below
    // stays true; a fresh backing keeps the full overwrite.
    #[cfg(target_arch = "aarch64")]
    let skip_dead = k_pass_only
        && r2_dead_pair_skip_enabled()
        && pair_in_block_mask == 127
        && useful_pairs_inclusive == 121
        && lo_size.is_multiple_of(128)
        && cascade_k_pairs_per_chunk(n_pairs).is_multiple_of(128)
        && compact.deltas.is_fully_initialized()
        && (0..n_chunks).all(|j| table.data[j * 256] == F128::ZERO);
    #[cfg(not(target_arch = "aarch64"))]
    let skip_dead = {
        let _ = k_pass_only;
        false
    };
    compact.dead_tail = skip_dead;
    #[cfg(target_arch = "aarch64")]
    let horner_weights = m == 32
        && lo_size == 1 << 14
        && pair_in_block_mask == 127
        && useful_pairs_inclusive == 121
        && periodic_padding
        && mlv_challenges[3..7]
            == crate::zerocheck::univariate_skip_optimized::medium_challenges_ghash();

    #[cfg(target_arch = "aarch64")]
    let horner_skip_first = r2_first_shift_skip_enabled();
    #[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
    let horner_prescale = r2_prescale_enabled();
    #[cfg(all(target_arch = "aarch64", not(target_feature = "sha3")))]
    let horner_prescale = false;

    // R2 C capture (`univariate_skip_optimized::ENV_NO_R2_C_CAPTURE`): round
    // one's kept C rows (taken here in every case, so none outlives this
    // round), used only on the ranked head-compacted window-scaled walk.
    let r2_capture = crate::zerocheck::univariate_skip_optimized::r2_c_capture_take();
    #[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
    let pure_wanted = r2_capture.is_some()
        && m == 32
        && lo_size.is_multiple_of(128)
        && skip_dead
        && horner_weights
        && periodic_padding
        && degen
        && zc_r2_tail_compact_enabled()
        && cascade_k_fused_enabled()
        && zc_k_one_table_enabled()
        && zc_r2_head_compact_enabled()
        && kernels::aarch64::r2_head_compact_supported()
        && kernels::aarch64::r2_full_xb_enabled()
        && table.fold_one_row(&[0xFF; 8]) == F128::ONE;
    #[cfg(not(all(target_arch = "aarch64", target_feature = "sha3")))]
    let pure_wanted = false;

    // R2-CS (`gpu_commit::ENV_NO_ZC_R2_CS`): the compact layout this sweep
    // writes when the arm runs, decided here (the same expressions as
    // `tail_compact` and `head_compact` below with the arm admitted) so the
    // arm's kernel can store the same bytes at the same offsets. Only the
    // ranked walk: dead pairs skipped, prefix chunks on the prescaled walk.
    #[cfg(all(target_os = "macos", target_arch = "aarch64", target_feature = "sha3"))]
    let cs_layout = if crate::gpu_commit::zc_r2_cs_enabled()
        && skip_dead
        && zc_r2_odd_offload_enabled()
        && horner_weights
        && horner_prescale
    {
        let tc = periodic_padding
            && zc_r2_tail_compact_enabled()
            && cascade_k_fused_enabled()
            && zc_k_one_table_enabled();
        let hc = tc
            && zc_r2_head_compact_enabled()
            && kernels::aarch64::r2_head_compact_supported()
            && degen
            && table.fold_one_row(&[0xFF; 8]) == F128::ONE;
        let tbb = if tc { (lo_size / 128).max(cascade_k_pairs_per_chunk(n_pairs) / 128) } else { 0 };
        use kernels::aarch64::{
            HC_ANCHOR_GAP, HC_ANCHOR_SHIFT, HC_ANCHORS_PER_BLOCK, HC_DELTA_BYTES_PER_BLOCK, HC_DELTA_GAP,
            HC_DELTA_SHIFT, TC_ANCHOR_GAP, TC_ANCHORS_PER_BLOCK, TC_DELTA_BYTES_PER_BLOCK, TC_DELTA_GAP,
        };
        Some(if hc {
            crate::gpu_commit::ZcR2CsLayout {
                hc: true,
                a_base: tbb * HC_ANCHOR_GAP + HC_ANCHOR_SHIFT,
                a_stride: HC_ANCHORS_PER_BLOCK,
                d_base: tbb * HC_DELTA_GAP + HC_DELTA_SHIFT,
                d_stride: HC_DELTA_BYTES_PER_BLOCK,
            }
        } else if tc {
            crate::gpu_commit::ZcR2CsLayout {
                hc: false,
                a_base: tbb * TC_ANCHOR_GAP,
                a_stride: TC_ANCHORS_PER_BLOCK,
                d_base: tbb * TC_DELTA_GAP,
                d_stride: TC_DELTA_BYTES_PER_BLOCK,
            }
        } else {
            crate::gpu_commit::ZcR2CsLayout { hc: false, a_base: 0, a_stride: 256, d_base: 0, d_stride: 2048 }
        })
    } else {
        None
    };
    #[cfg(all(target_os = "macos", target_arch = "aarch64", not(target_feature = "sha3")))]
    let cs_layout: Option<crate::gpu_commit::ZcR2CsLayout> = None;

    // R2-CS stores a timed prefix through this prove's existing views of
    // the two allocations, never a new one: the K premap above (anchors) and
    // round one's handed-over view of the donated deltas backing; the launch
    // checks that both exist and cover its prefix. `cs_live` = (anchors
    // base, deltas base, deltas bytes).
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let cs_live = cs_layout.map(|_| {
        (
            compact.anchors.as_ptr() as usize,
            compact.deltas.full_overwrite_ptr() as usize,
            compact.deltas.len(),
        )
    });

    // Same GPU round-two products arm as the incumbent sweep: it still
    // receives byte-identical anchors/deltas for every chunk and still owns
    // the *summed* `(p1, pinf)` for its prefix. The CPU keeps producing the
    // odd-parity halves on those chunks (the GPU's sums are not parity-split),
    // which is what makes `W1`/`W2` recoverable everywhere.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let gpu_job = crate::gpu_commit::launch_zc_r2_products_cs(
        a_packed,
        b_packed,
        &table.data,
        eq_lo,
        eq_hi,
        lo_size,
        hi_size,
        pair_in_block_mask,
        useful_pairs_inclusive,
        cs_layout,
        cs_live,
        ab_ready,
        pure_wanted,
    );
    // R2-CS owns this timed launch's prefix chunks (the CPU skips them).
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let r2cs = gpu_job.as_ref().is_some_and(|j| j.cs_active());
    // R2-CS calibration launch: (chunks, layout) for its compact-state oracle.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let cs_calib = gpu_job.as_ref().and_then(|j| j.cs_calibration());
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

    let mut partials: Vec<(F128, F128)> = crate::alloc_uninit_vec(hi_size);
    // [p1_odd, pinf_odd, W0, W3, W4, W5], eq_hi-weighted, one slot per chunk.
    let mut la_partials: Vec<[F128; 6]> = crate::alloc_uninit_vec(hi_size);
    // The fold table in seven pieces (`FLOCK_NO_R2_FOLD7`), for the full
    // chunks' ranked walks on the performance-core workers: one lookup fewer
    // per row, the same folded rows (none when a byte bank is not linear).
    #[cfg(target_arch = "aarch64")]
    let fold7 = if kernels::aarch64::r2_fold7_enabled() {
        kernels::aarch64::r2_fold7_table(&table.data)
    } else {
        None
    };
    #[cfg(target_arch = "aarch64")]
    let fold7_addr = fold7.as_ref().map_or(0usize, |t| t.as_ptr() as usize);
    // Tail-compacted layout: the dead-pair skip is on, every chunk's
    // producer runs its ranked periodic walk (the lookahead, prescaled and
    // Horner producers all rebase per block there) and the one-table fused
    // K kernel is the reader.
    #[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
    let tail_compact = skip_dead
        && periodic_padding
        && zc_r2_tail_compact_enabled()
        && cascade_k_fused_enabled()
        && zc_k_one_table_enabled();
    #[cfg(not(all(target_arch = "aarch64", target_feature = "sha3")))]
    let tail_compact = false;
    compact.tail_compact = tail_compact;
    // Full chunks (the suffix after the round-two GPU prefix, stolen prefix
    // chunks, every chunk without the arm) on the window-scaled walk
    // (`FLOCK_NO_R2_FULL_XB` keeps the prepared-weight walk): it needs the
    // fixed medium weights (`horner_weights`, which also pins the ranked
    // periodic block shape), the dead pairs skipped and the odd groups'
    // weights a constant multiple of the even ones' (checked when the table
    // is built). Same state bytes and slots as the prepared-weight walk.
    #[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
    let xb_rho = if horner_weights && skip_dead && kernels::aarch64::r2_full_xb_enabled() {
        kernels::aarch64::r2_full_xb_rho_prep(eq_lo)
    } else {
        None
    };
    #[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
    let xb_rho_addr = xb_rho.as_ref().map_or(0usize, |t| t.as_ptr() as usize);
    #[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
    let full_xb = xb_rho_addr != 0;
    #[cfg(all(target_arch = "aarch64", not(target_feature = "sha3")))]
    let full_xb = false;
    // The full-sweep group weights `eq_lo[2u + 1]` in prepared form, shared
    // by every chunk's ranked loop (skipped when `full_xb` uses window-scaled
    // weights; null without SHA3 or under `full_xb`, keeping fallbacks on the
    // parent loop).
    #[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
    let eq_prep = if full_xb { Vec::new() } else { kernels::aarch64::round2_eq_prep(eq_lo) };
    #[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
    let eq_prep_addr = if full_xb { 0usize } else { eq_prep.as_ptr() as usize };
    #[cfg(all(target_arch = "aarch64", not(target_feature = "sha3")))]
    let eq_prep_addr = 0usize;
    // Head-compacted layout on top: every chunk's producer must be one of the
    // two ranked walks that write it (the full lookahead walk, or the
    // prescaled ranked-invariant walk on GPU-offloaded chunks).
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let gpu_used = gpu_prefix > 0;
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    let gpu_used = false;
    #[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
    let head_compact = tail_compact
        && zc_r2_head_compact_enabled()
        && kernels::aarch64::r2_head_compact_supported()
        && degen
        && table.fold_one_row(&[0xFF; 8]) == F128::ONE
        && (!gpu_used || (odd_on_gpu && horner_weights && horner_prescale));
    #[cfg(not(all(target_arch = "aarch64", target_feature = "sha3")))]
    let head_compact = {
        let _ = gpu_used;
        false
    };
    let head_fail = std::sync::atomic::AtomicBool::new(false);
    let head_fail_addr = if head_compact { &head_fail as *const std::sync::atomic::AtomicBool as usize } else { 0 };
    #[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
    if tail_compact {
        // Room for the deepest rebase of either chunking (this one and the
        // K pass's) in front of the first block.
        compact.tail_base_blocks = (lo_size / 128).max(cascade_k_pairs_per_chunk(n_pairs) / 128);
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
    let (anchor_chunk_stride, delta_chunk_stride, anchor_base, delta_base) = if head_compact {
        compact.head_compact = true;
        compact.head_b = table.fold_one_row(&[0xFF; 8]);
        (
            lo_size / 128 * kernels::aarch64::HC_ANCHORS_PER_BLOCK,
            lo_size / 128 * kernels::aarch64::HC_DELTA_BYTES_PER_BLOCK,
            compact.anchor_index(0),
            compact.delta_offset(0),
        )
    } else if tail_compact {
        (
            lo_size / 128 * kernels::aarch64::TC_ANCHORS_PER_BLOCK,
            lo_size / 128 * kernels::aarch64::TC_DELTA_BYTES_PER_BLOCK,
            compact.anchor_index(0),
            compact.delta_offset(0),
        )
    } else {
        (anchor_chunk_size, delta_chunk_size, 0, 0)
    };
    #[cfg(not(all(target_arch = "aarch64", target_feature = "sha3")))]
    let (anchor_chunk_stride, delta_chunk_stride, anchor_base, delta_base) =
        (anchor_chunk_size, delta_chunk_size, 0usize, 0usize);
    // R2-CS: the arm stores the prefix chunks at the planned layout, which
    // is this sweep's by construction (same expressions, arm admitted). Were
    // it not, the CPU walks every chunk and the whole round is redone
    // serially once the arm has joined (`cs_redo_all`, below).
    // R2 C capture: the CPU's walks can leave the capture's groups out only
    // on the head-compacted window-scaled walk; a GPU prefix that left them
    // out while the CPU cannot is redone with the whole round below.
    let pure_cpu = pure_wanted && full_xb && head_compact;
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let gpu_pure = gpu_job.as_ref().is_some_and(|j| j.pure_active());
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    let gpu_pure = false;
    #[cfg(all(target_os = "macos", target_arch = "aarch64", target_feature = "sha3"))]
    let cs_redo_all = r2cs
        && (!cs_layout.is_some_and(|l| {
            l.hc == compact.head_compact
                && l.a_base == anchor_base
                && l.d_base == delta_base
                && l.a_stride * (lo_size / 128) == anchor_chunk_stride
                && l.d_stride * (lo_size / 128) == delta_chunk_stride
        }) || (gpu_pure && !pure_cpu));
    #[cfg(all(target_os = "macos", target_arch = "aarch64", target_feature = "sha3"))]
    let r2cs = r2cs && !cs_redo_all;
    // Every chunk the CPU walks is then a full window-scaled walk, and every
    // GPU chunk whose sums are used left the same groups out.
    #[cfg(all(target_os = "macos", target_arch = "aarch64", target_feature = "sha3"))]
    let skip_pure = pure_cpu
        && !cs_redo_all
        && !gpu_job.as_ref().is_some_and(|j| j.is_calibration())
        && (!gpu_used || (r2cs && gpu_pure));
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64", target_feature = "sha3")))]
    let skip_pure = {
        let _ = (pure_cpu, gpu_pure);
        false
    };
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if crate::gpu_commit::gpu_zc_r2_debug() {
        eprintln!(
            "[zc-r2] round-two layout: head_compact={} tail_compact={tail_compact} skip_dead={skip_dead} \
             odd_on_gpu={odd_on_gpu} horner_weights={horner_weights} prescale={horner_prescale} \
             gpu_prefix={gpu_prefix}/{hi_size} calibration={} r2cs={r2cs} ab_rebuild_from={} \
             c_capture={skip_pure}",
            compact.head_compact,
            gpu_job.as_ref().is_some_and(|j| j.is_calibration()),
            if rebuild.is_some() { rebuild_from.to_string() } else { "off".to_string() },
        );
    }
    let anchors_base = crate::epool::SyncPtr(unsafe { compact.anchors.as_mut_ptr().add(anchor_base) });
    let deltas_base = crate::epool::SyncPtr(unsafe { compact.deltas.full_overwrite_ptr().add(delta_base) });
    let partials_base = crate::epool::SyncPtr(partials.as_mut_ptr());
    let la_base = crate::epool::SyncPtr(la_partials.as_mut_ptr());
    // `stolen`: R2-CS steal (`gpu_commit::ENV_NO_R2_STEAL`), a prefix chunk
    // the CPU took from the GPU, walked in full as a chunk after the prefix.
    let chunk_body = |x_hi: usize, stolen: bool| {
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        let _ = stolen;
        // SAFETY: the queue (or the steal's claim) hands out each x_hi
        // exactly once; chunk x_hi exclusively owns its anchors/deltas ranges
        // and both partial slots.
        let (anchors, deltas) = unsafe {
            (
                anchors_base.ptr().add(x_hi * anchor_chunk_stride),
                deltas_base.ptr().add(x_hi * delta_chunk_stride),
            )
        };
        let pair_idx_base = x_hi * lo_size;
        let row_base = pair_idx_base * 2;
        let mut out = [F128::ZERO; 8];
        let head_fail_ptr = head_fail_addr as *const std::sync::atomic::AtomicBool;

        // R2-CS: the arm owns this prefix chunk outright (compact state,
        // products, lookahead terms); its slots are filled after the join.
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        if r2cs && x_hi < gpu_prefix && !stolen {
            // SAFETY: exclusive owner of both partial slots (see above).
            unsafe {
                *partials_base.ptr().add(x_hi) = (F128::ZERO, F128::ZERO);
                *la_base.ptr().add(x_hi) = [F128::ZERO; 6];
            }
            return;
        }

        #[cfg(target_arch = "aarch64")]
        {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            let full = stolen || x_hi >= gpu_prefix;
            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            let full = true;
            unsafe {
                let t = table.data.as_ptr().cast::<u8>();
                // Round-two A/B rebuild: a chunk past the published prefix
                // reads its rows from this worker's scratch, rebuilt now.
                debug_assert_eq!(row_base * n_chunks, x_hi * chunk_ab_bytes);
                #[cfg(target_feature = "sha3")]
                let quad = rebuild.as_ref().filter(|r| {
                    full && m == 32 && x_hi >= rebuild_from && r2_ab_quad_enabled()
                        && r.unit_bytes == 8192 && chunk_ab_bytes.is_multiple_of(r.unit_bytes)
                        && periodic_padding && pair_in_block_mask == 127
                        && useful_pairs_inclusive == 121 && pair_idx_base.is_multiple_of(128)
                        && degen && skip_dead && (full_xb || eq_prep_addr != 0)
                });
                #[cfg(not(target_feature = "sha3"))]
                let quad: Option<&crate::zerocheck::univariate_skip_optimized::Round2AbRebuild> = None;
                #[cfg(target_feature = "sha3")]
                if let Some(r) = quad {
                    let table7 = if fold7_addr != 0 && !kernels::aarch64::r2_on_efficiency_helper() {
                        fold7_addr
                    } else { 0 };
                    out = r2_rebuild_quad_walk(r, x_hi, t, table7, anchors, deltas,
                        eq_lo.as_ptr(), eq_prep_addr as *const F128, lo_size,
                        pair_idx_base, tail_compact, head_fail_ptr,
                        if full_xb { xb_rho_addr as *const F128 } else { core::ptr::null() },
                        skip_pure);
                }
                if quad.is_none() {
                let (ap, bp) = r2_rows(x_hi);
                if full && full_xb {
                    // Full chunk on the window-scaled walk
                    // (`FLOCK_NO_R2_FULL_XB`): the seven-piece table on the
                    // performance-core workers, the byte banks on the helpers.
                    #[cfg(target_feature = "sha3")]
                    kernels::aarch64::fold_round2_compact_chunk_neon_lookahead_8_xb(
                        t,
                        if fold7_addr != 0 && !kernels::aarch64::r2_on_efficiency_helper() {
                            fold7_addr as *const u8
                        } else {
                            core::ptr::null()
                        },
                        ap,
                        bp,
                        anchors,
                        deltas,
                        eq_lo.as_ptr(),
                        xb_rho_addr as *const F128,
                        lo_size,
                        degen,
                        tail_compact,
                        head_fail_ptr,
                        skip_pure,
                        out.as_mut_ptr(),
                    );
                } else if full && fold7_addr != 0 && !kernels::aarch64::r2_on_efficiency_helper() {
                    kernels::aarch64::fold_round2_compact_chunk_neon_lookahead_8_pieces7(
                        t,
                        fold7_addr as *const u8,
                        ap,
                        bp,
                        anchors,
                        deltas,
                        eq_lo.as_ptr(),
                        eq_prep_addr as *const F128,
                        lo_size,
                        pair_idx_base,
                        pair_in_block_mask,
                        useful_pairs_inclusive,
                        degen,
                        periodic_padding,
                        skip_dead,
                        tail_compact,
                        head_fail_ptr,
                        out.as_mut_ptr(),
                    );
                } else if full {
                    fold_round2_compact_chunk_neon_lookahead_8::<true, false>(
                        t,
                        ap,
                        bp,
                        anchors,
                        deltas,
                        eq_lo.as_ptr(),
                        eq_prep_addr as *const F128,
                        lo_size,
                        pair_idx_base,
                        pair_in_block_mask,
                        useful_pairs_inclusive,
                        degen,
                        periodic_padding,
                        skip_dead,
                        tail_compact,
                        head_fail_ptr,
                        out.as_mut_ptr(),
                    );
                } else if odd_on_gpu && horner_weights {
                    if horner_prescale {
                        #[cfg(target_feature = "sha3")]
                        fold_round2_compact_chunk_neon_prescaled_8(
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
                            skip_dead,
                            tail_compact,
                            head_fail_ptr,
                            out.as_mut_ptr(),
                        );
                    } else if horner_skip_first {
                        fold_round2_compact_chunk_neon_horner_8::<true>(
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
                            skip_dead,
                            tail_compact,
                            head_fail_ptr,
                            out.as_mut_ptr(),
                        );
                    } else {
                        fold_round2_compact_chunk_neon_horner_8::<false>(
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
                            skip_dead,
                            tail_compact,
                            head_fail_ptr,
                            out.as_mut_ptr(),
                        );
                    }
                    let eq_h = eq_hi[x_hi];
                    *partials_base.ptr().add(x_hi) = (F128::ZERO, F128::ZERO);
                    *la_base.ptr().add(x_hi) = [
                        F128::ZERO,
                        F128::ZERO,
                        eq_h * out[4],
                        eq_h * out[5],
                        eq_h * out[6],
                        eq_h * out[7],
                    ];
                    return;
                } else if odd_on_gpu {
                    fold_round2_compact_chunk_neon_lookahead_8::<false, true>(
                        t,
                        ap,
                        bp,
                        anchors,
                        deltas,
                        eq_lo.as_ptr(),
                        core::ptr::null(),
                        lo_size,
                        pair_idx_base,
                        pair_in_block_mask,
                        useful_pairs_inclusive,
                        degen,
                        periodic_padding,
                        skip_dead,
                        tail_compact,
                        head_fail_ptr,
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
                        core::ptr::null(),
                        lo_size,
                        pair_idx_base,
                        pair_in_block_mask,
                        useful_pairs_inclusive,
                        degen,
                        periodic_padding,
                        skip_dead,
                        tail_compact,
                        head_fail_ptr,
                        out.as_mut_ptr(),
                    );
                }
                }
            }
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            let anchors = unsafe { std::slice::from_raw_parts_mut(anchors, anchor_chunk_size) };
            // Round-two A/B rebuild: past the published prefix, the rebuilt
            // chunk (its rows from 0).
            let (rows_a, rows_b, rows_base) = if rebuild.is_some() && x_hi >= rebuild_from {
                let (pa, pb) = r2_rows(x_hi);
                // SAFETY: the scratch holds `chunk_ab_bytes` bytes of each.
                unsafe {
                    (
                        std::slice::from_raw_parts(pa, chunk_ab_bytes),
                        std::slice::from_raw_parts(pb, chunk_ab_bytes),
                        0,
                    )
                }
            } else {
                (a_packed, b_packed, row_base)
            };
            out = round2_lookahead_chunk_scalar(
                rows_a,
                rows_b,
                table,
                anchors,
                deltas,
                eq_lo,
                lo_size,
                rows_base,
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
    };
    // R2-CS steal (`gpu_commit::ENV_NO_R2_STEAL`): the queue walks the chunks
    // after the prefix; each main-pool worker that finds them claimed then
    // takes the GPU's chunks from the top of the prefix until the prefix's
    // signal has been seen. Otherwise the record's queue over every chunk.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    let steal = gpu_job.as_ref().and_then(|j| j.steal()).filter(|_| r2cs);
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if let Some(steal) = steal {
        crate::epool::run_hetero_chunks_then(hi_size - gpu_prefix, &|i| chunk_body(gpu_prefix + i, false), &|| {
            while let Some(x_hi) = steal.claim() {
                chunk_body(x_hi, true);
            }
        });
    } else {
        crate::epool::run_hetero_chunks(hi_size, |x_hi| chunk_body(x_hi, false));
    }
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    crate::epool::run_hetero_chunks(hi_size, |x_hi| chunk_body(x_hi, false));

    // Drain the GPU products arm. The arm returns `[p1_even, pinf_even,
    // p1_odd, pinf_odd]`; XORing the parities back together reproduces the
    // summed pair the incumbent contract delivered, and the odd half is
    // adopted as the lookahead's `W1`/`W2` state exactly when the CPU was told
    // to skip it.
    // R2 C capture: the arm failed after the CPU left the capture's groups
    // out of its chunks; the whole round is redone without the capture below.
    #[allow(unused_mut)]
    let mut capture_gpu_failed = false;
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if let Some(job) = gpu_job {
        let cpu_wall_ms = t_cpu_sweep.elapsed().as_secs_f64() * 1e3;
        let calib = job.is_calibration();
        // R2-CS steal: the GPU's chunks are those below the lowest chunk the
        // CPU took (the CPU walked the rest of the prefix).
        let prefix = job.steal().map_or(job.cpu_split(), |s| s.taken().0);
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
            crate::gpu_commit::ZcR2Result::PrefixCs { prefix: vals, lookahead, head_bad } => {
                // R2-CS: the arm owned these chunks outright, so their
                // lookahead slots 2..6 are its [W0, W3, W4, W5] as well, and its
                // head-pair flag joins the CPU walk's. (Without `r2cs` the CPU
                // walked them after all: the incumbent merge.)
                for (x_hi, (v, w)) in vals.iter().zip(&lookahead).enumerate() {
                    partials[x_hi] = (v[0] + v[2], v[1] + v[3]);
                    if r2cs {
                        la_partials[x_hi] = [v[2], v[3], w[0], w[1], w[2], w[3]];
                    } else if odd_on_gpu {
                        la_partials[x_hi][0] = v[2];
                        la_partials[x_hi][1] = v[3];
                    }
                }
                if r2cs && head_bad {
                    head_fail.store(true, std::sync::atomic::Ordering::Relaxed);
                }
            }
            crate::gpu_commit::ZcR2Result::Failed if skip_pure => capture_gpu_failed = true,
            crate::gpu_commit::ZcR2Result::Failed => {
                // Redo exactly the skipped prefix products — slower, still
                // exact. Throwaway anchor/delta scratch: the real ranges were
                // already written by the lookahead pass. The full-lookahead
                // monomorphization is used so the odd-parity slots are
                // recovered too when the CPU sweep skipped them. R2-CS: the
                // sweep skipped these chunks outright, so the redo writes their
                // real compact state and lookahead slots with the sweep's own
                // walk.
                let mut scr_anchors = vec![F128::ZERO; anchor_chunk_size];
                let mut scr_deltas = vec![0u8; delta_chunk_size];
                for x_hi in 0..prefix {
                    let pair_idx_base = x_hi * lo_size;
                    let (ap, bp) = r2_rows(x_hi);
                    let mut out = [F128::ZERO; 8];
                    let (anchors, deltas, dead, tail, head) = if r2cs {
                        unsafe {
                            (
                                anchors_base.ptr().add(x_hi * anchor_chunk_stride),
                                deltas_base.ptr().add(x_hi * delta_chunk_stride),
                                skip_dead,
                                tail_compact,
                                head_fail_addr as *const std::sync::atomic::AtomicBool,
                            )
                        }
                    } else {
                        (scr_anchors.as_mut_ptr(), scr_deltas.as_mut_ptr(), false, false, core::ptr::null())
                    };
                    unsafe {
                        fold_round2_compact_chunk_neon_lookahead_8::<true, false>(
                            table.data.as_ptr().cast::<u8>(),
                            ap,
                            bp,
                            anchors,
                            deltas,
                            eq_lo.as_ptr(),
                            eq_prep_addr as *const F128,
                            lo_size,
                            pair_idx_base,
                            pair_in_block_mask,
                            useful_pairs_inclusive,
                            degen,
                            periodic_padding,
                            dead,
                            tail,
                            head,
                            out.as_mut_ptr(),
                        );
                    }
                    let eq_h = eq_hi[x_hi];
                    partials[x_hi] = (
                        eq_h * (kappa * out[0] + out[2]),
                        eq_h * (kappa * out[1] + out[3]),
                    );
                    if r2cs {
                        la_partials[x_hi] = [
                            eq_h * out[2],
                            eq_h * out[3],
                            eq_h * out[4],
                            eq_h * out[5],
                            eq_h * out[6],
                            eq_h * out[7],
                        ];
                    } else if odd_on_gpu {
                        la_partials[x_hi][0] = eq_h * out[2];
                        la_partials[x_hi][1] = eq_h * out[3];
                    }
                }
            }
        }
    }

    // R2-CS stored at a layout this sweep does not write (never at the
    // ranked shape, see `cs_redo_all`): the arm has joined, so redo the whole
    // round serially with the full walk, every store and every slot.
    #[cfg(all(target_os = "macos", target_arch = "aarch64", target_feature = "sha3"))]
    if cs_redo_all {
        for x_hi in 0..hi_size {
            let pair_idx_base = x_hi * lo_size;
            let (ap, bp) = r2_rows(x_hi);
            let mut out = [F128::ZERO; 8];
            // SAFETY: every writer has joined; chunk x_hi's ranges, serially.
            unsafe {
                fold_round2_compact_chunk_neon_lookahead_8::<true, false>(
                    table.data.as_ptr().cast::<u8>(),
                    ap,
                    bp,
                    anchors_base.ptr().add(x_hi * anchor_chunk_stride),
                    deltas_base.ptr().add(x_hi * delta_chunk_stride),
                    eq_lo.as_ptr(),
                    eq_prep_addr as *const F128,
                    lo_size,
                    pair_idx_base,
                    pair_in_block_mask,
                    useful_pairs_inclusive,
                    degen,
                    periodic_padding,
                    skip_dead,
                    tail_compact,
                    head_fail_addr as *const std::sync::atomic::AtomicBool,
                    out.as_mut_ptr(),
                );
            }
            let eq_h = eq_hi[x_hi];
            partials[x_hi] = (eq_h * (kappa * out[0] + out[2]), eq_h * (kappa * out[1] + out[3]));
            la_partials[x_hi] = [
                eq_h * out[2],
                eq_h * out[3],
                eq_h * out[4],
                eq_h * out[5],
                eq_h * out[6],
                eq_h * out[7],
            ];
        }
    }

    // R2 C capture: a capture group whose b rows are not all ones (the head
    // flag), or a failed arm: redo the whole round serially without the
    // capture, every store and every slot, in the head-compacted layout or,
    // after a head failure, the tail-compacted one (never at the ranked shape
    // with a satisfying witness).
    #[allow(unused_mut)]
    let mut capture_sums = skip_pure;
    #[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
    if skip_pure && (capture_gpu_failed || head_fail.load(std::sync::atomic::Ordering::Relaxed)) {
        capture_sums = false;
        let to_tail = head_fail.load(std::sync::atomic::Ordering::Relaxed);
        let (a_base, d_base, a_stride, d_stride, head) = if to_tail {
            compact.head_compact = false;
            compact.head_b = F128::ZERO;
            (
                unsafe { compact.anchors.as_mut_ptr().add(compact.anchor_index(0)) },
                unsafe { compact.deltas.full_overwrite_ptr().add(compact.delta_offset(0)) },
                lo_size / 128 * kernels::aarch64::TC_ANCHORS_PER_BLOCK,
                lo_size / 128 * kernels::aarch64::TC_DELTA_BYTES_PER_BLOCK,
                core::ptr::null(),
            )
        } else {
            (
                anchors_base.ptr(),
                deltas_base.ptr(),
                anchor_chunk_stride,
                delta_chunk_stride,
                head_fail_addr as *const std::sync::atomic::AtomicBool,
            )
        };
        for x_hi in 0..hi_size {
            let (ap, bp) = r2_rows(x_hi);
            let mut out = [F128::ZERO; 8];
            // SAFETY: every writer has joined; chunk x_hi's ranges, serially.
            unsafe {
                fold_round2_compact_chunk_neon_lookahead_8::<true, false>(
                    table.data.as_ptr().cast::<u8>(),
                    ap,
                    bp,
                    a_base.add(x_hi * a_stride),
                    d_base.add(x_hi * d_stride),
                    eq_lo.as_ptr(),
                    eq_prep_addr as *const F128,
                    lo_size,
                    x_hi * lo_size,
                    pair_in_block_mask,
                    useful_pairs_inclusive,
                    degen,
                    periodic_padding,
                    skip_dead,
                    true,
                    head,
                    out.as_mut_ptr(),
                );
            }
            let eq_h = eq_hi[x_hi];
            partials[x_hi] = (eq_h * (kappa * out[0] + out[2]), eq_h * (kappa * out[1] + out[3]));
            la_partials[x_hi] = [
                eq_h * out[2],
                eq_h * out[3],
                eq_h * out[4],
                eq_h * out[5],
                eq_h * out[6],
                eq_h * out[7],
            ];
        }
    }

    // A head pair whose b rows are not all ones: the head-compacted state
    // cannot hold its b anchor and delta, so rewrite the whole state (anchors
    // and deltas only; the products above stand, they do not depend on the
    // layout) in the tail-compacted layout with the full lookahead walk.
    #[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
    if head_compact && compact.head_compact && head_fail.load(std::sync::atomic::Ordering::Relaxed) {
        compact.head_compact = false;
        compact.head_b = F128::ZERO;
        let a_base = unsafe { compact.anchors.as_mut_ptr().add(compact.anchor_index(0)) };
        let d_base = unsafe { compact.deltas.full_overwrite_ptr().add(compact.delta_offset(0)) };
        let a_stride = lo_size / 128 * kernels::aarch64::TC_ANCHORS_PER_BLOCK;
        let d_stride = lo_size / 128 * kernels::aarch64::TC_DELTA_BYTES_PER_BLOCK;
        // Serial: this path never runs at the ranked shape, and a plain loop
        // adds no queue instantiation to the build.
        for x_hi in 0..hi_size {
            let pair_idx_base = x_hi * lo_size;
            let (ap, bp) = r2_rows(x_hi);
            let mut out = [F128::ZERO; 8];
            // SAFETY: chunk x_hi's anchor and delta ranges, one chunk at a time.
            unsafe {
                fold_round2_compact_chunk_neon_lookahead_8::<true, false>(
                    table.data.as_ptr().cast::<u8>(),
                    ap,
                    bp,
                    a_base.add(x_hi * a_stride),
                    d_base.add(x_hi * d_stride),
                    eq_lo.as_ptr(),
                    eq_prep_addr as *const F128,
                    lo_size,
                    pair_idx_base,
                    pair_in_block_mask,
                    useful_pairs_inclusive,
                    degen,
                    periodic_padding,
                    skip_dead,
                    true,
                    core::ptr::null(),
                    out.as_mut_ptr(),
                );
            }
        }
    }

    // R2-CS calibration (warm-up prove, after every writer joined): the
    // arm's scratch compact state against the CPU's final compact state,
    // every live slot of the calibration chunks at the state's own
    // `anchor_index` / `delta_offset`; the verdict admits R2-CS for the
    // process (the lookahead partials were compared at the drain).
    #[cfg(all(target_os = "macos", target_arch = "aarch64", target_feature = "sha3"))]
    if let Some((chunks, layout)) = cs_calib {
        let laid_out = compact.head_compact == layout.hc
            && compact.anchor_index(0) == layout.a_base
            && compact.anchor_index(128) == layout.a_base + layout.a_stride
            && compact.delta_offset(0) == layout.d_base
            && compact.delta_offset(128) == layout.d_base + layout.d_stride;
        let deltas_ptr = compact.deltas.full_overwrite_ptr().cast_const();
        let res = if laid_out {
            // SAFETY: the joined sweep wrote the state; `deltas_ptr` spans
            // `compact.deltas.len()` bytes.
            unsafe {
                crate::gpu_commit::zc_r2_cs_oracle(
                    chunks * lo_size,
                    &compact.anchors,
                    deltas_ptr,
                    compact.deltas.len(),
                    |p| (p & pair_in_block_mask) < useful_pairs_inclusive,
                    |p| {
                        (
                            compact.anchor_index(p),
                            compact.delta_offset(p),
                            compact.head_compact && p % 128 < kernels::aarch64::HC_HEAD_PAIRS,
                        )
                    },
                )
            }
        } else {
            Err(format!(
                "planned layout {layout:?} is not the state's (head_compact={})",
                compact.head_compact
            ))
        };
        if let Err(e) = &res
            && crate::gpu_commit::gpu_zc_r2_debug()
        {
            eprintln!("[zc-r2] R2-CS ORACLE MISMATCH: {e} -> R2-CS disabled");
        }
        crate::gpu_commit::zc_r2_cs_verdict(res.is_ok(), chunks, hi_size);
    }

    // Both the standard and lookahead round-two producers promise identical
    // full delta coverage. Only after the CPU/GPU writers join may the compact
    // reconstruction expose the byte backing to its readers.
    // SAFETY: the lookahead writer has the same complete delta-store contract
    // as the standard writer. All hetero workers and the optional GPU-prefix
    // fallback have joined, covering every byte including padding rows.
    unsafe { compact.deltas.mark_fully_initialized() };

    #[allow(unused_mut)]
    let (mut sum1, sum_inf) = partials
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
    let r1_inv = retained_r1_inv.unwrap_or_else(|| r1.inv());
    #[allow(unused_mut)]
    let mut w1 = r1_inv * agg[0];
    let w2 = r1_inv * agg[1];
    #[allow(unused_mut)]
    let mut w0 = r1_inv * agg[2];
    // R2 C capture: the left-out groups' exact terms, from round one's kept
    // C rows (`r2_c_capture_terms`).
    if capture_sums && let Some(rows) = r2_capture.as_deref() {
        let (g1, t0, t1) = r2_c_capture_terms(table, rows, mlv_challenges);
        sum1 += g1;
        w0 += t0;
        w1 += t1;
    }
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
    deltas: *mut u8,
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

    let fold_pair = |x_lo: usize, anchors: &mut [F128]| {
        if ((pair_idx_base + x_lo) & pair_in_block_mask) >= useful_pairs_inclusive {
            anchors[2 * x_lo] = F128::ZERO;
            anchors[2 * x_lo + 1] = F128::ZERO;
            // SAFETY: the caller passes the unique raw destination for this
            // x_hi chunk; this pair owns two complete n_chunks rows.
            unsafe { core::ptr::write_bytes(deltas.add(2 * x_lo * n_chunks), 0, 2 * n_chunks) };
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
            // SAFETY: each indexed byte belongs to this chunk/pair and is
            // written exactly once before the outer join publishes it.
            unsafe {
                deltas
                    .add(2 * x_lo * n_chunks + j)
                    .write(a0_bytes[j] ^ a1_bytes[j]);
                deltas
                    .add((2 * x_lo + 1) * n_chunks + j)
                    .write(b0_bytes[j] ^ b1_bytes[j]);
            }
        }
        Some((a0, a1, b0, b1))
    };

    for u in 0..lo_size / 2 {
        let even = fold_pair(2 * u, anchors);
        let odd = fold_pair(2 * u + 1, anchors);
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
    assert!(!compact.dead_tail, "dead-pair compact state is K-pass only");
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
    let (m1, minf, la, _, _) =
        fold2_compact_and_round45_into_packed(compact, table, rho1, rho2, r_next4, a_out, b_out, false);
    (m1, minf, la)
}

/// Kill switch for the dead-tail-packed K output (`FLOCK_NO_ZC_K_DEAD_PACK`,
/// any value): the K pass writes every output, the structural zeros
/// included, and the tri pass reads them back. Packed, the K pass writes 61
/// of every 64 outputs (`kernels::aarch64::PACK_SUPER` layout) and the tri
/// pass reads those 61, taking the three zeros as zero registers. Same
/// values, same messages. Read once per process.
pub(crate) fn zc_k_dead_pack_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_ZC_K_DEAD_PACK").is_none())
}

/// Kill switch for the b-one-packed K b output (`FLOCK_NO_ZC_K_BONE_PACK`,
/// any value): with the dead-tail pack, the K pass writes b outputs 0..3 and
/// 59 of every 64 (one at the ranked shape: const-one wires of BLAKE3's
/// linear constraints) and the tri pass reads them back. Packed, the K pass
/// writes 56 b entries per 64 outputs (`kernels::aarch64::B_PACK_BLOCK`, 14
/// lines instead of 61/4 + ...: 56 lines per 256 outputs instead of 61) and
/// the tri pass takes those ones as one registers; it checks every skipped
/// value and reruns the pass unpacked if one is not one. Same values, same
/// messages. Read once per process.
pub(crate) fn zc_k_bone_pack_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_ZC_K_BONE_PACK").is_none())
}

/// [`fold2_compact_and_round45_into`] that, with `pack` and when the state
/// and split allow it (one-table fused kernel, dead-tail state, chunks of
/// whole four-block superblocks), writes the dead-tail-packed layout: the
/// first `n_groups / 256 * 244` entries of `a_out` / `b_out` (see
/// `kernels::aarch64::PACK_SUPER`); outputs 61..63 of every 64 are zero and
/// are not written. Returns whether it packed (otherwise every output is
/// written as usual) and whether b is in the b-one-packed layout instead
/// (the first `n_groups / 64 * 56` entries of `b_out`, see
/// [`zc_k_bone_pack_enabled`]).
#[allow(clippy::too_many_arguments)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn fold2_compact_and_round45_into_packed(
    compact: &UniSkipCompactFold,
    table: &UniSkipFoldTable,
    rho1: F128,
    rho2: F128,
    r_next4: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    pack: bool,
) -> (F128, F128, Round3Lookahead, bool, bool) {
    let (m1, minf, la, packed, bpacked, _) = fold2_compact_and_round45_into_packed_gpu(
        compact, table, rho1, rho2, r_next4, a_out, b_out, pack, false,
    );
    (m1, minf, la, packed, bpacked)
}

/// Outputs of the leading K chunks that the K pass GPU prefix arm folded (see
/// `gpu_commit::ENV_NO_ZC_K_GPU`): `len_a` / `len_b` entries of the packed a
/// and b-one-packed b layouts, held in the arm's process-lifetime buffers.
/// The same entries of `a_out` / `b_out` were not written; the tri pass reads
/// them from here. Valid until the next K pass. Also the leading outputs of
/// the composed 5+6 pass that its GPU prefix arm folded (see
/// `gpu_commit::ENV_NO_ZC_T56_GPU`), valid until that arm's next launch.
pub(crate) struct KGpuPrefix {
    a: *const F128,
    b: *const F128,
    len_a: usize,
    len_b: usize,
}

impl KGpuPrefix {
    /// SAFETY: call before the next K pass of the process (which rewrites
    /// the buffers).
    pub(crate) unsafe fn slices<'a>(&self) -> (&'a [F128], &'a [F128]) {
        unsafe {
            (
                std::slice::from_raw_parts(self.a, self.len_a),
                std::slice::from_raw_parts(self.b, self.len_b),
            )
        }
    }
}

/// [`fold2_compact_and_round45_into_packed`] with the K pass GPU prefix arm
/// when `gpu` is set (b-one-packed tail-compacted pass only). Also returns the
/// GPU prefix the tri pass must read in place of the leading entries of
/// `a_out` / `b_out`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fold2_compact_and_round45_into_packed_gpu(
    compact: &UniSkipCompactFold,
    table: &UniSkipFoldTable,
    rho1: F128,
    rho2: F128,
    r_next4: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    pack: bool,
    gpu: bool,
) -> (F128, F128, Round3Lookahead, bool, bool, Option<KGpuPrefix>) {
    let bpack = pack && zc_k_bone_pack_enabled();
    if bpack {
        if let Some(r) =
            fold2_compact_and_round45_into_packed_b(compact, table, rho1, rho2, r_next4, a_out, b_out, pack, true, gpu)
        {
            return r;
        }
    }
    fold2_compact_and_round45_into_packed_b(compact, table, rho1, rho2, r_next4, a_out, b_out, pack, false, false)
        .expect("a K pass without the b-one pack always completes")
}

/// [`fold2_compact_and_round45_into_packed`] with the b-one pack requested
/// by `bpack`; `None` when a b output the pack skips is not one (every output
/// is then rewritten by the caller's unpacked rerun).
#[allow(clippy::too_many_arguments)]
fn fold2_compact_and_round45_into_packed_b(
    compact: &UniSkipCompactFold,
    table: &UniSkipFoldTable,
    rho1: F128,
    rho2: F128,
    r_next4: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    pack: bool,
    bpack: bool,
    gpu: bool,
) -> Option<(F128, F128, Round3Lookahead, bool, bool, Option<KGpuPrefix>)> {
    let _ = gpu;
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
    assert!(!compact.tail_compact || compact.dead_tail);
    // Every K chunk's per-block rebase stays inside the buffers.
    assert!(!compact.tail_compact || compact.tail_base_blocks >= cascade_k_pairs_per_chunk(n_pairs) / 128);

    #[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
    let fused_k = cascade_k_fused_enabled();
    #[cfg(all(target_arch = "aarch64", not(target_feature = "sha3")))]
    let fused_k = false;
    #[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
    let one_table = fused_k && zc_k_one_table_enabled();
    #[cfg(not(all(target_arch = "aarch64", target_feature = "sha3")))]
    let one_table = false;
    assert!(
        !compact.tail_compact || one_table,
        "a tail-compacted state is read by the one-table fused K kernel only"
    );
    let (table_l1, table_l3) = if one_table {
        (table.scaled_linear(rho1), Vec::new())
    } else {
        let lambda1 = rho1 * (F128::ONE + rho2);
        let lambda3 = rho1 * rho2;
        (table.scaled_linear(lambda1), table.scaled_linear(lambda3))
    };

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
    let (kappa, retained_r_inv) =
        lookahead_kappa_and_retained_inv(r_par, lookahead_inverse_reuse_enabled());
    let eq_hi = &eq.hi;
    let eq_lo = &eq.lo;
    let out_chunk = 2 * lo_size;
    // A dead-pair state needs every chunk to start on a 128-pair block and
    // hold whole blocks (16 round-five groups each); the producer checked
    // the same split through `cascade_k_pairs_per_chunk`.
    assert!(
        !compact.dead_tail
            || (cfg!(target_arch = "aarch64")
                && 2 * out_chunk == cascade_k_pairs_per_chunk(n_pairs)
                && (2 * out_chunk).is_multiple_of(128)),
        "dead-pair compact state needs 128-pair-aligned K chunks"
    );
    #[cfg(target_arch = "aarch64")]
    let degen = r2_degen_enabled();

    // Weight each kernel output once. The odd-lane terms are shared by the
    // round-4 result and round-5 lookahead; kappa is common to all chunks and
    // can be applied after summing their even-lane terms.
    let mut partials: Vec<[F128; 8]> = vec![[F128::ZERO; 8]; hi_size];
    let a_base = crate::epool::SyncPtr(a_out.as_mut_ptr());
    let b_base = crate::epool::SyncPtr(b_out.as_mut_ptr());
    let partials_base = crate::epool::SyncPtr(partials.as_mut_ptr());
    // Packed: one-table fused kernel on a dead-tail state, every chunk a
    // whole number of four-block (256-output) superblocks.
    #[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
    let pack = pack && compact.dead_tail && one_table && out_chunk.is_multiple_of(256);
    #[cfg(not(all(target_arch = "aarch64", target_feature = "sha3")))]
    let pack = {
        let _ = pack;
        false
    };
    #[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
    let k_xb =
        one_table && kernels::aarch64::zc_k_xb_enabled() && kernels::aarch64::k_full_xb_weights(eq_lo, lo_size);
    #[cfg(not(all(target_arch = "aarch64", target_feature = "sha3")))]
    let k_xb = false;
    if std::env::var_os("FLOCK_ZC_K_XB_DEBUG").is_some() && one_table {
        let n_groups = lo_size / 2;
        let armed = k_xb;
        let first_window_ok = (0..n_groups.min(8)).all(|j| {
            j == 0 || eq_lo[2 * j + 1] == eq_lo[1] * F128::new(1u64 << (2 * j), 0)
        });
        eprintln!(
            "[zc-k-xb] armed={} n_groups={} first_window_ok={}",
            armed, n_groups, first_window_ok
        );
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
    let eq_prep = if one_table && !k_xb { kernels::aarch64::round2_eq_prep(eq_lo) } else { Vec::new() };
    // The scaled table in seven pieces (`FLOCK_ZC_K_FOLD7=1`, off by default) for the packed
    // chunks on the performance-core workers: one lookup fewer per tree, the
    // same values (none when a byte bank is not linear).
    #[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
    let fold7 = if one_table && pack && kernels::aarch64::zc_k_fold7_enabled() {
        kernels::aarch64::r2_fold7_table(&table_l1)
    } else {
        None
    };
    #[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
    let fold7_addr = fold7.as_ref().map_or(0usize, |t| t.as_ptr() as usize);
    // b-one pack (packed outputs only): each 64-output block's b takes
    // `B_PACK_BLOCK` entries; a kernel that meets a skipped b output that is
    // not one says so, and the whole pass is rerun without it.
    let bpack = bpack && pack;
    #[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
    let b_block = kernels::aarch64::B_PACK_BLOCK;
    #[cfg(not(all(target_arch = "aarch64", target_feature = "sha3")))]
    let b_block = 61;
    let b_failed = std::sync::atomic::AtomicBool::new(false);
    // K pass GPU prefix arm (`FLOCK_NO_ZC_K_GPU` kills): the GPU folds the
    // fixed leading chunks `0..cpu_base` into its own buffers, submitted
    // before the CPU queue, which starts at `cpu_base`. Only on the
    // tail-compacted b-one-packed pass (the arm's layouts). During the arm's
    // first launch (oracle probe) `cpu_base == 0` and the queue is exactly the
    // incumbent one.
    #[cfg(all(target_os = "macos", target_arch = "aarch64", target_feature = "sha3"))]
    let gpu_job = {
        let share = crate::gpu_commit::gpu_zc_k_share().min(hi_size);
        if gpu && bpack && compact.tail_compact && share > 0 {
            let share = crate::gpu_commit::zc_k_launch_share().min(hi_size);
            let hc = compact.head_compact;
            let layout = crate::gpu_commit::ZcKLayout {
                hc,
                a_base: compact.anchor_index(0),
                a_stride: if hc { kernels::aarch64::HC_ANCHORS_PER_BLOCK } else { kernels::aarch64::TC_ANCHORS_PER_BLOCK },
                a_shift: if hc { kernels::aarch64::HC_ANCHOR_SHIFT } else { 0 },
                d_base: compact.delta_offset(0),
                d_stride: if hc {
                    kernels::aarch64::HC_DELTA_BYTES_PER_BLOCK
                } else {
                    kernels::aarch64::TC_DELTA_BYTES_PER_BLOCK
                },
                d_shift: if hc { kernels::aarch64::HC_DELTA_SHIFT } else { 0 },
            };
            crate::gpu_commit::launch_zc_k_prefix(
                &compact.anchors,
                &compact.deltas,
                &layout,
                &table_l1,
                rho2,
                compact.head_b,
                eq_lo,
                eq_hi,
                share,
                out_chunk / 256 * 244,
                out_chunk / 64 * b_block,
            )
        } else {
            None
        }
    };
    #[cfg(all(target_os = "macos", target_arch = "aarch64", target_feature = "sha3"))]
    let cpu_base = gpu_job.as_ref().map_or(0, crate::gpu_commit::ZcKJob::cpu_base);
    #[cfg(not(all(target_os = "macos", target_arch = "aarch64", target_feature = "sha3")))]
    let cpu_base = 0usize;
    let chunk_body = |x_hi: usize| {
        // SAFETY: exclusive per-chunk ownership; the queue join publishes the
        // writes before the reduction below reads them. Packed: each of a
        // chunk's `out_chunk / 256` superblocks takes 244 entries.
        let (a_ptr, b_ptr) = unsafe {
            let o = if pack { x_hi * out_chunk / 256 * 244 } else { x_hi * out_chunk };
            let ob = if bpack { x_hi * out_chunk / 64 * b_block } else { o };
            (a_base.ptr().add(o), b_base.ptr().add(ob))
        };
        // Each output chunk covers `out_chunk` groups = 2·out_chunk pairs.
        let pair_base = 2 * x_hi * out_chunk;
        let mut outv = [F128::ZERO; 8];

        #[cfg(target_arch = "aarch64")]
        unsafe {
            // Tail-compacted state: whole 128-pair blocks of 244 anchors and
            // 1984 delta bytes (chunks start on a block, asserted above).
            #[cfg(target_feature = "sha3")]
            let (anchors, deltas) = if compact.tail_compact {
                // `anchor_index` / `delta_offset` of the chunk's first pair.
                (
                    compact.anchors.as_ptr().add(compact.anchor_index(pair_base)),
                    compact.deltas.as_ptr().add(compact.delta_offset(pair_base)),
                )
            } else {
                (
                    compact.anchors.as_ptr().add(2 * pair_base),
                    compact.deltas.as_ptr().add(pair_base * table.n_chunks * 2),
                )
            };
            #[cfg(not(target_feature = "sha3"))]
            let (anchors, deltas) = (
                compact.anchors.as_ptr().add(2 * pair_base),
                compact.deltas.as_ptr().add(pair_base * table.n_chunks * 2),
            );
            let (t1, t3) = (
                table_l1.as_ptr().cast::<u8>(),
                table_l3.as_ptr().cast::<u8>(),
            );
            let eq_ptr = eq_lo.as_ptr();
            #[cfg(target_feature = "sha3")]
            let prep_ptr = if k_xb { eq_lo.as_ptr() } else { eq_prep.as_ptr() };
            let out_ptr = outv.as_mut_ptr();
            match (fused_k, compact.dead_tail) {
                #[cfg(target_feature = "sha3")]
                (true, true) if one_table && pack => {
                    let ok = if fold7_addr != 0 && !kernels::aarch64::r2_on_efficiency_helper() {
                        if k_xb {
                            kernels::aarch64::fold2_compact_and_round45_chunk_neon_8_fused_r1_pieces7::<true>(
                                fold7_addr as *const u8, rho2, anchors, deltas, a_ptr, b_ptr, prep_ptr, lo_size,
                                degen, out_ptr, bpack, compact.tail_compact, compact.head_compact, compact.head_b,
                            )
                        } else {
                            kernels::aarch64::fold2_compact_and_round45_chunk_neon_8_fused_r1_pieces7::<false>(
                                fold7_addr as *const u8, rho2, anchors, deltas, a_ptr, b_ptr, prep_ptr, lo_size,
                                degen, out_ptr, bpack, compact.tail_compact, compact.head_compact, compact.head_b,
                            )
                        }
                    } else {
                        if k_xb {
                            fold2_compact_and_round45_chunk_neon_8_fused_r1::<true, true, false, true>(
                                t1, rho2, anchors, deltas, a_ptr, b_ptr, prep_ptr, lo_size, degen, out_ptr, bpack,
                                compact.tail_compact, compact.head_compact, compact.head_b,
                            )
                        } else {
                            fold2_compact_and_round45_chunk_neon_8_fused_r1::<true, true, false, false>(
                                t1, rho2, anchors, deltas, a_ptr, b_ptr, prep_ptr, lo_size, degen, out_ptr, bpack,
                                compact.tail_compact, compact.head_compact, compact.head_b,
                            )
                        }
                    };
                    if !ok {
                        b_failed.store(true, std::sync::atomic::Ordering::Relaxed);
                    }
                }
                #[cfg(target_feature = "sha3")]
                (true, true) if one_table => {
                    if k_xb {
                        fold2_compact_and_round45_chunk_neon_8_fused_r1::<true, false, false, true>(
                            t1, rho2, anchors, deltas, a_ptr, b_ptr, prep_ptr, lo_size, degen, out_ptr, false,
                            compact.tail_compact, compact.head_compact, compact.head_b,
                        );
                    } else {
                        fold2_compact_and_round45_chunk_neon_8_fused_r1::<true, false, false, false>(
                            t1, rho2, anchors, deltas, a_ptr, b_ptr, prep_ptr, lo_size, degen, out_ptr, false,
                            compact.tail_compact, compact.head_compact, compact.head_b,
                        );
                    }
                }
                #[cfg(target_feature = "sha3")]
                (true, false) if one_table => {
                    if k_xb {
                        fold2_compact_and_round45_chunk_neon_8_fused_r1::<false, false, false, true>(
                            t1, rho2, anchors, deltas, a_ptr, b_ptr, prep_ptr, lo_size, degen, out_ptr, false,
                            false, false, F128::ZERO,
                        );
                    } else {
                        fold2_compact_and_round45_chunk_neon_8_fused_r1::<false, false, false, false>(
                            t1, rho2, anchors, deltas, a_ptr, b_ptr, prep_ptr, lo_size, degen, out_ptr, false,
                            false, false, F128::ZERO,
                        );
                    }
                }
                #[cfg(target_feature = "sha3")]
                (true, true) => fold2_compact_and_round45_chunk_neon_8_fused::<true>(
                    t1, t3, rho2, anchors, deltas, a_ptr, b_ptr, eq_ptr, lo_size, degen, out_ptr,
                ),
                #[cfg(target_feature = "sha3")]
                (true, false) => fold2_compact_and_round45_chunk_neon_8_fused::<false>(
                    t1, t3, rho2, anchors, deltas, a_ptr, b_ptr, eq_ptr, lo_size, degen, out_ptr,
                ),
                (_, true) => fold2_compact_and_round45_chunk_neon_8::<true>(
                    t1, t3, rho2, anchors, deltas, a_ptr, b_ptr, eq_ptr, lo_size, degen, out_ptr,
                ),
                (_, false) => fold2_compact_and_round45_chunk_neon_8::<false>(
                    t1, t3, rho2, anchors, deltas, a_ptr, b_ptr, eq_ptr, lo_size, degen, out_ptr,
                ),
            }
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
        // SAFETY: each queue job owns its partial slot exclusively.
        unsafe {
            crate::field::f128_slice::scale_slice_into(&mut *partials_base.ptr().add(x_hi), &outv, eq_h);
        }
    };
    crate::epool::run_hetero_chunks(hi_size - cpu_base, |i| chunk_body(cpu_base + i));

    // Join the GPU prefix (it was submitted first and is normally done).
    #[allow(unused_mut)]
    let mut gpu_prefix: Option<KGpuPrefix> = None;
    #[cfg(all(target_os = "macos", target_arch = "aarch64", target_feature = "sha3"))]
    if let Some(job) = gpu_job {
        let calibration = job.calibration();
        let chunks = job.chunks();
        match crate::gpu_commit::zc_k_wait(job, hi_size) {
            crate::gpu_commit::ZcKResult::Prefix { parts, a, b, len_a, len_b, b_ones } => {
                if !b_ones {
                    // A b output of the GPU's chunks that the pack skips is
                    // not one: the pass reruns without the b-one pack, as
                    // for the CPU's chunks.
                    b_failed.store(true, std::sync::atomic::Ordering::Relaxed);
                } else if calibration {
                    // Oracle probe: the CPU folded every chunk; the GPU's
                    // outputs and weighted sums must equal the CPU's.
                    if !b_failed.load(std::sync::atomic::Ordering::Relaxed) {
                        // SAFETY: the joined job's buffers hold `len_a` / `len_b` entries.
                        let (ga, gb) = unsafe {
                            (std::slice::from_raw_parts(a, len_a), std::slice::from_raw_parts(b, len_b))
                        };
                        let equal = len_a <= a_out.len()
                            && len_b <= b_out.len()
                            && ga == &a_out[..len_a]
                            && gb == &b_out[..len_b]
                            && parts.as_slice() == &partials[..chunks];
                        crate::gpu_commit::zc_k_verdict(equal, chunks, hi_size);
                        // With the 5+6 GPU arm on, an admitted probe hands its
                        // buffers (byte-equal to the CPU's outputs above) to
                        // the 5+6 pass, so that arm's own first launch, its
                        // oracle, chains on them in this prove too.
                        if equal && crate::gpu_commit::gpu_zc_t56_enabled() {
                            gpu_prefix = Some(KGpuPrefix { a, b, len_a, len_b });
                        }
                    }
                } else {
                    partials[..chunks].copy_from_slice(&parts);
                    gpu_prefix = Some(KGpuPrefix { a, b, len_a, len_b });
                }
            }
            crate::gpu_commit::ZcKResult::Failed => {
                // The GPU failed after admission: the CPU folds the prefix
                // chunks with the same kernel (the arm is now disabled).
                crate::epool::run_hetero_chunks(cpu_base, &chunk_body);
            }
        }
    }
    if b_failed.into_inner() {
        return None;
    }

    let mut agg = [F128::ZERO; 8];
    for slot in &partials {
        for (a, v) in agg.iter_mut().zip(slot.iter()) {
            *a += *v;
        }
    }
    let sum1 = kappa * agg[0] + agg[2];
    let sum_inf = kappa * agg[1] + agg[3];
    // Every aggregate was accumulated on the odd lane's weight `r'·eq₅`, so a
    // single `r'⁻¹` puts all six back on `eq₅`.
    let r_inv = retained_r_inv.unwrap_or_else(|| r_par.inv());
    let w1 = r_inv * agg[2];
    let w2 = r_inv * agg[3];
    let w0 = r_inv * agg[4];
    let w3 = r_inv * agg[5];
    let w4 = r_inv * agg[6];
    let w5 = r_inv * agg[7];
    let la5 = Round3Lookahead {
        c: [w0, w0 + w1 + w2, w2, w3, w3 + w4 + w5, w5],
    };

    Some((r_next4[0] * sum1, sum_inf, la5, pack, bpack, gpu_prefix))
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
    let (kappa, retained_r_inv) =
        lookahead_kappa_and_retained_inv(r_par, lookahead_inverse_reuse_enabled());

    let chunk_in = 8 * lo_size; // four inputs per composed output
    let chunk_out = 2 * lo_size;
    let eq_lo = &eq.lo;
    let eq_hi = &eq.hi;

    if quarter >= (1usize << 20) && hi_size >= 8 {
        static DUMPED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if let Some(d) = std::env::var_os("FLOCK_ZK_DUMP")
            && !DUMPED.swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            let d = std::path::PathBuf::from(d);
            let _ = std::fs::create_dir_all(&d);
            let bytes = |v: &[F128]| -> Vec<u8> {
                v.iter().flat_map(|x| [x.lo.to_le_bytes(), x.hi.to_le_bytes()].concat()).collect()
            };
            let x0 = hi_size / 2;
            let rg = x0 * chunk_in..(x0 + 4) * chunk_in;
            let _ = std::fs::write(d.join("t3_a.bin"), bytes(&a[rg.clone()]));
            let _ = std::fs::write(d.join("t3_b.bin"), bytes(&b[rg]));
            let _ = std::fs::write(d.join("t3_eqlo.bin"), bytes(eq_lo));
            let _ = std::fs::write(d.join("t3_rho.bin"), bytes(&[rho3, rho4]));
        }
    }

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

    let mut agg = [F128::ZERO; 8];
    for (x_hi, outv) in partials.iter().enumerate() {
        let eq_h = eq_hi[x_hi];
        // The odd terms are shared by the round-6 sum and round-7 lookahead.
        for (aslot, &v) in agg.iter_mut().zip(outv.iter()) {
            *aslot += eq_h * v;
        }
    }
    let sum1 = kappa * agg[0] + agg[2];
    let sum_inf = kappa * agg[1] + agg[3];
    // Every aggregate was accumulated on the odd lane's weight `r''·eq₇`, so
    // a single `r''⁻¹` puts all six back on `eq₇`.
    let r_inv = retained_r_inv.unwrap_or_else(|| r_par.inv());
    let w1 = r_inv * agg[2];
    let w2 = r_inv * agg[3];
    let w0 = r_inv * agg[4];
    let w3 = r_inv * agg[5];
    let w4 = r_inv * agg[6];
    let w5 = r_inv * agg[7];
    let la7 = Round3Lookahead {
        c: [w0, w0 + w1 + w2, w2, w3, w3 + w4 + w5, w5],
    };

    (r_next6[0] * sum1, sum_inf, la7)
}

// ---------------------------------------------------------------------------
//
//
//
//
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lookahead2 {
    pub g1: [F128; 9],
    pub ginf: [F128; 9],
}

#[inline]
pub fn eval_lookahead2(la: &Lookahead2, u: F128, v: F128) -> (F128, F128) {
    let quad = |g0: F128, g1: F128, gi: F128, t: F128| g0 + t * (g0 + g1 + gi) + t * t * gi;
    let ev = |g: &[F128; 9]| {
        let h0 = quad(g[0], g[1], g[2], v);
        let h1 = quad(g[3], g[4], g[5], v);
        let hi = quad(g[6], g[7], g[8], v);
        quad(h0, h1, hi, u)
    };
    (ev(&la.g1), ev(&la.ginf))
}

#[cfg(test)]
pub(crate) static ZC_TRI_FORCED_OFF: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub(crate) fn zc_tri_enabled() -> bool {
    #[cfg(test)]
    if ZC_TRI_FORCED_OFF.load(std::sync::atomic::Ordering::Relaxed) {
        return false;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var_os("FLOCK_NO_ZC_TRI").is_some_and(|v| v == *"1"))
}

#[inline]
fn tri_n_hi(n_vars: usize) -> usize {
    SplitEqGhash::MAX_N_HI.min(n_vars.saturating_sub(8))
}

pub(crate) fn tri_chunk_scalar(
    a_in: &[F128],
    b_in: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    rhos: &[F128],
    eq_lo: &[F128],
) -> [F128; 26] {
    let per = 1usize << rhos.len();
    let fold = |v: &[F128]| {
        let mut cur = v.to_vec();
        for &rho in rhos {
            cur = cur.chunks_exact(2).map(|p| p[0] + rho * (p[0] + p[1])).collect();
        }
        cur[0]
    };
    for o in 0..a_out.len() {
        a_out[o] = fold(&a_in[o * per..(o + 1) * per]);
        b_out[o] = fold(&b_in[o * per..(o + 1) * per]);
    }
    let mut acc = [F128::ZERO; 26];
    for y in 0..a_out.len() / 8 {
        let w = eq_lo[4 * y + 3];
        let aw: [F128; 8] = std::array::from_fn(|i| w * a_out[8 * y + i]);
        let b: [F128; 8] = std::array::from_fn(|i| b_out[8 * y + i]);
        for j in 0..4 {
            acc[j] += aw[2 * j + 1] * b[2 * j + 1];
            acc[4 + j] += (aw[2 * j] + aw[2 * j + 1]) * (b[2 * j] + b[2 * j + 1]);
        }
        for k in 0..2 {
            let o = 4 * k;
            acc[8 + k] += aw[o + 2] * b[o + 2];
            let (ea, eb) = (aw[o] + aw[o + 2], b[o] + b[o + 2]);
            let (oa, ob) = (aw[o + 1] + aw[o + 3], b[o + 1] + b[o + 3]);
            acc[10 + k] += ea * eb;
            acc[12 + k] += oa * ob;
            acc[14 + k] += (ea + oa) * (eb + ob);
        }
        acc[16] += aw[4] * b[4];
        let sa: [F128; 4] = std::array::from_fn(|i| aw[i] + aw[i + 4]);
        let sb: [F128; 4] = std::array::from_fn(|i| b[i] + b[i + 4]);
        let grid = |s: &[F128; 4], iu: usize, iv: usize| {
            let row = |v: usize| match iu {
                0 => s[2 * v],
                1 => s[2 * v + 1],
                _ => s[2 * v] + s[2 * v + 1],
            };
            match iv {
                0 => row(0),
                1 => row(1),
                _ => row(0) + row(1),
            }
        };
        for iu in 0..3 {
            for iv in 0..3 {
                acc[17 + 3 * iu + iv] += grid(&sa, iu, iv) * grid(&sb, iu, iv);
            }
        }
    }
    acc
}

///
pub(crate) fn fold_cascade_t_into(
    a: &[F128],
    b: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    rhos: &[F128],
    r_next: &[F128],
) -> (F128, F128, Round3Lookahead, Lookahead2) {
    fold_cascade_t_into_packed(a, b, false, false, a_out, b_out, rhos, r_next)
}

/// [`fold_cascade_t_into`] whose input, with `packed`, is in the
/// dead-tail-packed layout of [`fold2_compact_and_round45_into_packed`]
/// (depth two only): `a` / `b` hold 244 entries per 256 logical inputs and
/// logical inputs 61..63 of every 64 are zero. With `bpacked` (and
/// `packed`), `b` is in the K pass's b-one-packed layout instead: 56 entries
/// per 64 logical inputs, inputs 0..3 and 59 of every 64 are one.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fold_cascade_t_into_packed(
    a: &[F128],
    b: &[F128],
    packed: bool,
    bpacked: bool,
    a_out: &mut [F128],
    b_out: &mut [F128],
    rhos: &[F128],
    r_next: &[F128],
) -> (F128, F128, Round3Lookahead, Lookahead2) {
    fold_cascade_t_into_packed_prefix(a, b, packed, bpacked, None, a_out, b_out, rhos, r_next)
}

/// Entries `lo..hi` of an input whose leading `pre.len()` entries come from
/// `pre` instead of `main` (a range across the boundary is assembled).
fn prefixed_input<'a>(main: &'a [F128], pre: &'a [F128], lo: usize, hi: usize) -> std::borrow::Cow<'a, [F128]> {
    let p = pre.len();
    if hi <= p {
        pre[lo..hi].into()
    } else if lo >= p {
        main[lo..hi].into()
    } else {
        [&pre[lo..], &main[p..hi]].concat().into()
    }
}

/// [`fold_cascade_t_into_packed`] whose leading input entries come from
/// `prefix = Some((pa, pb))` (the K pass GPU prefix, [`KGpuPrefix`], or the
/// 5+6 one): entries `0..pa.len()` of `a` and `0..pb.len()` of `b` are read
/// there, not in `a` / `b`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fold_cascade_t_into_packed_prefix(
    a: &[F128],
    b: &[F128],
    packed: bool,
    bpacked: bool,
    prefix: Option<(&[F128], &[F128])>,
    a_out: &mut [F128],
    b_out: &mut [F128],
    rhos: &[F128],
    r_next: &[F128],
) -> (F128, F128, Round3Lookahead, Lookahead2) {
    let (m1, minf, la, la2, carried) =
        fold_cascade_t_into_packed_prefix_gpu(a, b, packed, bpacked, prefix, a_out, b_out, rhos, r_next, false);
    // With `gpu` off, a prefix comes back only from the tri pass 0 carry
    // (`gpu_commit::ENV_NO_ZC_T0_CARRY`); this wrapper's callers read every
    // output entry, so write the carried leading outputs in.
    if let Some(p) = carried {
        // SAFETY: the arm's buffers are rewritten only by the next prove's
        // pass 0; the carried chunks were observed complete.
        let (pa, pb) = unsafe { p.slices() };
        a_out[..pa.len()].copy_from_slice(pa);
        b_out[..pb.len()].copy_from_slice(pb);
    }
    (m1, minf, la, la2)
}

/// Allocate tri-pass partial rows without clearing storage that each chunk
/// overwrites in full. The legacy switch restores the original zero fill.
fn tri_partial_storage(len: usize) -> Vec<std::mem::MaybeUninit<[F128; 26]>> {
    use std::mem::MaybeUninit;
    #[cfg(test)]
    {
        let mode = TRI_PARTIAL_TEST_MODE.with(|m| m.get());
        if mode == 1 {
            return vec![MaybeUninit::new([F128::ZERO; 26]); len];
        }
        if mode == 2 {
            return (0..len)
                .map(|row| MaybeUninit::new(std::array::from_fn(|lane| {
                    F128::new(
                        0xa5a5_6b6b_1234_5678 ^ (row as u64).wrapping_mul(0x9e37_79b9),
                        0x5a5a_9494_fedc_ba98 ^ (lane as u64).wrapping_mul(0x85eb_ca6b),
                    )
                })))
                .collect();
        }
        if mode == 3 {
            return vec![MaybeUninit::uninit(); len];
        }
    }
    static UNCLEARED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var_os("FLOCK_NO_ZC_TRI_PARTIALS_UNINIT").is_none()
    });
    if *UNCLEARED {
        vec![MaybeUninit::uninit(); len]
    } else {
        vec![MaybeUninit::new([F128::ZERO; 26]); len]
    }
}

/// Borrow only the prefix already initialized by the completed CPU queue.
///
/// # Safety
/// Every row in `0..len` must be fully written and no concurrent writer may
/// touch that prefix for the returned borrow's lifetime.
unsafe fn tri_partial_prefix(
    rows: &[std::mem::MaybeUninit<[F128; 26]>],
    len: usize,
) -> &[[F128; 26]] {
    let prefix = &rows[..len];
    // SAFETY: caller establishes initialization and exclusive completion;
    // MaybeUninit<T> has exactly T's layout and alignment.
    unsafe { std::slice::from_raw_parts(prefix.as_ptr().cast(), prefix.len()) }
}

/// Reuse the allocation after all CPU/GPU owners have initialized their rows.
///
/// # Safety
/// Every element must be fully initialized, with all writes joined. Abandoned
/// incomplete passes must drop the MaybeUninit vector instead of calling this.
unsafe fn tri_partials_assume_init(
    rows: Vec<std::mem::MaybeUninit<[F128; 26]>>,
) -> Vec<[F128; 26]> {
    let mut rows = std::mem::ManuallyDrop::new(rows);
    // SAFETY: caller establishes initialization; element layouts, lengths and
    // capacities are identical, and ManuallyDrop transfers the allocation once.
    let initialized: Vec<[F128; 26]> = unsafe {
        Vec::from_raw_parts(rows.as_mut_ptr().cast(), rows.len(), rows.capacity())
    };
    #[cfg(test)]
    if TRI_PARTIAL_TEST_MODE.with(|m| m.get() != 0) {
        TRI_PARTIAL_TEST_ROWS.with(|capture| *capture.borrow_mut() = Some(initialized.clone()));
    }
    initialized
}

#[cfg(test)]
std::thread_local! {
    static TRI_PARTIAL_TEST_MODE: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
    static TRI_PARTIAL_TEST_ROWS: std::cell::RefCell<Option<Vec<[F128; 26]>>> = const { std::cell::RefCell::new(None) };
}

/// Tri pass 0's claim queue with the GPU arm (see
/// `gpu_commit::ENV_NO_ZC_T0_GPU`). `hi_size` claims, one per queue job,
/// each under one mutex: `(g0, g1)` = (next chunk to take from the GPU,
/// lowest chunk the CPU has taken), always `g0 < g1` before a claim, so each
/// chunk is claimed once. A claim takes `g0` from the GPU when its encoder
/// has signalled completion (the job's `chunk_ready`), else folds `g1 - 1`
/// on the CPU and publishes that boundary so the GPU skips it. A GPU chunk's
/// outputs are copied into its output ranges and its 26 slots into its
/// partial row, exactly what `body` writes for it. At the arm's first
/// launch (oracle) the CPU folds every chunk and the GPU's outputs and
/// slots of every offered chunk must equal the CPU's. `FLOCK_ZC_T0_FORCE`
/// (test hook) keeps chunks below it for the GPU and waits for them.
///
/// Carry (`gpu_commit::ENV_NO_ZC_T0_CARRY`): outside the oracle and the
/// test hook, the chunks taken from the GPU are exactly `0..taken` (the GPU
/// side of the queue only advances `g0` from 0, one ready chunk at a time).
/// Their outputs are then not copied: they stay in the arm's buffers, and
/// the returned prefix (`taken * chunk_out` entries of each) is what tri
/// pass 1 reads in place of the leading entries of this pass's `a_out` /
/// `b_out`, which are left unwritten there. `None` = nothing carried (every
/// output range written, as before).
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[allow(clippy::too_many_arguments)]
fn tri0_gpu_queue(
    job: crate::gpu_commit::ZcT0Job,
    hi_size: usize,
    chunk_out: usize,
    a_base: crate::epool::SyncPtr<F128>,
    b_base: crate::epool::SyncPtr<F128>,
    partials_base: crate::epool::SyncPtr<std::mem::MaybeUninit<[F128; 26]>>,
    body: &(dyn Fn(usize) + Sync),
    weigh: Option<&[F128]>,
) -> Option<KGpuPrefix> {
    let calibration = job.calibration();
    // Raw rows for the oracle below.
    let weigh = weigh.filter(|_| !calibration);
    let force = if calibration { 0 } else { crate::gpu_commit::gpu_zc_t0_force().min(job.offer()) };
    let carry = !calibration && force == 0 && crate::gpu_commit::gpu_zc_t0_carry_enabled();
    let claims = std::sync::Mutex::new((0usize, hi_size, 0usize));
    crate::epool::run_hetero_chunks(hi_size, |_| {
        let (x_hi, from_gpu) = {
            let mut g = crate::lock_spin(&claims).unwrap_or_else(|e| e.into_inner());
            assert!(g.0 < g.1, "tri pass 0 claim queue overrun");
            if !calibration && job.chunk_ready(g.0) {
                g.0 += 1;
                g.2 += 1;
                (g.0 - 1, true)
            } else if !calibration && g.1 <= force {
                // Test hook: the rest is the GPU's; wait for it.
                g.0 += 1;
                if job.wait_chunk(g.0 - 1) {
                    g.2 += 1;
                    (g.0 - 1, true)
                } else {
                    (g.0 - 1, false)
                }
            } else {
                g.1 -= 1;
                if !calibration {
                    job.set_cut(g.1);
                }
                (g.1, false)
            }
        };
        if from_gpu {
            let (ga, gb, slots) = job.chunk_out(x_hi);
            // Tri partials weighted on their producers: this worker weighs
            // the GPU's row (`FLOCK_NO_TRI_WEIGHT_WORKERS`).
            let slots = match weigh {
                Some(eq_hi) => tri_weigh(eq_hi[x_hi], slots),
                None => slots,
            };
            // SAFETY: exclusive owner of chunk `x_hi`'s output ranges and
            // partial slot (claimed above); the queue join publishes them.
            unsafe {
                if !carry {
                    std::ptr::copy_nonoverlapping(ga.as_ptr(), a_base.ptr().add(x_hi * chunk_out), chunk_out);
                    std::ptr::copy_nonoverlapping(gb.as_ptr(), b_base.ptr().add(x_hi * chunk_out), chunk_out);
                }
                partials_base.ptr().add(x_hi).write(std::mem::MaybeUninit::new(slots));
            }
        } else {
            body(x_hi);
        }
    });
    let (g0, _, taken) = claims.into_inner().unwrap_or_else(|e| e.into_inner());
    // Carry: the GPU-taken chunks are `0..g0` (no forced claims), all of
    // them observed complete; hand them to the next pass uncopied.
    let carried = if carry && taken > 0 {
        assert_eq!(g0, taken, "tri pass 0 carry: GPU chunks are not a prefix");
        let (a, b) = job.outputs();
        if crate::gpu_commit::gpu_zc_t0_debug() {
            eprintln!(
                "[zc-t0-gpu] carry: chunks 0..{taken} handed to tri pass 1 uncopied ({} KiB not copied)",
                taken * 2 * chunk_out * std::mem::size_of::<F128>() / 1024
            );
        }
        Some(KGpuPrefix { a, b, len_a: taken * chunk_out, len_b: taken * chunk_out })
    } else {
        None
    };
    if calibration {
        let equal = job.wait_all()
            && (0..job.offer()).all(|c| {
                let (ga, gb, slots) = job.chunk_out(c);
                // SAFETY: the queue has drained; no other access.
                unsafe {
                    std::slice::from_raw_parts(a_base.ptr().add(c * chunk_out).cast_const(), chunk_out) == ga
                        && std::slice::from_raw_parts(b_base.ptr().add(c * chunk_out).cast_const(), chunk_out) == gb
                        && (&*partials_base.ptr().add(c)).assume_init_ref() == &slots
                }
            });
        crate::gpu_commit::zc_t0_verdict(equal, job.offer());
    }
    job.finish(taken, hi_size);
    carried
}

/// Tri partials weighted on their producers (`FLOCK_NO_TRI_WEIGHT_WORKERS`, any
/// value, keeps the record's combine). After a tri pass's chunks have joined,
/// the record's caller multiplies every chunk's 26 raw slots by the chunk's
/// `eq_hi` weight and sums them, serially (`hi_size * 26` products per pass).
/// With this on, the worker that produces a chunk's row (the CPU fold, the
/// steal, the tri pass 0 queue's copy of a ready GPU chunk) stores it already
/// weighted by the same products, the caller weights only the rows it copies
/// from an admitted GPU prefix, and the combine is a sum. Same products, the
/// same terms summed in GF(2^128) (exact, order-free), so the same 26 values.
/// Calibration launches (raw-row oracles) keep raw rows. Read once per process.
fn tri_weight_workers_enabled() -> bool {
    use std::sync::LazyLock;
    static ON: LazyLock<bool> = LazyLock::new(|| std::env::var_os("FLOCK_NO_TRI_WEIGHT_WORKERS").is_none());
    *ON
}

/// A chunk's 26 tri slots times its `eq_hi` weight (the record's combine's
/// products).
#[inline]
fn tri_weigh(eq_h: F128, v: [F128; 26]) -> [F128; 26] {
    let mut w = v;
    for x in w.iter_mut() {
        *x = eq_h * *x;
    }
    w
}

/// The 26 tri sums: of weighted rows (`weighted`) or of raw rows times their
/// `eq_hi` weights (the record's combine).
fn tri_combine(partials: &[[F128; 26]], eq_hi: &[F128], weighted: bool) -> [F128; 26] {
    let mut s = [F128::ZERO; 26];
    if weighted {
        for outv in partials {
            for (slot, &v) in s.iter_mut().zip(outv.iter()) {
                *slot += v;
            }
        }
    } else {
        for (x_hi, outv) in partials.iter().enumerate() {
            let eq_h = eq_hi[x_hi];
            for (slot, &v) in s.iter_mut().zip(outv.iter()) {
                *slot += eq_h * v;
            }
        }
    }
    s
}

/// `FLOCK_ZC_TAIL_ROUND_TIMING` (read once): the tri combine's timing line.
fn tri_combine_timing() -> bool {
    use std::sync::LazyLock;
    static ON: LazyLock<bool> = LazyLock::new(|| std::env::var_os("FLOCK_ZC_TAIL_ROUND_TIMING").is_some());
    *ON
}

/// Caller-first main-pool tri passes (`FLOCK_NO_ZC_TRI_CF`, any value,
/// restores the rayon `par_chunks_mut` collect; c1fbbc9e's piece). The tri
/// passes below the hetero floor run through
/// [`crate::epool::run_caller_first_chunks`]: the calling thread starts on
/// chunk 0 at once and never parks on a pool latch. Same `x_hi` chunk grid
/// and chunk closure; each chunk writes its own output range and its own
/// `partials[x_hi]`, combined in index order, so bytes are identical. Read
/// once per process.
fn zc_tri_caller_first_enabled() -> bool {
    use std::sync::LazyLock;
    static ON: LazyLock<bool> = LazyLock::new(|| std::env::var_os("FLOCK_NO_ZC_TRI_CF").is_none());
    *ON
}

/// [`fold_cascade_t_into_packed_prefix`] with the 5+6 GPU prefix arm when
/// `gpu` is set (see `gpu_commit::ENV_NO_ZC_T56_GPU`): on the ranked packed
/// b-one-packed depth-two pass whose `prefix` is the K arm's, the GPU folds
/// the leading chunks that lie wholly inside that prefix. Also returns the
/// arm's prefix (`None` unless it was used for this pass's outputs): its
/// `len` leading outputs of `a_out` / `b_out` were not written, and the next
/// tri pass reads them from there. Tri pass 0 returns its own GPU arm's
/// prefix the same way when it carries (`gpu_commit::ENV_NO_ZC_T0_CARRY`),
/// whatever `gpu` is.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fold_cascade_t_into_packed_prefix_gpu(
    a: &[F128],
    b: &[F128],
    packed: bool,
    bpacked: bool,
    prefix: Option<(&[F128], &[F128])>,
    a_out: &mut [F128],
    b_out: &mut [F128],
    rhos: &[F128],
    r_next: &[F128],
    gpu: bool,
) -> (F128, F128, Round3Lookahead, Lookahead2, Option<KGpuPrefix>) {
    let _ = gpu;
    let depth = rhos.len();
    assert!(depth == 2 || depth == 3);
    let bpacked = packed && bpacked;
    let n = if packed {
        assert_eq!(depth, 2, "a packed input is the K pass output (depth two)");
        assert!(a.len().is_multiple_of(244));
        a.len() / 244 * 256
    } else {
        a.len()
    };
    if bpacked {
        assert_eq!(b.len(), n / 64 * 56);
    } else {
        assert_eq!(b.len(), a.len());
    }
    let out_n = n >> depth;
    assert!(n.is_power_of_two() && out_n >= 8);
    assert_eq!(a_out.len(), out_n);
    assert_eq!(b_out.len(), out_n);
    assert_eq!(r_next.len(), out_n.trailing_zeros() as usize);
    let (r1, r2) = (r_next[1], r_next[2]);
    assert!(r1 != F128::ZERO && r2 != F128::ZERO, "tri cascade needs r1, r2 ≠ 0");

    let n_vars = r_next.len() - 1;
    let eq = SplitEqGhash::with_n_hi(&r_next[1..], tri_n_hi(n_vars));
    let lo_size = 1usize << eq.n_lo;
    let hi_size = 1usize << eq.n_hi;
    assert!(lo_size >= 4);
    assert_eq!(lo_size * hi_size * 2, out_n);
    let chunk_out = 2 * lo_size;
    let chunk_in = chunk_out << depth;
    let eq_lo = &eq.lo;
    let eq_hi = &eq.hi;

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    let nt_stores = {
        static NT_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        out_n >= (1usize << 20)
            && *NT_ENABLED.get_or_init(|| std::env::var_os("FLOCK_ZC_NT_LEGACY").is_none())
    };
    let rho_c = if depth == 3 { rhos[2] } else { F128::ZERO };
    // The group weights `eq_lo[4y + 3]` as prepared pairs, one table per
    // pass shared by every chunk.
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    let eq_prep = kernels::aarch64::round2_eq_prep(eq_lo);

    // Chunk `x_hi`'s input entries: `chunk_in` logical inputs, 244 entries
    // per 256 of them when packed.
    let per_chunk = if packed {
        assert!(chunk_in.is_multiple_of(256));
        chunk_in / 256 * 244
    } else {
        chunk_in
    };
    // The packed tri kernel tells the b layout from the slice length.
    let per_chunk_b = if bpacked { chunk_in / 64 * 56 } else { per_chunk };
    let (pre_a, pre_b) = prefix.unwrap_or((&[], &[]));
    assert!(pre_a.len() <= a.len() && pre_b.len() <= b.len());
    let inputs = |x_hi: usize| -> (std::borrow::Cow<'_, [F128]>, std::borrow::Cow<'_, [F128]>) {
        (
            prefixed_input(a, pre_a, x_hi * per_chunk, (x_hi + 1) * per_chunk),
            prefixed_input(b, pre_b, x_hi * per_chunk_b, (x_hi + 1) * per_chunk_b),
        )
    };
    let chunk = |a_in: &[F128], b_in: &[F128], a_o: &mut [F128], b_o: &mut [F128]| -> [F128; 26] {
        #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
        {
            if packed {
                use kernels::aarch64::fold_tri_and_message_packed_aarch64 as kp;
                return if nt_stores {
                    kp::<true>(a_in, b_in, a_o, b_o, rhos[0], rhos[1], eq_lo, &eq_prep)
                } else {
                    kp::<false>(a_in, b_in, a_o, b_o, rhos[0], rhos[1], eq_lo, &eq_prep)
                };
            }
            use kernels::aarch64::fold_tri_and_message_aarch64 as k;
            let ep = &eq_prep;
            match (depth, nt_stores) {
                (2, true) => k::<2, true, 0>(a_in, b_in, a_o, b_o, rhos[0], rhos[1], rho_c, eq_lo, ep),
                (2, false) => k::<2, false, 0>(a_in, b_in, a_o, b_o, rhos[0], rhos[1], rho_c, eq_lo, ep),
                (_, true) => k::<3, true, 0>(a_in, b_in, a_o, b_o, rhos[0], rhos[1], rho_c, eq_lo, ep),
                (_, false) => k::<3, false, 0>(a_in, b_in, a_o, b_o, rhos[0], rhos[1], rho_c, eq_lo, ep),
            }
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
        {
            assert!(!packed, "the packed K output is produced on AArch64 only");
            let _ = rho_c;
            tri_chunk_scalar(a_in, b_in, a_o, b_o, rhos, eq_lo)
        }
    };

    #[cfg(target_arch = "aarch64")]
    let hetero = out_n >= zc_tail_hetero_low_floor() && zc_tail_hetero_enabled();
    #[cfg(not(target_arch = "aarch64"))]
    let hetero = false;

    #[allow(unused_mut)]
    let mut gpu_prefix: Option<KGpuPrefix> = None;
    #[cfg(test)]
    let hetero = hetero || TRI_PARTIAL_TEST_MODE.with(|m| m.get() != 0);
    // Tri partials weighted on their producers (`FLOCK_NO_TRI_WEIGHT_WORKERS`).
    let ww = tri_weight_workers_enabled();
    let timing = tri_combine_timing();
    #[allow(unused_mut)]
    let mut rows_us = 0.0f64;
    let (partials, weighted): (Vec<[F128; 26]>, bool) = if hetero {
        let mut partials = tri_partial_storage(hi_size);
        let a_base = crate::epool::SyncPtr(a_out.as_mut_ptr());
        let b_base = crate::epool::SyncPtr(b_out.as_mut_ptr());
        let partials_base = crate::epool::SyncPtr(partials.as_mut_ptr());
        // 5+6 GPU prefix arm (`FLOCK_NO_ZC_T56_GPU` kills): the GPU folds
        // chunks `0..cpu_base` (those wholly inside the K arm's prefix) from
        // the K arm's buffers into its own, submitted before the CPU queue,
        // which starts at `cpu_base`. During the arm's first launch (oracle)
        // `cpu_base == 0` and the queue is exactly the incumbent one.
        // Late-GPU steal (`FLOCK_NO_ZC_T56_LOOP=1` keeps the record's join
        // below): in an admitted prove the launch, the share and the CPU
        // queue are the record's. If the GPU's buffer is complete when the
        // queue has drained, that is all. If not, the CPU does not wait: it
        // folds the GPU's chunks itself from the top of the share downward
        // and asks at every claim whether the buffer has completed.
        //
        // The seam: the boundary `lo` and the completion flag change only
        // under the claim lock; once completion is seen the boundary never
        // moves again. The kernel skips a chunk only at or above a boundary
        // value the CPU wrote, and every value written is the chunk the CPU
        // claimed with it, so a chunk below the final `lo` was never skipped
        // (the completed buffer folded it) and never claimed (no CPU write).
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        let loop_arm = match prefix {
            Some((pa, pb))
                if gpu
                    && crate::gpu_commit::zc_t56_loop_admitted()
                    && packed
                    && bpacked
                    && depth == 2
                    && hi_size == crate::gpu_commit::zc_t56::HI
                    && chunk_out == crate::gpu_commit::zc_t56::OUT
                    && per_chunk == crate::gpu_commit::zc_t56::IN_A
                    && per_chunk_b == crate::gpu_commit::zc_t56::IN_B =>
            {
                let chunks =
                    (pa.len() / per_chunk).min(pb.len() / per_chunk_b).min(crate::gpu_commit::gpu_zc_t56_cap());
                Some((pa, pb, chunks))
            }
            _ => None,
        };
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        if let Some((pa, pb, share)) = loop_arm {
            let fold = |x_hi: usize| {
                // SAFETY: exclusive per-chunk ownership (queue index or
                // claim); the queue join publishes the writes.
                let (a_o, b_o) = unsafe {
                    (
                        std::slice::from_raw_parts_mut(a_base.ptr().add(x_hi * chunk_out), chunk_out),
                        std::slice::from_raw_parts_mut(b_base.ptr().add(x_hi * chunk_out), chunk_out),
                    )
                };
                let (a_in, b_in) = inputs(x_hi);
                let outv = chunk(&a_in, &b_in, a_o, b_o);
                let outv = if ww { tri_weigh(eq_hi[x_hi], outv) } else { outv };
                // SAFETY: exclusive owner of this row; write a complete value
                // without forming a reference to uninitialized storage.
                unsafe {
                    partials_base.ptr().add(x_hi).write(std::mem::MaybeUninit::new(outv));
                }
            };
            // The record's launch and queue: the GPU's share first, the CPU
            // folds the chunks after it in ascending order.
            let job = crate::gpu_commit::launch_zc_t56_loop(pa, pb, share, rhos[0], rhos[1], eq_lo);
            let cpu_base = job.as_ref().map_or(0, |j| j.chunks());
            crate::epool::run_hetero_chunks(hi_size - cpu_base, |i| fold(cpu_base + i));
            if let Some(job) = job {
                let mut lo = cpu_base;
                let mut gpu_done = job.done();
                if !gpu_done {
                    // The GPU is late: steal from the top of its share.
                    let claims = std::sync::Mutex::new((cpu_base, false));
                    crate::epool::run_hetero_chunks(cpu_base, |_| {
                        let x_hi = {
                            let mut g = claims.lock().unwrap_or_else(|e| e.into_inner());
                            if g.1 || g.0 == 0 {
                                return;
                            }
                            if job.done() {
                                g.1 = true;
                                return;
                            }
                            // Test builds only (see `t56_loop_hold`, the
                            // constant 0 otherwise): do not steal below
                            // chunk `n`; wait for the GPU instead, so a slow
                            // GPU still keeps `min(n, share)` chunks and
                            // that path can be checked.
                            if g.0 <= t56_loop_hold() {
                                while !job.done() {
                                    std::hint::spin_loop();
                                }
                                g.1 = true;
                                return;
                            }
                            job.set_cut(g.0 - 1);
                            g.0 -= 1;
                            g.0
                        };
                        fold(x_hi);
                    });
                    (lo, gpu_done) = claims.into_inner().unwrap_or_else(|e| e.into_inner());
                    // Without the completion seen, the CPU folded every
                    // chunk and nothing of the GPU's is read.
                    assert!(gpu_done || lo == 0, "5+6 steal ended early");
                }
                // A buffer that signalled and still reports an error is not
                // trusted (the record's join checks the status too): the CPU
                // folds what it had left to the GPU; `finish` turns the arm
                // off.
                let gpu_ok = gpu_done && !job.failed();
                if gpu_done && !gpu_ok {
                    crate::epool::run_hetero_chunks(lo, &fold);
                }
                let taken = if gpu_ok { lo } else { 0 };
                let t_rows = timing.then(std::time::Instant::now);
                if ww && taken >= 64 {
                    // Tri partials weighted on their producers: the GPU's
                    // rows (each finished from its banks, then weighted) by
                    // the pool, eight rows per claim, instead of the caller.
                    let rows = |g: usize| {
                        for c in 8 * g..(8 * g + 8).min(taken) {
                            // SAFETY: row `c` belongs to claim `g` alone; the
                            // run's join publishes it.
                            unsafe {
                                partials_base
                                    .ptr()
                                    .add(c)
                                    .write(std::mem::MaybeUninit::new(tri_weigh(eq_hi[c], job.part(c))));
                            }
                        }
                    };
                    crate::epool::run_hetero_chunks_then(taken.div_ceil(8), &rows, &|| {});
                } else {
                    for c in 0..taken {
                        let part = job.part(c);
                        partials[c].write(if ww { tri_weigh(eq_hi[c], part) } else { part });
                    }
                }
                rows_us = t_rows.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e6);
                if taken > 0 {
                    let (a, b) = job.outputs();
                    gpu_prefix = Some(KGpuPrefix { a, b, len_a: taken * chunk_out, len_b: taken * chunk_out });
                }
                job.finish(taken, hi_size);
            }
            return {
                // SAFETY: the joined CPU suffix/steals and the copied GPU
                // prefix partition every row; failure paths recompute the rest.
                let partials = unsafe { tri_partials_assume_init(partials) };
                let t_comb = timing.then(std::time::Instant::now);
                let s = tri_combine(&partials, eq_hi, ww);
                let comb_us = t_comb.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e6);
                let (m1, minf, la, la2) = tri_finish(&s, r_next);
                if timing {
                    eprintln!(
                        "[zc-tail-rounds] tri combine (rows={hi_size}, weighted on workers={ww}): \
                         gpu rows {rows_us:.1} us + sum {comb_us:.1} us"
                    );
                }
                (m1, minf, la, la2, gpu_prefix)
            };
        }
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        let gpu_job = match prefix {
            Some((pa, pb))
                if gpu
                    && crate::gpu_commit::gpu_zc_t56_enabled()
                    && packed
                    && bpacked
                    && depth == 2
                    && hi_size == crate::gpu_commit::zc_t56::HI
                    && chunk_out == crate::gpu_commit::zc_t56::OUT
                    && per_chunk == crate::gpu_commit::zc_t56::IN_A
                    && per_chunk_b == crate::gpu_commit::zc_t56::IN_B =>
            {
                let chunks =
                    (pa.len() / per_chunk).min(pb.len() / per_chunk_b).min(crate::gpu_commit::gpu_zc_t56_cap());
                crate::gpu_commit::launch_zc_t56_prefix(pa, pb, chunks, rhos[0], rhos[1], eq_lo)
            }
            _ => None,
        };
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        let cpu_base = gpu_job.as_ref().map_or(0, crate::gpu_commit::ZcT56Job::cpu_base);
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        let cpu_base = 0usize;
        // One queue closure for both drains below (CPU chunks from `base`).
        let base = std::sync::atomic::AtomicUsize::new(cpu_base);
        // Tri partials weighted on their producers, unless a launch here is a
        // calibration (its oracle compares raw rows).
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        let ww_h = std::sync::atomic::AtomicBool::new(ww && !gpu_job.as_ref().is_some_and(|j| j.calibration()));
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        let ww_h = std::sync::atomic::AtomicBool::new(ww);
        let body = |i: usize| {
            let x_hi = base.load(std::sync::atomic::Ordering::Relaxed) + i;
            // SAFETY: exclusive per-chunk ownership; queue join publishes writes.
            let (a_o, b_o) = unsafe {
                (
                    std::slice::from_raw_parts_mut(a_base.ptr().add(x_hi * chunk_out), chunk_out),
                    std::slice::from_raw_parts_mut(b_base.ptr().add(x_hi * chunk_out), chunk_out),
                )
            };
            let (a_in, b_in) = inputs(x_hi);
            let outv = chunk(&a_in, &b_in, a_o, b_o);
            let outv =
                if ww_h.load(std::sync::atomic::Ordering::Relaxed) { tri_weigh(eq_hi[x_hi], outv) } else { outv };
            // SAFETY: exclusive owner; initialize the complete row by value.
            unsafe {
                partials_base.ptr().add(x_hi).write(std::mem::MaybeUninit::new(outv));
            }
        };
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        let t_cpu = std::time::Instant::now();
        // Tri pass 0 GPU arm (`FLOCK_NO_ZC_T0_GPU` kills): this pass's
        // leading inputs are the 5+6 arm's buffers; the GPU folds the chunks
        // wholly inside them from chunk 0 up while the queue claims from the
        // top, and the queue takes a chunk from the GPU only once its encoder
        // has signalled (see `tri0_gpu_queue`).
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        let t0_job = match prefix {
            Some((pa, pb))
                if depth == 3
                    && !packed
                    && cpu_base == 0
                    && crate::gpu_commit::gpu_zc_t0_enabled()
                    && hi_size == crate::gpu_commit::T0_HI
                    && chunk_out == crate::gpu_commit::T0_OUT
                    && chunk_in == crate::gpu_commit::T0_IN =>
            {
                crate::gpu_commit::zc_t0_fire(pa, pb, [rhos[0], rhos[1], rhos[2]], eq_lo)
            }
            _ => None,
        };
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        let t0_job: Option<()> = None;
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        if t0_job.as_ref().is_some_and(|j| j.calibration()) {
            ww_h.store(false, std::sync::atomic::Ordering::Relaxed);
        }
        match t0_job {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            Some(job) => {
                let weigh = if ww_h.load(std::sync::atomic::Ordering::Relaxed) { Some(eq_hi.as_slice()) } else { None };
                // Carry (`ENV_NO_ZC_T0_CARRY`): the GPU-taken leading
                // outputs stay in the arm's buffers; returned below for tri
                // pass 1, which reads them in place of `a_out` / `b_out`'s
                // leading entries (not written in that case).
                if let Some(p) = tri0_gpu_queue(job, hi_size, chunk_out, a_base, b_base, partials_base, &body, weigh) {
                    gpu_prefix = Some(p);
                }
            }
            _ => crate::epool::run_hetero_chunks(hi_size - cpu_base, &body),
        }
        // Join the GPU prefix (submitted first; normally done).
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        if let Some(job) = gpu_job {
            let calibration = job.calibration();
            let chunks = job.chunks();
            let cpu_ms = t_cpu.elapsed().as_secs_f64() * 1e3;
            match crate::gpu_commit::zc_t56_wait(job, hi_size, cpu_ms) {
                crate::gpu_commit::ZcT56Result::Prefix { parts, a, b, len } => {
                    if calibration {
                        // Oracle: the CPU folded every chunk; the GPU's
                        // outputs and slots must equal the CPU's.
                        // SAFETY: the joined job's buffers hold `len` entries each.
                        let (ga, gb) =
                            unsafe { (std::slice::from_raw_parts(a, len), std::slice::from_raw_parts(b, len)) };
                        let equal = len <= a_out.len()
                            && ga == &a_out[..len]
                            && gb == &b_out[..len]
                            // SAFETY: calibration uses cpu_base=0; the CPU
                            // queue has completed every row before this oracle.
                            && parts.as_slice() == unsafe { tri_partial_prefix(&partials, chunks) };
                        crate::gpu_commit::zc_t56_verdict(equal, chunks, hi_size);
                        // With the tri pass 0 arm on, an admitted oracle hands
                        // its buffers (byte-equal to the CPU's outputs above)
                        // to pass 0, so that arm's own first launch, its
                        // oracle, chains on them in this prove too.
                        if equal && crate::gpu_commit::gpu_zc_t0_enabled() {
                            gpu_prefix = Some(KGpuPrefix { a, b, len_a: len, len_b: len });
                        }
                    } else {
                        assert_eq!(parts.len(), chunks);
                        let ww_rows = ww_h.load(std::sync::atomic::Ordering::Relaxed);
                        let t_rows = timing.then(std::time::Instant::now);
                        for (c, (dst, value)) in partials[..chunks].iter_mut().zip(parts).enumerate() {
                            dst.write(if ww_rows { tri_weigh(eq_hi[c], value) } else { value });
                        }
                        rows_us = t_rows.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e6);
                        gpu_prefix = Some(KGpuPrefix { a, b, len_a: len, len_b: len });
                    }
                }
                crate::gpu_commit::ZcT56Result::Failed => {
                    // The GPU failed after admission: the CPU folds the
                    // prefix chunks (the arm is now disabled).
                    base.store(0, std::sync::atomic::Ordering::Relaxed);
                    crate::epool::run_hetero_chunks(cpu_base, &body);
                }
            }
        }
        // SAFETY: each CPU row is assigned in full; admitted GPU prefix rows
        // are copied in full, and any failed prefix was recomputed on the CPU.
        // The tri0 queue likewise writes each claimed row before returning.
        (unsafe { tri_partials_assume_init(partials) }, ww_h.into_inner())
    } else if hi_size == 1 {
        let (a_in, b_in) = inputs(0);
        (vec![chunk(&a_in, &b_in, a_out, b_out)], false)
    } else if zc_tri_caller_first_enabled() {
        // The `par_chunks_mut(chunk_out)` grid below: `hi_size` chunks.
        let mut partials: Vec<[F128; 26]> = vec![[F128::ZERO; 26]; hi_size];
        let a_base = crate::epool::SyncPtr(a_out.as_mut_ptr());
        let b_base = crate::epool::SyncPtr(b_out.as_mut_ptr());
        let partials_base = crate::epool::SyncPtr(partials.as_mut_ptr());
        crate::epool::run_caller_first_chunks(hi_size, &|x_hi| {
            // SAFETY: the drain runs each `x_hi` exactly once; its output
            // ranges are disjoint per chunk, and the drain publishes every
            // write before the combine below reads them.
            let (a_o, b_o) = unsafe {
                (
                    std::slice::from_raw_parts_mut(a_base.ptr().add(x_hi * chunk_out), chunk_out),
                    std::slice::from_raw_parts_mut(b_base.ptr().add(x_hi * chunk_out), chunk_out),
                )
            };
            let (a_in, b_in) = inputs(x_hi);
            let outv = chunk(&a_in, &b_in, a_o, b_o);
            let outv = if ww { tri_weigh(eq_hi[x_hi], outv) } else { outv };
            // SAFETY: exclusive owner of partials[x_hi].
            unsafe {
                *partials_base.ptr().add(x_hi) = outv;
            }
        });
        (partials, ww)
    } else {
        use rayon::prelude::*;
        (
            a_out
            .par_chunks_mut(chunk_out)
            .zip(b_out.par_chunks_mut(chunk_out))
            .enumerate()
            .map(|(x_hi, (a_o, b_o))| {
                let (a_in, b_in) = inputs(x_hi);
                chunk(&a_in, &b_in, a_o, b_o)
            })
            .collect::<Vec<_>>(),
            false,
        )
    };

    let t_comb = timing.then(std::time::Instant::now);
    let s = tri_combine(&partials, eq_hi, weighted);
    let comb_us = t_comb.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e6);
    let (m1, minf, la, la2) = tri_finish(&s, r_next);
    if timing {
        eprintln!(
            "[zc-tail-rounds] tri combine (rows={hi_size}, weighted on workers={weighted}): \
             gpu rows {rows_us:.1} us + sum {comb_us:.1} us"
        );
    }
    (m1, minf, la, la2, gpu_prefix)
}

/// Test hook of the 5+6 arm's late-GPU steal (see its claim queue). It exists only
/// in a build made with `FLOCK_ZC_T56_TEST_HOOK` set in the BUILD
/// environment (`option_env!`): such a build holds at chunk
/// `FLOCK_ZC_T56_LOOP_HOLD` (run time), or at the build variable's value when
/// that is unset. In every other build this is the constant 0 and nothing in
/// the process environment can change the claim queue.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn t56_loop_hold() -> usize {
    const HOOK: Option<&str> = option_env!("FLOCK_ZC_T56_TEST_HOOK");
    let Some(built) = HOOK else { return 0 };
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("FLOCK_ZC_T56_LOOP_HOLD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| built.parse().unwrap_or(0))
    })
}

/// Kill switch for the shared inversion of the tri-cascade finish:
/// `FLOCK_NO_ZC_TRI_BATCH_INV` (any value) inverts the two challenges
/// separately, as before. Same inverses. Read once per process.
fn zc_tri_batch_inv_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_ZC_TRI_BATCH_INV").is_none())
}

/// `(a^-1, b^-1)` with `F128::inv`'s convention (zero maps to zero). With
/// `batch` and both inputs nonzero it takes one inversion of `a * b` and two
/// products: `a^-1 = b * (a b)^-1`, `b^-1 = a * (a b)^-1`; otherwise it
/// inverts each input.
fn tri_inverse_pair(a: F128, b: F128, batch: bool) -> (F128, F128) {
    if batch {
        let ab = a * b;
        if ab != F128::ZERO {
            let inv_ab = ab.inv();
            return (b * inv_ab, a * inv_ab);
        }
    }
    (a.inv(), b.inv())
}

fn tri_finish(s: &[F128; 26], r_next: &[F128]) -> (F128, F128, Round3Lookahead, Lookahead2) {
    let (inv1, inv2) = tri_inverse_pair(r_next[1], r_next[2], zc_tri_batch_inv_enabled());
    let k1 = F128::ONE + inv1;
    let k2 = F128::ONE + inv2;
    let k12 = k1 * k2;
    let m1 = r_next[0] * (k12 * s[0] + k2 * s[1] + k1 * s[2] + s[3]);
    let minf = k12 * s[4] + k2 * s[5] + k1 * s[6] + s[7];
    let (p0, p1) = (k2 * inv1, inv1);
    let w0 = p0 * s[8] + p1 * s[9];
    let w1 = p0 * s[1] + p1 * s[3];
    let w2 = p0 * s[5] + p1 * s[7];
    let w3 = p0 * s[10] + p1 * s[11];
    let w4 = p0 * s[12] + p1 * s[13];
    let w5 = p0 * s[14] + p1 * s[15];
    let la = Round3Lookahead {
        c: [w0, w0 + w1 + w2, w2, w3, w3 + w4 + w5, w5],
    };
    let sc = inv1 * inv2;
    let g1_src = [s[16], s[9], s[11], s[2], s[3], s[13], s[6], s[7], s[15]];
    let la2 = Lookahead2 {
        g1: g1_src.map(|v| sc * v),
        ginf: std::array::from_fn(|i| sc * s[17 + i]),
    };
    (m1, minf, la, la2)
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
/// Two consecutive [`fold_in_place_pair`] binds (`r1`, then `r2`) in one pass:
/// per output `x` it computes `a01 = a[4x] + r1·(a[4x+1] + a[4x])`,
/// `a23 = a[4x+2] + r1·(a[4x+3] + a[4x+2])` and `a01 + r2·(a23 + a01)` — the
/// same field operations in the same order as the two passes, without storing
/// the half-size intermediate tables. Same for `b`.
pub fn fold2_in_place_pair(a: &mut Vec<F128>, b: &mut Vec<F128>, r1: F128, r2: F128) {
    let n = a.len();
    assert_eq!(b.len(), n);
    assert!(n.is_power_of_two() && n >= 4);
    let quarter = n / 4;
    for x in 0..quarter {
        let a01 = a[4 * x] + r1 * (a[4 * x + 1] + a[4 * x]);
        let a23 = a[4 * x + 2] + r1 * (a[4 * x + 3] + a[4 * x + 2]);
        let b01 = b[4 * x] + r1 * (b[4 * x + 1] + b[4 * x]);
        let b23 = b[4 * x + 2] + r1 * (b[4 * x + 3] + b[4 * x + 2]);
        a[x] = a01 + r2 * (a23 + a01);
        b[x] = b01 + r2 * (b23 + b01);
    }
    a.truncate(quarter);
    b.truncate(quarter);
}

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
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        const ZC_LOOP_INTEGRATION_PARKED: bool = false;
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
    } else if hi_size > 1 && zc_tail_caller_first_enabled() {
        // Caller-first drain of the same `x_hi` chunk grid (see
        // `zc_tail_caller_first_enabled`): each chunk writes its own output
        // ranges and its own partial slot; the sums are XORs of the same
        // per-chunk terms, so the order cannot change a bit.
        let mut partials: Vec<(F128, F128)> = vec![(F128::ZERO, F128::ZERO); hi_size];
        let a_base = crate::epool::SyncPtr(a_out.as_mut_ptr());
        let b_base = crate::epool::SyncPtr(b_out.as_mut_ptr());
        let partials_base = crate::epool::SyncPtr(partials.as_mut_ptr());
        crate::epool::run_caller_first_chunks(hi_size, &|x_hi| {
            // SAFETY: the drain runs each `x_hi` exactly once; output ranges
            // are disjoint per chunk and published before the sum below.
            let (a_o, b_o) = unsafe {
                (
                    std::slice::from_raw_parts_mut(a_base.ptr().add(x_hi * chunk_out), chunk_out),
                    std::slice::from_raw_parts_mut(b_base.ptr().add(x_hi * chunk_out), chunk_out),
                )
            };
            let (p1, pinf) = chunk_partial(
                &a[x_hi * chunk_in..(x_hi + 1) * chunk_in],
                &b[x_hi * chunk_in..(x_hi + 1) * chunk_in],
                a_o,
                b_o,
            );
            let eq_h = eq_hi[x_hi];
            // SAFETY: exclusive owner of partials[x_hi].
            unsafe {
                *partials_base.ptr().add(x_hi) = (eq_h * p1, eq_h * pinf);
            }
        });
        partials
            .iter()
            .fold((F128::ZERO, F128::ZERO), |(s1, sinf), &(c1, cinf)| (s1 + c1, sinf + cinf))
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

/// Caller-first main-pool zerocheck tail rounds below the hetero floor
/// (`FLOCK_NO_ZC_TAIL_CF`, any value, restores the rayon
/// `par_chunks_mut` reduce): the calling thread starts on chunk 0 at once and
/// never parks on a pool latch (the lincheck wide rounds' caller-first, one
/// phase later). Same chunk grid and chunk closure; the message sums are XORs
/// of the same per-chunk terms. Read once per process.
fn zc_tail_caller_first_enabled() -> bool {
    use std::sync::LazyLock;
    static ON: LazyLock<bool> = LazyLock::new(|| std::env::var_os("FLOCK_NO_ZC_TAIL_CF").is_none());
    *ON
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

    #[test]
    fn apple_lookahead_inverse_reuse_selector_is_literal_and_fail_closed() {
        assert!(apple_inverse_micro_selected(true, None));
        assert!(apple_inverse_micro_selected(true, Some("0")));
        assert!(apple_inverse_micro_selected(true, Some("true")));
        assert!(!apple_inverse_micro_selected(true, Some("1")));
        assert!(!apple_inverse_micro_selected(false, None));
        assert!(!apple_inverse_micro_selected(false, Some("0")));
    }

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

    /// `FLOCK_NO_RECOMPUTE_CACHE`: the kept lookahead inverses are the
    /// inverses, for repeated points, more points than slots, and both arms.
    #[test]
    fn lookahead_point_inv_equals_inv() {
        let mut rng = Rng::new(0x5EED);
        let points: Vec<F128> = (0..3 * LOOKAHEAD_INV_SLOTS)
            .map(|_| F128::new(rng.next_u64(), rng.next_u64()))
            .collect();
        for _ in 0..3 {
            for &r in &points {
                assert_eq!(lookahead_point_inv(r), r.inv());
                let (kappa, kept) = lookahead_kappa_and_retained_inv(r, true);
                assert_eq!(kept, Some(r.inv()));
                assert_eq!(kappa, F128::ONE + r.inv());
                assert_eq!(lookahead_kappa_and_retained_inv(r, false), ((F128::ONE + r) * r.inv(), None));
            }
        }
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

    /// Timing probe (ignored): the ranked-size cascade K pass and composed
    /// 5+6 pass with the b-one pack on and off, interleaved in one process
    /// on a synthetic compact state with the ranked b == 1 pairs. Prints the
    /// per-mode medians and the paired differences; asserts equal messages.
    /// `KBONE_LOG_GROUPS` (default 24) and `KBONE_SAMPLES` (default 21).
    #[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
    #[test]
    #[ignore = "timing probe"]
    fn k_bone_pack_timing() {
        let _ = rayon::ThreadPoolBuilder::new().num_threads(8).build_global();
        crate::set_calling_thread_prover_qos();
        let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let log_groups = env("KBONE_LOG_GROUPS", 24);
        let samples = env("KBONE_SAMPLES", 21);
        let n_groups = 1usize << log_groups;
        let n_pairs = 2 * n_groups;
        let mut rng = Rng::new(0x6b62_6f6e_6531);
        let table = UniSkipFoldTable::new(6, rng.f128());
        let mut anchors = rng.f128_vec(2 * n_pairs);
        let mut deltas = crate::scratch::ScratchBytes::take(16 * n_pairs);
        for chunk in deltas.chunks_exact_mut(8) {
            chunk.copy_from_slice(&rng.next_u64().to_le_bytes());
        }
        for p in 0..n_pairs {
            if matches!(p % 128, 0..=8 | 118 | 119) {
                anchors[2 * p + 1] = F128::ONE;
                deltas[16 * p + 8..16 * p + 16].fill(0);
            }
        }
        let compact = UniSkipCompactFold { anchors, deltas, dead_tail: true, tail_compact: false, tail_base_blocks: 0, head_compact: false, head_b: F128::ZERO };
        let (rho1, rho2) = (rng.f128(), rng.f128());
        let mut r_next4 = vec![F128::ONE; log_groups];
        for r in &mut r_next4[1..] {
            *r = rng.f128();
        }
        let rhos = [rng.f128(), rng.f128()];
        let mut r_next6 = vec![F128::ONE; log_groups - 2];
        for r in &mut r_next6[1..] {
            *r = rng.f128();
        }
        let mut a_out = vec![F128::ZERO; n_groups];
        let mut b_out = vec![F128::ZERO; n_groups];
        let mut a2 = vec![F128::ZERO; n_groups / 4];
        let mut b2 = vec![F128::ZERO; n_groups / 4];
        let mut run = |bpack: bool| {
            let t = std::time::Instant::now();
            let (m1, minf, la, packed, bpacked, _) = fold2_compact_and_round45_into_packed_b(
                &compact, &table, rho1, rho2, &r_next4, &mut a_out, &mut b_out, true, bpack, false,
            )
            .expect("ranked b == 1 pairs");
            let k_ms = t.elapsed().as_secs_f64() * 1e3;
            assert!(packed && bpacked == bpack);
            let main = n_groups / 256 * 244;
            let main_b = if bpacked { n_groups / 64 * 56 } else { main };
            let t = std::time::Instant::now();
            let (e1, einf, _, _) = fold_cascade_t_into_packed(
                &a_out[..main], &b_out[..main_b], true, bpacked, &mut a2, &mut b2, &rhos, &r_next6,
            );
            let c_ms = t.elapsed().as_secs_f64() * 1e3;
            (k_ms, c_ms, (m1, minf, la.c, e1, einf, a2[a2.len() - 1], b2[1]))
        };
        let want = run(false).2;
        assert_eq!(run(true).2, want);
        let (mut k_off, mut k_on, mut c_off, mut c_on) = (vec![], vec![], vec![], vec![]);
        for i in 0..samples {
            let order = if i % 2 == 0 { [false, true] } else { [true, false] };
            for bpack in order {
                let (k, c, v) = run(bpack);
                assert_eq!(v, want);
                if bpack {
                    k_on.push(k);
                    c_on.push(c);
                } else {
                    k_off.push(k);
                    c_off.push(c);
                }
            }
        }
        let med = |v: &[f64]| {
            let mut s = v.to_vec();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            s[s.len() / 2]
        };
        let diff = |on: &[f64], off: &[f64]| -> (f64, f64, f64, usize) {
            let d: Vec<f64> = on.iter().zip(off).map(|(a, b)| a - b).collect();
            let n = d.len() as f64;
            let mean = d.iter().sum::<f64>() / n;
            let sd = (d.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
            (mean, sd / n.sqrt(), med(&d), d.iter().filter(|&&x| x < 0.0).count())
        };
        let (km, kse, kmed, kn) = diff(&k_on, &k_off);
        let (cm, cse, cmed, cn) = diff(&c_on, &c_off);
        println!(
            "KBONE n_groups 2^{log_groups} samples {samples}: K off med {:.3} on med {:.3} ms, paired on-off mean {km:+.3} se {kse:.3} med {kmed:+.3} faster {kn}/{samples} | 5+6 off med {:.3} on med {:.3} ms, paired mean {cm:+.3} se {cse:.3} med {cmed:+.3} faster {cn}/{samples}",
            med(&k_off),
            med(&k_on),
            med(&c_off),
            med(&c_on),
        );
    }

    /// Timing probe (ignored): the ranked-size cascade K pass (b-one pack on)
    /// over the same synthetic dead-tail compact state in the dense and the
    /// tail-compacted layout, interleaved in one process. Asserts equal
    /// messages and outputs. `KBONE_LOG_GROUPS`, `KBONE_SAMPLES`.
    #[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
    #[test]
    #[ignore = "timing probe"]
    fn k_tail_compact_timing() {
        use kernels::aarch64::{TC_ANCHORS_PER_BLOCK, TC_DELTA_BYTES_PER_BLOCK};
        let _ = rayon::ThreadPoolBuilder::new().num_threads(8).build_global();
        crate::set_calling_thread_prover_qos();
        let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let log_groups = env("KBONE_LOG_GROUPS", 24);
        let samples = env("KBONE_SAMPLES", 21);
        let n_groups = 1usize << log_groups;
        let n_pairs = 2 * n_groups;
        let mut rng = Rng::new(0x7463_6f6d_7031);
        let table = UniSkipFoldTable::new(6, rng.f128());
        let mut anchors = rng.f128_vec(2 * n_pairs);
        let mut deltas = crate::scratch::ScratchBytes::take(16 * n_pairs);
        for chunk in deltas.chunks_exact_mut(8) {
            chunk.copy_from_slice(&rng.next_u64().to_le_bytes());
        }
        for p in 0..n_pairs {
            if matches!(p % 128, 0..=8 | 118 | 119) {
                anchors[2 * p + 1] = F128::ONE;
                deltas[16 * p + 8..16 * p + 16].fill(0);
            }
        }
        let dense = UniSkipCompactFold { anchors, deltas, dead_tail: true, tail_compact: false, tail_base_blocks: 0, head_compact: false, head_b: F128::ZERO };
        let mut tc = UniSkipCompactFold {
            anchors: vec![F128::new(3, 3); 2 * n_pairs],
            deltas: crate::scratch::ScratchBytes::take(16 * n_pairs),
            dead_tail: true,
            tail_compact: true,
            tail_base_blocks: cascade_k_pairs_per_chunk(n_pairs) / 128,
            head_compact: false,
            head_b: F128::ZERO,
        };
        for blk in 0..n_pairs / 128 {
            let (ia, id) = (tc.anchor_index(128 * blk), tc.delta_offset(128 * blk));
            tc.anchors[ia..ia + TC_ANCHORS_PER_BLOCK]
                .copy_from_slice(&dense.anchors[256 * blk..256 * blk + TC_ANCHORS_PER_BLOCK]);
            tc.deltas[id..id + TC_DELTA_BYTES_PER_BLOCK]
                .copy_from_slice(&dense.deltas[2048 * blk..2048 * blk + TC_DELTA_BYTES_PER_BLOCK]);
        }
        let (rho1, rho2) = (rng.f128(), rng.f128());
        let mut r_next4 = vec![F128::ONE; log_groups];
        for r in &mut r_next4[1..] {
            *r = rng.f128();
        }
        let mut a_out = vec![F128::ZERO; n_groups];
        let mut b_out = vec![F128::ZERO; n_groups];
        let mut run = |c: &UniSkipCompactFold| {
            let t = std::time::Instant::now();
            let (m1, minf, la, _, _, _) = fold2_compact_and_round45_into_packed_b(
                c, &table, rho1, rho2, &r_next4, &mut a_out, &mut b_out, true, true, false,
            )
            .expect("ranked b == 1 pairs");
            let k_ms = t.elapsed().as_secs_f64() * 1e3;
            (k_ms, (m1, minf, la.c, a_out[n_groups / 2 + 7], b_out[n_groups / 3 + 5]))
        };
        let want = run(&dense).1;
        assert_eq!(run(&tc).1, want);
        let (mut off, mut on) = (vec![], vec![]);
        for i in 0..samples {
            let order = if i % 2 == 0 { [false, true] } else { [true, false] };
            for compact in order {
                let (k, v) = run(if compact { &tc } else { &dense });
                assert_eq!(v, want);
                if compact {
                    on.push(k);
                } else {
                    off.push(k);
                }
            }
        }
        let med = |v: &[f64]| {
            let mut s = v.to_vec();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            s[s.len() / 2]
        };
        let d: Vec<f64> = on.iter().zip(&off).map(|(a, b)| a - b).collect();
        let n = d.len() as f64;
        let mean = d.iter().sum::<f64>() / n;
        let se = (d.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt() / n.sqrt();
        println!(
            "KTC n_groups 2^{log_groups} samples {samples}: K dense med {:.3} compacted med {:.3} ms, paired mean {mean:+.3} se {se:.3} med {:+.3} faster {}/{samples}",
            med(&off),
            med(&on),
            med(&d),
            d.iter().filter(|&&x| x < 0.0).count(),
        );
    }

    #[test]
    fn lookahead_kappa_inverse_reuse_matches_incumbent_algebra() {
        let mut rng = Rng::new(0x51a7_1e5e);
        let mut values = vec![F128::ONE, F128::generator()];
        values.extend((0..64).map(|_| {
            let mut value = rng.f128();
            if value == F128::ZERO {
                value = F128::ONE;
            }
            value
        }));

        for r in values {
            let (control_kappa, control_retained) = lookahead_kappa_and_retained_inv(r, false);
            let (candidate_kappa, candidate_retained) = lookahead_kappa_and_retained_inv(r, true);
            let candidate_inv = candidate_retained.expect("candidate retains r^-1");

            assert!(
                control_retained.is_none(),
                "control keeps the later inv site"
            );
            assert_eq!(candidate_inv, r.inv());
            assert_eq!(candidate_kappa, control_kappa);
            assert_eq!(candidate_kappa * r, F128::ONE + r);
        }
    }

    // ----------------------------------------------------------------------
    // Lagrange weights — algebraic properties.
    // ----------------------------------------------------------------------

    /// `Σ_i L_i(z) = 1` for all z. The polynomial `1` interpolates to constant
    /// Dropping the first challenge of an equality table by pair sums gives
    /// the fresh table of the shorter suffix, entry for entry (both builder
    /// routes: below and from three challenges), and the serial round on that
    /// table equals `round_pair_naive`; a chain of drops (as in the serial
    /// zerocheck tail) stays exact down to the one-entry table.
    #[test]
    fn eq_drop_first_matches_fresh_suffix_tables() {
        let mut rng = Rng::new(0x5a7a_11e9);
        for len in 1usize..=12 {
            let r: Vec<F128> = (0..len).map(|_| rng.f128()).collect();
            let mut eq = build_eq(&r);
            for d in 1..=len {
                eq = eq_drop_first(&eq);
                assert_eq!(eq, build_eq(&r[d..]), "len {len} drop {d}");
            }
            let n = 1usize << len;
            let a: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let b: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let mut rr = vec![F128::ONE];
            rr.extend_from_slice(&r[1..]);
            let tab = eq_drop_first(&build_eq(&r));
            assert_eq!(round_pair_naive_with_eq(&a, &b, rr[0], &tab), round_pair_naive(&a, &b, &rr));
            assert_eq!(
                round_pair_naive_with_eq(&a, &b, r[0], &build_eq(&r[1..])),
                round_pair_naive(&a, &b, &r)
            );
        }
    }

    #[test]
    fn fused_tail_fold_and_round_matches_two_passes() {
        let mut rng = Rng::new(0x7a11_f01d);
        for log_n in 2usize..=12 {
            let n = 1usize << log_n;
            let a: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let b: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let rho = rng.f128();
            let r_0 = rng.f128();
            let r: Vec<F128> = (0..log_n - 2).map(|_| rng.f128()).collect();
            let eq = build_eq(&r);
            let (mut a2, mut b2) = (a.clone(), b.clone());
            fold_in_place_pair(&mut a2, &mut b2, rho);
            let want = round_pair_naive_with_eq(&a2, &b2, r_0, &eq);
            let (mut a1, mut b1) = (a, b);
            let got = fold_in_place_and_round_pair_with_eq(&mut a1, &mut b1, rho, r_0, &eq);
            assert_eq!(got, want, "log_n {log_n}");
            assert_eq!((a1, b1), (a2, b2), "log_n {log_n}");
        }
    }

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

    #[test]
    fn lagrange_denominator_matches_every_node_in_both_domains() {
        for (k, &index) in LAGRANGE_DEN_INV_F8.iter().enumerate() {
            let ell = 1usize << k;
            for nodes in PHI_8_TABLE.chunks_exact(ell).take(2) {
                for &node in nodes {
                    let mut product = PHI_8_TABLE[index as usize];
                    for &other in nodes {
                        if other != node {
                            product *= node + other;
                        }
                    }
                    assert_eq!(product, F128::ONE);
                }
            }
        }
    }

    /// The normalized prefix/suffix products match the quadratic loop on both
    /// node domains, for random and node-coincident fold points.
    #[test]
    fn lagrange_fast_matches_reference() {
        let mut rng = Rng::new(42);
        for k_skip in 0..=8 {
            let ell = 1usize << k_skip;
            for t in 0..6 {
                let z = match t {
                    4 => PHI_8_TABLE[0],
                    5 => PHI_8_TABLE[(2 * ell - 1).min(255)],
                    _ => rng.f128(),
                };
                assert_eq!(
                    lagrange_weights_fast(k_skip, z, 0),
                    lagrange_weights_naive_reference(k_skip, z),
                    "S-domain mismatch at k_skip={k_skip}, t={t}"
                );
                if 2 * ell <= 256 {
                    assert_eq!(
                        lagrange_weights_fast(k_skip, z, ell),
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

    #[test]
    fn round_pair_deferred_matches_reduced_parent() {
        let mut rng = Rng::new(0xDEF3_22ED);
        for log_n in [1usize, 2, 3, 8, 13] {
            let a = rng.f128_vec(1usize << log_n);
            let b = rng.f128_vec(1usize << log_n);
            for mode in 0..4 {
                let r: Vec<F128> = (0..log_n)
                    .map(|i| match mode {
                        0 => F128::ZERO,
                        1 => F128::ONE,
                        2 if i & 1 == 0 => F128::ZERO,
                        2 => F128::ONE,
                        _ => rng.f128(),
                    })
                    .collect();
                let eq = build_eq(&r[1..]);
                let mut want = (F128::ZERO, F128::ZERO);
                for (i, &weight) in eq.iter().enumerate() {
                    want.0 += weight * a[2 * i + 1] * b[2 * i + 1];
                    want.1 += weight * (a[2 * i] + a[2 * i + 1]) * (b[2 * i] + b[2 * i + 1]);
                }
                want.0 = r[0] * want.0;
                assert_eq!(
                    round_pair_naive(&a, &b, &r),
                    want,
                    "log_n={log_n} mode={mode}"
                );
            }
        }
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

    /// Production donation oracle: both full-coverage R2 writers must accept
    /// a backing that exposes no slice initially, publish it only after every
    /// writer joins, and mint the exact provenance token on final recycle.
    #[test]
    fn standard_and_lookahead_r2_publish_full_overwrite_backing() {
        const STANDARD_TAG: u64 = 0xA8F1_2002_0000_1001;
        const LOOKAHEAD_TAG: u64 = 0xA8F1_2002_0000_1002;
        let f = la_fixture(16, 0xF011_C0DE_2002, true);
        let deltas_len = (1usize << (f.m - 6)) * f.table.n_chunks;
        let deltas_f128 = deltas_len / core::mem::size_of::<F128>();
        crate::scratch::clear();

        let (standard_expected, expected_m1, expected_mi) =
            uni_skip_fold_and_round_pair_compact_padded(
                &f.a_packed,
                &f.b_packed,
                f.m,
                6,
                &f.table,
                &f.mlv,
                &f.padding,
            );
        let standard_donor =
            ScratchBytes::from_f128_full_overwrite(vec![LA_POISON; deltas_f128], STANDARD_TAG);
        let (standard_got, got_m1, got_mi) =
            uni_skip_fold_and_round_pair_compact_padded_with_deltas(
                &f.a_packed,
                &f.b_packed,
                f.m,
                6,
                &f.table,
                &f.mlv,
                &f.padding,
                Some(standard_donor),
            );
        assert_eq!((got_m1, got_mi), (expected_m1, expected_mi));
        assert_eq!(standard_got.anchors, standard_expected.anchors);
        assert_eq!(&standard_got.deltas[..], &standard_expected.deltas[..]);
        standard_expected.recycle();
        standard_got.recycle();
        let (standard_held, standard_hit) =
            crate::scratch::take_f128_with_token(deltas_f128, STANDARD_TAG);
        assert!(standard_hit, "standard R2 must stage its completed backing");

        let (lookahead_expected, expected_m1, expected_mi, expected_la) =
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
        let lookahead_donor =
            ScratchBytes::from_f128_full_overwrite(vec![LA_POISON; deltas_f128], LOOKAHEAD_TAG);
        let (lookahead_got, got_m1, got_mi, got_la) =
            uni_skip_fold_and_round_pair_compact_padded_lookahead(
                &f.a_packed,
                &f.b_packed,
                f.m,
                6,
                &f.table,
                &f.mlv,
                &f.padding,
                Some(lookahead_donor),
            );
        assert_eq!(
            (got_m1, got_mi, got_la),
            (expected_m1, expected_mi, expected_la)
        );
        assert_eq!(lookahead_got.anchors, lookahead_expected.anchors);
        assert_eq!(&lookahead_got.deltas[..], &lookahead_expected.deltas[..]);
        lookahead_expected.recycle();
        lookahead_got.recycle();
        let (lookahead_held, lookahead_hit) =
            crate::scratch::take_f128_with_token(deltas_f128, LOOKAHEAD_TAG);
        assert!(
            lookahead_hit,
            "lookahead R2 must stage its completed backing"
        );

        crate::scratch::give_f128(standard_held);
        crate::scratch::give_f128(lookahead_held);
        crate::scratch::clear();
    }

    /// Round-two A/B rebuild (`univariate_skip_optimized::ENV_NO_R2_AB_REBUILD`):
    /// with the rows past the CPU rebuild boundary garbage and the producer
    /// registered, the lookahead sweep returns the same message, lookahead
    /// and compact state as over the complete buffers, it takes the
    /// registration, and it never publishes the rest: the walk read the
    /// rebuilt rows. Several prefixes and boundaries, ranked and dense.
    #[test]
    fn round2_ab_rebuild_matches_complete_buffers() {
        use crate::zerocheck::univariate_skip_optimized::{Round1AbProducer, round2_ab_rebuild_scope};
        for (m, ranked) in [(20usize, true), (16, false), (25, true)] {
            let f = la_fixture(m, 0x2AB0_0001 ^ (m as u64), ranked);
            let (hi_size, chunk_bytes) = round2_lookahead_chunk_geometry(m).expect("geometry");
            assert_eq!(hi_size * chunk_bytes, f.a_packed.len());
            let (want, want_m1, want_mi, want_la) = uni_skip_fold_and_round_pair_compact_padded_lookahead(
                &f.a_packed, &f.b_packed, f.m, 6, &f.table, &f.mlv, &f.padding, None,
            );
            // Units of one chunk, or of half a chunk when it splits evenly.
            let unit_bytes = if chunk_bytes.is_multiple_of(32) { chunk_bytes / 2 } else { chunk_bytes };
            // (published chunks, CPU rebuild boundary, garbage from): the
            // AB arm's prefix; complete buffers whose walk still rebuilds past
            // the cap (the warm-up calibration's registration); every chunk
            // rebuilt. The GPU arm (absent at these shapes) is the only
            // reader of the published rows past the boundary.
            let caps = [0usize, 1, hi_size * 43 / 100, hi_size - 1];
            let cases = caps
                .iter()
                .map(|&c| (c, c, c))
                .chain(caps.iter().map(|&c| (hi_size, c, c)))
                .chain([(hi_size * 43 / 100, 0, 0)]);
            for (cap, from, garbage) in cases {
                let published = cap * chunk_bytes;
                let rebuild_from = from * chunk_bytes;
                let garbage = garbage * chunk_bytes;
                let mut a = f.a_packed.clone();
                let mut b = f.b_packed.clone();
                a[garbage..].fill(0xA5);
                b[garbage..].fill(0x5A);
                let (a_ptr, b_ptr) = (a.as_mut_ptr() as usize, b.as_mut_ptr() as usize);
                let produce = |unit: usize, sa: *mut u8, sb: *mut u8, _primed: bool| {
                    let at = unit * unit_bytes;
                    let (da, db) = if sa.is_null() {
                        // SAFETY: the test's buffers, one unit each.
                        unsafe { ((a_ptr as *mut u8).add(at), (b_ptr as *mut u8).add(at)) }
                    } else {
                        (sa, sb)
                    };
                    // SAFETY: `unit_bytes` bytes at both ends.
                    unsafe {
                        core::ptr::copy_nonoverlapping(f.a_packed.as_ptr().add(at), da, unit_bytes);
                        core::ptr::copy_nonoverlapping(f.b_packed.as_ptr().add(at), db, unit_bytes);
                    }
                };
                let len = a.len();
                // SAFETY: the buffers outlive the scope; only `produce`
                // writes them (null scratch only).
                let scope = unsafe {
                    round2_ab_rebuild_scope(
                        a_ptr as *const u8,
                        b_ptr as *const u8,
                        len,
                        published,
                        rebuild_from,
                        Round1AbProducer { unit_bytes, produce: &produce },
                    )
                };
                let (got, got_m1, got_mi, got_la) = uni_skip_fold_and_round_pair_compact_padded_lookahead(
                    &a, &b, f.m, 6, &f.table, &f.mlv, &f.padding, None,
                );
                assert_eq!((got_m1, got_mi, got_la), (want_m1, want_mi, want_la), "message, m={m} cap={cap} from={from}");
                assert_eq!(got.anchors, want.anchors, "anchors, m={m} cap={cap} from={from}");
                assert_eq!(&got.deltas[..], &want.deltas[..], "deltas, m={m} cap={cap} from={from}");
                assert!(
                    crate::zerocheck::univariate_skip_optimized::round2_ab_rebuild_take(&a, &b).is_none(),
                    "round two must take its registration"
                );
                drop(scope);
                assert!(a[garbage..].iter().all(|&x| x == 0xA5), "a rows published, m={m} cap={cap} from={from}");
                assert!(b[garbage..].iter().all(|&x| x == 0x5A), "b rows published, m={m} cap={cap} from={from}");
                got.recycle();
            }
            want.recycle();
        }
    }


    #[test]
    #[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
    fn round2_ab_quad_matches_whole_chunk() {
        use crate::zerocheck::univariate_skip_optimized::{Round1AbProducer, round2_ab_rebuild_scope, round2_ab_rebuild_take};
        use std::sync::atomic::{AtomicUsize, Ordering};
        const LO: usize = 16384;
        const UNIT: usize = 8192;
        let mut rng = Rng::new(0x2AB0_512);
        let table = UniSkipFoldTable::new(6, rng.f128());
        let eq: Vec<_> = (0..LO).map(|_| rng.f128()).collect();
        let prep = kernels::aarch64::round2_eq_prep(&eq);
        let fold7 = kernels::aarch64::r2_fold7_table(&table.data).expect("linear table");
        let mut a = vec![0u8; 3 * LO * 16];
        let mut b = a.clone();
        // Generate every unit from its index; only each block's constant
        // all-ones b head may be retained when scratch is primed.
        let calls = AtomicUsize::new(0);
        let first_primed = AtomicUsize::new(0);
        let produce = |q: usize, sa: *mut u8, sb: *mut u8, primed: bool| {
            if calls.fetch_add(1, Ordering::Relaxed) == 0 && primed {
                first_primed.fetch_add(1, Ordering::Relaxed);
            }
            unsafe {
                for i in 0..UNIT {
                    let global = q * UNIT + i;
                    *sa.add(i) = (global.wrapping_mul(73) ^ (global >> 7) ^ (global >> 13)) as u8;
                    if !primed || i % 2048 >= 128 {
                        *sb.add(i) = if i % 2048 < 128 { 255 } else {
                            (global.wrapping_mul(31) ^ (global >> 5) ^ (global >> 11)) as u8
                        };
                    }
                }
            }
        };
        for q in 0..a.len() / UNIT {
            produce(q, unsafe { a.as_mut_ptr().add(q * UNIT) }, unsafe { b.as_mut_ptr().add(q * UNIT) }, false);
        }
        for x_hi in [1usize, 2] {
            for (tail, head_enabled) in [(false, false), (true, false), (true, true)] {
                for f7 in [0usize, fold7.as_ptr() as usize] {
                    calls.store(0, Ordering::Relaxed);
                    first_primed.store(0, Ordering::Relaxed);
                    let scope = unsafe { round2_ab_rebuild_scope(a.as_ptr(), b.as_ptr(), a.len(), 0, 0,
                        Round1AbProducer { unit_bytes: UNIT, produce: &produce }) };
                    let r = round2_ab_rebuild_take(&a, &b).unwrap();
                    // Nonzero allocation offsets catch compact pointer rebases.
                    let mut want_a = vec![LA_POISON; 2 * LO + 512];
                    let mut got_a = want_a.clone();
                    let mut want_d = vec![0xA5; 16 * LO + 8192];
                    let mut got_d = want_d.clone();
                    let want_head = std::sync::atomic::AtomicBool::new(false);
                    let got_head = std::sync::atomic::AtomicBool::new(false);
                    let mut want = [F128::ZERO; 8];
                    let offset_a = 256;
                    let offset_d = 4096;
                    let hp = |h: &std::sync::atomic::AtomicBool| if head_enabled { h as *const _ } else { core::ptr::null() };
                    unsafe {
                        let (ap, bp) = r2_rebuild_chunk(&r, x_hi, LO * 16);
                        if f7 != 0 {
                            kernels::aarch64::fold_round2_compact_chunk_neon_lookahead_8_pieces7(
                                table.data.as_ptr().cast(), f7 as *const u8, ap, bp,
                                want_a.as_mut_ptr().add(offset_a), want_d.as_mut_ptr().add(offset_d),
                                eq.as_ptr(), prep.as_ptr(), LO, x_hi * LO, 127, 121,
                                true, true, true, tail, hp(&want_head), want.as_mut_ptr());
                        } else {
                            fold_round2_compact_chunk_neon_lookahead_8::<true, false>(
                                table.data.as_ptr().cast(), ap, bp,
                                want_a.as_mut_ptr().add(offset_a), want_d.as_mut_ptr().add(offset_d),
                                eq.as_ptr(), prep.as_ptr(), LO, x_hi * LO, 127, 121,
                                true, true, true, tail, hp(&want_head), want.as_mut_ptr());
                        }
                        // A new registration must never inherit primed state.
                        calls.store(0, Ordering::Relaxed);
                        R2_AB_SCRATCH.with(|s| s.primed_id.set(0));
                        let got = r2_rebuild_quad_walk(&r, x_hi, table.data.as_ptr().cast(), f7,
                            got_a.as_mut_ptr().add(offset_a), got_d.as_mut_ptr().add(offset_d),
                            eq.as_ptr(), prep.as_ptr(), LO, x_hi * LO, tail, hp(&got_head), core::ptr::null(), false);
                        assert_eq!(got, want, "all eight sums: x_hi={x_hi} tail={tail} head={head_enabled} f7={f7}");
                    }
                    assert_eq!(got_a, want_a, "all live anchors and poison guards");
                    assert_eq!(got_d, want_d, "all live deltas and poison guards");
                    assert!(!want_head.load(Ordering::Relaxed));
                    assert!(!got_head.load(Ordering::Relaxed));
                    assert_eq!(calls.load(Ordering::Relaxed), LO * 16 / UNIT);
                    assert_eq!(first_primed.load(Ordering::Relaxed), 0);
                    assert_eq!(R2_AB_SCRATCH.with(|s| s.bytes.get()), UNIT,
                        "quad engagement: scratch must hold one producer unit per A/B half");
                    drop(scope);
                }
            }
        }
        eprintln!("quad differential: 12 cases, 8 sums, all compact bytes, nonzero offsets, scratch={} bytes", 2 * UNIT);
    }

    #[test]
    #[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
    fn round2_ab_quad_preserves_xb_kernel() {
        use crate::zerocheck::univariate_skip_optimized::{Round1AbProducer, round2_ab_rebuild_scope, round2_ab_rebuild_take};
        use std::sync::atomic::{AtomicBool, Ordering};
        const LO: usize = 2048;
        const UNIT: usize = 8192;
        let mut rng = Rng::new(0x2ab0_5842_512);
        let table = UniSkipFoldTable::new(6, rng.f128());
        let mut eq: Vec<_> = (0..LO).map(|_| rng.f128()).collect();
        let rho = rng.f128();
        for first in (0..LO / 2).step_by(32) {
            eq[2 * first + 3] = rho * eq[2 * first + 1];
            for t in 0..2 {
                for power in 1..16 {
                    eq[2 * (first + 2 * power + t) + 1] =
                        eq[2 * (first + t) + 1] * F128::new(1u64 << power, 0);
                }
            }
        }
        let prep = kernels::aarch64::round2_eq_prep(&eq);
        let xb = kernels::aarch64::r2_full_xb_rho_prep(&eq).expect("valid XB weights");
        let f7 = kernels::aarch64::r2_fold7_table(&table.data).expect("linear table");
        let a = vec![0u8; 3 * LO * 16];
        let b = a.clone();
        let produce = |q: usize, sa: *mut u8, sb: *mut u8, primed: bool| {
            unsafe {
                for i in 0..UNIT {
                    let at = q * UNIT + i;
                    *sa.add(i) = (at.wrapping_mul(73) ^ (at >> 9)) as u8;
                    if !primed || i % 2048 >= 128 {
                        *sb.add(i) = if i % 2048 < 128 { 255 } else {
                            (at.wrapping_mul(47) ^ (at >> 5)) as u8
                        };
                    }
                }
            }
        };
        for x_hi in [1usize, 2] {
            for (tail, head) in [(false, false), (true, false), (true, true)] {
                for pieces in [false, true] {
                    let scope = unsafe { round2_ab_rebuild_scope(a.as_ptr(), b.as_ptr(), a.len(), 0, 0,
                        Round1AbProducer { unit_bytes: UNIT, produce: &produce }) };
                    let r = round2_ab_rebuild_take(&a, &b).unwrap();
                    let mut want_a = vec![LA_POISON; 2 * LO + 512];
                    let mut got_a = want_a.clone();
                    let mut want_d = vec![0xa5u8; 16 * LO + 8192];
                    let mut got_d = want_d.clone();
                    let wh = AtomicBool::new(false);
                    let gh = AtomicBool::new(false);
                    let hp = |h: &AtomicBool| if head { h as *const _ } else { core::ptr::null() };
                    let table7 = if pieces { f7.as_ptr() as usize } else { 0 };
                    let mut want = [F128::ZERO; 8];
                    unsafe {
                        let (ap, bp) = r2_rebuild_chunk(&r, x_hi, LO * 16);
                        kernels::aarch64::fold_round2_compact_chunk_neon_lookahead_8_xb(
                            table.data.as_ptr().cast(), table7 as *const u8, ap, bp,
                            want_a.as_mut_ptr().add(256), want_d.as_mut_ptr().add(4096),
                            eq.as_ptr(), xb.as_ptr(), LO, true, tail, hp(&wh), false, want.as_mut_ptr());
                        let got = r2_rebuild_quad_walk(&r, x_hi, table.data.as_ptr().cast(), table7,
                            got_a.as_mut_ptr().add(256), got_d.as_mut_ptr().add(4096),
                            eq.as_ptr(), prep.as_ptr(), LO, x_hi * LO, tail, hp(&gh), xb.as_ptr(), false);
                        assert_eq!(got, want, "XB slots: {x_hi} {tail} {head} {pieces}");
                    }
                    assert_eq!(got_a, want_a, "XB anchors and guards");
                    assert_eq!(got_d, want_d, "XB deltas and guards");
                    assert_eq!(gh.load(Ordering::Relaxed), wh.load(Ordering::Relaxed));
                    assert_eq!(R2_AB_SCRATCH.with(|s| s.bytes.get()), UNIT);
                    drop(scope);
                }
            }
        }
        eprintln!("XB quad differential: 12 cases, all 8 sums and compact bytes");
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

    /// D1 — dead-pair skip. A K-pass-only round two at the ranked padding
    /// leaves the padding pairs' slots unwritten (poison survives on a
    /// donated initialized backing) and the cascade K pass then reproduces
    /// the dense state's outputs, round-four message and round-five
    /// lookahead exactly. m = 25 is the smallest shape whose K chunks hold
    /// whole 128-pair blocks.
    #[test]
    fn dead_pair_skip_matches_dense_k_pass() {
        // Random b rows (a head pair's b rows are not all ones: round two
        // falls back from the head-compacted layout) and the ranked
        // structure (head pairs 0..7 of every block have all-ones b rows).
        for ones_head in [false, true] {
            let mut f = la_fixture(25, 0xDEAD_0001, true);
            if ones_head {
                for blk in 0..f.b_packed.len() / 2048 {
                    f.b_packed[2048 * blk..2048 * blk + 128].fill(0xFF);
                }
            }
            let r_next4 = la_r_next4(&f);
            // 16 delta bytes per pair = one F128 per pair.
            let deltas_f128 = 1usize << (f.m - 7);
            let mut tc_seen = false;
            with_round2_settings(|| {
                crate::scratch::clear();
                let (dense, d1, di, dla) = uni_skip_fold_and_round_pair_compact_padded_lookahead(
                    &f.a_packed,
                    &f.b_packed,
                    f.m,
                    6,
                    &f.table,
                    &f.mlv,
                    &f.padding,
                    None,
                );
                assert!(!dense.dead_tail);
                let backing = ScratchBytes::from_initialized_f128(vec![LA_POISON; deltas_f128]);
                let (sparse, s1, si, sla) = uni_skip_fold_and_round_pair_compact_padded_lookahead_for(
                    &f.a_packed,
                    &f.b_packed,
                    f.m,
                    6,
                    &f.table,
                    &f.mlv,
                    &f.padding,
                    Some(backing),
                    true,
                    None,
                );
                let skip_on = std::env::var_os("FLOCK_NO_R2_DEADPAIR_SKIP").is_none();
                assert_eq!(sparse.dead_tail, cfg!(target_arch = "aarch64") && skip_on);
                assert_eq!((s1, si), (d1, di), "round-two message");
                assert_eq!(sla.c, dla.c, "round-three lookahead");
                let n_pairs = dense.len();
                let poison_bytes = {
                    let v = [LA_POISON];
                    unsafe { core::slice::from_raw_parts(v.as_ptr().cast::<u8>(), 16) }.to_vec()
                };
                // Other tests flip the round-two switches concurrently, so the
                // layout is checked for consistency, not predicted.
                assert!(!sparse.tail_compact || sparse.dead_tail);
                assert!(!sparse.head_compact || sparse.tail_compact);
                tc_seen |= sparse.tail_compact;
                if ones_head && sparse.tail_compact && !sparse.head_compact {
                    eprintln!("dead_pair_skip_matches_dense_k_pass: head-compacted layout not taken");
                }
                if !ones_head {
                    assert!(!sparse.head_compact, "random head b rows must fall back");
                }
                for p in 0..n_pairs {
                    let dead = (p & 127) >= 121;
                    let (sa, so) = (sparse.anchor_index(p), sparse.delta_offset(p));
                    if dead {
                        assert!(dense.anchors[2 * p..2 * p + 2].iter().all(|x| *x == F128::ZERO));
                        // The tail-compacted layout keeps delta slots for the
                        // dead pairs 121..123 only (they share pair 120's line).
                        // After a head-compaction fallback (random head b rows)
                        // they may hold the first layout's bytes: never read.
                        if sparse.dead_tail
                            && (!sparse.tail_compact || ((p & 127) < 124 && ones_head))
                        {
                            let sd = &sparse.deltas[so..so + 16];
                            assert_eq!(sd, &poison_bytes[..], "dead delta written at pair {p}");
                        }
                    } else if sparse.head_compact && (p & 127) < 8 {
                        // Head pair: a anchor and a delta only.
                        assert_eq!(sparse.anchors[sa], dense.anchors[2 * p], "head a anchor at {p}");
                        assert_eq!(dense.anchors[2 * p + 1], sparse.head_b, "head b anchor at {p}");
                        assert_eq!(&sparse.deltas[so..so + 8], &dense.deltas[16 * p..16 * p + 8]);
                        assert!(dense.deltas[16 * p + 8..16 * p + 16].iter().all(|&x| x == 0));
                    } else {
                        let sd = &sparse.deltas[so..so + 16];
                        assert_eq!(&sparse.anchors[sa..sa + 2], &dense.anchors[2 * p..2 * p + 2]);
                        assert_eq!(sd, &dense.deltas[16 * p..16 * p + 16], "delta at pair {p}");
                    }
                }
                if sparse.tail_compact {
                    // Nothing is written in front of or past the compacted blocks.
                    let lo = sparse.delta_offset(0);
                    // The compacted layouts exist only in the aarch64 kernels.
                    #[cfg(target_arch = "aarch64")]
                    let (hc_bpb, tc_bpb) = (
                        kernels::aarch64::HC_DELTA_BYTES_PER_BLOCK,
                        kernels::aarch64::TC_DELTA_BYTES_PER_BLOCK,
                    );
                    #[cfg(not(target_arch = "aarch64"))]
                    let (hc_bpb, tc_bpb): (usize, usize) =
                        unreachable!("tail compaction is an aarch64-only layout");
                    let per_block = if sparse.head_compact { hc_bpb } else { tc_bpb };
                    let hi = sparse.delta_offset(n_pairs - 128) + per_block;
                    assert!(sparse.deltas[..lo].chunks_exact(16).all(|c| c == &poison_bytes[..]));
                    assert!(sparse.deltas[hi..].chunks_exact(16).all(|c| c == &poison_bytes[..]));
                }
                let n_groups = n_pairs / 2;
                for &(rho1, rho2) in &[
                    (f.rho_grid[3], f.rho_grid[2]),
                    (F128::ONE, f.rho_grid[3]),
                    (F128::ZERO, F128::ONE),
                ] {
                    let mut a_d = vec![LA_POISON; n_groups];
                    let mut b_d = vec![LA_POISON; n_groups];
                    let (m1_d, mi_d, la_d) = fold2_compact_and_round45_into(
                        &dense, &f.table, rho1, rho2, &r_next4, &mut a_d, &mut b_d,
                    );
                    let mut a_s = vec![LA_POISON; n_groups];
                    let mut b_s = vec![LA_POISON; n_groups];
                    let (m1_s, mi_s, la_s) = fold2_compact_and_round45_into(
                        &sparse, &f.table, rho1, rho2, &r_next4, &mut a_s, &mut b_s,
                    );
                    assert_eq!((m1_s, mi_s), (m1_d, mi_d), "round-four message");
                    assert_eq!(la_s.c, la_d.c, "round-five lookahead");
                    assert!(a_s == a_d, "A'' outputs differ");
                    assert!(b_s == b_d, "B'' outputs differ");
                }
                dense.recycle();
                sparse.recycle();
                crate::scratch::clear();
            });
            if cfg!(all(target_arch = "aarch64", target_feature = "sha3")) && !tc_seen {
                eprintln!("dead_pair_skip_matches_dense_k_pass: tail-compacted layout not exercised (switch set)");
            }
            unsafe { std::env::set_var("FLOCK_NO_R2_DEADPAIR_SKIP", "1") };
            let backing = ScratchBytes::from_initialized_f128(vec![LA_POISON; deltas_f128]);
            let (off, _, _, _) = uni_skip_fold_and_round_pair_compact_padded_lookahead_for(
                &f.a_packed,
                &f.b_packed,
                f.m,
                6,
                &f.table,
                &f.mlv,
                &f.padding,
                Some(backing),
                true,
                None,
            );
            unsafe { std::env::remove_var("FLOCK_NO_R2_DEADPAIR_SKIP") };
            assert!(!off.dead_tail, "kill switch keeps the dense state");
            off.recycle();
            crate::scratch::clear();
        }
    }


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

    #[test]
    fn tri_pass_matches_sequential_rounds() {
        let mut rng = Rng::new(0x7121_0003);
        let special = |rng: &mut Rng, i: usize| match i % 4 {
            0 => F128::ZERO,
            1 => F128::ONE,
            _ => rng.f128(),
        };
        for depth in [2usize, 3] {
            for (case, log_out) in [3usize, 4, 5, 9, 11, 12, 17, 20].into_iter().enumerate() {
                if depth == 3 && log_out == 20 {
                    continue;
                }
                let n = 1usize << (log_out + depth);
                let a: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
                let b: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
                let rhos: Vec<F128> = (0..depth).map(|i| special(&mut rng, i + case)).collect();
                let mut r_next = vec![F128::ONE; log_out];
                for s in r_next[1..].iter_mut() {
                    *s = rng.f128();
                }
                let mut a_o = vec![LA_POISON; n >> depth];
                let mut b_o = vec![LA_POISON; n >> depth];
                let (m1, minf, la, la2) =
                    fold_cascade_t_into(&a, &b, &mut a_o, &mut b_o, &rhos, &r_next);
                let (mut fa, mut fb) = (a.clone(), b.clone());
                for &rho in &rhos {
                    fold_in_place_pair(&mut fa, &mut fb, rho);
                }
                assert!(fa == a_o && fb == b_o, "outputs depth {depth} log_out {log_out}");
                assert_eq!(
                    (m1, minf),
                    round_pair_naive(&fa, &fb, &r_next),
                    "exact depth {depth} log_out {log_out}"
                );
                for t in 0..3 {
                    let u = special(&mut rng, t + 1);
                    let v = special(&mut rng, t + 2);
                    let (mut ga, mut gb) = (fa.clone(), fb.clone());
                    fold_in_place_pair(&mut ga, &mut gb, u);
                    let mut r1 = vec![F128::ONE; log_out - 1];
                    r1[1..].copy_from_slice(&r_next[2..]);
                    assert_eq!(
                        eval_round3_lookahead(&la, u),
                        round_pair_naive(&ga, &gb, &r1),
                        "la depth {depth} log_out {log_out}"
                    );
                    fold_in_place_pair(&mut ga, &mut gb, v);
                    let mut r2 = vec![F128::ONE; log_out - 2];
                    r2[1..].copy_from_slice(&r_next[3..]);
                    assert_eq!(
                        eval_lookahead2(&la2, u, v),
                        round_pair_naive(&ga, &gb, &r2),
                        "la2 depth {depth} log_out {log_out}"
                    );
                }
                if log_out <= 12 {
                    let eq = SplitEqGhash::with_n_hi(&r_next[1..], 0);
                    let mut sa = vec![LA_POISON; n >> depth];
                    let mut sb = vec![LA_POISON; n >> depth];
                    let s = tri_chunk_scalar(&a, &b, &mut sa, &mut sb, &rhos, &eq.lo);
                    assert_eq!(tri_finish(&s, &r_next), (m1, minf, la, la2));
                }
            }
        }
    }

    /// Force the real heterogeneous CPU path at small sizes and compare every
    /// unweighted partial row, not only the final XOR reduction. A repeated
    /// poison could cancel in the reduction, so retain the full row snapshots.
    #[test]
    fn tri_partials_no_clear_matches_zero_and_poison() {
        struct Reset;
        impl Drop for Reset {
            fn drop(&mut self) {
                TRI_PARTIAL_TEST_MODE.with(|m| m.set(0));
                TRI_PARTIAL_TEST_ROWS.with(|r| *r.borrow_mut() = None);
            }
        }
        let _reset = Reset;
        let mut rng = Rng::new(0x7472_6970_6172_746c);
        for (depth, packed, bpacked) in [(2, false, false), (3, false, false), (2, true, false), (2, true, true)] {
            if packed && !cfg!(all(target_arch = "aarch64", target_feature = "aes")) {
                continue;
            }
            for log_out in [3usize, 8, 11] {
                let n = 1usize << (log_out + depth);
                if packed && n < 256 { continue; }
                let mut a = rng.f128_vec(n);
                let mut b = rng.f128_vec(n);
                if packed {
                    for i in 0..n {
                        if i % 64 >= 61 {
                            a[i] = F128::ZERO;
                            b[i] = F128::ZERO;
                        } else if bpacked && matches!(i % 64, 0..=3 | 59) {
                            b[i] = F128::ONE;
                        }
                    }
                    a = a.into_iter().enumerate().filter_map(|(i, v)| (i % 64 <= 60).then_some(v)).collect();
                    b = b.into_iter().enumerate().filter_map(|(i, v)| {
                        (i % 64 <= 60 && !(bpacked && matches!(i % 64, 0..=3 | 59))).then_some(v)
                    }).collect();
                }
                let rhos = rng.f128_vec(depth);
                let mut r_next = rng.f128_vec(log_out);
                r_next[0] = F128::ONE;
                r_next[1] = F128::new(7, 13);
                r_next[2] = F128::new(17, 19);
                let run = |mode| {
                    TRI_PARTIAL_TEST_MODE.with(|m| m.set(mode));
                    TRI_PARTIAL_TEST_ROWS.with(|r| *r.borrow_mut() = None);
                    let mut a_out = vec![LA_POISON; 1 << log_out];
                    let mut b_out = vec![LA_POISON; 1 << log_out];
                    let (m1, mi, la, la2, prefix) = fold_cascade_t_into_packed_prefix_gpu(
                        &a, &b, packed, bpacked, None, &mut a_out, &mut b_out,
                        &rhos, &r_next, false,
                    );
                    assert!(prefix.is_none());
                    let rows = TRI_PARTIAL_TEST_ROWS.with(|r| r.borrow_mut().take()).expect("heterogeneous rows captured");
                    (a_out, b_out, m1, mi, la, la2, rows)
                };
                let reference = run(1);
                assert_eq!(run(2), reference, "poison depth={depth} packed={packed} bpacked={bpacked} log_out={log_out}");
                assert_eq!(run(3), reference, "uninit depth={depth} packed={packed} bpacked={bpacked} log_out={log_out}");
            }
        }
    }

    /// Model only ownership handoffs, not Metal execution. Each producer
    /// writes a complete row; read a prefix only after it has been written.
    #[test]
    fn tri_partial_storage_prefix_and_failure_handoffs() {
        const N: usize = 17;
        let expected: Vec<[F128; 26]> = (0..N).map(|row| {
            std::array::from_fn(|lane| F128::new((row * 31 + lane + 1) as u64, (row * 71 + lane * 13 + 3) as u64))
        }).collect();
        for mode in [1, 2, 3] {
            TRI_PARTIAL_TEST_MODE.with(|m| m.set(mode));
            for prefix in [0, 1, N / 2, N] {
                for failed in [false, true] {
                    let mut rows = tri_partial_storage(N);
                    for row in prefix..N { rows[row].write(expected[row]); }
                    if failed {
                        // Failed or declined GPU: CPU recomputes its prefix.
                        for row in 0..prefix { rows[row].write(expected[row]); }
                    } else {
                        for row in 0..prefix { rows[row].write(expected[row]); }
                        // SAFETY: that prefix has just been fully written.
                        assert_eq!(unsafe { tri_partial_prefix(&rows, prefix) }, &expected[..prefix]);
                    }
                    // SAFETY: prefix and suffix partition all complete rows.
                    assert_eq!(unsafe { tri_partials_assume_init(rows) }, expected);
                }
            }
        }
        TRI_PARTIAL_TEST_MODE.with(|m| m.set(0));
        TRI_PARTIAL_TEST_ROWS.with(|r| *r.borrow_mut() = None);
        // Dropping incomplete storage is safe; it never assumes initialization.
        drop(tri_partial_storage(N));
    }

    #[test]
    fn tri_inverse_pair_matches_separate_inversions() {
        let mut rng = Rng::new(0x7121_1A57);
        let mut cases: Vec<(F128, F128)> = (0..64).map(|_| (rng.f128(), rng.f128())).collect();
        cases.push((F128::ZERO, rng.f128()));
        cases.push((rng.f128(), F128::ZERO));
        cases.push((F128::ZERO, F128::ZERO));
        cases.push((F128::ONE, F128::ONE));
        for (a, b) in cases {
            let want = (a.inv(), b.inv());
            assert_eq!(tri_inverse_pair(a, b, false), want);
            assert_eq!(tri_inverse_pair(a, b, true), want, "a={a:?} b={b:?}");
            if a != F128::ZERO {
                assert_eq!(a * tri_inverse_pair(a, b, true).0, F128::ONE);
            }
        }
    }

    #[test]
    fn tri_depth2_matches_incumbent_round67() {
        let mut rng = Rng::new(0x7121_0067);
        for log_out in [3usize, 6, 10, 14, 19] {
            let n = 4usize << log_out;
            let a: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let b: Vec<F128> = (0..n).map(|_| rng.f128()).collect();
            let (rho3, rho4) = (rng.f128(), rng.f128());
            let mut r_next6 = vec![F128::ONE; log_out];
            for s in r_next6[1..].iter_mut() {
                *s = rng.f128();
            }
            let mut a_i = vec![LA_POISON; n / 4];
            let mut b_i = vec![LA_POISON; n / 4];
            let (i1, iinf, ila) =
                fold2_plain_and_round67_into(&a, &b, &mut a_i, &mut b_i, rho3, rho4, &r_next6);
            let mut a_t = vec![LA_POISON; n / 4];
            let mut b_t = vec![LA_POISON; n / 4];
            let (t1, tinf, tla, _) =
                fold_cascade_t_into(&a, &b, &mut a_t, &mut b_t, &[rho3, rho4], &r_next6);
            assert!(a_i == a_t && b_i == b_t, "outputs log_out {log_out}");
            assert_eq!((i1, iinf, ila), (t1, tinf, tla), "messages log_out {log_out}");
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    #[test]
    #[ignore]
    fn t3_real_chunk_tri_matches() {
        let d: std::path::PathBuf = std::env::var_os("ZK_DIR").expect("ZK_DIR").into();
        let (a, b, eq_lo, rho3, rho4) = t3_probe::load(&d);
        let nck = a.len() / (8 * eq_lo.len());
        let ci = 8 * eq_lo.len();
        let co = 2 * eq_lo.len();
        let ep = kernels::aarch64::round2_eq_prep(&eq_lo);
        for c in 0..nck {
            let (ai, bi) = (&a[c * ci..(c + 1) * ci], &b[c * ci..(c + 1) * ci]);
            let mut a1 = vec![LA_POISON; co];
            let mut b1 = vec![LA_POISON; co];
            let inc = kernels::aarch64::fold2_and_message_lookahead_nt_expanded_aarch64(
                ai, bi, &mut a1, &mut b1, rho3, rho4, rho3 * rho4, &eq_lo,
            );
            let mut a2 = vec![LA_POISON; co];
            let mut b2 = vec![LA_POISON; co];
            let tri = kernels::aarch64::fold_tri_and_message_aarch64::<2, true, 0>(
                ai, bi, &mut a2, &mut b2, rho3, rho4, F128::ZERO, &eq_lo, &ep,
            );
            let mut a3 = vec![LA_POISON; co];
            let mut b3 = vec![LA_POISON; co];
            let sc = tri_chunk_scalar(ai, bi, &mut a3, &mut b3, &[rho3, rho4], &eq_lo);
            assert!(a1 == a2 && b1 == b2 && a2 == a3 && b2 == b3, "chunk {c} outputs");
            assert_eq!(tri, sc, "chunk {c} aggregates");
            let _ = inc;
        }
        println!("t3: {nck} real chunks: tri kernel == incumbent outputs == portable");
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    #[test]
    #[ignore]
    fn t3_tail_timing() {
        std::thread::spawn(t3_probe::run).join().unwrap();
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    mod t3_probe {
        use super::super::kernels::aarch64::*;
        use super::*;
        use std::time::Instant;

        pub(super) fn f128s(path: &std::path::Path) -> Vec<F128> {
            let b = std::fs::read(path).unwrap();
            b.chunks_exact(16)
                .map(|c| F128 {
                    lo: u64::from_le_bytes(c[..8].try_into().unwrap()),
                    hi: u64::from_le_bytes(c[8..].try_into().unwrap()),
                })
                .collect()
        }

        pub(super) fn load(d: &std::path::Path) -> (Vec<F128>, Vec<F128>, Vec<F128>, F128, F128) {
            let a = f128s(&d.join("t3_a.bin"));
            let b = f128s(&d.join("t3_b.bin"));
            let eq = f128s(&d.join("t3_eqlo.bin"));
            let r = f128s(&d.join("t3_rho.bin"));
            (a, b, eq, r[0], r[1])
        }

        fn qos() {
            // macOS-only call: elsewhere (aarch64 Linux test runs) a no-op.
            #[cfg(target_os = "macos")]
            {
                unsafe extern "C" {
                    fn pthread_set_qos_class_self_np(q: u32, p: i32) -> i32;
                }
                unsafe { pthread_set_qos_class_self_np(0x21, 0) };
            }
        }

        fn lq(v: &mut [f64]) -> (f64, f64) {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (v[v.len() / 4], v[v.len() / 2])
        }

        type Kern<'a> = Box<dyn FnMut(usize) + 'a>;

        fn bench(name: &str, forms: &mut [(String, Kern<'_>, f64)], chunks: usize, rounds: usize) {
            let mut t = vec![Vec::new(); forms.len()];
            for f in forms.iter_mut() {
                for c in 0..chunks {
                    (f.1)(c);
                }
            }
            for r in 0..rounds {
                for k in 0..forms.len() {
                    let i = (k + r) % forms.len();
                    let t0 = Instant::now();
                    for c in 0..chunks {
                        (forms[i].1)(c);
                    }
                    t[i].push(t0.elapsed().as_nanos() as f64 / (forms[i].2 * chunks as f64));
                }
            }
            println!("{name}");
            for (f, v) in forms.iter().zip(t.iter_mut()) {
                let (l, m) = lq(v);
                println!("  {:<52} lq {l:7.3}  med {m:7.3} ns/input-pair", f.0);
            }
        }

        pub(super) fn run() {
            qos();
            let mut rng = Rng::new(0x7121_7177);
            let (ra, rb, eq56, rho3, rho4) = match std::env::var_os("ZK_DIR") {
                Some(d) => load(std::path::Path::new(&d)),
                None => {
                    let ci = 4 * 8192;
                    (
                        (0..4 * ci).map(|_| rng.f128()).collect(),
                        (0..4 * ci).map(|_| rng.f128()).collect(),
                        (0..4096).map(|_| rng.f128()).collect(),
                        rng.f128(),
                        rng.f128(),
                    )
                }
            };
            let rho5 = rng.f128();
            let lo = eq56.len();
            let ci = 8 * lo;
            let co = 2 * lo;
            println!("t3 probe: lo {lo}, chunk_in {ci}, real={}", std::env::var_os("ZK_DIR").is_some());

            let a_c = &ra[..ci];
            let b_c = &rb[..ci];
            let mut ao = vec![F128::ZERO; co];
            let mut bo = vec![F128::ZERO; co];
            let ao_p = ao.as_mut_ptr();
            let bo_p = bo.as_mut_ptr();
            let eq8 = &eq56[..lo / 2];
            let (ep56, ep8) = (round2_eq_prep(&eq56), round2_eq_prep(eq8));
            macro_rules! outs {
                ($n:expr) => {
                    unsafe {
                        (
                            std::slice::from_raw_parts_mut(ao_p, $n),
                            std::slice::from_raw_parts_mut(bo_p, $n),
                        )
                    }
                };
            }
            let cif = ci as f64;
            let mut forms: Vec<(String, Kern<'_>, f64)> = vec![
                (
                    "5+6 incumbent (lookahead, NT, 16 prod/8 out)".into(),
                    Box::new(|_| {
                        let (x, y) = outs!(co);
                        std::hint::black_box(fold2_and_message_lookahead_nt_expanded_aarch64(
                            a_c, b_c, x, y, rho3, rho4, rho3 * rho4, &eq56,
                        ));
                    }),
                    cif,
                ),
                (
                    "7+8 incumbent kernel (lookahead, normal stores)".into(),
                    Box::new(|_| {
                        let (x, y) = outs!(co);
                        std::hint::black_box(fold2_and_message_lookahead_normal_expanded_aarch64(
                            a_c, b_c, x, y, rho3, rho4, rho3 * rho4, &eq56,
                        ));
                    }),
                    cif,
                ),
                (
                    "11+12 incumbent kernel (direct, normal stores)".into(),
                    Box::new(|_| {
                        let (x, y) = outs!(co);
                        std::hint::black_box(fold2_and_message_normal_expanded_aarch64(
                            a_c, b_c, x, y, rho3, rho4, rho3 * rho4, &eq56,
                        ));
                    }),
                    cif,
                ),
                (
                    "tri d2 NT (5+6 +la8, 26 prod/8 out)".into(),
                    Box::new(|_| {
                        let (x, y) = outs!(co);
                        std::hint::black_box(fold_tri_and_message_aarch64::<2, true, 0>(
                            a_c, b_c, x, y, rho3, rho4, F128::ZERO, &eq56, &ep56,
                        ));
                    }),
                    cif,
                ),
                (
                    "tri d2 NT, no products (ABL 1)".into(),
                    Box::new(|_| {
                        let (x, y) = outs!(co);
                        std::hint::black_box(fold_tri_and_message_aarch64::<2, true, 1>(
                            a_c, b_c, x, y, rho3, rho4, F128::ZERO, &eq56, &ep56,
                        ));
                    }),
                    cif,
                ),
                (
                    "tri d2 NT, no stores (ABL 2)".into(),
                    Box::new(|_| {
                        let (x, y) = outs!(co);
                        std::hint::black_box(fold_tri_and_message_aarch64::<2, true, 2>(
                            a_c, b_c, x, y, rho3, rho4, F128::ZERO, &eq56, &ep56,
                        ));
                    }),
                    cif,
                ),
                (
                    "tri d2, fold only (ABL 3)".into(),
                    Box::new(|_| {
                        let (x, y) = outs!(co);
                        std::hint::black_box(fold_tri_and_message_aarch64::<2, true, 3>(
                            a_c, b_c, x, y, rho3, rho4, F128::ZERO, &eq56, &ep56,
                        ));
                    }),
                    cif,
                ),
                (
                    "tri d3 normal (T pass, 26 prod/8 out)".into(),
                    Box::new(|_| {
                        let (x, y) = outs!(co / 2);
                        std::hint::black_box(fold_tri_and_message_aarch64::<3, false, 0>(
                            a_c, b_c, x, y, rho3, rho4, rho5, eq8, &ep8,
                        ));
                    }),
                    cif,
                ),
                (
                    "tri d3 NT (T pass)".into(),
                    Box::new(|_| {
                        let (x, y) = outs!(co / 2);
                        std::hint::black_box(fold_tri_and_message_aarch64::<3, true, 0>(
                            a_c, b_c, x, y, rho3, rho4, rho5, eq8, &ep8,
                        ));
                    }),
                    cif,
                ),
                (
                    "tri d3, no products (ABL 1)".into(),
                    Box::new(|_| {
                        let (x, y) = outs!(co / 2);
                        std::hint::black_box(fold_tri_and_message_aarch64::<3, false, 1>(
                            a_c, b_c, x, y, rho3, rho4, rho5, eq8, &ep8,
                        ));
                    }),
                    cif,
                ),
            ];
            bench("CACHE-RESIDENT (one chunk reused)", &mut forms, 1, 41);
            drop(forms);

            let big = 1usize << 24;
            let mut sa: Vec<F128> = Vec::with_capacity(big);
            let mut sb: Vec<F128> = Vec::with_capacity(big);
            for i in 0..big {
                sa.push(ra[i % ra.len()]);
                sb.push(rb[i % rb.len()]);
            }
            let mut oa = vec![F128::ZERO; big / 4];
            let mut ob = vec![F128::ZERO; big / 4];
            let (oa_p, ob_p) = (oa.as_mut_ptr(), ob.as_mut_ptr());
            let nchunks = big / ci;
            let (sa_r, sb_r) = (&sa, &sb);
            macro_rules! sl {
                ($c:expr, $per_out:expr) => {{
                    let o = $c * (ci / $per_out);
                    unsafe {
                        (
                            &sa_r[$c * ci..($c + 1) * ci],
                            &sb_r[$c * ci..($c + 1) * ci],
                            std::slice::from_raw_parts_mut(oa_p.add(o), ci / $per_out),
                            std::slice::from_raw_parts_mut(ob_p.add(o), ci / $per_out),
                        )
                    }
                }};
            }
            let mut forms: Vec<(String, Kern<'_>, f64)> = vec![
                (
                    "5+6 incumbent NT, DRAM".into(),
                    Box::new(|c| {
                        let (x, y, p, q) = sl!(c, 4);
                        std::hint::black_box(fold2_and_message_lookahead_nt_expanded_aarch64(
                            x, y, p, q, rho3, rho4, rho3 * rho4, &eq56,
                        ));
                    }),
                    cif,
                ),
                (
                    "tri d2 NT, DRAM".into(),
                    Box::new(|c| {
                        let (x, y, p, q) = sl!(c, 4);
                        std::hint::black_box(fold_tri_and_message_aarch64::<2, true, 0>(
                            x, y, p, q, rho3, rho4, F128::ZERO, &eq56, &ep56,
                        ));
                    }),
                    cif,
                ),
                (
                    "tri d2 NT no products, DRAM".into(),
                    Box::new(|c| {
                        let (x, y, p, q) = sl!(c, 4);
                        std::hint::black_box(fold_tri_and_message_aarch64::<2, true, 1>(
                            x, y, p, q, rho3, rho4, F128::ZERO, &eq56, &ep56,
                        ));
                    }),
                    cif,
                ),
                (
                    "tri d3 NT, DRAM".into(),
                    Box::new(|c| {
                        let (x, y, p, q) = sl!(c, 8);
                        std::hint::black_box(fold_tri_and_message_aarch64::<3, true, 0>(
                            x, y, p, q, rho3, rho4, rho5, eq8, &ep8,
                        ));
                    }),
                    cif,
                ),
            ];
            bench("DRAM STREAM (2^24 inputs/side, one core)", &mut forms, nchunks, 7);
        }
    }
}

// r567 archive marker: submission cadence variant.

// r577 cadence marker: candidate identity only; no runtime effect.

// r581: source-distinct cadence marker; no semantic effect.

// r584: source-distinct cadence marker
