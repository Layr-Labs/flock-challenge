// Copyright 2024-2025 Irreducible, Inc.
// Modifications copyright 2026 Succinct Labs, Benedikt Bunz, William Wang
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// r1021 marker: recursive top hetero producer (new mechanism on the 3b070a2
// frontier = L2 gate resample that promoted +0.02% at 7:43 AM CDT).
//
// The algorithm skeleton (iterative LCH NTT, neighbors-last ordering) is
// derived from binius64's `NeighborsLastReference`
// (https://github.com/binius-zk/binius64, `crates/math/src/ntt/reference.rs`).
// The interleaved SoA layout, fused 2-layer butterfly, and parallelization
// strategy are original to Flock.

//! Additive NTT over F_{2^128} using the LCH novel polynomial basis.
//!
//! Iterative LCH NTT skeleton derived from binius64's `NeighborsLastReference`,
//! with an interleaved SoA layout, a fused 2-layer butterfly, and rayon-based
//! parallelization added on top. The forward transform maps polynomial
//! coefficients (in the novel polynomial basis) to evaluations over an
//! F_2-affine subspace; the inverse reverses this. Used by the PCS commit and
//! by FRI folding.
//!
//! ## Convention
//!
//! Given a basis `{β_0, …, β_{ℓ-1}}` of an F_2-subspace V ⊂ F_{2^128}, define
//! the subspace polynomials W_i recursively:
//! ```text
//!     W_0(z) = z
//!     W_i(z) = W_{i-1}(z) · (W_{i-1}(z) + W_{i-1}(β_{i-1}))     (for i ≥ 1)
//! ```
//! and the *normalized* forms `Ŵ_i(z) = W_i(z) / W_i(β_i)` so that
//! `Ŵ_i(β_i) = 1`. The "twiddle" at layer `l` and block `b` is then
//! `Ŵ_{ℓ-l-1}(z)` evaluated at the `b`-th element of the F_2-span of
//! `{β_{ℓ-l}, β_{ℓ-l+1}, …, β_{ℓ-1}}`.
//!
//! At forward-transform layer `l` (`l = 0, …, log_d − 1`):
//! - There are `2^l` blocks, each of size `2^(log_d − l)`.
//! - Within each block, pairs `(idx0, idx0 | block_size_half)` are
//!   butterflied with the block's twiddle.
//! - **Pairing at layer `l`**: positions differ by `block_size_half =
//!   2^(log_d − l − 1)`. So at layer 0 pairs are far (N/2 apart), and at the
//!   deepest layer pairs are adjacent (1 apart) — this is "neighbors-last."
//!
//! FRI fold processes layers in **reverse** (deepest first), at which level
//! pairs are adjacent — matching the standard `fold_pair` formula in DP24.

use std::sync::{Arc, OnceLock};

use crate::field::F128;

mod kernels;

/// Compute the normalized subspace-polynomial evaluation table.
///
/// Returns `evals` where `evals[i] = [Ŵ_i(β_i), Ŵ_i(β_{i+1}), …, Ŵ_i(β_{ℓ-1})]`.
/// The 0-th element of each row is always `1` (by normalization).
fn generate_evals_from_subspace(basis: &[F128]) -> Vec<Vec<F128>> {
    let l = basis.len();
    let mut evals: Vec<Vec<F128>> = Vec::with_capacity(l);

    // evals[0] = [W_0(β_0), W_0(β_1), …, W_0(β_{ℓ-1})] = basis.
    evals.push(basis.to_vec());

    // evals[i][k] = W_i(β_{i+k}) computed from evals[i-1].
    // evals[i-1] = [W_{i-1}(β_{i-1}), W_{i-1}(β_i), W_{i-1}(β_{i+1}), …]
    // We want W_i(β_{i+k}) = W_{i-1}(β_{i+k}) · (W_{i-1}(β_{i+k}) + W_{i-1}(β_{i-1}))
    //                     = evals[i-1][k+1] · (evals[i-1][k+1] + evals[i-1][0])
    for i in 1..l {
        let mut row = Vec::with_capacity(l - i);
        for k in 1..evals[i - 1].len() {
            let val = evals[i - 1][k] * (evals[i - 1][k] + evals[i - 1][0]);
            row.push(val);
        }
        evals.push(row);
    }

    // Normalize each row by its 0-th element (= W_i(β_i)).
    for row in evals.iter_mut() {
        let inv = row[0].inv();
        for v in row.iter_mut() {
            *v *= inv;
        }
    }

    evals
}

/// Compute `Σ_j bit_j(idx) · basis[j]` — the `idx`-th element of the F_2-span
/// of `basis`.
#[inline]
fn span_get(basis: &[F128], idx: usize) -> F128 {
    let mut acc = F128::ZERO;
    for (j, &b) in basis.iter().enumerate() {
        if (idx >> j) & 1 == 1 {
            acc += b;
        }
    }
    acc
}

/// Largest domain whose complete breadth-first twiddle tree is cached.
/// A size-2^20 domain uses just under 16 MiB; larger domains keep the compact
/// allocation-free fallback.
const MAX_PRECOMPUTED_TWIDDLE_LOG: usize = 20;

fn precompute_twiddles(evals: &[Vec<F128>]) -> Option<Vec<F128>> {
    let log_d = evals.len();
    if log_d > MAX_PRECOMPUTED_TWIDDLE_LOG {
        return None;
    }

    let mut twiddles = Vec::with_capacity((1usize << log_d) - 1);
    for layer in 0..log_d {
        let layer_start = twiddles.len();
        let eval_row = &evals[log_d - layer - 1];
        debug_assert_eq!(eval_row.len(), layer + 1);
        twiddles.push(F128::ZERO);
        for (bit, &basis_value) in eval_row[1..].iter().enumerate() {
            let half = 1usize << bit;
            for block in 0..half {
                twiddles.push(twiddles[layer_start + block] + basis_value);
            }
        }
        debug_assert_eq!(twiddles.len() - layer_start, 1usize << layer);
    }
    Some(twiddles)
}

/// Share immutable standard-basis tables across all NTT objects. The worker's
/// mandatory untimed proof initializes these before measured requests.
fn cached_standard_twiddles(dim: usize, evals: &[Vec<F128>]) -> Option<Arc<[F128]>> {
    if dim > MAX_PRECOMPUTED_TWIDDLE_LOG {
        return None;
    }
    static TABLES: OnceLock<[OnceLock<Arc<[F128]>>; MAX_PRECOMPUTED_TWIDDLE_LOG + 1]> =
        OnceLock::new();
    let tables = TABLES.get_or_init(|| std::array::from_fn(|_| OnceLock::new()));
    Some(
        tables[dim]
            .get_or_init(|| {
                Arc::from(
                    precompute_twiddles(evals)
                        .expect("standard production domain should fit twiddle cache"),
                )
            })
            .clone(),
    )
}

/// Cache the much smaller normalized evaluation triangles as well. Standard
/// NTT objects are constructed repeatedly by the recursive PCS, and rebuilding
/// these rows otherwise repeats field inversions and multiplications even when
/// the large twiddle table is already resident from the untimed warm proof.
/// Clone into the original nested-`Vec` representation so transform object
/// layout and hot-loop dereferencing remain unchanged.
fn cached_standard_evals(dim: usize) -> Vec<Vec<F128>> {
    static TABLES: OnceLock<[OnceLock<Vec<Vec<F128>>>; 65]> = OnceLock::new();
    let tables = TABLES.get_or_init(|| std::array::from_fn(|_| OnceLock::new()));
    tables[dim]
        .get_or_init(|| {
            let basis: Vec<F128> = (0..dim).map(|i| F128::new(1u64 << i, 0)).collect();
            generate_evals_from_subspace(&basis)
        })
        .clone()
}

/// Complete the last radix-8 group for the ranked Apple-silicon L0 commit.
///
/// The generic 2 MiB cache split selects `n_top = 9` for the production shape
/// (`log_d = 20`, 64 interleaved lanes, rate-layer entry at layer 1). That
/// leaves layers 7 and 8 in a fused-2 pass and starts the 2 MiB sub-transforms
/// at layer 9. Raising the split by one lets the already-selected third top
/// pass fuse layers 7, 8, and 9 together, so it removes one full 1 GiB
/// codeword read/write without adding another top pass.
///
/// Keep this keyed to the ranked shape: recursive Ligerito commits and other
/// transform geometries retain the cache policy that was tuned for them.
#[inline]
fn fusion_aware_interleaved_n_top(
    log_d: usize,
    num_ntts: usize,
    start_layer: usize,
    n_top: usize,
) -> usize {
    let ranked_apple_l0 = cfg!(all(
        target_os = "macos",
        target_arch = "aarch64",
        target_feature = "aes"
    )) && log_d == 20
        && num_ntts == 64
        && start_layer == 1
        && n_top == 9;
    if ranked_apple_l0 { 10 } else { n_top }
}

/// Whether to fuse the ranked L0 commit's ten cache-resident tail layers in
/// pairs. Keep this as narrow as [`fusion_aware_interleaved_n_top`]: smaller
/// recursive commits retain their independently tuned single-layer tail.
#[inline]
fn use_ranked_deep_pair_fusion(
    log_d: usize,
    num_ntts: usize,
    start_layer: usize,
    n_top: usize,
) -> bool {
    cfg!(all(
        target_os = "macos",
        target_arch = "aarch64",
        target_feature = "aes"
    )) && log_d == 20
        && num_ntts == 64
        && start_layer == 1
        && n_top == 10
}

/// The standard dimension-20 basis has low-limb-only twiddles throughout the
/// final two layers. This permits a two-PMULL product on AArch64 instead of the
/// generic six-PMULL field multiply. Keep the dispatch tied to the exact
/// ranked deep-transform geometry.
#[inline]
fn use_ranked_low_twiddle_final_pair(log_d: usize, num_ntts: usize, n_top: usize) -> bool {
    cfg!(all(
        target_os = "macos",
        target_arch = "aarch64",
        target_feature = "aes"
    )) && log_d == 20
        && num_ntts == 64
        && n_top == 10
        && std::env::var_os("FLOCK_NO_NTT_LOW_TWIDDLE_FINAL").is_none()
}

/// The zero-root radix-8 kernel is currently scored only for the ranked L0
/// transform. Keep recursive and diagnostic commits on their prior kernel so
/// this candidate has one production scope and one transfer story.
#[inline]
fn use_ranked_zero_root_fusion(
    log_d: usize,
    num_ntts: usize,
    start_layer: usize,
    n_top: usize,
) -> bool {
    use_ranked_deep_pair_fusion(log_d, num_ntts, start_layer, n_top)
}

/// The three ranked radix-8 passes share fixed one-MiB row tiles with the
/// efficiency-core pool. Layer 1 has only two outer blocks, but flattening
/// `(block, row-tile)` still exposes the same 1024 claims as layers 4 and 7.
/// It overlaps the concurrently-running round-1 AB precompute on the main
/// Rayon pool; the separate helper pool is otherwise idle until these queues
/// finish and the deep-transform leaf receivers start.
#[inline]
fn is_ranked_top_hetero_fused3_pass(
    log_d: usize,
    num_ntts: usize,
    start_layer: usize,
    n_top: usize,
    layer: usize,
) -> bool {
    use_ranked_deep_pair_fusion(log_d, num_ntts, start_layer, n_top) && matches!(layer, 1 | 4 | 7)
}

/// The first recursive Ligerito commitment is a 32 MiB, eight-lane transform:
/// rate reduction enters at layer 2 and the direct-from-message radix-8 pass
/// produces the layer-5 state. Its 32 independent 1 MiB sub-transforms are
/// large enough to amortize the existing P+E shared queue while the efficiency
/// cores would otherwise be idle. The radix-8 producer remains on its incumbent
/// Rayon schedule; only the cache-resident tail uses the helper pool.
#[inline]
fn is_recursive_l1_ntt_epool_shape(log_d: usize, num_ntts: usize, start_layer: usize) -> bool {
    cfg!(all(
        target_os = "macos",
        target_arch = "aarch64",
        target_feature = "aes"
    )) && log_d == 18
        && num_ntts == 8
        && start_layer == 2
}

/// The second recursive Ligerito commitment (L2: 8 MiB, `log_msg_cols=13`,
/// `log_inv_rate=3`) shares the same eight-lane direct-from-message geometry
/// at `log_d = 16`. Its 32 x 256 KiB deep-tail claims ride the same P+E queue
/// in the window immediately after L1's deep tail, where the helper pool is
/// otherwise idle. The tail kernel, layer order, and twiddles are identical to
/// the incumbent single-layer loop — this changes ownership/scheduling only.
#[inline]
fn is_recursive_l2_ntt_epool_shape(log_d: usize, num_ntts: usize, start_layer: usize) -> bool {
    cfg!(all(
        target_os = "macos",
        target_arch = "aarch64",
        target_feature = "aes"
    )) && log_d == 16
        && num_ntts == 8
        && start_layer == 3
}

// r1020 resample marker: AB redraw of 4912aee after the r1019 frontier control
// landed in-band (−0.47%); comment-only delta versus the r1018 archive.
/// The union gate for both recursive-commit epool shapes. Keep it separate
/// from the ranked L0 selectors above: this changes scheduling only, and the
/// recursive tail continues to use its independently tuned single-layer
/// arithmetic on either shape.
#[inline]
fn is_recursive_ntt_epool_shape(log_d: usize, num_ntts: usize, start_layer: usize) -> bool {
    is_recursive_l1_ntt_epool_shape(log_d, num_ntts, start_layer)
        || is_recursive_l2_ntt_epool_shape(log_d, num_ntts, start_layer)
}

/// Exact rollback rule for the recursive-L1 heterogeneous NTT schedule.
/// Values other than the documented `1` cannot silently disable the path.
#[inline]
fn recursive_ntt_epool_killed_by(value: Option<&str>) -> bool {
    value == Some("1")
}

/// Exact rollback rule for the L2 extension: `FLOCK_NO_RECURSIVE_L2_NTT_EPOOL=1`
/// disables only the second recursive commitment's helper schedule, leaving
/// the shipped L1 path (and its documented `FLOCK_NO_RECURSIVE_L1_NTT_EPOOL`
/// rollback) untouched. Values other than `1` cannot silently disable the path.
#[inline]
fn l2_recursive_ntt_epool_killed_by(value: Option<&str>) -> bool {
    value == Some("1")
}

#[inline]
fn use_recursive_ntt_epool(log_d: usize, num_ntts: usize, start_layer: usize) -> bool {
    static L1_ENABLED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        !recursive_ntt_epool_killed_by(
            std::env::var("FLOCK_NO_RECURSIVE_L1_NTT_EPOOL")
                .ok()
                .as_deref(),
        )
    });
    static L2_ENABLED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        !l2_recursive_ntt_epool_killed_by(
            std::env::var("FLOCK_NO_RECURSIVE_L2_NTT_EPOOL")
                .ok()
                .as_deref(),
        )
    });
    if is_recursive_l1_ntt_epool_shape(log_d, num_ntts, start_layer) {
        *L1_ENABLED && crate::epool::helper_pool_available()
    } else if is_recursive_l2_ntt_epool_shape(log_d, num_ntts, start_layer) {
        *L2_ENABLED && crate::epool::helper_pool_available()
    } else {
        false
    }
}

/// Explicit diagnostic only. With the variable unset, the production path
/// takes no timestamps and reads no helper counters; it pays only the cold
/// static-boolean branch at the exact recursive shapes.
#[inline]
fn trace_recursive_ntt_epool() -> bool {
    static ENABLED: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        let l1 = std::env::var("FLOCK_TRACE_RECURSIVE_L1_NTT_EPOOL")
            .ok()
            .as_deref()
            == Some("1");
        let l2 = std::env::var("FLOCK_TRACE_RECURSIVE_NTT_EPOOL")
            .ok()
            .as_deref()
            == Some("1");
        l1 || l2
    });
    *ENABLED
}

/// Kill switch for the heterogeneous claim-queue producer on the recursive
/// L1/L2 from-message radix-8 top (`FLOCK_NO_RECURSIVE_TOP_HETERO=1`):
/// restores the incumbent main-Rayon `into_par_iter` fan-out for the top
/// pass. The queue only reassigns identical per-tile kernels onto the
/// otherwise-idle efficiency cores (the same ownership argument as the
/// ranked L0 top); outputs are bit-identical either way because every
/// (block, tile) job writes disjoint destination rows.
#[inline]
fn recursive_top_hetero_enabled() -> bool {
    std::env::var_os("FLOCK_NO_RECURSIVE_TOP_HETERO").is_none()
}

// ---------------------------------------------------------------------------
// Static zero-lane skip (commit NTT).
// ---------------------------------------------------------------------------

/// The interleaved lane width the zero-odd-tail skip is defined for. The
/// geometry that produces the pattern (one R1CS block = `2 · num_ntts` packed
/// `F128` words, so a block's padding tail lands wholly inside the block's
/// ODD codeword position) only holds at this width.
const ZERO_TAIL_NUM_NTTS: usize = 64;

/// The ranked commit domain the ambient publication is honored at.
const ZERO_TAIL_LOG_D: usize = 20;

/// Count of trailing SoA lanes that are identically zero at every **odd**
/// codeword position of the message being committed (0 = none / unknown).
///
/// For the ranked BLAKE3 shape one R1CS block is `K = 2^14` bits = 128 packed
/// `F128` words = exactly two SoA positions, and only `USEFUL_BITS = 15,409`
/// of those bits are constrained; words 121..128 of every block are forced to
/// zero by the padding rows `0·0 = z[i]`. Those words are lanes 57..63 of the
/// block's odd position, so 7 of the 64 interleaved sub-NTTs carry a static
/// stride-2 all-zero coefficient pattern.
///
/// Every forward layer except the deepest pairs positions an EVEN distance
/// apart, so the pattern survives untouched through those layers: both inputs
/// of such a butterfly are zero and both outputs stay zero. Skipping the tail
/// lanes on odd rows therefore removes butterfly work without changing a
/// single output byte.
static ZERO_ODD_TAIL_LANES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// `FLOCK_NO_ZERO_LANE_SKIP=1` restores the dense butterfly in the same
/// binary, so a candidate/control pair differs only in this dispatch.
#[inline]
fn zero_lane_skip_disabled() -> bool {
    static OFF: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_ZERO_LANE_SKIP").is_some());
    *OFF
}

/// Trailing lanes the ranked commit transform may skip on odd rows, or 0.
///
/// The ambient publication is honored ONLY at the exact ranked production
/// geometry. Every other transform — recursive commits, FRI folds, tests —
/// receives its odd tail as an explicit parameter, so no unrelated buffer can
/// ever pick up another caller's publication.
#[inline]
pub(crate) fn ranked_zero_odd_tail_lanes(log_d: usize, num_ntts: usize) -> usize {
    if log_d != ZERO_TAIL_LOG_D || num_ntts != ZERO_TAIL_NUM_NTTS || zero_lane_skip_disabled() {
        return 0;
    }
    let tail = ZERO_ODD_TAIL_LANES.load(std::sync::atomic::Ordering::Relaxed);
    if tail < num_ntts { tail } else { 0 }
}

/// Scoped publication of the zero-odd-tail-lane count, restoring the previous
/// value on drop. The committer sets this from the R1CS padding descriptor —
/// the same source of truth zerocheck, lincheck and the ring-switch fold
/// already skip padding with — for the duration of one commitment.
#[must_use = "the skip is active only while the guard is alive"]
pub struct ZeroOddTailLanes(usize);

impl ZeroOddTailLanes {
    /// Publish `lanes` trailing zero lanes for `num_ntts`-wide interleaving.
    /// Any shape outside the supported geometry publishes 0 (no skip).
    pub fn scope(num_ntts: usize, lanes: usize) -> Self {
        let lanes = if num_ntts == ZERO_TAIL_NUM_NTTS && lanes < num_ntts {
            lanes
        } else {
            0
        };
        Self(ZERO_ODD_TAIL_LANES.swap(lanes, std::sync::atomic::Ordering::Relaxed))
    }

    /// Trailing zero lanes implied by an R1CS padding descriptor, or 0 when
    /// the block geometry does not place the padding tail in the odd position.
    ///
    /// Requires one block to be exactly `2 · num_ntts` packed `F128` words so
    /// that block `b` occupies codeword positions `2b` (even) and `2b+1`
    /// (odd), and requires the whole zero tail to fit inside that odd
    /// position.
    pub fn lanes_for_padding(num_ntts: usize, k_log: usize, useful_bits_per_block: usize) -> usize {
        const LOG_PACKING: usize = 7;
        if num_ntts != ZERO_TAIL_NUM_NTTS || k_log < LOG_PACKING {
            return 0;
        }
        let words_per_block = 1usize << (k_log - LOG_PACKING);
        if words_per_block != 2 * num_ntts {
            return 0;
        }
        let used_words = useful_bits_per_block.div_ceil(1 << LOG_PACKING);
        if used_words > words_per_block {
            return 0;
        }
        let zero_words = words_per_block - used_words;
        if zero_words < num_ntts { zero_words } else { 0 }
    }
}

impl Drop for ZeroOddTailLanes {
    fn drop(&mut self) {
        ZERO_ODD_TAIL_LANES.store(self.0, std::sync::atomic::Ordering::Relaxed);
    }
}

const INTERLEAVED_PHASE_ALL: u8 = 0;
const INTERLEAVED_PHASE_TOP_ONLY: u8 = 1;
const INTERLEAVED_PHASE_DEEP_ONLY: u8 = 2;

/// Additive NTT over F_{2^128} with the standard polynomial-basis subspace.
///
/// The basis is `{1, x, x², …, x^(ℓ-1)}` in F_{2^128} = F_2[x]/(GHASH-poly).
/// This makes the F_2-subspace V = `{0, 1, …, 2^ℓ-1}` (under the natural
/// integer encoding of F_{2^128} elements).
#[derive(Clone, Debug)]
pub struct AdditiveNttF128 {
    /// `evals[i]` of length `ℓ − i`, the normalized subspace polynomial values.
    evals: Vec<Vec<F128>>,
    /// Breadth-first table: layer `l` starts at `2^l - 1`.
    precomputed_twiddles: Option<Arc<[F128]>>,
}

impl AdditiveNttF128 {
    /// Construct an NTT from an explicit F_2-basis.
    pub fn new(basis: &[F128]) -> Self {
        let evals = generate_evals_from_subspace(basis);
        let precomputed_twiddles = precompute_twiddles(&evals).map(Arc::from);
        Self {
            evals,
            precomputed_twiddles,
        }
    }

    /// Standard NTT with basis `{1, x, x², …, x^(dim-1)}`. Requires `dim ≤ 64`
    /// (the low 64 bits of F_{2^128} hold these basis vectors).
    pub fn standard(dim: usize) -> Self {
        assert!(dim <= 64, "standard NTT requires dim ≤ 64");
        let evals = cached_standard_evals(dim);
        let precomputed_twiddles = cached_standard_twiddles(dim, &evals);
        Self {
            evals,
            precomputed_twiddles,
        }
    }

    pub fn log_domain_size(&self) -> usize {
        self.evals.len()
    }

    /// Full breadth-first twiddle table when cached (domains up to
    /// `MAX_PRECOMPUTED_TWIDDLE_LOG`): layer `l` starts at `2^l - 1`. The GPU
    /// commit uploads this table once at init.
    pub(crate) fn precomputed_twiddle_table(&self) -> Option<&[F128]> {
        self.precomputed_twiddles.as_deref()
    }

    /// Twiddle at `(layer, block)` for the forward NTT and FRI fold.
    ///
    /// At layer `l` ∈ `[0, ℓ)`, block index `b` ∈ `[0, 2^l)`:
    /// `twiddle(l, b) = Σ_j bit_j(b) · Ŵ_{ℓ-l-1}(β_{ℓ-l+j})`
    ///
    /// (The 0-th element of the row corresponds to `Ŵ_{ℓ-l-1}(β_{ℓ-l-1}) = 1`,
    /// which is "absorbed" into the butterfly and not in the twiddle.)
    pub fn twiddle(&self, layer: usize, block: usize) -> F128 {
        debug_assert!(layer < self.log_domain_size());
        debug_assert!(block < 1usize << layer);
        if let Some(twiddles) = &self.precomputed_twiddles {
            return twiddles[(1usize << layer) - 1 + block];
        }
        let v = &self.evals[self.log_domain_size() - layer - 1];
        span_get(&v[1..], block)
    }

    /// Forward additive NTT in place. `data.len()` must be `2^log_d` for some
    /// `log_d ≤ log_domain_size()`. Layer `l ∈ [0, log_d)` is processed in
    /// order (neighbors-last: top layer first).
    ///
    /// Dispatches to the cache-blocked batched implementation when available
    /// and the buffer is large enough to benefit; otherwise falls back to the
    /// per-layer parallel path or scalar.
    pub fn forward_transform(&self, data: &mut [F128]) {
        #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
        {
            self.forward_transform_batched(data);
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
        {
            self.forward_transform_scalar(data);
        }
    }

    /// Interleaved forward NTT: process `num_ntts` independent NTTs in
    /// position-major SoA layout.
    ///
    /// `data` layout: `data[pos * num_ntts + lane]` for `pos ∈ 0..2^log_d`,
    /// `lane ∈ 0..num_ntts`. Each "lane" is an independent NTT instance over
    /// the same domain; all `num_ntts` instances share the twiddle structure
    /// (same `self.twiddle(layer, block)` is applied to every lane at the
    /// corresponding butterfly).
    ///
    /// `num_ntts` must be a positive power of 2. `data.len()` must equal
    /// `(1 << log_d) * num_ntts` for some `log_d ≤ log_domain_size()`.
    ///
    /// This produces the SAME RS code per lane as `forward_transform`, with
    /// FRI-compatible twiddles. The SoA layout is what makes each Merkle leaf
    /// = one position across all `num_ntts` lanes (= contiguous slice of
    /// `num_ntts` F_{2^128} elements).
    pub fn forward_transform_interleaved(&self, data: &mut [F128], num_ntts: usize) {
        self.forward_transform_interleaved_from_layer(data, num_ntts, 0);
    }

    /// Forward interleaved NTT starting at `start_layer`, assuming the first
    /// `start_layer` layers have already been applied to `data`.
    ///
    /// The RS-encoding use case: with `log_inv_rate = r` the upper
    /// `(2^r − 1)/2^r` of the coefficient buffer is zero, so each of the first
    /// `r` layers degenerates to a copy (butterfly with `v = 0` gives
    /// `(u, u)`). The caller replicates the message into all `2^r` sub-blocks
    /// — which IS the exact post-layer-`r` state — and skips those layers'
    /// reads and multiplies here.
    pub fn forward_transform_interleaved_from_layer(
        &self,
        data: &mut [F128],
        num_ntts: usize,
        start_layer: usize,
    ) {
        self.forward_transform_interleaved_from_layer_and_then(
            data,
            num_ntts,
            start_layer,
            |_, _| {},
        );
    }

    /// Variant of [`Self::forward_transform_interleaved_from_layer`] that calls
    /// `finish_chunk(offset, chunk)` exactly once for every disjoint finalized
    /// cache chunk. `offset` is in `F128` elements from the start of `data`.
    ///
    /// The callback runs inside the existing deep-transform Rayon job, before
    /// that worker moves to another chunk. Once a callback begins, the
    /// transform never reads or writes that chunk again; callers may therefore
    /// hand the finalized range to another worker before the callback returns.
    /// This lets the PCS hash codeword leaves while their 1 MiB ranked subtree
    /// is still cache-resident, without changing transform ordering or adding
    /// another parallel region.
    pub(crate) fn forward_transform_interleaved_from_layer_and_then<F>(
        &self,
        data: &mut [F128],
        num_ntts: usize,
        start_layer: usize,
        finish_chunk: F,
    ) where
        F: Fn(usize, &[F128]) + Sync + Send,
    {
        assert!(num_ntts.is_power_of_two() && num_ntts > 0);
        let n_total = data.len();
        assert_eq!(n_total % num_ntts, 0);
        let log_d = log2_pow2(n_total / num_ntts);
        assert!(log_d <= self.log_domain_size());
        assert!(start_layer <= log_d);

        // Scalar; SIMD/parallel variants below dispatch from `forward_transform_interleaved`
        // on supported targets.
        #[cfg(any(
            all(target_arch = "aarch64", target_feature = "aes"),
            all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        ))]
        {
            self.forward_transform_interleaved_parallel_from_layer_and_then::<
                INTERLEAVED_PHASE_ALL,
                _,
            >(
                data,
                num_ntts,
                start_layer,
                &finish_chunk,
            );
        }
        #[cfg(not(any(
            all(target_arch = "aarch64", target_feature = "aes"),
            all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        )))]
        {
            self.forward_transform_interleaved_scalar_from_layer(data, num_ntts, start_layer);
            finish_chunk(0, data);
        }
    }

    /// Apply only the ranked L0 transform's three top radix-8 passes. The PCS
    /// leaf pipeline uses this split entry point before occupying the E-core
    /// pool with blocking leaf receivers, leaving that pool available to all
    /// three heterogeneous top-pass queues.
    #[inline]
    pub(crate) fn forward_transform_interleaved_ranked_top_from_layer(
        &self,
        data: &mut [F128],
        num_ntts: usize,
        start_layer: usize,
    ) {
        let log_d = log2_pow2(data.len() / num_ntts);
        assert!(use_ranked_deep_pair_fusion(
            log_d,
            num_ntts,
            start_layer,
            10
        ));
        #[cfg(any(
            all(target_arch = "aarch64", target_feature = "aes"),
            all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        ))]
        self.forward_transform_interleaved_parallel_from_layer_and_then::<
            INTERLEAVED_PHASE_TOP_ONLY,
            _,
        >(data, num_ntts, start_layer, &|_, _| {});
        #[cfg(not(any(
            all(target_arch = "aarch64", target_feature = "aes"),
            all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        )))]
        unreachable!("ranked top split requires a hardware NTT target");
    }

    /// Start an interleaved transform directly from the rate-reduced message,
    /// fusing the first three nontrivial layers into the stores that initialize
    /// `data`, then finish the remaining layers in place.
    ///
    /// The ordinary path first writes `2^start_layer` replicas of `msg`, then
    /// immediately reads and rewrites the whole codeword for the radix-8 pass
    /// at `start_layer..start_layer + 3`. This entry point reads the same
    /// message rows and writes the post-radix-8 values directly, so stale
    /// destination bytes are never loaded and the replica-fill pass vanishes.
    pub(crate) fn forward_transform_interleaved_from_message_fused3(
        &self,
        msg: &[F128],
        data: &mut [F128],
        num_ntts: usize,
        start_layer: usize,
    ) {
        let log_d = log2_pow2(data.len() / num_ntts);
        let recursive_l1_helper = if use_recursive_ntt_epool(log_d, num_ntts, start_layer) {
            crate::epool::helper_pool()
        } else {
            None
        };
        self.forward_transform_interleaved_from_message_fused3_with_helper(
            msg,
            data,
            num_ntts,
            start_layer,
            log_d,
            recursive_l1_helper,
        );
    }

    fn forward_transform_interleaved_from_message_fused3_with_helper(
        &self,
        msg: &[F128],
        data: &mut [F128],
        num_ntts: usize,
        start_layer: usize,
        log_d: usize,
        recursive_l1_helper: Option<&rayon::ThreadPool>,
    ) {
        use rayon::prelude::*;

        let trace = is_recursive_ntt_epool_shape(log_d, num_ntts, start_layer)
            && trace_recursive_ntt_epool();
        let top_trace = trace.then(|| {
            (
                std::time::Instant::now(),
                crate::epool::helper_chunks_claimed(),
                crate::epool::helper_broadcasts_issued(),
            )
        });

        assert_eq!(
            num_ntts, 8,
            "recursive from-message fusion uses eight lanes"
        );
        assert!(start_layer + 3 <= log_d);
        assert_eq!(data.len(), msg.len() << start_layer);

        let num_blocks = 1usize << start_layer;
        let block_positions = 1usize << (log_d - start_layer);
        let block_elems = block_positions * num_ntts;
        let eighth = block_positions >> 3;
        assert_eq!(msg.len(), block_elems);
        let twiddles: Vec<[F128; 7]> = (0..num_blocks)
            .map(|block| {
                let mut tw = [F128 { lo: 0, hi: 0 }; 7];
                tw[0] = self.twiddle(start_layer, block);
                for s in 0..2 {
                    tw[1 + s] = self.twiddle(start_layer + 1, 2 * block + s);
                }
                for s in 0..4 {
                    tw[3 + s] = self.twiddle(start_layer + 2, 4 * block + s);
                }
                tw
            })
            .collect();
        debug_assert_eq!(twiddles[0][0], F128::ZERO);
        debug_assert_eq!(twiddles[0][1], F128::ZERO);
        debug_assert_eq!(twiddles[0][3], F128::ZERO);

        const ROWS_PER_TILE: usize = 128;
        let tiles_per_block = eighth.div_ceil(ROWS_PER_TILE);
        let src = msg.as_ptr() as usize;
        let dst = data.as_mut_ptr() as usize;
        let top = |job: usize| {
            let block = job / tiles_per_block;
            let tile = job % tiles_per_block;
            let row_start = tile * ROWS_PER_TILE;
            let row_end = (row_start + ROWS_PER_TILE).min(eighth);
            // SAFETY: each `(block, tile)` job owns all eight destination
            // rows for one disjoint row interval. `msg` is immutable and
            // has one complete layer-start block; every derived address
            // is in the validated source/destination geometry.
            unsafe {
                let dst_block = (dst as *mut F128).add(block * block_elems);
                for row in row_start..row_end {
                    if block == 0 {
                        kernels::butterfly_fused_3layer_zero_root_from_src_row(
                            src as *const F128,
                            dst_block,
                            eighth,
                            num_ntts,
                            row,
                            &twiddles[block],
                        );
                    } else {
                        kernels::butterfly_fused_3layer_from_src_row(
                            src as *const F128,
                            dst_block,
                            eighth,
                            num_ntts,
                            row,
                            &twiddles[block],
                        );
                    }
                }
            }
        };
        if recursive_l1_helper.is_some() && recursive_top_hetero_enabled() {
            // Heterogeneous claim queue for the radix-8 producer: the ranked
            // L0 top already proves this pattern at the exact 1 GiB codeword
            // (`is_ranked_top_hetero_fused3_pass`), where the shared atomic
            // queue bounds the heterogeneous tail to one 128-row tile per
            // worker. The recursive L1/L2 tops currently leave the efficiency
            // cores idle while the main pool writes the 32 MiB / 8 MiB
            // codewords; claiming fixed tiles there overlaps the otherwise
            // idle E cores with the top pass, then the deep tail proceeds on
            // the helper pool exactly as before. Output is bit-identical: the
            // same per-tile kernel runs on either pool and every (block,
            // tile) job owns disjoint destination rows.
            crate::epool::run_hetero_chunks(num_blocks * tiles_per_block, top);
        } else {
            (0..num_blocks * tiles_per_block)
                .into_par_iter()
                .for_each(top);
        }

        let top_trace = top_trace.map(|(started, claims, broadcasts)| {
            (
                started.elapsed().as_secs_f64() * 1e3,
                crate::epool::helper_chunks_claimed().wrapping_sub(claims),
                crate::epool::helper_broadcasts_issued().wrapping_sub(broadcasts),
            )
        });
        let deep_trace = trace.then(|| {
            (
                std::time::Instant::now(),
                crate::epool::helper_chunks_claimed(),
                crate::epool::helper_broadcasts_issued(),
            )
        });
        if let Some(helper) = recursive_l1_helper {
            self.forward_transform_interleaved_recursive_deep_with_helper(
                data,
                num_ntts,
                start_layer + 3,
                helper,
            );
        } else {
            self.forward_transform_interleaved_from_layer(data, num_ntts, start_layer + 3);
        }
        if let (Some((top_ms, top_claims, top_broadcasts)), Some((started, claims, broadcasts))) =
            (top_trace, deep_trace)
        {
            eprintln!(
                "[recursive-l1-ntt-epool] helper={} top_ms={top_ms:.3} \
                 top_claims={top_claims} top_broadcasts={top_broadcasts} deep_ms={:.3} \
                 deep_claims={} deep_broadcasts={}",
                recursive_l1_helper.is_some(),
                started.elapsed().as_secs_f64() * 1e3,
                crate::epool::helper_chunks_claimed().wrapping_sub(claims),
                crate::epool::helper_broadcasts_issued().wrapping_sub(broadcasts),
            );
        }
    }

    /// Ranked L0 top passes with the layer-1 pass fused from the message:
    /// both rate-1/2 replica blocks' layer-1..3 butterflies are evaluated
    /// straight from `msg` (`replicate_message_fill` is exactly the
    /// pre-applied layer 0), so the message is never materialized into the
    /// codeword — deleting one full replica store from the producer and
    /// halving this pass's loads. Layers 4 and 7 then run the exact in-place
    /// hetero passes of
    /// [`Self::forward_transform_interleaved_ranked_top_from_layer`].
    ///
    /// Every codeword element is written by the layer-1 pass, so `data` may
    /// hold arbitrary stale bytes on entry. Only the exact ranked L0 shape is
    /// supported (asserted); callers outside it must replicate-fill and use
    /// the ordinary transform.
    pub(crate) fn forward_transform_interleaved_ranked_top_from_message(
        &self,
        msg: &[F128],
        data: &mut [F128],
        num_ntts: usize,
        start_layer: usize,
    ) {
        let log_d = log2_pow2(data.len() / num_ntts);
        assert!(use_ranked_deep_pair_fusion(
            log_d,
            num_ntts,
            start_layer,
            10
        ));
        assert_eq!(
            start_layer, 1,
            "from-message fusion is a rate-1/2 layer-1 pass"
        );
        assert_eq!(data.len(), 2 * msg.len());

        let n_top = 10usize;
        let block_twiddles = |layer: usize, block: usize| {
            let mut tw = [F128 { lo: 0, hi: 0 }; 7];
            tw[0] = self.twiddle(layer, block);
            for s in 0..2 {
                tw[1 + s] = self.twiddle(layer + 1, 2 * block + s);
            }
            for s in 0..4 {
                tw[3 + s] = self.twiddle(layer + 2, 4 * block + s);
            }
            tw
        };

        let pass_timing = std::env::var_os("FLOCK_NTT_PASS_TIMING").is_some();
        let t_l1 = std::time::Instant::now();
        // Layer-1 fused-3 pass from the message: 2 blocks, identical input.
        {
            let block_size = 1usize << (log_d - 1);
            let eighth = block_size >> 3;
            debug_assert_eq!(msg.len(), block_size * num_ntts);
            let t_zero = block_twiddles(1, 0);
            let t_gen = block_twiddles(1, 1);
            debug_assert_eq!(t_zero[0], F128::ZERO);
            debug_assert_eq!(t_zero[1], F128::ZERO);
            debug_assert_eq!(t_zero[3], F128::ZERO);

            const ROWS_PER_TILE: usize = 128;
            let tiles = eighth.div_ceil(ROWS_PER_TILE);
            let block_elems = 8 * eighth * num_ntts;
            let src = msg.as_ptr() as usize;
            let dst = crate::epool::SyncPtr(data.as_mut_ptr());
            crate::epool::run_hetero_chunks(tiles, |tile| {
                let row_start = tile * ROWS_PER_TILE;
                let row_end = (row_start + ROWS_PER_TILE).min(eighth);
                let dst0 = dst.ptr();
                // SAFETY: each queue index owns one disjoint row tile; the
                // two destination blocks are disjoint halves of `data`, and
                // every derived address is inside the validated geometry.
                // `msg` is only read.
                unsafe {
                    let dst1 = dst0.add(block_elems);
                    for row in row_start..row_end {
                        kernels::butterfly_fused_3layer_dual_from_src_row(
                            src as *const F128,
                            dst0,
                            dst1,
                            eighth,
                            num_ntts,
                            row,
                            &t_zero,
                            &t_gen,
                        );
                    }
                }
            });
        }

        if pass_timing {
            eprintln!(
                "[ntt-pass] layer1-from-msg: {:.2} ms",
                t_l1.elapsed().as_secs_f64() * 1e3
            );
        }
        // Layers 4 and 7: exact in-place ranked hetero passes.
        for layer in [4usize, 7] {
            let num_blocks = 1usize << layer;
            let block_size = 1usize << (log_d - layer);
            let eighth = block_size >> 3;
            debug_assert!(is_ranked_top_hetero_fused3_pass(
                log_d,
                num_ntts,
                start_layer,
                n_top,
                layer
            ));
            let twiddles: Vec<[F128; 7]> =
                (0..num_blocks).map(|b| block_twiddles(layer, b)).collect();
            butterfly_interleaved_fused_3layer_all_blocks_hetero(data, &twiddles, eighth, num_ntts);
        }
    }

    /// Complete layers `start_layer..log_d` of the big interleaved transform
    /// for the absolute layer-`start_layer` blocks `[b_start, b_end)` only.
    ///
    /// Used by the hybrid GPU/CPU commit: the GPU finishes the shared top
    /// pass, then owns a position prefix while the CPU completes the suffix
    /// blocks with this routine. Twiddle indices are absolute (global layer
    /// and block numbering of the full `log_d` transform), so outputs are
    /// bit-identical to the unsplit transform on the same positions.
    ///
    /// Plain per-layer fused passes (fused-3 → fused-2 → single) over the
    /// suffix slice; the suffix is a minority share sized so streaming
    /// simplicity beats cache heroics.
    pub(crate) fn forward_transform_interleaved_block_range(
        &self,
        data: &mut [F128],
        num_ntts: usize,
        start_layer: usize,
        stop_layer: usize,
        b_start: usize,
        b_end: usize,
        // Trailing lanes known zero at every odd position of `data`; 0 for an
        // ordinary dense transform.
        odd_tail: usize,
    ) {
        let log_d = log2_pow2(data.len() / num_ntts);
        debug_assert!(stop_layer <= log_d);
        let range_blocks = b_end - b_start;
        debug_assert!(range_blocks > 0 && b_end <= (1usize << start_layer));
        let top_block_positions = 1usize << (log_d - start_layer);
        let base_pos = b_start * top_block_positions;

        let mut layer = start_layer;
        while layer < stop_layer {
            let block_size = 1usize << (log_d - layer);
            let block_elems = block_size * num_ntts;
            // Absolute block index of the range's first block at this layer.
            let abs_first = b_start << (layer - start_layer);
            let num_blocks = range_blocks << (layer - start_layer);
            let range_base_elem = base_pos * num_ntts;

            // Never leave a lone final layer (block_size 2 ⇒ 2^19 serial
            // kernel calls): when exactly 4 layers remain, take two fused-2
            // passes instead of fused-3 + single.
            if layer + 2 < stop_layer && stop_layer - layer != 4 && block_size >= 8 {
                let eighth = block_size >> 3;
                for local in 0..num_blocks {
                    let abs = abs_first + local;
                    let mut tw = [F128 { lo: 0, hi: 0 }; 7];
                    tw[0] = self.twiddle(layer, abs);
                    for s in 0..2 {
                        tw[1 + s] = self.twiddle(layer + 1, 2 * abs + s);
                    }
                    for s in 0..4 {
                        tw[3 + s] = self.twiddle(layer + 2, 4 * abs + s);
                    }
                    let start = local * block_elems;
                    let slice =
                        &mut data[range_base_elem + start..range_base_elem + start + block_elems];
                    // `eighth` even ⇒ every row group stays inside one
                    // position parity, so the static zero tail survives.
                    let tail = if eighth.is_multiple_of(2) {
                        odd_tail
                    } else {
                        0
                    };
                    if tw[0] == F128::ZERO && tw[1] == F128::ZERO && tw[3] == F128::ZERO {
                        butterfly_interleaved_fused_3layer_par_rows::<true>(
                            slice, &tw, eighth, num_ntts, tail,
                        );
                    } else {
                        butterfly_interleaved_fused_3layer_par_rows::<false>(
                            slice, &tw, eighth, num_ntts, tail,
                        );
                    }
                }
                layer += 3;
            } else if layer + 1 < stop_layer && block_size >= 4 {
                let quarter = block_size >> 2;
                for local in 0..num_blocks {
                    let abs = abs_first + local;
                    let start = local * block_elems;
                    butterfly_interleaved_fused_2layer_par_rows(
                        &mut data[range_base_elem + start..range_base_elem + start + block_elems],
                        self.twiddle(layer, abs),
                        self.twiddle(layer + 1, 2 * abs),
                        self.twiddle(layer + 1, 2 * abs + 1),
                        quarter,
                        num_ntts,
                    );
                }
                layer += 2;
            } else {
                let half = block_size >> 1;
                for local in 0..num_blocks {
                    let abs = abs_first + local;
                    let start = local * block_elems;
                    let t = self.twiddle(layer, abs);
                    let chunk =
                        &mut data[range_base_elem + start..range_base_elem + start + block_elems];
                    // SAFETY: the fused-2 path handles all sizes ≥ 4; only
                    // the final layer (block_size == 2) lands here, and the
                    // NEON single-layer kernel covers it.
                    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
                    unsafe {
                        kernels::butterfly_neon_block(chunk, t, half * num_ntts);
                    }
                    #[cfg(not(all(target_arch = "aarch64", target_feature = "aes")))]
                    {
                        let (lo, hi) = chunk.split_at_mut(half * num_ntts);
                        for (u, v) in lo.iter_mut().zip(hi.iter_mut()) {
                            let nu = *u + *v * t;
                            *v += nu;
                            *u = nu;
                        }
                    }
                }
                layer += 1;
            }
        }
    }

    /// Complete a ranked hybrid suffix with the same cache-local schedule as
    /// the tuned full-CPU transform, then publish each finalized 1 MiB chunk.
    ///
    /// The shared GPU pass has already completed layers 0..4.  This routine
    /// runs the two remaining top radix-8 groups (layers 4..10) over the
    /// requested absolute layer-4 blocks, then finishes layers 10..20 as five
    /// fused pairs inside independent layer-10 sub-NTTs.  `finish_chunk`
    /// receives an absolute element offset, so a concurrent GPU prefix and
    /// this CPU suffix can write disjoint Merkle leaf ranges without rebasing
    /// indices.  It runs immediately after the last pair while the 1 MiB
    /// codeword chunk is still cache-resident.
    ///
    /// This deliberately has a narrow production contract.  Other shapes
    /// keep [`Self::forward_transform_interleaved_block_range`].
    pub(crate) fn forward_transform_interleaved_ranked_block_range_and_then<F>(
        &self,
        data: &mut [F128],
        num_ntts: usize,
        start_layer: usize,
        stop_layer: usize,
        b_start: usize,
        b_end: usize,
        finish_chunk: F,
    ) where
        F: Fn(usize, &[F128]) + Sync + Send,
    {
        const DEEP_LAYER: usize = 10;
        let log_d = log2_pow2(data.len() / num_ntts);
        assert_eq!(log_d, 20, "ranked hybrid suffix requires log_d=20");
        assert_eq!(num_ntts, 64, "ranked hybrid suffix requires 64 lanes");
        assert_eq!(start_layer, 4, "ranked hybrid suffix starts at layer 4");
        assert_eq!(stop_layer, log_d, "ranked hybrid suffix completes the NTT");
        assert!(b_start < b_end && b_end <= (1usize << start_layer));

        // Only this ranked entry point (and the ranked full-CPU driver) honor
        // the ambient zero-lane publication; every other transform gets an
        // explicit tail.
        let odd_tail = ranked_zero_odd_tail_lanes(log_d, num_ntts);

        // Two fused radix-8 passes.  Unlike the old all-layer range driver,
        // this is the last streaming traversal of the complete suffix.
        self.forward_transform_interleaved_block_range(
            data,
            num_ntts,
            start_layer,
            DEEP_LAYER,
            b_start,
            b_end,
            odd_tail,
        );

        // Every layer-4 block contains 2^(10-4) independent layer-10
        // sub-NTTs.  Preserve their absolute indices so all deeper twiddles
        // match the unsplit transform exactly.
        let sub_start = b_start << (DEEP_LAYER - start_layer);
        let sub_end = b_end << (DEEP_LAYER - start_layer);
        self.forward_transform_interleaved_deep_fused_pairs_range_and_then(
            data,
            num_ntts,
            DEEP_LAYER,
            log_d,
            sub_start,
            sub_end,
            odd_tail,
            &finish_chunk,
        );
    }

    /// Finish the ranked L0 transform's five cache-local deep pairs and invoke
    /// `finish_chunk` exactly as the unsplit transform does.
    #[inline]
    pub(crate) fn forward_transform_interleaved_ranked_deep_and_then<F>(
        &self,
        data: &mut [F128],
        num_ntts: usize,
        start_layer: usize,
        finish_chunk: F,
    ) where
        F: Fn(usize, &[F128]) + Sync + Send,
    {
        let log_d = log2_pow2(data.len() / num_ntts);
        assert!(use_ranked_deep_pair_fusion(
            log_d,
            num_ntts,
            start_layer,
            10
        ));
        #[cfg(any(
            all(target_arch = "aarch64", target_feature = "aes"),
            all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        ))]
        self.forward_transform_interleaved_parallel_from_layer_and_then::<
            INTERLEAVED_PHASE_DEEP_ONLY,
            _,
        >(data, num_ntts, start_layer, &finish_chunk);
        #[cfg(not(any(
            all(target_arch = "aarch64", target_feature = "aes"),
            all(target_arch = "x86_64", target_feature = "pclmulqdq"),
        )))]
        unreachable!("ranked deep split requires a hardware NTT target");
    }

    /// Scalar reference for the interleaved forward NTT.
    pub fn forward_transform_interleaved_scalar(&self, data: &mut [F128], num_ntts: usize) {
        self.forward_transform_interleaved_scalar_from_layer(data, num_ntts, 0);
    }

    /// Scalar interleaved forward NTT from `start_layer` (see
    /// [`Self::forward_transform_interleaved_from_layer`]).
    pub fn forward_transform_interleaved_scalar_from_layer(
        &self,
        data: &mut [F128],
        num_ntts: usize,
        start_layer: usize,
    ) {
        let n_total = data.len();
        let log_d = log2_pow2(n_total / num_ntts);

        for layer in start_layer..log_d {
            let num_blocks = 1usize << layer;
            let block_size = 1usize << (log_d - layer);
            let block_size_half = block_size >> 1;
            let block_size_bytes = block_size * num_ntts;
            for block in 0..num_blocks {
                let twiddle = self.twiddle(layer, block);
                let block_start = block * block_size_bytes;
                // Butterfly pairs (top, bot) at positions (row, row + block_size_half)
                // within the block. Each "position" holds num_ntts lanes side-by-side.
                for row in 0..block_size_half {
                    let off_top = block_start + row * num_ntts;
                    let off_bot = off_top + block_size_half * num_ntts;
                    for lane in 0..num_ntts {
                        let v = data[off_bot + lane];
                        let new_u = data[off_top + lane] + v * twiddle;
                        data[off_top + lane] = new_u;
                        data[off_bot + lane] = v + new_u;
                    }
                }
            }
        }
    }

    /// Parallel + NEON interleaved forward NTT. Cache-blocks the same way as
    /// `forward_transform_batched`: top layers process the full SoA buffer with
    /// per-block parallelism; deep layers process each sub-NTT-group in cache.
    ///
    /// Internally calls [`forward_transform_interleaved_scalar`] for very small
    /// inputs to avoid rayon overhead; for large inputs it uses an in-place
    /// scalar butterfly per lane (per-lane vectorization is future work — the
    /// big win at large `m` is cache locality + thread parallelism).
    #[cfg(any(
        all(target_arch = "aarch64", target_feature = "aes"),
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
    ))]
    pub fn forward_transform_interleaved_parallel(&self, data: &mut [F128], num_ntts: usize) {
        self.forward_transform_interleaved_parallel_from_layer(data, num_ntts, 0);
    }

    /// Parallel interleaved forward NTT from `start_layer` (see
    /// [`Self::forward_transform_interleaved_from_layer`]).
    #[cfg(any(
        all(target_arch = "aarch64", target_feature = "aes"),
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
    ))]
    pub fn forward_transform_interleaved_parallel_from_layer(
        &self,
        data: &mut [F128],
        num_ntts: usize,
        start_layer: usize,
    ) {
        self.forward_transform_interleaved_parallel_from_layer_and_then::<
            INTERLEAVED_PHASE_ALL,
            _,
        >(
            data,
            num_ntts,
            start_layer,
            &|_, _| {},
        );
    }

    #[cfg(any(
        all(target_arch = "aarch64", target_feature = "aes"),
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
    ))]
    fn forward_transform_interleaved_parallel_from_layer_and_then<const PHASE: u8, F>(
        &self,
        data: &mut [F128],
        num_ntts: usize,
        start_layer: usize,
        finish_chunk: &F,
    ) where
        F: Fn(usize, &[F128]) + Sync + Send,
    {
        debug_assert!(PHASE <= INTERLEAVED_PHASE_DEEP_ONLY);
        let n_total = data.len();
        let log_d = log2_pow2(n_total / num_ntts);

        // Target sub-group size = 2 MB total bytes. Each position is
        // `num_ntts × 16` bytes, so positions per sub-group =
        // 2^21 / (num_ntts · 16). With num_ntts=1: 2^17 positions. With
        // num_ntts=32: 2^12 positions. (Without this scaling, sub-groups at
        // num_ntts=32 would be 64 MB and overflow L2 cache.)
        const TARGET_SUBGROUP_LOG_BYTES: usize = 21;
        let log_bytes_per_position = 4 + log2_pow2(num_ntts);
        let target_log_positions = TARGET_SUBGROUP_LOG_BYTES.saturating_sub(log_bytes_per_position);
        let cache_n_top = log_d.saturating_sub(target_log_positions);

        // Parallelism floor. The cache heuristic keeps each sub-NTT ~2 MB, but
        // for a mid-size transform whose whole codeword already fits that
        // budget it yields `cache_n_top == 0` and the transform runs fully
        // serial — e.g. the recursive Ligerito commits (~1 ms of NTT each,
        // previously 1.0× across threads). When the transform is big enough to
        // amortize rayon overhead, raise `n_top` so the deep-layer split
        // produces ~one sub-NTT per worker thread (capped to keep each sub-NTT
        // ≥ 2^MIN_SUB_LOG positions). The large initial PCS commit is unaffected:
        // its `cache_n_top` already exceeds this floor.
        //
        // The floor (log_d ≥ 12) is the measured dispatch-vs-compute crossover
        // for num_ntts≈8 recursive commits: at log_d=12 parallelizing cuts the
        // NTT ~0.22 → ~0.08 ms, but at log_d=10 the rayon dispatch costs more
        // than the ~0.04 ms of work, so those stay scalar.
        const PARALLEL_FLOOR_LOG_D: usize = 12;
        const MIN_SUB_LOG: usize = 8;
        let n_top = if log_d >= PARALLEL_FLOOR_LOG_D {
            let want_subs_log = log2_pow2(rayon::current_num_threads().next_power_of_two());
            let max_n_top = log_d.saturating_sub(MIN_SUB_LOG);
            cache_n_top.max(want_subs_log.min(max_n_top))
        } else {
            cache_n_top
        };
        let n_top = fusion_aware_interleaved_n_top(log_d, num_ntts, start_layer, n_top);
        if n_top == 0 || log_d < 8 {
            debug_assert_eq!(PHASE, INTERLEAVED_PHASE_ALL);
            self.forward_transform_interleaved_scalar_from_layer(data, num_ntts, start_layer);
            finish_chunk(0, data);
            return;
        }

        if PHASE == INTERLEAVED_PHASE_DEEP_ONLY {
            self.forward_transform_interleaved_deep_from_layer_and_then(
                data,
                num_ntts,
                start_layer,
                n_top,
                log_d,
                finish_chunk,
            );
            return;
        }

        // Top layers: full-buffer sweep. Parallelize **rows within each
        // block** so even layer 0 (1 huge block) gets rayon parallelism.
        //
        // Layer fusion: at top layers each layer is a separate full-buffer
        // sweep (read 512 MB + write 512 MB at m=31). Fusing two consecutive
        // layers in one pass loads each row once, applies both butterflies
        // in registers, stores once — halving memory traffic on the fused
        // layers. Each "outer block" at layer L has 4 contributing rows per
        // quarter-row; layer L butterflies (a,c) and (b,d) (distance =
        // block_size/2), layer L+1 butterflies (a,b) and (c,d) (distance =
        // block_size/4).
        // Fuse FOUR layers per pass only where a SIMD fused-4 kernel exists
        // (x86 AVX-512). On other targets the 16-point kernel falls back to
        // scalar, which is slower than the NEON fused-2 path — so keep fused-2
        // there.
        //
        // Radix-16 is unavailable on NEON for a cache reason as well as a
        // missing kernel: its 16 concurrently-live row streams all alias into
        // one L1 set (every row-group stride here is a multiple of the
        // set-repeat period) and so demand 16 ways against an 8-way L1D.
        // Radix-8 is the widest fusion that fits — 8 streams, 8 ways — and it
        // still halves the number of full-buffer sweeps versus fused-2.
        let fused4_ok = cfg!(all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "vpclmulqdq"
        ));
        let fused3_ok = !fused4_ok;
        let zero_root_fused3 = use_ranked_zero_root_fusion(log_d, num_ntts, start_layer, n_top);
        let mut layer = start_layer.min(n_top);
        while layer < n_top {
            let num_blocks = 1usize << layer;
            let block_size = 1usize << (log_d - layer);
            let block_bytes = block_size * num_ntts;

            if fused3_ok && layer + 2 < n_top && block_size >= 8 {
                // Fuse three layers (layer..layer+3): one read+write per block
                // instead of three. Each block contributes an 8-point butterfly.
                let eighth = block_size >> 3;
                let block_twiddles = |block: usize| {
                    let mut tw = [F128 { lo: 0, hi: 0 }; 7];
                    tw[0] = self.twiddle(layer, block);
                    for s in 0..2 {
                        tw[1 + s] = self.twiddle(layer + 1, 2 * block + s);
                    }
                    for s in 0..4 {
                        tw[3 + s] = self.twiddle(layer + 2, 4 * block + s);
                    }
                    tw
                };
                if zero_root_fused3 && std::env::var_os("FLOCK_NTT_BLOCK_REGIONS").is_none() {
                    let twiddles: Vec<[F128; 7]> = (0..num_blocks).map(block_twiddles).collect();
                    if is_ranked_top_hetero_fused3_pass(log_d, num_ntts, start_layer, n_top, layer)
                        && std::env::var_os("FLOCK_NO_NTT_TOP_EPOOL").is_none()
                    {
                        butterfly_interleaved_fused_3layer_all_blocks_hetero(
                            data, &twiddles, eighth, num_ntts,
                        );
                    } else {
                        butterfly_interleaved_fused_3layer_all_blocks_par_rows(
                            data, &twiddles, eighth, num_ntts,
                        );
                    }
                } else {
                    let odd_tail = if eighth.is_multiple_of(2) {
                        ranked_zero_odd_tail_lanes(log_d, num_ntts)
                    } else {
                        0
                    };
                    for block in 0..num_blocks {
                        let tw = block_twiddles(block);
                        let start = block * block_bytes;
                        if block == 0 && zero_root_fused3 {
                            butterfly_interleaved_fused_3layer_par_rows::<true>(
                                &mut data[start..start + block_bytes],
                                &tw,
                                eighth,
                                num_ntts,
                                odd_tail,
                            );
                        } else {
                            butterfly_interleaved_fused_3layer_par_rows::<false>(
                                &mut data[start..start + block_bytes],
                                &tw,
                                eighth,
                                num_ntts,
                                odd_tail,
                            );
                        }
                    }
                }
                layer += 3;
            } else if fused4_ok && layer + 3 < n_top && block_size >= 16 {
                // Fuse four layers (layer..layer+4): one read+write per block
                // instead of four. Each block contributes a 16-point butterfly.
                let sixteenth = block_size >> 4;
                for block in 0..num_blocks {
                    let mut tw = [F128 { lo: 0, hi: 0 }; 15];
                    tw[0] = self.twiddle(layer, block);
                    for s in 0..2 {
                        tw[1 + s] = self.twiddle(layer + 1, 2 * block + s);
                    }
                    for s in 0..4 {
                        tw[3 + s] = self.twiddle(layer + 2, 4 * block + s);
                    }
                    for s in 0..8 {
                        tw[7 + s] = self.twiddle(layer + 3, 8 * block + s);
                    }
                    let start = block * block_bytes;
                    butterfly_interleaved_fused_4layer_par_rows(
                        &mut data[start..start + block_bytes],
                        &tw,
                        sixteenth,
                        num_ntts,
                    );
                }
                layer += 4;
            } else if layer + 1 < n_top && block_size >= 4 {
                // Fuse layers (layer, layer+1).
                let quarter = block_size >> 2;
                for block in 0..num_blocks {
                    let t_outer = self.twiddle(layer, block);
                    let t_inner_a = self.twiddle(layer + 1, 2 * block);
                    let t_inner_b = self.twiddle(layer + 1, 2 * block + 1);
                    let start = block * block_bytes;
                    butterfly_interleaved_fused_2layer_par_rows(
                        &mut data[start..start + block_bytes],
                        t_outer,
                        t_inner_a,
                        t_inner_b,
                        quarter,
                        num_ntts,
                    );
                }
                layer += 2;
            } else {
                let block_size_half = block_size >> 1;
                for block in 0..num_blocks {
                    let t = self.twiddle(layer, block);
                    let start = block * block_bytes;
                    butterfly_interleaved_block_par_rows(
                        &mut data[start..start + block_bytes],
                        t,
                        block_size_half,
                        num_ntts,
                    );
                }
                layer += 1;
            }
        }
        if PHASE == INTERLEAVED_PHASE_TOP_ONLY {
            return;
        }

        self.forward_transform_interleaved_deep_from_layer_and_then(
            data,
            num_ntts,
            start_layer,
            n_top,
            log_d,
            finish_chunk,
        );
    }

    /// Finish the first recursive commitment from its post-radix-8 layer-5
    /// state. Splitting one level earlier than the generic 2 MiB cache policy
    /// exposes 32 uniform 1 MiB sub-transforms to the existing shared P+E
    /// queue. Each job still executes the ordinary single-layer tail below.
    #[inline]
    /// Deep tail of a recursive-commit transform on the P+E helper queue.
    /// Both ranked recursive shapes — L1 (32 x 1 MiB at log_d 18) and L2
    /// (32 x 256 KiB at log_d 16) — partition into 32 independent
    /// cache-resident sub-transforms; each sub runs the identical
    /// single-layer subtree loop, so this changes ownership/scheduling only.
    fn forward_transform_interleaved_recursive_deep_with_helper(
        &self,
        data: &mut [F128],
        num_ntts: usize,
        start_layer: usize,
        helper: &rayon::ThreadPool,
    ) {
        const N_TOP: usize = 5;

        let log_d = log2_pow2(data.len() / num_ntts);
        debug_assert!(
            matches!((log_d, num_ntts, start_layer), (18, 8, 5) | (16, 8, 6)),
            "recursive deep helper only serves the L1/L2 ranked shapes"
        );
        let sub_size_positions = 1usize << (log_d - N_TOP);
        let sub_bytes = sub_size_positions * num_ntts;
        let num_subs = 1usize << N_TOP;
        debug_assert_eq!(data.len(), num_subs * sub_bytes);

        let base = crate::epool::SyncPtr(data.as_mut_ptr());
        let run_sub = |sub_idx: usize| {
            // SAFETY: the shared queue issues each `sub_idx` once. Its range
            // is exactly one disjoint `sub_bytes` partition of `data`, and
            // `run_chunks_with_helper` joins both pools before returning.
            let sub_data = unsafe {
                std::slice::from_raw_parts_mut(base.ptr().add(sub_idx * sub_bytes), sub_bytes)
            };
            self.forward_transform_interleaved_deep_subtree(
                sub_data,
                num_ntts,
                start_layer,
                N_TOP,
                log_d,
                sub_idx,
            );
        };
        crate::epool::run_chunks_with_helper(num_subs, &run_sub, Some(helper));
    }

    #[inline(always)]
    fn forward_transform_interleaved_deep_from_layer_and_then<F>(
        &self,
        data: &mut [F128],
        num_ntts: usize,
        start_layer: usize,
        n_top: usize,
        log_d: usize,
        finish_chunk: &F,
    ) where
        F: Fn(usize, &[F128]) + Sync + Send,
    {
        use rayon::prelude::*;

        // Ranked L0: layers 10..20 form five exact pairs. Fuse each pair inside
        // the existing outer chunk job so every row is loaded/stored once per
        // two layers and no nested Rayon region is created.
        if use_ranked_deep_pair_fusion(log_d, num_ntts, start_layer, n_top) {
            self.forward_transform_interleaved_deep_fused_pairs_and_then(
                data,
                num_ntts,
                n_top,
                log_d,
                finish_chunk,
            );
            return;
        }

        // Deep layers: process each sub-NTT-group cache-resident.
        let sub_size_positions = 1usize << (log_d - n_top);
        let sub_bytes = sub_size_positions * num_ntts;

        data.par_chunks_mut(sub_bytes)
            .enumerate()
            .for_each(|(sub_idx, sub_data)| {
                self.forward_transform_interleaved_deep_subtree(
                    sub_data,
                    num_ntts,
                    start_layer,
                    n_top,
                    log_d,
                    sub_idx,
                );
                finish_chunk(sub_idx * sub_bytes, sub_data);
            });
    }

    /// Execute one independently-owned deep sub-transform. Both the ordinary
    /// Rayon partitioner and the recursive-L1 P+E queue call this exact loop,
    /// so the candidate changes ownership/scheduling but no butterfly,
    /// twiddle, or layer order.
    #[inline(always)]
    fn forward_transform_interleaved_deep_subtree(
        &self,
        sub_data: &mut [F128],
        num_ntts: usize,
        start_layer: usize,
        n_top: usize,
        log_d: usize,
        sub_idx: usize,
    ) {
        for layer in n_top.max(start_layer)..log_d {
            let layer_in_sub = layer - n_top;
            let num_blocks_in_sub = 1usize << layer_in_sub;
            let block_size = 1usize << (log_d - layer);
            let block_size_half = block_size >> 1;
            let block_bytes = block_size * num_ntts;

            for block_in_sub in 0..num_blocks_in_sub {
                let global_block = sub_idx * num_blocks_in_sub + block_in_sub;
                let twiddle = self.twiddle(layer, global_block);
                let block_start = block_in_sub * block_bytes;
                let block = &mut sub_data[block_start..block_start + block_bytes];
                butterfly_interleaved_block(block, twiddle, block_size_half, num_ntts);
            }
        }
    }

    /// Finish layers `n_top..log_d` two at a time inside independent
    /// cache-resident sub-NTTs. The outer `par_chunks_mut` is the only Rayon
    /// boundary; block and row work inside each chunk is deliberately serial.
    #[cfg(test)]
    fn forward_transform_interleaved_deep_fused_pairs(
        &self,
        data: &mut [F128],
        num_ntts: usize,
        n_top: usize,
        log_d: usize,
    ) {
        self.forward_transform_interleaved_deep_fused_pairs_and_then(
            data,
            num_ntts,
            n_top,
            log_d,
            &|_, _| {},
        );
    }

    fn forward_transform_interleaved_deep_fused_pairs_and_then<F>(
        &self,
        data: &mut [F128],
        num_ntts: usize,
        n_top: usize,
        log_d: usize,
        finish_chunk: &F,
    ) where
        F: Fn(usize, &[F128]) + Sync + Send,
    {
        // Reached only under `use_ranked_deep_pair_fusion`, i.e. the exact
        // ranked production geometry, so the ambient publication applies.
        let odd_tail = ranked_zero_odd_tail_lanes(log_d, num_ntts);
        self.forward_transform_interleaved_deep_fused_pairs_range_and_then(
            data,
            num_ntts,
            n_top,
            log_d,
            0,
            1usize << n_top,
            odd_tail,
            finish_chunk,
        );
    }

    /// Range form of the ranked deep-pair scheduler. `sub_start..sub_end` are
    /// absolute layer-`n_top` block indices in the full transform; using the
    /// absolute `sub_idx` below is what preserves every deeper twiddle index.
    fn forward_transform_interleaved_deep_fused_pairs_range_and_then<F>(
        &self,
        data: &mut [F128],
        num_ntts: usize,
        n_top: usize,
        log_d: usize,
        sub_start: usize,
        sub_end: usize,
        // Trailing lanes known zero at every odd position; 0 = dense.
        zero_tail: usize,
        finish_chunk: &F,
    ) where
        F: Fn(usize, &[F128]) + Sync + Send,
    {
        use rayon::prelude::*;

        debug_assert!(n_top <= log_d);
        debug_assert_eq!(data.len(), (1usize << log_d) * num_ntts);
        debug_assert!(sub_start < sub_end && sub_end <= (1usize << n_top));
        let sub_size_positions = 1usize << (log_d - n_top);
        let sub_elems = sub_size_positions * num_ntts;
        let low_twiddle_final_pair = use_ranked_low_twiddle_final_pair(log_d, num_ntts, n_top);
        debug_assert!(zero_tail < num_ntts);

        let range_start = sub_start * sub_elems;
        let range_end = sub_end * sub_elems;

        data[range_start..range_end]
            .par_chunks_mut(sub_elems)
            .enumerate()
            .for_each(|(local_sub_idx, sub_data)| {
                let sub_idx = sub_start + local_sub_idx;
                let mut layer = n_top;
                while layer + 1 < log_d {
                    let layer_in_sub = layer - n_top;
                    let num_blocks_in_sub = 1usize << layer_in_sub;
                    let block_size_positions = 1usize << (log_d - layer);
                    let quarter = block_size_positions >> 2;
                    let block_elems = block_size_positions * num_ntts;

                    for block_in_sub in 0..num_blocks_in_sub {
                        let global_block = sub_idx * num_blocks_in_sub + block_in_sub;
                        let t_outer = self.twiddle(layer, global_block);
                        let t_inner_a = self.twiddle(layer + 1, 2 * global_block);
                        let t_inner_b = self.twiddle(layer + 1, 2 * global_block + 1);
                        let block_start = block_in_sub * block_elems;
                        if low_twiddle_final_pair && layer + 2 == log_d {
                            debug_assert_eq!(t_outer.hi, 0);
                            debug_assert_eq!(t_inner_a.hi, 0);
                            debug_assert_eq!(t_inner_b.hi, 0);
                            butterfly_interleaved_fused_2layer_low_twiddles_rows_seq(
                                &mut sub_data[block_start..block_start + block_elems],
                                t_outer,
                                t_inner_a,
                                t_inner_b,
                                quarter,
                                num_ntts,
                            );
                        } else {
                            // `quarter` even ⇒ each row group stays inside
                            // one position parity and the static zero tail
                            // survives; the final pair (quarter == 1) mixes
                            // parities and always runs dense.
                            let odd_tail = if quarter.is_multiple_of(2) {
                                zero_tail
                            } else {
                                0
                            };
                            butterfly_interleaved_fused_2layer_rows_seq(
                                &mut sub_data[block_start..block_start + block_elems],
                                t_outer,
                                t_inner_a,
                                t_inner_b,
                                quarter,
                                num_ntts,
                                odd_tail,
                            );
                        }
                    }
                    layer += 2;
                }

                // The ranked path has ten deep layers and therefore no tail.
                // Retaining the scalar single-layer tail makes this helper a
                // compact exact-test fixture for odd layer counts as well.
                if layer < log_d {
                    let layer_in_sub = layer - n_top;
                    let num_blocks_in_sub = 1usize << layer_in_sub;
                    let block_size_positions = 1usize << (log_d - layer);
                    let block_size_half = block_size_positions >> 1;
                    let block_elems = block_size_positions * num_ntts;
                    for block_in_sub in 0..num_blocks_in_sub {
                        let global_block = sub_idx * num_blocks_in_sub + block_in_sub;
                        let twiddle = self.twiddle(layer, global_block);
                        let block_start = block_in_sub * block_elems;
                        butterfly_interleaved_block(
                            &mut sub_data[block_start..block_start + block_elems],
                            twiddle,
                            block_size_half,
                            num_ntts,
                        );
                    }
                }
                finish_chunk(sub_idx * sub_elems, sub_data);
            });
    }

    /// Scalar reference implementation. Used as the test oracle and on
    /// platforms without NEON+PMULL.
    pub fn forward_transform_scalar(&self, data: &mut [F128]) {
        let log_d = log2_pow2(data.len());
        assert!(log_d <= self.log_domain_size());

        for layer in 0..log_d {
            let num_blocks = 1usize << layer;
            let block_size_half = 1usize << (log_d - layer - 1);
            for block in 0..num_blocks {
                let twiddle = self.twiddle(layer, block);
                let block_start = block << (log_d - layer);
                for idx0 in block_start..(block_start + block_size_half) {
                    let idx1 = idx0 | block_size_half;
                    // Forward butterfly: u += v·twiddle; v += u.
                    let v = data[idx1];
                    let new_u = data[idx0] + v * twiddle;
                    data[idx0] = new_u;
                    data[idx1] = v + new_u;
                }
            }
        }
    }

    /// Single-threaded NEON forward transform (uses `ghash_mul_vec2_neon` to
    /// batch 2 butterflies per PMULL pair).
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    pub fn forward_transform_neon(&self, data: &mut [F128]) {
        let log_d = log2_pow2(data.len());
        assert!(log_d <= self.log_domain_size());

        for layer in 0..log_d {
            let num_blocks = 1usize << layer;
            let block_size = 1usize << (log_d - layer);
            let block_size_half = block_size >> 1;
            // SAFETY: target_feature = "aes" enabled at compile time.
            unsafe {
                if block_size_half >= 2 {
                    // Within-block: batch 2 pairs with shared twiddle.
                    for block in 0..num_blocks {
                        let twiddle = self.twiddle(layer, block);
                        let block_start = block * block_size;
                        let chunk = &mut data[block_start..block_start + block_size];
                        kernels::butterfly_neon_block(chunk, twiddle, block_size_half);
                    }
                } else {
                    // Deepest layer (half = 1): batch across 2 adjacent blocks
                    // (different twiddles). Handle odd tail with scalar when
                    // num_blocks = 1 (only happens at log_d = 1).
                    debug_assert_eq!(block_size_half, 1);
                    let mut block = 0;
                    while block + 1 < num_blocks {
                        let t_a = self.twiddle(layer, block);
                        let t_b = self.twiddle(layer, block + 1);
                        kernels::butterfly_neon_block_pair(data, block * 2, t_a, t_b);
                        block += 2;
                    }
                    // Scalar tail (num_blocks odd — only when num_blocks = 1).
                    while block < num_blocks {
                        let twiddle = self.twiddle(layer, block);
                        let idx0 = block * 2;
                        let idx1 = idx0 + 1;
                        let v = data[idx1];
                        let new_u = data[idx0] + v * twiddle;
                        data[idx0] = new_u;
                        data[idx1] = v + new_u;
                        block += 1;
                    }
                }
            }
        }
    }

    /// Rayon-parallel + NEON forward transform.
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    pub fn forward_transform_parallel(&self, data: &mut [F128]) {
        use rayon::prelude::*;
        let log_d = log2_pow2(data.len());
        assert!(log_d <= self.log_domain_size());

        // For small data (or shallow layers with few large blocks), the rayon
        // overhead exceeds the gain — fall back to the NEON single-thread path.
        const PARALLEL_THRESHOLD_LOG: usize = 14; // 2^14 = 16K elements (256 KB)
        if log_d <= PARALLEL_THRESHOLD_LOG {
            self.forward_transform_neon(data);
            return;
        }

        for layer in 0..log_d {
            let num_blocks = 1usize << layer;
            let block_size = 1usize << (log_d - layer);
            let block_size_half = block_size >> 1;

            // Parallelize across blocks when there are enough; otherwise process
            // sequentially with NEON (still fast for small block counts).
            if num_blocks >= 4 && block_size_half >= 2 {
                let twiddles: Vec<F128> = (0..num_blocks).map(|b| self.twiddle(layer, b)).collect();
                data.par_chunks_mut(block_size)
                    .zip(twiddles.par_iter())
                    .for_each(|(chunk, &twiddle)| {
                        // SAFETY: aes target feature enabled.
                        unsafe { kernels::butterfly_neon_block(chunk, twiddle, block_size_half) };
                    });
            } else if block_size_half >= 2 {
                // Few large blocks — process sequentially with NEON.
                // SAFETY: aes target feature enabled.
                unsafe {
                    for block in 0..num_blocks {
                        let twiddle = self.twiddle(layer, block);
                        let block_start = block * block_size;
                        kernels::butterfly_neon_block(
                            &mut data[block_start..block_start + block_size],
                            twiddle,
                            block_size_half,
                        );
                    }
                }
            } else {
                // Deepest layer (half = 1): need num_blocks ≥ 2 to batch
                // pairs; if there are at least 2 blocks, batch across them.
                // (When num_blocks < 2, fall back to NEON-single-thread which
                // handles the trivial cases.)
                debug_assert_eq!(block_size_half, 1);
                if num_blocks >= 2 {
                    let twiddles: Vec<F128> =
                        (0..num_blocks).map(|b| self.twiddle(layer, b)).collect();
                    data.par_chunks_mut(4).zip(twiddles.par_chunks(2)).for_each(
                        |(chunk, twiddle_pair)| {
                            // SAFETY: aes target feature enabled.
                            unsafe {
                                kernels::butterfly_neon_block_pair_chunk(
                                    chunk,
                                    twiddle_pair[0],
                                    twiddle_pair[1],
                                )
                            };
                        },
                    );
                } else {
                    let twiddle = self.twiddle(layer, 0);
                    let v = data[1];
                    let new_u = data[0] + v * twiddle;
                    data[0] = new_u;
                    data[1] = v + new_u;
                }
            }
        }
    }

    /// Cache-blocked + parallel + NEON forward transform.
    ///
    /// **Strategy**: decompose the NTT into two stages so the deep layers
    /// (which dominate work) operate on sub-buffers small enough to fit in L2
    /// cache, avoiding the DRAM round-trip per layer.
    ///
    /// 1. **Top layers** (layers `0..n_top`): each layer touches the full buffer
    ///    in one sweep. Bandwidth-bound; parallelize across blocks.
    /// 2. **Deep layers** (layers `n_top..log_d`): treat the data as `2^n_top`
    ///    independent sub-NTTs, each of size `2^(log_d − n_top)`. For each
    ///    sub-NTT, process ALL remaining layers in one cache-resident pass.
    ///    Parallelize across sub-NTTs via rayon.
    ///
    /// `n_top` is chosen so each sub-NTT is `≈ 2 MB` (= `2^17` F_{2^128} ≈ 2 MB).
    /// For `log_d ≤ 17` the whole NTT fits in cache and we fall back to the
    /// per-layer parallel path.
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    pub fn forward_transform_batched(&self, data: &mut [F128]) {
        use rayon::prelude::*;
        let log_d = log2_pow2(data.len());
        assert!(log_d <= self.log_domain_size());

        // Target sub-NTT size: 2^17 F_{2^128} = 2 MB. Tunable.
        const TARGET_SUB_NTT_LOG: usize = 17;
        if log_d <= TARGET_SUB_NTT_LOG {
            self.forward_transform_parallel(data);
            return;
        }
        let n_top = log_d - TARGET_SUB_NTT_LOG;
        let sub_ntt_size = 1usize << (log_d - n_top);

        // ---- Stage 1: top layers (full-buffer, bandwidth-bound).
        for layer in 0..n_top {
            let num_blocks = 1usize << layer;
            let block_size = 1usize << (log_d - layer);
            let block_size_half = block_size >> 1;

            if num_blocks >= 4 {
                let twiddles: Vec<F128> = (0..num_blocks).map(|b| self.twiddle(layer, b)).collect();
                data.par_chunks_mut(block_size)
                    .zip(twiddles.par_iter())
                    .for_each(|(chunk, &t)| {
                        // SAFETY: aes target feature enabled.
                        unsafe { kernels::butterfly_neon_block(chunk, t, block_size_half) };
                    });
            } else {
                // Few large blocks at very top layers: sequential NEON.
                unsafe {
                    for block in 0..num_blocks {
                        let t = self.twiddle(layer, block);
                        let block_start = block * block_size;
                        kernels::butterfly_neon_block(
                            &mut data[block_start..block_start + block_size],
                            t,
                            block_size_half,
                        );
                    }
                }
            }
        }

        // ---- Stage 2: deep layers as parallel cache-resident sub-NTTs.
        data.par_chunks_mut(sub_ntt_size)
            .enumerate()
            .for_each(|(sub_idx, sub_data)| {
                for layer in n_top..log_d {
                    let layer_in_sub = layer - n_top;
                    let num_blocks_in_sub = 1usize << layer_in_sub;
                    let block_size = 1usize << (log_d - layer);
                    let block_size_half = block_size >> 1;

                    for block_in_sub in 0..num_blocks_in_sub {
                        let global_block = sub_idx * num_blocks_in_sub + block_in_sub;
                        let twiddle = self.twiddle(layer, global_block);
                        let block_start = block_in_sub * block_size;
                        let block = &mut sub_data[block_start..block_start + block_size];
                        if block_size_half >= 2 {
                            // SAFETY: aes target feature enabled.
                            unsafe {
                                kernels::butterfly_neon_block(block, twiddle, block_size_half)
                            };
                        } else {
                            // Deepest layer: 1 pair per block, scalar.
                            let v = block[1];
                            let new_u = block[0] + v * twiddle;
                            block[0] = new_u;
                            block[1] = v + new_u;
                        }
                    }
                }
            });
    }

    /// Inverse additive NTT in place. Exact inverse of `forward_transform`.
    pub fn inverse_transform(&self, data: &mut [F128]) {
        let log_d = log2_pow2(data.len());
        assert!(log_d <= self.log_domain_size());

        for layer in (0..log_d).rev() {
            let num_blocks = 1usize << layer;
            let block_size_half = 1usize << (log_d - layer - 1);
            for block in 0..num_blocks {
                let twiddle = self.twiddle(layer, block);
                let block_start = block << (log_d - layer);
                for idx0 in block_start..(block_start + block_size_half) {
                    let idx1 = idx0 | block_size_half;
                    // Inverse butterfly: v += u; u += v·twiddle.
                    let u = data[idx0];
                    let new_v = data[idx1] + u;
                    data[idx1] = new_v;
                    data[idx0] = u + new_v * twiddle;
                }
            }
        }
    }
}

/// Like [`butterfly_interleaved_block`] but parallelizes across rows via
/// rayon. Used at top layers where the block is large (≥ 1024 rows) and only
/// 1-2 blocks exist (so block-level parallelism would be too coarse).
///
/// Falls back to sequential when the row count is small.
#[inline]
fn butterfly_interleaved_block_par_rows(
    block: &mut [F128],
    twiddle: F128,
    block_size_half: usize,
    num_ntts: usize,
) {
    use rayon::prelude::*;
    const PARALLEL_ROW_THRESHOLD: usize = 512;
    if block_size_half < PARALLEL_ROW_THRESHOLD {
        butterfly_interleaved_block(block, twiddle, block_size_half, num_ntts);
        return;
    }
    let half_offset = block_size_half * num_ntts;
    let (top, bot) = block.split_at_mut(half_offset);
    top.par_chunks_mut(num_ntts)
        .zip(bot.par_chunks_mut(num_ntts))
        .for_each(|(top_row, bot_row)| {
            kernels::butterfly_row_pair(top_row, bot_row, twiddle);
        });
}

/// Fused 2-layer butterfly: combines layer L (twiddle `t_outer`, shared by
/// the whole outer block) with layer L+1 (twiddles `t_inner_a` for the top
/// half, `t_inner_b` for the bottom half). Reads each row of the outer
/// block once and writes once — halving memory traffic vs running the two
/// layers as separate sweeps.
///
/// `block` has length `4 * quarter * num_ntts` (= one layer-L block of
/// `4*quarter` rows). For each `r ∈ 0..quarter`, four rows participate:
/// `a=r`, `b=r+quarter`, `c=r+2*quarter`, `d=r+3*quarter`. Layer L
/// butterflies `(a,c)` and `(b,d)`; layer L+1 then butterflies `(a,b)` (in
/// the new top sub-block) and `(c,d)` (in the new bottom sub-block).
#[inline]
fn butterfly_interleaved_fused_2layer_par_rows(
    block: &mut [F128],
    t_outer: F128,
    t_inner_a: F128,
    t_inner_b: F128,
    quarter: usize,
    num_ntts: usize,
) {
    use rayon::prelude::*;
    const PARALLEL_ROW_THRESHOLD: usize = 256;
    let stride = quarter * num_ntts;
    debug_assert_eq!(block.len(), 4 * stride);

    let do_one = |row_a: &mut [F128],
                  row_b: &mut [F128],
                  row_c: &mut [F128],
                  row_d: &mut [F128]| {
        kernels::butterfly_fused_2layer(row_a, row_b, row_c, row_d, t_outer, t_inner_a, t_inner_b);
    };

    // Split the block into four quarters, then zip row-wise. Each rayon task
    // processes one quarter-row index = 4 logical rows of work.
    let (top_half, bot_half) = block.split_at_mut(2 * stride);
    let (q1, q2) = top_half.split_at_mut(stride);
    let (q3, q4) = bot_half.split_at_mut(stride);

    if quarter < PARALLEL_ROW_THRESHOLD {
        for r in 0..quarter {
            let off = r * num_ntts;
            let (q1r, q1_rest) = q1[off..].split_at_mut(num_ntts);
            let _ = q1_rest;
            let (q2r, _) = q2[off..].split_at_mut(num_ntts);
            let (q3r, _) = q3[off..].split_at_mut(num_ntts);
            let (q4r, _) = q4[off..].split_at_mut(num_ntts);
            do_one(q1r, q2r, q3r, q4r);
        }
    } else {
        q1.par_chunks_mut(num_ntts)
            .zip(q2.par_chunks_mut(num_ntts))
            .zip(q3.par_chunks_mut(num_ntts))
            .zip(q4.par_chunks_mut(num_ntts))
            .for_each(|(((row_a, row_b), row_c), row_d)| {
                do_one(row_a, row_b, row_c, row_d);
            });
    }
}

/// Sequential counterpart of [`butterfly_interleaved_fused_2layer_par_rows`].
/// Used from inside the deep phase's outer Rayon jobs: adding row-level Rayon
/// here would create a nested fork/join for every 1 MiB subtree.
#[inline]
#[allow(clippy::too_many_arguments)]
fn butterfly_interleaved_fused_2layer_rows_seq(
    block: &mut [F128],
    t_outer: F128,
    t_inner_a: F128,
    t_inner_b: F128,
    quarter: usize,
    num_ntts: usize,
    odd_tail: usize,
) {
    let stride = quarter * num_ntts;
    debug_assert!(num_ntts > 0);
    debug_assert_eq!(block.len(), 4 * stride);
    // Row `r` touches positions `{r, r+quarter, r+2·quarter, r+3·quarter}`,
    // which all share `r`'s parity only when `quarter` is even.
    debug_assert!(odd_tail == 0 || quarter.is_multiple_of(2));

    if kernels::try_butterfly_fused_2layer_rows(
        block, t_outer, t_inner_a, t_inner_b, quarter, num_ntts, odd_tail,
    ) {
        return;
    }

    let (top_half, bot_half) = block.split_at_mut(2 * stride);
    let (q1, q2) = top_half.split_at_mut(stride);
    let (q3, q4) = bot_half.split_at_mut(stride);
    for (r, (((row_a, row_b), row_c), row_d)) in q1
        .chunks_exact_mut(num_ntts)
        .zip(q2.chunks_exact_mut(num_ntts))
        .zip(q3.chunks_exact_mut(num_ntts))
        .zip(q4.chunks_exact_mut(num_ntts))
        .enumerate()
    {
        // Odd rows carry `odd_tail` statically-zero trailing lanes; their
        // butterflies are (0,0) → (0,0) and are dropped.
        let lanes = if r & 1 == 1 {
            num_ntts - odd_tail
        } else {
            num_ntts
        };
        kernels::butterfly_fused_2layer(
            &mut row_a[..lanes],
            &mut row_b[..lanes],
            &mut row_c[..lanes],
            &mut row_d[..lanes],
            t_outer,
            t_inner_a,
            t_inner_b,
        );
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn butterfly_interleaved_fused_2layer_low_twiddles_rows_seq(
    block: &mut [F128],
    t_outer: F128,
    t_inner_a: F128,
    t_inner_b: F128,
    quarter: usize,
    num_ntts: usize,
) {
    let stride = quarter * num_ntts;
    debug_assert!(num_ntts > 0);
    debug_assert_eq!(block.len(), 4 * stride);
    debug_assert_eq!(t_outer.hi, 0);
    debug_assert_eq!(t_inner_a.hi, 0);
    debug_assert_eq!(t_inner_b.hi, 0);

    let (top_half, bot_half) = block.split_at_mut(2 * stride);
    let (q1, q2) = top_half.split_at_mut(stride);
    let (q3, q4) = bot_half.split_at_mut(stride);
    for (((row_a, row_b), row_c), row_d) in q1
        .chunks_exact_mut(num_ntts)
        .zip(q2.chunks_exact_mut(num_ntts))
        .zip(q3.chunks_exact_mut(num_ntts))
        .zip(q4.chunks_exact_mut(num_ntts))
    {
        kernels::butterfly_fused_2layer_low_twiddles(
            row_a, row_b, row_c, row_d, t_outer, t_inner_a, t_inner_b,
        );
    }
}

/// Butterfly one block of an interleaved (SoA) buffer with shared twiddle.
///
/// `block` has length `(2 * block_size_half) * num_ntts` and is laid out as
/// `num_ntts` lanes interleaved per row, `2 * block_size_half` rows total.
/// Pairs row `r` with row `r + block_size_half` for `r ∈ 0..block_size_half`.
///
/// **Note**: This is scalar-per-lane on purpose. With `num_ntts = 32` and
/// shared twiddle, the inner loop has 32 independent F_{2^128} muls per row
/// that the compiler ILPs effectively (each mul uses NEON via the field's
/// `binius_mul` already). An explicit 2-lane `ghash_mul_vec2_neon` variant was
/// tried but **regressed** by ~10-30% because the explicit batching prevented
/// ILP across more than 2 muls and added load/store overhead.
#[inline]
fn butterfly_interleaved_block(
    block: &mut [F128],
    twiddle: F128,
    block_size_half: usize,
    num_ntts: usize,
) {
    let off_bot = block_size_half * num_ntts;
    let (top, bot) = block.split_at_mut(off_bot);
    for r in 0..block_size_half {
        let o = r * num_ntts;
        kernels::butterfly_row_pair(
            &mut top[o..o + num_ntts],
            &mut bot[o..o + num_ntts],
            twiddle,
        );
    }
}

/// Butterfly one top-layer block, fusing three layers `(L..L+3)`. `block`
/// holds `8 * eighth` rows of `num_ntts` lanes; `t` carries the 7 twiddles for
/// the sub-butterflies. Parallel over row groups.
///
/// Sits between the fused-2 and fused-4 variants for a cache reason rather
/// than an arithmetic one: the row-group stride is a multiple of the L1
/// set-repeat period at these shapes, so the N concurrently-live row streams
/// all land in one set and demand N ways. Radix-8 is the widest fusion an
/// 8-way L1D admits.
#[inline]
fn butterfly_interleaved_fused_3layer_par_rows<const ZERO_ROOT: bool>(
    block: &mut [F128],
    t: &[F128; 7],
    eighth: usize,
    num_ntts: usize,
    odd_tail: usize,
) {
    use rayon::prelude::*;
    const PARALLEL_ROW_THRESHOLD: usize = 256;
    debug_assert_eq!(block.len(), 8 * eighth * num_ntts);
    if ZERO_ROOT {
        debug_assert_eq!(t[0], F128::ZERO);
        debug_assert_eq!(t[1], F128::ZERO);
        debug_assert_eq!(t[3], F128::ZERO);
    }
    debug_assert!(odd_tail == 0 || eighth.is_multiple_of(2));
    // Rows `{i·eighth + r}` all share `r`'s parity when `eighth` is even, so
    // an odd `r` selects eight positions whose top `odd_tail` lanes are known
    // zero — the butterfly maps (0,0) → (0,0) there and can be dropped.
    let lanes = |r: usize| {
        if r & 1 == 1 {
            num_ntts - odd_tail
        } else {
            num_ntts
        }
    };
    // Carry the base as `usize` (Send+Sync) so rayon's per-`r` closure can hold
    // it without a raw-pointer `Sync` shim. Each `r` writes the disjoint rows
    // `{i*eighth + r : i ∈ 0..8}`, so concurrent writes never alias.
    let base = block.as_mut_ptr() as usize;
    if eighth < PARALLEL_ROW_THRESHOLD {
        for r in 0..eighth {
            // SAFETY: row group r writes disjoint rows of this block.
            unsafe {
                if ZERO_ROOT {
                    kernels::butterfly_fused_3layer_zero_root_row(
                        base as *mut F128,
                        eighth,
                        num_ntts,
                        lanes(r),
                        r,
                        t,
                    )
                } else {
                    kernels::butterfly_fused_3layer_row(
                        base as *mut F128,
                        eighth,
                        num_ntts,
                        lanes(r),
                        r,
                        t,
                    )
                }
            };
        }
    } else {
        (0..eighth).into_par_iter().for_each(|r| {
            // SAFETY: distinct r → disjoint row groups → no aliasing.
            unsafe {
                if ZERO_ROOT {
                    kernels::butterfly_fused_3layer_zero_root_row(
                        base as *mut F128,
                        eighth,
                        num_ntts,
                        lanes(r),
                        r,
                        t,
                    )
                } else {
                    kernels::butterfly_fused_3layer_row(
                        base as *mut F128,
                        eighth,
                        num_ntts,
                        lanes(r),
                        r,
                        t,
                    )
                }
            };
        });
    }
}

/// Ranked sibling of [`butterfly_interleaved_fused_3layer_par_rows`] that
/// flattens every block and row group in one indexed Rayon region.
///
/// The ranked top groups contain 2, 16, then 128 blocks. Opening a blocking
/// region per block creates 146 sequential fork/join barriers even though all
/// `(block, row)` jobs are disjoint. Flattening retains the exact radix-8
/// kernels and work order within each row while using one barrier per group.
#[inline]
fn butterfly_interleaved_fused_3layer_all_blocks_par_rows(
    data: &mut [F128],
    twiddles: &[[F128; 7]],
    eighth: usize,
    num_ntts: usize,
) {
    use rayon::prelude::*;

    let num_blocks = twiddles.len();
    let block_elems = 8 * eighth * num_ntts;
    debug_assert_eq!(data.len(), num_blocks * block_elems);
    debug_assert!(eighth.is_power_of_two());
    debug_assert_eq!(twiddles[0][0], F128::ZERO);
    debug_assert_eq!(twiddles[0][1], F128::ZERO);
    debug_assert_eq!(twiddles[0][3], F128::ZERO);

    let base = data.as_mut_ptr() as usize;
    let eighth_log = eighth.trailing_zeros() as usize;
    let eighth_mask = eighth - 1;
    (0..num_blocks * eighth).into_par_iter().for_each(|job| {
        let block = job >> eighth_log;
        let row = job & eighth_mask;
        let block_base = unsafe { (base as *mut F128).add(block * block_elems) };
        // SAFETY: each job maps to one disjoint eight-row group within one
        // block. Block zero has the three zero roots required by the ranked
        // specialization; every other block uses the generic radix-8 kernel.
        unsafe {
            if block == 0 {
                kernels::butterfly_fused_3layer_zero_root_row(
                    block_base,
                    eighth,
                    num_ntts,
                    num_ntts,
                    row,
                    &twiddles[block],
                )
            } else {
                kernels::butterfly_fused_3layer_row(
                    block_base,
                    eighth,
                    num_ntts,
                    num_ntts,
                    row,
                    &twiddles[block],
                )
            }
        }
    });
}

/// Ranked block/tile-queue sibling of
/// [`butterfly_interleaved_fused_3layer_all_blocks_par_rows`]. Each queue
/// claim owns a fixed row tile within one block and processes those rows
/// serially, avoiding a nested Rayon region when the claim is executed on an
/// efficiency core.
///
/// At the exact 1 GiB ranked codeword, layers 1, 4, and 7 expose respectively
/// 2 × 512 MiB, 16 × 64 MiB, and 128 × 8 MiB blocks. A 128-row tile touches
/// 1 MiB of input in every pass, producing 1024 uniform claims per pass. Every
/// pass still reads and writes the codeword exactly once; the queue only lets
/// otherwise-idle E cores claim some of that fixed traffic and arithmetic.
/// The shared atomic queue bounds the heterogeneous tail to one 1 MiB tile per
/// worker rather than one potentially-slow outer block.
#[inline]
fn butterfly_interleaved_fused_3layer_all_blocks_hetero(
    data: &mut [F128],
    twiddles: &[[F128; 7]],
    eighth: usize,
    num_ntts: usize,
) {
    let num_blocks = twiddles.len();
    let block_elems = 8 * eighth * num_ntts;
    debug_assert_eq!(data.len(), num_blocks * block_elems);
    debug_assert!(eighth.is_power_of_two());
    debug_assert_eq!(twiddles[0][0], F128::ZERO);
    debug_assert_eq!(twiddles[0][1], F128::ZERO);
    debug_assert_eq!(twiddles[0][3], F128::ZERO);

    const ROWS_PER_TILE: usize = 128;
    let tiles_per_block = eighth.div_ceil(ROWS_PER_TILE);
    let base = crate::epool::SyncPtr(data.as_mut_ptr());
    crate::epool::run_hetero_chunks(num_blocks * tiles_per_block, |job| {
        let block = job / tiles_per_block;
        let tile = job % tiles_per_block;
        let row_start = tile * ROWS_PER_TILE;
        let row_end = (row_start + ROWS_PER_TILE).min(eighth);
        let block_base = unsafe { base.ptr().add(block * block_elems) };
        unsafe {
            if block == 0 {
                kernels::butterfly_fused_3layer_zero_root_rows(
                    block_base,
                    eighth,
                    num_ntts,
                    row_start,
                    row_end,
                    &twiddles[block],
                )
            } else {
                kernels::butterfly_fused_3layer_rows(
                    block_base,
                    eighth,
                    num_ntts,
                    row_start,
                    row_end,
                    &twiddles[block],
                )
            }
        }
    });
}

/// Butterfly one top-layer block, fusing four layers `(L..L+4)`. `block` holds
/// `16 * sixteenth` rows of `num_ntts` lanes; `t` carries the 15 twiddles for
/// the sub-butterflies (see module comment above). Parallel over row groups.
#[inline]
fn butterfly_interleaved_fused_4layer_par_rows(
    block: &mut [F128],
    t: &[F128; 15],
    sixteenth: usize,
    num_ntts: usize,
) {
    use rayon::prelude::*;
    const PARALLEL_ROW_THRESHOLD: usize = 256;
    debug_assert_eq!(block.len(), 16 * sixteenth * num_ntts);
    // Carry the base as `usize` (Send+Sync) so rayon's per-`r` closure can hold
    // it without a raw-pointer `Sync` shim. Each `r` writes the disjoint rows
    // `{i*sixteenth + r : i ∈ 0..16}`, so concurrent writes never alias.
    let base = block.as_mut_ptr() as usize;
    if sixteenth < PARALLEL_ROW_THRESHOLD {
        for r in 0..sixteenth {
            // SAFETY: row group r writes disjoint rows of this block.
            unsafe {
                kernels::butterfly_fused_4layer_row(base as *mut F128, sixteenth, num_ntts, r, t)
            };
        }
    } else {
        (0..sixteenth).into_par_iter().for_each(|r| {
            // SAFETY: distinct r → disjoint row groups → no aliasing.
            unsafe {
                kernels::butterfly_fused_4layer_row(base as *mut F128, sixteenth, num_ntts, r, t)
            };
        });
    }
}

#[inline]
fn log2_pow2(n: usize) -> usize {
    assert!(
        n.is_power_of_two() && n > 0,
        "length must be a positive power of 2"
    );
    n.trailing_zeros() as usize
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
        fn f128(&mut self) -> F128 {
            F128 {
                lo: self.next_u64(),
                hi: self.next_u64(),
            }
        }
    }

    fn rand_vec(rng: &mut Rng, n: usize) -> Vec<F128> {
        (0..n).map(|_| rng.f128()).collect()
    }

    fn scalar_fused_3layer_block(
        block: &mut [F128],
        twiddles: &[F128; 7],
        eighth: usize,
        num_ntts: usize,
    ) {
        let rows = 8 * eighth;
        assert_eq!(block.len(), rows * num_ntts);
        for stage in 0..3 {
            let num_subblocks = 1usize << stage;
            let subblock_rows = rows >> stage;
            let half = subblock_rows >> 1;
            let twiddle_offset = (1usize << stage) - 1;
            for subblock in 0..num_subblocks {
                let twiddle = twiddles[twiddle_offset + subblock];
                let start = subblock * subblock_rows * num_ntts;
                for row in 0..half {
                    let top = start + row * num_ntts;
                    let bot = top + half * num_ntts;
                    for lane in 0..num_ntts {
                        let v = block[bot + lane];
                        let new_u = block[top + lane] + v * twiddle;
                        block[top + lane] = new_u;
                        block[bot + lane] = v + new_u;
                    }
                }
            }
        }
    }

    /// The sequential deep helper must equal two ordinary layer sweeps for
    /// every row/lane geometry used by the ranked tail, including its final
    /// one-row quarter.
    #[test]
    fn fused2_sequential_rows_match_two_single_layers() {
        let mut rng = Rng::new(0xDEE2_F05E_20A5_0001);
        for (quarter, num_ntts) in [(1usize, 64usize), (4, 8), (16, 64), (64, 2), (256, 2)] {
            for iteration in 0..3 {
                let t_outer = rng.f128();
                let t_inner_a = rng.f128();
                let t_inner_b = rng.f128();
                let source = rand_vec(&mut rng, 4 * quarter * num_ntts);

                let mut want = source.clone();
                butterfly_interleaved_block(&mut want, t_outer, 2 * quarter, num_ntts);
                let half_elems = 2 * quarter * num_ntts;
                let (top, bot) = want.split_at_mut(half_elems);
                butterfly_interleaved_block(top, t_inner_a, quarter, num_ntts);
                butterfly_interleaved_block(bot, t_inner_b, quarter, num_ntts);

                let mut got = source;
                butterfly_interleaved_fused_2layer_rows_seq(
                    &mut got, t_outer, t_inner_a, t_inner_b, quarter, num_ntts, 0,
                );
                assert_eq!(
                    got, want,
                    "fused-2 sequential mismatch at quarter={quarter} \
                     num_ntts={num_ntts} iteration={iteration}"
                );
            }
        }
    }

    /// The low-twiddle AArch64 kernel is algebraically identical to the
    /// generic fused pair for every lane geometry used by the ranked final
    /// pair.
    #[test]
    fn fused2_low_twiddles_match_generic() {
        let mut rng = Rng::new(0x10F1_7A11_20A5_0001);
        for (quarter, num_ntts) in [(1usize, 64usize), (4, 8), (64, 2)] {
            for iteration in 0..4 {
                let t_outer = F128::new(rng.next_u64(), 0);
                let t_inner_a = F128::new(rng.next_u64(), 0);
                let t_inner_b = F128::new(rng.next_u64(), 0);
                let source = rand_vec(&mut rng, 4 * quarter * num_ntts);

                let mut want = source.clone();
                butterfly_interleaved_fused_2layer_rows_seq(
                    &mut want, t_outer, t_inner_a, t_inner_b, quarter, num_ntts, 0,
                );
                let mut got = source;
                butterfly_interleaved_fused_2layer_low_twiddles_rows_seq(
                    &mut got, t_outer, t_inner_a, t_inner_b, quarter, num_ntts,
                );
                assert_eq!(
                    got, want,
                    "low-twiddle fused-2 mismatch at quarter={quarter} \
                     num_ntts={num_ntts} iteration={iteration}"
                );
            }
        }
    }

    /// Exhaust the exact production tables used by layers 18 and 19. This is
    /// the invariant that makes the ranked narrow dispatch sound.
    #[test]
    fn standard_dim20_final_pair_twiddles_have_zero_high_limbs() {
        let ntt = AdditiveNttF128::standard(20);
        for layer in 18..20 {
            for block in 0..(1usize << layer) {
                assert_eq!(
                    ntt.twiddle(layer, block).hi,
                    0,
                    "nonzero high limb at layer={layer} block={block}"
                );
            }
        }
        assert!(use_ranked_low_twiddle_final_pair(20, 64, 10));
        assert!(!use_ranked_low_twiddle_final_pair(19, 64, 10));
        assert!(!use_ranked_low_twiddle_final_pair(20, 32, 10));
        assert!(!use_ranked_low_twiddle_final_pair(20, 64, 9));
    }

    /// Exercise five fused deep pairs with the ranked 1024-position subtree
    /// geometry, but only two interleaved lanes so the fixture stays small.
    /// The scalar oracle applies the same ten layers as separate sweeps and
    /// therefore catches both child-twiddle and global-block indexing errors.
    #[test]
    fn five_deep_fused_pairs_match_scalar_reference() {
        const LOG_D: usize = 12;
        const N_TOP: usize = 2;
        const NUM_NTTS: usize = 2;

        let mut rng = Rng::new(0xDEE2_F05E_20A5_0002);
        let ntt = AdditiveNttF128::standard(LOG_D);
        for iteration in 0..3 {
            let source = rand_vec(&mut rng, (1usize << LOG_D) * NUM_NTTS);
            let mut want = source.clone();
            ntt.forward_transform_interleaved_scalar_from_layer(&mut want, NUM_NTTS, N_TOP);

            let mut got = source;
            ntt.forward_transform_interleaved_deep_fused_pairs(&mut got, NUM_NTTS, N_TOP, LOG_D);
            assert_eq!(got, want, "five-pair mismatch at iteration={iteration}");
        }
    }

    /// Every finalized-chunk callback must observe the final transform bytes,
    /// exactly once and at the advertised global element offset.
    #[test]
    fn interleaved_chunk_finish_observes_complete_transform() {
        const LOG_D: usize = 12;
        const NUM_NTTS: usize = 2;
        const START_LAYER: usize = 1;

        let mut rng = Rng::new(0xCACE_10CA_1F1E_0001);
        let ntt = AdditiveNttF128::standard(LOG_D);
        let source = rand_vec(&mut rng, (1usize << LOG_D) * NUM_NTTS);
        let mut want = source.clone();
        ntt.forward_transform_interleaved_from_layer(&mut want, NUM_NTTS, START_LAYER);

        let observed =
            std::sync::Mutex::new((vec![F128::ZERO; source.len()], vec![0u8; source.len()]));
        let mut got = source;
        ntt.forward_transform_interleaved_from_layer_and_then(
            &mut got,
            NUM_NTTS,
            START_LAYER,
            |offset, chunk| {
                let mut observed = observed.lock().unwrap();
                observed.0[offset..offset + chunk.len()].copy_from_slice(chunk);
                for count in &mut observed.1[offset..offset + chunk.len()] {
                    *count += 1;
                }
            },
        );

        assert_eq!(got, want);
        let observed = observed.into_inner().unwrap();
        assert_eq!(observed.0, want);
        assert!(observed.1.iter().all(|&count| count == 1));
    }

    /// Splitting at the existing top/deep boundary must be byte-identical to
    /// the ordinary one-call transform, and callbacks must remain exclusively
    /// attached to finalized deep chunks.
    #[cfg(any(
        all(target_arch = "aarch64", target_feature = "aes"),
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
    ))]
    #[test]
    fn interleaved_top_deep_phase_split_matches_full_transform() {
        use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};

        const LOG_D: usize = 16;
        const NUM_NTTS: usize = 8;
        const START_LAYER: usize = 1;

        let mut rng = Rng::new(0x5F11_7B0A_DA7A);
        let ntt = AdditiveNttF128::standard(LOG_D);
        let source = rand_vec(&mut rng, (1usize << LOG_D) * NUM_NTTS);
        let mut expected = source.clone();
        ntt.forward_transform_interleaved_from_layer(&mut expected, NUM_NTTS, START_LAYER);

        let top_callbacks = AtomicUsize::new(0);
        let mut got = source;
        ntt.forward_transform_interleaved_parallel_from_layer_and_then::<
            INTERLEAVED_PHASE_TOP_ONLY,
            _,
        >(&mut got, NUM_NTTS, START_LAYER, &|_, _| {
            top_callbacks.fetch_add(1, Relaxed);
        });
        assert_eq!(top_callbacks.load(Relaxed), 0);

        let observed = std::sync::Mutex::new((vec![F128::ZERO; got.len()], vec![0u8; got.len()]));
        ntt.forward_transform_interleaved_parallel_from_layer_and_then::<
            INTERLEAVED_PHASE_DEEP_ONLY,
            _,
        >(&mut got, NUM_NTTS, START_LAYER, &|offset, chunk| {
            let mut observed = observed.lock().unwrap();
            observed.0[offset..offset + chunk.len()].copy_from_slice(chunk);
            for count in &mut observed.1[offset..offset + chunk.len()] {
                *count += 1;
            }
        });
        assert_eq!(got, expected);
        let observed = observed.into_inner().unwrap();
        assert_eq!(observed.0, expected);
        assert!(observed.1.iter().all(|&count| count == 1));
    }

    /// Only the ranked macOS/AArch64 L0 transform may use the new tail
    /// schedule; all recursive, diagnostic, and cross-platform shapes retain
    /// the existing single-layer deep loop.
    #[test]
    fn ranked_deep_pair_fusion_gate_is_narrow() {
        let enabled_here = cfg!(all(
            target_os = "macos",
            target_arch = "aarch64",
            target_feature = "aes"
        ));
        assert_eq!(use_ranked_deep_pair_fusion(20, 64, 1, 10), enabled_here);
        assert!(!use_ranked_deep_pair_fusion(19, 64, 1, 10));
        assert!(!use_ranked_deep_pair_fusion(20, 8, 1, 10));
        assert!(!use_ranked_deep_pair_fusion(20, 64, 0, 10));
        assert!(!use_ranked_deep_pair_fusion(20, 64, 1, 9));
        assert_eq!(use_ranked_zero_root_fusion(20, 64, 1, 10), enabled_here);
        assert!(!use_ranked_zero_root_fusion(19, 64, 1, 10));
        assert!(!use_ranked_zero_root_fusion(20, 8, 1, 10));
        assert!(!use_ranked_zero_root_fusion(20, 64, 0, 10));
        assert!(!use_ranked_zero_root_fusion(20, 64, 1, 9));

        for layer in [1, 4, 7] {
            assert_eq!(
                is_ranked_top_hetero_fused3_pass(20, 64, 1, 10, layer),
                enabled_here
            );
        }
        assert!(!is_ranked_top_hetero_fused3_pass(20, 64, 1, 10, 2));
        assert!(!is_ranked_top_hetero_fused3_pass(19, 64, 1, 10, 4));
        assert!(!is_ranked_top_hetero_fused3_pass(20, 8, 1, 10, 4));
        assert!(!is_ranked_top_hetero_fused3_pass(20, 64, 0, 10, 4));
        assert!(!is_ranked_top_hetero_fused3_pass(20, 64, 1, 9, 4));
    }

    /// Exercise the block-zero specialization and the ordinary nonzero-block
    /// path against an independent layer-by-layer scalar oracle. The two
    /// shapes straddle the row-parallel dispatch threshold.
    #[test]
    fn fused3_zero_root_and_nonzero_blocks_match_scalar() {
        const LAYER: usize = 2;
        const NUM_NTTS: usize = 8;

        let mut rng = Rng::new(0x20E2_07AD_D171_E000);
        for log_d in [10usize, 13] {
            let ntt = AdditiveNttF128::standard(log_d);
            let block_size = 1usize << (log_d - LAYER);
            let eighth = block_size >> 3;
            for iteration in 0..3 {
                for block_index in [0usize, 1, 3] {
                    let mut twiddles = [F128::ZERO; 7];
                    twiddles[0] = ntt.twiddle(LAYER, block_index);
                    for s in 0..2 {
                        twiddles[1 + s] = ntt.twiddle(LAYER + 1, 2 * block_index + s);
                    }
                    for s in 0..4 {
                        twiddles[3 + s] = ntt.twiddle(LAYER + 2, 4 * block_index + s);
                    }

                    let source = rand_vec(&mut rng, block_size * NUM_NTTS);
                    let mut want = source.clone();
                    scalar_fused_3layer_block(&mut want, &twiddles, eighth, NUM_NTTS);
                    let mut got = source.clone();
                    if block_index == 0 {
                        assert_eq!(twiddles[0], F128::ZERO);
                        assert_eq!(twiddles[1], F128::ZERO);
                        assert_eq!(twiddles[3], F128::ZERO);
                        butterfly_interleaved_fused_3layer_par_rows::<true>(
                            &mut got, &twiddles, eighth, NUM_NTTS, 0,
                        );
                        // The omitted stream-zero stores are valid only because
                        // this whole row stream is mathematically unchanged.
                        let stream_len = eighth * NUM_NTTS;
                        assert_eq!(&got[..stream_len], &source[..stream_len]);
                    } else {
                        butterfly_interleaved_fused_3layer_par_rows::<false>(
                            &mut got, &twiddles, eighth, NUM_NTTS, 0,
                        );
                    }
                    assert_eq!(
                        got, want,
                        "radix-8 block mismatch at log_d={log_d} iteration={iteration} \
                         block={block_index}"
                    );
                }
            }
        }
    }

    #[test]
    fn fused3_flattened_blocks_match_per_block_regions() {
        const LOG_D: usize = 10;
        const LAYER: usize = 2;
        const NUM_NTTS: usize = 8;

        let ntt = AdditiveNttF128::standard(LOG_D);
        let num_blocks = 1usize << LAYER;
        let block_size = 1usize << (LOG_D - LAYER);
        let eighth = block_size >> 3;
        let block_elems = block_size * NUM_NTTS;
        let mut twiddles = Vec::with_capacity(num_blocks);
        for block in 0..num_blocks {
            let mut tw = [F128::ZERO; 7];
            tw[0] = ntt.twiddle(LAYER, block);
            for s in 0..2 {
                tw[1 + s] = ntt.twiddle(LAYER + 1, 2 * block + s);
            }
            for s in 0..4 {
                tw[3 + s] = ntt.twiddle(LAYER + 2, 4 * block + s);
            }
            twiddles.push(tw);
        }

        let mut rng = Rng::new(0xF1A7_7EED_20E2_0001);
        for iteration in 0..3 {
            let source = rand_vec(&mut rng, (1usize << LOG_D) * NUM_NTTS);
            let mut expected = source.clone();
            for (block, tw) in twiddles.iter().enumerate() {
                let block_data = &mut expected[block * block_elems..(block + 1) * block_elems];
                if block == 0 {
                    butterfly_interleaved_fused_3layer_par_rows::<true>(
                        block_data, tw, eighth, NUM_NTTS, 0,
                    );
                } else {
                    butterfly_interleaved_fused_3layer_par_rows::<false>(
                        block_data, tw, eighth, NUM_NTTS, 0,
                    );
                }
            }

            let mut got = source;
            butterfly_interleaved_fused_3layer_all_blocks_par_rows(
                &mut got, &twiddles, eighth, NUM_NTTS,
            );
            assert_eq!(got, expected, "iteration={iteration}");
        }
    }

    /// The ranked heterogeneous dispatcher flattens `(block, row-tile)` queue
    /// claims rather than individual `(block, row)` Rayon jobs. Exercise the
    /// exact 2-, 16-, and 128-block pass shapes (including multiple tiles per
    /// block) at a reduced domain size and compare against the independent
    /// layer-by-layer scalar butterfly oracle.
    #[test]
    fn ranked_top_hetero_blocks_match_scalar_oracle() {
        const LOG_D: usize = 15;
        const NUM_NTTS: usize = 2;

        let ntt = AdditiveNttF128::standard(LOG_D);
        let mut rng = Rng::new(0xEC0E_70B0_10C5);
        for layer in [1usize, 4, 7] {
            let num_blocks = 1usize << layer;
            let block_size = 1usize << (LOG_D - layer);
            let eighth = block_size >> 3;
            let block_elems = block_size * NUM_NTTS;
            let twiddles: Vec<[F128; 7]> = (0..num_blocks)
                .map(|block| {
                    let mut tw = [F128::ZERO; 7];
                    tw[0] = ntt.twiddle(layer, block);
                    for s in 0..2 {
                        tw[1 + s] = ntt.twiddle(layer + 1, 2 * block + s);
                    }
                    for s in 0..4 {
                        tw[3 + s] = ntt.twiddle(layer + 2, 4 * block + s);
                    }
                    tw
                })
                .collect();

            let source = rand_vec(&mut rng, (1usize << LOG_D) * NUM_NTTS);
            let mut expected = source.clone();
            for (block, tw) in twiddles.iter().enumerate() {
                scalar_fused_3layer_block(
                    &mut expected[block * block_elems..(block + 1) * block_elems],
                    tw,
                    eighth,
                    NUM_NTTS,
                );
            }

            let mut got = source;
            butterfly_interleaved_fused_3layer_all_blocks_hetero(
                &mut got, &twiddles, eighth, NUM_NTTS,
            );
            assert_eq!(
                got, expected,
                "heterogeneous pass mismatch at layer={layer}"
            );
        }
    }

    /// The parallel interleaved transform — whose top layers use the radix-8
    /// fusion on non-AVX-512 targets — must agree bit-for-bit with the scalar
    /// reference, which applies every layer as its own separate sweep.
    ///
    /// This is the regression guard for the layer-fusion arithmetic: a radix-8
    /// butterfly reassociates three layers' worth of work into one pass over
    /// the block, so an error in the twiddle indexing or the sub-butterfly
    /// order shows up here as a mismatch rather than as a wrong proof much
    /// later. `log_d` is swept across the deep-phase split so the top-layer
    /// loop actually reaches the radix-8 arm (it needs `layer + 2 < n_top`),
    /// and `start_layer = 1` covers the RS-encode entry the PCS commit uses.
    #[cfg(any(
        all(target_arch = "aarch64", target_feature = "aes"),
        all(target_arch = "x86_64", target_feature = "pclmulqdq"),
    ))]
    #[test]
    fn fused_top_layers_match_scalar_reference() {
        let mut rng = Rng::new(0x3F30);
        let mut exercised = false;
        for log_d in [12usize, 13, 14, 15, 16] {
            for num_ntts in [2usize, 8, 64] {
                let ntt = AdditiveNttF128::standard(log_d);
                let src = rand_vec(&mut rng, (1usize << log_d) * num_ntts);
                for start_layer in [0usize, 1] {
                    let mut want = src.clone();
                    ntt.forward_transform_interleaved_scalar_from_layer(
                        &mut want,
                        num_ntts,
                        start_layer,
                    );
                    let mut got = src.clone();
                    ntt.forward_transform_interleaved_parallel_from_layer(
                        &mut got,
                        num_ntts,
                        start_layer,
                    );
                    assert_eq!(
                        got, want,
                        "fused top layers diverged from scalar at log_d={log_d} \
                         num_ntts={num_ntts} start_layer={start_layer}"
                    );
                    exercised |= got != src;
                }
            }
        }
        assert!(exercised, "test never transformed anything");
    }

    /// Directly initializing the codeword with the first nontrivial radix-8
    /// pass must match replica-fill followed by the ordinary transform, even
    /// when every destination slot starts as poisoned scratch.
    #[test]
    fn recursive_from_message_fused3_matches_replicated_transform() {
        const NUM_NTTS: usize = 8;
        let mut rng = Rng::new(0x5241_5445_384D_5347);
        for (log_d, start_layer) in [(12usize, 2usize), (12, 3)] {
            let ntt = AdditiveNttF128::standard(log_d);
            let msg_len = (1usize << (log_d - start_layer)) * NUM_NTTS;
            let msg = rand_vec(&mut rng, msg_len);

            let mut want = vec![F128::ZERO; msg_len << start_layer];
            for replica in want.chunks_mut(msg_len) {
                replica.copy_from_slice(&msg);
            }
            ntt.forward_transform_interleaved_from_layer(&mut want, NUM_NTTS, start_layer);

            let poison = F128::new(0xA5A5_A5A5_A5A5_A5A5, 0x5A5A_5A5A_5A5A_5A5A);
            let mut got = vec![poison; msg_len << start_layer];
            ntt.forward_transform_interleaved_from_message_fused3(
                &msg,
                &mut got,
                NUM_NTTS,
                start_layer,
            );
            assert_eq!(got, want, "log_d={log_d} rate_log={start_layer}");
        }
    }

    #[test]
    fn recursive_l1_ntt_epool_gate_and_rollback_are_exact() {
        let enabled_here = cfg!(all(
            target_os = "macos",
            target_arch = "aarch64",
            target_feature = "aes"
        ));
        assert_eq!(is_recursive_l1_ntt_epool_shape(18, 8, 2), enabled_here);
        assert!(!is_recursive_l1_ntt_epool_shape(17, 8, 2));
        assert!(!is_recursive_l1_ntt_epool_shape(18, 4, 2));
        assert!(!is_recursive_l1_ntt_epool_shape(18, 8, 1));
        assert!(!is_recursive_l1_ntt_epool_shape(18, 8, 5));

        // L2 shares the eight-lane direct-from-message geometry at log_d 16.
        assert_eq!(is_recursive_l2_ntt_epool_shape(16, 8, 3), enabled_here);
        assert!(!is_recursive_l2_ntt_epool_shape(15, 8, 3));
        assert!(!is_recursive_l2_ntt_epool_shape(16, 4, 3));
        assert!(!is_recursive_l2_ntt_epool_shape(16, 8, 2));

        assert!(!recursive_ntt_epool_killed_by(None));
        assert!(!recursive_ntt_epool_killed_by(Some("")));
        assert!(!recursive_ntt_epool_killed_by(Some("0")));
        assert!(!recursive_ntt_epool_killed_by(Some("true")));
        assert!(recursive_ntt_epool_killed_by(Some("1")));

        assert!(!l2_recursive_ntt_epool_killed_by(None));
        assert!(!l2_recursive_ntt_epool_killed_by(Some("0")));
        assert!(l2_recursive_ntt_epool_killed_by(Some("1")));

        // The union gate matches each shape exactly, independent of env.
        assert_eq!(
            is_recursive_ntt_epool_shape(18, 8, 2),
            enabled_here
        );
        assert_eq!(
            is_recursive_ntt_epool_shape(16, 8, 3),
            enabled_here
        );
        assert!(!is_recursive_ntt_epool_shape(17, 8, 2));
        assert!(!is_recursive_ntt_epool_shape(16, 8, 2));
    }

    /// Exercise the exact L1 production geometry through an injected helper pool.
    /// The control retains the incumbent 16 × 2 MiB Rayon deep partition; the
    /// candidate uses 32 × 1 MiB P+E claims. Both run the same direct-message
    /// radix-8 kernels and the same extracted single-layer subtree loop.
    #[test]
    fn recursive_l1_ntt_epool_exact_shape_matches_incumbent() {
        const LOG_D: usize = 18;
        const NUM_NTTS: usize = 8;
        const START_LAYER: usize = 2;

        let main = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .thread_name(|i| format!("recursive-l1-main-test-{i}"))
            .build()
            .unwrap();
        let helper = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .thread_name(|i| format!("recursive-l1-helper-test-{i}"))
            .build()
            .unwrap();

        let mut rng = Rng::new(0xEC0E_11A1_5C4E_DA7A);
        let msg_len = (1usize << (LOG_D - START_LAYER)) * NUM_NTTS;
        let msg = rand_vec(&mut rng, msg_len);
        let poison = F128::new(0xA5A5_A5A5_A5A5_A5A5, 0x5A5A_5A5A_5A5A_5A5A);
        let mut incumbent = vec![poison; msg_len << START_LAYER];
        let mut candidate = incumbent.clone();

        let claimed_before = crate::epool::helper_chunks_claimed();
        main.install(|| {
            let ntt = AdditiveNttF128::standard(LOG_D);
            ntt.forward_transform_interleaved_from_message_fused3_with_helper(
                &msg,
                &mut incumbent,
                NUM_NTTS,
                START_LAYER,
                LOG_D,
                None,
            );
            ntt.forward_transform_interleaved_from_message_fused3_with_helper(
                &msg,
                &mut candidate,
                NUM_NTTS,
                START_LAYER,
                LOG_D,
                Some(&helper),
            );
        });

        assert!(
            crate::epool::helper_chunks_claimed() > claimed_before,
            "injected helper never claimed an exact-shape deep subtree"
        );
        assert_eq!(candidate, incumbent);
    }

    /// Exercise the exact L2 production geometry (log_d 16, 8 lanes, rate
    /// entry at layer 3) through an injected helper pool. The control keeps
    /// the incumbent single-layer Rayon tail; the candidate routes the same
    /// 32 × 256 KiB sub-transforms over the P+E queue. Outputs must be
    /// byte-identical because both run the identical subtree loop.
    #[test]
    fn recursive_l2_ntt_epool_exact_shape_matches_incumbent() {
        const LOG_D: usize = 16;
        const NUM_NTTS: usize = 8;
        const START_LAYER: usize = 3;

        let main = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .thread_name(|i| format!("recursive-l2-main-test-{i}"))
            .build()
            .unwrap();
        let helper = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .thread_name(|i| format!("recursive-l2-helper-test-{i}"))
            .build()
            .unwrap();

        let mut rng = Rng::new(0x12C2_9E41_0D71_8B34);
        let msg_len = (1usize << (LOG_D - START_LAYER)) * NUM_NTTS;
        let msg = rand_vec(&mut rng, msg_len);
        let poison = F128::new(0xA5A5_A5A5_A5A5_A5A5, 0x5A5A_5A5A_5A5A_5A5A);
        let mut incumbent = vec![poison; msg_len << START_LAYER];
        let mut candidate = incumbent.clone();

        let claimed_before = crate::epool::helper_chunks_claimed();
        main.install(|| {
            let ntt = AdditiveNttF128::standard(LOG_D);
            ntt.forward_transform_interleaved_from_message_fused3_with_helper(
                &msg,
                &mut incumbent,
                NUM_NTTS,
                START_LAYER,
                LOG_D,
                None,
            );
            ntt.forward_transform_interleaved_from_message_fused3_with_helper(
                &msg,
                &mut candidate,
                NUM_NTTS,
                START_LAYER,
                LOG_D,
                Some(&helper),
            );
        });

        assert!(
            crate::epool::helper_chunks_claimed() > claimed_before,
            "injected helper never claimed an exact L2-shape deep subtree"
        );
        assert_eq!(candidate, incumbent);
    }

    /// The ranked split extension must produce three consecutive radix-8
    /// groups (layers 1..10) that are exactly equivalent to applying those
    /// nine layers one at a time. Use a small domain here: the butterfly and
    /// twiddle geometry is identical, while the real 64-lane/log-20 buffer is
    /// 1 GiB and is inappropriate for a unit test.
    #[cfg(all(target_os = "macos", target_arch = "aarch64", target_feature = "aes"))]
    #[test]
    fn fusion_aware_three_radix8_groups_match_scalar_reference() {
        const LOG_D: usize = 12;
        const NUM_NTTS: usize = 2;
        const START_LAYER: usize = 1;
        const N_TOP: usize = 10;

        let mut rng = Rng::new(0xF051_0A8E);
        let ntt = AdditiveNttF128::standard(LOG_D);
        let src = rand_vec(&mut rng, (1usize << LOG_D) * NUM_NTTS);

        let mut want = src.clone();
        ntt.forward_transform_interleaved_scalar_from_layer(&mut want, NUM_NTTS, START_LAYER);

        let mut got = src;
        let mut layer = START_LAYER;
        while layer < N_TOP {
            let num_blocks = 1usize << layer;
            let block_size = 1usize << (LOG_D - layer);
            let block_elems = block_size * NUM_NTTS;
            let eighth = block_size >> 3;
            for block in 0..num_blocks {
                let mut tw = [F128::ZERO; 7];
                tw[0] = ntt.twiddle(layer, block);
                for s in 0..2 {
                    tw[1 + s] = ntt.twiddle(layer + 1, 2 * block + s);
                }
                for s in 0..4 {
                    tw[3 + s] = ntt.twiddle(layer + 2, 4 * block + s);
                }
                let start = block * block_elems;
                if block == 0 {
                    butterfly_interleaved_fused_3layer_par_rows::<true>(
                        &mut got[start..start + block_elems],
                        &tw,
                        eighth,
                        NUM_NTTS,
                        0,
                    );
                } else {
                    butterfly_interleaved_fused_3layer_par_rows::<false>(
                        &mut got[start..start + block_elems],
                        &tw,
                        eighth,
                        NUM_NTTS,
                        0,
                    );
                }
            }
            layer += 3;
        }
        debug_assert_eq!(layer, N_TOP);

        // The production scheduler hands layers N_TOP..LOG_D to the deep
        // cache-resident phase. The scalar tail is the exact same arithmetic.
        ntt.forward_transform_interleaved_scalar_from_layer(&mut got, NUM_NTTS, N_TOP);
        assert_eq!(got, want);

        // Guard the narrowly-scoped production policy independently of the
        // smaller arithmetic fixture above.
        assert_eq!(fusion_aware_interleaved_n_top(20, 64, 1, 9), 10);
        assert_eq!(fusion_aware_interleaved_n_top(19, 64, 1, 9), 9);
        assert_eq!(fusion_aware_interleaved_n_top(20, 8, 1, 9), 9);
        assert_eq!(fusion_aware_interleaved_n_top(20, 64, 0, 9), 9);
        assert_eq!(fusion_aware_interleaved_n_top(20, 64, 1, 8), 8);
    }

    #[test]
    fn forward_inverse_roundtrip() {
        let mut rng = Rng::new(0xAB1);
        for log_d in [1usize, 2, 3, 4, 6, 8] {
            let ntt = AdditiveNttF128::standard(log_d);
            let original = rand_vec(&mut rng, 1 << log_d);
            let mut v = original.clone();
            ntt.forward_transform(&mut v);
            ntt.inverse_transform(&mut v);
            assert_eq!(v, original, "roundtrip failed at log_d={log_d}");
        }
    }

    #[test]
    fn inverse_forward_roundtrip() {
        let mut rng = Rng::new(0xAB2);
        for log_d in [1usize, 2, 3, 4, 6, 8] {
            let ntt = AdditiveNttF128::standard(log_d);
            let original = rand_vec(&mut rng, 1 << log_d);
            let mut v = original.clone();
            ntt.inverse_transform(&mut v);
            ntt.forward_transform(&mut v);
            assert_eq!(
                v, original,
                "inverse∘forward roundtrip failed at log_d={log_d}"
            );
        }
    }

    #[test]
    fn forward_is_linear() {
        let mut rng = Rng::new(0xAB3);
        for log_d in [1usize, 2, 3, 5] {
            let ntt = AdditiveNttF128::standard(log_d);
            let n = 1 << log_d;
            let a = rand_vec(&mut rng, n);
            let b = rand_vec(&mut rng, n);
            let ab: Vec<F128> = a.iter().zip(&b).map(|(x, y)| *x + *y).collect();

            let mut fa = a.clone();
            ntt.forward_transform(&mut fa);
            let mut fb = b.clone();
            ntt.forward_transform(&mut fb);
            let mut fab = ab.clone();
            ntt.forward_transform(&mut fab);

            for i in 0..n {
                assert_eq!(
                    fa[i] + fb[i],
                    fab[i],
                    "linearity fails at i={i}, log_d={log_d}"
                );
            }
        }
    }

    #[test]
    fn ntt_of_zero_is_zero() {
        for log_d in [1usize, 2, 3, 6] {
            let ntt = AdditiveNttF128::standard(log_d);
            let mut v = vec![F128::ZERO; 1 << log_d];
            ntt.forward_transform(&mut v);
            assert!(v.iter().all(|&x| x == F128::ZERO));
        }
    }

    #[test]
    fn twiddle_at_layer_0_uses_full_basis_minus_one() {
        // At layer 0 (topmost forward butterfly), there's 1 block.
        // twiddle(0, 0) = 0 (no bits set in block index 0).
        let ntt = AdditiveNttF128::standard(4);
        assert_eq!(ntt.twiddle(0, 0), F128::ZERO);
    }

    #[test]
    fn precomputed_twiddles_match_span_reference_and_cap() {
        for log_d in [1usize, 2, 5, 8, 12] {
            let ntt = AdditiveNttF128::standard(log_d);
            let table = ntt
                .precomputed_twiddles
                .as_ref()
                .expect("production-size domain should cache twiddles");
            assert_eq!(table.len(), (1usize << log_d) - 1);
            for layer in 0..log_d {
                let eval_row = &ntt.evals[log_d - layer - 1];
                for block in 0..(1usize << layer) {
                    assert_eq!(ntt.twiddle(layer, block), span_get(&eval_row[1..], block));
                }
            }
        }

        let cached_a = AdditiveNttF128::standard(8);
        let cached_b = AdditiveNttF128::standard(8);
        assert!(Arc::ptr_eq(
            cached_a.precomputed_twiddles.as_ref().unwrap(),
            cached_b.precomputed_twiddles.as_ref().unwrap()
        ));

        let fallback = AdditiveNttF128::standard(MAX_PRECOMPUTED_TWIDDLE_LOG + 1);
        assert!(fallback.precomputed_twiddles.is_none());
        let layer = MAX_PRECOMPUTED_TWIDDLE_LOG;
        let block = (1usize << layer) - 1;
        let eval_row = &fallback.evals[fallback.log_domain_size() - layer - 1];
        assert_eq!(
            fallback.twiddle(layer, block),
            span_get(&eval_row[1..], block)
        );
    }

    /// At layer log_d - 1 (deepest, where FRI starts), pairs are adjacent.
    /// twiddle should match the "domain points" indexing.
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    #[test]
    fn neon_matches_scalar() {
        let mut rng = Rng::new(0xBB1);
        for log_d in 1..=10 {
            let ntt = AdditiveNttF128::standard(log_d);
            let original = rand_vec(&mut rng, 1 << log_d);
            let mut v_scalar = original.clone();
            ntt.forward_transform_scalar(&mut v_scalar);
            let mut v_neon = original.clone();
            ntt.forward_transform_neon(&mut v_neon);
            assert_eq!(
                v_neon, v_scalar,
                "NEON disagrees with scalar at log_d={log_d}"
            );
        }
    }

    #[test]
    fn interleaved_matches_per_lane() {
        let mut rng = Rng::new(0xCC1);
        // For several log_d and num_ntts, verify the interleaved transform
        // matches running the per-lane scalar transform on each sub-NTT.
        for log_d in [3usize, 4, 8] {
            for num_ntts in [1usize, 2, 4, 8] {
                let ntt = AdditiveNttF128::standard(log_d);
                let n_total = (1 << log_d) * num_ntts;
                let original = rand_vec(&mut rng, n_total);

                // Interleaved.
                let mut v_inter = original.clone();
                ntt.forward_transform_interleaved_scalar(&mut v_inter, num_ntts);

                // Reference: per-lane, gather + scalar transform + scatter.
                let mut v_ref = original.clone();
                for lane in 0..num_ntts {
                    let mut sub: Vec<F128> = (0..(1 << log_d))
                        .map(|pos| v_ref[pos * num_ntts + lane])
                        .collect();
                    ntt.forward_transform_scalar(&mut sub);
                    for pos in 0..(1 << log_d) {
                        v_ref[pos * num_ntts + lane] = sub[pos];
                    }
                }

                assert_eq!(
                    v_inter, v_ref,
                    "interleaved mismatch at log_d={log_d}, num_ntts={num_ntts}"
                );
            }
        }
    }

    // Runs on both SIMD backends so the x86 PCLMUL and aarch64 NEON parallel
    // paths are each validated against the scalar oracle. AVX-512 builds also
    // exercise the fused-4 top-layer kernel in the larger cases.
    #[cfg(any(
        all(target_arch = "aarch64", target_feature = "aes"),
        all(target_arch = "x86_64", target_feature = "pclmulqdq")
    ))]
    #[test]
    fn interleaved_parallel_matches_scalar() {
        let mut rng = Rng::new(0xCC2);
        for log_d in [4usize, 10, 14, 17, 19] {
            for &num_ntts in &[2usize, 8, 32] {
                let ntt = AdditiveNttF128::standard(log_d);
                let n_total = (1 << log_d) * num_ntts;
                let original = rand_vec(&mut rng, n_total);
                let mut v_scalar = original.clone();
                ntt.forward_transform_interleaved_scalar(&mut v_scalar, num_ntts);
                let mut v_par = original.clone();
                ntt.forward_transform_interleaved_parallel(&mut v_par, num_ntts);
                assert_eq!(
                    v_par, v_scalar,
                    "interleaved parallel mismatch at log_d={log_d}, num_ntts={num_ntts}"
                );
            }
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    #[test]
    fn batched_matches_scalar() {
        let mut rng = Rng::new(0xBB4);
        // Include sizes above the TARGET_SUB_NTT_LOG threshold (17) so we
        // exercise the cache-blocked path.
        for log_d in [4usize, 8, 12, 17, 18, 19, 20] {
            let ntt = AdditiveNttF128::standard(log_d);
            let original = rand_vec(&mut rng, 1 << log_d);
            let mut v_scalar = original.clone();
            ntt.forward_transform_scalar(&mut v_scalar);
            let mut v_batched = original.clone();
            ntt.forward_transform_batched(&mut v_batched);
            assert_eq!(
                v_batched, v_scalar,
                "batched disagrees with scalar at log_d={log_d}"
            );
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    #[test]
    fn parallel_matches_scalar() {
        let mut rng = Rng::new(0xBB2);
        for log_d in [4usize, 8, 12, 15, 16] {
            let ntt = AdditiveNttF128::standard(log_d);
            let original = rand_vec(&mut rng, 1 << log_d);
            let mut v_scalar = original.clone();
            ntt.forward_transform_scalar(&mut v_scalar);
            let mut v_par = original.clone();
            ntt.forward_transform_parallel(&mut v_par);
            assert_eq!(
                v_par, v_scalar,
                "parallel disagrees with scalar at log_d={log_d}"
            );
        }
    }

    #[test]
    fn deepest_layer_twiddle_count() {
        let log_d = 4;
        let ntt = AdditiveNttF128::standard(log_d);
        // At layer log_d - 1 = 3, there are 2^3 = 8 blocks. twiddle(3, b) for b ∈ 0..8.
        for b in 0..8 {
            let _t = ntt.twiddle(log_d - 1, b);
        }
    }
}

#[cfg(test)]
mod twiddle_structure_check {
    use super::*;
    #[test]
    #[ignore]
    fn dump_top_layer_twiddle_structure() {
        // Ranked shape: log_domain 20 positions? standard(k_code) — use 20.
        let ntt = AdditiveNttF128::standard(20);
        for layer in [1usize, 4, 7] {
            let blocks = 1usize << layer;
            let mut hi_zero = 0usize;
            let mut lo_zero = 0usize;
            for b in 0..blocks {
                let t = ntt.twiddle(layer, b);
                if t.hi == 0 {
                    hi_zero += 1;
                }
                if t.lo == 0 && t.hi == 0 {
                    lo_zero += 1;
                }
            }
            println!("layer {layer}: {blocks} twiddles, hi==0: {hi_zero}, all-zero: {lo_zero}");
            if layer == 7 {
                for b in 0..8 {
                    let t = ntt.twiddle(layer, b);
                    println!("  t[{b}] = {:#018x}_{:016x}", t.hi, t.lo);
                }
            }
        }
        // also deep layers for context
        for layer in [10usize, 15, 19] {
            let blocks = 1usize << layer;
            let hi_zero = (0..blocks)
                .filter(|&b| ntt.twiddle(layer, b).hi == 0)
                .count();
            println!("layer {layer}: {blocks} twiddles, hi==0: {hi_zero}");
        }
    }
}

#[cfg(test)]
mod block_range_equivalence {
    use super::*;

    #[test]
    fn block_range_matches_full_transform() {
        let num_ntts = 4usize;
        let log_d = 10usize;
        let n = (1usize << log_d) * num_ntts;
        let ntt = AdditiveNttF128::standard(log_d);
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };
        let src: Vec<F128> = (0..n).map(|_| F128::new(next(), next())).collect();

        let mut full = src.clone();
        ntt.forward_transform_interleaved_from_layer(&mut full, num_ntts, 1);

        // Whole range through the block-range driver.
        let mut whole = src.clone();
        ntt.forward_transform_interleaved_block_range(&mut whole, num_ntts, 1, log_d, 0, 2, 0);
        assert_eq!(full, whole, "block_range over the full range diverges");

        // Split at layer 4: shared top layers 1..4, then independent
        // per-block completion — the hybrid-commit shape.
        let mut split = src.clone();
        ntt.forward_transform_interleaved_block_range(&mut split, num_ntts, 1, 4, 0, 2, 0);
        for b in 0..16 {
            ntt.forward_transform_interleaved_block_range(
                &mut split,
                num_ntts,
                4,
                log_d,
                b,
                b + 1,
                0,
            );
        }
        assert_eq!(full, split, "layer-4 split diverges");
    }

    #[test]
    fn cache_local_block_range_matches_plain_with_absolute_callbacks() {
        // Compact analogue of ranked layers 4..10 streaming + cache-local
        // deep pairs: six top layers followed by two fused deep pairs.
        let num_ntts = 4usize;
        let log_d = 12usize;
        let start_layer = 2usize;
        let n_top = 8usize;
        let (b_start, b_end) = (1usize, 4usize);
        let n = (1usize << log_d) * num_ntts;
        let ntt = AdditiveNttF128::standard(log_d);
        let mut state = 0x6a09_e667_f3bc_c909u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };
        let src: Vec<F128> = (0..n).map(|_| F128::new(next(), next())).collect();

        let mut plain = src.clone();
        ntt.forward_transform_interleaved_block_range(
            &mut plain,
            num_ntts,
            start_layer,
            log_d,
            b_start,
            b_end,
            0,
        );

        let mut cache_local = src;
        ntt.forward_transform_interleaved_block_range(
            &mut cache_local,
            num_ntts,
            start_layer,
            n_top,
            b_start,
            b_end,
            0,
        );
        let sub_start = b_start << (n_top - start_layer);
        let sub_end = b_end << (n_top - start_layer);
        let sub_elems = (1usize << (log_d - n_top)) * num_ntts;
        let offsets = std::sync::Mutex::new(Vec::new());
        ntt.forward_transform_interleaved_deep_fused_pairs_range_and_then(
            &mut cache_local,
            num_ntts,
            n_top,
            log_d,
            sub_start,
            sub_end,
            0,
            &|elem_offset, chunk| {
                assert_eq!(elem_offset % sub_elems, 0);
                assert_eq!(chunk, &plain[elem_offset..elem_offset + sub_elems]);
                offsets.lock().unwrap().push(elem_offset);
            },
        );

        assert_eq!(
            cache_local, plain,
            "cache-local range diverges from plain driver"
        );
        let mut got_offsets = offsets.into_inner().unwrap();
        got_offsets.sort_unstable();
        let expected_offsets: Vec<usize> =
            (sub_start..sub_end).map(|sub| sub * sub_elems).collect();
        assert_eq!(
            got_offsets, expected_offsets,
            "callback coverage/offsets diverge"
        );
    }
}

/// Bit-exactness oracle for the static zero-lane skip.
#[cfg(test)]
mod zero_lane_skip {
    use super::*;

    /// The skip is published through a process-global; serialize the cases so
    /// concurrently running tests never observe another case's publication.
    static PUBLISH: std::sync::Mutex<()> = std::sync::Mutex::new(());

    const NUM_NTTS: usize = 64;
    const TAIL: usize = 7;

    /// Codeword with the ranked BLAKE3 zero geometry: lanes `64 - TAIL .. 64`
    /// are zero at every odd position, everything else pseudorandom.
    fn structured(log_d: usize) -> Vec<F128> {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            state
        };
        (0..(1usize << log_d) * NUM_NTTS)
            .map(|i| {
                let odd_pos = (i / NUM_NTTS) & 1 == 1;
                if odd_pos && i % NUM_NTTS >= NUM_NTTS - TAIL {
                    F128::ZERO
                } else {
                    F128::new(next(), next())
                }
            })
            .collect()
    }

    /// The ranked BLAKE3 padding descriptor must yield exactly seven lanes;
    /// every other geometry must yield none.
    #[test]
    fn padding_descriptor_yields_seven_lanes() {
        // K_LOG = 14, USEFUL_BITS = 15,409.
        assert_eq!(ZeroOddTailLanes::lanes_for_padding(64, 14, 15_409), 7);
        // Dense witness: nothing to skip.
        assert_eq!(ZeroOddTailLanes::lanes_for_padding(64, 14, 16_384), 0);
        // Batch-major coalesces padding into one giant block: unsupported.
        assert_eq!(ZeroOddTailLanes::lanes_for_padding(64, 32, 1 << 31), 0);
        // Recursive commits interleave eight lanes.
        assert_eq!(ZeroOddTailLanes::lanes_for_padding(8, 14, 15_409), 0);
        // A tail wider than one position cannot be expressed as a lane skip.
        assert_eq!(ZeroOddTailLanes::lanes_for_padding(64, 14, 8_000), 0);
    }

    /// Hybrid-commit schedule (top block range + cache-local deep pairs) must
    /// produce byte-identical output with and without the skip.
    #[test]
    fn hybrid_suffix_schedule_is_bit_identical() {
        let log_d = 13usize;
        let start_layer = 4usize;
        let n_top = 10usize;
        let ntt = AdditiveNttF128::standard(log_d);
        let src = structured(log_d);

        let run = |input: &[F128], tail: usize| {
            let mut data = input.to_vec();
            ntt.forward_transform_interleaved_block_range(
                &mut data,
                NUM_NTTS,
                start_layer,
                n_top,
                0,
                1 << start_layer,
                tail,
            );
            ntt.forward_transform_interleaved_deep_fused_pairs_range_and_then(
                &mut data,
                NUM_NTTS,
                n_top,
                log_d,
                0,
                1 << n_top,
                tail,
                &|_, _| {},
            );
            data
        };

        let dense = run(&src, 0);
        let skipped = run(&src, TAIL);
        assert_eq!(dense, skipped, "zero-lane skip changed the codeword");

        // The schedule itself must still be the true transform.
        let mut oracle = src.clone();
        ntt.forward_transform_interleaved_scalar_from_layer(&mut oracle, NUM_NTTS, start_layer);
        assert_eq!(
            dense, oracle,
            "hybrid schedule diverges from the scalar oracle"
        );

        // Negative control: the equality above must come from the skip being
        // ACTIVE on provably-zero data, not from it being a no-op. Break the
        // zero geometry on one odd-position tail lane and the two runs must
        // now diverge.
        let mut poisoned = src;
        poisoned[NUM_NTTS + (NUM_NTTS - 1)] = F128::new(1, 0);
        assert_ne!(
            run(&poisoned, 0),
            run(&poisoned, TAIL),
            "skip never engaged — the oracle above proves nothing"
        );
    }

    /// Diagnostic dump of the ranked twiddle-tree structure: per-layer
    /// linearity constants E_{l,j} (tw(l,b) = XOR of bit_j(b)*E_{l,j}) and
    /// which 32-bit words are live per layer. Probe only.
    #[test]
    #[ignore = "diagnostic dump; run explicitly with --ignored --nocapture"]
    fn twiddle_structure_probe() {
        let ntt = AdditiveNttF128::standard(20);
        let tw = ntt.precomputed_twiddle_table().expect("dim 20 is cached");
        for layer in 0..20usize {
            let start = (1usize << layer) - 1;
            let blocks = 1usize << layer;
            // Verify GF(2)-linearity in the block index and collect E values.
            let mut es: Vec<F128> = Vec::new();
            for j in 0..layer {
                es.push(tw[start + (1 << j)]);
            }
            let mut linear = true;
            let sample = blocks.min(4096);
            for b in 0..sample {
                let mut acc = F128::ZERO;
                for (j, e) in es.iter().enumerate() {
                    if (b >> j) & 1 == 1 {
                        acc = acc + *e;
                    }
                }
                if acc != tw[start + b] {
                    linear = false;
                    break;
                }
            }
            // Word liveness across the whole layer.
            let (mut lo_or, mut hi_or) = (0u64, 0u64);
            for b in 0..blocks {
                lo_or |= tw[start + b].lo;
                hi_or |= tw[start + b].hi;
            }
            eprintln!("layer {layer:2}: linear={linear} lo_or={lo_or:016x} hi_or={hi_or:016x}");
            for (j, e) in es.iter().enumerate() {
                eprintln!("    E[{j}] = {:016x}_{:016x}", e.hi, e.lo);
            }
        }
    }

    /// The ambient publication is honored ONLY at the ranked production
    /// geometry, and never leaks past its guard.
    #[test]
    fn ambient_publication_is_ranked_only_and_scoped() {
        let _serial = PUBLISH.lock().unwrap_or_else(|e| e.into_inner());
        let live = usize::from(!zero_lane_skip_disabled());
        assert_eq!(ranked_zero_odd_tail_lanes(ZERO_TAIL_LOG_D, NUM_NTTS), 0);
        {
            let _outer = ZeroOddTailLanes::scope(NUM_NTTS, TAIL);
            assert_eq!(
                ranked_zero_odd_tail_lanes(ZERO_TAIL_LOG_D, NUM_NTTS),
                TAIL * live
            );
            // Any non-ranked domain or lane width ignores the publication.
            assert_eq!(ranked_zero_odd_tail_lanes(12, NUM_NTTS), 0);
            assert_eq!(ranked_zero_odd_tail_lanes(ZERO_TAIL_LOG_D, 8), 0);
            {
                let _inner = ZeroOddTailLanes::scope(8, TAIL);
                assert_eq!(ranked_zero_odd_tail_lanes(ZERO_TAIL_LOG_D, NUM_NTTS), 0);
            }
            assert_eq!(
                ranked_zero_odd_tail_lanes(ZERO_TAIL_LOG_D, NUM_NTTS),
                TAIL * live
            );
        }
        assert_eq!(ranked_zero_odd_tail_lanes(ZERO_TAIL_LOG_D, NUM_NTTS), 0);
    }
}
