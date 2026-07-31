// Copyright 2024-2025 Irreducible, Inc.
// Modifications copyright 2026 Succinct Labs, Benedikt Bunz, William Wang
// SPDX-License-Identifier: Apache-2.0 OR MIT
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
        let basis: Vec<F128> = (0..dim).map(|i| F128::new(1u64 << i, 0)).collect();
        let evals = generate_evals_from_subspace(&basis);
        let precomputed_twiddles = cached_standard_twiddles(dim, &evals);
        Self {
            evals,
            precomputed_twiddles,
        }
    }

    pub fn log_domain_size(&self) -> usize {
        self.evals.len()
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
                    for block in 0..num_blocks {
                        let tw = block_twiddles(block);
                        let start = block * block_bytes;
                        if block == 0 && zero_root_fused3 {
                            butterfly_interleaved_fused_3layer_par_rows::<true>(
                                &mut data[start..start + block_bytes],
                                &tw,
                                eighth,
                                num_ntts,
                            );
                        } else {
                            butterfly_interleaved_fused_3layer_par_rows::<false>(
                                &mut data[start..start + block_bytes],
                                &tw,
                                eighth,
                                num_ntts,
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
                finish_chunk(sub_idx * sub_bytes, sub_data);
            });
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
        use rayon::prelude::*;

        debug_assert!(n_top <= log_d);
        debug_assert_eq!(data.len(), (1usize << log_d) * num_ntts);
        let sub_size_positions = 1usize << (log_d - n_top);
        let sub_elems = sub_size_positions * num_ntts;

        data.par_chunks_mut(sub_elems)
            .enumerate()
            .for_each(|(sub_idx, sub_data)| {
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
                        butterfly_interleaved_fused_2layer_rows_seq(
                            &mut sub_data[block_start..block_start + block_elems],
                            t_outer,
                            t_inner_a,
                            t_inner_b,
                            quarter,
                            num_ntts,
                        );
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
fn butterfly_interleaved_fused_2layer_rows_seq(
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

    let (top_half, bot_half) = block.split_at_mut(2 * stride);
    let (q1, q2) = top_half.split_at_mut(stride);
    let (q3, q4) = bot_half.split_at_mut(stride);
    for (((row_a, row_b), row_c), row_d) in q1
        .chunks_exact_mut(num_ntts)
        .zip(q2.chunks_exact_mut(num_ntts))
        .zip(q3.chunks_exact_mut(num_ntts))
        .zip(q4.chunks_exact_mut(num_ntts))
    {
        kernels::butterfly_fused_2layer(row_a, row_b, row_c, row_d, t_outer, t_inner_a, t_inner_b);
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
) {
    use rayon::prelude::*;
    const PARALLEL_ROW_THRESHOLD: usize = 256;
    debug_assert_eq!(block.len(), 8 * eighth * num_ntts);
    if ZERO_ROOT {
        debug_assert_eq!(t[0], F128::ZERO);
        debug_assert_eq!(t[1], F128::ZERO);
        debug_assert_eq!(t[3], F128::ZERO);
    }
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
                        r,
                        t,
                    )
                } else {
                    kernels::butterfly_fused_3layer_row(base as *mut F128, eighth, num_ntts, r, t)
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
                        r,
                        t,
                    )
                } else {
                    kernels::butterfly_fused_3layer_row(base as *mut F128, eighth, num_ntts, r, t)
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
                    row,
                    &twiddles[block],
                )
            } else {
                kernels::butterfly_fused_3layer_row(
                    block_base,
                    eighth,
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
        for row in row_start..row_end {
            // SAFETY: each queue index maps to one disjoint `(block, row)`
            // tile. Rows within a tile are serial and all derived addresses
            // are in the validated block range.
            unsafe {
                if block == 0 {
                    kernels::butterfly_fused_3layer_zero_root_row(
                        block_base,
                        eighth,
                        num_ntts,
                        row,
                        &twiddles[block],
                    )
                } else {
                    kernels::butterfly_fused_3layer_row(
                        block_base,
                        eighth,
                        num_ntts,
                        row,
                        &twiddles[block],
                    )
                }
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
        for (quarter, num_ntts) in [(1usize, 64usize), (4, 8), (64, 2), (256, 2)] {
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
                    &mut got, t_outer, t_inner_a, t_inner_b, quarter, num_ntts,
                );
                assert_eq!(
                    got, want,
                    "fused-2 sequential mismatch at quarter={quarter} \
                     num_ntts={num_ntts} iteration={iteration}"
                );
            }
        }
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
                            &mut got, &twiddles, eighth, NUM_NTTS,
                        );
                        // The omitted stream-zero stores are valid only because
                        // this whole row stream is mathematically unchanged.
                        let stream_len = eighth * NUM_NTTS;
                        assert_eq!(&got[..stream_len], &source[..stream_len]);
                    } else {
                        butterfly_interleaved_fused_3layer_par_rows::<false>(
                            &mut got, &twiddles, eighth, NUM_NTTS,
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
                        block_data, tw, eighth, NUM_NTTS,
                    );
                } else {
                    butterfly_interleaved_fused_3layer_par_rows::<false>(
                        block_data, tw, eighth, NUM_NTTS,
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
                    );
                } else {
                    butterfly_interleaved_fused_3layer_par_rows::<false>(
                        &mut got[start..start + block_elems],
                        &tw,
                        eighth,
                        NUM_NTTS,
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
