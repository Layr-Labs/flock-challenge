//! PCS commit phase: pack → RS encode (additive NTT) → Merkle root.
//!
//! Uses [`AdditiveNttF128`], the binius-style LCH NTT with neighbors-last
//! pairing. The commit produces a non-systematic RS codeword (treating the
//! packed witness as novel-basis coefficients, zero-padded to the larger
//! domain, then forward-NTT'd).
//!
//! ## Layout
//!
//! With parameters `(m, log_inv_rate)`:
//! - `log_msg_len = m − LOG_PACKING` (= log2 of packed witness length)
//! - `k_code      = log_msg_len + log_inv_rate` (= log2 of codeword length)
//!
//! The codeword is a flat sequence of `2^k_code` F_{2^128} elements. Each
//! Merkle leaf is **one** F_{2^128} element = 16 bytes.

use crate::field::F128;
use crate::merkle::{self, Hash, HashKind};
use crate::ntt::AdditiveNttF128;
use crate::pcs::pack::LOG_PACKING;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// PCS configuration. Polynomial-basis subspace `{1, x, x², …}` for the NTT.
///
/// Interleaved RS: the packed witness is split into `2^log_batch_size`
/// independent sub-NTTs of size `2^log_dim` each. Each Merkle leaf holds one
/// codeword position across all `2^log_batch_size` lanes
/// (`2^log_batch_size · 16` bytes per leaf). This trades leaf-call SHA-256
/// overhead (was 16 B leaves, now 512 B leaves at default `log_batch_size=5`)
/// for much fewer Merkle nodes and better scaling to large `m`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PcsParams {
    pub m: usize,
    pub log_inv_rate: usize,
    /// Number of parallel sub-NTTs = `2^log_batch_size`. Default 5 (= 32 lanes).
    pub log_batch_size: usize,
    /// Ligerito parameter profile (fast/slim/secure). Selects which embedded
    /// security config (queries, OOD samples, grinding schedule) drives the
    /// PCS opening; must agree with `log_inv_rate`
    /// (`profile.log_inv_rate() == log_inv_rate`). Defaults to `Fast`.
    #[serde(default)]
    pub profile: crate::pcs::ligerito::LigeritoProfile,
    /// Hash backing the Merkle commitment. Defaults to SHA-256, so params
    /// serialized before this option existed deserialize unchanged.
    ///
    /// The verifier must be given the same value the prover committed under —
    /// it is carried in [`Commitment`] alongside the root for exactly that
    /// reason.
    #[serde(default)]
    pub merkle_hash: HashKind,
}

impl PcsParams {
    /// Total log message length (= log2 packed witness length).
    pub fn log_msg_len(&self) -> usize {
        self.m - LOG_PACKING
    }
    /// Per-sub-NTT log dimension (= number of "position" coords).
    pub fn log_dim(&self) -> usize {
        self.log_msg_len() - self.log_batch_size
    }
    /// Codeword size (log) per sub-NTT.
    pub fn k_code(&self) -> usize {
        self.log_dim() + self.log_inv_rate
    }
    /// Number of Merkle leaves (= per-sub-NTT codeword length).
    pub fn n_positions(&self) -> usize {
        1usize << self.k_code()
    }
    /// `num_ntts` = `2^log_batch_size`.
    pub fn num_ntts(&self) -> usize {
        1usize << self.log_batch_size
    }
    /// Total codeword length in F_{2^128} elements
    /// (= `n_positions() * num_ntts()`).
    pub fn codeword_len_f128(&self) -> usize {
        self.n_positions() * self.num_ntts()
    }
    /// `log_2` of the F_{2^128} count per **initial** Merkle leaf
    /// (= `log_batch_size`; just the row-batch lanes per position).
    pub fn log_leaf_f128_count(&self) -> usize {
        self.log_batch_size
    }
    /// Number of initial-tree Merkle leaves
    /// (= `codeword_len_f128() / 2^log_batch_size = 2^k_code`).
    pub fn n_leaves(&self) -> usize {
        self.codeword_len_f128() >> self.log_leaf_f128_count()
    }
    /// Merkle leaf size in bytes = `num_ntts() * 16`.
    pub fn leaf_size_bytes(&self) -> usize {
        16usize << self.log_leaf_f128_count()
    }

    /// Ligerito prover config for these params.
    ///
    /// Prefer this over calling [`ligerito::prover_config_for`] directly: the
    /// embedded security config carries its own `hash` field, but the Merkle
    /// hash the opening must use is the one the *commitment* was built under.
    /// This stamps `self.merkle_hash` over it, so the L0 tree and every
    /// recursive level cannot end up on different hashes.
    ///
    /// [`ligerito::prover_config_for`]: crate::pcs::ligerito::prover_config_for
    pub fn ligerito_prover_config(&self) -> Result<crate::pcs::ligerito::ProverConfig, String> {
        let mut cfg = crate::pcs::ligerito::prover_config_for(
            self.log_msg_len(),
            self.log_batch_size,
            self.profile,
        )?;
        cfg.merkle_hash = self.merkle_hash;
        Ok(cfg)
    }

    /// Verifier-side counterpart to [`Self::ligerito_prover_config`], stamped
    /// with the same Merkle hash for the same reason.
    pub fn ligerito_verifier_config(&self) -> Result<crate::pcs::ligerito::VerifierConfig, String> {
        let mut cfg = crate::pcs::ligerito::verifier_config_for(
            self.log_msg_len(),
            self.log_batch_size,
            self.profile,
        )?;
        cfg.merkle_hash = self.merkle_hash;
        Ok(cfg)
    }

    fn validate(&self) {
        assert!(
            self.m >= LOG_PACKING + self.log_batch_size,
            "m={} too small (need m ≥ LOG_PACKING + log_batch_size = {})",
            self.m,
            LOG_PACKING + self.log_batch_size,
        );
        assert!(
            self.log_inv_rate >= 1,
            "log_inv_rate must be ≥ 1 for a non-trivial RS code",
        );
    }
}

/// Public commitment (Merkle root + params).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Commitment {
    pub root: Hash,
    pub params: PcsParams,
}

/// Prover-side state retained after commit for use in the opening phase.
///
/// **The packed witness is NOT stored here.** The caller is responsible for
/// retaining its own copy of the packed witness across commit + open. This
/// avoids ~4 GB of duplication at large `m`, dropping peak commit memory by
/// a factor of ~1.5 (e.g. at m=35: 13 GB → 9 GB).
pub struct ProverData {
    pub codeword: Vec<F128>,
    pub merkle_tree: Vec<Hash>,
}

// Recycle the codeword buffer (the prover's largest single allocation —
// 128 MB at m = 29) through the scratch pool instead of unmapping it, and
// the Merkle tree buffer through the commit-local tree pool (the GPU
// wrap-cache keys MTLBuffer wraps by base pointer — see [`TREE_POOL`]).
impl Drop for ProverData {
    fn drop(&mut self) {
        crate::scratch::give_f128(std::mem::take(&mut self.codeword));
        give_tree(std::mem::take(&mut self.merkle_tree));
    }
}

/// Commit to a witness in **F_{2^128}-packed** form (polynomial basis: bit
/// `r` of `z_packed[i]` = logical bit `i·128 + r`).
///
/// Uses **interleaved RS encoding**: `num_ntts = 2^log_batch_size` independent
/// sub-NTTs share the same domain and twiddles, processed via the SoA
/// interleaved transform. The codeword is stored position-major SoA
/// (`codeword[pos · num_ntts + lane]`); each Merkle leaf is one position =
/// `num_ntts` F_{2^128} = `num_ntts · 16` bytes.
///
/// **Takes the witness by reference**. The returned [`ProverData`] does NOT
/// retain a copy of the packed witness — the caller is responsible for
/// keeping its own copy across commit + open. This frees ~4 GB during the
/// NTT/Merkle phase at large `m`.
///
/// `z_packed.len()` must equal `2^(m - LOG_PACKING) = 2^(m - 7)`.
pub fn commit(z_packed: &[F128], params: &PcsParams) -> (Commitment, ProverData) {
    params.validate();
    assert_eq!(z_packed.len(), 1usize << params.log_msg_len());

    let num_ntts = params.num_ntts();
    let n_positions = params.n_positions();
    let codeword_len = n_positions * num_ntts;

    // ---- Codeword buffer (SoA): codeword[pos * num_ntts + lane].
    // The semantic RS encoder overwrites every slot, so take a stale resident
    // scratch buffer without zeroing it. This is 1 GiB at the m=32 benchmark;
    // avoiding an eager initialization pass is material.
    let codeword = crate::scratch::take_f128(codeword_len);
    commit_into(z_packed, params, codeword)
}

/// Like [`commit`], but reuses a caller-provided codeword buffer instead of
/// allocating its own. The buffer must have length `codeword_len`; its
/// CONTENTS may be arbitrary (uninit/stale) — every slot is written by the RS
/// encoder. Buffers from [`prefault_codeword_during`] or the scratch pool are
/// already resident, so no write faults.
pub fn commit_into(
    z_packed: &[F128],
    params: &PcsParams,
    codeword: Vec<F128>,
) -> (Commitment, ProverData) {
    params.validate();
    assert_eq!(z_packed.len(), 1usize << params.log_msg_len());
    let codeword_len = params.n_positions() * params.num_ntts();
    assert_eq!(
        codeword.len(),
        codeword_len,
        "commit_into: prebuilt codeword buffer has wrong length"
    );

    finalize_commit(codeword, z_packed, params)
}

/// Widest-available rayon pool for hash-throughput-bound bulk hashing (all
/// logical cores, including efficiency cores that otherwise idle).
/// Built lazily — the ranked worker's untimed warm-up prove pays the spawn.
/// Also borrowed by the open phase's `b_combined` build (`pcs.rs`), which is
/// the one open section that scales with E-core issue capacity.
pub(crate) fn wide_hash_pool() -> &'static rayon::ThreadPool {
    use std::sync::LazyLock;
    static POOL: LazyLock<rayon::ThreadPool> = LazyLock::new(|| {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("wide hash pool")
    });
    &POOL
}

/// Shared tail of [`commit`] / [`commit_into`]: interleaved forward additive
/// NTT (RS-encode every lane) then the initial Merkle tree over codeword rows.
#[allow(
    clippy::collapsible_if,
    clippy::manual_is_multiple_of,
    clippy::manual_slice_size_calculation
)] // Preserve the ranked CPU control-flow and arithmetic source shape.
fn finalize_commit(
    mut codeword: Vec<F128>,
    z_packed: &[F128],
    params: &PcsParams,
) -> (Commitment, ProverData) {
    let timing = std::env::var_os("FLOCK_COMMIT_TIMING").is_some();
    let ntt = AdditiveNttF128::standard(params.k_code());

    // ---- GPU-hybrid streaming commit (BLAKE3 Merkle on Metal, leaves hashed
    // in ordered 1/8th-codeword chunks as the NTT deep pass completes them).
    // Production gates: BLAKE3 trees of ≥ 2^18 leaves with a batchable leaf
    // size, GPU Merkle pipeline available (Metal present, not latched off,
    // FLOCK_NO_GPU / FLOCK_NO_GPU_MERKLE unset). Everything else — and any
    // in-flight GPU failure — lands on the existing pure-CPU path below.
    if params.merkle_hash == HashKind::Blake3
        && params.n_leaves() >= (1 << 18)
        && merkle::blake3_leaf_size_is_batchable(params.leaf_size_bytes())
        && crate::gpu::merkle::available()
    {
        if let Some((merkle_tree, root)) = gpu_streamed_commit(
            &ntt,
            &mut codeword,
            z_packed,
            params,
            N_STREAM_CHUNKS,
            GPU_STOP_NODES,
            timing,
        ) {
            return (
                Commitment {
                    root,
                    params: params.clone(),
                },
                ProverData {
                    codeword,
                    merkle_tree,
                },
            );
        }
    }

    // ---- Merkle rewrite 1: stream leaf hashes into the same 2n−1 tree as
    // each NTT deep-pass sub-group retires (regular stores, same worker).
    // Parents fold after the encode join via [`build_upper_levels`] — the
    // same `hash_pairs_level` sequence `merkle_tree()` uses. No
    // `wide_hash_pool` (same-width hop / nested-install footgun). No Metal.
    let n_leaves = params.n_leaves();
    let kind = params.merkle_hash;
    let leaf_size = params.leaf_size_bytes();
    let num_ntts = params.num_ntts();
    // Pooled like every other prove-cycle buffer: `ProverData::drop` returns
    // the tree to TREE_POOL, so on all proves after the first the 64 MiB
    // (ranked shape) allocation is already resident — no mmap/fault-in here
    // and no munmap/TLB-shootdown at drop. Same write-before-read contract
    // as the uninit alloc this replaces.
    let mut merkle_tree: Vec<Hash> = take_tree(2 * n_leaves - 1);
    let tree_addr = merkle_tree.as_mut_ptr() as usize;

    // Subtree parents while hot: after a deep-pass sub-group's leaves are
    // hashed (still on this worker, still L2-resident), fold that sub-group's
    // own Merkle subtree — every level whose nodes depend only on this
    // sub-group's leaves — into the same flat tree. Sub-groups are disjoint,
    // power-of-two sized and aligned at the ranked shape (512 × 2048 leaves),
    // so each subtree is self-contained through `log2(len)` levels and its
    // node ranges at every level are pairwise disjoint across workers. The
    // post-join `build_upper_levels` then starts from the sub-group roots
    // instead of re-reading the whole 32 MiB leaf level (and 16+8+… MiB of
    // parents) cold, level by level, with a rayon barrier per level.
    // `local_levels` records the fold depth every sub-group achieved
    // (usize::MAX = unset); any sub-group that cannot fold (unaligned, not a
    // power of two, mismatched depth) sets it to 0 and the incumbent full
    // upper-level build runs — those levels are simply rewritten, so a partial
    // local fold is never wrong, only wasted.
    let subtree_parents = subtree_parents_enabled();
    let regroup_subtree_parents =
        subtree_parent_regroup_enabled(n_leaves, num_ntts, leaf_size, kind);
    let local_levels = AtomicUsize::new(usize::MAX);

    let t_ntt = std::time::Instant::now();
    ntt.rs_encode_interleaved_on_range_done(
        z_packed,
        &mut codeword,
        num_ntts,
        &|range, sub_data| {
            debug_assert_eq!(sub_data.len(), range.len() * num_ntts);
            // Zero-copy: F128 is repr(C, align(16)) lo||hi LE — same bytes as
            // the one-shot `merkle_tree` cast in the previous barrier path.
            let bytes: &[u8] = unsafe {
                core::slice::from_raw_parts(
                    sub_data.as_ptr() as *const u8,
                    sub_data.len() * core::mem::size_of::<F128>(),
                )
            };
            // SAFETY: each deep-pass sub-group maps to a disjoint leaf-index
            // range; only this worker writes `tree[range]`, and NTT writes to
            // those leaves have retired on this thread.
            let out = unsafe {
                core::slice::from_raw_parts_mut(
                    (tree_addr as *mut Hash).add(range.start),
                    range.len(),
                )
            };
            merkle::hash_leaves_serial(bytes, leaf_size, out, kind);

            if !subtree_parents {
                return;
            }
            // The deep block-fused schedule retires sixteen 128-leaf blocks
            // per 2048-leaf subgroup. Keep hashing each block's leaves at
            // retirement for producer/consumer overlap, but fold parents
            // only when the subgroup's final FIFO block arrives. At that
            // point all 2048 leaf CVs are ready, so one depth-11 fold fills
            // the BLAKE3 SIMD lanes that sixteen depth-7 folds leave sparse.
            let Some(parent_range) = local_parent_fold_range(&range, regroup_subtree_parents)
            else {
                return;
            };
            let len = parent_range.len();
            let depth = if len.is_power_of_two()
                && len >= 2
                && parent_range.start % len == 0
                && n_leaves % len == 0
            {
                len.trailing_zeros() as usize
            } else {
                0
            };
            // Publish this sub-group's depth; every sub-group must agree.
            let seen = match local_levels.compare_exchange(
                usize::MAX,
                depth,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => depth,
                Err(prev) => prev,
            };
            if seen != depth {
                local_levels.store(0, Ordering::Release);
            }
            if depth == 0 || seen == 0 {
                return;
            }
            // Level j (j ≥ 1) has n_leaves >> j nodes and starts at flat
            // offset 2·n_leaves − 2·(n_leaves >> j); this sub-group owns
            // nodes [start >> j, (start + len) >> j) of it.
            let mut lvl_read_off = parent_range.start; // level 0 = leaves
            let mut lvl_read_len = len;
            for j in 1..=depth {
                let nodes_j = n_leaves >> j;
                let base_j = 2 * n_leaves - 2 * nodes_j;
                let write_off = base_j + (parent_range.start >> j);
                let write_len = len >> j;
                // SAFETY: the read range is this worker's own just-written
                // nodes of level j−1 (leaves for j = 1); the write range is
                // this worker's own disjoint slice of level j. Both are inside
                // the 2·n_leaves−1 tree and never alias each other.
                let (read, write) = unsafe {
                    (
                        core::slice::from_raw_parts(
                            (tree_addr as *const Hash).add(lvl_read_off),
                            lvl_read_len,
                        ),
                        core::slice::from_raw_parts_mut(
                            (tree_addr as *mut Hash).add(write_off),
                            write_len,
                        ),
                    )
                };
                merkle::hash_pairs_level_serial(read, write, kind);
                lvl_read_off = write_off;
                lvl_read_len = write_len;
            }
        },
    );
    if timing {
        eprintln!(
            "[commit-timing] ntt: {:.2} ms",
            t_ntt.elapsed().as_secs_f64() * 1e3
        );
    }
    let t_merkle = std::time::Instant::now();
    let folded = match local_levels.load(Ordering::Acquire) {
        usize::MAX | 0 => 0,
        d => d,
    };
    build_upper_levels(&mut merkle_tree, n_leaves, n_leaves >> folded, kind);
    let root = merkle_tree[2 * n_leaves - 2];
    if timing {
        eprintln!(
            "[commit-timing] merkle: {:.2} ms (subtree levels folded in-callback: {})",
            t_merkle.elapsed().as_secs_f64() * 1e3,
            folded
        );
    }

    (
        Commitment {
            root,
            params: params.clone(),
        },
        ProverData {
            codeword,
            merkle_tree,
        },
    )
}

// ---------------------------------------------------------------------------
// GPU-hybrid streaming commit (Metal BLAKE3 Merkle).
//
// The season-1 GPU URM offload lost to its fixed costs (~9 ms/prove of buffer
// wiring + clock ramp) because it fired once per prove. Season-2 shares one
// warm/wired GPU cycle between TWO phases: (A) Merkle leaf+lower-tree hashing
// streamed here, chunk by chunk, while the NTT deep pass is still running,
// then (B) the zerocheck round-1 URM split ~100 ms later. The Merkle stream
// doubles as the clock keepalive for (B): command buffers land every ~5-10 ms
// through the whole commit, so no dedicated keepalive thread exists this
// season (see `flock-prover`'s prove entry).
// ---------------------------------------------------------------------------

/// Ordered NTT super-chunks per commit = GPU leaf command buffers per commit.
/// At the ranked m=32 geometry (1 GiB codeword) each chunk is 128 MiB — well
/// above the ≥64 MiB per-cb floor (0.2 ms fixed cost per cb).
const N_STREAM_CHUNKS: usize = 8;

/// The GPU builds parent levels while their node count is ≥ this; the CPU
/// hashes the remaining top (2^14 → root is < 2^14 pair hashes — sub-ms).
const GPU_STOP_NODES: usize = 1 << 14;

/// Tree over-allocation, in nodes: one 16 KiB page (512 × 32 B) so
/// `gpu::merkle::begin`'s page-coverage check always passes for the real
/// node range (see the allocation site in [`gpu_streamed_commit`]).
const TREE_PAD_NODES: usize = 512;

/// How many of the [`N_STREAM_CHUNKS`] leaf chunks the GPU takes; the rest
/// are left for the post-NTT CPU join. Default: all (the Merkle kernels
/// outpace the deep-pass cadence by ~5×; the CPU join is the degraded mode).
/// Refined per prove by [`note_merkle_calibration`]; the warmup calibration
/// in `flock-prover` seeds the CPU/GPU rates below.
static MERKLE_GPU_CHUNK_SHARE: AtomicUsize = AtomicUsize::new(N_STREAM_CHUNKS);
/// Measured CPU leaf-hash rate, ns per MiB of leaf data (0 = unknown).
static MERKLE_CPU_NS_PER_MIB: AtomicU64 = AtomicU64::new(0);
/// Measured GPU full-turnaround leaf rate, ns per MiB (0 = unknown).
static MERKLE_GPU_NS_PER_MIB: AtomicU64 = AtomicU64::new(0);

/// Season-1 latch rule, Merkle edition: a GPU slower than 6× the CPU on the
/// same leaves is a broken/contended GPU — turn the pipeline off for the
/// process lifetime rather than re-probing every prove.
const MERKLE_LATCH_FACTOR: u64 = 6;

/// Test/diagnostic escape hatches around the calibration atomics.
pub fn gpu_merkle_split() -> usize {
    MERKLE_GPU_CHUNK_SHARE.load(Ordering::Relaxed)
}
pub fn set_gpu_merkle_split(chunks: usize) {
    MERKLE_GPU_CHUNK_SHARE.store(chunks.min(N_STREAM_CHUNKS), Ordering::Relaxed);
}

/// Pool of page-stable Merkle tree buffers. The GPU wrap-cache keys MTLBuffer
/// wraps by base pointer, so the tree allocation (64 MiB at the ranked shape,
/// ≥ the 64 MiB wrap-cache floor) must survive across proves — dropping and
/// re-allocating it would churn a fresh wire+wrap every prove (season-1 rule).
static TREE_POOL: Mutex<Vec<Vec<Hash>>> = Mutex::new(Vec::new());

pub(crate) fn take_tree(total_nodes: usize) -> Vec<Hash> {
    if let Ok(mut pool) = TREE_POOL.lock() {
        // Smallest-fit: an L1 (16 MiB) take must not steal the parked L0
        // (64 MiB) tree. First-fit would force a fresh mmap of the 64 MiB
        // buffer on the next commit — the exact cost this pool exists to
        // delete.
        let mut best: Option<usize> = None;
        for (i, v) in pool.iter().enumerate() {
            if v.capacity() >= total_nodes && best.is_none_or(|b| v.capacity() < pool[b].capacity())
            {
                best = Some(i);
            }
        }
        if let Some(idx) = best {
            let mut v = pool.swap_remove(idx);
            // SAFETY: capacity checked; Hash is plain bytes, every node is
            // written before it is read (same contract as merkle_tree()).
            unsafe { v.set_len(total_nodes) };
            return v;
        }
    }
    crate::alloc_uninit_vec(total_nodes)
}

pub(crate) fn give_tree(mut tree: Vec<Hash>) {
    // Only park allocations big enough to matter for wrap-cache stability.
    // Floor at 2^16 nodes (~4 MiB) so the ranked L2 Ligerito tree (2^16
    // leaves, 131071 nodes) is recycled alongside L0 (64 MiB) and L1 (16 MiB).
    if tree.capacity() < (1 << 16) {
        return;
    }
    tree.clear();
    if let Ok(mut pool) = TREE_POOL.lock() {
        // Cap 3: ranked L0 (64 MiB) + L1 (16 MiB) + L2 (4 MiB) all park
        // after untimed warmup; the timed prove takes all three without
        // a fresh mmap/fault. Trees are sequential (L_i dropped before
        // L_{i+1} is committed), so at most three are ever parked at once.
        if pool.len() < 3 {
            pool.push(tree);
        }
    }
}

/// Record a full prove's Merkle turnaround and refine the CPU/GPU chunk
/// split. All rates are FULL turnaround (wall time around commit/wait), not
/// GPU timestamps — season-1's calibration rule.
#[allow(clippy::too_many_arguments)]
fn note_merkle_calibration(
    gpu_chunks: usize,
    cpu_chunks: usize,
    chunk_mib: f64,
    gpu_busy_seconds: f64,
    wait_seconds: f64,
    ntt_seconds: f64,
    cpu_join_seconds: f64,
    n_chunks: usize,
) {
    let to_ns_per_mib = |seconds: f64, chunks: usize| -> u64 {
        if chunks == 0 || chunk_mib <= 0.0 {
            return 0;
        }
        (seconds * 1e9 / (chunks as f64 * chunk_mib)) as u64
    };
    if cpu_chunks > 0 {
        let ns = to_ns_per_mib(cpu_join_seconds, cpu_chunks);
        if ns > 0 {
            MERKLE_CPU_NS_PER_MIB.store(ns, Ordering::Relaxed);
        }
    }
    // GPU rate: busy time per chunk, but never better than what the wall
    // clock proves (wait after the CPU was already done).
    if gpu_chunks > 0 {
        let busy = to_ns_per_mib(gpu_busy_seconds, gpu_chunks);
        let walled = to_ns_per_mib(wait_seconds, gpu_chunks);
        let ns = busy.max(walled);
        if ns > 0 {
            MERKLE_GPU_NS_PER_MIB.store(ns, Ordering::Relaxed);
        }
    }
    let t_c = MERKLE_CPU_NS_PER_MIB.load(Ordering::Relaxed);
    let t_g = MERKLE_GPU_NS_PER_MIB.load(Ordering::Relaxed);
    if t_c == 0 || t_g == 0 {
        return; // not enough data to re-split
    }
    if t_g > MERKLE_LATCH_FACTOR * t_c {
        crate::gpu::gpu_dbg_trace(&format!(
            "merkle calib: LATCH OFF (gpu {t_g} ns/MiB > {MERKLE_LATCH_FACTOR}x cpu {t_c})"
        ));
        crate::gpu::merkle::set_enabled(false);
        return;
    }
    // Split so both sides finish together: the GPU works during the NTT
    // (span ntt_seconds) and after it; the CPU join only starts after the
    // NTT. g·t_g − ntt ≈ (n − g)·t_c  ⇒  g = (ntt + n·t_c) / (t_g + t_c).
    let t_g_chunk = t_g as f64 * 1e-9 * chunk_mib;
    let t_c_chunk = t_c as f64 * 1e-9 * chunk_mib;
    let g = ((ntt_seconds + n_chunks as f64 * t_c_chunk) / (t_g_chunk + t_c_chunk))
        .floor()
        .clamp(0.0, n_chunks as f64) as usize;
    MERKLE_GPU_CHUNK_SHARE.store(g.min(N_STREAM_CHUNKS), Ordering::Relaxed);
    crate::gpu::gpu_dbg_trace(&format!(
        "merkle calib: t_c={t_c} t_g={t_g} ns/MiB ntt={:.1}ms -> share {g}/{n_chunks}",
        ntt_seconds * 1e3
    ));
}

/// Hash the CPU-owned leaf chunks after the NTT: wide pool, atomic chunk
/// cursor. The GPU and this join write **disjoint** `tree[..n_leaves]`
/// sub-slices — chunk ownership was fixed when each chunk either was or was
/// not committed to the GPU during the deep pass.
fn cpu_join_hash_leaves(
    codeword_bytes: &[u8],
    leaf_size: usize,
    leaves_out: &mut [Hash],
    ranges: &[core::ops::Range<usize>],
    kind: HashKind,
) {
    let cursor = AtomicUsize::new(0);
    let base = leaves_out.as_mut_ptr() as usize;
    wide_hash_pool().broadcast(|_| {
        loop {
            let i = cursor.fetch_add(1, Ordering::Relaxed);
            let Some(r) = ranges.get(i) else { break };
            // SAFETY: the claimed chunk index is unique (fetch_add), ranges
            // are disjoint sub-ranges of `leaves_out`, and no GPU-owned chunk
            // appears in `ranges` — so this mutable view aliases nothing.
            let out = unsafe {
                core::slice::from_raw_parts_mut((base as *mut Hash).add(r.start), r.len())
            };
            merkle::hash_leaves(
                &codeword_bytes[r.start * leaf_size..r.end * leaf_size],
                leaf_size,
                out,
                kind,
            );
        }
    });
}

/// `FLOCK_NO_MERKLE_SUBTREE_PARENTS=1` disables the in-callback subtree fold
/// (exact A/B control: the full upper-level build then runs as before).
/// Resolved once per process.
pub(crate) fn subtree_parents_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_MERKLE_SUBTREE_PARENTS").is_none());
    *ON
}

const RANKED_PARENT_BLOCK_LEAVES: usize = 128;
const RANKED_PARENT_SUBGROUP_LEAVES: usize = 2048;

/// The promoted L0 deep pass finalizes sixteen adjacent 128-leaf blocks for
/// every 2048-leaf subgroup. Regroup only that exact production shape; every
/// recursive, portable, alternate-hash, and diagnostic geometry keeps the
/// established callback-local fold.
#[inline]
fn subtree_parent_regroup_selected(
    n_leaves: usize,
    num_ntts: usize,
    leaf_size: usize,
    kind: HashKind,
    disabled: bool,
) -> bool {
    !disabled
        && n_leaves == (1 << 20)
        && num_ntts == 64
        && leaf_size == 1024
        && kind == HashKind::Blake3
}

fn subtree_parent_regroup_enabled(
    n_leaves: usize,
    num_ntts: usize,
    leaf_size: usize,
    kind: HashKind,
) -> bool {
    static DISABLED: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_MERKLE_SUBTREE_REGROUP").is_some());
    subtree_parent_regroup_selected(n_leaves, num_ntts, leaf_size, kind, *DISABLED)
}

/// Select the leaf range whose local parents may now be folded.
///
/// `None` means this is an intermediate block: its leaf CVs are complete, but
/// parent folding waits for the subgroup's final FIFO block. Any unexpected
/// callback geometry retains the incumbent per-range behavior.
#[inline]
fn local_parent_fold_range(
    range: &core::ops::Range<usize>,
    regroup: bool,
) -> Option<core::ops::Range<usize>> {
    if !regroup
        || range.len() != RANKED_PARENT_BLOCK_LEAVES
        || !range.start.is_multiple_of(RANKED_PARENT_BLOCK_LEAVES)
    {
        return Some(range.clone());
    }
    if !range.end.is_multiple_of(RANKED_PARENT_SUBGROUP_LEAVES) {
        return None;
    }
    Some(range.end - RANKED_PARENT_SUBGROUP_LEAVES..range.end)
}

/// `FLOCK_NO_LIG_FUSED_COMMIT=1` restores the Ligerito recursive commits'
/// incumbent shape (full `rs_encode_interleaved`, then a separate parallel
/// `fill_merkle_tree` pass over the codeword). Resolved once per process.
pub(crate) fn lig_fused_commit_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_LIG_FUSED_COMMIT").is_none());
    *ON
}

/// Fused encode + leaf hash + local subtree parents for the Ligerito
/// recursive commits — the same shape the L0 commit above uses, factored for
/// the recursive levels (whose leaf is one SoA position = `num_ntts` F128).
///
/// The NTT's deep pass fires `on_range_done` once per finalized sub-group on
/// the worker that finished it; that worker hashes the sub-group's leaves
/// while the codeword slice is still cache-resident and folds the sub-group's
/// own Merkle subtree (every level whose nodes depend only on those leaves)
/// into the flat tree. Returns the folded depth (0 = nothing folded); the
/// caller finishes with `build_upper_levels(tree, n_leaves, n_leaves >>
/// depth, kind)`. Bit-identical to encode-then-`fill_merkle_tree`: same
/// leaves, same pair hashes, only the pass structure changes.
#[allow(
    clippy::manual_is_multiple_of,
    clippy::manual_slice_size_calculation,
    clippy::too_many_arguments
)]
pub(crate) fn fused_encode_leaves_subtree(
    ntt: &AdditiveNttF128,
    msg: &[F128],
    codeword: &mut [F128],
    num_ntts: usize,
    tree: &mut [Hash],
    n_leaves: usize,
    leaf_size: usize,
    kind: HashKind,
) -> usize {
    debug_assert_eq!(codeword.len(), n_leaves * num_ntts);
    debug_assert_eq!(leaf_size, num_ntts * core::mem::size_of::<F128>());
    debug_assert_eq!(tree.len(), 2 * n_leaves - 1);
    let tree_addr = tree.as_mut_ptr() as usize;
    let subtree_parents = subtree_parents_enabled();
    let local_levels = AtomicUsize::new(usize::MAX);
    ntt.rs_encode_interleaved_on_range_done(msg, codeword, num_ntts, &|range, sub_data| {
        debug_assert_eq!(sub_data.len(), range.len() * num_ntts);
        let bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(
                sub_data.as_ptr() as *const u8,
                sub_data.len() * core::mem::size_of::<F128>(),
            )
        };
        // SAFETY: each finalized sub-group maps to a disjoint leaf-index
        // range; only this worker writes `tree[range]`, and the NTT writes to
        // those leaves have retired on this thread.
        let out = unsafe {
            core::slice::from_raw_parts_mut((tree_addr as *mut Hash).add(range.start), range.len())
        };
        merkle::hash_leaves_serial(bytes, leaf_size, out, kind);
        if !subtree_parents {
            return;
        }
        let len = range.len();
        let depth =
            if len.is_power_of_two() && len >= 2 && range.start % len == 0 && n_leaves % len == 0 {
                len.trailing_zeros() as usize
            } else {
                0
            };
        let seen = match local_levels.compare_exchange(
            usize::MAX,
            depth,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => depth,
            Err(prev) => prev,
        };
        if seen != depth {
            local_levels.store(0, Ordering::Release);
        }
        if depth == 0 || seen == 0 {
            return;
        }
        let mut lvl_read_off = range.start;
        let mut lvl_read_len = len;
        for j in 1..=depth {
            let nodes_j = n_leaves >> j;
            let base_j = 2 * n_leaves - 2 * nodes_j;
            let write_off = base_j + (range.start >> j);
            let write_len = len >> j;
            // SAFETY: read = this worker's own just-written nodes of level
            // j−1; write = this worker's own disjoint slice of level j; both
            // inside the 2·n_leaves−1 tree, never aliasing.
            let (read, write) = unsafe {
                (
                    core::slice::from_raw_parts(
                        (tree_addr as *const Hash).add(lvl_read_off),
                        lvl_read_len,
                    ),
                    core::slice::from_raw_parts_mut(
                        (tree_addr as *mut Hash).add(write_off),
                        write_len,
                    ),
                )
            };
            merkle::hash_pairs_level_serial(read, write, kind);
            lvl_read_off = write_off;
            lvl_read_len = write_len;
        }
    });
    match local_levels.load(Ordering::Acquire) {
        usize::MAX | 0 => 0,
        d => d,
    }
}

/// Build tree levels from the level with `from_nodes` nodes up to the root,
/// reading the already-written level below. Flat-layout offsets: the level
/// with `s` nodes starts at `2·n_leaves − 2·s`.
pub(crate) fn build_upper_levels(
    tree: &mut [Hash],
    n_leaves: usize,
    from_nodes: usize,
    kind: HashKind,
) {
    let mut s = from_nodes;
    while s > 1 {
        let read_start = 2 * n_leaves - 2 * s;
        let (read, rest) = tree[read_start..].split_at_mut(s);
        let write = &mut rest[..s / 2];
        merkle::hash_pairs_level(read, write, kind);
        s >>= 1;
    }
}

/// The GPU-hybrid streaming commit body. Returns `None` only when the GPU
/// session cannot even begin (alignment/coverage refused — nothing computed
/// yet, caller falls through to the pure-CPU path). Any failure AFTER the
/// session starts is repaired here on the CPU (the codeword is always fully
/// encoded by the CPU regardless of GPU health) and still returns `Some`.
#[allow(clippy::ptr_arg)] // The owned Vec is pooled/truncated by this path.
fn gpu_streamed_commit(
    ntt: &AdditiveNttF128,
    codeword: &mut Vec<F128>,
    z_packed: &[F128],
    params: &PcsParams,
    n_chunks: usize,
    stop_nodes: usize,
    timing: bool,
) -> Option<(Vec<Hash>, Hash)> {
    use crate::gpu;

    let n_leaves = params.n_leaves();
    let leaf_size = params.leaf_size_bytes();
    let num_ntts = params.num_ntts();
    let total_nodes = 2 * n_leaves - 1;
    let kind = params.merkle_hash;

    // Over-allocate by one 16 KiB page of nodes: gpu::merkle::begin requires
    // floor_page(tree_len·32) to cover every GPU-written node, and the last
    // real node may otherwise sit in a partially-covered tail page (matters
    // for small `stop_nodes`; at the production 2^14 the slack is ~512 KiB).
    // Truncated back to `total_nodes` before returning.
    let mut tree: Vec<Hash> = take_tree(total_nodes + TREE_PAD_NODES);

    let cw_ptr = codeword.as_ptr() as *const u8;
    let cw_len = codeword.len() * core::mem::size_of::<F128>();
    // SAFETY (begin): the session holds raw pointers, not borrows. The GPU
    // only ever reads codeword ranges whose NTT super-chunk barrier has
    // passed (final data, no further CPU writes), and only ever writes tree
    // slots the CPU join does not touch. Both buffers outlive the session
    // (finish() is called below before either can drop).
    let mut session = match unsafe {
        gpu::merkle::begin(
            core::slice::from_raw_parts(cw_ptr, cw_len),
            leaf_size,
            tree.as_mut_ptr(),
            tree.len(),
            stop_nodes,
        )
    } {
        Some(s) => s,
        None => {
            // Alignment/coverage refused — no latch, nothing computed yet.
            crate::gpu::gpu_dbg_trace("merkle: begin() refused (alignment/coverage); CPU path");
            give_tree(tree);
            return None;
        }
    };

    // Async prewire of the codeword + tree buffers while the NTT's top
    // layers run (~30 ms window): the driver's page wiring for ~1.1 GiB
    // otherwise stalls the first leaf command buffer (season-1: ~23 ms).
    // The witness a/b/z buffers are prewired at prove entry in flock-prover.
    let prewire = {
        let cw_addr = cw_ptr as usize;
        let tree_addr = tree.as_ptr() as usize;
        let tree_bytes = tree.len() * core::mem::size_of::<Hash>();
        std::thread::Builder::new()
            .name("flock-gpu-prewire-commit".into())
            .spawn(move || {
                // SAFETY: both buffers outlive this thread — it is joined
                // before finish(), below, and the buffers live to end of fn.
                unsafe {
                    gpu::prewire(core::slice::from_raw_parts(cw_addr as *const u8, cw_len));
                    gpu::prewire(core::slice::from_raw_parts(
                        tree_addr as *const u8,
                        tree_bytes,
                    ));
                }
            })
            .ok()
    };

    // ---- NTT with ordered chunk streaming. Chunk ownership is decided HERE,
    // once, per chunk: committed to the GPU (chunks 0..gpu_share, while
    // healthy) or pushed to the CPU-join list. This is what keeps the two
    // sides' tree writes disjoint.
    let gpu_share = MERKLE_GPU_CHUNK_SHARE.load(Ordering::Relaxed).min(n_chunks);
    let mut gpu_ranges: Vec<core::ops::Range<usize>> = Vec::with_capacity(n_chunks);
    let mut cpu_ranges: Vec<core::ops::Range<usize>> = Vec::with_capacity(n_chunks);
    let mut gpu_failed = false;

    let t_ntt = std::time::Instant::now();
    ntt.rs_encode_interleaved_streamed(
        z_packed,
        codeword,
        num_ntts,
        n_chunks,
        &mut |idx, range| {
            let give_gpu = !gpu_failed && idx < gpu_share;
            if give_gpu && session.commit_leaves(range.start, range.end) {
                gpu_ranges.push(range);
            } else {
                if give_gpu {
                    // commit_leaves failed and latched; everything from here
                    // on (and the repair below) is CPU work.
                    gpu_failed = true;
                }
                cpu_ranges.push(range);
            }
        },
    );
    let ntt_seconds = t_ntt.elapsed().as_secs_f64();

    let codeword_bytes: &[u8] = unsafe { core::slice::from_raw_parts(cw_ptr, cw_len) };

    // ---- CPU join on the leftover chunks (wide pool, atomic chunk cursor).
    let t_join = std::time::Instant::now();
    if !cpu_ranges.is_empty() {
        cpu_join_hash_leaves(
            codeword_bytes,
            leaf_size,
            &mut tree[..n_leaves],
            &cpu_ranges,
            kind,
        );
    }
    let cpu_join_seconds = t_join.elapsed().as_secs_f64();

    // ---- GPU parent levels (node count ≥ stop_nodes). Must be committed
    // only after every leaf is in place: GPU leaf cbs are queue-ordered
    // ahead of this cb, and the CPU join writes completed above (shared
    // storage is coherent at command-buffer commit).
    let do_gpu_parents = !gpu_failed && n_leaves / 2 >= stop_nodes;
    let mut committed_parents = false;
    if do_gpu_parents {
        if session.commit_parent_levels() {
            committed_parents = true;
        } else {
            gpu_failed = true;
        }
    }

    if let Some(h) = prewire {
        let _ = h.join();
    }

    // ---- Wait for the GPU (leaf cbs + parent cb).
    let t_wait = std::time::Instant::now();
    let gpu_seconds = session.finish();
    let wait_seconds = t_wait.elapsed().as_secs_f64();

    match gpu_seconds {
        Some(gpu_busy) if !gpu_failed => {
            // ---- CPU top: from stop_nodes (or from the leaves when the tree
            // was too small for GPU parent levels) up to the root.
            let from_nodes = if committed_parents {
                stop_nodes
            } else {
                n_leaves
            };
            build_upper_levels(&mut tree, n_leaves, from_nodes, kind);

            // Calibrate only on production-scale commits: on tiny (test)
            // shapes the per-cb fixed cost dominates and would poison the
            // rates (or spuriously latch the pipeline off process-wide).
            if cw_len >= (64 << 20) {
                let chunk_mib = (cw_len as f64 / n_chunks as f64) / (1024.0 * 1024.0);
                note_merkle_calibration(
                    gpu_ranges.len(),
                    cpu_ranges.len(),
                    chunk_mib,
                    gpu_busy,
                    wait_seconds,
                    ntt_seconds,
                    cpu_join_seconds,
                    n_chunks,
                );
            }
            if timing {
                eprintln!(
                    "[commit-timing] gpu-hybrid: ntt {:.2} ms (gpu {}/{} chunks, busy {:.2} ms), cpu-join {:.2} ms, wait {:.2} ms",
                    ntt_seconds * 1e3,
                    gpu_ranges.len(),
                    n_chunks,
                    gpu_busy * 1e3,
                    cpu_join_seconds * 1e3,
                    wait_seconds * 1e3,
                );
            }
            gpu::gpu_dbg_trace(&format!(
                "merkle: ntt={:.2}ms gpu_chunks={} cpu_chunks={} busy={:.2}ms join={:.2}ms wait={:.2}ms",
                ntt_seconds * 1e3,
                gpu_ranges.len(),
                cpu_ranges.len(),
                gpu_busy * 1e3,
                cpu_join_seconds * 1e3,
                wait_seconds * 1e3,
            ));
        }
        _ => {
            // GPU failed mid-flight (latched inside gpu::merkle). The
            // codeword is fully encoded — rebuild the whole tree on the CPU.
            // GPU-written tree contents are untrusted; recompute everything.
            gpu::gpu_dbg_trace("merkle: GPU FAILED mid-flight; full CPU tree rebuild");
            wide_hash_pool().install(|| {
                merkle::hash_leaves(codeword_bytes, leaf_size, &mut tree[..n_leaves], kind);
                build_upper_levels(&mut tree, n_leaves, n_leaves, kind);
            });
        }
    }

    tree.truncate(total_nodes);
    let root = tree[total_nodes - 1];
    Some((tree, root))
}

/// One-time GPU Merkle warmup calibration, called from the per-hash Setup
/// constructors (the ranked worker's untimed window). Forces Metal context +
/// pipeline compilation (~45 ms, once), then measures CPU vs GPU leaf-hash
/// rates on a synthetic 64 MiB buffer and seeds the split atomics — so the
/// FIRST measured prove already runs with a sane split, and a broken GPU is
/// latched off before it can cost a prove anything.
pub fn gpu_merkle_warmup_calibrate() {
    use crate::gpu;
    if !gpu::metal_available() || !gpu::merkle::available() {
        return;
    }
    const WARM_F128: usize = 1 << 22; // 64 MiB of leaf data
    let leaf_size = 1024usize;
    let n_leaves = WARM_F128 * 16 / leaf_size; // 65536
    let mut data = crate::scratch::take_f128(WARM_F128);
    // Deterministic contents (data-independent timing for BLAKE3, but the
    // buffer must be initialized and resident).
    data.fill(F128::ZERO);
    let mut tree: Vec<Hash> = take_tree(2 * n_leaves - 1 + TREE_PAD_NODES);

    // CPU rate: same batched kernels the join uses, on the wide pool.
    let t_cpu = std::time::Instant::now();
    {
        let bytes: &[u8] =
            unsafe { core::slice::from_raw_parts(data.as_ptr() as *const u8, WARM_F128 * 16) };
        wide_hash_pool().install(|| {
            merkle::hash_leaves(bytes, leaf_size, &mut tree[..n_leaves], HashKind::Blake3);
        });
    }
    let cpu_seconds = t_cpu.elapsed().as_secs_f64();

    // GPU rate: two passes (first pays wiring + clock ramp), keep the
    // second — but calibrate on FULL turnaround of that pass.
    let mut gpu_seconds_best = f64::INFINITY;
    for _pass in 0..2 {
        let bytes: &[u8] =
            unsafe { core::slice::from_raw_parts(data.as_ptr() as *const u8, WARM_F128 * 16) };
        // SAFETY: `data` and `tree` outlive the session; no CPU writes race
        // the GPU (the CPU pass above is complete).
        let Some(mut session) = (unsafe {
            gpu::merkle::begin(
                bytes,
                leaf_size,
                tree.as_mut_ptr(),
                tree.len(),
                2 * n_leaves, // leaves only — no parent levels in the probe
            )
        }) else {
            break;
        };
        let t_gpu = std::time::Instant::now();
        let quarter = n_leaves / 4;
        let mut ok = true;
        for c in 0..4 {
            ok = ok && session.commit_leaves(c * quarter, (c + 1) * quarter);
        }
        let finished = session.finish();
        if ok && finished.is_some() {
            gpu_seconds_best = gpu_seconds_best.min(t_gpu.elapsed().as_secs_f64());
        } else {
            break; // failure latched inside gpu::merkle
        }
    }

    let mib = (WARM_F128 * 16) as f64 / (1024.0 * 1024.0);
    let cpu_ns = (cpu_seconds * 1e9 / mib) as u64;
    if cpu_ns > 0 {
        MERKLE_CPU_NS_PER_MIB.store(cpu_ns, Ordering::Relaxed);
    }
    if gpu_seconds_best.is_finite() {
        let gpu_ns = (gpu_seconds_best * 1e9 / mib) as u64;
        if gpu_ns > 0 {
            MERKLE_GPU_NS_PER_MIB.store(gpu_ns, Ordering::Relaxed);
        }
        if gpu_ns > MERKLE_LATCH_FACTOR * cpu_ns {
            gpu::gpu_dbg_trace(&format!(
                "merkle warmup: LATCH OFF (gpu {gpu_ns} ns/MiB, cpu {cpu_ns})"
            ));
            gpu::merkle::set_enabled(false);
        } else {
            // GPU healthy: default to the full share; the NTT overlap makes
            // the GPU strictly cheaper than the CPU join for every chunk the
            // deep pass can feed it in time.
            MERKLE_GPU_CHUNK_SHARE.store(N_STREAM_CHUNKS, Ordering::Relaxed);
        }
        gpu::gpu_dbg_trace(&format!(
            "merkle warmup: cpu {cpu_ns} ns/MiB, gpu {gpu_ns} ns/MiB"
        ));
    }
    give_tree(tree);
    crate::scratch::give_f128(data);
}

/// Tag the current thread as background QoS. On macOS the scheduler then
/// strongly prefers efficiency (E) cores — ideal for the fault/bandwidth-bound
/// codeword pre-fault, which we want OFF the performance cores running witness
/// generation. No-op on other platforms.
#[cfg(target_os = "macos")]
fn set_background_qos() {
    // QOS_CLASS_BACKGROUND = 0x09. Declared inline to avoid a libc dependency.
    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }
    unsafe {
        let _ = pthread_set_qos_class_self_np(0x09, 0);
    }
}
#[cfg(not(target_os = "macos"))]
fn set_background_qos() {}

/// Allocate + zero-fill (pre-fault) the codeword buffer that [`commit_into`]
/// will consume, on a background-QoS (E-core) thread, **while** `gen` runs on
/// the caller's performance threads. Returns `(Some(buf), gen_result)`.
///
/// The codeword alloc is page-fault-bound (first-touch of a fresh 64–512 MB
/// buffer) and scales ~1.0×, so overlapping it with witness generation hides it
/// almost entirely (measured ~99% at m=29 — see `benches/ecore_offload_probe`).
///
/// **Gated for honest single-threaded behavior:** when the rayon pool has ≤ 1
/// thread (i.e. `RAYON_NUM_THREADS=1`), this spawns **zero** OS threads — it
/// runs `gen` and returns `None`, leaving [`commit`] to allocate inline. The
/// whole offload is therefore invisible to truly-serial runs.
pub fn prefault_codeword_during<R>(
    params: &PcsParams,
    generate: impl FnOnce() -> R,
) -> (Option<Vec<F128>>, R) {
    if rayon::current_num_threads() <= 1 || std::env::var_os("FLOCK_NO_PREFAULT").is_some() {
        // Truly single-threaded (or explicitly disabled): no extra OS thread;
        // commit allocates inline. FLOCK_NO_PREFAULT lets benchmarks A/B the
        // offload and keeps fixed-thread-count sweeps honest.
        return (None, generate());
    }
    let codeword_len = params.n_positions() * params.num_ntts();
    // Warm path: a pooled buffer is already resident — there is nothing to
    // pre-fault, and commit_into writes every slot itself. Skip the thread.
    if let Some(buf) = crate::scratch::try_take_f128(codeword_len) {
        return (Some(buf), generate());
    }
    // Cold path: allocate + first-touch on a background-QoS thread, hidden
    // under witness generation. (commit_into rewrites all slots, so the
    // zero values themselves don't matter — the page faults do.)
    std::thread::scope(|s| {
        let h = s.spawn(move || {
            set_background_qos();
            let mut buf: Vec<F128> = crate::alloc_uninit_f128_vec(codeword_len);
            unsafe {
                std::ptr::write_bytes(buf.as_mut_ptr(), 0u8, codeword_len);
            }
            buf
        });
        let r = generate();
        (Some(h.join().unwrap()), r)
    })
}

#[cfg(test)]
mod tests {
    /// The exact ranked selector is narrow, and regrouping sixteen retired
    /// blocks into one 2048-leaf parent fold reproduces every flat-tree node.
    #[test]
    fn subtree_parent_regroup_matches_full_tree_node_for_node() {
        use super::*;

        assert!(subtree_parent_regroup_selected(
            1 << 20,
            64,
            1024,
            HashKind::Blake3,
            false,
        ));
        for selected in [
            subtree_parent_regroup_selected((1 << 20) - 1, 64, 1024, HashKind::Blake3, false),
            subtree_parent_regroup_selected(1 << 20, 32, 1024, HashKind::Blake3, false),
            subtree_parent_regroup_selected(1 << 20, 64, 512, HashKind::Blake3, false),
            subtree_parent_regroup_selected(1 << 20, 64, 1024, HashKind::Sha256, false),
            subtree_parent_regroup_selected(1 << 20, 64, 1024, HashKind::Blake3, true),
        ] {
            assert!(!selected);
        }

        let first = 0..RANKED_PARENT_BLOCK_LEAVES;
        assert_eq!(local_parent_fold_range(&first, false), Some(first.clone()));
        assert_eq!(local_parent_fold_range(&first, true), None);
        let final_first = RANKED_PARENT_SUBGROUP_LEAVES - RANKED_PARENT_BLOCK_LEAVES
            ..RANKED_PARENT_SUBGROUP_LEAVES;
        assert_eq!(
            local_parent_fold_range(&final_first, true),
            Some(0..RANKED_PARENT_SUBGROUP_LEAVES),
        );
        let unexpected = 0..256;
        assert_eq!(local_parent_fold_range(&unexpected, true), Some(unexpected),);

        const N_LEAVES: usize = 2 * RANKED_PARENT_SUBGROUP_LEAVES;
        const LEAF_SIZE: usize = 1024;
        let mut bytes = vec![0u8; N_LEAVES * LEAF_SIZE];
        for (i, byte) in bytes.iter_mut().enumerate() {
            *byte = (i as u64)
                .wrapping_mul(0x9E37_79B9)
                .rotate_left((i & 31) as u32) as u8;
        }
        let oracle = merkle::merkle_tree(&bytes, N_LEAVES, HashKind::Blake3);
        let mut tree = vec![[0u8; 32]; 2 * N_LEAVES - 1];
        let mut parent_ranges = Vec::new();

        for start in (0..N_LEAVES).step_by(RANKED_PARENT_BLOCK_LEAVES) {
            let range = start..start + RANKED_PARENT_BLOCK_LEAVES;
            merkle::hash_leaves_serial(
                &bytes[range.start * LEAF_SIZE..range.end * LEAF_SIZE],
                LEAF_SIZE,
                &mut tree[range.clone()],
                HashKind::Blake3,
            );
            let Some(parent_range) = local_parent_fold_range(&range, true) else {
                continue;
            };
            parent_ranges.push(parent_range.clone());
            let depth = parent_range.len().trailing_zeros() as usize;
            let mut lvl_read_off = parent_range.start;
            let mut lvl_read_len = parent_range.len();
            for j in 1..=depth {
                let nodes_j = N_LEAVES >> j;
                let base_j = 2 * N_LEAVES - 2 * nodes_j;
                let write_off = base_j + (parent_range.start >> j);
                let write_len = parent_range.len() >> j;
                let (before_write, write_and_after) = tree.split_at_mut(write_off);
                merkle::hash_pairs_level_serial(
                    &before_write[lvl_read_off..lvl_read_off + lvl_read_len],
                    &mut write_and_after[..write_len],
                    HashKind::Blake3,
                );
                lvl_read_off = write_off;
                lvl_read_len = write_len;
            }
        }
        assert_eq!(
            parent_ranges,
            [
                0..RANKED_PARENT_SUBGROUP_LEAVES,
                RANKED_PARENT_SUBGROUP_LEAVES..N_LEAVES,
            ],
        );
        let platform_calls = |mut nodes: usize| {
            let mut calls = 0;
            while nodes > 1 {
                nodes >>= 1;
                calls += nodes.div_ceil(16);
            }
            calls
        };
        let old_calls = 8_192 * platform_calls(RANKED_PARENT_BLOCK_LEAVES) + platform_calls(8_192);
        let new_calls = 512 * platform_calls(RANKED_PARENT_SUBGROUP_LEAVES) + platform_calls(512);
        assert_eq!(
            (old_calls, new_calls, old_calls - new_calls),
            (90_627, 67_107, 23_520),
        );

        build_upper_levels(
            &mut tree,
            N_LEAVES,
            N_LEAVES >> RANKED_PARENT_SUBGROUP_LEAVES.trailing_zeros(),
            HashKind::Blake3,
        );
        assert_eq!(tree, oracle);
    }

    /// The fused recursive-commit route (encode → per-sub-group serial leaves
    /// + local subtree parents → upper levels) must produce the identical
    /// codeword AND identical flat Merkle tree (every node) as the incumbent
    /// encode-then-`fill_merkle_tree`, across the Ligerito recursive shapes
    /// (num_ntts = 8, rate 1/4, log_d 10..16), rate 1/2, odd lane counts, and
    /// the tiny scalar-NTT shapes where the callback fires once.
    #[test]
    fn fused_recursive_commit_matches_encode_then_fill() {
        use super::*;
        use crate::merkle::fill_merkle_tree;
        for &(log_d, num_ntts, log_inv_rate) in &[
            (10usize, 8usize, 2usize),
            (11, 8, 2),
            (12, 8, 2),
            (13, 8, 2),
            (14, 8, 2),
            (16, 8, 2),
            (12, 8, 1),
            (12, 4, 2),
            (12, 1, 1),
            (9, 2, 1),
            (8, 16, 2),
            // Ranked recursive rates: L2 = 1/8 (lone-top-layer bump fires),
            // L3 = 1/16, L4 = 1/32, L5 = 1/64.
            (16, 8, 3),
            (14, 8, 4),
            (12, 8, 5),
            (10, 8, 6),
            (13, 8, 3),
        ] {
            for &kind in &[HashKind::Blake3, HashKind::Sha256] {
                let ntt = AdditiveNttF128::standard(log_d);
                let msg: Vec<F128> = (0..((1usize << (log_d - log_inv_rate)) * num_ntts))
                    .map(|i| {
                        F128::new(
                            (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ log_d as u64,
                            (i as u64).wrapping_mul(0xD1B5_4A32_D192_ED03) ^ num_ntts as u64,
                        )
                    })
                    .collect();
                let n_leaves = 1usize << log_d;
                let leaf = num_ntts * 16;
                let mut cw_a = vec![F128::ZERO; n_leaves * num_ntts];
                ntt.rs_encode_interleaved(&msg, &mut cw_a, num_ntts);
                let mut tree_a = vec![Hash::default(); 2 * n_leaves - 1];
                let bytes_a = unsafe {
                    core::slice::from_raw_parts(cw_a.as_ptr() as *const u8, cw_a.len() * 16)
                };
                fill_merkle_tree(&mut tree_a, bytes_a, n_leaves, kind);

                let mut cw_b = vec![F128::new(u64::MAX, u64::MAX); n_leaves * num_ntts];
                let mut tree_b: Vec<Hash> = vec![[0xAAu8; 32]; 2 * n_leaves - 1];
                let folded = fused_encode_leaves_subtree(
                    &ntt,
                    &msg,
                    &mut cw_b,
                    num_ntts,
                    &mut tree_b,
                    n_leaves,
                    leaf,
                    kind,
                );
                build_upper_levels(&mut tree_b, n_leaves, n_leaves >> folded, kind);
                assert_eq!(
                    cw_a, cw_b,
                    "codeword log_d={log_d} n={num_ntts} rate={log_inv_rate}"
                );
                assert_eq!(
                    tree_a, tree_b,
                    "tree log_d={log_d} n={num_ntts} rate={log_inv_rate} kind={kind:?} folded={folded}"
                );
            }
        }
    }

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
    }

    /// The Ligerito configs derived from `PcsParams` must carry the params'
    /// Merkle hash, not the embedded security config's `hash` field. If they
    /// diverge, the L0 commitment and the recursive levels are built under
    /// different hashes and nothing verifies — silently, and only at the
    /// geometries that reach recursion.
    #[test]
    fn ligerito_configs_inherit_the_params_merkle_hash() {
        let mut params = default_params(22);
        params.log_batch_size = 6;

        assert_eq!(params.merkle_hash, HashKind::Sha256);
        assert_eq!(
            params.ligerito_prover_config().unwrap().merkle_hash,
            HashKind::Sha256
        );

        params.merkle_hash = HashKind::Blake3;
        assert_eq!(
            params.ligerito_prover_config().unwrap().merkle_hash,
            HashKind::Blake3,
            "prover config must follow PcsParams, not the embedded TOML"
        );
        assert_eq!(
            params.ligerito_verifier_config().unwrap().merkle_hash,
            HashKind::Blake3,
            "verifier config must follow PcsParams, not the embedded TOML"
        );
    }

    fn default_params(m: usize) -> PcsParams {
        PcsParams {
            m,
            log_inv_rate: 1,
            log_batch_size: 1,
            profile: Default::default(),
            merkle_hash: Default::default(),
        }
    }

    /// The replicate-fill + start-at-layer-`log_inv_rate` fast path must be
    /// byte-identical to the definitional encoding: zero-padded coefficients
    /// through the FULL forward NTT. Covers rate 1/2 and 1/4 and both
    /// interleaving widths.
    #[test]
    fn commit_matches_full_ntt_oracle() {
        use crate::ntt::AdditiveNttF128;
        let mut rng = Rng::new(0xFEED);
        for (m, log_inv_rate, log_batch_size) in [(10, 1, 1), (12, 1, 2), (12, 2, 1), (14, 2, 3)] {
            let params = PcsParams {
                m,
                log_inv_rate,
                log_batch_size,
                profile: Default::default(),
                merkle_hash: Default::default(),
            };
            let z = rng.bits(1 << m);
            let z_packed = super::super::pack::pack_witness(&z, m);

            let (commitment, pd) = commit(&z_packed, &params);

            // Oracle: explicit [z, 0, …, 0] coefficients, full NTT from layer 0.
            let mut oracle = vec![F128::ZERO; params.codeword_len_f128()];
            oracle[..z_packed.len()].copy_from_slice(&z_packed);
            let ntt = AdditiveNttF128::standard(params.k_code());
            ntt.forward_transform_interleaved(&mut oracle, params.num_ntts());

            assert_eq!(
                pd.codeword, oracle,
                "codeword mismatch at m={m} r={log_inv_rate}"
            );
            let oracle_bytes: &[u8] = unsafe {
                core::slice::from_raw_parts(oracle.as_ptr() as *const u8, oracle.len() * 16)
            };
            let oracle_root =
                *crate::merkle::merkle_tree(oracle_bytes, params.n_leaves(), params.merkle_hash)
                    .last()
                    .unwrap();
            assert_eq!(
                commitment.root, oracle_root,
                "root mismatch at m={m} r={log_inv_rate}"
            );
        }
    }

    #[test]
    fn commit_runs_and_produces_root() {
        let mut rng = Rng::new(42);
        for m in [8usize, 10, 12] {
            let z = rng.bits(1 << m);
            let z_packed = super::super::pack::pack_witness(&z, m);
            let params = default_params(m);
            let (commitment, prover_data) = commit(&z_packed, &params);
            assert_eq!(prover_data.codeword.len(), params.codeword_len_f128());
            assert_eq!(
                prover_data.merkle_tree.last().copied().unwrap(),
                commitment.root
            );
            assert_eq!(z_packed.len(), 1 << params.log_msg_len());
        }
    }

    #[test]
    fn commit_is_deterministic() {
        let mut rng = Rng::new(7);
        let m = 10;
        let z = rng.bits(1 << m);
        let z_packed = super::super::pack::pack_witness(&z, m);
        let params = default_params(m);
        let (c1, _) = commit(&z_packed, &params);
        let (c2, _) = commit(&z_packed, &params);
        assert_eq!(c1.root, c2.root);
    }

    #[test]
    fn commit_root_sensitive_to_witness() {
        let mut rng = Rng::new(99);
        let m = 10;
        let mut z = rng.bits(1 << m);
        let params = default_params(m);
        let (c1, _) = commit(&super::super::pack::pack_witness(&z, m), &params);
        z[7] ^= true;
        let (c2, _) = commit(&super::super::pack::pack_witness(&z, m), &params);
        assert_ne!(c1.root, c2.root);
    }

    #[test]
    fn rs_encoding_is_linear() {
        let mut rng = Rng::new(123);
        let m = 9;
        let params = default_params(m);
        let z1 = rng.bits(1 << m);
        let z2 = rng.bits(1 << m);
        let z_xor: Vec<bool> = z1.iter().zip(&z2).map(|(a, b)| a ^ b).collect();
        let pack = |z: &[bool]| super::super::pack::pack_witness(z, m);
        let (_, pd1) = commit(&pack(&z1), &params);
        let (_, pd2) = commit(&pack(&z2), &params);
        let (_, pd_x) = commit(&pack(&z_xor), &params);
        for (i, (&c1, &c2)) in pd1.codeword.iter().zip(&pd2.codeword).enumerate() {
            assert_eq!(c1 + c2, pd_x.codeword[i], "linearity fails at i={i}");
        }
    }

    #[test]
    fn codeword_doubles_message_length() {
        let mut rng = Rng::new(2);
        let m = 10;
        let params = default_params(m);
        let z = rng.bits(1 << m);
        let z_packed = super::super::pack::pack_witness(&z, m);
        let (_, pd) = commit(&z_packed, &params);
        assert_eq!(pd.codeword.len(), 2 * z_packed.len());
    }

    /// Merkle rewrite 1: CPU streamed commit (hash-as-NTT-finishes + parent
    /// fold) must be node-for-node identical to one-shot `merkle_tree()` on
    /// the same codeword. Reduced geometry, ranked 1024 B leaf. Does not
    /// go through Metal (`gpu_hybrid_commit_matches_pure_cpu` skips here).
    ///
    /// Soundness: `pd.codeword` must also match the definitional zero-padded
    /// full interleaved NTT of the same packed message (same oracle as
    /// [`commit_matches_full_ntt_oracle`]). Hashing `pd.codeword` alone
    /// cannot hide a wrong encode. Both shapes stay: m=20 scalar deep pass
    /// and m=24 parallel deep pass (`on_range_done` per sub-group).
    #[test]
    fn cpu_streamed_commit_matches_merkle_tree_node_for_node() {
        let mut rng = Rng::new(0x51EA);
        // m=20 / log_batch=6 → k_code=8, 256 leaves (scalar deep pass).
        // m=24 / log_batch=6 → k_code=12, 4096 leaves (parallel deep pass).
        for m in [20usize, 24] {
            let params = PcsParams {
                m,
                log_inv_rate: 1,
                log_batch_size: 6,
                profile: Default::default(),
                merkle_hash: HashKind::Blake3,
            };
            let z = rng.bits(1 << m);
            let z_packed = super::super::pack::pack_witness(&z, m);
            let (commitment, pd) = commit(&z_packed, &params);

            // Independent encode oracle (same construction as
            // `commit_matches_full_ntt_oracle`): zero-pad + full interleaved
            // NTT. The streamed path uses `rs_encode_interleaved_on_range_done`;
            // a wrong encode must not pass just because the tree hashes
            // `pd.codeword`.
            let mut encode_oracle = vec![F128::ZERO; params.codeword_len_f128()];
            encode_oracle[..z_packed.len()].copy_from_slice(&z_packed);
            let ntt = AdditiveNttF128::standard(params.k_code());
            ntt.forward_transform_interleaved(&mut encode_oracle, params.num_ntts());
            assert_eq!(
                pd.codeword,
                encode_oracle,
                "streamed codeword != full-NTT encode at m={m} r={} ntts={}",
                params.log_inv_rate,
                params.num_ntts()
            );

            let codeword_bytes: &[u8] = unsafe {
                core::slice::from_raw_parts(
                    pd.codeword.as_ptr() as *const u8,
                    pd.codeword.len() * core::mem::size_of::<F128>(),
                )
            };
            let oracle = merkle::merkle_tree(codeword_bytes, params.n_leaves(), HashKind::Blake3);
            assert_eq!(
                pd.merkle_tree.len(),
                2 * params.n_leaves() - 1,
                "flat 2n-1 layout at m={m}"
            );
            assert_eq!(
                pd.merkle_tree, oracle,
                "streamed CPU tree != merkle_tree() node-for-node at m={m}"
            );
            assert_eq!(
                commitment.root,
                oracle[2 * params.n_leaves() - 2],
                "root at 2n-2 at m={m}"
            );
        }
    }

    /// **GPU-hybrid acceptance**: the streamed GPU commit (leaf chunks + GPU
    /// parent levels + CPU top) must produce the exact tree and root the
    /// pure-CPU path produces — every node, not just the root. Forced through
    /// `gpu_streamed_commit` directly (production gates require ≥ 2^18
    /// leaves; here small `stop_nodes` values exercise the GPU parent levels
    /// and the CPU top at m≈20). SKIPS (does not fail) without Metal.
    #[test]
    fn gpu_hybrid_commit_matches_pure_cpu() {
        if !crate::gpu::merkle::available() {
            eprintln!("SKIP gpu_hybrid_commit_matches_pure_cpu: Metal unavailable");
            return;
        }
        let mut rng = Rng::new(0x69B0);
        // (m, stop_nodes): stop_nodes = 4 exercises deep GPU parent levels +
        // CPU top; a huge stop_nodes exercises the leaves-only + full-CPU-top
        // shape; log_batch_size = 6 gives the production 1024-B leaf.
        for (m, stop_nodes) in [(20usize, 4usize), (20, 1 << 30), (21, 64), (24, 64)] {
            let params = PcsParams {
                m,
                log_inv_rate: 1,
                log_batch_size: 6,
                profile: Default::default(),
                merkle_hash: HashKind::Blake3,
            };
            let z = rng.bits(1 << m);
            let z_packed = super::super::pack::pack_witness(&z, m);

            // CPU oracle: the normal path (gates keep it off the GPU here).
            let (cpu_commitment, cpu_pd) = commit(&z_packed, &params);

            // Forced GPU-hybrid.
            let ntt = AdditiveNttF128::standard(params.k_code());
            let mut codeword = crate::scratch::take_f128(params.codeword_len_f128());
            let (gpu_tree, gpu_root) = gpu_streamed_commit(
                &ntt,
                &mut codeword,
                &z_packed,
                &params,
                N_STREAM_CHUNKS,
                stop_nodes,
                false,
            )
            .expect("gpu_streamed_commit refused to start (begin() returned None)");

            assert_eq!(
                codeword, cpu_pd.codeword,
                "codeword mismatch at m={m} (streamed NTT)"
            );
            assert_eq!(
                gpu_tree, cpu_pd.merkle_tree,
                "full tree mismatch at m={m} stop_nodes={stop_nodes}"
            );
            assert_eq!(
                gpu_root, cpu_commitment.root,
                "root mismatch at m={m} stop_nodes={stop_nodes}"
            );
            assert!(
                crate::gpu::merkle::available(),
                "GPU latched off during the hybrid commit at m={m} — a command \
                 buffer failed mid-flight and the CPU repair masked it"
            );
            crate::scratch::give_f128(codeword);
            give_tree(gpu_tree);
        }
    }

    /// Production-scale smoke (256 MiB codeword, 2^18 leaves — the real
    /// `finalize_commit` gate threshold): full commit through the public
    /// entry, GPU-hybrid vs FLOCK-env-free CPU oracle. Ignored by default
    /// (allocates ~0.5 GiB); run explicitly with `-- --ignored`.
    /// SKIPS without Metal.
    #[test]
    #[ignore]
    fn gpu_hybrid_commit_production_scale_smoke() {
        if !crate::gpu::merkle::available() {
            eprintln!("SKIP gpu_hybrid_commit_production_scale_smoke: Metal unavailable");
            return;
        }
        // The warmup calibration Blake3Setup runs at m ≥ 26: must seed the
        // rate atomics, keep the pipeline enabled on a healthy GPU, and leave
        // the full GPU share in place.
        gpu_merkle_warmup_calibrate();
        assert!(
            crate::gpu::merkle::available(),
            "warmup calibration latched a healthy GPU off"
        );
        assert_eq!(gpu_merkle_split(), N_STREAM_CHUNKS);
        assert!(MERKLE_CPU_NS_PER_MIB.load(Ordering::Relaxed) > 0);
        assert!(MERKLE_GPU_NS_PER_MIB.load(Ordering::Relaxed) > 0);
        let m = 30usize;
        let params = PcsParams {
            m,
            log_inv_rate: 1,
            log_batch_size: 6,
            profile: Default::default(),
            merkle_hash: HashKind::Blake3,
        };
        assert!(params.n_leaves() >= (1 << 18), "must clear the real gate");
        // Random packed witness straight in F128 (2^23 × 16 B): bit-packing
        // 2^30 bools would dominate the test for no coverage.
        let mut rng = Rng::new(0x5CA1E);
        let mut z_packed: Vec<F128> = crate::alloc_uninit_vec(1usize << params.log_msg_len());
        for v in z_packed.iter_mut() {
            *v = F128 {
                lo: rng.next_u64(),
                hi: rng.next_u64(),
            };
        }

        // Public entry — exercises the production gates, the streamed NTT,
        // prewire, calibration-size guard, GPU parents, CPU top.
        let (commitment, pd) = commit(&z_packed, &params);

        // CPU oracle on the SAME encoded codeword.
        let bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(pd.codeword.as_ptr() as *const u8, pd.codeword.len() * 16)
        };
        let oracle = crate::merkle::merkle_tree(bytes, params.n_leaves(), HashKind::Blake3);
        assert_eq!(pd.merkle_tree, oracle, "hybrid tree != CPU oracle at m={m}");
        assert_eq!(commitment.root, *oracle.last().unwrap());
        assert!(
            crate::gpu::merkle::available(),
            "GPU latched off during the production-scale hybrid commit"
        );
    }

    /// TREE_POOL smallest-fit: a small take must reuse the small parked
    /// buffer, not steal the large one (L1 must not evict L0).
    #[test]
    fn tree_pool_smallest_fit_reuses_matching_buffer() {
        // Both sizes clear give_tree's 1<<18-node (8 MiB) floor.
        let n_small = 1 << 18;
        let n_large = 1 << 19;
        {
            let mut pool = TREE_POOL.lock().unwrap();
            pool.clear();
        }
        let small = take_tree(n_small);
        let ptr_small = small.as_ptr();
        give_tree(small);
        let large = take_tree(n_large);
        let ptr_large = large.as_ptr();
        give_tree(large);
        let again_small = take_tree(n_small);
        assert_eq!(
            again_small.as_ptr(),
            ptr_small,
            "small take must reuse the small parked tree"
        );
        assert_ne!(
            again_small.as_ptr(),
            ptr_large,
            "small take must not steal the large parked tree"
        );
        give_tree(again_small);
        let again_large = take_tree(n_large);
        assert_eq!(
            again_large.as_ptr(),
            ptr_large,
            "large take must reuse the large parked tree"
        );
        give_tree(again_large);
        {
            let mut pool = TREE_POOL.lock().unwrap();
            pool.clear();
        }
    }

    /// TREE_POOL cap 3 + lowered floor: the ranked L2 Ligerito tree
    /// (2^16 leaves → 2·2^16−1 = 131071 nodes, ~4 MiB) must now be parked
    /// by give_tree (previously refused by the 2^18 floor), and three
    /// trees (L0/L1/L2 sizes) must coexist in the pool.
    #[test]
    fn tree_pool_parks_l2_and_caps_at_three() {
        let n_l2 = 2 * (1 << 16) - 1; // 131071 — ranked L2
        let n_l1 = 2 * (1 << 18) - 1; // ranked L1
        let n_l0 = 2 * (1 << 20) - 1; // ranked L0
        {
            let mut pool = TREE_POOL.lock().unwrap();
            pool.clear();
        }
        // L2-sized tree must now be parked (was refused by old 2^18 floor).
        let l2 = take_tree(n_l2);
        let ptr_l2 = l2.as_ptr();
        give_tree(l2);
        let l1 = take_tree(n_l1);
        let ptr_l1 = l1.as_ptr();
        give_tree(l1);
        let l0 = take_tree(n_l0);
        let ptr_l0 = l0.as_ptr();
        give_tree(l0);
        // All three should be parked (cap 3).
        {
            let pool = TREE_POOL.lock().unwrap();
            assert_eq!(pool.len(), 3, "cap 3: L0+L1+L2 must all park");
        }
        // Smallest-fit: taking L2 must reuse the L2 buffer, not L0/L1.
        let again_l2 = take_tree(n_l2);
        assert_eq!(
            again_l2.as_ptr(),
            ptr_l2,
            "L2 take must reuse the small L2 buffer"
        );
        // A fourth give_tree must be refused (cap 3, pool full after re-give).
        give_tree(again_l2);
        {
            let pool = TREE_POOL.lock().unwrap();
            assert_eq!(pool.len(), 3, "cap 3: fourth tree must be refused");
        }
        {
            let mut pool = TREE_POOL.lock().unwrap();
            pool.clear();
        }
        // Suppress unused warnings for ptr_l0/l1 used in capacity reasoning.
        let _ = (ptr_l0, ptr_l1);
    }
}
