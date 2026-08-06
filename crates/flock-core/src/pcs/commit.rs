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
    pub codeword: CodewordBuf,
    pub merkle_tree: MerkleTreeBuf,
}

/// Storage for the L0 codeword. Normally a pooled `Vec` (CPU commit); with
/// the GPU commit latched on it is a view into the process-persistent Metal
/// staging buffer instead (unified memory — CPU reads during the open are
/// ordinary cached reads). Derefs to `[F128]` either way.
pub enum CodewordBuf {
    Cpu(Vec<F128>),
    Gpu(crate::gpu_commit::GpuCodeword),
}

/// Storage for the L0 Merkle tree. Ranked GPU commitments leave the tree in
/// their persistent host-visible Metal buffer, avoiding a 64 MiB copy-out.
pub enum MerkleTreeBuf {
    Cpu(Vec<Hash>),
    Gpu(crate::gpu_commit::GpuMerkleTree),
}

impl core::ops::Deref for MerkleTreeBuf {
    type Target = [Hash];
    fn deref(&self) -> &[Hash] {
        match self {
            MerkleTreeBuf::Cpu(v) => v,
            MerkleTreeBuf::Gpu(g) => g,
        }
    }
}

impl core::fmt::Debug for MerkleTreeBuf {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MerkleTreeBuf")
            .field("len", &self.len())
            .finish()
    }
}

impl PartialEq<Vec<Hash>> for MerkleTreeBuf {
    fn eq(&self, other: &Vec<Hash>) -> bool {
        &**self == other.as_slice()
    }
}

impl PartialEq<MerkleTreeBuf> for Vec<Hash> {
    fn eq(&self, other: &MerkleTreeBuf) -> bool {
        self.as_slice() == &**other
    }
}

impl core::ops::Deref for CodewordBuf {
    type Target = [F128];
    fn deref(&self) -> &[F128] {
        match self {
            CodewordBuf::Cpu(v) => v,
            CodewordBuf::Gpu(g) => g,
        }
    }
}

// Recycle the codeword buffer (the prover's largest single allocation —
// 128 MB at m = 29) through the scratch pool instead of unmapping it (the
// GPU-staging variant hands the staging buffer back to the latch on drop),
// and ranked-size trees through the GPU commit's tree pool (keeps the 64 MiB
// copy-out target page-resident across the warmup and timed proves).
impl Drop for ProverData {
    fn drop(&mut self) {
        // Drop the borrowed GPU-tree view before releasing the staging lease
        // carried by the codeword. This keeps both persistent GPU buffers
        // protected for the complete lifetime of ProverData.
        if let MerkleTreeBuf::Cpu(tree) =
            std::mem::replace(&mut self.merkle_tree, MerkleTreeBuf::Cpu(Vec::new()))
        {
            crate::gpu_commit::give_tree(tree);
        }
        match std::mem::replace(&mut self.codeword, CodewordBuf::Cpu(Vec::new())) {
            CodewordBuf::Cpu(v) => crate::scratch::give_f128(v),
            CodewordBuf::Gpu(g) => drop(g),
        }
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
    // Copy first 2^log_msg_len positions from packed witness; zero-pad the rest.
    //
    // At large m the codeword buffer is huge (128 MB at m=29, 512 MB at m=31).
    // `vec![F128::ZERO; n]` would eagerly zero all 128 MB upfront, then
    // immediately overwrite the lower half with `z_packed` — half the zero-fill
    // is wasted. Instead allocate uninit, write each half exactly once: copy
    // `z_packed` into the lower half, and zero-fill JUST the upper half (the
    // RS-encoding zero coefficients that the NTT's first-layer butterfly will
    // read). Saves ~64 MB of memory writes at m=29 (~9 ms).
    if crate::gpu_commit::gpu_commit_latched_on()
        && ranked_from_message_supported_len(params, codeword_len, z_packed)
    {
        // Latched Metal graph reads z directly into persistent staging.
        // CPU fallback buffer is allocated lazily only if Metal fails.
        return finalize_commit_impl(z_packed, Vec::new(), params, true);
    }
    let codeword = crate::scratch::take_f128(codeword_len);
    commit_into(z_packed, params, codeword)
}

/// Like [`commit`], but reuses a caller-provided codeword buffer instead of
/// allocating its own. The buffer must have length `codeword_len`; its
/// CONTENTS may be arbitrary (uninit/stale) — every slot is written here:
/// `z_packed` is replicated into all `2^log_inv_rate` sub-blocks (the exact
/// state after the first `log_inv_rate` NTT layers on `[z, 0, …, 0]`), in
/// parallel. Buffers from [`prefault_codeword_during`] or the scratch pool
/// are already resident, so no write faults.
pub fn commit_into(
    z_packed: &[F128],
    params: &PcsParams,
    mut codeword: Vec<F128>,
) -> (Commitment, ProverData) {
    params.validate();
    assert_eq!(z_packed.len(), 1usize << params.log_msg_len());
    let codeword_len = params.n_positions() * params.num_ntts();
    assert_eq!(
        codeword.len(),
        codeword_len,
        "commit_into: prebuilt codeword buffer has wrong length"
    );

    // RS encoding of [z, 0, …, 0] starts with `log_inv_rate` butterfly layers
    // whose bottom inputs are all zero — each is a pure copy, so after those
    // layers the buffer holds 2^log_inv_rate replicas of z. Write that state
    // directly (replicating z costs the same writes as the zero-fill it
    // replaces) and start the NTT at layer `log_inv_rate`, skipping those
    // layers' full-buffer reads and multiplies.
    //
    // Ranked rate-1/2 exception: the split top pass can synthesize both
    // replicas from `z_packed` directly (see
    // `forward_transform_interleaved_ranked_top_from_message`), deleting one
    // full replica store here and halving that pass's loads. Every other
    // shape writes the replica state and starts the NTT at `log_inv_rate`.
    if ranked_from_message_supported(params, &codeword, z_packed) {
        return finalize_commit_impl(z_packed, codeword, params, true);
    }
    replicate_message_fill(&mut codeword, z_packed);

    finalize_commit_impl(z_packed, codeword, params, false)
}

/// Whether the ranked rate-1/2 commit will fuse the replicate-fill into the
/// first NTT top pass (see
/// `forward_transform_interleaved_ranked_top_from_message`). Mirrors every
/// condition under which `finalize_commit` actually reaches the split ranked
/// top; witness producers use this to decide whether writing the replica
/// state themselves is still worthwhile. `FLOCK_NO_NTT_FROM_MSG=1` is the
/// exact A/B control restoring the hot-codeword replicate path.
pub fn use_ranked_from_message_commit(params: &PcsParams) -> bool {
    use_ranked_ntt_merkle_leaf_pipeline(params)
        && crate::epool::epool().is_some()
        && rayon::current_num_threads() > 1
        && params.log_inv_rate == 1
        && std::env::var_os("FLOCK_NO_NTT_TOP_EPOOL").is_none()
        && std::env::var_os("FLOCK_NTT_BLOCK_REGIONS").is_none()
        && std::env::var_os("FLOCK_NO_NTT_FROM_MSG").is_none()
}

/// [`use_ranked_from_message_commit`] plus the buffer-geometry check
/// [`commit_into`] needs before taking the fused path.
fn ranked_from_message_supported_len(
    params: &PcsParams,
    codeword_len: usize,
    z_packed: &[F128],
) -> bool {
    use_ranked_from_message_commit(params) && codeword_len == 2 * z_packed.len()
}

fn ranked_from_message_supported(params: &PcsParams, codeword: &[F128], z_packed: &[F128]) -> bool {
    ranked_from_message_supported_len(params, codeword.len(), z_packed)
}

/// Commit from a codeword whose first `log_inv_rate` trivial NTT layers have
/// already been applied.
///
/// The caller must initialize every element to the same state produced by
/// [`replicate_message_fill`] before calling this function. This entry point is
/// used when a witness producer writes those replicas directly while its input
/// rows are still cache-resident; unlike [`commit_into`], it performs no
/// witness-to-codeword fill and begins with the remaining NTT layers.
pub fn commit_preinitialized(
    z_packed: &[F128],
    codeword: Vec<F128>,
    params: &PcsParams,
) -> (Commitment, ProverData) {
    params.validate();
    assert_eq!(z_packed.len(), 1usize << params.log_msg_len());
    let codeword_len = params.n_positions() * params.num_ntts();
    assert_eq!(
        codeword.len(),
        codeword_len,
        "commit_preinitialized: codeword buffer has wrong length"
    );
    finalize_commit_impl(z_packed, codeword, params, false)
}

/// Complete a ranked commitment whose GPU from-`z` layers 0..3 were streamed
/// during witness generation. On any stream/Metal failure this takes the
/// unchanged stale codeword through the exact ordinary from-message CPU path.
#[doc(hidden)]
pub fn commit_from_streamed_first_pass(
    z_packed: &[F128],
    codeword: Vec<F128>,
    params: &PcsParams,
    stream: crate::gpu_commit::FromZFirstPassStream,
) -> (Commitment, ProverData) {
    params.validate();
    assert_eq!(z_packed.len(), 1usize << params.log_msg_len());
    // Empty marker: the latched caller omitted the speculative CPU fallback
    // buffer; every CPU fallback gate hydrates it from the scratch pool.
    assert!(
        codeword.is_empty() || codeword.len() == params.codeword_len_f128(),
        "streamed commit: codeword buffer has wrong length"
    );
    let (codeword, merkle_tree) = crate::gpu_commit::finish_from_z_first_pass_or_fallback(
        stream,
        z_packed,
        codeword,
        params,
        |cw| cpu_transform_and_tree(cw, params, Some(z_packed)),
    );
    let root = *merkle_tree.last().expect("merkle tree non-empty");
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

/// Process CPU (user+system) in ms for `FLOCK_COMMIT_TIMING` diagnostics.
/// Direct `getrusage(RUSAGE_SELF)`; diagnostics-only, 0.0 off macOS.
pub(crate) fn commit_cpu_ms() -> f64 {
    #[cfg(target_os = "macos")]
    {
        #[repr(C)]
        struct Timeval {
            sec: i64,
            usec: i32,
        }
        #[repr(C)]
        struct Rusage {
            utime: Timeval,
            stime: Timeval,
            other: [i64; 14],
        }
        unsafe extern "C" {
            fn getrusage(who: i32, usage: *mut Rusage) -> i32;
        }
        let mut ru = Rusage {
            utime: Timeval { sec: 0, usec: 0 },
            stime: Timeval { sec: 0, usec: 0 },
            other: [0; 14],
        };
        if unsafe { getrusage(0, &mut ru) } == 0 {
            return (ru.utime.sec + ru.stime.sec) as f64 * 1e3
                + (ru.utime.usec + ru.stime.usec) as f64 * 1e-3;
        }
    }
    0.0
}

/// Fill `codeword` with `2^r` replicas of `msg` (`r = log2(codeword.len() /
/// msg.len())`) — the exact state after the first `r` forward-NTT layers on
/// the zero-padded coefficient vector `[msg, 0, …, 0]`. Pair with
/// `forward_transform_interleaved_from_layer(…, r)`. Every slot of `codeword`
/// is written (input contents may be stale/uninit).
#[doc(hidden)]
pub fn replicate_message_fill(codeword: &mut [F128], msg: &[F128]) {
    use rayon::prelude::*;
    let msg_len = msg.len();
    debug_assert!(codeword.len().is_multiple_of(msg_len));
    const COPY_CHUNK: usize = 1 << 16;
    if msg_len >= COPY_CHUNK {
        // Both are powers of two, so chunks never straddle a replica boundary.
        codeword
            .par_chunks_mut(COPY_CHUNK)
            .enumerate()
            .for_each(|(i, dst)| {
                let src_off = (i * COPY_CHUNK) % msg_len;
                dst.copy_from_slice(&msg[src_off..src_off + dst.len()]);
            });
    } else {
        for rep in codeword.chunks_mut(msg_len) {
            rep.copy_from_slice(msg);
        }
    }
}

/// Exact ranked geometry for the cache-local NTT-to-Merkle leaf pipeline.
/// Alternate hashes, profiles, rates, and recursive commits retain the existing
/// independently scheduled transform and leaf pass.
#[inline]
fn is_ranked_ntt_merkle_leaf_pipeline_shape(params: &PcsParams) -> bool {
    cfg!(all(
        target_os = "macos",
        target_arch = "aarch64",
        target_feature = "aes"
    )) && params.m == 32
        && params.log_inv_rate == 1
        && params.log_batch_size == 6
        && params.profile == crate::pcs::ligerito::LigeritoProfile::Fast
        && params.merkle_hash == HashKind::Blake3
}

#[inline]
fn use_ranked_ntt_merkle_leaf_pipeline(params: &PcsParams) -> bool {
    is_ranked_ntt_merkle_leaf_pipeline_shape(params)
        && std::env::var_os("FLOCK_NO_NTT_MERKLE_PIPELINE").is_none()
}

#[derive(Clone, Copy)]
struct RankedLeafJob {
    elem_offset: usize,
    elem_len: usize,
}

/// Finish the ranked NTT and hash each finalized 1 MiB subtree before it goes
/// cold. P-core transform jobs offer leaf work to a bounded queue drained by
/// the existing utility-QoS E-core pool; when that queue is full they hash the
/// just-finished subtree inline. Thus the main NTT never waits for queue space,
/// the helper pool retains the frontier's extra leaf throughput, and at most a
/// small L2-sized window of completed codeword data awaits hashing.
fn ranked_ntt_with_pipelined_leaves(
    ntt: &AdditiveNttF128,
    codeword: &mut [F128],
    params: &PcsParams,
    tree: &mut [Hash],
    helper: &rayon::ThreadPool,
    from_message: Option<&[F128]>,
) -> usize {
    use rayon::prelude::*;
    use std::sync::Mutex;
    use std::sync::mpsc::{TrySendError, sync_channel};

    let num_ntts = params.num_ntts();
    assert_eq!(num_ntts, 64);
    let num_leaves = codeword.len() / num_ntts;
    assert_eq!(tree.len(), 2 * num_leaves - 1);
    // Stop at four roots (128 contiguous bytes) per 1,024-leaf job. These eight
    // local levels cover 1,020 of each subtree's 1,023 parent nodes while
    // leaving the shared top for the level-wide builder.
    const LOCAL_PARENT_LEVELS: usize = 8;
    let local_parent_levels = if std::env::var_os("FLOCK_NO_MERKLE_SUBTREE_PARENTS").is_some() {
        0
    } else {
        LOCAL_PARENT_LEVELS
    };
    let codeword_base = crate::epool::SyncPtr(codeword.as_mut_ptr());
    let tree_base = crate::epool::SyncPtr(tree.as_mut_ptr());

    let hash_job = |job: RankedLeafJob| {
        assert_eq!(job.elem_offset % num_ntts, 0);
        assert_eq!(job.elem_len % num_ntts, 0);
        let leaf_start = job.elem_offset / num_ntts;
        let leaf_len = job.elem_len / num_ntts;
        assert_eq!(leaf_start % (1 << local_parent_levels), 0);
        assert_eq!(leaf_len % (1 << local_parent_levels), 0);
        // SAFETY: the NTT publishes a job only after the corresponding mutable
        // subtree is finalized and never touched again. Every subtree is
        // published exactly once; offsets are disjoint and cover the codeword,
        // so both the immutable input ranges and mutable leaf-output ranges are
        // pairwise disjoint and in bounds. Channel send/receive synchronizes the
        // completed NTT writes before a helper worker reads them.
        unsafe {
            let elems =
                core::slice::from_raw_parts(codeword_base.ptr().add(job.elem_offset), job.elem_len);
            let bytes = core::slice::from_raw_parts(
                elems.as_ptr().cast::<u8>(),
                core::mem::size_of_val(elems),
            );
            let outs = core::slice::from_raw_parts_mut(tree_base.ptr().add(leaf_start), leaf_len);
            merkle::hash_ranked_blake3_leaf_chunk(bytes, outs);

            // Build the aligned local subtree while its leaf range is still
            // hot. At every level, different jobs own disjoint read and write
            // ranges. Only the small shared top remains after the job barrier.
            let mut read_level_start = 0usize;
            let mut read_level_len = num_leaves;
            let mut local_start = leaf_start;
            let mut local_len = leaf_len;
            for _ in 0..local_parent_levels {
                let write_level_start = read_level_start + read_level_len;
                let write_start = write_level_start + (local_start >> 1);
                let write_len = local_len >> 1;
                let read = core::slice::from_raw_parts(
                    tree_base.ptr().add(read_level_start + local_start),
                    local_len,
                );
                let write =
                    core::slice::from_raw_parts_mut(tree_base.ptr().add(write_start), write_len);
                merkle::hash_ranked_blake3_parent_chunk(read, write);
                read_level_start = write_level_start;
                read_level_len >>= 1;
                local_start >>= 1;
                local_len >>= 1;
            }
        }
    };

    // Two queued subtrees per helper keep all four E-cores fed while bounding
    // not-yet-hashed input to 8 MiB on the ranked 4-E-core host.
    let queue_capacity = (2 * helper.current_num_threads()).max(1);
    let (sender, receiver) = sync_channel::<RankedLeafJob>(queue_capacity);
    let receiver = Mutex::new(receiver);

    // The exact ranked top passes can borrow the E-core pool themselves. Run
    // them before starting the blocking leaf receivers; otherwise every helper
    // worker would be parked on `recv` and a nested top-pass broadcast could
    // never begin. The deep transform and callback-driven leaf pipeline below
    // retain their existing overlap and scheduling.
    let split_ranked_top = is_ranked_ntt_merkle_leaf_pipeline_shape(params)
        && std::env::var_os("FLOCK_NO_NTT_TOP_EPOOL").is_none()
        && std::env::var_os("FLOCK_NTT_BLOCK_REGIONS").is_none();
    if split_ranked_top {
        match from_message {
            Some(msg) => ntt.forward_transform_interleaved_ranked_top_from_message(
                msg,
                codeword,
                num_ntts,
                params.log_inv_rate,
            ),
            None => ntt.forward_transform_interleaved_ranked_top_from_layer(
                codeword,
                num_ntts,
                params.log_inv_rate,
            ),
        }
    } else if let Some(msg) = from_message {
        // Gate mismatch fallback: materialize the replica state the ordinary
        // way; the unsplit transform below starts at `log_inv_rate`.
        replicate_message_fill(codeword, msg);
    }

    std::thread::scope(|scope| {
        let helper_manager = scope.spawn(|| {
            helper.broadcast(|_| {
                loop {
                    let job = receiver.lock().unwrap().recv();
                    match job {
                        Ok(job) => hash_job(job),
                        Err(_) => break,
                    }
                }
            });
        });

        let finish_chunk = |elem_offset, chunk: &[F128]| {
            let job = RankedLeafJob {
                elem_offset,
                elem_len: chunk.len(),
            };
            match sender.try_send(job) {
                Ok(()) => {}
                Err(TrySendError::Full(job) | TrySendError::Disconnected(job)) => hash_job(job),
            }
        };
        if split_ranked_top {
            ntt.forward_transform_interleaved_ranked_deep_and_then(
                codeword,
                num_ntts,
                params.log_inv_rate,
                finish_chunk,
            );
        } else {
            ntt.forward_transform_interleaved_from_layer_and_then(
                codeword,
                num_ntts,
                params.log_inv_rate,
                finish_chunk,
            );
        }
        drop(sender);

        // No more jobs can arrive. Pull any bounded queue tail away from the
        // E-cores and let the full main pool finish it while already-claimed E
        // jobs complete concurrently.
        let mut tail = Vec::with_capacity(queue_capacity);
        {
            let receiver = receiver.lock().unwrap();
            while let Ok(job) = receiver.try_recv() {
                tail.push(job);
            }
        }
        tail.into_par_iter().for_each(|job| hash_job(job));
        helper_manager
            .join()
            .expect("ranked NTT-to-Merkle helper manager panicked");
    });
    local_parent_levels
}

/// Shared tail of [`commit`] / [`commit_into`]: interleaved forward additive
/// NTT (RS-encode every lane) then the initial Merkle tree over codeword rows.
///
/// `from_message = true` is the ranked rate-1/2 fusion: `codeword` holds
/// arbitrary STALE bytes and both replicas are synthesized from `z_packed`
/// directly — by the split ranked top pass on the CPU (see
/// `forward_transform_interleaved_ranked_top_from_message`), or inherently by
/// the GPU graph, whose first pass always reads only `z_packed`. With
/// `from_message = false` the caller materialized the exact
/// [`replicate_message_fill`] state.
///
/// GPU latch (see [`crate::gpu_commit`]): the GPU graph never writes
/// `z_packed` or `codeword`, so on any Metal failure the CPU closure below
/// runs on the untouched inputs and reproduces the exact CPU result for BOTH
/// `from_message` modes — the fallback keeps the frontier's from-message
/// fusion rather than forcing a re-replication.
fn finalize_commit_impl(
    z_packed: &[F128],
    codeword: Vec<F128>,
    params: &PcsParams,
    from_message: bool,
) -> (Commitment, ProverData) {
    let (codeword, merkle_tree) =
        crate::gpu_commit::commit_l0_or_fallback(z_packed, codeword, params, |cw| {
            cpu_transform_and_tree(cw, params, from_message.then_some(z_packed))
        });
    let root = *merkle_tree.last().expect("merkle tree non-empty");
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

/// The CPU commit pipeline: interleaved forward NTT from the rate layers,
/// then the L0 Merkle tree. This is the (only) path on non-ranked shapes and
/// the latched fallback for the GPU offload.
///
/// When `from_message` is `Some(z)`, `codeword` holds arbitrary stale bytes
/// and the ranked split top pass synthesizes both replicas from `z` directly
/// (gate: [`ranked_from_message_supported`]). Paths that cannot fuse fall
/// back to an explicit [`replicate_message_fill`] first.
pub(crate) fn cpu_transform_and_tree(
    codeword: &mut [F128],
    params: &PcsParams,
    from_message: Option<&[F128]>,
) -> Vec<Hash> {
    let helper = if use_ranked_ntt_merkle_leaf_pipeline(params) && rayon::current_num_threads() > 1
    {
        crate::epool::epool()
    } else {
        None
    };
    let pipelined_leaves = helper.is_some();
    let mut prehashed_tree = helper.map(|_| {
        // Ranked: 64 MiB flat tree. Allocation is uninitialized, so only the
        // 32 MiB leaf prefix is page-touched during the NTT; the internal half
        // remains untouched until the normal parent-level build below. This
        // advances allocation lifetime but does not raise the commit's final
        // codeword+tree peak alongside the retained prover scratch pools.
        let total_nodes = 2 * params.n_leaves() - 1;
        crate::alloc_uninit_vec::<Hash>(total_nodes)
    });
    let timing = std::env::var_os("FLOCK_COMMIT_TIMING").is_some();
    let cpu_ntt0 = timing.then(commit_cpu_ms);
    let t_ntt = std::time::Instant::now();
    let mut prehashed_parent_levels = 0usize;
    // ---- Interleaved forward additive NTT: 2^log_batch_size independent
    // sub-NTTs with shared twiddles. Each sub-NTT operates on its lane of the
    // SoA buffer. The first `log_inv_rate` layers were pre-applied by the
    // caller's replicate-fill (commit_into), so start past them.
    let ntt = AdditiveNttF128::standard(params.k_code());
    if let (Some(helper), Some(tree)) = (helper, prehashed_tree.as_mut()) {
        prehashed_parent_levels =
            ranked_ntt_with_pipelined_leaves(&ntt, codeword, params, tree, helper, from_message);
    } else {
        // No pipeline → no split ranked top; materialize the replicas the
        // ordinary way before the full transform.
        if let Some(msg) = from_message {
            replicate_message_fill(codeword, msg);
        }
        ntt.forward_transform_interleaved_from_layer(
            codeword,
            params.num_ntts(),
            params.log_inv_rate,
        );
    }
    if timing {
        let phase = if pipelined_leaves {
            "ntt+merkle-leaves"
        } else {
            "ntt"
        };
        let wall = t_ntt.elapsed().as_secs_f64() * 1e3;
        let cpu = commit_cpu_ms() - cpu_ntt0.unwrap_or(0.0);
        eprintln!("[commit-timing] {phase}: {wall:.2} ms cpu={cpu:.1}");
    }
    let cpu_merkle0 = timing.then(commit_cpu_ms);
    let t_merkle = std::time::Instant::now();

    // Initial tree: one leaf per codeword position, each containing the
    // row-batch lanes (num_ntts F_{2^128} values = 2^log_batch_size). This is
    // Ligerito's L0 commitment.
    let merkle_tree = if let Some(tree) = prehashed_tree {
        merkle::merkle_tree_from_prehashed_level(
            tree,
            params.n_leaves(),
            params.merkle_hash,
            prehashed_parent_levels,
        )
    } else {
        // Zero-copy: F128 is repr(C, align(16)) with two u64s laid out
        // little-endian, matching the canonical leaf serialization.
        let codeword_bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(
                codeword.as_ptr() as *const u8,
                codeword.len() * core::mem::size_of::<F128>(),
            )
        };
        merkle::merkle_tree(codeword_bytes, params.n_leaves(), params.merkle_hash)
    };
    if timing {
        let phase = if pipelined_leaves {
            "merkle-parents"
        } else {
            "merkle"
        };
        let wall = t_merkle.elapsed().as_secs_f64() * 1e3;
        let cpu = commit_cpu_ms() - cpu_merkle0.unwrap_or(0.0);
        eprintln!("[commit-timing] {phase}: {wall:.2} ms cpu={cpu:.1}");
    }

    merkle_tree
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
    // Warmup selected persistent Metal staging: do not pull the unused
    // 1 GiB CPU codeword into the timed witness phase.
    if crate::gpu_commit::gpu_commit_latched_on() && use_ranked_from_message_commit(params) {
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
        fn f128(&mut self) -> F128 {
            F128 {
                lo: self.next_u64(),
                hi: self.next_u64(),
            }
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

    #[test]
    fn ranked_ntt_merkle_pipeline_gate_is_narrow() {
        let params = PcsParams {
            m: 32,
            log_inv_rate: 1,
            log_batch_size: 6,
            profile: crate::pcs::ligerito::LigeritoProfile::Fast,
            merkle_hash: HashKind::Blake3,
        };
        let enabled_here = cfg!(all(
            target_os = "macos",
            target_arch = "aarch64",
            target_feature = "aes"
        ));
        assert_eq!(
            is_ranked_ntt_merkle_leaf_pipeline_shape(&params),
            enabled_here
        );

        let mut changed = params.clone();
        changed.m = 31;
        assert!(!is_ranked_ntt_merkle_leaf_pipeline_shape(&changed));
        let mut changed = params.clone();
        changed.log_inv_rate = 2;
        assert!(!is_ranked_ntt_merkle_leaf_pipeline_shape(&changed));
        let mut changed = params.clone();
        changed.log_batch_size = 5;
        assert!(!is_ranked_ntt_merkle_leaf_pipeline_shape(&changed));
        let mut changed = params.clone();
        changed.profile = crate::pcs::ligerito::LigeritoProfile::Secure;
        assert!(!is_ranked_ntt_merkle_leaf_pipeline_shape(&changed));
        let mut changed = params;
        changed.merkle_hash = HashKind::Sha256;
        assert!(!is_ranked_ntt_merkle_leaf_pipeline_shape(&changed));
    }

    #[test]
    fn pipelined_ntt_leaves_match_separate_oracle() {
        // log_d=12 is large enough to split into multiple finalized chunks on
        // ordinary test hosts. This exercises concurrent helper receives,
        // bounded-queue overflow to inline hashing, and the post-NTT tail
        // drain, rather than only the scalar one-callback fallback.
        let params = PcsParams {
            m: 24,
            log_inv_rate: 1,
            log_batch_size: 6,
            profile: crate::pcs::ligerito::LigeritoProfile::Fast,
            merkle_hash: HashKind::Blake3,
        };
        let mut rng = Rng::new(0xCA5E_10CA_C011_1701);
        let message: Vec<F128> = (0..1usize << params.log_msg_len())
            .map(|_| rng.f128())
            .collect();

        let mut source = vec![F128::ZERO; params.codeword_len_f128()];
        replicate_message_fill(&mut source, &message);
        let ntt = AdditiveNttF128::standard(params.k_code());

        let mut expect_codeword = source.clone();
        ntt.forward_transform_interleaved_from_layer(
            &mut expect_codeword,
            params.num_ntts(),
            params.log_inv_rate,
        );
        let expect_bytes = unsafe {
            core::slice::from_raw_parts(
                expect_codeword.as_ptr().cast::<u8>(),
                core::mem::size_of_val(expect_codeword.as_slice()),
            )
        };
        let expect_tree = merkle::merkle_tree(expect_bytes, params.n_leaves(), HashKind::Blake3);

        let helper = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let mut got_codeword = source;
        let mut got_tree = vec![[0u8; 32]; 2 * params.n_leaves() - 1];
        let prehashed_parent_levels = ranked_ntt_with_pipelined_leaves(
            &ntt,
            &mut got_codeword,
            &params,
            &mut got_tree,
            &helper,
            None,
        );
        let got_tree = merkle::merkle_tree_from_prehashed_level(
            got_tree,
            params.n_leaves(),
            HashKind::Blake3,
            prehashed_parent_levels,
        );

        assert_eq!(got_codeword, expect_codeword);
        assert_eq!(got_tree, expect_tree);
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

            assert!(
                pd.codeword[..] == oracle[..],
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
        for (i, (&c1, &c2)) in pd1.codeword.iter().zip(pd2.codeword.iter()).enumerate() {
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
}
