//! Bit-packing and R1CS-row helpers shared by the monolithic hash R1CS
//! modules (`sha2`, `blake3`, `keccak`). The shared `prove_fast`
//! orchestration lives in [`crate::prover::prove_fast_from_witness`].

use std::sync::OnceLock;

use flock_core::bits::transpose_8_u64s_to_64_bytes;
use flock_core::field::F128;
use flock_core::r1cs::{BlockR1cs, SparseBinaryMatrix, WitnessLayout};

/// OR the low 32 bits of `val` into `buf` starting at bit-offset `bit_off`.
/// Handles u64 straddling when `bit_off % 64 > 32`.
#[inline(always)]
pub(crate) fn or_u32_at_bit(buf: &mut [u64], bit_off: usize, val: u32) {
    let u64_idx = bit_off >> 6;
    let shift = bit_off & 63;
    buf[u64_idx] |= (val as u64) << shift;
    if shift > 32 {
        buf[u64_idx + 1] |= (val as u64) >> (64 - shift);
    }
}

/// Set bit `bit_off` of `buf` (low-bit-first within each u64).
#[inline(always)]
pub(crate) fn or_bit_at(buf: &mut [u64], bit_off: usize) {
    buf[bit_off >> 6] |= 1u64 << (bit_off & 63);
}

/// A `64·NW`-bit record composed in registers and OR-flushed into the block
/// buffer once.
///
/// Hash witness builders write groups of adjacent sub-word fields (e.g.
/// 31-bit carry slots) with `or_u32_at_bit`; back-to-back fields hit the
/// same u64 word, serializing on store-to-load forwarding, with a straddle
/// branch per call. Composing the group in registers (const positions,
/// branchless) and flushing with one `NW + 1`-word shifted OR pass turns
/// ~2 read-modify-writes per field into `NW + 1` per group.
pub(crate) struct BitRecord<const NW: usize> {
    w: [u64; NW],
}

impl<const NW: usize> BitRecord<NW> {
    #[inline(always)]
    pub(crate) fn new() -> Self {
        Self { w: [0u64; NW] }
    }

    #[inline(always)]
    pub(crate) fn words(&self) -> &[u64; NW] {
        &self.w
    }

    /// OR a (pre-masked) value into record bits `[POS, POS + width)`.
    /// `POS` is const so the straddle branch and shifts fold at compile time.
    #[inline(always)]
    pub(crate) fn push<const POS: usize>(&mut self, val: u32) {
        let v = val as u64;
        let idx = POS >> 6;
        let s = POS & 63;
        self.w[idx] |= v << s;
        if s > 32 {
            self.w[idx + 1] |= v >> (64 - s);
        }
    }

    /// OR the record into `buf` starting at bit `base_bit`.
    #[inline(always)]
    pub(crate) fn flush(&self, buf: &mut [u64], base_bit: usize) {
        let bi = base_bit >> 6;
        let s = base_bit & 63;
        let mut spill = 0u64;
        for j in 0..NW {
            buf[bi + j] |= (self.w[j] << s) | spill;
            // `(x >> 1) >> (63 - s)` = `x >> (64 - s)` without the s = 0 UB.
            spill = (self.w[j] >> 1) >> (63 - s);
        }
        buf[bi + NW] |= spill;
    }
}

/// One 32-bit ADD's witness parts: `(sum, left, right, carry_aux)` with
/// `left/right/carry_aux` masked to the low 31 bits (bit 31 is the discarded
/// mod-2³² carry-out; the carry slot is 31 bits wide).
#[inline(always)]
pub(crate) fn add_carry_parts(x: u32, y: u32) -> (u32, u32, u32, u32) {
    let sum = x.wrapping_add(y);
    let cin = sum ^ x ^ y;
    const MASK_LO31: u32 = 0x7FFF_FFFF;
    let left = (x ^ cin) & MASK_LO31;
    let right = (y ^ cin) & MASK_LO31;
    let carry_aux = left & right;
    (sum, left, right, carry_aux)
}

// ---------------------------------------------------------------------------
// Shared R1CS helpers: empty matrix, identity, BlockR1cs stub builder.
//
// The K_LOG=16 hash encoders all use empty A_0/B_0 matrices (constraint
// definition lives in their LincheckCircuit walkers) and C_0 = I_K. These
// three helpers were duplicated across keccak.rs, blake3.rs, sha2.rs.
// ---------------------------------------------------------------------------

/// K × K sparse matrix with no nonzero entries. Used as an `a_0`/`b_0` stub
/// when the constraint definition lives in a `LincheckCircuit` walker.
pub(crate) fn empty_matrix(k: usize) -> SparseBinaryMatrix {
    SparseBinaryMatrix {
        num_rows: k,
        num_cols: k,
        rows: vec![Vec::new(); k],
    }
}

/// K × K identity sparse matrix.
pub(crate) fn identity(k: usize) -> SparseBinaryMatrix {
    SparseBinaryMatrix {
        num_rows: k,
        num_cols: k,
        rows: (0..k).map(|i| vec![i]).collect(),
    }
}

/// Build a `BlockR1cs` shell with empty A_0, B_0 stubs and C_0 = I_K. The
/// constraint definition lives in a per-hash `LincheckCircuit` walker. Used
/// by Keccak.
pub(crate) fn build_block_r1cs_empty_stub(
    n_blocks_log: usize,
    k_log: usize,
    k_skip: usize,
    useful_bits: usize,
) -> BlockR1cs {
    let k = 1usize << k_log;
    // Empty-stub R1CS carry their constraints (and constant-wire pin) on a
    // per-hash `LincheckCircuit` walker, so no R1CS-level `const_pin` here.
    build_block_r1cs_with_matrices(
        n_blocks_log,
        k_log,
        k_skip,
        useful_bits,
        empty_matrix(k),
        empty_matrix(k),
        None,
    )
}

/// Build a `BlockR1cs` with caller-supplied A_0, B_0 sparse matrices and
/// C_0 = I_K. Used by BLAKE3 and SHA-2 (they materialize real A_0/B_0 via
/// their `build_matrices`).
///
/// `useful_bits ≤ 2^k_log` declares how many rows of each block carry real
/// data; the remainder is zero padding (URM can skip work over those).
///
/// `const_pin` is the column of the constant-one wire to pin to 1 across all
/// blocks (closing the all-zero soundness gap — see `docs/const-wire-pin.md`),
/// or `None`. It is propagated into the CSC / sparse `LincheckCircuit` this
/// R1CS builds. Encoders that set it MUST fill padding blocks with valid
/// (constant = 1) computations.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_block_r1cs_with_matrices(
    n_blocks_log: usize,
    k_log: usize,
    k_skip: usize,
    useful_bits: usize,
    a_0: SparseBinaryMatrix,
    b_0: SparseBinaryMatrix,
    const_pin: Option<usize>,
) -> BlockR1cs {
    assert!(
        n_blocks_log >= 3,
        "lincheck needs n_outer ≥ 8 — pick n_blocks_log ≥ 3"
    );
    let k = 1usize << k_log;
    assert!(
        useful_bits <= k,
        "useful_bits ({useful_bits}) must be ≤ 2^k_log ({k})"
    );
    BlockR1cs {
        m: k_log + n_blocks_log,
        k_log,
        k_skip,
        useful_bits,
        a_0,
        b_0,
        c_0: identity(k),
        layout: WitnessLayout::RowMajor,
        const_pin,
        digest_cache: OnceLock::new(),
        csc_cache: OnceLock::new(),
    }
}

// ---------------------------------------------------------------------------
// Generic witness packing driver.
//
// All three hash encoders (keccak, blake3, sha2) had identical chunked
// parallel iteration + bit-transpose-to-stripe boilerplate around their
// per-block witness builder. This driver captures that shape; each hash
// passes its `per_block` closure that fills 3 length-`U64_PER_BLOCK`
// buffers (z, a, b) from one input.
// ---------------------------------------------------------------------------

/// Drive the parallel chunked witness build for `n_blocks` instances padded
/// to `2^n_blocks_log` slots. Returns `(z, a, b, z_lincheck)` packed in
/// F128 form (z/a/b) and byte-stripe form (z_lincheck).
///
/// `per_block(initial, z_u64, a_u64, b_u64)` populates one block's worth of
/// `(z, a, b)` data — 3 zero-initialized `u64`-buffers of length `K / 64`.
/// `K` is derived from `k_log`. `initial_states.len()` may be less than
/// `2^n_blocks_log`.
///
/// `padding` controls what fills the trailing `2^n_blocks_log −
/// initial_states.len()` slots:
/// - `None` — leave them all-zero (trivial constraint satisfaction).
/// - `Some(p)` — build a real block from `p` in every padding slot. Encoders
///   that pin a constant wire need this so the constant column is all-ones
///   across *every* batched instance (see `docs/const-wire-pin.md`); for keccak
///   the padding input is the all-zero state, whose witness is `keccak_f(0)`.
pub(crate) fn drive_witness_packed_and_lincheck<S: Sync, F>(
    initial_states: &[S],
    padding: Option<&S>,
    n_blocks_log: usize,
    k_log: usize,
    per_block: F,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>)
where
    F: Fn(&S, &mut [u64], &mut [u64], &mut [u64]) + Sync,
{
    let (z, a, b, stripe, stream) = drive_witness_packed_and_lincheck_impl::<false, false, S, F>(
        initial_states,
        padding,
        n_blocks_log,
        k_log,
        1usize << k_log,
        None,
        None,
        per_block,
    );
    debug_assert!(stream.is_none());
    (z, a, b, stripe)
}

/// Full-write variant of [`drive_witness_packed_and_lincheck`]. It skips the
/// group zero pass, so `per_block` MUST overwrite every word of all three
/// buffers before returning. Padding is mandatory: this ensures the callback
/// runs for every allocated slot, including the tail of a non-power-of-two
/// input batch.
pub(crate) fn drive_witness_packed_and_lincheck_full_write<S: Sync, F>(
    initial_states: &[S],
    padding: &S,
    n_blocks_log: usize,
    k_log: usize,
    useful_bits: usize,
    per_block: F,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>)
where
    F: Fn(&S, &mut [u64], &mut [u64], &mut [u64]) + Sync,
{
    let (z, a, b, stripe, stream) = drive_witness_packed_and_lincheck_impl::<true, false, S, F>(
        initial_states,
        Some(padding),
        n_blocks_log,
        k_log,
        useful_bits,
        None,
        None,
        per_block,
    );
    debug_assert!(stream.is_none());
    (z, a, b, stripe)
}

/// Ranked full-write witness driver that publishes eight coarse readiness
/// bands to the latched Metal from-`z` pass. Returns `None` for the stream on
/// warmup/non-Metal/fallback paths while preserving identical witness output.
pub(crate) fn drive_witness_packed_and_lincheck_full_write_streamed<S: Sync, F>(
    initial_states: &[S],
    padding: &S,
    n_blocks_log: usize,
    k_log: usize,
    useful_bits: usize,
    pcs_params: &flock_core::pcs::PcsParams,
    per_block: F,
) -> (
    Vec<F128>,
    Vec<F128>,
    Vec<F128>,
    Vec<u8>,
    Option<flock_core::gpu_commit::FromZFirstPassStream>,
)
where
    F: Fn(&S, &mut [u64], &mut [u64], &mut [u64]) + Sync,
{
    drive_witness_packed_and_lincheck_impl::<true, false, S, F>(
        initial_states,
        Some(padding),
        n_blocks_log,
        k_log,
        useful_bits,
        None,
        Some(pcs_params),
        per_block,
    )
}

#[derive(Clone, Copy)]
struct Rate2CodewordPtr(*mut F128);
// SAFETY: the only use is the indexed group writer below. Each group owns
// disjoint ranges in both replicas, and the parallel iterator joins before the
// original mutable codeword borrow becomes usable again.
unsafe impl Send for Rate2CodewordPtr {}
unsafe impl Sync for Rate2CodewordPtr {}
impl Rate2CodewordPtr {
    /// Avoid closure field-capture turning this back into a bare non-Send ptr.
    fn get(self) -> *mut F128 {
        self.0
    }
}

#[inline(always)]
fn ranked_stream_group_index(
    job: usize,
    band: usize,
    groups_per_segment: usize,
    groups_per_band: usize,
) -> usize {
    let segment = job / groups_per_band;
    let local = job % groups_per_band;
    segment * groups_per_segment + band * groups_per_band + local
}

/// Full-write row-major witness driver that also emits the exact rate-1/2
/// pre-NTT codeword `[z, z]`. Each worker copies its completed `z` group into
/// the two disjoint replica ranges while that group is still cache-resident.
///
/// `codeword` must have exactly twice the packed-witness length. As with the
/// scratch buffers allocated by the driver, its old contents may be stale:
/// every element is overwritten before this function returns.
pub(crate) fn drive_witness_packed_and_lincheck_full_write_with_rate2_codeword<S: Sync, F>(
    initial_states: &[S],
    padding: &S,
    n_blocks_log: usize,
    k_log: usize,
    useful_bits: usize,
    codeword: &mut [F128],
    per_block: F,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>)
where
    F: Fn(&S, &mut [u64], &mut [u64], &mut [u64]) + Sync,
{
    let (z, a, b, stripe, stream) = drive_witness_packed_and_lincheck_impl::<true, true, S, F>(
        initial_states,
        Some(padding),
        n_blocks_log,
        k_log,
        useful_bits,
        Some(codeword),
        None,
        per_block,
    );
    debug_assert!(stream.is_none());
    (z, a, b, stripe)
}

/// Kill switch for the hetero (E-core) drain of the witness-generation
/// groups: `FLOCK_NO_WITGEN_HETERO=1` restores the incumbent main-pool rayon
/// pass. Bit-identical output either way — the drain only changes *which*
/// core processes a group, never what the group writes.
fn witgen_hetero_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_WITGEN_HETERO").is_none());
    *ON
}

/// Engagement tracer: `FLOCK_WITGEN_HETERO_TRACE=1` prints the helper-pool
/// chunk-claim delta around each witness drain (epool::helper_chunks_claimed
/// pattern — engagement evidence for hetero claims).
pub(crate) fn witgen_hetero_trace() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_WITGEN_HETERO_TRACE").is_some());
    *ON
}

/// Oracle/debug arm: `FLOCK_WITGEN_HETERO_MAIN_ONLY=1` drains the groups
/// through the shared atomic queue but WITHOUT the efficiency-core pool —
/// isolates queue-vs-slab scheduling from the E-core addition.
fn witgen_hetero_main_only() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_WITGEN_HETERO_MAIN_ONLY").is_some());
    *ON
}

/// Slab size for the hetero drain: one claim covers WITGEN_HETERO_SLAB
/// consecutive groups, preserving the incumbent's long ascending store runs
/// per worker while still letting the efficiency cores pull slabs. 64 groups
/// = 4 MiB of z per slab at K_LOG=14; 32,768 groups → 512 slabs ≫
/// EPOOL_MIN_CHUNKS. (Measured on b844d53: single-group claims interleave 14
/// writers' store streams and cost +2.5 ms of pure scheduling damage;
/// 64-group slabs recover it and the E-cores convert −1.3 ms.)
pub(crate) const WITGEN_HETERO_SLAB: usize = 64;

/// Oracle arm: `FLOCK_WITGEN_HETERO_SINGLE=1` drains single groups per claim
/// (the naive `run_hetero_chunks` shape) instead of 64-group slabs.
fn witgen_hetero_single() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_WITGEN_HETERO_SINGLE").is_some());
    *ON
}

/// W-H1 scheduling shim: drain `n_jobs` group jobs either on the incumbent
/// main-pool sweep (`with_max_len(256)`) or through the hetero E-core queue
/// in 64-group slabs. Group `g` owns disjoint ranges in every witness
/// buffer, so the output is byte-identical regardless of which pool claims
/// a group; the drain's completion join publishes every write.
pub(crate) fn drain_group_jobs<F>(n_jobs: usize, f: &F)
where
    F: Fn(usize) + Sync,
{
    use rayon::prelude::*;
    if !witgen_hetero_enabled() {
        (0..n_jobs).into_par_iter().with_max_len(256).for_each(f);
        return;
    }
    if witgen_hetero_main_only() {
        flock_core::epool::run_chunks_with_helper(n_jobs, f, None);
    } else if witgen_hetero_single() {
        flock_core::epool::run_hetero_chunks(n_jobs, f);
    } else {
        let n_slabs = n_jobs.div_ceil(WITGEN_HETERO_SLAB);
        flock_core::epool::run_hetero_chunks(n_slabs, &|s: usize| {
            let lo = s * WITGEN_HETERO_SLAB;
            let hi = (lo + WITGEN_HETERO_SLAB).min(n_jobs);
            for g in lo..hi {
                f(g);
            }
        });
    }
}

/// Slab-granular sibling of [`drain_group_jobs`] for the continuous-queue
/// streamed drain (blake3 witgen item A): `f` receives SLAB indices — the
/// caller owns the slab→groups mapping and any per-slab completion
/// accounting (band counters + in-order Metal submits) — claimed one at a
/// time from a single shared queue in ascending order. Same pool selection
/// as [`drain_group_jobs`]: hetero P+E queue by default,
/// `FLOCK_WITGEN_HETERO_MAIN_ONLY=1` for the queue-without-E-cores oracle,
/// `FLOCK_NO_WITGEN_HETERO=1` for a main-pool-only claim loop (still
/// single-queue so the caller's completion accounting stays exact).
pub(crate) fn drain_witgen_slabs<F>(n_slabs: usize, f: &F)
where
    F: Fn(usize) + Sync,
{
    if !witgen_hetero_enabled() || witgen_hetero_main_only() {
        flock_core::epool::run_chunks_with_helper(n_slabs, f, None);
    } else {
        flock_core::epool::run_hetero_chunks(n_slabs, f);
    }
}

fn drive_witness_packed_and_lincheck_impl<
    const PER_BLOCK_FULLY_WRITES: bool,
    const EMIT_RATE2_CODEWORD: bool,
    S: Sync,
    F,
>(
    initial_states: &[S],
    padding: Option<&S>,
    n_blocks_log: usize,
    k_log: usize,
    stripe_useful_bits: usize,
    rate2_codeword: Option<&mut [F128]>,
    stream_params: Option<&flock_core::pcs::PcsParams>,
    per_block: F,
) -> (
    Vec<F128>,
    Vec<F128>,
    Vec<F128>,
    Vec<u8>,
    Option<flock_core::gpu_commit::FromZFirstPassStream>,
)
where
    F: Fn(&S, &mut [u64], &mut [u64], &mut [u64]) + Sync,
{
    let k = 1usize << k_log;
    let f128_per_block = k / 128;
    let u64_per_block = k / 64;
    let n_total = 1usize << n_blocks_log;
    let n_blocks = initial_states.len();
    assert!(stripe_useful_bits <= k);
    assert!(
        n_blocks <= n_total,
        "{n_blocks} blocks > 2^{n_blocks_log} = {n_total} slots"
    );
    assert!(
        n_total >= 8 && n_total.is_multiple_of(8),
        "lincheck stripe layout requires n_total ≥ 8 and divisible by 8"
    );
    assert!(
        !PER_BLOCK_FULLY_WRITES || padding.is_some(),
        "full-write witness generation requires a padding block"
    );

    let total_f128 = n_total * f128_per_block;
    assert_eq!(
        rate2_codeword.is_some(),
        EMIT_RATE2_CODEWORD,
        "rate-1/2 codeword presence must match the driver specialization"
    );
    let rate2_codeword = rate2_codeword.map(|codeword| {
        assert_eq!(
            codeword.len(),
            2 * total_f128,
            "rate-1/2 codeword must contain exactly two packed-witness replicas"
        );
        Rate2CodewordPtr(codeword.as_mut_ptr())
    });
    // z/a/b are allocated uninitialized. Ordinary OR-based builders zero each
    // 8-block group inside the parallel loop; full-write builders initialize
    // every word directly and skip that pass. `z_lincheck` comes from the
    // scratch byte pool (UNINITIALIZED, possibly stale): the transpose below
    // writes every byte of every group before anything reads it, and the
    // caller returns it via `scratch::give_u8` after lincheck so the next
    // prove reuses resident pages instead of re-faulting 2^(m-3) bytes.
    let mut z = flock_core::scratch::take_f128(total_f128);
    let mut a = flock_core::scratch::take_f128(total_f128);
    let mut b = flock_core::scratch::take_f128(total_f128);
    let mut z_lincheck = flock_core::scratch::take_u8((n_total / 8) * k);

    let mut stream = stream_params.and_then(|params| {
        // SAFETY: z's allocation/address stays fixed until the returned stream
        // is consumed by commit. No range is submitted until all eight source
        // segments for it have been fully initialized below.
        unsafe {
            flock_core::gpu_commit::begin_from_z_first_pass_stream(z.as_mut_ptr(), z.len(), params)
        }
    });
    // The range-to-witness mapping below is the ranked BLAKE3 geometry:
    // 2^18 blocks × 2^14 bits, grouped by eight blocks, with eight from-z
    // source segments. A mismatched caller simply retains the ordinary loop.
    if stream.is_some() && (n_total != 1 << 18 || k_log != 14 || EMIT_RATE2_CODEWORD) {
        stream = None;
    }

    #[derive(Clone, Copy)]
    struct F128WritePtr(*mut F128);
    unsafe impl Send for F128WritePtr {}
    unsafe impl Sync for F128WritePtr {}
    impl F128WritePtr {
        fn get(self) -> *mut F128 {
            self.0
        }
    }
    #[derive(Clone, Copy)]
    struct U8WritePtr(*mut u8);
    unsafe impl Send for U8WritePtr {}
    unsafe impl Sync for U8WritePtr {}
    impl U8WritePtr {
        fn get(self) -> *mut u8 {
            self.0
        }
    }

    let group_f128 = 8 * f128_per_block;
    let z_base = F128WritePtr(z.as_mut_ptr());
    let a_base = F128WritePtr(a.as_mut_ptr());
    let b_base = F128WritePtr(b.as_mut_ptr());
    let stripe_base = U8WritePtr(z_lincheck.as_mut_ptr());

    let process_group = |g: usize| {
        // SAFETY: each scheduled group index occurs exactly once. Every
        // group owns disjoint z/a/b ranges and one disjoint stripe.
        let (z_grp, a_grp, b_grp, stripe) = unsafe {
            (
                std::slice::from_raw_parts_mut(z_base.get().add(g * group_f128), group_f128),
                std::slice::from_raw_parts_mut(a_base.get().add(g * group_f128), group_f128),
                std::slice::from_raw_parts_mut(b_base.get().add(g * group_f128), group_f128),
                std::slice::from_raw_parts_mut(stripe_base.get().add(g * k), k),
            )
        };
        // Ordinary per-block builders OR 1-bits into pre-zeroed words; any
        // slot left unbuilt (no padding block) stays zero, which the
        // lincheck transpose below reads correctly. Full-write builders
        // skip this pass and must initialize every word themselves.
        // SAFETY: F128 is `Copy` (no Drop) and the all-zero bit pattern is
        // the valid `F128::ZERO`, so a byte memset is a correct init.
        if !PER_BLOCK_FULLY_WRITES {
            unsafe {
                std::ptr::write_bytes(z_grp.as_mut_ptr(), 0, z_grp.len());
                std::ptr::write_bytes(a_grp.as_mut_ptr(), 0, a_grp.len());
                std::ptr::write_bytes(b_grp.as_mut_ptr(), 0, b_grp.len());
            }
        }
        for k_in in 0..8 {
            let global_idx = 8 * g + k_in;
            let init: &S = if global_idx < n_blocks {
                &initial_states[global_idx]
            } else if let Some(p) = padding {
                // Fill the padding slot with a real block so its constant
                // wire is set (see `padding` docs above).
                p
            } else {
                // No padding block — leave this slot zero.
                continue;
            };
            let z_chunk = &mut z_grp[k_in * f128_per_block..(k_in + 1) * f128_per_block];
            let a_chunk = &mut a_grp[k_in * f128_per_block..(k_in + 1) * f128_per_block];
            let b_chunk = &mut b_grp[k_in * f128_per_block..(k_in + 1) * f128_per_block];
            // SAFETY: F128 is `repr(C, align(16))` with two `u64` fields in
            // LE order — same byte layout as a u64 pair.
            let z_u64: &mut [u64] = unsafe {
                std::slice::from_raw_parts_mut(z_chunk.as_mut_ptr() as *mut u64, z_chunk.len() * 2)
            };
            let a_u64: &mut [u64] = unsafe {
                std::slice::from_raw_parts_mut(a_chunk.as_mut_ptr() as *mut u64, a_chunk.len() * 2)
            };
            let b_u64: &mut [u64] = unsafe {
                std::slice::from_raw_parts_mut(b_chunk.as_mut_ptr() as *mut u64, b_chunk.len() * 2)
            };
            per_block(init, z_u64, a_u64, b_u64);
        }

        // Bit-transpose 8 z chunks into the lincheck stripe.
        let z_u64_all: &[u64] =
            unsafe { std::slice::from_raw_parts(z_grp.as_ptr() as *const u64, z_grp.len() * 2) };
        // Padded lincheck fold reads only stripe[..useful_bits]
        // (`partial_fold_packed_z_fast_padded`). Ranked Blake3 defaults
        // useful_bits=USEFUL_BITS (15409) with k=16384, so the tail past
        // useful_words*64 is never observed on the timed path.
        //
        // `take_u8` is write-before-read / stale-pool (scratch.rs): skipping
        // the tail memset leaves pool garbage in stripe[useful_words*64..].
        // Production fold never loads that range; unit oracles that compare
        // full-k stripes still need an honest zero pad. Gate the memset on
        // `cfg(test)` so release/ranked proves elide ~960 B/stripe × n_stripes
        // while `cargo test` keeps the full-stripe contract.
        let useful_words = stripe_useful_bits.div_ceil(64);
        for i in 0..useful_words {
            let lanes: [u64; 8] = [
                z_u64_all[0 * u64_per_block + i],
                z_u64_all[u64_per_block + i],
                z_u64_all[2 * u64_per_block + i],
                z_u64_all[3 * u64_per_block + i],
                z_u64_all[4 * u64_per_block + i],
                z_u64_all[5 * u64_per_block + i],
                z_u64_all[6 * u64_per_block + i],
                z_u64_all[7 * u64_per_block + i],
            ];
            transpose_8_u64s_to_64_bytes(&lanes, &mut stripe[i * 64..i * 64 + 64]);
        }
        #[cfg(test)]
        {
            stripe[useful_words * 64..].fill(0);
        }

        if EMIT_RATE2_CODEWORD {
            let codeword =
                rate2_codeword.expect("rate-1/2 driver specialization requires a codeword");
            let elem_offset = g * z_grp.len();
            // SAFETY: group `g` owns `z[elem_offset..elem_offset+len]`.
            // The same group exclusively owns those offsets in each of the
            // two codeword replicas. Groups are disjoint, cover all of z,
            // and the parallel iterator joins before `codeword` is used
            // again. Both source and destinations are in bounds and do not
            // overlap.
            unsafe {
                let dst = codeword.get();
                std::ptr::copy_nonoverlapping(z_grp.as_ptr(), dst.add(elem_offset), z_grp.len());
                std::ptr::copy_nonoverlapping(
                    z_grp.as_ptr(),
                    dst.add(total_f128 + elem_offset),
                    z_grp.len(),
                );
            }
        }
    };

    let n_groups = n_total / 8;
    // W-H1 engagement evidence: E-core slab claims across the whole witness
    // drain (both streamed bands and the plain sweep route through the same
    // shim, so one delta covers whichever arm runs).
    let claimed_before = witgen_hetero_trace().then(flock_core::epool::helper_chunks_claimed);
    if let Some(stream) = &mut stream {
        const SEGMENTS: usize = 8;
        const BANDS: usize = 8;
        let groups_per_segment = n_groups / SEGMENTS;
        let groups_per_band = groups_per_segment / BANDS;
        let r_total = 1usize << 16;
        let r_per_band = r_total / BANDS;
        debug_assert_eq!(groups_per_segment, 4096);
        debug_assert_eq!(groups_per_band, 512);
        for band in 0..BANDS {
            // W-H1: the band's jobs drain through the same slab shim. A slab
            // never straddles a segment (512 % 64 == 0), so consecutive jobs
            // stay consecutive groups and the long ascending store runs are
            // preserved. The queue join below still bounds the band before
            // submission, exactly like the incumbent Rayon join.
            let band_job = |job: usize| {
                let g = ranked_stream_group_index(job, band, groups_per_segment, groups_per_band);
                process_group(g);
            };
            drain_group_jobs(SEGMENTS * groups_per_band, &band_job);
            // The queue/Rayon join above publishes every CPU write in this
            // band; command-buffer submission then makes those shared-memory
            // pages visible to Metal before it starts the range.
            std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
            stream.submit_ready_range(band * r_per_band, r_per_band);
        }
    } else {
        drain_group_jobs(n_groups, &process_group);
    }
    if let Some(before) = claimed_before {
        eprintln!(
            "[witgen-hetero] groups={n_groups} helper-claims={}",
            flock_core::epool::helper_chunks_claimed() - before
        );
    }

    (z, a, b, z_lincheck, stream)
}

#[cfg(test)]
mod streamed_first_pass_tests {
    use super::ranked_stream_group_index;

    #[test]
    fn ranked_bands_cover_each_group_and_every_published_tile_is_ready() {
        const SEGMENTS: usize = 8;
        const BANDS: usize = 8;
        const GROUPS_PER_SEGMENT: usize = 4096;
        const GROUPS_PER_BAND: usize = 512;
        const R_PER_BAND: usize = 8192;
        let mut seen = vec![false; SEGMENTS * GROUPS_PER_SEGMENT];

        for band in 0..BANDS {
            for job in 0..SEGMENTS * GROUPS_PER_BAND {
                let g = ranked_stream_group_index(job, band, GROUPS_PER_SEGMENT, GROUPS_PER_BAND);
                assert!(
                    !std::mem::replace(&mut seen[g], true),
                    "duplicate group {g}"
                );
            }

            // One witness group is eight compressions = sixteen NTT
            // positions. For every tile r published in this band, all eight
            // from-z source segments therefore land in groups completed by
            // the enumeration above.
            for r in band * R_PER_BAND..(band + 1) * R_PER_BAND {
                for segment in 0..SEGMENTS {
                    let g = segment * GROUPS_PER_SEGMENT + r / 16;
                    assert!(seen[g], "band {band} published unreadied group {g}");

                    // Binding z at byte offset r_start*64*sizeof(F128) makes
                    // the kernel's local-r address equal its global address.
                    for lane in [0usize, 63] {
                        let local_r = r - band * R_PER_BAND;
                        let offset_elems = band * R_PER_BAND * 64;
                        let local = ((segment << 16) + local_r) * 64 + lane;
                        let global = ((segment << 16) + r) * 64 + lane;
                        assert_eq!(offset_elems + local, global);
                    }
                }
            }
        }
        assert!(seen.into_iter().all(|done| done));
    }
}

/// Sort `v` and remove pairs of duplicates (GF(2) cancellation). Keeps R1CS
/// rows in canonical (sorted, square-free) form.
pub(crate) fn xor_dedup(mut v: Vec<usize>) -> Vec<usize> {
    v.sort();
    let mut out = Vec::with_capacity(v.len());
    let mut i = 0;
    while i < v.len() {
        let val = v[i];
        let mut count = 0;
        while i < v.len() && v[i] == val {
            count += 1;
            i += 1;
        }
        if count % 2 == 1 {
            out.push(val);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Batch-major (WitnessLayout::BatchMajor) witness-producer plumbing.
//
// The batch-major producers simulate V = 8 instances in lockstep and write
// witness words directly at their batch-major addresses: the word-row for
// block-u64 index `w` across the 8 instances is exactly one 128-byte
// chunk-row (= one cache line) at dest word `((w >> 1) << n_log) + o0`,
// stored non-temporally (dest lines are fully overwritten and not re-read
// soon, so write-allocate reads are pure waste). V = 8 also equals the
// lincheck stripe group, so the byte-stripe is transposed from the in-flight
// rows at zero extra reads.
//
// Producer contract: chunk-columns `[0, useful_chunks)` are FULLY written
// every call; the padding suffix `[useful_chunks, k/128)` columns are never
// written (the generators zero that contiguous buffer suffix themselves, so
// recycled scratch buffers stay valid).
// ---------------------------------------------------------------------------

/// Instances per lockstep group (= one lincheck-stripe group; one chunk-row
/// emission = 128 B).
pub(crate) const BM_V: usize = 8;
/// One interleaved word-row: the same block-u64 index across BM_V instances.
pub(crate) type BmRow = [u64; BM_V];

/// Raw-pointer wrapper for the disjoint per-group strided writes.
#[derive(Copy, Clone)]
pub(crate) struct SendPtr(pub *mut u64);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    /// Method (not field) access so `move` closures capture the whole
    /// wrapper — field capture would move the bare `*mut u64` (`!Send`).
    pub(crate) fn get(self) -> *mut u64 {
        self.0
    }
}

/// V-wide `or_u32_at_bit`: OR the V instances' 32-bit values into row `w`
/// (and `w + 1` on straddle) at bit offset `bit`.
#[inline(always)]
pub(crate) fn or_u32_row(rows: &mut [BmRow], bit: usize, vals: &[u32; BM_V]) {
    let w = bit >> 6;
    let s = bit & 63;
    for j in 0..BM_V {
        rows[w][j] |= (vals[j] as u64) << s;
    }
    if s > 32 {
        for j in 0..BM_V {
            rows[w + 1][j] |= (vals[j] as u64) >> (64 - s);
        }
    }
}

/// V-wide `or_bit_at`: set bit `bit` in every instance's row.
#[inline(always)]
pub(crate) fn or_bit_row(rows: &mut [BmRow], bit: usize) {
    let w = bit >> 6;
    let s = bit & 63;
    for j in 0..BM_V {
        rows[w][j] |= 1u64 << s;
    }
}

/// Non-temporal store of one interleaved 128-byte chunk-row.
#[inline(always)]
pub(crate) unsafe fn nt_store_row(src: *const u64, dst: *mut u64) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        std::arch::asm!(
            "ldp {t0:q}, {t1:q}, [{s}]",
            "stnp {t0:q}, {t1:q}, [{d}]",
            "ldp {t0:q}, {t1:q}, [{s}, #32]",
            "stnp {t0:q}, {t1:q}, [{d}, #32]",
            "ldp {t0:q}, {t1:q}, [{s}, #64]",
            "stnp {t0:q}, {t1:q}, [{d}, #64]",
            "ldp {t0:q}, {t1:q}, [{s}, #96]",
            "stnp {t0:q}, {t1:q}, [{d}, #96]",
            s = in(reg) src, d = in(reg) dst,
            t0 = out(vreg) _, t1 = out(vreg) _,
            options(nostack),
        );
    }
    #[cfg(not(target_arch = "aarch64"))]
    unsafe {
        std::ptr::copy_nonoverlapping(src, dst, 2 * BM_V);
    }
}

/// NT-flush `useful_chunks` chunk-rows of an interleaved row buffer to the
/// batch-major destination (dest word index `(c << n_log) + o0`).
///
/// SAFETY: caller guarantees dest sizing and per-group disjointness.
#[inline]
pub(crate) unsafe fn flush_rows_nt(
    rows: &[BmRow],
    dest: *mut u64,
    o0: usize,
    n_log: usize,
    useful_chunks: usize,
) {
    debug_assert!(2 * useful_chunks <= rows.len());
    for c in 0..useful_chunks {
        let even = &rows[2 * c];
        let odd = &rows[2 * c + 1];
        let mut buf = [0u64; 2 * BM_V];
        for j in 0..BM_V {
            buf[2 * j] = even[j];
            buf[2 * j + 1] = odd[j];
        }
        unsafe {
            nt_store_row(buf.as_ptr(), dest.add(((c << n_log) + o0) * 2));
        }
    }
}

/// Transpose the z rows into the lincheck byte-stripe for one V = 8 group.
/// Only `useful_words` rows are written (the stripe tail stays zero).
#[inline]
pub(crate) unsafe fn stripe_from_rows(
    rows: &[BmRow],
    stripe: *mut u8,
    o0: usize,
    u64_per_block: usize,
    useful_words: usize,
) {
    let base = (o0 / 8) * u64_per_block * 64;
    for (w, row) in rows.iter().enumerate().take(useful_words) {
        let out = unsafe { std::slice::from_raw_parts_mut(stripe.add(base + w * 64), 64) };
        transpose_8_u64s_to_64_bytes(row, out);
    }
}

/// V-wide [`add_carry_parts`]: per-instance `(sum, left, right, carry_aux)`.
#[inline(always)]
pub(crate) fn add_carry_parts_v(
    x: &[u32; BM_V],
    y: &[u32; BM_V],
) -> ([u32; BM_V], [u32; BM_V], [u32; BM_V], [u32; BM_V]) {
    const MASK_LO31: u32 = 0x7FFF_FFFF;
    let mut sum = [0u32; BM_V];
    let mut left = [0u32; BM_V];
    let mut right = [0u32; BM_V];
    let mut carry = [0u32; BM_V];
    for j in 0..BM_V {
        let s = x[j].wrapping_add(y[j]);
        let cin = s ^ x[j] ^ y[j];
        let l = (x[j] ^ cin) & MASK_LO31;
        let r = (y[j] ^ cin) & MASK_LO31;
        sum[j] = s;
        left[j] = l;
        right[j] = r;
        carry[j] = l & r;
    }
    (sum, left, right, carry)
}

/// Shared driver for the interleaved-row batch-major producers (sha2,
/// blake3 — the bit-packed encoders): parallel over V-instance groups, each
/// group builds its rows via `per_group(group_inputs, rows)` then NT-flushes
/// the useful chunks and transposes the stripe. Padding slots use `padding`
/// (required, matching the row-major driver's const-wire-pin behavior).
///
/// Returns `(z, a, b, stripe)`; z/a/b come from the scratch pool with the
/// padding suffix zeroed (the producers fully write the useful prefix).
pub(crate) fn drive_witness_batch_major<S: Sync, F>(
    inputs: &[S],
    padding: &S,
    n_blocks_log: usize,
    k_log: usize,
    useful_bits: usize,
    per_group: F,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>)
where
    F: Fn([&S; BM_V], &mut [BmRow], &mut [BmRow], &mut [BmRow]) + Sync + Send,
{
    use rayon::prelude::*;

    let n_total = 1usize << n_blocks_log;
    assert!(inputs.len() <= n_total);
    assert!(n_total >= BM_V);
    let u64_per_block = (1usize << k_log) / 64;
    let useful_chunks = useful_bits.div_ceil(128);
    let useful_words = useful_bits.div_ceil(64);
    let total_f128 = n_total * (u64_per_block / 2);

    let mut z = flock_core::scratch::take_f128(total_f128);
    let mut a = flock_core::scratch::take_f128(total_f128);
    let mut b = flock_core::scratch::take_f128(total_f128);
    let stripe = vec![0u8; n_total * u64_per_block * 8];
    // Zero the padding suffix (contiguous chunk-columns >= useful_chunks);
    // the producers fully rewrite the useful prefix every call.
    let tail = useful_chunks << n_blocks_log;
    for buf in [&mut z, &mut a, &mut b] {
        buf[tail..]
            .par_chunks_mut(1 << 16)
            .for_each(|c| c.fill(F128::ZERO));
    }

    let (zp, ap, bp) = (
        SendPtr(z.as_mut_ptr() as *mut u64),
        SendPtr(a.as_mut_ptr() as *mut u64),
        SendPtr(b.as_mut_ptr() as *mut u64),
    );
    let sp = SendPtr(stripe.as_ptr() as *mut u64);
    let inputs_ref = inputs;

    (0..n_total / BM_V).into_par_iter().for_each_init(
        || {
            (
                vec![[0u64; BM_V]; u64_per_block],
                vec![[0u64; BM_V]; u64_per_block],
                vec![[0u64; BM_V]; u64_per_block],
            )
        },
        move |(rz, ra, rb), g| {
            rz[..useful_words].fill([0u64; BM_V]);
            ra[..useful_words].fill([0u64; BM_V]);
            rb[..useful_words].fill([0u64; BM_V]);
            let o0 = g * BM_V;
            let group: [&S; BM_V] =
                std::array::from_fn(|j| inputs_ref.get(o0 + j).unwrap_or(padding));
            per_group(group, rz, ra, rb);
            // SAFETY: disjoint instance ranges per group; suffix pre-zeroed.
            unsafe {
                flush_rows_nt(rz, zp.get(), o0, n_blocks_log, useful_chunks);
                flush_rows_nt(ra, ap.get(), o0, n_blocks_log, useful_chunks);
                flush_rows_nt(rb, bp.get(), o0, n_blocks_log, useful_chunks);
                stripe_from_rows(rz, sp.get() as *mut u8, o0, u64_per_block, useful_words);
            }
        },
    );

    (z, a, b, stripe)
}
