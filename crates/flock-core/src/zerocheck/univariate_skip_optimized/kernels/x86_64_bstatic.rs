//! Static-B round-1 shift-reduce kernel for x86 AVX-512/GFNI.
//!
//! Port of the mac track's static-B mechanism (its `FLOCK_NO_FAST_SHIFT_REDUCE`
//! kernel) to the fused AVX-512/GFNI kernel this track runs on Sapphire
//! Rapids. The BLAKE3 circuit fixes most bytes of the round-1 **b operand**
//! independently of the block inputs (const-one wires and structural zeros:
//! see [`x86_64_bstatic_plan`]). Because the §2.1 inverse-NTT/LDE table apply
//! `T(b) = ⊕_j π_j(T₀[b_j])` is GF(2)-linear in the packed row, a K-row whose
//! b word matches its plan needs only
//!
//! ```text
//! T(b) = T(expected) ⊕ ⊕_{j ∈ vary} π_j(T₀[b_j])
//! ```
//!
//! i.e. one aligned 64-byte load of a precomputed image plus 0..6 table rows,
//! instead of the full 8-row apply (8 loads + 7 lane permutes + 7 XORs). Rows
//! whose b word is structurally zero contribute nothing and are skipped
//! outright; rows whose b word is fully static additionally take a
//! `x^K`-prescaled image so the `x^K` GFNI multiply disappears too.
//!
//! Every planned row still checks `(b_word & mask) == expected` and falls
//! back to the generic row on a miss, so the output is bit-identical to
//! [`super::x86_64::shift_reduce_inner_ab_x86_avx512`] for ANY witness — the
//! plan is a performance hint, never a correctness assumption. The generic
//! rows are the incumbent computation verbatim.
//!
//! Kill switch: `FLOCK_NO_FAST_SHIFT_REDUCE=1` keeps the incumbent kernel.

use core::arch::x86_64::*;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::super::{F8, InvNttTableByteSingleGf8, N_CHUNKS};
use super::x86_64_bstatic_plan::{
    BSTATIC_BLOCKS, BSTATIC_PLAN, BstaticRow, ROW_GENERIC, ROW_STATIC, ROW_ZERO,
};

/// Precomputed `T(expected)` images, one 64-byte row per planned K-row.
/// Rows whose b word is fully static (`vary == 0`) hold `T(expected) · x^K`
/// (elementwise GF(2⁸)), everything else holds the plain `T(expected)`.
/// Unplanned rows are zero.
#[repr(C, align(64))]
pub(crate) struct BstaticPartials {
    rows: [[[u8; 64]; 8]; BSTATIC_BLOCKS],
    /// Resolved once, at cache build: use the paired two-image apply
    /// ([`apply_full_2img`]) for the a-side row transforms. Process-invariant
    /// (kill switch + table shape), so the per-window kernel reads a plain
    /// bool from an already-resident line instead of entering a `OnceLock`.
    img2: bool,
}

/// Process-wide cache of the partial images together with a fingerprint of the
/// table they were derived from. The eight one-hot rows `T₀[1 << t]`
/// determine `T₀` completely (GF(2)-linearity), so they are the fingerprint.
struct Cached {
    fingerprint: [u8; 8 * 64],
    partials: Box<BstaticPartials>,
}

static CACHE: OnceLock<Cached> = OnceLock::new();
/// `data_ptr()` of the last table verified against `CACHE` — the O(1) fast
/// path for the per-block streaming caller.
static VERIFIED_PTR: AtomicUsize = AtomicUsize::new(0);

fn fingerprint_of(inv_table: &InvNttTableByteSingleGf8) -> [u8; 8 * 64] {
    let mut fp = [0u8; 8 * 64];
    let base = inv_table.data_ptr();
    for t in 0..8 {
        // SAFETY: the table has 256 rows of `ell == 64` readable bytes.
        let row = unsafe { core::slice::from_raw_parts(base.add((1usize << t) * 64), 64) };
        fp[t * 64..(t + 1) * 64].copy_from_slice(row);
    }
    fp
}

fn build_partials(inv_table: &InvNttTableByteSingleGf8) -> Box<BstaticPartials> {
    let mut out = Box::new(BstaticPartials {
        rows: [[[0u8; 64]; 8]; BSTATIC_BLOCKS],
        img2: apply_2img_for(inv_table),
    });
    let mut f = [F8::ZERO; 64];
    for blk in 0..BSTATIC_BLOCKS {
        for k in 0..8 {
            let p = BSTATIC_PLAN[blk][k];
            if p.kind != ROW_STATIC {
                continue;
            }
            inv_table.apply(&p.expected.to_le_bytes(), &mut f);
            let scale = if p.vary == 0 { F8(1u8 << k) } else { F8::ONE };
            for i in 0..64 {
                out.rows[blk][k][i] = (f[i] * scale).0;
            }
        }
    }
    out
}

/// Resolve the static-B partial images for `inv_table`, or `None` when the
/// mechanism is disabled or the table is not the one the cache was built
/// from (a differently-shaped table would make the images meaningless; the
/// caller then runs the incumbent kernel, which is always correct).
pub(crate) fn prepare_bstatic(
    inv_table: &InvNttTableByteSingleGf8,
) -> Option<&'static BstaticPartials> {
    if !fast_shift_reduce_enabled()
        || inv_table.k != 6
        || inv_table.ell != 64
        || inv_table.n_chunks != 8
    {
        return None;
    }
    let ptr = inv_table.data_ptr() as usize;
    if VERIFIED_PTR.load(Ordering::Acquire) == ptr {
        // SAFETY of the logic: VERIFIED_PTR is only ever set after CACHE is
        // initialised and the pointed-to table's fingerprint matched it.
        return CACHE.get().map(|c| &*c.partials);
    }
    let cached = CACHE.get_or_init(|| Cached {
        fingerprint: fingerprint_of(inv_table),
        partials: build_partials(inv_table),
    });
    if cached.fingerprint == fingerprint_of(inv_table)
        && cached.partials.img2 == apply_2img_for(inv_table)
    {
        VERIFIED_PTR.store(ptr, Ordering::Release);
        Some(&*cached.partials)
    } else {
        None
    }
}

/// Whether this table can serve the paired two-image apply. Both terms are
/// process-invariant for a given table, so the result is cached inside
/// [`BstaticPartials`] and re-checked only once per buffer in
/// [`prepare_bstatic`], never per window.
fn apply_2img_for(inv_table: &InvNttTableByteSingleGf8) -> bool {
    bstatic_apply_2img_enabled() && inv_table.has_second_image()
}

/// `FLOCK_NO_BSTATIC_APPLY_2IMG=1` restores the one-image `perm_row` apply in
/// the static-B kernel (same-binary A/B); the ranked worker's cleared env
/// never sets it.
pub(crate) fn bstatic_apply_2img_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_BSTATIC_APPLY_2IMG").is_none())
}

/// `FLOCK_NO_FAST_SHIFT_REDUCE=1` restores the incumbent kernel (same-binary
/// A/B); the ranked worker's cleared env never sets it.
pub(crate) fn fast_shift_reduce_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_FAST_SHIFT_REDUCE").is_none())
}

/// `FLOCK_NO_BSTATIC_28=1` keeps block 28 on the incumbent generic kernel.
/// Ranked env is cleared, so the specialised mixed body runs for blk 28.
fn bstatic_28_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_BSTATIC_28").is_none())
}

/// Blocks whose specialised kernel measured faster than the incumbent on the
/// AVX-512 box (single-thread hot, production entry point, plan-shaped
/// inputs): 0 and 1 (every K-row's b word is all-ones: 0.70×), 30 (seven
/// structurally-zero rows plus one fully static row: 0.23×), 31 (all eight
/// rows zero: 0.10×). Blocks with a mix of static and generic rows measured
/// 1.03–1.16× on this generation of the kernel body (the fully unrolled
/// specialised body pays ~3.5 ns/call over the incumbent's compact loop, more
/// than its partial-row savings recover), so they stay on the incumbent.
#[cfg(test)]
pub(crate) const BSTATIC_LIVE: [bool; BSTATIC_BLOCKS] = {
    let mut t = [false; BSTATIC_BLOCKS];
    t[0] = true;
    t[1] = true;
    t[28] = true;
    t[30] = true;
    t[31] = true;
    t
};

/// Qword permutation `q ↦ q ^ j` — the §2.1 collapse permutation
/// `π_j(i') = i' ⊕ 8j` expressed on 64-bit lanes (lane `c ^ (j >> 1)` plus the
/// odd-`j` 64-bit half swap of the incumbent apply combine into `q ^ j`).
#[repr(C, align(64))]
struct PermIdx([[u64; 8]; 8]);
static PERM_IDX: PermIdx = {
    let mut t = [[0u64; 8]; 8];
    let mut j = 0;
    while j < 8 {
        let mut q = 0;
        while q < 8 {
            t[j][q] = (q ^ j) as u64;
            q += 1;
        }
        j += 1;
    }
    PermIdx(t)
};

/// `π_j` applied to one table row held in a register.
///
/// (Plain `#[inline(always)]`: this module only compiles when `avx512f` is a
/// baseline feature of the build, and `#[inline(always)]` cannot be combined
/// with `#[target_feature]` — rust#145574.)
#[inline(always)]
unsafe fn perm_row(v: __m512i, j: usize) -> __m512i {
    if j == 0 {
        v
    } else {
        // SAFETY: PERM_IDX is 64-byte aligned static data.
        unsafe {
            _mm512_permutexvar_epi64(
                _mm512_load_si512(PERM_IDX.0[j].as_ptr() as *const __m512i),
                v,
            )
        }
    }
}

/// Full inverse-NTT apply of one 8-byte packed row (bit-identical to
/// `InvNttTableByteSingleGf8::apply_x86_avx512_register_unchecked`).
#[inline(always)]
unsafe fn apply_full(table: *const u8, bytes: *const u8) -> __m512i {
    // SAFETY: caller guarantees eight readable bytes at `bytes` and a table of
    // 256 rows × 64 readable bytes.
    unsafe {
        let mut acc = _mm512_loadu_si512(table.add(*bytes as usize * 64) as *const __m512i);
        for j in 1..8 {
            let row = _mm512_loadu_si512(table.add(*bytes.add(j) as usize * 64) as *const __m512i);
            acc = _mm512_xor_si512(acc, perm_row(row, j));
        }
        acc
    }
}

/// Two-image twin of [`apply_full`]: identical value, first butterfly level
/// folded into the loads.
///
/// The apply is `⊕_j σ_{8j}(T₀[b_j])` with `σ_s(v)[i] = v[i ^ s]`, an F₂-linear
/// coordinate permutation composing as `σ_s ∘ σ_t = σ_{s^t}`. Taking the
/// odd-`j` rows from the σ₈ image gives `U_c = T₀[b_{2c}] ⊕ σ₈(T₀[b_{2c+1}])`,
/// and the whole apply is `(U₀ ⊕ σ₃₂(U₂)) ⊕ σ₁₆(U₁ ⊕ σ₃₂(U₃))` — three
/// 128-bit-lane shuffles (σ₁₆ = imm 0xB1, σ₃₂ = imm 0x4E) instead of the
/// one-image form's seven `vpermq`. Same eight loads, same seven XORs; the
/// port-5-only stream per apply drops 7 → 3, and the seven `PERM_IDX` index
/// vectors the one-image form pins in ZMM registers are no longer needed.
///
/// This is [`crate::ntt::InvNttTableByteSingleGf8::apply_x86_avx512_register_2img_unchecked`]
/// specialised to the raw table pointers the static-B kernel already holds.
///
/// # Safety
/// As for [`apply_full`], plus: `table8` must be the σ₈ second image of
/// `table` (`inv_table.half_swapped_data_ptr()`, guarded by
/// `has_second_image()`).
#[inline(always)]
unsafe fn apply_full_2img(table: *const u8, table8: *const u8, bytes: *const u8) -> __m512i {
    // SAFETY: caller guarantees eight readable bytes at `bytes` and two
    // images of 256 rows × 64 readable bytes.
    unsafe {
        let row = |img: *const u8, b: usize| {
            _mm512_loadu_si512(img.add(*bytes.add(b) as usize * 64) as *const __m512i)
        };
        let u0 = _mm512_xor_si512(row(table, 0), row(table8, 1));
        let u1 = _mm512_xor_si512(row(table, 2), row(table8, 3));
        let u2 = _mm512_xor_si512(row(table, 4), row(table8, 5));
        let u3 = _mm512_xor_si512(row(table, 6), row(table8, 7));
        let even = _mm512_xor_si512(u0, _mm512_shuffle_i64x2::<0x4E>(u2, u2));
        let odd = _mm512_xor_si512(u1, _mm512_shuffle_i64x2::<0x4E>(u3, u3));
        _mm512_xor_si512(even, _mm512_shuffle_i64x2::<0xB1>(odd, odd))
    }
}

/// The eight fully-static K-rows of an `ALL_FULLY_STATIC` block: each row's
/// a-side apply times its `x^K`-prescaled `T(expected)` image, XOR-accumulated.
/// Identical arithmetic in both instantiations — `IMG2` only selects which of
/// the two bit-identical apply forms produces `av`.
///
/// # Safety
/// As for [`kernel`]; `parts` addresses the block's eight 64-byte-aligned
/// partial images and `a_base` its eight packed a-rows. With `IMG2`, as for
/// [`apply_full_2img`].
#[inline(always)]
unsafe fn all_static_acc<const IMG2: bool>(
    table: *const u8,
    table8: *const u8,
    a_base: *const u8,
    parts: *const u8,
) -> __m512i {
    // SAFETY: forwarded from the caller's contract.
    unsafe {
        let mut acc = _mm512_setzero_si512();
        for k in 0..8usize {
            let av = if IMG2 {
                apply_full_2img(table, table8, a_base.add(k * N_CHUNKS))
            } else {
                apply_full(table, a_base.add(k * N_CHUNKS))
            };
            let part = _mm512_load_si512(parts.add(k * 64) as *const __m512i);
            acc = _mm512_xor_si512(acc, _mm512_gf2p8mul_epi8(av, part));
        }
        acc
    }
}

/// One `(BLK)`-specialised static-B kernel call. Same contract as
/// [`super::x86_64::shift_reduce_inner_ab_x86_avx512`]; `byte_base_b` is
/// `chunk_byte_base + b_med * N_CHUNKS * 8`.
///
/// Returns `false` (having written nothing) when any planned K-row's b word
/// disagrees with the plan; the caller then runs the incumbent kernel for the
/// whole window. Checking all eight rows up front keeps the specialised body
/// a single straight-line block (no per-row fallback paths for LLVM to hoist
/// loads across and spill), at the cost of the rare all-or-nothing miss.
///
/// # Safety
/// `gfni`, `avx512f`, `avx512bw` must be available; `a_packed`/`b_packed` must
/// hold 8 readable bytes at every K-row offset; the table must have the
/// protocol shape (`ell = 64`, `n_chunks = 8`) and `partials` must have been
/// built from it.
/// (No `#[target_feature]`: this module only compiles when gfni/avx512f/bw are
/// baseline features of the build, and `#[inline(always)]` cannot be combined
/// with `#[target_feature]` — rust#145574. Inlining the live bodies into the
/// hot dispatcher keeps them out of cold call targets.)
#[inline(always)]
unsafe fn kernel<const BLK: usize>(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    byte_base_b: usize,
    partials: &BstaticPartials,
    out: &mut [u8; 64],
    nt: u8,
) -> bool {
    let table = inv_table.data_ptr();
    // Paired two-image apply, resolved once per buffer into `partials`
    // (never a per-window `OnceLock` entry). `table8` is only formed when the
    // mode is live, i.e. when the table actually carries the σ₈ image.
    let img2 = partials.img2;
    let table8 = if img2 {
        inv_table.half_swapped_data_ptr()
    } else {
        table
    };
    // SAFETY: see the function contract; every load below is either an
    // 8-byte packed-row read the caller vouches for, a table-row read (u8
    // index into 256 rows of 64 bytes), or an aligned read of static/partial
    // data.
    unsafe {
        let a_base = a_packed.as_ptr().add(byte_base_b);
        let b_base = b_packed.as_ptr().add(byte_base_b);
        // 1. Sniff: every planned row must match its (mask, expected).
        let mut b_words = [0u64; 8];
        let mut hit = true;
        macro_rules! sniff {
            ($k:literal) => {{
                let p: BstaticRow = BSTATIC_PLAN[BLK][$k];
                if p.kind != ROW_GENERIC {
                    let w = u64::from_le(core::ptr::read_unaligned(
                        b_base.add($k * N_CHUNKS) as *const u64
                    ));
                    b_words[$k] = w;
                    hit &= (w & p.mask) == p.expected;
                }
            }};
        }
        sniff!(0);
        sniff!(1);
        sniff!(2);
        sniff!(3);
        sniff!(4);
        sniff!(5);
        sniff!(6);
        sniff!(7);
        if !hit {
            return false;
        }
        // 2a. All eight rows fully static (blocks 0 and 1): a compact loop —
        //     LLVM keeps it rolled like the incumbent's, which avoids the
        //     unrolled body's register spills.
        const ALL_FULLY_STATIC: [bool; 33] = {
            let mut t = [false; 33];
            let mut b = 0;
            while b < 33 {
                let mut all = true;
                let mut k = 0;
                while k < 8 {
                    let p = BSTATIC_PLAN[b][k];
                    if p.kind != ROW_STATIC || p.vary != 0 {
                        all = false;
                    }
                    k += 1;
                }
                t[b] = all;
                b += 1;
            }
            t
        };
        if ALL_FULLY_STATIC[BLK] {
            // The apply form is loop-invariant, so it is a const-generic
            // parameter of the row loop, not a per-row branch.
            let parts = partials.rows[BLK].as_ptr() as *const u8;
            let acc = if img2 {
                all_static_acc::<true>(table, table8, a_base, parts)
            } else {
                all_static_acc::<false>(table, table8, a_base, parts)
            };
            super::x86_64::store_out64(out, acc, nt);
            return true;
        }
        // 2b. Straight-line specialised body.
        let mut acc = _mm512_setzero_si512();
        macro_rules! row {
            ($k:literal) => {{
                let p: BstaticRow = BSTATIC_PLAN[BLK][$k];
                let a_ptr = a_base.add($k * N_CHUNKS);
                if p.kind == ROW_ZERO {
                    // b row is structurally zero ⇒ contributes nothing.
                } else if p.kind == ROW_STATIC {
                    let av = if img2 {
                        apply_full_2img(table, table8, a_ptr)
                    } else {
                        apply_full(table, a_ptr)
                    };
                    let part = _mm512_load_si512(partials.rows[BLK][$k].as_ptr() as *const __m512i);
                    if p.vary == 0 {
                        // `part` is already `T(expected) · x^K`.
                        acc = _mm512_xor_si512(acc, _mm512_gf2p8mul_epi8(av, part));
                    } else {
                        let b_word = b_words[$k];
                        let mut bv = part;
                        let mut v = p.vary;
                        while v != 0 {
                            let j = v.trailing_zeros() as usize;
                            let byte = (b_word >> (8 * j)) as u8 as usize;
                            let r = _mm512_loadu_si512(table.add(byte * 64) as *const __m512i);
                            bv = _mm512_xor_si512(bv, perm_row(r, j));
                            v &= v - 1;
                        }
                        let prod = _mm512_gf2p8mul_epi8(av, bv);
                        let scaled = if $k == 0 {
                            prod
                        } else {
                            _mm512_gf2p8mul_epi8(prod, _mm512_set1_epi8((1u8 << $k) as i8))
                        };
                        acc = _mm512_xor_si512(acc, scaled);
                    }
                } else {
                    let av = apply_full(table, a_ptr);
                    let bv = apply_full(table, b_base.add($k * N_CHUNKS));
                    let prod = _mm512_gf2p8mul_epi8(av, bv);
                    let scaled = if $k == 0 {
                        prod
                    } else {
                        _mm512_gf2p8mul_epi8(prod, _mm512_set1_epi8((1u8 << $k) as i8))
                    };
                    acc = _mm512_xor_si512(acc, scaled);
                }
            }};
        }
        row!(0);
        row!(1);
        row!(2);
        row!(3);
        row!(4);
        row!(5);
        row!(6);
        row!(7);
        super::x86_64::store_out64(out, acc, nt);
        true
    }
}

/// Dispatch one `(w, b_med)` window through its specialised plan. Returns
/// `false` when the position has no live plan or its b words miss the plan;
/// the caller must then run the incumbent kernel (nothing has been written).
///
/// Direct compares rather than a jump table: the live blocks recur at fixed
/// positions of the 32-window BLAKE3 sequence, which a history-based branch
/// predictor learns, whereas an indirect target that changes every few dozen
/// calls does not stay predicted.
///
/// Out of line on purpose: the incumbent kernel's caller stays lean (its
/// prologue and register allocation are unaffected by the specialised bodies),
/// and the live blocks pay one direct, well-predicted call.
///
/// # Safety
/// As for [`kernel`].
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn shift_reduce_inner_ab_x86_avx512_bstatic(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    w: usize,
    partials: &BstaticPartials,
    out: &mut [u8; 64],
    nt: u8,
) -> bool {
    if w > 1 || b_med >= 16 {
        return false;
    }
    // SAFETY: forwarded from the caller's contract.
    unsafe {
        shift_reduce_inner_ab_x86_avx512_bstatic_at(
            a_packed,
            b_packed,
            inv_table,
            chunk_byte_base + b_med * N_CHUNKS * 8,
            w * 16 + b_med,
            partials,
            out,
            nt,
        )
    }
}

/// [`shift_reduce_inner_ab_x86_avx512_bstatic`] addressed directly by the
/// window's absolute byte offset and its global block index
/// `blk = w * 16 + b_med`.
///
/// # Safety
/// As for [`kernel`].
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn shift_reduce_inner_ab_x86_avx512_bstatic_at(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    byte_base_b: usize,
    blk: usize,
    partials: &BstaticPartials,
    out: &mut [u8; 64],
    nt: u8,
) -> bool {
    // SAFETY: forwarded from the caller's contract.
    unsafe {
        if blk == 31 {
            kernel::<31>(
                a_packed,
                b_packed,
                inv_table,
                byte_base_b,
                partials,
                out,
                nt,
            )
        } else if blk == 30 {
            kernel::<30>(
                a_packed,
                b_packed,
                inv_table,
                byte_base_b,
                partials,
                out,
                nt,
            )
        } else if blk <= 1 {
            // Blocks 0 and 1 carry the identical plan (all-ones b on every
            // row), so they share one body and one set of partial images.
            kernel::<0>(
                a_packed,
                b_packed,
                inv_table,
                byte_base_b,
                partials,
                out,
                nt,
            )
        } else if blk == 28 && bstatic_28_enabled() {
            // Five STATIC rows + three GENERIC. The unrolled mixed body
            // measured slower on typical mixed blocks; 28 is the most
            // static mixed plan. Kill restores the incumbent for this blk.
            kernel::<28>(
                a_packed,
                b_packed,
                inv_table,
                byte_base_b,
                partials,
                out,
                nt,
            )
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::x86_64::shift_reduce_inner_ab_x86_avx512;
    use super::super::x86_64_bstatic_plan::BSTATIC_GENERIC_PLAN;
    use super::*;
    use crate::ntt::AdditiveNttGf8;
    use crate::zerocheck::univariate_skip_optimized::K_SKIP;

    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    fn standard_table() -> InvNttTableByteSingleGf8 {
        let ntt_s = AdditiveNttGf8::new(K_SKIP, F8::ZERO);
        let ntt_l = AdditiveNttGf8::new(K_SKIP, F8(1u8 << K_SKIP));
        InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l)
    }

    /// Every plan (all 32 BLAKE3 blocks + the generic control), on b words
    /// that hit the plan (`expected | random ∉ mask`), miss it (fully random),
    /// or are all-zero / all-ones, must match the incumbent kernel bit for bit.
    #[test]
    fn bstatic_matches_incumbent_all_plans() {
        let inv_table = standard_table();
        let partials = build_partials(&inv_table);
        let mut rng = Rng(0xB57A_71C0);
        // 4 outer windows of 16 b_med windows of 8 K-rows of 8 bytes.
        const OUTER: usize = 1024;
        let n_windows = 4;
        for plan in 0..=BSTATIC_GENERIC_PLAN {
            let (w, b_med) = if plan == BSTATIC_GENERIC_PLAN {
                (0, 3)
            } else {
                (plan / 16, plan % 16)
            };
            for mode in 0..4 {
                let mut a = vec![0u8; OUTER * n_windows];
                let mut b = vec![0u8; OUTER * n_windows];
                for x in 0..n_windows {
                    for bm in 0..16 {
                        for k in 0..8 {
                            let off = x * OUTER + bm * 64 + k * 8;
                            let a_word = rng.next_u64();
                            let p = BSTATIC_PLAN[plan.min(BSTATIC_BLOCKS - 1)][k];
                            let b_word = match mode {
                                0 => (p.expected & p.mask) | (rng.next_u64() & !p.mask), // hit
                                1 => rng.next_u64(), // miss (mostly)
                                2 => 0,
                                _ => u64::MAX,
                            };
                            a[off..off + 8].copy_from_slice(&a_word.to_le_bytes());
                            b[off..off + 8].copy_from_slice(&b_word.to_le_bytes());
                        }
                    }
                }
                for x in 0..n_windows {
                    let mut got = [0u8; 64];
                    let mut want = [0u8; 64];
                    // SAFETY: test runs only where the cfg gate compiled this
                    // module (avx512f/avx512bw/gfni baseline).
                    unsafe {
                        let byte_base_b = x * OUTER + b_med * N_CHUNKS * 8;
                        macro_rules! run {
                            ($($i:literal),*) => {
                                match plan {
                                    $($i => kernel::<$i>(&a, &b, &inv_table, byte_base_b, &partials, &mut got, 0),)*
                                    _ => unreachable!(),
                                }
                            };
                        }
                        let ran = run!(
                            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                            20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
                        );
                        shift_reduce_inner_ab_x86_avx512(
                            &a,
                            &b,
                            &inv_table,
                            x * OUTER,
                            b_med,
                            &mut want,
                            0,
                        );
                        // mode 0 (plan-shaped hits) must take the specialised
                        // path; the generic control plan always runs.
                        if mode == 0 || plan == BSTATIC_GENERIC_PLAN {
                            assert!(
                                ran,
                                "plan {plan} mode {mode} window {x} unexpectedly missed"
                            );
                        }
                        if !ran {
                            got = want; // caller contract: incumbent runs on a miss
                        }
                        // The public dispatcher must agree too (w/b_med → plan).
                        if plan != BSTATIC_GENERIC_PLAN {
                            let mut got2 = [0u8; 64];
                            let ran2 = shift_reduce_inner_ab_x86_avx512_bstatic(
                                &a,
                                &b,
                                &inv_table,
                                x * OUTER,
                                b_med,
                                w,
                                &partials,
                                &mut got2,
                                0,
                            );
                            let live = BSTATIC_LIVE[plan];
                            assert_eq!(
                                ran2,
                                ran && live,
                                "dispatcher/kernel disagree on plan {plan} mode {mode}"
                            );
                            if ran2 {
                                assert_eq!(
                                    got2, want,
                                    "dispatcher plan {plan} mode {mode} window {x}"
                                );
                            }
                        }
                    }
                    assert_eq!(got, want, "plan {plan} mode {mode} window {x}");
                }
            }
        }
    }

    /// The prescaled fully-static images equal `T(expected) · x^K` and the
    /// plain ones equal `T(expected)`; unplanned rows are zero.
    #[test]
    fn partials_shape() {
        let inv_table = standard_table();
        let partials = build_partials(&inv_table);
        let mut f = [F8::ZERO; 64];
        for blk in 0..BSTATIC_BLOCKS {
            for k in 0..8 {
                let p = BSTATIC_PLAN[blk][k];
                if p.kind != ROW_STATIC {
                    assert_eq!(partials.rows[blk][k], [0u8; 64]);
                    continue;
                }
                inv_table.apply(&p.expected.to_le_bytes(), &mut f);
                for i in 0..64 {
                    let want = if p.vary == 0 {
                        f[i] * F8(1u8 << k)
                    } else {
                        f[i]
                    };
                    assert_eq!(partials.rows[blk][k][i], want.0, "blk {blk} k {k} lane {i}");
                }
            }
        }
    }

    /// `prepare_bstatic` accepts the protocol table (and its identical twin)
    /// and rejects a table with a different image.
    #[test]
    fn prepare_bstatic_fingerprint() {
        if !fast_shift_reduce_enabled() {
            return;
        }
        let t1 = standard_table();
        let t2 = standard_table();
        assert!(prepare_bstatic(&t1).is_some());
        assert!(prepare_bstatic(&t2).is_some());
        // A table over a different output coset has different basis rows.
        let ntt_s = AdditiveNttGf8::new(K_SKIP, F8::ZERO);
        let ntt_l = AdditiveNttGf8::new(K_SKIP, F8(2u8 << K_SKIP));
        let t3 = InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);
        assert!(prepare_bstatic(&t3).is_none());
        // And the original is accepted again afterwards.
        assert!(prepare_bstatic(&t1).is_some());
    }

    /// The `q ↦ q ^ j` qword permutation reproduces the incumbent apply's
    /// lane shuffle + half swap for every byte position.
    #[test]
    fn perm_row_matches_incumbent_apply() {
        let inv_table = standard_table();
        let mut rng = Rng(0x9E4D);
        for _ in 0..256 {
            let word = rng.next_u64().to_le_bytes();
            let mut want = [F8::ZERO; 64];
            inv_table.apply(&word, &mut want);
            // SAFETY: cfg-gated module ⇒ avx512f baseline.
            let got = unsafe {
                let v = apply_full(inv_table.data_ptr(), word.as_ptr());
                let mut o = [0u8; 64];
                _mm512_storeu_si512(o.as_mut_ptr() as *mut __m512i, v);
                o
            };
            let want: [u8; 64] = core::array::from_fn(|i| want[i].0);
            assert_eq!(got, want);
        }
    }
}

#[cfg(test)]
mod microbench {
    //! Single-thread hot-cache cost per call, incumbent vs specialised plan.
    //! `cargo test --release -p flock-core bstatic_microbench -- --ignored --nocapture`
    use super::super::x86_64::shift_reduce_inner_ab_x86_avx512;
    use super::*;
    use crate::ntt::AdditiveNttGf8;
    use crate::zerocheck::univariate_skip_optimized::K_SKIP;

    // ---- experimental bodies kept for the measurements quoted in the
    // ---- submission note (both lost to the incumbent in the whole-cycle
    // ---- bench: data-driven rolled 1.08×, per-block rolled 1.09×) ----
    /// `x^k` as a broadcast F₈ byte, for the runtime-`k` rolled kernel.
    #[repr(C, align(64))]
    struct XkTable([[u8; 64]; 8]);
    static XK: XkTable = {
        let mut t = [[0u8; 64]; 8];
        let mut k = 0;
        while k < 8 {
            let mut i = 0;
            while i < 64 {
                t[k][i] = 1u8 << k;
                i += 1;
            }
            k += 1;
        }
        XkTable(t)
    };

    /// EXPERIMENTAL single-body, data-driven static-B kernel: the plan and the
    /// partial images are runtime parameters and the K loop stays rolled, so one
    /// ~1.5 KiB body serves every block (no per-block code footprint). Its
    /// per-row `kind` branches follow the fixed BLAKE3 block sequence.
    ///
    /// # Safety
    /// As for [`kernel`].
    #[target_feature(enable = "gfni,avx512f,avx512bw")]
    unsafe fn kernel_dyn(
        a_packed: &[u8],
        b_packed: &[u8],
        inv_table: &InvNttTableByteSingleGf8,
        byte_base_b: usize,
        plan: &[BstaticRow; 8],
        parts: &[[u8; 64]; 8],
        out: &mut [u8; 64],
    ) -> bool {
        let table = inv_table.data_ptr();
        // SAFETY: see [`kernel`].
        unsafe {
            let a_base = a_packed.as_ptr().add(byte_base_b);
            let b_base = b_packed.as_ptr().add(byte_base_b);
            let mut b_words = [0u64; 8];
            let mut hit = true;
            for k in 0..8 {
                let p = plan[k];
                if p.kind != ROW_GENERIC {
                    let w = u64::from_le(core::ptr::read_unaligned(
                        b_base.add(k * N_CHUNKS) as *const u64
                    ));
                    b_words[k] = w;
                    hit &= (w & p.mask) == p.expected;
                }
            }
            if !hit {
                return false;
            }
            let mut acc = _mm512_setzero_si512();
            // `black_box` keeps the K loop rolled (one compact body).
            for k in 0..core::hint::black_box(8usize) {
                let p = plan[k];
                if p.kind == ROW_ZERO {
                    continue;
                }
                let av = apply_full(table, a_base.add(k * N_CHUNKS));
                if p.kind == ROW_STATIC {
                    let mut bv = _mm512_load_si512(parts[k].as_ptr() as *const __m512i);
                    if p.vary == 0 {
                        acc = _mm512_xor_si512(acc, _mm512_gf2p8mul_epi8(av, bv));
                        continue;
                    }
                    let b_word = b_words[k];
                    let mut v = p.vary;
                    while v != 0 {
                        let j = v.trailing_zeros() as usize;
                        let byte = (b_word >> (8 * j)) as u8 as usize;
                        let r = _mm512_loadu_si512(table.add(byte * 64) as *const __m512i);
                        bv = _mm512_xor_si512(bv, perm_row(r, j));
                        v &= v - 1;
                    }
                    let prod = _mm512_gf2p8mul_epi8(av, bv);
                    let scaled = if k == 0 {
                        prod
                    } else {
                        _mm512_gf2p8mul_epi8(
                            prod,
                            _mm512_load_si512(XK.0[k].as_ptr() as *const __m512i),
                        )
                    };
                    acc = _mm512_xor_si512(acc, scaled);
                } else {
                    let bv = apply_full(table, b_base.add(k * N_CHUNKS));
                    let prod = _mm512_gf2p8mul_epi8(av, bv);
                    let scaled = if k == 0 {
                        prod
                    } else {
                        _mm512_gf2p8mul_epi8(
                            prod,
                            _mm512_load_si512(XK.0[k].as_ptr() as *const __m512i),
                        )
                    };
                    acc = _mm512_xor_si512(acc, scaled);
                }
            }
            _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, acc);
            true
        }
    }

    /// Per-block monomorphized rolled body (experiment): same code as
    /// [`kernel_dyn`] but the plan/partials are compile-time selected, so every
    /// branch inside has a period-8 outcome pattern at its own code address.
    #[target_feature(enable = "gfni,avx512f,avx512bw")]
    unsafe fn kernel_rolled<const BLK: usize>(
        a_packed: &[u8],
        b_packed: &[u8],
        inv_table: &InvNttTableByteSingleGf8,
        byte_base_b: usize,
        partials: &BstaticPartials,
        out: &mut [u8; 64],
    ) -> bool {
        // SAFETY: forwarded.
        unsafe {
            kernel_dyn(
                a_packed,
                b_packed,
                inv_table,
                byte_base_b,
                &BSTATIC_PLAN[BLK],
                &partials.rows[BLK],
                out,
            )
        }
    }

    /// Whole-cycle cost: the 32 BLAKE3 windows in production order, per call,
    /// incumbent vs the live-plan dispatch vs the rolled data-driven body.
    #[test]
    #[ignore]
    fn bstatic_cycle_bench() {
        let ntt_s = AdditiveNttGf8::new(K_SKIP, F8::ZERO);
        let ntt_l = AdditiveNttGf8::new(K_SKIP, F8(1u8 << K_SKIP));
        let inv_table = InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);
        let partials: &'static BstaticPartials = Box::leak(build_partials(&inv_table));
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut rnd = || {
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        // 32 blocks of 2 windows each, plan-shaped b words in every (w, b_med, k).
        const BLOCK: usize = 2048;
        let n_blocks = 32;
        let mut a = vec![0u8; BLOCK * n_blocks];
        let mut b = vec![0u8; BLOCK * n_blocks];
        for blk in 0..n_blocks {
            for w in 0..2 {
                for bm in 0..16 {
                    for k in 0..8 {
                        let off = blk * BLOCK + w * 1024 + bm * 64 + k * 8;
                        let p = BSTATIC_PLAN[w * 16 + bm][k];
                        a[off..off + 8].copy_from_slice(&rnd().to_le_bytes());
                        b[off..off + 8].copy_from_slice(
                            &((p.expected & p.mask) | (rnd() & !p.mask)).to_le_bytes(),
                        );
                    }
                }
            }
        }
        let iters = 20_000usize; // × 32 calls
        let mut out = [0u8; 64];
        let mut a_col = [F8::ZERO; 64];
        let mut b_col = [F8::ZERO; 64];
        let mut sink = 0u64;
        for round in 0..3 {
            let t = std::time::Instant::now();
            for i in 0..iters {
                let base = (i % n_blocks) * BLOCK;
                for w in 0..2 {
                    for bm in 0..16 {
                        super::super::shift_reduce_inner_ab(
                            &a,
                            &b,
                            &inv_table,
                            base + w * 1024,
                            bm,
                            &mut out,
                            &mut a_col,
                            &mut b_col,
                            None,
                            0,
                        );
                        sink ^= out[0] as u64;
                    }
                }
            }
            let off_ns = t.elapsed().as_secs_f64() * 1e9 / (iters * 32) as f64;
            let t = std::time::Instant::now();
            for i in 0..iters {
                let base = (i % n_blocks) * BLOCK;
                for w in 0..2 {
                    for bm in 0..16 {
                        super::super::shift_reduce_inner_ab(
                            &a,
                            &b,
                            &inv_table,
                            base + w * 1024,
                            bm,
                            &mut out,
                            &mut a_col,
                            &mut b_col,
                            Some((w, partials)),
                            0,
                        );
                        sink ^= out[0] as u64;
                    }
                }
            }
            let live_ns = t.elapsed().as_secs_f64() * 1e9 / (iters * 32) as f64;
            let t = std::time::Instant::now();
            for i in 0..iters {
                let base = (i % n_blocks) * BLOCK;
                for w in 0..2 {
                    for bm in 0..16 {
                        let blk = w * 16 + bm;
                        let ok = unsafe {
                            kernel_dyn(
                                &a,
                                &b,
                                &inv_table,
                                base + w * 1024 + bm * 64,
                                &BSTATIC_PLAN[blk],
                                &partials.rows[blk],
                                &mut out,
                            )
                        };
                        assert!(ok);
                        sink ^= out[0] as u64;
                    }
                }
            }
            let dyn_ns = t.elapsed().as_secs_f64() * 1e9 / (iters * 32) as f64;
            let t = std::time::Instant::now();
            for i in 0..iters {
                let base = (i % n_blocks) * BLOCK;
                for w in 0..2 {
                    for bm in 0..16 {
                        let blk = w * 16 + bm;
                        let bb = base + w * 1024 + bm * 64;
                        macro_rules! run { ($($i:literal),*) => { match blk { $($i => unsafe { kernel_rolled::<$i>(&a, &b, &inv_table, bb, partials, &mut out) },)* _ => unreachable!() } }; }
                        let ok = run!(
                            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                            20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
                        );
                        assert!(ok);
                        sink ^= out[0] as u64;
                    }
                }
            }
            let rolled_ns = t.elapsed().as_secs_f64() * 1e9 / (iters * 32) as f64;
            println!(
                "cycle round {round}: incumbent {off_ns:6.2} ns/call  live-plan {live_ns:6.2} ({:.3})  dyn-all-blocks {dyn_ns:6.2} ({:.3})  rolled-per-blk-all {rolled_ns:6.2} ({:.3})  sink {sink}",
                live_ns / off_ns,
                dyn_ns / off_ns,
                rolled_ns / off_ns
            );
        }
        // correctness of kernel_dyn over the whole cycle
        for blk in 0..n_blocks {
            let base = blk * BLOCK;
            for w in 0..2 {
                for bm in 0..16 {
                    let plan_i = w * 16 + bm;
                    let mut want = [0u8; 64];
                    let mut got = [0u8; 64];
                    unsafe {
                        shift_reduce_inner_ab_x86_avx512(
                            &a,
                            &b,
                            &inv_table,
                            base + w * 1024,
                            bm,
                            &mut want,
                            0,
                        );
                        assert!(kernel_dyn(
                            &a,
                            &b,
                            &inv_table,
                            base + w * 1024 + bm * 64,
                            &BSTATIC_PLAN[plan_i],
                            &partials.rows[plan_i],
                            &mut got
                        ));
                    }
                    assert_eq!(got, want, "kernel_dyn mismatch blk {blk} w {w} bm {bm}");
                }
            }
        }
    }

    #[test]
    #[ignore]
    fn bstatic_microbench() {
        let ntt_s = AdditiveNttGf8::new(K_SKIP, F8::ZERO);
        let ntt_l = AdditiveNttGf8::new(K_SKIP, F8(1u8 << K_SKIP));
        let inv_table = InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);
        let partials: &'static BstaticPartials = Box::leak(build_partials(&inv_table));
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut rnd = || {
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        // 64 outer windows (64 KiB each of a and b) so the input stream is
        // L2-resident like the streaming seam, not a single hot line.
        const OUTER: usize = 1024;
        let n_windows = 64;
        let iters = 200_000usize;
        let sweep: Vec<(usize, bool)> = if std::env::var_os("BSTATIC_SWEEP_ALL").is_some() {
            (0..32).map(|b| (b, true)).collect()
        } else {
            vec![
                (0usize, true),
                (1, true),
                (3, true),
                (12, true),
                (20, true),
                (30, true),
                (31, true),
                (3, false),
                (12, false),
            ]
        };
        for &(plan, hits) in &sweep {
            let (w, b_med) = (plan / 16, plan % 16);
            let mut a = vec![0u8; OUTER * n_windows];
            let mut b = vec![0u8; OUTER * n_windows];
            for x in 0..n_windows {
                for bm in 0..16 {
                    for k in 0..8 {
                        let off = x * OUTER + bm * 64 + k * 8;
                        let p = BSTATIC_PLAN[plan][k];
                        let a_word = rnd();
                        let b_word = if hits {
                            (p.expected & p.mask) | (rnd() & !p.mask)
                        } else {
                            rnd()
                        };
                        a[off..off + 8].copy_from_slice(&a_word.to_le_bytes());
                        b[off..off + 8].copy_from_slice(&b_word.to_le_bytes());
                    }
                }
            }
            let mut out = [0u8; 64];
            let mut sink = 0u64;
            let mut a_col = [F8::ZERO; 64];
            let mut b_col = [F8::ZERO; 64];
            // production entry, hint off vs on
            let t = std::time::Instant::now();
            for i in 0..iters {
                let x = i % n_windows;
                super::super::shift_reduce_inner_ab(
                    &a,
                    &b,
                    &inv_table,
                    x * OUTER,
                    b_med,
                    &mut out,
                    &mut a_col,
                    &mut b_col,
                    None,
                    0,
                );
                sink ^= out[0] as u64 ^ (out[63] as u64) << 8;
            }
            let prod_off = t.elapsed().as_secs_f64() * 1e9 / iters as f64;
            let t = std::time::Instant::now();
            for i in 0..iters {
                let x = i % n_windows;
                super::super::shift_reduce_inner_ab(
                    &a,
                    &b,
                    &inv_table,
                    x * OUTER,
                    b_med,
                    &mut out,
                    &mut a_col,
                    &mut b_col,
                    Some((w, partials)),
                    0,
                );
                sink ^= out[0] as u64 ^ (out[63] as u64) << 8;
            }
            let prod_on = t.elapsed().as_secs_f64() * 1e9 / iters as f64;
            println!(
                "plan {plan:2} hits={hits:5}: PRODUCTION entry hint-off {prod_off:6.1}  hint-on {prod_on:6.1} ns/call  ratio {:.2}",
                prod_on / prod_off
            );
            // incumbent
            let t = std::time::Instant::now();
            for i in 0..iters {
                let x = i % n_windows;
                unsafe {
                    shift_reduce_inner_ab_x86_avx512(
                        &a,
                        &b,
                        &inv_table,
                        x * OUTER,
                        b_med,
                        &mut out,
                        0,
                    );
                }
                sink ^= out[0] as u64 ^ (out[63] as u64) << 8;
            }
            let inc = t.elapsed().as_secs_f64() * 1e9 / iters as f64;
            // specialised
            let t = std::time::Instant::now();
            for i in 0..iters {
                let x = i % n_windows;
                unsafe {
                    let ok = shift_reduce_inner_ab_x86_avx512_bstatic(
                        &a,
                        &b,
                        &inv_table,
                        x * OUTER,
                        b_med,
                        w,
                        &partials,
                        &mut out,
                        0,
                    );
                    if !ok {
                        shift_reduce_inner_ab_x86_avx512(
                            &a,
                            &b,
                            &inv_table,
                            x * OUTER,
                            b_med,
                            &mut out,
                            0,
                        );
                    }
                }
                sink ^= out[0] as u64 ^ (out[63] as u64) << 8;
            }
            let spec = t.elapsed().as_secs_f64() * 1e9 / iters as f64;
            // direct generic-control body (no dispatch): isolates call/dispatch overhead
            let t = std::time::Instant::now();
            for i in 0..iters {
                let x = i % n_windows;
                unsafe {
                    let byte_base_b = x * OUTER + b_med * N_CHUNKS * 8;
                    let ok = kernel::<32>(&a, &b, &inv_table, byte_base_b, &partials, &mut out, 0);
                    assert!(ok);
                }
                sink ^= out[0] as u64 ^ (out[63] as u64) << 8;
            }
            let gen32 = t.elapsed().as_secs_f64() * 1e9 / iters as f64;
            // direct specialised body (no dispatch)
            let t = std::time::Instant::now();
            for i in 0..iters {
                let x = i % n_windows;
                unsafe {
                    let byte_base_b = x * OUTER + b_med * N_CHUNKS * 8;
                    macro_rules! run {
                        ($($i:literal),*) => { match plan { $($i => kernel::<$i>(&a, &b, &inv_table, byte_base_b, &partials, &mut out, 0),)* _ => unreachable!() } };
                    }
                    let ok = run!(
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
                    );
                    if !ok {
                        shift_reduce_inner_ab_x86_avx512(
                            &a,
                            &b,
                            &inv_table,
                            x * OUTER,
                            b_med,
                            &mut out,
                            0,
                        );
                    }
                }
                sink ^= out[0] as u64 ^ (out[63] as u64) << 8;
            }
            let direct = t.elapsed().as_secs_f64() * 1e9 / iters as f64;
            println!(
                "plan {plan:2} hits={hits:5}: incumbent {inc:6.1}  dispatch+spec {spec:6.1}  direct-spec {direct:6.1}  direct-generic32 {gen32:6.1} ns/call  (sink {sink})"
            );
        }
    }
}
