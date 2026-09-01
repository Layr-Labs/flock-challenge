use super::{F8, InvNttTableByteSingleGf8};

mod portable;

#[cfg(all(test, target_arch = "aarch64"))]
pub(super) use portable::bit_transpose_64bytes_scalar;
#[cfg(all(
    test,
    any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "gfni")
    )
))]
pub(super) use portable::shift_reduce_inner_ab_scalar;

#[cfg(target_arch = "aarch64")]
pub(super) mod aarch64;

#[cfg(target_arch = "x86_64")]
pub(super) mod x86_64;

/// Ranked-static B-complement projection (x86 AVX-512/GFNI only).
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
pub(super) mod x86_64_bcomplement;
/// Static-B round-1 kernel (x86 AVX-512/GFNI only). See `x86_64_bstatic.rs`.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
pub(super) mod x86_64_bstatic;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
mod x86_64_bstatic_plan;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
pub(super) use x86_64_bstatic::{BstaticPartials, prepare_bstatic};

/// Static-B partial images: an empty placeholder where the kernel does not
/// exist, so callers can thread the hint unconditionally.
#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512f",
    target_feature = "avx512bw"
)))]
pub(super) struct BstaticPartials;

/// Static-B is only available on the x86 AVX-512/GFNI kernel; elsewhere the
/// hint is always absent and the incumbent kernels run unchanged.
#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512f",
    target_feature = "avx512bw"
)))]
#[inline]
pub(super) fn prepare_bstatic(
    _inv_table: &InvNttTableByteSingleGf8,
) -> Option<&'static BstaticPartials> {
    None
}

/// The BLAKE3 medium-window indices whose static-B plan can be live. The
/// outlined static-B dispatcher returns `false` without writing for every
/// other window, so a caller that transforms many blocks at one window index
/// resolves this once instead of entering the dispatcher per block.
#[inline]
pub(super) fn bstatic_window_live(blk: usize) -> bool {
    blk <= 1 || blk == 28 || blk == 30 || blk == 31
}

/// Per-window static-B hint: the BLAKE3 outer-window index `w ∈ {0, 1}` of
/// the 8192-bit window being transformed plus the resolved partial images.
/// `None` ⇒ incumbent kernel. Only a performance hint: the static kernel
/// verifies every planned row against the actual b word.
pub(super) type BstaticHint = Option<(usize, &'static BstaticPartials)>;

/// Process-invariant mode switches for the x86 round-1 AB kernel. The
/// streaming witness path resolves these once, rather than entering three
/// `LazyLock` accessors for every 64-byte window.
#[derive(Clone, Copy)]
pub(super) struct ShiftReducePlan {
    img2: bool,
    #[allow(dead_code)] // Reserved by the matching rollback selector.
    pidx: bool,
    #[allow(dead_code)] // Reserved by the matching rollback selector.
    offw: bool,
}

impl ShiftReducePlan {
    /// True when the kernel body addresses the table through the σ₈ second
    /// image, i.e. when the resolved image pointers are the ones it reads.
    #[inline]
    pub(super) fn uses_images(self) -> bool {
        self.img2
    }
}

#[inline]
pub(super) fn prepare_shift_reduce(inv_table: &InvNttTableByteSingleGf8) -> ShiftReducePlan {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    {
        let img2 = x86_64::urm_apply_2img_enabled() && inv_table.has_second_image();
        let pidx = img2 && x86_64::urm_pidx_enabled();
        let offw = pidx && x86_64::urm_offw_enabled();
        ShiftReducePlan { img2, pidx, offw }
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    )))]
    {
        let _ = inv_table;
        ShiftReducePlan {
            img2: false,
            pidx: false,
            offw: false,
        }
    }
}

#[inline]
pub(super) fn bit_transpose_64bytes(input: &[u8; 64], output: &mut [u8; 64]) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: aarch64 statically guarantees NEON.
    unsafe {
        aarch64::bit_transpose_64bytes_neon(input, output);
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi"
    ))]
    // SAFETY: all target features required by the kernel are enabled.
    unsafe {
        x86_64::bit_transpose_64bytes_avx512(input, output);
    }

    #[cfg(not(any(
        target_arch = "aarch64",
        all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512vbmi"
        )
    )))]
    portable::bit_transpose_64bytes_scalar(input, output);
}

#[allow(clippy::too_many_arguments)]
pub(super) fn shift_reduce_inner_ab(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; 64],
    a_col: &mut [F8],
    b_col: &mut [F8],
    bstatic: BstaticHint,
    nt: u8,
) {
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    )))]
    let _ = (bstatic, nt);

    #[cfg(target_arch = "aarch64")]
    {
        let _ = (a_col, b_col);
        aarch64::shift_reduce_inner_ab_fused_neon(
            a_packed,
            b_packed,
            inv_table,
            chunk_byte_base,
            b_med,
            out,
        );
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    {
        let _ = (a_col, b_col);
        // SAFETY: all required target features are enabled at compile time.
        unsafe {
            if let Some((w, partials)) = bstatic {
                if x86_64_bstatic::shift_reduce_inner_ab_x86_avx512_bstatic(
                    a_packed,
                    b_packed,
                    inv_table,
                    chunk_byte_base,
                    b_med,
                    w,
                    partials,
                    out,
                    nt,
                ) {
                    return;
                }
            }
            x86_64::shift_reduce_inner_ab_x86_avx512(
                a_packed,
                b_packed,
                inv_table,
                chunk_byte_base,
                b_med,
                out,
                nt,
            );
        }
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        not(all(target_feature = "avx512f", target_feature = "avx512bw"))
    ))]
    // SAFETY: gfni is enabled at compile time; SSE2 is baseline on x86_64.
    unsafe {
        x86_64::shift_reduce_inner_ab_x86_sse(
            a_packed,
            b_packed,
            inv_table,
            chunk_byte_base,
            b_med,
            out,
            a_col,
            b_col,
        );
    }

    #[cfg(not(any(
        target_arch = "aarch64",
        all(target_arch = "x86_64", target_feature = "gfni")
    )))]
    portable::shift_reduce_inner_ab_scalar(
        a_packed,
        b_packed,
        inv_table,
        chunk_byte_base,
        b_med,
        out,
        a_col,
        b_col,
    );
}

/// True when [`shift_reduce_inner_ab_at`] with this `prepared` plan and
/// `bstatic` presence would take the pre-scaled-offset (pidx + offw) x86
/// body for window `blk` — i.e. when the caller may replace that call with
/// the split offsets-build / offsets-consume pair and get identical bytes.
#[inline]
#[allow(unused_variables)]
pub(super) fn shift_reduce_offsets_eligible(
    prepared: ShiftReducePlan,
    bstatic: bool,
    blk: usize,
) -> bool {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    {
        prepared.img2
            && prepared.pidx
            && prepared.offw
            && x86_64::urm_off_arena_enabled()
            && !(bstatic && bstatic_window_live(blk))
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    )))]
    {
        false
    }
}

/// Single-window twin of [`shift_reduce_inner_ab`] addressed by an absolute
/// byte offset plus the global BLAKE3 medium-window index
/// `blk = w * 16 + b_med`, for callers that hold one 64-byte window's bytes
/// on their own rather than a whole packed block. Bit-identical to
/// [`shift_reduce_inner_ab`] on the same bytes with the same `blk`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn shift_reduce_inner_ab_at(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    byte_base: usize,
    blk: usize,
    out: &mut [u8; 64],
    bstatic: Option<&'static BstaticPartials>,
    prepared: ShiftReducePlan,
    nt: u8,
    imgs: (*const u8, *const u8),
) {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    ))]
    {
        // SAFETY: all required target features are enabled at compile time,
        // and the caller vouches for 64 readable bytes at `byte_base`.
        unsafe {
            if let Some(partials) = bstatic {
                // The outlined static-B dispatcher (`inline(never)`) is live
                // only for windows 0, 1, 28, 30, 31 — every other `blk` returns
                // false without writing. The streaming producer still called
                // it 28/32 of the time. Gate here so those windows stay on
                // the prepared generic body with no extra call. Unexpected
                // `blk` keeps the incumbent generic path.
                if bstatic_window_live(blk)
                    && x86_64_bstatic::shift_reduce_inner_ab_x86_avx512_bstatic_at(
                        a_packed, b_packed, inv_table, byte_base, blk, partials, out, nt,
                    )
                {
                    return;
                }
            }
            x86_64::shift_reduce_inner_ab_x86_avx512_prepared(
                a_packed,
                b_packed,
                inv_table,
                byte_base,
                0,
                out,
                nt,
                prepared.img2,
                prepared.pidx,
                prepared.offw,
                imgs,
            );
        }
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "gfni",
        target_feature = "avx512f",
        target_feature = "avx512bw"
    )))]
    {
        let _ = (blk, bstatic, prepared, imgs);
        let mut a_col = [F8::ZERO; 64];
        let mut b_col = [F8::ZERO; 64];
        shift_reduce_inner_ab(
            a_packed, b_packed, inv_table, byte_base, 0, out, &mut a_col, &mut b_col, None, nt,
        );
    }
}

/// Paired-window variant: computes windows `b_med` and `b_med + 1` in one
/// call. On aarch64 this takes the interleaved two-window NEON wavefront
/// (`shift_reduce_inner_ab_fused_neon_x2`); everywhere else it decays to two
/// sequential single-window calls. Bit-identical to two calls of
/// [`shift_reduce_inner_ab`] on every path.
#[allow(clippy::too_many_arguments)]
pub(super) fn shift_reduce_inner_ab_x2(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out0: &mut [u8; 64],
    out1: &mut [u8; 64],
    a_col: &mut [F8],
    b_col: &mut [F8],
    bstatic: BstaticHint,
    nt: u8,
) {
    #[cfg(target_arch = "aarch64")]
    {
        let _ = (a_col, b_col, bstatic, nt);
        aarch64::shift_reduce_inner_ab_fused_neon_x2(
            a_packed,
            b_packed,
            inv_table,
            chunk_byte_base,
            b_med,
            out0,
            out1,
        );
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        shift_reduce_inner_ab(
            a_packed,
            b_packed,
            inv_table,
            chunk_byte_base,
            b_med,
            out0,
            a_col,
            b_col,
            bstatic,
            nt,
        );
        shift_reduce_inner_ab(
            a_packed,
            b_packed,
            inv_table,
            chunk_byte_base,
            b_med + 1,
            out1,
            a_col,
            b_col,
            bstatic,
            nt,
        );
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn accumulate_convert(
    chunk_ab_bytes: &[[u8; 64]; 16],
    chunk_c_bytes: &[[u8; 64]; 16],
    n_b_med: usize,
    convert: &[super::F128],
    eq_lo_val: super::F128,
    partial_ab: &mut [super::F128; 64],
    partial_c: &mut [super::F128; 64],
) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: aarch64 statically guarantees NEON and the fixed arrays cover
    // all table-selected loads.
    unsafe {
        aarch64::accumulate_convert(
            chunk_ab_bytes,
            chunk_c_bytes,
            n_b_med,
            convert,
            eq_lo_val,
            partial_ab,
            partial_c,
        );
    }

    #[cfg(not(target_arch = "aarch64"))]
    portable::accumulate_convert(
        chunk_ab_bytes,
        chunk_c_bytes,
        n_b_med,
        convert,
        eq_lo_val,
        partial_ab,
        partial_c,
    );
}
/// GFNI bit-matrix twin of [`accumulate_convert_ab`] for the eq-folded
/// drain: `mats` is the 16x16 qword matrix block for the current `w` slice
/// (convert basis pre-scaled by `eq_top[w]`), `bank_planes` the byte-plane
/// bank for the current `u`. Bit-identical sums; see the kernel.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
#[inline]
pub(super) fn accumulate_convert_ab_nomul_gfni(
    chunk_ab_bytes: &[[u8; 64]; 16],
    n_b_med: usize,
    mats: &[u64; 256],
    bank_planes: &mut [u8; 16 * 64],
) {
    // SAFETY: the cfg gate guarantees the SIMD features and the fixed arrays
    // cover every 64-byte load/store.
    unsafe {
        x86_64::accumulate_convert_ab_nomul_x86_gfni(chunk_ab_bytes, n_b_med, mats, bank_planes);
    }
}

/// Ranked residual twin: evaluate absolute medium rows 2..`n_b_med`.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
#[inline]
pub(super) fn accumulate_convert_ab_nomul_gfni_range2(
    chunk_ab_bytes: &[[u8; 64]; 16],
    n_b_med: usize,
    mats: &[u64; 256],
    bank_planes: &mut [u8; 16 * 64],
) {
    unsafe {
        x86_64::accumulate_convert_ab_nomul_x86_gfni_range2(
            chunk_ab_bytes,
            n_b_med,
            mats,
            bank_planes,
        );
    }
}

/// First-visit twin of [`accumulate_convert_ab_nomul_gfni`]. Every output
/// plane is assigned without observing the bank's previous bytes.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
#[inline]
pub(super) fn write_convert_ab_nomul_gfni(
    chunk_ab_bytes: &[[u8; 64]; 16],
    n_b_med: usize,
    mats: &[u64; 256],
    bank_planes: &mut [u8; 16 * 64],
) {
    // SAFETY: the cfg gate guarantees the SIMD features and the fixed arrays
    // cover every 64-byte load/store. The caller proves write-before-read.
    unsafe {
        x86_64::write_convert_ab_nomul_x86_gfni(chunk_ab_bytes, n_b_med, mats, bank_planes);
    }
}

/// First-write twin of [`accumulate_convert_ab_nomul_gfni_range2`].
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
#[inline]
pub(super) fn write_convert_ab_nomul_gfni_range2(
    chunk_ab_bytes: &[[u8; 64]; 16],
    n_b_med: usize,
    mats: &[u64; 256],
    bank_planes: &mut [u8; 16 * 64],
) {
    unsafe {
        x86_64::write_convert_ab_nomul_x86_gfni_range2(chunk_ab_bytes, n_b_med, mats, bank_planes);
    }
}

/// The incumbent two-window prefetch schedule. `next_window` points at the
/// first live row (`first`) so dense and ranked-compact storage share the
/// same relative hint walk. It is used only for nonfaulting hints, never as a
/// slice or a load/store pointer. An empty range disables hints, including
/// when this pointer is null.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
pub(super) struct AbDirectPrefetch {
    pub(super) next_window: *const u8,
    pub(super) first: usize,
    pub(super) end: usize,
    pub(super) spread: bool,
}

/// Direct-input twin of the ranked GFNI drains. The slice contains ONLY
/// absolute medium rows `first_b_med..n_b_med`, not the entire window.
/// Matrix indexing remains absolute; omitted prefix/tail rows are never read.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
#[inline]
pub(super) fn convert_ab_nomul_gfni_direct<const FIRST_WRITE: bool>(
    live_rows: &[u8],
    first_b_med: usize,
    n_b_med: usize,
    mats: &[u64; 256],
    bank_planes: &mut [u8; 16 * 64],
    prefetch: &AbDirectPrefetch,
) {
    // SAFETY: each arm checks the precise live span for its fixed row range;
    // the plane/matrix arrays cover all accesses and the cfg supplies GFNI.
    match (first_b_med, n_b_med) {
        (2, 16) => {
            assert_eq!(live_rows.len(), 14 * 64);
            unsafe {
                x86_64::convert_ab_nomul_x86_gfni_direct::<2, 16, FIRST_WRITE>(
                    live_rows,
                    mats,
                    bank_planes,
                    prefetch,
                );
            }
        }
        (0, 15) => {
            assert_eq!(live_rows.len(), 15 * 64);
            unsafe {
                x86_64::convert_ab_nomul_x86_gfni_direct::<0, 15, FIRST_WRITE>(
                    live_rows,
                    mats,
                    bank_planes,
                    prefetch,
                );
            }
        }
        _ => unreachable!("direct AB input requires the ranked residual row pair"),
    }
}

/// Reassemble one byte-plane C bank (`[plane][lane]`) into its 64 F128 lanes:
/// `out[lane] = sum_k plane[k][lane] << 8k` over the low eight planes for
/// `lo` and the high eight for `hi`. Run once per bank per band by the fused
/// GFNI drain.
#[inline]
pub(super) fn c_plane_bank_to_f128(bank_planes: &[u8; 16 * 64], out: &mut [super::F128; 64]) {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi",
        target_feature = "vpclmulqdq"
    ))]
    // SAFETY: the cfg gate supplies the features; both arrays are fixed-size.
    unsafe {
        x86_64::c_plane_bank_to_f128_x86_avx512(bank_planes, out);
        return;
    }

    #[allow(unreachable_code)]
    for (lane, slot) in out.iter_mut().enumerate() {
        let mut lo = 0u64;
        let mut hi = 0u64;
        for k in 0..8 {
            lo |= (bank_planes[k * 64 + lane] as u64) << (8 * k);
        }
        for k in 8..16 {
            hi |= (bank_planes[k * 64 + lane] as u64) << (8 * (k - 8));
        }
        *slot = super::F128 { lo, hi };
    }
}

/// Ascending bulk fetch of one four-window C group into the worker staging
/// buffer; see the kernel.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
#[inline]
pub(super) fn stage_c_group(src: &[u8], dst: &mut [u8; 4 * 16 * 64]) {
    // SAFETY: the cfg gate supplies the features; both sides are bounded.
    unsafe {
        x86_64::stage_c_group_x86_avx512(src, dst);
    }
}

/// Fused GFNI DirectFold4 C drain: the whole bit-transpose + mask-build +
/// nibble-LUT chain of [`accumulate_c_banks_prebuilt`] replaced by two bit
/// transposes and a bit-matrix product straight off `c_packed`. `c_group` is
/// the group's four contiguous 1 KiB windows, `n_b_med` their live medium-row
/// counts, `mats` the group's 32 matrix qwords, `plane_banks` the byte-plane
/// C bank store. Bit-identical banks; see the kernel.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
#[inline]
/// First-write form of [`accumulate_c_banks_fold4_fused_gfni`], for the first
/// live group of a band.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
#[inline]
pub(super) fn write_c_banks_fold4_fused_gfni(
    c_group: &[u8],
    n_b_med: &[usize; 4],
    mats: &[u64; super::C_FOLD4_MATS_PER_GROUP],
    plane_banks: &mut [u8; super::C_PLANE_BANK_BYTES],
) {
    // SAFETY: the cfg gate guarantees the SIMD features; the caller's slice
    // covers the group's four windows and the plane store is fixed-size.
    unsafe {
        x86_64::write_c_banks_fold4_fused_x86_gfni(c_group, n_b_med, mats, plane_banks);
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
pub(super) fn accumulate_c_banks_fold4_fused_gfni(
    c_group: &[u8],
    n_b_med: &[usize; 4],
    mats: &[u64; super::C_FOLD4_MATS_PER_GROUP],
    plane_banks: &mut [u8; super::C_PLANE_BANK_BYTES],
) {
    // SAFETY: the cfg gate guarantees the SIMD features; the caller's slice
    // covers the group's four windows and the plane store is fixed-size.
    unsafe {
        x86_64::accumulate_c_banks_fold4_fused_x86_gfni(c_group, n_b_med, mats, plane_banks);
    }
}

#[inline]
pub(super) fn accumulate_convert_ab(
    chunk_ab_bytes: &[[u8; 64]; 16],
    n_b_med: usize,
    convert: &[super::F128],
    eq_lo_val: super::F128,
    partial_ab: &mut [super::F128; 64],
) {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    // SAFETY: the cfg gate supplies the SIMD features and fixed arrays bound all accesses.
    unsafe {
        if ab_nibble_lut_enabled() {
            x86_64::accumulate_convert_ab_x86_avx512_nibble(
                chunk_ab_bytes,
                n_b_med,
                convert,
                eq_lo_val,
                partial_ab,
            );
        } else {
            x86_64::accumulate_convert_ab_x86_avx512(
                chunk_ab_bytes,
                n_b_med,
                convert,
                eq_lo_val,
                partial_ab,
            );
        }
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    )))]
    for lane in 0..64 {
        let mut converted_ab = super::F128::ZERO;
        for b_med in 0..n_b_med {
            converted_ab += convert[b_med * 256 + usize::from(chunk_ab_bytes[b_med][lane])];
        }
        partial_ab[lane] += converted_ab * eq_lo_val;
    }
}

/// Ranked default is the nibble-LUT AB convert. `FLOCK_NO_AB_NIBBLE=1`
/// restores the scalar 256-entry GPR kernel for one-process A/B; the ranked
/// worker's cleared env never sets it. Production `convert[b][v] = γ^b · φ₈(v)`
/// is F₂-linear in `v`, so a byte lookup equals the XOR of two nibble lookups.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
fn ab_nibble_lut_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_AB_NIBBLE").is_none())
}

/// Ranked default is the nibble-LUT drain. `FLOCK_NO_C_NIBBLE=1` restores the
/// gather kernel for one-process A/B; the ranked worker's cleared env never
/// sets it. Production mask tables are F₂-linear XOR-doubling subset sums, so
/// a byte lookup equals the XOR of two nibble lookups. The nibble kernel's
/// mask build is a 64-lane `vptestmb` pack (AVX-512BW); the gather kill-switch
/// path stays AVX-512F-only.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
fn c_nibble_lut_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_C_NIBBLE").is_none())
}

/// SoA form of the four 16-entry F₂-linear nibble tables used by DirectC.
/// One table depends only on `eq_lo`, so round one can build it once and reuse
/// it across every `x_hi` instead of reconstructing it in the hot drain.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
#[repr(C, align(64))]
pub(super) struct CBankNibbleLut {
    pub(super) lo_n0_lo: [u64; 16],
    pub(super) lo_n0_hi: [u64; 16],
    pub(super) lo_n1_lo: [u64; 16],
    pub(super) lo_n1_hi: [u64; 16],
    pub(super) hi_n0_lo: [u64; 16],
    pub(super) hi_n0_hi: [u64; 16],
    pub(super) hi_n1_lo: [u64; 16],
    pub(super) hi_n1_hi: [u64; 16],
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
impl CBankNibbleLut {
    fn new(mask_tables: &[super::F128]) -> Self {
        debug_assert_eq!(mask_tables.len(), 512);
        let (t_lo, t_hi) = mask_tables.split_at(256);
        let mut lut = Self {
            lo_n0_lo: [0; 16],
            lo_n0_hi: [0; 16],
            lo_n1_lo: [0; 16],
            lo_n1_hi: [0; 16],
            hi_n0_lo: [0; 16],
            hi_n0_hi: [0; 16],
            hi_n1_lo: [0; 16],
            hi_n1_hi: [0; 16],
        };
        for i in 0..16 {
            lut.lo_n0_lo[i] = t_lo[i].lo;
            lut.lo_n0_hi[i] = t_lo[i].hi;
            lut.lo_n1_lo[i] = t_lo[i << 4].lo;
            lut.lo_n1_hi[i] = t_lo[i << 4].hi;
            lut.hi_n0_lo[i] = t_hi[i].lo;
            lut.hi_n0_hi[i] = t_hi[i].hi;
            lut.hi_n1_lo[i] = t_hi[i << 4].lo;
            lut.hi_n1_hi[i] = t_hi[i << 4].hi;
        }
        lut
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
pub(super) fn build_c_bank_nibble_luts(mask_tables: &[super::F128]) -> Vec<CBankNibbleLut> {
    debug_assert_eq!(mask_tables.len() % 512, 0);
    mask_tables
        .chunks_exact(512)
        .map(CBankNibbleLut::new)
        .collect()
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
#[inline]
pub(super) fn accumulate_c_banks_prebuilt(
    c_block: &[u8; 16 * 64],
    n_b_med: usize,
    mask_tables: &[super::F128],
    lut: &CBankNibbleLut,
    partial_c: &mut [[super::F128; 64]; 8],
) {
    // SAFETY: cfg supplies the features and fixed arrays bound all accesses.
    unsafe {
        if c_nibble_lut_enabled() {
            x86_64::accumulate_c_banks_x86_avx512_nibble_prebuilt(c_block, n_b_med, lut, partial_c);
        } else {
            x86_64::accumulate_c_banks_x86_avx512(c_block, n_b_med, mask_tables, partial_c);
        }
    }
}

/// Accumulate the eight alpha-free C banks from already-transposed C rows.
#[inline]
pub(super) fn accumulate_c_banks(
    c_block: &[u8; 16 * 64],
    n_b_med: usize,
    mask_tables: &[super::F128],
    partial_c: &mut [[super::F128; 64]; 8],
) {
    debug_assert!(n_b_med <= 16);
    debug_assert_eq!(mask_tables.len(), 512);

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    // SAFETY: the cfg gate supplies the SIMD features; the fixed-size C and
    // partial arrays and the asserted 512-entry table bound every access.
    unsafe {
        if c_nibble_lut_enabled() {
            x86_64::accumulate_c_banks_x86_avx512_nibble(c_block, n_b_med, mask_tables, partial_c);
        } else {
            x86_64::accumulate_c_banks_x86_avx512(c_block, n_b_med, mask_tables, partial_c);
        }
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    )))]
    accumulate_c_banks_scalar(c_block, n_b_med, mask_tables, partial_c);
}

/// Portable DirectC drain and cross-check oracle for the AVX-512 kernel.
#[allow(dead_code)] // unused outside tests in native AVX-512 builds
#[inline]
pub(super) fn accumulate_c_banks_scalar(
    c_block: &[u8; 16 * 64],
    n_b_med: usize,
    mask_tables: &[super::F128],
    partial_c: &mut [[super::F128; 64]; 8],
) {
    debug_assert!(n_b_med <= 16);
    debug_assert_eq!(mask_tables.len(), 512);
    let (t_lo, t_hi) = mask_tables.split_at(256);
    for lane in 0..64 {
        let mut masks = [0u16; 8];
        for b_med in 0..n_b_med {
            let c = c_block[b_med * 64 + lane];
            for (bank, mask) in masks.iter_mut().enumerate() {
                *mask |= u16::from((c >> bank) & 1) << b_med;
            }
        }
        for (bank, mask) in partial_c.iter_mut().zip(masks) {
            bank[lane] += t_lo[usize::from(mask & 0xff)] + t_hi[usize::from(mask >> 8)];
        }
    }
}
#[allow(dead_code, clippy::too_many_arguments)]
#[inline]
pub(super) fn accumulate_convert_with_s_hat_v(
    chunk_ab_bytes: &[[u8; 64]; 16],
    chunk_c_bytes: &[[u8; 64]; 16],
    n_b_med: usize,
    convert: &[super::F128],
    eq_lo_val: super::F128,
    partial_ab: &mut [super::F128; 64],
    partial_c_0: &mut [super::F128; 64],
    partial_c_1: &mut [super::F128; 64],
) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: aarch64 statically guarantees NEON and the fixed arrays cover
    // all table-selected loads.
    unsafe {
        aarch64::accumulate_convert_with_s_hat_v(
            chunk_ab_bytes,
            chunk_c_bytes,
            n_b_med,
            convert,
            eq_lo_val,
            partial_ab,
            partial_c_0,
            partial_c_1,
        );
    }

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    // SAFETY: the cfg gate guarantees the SIMD features and the fixed arrays
    // cover every four-lane load/store.
    unsafe {
        x86_64::accumulate_convert_with_s_hat_v_x86_avx512(
            chunk_ab_bytes,
            chunk_c_bytes,
            n_b_med,
            convert,
            eq_lo_val,
            partial_ab,
            partial_c_0,
            partial_c_1,
        );
    }

    #[cfg(not(any(
        target_arch = "aarch64",
        all(
            target_arch = "x86_64",
            target_feature = "avx512f",
            target_feature = "vpclmulqdq"
        )
    )))]
    portable::accumulate_convert_with_s_hat_v(
        chunk_ab_bytes,
        chunk_c_bytes,
        n_b_med,
        convert,
        eq_lo_val,
        partial_ab,
        partial_c_0,
        partial_c_1,
    );
}

#[cfg(test)]
mod nibble_algebra_tests {
    use super::super::F128;
    use super::accumulate_c_banks_scalar;

    fn xor_doubling_tables() -> [F128; 512] {
        let mut tables = [F128::ZERO; 512];
        let mut basis = [F128::ZERO; 16];
        basis[0] = F128::new(0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210);
        for b in 1..16 {
            let x = basis[b - 1];
            basis[b] = F128::new(
                x.lo.rotate_left(1) ^ x.hi.wrapping_mul(0x9e37_79b9_7f4a_7c15),
                x.hi.rotate_left(1) ^ x.lo.wrapping_mul(0xd6e8_feb8_6659_fd93),
            );
        }
        let (t_lo, t_hi) = tables.split_at_mut(256);
        for (half, table) in [t_lo, t_hi].into_iter().enumerate() {
            table[0] = F128::ZERO;
            for b in 0..8 {
                let add = basis[half * 8 + b];
                let (done, rest) = table.split_at_mut(1 << b);
                for (out, seen) in rest[..1 << b].iter_mut().zip(done.iter()) {
                    *out = F128::new(seen.lo ^ add.lo, seen.hi ^ add.hi);
                }
            }
        }
        tables
    }

    #[test]
    fn xor_doubling_byte_equals_nibble_xor() {
        let tables = xor_doubling_tables();
        let (t_lo, t_hi) = tables.split_at(256);
        for byte in 0usize..256 {
            let n0 = byte & 0xf;
            let n1 = (byte >> 4) << 4;
            assert_eq!(
                t_lo[byte],
                F128::new(t_lo[n0].lo ^ t_lo[n1].lo, t_lo[n0].hi ^ t_lo[n1].hi),
                "t_lo byte={byte}"
            );
            assert_eq!(
                t_hi[byte],
                F128::new(t_hi[n0].lo ^ t_hi[n1].lo, t_hi[n0].hi ^ t_hi[n1].hi),
                "t_hi byte={byte}"
            );
        }
    }

    fn nibble_drain(
        c_block: &[u8; 16 * 64],
        n_b_med: usize,
        mask_tables: &[F128],
        partial_c: &mut [[F128; 64]; 8],
    ) {
        let (t_lo, t_hi) = mask_tables.split_at(256);
        for lane in 0..64 {
            let mut masks = [0u16; 8];
            for b_med in 0..n_b_med {
                let c = c_block[b_med * 64 + lane];
                for (bank, mask) in masks.iter_mut().enumerate() {
                    *mask |= u16::from((c >> bank) & 1) << b_med;
                }
            }
            for (bank, mask) in partial_c.iter_mut().zip(masks) {
                let lo = usize::from(mask & 0xff);
                let hi = usize::from(mask >> 8);
                bank[lane] +=
                    t_lo[lo & 0xf] + t_lo[(lo >> 4) << 4] + t_hi[hi & 0xf] + t_hi[(hi >> 4) << 4];
            }
        }
    }

    #[test]
    fn nibble_drain_matches_scalar_on_linear_tables() {
        let mut c_block = [0u8; 16 * 64];
        for (i, byte) in c_block.iter_mut().enumerate() {
            *byte = (i as u8).wrapping_mul(0x9d).rotate_left((i & 7) as u32) ^ (i >> 3) as u8;
        }
        let tables = xor_doubling_tables();
        for n_b_med in 0..=16 {
            let mut scalar = [[F128::ZERO; 64]; 8];
            let mut nibble = [[F128::ZERO; 64]; 8];
            for (bank_index, bank) in scalar.iter_mut().enumerate() {
                for (lane, value) in bank.iter_mut().enumerate() {
                    *value = F128::new(
                        ((bank_index * 64 + lane) as u64).wrapping_mul(0xa24b_aed4_963e_e407),
                        ((lane * 8 + bank_index) as u64).wrapping_mul(0x9fb2_1c65_1e98_df25),
                    );
                }
            }
            nibble.copy_from_slice(&scalar);
            accumulate_c_banks_scalar(&c_block, n_b_med, &tables, &mut scalar);
            nibble_drain(&c_block, n_b_med, &tables, &mut nibble);
            assert_eq!(nibble, scalar, "n_b_med={n_b_med}");
        }
    }

    fn ab_nibble_convert(
        chunk_ab_bytes: &[[u8; 64]; 16],
        n_b_med: usize,
        convert: &[F128],
        eq_lo_val: F128,
        partial_ab: &mut [F128; 64],
    ) {
        for lane in 0..64 {
            let mut converted = F128::ZERO;
            for b_med in 0..n_b_med {
                let byte = usize::from(chunk_ab_bytes[b_med][lane]);
                let base = b_med * 256;
                converted += convert[base + (byte & 0xf)] + convert[base + ((byte >> 4) << 4)];
            }
            partial_ab[lane] += converted * eq_lo_val;
        }
    }

    fn ab_byte_convert(
        chunk_ab_bytes: &[[u8; 64]; 16],
        n_b_med: usize,
        convert: &[F128],
        eq_lo_val: F128,
        partial_ab: &mut [F128; 64],
    ) {
        for lane in 0..64 {
            let mut converted = F128::ZERO;
            for b_med in 0..n_b_med {
                converted += convert[b_med * 256 + usize::from(chunk_ab_bytes[b_med][lane])];
            }
            partial_ab[lane] += converted * eq_lo_val;
        }
    }

    #[test]
    fn convert_table_byte_equals_nibble_xor() {
        let convert = super::super::convert_table();
        for b in 0..16 {
            let base = b * 256;
            for byte in 0usize..256 {
                let n0 = byte & 0xf;
                let n1 = (byte >> 4) << 4;
                assert_eq!(
                    convert[base + byte],
                    convert[base + n0] + convert[base + n1],
                    "b={b} byte={byte:#04x}"
                );
            }
        }
    }

    #[test]
    fn ab_nibble_matches_byte_lookup_on_convert_table() {
        let convert = super::super::convert_table();
        let mut chunk_ab_bytes = [[0u8; 64]; 16];
        for b_med in 0..16 {
            for lane in 0..64 {
                chunk_ab_bytes[b_med][lane] = (b_med * 17 + lane * 13) as u8 ^ 0x5a;
            }
        }
        let eq = F128::new(0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210);
        for n_b_med in 0..=16 {
            let mut seed = [F128::ZERO; 64];
            for (lane, value) in seed.iter_mut().enumerate() {
                *value = F128::new(
                    (lane as u64).wrapping_mul(0xa24b_aed4_963e_e407),
                    (lane as u64).wrapping_mul(0x9fb2_1c65_1e98_df25),
                );
            }
            let mut byte_acc = seed;
            let mut nibble_acc = seed;
            ab_byte_convert(&chunk_ab_bytes, n_b_med, convert, eq, &mut byte_acc);
            ab_nibble_convert(&chunk_ab_bytes, n_b_med, convert, eq, &mut nibble_acc);
            assert_eq!(nibble_acc, byte_acc, "n_b_med={n_b_med}");
        }
    }
}
