#[cfg(all(
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
use super::super::{C_FOLD4_MATS_PER_GROUP, C_PLANE_BANK_BYTES, N_C_BANKS, N_C_Q};
#[cfg(all(target_feature = "avx512f", target_feature = "vpclmulqdq"))]
use super::super::{ELL, F128, N_MEDIUM};
#[cfg(target_feature = "gfni")]
use super::super::{F8, InvNttTableByteSingleGf8, N_CHUNKS};

/// algorithm. `_mm512_permutexvar_epi8` does the byte-gather (NEON `vqtbl4q`)
/// in one instruction; the three masked bit-swap rounds (distances 7/14/28)
/// are identical to the NEON version, applied to all eight 64-bit lanes at once.
///
/// Replaces `bit_transpose_64bytes_scalar` (512 branchy bit ops/call) — which
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi"
))]
#[target_feature(enable = "avx512vbmi,avx512bw,avx512f")]
pub(crate) unsafe fn bit_transpose_64bytes_avx512(input: &[u8; 64], output: &mut [u8; 64]) {
    use core::arch::x86_64::*;
    // Gather index = NEON IDX0 ++ IDX1 ++ IDX2 ++ IDX3 (the 8×8 byte transpose).
    #[rustfmt::skip]
    const IDX: [i8; 64] = [
        0, 8, 16, 24, 32, 40, 48, 56,  1, 9, 17, 25, 33, 41, 49, 57,
        2, 10, 18, 26, 34, 42, 50, 58,  3, 11, 19, 27, 35, 43, 51, 59,
        4, 12, 20, 28, 36, 44, 52, 60,  5, 13, 21, 29, 37, 45, 53, 61,
        6, 14, 22, 30, 38, 46, 54, 62,  7, 15, 23, 31, 39, 47, 55, 63,
    ];
    unsafe {
        let inp = _mm512_loadu_si512(input.as_ptr() as *const __m512i);
        let idx = _mm512_loadu_si512(IDX.as_ptr() as *const __m512i);
        let mut y = _mm512_permutexvar_epi8(idx, inp); // y[i] = input[IDX[i]]

        let mask1 = _mm512_set1_epi64(0x00AA00AA00AA00AAu64 as i64);
        let mask2 = _mm512_set1_epi64(0x0000CCCC0000CCCCu64 as i64);
        let mask3 = _mm512_set1_epi64(0x00000000F0F0F0F0u64 as i64);

        let t = _mm512_and_si512(_mm512_xor_si512(y, _mm512_srli_epi64::<7>(y)), mask1);
        y = _mm512_xor_si512(y, _mm512_xor_si512(t, _mm512_slli_epi64::<7>(t)));
        let t = _mm512_and_si512(_mm512_xor_si512(y, _mm512_srli_epi64::<14>(y)), mask2);
        y = _mm512_xor_si512(y, _mm512_xor_si512(t, _mm512_slli_epi64::<14>(t)));
        let t = _mm512_and_si512(_mm512_xor_si512(y, _mm512_srli_epi64::<28>(y)), mask3);
        y = _mm512_xor_si512(y, _mm512_xor_si512(t, _mm512_slli_epi64::<28>(t)));

        _mm512_storeu_si512(output.as_mut_ptr() as *mut __m512i, y);
    }
}

/// SSE/GFNI x86 kernel. The inverse-NTT apply uses its best available x86 path,
/// writes two 64-byte columns, and this kernel multiplies them four XMM chunks
/// at a time. Kept as the fallback for GFNI CPUs without AVX-512.
#[inline]
#[allow(dead_code)] // unused in native AVX-512 builds; exercised by its oracle test
#[cfg(all(target_arch = "x86_64", target_feature = "gfni"))]
#[target_feature(enable = "gfni,sse2")]
pub(crate) unsafe fn shift_reduce_inner_ab_x86_sse(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; 64],
    a_col: &mut [F8],
    b_col: &mut [F8],
) {
    use core::arch::x86_64::*;
    let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;

    // SAFETY: function carries gfni+sse2; raw loads/stores stay within the
    // validated `a_col`/`b_col` (len ELL) and `out` ([u8; 64]) buffers.
    unsafe {
        // 4 byte-accumulators × 16 lanes = ELL = 64 lanes, reduced F_8 values.
        let mut acc = [_mm_setzero_si128(); 4];
        for k in 0..8usize {
            let chunk_off = byte_base_b + k * N_CHUNKS;
            inv_table.apply(&a_packed[chunk_off..chunk_off + N_CHUNKS], a_col);
            inv_table.apply(&b_packed[chunk_off..chunk_off + N_CHUNKS], b_col);
            let a_ptr = a_col.as_ptr() as *const u8;
            let b_ptr = b_col.as_ptr() as *const u8;
            let xk = _mm_set1_epi8((1u8 << k) as i8); // x^k as an F_8 byte; k=0 ⇒ 1
            for c in 0..4usize {
                let av = _mm_loadu_si128(a_ptr.add(c * 16) as *const __m128i);
                let bv = _mm_loadu_si128(b_ptr.add(c * 16) as *const __m128i);
                // y = (a·b) · x^k in F_8. For k=0, xk=1 ⇒ second mul is identity.
                let y = _mm_gf2p8mul_epi8(_mm_gf2p8mul_epi8(av, bv), xk);
                acc[c] = _mm_xor_si128(acc[c], y);
            }
        }
        let out_ptr = out.as_mut_ptr();
        for c in 0..4usize {
            _mm_storeu_si128(out_ptr.add(c * 16) as *mut __m128i, acc[c]);
        }
    }
}

/// `FLOCK_NO_URM_APPLY_2IMG=1` restores the one-image inverse-NTT table
/// apply (ten port-5 shuffles per apply) in the shift-reduce AB kernel.
/// Resolved once per process.
#[allow(dead_code)] // Retained same-binary rollback selector.
pub(crate) fn urm_apply_2img_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_URM_APPLY_2IMG").is_none());
    *ON
}

/// `FLOCK_NO_URM_PIDX=1` restores per-byte index scaling (`movzbl` + `shl $6`)
/// in the shift-reduce AB kernel instead of the pre-scaled `u16` offset
/// buffer. Resolved once per process.
#[allow(dead_code)] // Retained same-binary rollback selector.
pub(crate) fn urm_pidx_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_URM_PIDX").is_none());
    *ON
}

/// `FLOCK_NO_URM_OFF_ARENA=1` restores the incumbent in-kernel offset
/// prologue in the shift-reduce AB kernel (and disables the producer-side
/// fused offset arena that feeds the split consume body). Resolved once per
/// process.
#[allow(dead_code)] // Retained same-binary rollback selector.
pub(crate) fn urm_off_arena_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_URM_OFF_ARENA").is_none());
    *ON
}

/// `FLOCK_NO_URM_OFFW=1` restores the eight separate 16-bit reads of the
/// pre-scaled offset buffer in the shift-reduce AB kernel instead of two
/// 64-bit reads split with shifts. Resolved once per process.
#[allow(dead_code)] // Retained same-binary rollback selector.
pub(crate) fn urm_offw_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_URM_OFFW").is_none());
    *ON
}

/// Terminal 64-byte store for the shift-reduce AB kernels. `nt` selects the
/// store class, decided once per precompute call by the producer:
/// - `0`: temporal `storeu` (the incumbent; all in-fold callers).
/// - `1`: four XMM non-temporal streams (16-aligned destination).
/// - `2`: one ZMM non-temporal stream (64-aligned destination).
/// NT callers own the visibility contract: writes cross a thread boundary
/// only after an `_mm_sfence()` on the producing thread (see
/// [`super::super::abinner_publish_fence`]).
#[inline(always)]
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
pub(crate) unsafe fn store_out64(out: &mut [u8; 64], acc: core::arch::x86_64::__m512i, nt: u8) {
    use core::arch::x86_64::*;
    // SAFETY: `out` is 64 writable bytes; alignment per the `nt` contract.
    unsafe {
        if nt == 2 {
            _mm512_stream_si512(out.as_mut_ptr() as *mut __m512i, acc);
        } else {
            store_out64_split(out, acc, nt);
        }
    }
}

/// The `nt = 1` and `nt = 0` store classes of [`store_out64`], out of line so
/// the ZMM-stream class does not carry their register allocation into the
/// kernels that inline it.
///
/// # Safety
/// As for [`store_out64`].
#[inline(never)]
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn store_out64_split(out: &mut [u8; 64], acc: core::arch::x86_64::__m512i, nt: u8) {
    use core::arch::x86_64::*;
    // SAFETY: forwarded from [`store_out64`]'s contract.
    unsafe {
        if nt == 1 {
            let p = out.as_mut_ptr();
            _mm_stream_si128(p as *mut __m128i, _mm512_extracti32x4_epi32::<0>(acc));
            _mm_stream_si128(
                p.add(16) as *mut __m128i,
                _mm512_extracti32x4_epi32::<1>(acc),
            );
            _mm_stream_si128(
                p.add(32) as *mut __m128i,
                _mm512_extracti32x4_epi32::<2>(acc),
            );
            _mm_stream_si128(
                p.add(48) as *mut __m128i,
                _mm512_extracti32x4_epi32::<3>(acc),
            );
        } else {
            _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, acc);
        }
    }
}

/// Pre-scaled-index twin of [`shift_reduce_inner_ab_x86_avx512`] (two-image
/// apply only). Bit-identical output — every table address is the same — but
/// the eight-per-apply `movzbl` + `shl $0x6` index pairs become a single
/// `movzwl` each, because the window's 128 input bytes are widened to `u16`
/// and multiplied by the row stride 64 up front with four `vpmovzxbw` +
/// `vpsllw` + store triples.
///
/// which is what makes the two SMT siblings add up to one core), the incumbent
/// loop branch, so trading 128 shifts for 4 loads + 4 shifts + 4 stores moves
/// work off the two ports that are actually full onto ones that are not.
///
/// reloads of the offset buffer do not stall on the 64-byte stores.
#[inline]
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
#[target_feature(enable = "gfni,avx512f,avx512bw")]
pub(crate) unsafe fn shift_reduce_inner_ab_x86_avx512_pidx(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; 64],
    nt: u8,
    offw: bool,
    imgs: (*const u8, *const u8),
) {
    use core::arch::x86_64::*;
    let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;

    #[repr(align(64))]
    struct Off([u16; 128]);

    // SAFETY: the caller's packed-input bounds guarantee 64 readable bytes at
    // `byte_base_b` in both A and B (the incumbent kernel reads exactly that
    // range across its eight K rows). `off` is a fully written 128-u16 stack
    // buffer, and `out` is one writable ZMM register.
    unsafe {
        let a0 = a_packed.as_ptr().add(byte_base_b);
        let b0 = b_packed.as_ptr().add(byte_base_b);

        let mut off = core::mem::MaybeUninit::<Off>::uninit();
        let op = core::ptr::addr_of_mut!((*off.as_mut_ptr()).0) as *mut u16;
        // byte -> byte * 64, 32 lanes at a time.
        let scale = |p: *const u8| {
            _mm512_slli_epi16::<6>(_mm512_cvtepu8_epi16(_mm256_loadu_si256(
                p as *const __m256i,
            )))
        };
        _mm512_store_si512(op as *mut __m512i, scale(a0));
        _mm512_store_si512(op.add(32) as *mut __m512i, scale(a0.add(32)));
        _mm512_store_si512(op.add(64) as *mut __m512i, scale(b0));
        _mm512_store_si512(op.add(96) as *mut __m512i, scale(b0.add(32)));

        let acc = if offw {
            horner_2img_offw(imgs, op)
        } else {
            horner_2img_off_narrow(inv_table, op)
        };
        store_out64(out, acc, nt);
    }
}

/// The pre-scaled-offset PROLOGUE of
/// [`shift_reduce_inner_ab_x86_avx512_pidx`] alone: widen the window's
/// 128 a/b bytes to `u16` and multiply by the row stride 64, into a
/// caller-owned 128-`u16` block. Identical stores to the incumbent body's
/// stack staging; splitting it out lets a producer build several blocks'
/// offsets BEFORE any consume, so the consuming loads never sit in the
/// shadow of their own ZMM stores.
///
/// # Safety
/// 64 readable bytes at `a0` and `b0`; `op` is 64-byte aligned with 128
/// writable `u16`s.
#[inline]
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
#[target_feature(enable = "gfni,avx512f,avx512bw")]
pub(crate) unsafe fn shift_reduce_ab_offsets_build(a0: *const u8, b0: *const u8, op: *mut u16) {
    use core::arch::x86_64::*;
    // SAFETY: forwarded from this function's contract.
    unsafe {
        let scale = |p: *const u8| {
            _mm512_slli_epi16::<6>(_mm512_cvtepu8_epi16(_mm256_loadu_si256(
                p as *const __m256i,
            )))
        };
        _mm512_store_si512(op as *mut __m512i, scale(a0));
        _mm512_store_si512(op.add(32) as *mut __m512i, scale(a0.add(32)));
        _mm512_store_si512(op.add(64) as *mut __m512i, scale(b0));
        _mm512_store_si512(op.add(96) as *mut __m512i, scale(b0.add(32)));
    }
}

/// The CONSUME half of [`shift_reduce_inner_ab_x86_avx512_pidx`] (two-image
/// wide-read Horner + terminal store), fed from offsets prebuilt by
/// [`shift_reduce_ab_offsets_build`]. Bit-identical output: identical table
/// addresses, identical arithmetic, identical store class.
///
/// # Safety
/// `op` holds this window-block's 128 pre-scaled offsets; `imgs` are the
/// table's base and σ₈ image pointers; `out`/`nt` as for [`store_out64`].
#[inline]
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
#[target_feature(enable = "gfni,avx512f,avx512bw")]
pub(crate) unsafe fn shift_reduce_inner_ab_x86_avx512_from_off(
    op: *const u16,
    out: &mut [u8; 64],
    nt: u8,
    imgs: (*const u8, *const u8),
) {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        let acc = horner_2img_offw(imgs, op);
        store_out64(out, acc, nt);
    }
}

/// Fixed-`nt=2` twin of [`shift_reduce_inner_ab_x86_avx512_from_off`] for the
/// measured ranked offset consumer. The destination class is part of that
/// producer's contract, so the terminal store is one unconditional ZMM
/// non-temporal stream instead of entering [`store_out64`]'s selector.
///
/// # Safety
/// As for [`shift_reduce_inner_ab_x86_avx512_from_off`], with `out` additionally
/// 64-byte aligned. The caller must publish an `_mm_sfence()` before another
/// thread observes the output.
#[inline]
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
#[target_feature(enable = "gfni,avx512f,avx512bw")]
pub(crate) unsafe fn shift_reduce_inner_ab_x86_avx512_from_off_nt2(
    op: *const u16,
    out: &mut [u8; 64],
    imgs: (*const u8, *const u8),
) {
    use core::arch::x86_64::*;
    // SAFETY: forwarded from this function's contract. Unlike the generic
    // twin, ranked nt=2 makes alignment/store class invariant by construction.
    unsafe {
        let acc = horner_2img_offw(imgs, op);
        _mm512_stream_si512(out.as_mut_ptr() as *mut __m512i, acc);
    }
}

/// Ranked residual-AB fixed-stream-store leaf. The two admitted masks are
/// block 2's K2..7 and block 29's K0..3. Dispatch is outside the arithmetic
/// body, so each arm statically deletes the omitted inverse-table applies.
#[inline(always)]
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
pub(crate) unsafe fn shift_reduce_inner_ab_x86_avx512_from_off_nt2_residual(
    op: *const u16,
    out: &mut [u8; 64],
    imgs: (*const u8, *const u8),
    keep: u8,
) {
    use core::arch::x86_64::*;
    unsafe {
        let acc = match keep {
            0xfc => residual_2img_offw_k2_7(imgs, op),
            0x0f => residual_2img_offw_k0_3(imgs, op),
            _ => core::hint::unreachable_unchecked(),
        };
        _mm512_stream_si512(out.as_mut_ptr() as *mut __m512i, acc);
    }
}

/// Horner over x: Σ_k x^k·y_k = y_0 + x·(y_1 + x·(… + x·y_7)). Same count of
/// `vgf2p8mulb` as the explicit x^k form (8 products + 7 scalings), but the
/// multiplier is the loop-invariant x = 0x02, so the per-iteration `mov $1` /
/// `shl %cl` / `vpbroadcastb` that rebuilt x^k disappear. GF(2^8)
/// multiplication is associative and distributes over XOR, so the value is
/// bit-identical.
///
/// The eight pre-scaled `u16` offsets of every apply are fetched as two
/// 64-bit reads and split with shifts. `imgs` are the table images the caller
/// resolved for this run of windows.
///
/// # Safety
/// As for [`shift_reduce_inner_ab_x86_avx512_pidx`], with `imgs` the table's
/// base and σ₈ image pointers.
#[inline(always)]
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
unsafe fn horner_2img_offw(
    imgs: (*const u8, *const u8),
    op: *const u16,
) -> core::arch::x86_64::__m512i {
    use core::arch::x86_64::*;
    // SAFETY: forwarded from the caller's contract.
    unsafe {
        let apply = |o: *const u16| {
            crate::ntt::inv_table::apply_x86_avx512_register_2img_offw_at(imgs.0, imgs.1, o)
        };
        let xb = _mm512_set1_epi8(2);
        let mut acc = _mm512_gf2p8mul_epi8(apply(op.add(7 * 8)), apply(op.add(64 + 7 * 8)));
        for k in (0..7usize).rev() {
            let av = apply(op.add(k * 8));
            let bv = apply(op.add(64 + k * 8));
            let product = _mm512_gf2p8mul_epi8(av, bv);
            acc = _mm512_xor_si512(_mm512_gf2p8mul_epi8(acc, xb), product);
        }
        acc
    }
}

/// Explicit ranked block-2 subset of [`horner_2img_offw`]. Keeping literal
/// K indices here prevents LLVM from retaining a six-trip control loop.
#[inline(always)]
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
unsafe fn residual_2img_offw_k2_7(
    imgs: (*const u8, *const u8),
    op: *const u16,
) -> core::arch::x86_64::__m512i {
    use core::arch::x86_64::*;
    unsafe {
        let apply = |o: *const u16| {
            crate::ntt::inv_table::apply_x86_avx512_register_2img_offw_at(imgs.0, imgs.1, o)
        };
        macro_rules! scaled {
            ($k:literal) => {{
                let product =
                    _mm512_gf2p8mul_epi8(apply(op.add($k * 8)), apply(op.add(64 + $k * 8)));
                _mm512_gf2p8mul_epi8(product, _mm512_set1_epi8((1u8 << $k) as i8))
            }};
        }
        let p2 = scaled!(2);
        let p3 = scaled!(3);
        let p4 = scaled!(4);
        let p5 = scaled!(5);
        let p6 = scaled!(6);
        let p7 = scaled!(7);
        _mm512_xor_si512(
            _mm512_xor_si512(p2, p3),
            _mm512_xor_si512(_mm512_xor_si512(p4, p5), _mm512_xor_si512(p6, p7)),
        )
    }
}

/// Ranked window 30: only K0 is live (A words 480–481, B = 0x0001_ffff_ffff_ffff
/// then zeros). Horner of y_1..y_7 = 0 collapses to y_0 = T(A0)·T(B0).
///
/// # Safety
/// `a`/`b` each supply eight readable bytes at offset 0; `out` is 64-byte
/// aligned; `imgs` are the table's base and σ₈ image. Caller sfence.
#[inline(never)]
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
#[target_feature(enable = "gfni,avx512f,avx512bw")]
pub(crate) unsafe fn shift_reduce_inner_ab_x86_avx512_window30_k0(
    a: &[u8; 64],
    b: &[u8; 64],
    out: &mut [u8; 64],
    imgs: (*const u8, *const u8),
) {
    use core::arch::x86_64::*;
    unsafe {
        let apply = |bytes: *const u8| {
            let row = |img: *const u8, i: usize| {
                _mm512_loadu_si512(img.add(*bytes.add(i) as usize * 64).cast::<__m512i>())
            };
            let u0 = _mm512_xor_si512(row(imgs.0, 0), row(imgs.1, 1));
            let u1 = _mm512_xor_si512(row(imgs.0, 2), row(imgs.1, 3));
            let u2 = _mm512_xor_si512(row(imgs.0, 4), row(imgs.1, 5));
            let u3 = _mm512_xor_si512(row(imgs.0, 6), row(imgs.1, 7));
            let even = _mm512_xor_si512(u0, _mm512_shuffle_i64x2::<0x4E>(u2, u2));
            let odd = _mm512_xor_si512(u1, _mm512_shuffle_i64x2::<0x4E>(u3, u3));
            _mm512_xor_si512(even, _mm512_shuffle_i64x2::<0xB1>(odd, odd))
        };
        let acc = _mm512_gf2p8mul_epi8(apply(a.as_ptr()), apply(b.as_ptr()));
        _mm512_stream_si512(out.as_mut_ptr().cast::<__m512i>(), acc);
    }
}

/// Explicit ranked block-29 subset of [`horner_2img_offw`].
#[inline(always)]
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
unsafe fn residual_2img_offw_k0_3(
    imgs: (*const u8, *const u8),
    op: *const u16,
) -> core::arch::x86_64::__m512i {
    use core::arch::x86_64::*;
    unsafe {
        let apply = |o: *const u16| {
            crate::ntt::inv_table::apply_x86_avx512_register_2img_offw_at(imgs.0, imgs.1, o)
        };
        let p0 = _mm512_gf2p8mul_epi8(apply(op), apply(op.add(64)));
        macro_rules! scaled {
            ($k:literal) => {{
                let product =
                    _mm512_gf2p8mul_epi8(apply(op.add($k * 8)), apply(op.add(64 + $k * 8)));
                _mm512_gf2p8mul_epi8(product, _mm512_set1_epi8((1u8 << $k) as i8))
            }};
        }
        _mm512_xor_si512(
            _mm512_xor_si512(p0, scaled!(1)),
            _mm512_xor_si512(scaled!(2), scaled!(3)),
        )
    }
}

/// [`horner_2img_offw`] with the eight offsets read separately as 16-bit
/// values, out of line so the wide-read form does not carry its register
/// allocation. Identical value and identical table addresses.
/// `FLOCK_NO_URM_OFFW=1` restores it.
///
/// # Safety
/// As for [`horner_2img_offw`].
#[inline(never)]
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
#[target_feature(enable = "gfni,avx512f,avx512bw")]
unsafe fn horner_2img_off_narrow(
    inv_table: &InvNttTableByteSingleGf8,
    op: *const u16,
) -> core::arch::x86_64::__m512i {
    use core::arch::x86_64::*;
    // SAFETY: forwarded from the caller's contract.
    unsafe {
        let apply = |o: *const u16| inv_table.apply_x86_avx512_register_2img_off_unchecked(o);
        let xb = _mm512_set1_epi8(2);
        let mut acc = _mm512_gf2p8mul_epi8(apply(op.add(7 * 8)), apply(op.add(64 + 7 * 8)));
        for k in (0..7usize).rev() {
            let av = apply(op.add(k * 8));
            let bv = apply(op.add(64 + k * 8));
            let product = _mm512_gf2p8mul_epi8(av, bv);
            acc = _mm512_xor_si512(_mm512_gf2p8mul_epi8(acc, xb), product);
        }
        acc
    }
}

/// Fused AVX-512/GFNI x86 kernel. Each inverse-NTT apply returns all 64 F_8
/// evaluations in one ZMM register; the product and x^k scaling stay 64-wide
/// and register-resident through the final XOR accumulation.
#[inline]
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
#[target_feature(enable = "gfni,avx512f,avx512bw")]
pub(crate) unsafe fn shift_reduce_inner_ab_x86_avx512(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; 64],
    nt: u8,
) {
    let img2 = urm_apply_2img_enabled() && inv_table.has_second_image();
    let pidx = img2 && urm_pidx_enabled();
    let offw = pidx && urm_offw_enabled();
    let imgs = if img2 {
        inv_table.image_ptrs()
    } else {
        (core::ptr::null(), core::ptr::null())
    };
    // SAFETY: same contract as this function; the mode flags merely cache
    // the three process-invariant selectors used by the incumbent body.
    unsafe {
        shift_reduce_inner_ab_x86_avx512_prepared(
            a_packed,
            b_packed,
            inv_table,
            chunk_byte_base,
            b_med,
            out,
            nt,
            img2,
            pidx,
            offw,
            imgs,
        );
    }
}

/// [`shift_reduce_inner_ab_x86_avx512`] with its process-invariant mode
/// switches already resolved. Arithmetic, loads and stores are identical.
#[inline]
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
#[target_feature(enable = "gfni,avx512f,avx512bw")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn shift_reduce_inner_ab_x86_avx512_prepared(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; 64],
    nt: u8,
    img2: bool,
    pidx: bool,
    offw: bool,
    imgs: (*const u8, *const u8),
) {
    let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;

    // Two-image apply: 3 port-5 shuffles per table apply instead of 10 (see
    // `apply_x86_avx512_register_2img_unchecked`). Sixteen applies per
    // window over the whole packed input make this the largest port-5-only
    // uop stream in the prover. `FLOCK_NO_URM_APPLY_2IMG=1` restores the
    // one-image form (exact same-binary A/B).
    if img2 && pidx {
        // SAFETY: same contract as this function; `img2` proves the σ₈ image.
        unsafe {
            shift_reduce_inner_ab_x86_avx512_pidx(
                a_packed,
                b_packed,
                inv_table,
                chunk_byte_base,
                b_med,
                out,
                nt,
                offw,
                imgs,
            );
        }
        return;
    }
    // SAFETY: same contract as this function.
    unsafe {
        shift_reduce_inner_ab_x86_avx512_rows(
            a_packed,
            b_packed,
            inv_table,
            byte_base_b,
            out,
            nt,
            img2,
        );
    }
}

/// The explicit-x^k row loop of [`shift_reduce_inner_ab_x86_avx512_prepared`]
/// — the form used whenever the pre-scaled-index apply is not selected. Out of
/// line so the pre-scaled body does not carry its register allocation.
/// `FLOCK_NO_URM_PIDX=1` / `FLOCK_NO_URM_APPLY_2IMG=1` restore it.
///
/// # Safety
/// As for [`shift_reduce_inner_ab_x86_avx512_prepared`]; `byte_base_b` is
/// `chunk_byte_base + b_med * N_CHUNKS * 8`.
#[inline(never)]
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "gfni",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
#[target_feature(enable = "gfni,avx512f,avx512bw")]
unsafe fn shift_reduce_inner_ab_x86_avx512_rows(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    byte_base_b: usize,
    out: &mut [u8; 64],
    nt: u8,
    img2: bool,
) {
    use core::arch::x86_64::*;
    // SAFETY: the caller's packed-input bounds guarantee 8 readable bytes at
    // every K-row offset. The table has the protocol-fixed ell=64/chunks=8
    // shape (and carries the σ₈ image when `img2`), and `out` is exactly one
    // writable ZMM register.
    unsafe {
        let mut acc = _mm512_setzero_si512();
        for k in 0..8usize {
            let off = byte_base_b + k * N_CHUNKS;
            let (av, bv) = if img2 {
                (
                    inv_table.apply_x86_avx512_register_2img_unchecked(a_packed.as_ptr().add(off)),
                    inv_table.apply_x86_avx512_register_2img_unchecked(b_packed.as_ptr().add(off)),
                )
            } else {
                (
                    inv_table.apply_x86_avx512_register_unchecked(a_packed.as_ptr().add(off)),
                    inv_table.apply_x86_avx512_register_unchecked(b_packed.as_ptr().add(off)),
                )
            };
            let product = _mm512_gf2p8mul_epi8(av, bv);
            // x^0 is the multiplicative identity, so avoid one GFNI operation
            // for the first row.
            let scaled = if k == 0 {
                product
            } else {
                _mm512_gf2p8mul_epi8(product, _mm512_set1_epi8((1u8 << k) as i8))
            };
            acc = _mm512_xor_si512(acc, scaled);
        }
        store_out64(out, acc, nt);
    }
}

/// Scalar 256-entry GPR convert. Kill-switch / oracle for the nibble kernel.
/// The C side is table-free (see `kernels::accumulate_c_banks`).
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(crate) unsafe fn accumulate_convert_ab_x86_avx512(
    chunk_ab_bytes: &[[u8; ELL]; 1 << N_MEDIUM],
    n_b_med: usize,
    convert: &[F128],
    eq_lo_val: F128,
    partial_ab: &mut [F128; ELL],
) {
    use crate::field::gf2_128::x86_64::{f128x4_set, ghash_mul_x4};
    use core::arch::x86_64::*;
    debug_assert!(n_b_med <= 1 << N_MEDIUM);
    debug_assert_eq!(ELL % 4, 0);

    // SAFETY: the fixed-size input/partial arrays contain every four-lane load
    // and store below. Convert indices are `b_med * 256 + u8`, bounded by the
    // 16*256-entry table. The cfg gate supplies both required target features.
    unsafe {
        let eq = f128x4_set(eq_lo_val, eq_lo_val, eq_lo_val, eq_lo_val);
        for lane in (0..ELL).step_by(4) {
            let mut cf_ab = [F128::ZERO; 4];
            for b_med in 0..n_b_med {
                let table_base = b_med * 256;
                for j in 0..4 {
                    let v_ab = chunk_ab_bytes[b_med][lane + j] as usize;
                    cf_ab[j] += convert[table_base + v_ab];
                }
            }

            let scaled_ab = ghash_mul_x4(f128x4_set(cf_ab[0], cf_ab[1], cf_ab[2], cf_ab[3]), eq);

            let ab_ptr = partial_ab.as_mut_ptr().add(lane) as *mut __m512i;
            _mm512_storeu_si512(
                ab_ptr,
                _mm512_xor_si512(_mm512_loadu_si512(ab_ptr), scaled_ab),
            );
        }
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
#[repr(C, align(64))]
struct ConvertNibbleLut {
    n0_lo: [[u64; 16]; 16],
    n0_hi: [[u64; 16]; 16],
    n1_lo: [[u64; 16]; 16],
    n1_hi: [[u64; 16]; 16],
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
fn build_convert_nibble_lut(convert: &[F128]) -> ConvertNibbleLut {
    debug_assert!(convert.len() >= 16 * 256);
    let mut lut = ConvertNibbleLut {
        n0_lo: [[0; 16]; 16],
        n0_hi: [[0; 16]; 16],
        n1_lo: [[0; 16]; 16],
        n1_hi: [[0; 16]; 16],
    };
    for b in 0..16 {
        let base = b * 256;
        for i in 0..16 {
            lut.n0_lo[b][i] = convert[base + i].lo;
            lut.n0_hi[b][i] = convert[base + i].hi;
            lut.n1_lo[b][i] = convert[base + (i << 4)].lo;
            lut.n1_hi[b][i] = convert[base + (i << 4)].hi;
        }
    }
    lut
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
fn production_convert_nibble_lut() -> &'static ConvertNibbleLut {
    static LUT: std::sync::OnceLock<ConvertNibbleLut> = std::sync::OnceLock::new();
    LUT.get_or_init(|| build_convert_nibble_lut(super::super::convert_table()))
}

/// F₂-linear nibble-LUT AB convert. Production `convert[b][v] = γ^b · φ₈(v)`
/// is F₂-linear in the bits of `v`: `φ₈` is a field homomorphism (AES-GF(256)
/// addition is XOR) and multiplication by `γ^b` is F₂-linear, so
/// `T[byte] = T[n0] ⊕ T[n1 << 4]`. Same 16-entry SoA `vpermi2q` path as the
/// C-drain nibble kernel; replaces 4 scalar 256-entry GPR loads per 4-lane
/// group. Bit-identical to [`accumulate_convert_ab_x86_avx512`] on linear
/// tables.
#[inline]
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(crate) unsafe fn accumulate_convert_ab_x86_avx512_nibble(
    chunk_ab_bytes: &[[u8; ELL]; 1 << N_MEDIUM],
    n_b_med: usize,
    convert: &[F128],
    eq_lo_val: F128,
    partial_ab: &mut [F128; ELL],
) {
    use crate::field::gf2_128::x86_64::{f128x4_set, ghash_mul_x4};
    use core::arch::x86_64::*;
    debug_assert!(n_b_med <= 1 << N_MEDIUM);
    debug_assert!(convert.len() >= n_b_med * 256);

    let owned;
    let production = super::super::convert_table();
    let lut: &ConvertNibbleLut =
        if convert.as_ptr() == production.as_ptr() && convert.len() == production.len() {
            production_convert_nibble_lut()
        } else {
            owned = build_convert_nibble_lut(convert);
            &owned
        };

    #[inline(always)]
    unsafe fn lookup8(idx: __m512i, table: *const u64) -> __m512i {
        unsafe {
            let a = _mm512_load_si512(table as *const __m512i);
            let b = _mm512_load_si512(table.add(8) as *const __m512i);
            _mm512_permutex2var_epi64(a, idx, b)
        }
    }

    #[inline(always)]
    unsafe fn interleave_aos(los: __m512i, his: __m512i) -> (__m512i, __m512i) {
        unsafe {
            let idx0 = _mm512_set_epi64(11, 3, 10, 2, 9, 1, 8, 0);
            let idx1 = _mm512_set_epi64(15, 7, 14, 6, 13, 5, 12, 4);
            (
                _mm512_permutex2var_epi64(los, idx0, his),
                _mm512_permutex2var_epi64(los, idx1, his),
            )
        }
    }

    // SAFETY: each 16-byte row load is inside a 64-byte AB row; nibble
    // indices are 0..=15 by AND-0xf so every vpermi2q stays in a 16-entry
    // table half. Eq scaling and partial XOR cover four F128s per store.
    unsafe {
        let nibble_mask = _mm512_set1_epi32(0xf);
        let eq = f128x4_set(eq_lo_val, eq_lo_val, eq_lo_val, eq_lo_val);
        for lane_base in (0..ELL).step_by(16) {
            let mut los = [_mm512_setzero_si512(); 2];
            let mut his = [_mm512_setzero_si512(); 2];
            for b_med in 0..n_b_med {
                let row_ptr = chunk_ab_bytes[b_med].as_ptr().add(lane_base);
                let row_bytes = _mm_loadu_si128(row_ptr as *const __m128i);
                let row = _mm512_cvtepu8_epi32(row_bytes);
                let n0 = _mm512_and_si512(row, nibble_mask);
                let n1 = _mm512_and_si512(_mm512_srli_epi32::<4>(row), nibble_mask);
                for group in 0..2 {
                    let n0_8 = if group == 0 {
                        _mm512_cvtepu32_epi64(_mm512_castsi512_si256(n0))
                    } else {
                        _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64::<1>(n0))
                    };
                    let n1_8 = if group == 0 {
                        _mm512_cvtepu32_epi64(_mm512_castsi512_si256(n1))
                    } else {
                        _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64::<1>(n1))
                    };
                    los[group] = _mm512_xor_si512(
                        los[group],
                        _mm512_xor_si512(
                            lookup8(n0_8, lut.n0_lo[b_med].as_ptr()),
                            lookup8(n1_8, lut.n1_lo[b_med].as_ptr()),
                        ),
                    );
                    his[group] = _mm512_xor_si512(
                        his[group],
                        _mm512_xor_si512(
                            lookup8(n0_8, lut.n0_hi[b_med].as_ptr()),
                            lookup8(n1_8, lut.n1_hi[b_med].as_ptr()),
                        ),
                    );
                }
            }
            for group in 0..2 {
                let (aos0, aos1) = interleave_aos(los[group], his[group]);
                let scaled0 = ghash_mul_x4(aos0, eq);
                let scaled1 = ghash_mul_x4(aos1, eq);
                let partial_ptr =
                    partial_ab.as_mut_ptr().add(lane_base + group * 8) as *mut __m512i;
                _mm512_storeu_si512(
                    partial_ptr,
                    _mm512_xor_si512(_mm512_loadu_si512(partial_ptr), scaled0),
                );
                _mm512_storeu_si512(
                    partial_ptr.add(1),
                    _mm512_xor_si512(_mm512_loadu_si512(partial_ptr.add(1)), scaled1),
                );
            }
        }
    }
}

/// AVX-512 DirectC drain. Mask construction works on 16 output lanes at once:
/// each C row is widened from bytes to dwords, then eight `vptestmd` masks
/// select the row's bit into the corresponding bank accumulator. Four F128s
/// are updated per ZMM using two L1-resident qword gathers (one per table).
///
/// This deliberately requires only the dispatch contract's AVX-512F. In
/// particular, it does not silently depend on AVX-512BW or VBMI even though
/// the ranked machine also provides those extensions.
#[inline]
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(crate) unsafe fn accumulate_c_banks_x86_avx512(
    c_block: &[u8; 16 * ELL],
    n_b_med: usize,
    mask_tables: &[F128],
    partial_c: &mut [[F128; ELL]; 8],
) {
    use core::arch::x86_64::*;
    debug_assert_eq!(ELL, 64);
    debug_assert!(n_b_med <= 1 << N_MEDIUM);
    debug_assert_eq!(mask_tables.len(), 512);

    let (t_lo, t_hi) = mask_tables.split_at(256);

    // SAFETY: each row load reads 16 bytes from a 64-byte row of `c_block`;
    // n_b_med <= 16 bounds the row index. Each mask is at most 0xffff, and
    // its low/high bytes therefore index the two 256-entry table halves.
    // Gather indices address the two qwords of four selected F128 values.
    // Each partial load/store covers exactly four lanes in a 64-lane bank.
    unsafe {
        for lane_base in (0..ELL).step_by(16) {
            let mut masks = [_mm512_setzero_si512(); 8];
            let bank_bits = [
                _mm512_set1_epi32(1),
                _mm512_set1_epi32(2),
                _mm512_set1_epi32(4),
                _mm512_set1_epi32(8),
                _mm512_set1_epi32(16),
                _mm512_set1_epi32(32),
                _mm512_set1_epi32(64),
                _mm512_set1_epi32(128),
            ];

            for b_med in 0..n_b_med {
                let row_ptr = c_block.as_ptr().add(b_med * ELL + lane_base);
                let row_bytes = _mm_loadu_si128(row_ptr as *const __m128i);
                let row = _mm512_cvtepu8_epi32(row_bytes);
                let weight = _mm512_set1_epi32((1u32 << b_med) as i32);

                for (mask, bank_bit) in masks.iter_mut().zip(bank_bits) {
                    let selected = _mm512_test_epi32_mask(row, bank_bit);
                    *mask = _mm512_mask_or_epi32(*mask, selected, *mask, weight);
                }
            }

            for (bank, mask) in partial_c.iter_mut().zip(masks) {
                let mut mask_words = [0u32; 16];
                _mm512_storeu_si512(mask_words.as_mut_ptr() as *mut __m512i, mask);

                for lane_in_group in (0..16).step_by(4) {
                    let m0 = i64::from(mask_words[lane_in_group]);
                    let m1 = i64::from(mask_words[lane_in_group + 1]);
                    let m2 = i64::from(mask_words[lane_in_group + 2]);
                    let m3 = i64::from(mask_words[lane_in_group + 3]);

                    // F128 is two adjacent u64s. These qword indices preserve
                    // the in-memory [lo, hi] order for four gathered values.
                    let lo_indices = _mm512_set_epi64(
                        2 * (m3 & 0xff) + 1,
                        2 * (m3 & 0xff),
                        2 * (m2 & 0xff) + 1,
                        2 * (m2 & 0xff),
                        2 * (m1 & 0xff) + 1,
                        2 * (m1 & 0xff),
                        2 * (m0 & 0xff) + 1,
                        2 * (m0 & 0xff),
                    );
                    let hi_indices = _mm512_set_epi64(
                        2 * (m3 >> 8) + 1,
                        2 * (m3 >> 8),
                        2 * (m2 >> 8) + 1,
                        2 * (m2 >> 8),
                        2 * (m1 >> 8) + 1,
                        2 * (m1 >> 8),
                        2 * (m0 >> 8) + 1,
                        2 * (m0 >> 8),
                    );

                    let from_lo =
                        _mm512_i64gather_epi64::<8>(lo_indices, t_lo.as_ptr() as *const i64);
                    let from_hi =
                        _mm512_i64gather_epi64::<8>(hi_indices, t_hi.as_ptr() as *const i64);
                    let partial_ptr =
                        bank.as_mut_ptr().add(lane_base + lane_in_group) as *mut __m512i;
                    let updated = _mm512_xor_si512(
                        _mm512_loadu_si512(partial_ptr),
                        _mm512_xor_si512(from_lo, from_hi),
                    );
                    _mm512_storeu_si512(partial_ptr, updated);
                }
            }
        }
    }
}

/// F₂-linear nibble-LUT drain. Production `mask_tables` are XOR-doubling
/// subset sums, so `T[byte] = T[n0] ⊕ T[n1 << 4]`. Each 16-entry nibble table
/// is 16 u64 halves = two ZMM, looked up with `vpermi2q` (AVX-512F, 3c/1c on
/// Sapphire Rapids) instead of `vpgatherqq` plus a SIMD→GPR index spill.
///
/// Bit-identical to the gather kernel on linear tables; GF(2) XOR reordering
/// of independent nibble contributions is the only algebraic change.
#[inline]
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
#[target_feature(enable = "avx512f,avx512bw,vpclmulqdq")]
pub(crate) unsafe fn accumulate_c_banks_x86_avx512_nibble(
    c_block: &[u8; 16 * ELL],
    n_b_med: usize,
    mask_tables: &[F128],
    partial_c: &mut [[F128; ELL]; 8],
) {
    debug_assert_eq!(mask_tables.len(), 512);
    let lut = super::CBankNibbleLut::new(mask_tables);
    unsafe { accumulate_c_banks_x86_avx512_nibble_prebuilt(c_block, n_b_med, &lut, partial_c) }
}

/// DirectC nibble drain with the eq-dependent table already materialized.
///
/// Mask build is a 64-lane AVX-512BW `vptestmb` pack: one ZMM load of the
/// full C row, eight byte-tests, eight `maskz_set1` ORs. The 16-bit subset
/// index is split into two u8 planes (`b_med 0..7` / `8..15`) because a
/// 16-bit weight does not fit in a byte; n0|n1 come from the lo plane and
/// n2|n3 from the hi plane — the same four nibbles the 16-lane `vptestmd`
/// path accumulated in i32. LUT half is unchanged (`vpermi2q` on the
/// existing SoA tables).
#[inline]
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
#[target_feature(enable = "avx512f,avx512bw,vpclmulqdq")]
pub(crate) unsafe fn accumulate_c_banks_x86_avx512_nibble_prebuilt(
    c_block: &[u8; 16 * ELL],
    n_b_med: usize,
    lut: &super::CBankNibbleLut,
    partial_c: &mut [[F128; ELL]; 8],
) {
    use core::arch::x86_64::*;
    debug_assert_eq!(ELL, 64);
    debug_assert!(n_b_med <= 1 << N_MEDIUM);

    #[inline(always)]
    unsafe fn lookup8(idx: __m512i, table: *const u64) -> __m512i {
        unsafe {
            let a = _mm512_load_si512(table as *const __m512i);
            let b = _mm512_load_si512(table.add(8) as *const __m512i);
            _mm512_permutex2var_epi64(a, idx, b)
        }
    }

    #[inline(always)]
    unsafe fn interleave_aos(los: __m512i, his: __m512i) -> (__m512i, __m512i) {
        unsafe {
            let idx0 = _mm512_set_epi64(11, 3, 10, 2, 9, 1, 8, 0);
            let idx1 = _mm512_set_epi64(15, 7, 14, 6, 13, 5, 12, 4);
            (
                _mm512_permutex2var_epi64(los, idx0, his),
                _mm512_permutex2var_epi64(los, idx1, his),
            )
        }
    }

    #[inline(always)]
    unsafe fn extract16(v: __m512i, chunk: usize) -> __m128i {
        unsafe {
            match chunk {
                0 => _mm512_castsi512_si128(v),
                1 => _mm512_extracti32x4_epi32::<1>(v),
                2 => _mm512_extracti32x4_epi32::<2>(v),
                _ => _mm512_extracti32x4_epi32::<3>(v),
            }
        }
    }

    // SAFETY: each `b_med` load reads a full 64-byte C row (`ELL == 64`);
    // `n_b_med <= 16` bounds the row. Bank bits are the eight C-byte bits.
    // Lo/hi planes reconstruct the same 16-bit subset index the i32 path
    // built, so nibble indices stay in 0..=15 and every vpermi2q stays
    // inside a 16-entry table half. Partial stores cover eight F128 lanes.
    unsafe {
        let nibble_mask = _mm512_set1_epi32(0xf);
        let bank_bits = [
            _mm512_set1_epi8(1),
            _mm512_set1_epi8(2),
            _mm512_set1_epi8(4),
            _mm512_set1_epi8(8),
            _mm512_set1_epi8(16),
            _mm512_set1_epi8(32),
            _mm512_set1_epi8(64),
            _mm512_set1_epi8(-128), // 1 << 7 as i8
        ];

        let mut lo_masks = [_mm512_setzero_si512(); 8];
        let mut hi_masks = [_mm512_setzero_si512(); 8];

        for b_med in 0..n_b_med {
            let row = _mm512_loadu_si512(c_block.as_ptr().add(b_med * ELL) as *const __m512i);
            let hi_plane = b_med >= 8;
            let shift = if hi_plane { b_med - 8 } else { b_med };
            let weight = (1u8 << shift) as i8;
            let dest = if hi_plane {
                &mut hi_masks
            } else {
                &mut lo_masks
            };
            for (mask, bank_bit) in dest.iter_mut().zip(bank_bits) {
                let selected = _mm512_test_epi8_mask(row, bank_bit);
                *mask = _mm512_or_si512(*mask, _mm512_maskz_set1_epi8(selected, weight));
            }
        }

        for (bank, (lo, hi)) in partial_c
            .iter_mut()
            .zip(lo_masks.iter().zip(hi_masks.iter()))
        {
            for chunk in 0..4 {
                let lane_base = chunk * 16;
                let lo16 = _mm512_cvtepu8_epi32(extract16(*lo, chunk));
                let hi16 = _mm512_cvtepu8_epi32(extract16(*hi, chunk));
                let n0 = _mm512_and_si512(lo16, nibble_mask);
                let n1 = _mm512_and_si512(_mm512_srli_epi32::<4>(lo16), nibble_mask);
                let n2 = _mm512_and_si512(hi16, nibble_mask);
                let n3 = _mm512_and_si512(_mm512_srli_epi32::<4>(hi16), nibble_mask);

                for group in 0..2 {
                    let n0_8 = if group == 0 {
                        _mm512_cvtepu32_epi64(_mm512_castsi512_si256(n0))
                    } else {
                        _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64::<1>(n0))
                    };
                    let n1_8 = if group == 0 {
                        _mm512_cvtepu32_epi64(_mm512_castsi512_si256(n1))
                    } else {
                        _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64::<1>(n1))
                    };
                    let n2_8 = if group == 0 {
                        _mm512_cvtepu32_epi64(_mm512_castsi512_si256(n2))
                    } else {
                        _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64::<1>(n2))
                    };
                    let n3_8 = if group == 0 {
                        _mm512_cvtepu32_epi64(_mm512_castsi512_si256(n3))
                    } else {
                        _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64::<1>(n3))
                    };

                    let los = _mm512_xor_si512(
                        _mm512_xor_si512(
                            lookup8(n0_8, lut.lo_n0_lo.as_ptr()),
                            lookup8(n1_8, lut.lo_n1_lo.as_ptr()),
                        ),
                        _mm512_xor_si512(
                            lookup8(n2_8, lut.hi_n0_lo.as_ptr()),
                            lookup8(n3_8, lut.hi_n1_lo.as_ptr()),
                        ),
                    );
                    let his = _mm512_xor_si512(
                        _mm512_xor_si512(
                            lookup8(n0_8, lut.lo_n0_hi.as_ptr()),
                            lookup8(n1_8, lut.lo_n1_hi.as_ptr()),
                        ),
                        _mm512_xor_si512(
                            lookup8(n2_8, lut.hi_n0_hi.as_ptr()),
                            lookup8(n3_8, lut.hi_n1_hi.as_ptr()),
                        ),
                    );
                    let (aos0, aos1) = interleave_aos(los, his);
                    let partial_ptr = bank.as_mut_ptr().add(lane_base + group * 8) as *mut __m512i;
                    _mm512_storeu_si512(
                        partial_ptr,
                        _mm512_xor_si512(_mm512_loadu_si512(partial_ptr), aos0),
                    );
                    _mm512_storeu_si512(
                        partial_ptr.add(1),
                        _mm512_xor_si512(_mm512_loadu_si512(partial_ptr.add(1)), aos1),
                    );
                }
            }
        }
    }
}

/// Legacy two-bank capture kernel retained for compatibility with the old
/// entry point. DirectC does not call it.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
#[target_feature(enable = "avx512f,vpclmulqdq")]
pub(crate) unsafe fn accumulate_convert_with_s_hat_v_x86_avx512(
    chunk_ab_bytes: &[[u8; ELL]; 1 << N_MEDIUM],
    chunk_c_bytes: &[[u8; ELL]; 1 << N_MEDIUM],
    n_b_med: usize,
    convert: &[F128],
    eq_lo_val: F128,
    partial_ab: &mut [F128; ELL],
    partial_c_0: &mut [F128; ELL],
    partial_c_1: &mut [F128; ELL],
) {
    for lane in 0..ELL {
        let mut converted_ab = F128::ZERO;
        let mut converted_c_0 = F128::ZERO;
        let mut converted_c_1 = F128::ZERO;
        for b_med in 0..n_b_med {
            let table_base = b_med * 256;
            let c = usize::from(chunk_c_bytes[b_med][lane]);
            converted_ab += convert[table_base + usize::from(chunk_ab_bytes[b_med][lane])];
            converted_c_0 += convert[table_base + (c & 0x55)];
            converted_c_1 += convert[table_base + (c & 0xaa)];
        }
        partial_ab[lane] += converted_ab * eq_lo_val;
        partial_c_0[lane] += converted_c_0 * eq_lo_val;
        partial_c_1[lane] += converted_c_1 * eq_lo_val;
    }
}

#[cfg(all(
    test,
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
mod tests {
    use super::super::accumulate_c_banks_scalar;
    use super::{
        F128, accumulate_c_banks_x86_avx512, accumulate_c_banks_x86_avx512_nibble,
        accumulate_convert_ab_x86_avx512, accumulate_convert_ab_x86_avx512_nibble,
    };
    #[cfg(target_feature = "gfni")]
    use super::{
        accumulate_convert_ab_nomul_x86_gfni, accumulate_convert_ab_nomul_x86_gfni_dynamic,
        accumulate_convert_ab_nomul_x86_gfni_fixed,
    };

    #[test]
    fn accumulate_c_banks_avx512_matches_scalar() {
        let mut c_block = [0u8; 16 * 64];
        for (i, byte) in c_block.iter_mut().enumerate() {
            *byte = (i as u8).wrapping_mul(0x9d).rotate_left((i & 7) as u32) ^ (i >> 3) as u8;
        }

        let mut mask_tables = [F128::ZERO; 512];
        for (i, value) in mask_tables.iter_mut().enumerate() {
            let x = i as u64;
            *value = F128::new(
                x.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0x0123_4567_89ab_cdef,
                x.rotate_left(29).wrapping_mul(0xd6e8_feb8_6659_fd93) ^ 0xfedc_ba98_7654_3210,
            );
        }

        for n_b_med in 0..=16 {
            let mut scalar = [[F128::ZERO; 64]; 8];
            for (bank_index, bank) in scalar.iter_mut().enumerate() {
                for (lane, value) in bank.iter_mut().enumerate() {
                    *value = F128::new(
                        ((bank_index * 64 + lane) as u64).wrapping_mul(0xa24b_aed4_963e_e407),
                        ((lane * 8 + bank_index) as u64).wrapping_mul(0x9fb2_1c65_1e98_df25),
                    );
                }
            }
            let mut simd = scalar;

            accumulate_c_banks_scalar(&c_block, n_b_med, &mask_tables, &mut scalar);
            // SAFETY: this test is compiled only when the kernel's target
            // features are statically enabled; all buffers have exact sizes.
            unsafe {
                accumulate_c_banks_x86_avx512(&c_block, n_b_med, &mask_tables, &mut simd);
            }
            assert_eq!(simd, scalar, "n_b_med={n_b_med}");
        }
    }

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

    fn nontrivial_partials() -> [[F128; 64]; 8] {
        let mut partials = [[F128::ZERO; 64]; 8];
        for (bank_index, bank) in partials.iter_mut().enumerate() {
            for (lane, value) in bank.iter_mut().enumerate() {
                *value = F128::new(
                    ((bank_index * 64 + lane) as u64).wrapping_mul(0xa24b_aed4_963e_e407),
                    ((lane * 8 + bank_index) as u64).wrapping_mul(0x9fb2_1c65_1e98_df25),
                );
            }
        }
        partials
    }

    fn ranked_shape_c_block() -> [u8; 16 * 64] {
        let mut c_block = [0u8; 16 * 64];
        for (i, byte) in c_block.iter_mut().enumerate() {
            *byte = (i as u8).wrapping_mul(0x9d).rotate_left((i & 7) as u32) ^ (i >> 3) as u8;
        }
        c_block
    }

    #[test]
    fn accumulate_c_banks_nibble_matches_scalar_linear_tables() {
        let c_block = ranked_shape_c_block();
        let mask_tables = xor_doubling_tables();

        for n_b_med in 0..=16 {
            let mut scalar = nontrivial_partials();
            let mut nibble = scalar;
            let mut gather = scalar;

            accumulate_c_banks_scalar(&c_block, n_b_med, &mask_tables, &mut scalar);
            unsafe {
                accumulate_c_banks_x86_avx512_nibble(&c_block, n_b_med, &mask_tables, &mut nibble);
                accumulate_c_banks_x86_avx512(&c_block, n_b_med, &mask_tables, &mut gather);
            }
            assert_eq!(nibble, scalar, "nibble vs scalar n_b_med={n_b_med}");
            assert_eq!(gather, scalar, "gather vs scalar n_b_med={n_b_med}");
        }
    }

    #[test]
    fn accumulate_convert_ab_nibble_matches_gpr_on_convert_table() {
        let convert = super::super::super::convert_table();
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
            let mut gpr = seed;
            let mut nibble = seed;
            unsafe {
                accumulate_convert_ab_x86_avx512(&chunk_ab_bytes, n_b_med, convert, eq, &mut gpr);
                accumulate_convert_ab_x86_avx512_nibble(
                    &chunk_ab_bytes,
                    n_b_med,
                    convert,
                    eq,
                    &mut nibble,
                );
            }
            assert_eq!(nibble, gpr, "n_b_med={n_b_med}");
        }
    }

    #[cfg(target_feature = "gfni")]
    #[test]
    fn accumulate_convert_ab_gfni_fixed_rows_match_dynamic() {
        let mut chunk_ab_bytes = [[0u8; 64]; 16];
        for (row, bytes) in chunk_ab_bytes.iter_mut().enumerate() {
            for (lane, byte) in bytes.iter_mut().enumerate() {
                *byte = (row as u8).wrapping_mul(0x9d)
                    ^ (lane as u8).wrapping_mul(0x53)
                    ^ ((row * 17 + lane * 29) >> 3) as u8;
            }
        }
        let mut mats = [0u64; 256];
        for (i, matrix) in mats.iter_mut().enumerate() {
            *matrix = (i as u64)
                .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                .rotate_left((i & 63) as u32)
                ^ 0xd6e8_feb8_6659_fd93;
        }
        let mut seed = [0u8; 16 * 64];
        for (i, byte) in seed.iter_mut().enumerate() {
            *byte = (i as u8).wrapping_mul(0xa7) ^ (i >> 2) as u8;
        }

        for n_b_med in 0..=16 {
            let mut dynamic = seed;
            let mut dispatched = seed;
            // SAFETY: this test is compiled only with the kernel's complete
            // target feature set and every buffer has its exact fixed size.
            unsafe {
                accumulate_convert_ab_nomul_x86_gfni_dynamic(
                    &chunk_ab_bytes,
                    n_b_med,
                    &mats,
                    &mut dynamic,
                );
                accumulate_convert_ab_nomul_x86_gfni(
                    &chunk_ab_bytes,
                    n_b_med,
                    &mats,
                    &mut dispatched,
                );
            }
            assert_eq!(dispatched, dynamic, "dispatch n_b_med={n_b_med}");
        }

        for n_b_med in [15usize, 16] {
            let mut dynamic = seed;
            let mut fixed = seed;
            unsafe {
                accumulate_convert_ab_nomul_x86_gfni_dynamic(
                    &chunk_ab_bytes,
                    n_b_med,
                    &mats,
                    &mut dynamic,
                );
                if n_b_med == 15 {
                    accumulate_convert_ab_nomul_x86_gfni_fixed::<15, false>(
                        &chunk_ab_bytes,
                        &mats,
                        &mut fixed,
                    );
                } else {
                    accumulate_convert_ab_nomul_x86_gfni_fixed::<16, false>(
                        &chunk_ab_bytes,
                        &mats,
                        &mut fixed,
                    );
                }
            }
            assert_eq!(fixed, dynamic, "fixed n_b_med={n_b_med}");
        }
    }
}

/// /// GFNI twin of [`accumulate_convert_ab_nomul_x86_avx512`]: the 256-entry
/// byte sub-tables are F2-linear (XOR-composed from eight basis entries with
/// `T[0] = 0`), so each sub-table IS sixteen 8×8 bit matrices — one per
/// output byte of the F128 — and one `VGF2P8AFFINEQB` evaluates a matrix on
/// all 64 lanes' bytes at once with no table loads at all. The bank lives in
/// byte-plane-major form (`bank_planes[k*64 + lane]` = byte `k` of lane's
/// F128 accumulator); the caller transposes once per band. Encoding
/// (hardware-verified): `out.bit[i] = parity(mat.byte[7-i] & in)`. The
/// accumulated planes are the table path's sums with the XOR terms
/// reassociated — bit-identical.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
#[cold]
#[inline(never)]
#[target_feature(enable = "avx512f,gfni")]
unsafe fn accumulate_convert_ab_nomul_x86_gfni_dynamic(
    chunk_ab_bytes: &[[u8; ELL]; 1 << N_MEDIUM],
    n_b_med: usize,
    mats: &[u64; 256],
    bank_planes: &mut [u8; 16 * ELL],
) {
    // SAFETY: forwarded unchanged to the shared target-feature body; this
    // arm preserves and XOR-accumulates the bank's existing planes.
    unsafe {
        accumulate_convert_ab_nomul_x86_gfni_impl::<false>(
            chunk_ab_bytes,
            n_b_med,
            mats,
            bank_planes,
        );
    }
}

/// First-visit twin of [`accumulate_convert_ab_nomul_x86_gfni`]. The caller
/// proves that every plane in `bank_planes` is dead, so the sixteen output
/// chains start from register zero and overwrite all 1 KiB without loading it.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
#[target_feature(enable = "avx512f,gfni")]
pub(crate) unsafe fn write_convert_ab_nomul_x86_gfni(
    chunk_ab_bytes: &[[u8; ELL]; 1 << N_MEDIUM],
    n_b_med: usize,
    mats: &[u64; 256],
    bank_planes: &mut [u8; 16 * ELL],
) {
    // SAFETY: each callee has this entry point's fixed-array and feature
    // contract. Ranked 15/16 use the same const-N body as accumulate, with
    // FIRST_WRITE so the first visit does not load dead planes. Counts
    // outside that pair keep the generic first-write impl.
    unsafe {
        if !r1_ab_write_fixed_enabled() {
            accumulate_convert_ab_nomul_x86_gfni_impl::<true>(
                chunk_ab_bytes,
                n_b_med,
                mats,
                bank_planes,
            );
            return;
        }
        match n_b_med {
            15 => accumulate_convert_ab_nomul_x86_gfni_fixed::<15, true>(
                chunk_ab_bytes,
                mats,
                bank_planes,
            ),
            16 => accumulate_convert_ab_nomul_x86_gfni_fixed::<16, true>(
                chunk_ab_bytes,
                mats,
                bank_planes,
            ),
            _ => accumulate_convert_ab_nomul_x86_gfni_impl::<true>(
                chunk_ab_bytes,
                n_b_med,
                mats,
                bank_planes,
            ),
        }
    }
}

/// First-write ranked residual body: rows 0 and 1 are supplied by the
/// identity-C contribution, so this evaluates absolute medium rows 2..N.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
#[inline]
#[target_feature(enable = "avx512f,gfni")]
pub(crate) unsafe fn write_convert_ab_nomul_x86_gfni_range2(
    chunk_ab_bytes: &[[u8; ELL]; 1 << N_MEDIUM],
    n_b_med: usize,
    mats: &[u64; 256],
    bank_planes: &mut [u8; 16 * ELL],
) {
    unsafe {
        match n_b_med {
            15 => accumulate_convert_ab_nomul_x86_gfni_fixed_range2::<15, true>(
                chunk_ab_bytes,
                mats,
                bank_planes,
            ),
            16 => accumulate_convert_ab_nomul_x86_gfni_fixed_range2::<16, true>(
                chunk_ab_bytes,
                mats,
                bank_planes,
            ),
            _ => core::hint::unreachable_unchecked(),
        }
    }
}

/// Accumulating twin of [`write_convert_ab_nomul_x86_gfni_range2`].
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
#[inline]
#[target_feature(enable = "avx512f,gfni")]
pub(crate) unsafe fn accumulate_convert_ab_nomul_x86_gfni_range2(
    chunk_ab_bytes: &[[u8; ELL]; 1 << N_MEDIUM],
    n_b_med: usize,
    mats: &[u64; 256],
    bank_planes: &mut [u8; 16 * ELL],
) {
    unsafe {
        match n_b_med {
            15 => accumulate_convert_ab_nomul_x86_gfni_fixed_range2::<15, false>(
                chunk_ab_bytes,
                mats,
                bank_planes,
            ),
            16 => accumulate_convert_ab_nomul_x86_gfni_fixed_range2::<16, false>(
                chunk_ab_bytes,
                mats,
                bank_planes,
            ),
            _ => core::hint::unreachable_unchecked(),
        }
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
#[inline(never)]
#[target_feature(enable = "avx512f,gfni")]
unsafe fn accumulate_convert_ab_nomul_x86_gfni_fixed_range2<
    const N: usize,
    const FIRST_WRITE: bool,
>(
    chunk_ab_bytes: &[[u8; ELL]; 1 << N_MEDIUM],
    mats: &[u64; 256],
    bank_planes: &mut [u8; 16 * ELL],
) {
    use core::arch::x86_64::*;
    debug_assert!(N == 15 || N == 16);
    unsafe {
        let mut rows = [_mm512_setzero_si512(); 1 << N_MEDIUM];
        for bm in 2..N {
            rows[bm] = _mm512_loadu_si512(chunk_ab_bytes[bm].as_ptr() as *const __m512i);
        }
        for k in 0..16 {
            let plane_ptr = bank_planes.as_mut_ptr().add(k * ELL) as *mut __m512i;
            let mut acc = if FIRST_WRITE {
                _mm512_setzero_si512()
            } else {
                _mm512_loadu_si512(plane_ptr as *const __m512i)
            };
            let mut bm = 2;
            while bm + 1 < N {
                let g0 = _mm512_gf2p8affine_epi64_epi8::<0>(
                    rows[bm],
                    _mm512_set1_epi64(mats[bm * 16 + k] as i64),
                );
                let g1 = _mm512_gf2p8affine_epi64_epi8::<0>(
                    rows[bm + 1],
                    _mm512_set1_epi64(mats[(bm + 1) * 16 + k] as i64),
                );
                acc = _mm512_ternarylogic_epi64::<0x96>(acc, g0, g1);
                bm += 2;
            }
            if bm < N {
                let g = _mm512_gf2p8affine_epi64_epi8::<0>(
                    rows[bm],
                    _mm512_set1_epi64(mats[bm * 16 + k] as i64),
                );
                acc = _mm512_xor_si512(acc, g);
            }
            _mm512_storeu_si512(plane_ptr, acc);
        }
    }
}

#[inline]
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
#[target_feature(enable = "avx512f,gfni")]
unsafe fn accumulate_convert_ab_nomul_x86_gfni_impl<const FIRST_WRITE: bool>(
    chunk_ab_bytes: &[[u8; ELL]; 1 << N_MEDIUM],
    n_b_med: usize,
    mats: &[u64; 256],
    bank_planes: &mut [u8; 16 * ELL],
) {
    use core::arch::x86_64::*;
    debug_assert!(n_b_med <= 1 << N_MEDIUM);
    // SAFETY: the fixed-size input/plane arrays contain every 64-byte load
    // and store below; `mats` is exactly the 16×16 qword matrix block for
    // this table slice. The cfg gate supplies the required target features.
    unsafe {
        let mut rows = [_mm512_setzero_si512(); 1 << N_MEDIUM];
        for (bm, row) in rows.iter_mut().enumerate().take(n_b_med) {
            *row = _mm512_loadu_si512(chunk_ab_bytes[bm].as_ptr() as *const __m512i);
        }
        for k in 0..16 {
            let plane_ptr = bank_planes.as_mut_ptr().add(k * ELL) as *mut __m512i;
            let mut acc = if FIRST_WRITE {
                _mm512_setzero_si512()
            } else {
                _mm512_loadu_si512(plane_ptr as *const __m512i)
            };
            let mut bm = 0;
            // Two GFNI products fold into the accumulator per VPTERNLOGQ
            // (imm 0x96 = a ^ b ^ c); sixteen independent plane chains keep
            // the accumulation latency off the critical path.
            while bm + 1 < n_b_med {
                let g0 = _mm512_gf2p8affine_epi64_epi8::<0>(
                    rows[bm],
                    _mm512_set1_epi64(mats[bm * 16 + k] as i64),
                );
                let g1 = _mm512_gf2p8affine_epi64_epi8::<0>(
                    rows[bm + 1],
                    _mm512_set1_epi64(mats[(bm + 1) * 16 + k] as i64),
                );
                acc = _mm512_ternarylogic_epi64::<0x96>(acc, g0, g1);
                bm += 2;
            }
            if bm < n_b_med {
                let g = _mm512_gf2p8affine_epi64_epi8::<0>(
                    rows[bm],
                    _mm512_set1_epi64(mats[bm * 16 + k] as i64),
                );
                acc = _mm512_xor_si512(acc, g);
            }
            _mm512_storeu_si512(plane_ptr, acc);
        }
    }
}

/// Ranked first-write 15/16 uses the same const-N GFNI body as accumulate.
/// `FLOCK_NO_R1_AB_WRITE_FIXED=1` restores the generic first-write impl.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
#[inline]
fn r1_ab_write_fixed_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var_os("FLOCK_NO_R1_AB_WRITE_FIXED").is_none()
    });
    *ON
}

/// Ranked fixed-row body. BLAKE3's two padding classes contain exactly
/// fifteen and sixteen live medium rows; making that count a monomorphized
/// constant removes LLVM's per-plane bounds ladder from the GFNI battery.
/// `FIRST_WRITE` starts each plane from a zero register instead of loading
/// dead bytes — the same split range2 already uses.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
#[inline(never)]
#[target_feature(enable = "avx512f,gfni")]
unsafe fn accumulate_convert_ab_nomul_x86_gfni_fixed<const N: usize, const FIRST_WRITE: bool>(
    chunk_ab_bytes: &[[u8; ELL]; 1 << N_MEDIUM],
    mats: &[u64; 256],
    bank_planes: &mut [u8; 16 * ELL],
) {
    use core::arch::x86_64::*;
    debug_assert!(N == 15 || N == 16);
    // SAFETY: the fixed arrays cover every row/plane load and store. `N` is
    // one of the two ranked live-row counts and the cfg gate supplies GFNI.
    unsafe {
        let mut rows = [_mm512_setzero_si512(); 1 << N_MEDIUM];
        for bm in 0..N {
            rows[bm] = _mm512_loadu_si512(chunk_ab_bytes[bm].as_ptr() as *const __m512i);
        }
        for k in 0..16 {
            let plane_ptr = bank_planes.as_mut_ptr().add(k * ELL) as *mut __m512i;
            let mut acc = if FIRST_WRITE {
                _mm512_setzero_si512()
            } else {
                _mm512_loadu_si512(plane_ptr as *const __m512i)
            };
            let mut bm = 0;
            while bm + 1 < N {
                let g0 = _mm512_gf2p8affine_epi64_epi8::<0>(
                    rows[bm],
                    _mm512_set1_epi64(mats[bm * 16 + k] as i64),
                );
                let g1 = _mm512_gf2p8affine_epi64_epi8::<0>(
                    rows[bm + 1],
                    _mm512_set1_epi64(mats[(bm + 1) * 16 + k] as i64),
                );
                acc = _mm512_ternarylogic_epi64::<0x96>(acc, g0, g1);
                bm += 2;
            }
            if bm < N {
                let g = _mm512_gf2p8affine_epi64_epi8::<0>(
                    rows[bm],
                    _mm512_set1_epi64(mats[bm * 16 + k] as i64),
                );
                acc = _mm512_xor_si512(acc, g);
            }
            _mm512_storeu_si512(plane_ptr, acc);
        }
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
#[inline]
#[target_feature(enable = "avx512f,gfni")]
pub(crate) unsafe fn accumulate_convert_ab_nomul_x86_gfni(
    chunk_ab_bytes: &[[u8; ELL]; 1 << N_MEDIUM],
    n_b_med: usize,
    mats: &[u64; 256],
    bank_planes: &mut [u8; 16 * ELL],
) {
    // SAFETY: each callee has this entry point's fixed-array and feature
    // contract. Counts outside the ranked pair retain the incumbent body.
    unsafe {
        match n_b_med {
            15 => accumulate_convert_ab_nomul_x86_gfni_fixed::<15, false>(
                chunk_ab_bytes,
                mats,
                bank_planes,
            ),
            16 => accumulate_convert_ab_nomul_x86_gfni_fixed::<16, false>(
                chunk_ab_bytes,
                mats,
                bank_planes,
            ),
            _ => accumulate_convert_ab_nomul_x86_gfni_dynamic(
                chunk_ab_bytes,
                n_b_med,
                mats,
                bank_planes,
            ),
        }
    }
}

/// Same ranked GFNI row/plane chains as the staged entry points, loading
/// only the producer's initialized `FIRST..N` span. The caller proves the
/// span length and that FIRST_WRITE overwrites every plane before a read.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
#[inline(never)]
#[target_feature(enable = "avx512f,gfni")]
pub(super) unsafe fn convert_ab_nomul_x86_gfni_direct<
    const FIRST: usize,
    const N: usize,
    const FIRST_WRITE: bool,
>(
    live_rows: &[u8],
    mats: &[u64; 256],
    bank_planes: &mut [u8; 16 * ELL],
    prefetch: &super::AbDirectPrefetch,
) {
    use core::arch::x86_64::*;
    debug_assert!((FIRST == 2 && N == 16) || (FIRST == 0 && N == 15));
    debug_assert_eq!(live_rows.len(), (N - FIRST) * ELL);
    // SAFETY: the wrapper checks all (N - FIRST) live rows. Input loads use
    // relative offsets, while matrix/row indices retain their absolute bm.
    // The fixed output array covers all sixteen 64-byte plane stores. The
    // prefetch pointer is only used by nonfaulting hints via wrapping_add.
    unsafe {
        let pf_one = |bm: usize| {
            _mm_prefetch(
                prefetch
                    .next_window
                    .wrapping_add((bm - prefetch.first) * ELL)
                    .cast::<i8>(),
                _MM_HINT_T0,
            );
        };
        if !prefetch.spread {
            for bm in prefetch.first..prefetch.end {
                pf_one(bm);
            }
        }
        let mut rows = [_mm512_setzero_si512(); 1 << N_MEDIUM];
        for bm in FIRST..N {
            // Preserve the old copy-loop hint/load interleave, now beside
            // the first and only demand load of the original input line.
            if prefetch.spread && bm >= prefetch.first && bm < prefetch.end {
                pf_one(bm);
            }
            rows[bm] =
                _mm512_loadu_si512(live_rows.as_ptr().add((bm - FIRST) * ELL) as *const __m512i);
        }
        if prefetch.spread {
            for bm in N.max(prefetch.first)..prefetch.end {
                pf_one(bm);
            }
        }
        for k in 0..16 {
            let plane_ptr = bank_planes.as_mut_ptr().add(k * ELL) as *mut __m512i;
            let mut acc = if FIRST_WRITE {
                _mm512_setzero_si512()
            } else {
                _mm512_loadu_si512(plane_ptr as *const __m512i)
            };
            let mut bm = FIRST;
            while bm + 1 < N {
                let g0 = _mm512_gf2p8affine_epi64_epi8::<0>(
                    rows[bm],
                    _mm512_set1_epi64(mats[bm * 16 + k] as i64),
                );
                let g1 = _mm512_gf2p8affine_epi64_epi8::<0>(
                    rows[bm + 1],
                    _mm512_set1_epi64(mats[(bm + 1) * 16 + k] as i64),
                );
                acc = _mm512_ternarylogic_epi64::<0x96>(acc, g0, g1);
                bm += 2;
            }
            if bm < N {
                let g = _mm512_gf2p8affine_epi64_epi8::<0>(
                    rows[bm],
                    _mm512_set1_epi64(mats[bm * 16 + k] as i64),
                );
                acc = _mm512_xor_si512(acc, g);
            }
            _mm512_storeu_si512(plane_ptr, acc);
        }
    }
}

/// 8x64 byte transpose: eight 64-byte rows in, eight registers out with
/// `out[k].byte[8L + b] = rows[REV ? 7 - b : b][8k + L]` — i.e. output
/// register `k` holds, in qword `L`, the eight rows' byte `8k + L`.
///
/// Three `vpermt2q` stages swap register index bit `d` with qword index bit
/// `d` (a plain 8x8 qword transpose), then one `vpermb` finishes the
/// sub-qword byte transpose. `REV` reverses the row order, which is what
/// cancels the `A.byte[7-i]` indexing of `VGF2P8AFFINEQB` when the result is
/// fed back in as a bit-transpose matrix.
#[inline]
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi"
))]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
unsafe fn byte_transpose_8x64<const REV: bool>(
    rows: [core::arch::x86_64::__m512i; 8],
) -> [core::arch::x86_64::__m512i; 8] {
    use core::arch::x86_64::*;
    #[rustfmt::skip]
    const IDX: [i8; 64] = [
         0,  8, 16, 24, 32, 40, 48, 56,
         1,  9, 17, 25, 33, 41, 49, 57,
         2, 10, 18, 26, 34, 42, 50, 58,
         3, 11, 19, 27, 35, 43, 51, 59,
         4, 12, 20, 28, 36, 44, 52, 60,
         5, 13, 21, 29, 37, 45, 53, 61,
         6, 14, 22, 30, 38, 46, 54, 62,
         7, 15, 23, 31, 39, 47, 55, 63,
    ];
    #[rustfmt::skip]
    const IDX_REV: [i8; 64] = [
        56, 48, 40, 32, 24, 16,  8,  0,
        57, 49, 41, 33, 25, 17,  9,  1,
        58, 50, 42, 34, 26, 18, 10,  2,
        59, 51, 43, 35, 27, 19, 11,  3,
        60, 52, 44, 36, 28, 20, 12,  4,
        61, 53, 45, 37, 29, 21, 13,  5,
        62, 54, 46, 38, 30, 22, 14,  6,
        63, 55, 47, 39, 31, 23, 15,  7,
    ];
    const T4A: [i64; 8] = [0, 1, 2, 3, 8, 9, 10, 11];
    const T4B: [i64; 8] = [4, 5, 6, 7, 12, 13, 14, 15];
    const T2A: [i64; 8] = [0, 1, 8, 9, 4, 5, 12, 13];
    const T2B: [i64; 8] = [2, 3, 10, 11, 6, 7, 14, 15];
    const T1A: [i64; 8] = [0, 8, 2, 10, 4, 12, 6, 14];
    const T1B: [i64; 8] = [1, 9, 3, 11, 5, 13, 7, 15];

    // SAFETY: only register-to-register shuffles plus loads of the fixed
    // 64-byte index constants; the cfg gate supplies the target features.
    unsafe {
        let mut cur = rows;
        for (a, b, d) in [(T4A, T4B, 4usize), (T2A, T2B, 2), (T1A, T1B, 1)] {
            let ia = _mm512_loadu_si512(a.as_ptr() as *const __m512i);
            let ib = _mm512_loadu_si512(b.as_ptr() as *const __m512i);
            let mut next = [_mm512_setzero_si512(); 8];
            for r in 0..8usize {
                if r & d == 0 {
                    let x = cur[r];
                    let y = cur[r | d];
                    next[r] = _mm512_permutex2var_epi64(x, ia, y);
                    next[r | d] = _mm512_permutex2var_epi64(x, ib, y);
                }
            }
            cur = next;
        }
        let table = if REV { IDX_REV.as_ptr() } else { IDX.as_ptr() };
        let idx = _mm512_loadu_si512(table as *const __m512i);
        let mut out = [_mm512_setzero_si512(); 8];
        for k in 0..8usize {
            out[k] = _mm512_permutexvar_epi8(idx, cur[k]);
        }
        out
    }
}

/// Ascending bulk fetch of one four-window C group into the worker's staging
/// buffer.
///
/// The drain consumes rows in `(q, half, window, j)` order, which walks
/// `c_packed` in a permuted stride-256 pattern and leaves its DRAM misses
/// uncovered at the end of the group. Staging instead issues all 64 line
/// loads up front in strict address order — one linear sweep, since the
/// DirectFold4 group's four windows are contiguous — and the caller runs it
/// *before* the group's AB completion, so the misses retire underneath work
/// that does not depend on them.
#[inline]
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn stage_c_group_x86_avx512(src: &[u8], dst: &mut [u8; 4 * 16 * ELL]) {
    use core::arch::x86_64::*;
    debug_assert!(src.len() >= 4 * 16 * ELL);
    // SAFETY: both sides cover exactly the group's 4 KiB; `dst` is 64-byte
    // aligned by construction in the worker state.
    unsafe {
        let s = src.as_ptr();
        let d = dst.as_mut_ptr();
        for line in 0..(4 * 16 * ELL / 64) {
            _mm512_store_si512(
                d.add(line * 64) as *mut __m512i,
                _mm512_loadu_si512(s.add(line * 64) as *const __m512i),
            );
        }
    }
}

/// Fused GFNI DirectFold4 C drain — the last gather-style kernel of round 1.
///
/// The incumbent path per four-window group is: 64 `bit_transpose_64bytes`
/// calls, then per `q` a `vptestmd`/`vpord` mask build (16 rows x 8 banks x 4
/// dword groups) and 512 `vpermi2q` nibble probes of a 1 KiB per-group LUT.
/// Every stage of that is F2-linear in the C bits, so the whole composition
/// collapses into two bit transposes and one bit-matrix product:
///
/// * **Mask build is a transpose.** Bank `k`'s mask bit `b` is bit `k` of
///   drain row `b`, and drain row `b` is itself
///   `bit_transpose_64bytes(c_in_b)`, i.e.
///   `row_b[lane].bit[k] = c_in_b[8k + lane/8].bit[lane%8]`. Composing the
///   two gives `mask[k][lane].bit[b] = c_in_b[8k + lane/8].bit[lane%8]` — so
///   transposing the eight raw `c_packed` rows once (8x8 qword transpose,
///   then a reversed 8x8 byte transpose, then a per-qword 8x8 bit transpose)
///   lands the sixteen mask vectors directly, one ZMM per bank, lanes already
///   in natural order. The per-qword bit transpose is itself one
///   `VGF2P8AFFINEQB`: feeding the DATA as the matrix operand and
///   `0x8040201008040201` as the vector gives `out.byte[t].bit[b] =
///   A.byte[7-b].bit[t]`, which is why the byte transpose reverses rows.
/// * **Table lookup is a bit-matrix product.** The synthetic fold4 mask
///   tables are F2-linear subset sums, so each 8-bit mask half IS sixteen
///   8x8 bit matrices (`build_c_fold4_gfni_mats`) and one `VGF2P8AFFINEQB`
///   produces 64 lanes' worth of one output byte with no table load at all.
///   Both halves fold into the plane accumulator with a single `vpternlogq`.
///
/// Banks live byte-plane-major (`[q][bank][plane][lane]`); the caller
/// reassembles F128 lanes once per band. Everything is a reassociation of the
/// same XOR terms, so the banks are bit-identical to the incumbent drain's.
#[inline]
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni")]
pub(crate) unsafe fn accumulate_c_banks_fold4_fused_x86_gfni(
    c_group: &[u8],
    n_b_med: &[usize; 4],
    mats: &[u64; C_FOLD4_MATS_PER_GROUP],
    plane_banks: &mut [u8; C_PLANE_BANK_BYTES],
) {
    // SAFETY: same contract as this function's own.
    unsafe {
        accumulate_c_banks_fold4_fused_x86_gfni_impl::<false>(c_group, n_b_med, mats, plane_banks);
    }
}

/// First-write twin of [`accumulate_c_banks_fold4_fused_x86_gfni`]. The
/// kernel's nested loops store every output plane on every call — a dead row
/// contributes a zero mask rather than being skipped — so the first live group
/// of a band can take an all-zero register as the prior value instead of
/// loading the plane. In characteristic two the final plane values are
/// identical, so the reassembled `partial_c4` is bit-for-bit the incumbent's.
///
/// # Safety
/// As for [`accumulate_c_banks_fold4_fused_x86_gfni`], plus: the caller must
/// have established that no earlier group of this band has written the bank.
#[inline]
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni")]
pub(crate) unsafe fn write_c_banks_fold4_fused_x86_gfni(
    c_group: &[u8],
    n_b_med: &[usize; 4],
    mats: &[u64; C_FOLD4_MATS_PER_GROUP],
    plane_banks: &mut [u8; C_PLANE_BANK_BYTES],
) {
    // SAFETY: same contract as the accumulating twin.
    unsafe {
        accumulate_c_banks_fold4_fused_x86_gfni_impl::<true>(c_group, n_b_med, mats, plane_banks);
    }
}

#[inline]
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq",
    target_feature = "gfni"
))]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi,gfni")]
unsafe fn accumulate_c_banks_fold4_fused_x86_gfni_impl<const FIRST_WRITE: bool>(
    c_group: &[u8],
    n_b_med: &[usize; 4],
    mats: &[u64; C_FOLD4_MATS_PER_GROUP],
    plane_banks: &mut [u8; C_PLANE_BANK_BYTES],
) {
    use core::arch::x86_64::*;
    const ROW_BYTES: usize = ELL; // one medium row = 64 packed C bytes
    const WINDOW_BYTES: usize = 16 * ROW_BYTES; // 16 medium rows per window
    debug_assert!(c_group.len() >= 4 * WINDOW_BYTES);

    // `byte[j] = 1 << j` — the vector operand that turns VGF2P8AFFINEQB's
    // matrix operand into a per-qword 8x8 bit transpose.
    const BIT_TRANSPOSE_ID: i64 = 0x8040_2010_0804_0201u64 as i64;

    // SAFETY: every load reads one 64-byte row inside the caller's 4 KiB
    // group slice (`w < 4`, `b_med < 16`); every store covers one 64-byte
    // plane inside the fixed-size `plane_banks` array; the cfg gate supplies
    // all required target features.
    unsafe {
        let ident = _mm512_set1_epi64(BIT_TRANSPOSE_ID);
        let planes = plane_banks.as_mut_ptr();
        let base = c_group.as_ptr();

        for q in 0..N_C_Q {
            let mut masks = [[_mm512_setzero_si512(); N_C_BANKS]; 2];
            for (half, mask_half) in masks.iter_mut().enumerate() {
                // Drain row `8*half + 4*wj + j` is window `2*half + wj`'s
                // medium row `4*j + q`; dead rows stay zero exactly as the
                // incumbent's `dst.fill(0)` leaves them.
                let mut rows = [_mm512_setzero_si512(); 8];
                for wj in 0..2usize {
                    let w = 2 * half + wj;
                    let live = n_b_med[w];
                    for j in 0..4usize {
                        let b_med = 4 * j + q;
                        if b_med < live {
                            rows[4 * wj + j] =
                                _mm512_loadu_si512(base.add(w * WINDOW_BYTES + b_med * ROW_BYTES)
                                    as *const __m512i);
                        }
                    }
                }
                let cols = byte_transpose_8x64::<true>(rows);
                for (bank, slot) in mask_half.iter_mut().enumerate() {
                    *slot = _mm512_gf2p8affine_epi64_epi8::<0>(ident, cols[bank]);
                }
            }
            // One VGF2P8AFFINEQB per (mask half, output byte plane); both
            // halves fold into the plane with one vpternlogq (0x96 = a^b^c).
            for plane in 0..16usize {
                let m_lo = _mm512_set1_epi64(mats[plane] as i64);
                let m_hi = _mm512_set1_epi64(mats[16 + plane] as i64);
                for bank in 0..N_C_BANKS {
                    let g_lo = _mm512_gf2p8affine_epi64_epi8::<0>(masks[0][bank], m_lo);
                    let g_hi = _mm512_gf2p8affine_epi64_epi8::<0>(masks[1][bank], m_hi);
                    let ptr =
                        planes.add(((q * N_C_BANKS + bank) * 16 + plane) * ELL) as *mut __m512i;
                    _mm512_storeu_si512(
                        ptr,
                        _mm512_ternarylogic_epi64::<0x96>(
                            if FIRST_WRITE {
                                _mm512_setzero_si512()
                            } else {
                                _mm512_loadu_si512(ptr as *const __m512i)
                            },
                            g_lo,
                            g_hi,
                        ),
                    );
                }
            }
        }
    }
}

/// Reassemble one byte-plane C bank into its 64 F128 lanes.
///
/// The store is `lane -> [lo, hi]` with `lo = sum_k plane[k][lane] << 8k`, so
/// the low eight planes byte-transpose straight into the lanes' `lo` qwords
/// (`byte_transpose_8x64` output register `k`, qword `L`, is lane `8k + L`)
/// and the high eight into their `hi` qwords; two `vpermt2q` per output pair
/// interleave them back to AoS F128. Replaces a 1024-iteration strided scalar
/// gather per bank, run 32 times per band.
#[inline]
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi",
    target_feature = "vpclmulqdq"
))]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
pub(crate) unsafe fn c_plane_bank_to_f128_x86_avx512(
    bank_planes: &[u8; 16 * ELL],
    out: &mut [F128; ELL],
) {
    use core::arch::x86_64::*;
    // SAFETY: sixteen 64-byte plane loads cover `bank_planes` exactly and the
    // sixteen 64-byte stores cover `out` (64 F128 = 1024 bytes) exactly.
    unsafe {
        let src = bank_planes.as_ptr();
        let lo_rows: [__m512i; 8] =
            std::array::from_fn(|k| _mm512_loadu_si512(src.add(k * ELL) as *const __m512i));
        let hi_rows: [__m512i; 8] =
            std::array::from_fn(|k| _mm512_loadu_si512(src.add((8 + k) * ELL) as *const __m512i));
        let los = byte_transpose_8x64::<false>(lo_rows);
        let his = byte_transpose_8x64::<false>(hi_rows);
        let idx0 = _mm512_set_epi64(11, 3, 10, 2, 9, 1, 8, 0);
        let idx1 = _mm512_set_epi64(15, 7, 14, 6, 13, 5, 12, 4);
        let dst = out.as_mut_ptr() as *mut __m512i;
        for k in 0..8usize {
            _mm512_storeu_si512(
                dst.add(2 * k),
                _mm512_permutex2var_epi64(los[k], idx0, his[k]),
            );
            _mm512_storeu_si512(
                dst.add(2 * k + 1),
                _mm512_permutex2var_epi64(los[k], idx1, his[k]),
            );
        }
    }
}
