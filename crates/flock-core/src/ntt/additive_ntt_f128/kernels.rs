use crate::field::F128;

mod portable;

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
mod aarch64;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
mod x86_64;

#[inline]
pub(super) fn butterfly_row_pair(top: &mut [F128], bot: &mut [F128], twiddle: F128) {
    debug_assert_eq!(top.len(), bot.len());

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    unsafe {
        x86_64::butterfly_row_pair(top, bot, twiddle);
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    )))]
    portable::butterfly_row_pair(top, bot, twiddle);
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn butterfly_fused_2layer(
    a: &mut [F128],
    b: &mut [F128],
    c: &mut [F128],
    d: &mut [F128],
    t_outer: F128,
    t_inner_a: F128,
    t_inner_b: F128,
) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), c.len());
    debug_assert_eq!(a.len(), d.len());

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    unsafe {
        x86_64::butterfly_fused_2layer(a, b, c, d, t_outer, t_inner_a, t_inner_b);
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    )))]
    portable::butterfly_fused_2layer(a, b, c, d, t_outer, t_inner_a, t_inner_b);
}

#[allow(dead_code)]
#[inline]
pub(super) unsafe fn butterfly_fused_4layer_row(
    ptr: *mut F128,
    sixteenth: usize,
    num_ntts: usize,
    r: usize,
    twiddles: &[F128; 15],
) {
    unsafe {
        butterfly_fused_4layer_rows(ptr, sixteenth, num_ntts, r, 1, twiddles);
    }
}

#[inline]
pub(super) unsafe fn butterfly_fused_4layer_rows(
    ptr: *mut F128,
    sixteenth: usize,
    num_ntts: usize,
    r_start: usize,
    r_count: usize,
    twiddles: &[F128; 15],
) {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    ))]
    unsafe {
        for r in r_start..(r_start + r_count) {
            x86_64::butterfly_fused_4layer_row(ptr, sixteenth, num_ntts, r, twiddles);
        }
    }

    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "vpclmulqdq"
    )))]
    unsafe {
        portable::butterfly_fused_4layer_rows(ptr, sixteenth, num_ntts, r_start, r_count, twiddles);
    }
}

#[inline]
pub(super) unsafe fn butterfly_fused_3layer_row(
    ptr: *mut F128,
    eighth: usize,
    num_ntts: usize,
    r: usize,
    twiddles: &[F128; 7],
) {
    unsafe {
        portable::butterfly_fused_3layer_row(ptr, eighth, num_ntts, r, twiddles);
    }
}

#[allow(dead_code, clippy::too_many_arguments)]
#[inline]
pub(super) unsafe fn butterfly_fused_4layer_dual_from_src_row(
    src_ptr: *const F128,
    dst0_ptr: *mut F128,
    dst1_ptr: *mut F128,
    sixteenth: usize,
    num_ntts: usize,
    r: usize,
    tw0: &[F128; 15],
    tw1: &[F128; 15],
) {
    unsafe {
        portable::butterfly_fused_4layer_dual_from_src_rows(
            src_ptr, dst0_ptr, dst1_ptr, sixteenth, num_ntts, r, 1, tw0, tw1,
        );
    }
}

#[allow(dead_code, clippy::too_many_arguments)]
#[inline]
pub(super) unsafe fn butterfly_fused_4layer_dual_from_src_rows(
    src_ptr: *const F128,
    dst0_ptr: *mut F128,
    dst1_ptr: *mut F128,
    sixteenth: usize,
    num_ntts: usize,
    r_start: usize,
    r_count: usize,
    tw0: &[F128; 15],
    tw1: &[F128; 15],
) {
    unsafe {
        portable::butterfly_fused_4layer_dual_from_src_rows(
            src_ptr, dst0_ptr, dst1_ptr, sixteenth, num_ntts, r_start, r_count, tw0, tw1,
        );
    }
}

#[allow(dead_code, clippy::too_many_arguments)]
#[inline]
pub(super) unsafe fn butterfly_fused_4layer_row_from_replicated_msg(
    msg_ptr: *const F128,
    msg_len: usize,
    dst_off: usize,
    dst_ptr: *mut F128,
    sixteenth: usize,
    num_ntts: usize,
    r: usize,
    twiddles: &[F128; 15],
) {
    unsafe {
        portable::butterfly_fused_4layer_row_from_replicated_msg(
            msg_ptr, msg_len, dst_off, dst_ptr, sixteenth, num_ntts, r, twiddles,
        );
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline]
pub(super) unsafe fn butterfly_neon_block(chunk: &mut [F128], twiddle: F128, half: usize) {
    unsafe { aarch64::butterfly_block(chunk, twiddle, half) }
}

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline]
pub(super) unsafe fn butterfly_neon_block_pair(
    data: &mut [F128],
    base: usize,
    t_a: F128,
    t_b: F128,
) {
    unsafe { aarch64::butterfly_block_pair(&mut data[base..base + 4], t_a, t_b) }
}

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[inline]
pub(super) unsafe fn butterfly_neon_block_pair_chunk(chunk: &mut [F128], t_a: F128, t_b: F128) {
    unsafe { aarch64::butterfly_block_pair(chunk, t_a, t_b) }
}
