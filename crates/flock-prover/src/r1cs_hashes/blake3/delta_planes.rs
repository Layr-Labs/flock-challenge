pub(crate) unsafe fn build_quad_delta_planes<const PRIMED: bool>(
    input: QuadInput<'_>, even: *mut u32, delta: *mut u32, sa: *mut u32, sb: *mut u32,
) {
    unsafe {
        if ab_rebuild_enabled() {
            build_quad_split_neon_impl::<false, true, PRIMED, true, true, true, true, true, true>(
                input, core::ptr::null_mut(), even, delta, sa, sb);
        } else {
            build_quad_split_neon_impl::<false, true, PRIMED, true, true, false, false, false, true>(
                input, core::ptr::null_mut(), even, delta, sa, sb);
        }
    }
}
