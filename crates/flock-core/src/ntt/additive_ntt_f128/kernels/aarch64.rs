use crate::field::F128;

// The dense fused-three-layer kernel below deliberately owns the complete
// physical register allocation. LLVM's inlined portable graph retains the
// exact arithmetic but spills invariant twiddle halves inside the lane loop.
//
// This leaf is Apple-only because the symbol spelling and section directives
// are Mach-O specific. Other AArch64 targets retain the portable dense path.
// EOR3 is part of FEAT_SHA3, so the leaf is compiled only when both that
// feature and this module's enclosing AES/PMULL feature are available.
#[cfg(all(target_vendor = "apple", target_feature = "sha3"))]
core::arch::global_asm!(
    r#"
    .section __TEXT,__text,regular,pure_instructions
    .p2align 2
    .arch_extension aes
    .arch_extension sha3

    .macro FLOCK_FUSED3_MUL2 u0, v0x, tl0, th0, u1, v1x, tl1, th1
        pmull   v24.1q, \v0x\().1d, \tl0\().1d
        pmull   v25.1q, \v0x\().1d, \th0\().1d
        pmull2  v26.1q, \v0x\().2d, \tl0\().2d
        pmull2  v27.1q, \v0x\().2d, \th0\().2d
        pmull   v28.1q, \v1x\().1d, \tl1\().1d
        pmull   v29.1q, \v1x\().1d, \th1\().1d
        pmull2  v30.1q, \v1x\().2d, \tl1\().2d
        pmull2  v31.1q, \v1x\().2d, \th1\().2d

        eor     v25.16b, v25.16b, v26.16b
        eor     v29.16b, v29.16b, v30.16b
        ext     v26.16b, v22.16b, v27.16b, #8
        ext     v30.16b, v22.16b, v31.16b, #8
        pmull2  v27.1q, v27.2d, v23.2d
        pmull2  v31.1q, v31.2d, v23.2d
        eor3    v25.16b, v25.16b, v26.16b, v27.16b
        eor3    v29.16b, v29.16b, v30.16b, v31.16b

        ext     v26.16b, v22.16b, v25.16b, #8
        ext     v30.16b, v22.16b, v29.16b, #8
        pmull2  v27.1q, v25.2d, v23.2d
        pmull2  v31.1q, v29.2d, v23.2d
        eor     v24.16b, v24.16b, \u0\().16b
        eor     v28.16b, v28.16b, \u1\().16b
        eor3    \u0\().16b, v24.16b, v26.16b, v27.16b
        eor3    \u1\().16b, v28.16b, v30.16b, v31.16b
        eor     \v0x\().16b, \v0x\().16b, \u0\().16b
        eor     \v1x\().16b, \v1x\().16b, \u1\().16b
    .endm

    .globl _flock_ntt_fused3_dense_qresident
    .private_extern _flock_ntt_fused3_dense_qresident
_flock_ntt_fused3_dense_qresident:
    cbz     x2, 2f

    sub     sp, sp, #64
    stp     d8, d9, [sp, #0]
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp, #32]
    stp     d14, d15, [sp, #48]

    mov     x8, x2
    ld2r    {{v8.2d, v9.2d}}, [x3], #16
    ld2r    {{v10.2d, v11.2d}}, [x3], #16
    ld2r    {{v12.2d, v13.2d}}, [x3], #16
    ld2r    {{v14.2d, v15.2d}}, [x3], #16
    ld2r    {{v16.2d, v17.2d}}, [x3], #16
    ld2r    {{v18.2d, v19.2d}}, [x3], #16
    ld2r    {{v20.2d, v21.2d}}, [x3], #16
    movi    v22.2d, #0
    mov     x10, #0x87
    dup     v23.2d, x10

    mov     x9, x1
    add     x1, x0, x9
    add     x2, x1, x9
    add     x3, x2, x9
    add     x4, x3, x9
    add     x5, x4, x9
    add     x6, x5, x9
    add     x7, x6, x9

1:
    ldr     q0, [x0]
    ldr     q1, [x1]
    ldr     q2, [x2]
    ldr     q3, [x3]
    ldr     q4, [x4]
    ldr     q5, [x5]
    ldr     q6, [x6]
    ldr     q7, [x7]

    FLOCK_FUSED3_MUL2 v0, v4, v8, v9, v1, v5, v8, v9
    FLOCK_FUSED3_MUL2 v2, v6, v8, v9, v3, v7, v8, v9
    FLOCK_FUSED3_MUL2 v0, v2, v10, v11, v1, v3, v10, v11
    FLOCK_FUSED3_MUL2 v4, v6, v12, v13, v5, v7, v12, v13
    FLOCK_FUSED3_MUL2 v0, v1, v14, v15, v2, v3, v16, v17
    FLOCK_FUSED3_MUL2 v4, v5, v18, v19, v6, v7, v20, v21

    str     q0, [x0], #16
    str     q1, [x1], #16
    str     q2, [x2], #16
    str     q3, [x3], #16
    str     q4, [x4], #16
    str     q5, [x5], #16
    str     q6, [x6], #16
    str     q7, [x7], #16
    subs    x8, x8, #1
    b.ne    1b

    ldp     d8, d9, [sp, #0]
    ldp     d10, d11, [sp, #16]
    ldp     d12, d13, [sp, #32]
    ldp     d14, d15, [sp, #48]
    add     sp, sp, #64
2:
    ret
    "#,
);

#[cfg(all(target_vendor = "apple", target_feature = "sha3"))]
unsafe extern "C" {
    fn flock_ntt_fused3_dense_qresident(
        row0: *mut F128,
        row_stride_bytes: usize,
        active_lanes: usize,
        twiddles: *const F128,
    );
}

/// Process one dense in-place fused-three-layer row group with a fixed
/// register allocation.
///
/// # Safety
/// The caller guarantees that all eight selected rows and the active lane
/// prefix are valid and exclusively writable.
#[cfg(all(target_vendor = "apple", target_feature = "sha3"))]
#[inline]
pub(super) unsafe fn butterfly_fused_3layer_row_dense_qresident(
    ptr: *mut F128,
    eighth: usize,
    num_ntts: usize,
    r: usize,
    twiddles: &[F128; 7],
    active_lanes: usize,
) {
    debug_assert!(active_lanes <= num_ntts);
    debug_assert!(r < eighth);
    let row0 = unsafe { ptr.add(r * num_ntts) };
    let row_stride_bytes = eighth * num_ntts * core::mem::size_of::<F128>();
    unsafe {
        flock_ntt_fused3_dense_qresident(row0, row_stride_bytes, active_lanes, twiddles.as_ptr())
    }
}

/// Process two butterflies at a time within a block sharing one twiddle.
///
/// # Safety
/// Requires the `aes` target feature.
#[target_feature(enable = "aes")]
pub(super) unsafe fn butterfly_block(chunk: &mut [F128], twiddle: F128, half: usize) {
    use crate::field::gf2_128::aarch64::ghash_mul_vec2_neon;

    debug_assert!(half >= 2);
    debug_assert_eq!(chunk.len(), 2 * half);
    let mut idx0 = 0;
    while idx0 < half {
        let idx1 = idx0 + half;
        let u_a = chunk[idx0];
        let v_a = chunk[idx1];
        let u_b = chunk[idx0 + 1];
        let v_b = chunk[idx1 + 1];

        // SAFETY: caller guarantees the aes target feature.
        let product = unsafe { ghash_mul_vec2_neon([twiddle, twiddle], [v_a, v_b]) };
        let new_u_a = F128 {
            lo: u_a.lo ^ product[0].lo,
            hi: u_a.hi ^ product[0].hi,
        };
        let new_u_b = F128 {
            lo: u_b.lo ^ product[1].lo,
            hi: u_b.hi ^ product[1].hi,
        };
        let new_v_a = F128 {
            lo: v_a.lo ^ new_u_a.lo,
            hi: v_a.hi ^ new_u_a.hi,
        };
        let new_v_b = F128 {
            lo: v_b.lo ^ new_u_b.lo,
            hi: v_b.hi ^ new_u_b.hi,
        };

        chunk[idx0] = new_u_a;
        chunk[idx1] = new_v_a;
        chunk[idx0 + 1] = new_u_b;
        chunk[idx1 + 1] = new_v_b;
        idx0 += 2;
    }
}

/// Process the single pair in each of two adjacent blocks with distinct
/// twiddles.
///
/// # Safety
/// Requires the `aes` target feature.
#[target_feature(enable = "aes")]
pub(super) unsafe fn butterfly_block_pair(chunk: &mut [F128], t_a: F128, t_b: F128) {
    use crate::field::gf2_128::aarch64::ghash_mul_vec2_neon;

    debug_assert_eq!(chunk.len(), 4);
    let u_a = chunk[0];
    let v_a = chunk[1];
    let u_b = chunk[2];
    let v_b = chunk[3];

    // SAFETY: caller guarantees the aes target feature.
    let product = unsafe { ghash_mul_vec2_neon([t_a, t_b], [v_a, v_b]) };
    let new_u_a = F128 {
        lo: u_a.lo ^ product[0].lo,
        hi: u_a.hi ^ product[0].hi,
    };
    let new_u_b = F128 {
        lo: u_b.lo ^ product[1].lo,
        hi: u_b.hi ^ product[1].hi,
    };
    let new_v_a = F128 {
        lo: v_a.lo ^ new_u_a.lo,
        hi: v_a.hi ^ new_u_a.hi,
    };
    let new_v_b = F128 {
        lo: v_b.lo ^ new_u_b.lo,
        hi: v_b.hi ^ new_u_b.hi,
    };

    chunk[0] = new_u_a;
    chunk[1] = new_v_a;
    chunk[2] = new_u_b;
    chunk[3] = new_v_b;
}
