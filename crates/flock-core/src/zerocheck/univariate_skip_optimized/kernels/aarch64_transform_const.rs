/// Compile-time selection for transforms with a structurally fixed byte tail.
pub(crate) const ABV_TRANSFORM_CONST: u32 = 1 << 18;

/// The ordinary z map with bytes NB..8 fixed to 0xff, NB = 5 or 6.
/// The caller's existing static-B guard establishes these bytes. Thus the
/// {6,7} lookup is zero, the {5,7} lookup is also zero for NB = 5, and the
/// all-byte term is the constant-one vector. All other terms are unchanged.
#[inline(always)]
unsafe fn apply_word_z_fixed_tail<const NB: u32>(
    t0: *const u8,
    kara: *const u8,
    w: u64,
) -> (
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        let z = kara.add(2 * AB_KARA_IMAGE_BYTES);
        let ys = w ^ (w >> 8);
        let yl1 = w ^ (w >> 16);
        let yl2 = w ^ (w >> 32);
        let ysl1 = ys ^ (ys >> 16);
        let ysl2 = ys ^ (ys >> 32);
        let yl12 = yl1 ^ (yl1 >> 32);
        let yall = ysl1 ^ (ysl1 >> 32);
        let ld = |base: *const u8, off: usize| vld1q_u8(base.add(off));
        let r0 = t0.add(ab_z_idx::<0, 6>(yall));
        let (l0, l1, l2, l3) = (ld(r0, 0), ld(r0, 16), ld(r0, 32), ld(r0, 48));
        let rec = |i: usize| -> *const u8 {
            let a: usize;
            core::arch::asm!(
                "add {a}, {b}, {i}",
                a = lateout(reg) a,
                b = in(reg) z.add(AB_Z_REC),
                i = in(reg) i,
                options(pure, nomem, nostack, preserves_flags),
            );
            a as *const u8
        };
        let a = rec(ab_z_idx::<2, 7>(ysl2));
        let (m0, m2) = (ld(a, 0), ld(a, 16));
        let a = rec(ab_z_idx::<4, 7>(ysl1));
        let (n0, n1) = (ld(a, 32), ld(a, 48));
        let a = rec(ab_z_idx::<1, 7>(yl12));
        let (d01, d23) = (ld(a, 64), ld(a, 80));
        let a = rec(ab_z_idx::<3, 7>(yl2));
        let (e0, e2) = (ld(a, 96), ld(a, 112));
        let one = vdupq_n_u8(1);
        let fg = if NB == 6 {
            veorq_u8(ld(z.add(AB_Z_SL2), ab_z_idx::<5, 4>(yl1)), one)
        } else {
            one
        };
        let p = vreinterpretq_u64_u8(xor3_u8(d01, e0, fg));
        let q = vreinterpretq_u64_u8(xor3_u8(d23, e2, fg));
        let b0 = vreinterpretq_u8_u64(vzip1q_u64(p, p));
        let b1 = vreinterpretq_u8_u64(vzip2q_u64(p, p));
        let b2 = vreinterpretq_u8_u64(vzip1q_u64(q, q));
        let b3 = vreinterpretq_u8_u64(vzip2q_u64(q, q));
        (
            veorq_u8(xor3_u8(l0, m0, n0), b0),
            veorq_u8(xor3_u8(l1, m0, n1), b1),
            veorq_u8(xor3_u8(l2, m2, n0), b2),
            veorq_u8(xor3_u8(l3, m2, n1), b3),
        )
    }
}

/// Half-pair transform of only the low NB bytes, NB = 5 or 6. The high
/// pair is zero; its full-row and sigma lookups disappear. For NB = 5 the
/// next odd byte is zero as well. The caller adds the existing fixed partial.
#[inline(always)]
unsafe fn apply_word_hz_low<const NB: u32>(
    t0: *const u8,
    sg: *const u8,
    w: u64,
) -> (
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        let v = w & ((1u64 << (8 * NB)) - 1);
        let y = v ^ (v >> 8);
        let ld = |base: *const u8, off: usize| vld1q_u8(base.add(off));
        let at = |base: *const u8, i: usize| -> *const u8 {
            let a: usize;
            core::arch::asm!(
                "add {a}, {b}, {i}",
                a = lateout(reg) a,
                b = in(reg) base,
                i = in(reg) i,
                options(pure, nomem, nostack, preserves_flags),
            );
            a as *const u8
        };
        let r0 = at(t0, ab_z_idx::<0, 6>(y));
        let r1 = at(t0, ab_z_idx::<2, 6>(y));
        let r2 = at(t0, ab_z_idx::<4, 6>(y));
        let s0 = at(sg, ab_z_idx::<1, 5>(v));
        let s1 = at(sg, ab_z_idx::<3, 5>(v));
        let (p0, q0) = (ld(s0, 0), ld(s0, 16));
        let (p1, q1) = (ld(s1, 0), ld(s1, 16));
        let (p2, q2) = if NB == 6 {
            let s2 = at(sg, ab_z_idx::<5, 5>(v));
            (ld(s2, 0), ld(s2, 16))
        } else {
            (vdupq_n_u8(0), vdupq_n_u8(0))
        };
        let x = vreinterpretq_u64_u8(xor3_u8(p0, q2, vextq_u8::<8>(p1, p1)));
        let yv = vreinterpretq_u64_u8(xor3_u8(q0, p2, vextq_u8::<8>(q1, q1)));
        let t0d = vreinterpretq_u8_u64(vzip1q_u64(x, x));
        let t1d = vreinterpretq_u8_u64(vzip2q_u64(x, x));
        let t2d = vreinterpretq_u8_u64(vzip1q_u64(yv, yv));
        let t3d = vreinterpretq_u8_u64(vzip2q_u64(yv, yv));
        (
            veorq_u8(xor3_u8(ld(r0, 0), ld(r1, 16), ld(r2, 32)), t0d),
            veorq_u8(xor3_u8(ld(r0, 16), ld(r1, 0), ld(r2, 48)), t1d),
            veorq_u8(xor3_u8(ld(r0, 32), ld(r1, 48), ld(r2, 0)), t2d),
            veorq_u8(xor3_u8(ld(r0, 48), ld(r1, 32), ld(r2, 16)), t3d),
        )
    }
}
