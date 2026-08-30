use crate::field::F128;

// ---------------------------------------------------------------------------
// Vector-resident fused radix-8 rows.
//
// The portable fused-three-layer row kernel expresses its chain in terms of
// `F128` values: `values[u] + values[v] * twiddle`. Each `*` calls
// `ghash_mul_binius`, which repacks its NEON accumulator into
// `F128 { lo, hi }` on return, and each `+` is then a pair of scalar `u64`
// XORs.
//
// The kernels below keep all row values, and every intermediate, in
// `uint64x2_t` for the whole three-layer and two-layer chains. Values enter
// and leave vector registers exactly once per row group; XOR becomes `veorq_u64`
// (or `veor3q_u64`) and the multiply never repacks or extracts to scalar registers.
//
// Multiplications with constant/twiddle values prepare `t_swap = vextq_u64::<1>(t, t)`
// once outside the lane loop, allowing cross terms and reductions to issue as pure
// vector `pmull` / `pmull2` operations with zero scalar-vector register transfers.
// ---------------------------------------------------------------------------

/// Carry-less multiply of lower 64-bit lanes, result in a q register.
#[inline(always)]
unsafe fn pmull_ll(
    a: core::arch::aarch64::uint64x2_t,
    b: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    use core::arch::aarch64::*;
    unsafe {
        core::mem::transmute::<u128, uint64x2_t>(vmull_p64(
            vgetq_lane_u64::<0>(a),
            vgetq_lane_u64::<0>(b),
        ))
    }
}

/// Carry-less multiply of upper 64-bit lanes, result in a q register.
#[inline(always)]
unsafe fn pmull_hh(
    a: core::arch::aarch64::uint64x2_t,
    b: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    use core::arch::aarch64::*;
    unsafe {
        core::mem::transmute::<u128, uint64x2_t>(vmull_high_p64(
            vreinterpretq_p64_u64(a),
            vreinterpretq_p64_u64(b),
        ))
    }
}

#[cfg(target_feature = "sha3")]
#[inline(always)]
unsafe fn xor3(
    a: core::arch::aarch64::uint64x2_t,
    b: core::arch::aarch64::uint64x2_t,
    c: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    unsafe { core::arch::aarch64::veor3q_u64(a, b, c) }
}

#[cfg(not(target_feature = "sha3"))]
#[inline(always)]
unsafe fn xor3(
    a: core::arch::aarch64::uint64x2_t,
    b: core::arch::aarch64::uint64x2_t,
    c: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    use core::arch::aarch64::*;
    unsafe { veorq_u64(a, veorq_u64(b, c)) }
}

/// GHASH multiply with both operands and the result held in q registers.
/// Uses pre-swapped twiddle `b_swap = vextq_u64::<1>(b, b)`, `c87 = vdupq_n_u64(0x87)`,
/// and `zero = vdupq_n_u64(0)` to avoid any scalar register extraction.
#[inline(always)]
unsafe fn mul_q_prepared(
    a: core::arch::aarch64::uint64x2_t,
    b: core::arch::aarch64::uint64x2_t,
    b_swap: core::arch::aarch64::uint64x2_t,
    c87: core::arch::aarch64::uint64x2_t,
    zero: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    use core::arch::aarch64::*;
    unsafe {
        let t0 = pmull_ll(a, b);
        let t1a = pmull_ll(a, b_swap);
        let t1b = pmull_hh(a, b_swap);
        let t2 = pmull_hh(a, b);
        let t1_cross = veorq_u64(t1a, t1b);

        // t1 += x^64 · t2 (mod p): {0, t2.lo} places t2.lo into t1.hi.
        let t2_red = pmull_hh(t2, c87);
        let t1 = xor3(
            t1_cross,
            vextq_u64::<1>(zero, t2),
            t2_red,
        );

        // t0 += x^64 · t1 (mod p).
        let t1_red = pmull_hh(t1, c87);
        xor3(
            t0,
            vextq_u64::<1>(zero, t1),
            t1_red,
        )
    }
}

#[inline(always)]
unsafe fn mul_q(
    a: core::arch::aarch64::uint64x2_t,
    b: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    use core::arch::aarch64::*;
    unsafe {
        let zero = vdupq_n_u64(0);
        let c87 = vdupq_n_u64(0x87);
        let b_swap = vextq_u64::<1>(b, b);
        mul_q_prepared(a, b, b_swap, c87, zero)
    }
}

/// One forward butterfly, fully in q registers with prepared twiddle constants:
/// `u' = u + v·t`, `v' = v + u'`.
#[inline(always)]
unsafe fn butterfly_q_prepared(
    u: core::arch::aarch64::uint64x2_t,
    v: core::arch::aarch64::uint64x2_t,
    t: core::arch::aarch64::uint64x2_t,
    t_swap: core::arch::aarch64::uint64x2_t,
    c87: core::arch::aarch64::uint64x2_t,
    zero: core::arch::aarch64::uint64x2_t,
) -> (
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        let new_u = veorq_u64(u, mul_q_prepared(v, t, t_swap, c87, zero));
        (new_u, veorq_u64(v, new_u))
    }
}

/// One forward butterfly, fully in q registers:
/// `u' = u + v·t`, `v' = v + u'`.
#[inline(always)]
unsafe fn butterfly_q(
    u: core::arch::aarch64::uint64x2_t,
    v: core::arch::aarch64::uint64x2_t,
    t: core::arch::aarch64::uint64x2_t,
) -> (
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        let new_u = veorq_u64(u, mul_q(v, t));
        (new_u, veorq_u64(v, new_u))
    }
}

/// Zero-twiddle butterfly: `u' = u`, `v' = v + u`.
#[inline(always)]
unsafe fn butterfly_zero_q(
    u: core::arch::aarch64::uint64x2_t,
    v: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    unsafe { core::arch::aarch64::veorq_u64(v, u) }
}

#[target_feature(enable = "aes")]
unsafe fn butterfly_fused_3layer_row_with_q(
    ptr: *mut F128,
    eighth: usize,
    num_ntts: usize,
    lanes: usize,
    r: usize,
    t: &[core::arch::aarch64::uint64x2_t; 7],
) {
    use core::arch::aarch64::*;
    unsafe {
        let zero = vdupq_n_u64(0);
        let c87 = vdupq_n_u64(0x87);
        let t_swap: [uint64x2_t; 7] = core::array::from_fn(|i| vextq_u64::<1>(t[i], t[i]));
        let step = eighth * num_ntts;
        let base_r = ptr.add(r * num_ntts);

        let mut lane = 0;
        while lane + 1 < lanes {
            let base0 = base_r.add(lane);
            let base1 = base_r.add(lane + 1);

            let mut v0: [uint64x2_t; 8] =
                core::array::from_fn(|i| vld1q_u64(base0.add(i * step).cast::<u64>()));
            let mut v1: [uint64x2_t; 8] =
                core::array::from_fn(|i| vld1q_u64(base1.add(i * step).cast::<u64>()));

            // Layer L: stride 4, shared twiddle t[0].
            for i in 0..4 {
                let (a0, b0) = butterfly_q_prepared(v0[i], v0[i + 4], t[0], t_swap[0], c87, zero);
                let (a1, b1) = butterfly_q_prepared(v1[i], v1[i + 4], t[0], t_swap[0], c87, zero);
                v0[i] = a0;
                v0[i + 4] = b0;
                v1[i] = a1;
                v1[i + 4] = b1;
            }
            // Layer L+1: stride 2, twiddles t[1], t[2] per half.
            for s in 0..2 {
                let ts = t[1 + s];
                let ts_swap = t_swap[1 + s];
                for i in 0..2 {
                    let (u, w) = (4 * s + i, 4 * s + i + 2);
                    let (a0, b0) = butterfly_q_prepared(v0[u], v0[w], ts, ts_swap, c87, zero);
                    let (a1, b1) = butterfly_q_prepared(v1[u], v1[w], ts, ts_swap, c87, zero);
                    v0[u] = a0;
                    v0[w] = b0;
                    v1[u] = a1;
                    v1[w] = b1;
                }
            }
            // Layer L+2: stride 1, twiddles t[3..7] per quarter.
            for s in 0..4 {
                let ts = t[3 + s];
                let ts_swap = t_swap[3 + s];
                let (a0, b0) = butterfly_q_prepared(v0[2 * s], v0[2 * s + 1], ts, ts_swap, c87, zero);
                let (a1, b1) = butterfly_q_prepared(v1[2 * s], v1[2 * s + 1], ts, ts_swap, c87, zero);
                v0[2 * s] = a0;
                v0[2 * s + 1] = b0;
                v1[2 * s] = a1;
                v1[2 * s + 1] = b1;
            }

            for (i, value) in v0.iter().enumerate() {
                vst1q_u64(base0.add(i * step).cast::<u64>(), *value);
            }
            for (i, value) in v1.iter().enumerate() {
                vst1q_u64(base1.add(i * step).cast::<u64>(), *value);
            }
            lane += 2;
        }

        if lane < lanes {
            let base = base_r.add(lane);
            let mut v: [uint64x2_t; 8] =
                core::array::from_fn(|i| vld1q_u64(base.add(i * step).cast::<u64>()));

            // Layer L: stride 4, shared twiddle t[0].
            for i in 0..4 {
                let (a, b) = butterfly_q_prepared(v[i], v[i + 4], t[0], t_swap[0], c87, zero);
                v[i] = a;
                v[i + 4] = b;
            }
            // Layer L+1: stride 2, twiddles t[1], t[2] per half.
            for s in 0..2 {
                let ts = t[1 + s];
                let ts_swap = t_swap[1 + s];
                for i in 0..2 {
                    let (u, w) = (4 * s + i, 4 * s + i + 2);
                    let (a, b) = butterfly_q_prepared(v[u], v[w], ts, ts_swap, c87, zero);
                    v[u] = a;
                    v[w] = b;
                }
            }
            // Layer L+2: stride 1, twiddles t[3..7] per quarter.
            for s in 0..4 {
                let ts = t[3 + s];
                let ts_swap = t_swap[3 + s];
                let (a, b) = butterfly_q_prepared(v[2 * s], v[2 * s + 1], ts, ts_swap, c87, zero);
                v[2 * s] = a;
                v[2 * s + 1] = b;
            }

            for (i, value) in v.iter().enumerate() {
                vst1q_u64(base.add(i * step).cast::<u64>(), *value);
            }
        }
    }
}

/// Vector-resident rate-1/2 first-pass row kernel with staged streaming
/// stores. Loads the radix-8 row group from `src` once, evaluates both
/// layer-1 blocks' fused-3 butterflies on those registers (zero-root set for
/// `dst0`, general set for `dst1`), staging all outputs in two L1-resident
/// stack tiles, then emits every destination row's lane run as one
/// contiguous `stnp` burst.
///
/// # Safety
/// Row/lane geometry valid for all three pointers, disjoint row groups
/// across concurrent calls, `num_ntts ≤ 64`, 16 B-aligned destinations, and
/// `t_zero[0] == t_zero[1] == t_zero[3] == 0`.
#[target_feature(enable = "aes")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn butterfly_fused_3layer_dual_from_src_row(
    src: *const F128,
    dst0: *mut F128,
    dst1: *mut F128,
    eighth: usize,
    num_ntts: usize,
    r: usize,
    t_zero: &[F128; 7],
    t_gen: &[F128; 7],
) {
    use core::arch::aarch64::*;

    #[inline(always)]
    fn nt_bursts() -> bool {
        use std::sync::LazyLock;
        static ON: LazyLock<bool> =
            LazyLock::new(|| std::env::var_os("FLOCK_FROM_MSG_PLAIN_ST").is_none());
        *ON
    }

    #[inline(always)]
    unsafe fn store_pair_nt(dst: *mut F128, x: uint64x2_t, y: uint64x2_t) {
        unsafe {
            core::arch::asm!(
                "stnp {x:q}, {y:q}, [{dst}]",
                dst = in(reg) dst,
                x = in(vreg) x,
                y = in(vreg) y,
                options(nostack, preserves_flags),
            );
        }
    }

    unsafe {
        debug_assert_eq!(t_zero[0], F128::ZERO);
        debug_assert_eq!(t_zero[1], F128::ZERO);
        debug_assert_eq!(t_zero[3], F128::ZERO);
        debug_assert!(num_ntts <= 64 && num_ntts.is_multiple_of(2));

        let zero = vdupq_n_u64(0);
        let c87 = vdupq_n_u64(0x87);

        let z2 = vld1q_u64((&raw const t_zero[2]).cast::<u64>());
        let z4 = vld1q_u64((&raw const t_zero[4]).cast::<u64>());
        let z5 = vld1q_u64((&raw const t_zero[5]).cast::<u64>());
        let z6 = vld1q_u64((&raw const t_zero[6]).cast::<u64>());
        let z2_swap = vextq_u64::<1>(z2, z2);
        let z4_swap = vextq_u64::<1>(z4, z4);
        let z5_swap = vextq_u64::<1>(z5, z5);
        let z6_swap = vextq_u64::<1>(z6, z6);

        let g: [uint64x2_t; 7] =
            core::array::from_fn(|i| vld1q_u64((&raw const t_gen[i]).cast::<u64>()));
        let g_swap: [uint64x2_t; 7] = core::array::from_fn(|i| vextq_u64::<1>(g[i], g[i]));

        let mut stage0 = [F128 { lo: 0, hi: 0 }; 512];
        let mut stage1 = [F128 { lo: 0, hi: 0 }; 512];

        let off = r * num_ntts;
        let step = eighth * num_ntts;
        let use_nt_bursts = nt_bursts();
        for lane in 0..num_ntts {
            let src_base = src.add(off + lane);
            let loaded: [uint64x2_t; 8] =
                core::array::from_fn(|i| vld1q_u64(src_base.add(i * step).cast::<u64>()));

            // Block 0: zero-root chain
            {
                let mut v = loaded;
                for i in 0..4 {
                    v[i + 4] = butterfly_zero_q(v[i], v[i + 4]);
                }
                for i in 0..2 {
                    v[i + 2] = butterfly_zero_q(v[i], v[i + 2]);
                }
                let (a, b) = butterfly_q_prepared(v[4], v[6], z2, z2_swap, c87, zero);
                v[4] = a;
                v[6] = b;
                let (a, b) = butterfly_q_prepared(v[5], v[7], z2, z2_swap, c87, zero);
                v[5] = a;
                v[7] = b;
                v[1] = butterfly_zero_q(v[0], v[1]);
                let (a, b) = butterfly_q_prepared(v[2], v[3], z4, z4_swap, c87, zero);
                v[2] = a;
                v[3] = b;
                let (a, b) = butterfly_q_prepared(v[4], v[5], z5, z5_swap, c87, zero);
                v[4] = a;
                v[5] = b;
                let (a, b) = butterfly_q_prepared(v[6], v[7], z6, z6_swap, c87, zero);
                v[6] = a;
                v[7] = b;
                for (i, value) in v.iter().enumerate() {
                    vst1q_u64(
                        stage0.as_mut_ptr().add(i * num_ntts + lane).cast::<u64>(),
                        *value,
                    );
                }
            }

            // Block 1: general chain
            {
                let mut v = loaded;
                for i in 0..4 {
                    let (a, b) = butterfly_q_prepared(v[i], v[i + 4], g[0], g_swap[0], c87, zero);
                    v[i] = a;
                    v[i + 4] = b;
                }
                for s in 0..2 {
                    let gs = g[1 + s];
                    let gs_swap = g_swap[1 + s];
                    for i in 0..2 {
                        let (u, w) = (4 * s + i, 4 * s + i + 2);
                        let (a, b) = butterfly_q_prepared(v[u], v[w], gs, gs_swap, c87, zero);
                        v[u] = a;
                        v[w] = b;
                    }
                }
                for s in 0..4 {
                    let gs = g[3 + s];
                    let gs_swap = g_swap[3 + s];
                    let (a, b) = butterfly_q_prepared(v[2 * s], v[2 * s + 1], gs, gs_swap, c87, zero);
                    v[2 * s] = a;
                    v[2 * s + 1] = b;
                }
                for (i, value) in v.iter().enumerate() {
                    vst1q_u64(
                        stage1.as_mut_ptr().add(i * num_ntts + lane).cast::<u64>(),
                        *value,
                    );
                }
            }
        }

        // Emit each destination row's full lane run as one sequential non-temporal burst.
        for i in 0..8 {
            let s0 = stage0.as_ptr().add(i * num_ntts);
            let s1 = stage1.as_ptr().add(i * num_ntts);
            let d0 = dst0.add(off + i * step);
            let d1 = dst1.add(off + i * step);
            let mut lane = 0;
            while lane < num_ntts {
                let x = vld1q_u64(s0.add(lane).cast::<u64>());
                let y = vld1q_u64(s0.add(lane + 1).cast::<u64>());
                if use_nt_bursts {
                    store_pair_nt(d0.add(lane), x, y);
                } else {
                    vst1q_u64(d0.add(lane).cast::<u64>(), x);
                    vst1q_u64(d0.add(lane + 1).cast::<u64>(), y);
                }
                let x = vld1q_u64(s1.add(lane).cast::<u64>());
                let y = vld1q_u64(s1.add(lane + 1).cast::<u64>());
                if use_nt_bursts {
                    store_pair_nt(d1.add(lane), x, y);
                } else {
                    vst1q_u64(d1.add(lane).cast::<u64>(), x);
                    vst1q_u64(d1.add(lane + 1).cast::<u64>(), y);
                }
                lane += 2;
            }
        }
    }
}

/// Out-of-place radix-8 row used by recursive from-message commitments.
#[target_feature(enable = "aes")]
unsafe fn butterfly_fused_3layer_from_src_row_impl<const ZERO_ROOT: bool>(
    src: *const F128,
    dst: *mut F128,
    eighth: usize,
    num_ntts: usize,
    r: usize,
    twiddles: &[F128; 7],
) {
    use core::arch::aarch64::*;

    #[inline(always)]
    unsafe fn store_pair_nt(dst: *mut F128, x: uint64x2_t, y: uint64x2_t) {
        unsafe {
            core::arch::asm!(
                "stnp {x:q}, {y:q}, [{dst}]",
                dst = in(reg) dst,
                x = in(vreg) x,
                y = in(vreg) y,
                options(nostack, preserves_flags),
            );
        }
    }

    unsafe {
        debug_assert_eq!(num_ntts, 8);
        if ZERO_ROOT {
            debug_assert_eq!(twiddles[0], F128::ZERO);
            debug_assert_eq!(twiddles[1], F128::ZERO);
            debug_assert_eq!(twiddles[3], F128::ZERO);
        }
        let zero = vdupq_n_u64(0);
        let c87 = vdupq_n_u64(0x87);
        let t: [uint64x2_t; 7] =
            core::array::from_fn(|i| vld1q_u64((&raw const twiddles[i]).cast::<u64>()));
        let t_swap: [uint64x2_t; 7] = core::array::from_fn(|i| vextq_u64::<1>(t[i], t[i]));
        let mut stage = [F128 { lo: 0, hi: 0 }; 64];
        let off = r * num_ntts;
        let step = eighth * num_ntts;

        for lane in 0..num_ntts {
            let src_base = src.add(off + lane);
            let mut v: [uint64x2_t; 8] =
                core::array::from_fn(|i| vld1q_u64(src_base.add(i * step).cast::<u64>()));

            if ZERO_ROOT {
                for i in 0..4 {
                    v[i + 4] = butterfly_zero_q(v[i], v[i + 4]);
                }
                for i in 0..2 {
                    v[i + 2] = butterfly_zero_q(v[i], v[i + 2]);
                }
                let (a, b) = butterfly_q_prepared(v[4], v[6], t[2], t_swap[2], c87, zero);
                v[4] = a;
                v[6] = b;
                let (a, b) = butterfly_q_prepared(v[5], v[7], t[2], t_swap[2], c87, zero);
                v[5] = a;
                v[7] = b;
                v[1] = butterfly_zero_q(v[0], v[1]);
                let (a, b) = butterfly_q_prepared(v[2], v[3], t[4], t_swap[4], c87, zero);
                v[2] = a;
                v[3] = b;
                let (a, b) = butterfly_q_prepared(v[4], v[5], t[5], t_swap[5], c87, zero);
                v[4] = a;
                v[5] = b;
                let (a, b) = butterfly_q_prepared(v[6], v[7], t[6], t_swap[6], c87, zero);
                v[6] = a;
                v[7] = b;
            } else {
                for i in 0..4 {
                    let (a, b) = butterfly_q_prepared(v[i], v[i + 4], t[0], t_swap[0], c87, zero);
                    v[i] = a;
                    v[i + 4] = b;
                }
                for s in 0..2 {
                    let ts = t[1 + s];
                    let ts_swap = t_swap[1 + s];
                    for i in 0..2 {
                        let (u, w) = (4 * s + i, 4 * s + i + 2);
                        let (a, b) = butterfly_q_prepared(v[u], v[w], ts, ts_swap, c87, zero);
                        v[u] = a;
                        v[w] = b;
                    }
                }
                for s in 0..4 {
                    let ts = t[3 + s];
                    let ts_swap = t_swap[3 + s];
                    let (a, b) = butterfly_q_prepared(v[2 * s], v[2 * s + 1], ts, ts_swap, c87, zero);
                    v[2 * s] = a;
                    v[2 * s + 1] = b;
                }
            }

            for (i, value) in v.iter().enumerate() {
                vst1q_u64(
                    stage.as_mut_ptr().add(i * num_ntts + lane).cast::<u64>(),
                    *value,
                );
            }
        }

        for i in 0..8 {
            let src_row = stage.as_ptr().add(i * num_ntts);
            let dst_row = dst.add(off + i * step);
            let mut lane = 0;
            while lane < num_ntts {
                let x = vld1q_u64(src_row.add(lane).cast::<u64>());
                let y = vld1q_u64(src_row.add(lane + 1).cast::<u64>());
                store_pair_nt(dst_row.add(lane), x, y);
                lane += 2;
            }
        }
    }
}

#[target_feature(enable = "aes")]
pub(super) unsafe fn butterfly_fused_3layer_from_src_row(
    src: *const F128,
    dst: *mut F128,
    eighth: usize,
    num_ntts: usize,
    r: usize,
    twiddles: &[F128; 7],
) {
    unsafe {
        butterfly_fused_3layer_from_src_row_impl::<false>(src, dst, eighth, num_ntts, r, twiddles);
    }
}

#[target_feature(enable = "aes")]
pub(super) unsafe fn butterfly_fused_3layer_zero_root_from_src_row(
    src: *const F128,
    dst: *mut F128,
    eighth: usize,
    num_ntts: usize,
    r: usize,
    twiddles: &[F128; 7],
) {
    unsafe {
        butterfly_fused_3layer_from_src_row_impl::<true>(src, dst, eighth, num_ntts, r, twiddles);
    }
}

#[target_feature(enable = "aes")]
pub(super) unsafe fn butterfly_fused_3layer_row(
    ptr: *mut F128,
    eighth: usize,
    num_ntts: usize,
    lanes: usize,
    r: usize,
    twiddles: &[F128; 7],
) {
    use core::arch::aarch64::*;
    unsafe {
        let t: [uint64x2_t; 7] =
            core::array::from_fn(|i| vld1q_u64((&raw const twiddles[i]).cast::<u64>()));
        butterfly_fused_3layer_row_with_q(ptr, eighth, num_ntts, lanes, r, &t);
    }
}

#[target_feature(enable = "aes")]
pub(super) unsafe fn butterfly_fused_3layer_rows(
    ptr: *mut F128,
    eighth: usize,
    num_ntts: usize,
    row_start: usize,
    row_end: usize,
    twiddles: &[F128; 7],
) {
    use core::arch::aarch64::*;
    unsafe {
        let t: [uint64x2_t; 7] =
            core::array::from_fn(|i| vld1q_u64((&raw const twiddles[i]).cast::<u64>()));
        for r in row_start..row_end {
            butterfly_fused_3layer_row_with_q(ptr, eighth, num_ntts, num_ntts, r, &t);
        }
    }
}

#[target_feature(enable = "aes")]
unsafe fn butterfly_fused_3layer_zero_root_row_with_q(
    ptr: *mut F128,
    eighth: usize,
    num_ntts: usize,
    lanes: usize,
    r: usize,
    t2: core::arch::aarch64::uint64x2_t,
    t4: core::arch::aarch64::uint64x2_t,
    t5: core::arch::aarch64::uint64x2_t,
    t6: core::arch::aarch64::uint64x2_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        let zero = vdupq_n_u64(0);
        let c87 = vdupq_n_u64(0x87);
        let t2_swap = vextq_u64::<1>(t2, t2);
        let t4_swap = vextq_u64::<1>(t4, t4);
        let t5_swap = vextq_u64::<1>(t5, t5);
        let t6_swap = vextq_u64::<1>(t6, t6);
        let step = eighth * num_ntts;
        let base_r = ptr.add(r * num_ntts);

        let mut lane = 0;
        while lane + 1 < lanes {
            let base0 = base_r.add(lane);
            let base1 = base_r.add(lane + 1);

            let mut v0: [uint64x2_t; 8] =
                core::array::from_fn(|i| vld1q_u64(base0.add(i * step).cast::<u64>()));
            let mut v1: [uint64x2_t; 8] =
                core::array::from_fn(|i| vld1q_u64(base1.add(i * step).cast::<u64>()));

            // Layer L, t[0] = 0: four XOR-only butterflies.
            for i in 0..4 {
                v0[i + 4] = butterfly_zero_q(v0[i], v0[i + 4]);
                v1[i + 4] = butterfly_zero_q(v1[i], v1[i + 4]);
            }
            // Layer L+1: top half t[1] = 0, bottom half t[2] general.
            for i in 0..2 {
                v0[i + 2] = butterfly_zero_q(v0[i], v0[i + 2]);
                v1[i + 2] = butterfly_zero_q(v1[i], v1[i + 2]);
            }
            let (a0, b0) = butterfly_q_prepared(v0[4], v0[6], t2, t2_swap, c87, zero);
            let (a1, b1) = butterfly_q_prepared(v1[4], v1[6], t2, t2_swap, c87, zero);
            v0[4] = a0;
            v0[6] = b0;
            v1[4] = a1;
            v1[6] = b1;

            let (a0, b0) = butterfly_q_prepared(v0[5], v0[7], t2, t2_swap, c87, zero);
            let (a1, b1) = butterfly_q_prepared(v1[5], v1[7], t2, t2_swap, c87, zero);
            v0[5] = a0;
            v0[7] = b0;
            v1[5] = a1;
            v1[7] = b1;

            // Layer L+2: first quarter t[3] = 0.
            v0[1] = butterfly_zero_q(v0[0], v0[1]);
            v1[1] = butterfly_zero_q(v1[0], v1[1]);

            let (a0, b0) = butterfly_q_prepared(v0[2], v0[3], t4, t4_swap, c87, zero);
            let (a1, b1) = butterfly_q_prepared(v1[2], v1[3], t4, t4_swap, c87, zero);
            v0[2] = a0;
            v0[3] = b0;
            v1[2] = a1;
            v1[3] = b1;

            let (a0, b0) = butterfly_q_prepared(v0[4], v0[5], t5, t5_swap, c87, zero);
            let (a1, b1) = butterfly_q_prepared(v1[4], v1[5], t5, t5_swap, c87, zero);
            v0[4] = a0;
            v0[5] = b0;
            v1[4] = a1;
            v1[5] = b1;

            let (a0, b0) = butterfly_q_prepared(v0[6], v0[7], t6, t6_swap, c87, zero);
            let (a1, b1) = butterfly_q_prepared(v1[6], v1[7], t6, t6_swap, c87, zero);
            v0[6] = a0;
            v0[7] = b0;
            v1[6] = a1;
            v1[7] = b1;

            // v[0] is unchanged by all three layers; skip its store.
            for (i, value) in v0.iter().enumerate().skip(1) {
                vst1q_u64(base0.add(i * step).cast::<u64>(), *value);
            }
            for (i, value) in v1.iter().enumerate().skip(1) {
                vst1q_u64(base1.add(i * step).cast::<u64>(), *value);
            }
            lane += 2;
        }

        if lane < lanes {
            let base = base_r.add(lane);
            let mut v: [uint64x2_t; 8] =
                core::array::from_fn(|i| vld1q_u64(base.add(i * step).cast::<u64>()));

            // Layer L, t[0] = 0: four XOR-only butterflies.
            for i in 0..4 {
                v[i + 4] = butterfly_zero_q(v[i], v[i + 4]);
            }
            // Layer L+1: top half t[1] = 0, bottom half t[2] general.
            for i in 0..2 {
                v[i + 2] = butterfly_zero_q(v[i], v[i + 2]);
            }
            let (a, b) = butterfly_q_prepared(v[4], v[6], t2, t2_swap, c87, zero);
            v[4] = a;
            v[6] = b;
            let (a, b) = butterfly_q_prepared(v[5], v[7], t2, t2_swap, c87, zero);
            v[5] = a;
            v[7] = b;
            // Layer L+2: first quarter t[3] = 0.
            v[1] = butterfly_zero_q(v[0], v[1]);
            let (a, b) = butterfly_q_prepared(v[2], v[3], t4, t4_swap, c87, zero);
            v[2] = a;
            v[3] = b;
            let (a, b) = butterfly_q_prepared(v[4], v[5], t5, t5_swap, c87, zero);
            v[4] = a;
            v[5] = b;
            let (a, b) = butterfly_q_prepared(v[6], v[7], t6, t6_swap, c87, zero);
            v[6] = a;
            v[7] = b;

            // v[0] is unchanged by all three layers; skip its store.
            for (i, value) in v.iter().enumerate().skip(1) {
                vst1q_u64(base.add(i * step).cast::<u64>(), *value);
            }
        }
    }
}

#[target_feature(enable = "aes")]
pub(super) unsafe fn butterfly_fused_3layer_zero_root_row(
    ptr: *mut F128,
    eighth: usize,
    num_ntts: usize,
    lanes: usize,
    r: usize,
    twiddles: &[F128; 7],
) {
    use core::arch::aarch64::*;
    unsafe {
        debug_assert_eq!(twiddles[0], F128::ZERO);
        debug_assert_eq!(twiddles[1], F128::ZERO);
        debug_assert_eq!(twiddles[3], F128::ZERO);
        let t2 = vld1q_u64((&raw const twiddles[2]).cast::<u64>());
        let t4 = vld1q_u64((&raw const twiddles[4]).cast::<u64>());
        let t5 = vld1q_u64((&raw const twiddles[5]).cast::<u64>());
        let t6 = vld1q_u64((&raw const twiddles[6]).cast::<u64>());
        butterfly_fused_3layer_zero_root_row_with_q(
            ptr, eighth, num_ntts, lanes, r, t2, t4, t5, t6,
        );
    }
}

#[target_feature(enable = "aes")]
pub(super) unsafe fn butterfly_fused_3layer_zero_root_rows(
    ptr: *mut F128,
    eighth: usize,
    num_ntts: usize,
    row_start: usize,
    row_end: usize,
    twiddles: &[F128; 7],
) {
    use core::arch::aarch64::*;
    unsafe {
        debug_assert_eq!(twiddles[0], F128::ZERO);
        debug_assert_eq!(twiddles[1], F128::ZERO);
        debug_assert_eq!(twiddles[3], F128::ZERO);
        let t2 = vld1q_u64((&raw const twiddles[2]).cast::<u64>());
        let t4 = vld1q_u64((&raw const twiddles[4]).cast::<u64>());
        let t5 = vld1q_u64((&raw const twiddles[5]).cast::<u64>());
        let t6 = vld1q_u64((&raw const twiddles[6]).cast::<u64>());
        for r in row_start..row_end {
            butterfly_fused_3layer_zero_root_row_with_q(
                ptr, eighth, num_ntts, num_ntts, r, t2, t4, t5, t6,
            );
        }
    }
}

/// Two low-constant products with everything held in q registers: product
/// `i` is `v_i · t_i` where `t_i.hi == 0` and `ts_i` holds `t_i.lo` in both
/// lanes.
#[inline(always)]
unsafe fn mul_low_pair_q(
    v0: core::arch::aarch64::uint64x2_t,
    v1: core::arch::aarch64::uint64x2_t,
    ts0: core::arch::aarch64::uint64x2_t,
    ts1: core::arch::aarch64::uint64x2_t,
) -> (
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        let p0_ll = pmull_ll(v0, ts0);
        let p0_hl = pmull_hh(v0, ts0);
        let p1_ll = pmull_ll(v1, ts1);
        let p1_hl = pmull_hh(v1, ts1);

        let r0 = vzip1q_u64(p0_ll, p1_ll);
        let r1 = veorq_u64(vzip2q_u64(p0_ll, p1_ll), vzip1q_u64(p0_hl, p1_hl));
        let r2 = vzip2q_u64(p0_hl, p1_hl);

        let folded_lo = xor3(
            r2,
            vshlq_n_u64::<1>(r2),
            veorq_u64(vshlq_n_u64::<2>(r2), vshlq_n_u64::<7>(r2)),
        );
        let folded_hi = xor3(
            vshrq_n_u64::<63>(r2),
            vshrq_n_u64::<62>(r2),
            vshrq_n_u64::<57>(r2),
        );
        let out_lo = veorq_u64(r0, folded_lo);
        let out_hi = veorq_u64(r1, folded_hi);

        (vzip1q_u64(out_lo, out_hi), vzip2q_u64(out_lo, out_hi))
    }
}

/// Fused two-layer butterfly specialized for three low-limb-only twiddles.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "aes")]
#[inline]
pub(super) unsafe fn butterfly_fused_2layer_low_twiddles(
    a: &mut [F128],
    b: &mut [F128],
    c: &mut [F128],
    d: &mut [F128],
    t_outer: F128,
    t_inner_a: F128,
    t_inner_b: F128,
) {
    use core::arch::aarch64::*;

    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), c.len());
    debug_assert_eq!(a.len(), d.len());
    debug_assert_eq!(t_outer.hi, 0);
    debug_assert_eq!(t_inner_a.hi, 0);
    debug_assert_eq!(t_inner_b.hi, 0);

    unsafe {
        let to = vdupq_n_u64(t_outer.lo);
        let ta = vdupq_n_u64(t_inner_a.lo);
        let tb = vdupq_n_u64(t_inner_b.lo);
        for lane in 0..a.len() {
            let mut xa = vld1q_u64((&raw const a[lane]).cast::<u64>());
            let mut xb = vld1q_u64((&raw const b[lane]).cast::<u64>());
            let mut xc = vld1q_u64((&raw const c[lane]).cast::<u64>());
            let mut xd = vld1q_u64((&raw const d[lane]).cast::<u64>());

            // Layer L: (a,c) and (b,d) share t_outer.
            let (o0, o1) = mul_low_pair_q(xc, xd, to, to);
            xa = veorq_u64(xa, o0);
            xc = veorq_u64(xc, xa);
            xb = veorq_u64(xb, o1);
            xd = veorq_u64(xd, xb);

            // Layer L+1: (a,b) under t_inner_a, (c,d) under t_inner_b.
            let (i0, i1) = mul_low_pair_q(xb, xd, ta, tb);
            xa = veorq_u64(xa, i0);
            xb = veorq_u64(xb, xa);
            xc = veorq_u64(xc, i1);
            xd = veorq_u64(xd, xc);

            vst1q_u64((&raw mut a[lane]).cast::<u64>(), xa);
            vst1q_u64((&raw mut b[lane]).cast::<u64>(), xb);
            vst1q_u64((&raw mut c[lane]).cast::<u64>(), xc);
            vst1q_u64((&raw mut d[lane]).cast::<u64>(), xd);
        }
    }
}

/// Process two butterflies at a time within a block sharing one twiddle.
#[target_feature(enable = "aes")]
pub(super) unsafe fn butterfly_block(chunk: &mut [F128], twiddle: F128, half: usize) {
    use core::arch::aarch64::*;

    debug_assert!(half >= 2);
    debug_assert_eq!(chunk.len(), 2 * half);
    unsafe {
        let zero = vdupq_n_u64(0);
        let c87 = vdupq_n_u64(0x87);
        let t_q = vld1q_u64((&raw const twiddle).cast::<u64>());
        let t_swap = vextq_u64::<1>(t_q, t_q);

        let ptr = chunk.as_mut_ptr();
        let mut idx0 = 0;
        while idx0 + 1 < half {
            let idx1 = idx0 + half;
            let u_a = vld1q_u64(ptr.add(idx0).cast::<u64>());
            let v_a = vld1q_u64(ptr.add(idx1).cast::<u64>());
            let u_b = vld1q_u64(ptr.add(idx0 + 1).cast::<u64>());
            let v_b = vld1q_u64(ptr.add(idx1 + 1).cast::<u64>());

            let (new_u_a, new_v_a) = butterfly_q_prepared(u_a, v_a, t_q, t_swap, c87, zero);
            let (new_u_b, new_v_b) = butterfly_q_prepared(u_b, v_b, t_q, t_swap, c87, zero);

            vst1q_u64(ptr.add(idx0).cast::<u64>(), new_u_a);
            vst1q_u64(ptr.add(idx1).cast::<u64>(), new_v_a);
            vst1q_u64(ptr.add(idx0 + 1).cast::<u64>(), new_u_b);
            vst1q_u64(ptr.add(idx1 + 1).cast::<u64>(), new_v_b);

            idx0 += 2;
        }

        if idx0 < half {
            let idx1 = idx0 + half;
            let u = vld1q_u64(ptr.add(idx0).cast::<u64>());
            let v = vld1q_u64(ptr.add(idx1).cast::<u64>());
            let (new_u, new_v) = butterfly_q_prepared(u, v, t_q, t_swap, c87, zero);
            vst1q_u64(ptr.add(idx0).cast::<u64>(), new_u);
            vst1q_u64(ptr.add(idx1).cast::<u64>(), new_v);
        }
    }
}

/// Process the single pair in each of two adjacent blocks with distinct
/// twiddles.
#[target_feature(enable = "aes")]
pub(super) unsafe fn butterfly_block_pair(chunk: &mut [F128], t_a: F128, t_b: F128) {
    use core::arch::aarch64::*;

    debug_assert_eq!(chunk.len(), 4);
    unsafe {
        let zero = vdupq_n_u64(0);
        let c87 = vdupq_n_u64(0x87);
        let ta_q = vld1q_u64((&raw const t_a).cast::<u64>());
        let tb_q = vld1q_u64((&raw const t_b).cast::<u64>());
        let ta_swap = vextq_u64::<1>(ta_q, ta_q);
        let tb_swap = vextq_u64::<1>(tb_q, tb_q);

        let ptr = chunk.as_mut_ptr();
        let u_a = vld1q_u64(ptr.cast::<u64>());
        let v_a = vld1q_u64(ptr.add(1).cast::<u64>());
        let u_b = vld1q_u64(ptr.add(2).cast::<u64>());
        let v_b = vld1q_u64(ptr.add(3).cast::<u64>());

        let (new_u_a, new_v_a) = butterfly_q_prepared(u_a, v_a, ta_q, ta_swap, c87, zero);
        let (new_u_b, new_v_b) = butterfly_q_prepared(u_b, v_b, tb_q, tb_swap, c87, zero);

        vst1q_u64(ptr.cast::<u64>(), new_u_a);
        vst1q_u64(ptr.add(1).cast::<u64>(), new_v_a);
        vst1q_u64(ptr.add(2).cast::<u64>(), new_u_b);
        vst1q_u64(ptr.add(3).cast::<u64>(), new_v_b);
    }
}

/// Vector-resident twin of `portable::butterfly_fused_2layer`.
#[target_feature(enable = "aes")]
unsafe fn butterfly_fused_2layer_row_with_q(
    a: &mut [F128],
    b: &mut [F128],
    c: &mut [F128],
    d: &mut [F128],
    to: core::arch::aarch64::uint64x2_t,
    ta: core::arch::aarch64::uint64x2_t,
    tb: core::arch::aarch64::uint64x2_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        let zero = vdupq_n_u64(0);
        let c87 = vdupq_n_u64(0x87);
        let to_swap = vextq_u64::<1>(to, to);
        let ta_swap = vextq_u64::<1>(ta, ta);
        let tb_swap = vextq_u64::<1>(tb, tb);

        let len = a.len();
        let mut lane = 0;
        while lane + 1 < len {
            let mut xa0 = vld1q_u64((&raw const a[lane]).cast::<u64>());
            let mut xb0 = vld1q_u64((&raw const b[lane]).cast::<u64>());
            let mut xc0 = vld1q_u64((&raw const c[lane]).cast::<u64>());
            let mut xd0 = vld1q_u64((&raw const d[lane]).cast::<u64>());

            let mut xa1 = vld1q_u64((&raw const a[lane + 1]).cast::<u64>());
            let mut xb1 = vld1q_u64((&raw const b[lane + 1]).cast::<u64>());
            let mut xc1 = vld1q_u64((&raw const c[lane + 1]).cast::<u64>());
            let mut xd1 = vld1q_u64((&raw const d[lane + 1]).cast::<u64>());

            // Layer L: (a,c) and (b,d) share t_outer.
            let (na0, nc0) = butterfly_q_prepared(xa0, xc0, to, to_swap, c87, zero);
            let (nb0, nd0) = butterfly_q_prepared(xb0, xd0, to, to_swap, c87, zero);
            let (na1, nc1) = butterfly_q_prepared(xa1, xc1, to, to_swap, c87, zero);
            let (nb1, nd1) = butterfly_q_prepared(xb1, xd1, to, to_swap, c87, zero);

            // Layer L+1: (a,b) under t_inner_a, (c,d) under t_inner_b.
            let (fa0, fb0) = butterfly_q_prepared(na0, nb0, ta, ta_swap, c87, zero);
            let (fc0, fd0) = butterfly_q_prepared(nc0, nd0, tb, tb_swap, c87, zero);
            let (fa1, fb1) = butterfly_q_prepared(na1, nb1, ta, ta_swap, c87, zero);
            let (fc1, fd1) = butterfly_q_prepared(nc1, nd1, tb, tb_swap, c87, zero);

            vst1q_u64((&raw mut a[lane]).cast::<u64>(), fa0);
            vst1q_u64((&raw mut b[lane]).cast::<u64>(), fb0);
            vst1q_u64((&raw mut c[lane]).cast::<u64>(), fc0);
            vst1q_u64((&raw mut d[lane]).cast::<u64>(), fd0);

            vst1q_u64((&raw mut a[lane + 1]).cast::<u64>(), fa1);
            vst1q_u64((&raw mut b[lane + 1]).cast::<u64>(), fb1);
            vst1q_u64((&raw mut c[lane + 1]).cast::<u64>(), fc1);
            vst1q_u64((&raw mut d[lane + 1]).cast::<u64>(), fd1);

            lane += 2;
        }

        if lane < len {
            let mut xa = vld1q_u64((&raw const a[lane]).cast::<u64>());
            let mut xb = vld1q_u64((&raw const b[lane]).cast::<u64>());
            let mut xc = vld1q_u64((&raw const c[lane]).cast::<u64>());
            let mut xd = vld1q_u64((&raw const d[lane]).cast::<u64>());

            // Layer L: (a,c) and (b,d) share t_outer.
            let (na, nc) = butterfly_q_prepared(xa, xc, to, to_swap, c87, zero);
            let (nb, nd) = butterfly_q_prepared(xb, xd, to, to_swap, c87, zero);
            // Layer L+1: (a,b) under t_inner_a, (c,d) under t_inner_b.
            let (fa, fb) = butterfly_q_prepared(na, nb, ta, ta_swap, c87, zero);
            let (fc, fd) = butterfly_q_prepared(nc, nd, tb, tb_swap, c87, zero);

            vst1q_u64((&raw mut a[lane]).cast::<u64>(), fa);
            vst1q_u64((&raw mut b[lane]).cast::<u64>(), fb);
            vst1q_u64((&raw mut c[lane]).cast::<u64>(), fc);
            vst1q_u64((&raw mut d[lane]).cast::<u64>(), fd);
        }
    }
}

#[target_feature(enable = "aes")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn butterfly_fused_2layer(
    a: &mut [F128],
    b: &mut [F128],
    c: &mut [F128],
    d: &mut [F128],
    t_outer: F128,
    t_inner_a: F128,
    t_inner_b: F128,
) {
    use core::arch::aarch64::*;
    unsafe {
        let to = vld1q_u64((&raw const t_outer).cast::<u64>());
        let ta = vld1q_u64((&raw const t_inner_a).cast::<u64>());
        let tb = vld1q_u64((&raw const t_inner_b).cast::<u64>());
        butterfly_fused_2layer_row_with_q(a, b, c, d, to, ta, tb);
    }
}

/// Process every row group in one fused-two-layer block while keeping the
/// three twiddles in q registers across the block.
#[target_feature(enable = "aes")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn butterfly_fused_2layer_rows(
    block: &mut [F128],
    t_outer: F128,
    t_inner_a: F128,
    t_inner_b: F128,
    quarter: usize,
    num_ntts: usize,
    odd_tail: usize,
) {
    use core::arch::aarch64::*;
    unsafe {
        let to = vld1q_u64((&raw const t_outer).cast::<u64>());
        let ta = vld1q_u64((&raw const t_inner_a).cast::<u64>());
        let tb = vld1q_u64((&raw const t_inner_b).cast::<u64>());
        let stride = quarter * num_ntts;
        let (top_half, bot_half) = block.split_at_mut(2 * stride);
        let (q1, q2) = top_half.split_at_mut(stride);
        let (q3, q4) = bot_half.split_at_mut(stride);

        for (r, (((row_a, row_b), row_c), row_d)) in q1
            .chunks_exact_mut(num_ntts)
            .zip(q2.chunks_exact_mut(num_ntts))
            .zip(q3.chunks_exact_mut(num_ntts))
            .zip(q4.chunks_exact_mut(num_ntts))
            .enumerate()
        {
            let lanes = if r & 1 == 1 {
                num_ntts - odd_tail
            } else {
                num_ntts
            };
            butterfly_fused_2layer_row_with_q(
                &mut row_a[..lanes],
                &mut row_b[..lanes],
                &mut row_c[..lanes],
                &mut row_d[..lanes],
                to,
                ta,
                tb,
            );
        }
    }
}
