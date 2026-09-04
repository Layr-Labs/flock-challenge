use crate::field::F128;

// ---------------------------------------------------------------------------
// Vector-resident fused radix-8 rows.
//
// The portable fused-three-layer row kernel expresses its chain in terms of
// `F128` values: `values[u] + values[v] * twiddle`. Each `*` calls
// `ghash_mul_binius`, which repacks its NEON accumulator into
// `F128 { lo, hi }` on return, and each `+` is then a pair of scalar `u64`
// XORs. Because `F128` is a two-word struct, the twenty-one multiplies and
// forty-two XORs of one radix-8 row group cross the general-purpose/vector
// register boundary repeatedly. Disassembling the shipped kernel shows the
// cost directly: alongside its 72 PMULLs the loop carries 24 `fmov`, 12
// scalar `eor`, 11 `dup.2d` and 5 `mov.d` — roughly fifty instructions that
// exist only to shuttle values between register files.
//
// The kernels below keep all eight row values, and every intermediate, in
// `uint64x2_t` for the whole three-layer chain. Values enter and leave
// vector registers exactly once per row group; XOR becomes `veorq_u64` and
// the multiply never repacks. The arithmetic is the same GF(2^128) multiply
// and the same butterfly order, so output is bit-identical — `F128` is
// `#[repr(C, align(16))]` over `{ lo, hi }`, which is precisely the
// little-endian `uint64x2_t` lane order, making the loads and stores
// reinterpretations rather than conversions.
// ---------------------------------------------------------------------------

/// Carry-less multiply of two 64-bit lanes, result in a q register.
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

/// `pmull` of a lane against the fixed reduction constant `0x87`.
#[inline(always)]
unsafe fn pmull_87(x: u64) -> core::arch::aarch64::uint64x2_t {
    use core::arch::aarch64::*;
    unsafe { core::mem::transmute::<u128, uint64x2_t>(vmull_p64(x, 0x87)) }
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
///
/// Same algorithm as [`crate::field::gf2_128::aarch64::ghash_mul_binius`] —
/// four schoolbook PMULLs and the two-stage recursive reduction — but it
/// neither accepts nor returns an `F128`, so no lane ever moves to a
/// general-purpose register on the way in or out. `vmull_high_p64` reads the
/// high lanes in place, so the operand extraction the scalar form needs
/// disappears too.
#[inline(always)]
unsafe fn mul_q(
    a: core::arch::aarch64::uint64x2_t,
    b: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    use core::arch::aarch64::*;
    unsafe {
        let zero = vdupq_n_u64(0);

        let t0 = pmull_ll(a, b);
        let t1a = core::mem::transmute::<u128, uint64x2_t>(vmull_p64(
            vgetq_lane_u64::<0>(a),
            vgetq_lane_u64::<1>(b),
        ));
        let t1b = core::mem::transmute::<u128, uint64x2_t>(vmull_p64(
            vgetq_lane_u64::<1>(a),
            vgetq_lane_u64::<0>(b),
        ));
        let t2 = core::mem::transmute::<u128, uint64x2_t>(vmull_high_p64(
            vreinterpretq_p64_u64(a),
            vreinterpretq_p64_u64(b),
        ));
        let t1_cross = veorq_u64(t1a, t1b);

        // t1 += x^64 · t2 (mod p): {0, t2.lo} places t2.lo into t1.hi.
        let t1 = xor3(
            t1_cross,
            vextq_u64::<1>(zero, t2),
            pmull_87(vgetq_lane_u64::<1>(t2)),
        );

        // t0 += x^64 · t1 (mod p).
        xor3(
            t0,
            vextq_u64::<1>(zero, t1),
            pmull_87(vgetq_lane_u64::<1>(t1)),
        )
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
        for lane in 0..lanes {
            let base = ptr.add(r * num_ntts + lane);
            let step = eighth * num_ntts;

            let mut v: [uint64x2_t; 8] =
                core::array::from_fn(|i| vld1q_u64(base.add(i * step).cast::<u64>()));

            // Layer L: stride 4, shared twiddle t[0].
            for i in 0..4 {
                let (a, b) = butterfly_q(v[i], v[i + 4], t[0]);
                v[i] = a;
                v[i + 4] = b;
            }
            // Layer L+1: stride 2, twiddles t[1], t[2] per half.
            for s in 0..2 {
                for i in 0..2 {
                    let (u, w) = (4 * s + i, 4 * s + i + 2);
                    let (a, b) = butterfly_q(v[u], v[w], t[1 + s]);
                    v[u] = a;
                    v[w] = b;
                }
            }
            // Layer L+2: stride 1, twiddles t[3..7] per quarter.
            for s in 0..4 {
                let (a, b) = butterfly_q(v[2 * s], v[2 * s + 1], t[3 + s]);
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
/// The staging exists for the store microarchitecture, not the arithmetic:
/// per-lane scatter stores interleave sixteen 16 B streams spaced 64 MiB
/// apart, which defeats the streaming-store detector and pays a full RFO
/// read on ~1 GiB of destination lines (measured −4% end-to-end, note
/// `516b4da`). Sequential 1 KiB non-temporal bursts per row restore the
/// fill-like store behavior the replicate path enjoyed.
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
        use std::sync::OnceLock;
        static ON: OnceLock<bool> = OnceLock::new();
        // Diagnostics toggle while pricing the burst store flavor:
        // FLOCK_FROM_MSG_PLAIN_ST=1 uses ordinary vector stores.
        *ON.get_or_init(|| std::env::var_os("FLOCK_FROM_MSG_PLAIN_ST").is_none())
    }

    #[inline(always)]
    unsafe fn store_pair_nt(dst: *mut F128, x: uint64x2_t, y: uint64x2_t) {
        // Best-effort non-temporal pair hint; same idiom as the zerocheck
        // fold kernels' `stnp` arm.
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

        let z2 = vld1q_u64((&raw const t_zero[2]).cast::<u64>());
        let z4 = vld1q_u64((&raw const t_zero[4]).cast::<u64>());
        let z5 = vld1q_u64((&raw const t_zero[5]).cast::<u64>());
        let z6 = vld1q_u64((&raw const t_zero[6]).cast::<u64>());
        let g: [uint64x2_t; 7] =
            core::array::from_fn(|i| vld1q_u64((&raw const t_gen[i]).cast::<u64>()));

        // Two 8 KiB staging tiles (8 rows × ≤64 lanes × 16 B): L1-resident,
        // written per lane, read back once for the sequential row bursts.
        let mut stage0 = [F128 { lo: 0, hi: 0 }; 512];
        let mut stage1 = [F128 { lo: 0, hi: 0 }; 512];

        let off = r * num_ntts;
        let step = eighth * num_ntts;
        // The store flavor is process-constant. Resolve the diagnostic
        // killswitch once per row group rather than re-reading the OnceLock
        // for every destination pair in the burst loop below.
        let use_nt_bursts = nt_bursts();
        for lane in 0..num_ntts {
            let src_base = src.add(off + lane);
            let loaded: [uint64x2_t; 8] =
                core::array::from_fn(|i| vld1q_u64(src_base.add(i * step).cast::<u64>()));

            // Block 0: zero-root chain (mirrors
            // `butterfly_fused_3layer_zero_root_row`, but keeps v[0]).
            {
                let mut v = loaded;
                for i in 0..4 {
                    v[i + 4] = butterfly_zero_q(v[i], v[i + 4]);
                }
                for i in 0..2 {
                    v[i + 2] = butterfly_zero_q(v[i], v[i + 2]);
                }
                let (a, b) = butterfly_q(v[4], v[6], z2);
                v[4] = a;
                v[6] = b;
                let (a, b) = butterfly_q(v[5], v[7], z2);
                v[5] = a;
                v[7] = b;
                v[1] = butterfly_zero_q(v[0], v[1]);
                let (a, b) = butterfly_q(v[2], v[3], z4);
                v[2] = a;
                v[3] = b;
                let (a, b) = butterfly_q(v[4], v[5], z5);
                v[4] = a;
                v[5] = b;
                let (a, b) = butterfly_q(v[6], v[7], z6);
                v[6] = a;
                v[7] = b;
                for (i, value) in v.iter().enumerate() {
                    vst1q_u64(
                        stage0.as_mut_ptr().add(i * num_ntts + lane).cast::<u64>(),
                        *value,
                    );
                }
            }

            // Block 1: general chain (mirrors `butterfly_fused_3layer_row`).
            {
                let mut v = loaded;
                for i in 0..4 {
                    let (a, b) = butterfly_q(v[i], v[i + 4], g[0]);
                    v[i] = a;
                    v[i + 4] = b;
                }
                for s in 0..2 {
                    for i in 0..2 {
                        let (u, w) = (4 * s + i, 4 * s + i + 2);
                        let (a, b) = butterfly_q(v[u], v[w], g[1 + s]);
                        v[u] = a;
                        v[w] = b;
                    }
                }
                for s in 0..4 {
                    let (a, b) = butterfly_q(v[2 * s], v[2 * s + 1], g[3 + s]);
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

        // Emit each destination row's full lane run as one sequential
        // non-temporal burst (num_ntts/2 store-pairs of 32 B each).
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
/// Values are staged as one contiguous lane run per output row before the
/// non-temporal stores, avoiding a read-for-ownership of the stale codeword.
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
        let t: [uint64x2_t; 7] =
            core::array::from_fn(|i| vld1q_u64((&raw const twiddles[i]).cast::<u64>()));
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
                let (a, b) = butterfly_q(v[4], v[6], t[2]);
                v[4] = a;
                v[6] = b;
                let (a, b) = butterfly_q(v[5], v[7], t[2]);
                v[5] = a;
                v[7] = b;
                v[1] = butterfly_zero_q(v[0], v[1]);
                let (a, b) = butterfly_q(v[2], v[3], t[4]);
                v[2] = a;
                v[3] = b;
                let (a, b) = butterfly_q(v[4], v[5], t[5]);
                v[4] = a;
                v[5] = b;
                let (a, b) = butterfly_q(v[6], v[7], t[6]);
                v[6] = a;
                v[7] = b;
            } else {
                for i in 0..4 {
                    let (a, b) = butterfly_q(v[i], v[i + 4], t[0]);
                    v[i] = a;
                    v[i + 4] = b;
                }
                for s in 0..2 {
                    for i in 0..2 {
                        let (u, w) = (4 * s + i, 4 * s + i + 2);
                        let (a, b) = butterfly_q(v[u], v[w], t[1 + s]);
                        v[u] = a;
                        v[w] = b;
                    }
                }
                for s in 0..4 {
                    let (a, b) = butterfly_q(v[2 * s], v[2 * s + 1], t[3 + s]);
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

        // Every destination row is a short contiguous burst (128 B for the
        // ranked recursive geometry), so the stale allocation is written
        // without an RFO read.
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

/// Vector-resident twin of `portable::butterfly_fused_3layer_row`.
///
/// # Safety
/// Same contract as the portable form: the caller guarantees every selected
/// row and lane is valid and that concurrent calls own disjoint row groups.
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

/// Process a contiguous tile of row groups while keeping the seven twiddles
/// in vector registers across the tile.
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

/// Vector-resident twin of `portable::butterfly_fused_3layer_zero_root_row`.
///
/// Twiddles 0, 1 and 3 are zero on the root spine, so those butterflies are
/// XOR-only and `v[0]` is provably unchanged — it is therefore not stored.
///
/// # Safety
/// As [`butterfly_fused_3layer_row`], and the caller additionally guarantees
/// `twiddles[0] == twiddles[1] == twiddles[3] == 0`.
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
        for lane in 0..lanes {
            let base = ptr.add(r * num_ntts + lane);
            let step = eighth * num_ntts;

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
            let (a, b) = butterfly_q(v[4], v[6], t2);
            v[4] = a;
            v[6] = b;
            let (a, b) = butterfly_q(v[5], v[7], t2);
            v[5] = a;
            v[7] = b;
            // Layer L+2: first quarter t[3] = 0.
            v[1] = butterfly_zero_q(v[0], v[1]);
            let (a, b) = butterfly_q(v[2], v[3], t4);
            v[2] = a;
            v[3] = b;
            let (a, b) = butterfly_q(v[4], v[5], t5);
            v[4] = a;
            v[5] = b;
            let (a, b) = butterfly_q(v[6], v[7], t6);
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
/// lanes. Same unreduced-word layout and shift-fold reduction as
/// [`crate::field::gf2_128::aarch64::ghash_mul_low_constants_vec2_neon`]
/// (`r3 = 0`, single fold, no overflow correction), but the operands arrive
/// as `uint64x2_t` and the products leave as `uint64x2_t`, so no lane ever
/// visits a general-purpose register.
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
        // With t.hi == 0 each product needs only ll = v.lo·t.lo and
        // hl = v.hi·t.lo; `vmull_high_p64` reads v.hi in place.
        let p0_ll = pmull_ll(v0, ts0);
        let p0_hl = core::mem::transmute::<u128, uint64x2_t>(vmull_high_p64(
            vreinterpretq_p64_u64(v0),
            vreinterpretq_p64_u64(ts0),
        ));
        let p1_ll = pmull_ll(v1, ts1);
        let p1_hl = core::mem::transmute::<u128, uint64x2_t>(vmull_high_p64(
            vreinterpretq_p64_u64(v1),
            vreinterpretq_p64_u64(ts1),
        ));

        // Lane-paired unreduced words: r0 = ll.lo, r1 = ll.hi ^ hl.lo,
        // r2 = hl.hi, r3 = 0.
        let r0 = vzip1q_u64(p0_ll, p1_ll);
        let r1 = veorq_u64(vzip2q_u64(p0_ll, p1_ll), vzip1q_u64(p0_hl, p1_hl));
        let r2 = vzip2q_u64(p0_hl, p1_hl);

        // Fold r2·x^128 with x^128 = x^7 + x^2 + x + 1. Since r3 is zero,
        // there is no second overflow correction.
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

        // Unzip back to per-product `{ lo, hi }` lane order.
        (vzip1q_u64(out_lo, out_hi), vzip2q_u64(out_lo, out_hi))
    }
}

/// Fused two-layer butterfly specialized for three low-limb-only twiddles.
/// Two products are issued together at each stage, using four PMULLs instead
/// of twelve for the pair under the generic Binius field multiplier, and —
/// like [`butterfly_fused_2layer`] below — the four row values and every
/// intermediate stay in `uint64x2_t` across both layers, so the butterfly
/// XORs are `veorq_u64` rather than scalar `u64` pairs. Same multiplies,
/// same butterfly order, bit-identical output.
///
/// # Safety
/// Requires the `aes` target feature.
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

    // SAFETY: this function carries aes and all constants have zero high
    // limbs by the ranked gate and assertions above.
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

/// Vector-resident twin of `portable::butterfly_fused_2layer` — the deep
/// fused-pair chain held in q registers end to end.
///
/// The four generic deep pairs of the ranked L0 transform (layers 10..18)
/// run the portable `F128`-typed chain: four `ghash_mul_binius` calls per
/// lane, each repacking its NEON accumulator into `{ lo, hi }`, with every
/// butterfly `+` done as scalar `u64` XOR pairs — the same
/// register-file shuttling the fused-3-layer row kernels above eliminate
/// for the top passes. Here the four row values and every intermediate stay
/// in `uint64x2_t` across both layers; values enter and leave vector
/// registers exactly once per lane. Same multiplies, same butterfly order,
/// bit-identical output.
///
/// # Safety
/// Same contract as the portable form; caller guarantees equal slice
/// lengths (asserted upstream) and the `aes` target feature.
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
        for lane in 0..a.len() {
            let xa = vld1q_u64((&raw const a[lane]).cast::<u64>());
            let xb = vld1q_u64((&raw const b[lane]).cast::<u64>());
            let xc = vld1q_u64((&raw const c[lane]).cast::<u64>());
            let xd = vld1q_u64((&raw const d[lane]).cast::<u64>());

            // Layer L: (a,c) and (b,d) share t_outer.
            let (na, nc) = butterfly_q(xa, xc, to);
            let (nb, nd) = butterfly_q(xb, xd, to);
            // Layer L+1: (a,b) under t_inner_a, (c,d) under t_inner_b.
            let (fa, fb) = butterfly_q(na, nb, ta);
            let (fc, fd) = butterfly_q(nc, nd, tb);

            vst1q_u64((&raw mut a[lane]).cast::<u64>(), fa);
            vst1q_u64((&raw mut b[lane]).cast::<u64>(), fb);
            vst1q_u64((&raw mut c[lane]).cast::<u64>(), fc);
            vst1q_u64((&raw mut d[lane]).cast::<u64>(), fd);
        }
    }
}

#[target_feature(enable = "aes")]
#[allow(clippy::too_many_arguments)]
/// # Safety
/// Same contract as the portable form; the caller guarantees equal slice
/// lengths and the `aes` target feature.
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
///
/// # Safety
/// The caller guarantees the asserted four-quarter geometry, that any odd
/// tail is valid for each row group, and the `aes` target feature.
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
