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

/// `acc ^ (a.lo * b.lo)` as one macro-fused `pmull` + `eor` SIMD issue slot on Apple P-cores.
#[inline(always)]
unsafe fn fused_pmull_lo_acc(
    acc: core::arch::aarch64::uint64x2_t,
    a: core::arch::aarch64::uint64x2_t,
    b: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    let out: core::arch::aarch64::uint64x2_t;
    unsafe {
        core::arch::asm!(
            "pmull {o:v}.1q, {a:v}.1d, {b:v}.1d",
            "eor {o:v}.16b, {o:v}.16b, {c:v}.16b",
            a = in(vreg) a,
            b = in(vreg) b,
            c = in(vreg) acc,
            o = out(vreg) out,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    out
}

/// `acc ^ (a.hi * b.hi)` as one macro-fused `pmull2` + `eor` SIMD issue slot on Apple P-cores.
#[inline(always)]
unsafe fn fused_pmull_hi_acc(
    acc: core::arch::aarch64::uint64x2_t,
    a: core::arch::aarch64::uint64x2_t,
    b: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    let out: core::arch::aarch64::uint64x2_t;
    unsafe {
        core::arch::asm!(
            "pmull2 {o:v}.1q, {a:v}.2d, {b:v}.2d",
            "eor {o:v}.16b, {o:v}.16b, {c:v}.16b",
            a = in(vreg) a,
            b = in(vreg) b,
            c = in(vreg) acc,
            o = out(vreg) out,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    out
}

/// `(x.lo * y.lo) ^ (x.hi * y.hi)` as `pmull` + macro-fused `pmull2`+`eor` (two issue slots).
#[inline(always)]
unsafe fn cross_sum_q(
    x: core::arch::aarch64::uint64x2_t,
    y: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    let out: core::arch::aarch64::uint64x2_t;
    unsafe {
        core::arch::asm!(
            "pmull {p:v}.1q, {x:v}.1d, {y:v}.1d",
            "pmull2 {o:v}.1q, {x:v}.2d, {y:v}.2d",
            "eor {o:v}.16b, {o:v}.16b, {p:v}.16b",
            x = in(vreg) x,
            y = in(vreg) y,
            p = out(vreg) _,
            o = out(vreg) out,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    out
}

/// `x^64 · m (mod p)` added into `l`: `{0, m.lo}` places `m.lo` into the
/// high limb and `m.hi · x^128 = m.hi · (x^7 + x^2 + x + 1)`, a product of
/// at most 71 bits, lands in the low limbs. The map `m ↦ x^64 · m` is linear,
/// so a sum of pending halves can be folded once.
///
/// The three terms `l`, `{0, m.lo}` and `m.hi · 0x87` are summed by one
/// three-input `eor3` ([`xor3`]): `ext`, `pmull2` ([`pmull_hi_q`]), `eor3`.
#[cfg(target_feature = "sha3")]
#[inline(always)]
unsafe fn fold_x64_into(
    l: core::arch::aarch64::uint64x2_t,
    m: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    use core::arch::aarch64::*;
    unsafe { xor3(l, vextq_u64::<1>(vdupq_n_u64(0), m), pmull_hi_q(m, vdupq_n_u64(0x87))) }
}

/// `a.hi * b.hi` as one `pmull2`, with `b` taken as a vector operand (so a
/// splatted constant stays one hoisted register).
#[cfg(target_feature = "sha3")]
#[inline(always)]
unsafe fn pmull_hi_q(
    a: core::arch::aarch64::uint64x2_t,
    b: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    let out: core::arch::aarch64::uint64x2_t;
    unsafe {
        core::arch::asm!(
            "pmull2 {o:v}.1q, {a:v}.2d, {b:v}.2d",
            a = in(vreg) a,
            b = in(vreg) b,
            o = lateout(vreg) out,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    out
}

/// [`fold_x64_into`] without the `sha3` extension: `ext` + `eor`, then the
/// `pmull2` + `eor` pair.
#[cfg(not(target_feature = "sha3"))]
#[inline(always)]
unsafe fn fold_x64_into(
    l: core::arch::aarch64::uint64x2_t,
    m: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    use core::arch::aarch64::*;
    unsafe {
        let s = veorq_u64(l, vextq_u64::<1>(vdupq_n_u64(0), m));
        fused_pmull_hi_acc(s, m, vdupq_n_u64(0x87))
    }
}

/// Eager butterfly with zero pending half: `u' = u + v·t`, `v' = v + u'`.
#[inline(always)]
unsafe fn butterfly_fold_zero_q(
    u: core::arch::aarch64::uint64x2_t,
    v: core::arch::aarch64::uint64x2_t,
    t: core::arch::aarch64::uint64x2_t,
) -> (
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        let t_sw = vextq_u64::<1>(t, t);
        let u_lo = fused_pmull_lo_acc(u, v, t);
        let cross = cross_sum_q(v, t_sw);
        let t2 = core::mem::transmute::<u128, uint64x2_t>(vmull_high_p64(
            vreinterpretq_p64_u64(v),
            vreinterpretq_p64_u64(t),
        ));
        let mid = fold_x64_into(cross, t2);
        let new_u = fold_x64_into(u_lo, mid);
        (new_u, veorq_u64(v, new_u))
    }
}

/// Lazy butterfly with zero pending half.
#[inline(always)]
unsafe fn butterfly_lazy_zero_q(
    u: core::arch::aarch64::uint64x2_t,
    v: core::arch::aarch64::uint64x2_t,
    t: core::arch::aarch64::uint64x2_t,
) -> (
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        let t_sw = vextq_u64::<1>(t, t);
        let new_u = fused_pmull_lo_acc(u, v, t);
        let cross = cross_sum_q(v, t_sw);
        let t2 = core::mem::transmute::<u128, uint64x2_t>(vmull_high_p64(
            vreinterpretq_p64_u64(v),
            vreinterpretq_p64_u64(t),
        ));
        let mid = fold_x64_into(cross, t2);
        (new_u, veorq_u64(v, new_u), mid)
    }
}

/// Eager butterfly built on the split multiply: `u' = u + v·t` with the
/// product's x^64 half folded at once (plus any `pending` half `u` already
/// carries), `v' = v + u'`. `v` must have no pending half.
#[inline(always)]
unsafe fn butterfly_fold_q(
    u: core::arch::aarch64::uint64x2_t,
    v: core::arch::aarch64::uint64x2_t,
    t: core::arch::aarch64::uint64x2_t,
    pending: core::arch::aarch64::uint64x2_t,
) -> (
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        let t_sw = vextq_u64::<1>(t, t);
        let u_lo = fused_pmull_lo_acc(u, v, t);
        let cross = fused_pmull_hi_acc(fused_pmull_lo_acc(pending, v, t_sw), v, t_sw);
        let t2 = core::mem::transmute::<u128, uint64x2_t>(vmull_high_p64(
            vreinterpretq_p64_u64(v),
            vreinterpretq_p64_u64(t),
        ));
        let mid = fold_x64_into(cross, t2);
        let new_u = fold_x64_into(u_lo, mid);
        (new_u, veorq_u64(v, new_u))
    }
}

/// Lazy butterfly: as [`butterfly_fold_q`], but the summed x^64 half is
/// returned unfolded instead. Both outputs then carry that same pending
/// half (`v' = v + u'`), so the caller must fold it into each before the
/// row is a multiplicand or is stored.
#[inline(always)]
unsafe fn butterfly_lazy_q(
    u: core::arch::aarch64::uint64x2_t,
    v: core::arch::aarch64::uint64x2_t,
    t: core::arch::aarch64::uint64x2_t,
    pending: core::arch::aarch64::uint64x2_t,
) -> (
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        let t_sw = vextq_u64::<1>(t, t);
        let new_u = fused_pmull_lo_acc(u, v, t);
        let cross = fused_pmull_hi_acc(fused_pmull_lo_acc(pending, v, t_sw), v, t_sw);
        let t2 = core::mem::transmute::<u128, uint64x2_t>(vmull_high_p64(
            vreinterpretq_p64_u64(v),
            vreinterpretq_p64_u64(t),
        ));
        let mid = fold_x64_into(cross, t2);
        (new_u, veorq_u64(v, new_u), mid)
    }
}

/// General radix-8 chain (three layers, twiddles `t[0]`, `t[1..3]`,
/// `t[3..7]`) with lazy x^64 folds: 8 folds for the 12 products instead of
/// 12. Layer L leaves rows 0 and 1 unfolded (their halves `p0`, `p1` ride
/// into rows 4 and 5); layer L+1 leaves rows 0 and 4 unfolded (into rows 2
/// and 6) and folds rows 1 and 5 together with their layer-L halves;
/// layer L+2 folds every row once. Every multiplicand (rows 4..8 at L,
/// 2, 3, 6, 7 at L+1, odd rows at L+2) and every output is folded.
#[inline(always)]
unsafe fn fused3_general_q(
    v: &mut [core::arch::aarch64::uint64x2_t; 8],
    t: &[core::arch::aarch64::uint64x2_t; 7],
) {
    unsafe {
        // Layer L: stride 4, shared twiddle t[0].
        let (a, b, p0) = butterfly_lazy_zero_q(v[0], v[4], t[0]);
        v[0] = a;
        v[4] = b;
        let (a, b, p1) = butterfly_lazy_zero_q(v[1], v[5], t[0]);
        v[1] = a;
        v[5] = b;
        for i in 2..4 {
            let (a, b) = butterfly_fold_zero_q(v[i], v[i + 4], t[0]);
            v[i] = a;
            v[i + 4] = b;
        }
        // Layer L+1: stride 2, twiddles t[1], t[2] per half. Rows 0/4 carry
        // p0 and rows 1/5 carry p1; rows 2, 3, 6, 7 are folded.
        let (a, b, q0) = butterfly_lazy_q(v[0], v[2], t[1], p0);
        v[0] = a;
        v[2] = b;
        let (a, b) = butterfly_fold_q(v[1], v[3], t[1], p1);
        v[1] = a;
        v[3] = b;
        let (a, b, q4) = butterfly_lazy_q(v[4], v[6], t[2], p0);
        v[4] = a;
        v[6] = b;
        let (a, b) = butterfly_fold_q(v[5], v[7], t[2], p1);
        v[5] = a;
        v[7] = b;
        // Layer L+2: stride 1, twiddles t[3..7] per quarter. Rows 0/2 carry
        // q0 and rows 4/6 carry q4; odd rows are folded.
        for (s, pending) in [q0, q0, q4, q4].into_iter().enumerate() {
            let (a, b) = butterfly_fold_q(v[2 * s], v[2 * s + 1], t[3 + s], pending);
            v[2 * s] = a;
            v[2 * s + 1] = b;
        }
    }
}

/// Zero-root radix-8 chain (twiddles 0, 1 and 3 are zero) with lazy x^64
/// folds: 4 folds for the 5 products instead of 5. Row 4's layer-L+1 half
/// rides into row 6 and is folded with each row's layer-L+2 product.
/// `v[0]` is unchanged.
#[inline(always)]
unsafe fn fused3_zero_root_q(
    v: &mut [core::arch::aarch64::uint64x2_t; 8],
    t2: core::arch::aarch64::uint64x2_t,
    t4: core::arch::aarch64::uint64x2_t,
    t5: core::arch::aarch64::uint64x2_t,
    t6: core::arch::aarch64::uint64x2_t,
) {
    unsafe {
        // Layer L, t[0] = 0: four XOR-only butterflies.
        for i in 0..4 {
            v[i + 4] = butterfly_zero_q(v[i], v[i + 4]);
        }
        // Layer L+1: top half t[1] = 0, bottom half t[2] general.
        for i in 0..2 {
            v[i + 2] = butterfly_zero_q(v[i], v[i + 2]);
        }
        let (a, b, p4) = butterfly_lazy_zero_q(v[4], v[6], t2);
        v[4] = a;
        v[6] = b;
        let (a, b) = butterfly_fold_zero_q(v[5], v[7], t2);
        v[5] = a;
        v[7] = b;
        // Layer L+2: first quarter t[3] = 0; rows 4/6 carry p4.
        v[1] = butterfly_zero_q(v[0], v[1]);
        let (a, b) = butterfly_fold_zero_q(v[2], v[3], t4);
        v[2] = a;
        v[3] = b;
        let (a, b) = butterfly_fold_q(v[4], v[5], t5, p4);
        v[4] = a;
        v[5] = b;
        let (a, b) = butterfly_fold_q(v[6], v[7], t6, p4);
        v[6] = a;
        v[7] = b;
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

            fused3_general_q(&mut v, t);

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
        let ext = use_const_ext();
        let (pz2, pz4, pz5, pz6) = (
            three_coeffs(z2),
            three_coeffs(z4),
            three_coeffs(z5),
            three_coeffs(z6),
        );
        let pg: [Prep; 7] = core::array::from_fn(|i| three_coeffs(g[i]));

        // Two 8 KiB staging tiles (8 rows × ≤64 lanes × 16 B): L1-resident,
        // written per lane, read back once for the sequential row bursts.
        let mut stage0 = [core::mem::MaybeUninit::<F128>::uninit(); 512];
        let mut stage1 = [core::mem::MaybeUninit::<F128>::uninit(); 512];

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
                if ext {
                    fused3_zero_root_prepared_q(&mut v, pz2, pz4, pz5, pz6);
                } else {
                    fused3_zero_root_q(&mut v, z2, z4, z5, z6);
                }
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
                if ext {
                    fused3_prepared_q(&mut v, &pg);
                } else {
                    fused3_general_q(&mut v, &g);
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

/// Out-of-place radix-8 tile used by recursive from-message commitments.
/// Values are staged as one contiguous lane run per output row before the
/// non-temporal stores, avoiding a read-for-ownership of the stale codeword.
/// Twiddles and the 1 KiB stage are retained across the tile's row groups.
///
/// # Safety
/// Source and destination must be disjoint allocations that each cover
/// `8 * eighth * num_ntts` elements. The source must be initialized,
/// `num_ntts == 8`, and `row_start <= row_end <= eighth`. The caller must
/// own every selected destination row until this call returns. For the
/// zero-root form, twiddles 0, 1 and 3 must be zero.
#[target_feature(enable = "aes")]
unsafe fn butterfly_fused_3layer_from_src_rows_impl<const ZERO_ROOT: bool>(
    src: *const F128,
    dst: *mut F128,
    eighth: usize,
    num_ntts: usize,
    row_start: usize,
    row_end: usize,
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
        // The caller guarantees eight lanes; keep the width constant in release builds.
        let num_ntts = 8;
        debug_assert!(row_start <= row_end && row_end <= eighth);
        if ZERO_ROOT {
            debug_assert_eq!(twiddles[0], F128::ZERO);
            debug_assert_eq!(twiddles[1], F128::ZERO);
            debug_assert_eq!(twiddles[3], F128::ZERO);
        }
        let t: [uint64x2_t; 7] =
            core::array::from_fn(|i| vld1q_u64((&raw const twiddles[i]).cast::<u64>()));
        let ext = use_const_ext();
        let prep: [Prep; 7] = core::array::from_fn(|i| three_coeffs(t[i]));
        let mut stage = [core::mem::MaybeUninit::<F128>::uninit(); 64];
        let step = eighth * num_ntts;

        for r in row_start..row_end {
            let off = r * num_ntts;
            for lane in 0..num_ntts {
                let src_base = src.add(off + lane);
                let mut v: [uint64x2_t; 8] =
                    core::array::from_fn(|i| vld1q_u64(src_base.add(i * step).cast::<u64>()));

                if ext {
                    if ZERO_ROOT {
                        fused3_zero_root_prepared_q(&mut v, prep[2], prep[4], prep[5], prep[6]);
                    } else {
                        fused3_prepared_q(&mut v, &prep);
                    }
                } else if ZERO_ROOT {
                    fused3_zero_root_q(&mut v, t[2], t[4], t[5], t[6]);
                } else {
                    fused3_general_q(&mut v, &t);
                }

                for (i, value) in v.iter().enumerate() {
                    vst1q_u64(
                        stage.as_mut_ptr().add(i * num_ntts + lane).cast::<u64>(),
                        *value,
                    );
                }
            }

            // All 64 stage values are initialized above before this drain.
            // Drain each 128 B row before the next group reuses the stage.
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
}

#[cfg(test)]
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
        butterfly_fused_3layer_from_src_rows_impl::<false>(
            src,
            dst,
            eighth,
            num_ntts,
            r,
            r + 1,
            twiddles,
        );
    }
}

#[cfg(test)]
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
        butterfly_fused_3layer_from_src_rows_impl::<true>(
            src,
            dst,
            eighth,
            num_ntts,
            r,
            r + 1,
            twiddles,
        );
    }
}

#[target_feature(enable = "aes")]
pub(super) unsafe fn butterfly_fused_3layer_from_src_rows(
    src: *const F128,
    dst: *mut F128,
    eighth: usize,
    num_ntts: usize,
    row_start: usize,
    row_end: usize,
    twiddles: &[F128; 7],
) {
    unsafe {
        butterfly_fused_3layer_from_src_rows_impl::<false>(
            src, dst, eighth, num_ntts, row_start, row_end, twiddles,
        );
    }
}

#[target_feature(enable = "aes")]
pub(super) unsafe fn butterfly_fused_3layer_zero_root_from_src_rows(
    src: *const F128,
    dst: *mut F128,
    eighth: usize,
    num_ntts: usize,
    row_start: usize,
    row_end: usize,
    twiddles: &[F128; 7],
) {
    unsafe {
        butterfly_fused_3layer_from_src_rows_impl::<true>(
            src, dst, eighth, num_ntts, row_start, row_end, twiddles,
        );
    }
}

// Fixed-twiddle coefficients shared across the generic radix-8 row.
#[inline(always)]
unsafe fn three_coeffs(
    t: core::arch::aarch64::uint64x2_t,
) -> (
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
) {
    unsafe {
        use core::arch::aarch64::*;
        let r = core::mem::transmute::<u128, uint64x2_t>(vmull_high_p64(
            vreinterpretq_p64_u64(t),
            vreinterpretq_p64_u64(vdupq_n_u64(0x87)),
        ));
        let lo = vzip1q_u64(t, r);
        let mid = veorq_u64(vextq_u64::<1>(t, t), vzip2q_u64(vdupq_n_u64(0), r));
        (lo, mid)
    }
}
#[inline(always)]
unsafe fn three_lazy(
    u: core::arch::aarch64::uint64x2_t,
    v: core::arch::aarch64::uint64x2_t,
    t: (
        core::arch::aarch64::uint64x2_t,
        core::arch::aarch64::uint64x2_t,
    ),
    p: core::arch::aarch64::uint64x2_t,
) -> (
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
) {
    unsafe {
        use core::arch::aarch64::*;
        let nu = fused_pmull_hi_acc(fused_pmull_lo_acc(u, v, t.0), v, t.0);
        let mid = fused_pmull_hi_acc(fused_pmull_lo_acc(p, v, t.1), v, t.1);
        (nu, veorq_u64(v, nu), mid)
    }
}
#[inline(always)]
unsafe fn three_fold(
    u: core::arch::aarch64::uint64x2_t,
    v: core::arch::aarch64::uint64x2_t,
    t: (
        core::arch::aarch64::uint64x2_t,
        core::arch::aarch64::uint64x2_t,
    ),
    p: core::arch::aarch64::uint64x2_t,
) -> (
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
) {
    unsafe {
        use core::arch::aarch64::*;
        let (nu, _, mid) = three_lazy(u, v, t, p);
        let fu = fold_x64_into(nu, mid);
        (fu, veorq_u64(v, fu))
    }
}
/// [`three_lazy`] with a zero pending half: the x^64 half is the bare cross
/// sum `v.lo * t.1.lo + v.hi * t.1.hi` ([`cross_sum_q`]) instead of that sum
/// accumulated into a zero register. Same products, same values.
#[inline(always)]
unsafe fn three_lazy_zero(
    u: core::arch::aarch64::uint64x2_t,
    v: core::arch::aarch64::uint64x2_t,
    t: (
        core::arch::aarch64::uint64x2_t,
        core::arch::aarch64::uint64x2_t,
    ),
) -> (
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
) {
    unsafe {
        use core::arch::aarch64::*;
        let nu = fused_pmull_hi_acc(fused_pmull_lo_acc(u, v, t.0), v, t.0);
        let mid = cross_sum_q(v, t.1);
        (nu, veorq_u64(v, nu), mid)
    }
}
/// [`three_fold`] with a zero pending half (see [`three_lazy_zero`]).
#[inline(always)]
unsafe fn three_fold_zero(
    u: core::arch::aarch64::uint64x2_t,
    v: core::arch::aarch64::uint64x2_t,
    t: (
        core::arch::aarch64::uint64x2_t,
        core::arch::aarch64::uint64x2_t,
    ),
) -> (
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
) {
    unsafe {
        use core::arch::aarch64::*;
        let (nu, _, mid) = three_lazy_zero(u, v, t);
        let fu = fold_x64_into(nu, mid);
        (fu, veorq_u64(v, fu))
    }
}
#[inline(always)]
unsafe fn fused3_prepared_q(
    v: &mut [core::arch::aarch64::uint64x2_t; 8],
    t: &[(
        core::arch::aarch64::uint64x2_t,
        core::arch::aarch64::uint64x2_t,
    ); 7],
) {
    unsafe {
        use core::arch::aarch64::*;
        let (a, b, p0) = three_lazy_zero(v[0], v[4], t[0]);
        v[0] = a;
        v[4] = b;
        let (a, b, p1) = three_lazy_zero(v[1], v[5], t[0]);
        v[1] = a;
        v[5] = b;
        for i in 2..4 {
            let (a, b) = three_fold_zero(v[i], v[i + 4], t[0]);
            v[i] = a;
            v[i + 4] = b;
        }
        let (a, b, q0) = three_lazy(v[0], v[2], t[1], p0);
        v[0] = a;
        v[2] = b;
        let (a, b) = three_fold(v[1], v[3], t[1], p1);
        v[1] = a;
        v[3] = b;
        let (a, b, q4) = three_lazy(v[4], v[6], t[2], p0);
        v[4] = a;
        v[6] = b;
        let (a, b) = three_fold(v[5], v[7], t[2], p1);
        v[5] = a;
        v[7] = b;
        for (s, pending) in [q0, q0, q4, q4].into_iter().enumerate() {
            let (a, b) = three_fold(v[2 * s], v[2 * s + 1], t[3 + s], pending);
            v[2 * s] = a;
            v[2 * s + 1] = b;
        }
    }
}

#[inline]
fn use_three_const_fold() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_NTT_THREE_CONST_FOLD").is_none());
    static DEBUG: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NTT_THREE_CONST_FOLD_DEBUG").is_some());
    let enabled = *ON;
    if *DEBUG {
        static HITS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let hits = HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if hits.is_power_of_two() || hits.is_multiple_of(64) {
            eprintln!("[ntt-three-const-fold] enabled={enabled} groups={hits}");
        }
    }
    enabled
}
/// Extension of the fixed-twiddle (prepared) radix-8 arithmetic to the
/// kernels h10 left on the split-multiply path: zero-root rows (in place and
/// from source), the general from-source rows used by the recursive
/// from-message commits, the rate-1/2 dual first pass, and the single-layer
/// block. Same field values, canonical outputs: bit-identical.
/// `FLOCK_NO_NTT_CONST_EXT=1` restores the split-multiply kernels.
#[inline]
fn use_const_ext() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_NTT_CONST_EXT").is_none());
    *ON
}

type Prep = (
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
);

/// Prepared twin of [`fused3_zero_root_q`] (twiddles 0, 1, 3 zero).
#[inline(always)]
unsafe fn fused3_zero_root_prepared_q(
    v: &mut [core::arch::aarch64::uint64x2_t; 8],
    t2: Prep,
    t4: Prep,
    t5: Prep,
    t6: Prep,
) {
    unsafe {
        use core::arch::aarch64::*;
        for i in 0..4 {
            v[i + 4] = butterfly_zero_q(v[i], v[i + 4]);
        }
        for i in 0..2 {
            v[i + 2] = butterfly_zero_q(v[i], v[i + 2]);
        }
        let (a, b, p4) = three_lazy_zero(v[4], v[6], t2);
        v[4] = a;
        v[6] = b;
        let (a, b) = three_fold_zero(v[5], v[7], t2);
        v[5] = a;
        v[7] = b;
        v[1] = butterfly_zero_q(v[0], v[1]);
        let (a, b) = three_fold_zero(v[2], v[3], t4);
        v[2] = a;
        v[3] = b;
        let (a, b) = three_fold(v[4], v[5], t5, p4);
        v[4] = a;
        v[5] = b;
        let (a, b) = three_fold(v[6], v[7], t6, p4);
        v[6] = a;
        v[7] = b;
    }
}

#[target_feature(enable = "aes")]
#[inline]
unsafe fn butterfly_fused_3layer_row_with_const_q(
    ptr: *mut F128,
    eighth: usize,
    num_ntts: usize,
    lanes: usize,
    r: usize,
    t: &[(
        core::arch::aarch64::uint64x2_t,
        core::arch::aarch64::uint64x2_t,
    ); 7],
) {
    use core::arch::aarch64::*;
    unsafe {
        // Each of the eight rows is addressed from its end at one shared
        // negative byte offset, which is also the loop counter.
        let step = eighth * num_ntts;
        let ends: [*mut u8; 8] =
            core::array::from_fn(|i| ptr.add(r * num_ntts + i * step + lanes).cast::<u8>());
        let mut off = -((lanes * core::mem::size_of::<F128>()) as isize);
        while off != 0 {
            let mut v: [uint64x2_t; 8] =
                core::array::from_fn(|i| vld1q_u64(ends[i].offset(off).cast::<u64>()));
            fused3_prepared_q(&mut v, t);
            for (i, value) in v.into_iter().enumerate() {
                vst1q_u64(ends[i].offset(off).cast::<u64>(), value);
            }
            off = next_lane_off(off);
        }
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
        if use_three_const_fold() {
            let prepared = core::array::from_fn(|i| three_coeffs(t[i]));
            butterfly_fused_3layer_row_with_const_q(ptr, eighth, num_ntts, lanes, r, &prepared);
        } else {
            butterfly_fused_3layer_row_with_q(ptr, eighth, num_ntts, lanes, r, &t);
        }
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
        if use_three_const_fold() {
            let prepared = core::array::from_fn(|i| three_coeffs(t[i]));
            for r in row_start..row_end {
                butterfly_fused_3layer_row_with_const_q(
                    ptr, eighth, num_ntts, num_ntts, r, &prepared,
                );
            }
        } else {
            for r in row_start..row_end {
                butterfly_fused_3layer_row_with_q(ptr, eighth, num_ntts, num_ntts, r, &t);
            }
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

            if use_const_ext() {
                fused3_zero_root_prepared_q(
                    &mut v,
                    three_coeffs(t2),
                    three_coeffs(t4),
                    three_coeffs(t5),
                    three_coeffs(t6),
                );
            } else {
                fused3_zero_root_q(&mut v, t2, t4, t5, t6);
            }

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
unsafe fn mul_low_pair_q<const DIRECT: bool>(
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
        if DIRECT {
            // With t.hi == 0 the product is ll + x^64*hl. Keep each
            // product in its own vector and use the existing exact x^64
            // fold instead of packing, shift-folding and unpacking lanes.
            let lo0 = pmull_ll(v0, ts0);
            let hi0 = core::mem::transmute::<u128, uint64x2_t>(vmull_high_p64(
                vreinterpretq_p64_u64(v0),
                vreinterpretq_p64_u64(ts0),
            ));
            let lo1 = pmull_ll(v1, ts1);
            let hi1 = core::mem::transmute::<u128, uint64x2_t>(vmull_high_p64(
                vreinterpretq_p64_u64(v1),
                vreinterpretq_p64_u64(ts1),
            ));
            return (fold_x64_into(lo0, hi0), fold_x64_into(lo1, hi1));
        }

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
/// The default kernel shares a pending polynomial half across both layers.
/// Its eager direct-reduction and shift-reduction alternatives remain available.
/// Like [`butterfly_fused_2layer`] below, the four row values and every
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
    static DIRECT: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_NTT_LOW_PMULL").is_none());
    static LAZY: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_NTT_LOW_LAZY").is_none());
    static DEBUG: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NTT_LOW_PMULL_DEBUG").is_some());
    let direct = *DIRECT;
    let lazy = direct && *LAZY;
    if *DEBUG {
        static HITS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let hits = HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if hits.is_power_of_two() {
            eprintln!("[ntt-low-pmull] direct={direct} lazy={lazy} blocks={hits}");
        }
    }
    unsafe {
        if lazy {
            butterfly_fused_2layer_low_lazy(a, b, c, d, t_outer, t_inner_a, t_inner_b);
        } else if direct {
            butterfly_fused_2layer_low_impl::<true>(a, b, c, d, t_outer, t_inner_a, t_inner_b);
        } else {
            butterfly_fused_2layer_low_impl::<false>(a, b, c, d, t_outer, t_inner_a, t_inner_b);
        }
    }
}

#[target_feature(enable = "aes")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn butterfly_fused_2layer_low_impl<const DIRECT: bool>(
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
            let (o0, o1) = mul_low_pair_q::<DIRECT>(xc, xd, to, to);
            xa = veorq_u64(xa, o0);
            xc = veorq_u64(xc, xa);
            xb = veorq_u64(xb, o1);
            xd = veorq_u64(xd, xb);

            // Layer L+1: (a,b) under t_inner_a, (c,d) under t_inner_b.
            let (i0, i1) = mul_low_pair_q::<DIRECT>(xb, xd, ta, tb);
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

// Carry the x^64 half of the first outer product until the inner layer.
// The two descendants share that pending half. The other outer product
// must be reduced before it becomes an inner-layer multiplicand.
#[inline(always)]
unsafe fn butterfly_low_lazy_q(
    u: core::arch::aarch64::uint64x2_t,
    v: core::arch::aarch64::uint64x2_t,
    t: core::arch::aarch64::uint64x2_t,
    p: core::arch::aarch64::uint64x2_t,
) -> (
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
) {
    unsafe {
        use core::arch::aarch64::*;
        let hi = core::mem::transmute::<u128, uint64x2_t>(vmull_high_p64(
            vreinterpretq_p64_u64(v),
            vreinterpretq_p64_u64(t),
        ));
        let nu = fused_pmull_lo_acc(u, v, t);
        (nu, veorq_u64(v, nu), veorq_u64(hi, p))
    }
}

#[inline(always)]
unsafe fn butterfly_low_fold_q(
    u: core::arch::aarch64::uint64x2_t,
    v: core::arch::aarch64::uint64x2_t,
    t: core::arch::aarch64::uint64x2_t,
    p: core::arch::aarch64::uint64x2_t,
) -> (
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
) {
    unsafe {
        use core::arch::aarch64::*;
        let (nu, _, mid) = butterfly_low_lazy_q(u, v, t, p);
        let fu = fold_x64_into(nu, mid);
        (fu, veorq_u64(v, fu))
    }
}

#[target_feature(enable = "aes")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn butterfly_fused_2layer_low_lazy(
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
    debug_assert_eq!(t_outer.hi | t_inner_a.hi | t_inner_b.hi, 0);
    unsafe {
        use core::arch::aarch64::*;
        let to = vdupq_n_u64(t_outer.lo);
        let ta = vdupq_n_u64(t_inner_a.lo);
        let tb = vdupq_n_u64(t_inner_b.lo);
        let zero = vdupq_n_u64(0);
        let n = a.len();
        let [ea, eb, ec, ed] = row_ends([a, b, c, d]);
        let mut off = -((n * core::mem::size_of::<F128>()) as isize);
        while off != 0 {
            let xa = vld1q_u64(ea.offset(off).cast::<u64>());
            let xb = vld1q_u64(eb.offset(off).cast::<u64>());
            let xc = vld1q_u64(ec.offset(off).cast::<u64>());
            let xd = vld1q_u64(ed.offset(off).cast::<u64>());
            let (na, nc, p) = butterfly_low_lazy_q(xa, xc, to, zero);
            let (nb, nd) = butterfly_low_fold_q(xb, xd, to, zero);
            let (fa, fb) = butterfly_low_fold_q(na, nb, ta, p);
            let (fc, fd) = butterfly_low_fold_q(nc, nd, tb, p);
            vst1q_u64(ea.offset(off).cast::<u64>(), fa);
            vst1q_u64(eb.offset(off).cast::<u64>(), fb);
            vst1q_u64(ec.offset(off).cast::<u64>(), fc);
            vst1q_u64(ed.offset(off).cast::<u64>(), fd);
            off = next_lane_off(off);
        }
    }
}

/// `off + 16` (one `F128` lane), opaque to the optimizer: the lane loops then
/// keep `off` as their only induction variable, address every row at
/// `[end, off]` and test `off` itself for the exit (`add` + `cbnz` per lane).
#[inline(always)]
fn next_lane_off(off: isize) -> isize {
    const _: () = assert!(core::mem::size_of::<F128>() == 16);
    let mut o = off;
    // SAFETY: register-only arithmetic.
    unsafe {
        core::arch::asm!(
            "add {o}, {o}, #16",
            o = inout(reg) o,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    o
}

/// One-past-the-end byte pointers of four equal-length rows. The lane loops
/// address lane `l` of a row of `n` lanes at the negative byte offset
/// `16 (l - n)` from its end, so one register is both the shared address
/// offset and the loop counter (it reaches zero after the last lane).
///
/// Panics unless the four rows have equal length (the callers slice all four
/// to one lane count, so the check folds away in the lane kernels).
#[inline(always)]
fn row_ends(rows: [&mut [F128]; 4]) -> [*mut u8; 4] {
    let n = rows[0].len();
    assert!(rows.iter().all(|row| row.len() == n), "lane rows must have equal length");
    // SAFETY: one past the end of each row.
    rows.map(|row| unsafe { row.as_mut_ptr().add(n).cast::<u8>() })
}

/// Whether [`butterfly_fused_2layer_low_twiddles`] runs its lazy form
/// (`FLOCK_NO_NTT_LOW_PMULL` and `FLOCK_NO_NTT_LOW_LAZY` unset) with its
/// per-block debug counter off (`FLOCK_NTT_LOW_PMULL_DEBUG` unset). Read once
/// per process.
pub(super) fn low_lazy_blocks_selected() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("FLOCK_NO_NTT_LOW_PMULL").is_none()
            && std::env::var_os("FLOCK_NO_NTT_LOW_LAZY").is_none()
            && std::env::var_os("FLOCK_NTT_LOW_PMULL_DEBUG").is_none()
    })
}

/// Final fused pair (`quarter == 1`) over consecutive four-row blocks
/// `[a, b, c, d]` of `num_ntts` lanes each. `twiddles(i)` returns block `i`'s
/// `(t_outer, t_inner_a, t_inner_b)`, all with a zero high limb.
///
/// Rows b and d must be zero in the trailing `zero_tail` lanes. There layer L
/// gives `a1 = a + c·t_outer`, `c1 = c + a1` and leaves b and d at zero, so
/// layer L+1 gives `a2 = b2 = a1` and `c2 = d2 = c1`: one fully reduced
/// low-limb product per lane. The other lanes run the lazy lane body of
/// [`butterfly_fused_2layer_low_lazy`].
///
/// # Safety
/// Requires the `aes` target feature.
#[target_feature(enable = "aes")]
pub(super) unsafe fn butterfly_fused_2layer_low_lazy_zero_tail_blocks<F>(
    blocks: &mut [F128],
    num_ntts: usize,
    zero_tail: usize,
    twiddles: F,
) where
    F: Fn(usize) -> (F128, F128, F128),
{
    debug_assert!(zero_tail < num_ntts);
    debug_assert!(blocks.len().is_multiple_of(4 * num_ntts));
    let dense = num_ntts - zero_tail;
    unsafe {
        use core::arch::aarch64::*;
        let zero = vdupq_n_u64(0);
        for (i, block) in blocks.chunks_exact_mut(4 * num_ntts).enumerate() {
            let (t_outer, t_inner_a, t_inner_b) = twiddles(i);
            let (a, rest) = block.split_at_mut(num_ntts);
            let (b, rest) = rest.split_at_mut(num_ntts);
            let (c, d) = rest.split_at_mut(num_ntts);
            let (a, a_tail) = a.split_at_mut(dense);
            let (b, b_tail) = b.split_at_mut(dense);
            let (c, c_tail) = c.split_at_mut(dense);
            let (d, d_tail) = d.split_at_mut(dense);
            butterfly_fused_2layer_low_lazy(a, b, c, d, t_outer, t_inner_a, t_inner_b);

            let to = vdupq_n_u64(t_outer.lo);
            debug_assert!(
                b_tail.iter().chain(d_tail.iter()).all(|x| *x == F128::ZERO),
                "zero-tail final pair needs zero b and d rows in the tail lanes"
            );
            let [ea, eb, ec, ed] = row_ends([a_tail, b_tail, c_tail, d_tail]);
            let mut off = -((zero_tail * core::mem::size_of::<F128>()) as isize);
            while off != 0 {
                let xa = vld1q_u64(ea.offset(off).cast::<u64>());
                let xc = vld1q_u64(ec.offset(off).cast::<u64>());
                let (fa, fc) = butterfly_low_fold_q(xa, xc, to, zero);
                vst1q_u64(ea.offset(off).cast::<u64>(), fa);
                vst1q_u64(eb.offset(off).cast::<u64>(), fa);
                vst1q_u64(ec.offset(off).cast::<u64>(), fc);
                vst1q_u64(ed.offset(off).cast::<u64>(), fc);
                off = next_lane_off(off);
            }
        }
    }
}

///
/// # Safety
#[target_feature(enable = "aes")]
pub(super) unsafe fn butterfly_fused_2layer_low_inner_rows(
    block: &mut [F128],
    t_outer: F128,
    t_inner_a: F128,
    t_inner_b: F128,
    quarter: usize,
    num_ntts: usize,
) {
    use core::arch::aarch64::*;

    debug_assert_eq!(t_inner_a.hi, 0);
    debug_assert_eq!(t_inner_b.hi, 0);
    debug_assert_eq!(block.len(), 4 * quarter * num_ntts);
    unsafe {
        let to = const_coeffs(vld1q_u64((&raw const t_outer).cast::<u64>()));
        let ta = vdupq_n_u64(t_inner_a.lo);
        let tb = vdupq_n_u64(t_inner_b.lo);
        let stride = quarter * num_ntts;
        let (top_half, bot_half) = block.split_at_mut(2 * stride);
        let (q1, q2) = top_half.split_at_mut(stride);
        let (q3, q4) = bot_half.split_at_mut(stride);
        for (((a, b), c), d) in q1
            .iter_mut()
            .zip(q2.iter_mut())
            .zip(q3.iter_mut())
            .zip(q4.iter_mut())
        {
            let xa = vld1q_u64((&raw const *a).cast::<u64>());
            let xb = vld1q_u64((&raw const *b).cast::<u64>());
            let xc = vld1q_u64((&raw const *c).cast::<u64>());
            let xd = vld1q_u64((&raw const *d).cast::<u64>());
            let (na, nc, p) = const_lazy_zero(xa, xc, to);
            let (nb, nd) = const_fold_zero(xb, xd, to);
            // Layer L+1: (a,b) under t_inner_a, (c,d) under t_inner_b.
            let (fa, fb) = butterfly_low_fold_q(na, nb, ta, p);
            let (fc, fd) = butterfly_low_fold_q(nc, nd, tb, p);
            vst1q_u64((&raw mut *a).cast::<u64>(), fa);
            vst1q_u64((&raw mut *b).cast::<u64>(), fb);
            vst1q_u64((&raw mut *c).cast::<u64>(), fc);
            vst1q_u64((&raw mut *d).cast::<u64>(), fd);
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
    if use_const_ext() {
        use core::arch::aarch64::*;
        unsafe {
            let t = vld1q_u64((&raw const twiddle).cast::<u64>());
            let prep = three_coeffs(t);
            let base = chunk.as_mut_ptr();
            for idx0 in 0..half {
                let pu = base.add(idx0).cast::<u64>();
                let pv = base.add(idx0 + half).cast::<u64>();
                let (nu, nv) = three_fold_zero(vld1q_u64(pu), vld1q_u64(pv), prep);
                vst1q_u64(pu, nu);
                vst1q_u64(pv, nv);
            }
        }
        return;
    }
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

/// Prepared-twiddle single-layer row butterfly (`FLOCK_NO_NTT_ROW_PAIR_PREP=1`
/// restores the portable per-lane chain).
#[inline]
pub(super) fn use_row_pair_prepared() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_NTT_ROW_PAIR_PREP").is_none());
    *ON
}

/// Single-layer butterfly of two interleaved rows sharing one twiddle:
/// `top' = top + bot * twiddle`, `bot' = bot + top'`, lane by lane.
///
/// The row pair shares its twiddle across every lane, so the fixed-twiddle
/// coefficients ([`three_coeffs`]) are prepared once per call and each lane
/// costs four data PMULLs plus one fold, all in the vector register file.
/// The portable form this replaces runs the generic field multiply per lane
/// and repacks every product through a `{ lo, hi }` struct. Lanes stay
/// independent (one product per lane), so consecutive lanes still overlap in
/// the out-of-order window. Same GF(2^128) products, canonical outputs:
/// bit-identical to `portable::butterfly_row_pair`.
///
/// # Safety
/// Requires the `aes` target feature; `top` and `bot` must have equal length.
#[target_feature(enable = "aes")]
pub(super) unsafe fn butterfly_row_pair(top: &mut [F128], bot: &mut [F128], twiddle: F128) {
    use core::arch::aarch64::*;

    debug_assert_eq!(top.len(), bot.len());
    let lanes = top.len().min(bot.len());
    unsafe {
        let t = vld1q_u64((&raw const twiddle).cast::<u64>());
        let (t_lo, t_mid) = three_coeffs(t);
        let pt = top.as_mut_ptr();
        let pb = bot.as_mut_ptr();
        for lane in 0..lanes {
            let pu = pt.add(lane).cast::<u64>();
            let pv = pb.add(lane).cast::<u64>();
            let u = vld1q_u64(pu);
            let v = vld1q_u64(pv);
            // L = u + v.lo * t.lo + v.hi * R.lo, M = v.lo * t.hi + v.hi * (t.lo + R.hi).
            let l = fused_pmull_hi_acc(fused_pmull_lo_acc(u, v, t_lo), v, t_lo);
            let m = cross_sum_q(v, t_mid);
            let nu = fold_x64_into(l, m);
            vst1q_u64(pu, nu);
            vst1q_u64(pv, veorq_u64(v, nu));
        }
    }
}

#[cfg(test)]
mod row_pair_probe {
    use super::*;

    fn splitmix(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn rand_f128(state: &mut u64) -> F128 {
        F128 {
            lo: splitmix(state),
            hi: splitmix(state),
        }
    }

    /// Vector-resident row butterfly on the split multiply (six PMULLs per
    /// lane): the comparison point for the prepared form.
    #[target_feature(enable = "aes")]
    unsafe fn row_pair_split(top: &mut [F128], bot: &mut [F128], twiddle: F128) {
        use core::arch::aarch64::*;
        unsafe {
            let t = vld1q_u64((&raw const twiddle).cast::<u64>());
            for lane in 0..top.len() {
                let pu = top.as_mut_ptr().add(lane).cast::<u64>();
                let pv = bot.as_mut_ptr().add(lane).cast::<u64>();
                let (nu, nv) = butterfly_fold_zero_q(vld1q_u64(pu), vld1q_u64(pv), t);
                vst1q_u64(pu, nu);
                vst1q_u64(pv, nv);
            }
        }
    }

    /// All three forms agree bit for bit when the whole block is one call.
    #[test]
    fn whole_block_call_matches_per_row_portable() {
        let mut state = 0x424C_4F43_4B5F_5250;
        for &(rows_half, lanes) in &[(1usize, 8usize), (3, 8), (4096, 8), (64, 64), (5, 7)] {
            let n = rows_half * lanes;
            let top: Vec<F128> = (0..n).map(|_| rand_f128(&mut state)).collect();
            let bot: Vec<F128> = (0..n).map(|_| rand_f128(&mut state)).collect();
            let twiddle = rand_f128(&mut state);

            let (mut want_top, mut want_bot) = (top.clone(), bot.clone());
            for (tr, br) in want_top
                .chunks_exact_mut(lanes)
                .zip(want_bot.chunks_exact_mut(lanes))
            {
                super::super::portable::butterfly_row_pair(tr, br, twiddle);
            }

            let (mut got_top, mut got_bot) = (top.clone(), bot.clone());
            // SAFETY: equal-length halves; the module is compiled with `aes`.
            unsafe { butterfly_row_pair(&mut got_top, &mut got_bot, twiddle) };
            assert_eq!(got_top, want_top, "rows_half={rows_half} lanes={lanes}");
            assert_eq!(got_bot, want_bot, "rows_half={rows_half} lanes={lanes}");

            let (mut split_top, mut split_bot) = (top, bot);
            // SAFETY: as above.
            unsafe { row_pair_split(&mut split_top, &mut split_bot, twiddle) };
            assert_eq!(split_top, want_top);
            assert_eq!(split_bot, want_bot);
        }
    }

    /// Local timing probe (not a correctness test). Shapes: the recursive-L1
    /// lead layer (4096 row pairs of 8 lanes per 1 MiB sub-transform), an
    /// L1-resident block, and the 64-lane interleaving. Run with
    /// `cargo test --profile challenge -p flock-core --lib neon_row_pair_timing -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn neon_row_pair_timing() {
        let mut state = 0x5449_4D45_5F52_4F57;
        for &(rows_half, lanes) in &[(4096usize, 8usize), (64, 8), (64, 64), (4096, 64)] {
            let n = rows_half * lanes;
            let base_top: Vec<F128> = (0..n).map(|_| rand_f128(&mut state)).collect();
            let base_bot: Vec<F128> = (0..n).map(|_| rand_f128(&mut state)).collect();
            let twiddle = rand_f128(&mut state);
            let reps = ((1usize << 25) / n).max(1);
            let names = ["portable/row", "split/row", "prepared/row", "prepared/block"];
            let mut best = [f64::MAX; 4];
            for _trial in 0..15 {
                for which in 0..4 {
                    let (mut top, mut bot) = (base_top.clone(), base_bot.clone());
                    let t0 = std::time::Instant::now();
                    for _ in 0..reps {
                        if which == 3 {
                            // SAFETY: equal-length halves; `aes` via cfg.
                            unsafe { butterfly_row_pair(&mut top, &mut bot, twiddle) };
                        } else {
                            for (tr, br) in
                                top.chunks_exact_mut(lanes).zip(bot.chunks_exact_mut(lanes))
                            {
                                match which {
                                    0 => super::super::portable::butterfly_row_pair(tr, br, twiddle),
                                    // SAFETY: equal-length rows; `aes` via cfg.
                                    1 => unsafe { row_pair_split(tr, br, twiddle) },
                                    _ => unsafe { butterfly_row_pair(tr, br, twiddle) },
                                }
                            }
                        }
                        std::hint::black_box((&mut top, &mut bot));
                    }
                    let ns = t0.elapsed().as_secs_f64() * 1e9 / (reps * n) as f64;
                    best[which] = best[which].min(ns);
                }
            }
            let line: Vec<String> = (0..4)
                .map(|i| format!("{} {:.3} ({:+.1}%)", names[i], best[i], (best[i] / best[0] - 1.0) * 100.0))
                .collect();
            eprintln!(
                "[row-pair-timing] rows_half={rows_half} lanes={lanes} ns/butterfly (best of 15): {}",
                line.join(", ")
            );
        }
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

// For a fixed t, precompute R = t.hi * 0x87. Then v*t is represented as
// L + x^64*M with L = v.lo*t.lo + v.hi*R.lo and
// M = v.lo*t.hi + v.hi*(t.lo + R.hi). One x^64 fold completes reduction.
// The existing pending-half butterfly invariant applies to this representation.
#[inline(always)]
unsafe fn const_coeffs(
    t: core::arch::aarch64::uint64x2_t,
) -> (
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
) {
    unsafe {
        use core::arch::aarch64::*;
        let r = core::mem::transmute::<u128, uint64x2_t>(vmull_high_p64(
            vreinterpretq_p64_u64(t),
            vreinterpretq_p64_u64(vdupq_n_u64(0x87)),
        ));
        let lo = vzip1q_u64(t, r);
        let mid = veorq_u64(vextq_u64::<1>(t, t), vzip2q_u64(vdupq_n_u64(0), r));
        (lo, mid)
    }
}

#[inline(always)]
unsafe fn const_lazy(
    u: core::arch::aarch64::uint64x2_t,
    v: core::arch::aarch64::uint64x2_t,
    t: (
        core::arch::aarch64::uint64x2_t,
        core::arch::aarch64::uint64x2_t,
    ),
    p: core::arch::aarch64::uint64x2_t,
) -> (
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
) {
    unsafe {
        use core::arch::aarch64::*;
        let nu = fused_pmull_hi_acc(fused_pmull_lo_acc(u, v, t.0), v, t.0);
        let mid = fused_pmull_hi_acc(fused_pmull_lo_acc(p, v, t.1), v, t.1);
        (nu, veorq_u64(v, nu), mid)
    }
}

#[inline(always)]
unsafe fn const_fold(
    u: core::arch::aarch64::uint64x2_t,
    v: core::arch::aarch64::uint64x2_t,
    t: (
        core::arch::aarch64::uint64x2_t,
        core::arch::aarch64::uint64x2_t,
    ),
    p: core::arch::aarch64::uint64x2_t,
) -> (
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
) {
    unsafe {
        use core::arch::aarch64::*;
        let (nu, _, mid) = const_lazy(u, v, t, p);
        let fu = fold_x64_into(nu, mid);
        (fu, veorq_u64(v, fu))
    }
}
/// [`const_lazy`] with a zero pending half: the x^64 half is the bare cross
/// sum `v.lo * t.1.lo + v.hi * t.1.hi` ([`cross_sum_q`]) instead of that sum
/// accumulated into a zero register. Same products, same values.
#[inline(always)]
unsafe fn const_lazy_zero(
    u: core::arch::aarch64::uint64x2_t,
    v: core::arch::aarch64::uint64x2_t,
    t: (
        core::arch::aarch64::uint64x2_t,
        core::arch::aarch64::uint64x2_t,
    ),
) -> (
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
) {
    unsafe {
        use core::arch::aarch64::*;
        let nu = fused_pmull_hi_acc(fused_pmull_lo_acc(u, v, t.0), v, t.0);
        let mid = cross_sum_q(v, t.1);
        (nu, veorq_u64(v, nu), mid)
    }
}
/// [`const_fold`] with a zero pending half (see [`const_lazy_zero`]).
#[inline(always)]
unsafe fn const_fold_zero(
    u: core::arch::aarch64::uint64x2_t,
    v: core::arch::aarch64::uint64x2_t,
    t: (
        core::arch::aarch64::uint64x2_t,
        core::arch::aarch64::uint64x2_t,
    ),
) -> (
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
) {
    unsafe {
        use core::arch::aarch64::*;
        let (nu, _, mid) = const_lazy_zero(u, v, t);
        let fu = fold_x64_into(nu, mid);
        (fu, veorq_u64(v, fu))
    }
}

#[target_feature(enable = "aes")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn butterfly_fused_2layer_row_with_const_q(
    a: &mut [F128],
    b: &mut [F128],
    c: &mut [F128],
    d: &mut [F128],
    to: (
        core::arch::aarch64::uint64x2_t,
        core::arch::aarch64::uint64x2_t,
    ),
    ta: (
        core::arch::aarch64::uint64x2_t,
        core::arch::aarch64::uint64x2_t,
    ),
    tb: (
        core::arch::aarch64::uint64x2_t,
        core::arch::aarch64::uint64x2_t,
    ),
) {
    unsafe {
        use core::arch::aarch64::*;
        let n = a.len();
        let [ea, eb, ec, ed] = row_ends([a, b, c, d]);
        let mut off = -((n * core::mem::size_of::<F128>()) as isize);
        while off != 0 {
            let xa = vld1q_u64(ea.offset(off).cast::<u64>());
            let xb = vld1q_u64(eb.offset(off).cast::<u64>());
            let xc = vld1q_u64(ec.offset(off).cast::<u64>());
            let xd = vld1q_u64(ed.offset(off).cast::<u64>());
            let (na, nc, p) = const_lazy_zero(xa, xc, to);
            let (nb, nd) = const_fold_zero(xb, xd, to);
            let (fa, fb) = const_fold(na, nb, ta, p);
            let (fc, fd) = const_fold(nc, nd, tb, p);
            vst1q_u64(ea.offset(off).cast::<u64>(), fa);
            vst1q_u64(eb.offset(off).cast::<u64>(), fb);
            vst1q_u64(ec.offset(off).cast::<u64>(), fc);
            vst1q_u64(ed.offset(off).cast::<u64>(), fd);
            off = next_lane_off(off);
        }
    }
}

#[inline]
fn use_const_fold() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_NTT_CONST_FOLD").is_none());
    static DEBUG: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NTT_CONST_FOLD_DEBUG").is_some());
    let enabled = *ON;
    if *DEBUG {
        static HITS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let hits = HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if hits.is_power_of_two() {
            eprintln!("[ntt-const-fold] enabled={enabled} blocks={hits}");
        }
    }
    enabled
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

            // Layer L: (a,c) and (b,d) share t_outer. Row a keeps its
            // x^64 half `p` pending (row c inherits it); b is folded, so
            // 3 folds serve the 4 products.
            let (na, nc, p) = butterfly_lazy_zero_q(xa, xc, to);
            let (nb, nd) = butterfly_fold_zero_q(xb, xd, to);
            // Layer L+1: (a,b) under t_inner_a, (c,d) under t_inner_b; the
            // multiplicands b and d are folded, a and c fold `p` here.
            let (fa, fb) = butterfly_fold_q(na, nb, ta, p);
            let (fc, fd) = butterfly_fold_q(nc, nd, tb, p);

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
        if use_const_fold() {
            butterfly_fused_2layer_row_with_const_q(
                a,
                b,
                c,
                d,
                const_coeffs(to),
                const_coeffs(ta),
                const_coeffs(tb),
            );
        } else {
            butterfly_fused_2layer_row_with_q(a, b, c, d, to, ta, tb);
        }
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
    unsafe {
        if use_const_fold() {
            butterfly_fused_2layer_rows_impl::<true>(
                block, t_outer, t_inner_a, t_inner_b, quarter, num_ntts, odd_tail,
            );
        } else {
            butterfly_fused_2layer_rows_impl::<false>(
                block, t_outer, t_inner_a, t_inner_b, quarter, num_ntts, odd_tail,
            );
        }
    }
}

#[cfg(test)]
mod three_const_fold_tests {
    use super::*;
    fn next(s: &mut u64) -> u64 {
        *s = s.wrapping_add(0x9e3779b97f4a7c15);
        let mut x = *s;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }
    fn scalar_mul(a: F128, b: F128) -> F128 {
        let mut x = (a.lo as u128) | ((a.hi as u128) << 64);
        let mut y = (b.lo as u128) | ((b.hi as u128) << 64);
        let mut z = 0u128;
        for _ in 0..128 {
            if y & 1 != 0 {
                z ^= x;
            }
            let carry = x >> 127;
            x = (x << 1) ^ (carry * 0x87);
            y >>= 1;
        }
        F128::new(z as u64, (z >> 64) as u64)
    }
    #[test]
    fn row_matches_independent_polynomial_oracle() {
        let mut seen = [0usize; 2];
        for seed in 0..16u64 {
            for n in [1usize, 2, 3, 31, 64, 129, 257] {
                let mut state = 0xbe51d7a33 ^ seed;
                let input: Vec<_> = (0..8 * n)
                    .map(|_| F128::new(next(&mut state), next(&mut state)))
                    .collect();
                let sets = [
                    [F128::ZERO; 7],
                    [F128::ONE; 7],
                    [F128::new(u64::MAX, 0); 7],
                    [F128::new(u64::MAX, u64::MAX); 7],
                    core::array::from_fn(|_| F128::new(0, next(&mut state))),
                    core::array::from_fn(|_| F128::new(next(&mut state), next(&mut state))),
                ];
                for t in sets {
                    let mut expected = input.clone();
                    for layer in 0..3 {
                        let stride = 4 >> layer;
                        for block in 0..1usize << layer {
                            let tw = t[(1 << layer) - 1 + block];
                            for row in 0..stride {
                                for lane in 0..n {
                                    let u = (2 * stride * block + row) * n + lane;
                                    let v = u + stride * n;
                                    let nu = expected[u] + scalar_mul(expected[v], tw);
                                    expected[v] += nu;
                                    expected[u] = nu;
                                }
                            }
                        }
                    }
                    for mode in 0..2 {
                        seen[mode] += 1;
                        let mut actual = input.clone();
                        unsafe {
                            use core::arch::aarch64::*;
                            let tv: [uint64x2_t; 7] = core::array::from_fn(|i| {
                                vld1q_u64((&raw const t[i]).cast::<u64>())
                            });
                            if mode == 1 {
                                let prepared = core::array::from_fn(|i| three_coeffs(tv[i]));
                                butterfly_fused_3layer_row_with_const_q(
                                    actual.as_mut_ptr(),
                                    1,
                                    n,
                                    n,
                                    0,
                                    &prepared,
                                );
                            } else {
                                butterfly_fused_3layer_row_with_q(
                                    actual.as_mut_ptr(),
                                    1,
                                    n,
                                    n,
                                    0,
                                    &tv,
                                );
                            }
                        }
                        for (a, b) in actual.iter().zip(&expected) {
                            assert_eq!(
                                (a.lo, a.hi),
                                (b.lo, b.hi),
                                "seed={seed},lanes={n},mode={mode}"
                            );
                        }
                    }
                }
            }
        }
        assert_eq!(seen, [672; 2]);
    }
}

#[target_feature(enable = "aes")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn butterfly_fused_2layer_rows_impl<const PREPARED: bool>(
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
        let prepared = if PREPARED {
            Some((const_coeffs(to), const_coeffs(ta), const_coeffs(tb)))
        } else {
            None
        };
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
            if let Some((to, ta, tb)) = prepared {
                butterfly_fused_2layer_row_with_const_q(
                    &mut row_a[..lanes],
                    &mut row_b[..lanes],
                    &mut row_c[..lanes],
                    &mut row_d[..lanes],
                    to,
                    ta,
                    tb,
                );
            } else {
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
}

#[cfg(test)]
mod const_fold_tests {
    use super::*;

    fn scalar_mul(a: F128, b: F128) -> F128 {
        let mut x = a.lo as u128 | ((a.hi as u128) << 64);
        let mut y = b.lo as u128 | ((b.hi as u128) << 64);
        let mut z = 0u128;
        for _ in 0..128 {
            if y & 1 != 0 {
                z ^= x;
            }
            let carry = x >> 127;
            x = (x << 1) ^ (carry * 0x87);
            y >>= 1;
        }
        F128::new(z as u64, (z >> 64) as u64)
    }

    fn random(s: &mut u64) -> u64 {
        *s = s.wrapping_add(0x9e3779b97f4a7c15);
        let mut x = *s;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }

    #[test]
    fn full_pair_matches_independent_polynomial_oracle() {
        let mut cases = 0;
        let mut modes_seen = [0usize; 2];
        for seed in 0..16u64 {
            for n in [1usize, 2, 3, 31, 64, 129, 257] {
                let mut state = 0xbe51d7a33 ^ seed;
                let input: Vec<F128> = (0..4 * n)
                    .map(|_| F128::new(random(&mut state), random(&mut state)))
                    .collect();
                let sets = [
                    [F128::ZERO; 3],
                    [F128::ONE; 3],
                    [F128::new(u64::MAX, 0); 3],
                    [F128::new(u64::MAX, u64::MAX); 3],
                    core::array::from_fn(|_| F128::new(0, random(&mut state))),
                    core::array::from_fn(|_| F128::new(random(&mut state), random(&mut state))),
                ];
                for t in sets {
                    let mut expected = vec![F128::ZERO; 4 * n];
                    for i in 0..n {
                        let a = input[i] + scalar_mul(input[2 * n + i], t[0]);
                        let c = input[2 * n + i] + a;
                        let b = input[n + i] + scalar_mul(input[3 * n + i], t[0]);
                        let d = input[3 * n + i] + b;
                        let aa = a + scalar_mul(b, t[1]);
                        let bb = b + aa;
                        let cc = c + scalar_mul(d, t[2]);
                        let dd = d + cc;
                        expected[i] = aa;
                        expected[n + i] = bb;
                        expected[2 * n + i] = cc;
                        expected[3 * n + i] = dd;
                    }
                    for mode in 0..2 {
                        modes_seen[mode] += 1;
                        let mut actual = input.clone();
                        let (a, rest) = actual.split_at_mut(n);
                        let (b, rest) = rest.split_at_mut(n);
                        let (c, d) = rest.split_at_mut(n);
                        unsafe {
                            use core::arch::aarch64::*;
                            let to = vld1q_u64((&raw const t[0]).cast::<u64>());
                            let ta = vld1q_u64((&raw const t[1]).cast::<u64>());
                            let tb = vld1q_u64((&raw const t[2]).cast::<u64>());
                            if mode == 1 {
                                butterfly_fused_2layer_row_with_const_q(
                                    a,
                                    b,
                                    c,
                                    d,
                                    const_coeffs(to),
                                    const_coeffs(ta),
                                    const_coeffs(tb),
                                );
                            } else {
                                butterfly_fused_2layer_row_with_q(a, b, c, d, to, ta, tb);
                            }
                        }
                        for (actual, expected) in actual.iter().zip(&expected) {
                            assert_eq!(
                                (actual.lo, actual.hi),
                                (expected.lo, expected.hi),
                                "seed={seed}, lanes={n}, mode={mode}"
                            );
                        }
                    }
                    cases += 1;
                }
            }
        }
        assert_eq!(cases, 672);
        assert_eq!(modes_seen, [672; 2]);
    }
}

#[cfg(test)]
mod low_pmull_tests {
    use super::*;

    fn scalar_mul(a: F128, b: F128) -> F128 {
        let mut x = a.lo as u128 | ((a.hi as u128) << 64);
        let mut y = b.lo as u128 | ((b.hi as u128) << 64);
        let mut z = 0u128;
        for _ in 0..128 {
            if y & 1 != 0 {
                z ^= x;
            }
            let carry = x >> 127;
            x = (x << 1) ^ (carry * 0x87);
            y >>= 1;
        }
        F128::new(z as u64, (z >> 64) as u64)
    }

    fn random(s: &mut u64) -> u64 {
        *s = s.wrapping_add(0x9e3779b97f4a7c15);
        let mut x = *s;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }

    #[test]
    fn low_pair_matches_independent_polynomial_oracle() {
        let mut cases = 0;
        let mut modes_seen = [0usize; 3];
        for seed in 0..16u64 {
            for n in [1usize, 2, 3, 31, 64, 129, 257] {
                let mut state = 0xbe51d7a33 ^ seed;
                let input: Vec<F128> = (0..4 * n)
                    .map(|_| F128::new(random(&mut state), random(&mut state)))
                    .collect();
                let sets = [
                    [F128::ZERO; 3],
                    [F128::ONE; 3],
                    [F128::new(u64::MAX, 0); 3],
                    core::array::from_fn(|_| F128::new(random(&mut state), 0)),
                ];
                for t in sets {
                    let mut expected = vec![F128::ZERO; 4 * n];
                    for i in 0..n {
                        let a = input[i] + scalar_mul(input[2 * n + i], t[0]);
                        let c = input[2 * n + i] + a;
                        let b = input[n + i] + scalar_mul(input[3 * n + i], t[0]);
                        let d = input[3 * n + i] + b;
                        let aa = a + scalar_mul(b, t[1]);
                        let bb = b + aa;
                        let cc = c + scalar_mul(d, t[2]);
                        let dd = d + cc;
                        expected[i] = aa;
                        expected[n + i] = bb;
                        expected[2 * n + i] = cc;
                        expected[3 * n + i] = dd;
                    }
                    for mode in 0..3 {
                        modes_seen[mode] += 1;
                        let mut actual = input.clone();
                        let (a, rest) = actual.split_at_mut(n);
                        let (b, rest) = rest.split_at_mut(n);
                        let (c, d) = rest.split_at_mut(n);
                        unsafe {
                            if mode == 2 {
                                butterfly_fused_2layer_low_lazy(a, b, c, d, t[0], t[1], t[2]);
                            } else if mode == 1 {
                                butterfly_fused_2layer_low_impl::<true>(
                                    a, b, c, d, t[0], t[1], t[2],
                                );
                            } else {
                                butterfly_fused_2layer_low_impl::<false>(
                                    a, b, c, d, t[0], t[1], t[2],
                                );
                            }
                        }
                        for (actual, expected) in actual.iter().zip(&expected) {
                            assert_eq!(
                                (actual.lo, actual.hi),
                                (expected.lo, expected.hi),
                                "seed={seed}, lanes={n}, mode={mode}"
                            );
                        }
                    }
                    cases += 1;
                }
            }
        }
        assert_eq!(cases, 448);
        assert_eq!(modes_seen, [448; 3]);
    }
}

#[cfg(test)]
mod zero_pending_tests {
    use super::*;

    fn next(s: &mut u64) -> u64 {
        *s = s.wrapping_add(0x9e3779b97f4a7c15);
        let mut x = *s;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }

    fn eq(a: core::arch::aarch64::uint64x2_t, b: core::arch::aarch64::uint64x2_t) -> bool {
        use core::arch::aarch64::*;
        unsafe { vgetq_lane_u64::<0>(a) == vgetq_lane_u64::<0>(b) && vgetq_lane_u64::<1>(a) == vgetq_lane_u64::<1>(b) }
    }

    /// The zero-pending twins give exactly the outputs of the general helpers
    /// called with a zero pending half: lazy (both rows and the x^64 half) and
    /// folded, for the radix-8 (`three_*`) and the deep-pair (`const_*`) forms.
    #[test]
    fn zero_pending_twins_match_zero_accumulated_forms() {
        use core::arch::aarch64::*;
        let specials = [0u64, 1, 0x87, u64::MAX, 1 << 63];
        let mut state = 0x5eed_0f_2e40u64;
        let mut cases = 0usize;
        for i in 0..4096usize {
            let mut word = |k: usize| {
                if (i >> k) % 7 == 0 { specials[(i + k) % specials.len()] } else { next(&mut state) }
            };
            let (u0, u1, v0, v1, t0, t1) = (word(0), word(1), word(2), word(3), word(4), word(5));
            unsafe {
                let u = vcombine_u64(vcreate_u64(u0), vcreate_u64(u1));
                let v = vcombine_u64(vcreate_u64(v0), vcreate_u64(v1));
                let t = vcombine_u64(vcreate_u64(t0), vcreate_u64(t1));
                let zero = vdupq_n_u64(0);

                let prep = three_coeffs(t);
                let (a, b, m) = three_lazy(u, v, prep, zero);
                let (a0, b0, m0) = three_lazy_zero(u, v, prep);
                assert!(eq(a, a0) && eq(b, b0) && eq(m, m0), "three_lazy case {i}");
                let (a, b) = three_fold(u, v, prep, zero);
                let (a0, b0) = three_fold_zero(u, v, prep);
                assert!(eq(a, a0) && eq(b, b0), "three_fold case {i}");

                let prep = const_coeffs(t);
                let (a, b, m) = const_lazy(u, v, prep, zero);
                let (a0, b0, m0) = const_lazy_zero(u, v, prep);
                assert!(eq(a, a0) && eq(b, b0) && eq(m, m0), "const_lazy case {i}");
                let (a, b) = const_fold(u, v, prep, zero);
                let (a0, b0) = const_fold_zero(u, v, prep);
                assert!(eq(a, a0) && eq(b, b0), "const_fold case {i}");
            }
            cases += 1;
        }
        assert_eq!(cases, 4096);
    }
}

#[cfg(test)]
mod fold_x64_tests {
    use super::*;

    fn next(s: &mut u64) -> u64 {
        *s = s.wrapping_add(0x9e3779b97f4a7c15);
        let mut x = *s;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }

    /// `l + x^64 · m` on plain words: `m.lo` moves to the high limb and
    /// `m.hi · x^128 = m.hi · 0x87` is a carry-less product of at most 71 bits.
    fn fold_reference(l: [u64; 2], m: [u64; 2]) -> [u64; 2] {
        let mut prod = 0u128;
        for bit in 0..64 {
            if (m[1] >> bit) & 1 == 1 {
                prod ^= 0x87u128 << bit;
            }
        }
        [l[0] ^ prod as u64, l[1] ^ m[0] ^ (prod >> 64) as u64]
    }

    /// The three-term fold equals the reference on random and edge words.
    #[test]
    fn fold_x64_into_matches_reference() {
        use core::arch::aarch64::*;
        let specials = [0u64, 1, 0x87, u64::MAX, 1 << 63, 0x8000_0000_0000_0087];
        let mut state = 0xf01d_64u64;
        for i in 0..8192usize {
            let mut word = |k: usize| {
                if (i >> k) % 5 == 0 { specials[(i + 3 * k) % specials.len()] } else { next(&mut state) }
            };
            let (l, m) = ([word(0), word(1)], [word(2), word(3)]);
            let got = unsafe {
                let r = fold_x64_into(
                    vcombine_u64(vcreate_u64(l[0]), vcreate_u64(l[1])),
                    vcombine_u64(vcreate_u64(m[0]), vcreate_u64(m[1])),
                );
                [vgetq_lane_u64::<0>(r), vgetq_lane_u64::<1>(r)]
            };
            assert_eq!(got, fold_reference(l, m), "case {i}: l={l:x?} m={m:x?}");
        }
    }
}

#[cfg(test)]
mod lane_loop_tests {
    use super::*;

    fn next(s: &mut u64) -> u64 {
        *s = s.wrapping_add(0x9e3779b97f4a7c15);
        let mut x = *s;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }

    fn mul(a: F128, b: F128) -> F128 {
        let (mut x, mut y) = (a.lo as u128 | ((a.hi as u128) << 64), b.lo as u128 | ((b.hi as u128) << 64));
        let mut z = 0u128;
        for _ in 0..128 {
            if y & 1 != 0 {
                z ^= x;
            }
            x = (x << 1) ^ ((x >> 127) * 0x87);
            y >>= 1;
        }
        F128::new(z as u64, (z >> 64) as u64)
    }

    /// `u' = u + v·t`, `v' = v + u'` on `data[u]`, `data[v]`.
    fn bf(data: &mut [F128], u: usize, v: usize, t: F128) {
        let nu = data[u] + mul(data[v], t);
        data[v] += nu;
        data[u] = nu;
    }

    /// The deep-pair rows (`butterfly_fused_2layer_rows`, prepared form) and the
    /// prepared radix-8 row against a per-lane scalar chain, with lane counts
    /// below the row width: every processed lane matches, every other lane
    /// keeps its poison.
    #[test]
    fn lane_loops_match_scalar_and_leave_skipped_lanes() {
        let poison = F128::new(0xA5A5_A5A5_A5A5_A5A5, 0x5A5A_5A5A_5A5A_5A5A);
        let mut state = 0x1a7e_100bu64;
        let mut cases = 0usize;
        for num_ntts in [1usize, 3, 8, 64] {
            for odd_tail in [0usize, 1, num_ntts.saturating_sub(1), num_ntts / 2] {
                if odd_tail >= num_ntts {
                    continue;
                }
                // Deep pair: quarter 2 (row groups 0 and 1; group 1 is odd).
                let quarter = 2;
                let len = 4 * quarter * num_ntts;
                let mut input: Vec<F128> =
                    (0..len).map(|_| F128::new(next(&mut state), next(&mut state))).collect();
                // Odd row groups' tail lanes are not processed: poison them.
                for q in 0..4 {
                    for lane in num_ntts - odd_tail..num_ntts {
                        input[(q * quarter + 1) * num_ntts + lane] = poison;
                    }
                }
                let t: [F128; 3] = core::array::from_fn(|_| F128::new(next(&mut state), next(&mut state)));
                let mut expected = input.clone();
                for r in 0..quarter {
                    let lanes = if r & 1 == 1 { num_ntts - odd_tail } else { num_ntts };
                    for lane in 0..lanes {
                        let at = |q: usize| (q * quarter + r) * num_ntts + lane;
                        bf(&mut expected, at(0), at(2), t[0]);
                        bf(&mut expected, at(1), at(3), t[0]);
                        bf(&mut expected, at(0), at(1), t[1]);
                        bf(&mut expected, at(2), at(3), t[2]);
                    }
                }
                let mut actual = input.clone();
                unsafe {
                    butterfly_fused_2layer_rows_impl::<true>(&mut actual, t[0], t[1], t[2], quarter, num_ntts, odd_tail);
                }
                assert!(actual == expected, "deep pair num_ntts={num_ntts} odd_tail={odd_tail}");

                // Radix-8 row, eighth 1: lanes 0..lanes of row 0 of eight rows.
                for lanes in [0usize, 1, num_ntts - odd_tail, num_ntts] {
                    let mut input: Vec<F128> =
                        (0..8 * num_ntts).map(|_| F128::new(next(&mut state), next(&mut state))).collect();
                    for i in 0..8 {
                        for lane in lanes..num_ntts {
                            input[i * num_ntts + lane] = poison;
                        }
                    }
                    let tw: [F128; 7] = core::array::from_fn(|_| F128::new(next(&mut state), next(&mut state)));
                    let mut expected = input.clone();
                    for lane in 0..lanes {
                        let at = |i: usize| i * num_ntts + lane;
                        for layer in 0..3 {
                            let stride = 4 >> layer;
                            for block in 0..1usize << layer {
                                for row in 0..stride {
                                    let u = 2 * stride * block + row;
                                    bf(&mut expected, at(u), at(u + stride), tw[(1 << layer) - 1 + block]);
                                }
                            }
                        }
                    }
                    let mut actual = input.clone();
                    unsafe {
                        use core::arch::aarch64::*;
                        let prepared: [Prep; 7] = core::array::from_fn(|i| {
                            three_coeffs(vld1q_u64((&raw const tw[i]).cast::<u64>()))
                        });
                        butterfly_fused_3layer_row_with_const_q(actual.as_mut_ptr(), 1, num_ntts, lanes, 0, &prepared);
                    }
                    assert!(actual == expected, "radix-8 num_ntts={num_ntts} lanes={lanes}");
                }
                cases += 1;
            }
        }
        assert!(cases >= 12, "{cases}");
    }

    /// The zero-tail final pair: dense lanes and tail lanes against the
    /// per-lane scalar chain, over several blocks.
    #[test]
    fn zero_tail_final_pair_lane_loops_match_scalar() {
        let mut state = 0xf1_9a1u64;
        for num_ntts in [1usize, 2, 8, 64] {
            for zero_tail in 0..num_ntts.min(9) {
                let blocks = 3;
                let mut input: Vec<F128> =
                    (0..4 * num_ntts * blocks).map(|_| F128::new(next(&mut state), next(&mut state))).collect();
                for blk in 0..blocks {
                    for q in [1usize, 3] {
                        for lane in num_ntts - zero_tail..num_ntts {
                            input[(4 * blk + q) * num_ntts + lane] = F128::ZERO;
                        }
                    }
                }
                let tw: Vec<(F128, F128, F128)> = (0..blocks)
                    .map(|_| (F128::new(next(&mut state), 0), F128::new(next(&mut state), 0), F128::new(next(&mut state), 0)))
                    .collect();
                let mut expected = input.clone();
                for (blk, &(to, ta, tb)) in tw.iter().enumerate() {
                    for lane in 0..num_ntts {
                        let at = |q: usize| (4 * blk + q) * num_ntts + lane;
                        bf(&mut expected, at(0), at(2), to);
                        bf(&mut expected, at(1), at(3), to);
                        bf(&mut expected, at(0), at(1), ta);
                        bf(&mut expected, at(2), at(3), tb);
                    }
                }
                let mut actual = input.clone();
                unsafe {
                    butterfly_fused_2layer_low_lazy_zero_tail_blocks(&mut actual, num_ntts, zero_tail, |i| tw[i]);
                }
                assert!(actual == expected, "num_ntts={num_ntts} zero_tail={zero_tail}");
            }
        }
    }
}
