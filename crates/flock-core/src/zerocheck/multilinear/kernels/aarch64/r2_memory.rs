//! Complete live delta lines and bounded descending input hints.
use super::*;
use core::sync::atomic::{AtomicU8, Ordering};

static MODE: AtomicU8 = AtomicU8::new(3);

/// Read the two diagnostic switches during explicit pool startup.
pub(crate) fn init() {
    let enabled = |key| std::env::var_os(key).is_none_or(|v| v != "1");
    let mode = u8::from(enabled("FLOCK_NO_R2_STORE_BURST"))
        | (u8::from(enabled("FLOCK_NO_R2_PREFETCH")) << 1);
    MODE.store(mode, Ordering::Relaxed);
}

#[inline(always)]
pub(super) fn mode() -> u8 {
    MODE.load(Ordering::Relaxed)
}

#[repr(align(128))]
struct DeltaBlock {
    _bytes: [core::mem::MaybeUninit<u8>; 2048],
}

struct BurstPlan {
    scratch: *mut u8,
    base: usize,
    start: usize,
    end: usize,
}

impl BurstPlan {
    #[inline(always)]
    fn empty(scratch: *mut u8) -> Self {
        Self {
            scratch,
            base: 0,
            start: 0,
            end: 0,
        }
    }

    #[inline(always)]
    fn begin(&mut self, base: *mut u8, head: bool, enabled: bool) {
        self.base = base as usize;
        // Head deltas and all partial lines stay on the direct-store path.
        // A group contains two adjacent pairs; require group alignment so
        // its stores cannot straddle a burst boundary.
        let first = self.base + if head { 64 } else { 0 };
        let last = self.base + if head { 1872 } else { 1936 };
        self.start = (first + 127) & !127;
        self.end = last & !127;
        if !enabled || (self.base & 31) != 0 {
            self.end = self.start;
        }
    }

    #[inline(always)]
    unsafe fn store<const OFF: usize>(&self, dst: *mut u8, a0: u64, a1: u64, b0: u64, b1: u64) {
        unsafe {
            let dst = dst.wrapping_add(OFF);
            store_delta_pair_gpr(self.target(dst), a0, a1, b0, b1);
        }
    }

    #[inline(always)]
    unsafe fn target(&self, dst: *mut u8) -> *mut u8 {
        let address = dst as usize;
        if address.wrapping_sub(self.start) < self.end - self.start {
            unsafe { self.scratch.add(address - self.base) }
        } else {
            dst
        }
    }

    #[inline(always)]
    unsafe fn flush(&self) {
        unsafe {
            let mut address = self.start;
            while address < self.end {
                copy_line(self.scratch.add(address - self.base), address as *mut u8);
                address += 128;
            }
        }
    }
}

#[inline(always)]
unsafe fn copy_line(src: *const u8, dst: *mut u8) {
    use core::arch::aarch64::uint64x2_t;
    let (_a, _b, _c, _d, _e, _f, _g, _h): (
        uint64x2_t, uint64x2_t, uint64x2_t, uint64x2_t,
        uint64x2_t, uint64x2_t, uint64x2_t, uint64x2_t,
    );
    unsafe {
        core::arch::asm!(
            "ldp {a:q}, {b:q}, [{src}]",
            "ldp {c:q}, {d:q}, [{src}, #32]",
            "ldp {e:q}, {f:q}, [{src}, #64]",
            "ldp {g:q}, {h:q}, [{src}, #96]",
            "stnp {a:q}, {b:q}, [{dst}]",
            "stnp {c:q}, {d:q}, [{dst}, #32]",
            "stnp {e:q}, {f:q}, [{dst}, #64]",
            "stnp {g:q}, {h:q}, [{dst}, #96]",
            src = in(reg) src,
            dst = in(reg) dst,
            a = out(vreg) _a, b = out(vreg) _b,
            c = out(vreg) _c, d = out(vreg) _d,
            e = out(vreg) _e, f = out(vreg) _f,
            g = out(vreg) _g, h = out(vreg) _h,
            options(nostack, preserves_flags),
        );
    }
}

#[inline(always)]
unsafe fn prefetch_sources(a: *const u64, b: *const u64) {
    unsafe {
        core::arch::asm!(
            "prfm pldl1keep, [{a}]",
            "prfm pldl1keep, [{b}]",
            a = in(reg) a, b = in(reg) b,
            options(readonly, nostack, preserves_flags),
        );
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
#[target_feature(enable = "aes")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn candidate<const BURST: bool, const PREFETCH: bool>(
    table_data: *const u8,
    a_packed: *const u8,
    b_packed: *const u8,
    anchors: *mut F128,
    deltas: *mut u8,
    eq_lo: *const F128,
    lo_size: usize,
    pair_idx_base: usize,
    pair_in_block_mask: usize,
    useful_pairs_inclusive: usize,
    degen: bool,
    skip_dead: bool,
    tail_compact: bool,
    head_fail: *const core::sync::atomic::AtomicBool,
    out: *mut F128,
) {
    use core::arch::aarch64::*;
    const GPR_DELTA: bool = true;
    const FIXED_PADS: bool = true;
    const ASM_OFF: bool = true;
    const RANKED_INV: bool = true;

    #[inline(always)]
    unsafe fn store_anchor_pair_nt(dst: *mut F128, a: uint64x2_t, b: uint64x2_t) {
        unsafe {
            core::arch::asm!(
                "stnp {a:q}, {b:q}, [{dst}]",
                dst = in(reg) dst,
                a = in(vreg) a,
                b = in(vreg) b,
                options(nostack, preserves_flags),
            );
        }
    }

    unsafe {
        let zero = vdupq_n_u64(0);
        let k87 = vdupq_n_u64(0x87);
        let tri_zero = TriNeon {
            ll: zero,
            cross: zero,
            hh: zero,
        };
        let window_tri = r2_window_tri_enabled();
        let mut sum0 = tri_zero;
        let mut sum3 = tri_zero;
        let mut sum4 = tri_zero;
        let mut sum5 = tri_zero;
        let b_ones = fold_row_q(table_data, [0xffu8; 8].as_ptr());
        let ones_is_one = is_zero_q(veorq_u64(
            b_ones,
            core::mem::transmute::<F128, uint64x2_t>(F128::ONE),
        ));

        let aligned_padding = pair_in_block_mask == 127
            && useful_pairs_inclusive == 121
            && (pair_idx_base & 127) == 0
            && lo_size.is_multiple_of(128);
        let fixed_pads = FIXED_PADS && aligned_padding;
        let init_first = r2p_fused_enabled();
        let tail_fused = r2p_tail_fused_enabled();
        assert!(
            !tail_compact
                || (skip_dead && (pair_idx_base & 127) == 0 && lo_size.is_multiple_of(128))
        );
        let head = !head_fail.is_null();
        assert!(!head || (tail_compact && ASM_OFF && RANKED_INV && FIXED_PADS && GPR_DELTA && fixed_pads));
        assert!(!head || (degen && ones_is_one && r2p_fused_enabled()));
        let mut head_ok = true;
        let (anchors0, deltas0) = (anchors, deltas);
        // Only complete live body lines use this uninitialized local storage.
        let mut scratch = core::mem::MaybeUninit::<DeltaBlock>::uninit();
        let mut burst = BurstPlan::empty(scratch.as_mut_ptr().cast::<u8>());
        for first in (0..lo_size / 2).step_by(32) {
            if BURST && (first & 63) == 0 {
                let stride = if head {
                    HC_DELTA_BYTES_PER_BLOCK
                } else if tail_compact {
                    TC_DELTA_BYTES_PER_BLOCK
                } else {
                    2048
                };
                let base = deltas0.add(stride * (first / 64));
                burst.begin(base, head, fixed_pads && skip_dead);
            }
            let (anchors, deltas) = if head {
                (
                    anchors.wrapping_sub(HC_ANCHOR_GAP * (first / 64) + HC_ANCHOR_SHIFT),
                    deltas.wrapping_sub(HC_DELTA_GAP * (first / 64) + HC_DELTA_SHIFT),
                )
            } else if tail_compact {
                (
                    anchors.wrapping_sub(TC_ANCHOR_GAP * (first / 64)),
                    deltas.wrapping_sub(TC_DELTA_GAP * (first / 64)),
                )
            } else {
                (anchors, deltas)
            };
            let head_win = head && (first & 63) == 0;
            for t in 0..2 {
                let mut w0 = tri_zero;
                let mut w3 = tri_zero;
                let mut w4 = tri_zero;
                let mut w5 = tri_zero;
                let b_end = if aligned_padding && (first & 63) == 32 {
                    15 - t
                } else {
                    16
                };
                if !skip_dead {
                    for pos in (b_end..16).rev() {
                        let x_lo = 2 * (first + 2 * pos + t);
                        store_anchor_pair_nt(anchors.add(2 * x_lo), zero, zero);
                        vst1q_u64(deltas.add(x_lo * 16).cast::<u64>(), zero);
                        store_anchor_pair_nt(anchors.add(2 * (x_lo + 1)), zero, zero);
                        vst1q_u64(deltas.add((x_lo + 1) * 16).cast::<u64>(), zero);
                    }
                }
                let mut xb = vdupq_n_u64(1u64 << (b_end - 1));
                macro_rules! group {
                    ($pos:expr, $pads:expr) => {
                        group!(@ $pos, $pads, false)
                    };
                    (@ $pos:expr, $pads:expr, $init:literal) => {
                        'group: {
                            let u = first + 2 * $pos + t;
                            let x_lo0 = 2 * u;
                            let x_lo1 = x_lo0 + 1;
                            let (pad0, pad1): (bool, bool) = ($pads)(x_lo0, x_lo1);
                            let pair = |x_lo: usize, pad: bool| {
                                if pad {
                                    if !skip_dead {
                                        store_anchor_pair_nt(anchors.add(2 * x_lo), zero, zero);
                                        vst1q_u64(deltas.add(x_lo * 16).cast::<u64>(), zero);
                                    }
                                    return (zero, zero, 0u64, 0u64, false);
                                }
                                let a_row = a_packed.add(2 * x_lo * 8).cast::<u64>();
                                let b_row = b_packed.add(2 * x_lo * 8).cast::<u64>();
                                let (a_even, a_odd, b_even, b_odd) = if GPR_DELTA {
                                    let (a_even, a_odd) = load_code_pair_gpr(a_row);
                                    let (b_even, b_odd) = load_code_pair_gpr(b_row);
                                    (a_even, a_odd, b_even, b_odd)
                                } else {
                                    (
                                        u64::from_le(core::ptr::read_unaligned(a_row)),
                                        u64::from_le(core::ptr::read_unaligned(a_row.add(1))),
                                        u64::from_le(core::ptr::read_unaligned(b_row)),
                                        u64::from_le(core::ptr::read_unaligned(b_row.add(1))),
                                    )
                                };
                                let deg = degen && (b_even & b_odd) == u64::MAX;
                                let (a, b) = if deg {
                                    (fold_row_code_q(table_data, a_even), b_ones)
                                } else {
                                    fold_two_row_codes_q(table_data, a_even, b_even)
                                };
                                store_anchor_pair_nt(anchors.add(2 * x_lo), a, b);
                                if GPR_DELTA {
                                    if BURST {
                                        burst.store::<0>(deltas.add(x_lo * 16), a_even, a_odd, b_even, b_odd);
                                    } else {
                                    store_delta_pair_gpr(
                                        deltas.add(x_lo * 16),
                                        a_even,
                                        a_odd,
                                        b_even,
                                        b_odd,
                                    );
                                    }
                                } else {
                                    let delta_pair = core::mem::transmute::<[u64; 2], uint64x2_t>([
                                        a_even ^ a_odd,
                                        b_even ^ b_odd,
                                    ]);
                                    vst1q_u64(deltas.add(x_lo * 16).cast::<u64>(), delta_pair);
                                }
                                (a, b, a_odd, b_odd, deg)
                            };
                            let (a0, b0, a1c, b1c, deg0) = pair(x_lo0, pad0);
                            let (a2, b2, a3c, b3c, deg1) = pair(x_lo1, pad1);
                            if pad0 & pad1 {
                                break 'group;
                            }
                            if deg0 & deg1 & ones_is_one {
                                w0.ll = veorq_u64(w0.ll, mul_xb_q(a2, xb, k87));
                            } else {
                                let e_a = veorq_u64(a0, a2);
                                let e_b = veorq_u64(b0, b2);
                                if !pad1 {
                                    if $init && init_first {
                                        tri_set_q(&mut w0, mul_xb_q(a2, xb, k87), b2);
                                    } else {
                                        tri_acc_q_crossfused(&mut w0, mul_xb_q(a2, xb, k87), b2);
                                    }
                                }
                                let e_s = mul_xb_q(e_a, xb, k87);
                                if $init && init_first {
                                        tri_set_q(&mut w3, e_s, e_b);
                                    } else {
                                        tri_acc_q_crossfused(&mut w3, e_s, e_b);
                                    }
                                let (o_a, o_b) =
                                    fold_two_codes_paired_q(table_data, a1c ^ a3c, b1c ^ b3c);
                                let o_s = mul_xb_q(o_a, xb, k87);
                                if $init && init_first {
                                        tri_set_q(&mut w4, o_s, o_b);
                                    } else {
                                        tri_acc_q_crossfused(&mut w4, o_s, o_b);
                                    }
                                let (s5, b5) = (veorq_u64(e_s, o_s), veorq_u64(e_b, o_b));
                                if $init && init_first {
                                    tri_set_q(&mut w5, s5, b5);
                                } else {
                                    tri_acc_q_crossfused(&mut w5, s5, b5);
                                }
                            }
                        }
                        xb = vshrq_n_u64::<1>(xb);
                    };
                }
                let generic_pads = |x0: usize, x1: usize| {
                    (
                        ((pair_idx_base + x0) & pair_in_block_mask) >= useful_pairs_inclusive,
                        ((pair_idx_base + x1) & pair_in_block_mask) >= useful_pairs_inclusive,
                    )
                };
                if fixed_pads {
                    let boundary = b_end == 15;
                    group!(@ b_end - 1, |_: usize, _: usize| (false, boundary), true);
                    if ASM_OFF {
                        let top = first + 2 * (b_end - 2) + t;
                        let mut pa = a_packed.add(32 * top).cast::<u64>();
                        let mut pb = b_packed.add(32 * top).cast::<u64>();
                        let mut pn = anchors.add(4 * top);
                        let mut pd = deltas.add(32 * top);
                        macro_rules! pair_off {
                            ($code:literal, $anc:literal, $degen:expr, $delta:expr) => {{
                                let (a_even, a_odd) = load_code_pair_gpr_off::<$code>(pa);
                                let (b_even, b_odd) = load_code_pair_gpr_off::<$code>(pb);
                                let deg = $degen && (b_even & b_odd) == u64::MAX;
                                let (a, b) = if deg {
                                    (fold_row_code_q(table_data, a_even), b_ones)
                                } else {
                                    fold_two_row_codes_q(table_data, a_even, b_even)
                                };
                                store_anchor_pair_nt_off::<$anc>(pn, a, b);
                                store_delta_pair_gpr_off::<$code>($delta, a_even, a_odd, b_even, b_odd);
                                (a, b, a_odd, b_odd, deg)
                            }};
                        }
                        macro_rules! ranked {
                            ($degen:expr, $ones:expr, $k:expr) => {
                                ranked!($degen, $ones, $k, false)
                            };
                            ($degen:expr, $ones:expr, $k:expr, $fused:literal) => {
                                ranked!(@ $degen, $ones, $k, $fused, false)
                            };
                            (@ $degen:expr, $ones:expr, $k:expr, $fused:literal, $head:literal) => {{
                                macro_rules! tacc {
                                    ($acc:expr, $x:expr, $y:expr) => {
                                        if $fused {
                                            tri_acc_q($acc, $x, $y)
                                        } else {
                                            tri_acc_q_crossfused($acc, $x, $y)
                                        }
                                    };
                                }
                                let mut pos_left = b_end - 1;
                                macro_rules! one_group {
                                    () => {{
                                    pos_left -= 1;
                                    // Groups descend by 64 bytes; keep both hints inside this input chunk.
                                    if PREFETCH && first + 2 * pos_left + t >= 4 {
                                        prefetch_sources(pa.wrapping_sub(16), pb.wrapping_sub(16));
                                    }
                                    let delta_group = if BURST { burst.target(pd) } else { pd };
                                    let ((a0, b0, a1c, b1c, deg0), (a2, b2, a3c, b3c, deg1)) =
                                        if $head && head_win && pos_left <= 1 {
                                            let u_in = 2 * pos_left + t;
                                            let pa8 = pa.cast::<u8>();
                                            let pb8 = pb.cast::<u8>();
                                            let pair_h = |code: usize| {
                                                let a_even = u64::from_le(core::ptr::read_unaligned(pa8.add(code).cast::<u64>()));
                                                let a_odd = u64::from_le(core::ptr::read_unaligned(pa8.add(code + 8).cast::<u64>()));
                                                let b_even = u64::from_le(core::ptr::read_unaligned(pb8.add(code).cast::<u64>()));
                                                let b_odd = u64::from_le(core::ptr::read_unaligned(pb8.add(code + 8).cast::<u64>()));
                                                let ones = (b_even & b_odd) == u64::MAX;
                                                let deg = $degen && ones;
                                                let (a, b) = if deg {
                                                    (fold_row_code_q(table_data, a_even), b_ones)
                                                } else {
                                                    fold_two_row_codes_q(table_data, a_even, b_even)
                                                };
                                                ((a, b, a_odd, b_odd, deg), a_even, ones)
                                            };
                                            let (r0, a_even0, ones0) = pair_h(0);
                                            let (r1, a_even1, ones1) = pair_h(16);
                                            store_anchor_pair_nt(
                                                anchors0.add(HC_ANCHORS_PER_BLOCK * (first / 64) + 2 * u_in),
                                                r0.0,
                                                r1.0,
                                            );
                                            store_delta_pair_gpr(
                                                deltas0.add(HC_DELTA_BYTES_PER_BLOCK * (first / 64) + 16 * u_in),
                                                a_even0,
                                                r0.2,
                                                a_even1,
                                                r1.2,
                                            );
                                            head_ok &= ones0 & ones1;
                                            (r0, r1)
                                        } else {
                                            (pair_off!(0, 0, $degen, delta_group), pair_off!(16, 32, $degen, delta_group))
                                        };
                                    if deg0 & deg1 & $ones {
                                        w0.ll = veorq_u64(w0.ll, mul_xb_q(a2, xb, $k));
                                    } else {
                                        let e_a = veorq_u64(a0, a2);
                                        let e_b = veorq_u64(b0, b2);
                                        tacc!(&mut w0, mul_xb_q(a2, xb, $k), b2);
                                        let e_s = mul_xb_q(e_a, xb, $k);
                                        tacc!(&mut w3, e_s, e_b);
                                        let (o_a, o_b) =
                                            fold_two_codes_paired_q(table_data, a1c ^ a3c, b1c ^ b3c);
                                        let o_s = mul_xb_q(o_a, xb, $k);
                                        tacc!(&mut w4, o_s, o_b);
                                        tacc!(&mut w5, veorq_u64(e_s, o_s), veorq_u64(e_b, o_b));
                                    }
                                    xb = vshrq_n_u64::<1>(xb);
                                    pa = pa.wrapping_sub(8);
                                    pb = pb.wrapping_sub(8);
                                    pn = pn.wrapping_sub(8);
                                    pd = pd.wrapping_sub(64);
                                    core::arch::asm!(
                                        "// {0} {1} {2} {3}",
                                        inout(reg) pa,
                                        inout(reg) pb,
                                        inout(reg) pn,
                                        inout(reg) pd,
                                        options(pure, nomem, nostack, preserves_flags),
                                    );
                                    }};
                                }
                                let n_groups = b_end - 1;
                                if $fused {
                                    let mut g = 0usize;
                                    while g + 2 <= n_groups {
                                        one_group!();
                                        one_group!();
                                        g += 2;
                                    }
                                    if g < n_groups {
                                        one_group!();
                                    }
                                } else {
                                    for _ in 0..n_groups {
                                        one_group!();
                                    }
                                }
                            }};
                        }
                        if RANKED_INV {
                            let k: uint64x2_t;
                            core::arch::asm!(
                                "mov {t:w}, #0x87",
                                "dup {k:v}.2d, {t:x}",
                                t = out(reg) _,
                                k = out(vreg) k,
                                options(nomem, nostack, preserves_flags),
                            );
                            match (degen, ones_is_one, r2p_fused_enabled()) {
                                (true, true, true) => ranked!(@ true, true, k, true, true),
                                (true, false, true) => ranked!(true, false, k, true),
                                (false, _, true) => ranked!(false, false, k, true),
                                (true, true, false) => ranked!(true, true, k),
                                (true, false, false) => ranked!(true, false, k),
                                (false, _, false) => ranked!(false, false, k),
                            }
                        } else {
                            ranked!(degen, ones_is_one, k87);
                        }
                    } else {
                        for pos in (0..b_end - 1).rev() {
                            group!(pos, |_: usize, _: usize| (false, false));
                        }
                    }
                } else {
                    for pos in (0..b_end).rev() {
                        group!(pos, generic_pads);
                    }
                }
                let lambda = vld1q_u64(eq_lo.add(2 * (first + t) + 1).cast::<u64>());
                if window_tri && tail_fused {
                    tri_acc_q(&mut sum0, lambda, reduce_tri_binius_xor_q(w0.ll, w0.cross, w0.hh, zero));
                    tri_acc_q(&mut sum3, lambda, reduce_tri_binius_xor_q(w3.ll, w3.cross, w3.hh, zero));
                    tri_acc_q(&mut sum4, lambda, reduce_tri_binius_xor_q(w4.ll, w4.cross, w4.hh, zero));
                    tri_acc_q(&mut sum5, lambda, reduce_tri_binius_xor_q(w5.ll, w5.cross, w5.hh, zero));
                } else if window_tri {
                    tri_acc_q_crossfused(&mut sum0, lambda, reduce_tri_binius_xor_q(w0.ll, w0.cross, w0.hh, zero));
                    tri_acc_q_crossfused(&mut sum3, lambda, reduce_tri_binius_xor_q(w3.ll, w3.cross, w3.hh, zero));
                    tri_acc_q_crossfused(&mut sum4, lambda, reduce_tri_binius_xor_q(w4.ll, w4.cross, w4.hh, zero));
                    tri_acc_q_crossfused(&mut sum5, lambda, reduce_tri_binius_xor_q(w5.ll, w5.cross, w5.hh, zero));
                } else {
                    sum0.ll = veorq_u64(sum0.ll, mul_q(lambda, reduce_tri_q(w0)));
                    sum3.ll = veorq_u64(sum3.ll, mul_q(lambda, reduce_tri_q(w3)));
                    sum4.ll = veorq_u64(sum4.ll, mul_q(lambda, reduce_tri_q(w4)));
                    sum5.ll = veorq_u64(sum5.ll, mul_q(lambda, reduce_tri_q(w5)));
                }
            }
            // Both windows and their two parity walks have filled the block.
            if BURST && (first & 63) == 32 {
                burst.flush();
            }
        }
        if head && !head_ok {
            (*head_fail).store(true, core::sync::atomic::Ordering::Relaxed);
        }
        for i in 0..4 {
            *out.add(i) = F128::ZERO;
        }
        *out.add(4) = core::mem::transmute::<uint64x2_t, F128>(reduce_tri_q(sum0));
        *out.add(5) = core::mem::transmute::<uint64x2_t, F128>(reduce_tri_q(sum3));
        *out.add(6) = core::mem::transmute::<uint64x2_t, F128>(reduce_tri_q(sum4));
        *out.add(7) = core::mem::transmute::<uint64x2_t, F128>(reduce_tri_q(sum5));
    }
}

#[cfg(test)]
mod tests;
