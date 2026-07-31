use crate::field::{F128, F256Unreduced};

/// NEON one-row fold: 8 aligned 16-byte loads + 8 XORs, hand-unrolled for
/// `n_chunks = 8` (the k_skip=6 protocol size). Returns the folded F128.
///
/// The table is `Vec<F128>` with each entry 16-byte aligned (F128 is
/// `repr(C, align(16))`), so every `vld1q_u8` lands on an aligned address.
///
/// # Safety
/// Caller must guarantee `table_data` points to ≥ 8 × 256 × 16 valid bytes
/// (an `n_chunks ≥ 8` table) and `bytes_ptr` to ≥ 8 valid bytes.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) unsafe fn fold_one_row_neon_unchecked_8(
    table_data: *const u8,
    bytes_ptr: *const u8,
) -> F128 {
    use core::arch::aarch64::*;
    unsafe {
        // Constant skip-fibers are common in hash R1CS layouts (especially
        // B=1 identity rows). Their multilinear fold is known without any of
        // the eight indexed table loads.
        let packed = (bytes_ptr as *const u64).read_unaligned();
        if packed == 0 {
            return F128::ZERO;
        }
        if packed == u64::MAX {
            return F128::ONE;
        }

        const STRIDE: usize = 256 * 16;
        let mut acc = vld1q_u8(table_data.add((*bytes_ptr) as usize * 16));
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(1 * STRIDE + (*bytes_ptr.add(1)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(2 * STRIDE + (*bytes_ptr.add(2)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(3 * STRIDE + (*bytes_ptr.add(3)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(4 * STRIDE + (*bytes_ptr.add(4)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(5 * STRIDE + (*bytes_ptr.add(5)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(6 * STRIDE + (*bytes_ptr.add(6)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(7 * STRIDE + (*bytes_ptr.add(7)) as usize * 16)),
        );
        let acc_u64 = vreinterpretq_u64_u8(acc);
        F128 {
            lo: vgetq_lane_u64::<0>(acc_u64),
            hi: vgetq_lane_u64::<1>(acc_u64),
        }
    }
}

/// Vector-returning variant of [`fold_one_row_neon_unchecked_8`]: same
/// constant-fiber early-exits and 8-load XOR tree, but keeps the accumulator
/// in a NEON register so the caller can issue a q-form non-temporal store
/// without a GPR round trip. Early-exit values match the table XORs exactly:
/// an all-zero row folds to 0 (all chunk tables index 0 = 0) and an all-ones
/// row folds to 1 (Lagrange partition of unity over the skip domain).
///
/// # Safety
/// Same contract as [`fold_one_row_neon_unchecked_8`].
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn fold_one_row_neon_vec_8(
    table_data: *const u8,
    bytes_ptr: *const u8,
) -> core::arch::aarch64::uint8x16_t {
    use core::arch::aarch64::*;
    unsafe {
        let packed = (bytes_ptr as *const u64).read_unaligned();
        if packed == 0 {
            return vdupq_n_u8(0);
        }
        if packed == u64::MAX {
            return vec_from_f128(F128::ONE);
        }

        const STRIDE: usize = 256 * 16;
        let mut acc = vld1q_u8(table_data.add((*bytes_ptr) as usize * 16));
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(1 * STRIDE + (*bytes_ptr.add(1)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(2 * STRIDE + (*bytes_ptr.add(2)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(3 * STRIDE + (*bytes_ptr.add(3)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(4 * STRIDE + (*bytes_ptr.add(4)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(5 * STRIDE + (*bytes_ptr.add(5)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(6 * STRIDE + (*bytes_ptr.add(6)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(7 * STRIDE + (*bytes_ptr.add(7)) as usize * 16)),
        );
        acc
    }
}

/// Extract a NEON accumulator into the scalar `F128` the GHASH mul path uses.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn f128_from_vec(v: core::arch::aarch64::uint8x16_t) -> F128 {
    use core::arch::aarch64::*;
    // SAFETY: aarch64 statically guarantees NEON.
    unsafe {
        let v64 = vreinterpretq_u64_u8(v);
        F128 {
            lo: vgetq_lane_u64::<0>(v64),
            hi: vgetq_lane_u64::<1>(v64),
        }
    }
}

/// Reinterpret a scalar `F128` as a NEON register for a q-form store.
/// `F128` is `repr(C, align(16))` `{lo, hi}` little-endian — the same 16
/// bytes the store must publish.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn vec_from_f128(x: F128) -> core::arch::aarch64::uint8x16_t {
    // SAFETY: F128 and uint8x16_t are both plain 16-byte values; the byte
    // layout (lo LE ‖ hi LE) matches the in-memory representation stores use.
    unsafe { core::mem::transmute(x) }
}

/// Non-temporal store of two adjacent F128s (32 B) from NEON registers.
///
/// The round-2 outputs `a_mlv`/`b_mlv` total 2 × 1 GiB and are next read only
/// after a full Fiat–Shamir round trip — far beyond any cache — so bypassing
/// write-allocate skips one DRAM read-for-ownership per destination line
/// (measured: `stnp q,q` 140 GB/s vs `stp` 44.5 GB/s single-core on M4;
/// +36 % at 10 threads). No Rust intrinsic emits `stnp`; raw `asm!`.
///
/// # Safety
/// `dst` must be valid for 32 bytes of writes and 16-byte aligned.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn store_nt_q_pair(
    dst: *mut F128,
    v0: core::arch::aarch64::uint8x16_t,
    v1: core::arch::aarch64::uint8x16_t,
) {
    unsafe {
        core::arch::asm!(
            "stnp {a:q}, {b:q}, [{p}]",
            a = in(vreg) v0,
            b = in(vreg) v1,
            p = in(reg) dst,
            options(nostack, preserves_flags)
        );
    }
}

/// Non-temporal zero of two adjacent F128s (32 B) at `dst` via `xzr` pairs.
///
/// # Safety
/// `dst` must be valid for 32 bytes of writes and 8-byte aligned.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn store_nt_f128_pair_zero(dst: *mut F128) {
    unsafe {
        core::arch::asm!(
            "stnp xzr, xzr, [{p}]",
            "stnp xzr, xzr, [{p}, #16]",
            p = in(reg) dst,
            options(nostack, preserves_flags)
        );
    }
}

/// Raw-pointer leaf for one tail-round `x_hi` chunk: fold `a`/`b` at
/// `r_fold` AND build the next round message from the register values,
/// publishing the folded pairs with `stnp q,q` non-temporal stores.
///
/// The generic tail path (`fold_pairs` + reload loop) stores with
/// write-allocate and then re-reads the just-written pairs to compute the
/// message. For the large early tail rounds the outputs are next read only
/// after a Fiat–Shamir round trip — far beyond any cache — so the
/// write-allocate costs one hidden DRAM read per output line and the
/// cache-hot reload buys nothing. This leaf keeps the folded values in
/// registers for the message terms, so `stnp` becomes legal — the same
/// shape as [`round2_chunk_raw_neon`]'s non-temporal publication.
///
/// Arithmetic is identical to the generic path: the fold uses the same
/// `ghash_mul_vec2_neon` pair fold as `f128_slice::fold_pairs`, the message
/// terms are the same reduced products, and the accumulators use the same
/// unreduced-XOR-then-reduce schedule. Output and message bits match the
/// generic path exactly.
///
/// # Safety
/// Requires the `aes` target feature. `a_in`/`b_in` must be valid for
/// `4 * lo_size` F128 reads, `a_out`/`b_out` for `2 * lo_size` F128 writes
/// (16-byte aligned), and `eq_lo` for `lo_size` F128 reads.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
#[target_feature(enable = "aes")]
pub(crate) unsafe fn tail_fold_chunk_nt_neon(
    a_in: *const F128,
    b_in: *const F128,
    a_out: *mut F128,
    b_out: *mut F128,
    eq_lo: *const F128,
    lo_size: usize,
    r_fold: F128,
) -> (F128, F128) {
    use crate::field::gf2_128::aarch64::ghash_mul_vec2_neon;
    unsafe {
        let mut p1_acc = F256Unreduced::ZERO;
        let mut pinf_acc = F256Unreduced::ZERO;

        let mut src_a = a_in;
        let mut src_b = b_in;
        let mut dst_a = a_out;
        let mut dst_b = b_out;
        let mut eq_ptr = eq_lo;
        let mut remaining = lo_size;

        while remaining != 0 {
            let ae0 = src_a.read();
            let ao0 = src_a.add(1).read();
            let ae1 = src_a.add(2).read();
            let ao1 = src_a.add(3).read();
            let be0 = src_b.read();
            let bo0 = src_b.add(1).read();
            let be1 = src_b.add(2).read();
            let bo1 = src_b.add(3).read();

            // fold(e, o) = e + r_fold · (e ⊕ o), two lanes per array.
            let pa = ghash_mul_vec2_neon(
                [r_fold, r_fold],
                [
                    F128 { lo: ae0.lo ^ ao0.lo, hi: ae0.hi ^ ao0.hi },
                    F128 { lo: ae1.lo ^ ao1.lo, hi: ae1.hi ^ ao1.hi },
                ],
            );
            let pb = ghash_mul_vec2_neon(
                [r_fold, r_fold],
                [
                    F128 { lo: be0.lo ^ bo0.lo, hi: be0.hi ^ bo0.hi },
                    F128 { lo: be1.lo ^ bo1.lo, hi: be1.hi ^ bo1.hi },
                ],
            );
            let a0 = F128 { lo: ae0.lo ^ pa[0].lo, hi: ae0.hi ^ pa[0].hi };
            let a1 = F128 { lo: ae1.lo ^ pa[1].lo, hi: ae1.hi ^ pa[1].hi };
            let b0 = F128 { lo: be0.lo ^ pb[0].lo, hi: be0.hi ^ pb[0].hi };
            let b1 = F128 { lo: be1.lo ^ pb[1].lo, hi: be1.hi ^ pb[1].hi };

            store_nt_q_pair(dst_a, vec_from_f128(a0), vec_from_f128(a1));
            store_nt_q_pair(dst_b, vec_from_f128(b0), vec_from_f128(b1));

            // g1 = a1·b1, g_inf = (a0+a1)(b0+b1) — from registers, no reload.
            let g = ghash_mul_vec2_neon([a1, a0 + a1], [b1, b0 + b1]);
            let eq_l = eq_ptr.read();
            p1_acc ^= eq_l.mul_unreduced(g[0]);
            pinf_acc ^= eq_l.mul_unreduced(g[1]);

            src_a = src_a.add(4);
            src_b = src_b.add(4);
            dst_a = dst_a.add(2);
            dst_b = dst_b.add(2);
            eq_ptr = eq_ptr.add(1);
            remaining -= 1;
        }

        (p1_acc.reduce(), pinf_acc.reduce())
    }
}

// ---------------------------------------------------------------------------
// q-form (all-NEON) tail kernel.
// ---------------------------------------------------------------------------
//
// The scalar-struct tail leaf above keeps `F128 {lo, hi}` values in GPRs:
// every PMULL operand costs a `fmov d, x` transfer in, every folded value
// costs two `fmov x, d` transfers out, and the pair XORs run on the GPR
// side. Disassembly shows ~30 transfer/GPR ops per x_lo — and the big tail
// rounds are µop-bound, not DRAM-bound (n=2^26 round moves 3 GiB in
// ~14.6 ms ≈ 220 GB/s, well under the machine's streaming bandwidth).
//
// The q-form kernel below keeps everything in NEON registers end to end:
// `ldr q` loads, `eor.16b` pair XORs, PMULL/PMULL2 sourced from vector
// lanes (the cross terms use an `ext`-swapped multiplier), the same
// lane-paired vectorised reduction as `ghash_mul_vec2_neon`, and `stnp q,q`
// publication straight from the fold registers. Zero GPR round trips in
// the loop body. Values are bit-identical to the scalar-struct leaf: the
// fold muls, message muls and the unreduced-XOR-then-reduce accumulation
// compute the same GF(2^128) products, which are canonical.

/// `low64(a) · low64(b)` carry-less, operands staying in NEON registers.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn pmull_q_lo(
    a: core::arch::aarch64::uint64x2_t,
    b: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    use core::arch::aarch64::*;
    // SAFETY: caller carries the aes target feature; lane-0 extraction feeding
    // vmull_p64 stays in the FP register file (PMULL d-form).
    unsafe {
        core::mem::transmute::<u128, uint64x2_t>(vmull_p64(
            vgetq_lane_u64::<0>(a),
            vgetq_lane_u64::<0>(b),
        ))
    }
}

/// `high64(a) · high64(b)` carry-less (PMULL2), operands in NEON registers.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn pmull_q_hi(
    a: core::arch::aarch64::uint64x2_t,
    b: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    use core::arch::aarch64::*;
    // SAFETY: caller carries the aes target feature (PMULL2).
    unsafe {
        core::mem::transmute::<u128, uint64x2_t>(vmull_high_p64(
            vreinterpretq_p64_u64(a),
            vreinterpretq_p64_u64(b),
        ))
    }
}

/// Two reduced GHASH muls `(x0·y0, x1·y1)` entirely in q-form. Each operand
/// is an F128 packed `[lo, hi]` in a `uint64x2_t`; `y0_swap`/`y1_swap` must
/// be `vext(y, y, 1)` (caller hoists them when a multiplier is shared).
/// Same 8-PMULL schoolbook + lane-paired shift-XOR reduction as
/// [`ghash_mul_vec2_neon`] — bit-identical results.
///
/// # Safety
/// Requires the `aes` target feature.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn ghash_mul2_q(
    x0: core::arch::aarch64::uint64x2_t,
    y0: core::arch::aarch64::uint64x2_t,
    y0_swap: core::arch::aarch64::uint64x2_t,
    x1: core::arch::aarch64::uint64x2_t,
    y1: core::arch::aarch64::uint64x2_t,
    y1_swap: core::arch::aarch64::uint64x2_t,
) -> (
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
) {
    use core::arch::aarch64::*;
    // SAFETY: caller carries the aes target feature.
    unsafe {
        // 8 independent schoolbook PMULLs (4 per mul).
        let p0_ll = pmull_q_lo(x0, y0);
        let p0_lh = pmull_q_lo(x0, y0_swap); // x0.lo · y0.hi
        let p0_hl = pmull_q_hi(x0, y0_swap); // x0.hi · y0.lo
        let p0_hh = pmull_q_hi(x0, y0);
        let p1_ll = pmull_q_lo(x1, y1);
        let p1_lh = pmull_q_lo(x1, y1_swap);
        let p1_hl = pmull_q_hi(x1, y1_swap);
        let p1_hh = pmull_q_hi(x1, y1);

        // Per-mul cross terms (lh + hl).
        let c0 = veorq_u64(p0_lh, p0_hl);
        let c1 = veorq_u64(p1_lh, p1_hl);

        // Lane-paired (mul0, mul1) layout for each 64-bit word position.
        let r0 = vzip1q_u64(p0_ll, p1_ll);
        let ll_hi = vzip2q_u64(p0_ll, p1_ll);
        let c_lo = vzip1q_u64(c0, c1);
        let r1 = veorq_u64(ll_hi, c_lo);
        let hh_lo = vzip1q_u64(p0_hh, p1_hh);
        let c_hi = vzip2q_u64(c0, c1);
        let r2 = veorq_u64(hh_lo, c_hi);
        let r3 = vzip2q_u64(p0_hh, p1_hh);

        // Vectorised GHASH reduction (identical to ghash_mul_vec2_neon).
        let s1_lo = vshlq_n_u64::<1>(r2);
        let s1_hi = veorq_u64(vshlq_n_u64::<1>(r3), vshrq_n_u64::<63>(r2));
        let s2_lo = vshlq_n_u64::<2>(r2);
        let s2_hi = veorq_u64(vshlq_n_u64::<2>(r3), vshrq_n_u64::<62>(r2));
        let s7_lo = vshlq_n_u64::<7>(r2);
        let s7_hi = veorq_u64(vshlq_n_u64::<7>(r3), vshrq_n_u64::<57>(r2));

        let t_lo = veorq_u64(veorq_u64(r2, s1_lo), veorq_u64(s2_lo, s7_lo));
        let t_hi = veorq_u64(veorq_u64(r3, s1_hi), veorq_u64(s2_hi, s7_hi));

        let ov = veorq_u64(
            veorq_u64(vshrq_n_u64::<63>(r3), vshrq_n_u64::<62>(r3)),
            vshrq_n_u64::<57>(r3),
        );
        let corr = veorq_u64(
            veorq_u64(ov, vshlq_n_u64::<1>(ov)),
            veorq_u64(vshlq_n_u64::<2>(ov), vshlq_n_u64::<7>(ov)),
        );

        let final_lo = veorq_u64(veorq_u64(r0, t_lo), corr);
        let final_hi = veorq_u64(r1, t_hi);

        // Unpack lane-paired → per-mul [lo, hi] q registers.
        (
            vzip1q_u64(final_lo, final_hi),
            vzip2q_u64(final_lo, final_hi),
        )
    }
}

/// Unreduced 256-bit carry-less `x · y` XOR-accumulated into
/// `(acc_lo = [r0, r1], acc_hi = [r2, r3])` — the q-form equivalent of
/// `acc ^= x.mul_unreduced(y)`. `y_swap` must be `vext(y, y, 1)`.
///
/// # Safety
/// Requires the `aes` target feature.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn mul_unred_acc_q(
    x: core::arch::aarch64::uint64x2_t,
    y: core::arch::aarch64::uint64x2_t,
    y_swap: core::arch::aarch64::uint64x2_t,
    acc_lo: &mut core::arch::aarch64::uint64x2_t,
    acc_hi: &mut core::arch::aarch64::uint64x2_t,
) {
    use core::arch::aarch64::*;
    // SAFETY: caller carries the aes target feature.
    unsafe {
        let ll = pmull_q_lo(x, y);
        let lh = pmull_q_lo(x, y_swap);
        let hl = pmull_q_hi(x, y_swap);
        let hh = pmull_q_hi(x, y);
        let cross = veorq_u64(lh, hl);
        let zero = vdupq_n_u64(0);
        // r0 = ll.lo, r1 = ll.hi ^ cross.lo, r2 = hh.lo ^ cross.hi, r3 = hh.hi.
        *acc_lo = veorq_u64(*acc_lo, veorq_u64(ll, vextq_u64::<1>(zero, cross)));
        *acc_hi = veorq_u64(*acc_hi, veorq_u64(hh, vextq_u64::<1>(cross, zero)));
    }
}

/// Regular (write-allocate) store of two adjacent F128s from NEON registers —
/// the cache-friendly counterpart of [`store_nt_q_pair`] for rounds whose
/// output is small enough to be re-read from cache next round.
///
/// # Safety
/// `dst` must be valid for 32 bytes of writes and 16-byte aligned.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn store_q_pair(
    dst: *mut F128,
    v0: core::arch::aarch64::uint8x16_t,
    v1: core::arch::aarch64::uint8x16_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        vst1q_u8(dst as *mut u8, v0);
        vst1q_u8(dst.add(1) as *mut u8, v1);
    }
}

/// Non-temporal load of two adjacent F128s (32 B) into NEON registers via
/// `ldnp`. The tail-round inputs are streamed exactly once (the folded
/// output supersedes them), so the no-allocate hint spares SLC capacity for
/// the eq tables and co-running phases. `pure`/`readonly` keeps the
/// scheduler free to hoist the loads.
///
/// # Safety
/// `src` must be valid for 32 bytes of reads and 16-byte aligned.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn load_nt_q_pair(
    src: *const u64,
) -> (
    core::arch::aarch64::uint64x2_t,
    core::arch::aarch64::uint64x2_t,
) {
    let v0: core::arch::aarch64::uint64x2_t;
    let v1: core::arch::aarch64::uint64x2_t;
    unsafe {
        core::arch::asm!(
            "ldnp {a:q}, {b:q}, [{p}]",
            a = out(vreg) v0,
            b = out(vreg) v1,
            p = in(reg) src,
            options(pure, readonly, nostack, preserves_flags)
        );
    }
    (v0, v1)
}

/// q-form rewrite of [`tail_fold_chunk_nt_neon`]: identical I/O contract,
/// identical output bits, ~zero GPR↔NEON traffic in the loop body. See the
/// module comment above the helpers for the rationale.
///
/// `NT_STORE` selects `stnp` publication (large rounds, output beyond cache
/// reach) vs regular stores (mid rounds, output re-read from cache next
/// round). `NT_LOAD` selects `ldnp` input streaming (inputs are dead after
/// the fold — hint the caches not to keep them).
///
/// # Safety
/// Same contract as [`tail_fold_chunk_nt_neon`]: `a_in`/`b_in` valid for
/// `4 * lo_size` F128 reads, `a_out`/`b_out` for `2 * lo_size` F128 writes
/// (16-byte aligned), `eq_lo` for `lo_size` F128 reads, `aes` enabled.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
#[target_feature(enable = "aes")]
pub(crate) unsafe fn tail_fold_chunk_q<const NT_STORE: bool, const NT_LOAD: bool>(
    a_in: *const F128,
    b_in: *const F128,
    a_out: *mut F128,
    b_out: *mut F128,
    eq_lo: *const F128,
    lo_size: usize,
    r_fold: F128,
) -> (F128, F128) {
    use core::arch::aarch64::*;
    unsafe {
        // F128 is repr(C, align(16)) {lo, hi} little-endian → lanes [lo, hi].
        let vr = core::mem::transmute::<F128, uint64x2_t>(r_fold);
        let vr_swap = vextq_u64::<1>(vr, vr);

        let mut p1_lo = vdupq_n_u64(0);
        let mut p1_hi = vdupq_n_u64(0);
        let mut pinf_lo = vdupq_n_u64(0);
        let mut pinf_hi = vdupq_n_u64(0);

        let mut src_a = a_in as *const u64;
        let mut src_b = b_in as *const u64;
        let mut dst_a = a_out;
        let mut dst_b = b_out;
        let mut eq_ptr = eq_lo as *const u64;
        let mut remaining = lo_size;

        while remaining != 0 {
            let (ae0, ao0) = if NT_LOAD {
                load_nt_q_pair(src_a)
            } else {
                (vld1q_u64(src_a), vld1q_u64(src_a.add(2)))
            };
            let (ae1, ao1) = if NT_LOAD {
                load_nt_q_pair(src_a.add(4))
            } else {
                (vld1q_u64(src_a.add(4)), vld1q_u64(src_a.add(6)))
            };
            let (be0, bo0) = if NT_LOAD {
                load_nt_q_pair(src_b)
            } else {
                (vld1q_u64(src_b), vld1q_u64(src_b.add(2)))
            };
            let (be1, bo1) = if NT_LOAD {
                load_nt_q_pair(src_b.add(4))
            } else {
                (vld1q_u64(src_b.add(4)), vld1q_u64(src_b.add(6)))
            };

            // fold(e, o) = e + r_fold · (e ⊕ o), two lanes per array.
            let da0 = veorq_u64(ae0, ao0);
            let da1 = veorq_u64(ae1, ao1);
            let (pa0, pa1) = ghash_mul2_q(da0, vr, vr_swap, da1, vr, vr_swap);
            let a0 = veorq_u64(ae0, pa0);
            let a1 = veorq_u64(ae1, pa1);

            let db0 = veorq_u64(be0, bo0);
            let db1 = veorq_u64(be1, bo1);
            let (pb0, pb1) = ghash_mul2_q(db0, vr, vr_swap, db1, vr, vr_swap);
            let b0 = veorq_u64(be0, pb0);
            let b1 = veorq_u64(be1, pb1);

            if NT_STORE {
                store_nt_q_pair(dst_a, vreinterpretq_u8_u64(a0), vreinterpretq_u8_u64(a1));
                store_nt_q_pair(dst_b, vreinterpretq_u8_u64(b0), vreinterpretq_u8_u64(b1));
            } else {
                store_q_pair(dst_a, vreinterpretq_u8_u64(a0), vreinterpretq_u8_u64(a1));
                store_q_pair(dst_b, vreinterpretq_u8_u64(b0), vreinterpretq_u8_u64(b1));
            }

            // g1 = a1·b1, g_inf = (a0+a1)(b0+b1) — straight from registers.
            let sa = veorq_u64(a0, a1);
            let sb = veorq_u64(b0, b1);
            let b1_swap = vextq_u64::<1>(b1, b1);
            let sb_swap = vextq_u64::<1>(sb, sb);
            let (g1, g_inf) = ghash_mul2_q(a1, b1, b1_swap, sa, sb, sb_swap);

            let eq_l = vld1q_u64(eq_ptr);
            let eq_swap = vextq_u64::<1>(eq_l, eq_l);
            mul_unred_acc_q(g1, eq_l, eq_swap, &mut p1_lo, &mut p1_hi);
            mul_unred_acc_q(g_inf, eq_l, eq_swap, &mut pinf_lo, &mut pinf_hi);

            src_a = src_a.add(8);
            src_b = src_b.add(8);
            dst_a = dst_a.add(2);
            dst_b = dst_b.add(2);
            eq_ptr = eq_ptr.add(2);
            remaining -= 1;
        }

        let p1 = F256Unreduced {
            r0: vgetq_lane_u64::<0>(p1_lo),
            r1: vgetq_lane_u64::<1>(p1_lo),
            r2: vgetq_lane_u64::<0>(p1_hi),
            r3: vgetq_lane_u64::<1>(p1_hi),
        }
        .reduce();
        let pinf = F256Unreduced {
            r0: vgetq_lane_u64::<0>(pinf_lo),
            r1: vgetq_lane_u64::<1>(pinf_lo),
            r2: vgetq_lane_u64::<0>(pinf_hi),
            r3: vgetq_lane_u64::<1>(pinf_hi),
        }
        .reduce();
        (p1, pinf)
    }
}

/// Raw-pointer leaf for one round-2 `x_hi` chunk at the protocol-fixed
/// `k_skip = 6` / eight-table-chunk geometry. Keeping the loop in a noinline
/// leaf frees the surrounding Rayon closure's capture state from the hot
/// table-lookup and GHASH dependency chains.
///
/// The outputs `a_mlv`/`b_mlv` (2 × 1 GiB at the ranked shape) are next read
/// only after a Fiat–Shamir round trip, so they are published with q-form
/// non-temporal stores (no write-allocate). The message terms use the
/// register values, and the constant-fiber shortcuts are preserved: the row
/// fold early-exits on all-zero/all-ones rows, and a `b ≡ 1` pair skips the
/// message muls (`b(1) = 1`, `b(∞) = 0` in characteristic two).
///
/// # Safety
/// Requires the `aes` target feature and valid ranges for `2 * lo_size` packed
/// rows and output elements, plus `lo_size` initialized equality weights.
#[allow(clippy::too_many_arguments)]
#[cfg(target_arch = "aarch64")]
#[inline(never)]
#[target_feature(enable = "aes")]
pub(crate) unsafe fn round2_chunk_raw_neon(
    table_data: *const u8,
    a_packed: *const u8,
    b_packed: *const u8,
    a_out: *mut F128,
    b_out: *mut F128,
    eq_lo: *const F128,
    lo_size: usize,
    pair_idx_base: usize,
    pair_in_block_mask: usize,
    useful_pairs_inclusive: usize,
) -> (F128, F128) {
    unsafe {
        let mut p1_acc = F256Unreduced::ZERO;
        let mut pinf_acc = F256Unreduced::ZERO;

        let mut a_src = a_packed;
        let mut b_src = b_packed;
        let mut a_dst = a_out;
        let mut b_dst = b_out;
        let mut eq_ptr = eq_lo;
        let mut pair_idx = pair_idx_base;
        let mut remaining = lo_size;

        while remaining != 0 {
            if (pair_idx & pair_in_block_mask) >= useful_pairs_inclusive {
                store_nt_f128_pair_zero(a_dst);
                store_nt_f128_pair_zero(b_dst);
            } else {
                let a0v = fold_one_row_neon_vec_8(table_data, a_src);
                let b0v = fold_one_row_neon_vec_8(table_data, b_src);
                let a1v = fold_one_row_neon_vec_8(table_data, a_src.add(8));
                let b1v = fold_one_row_neon_vec_8(table_data, b_src.add(8));

                store_nt_q_pair(a_dst, a0v, a1v);
                store_nt_q_pair(b_dst, b0v, b1v);

                let a0 = f128_from_vec(a0v);
                let b0 = f128_from_vec(b0v);
                let a1 = f128_from_vec(a1v);
                let b1 = f128_from_vec(b1v);

                let eq_l = eq_ptr.read();
                if b0 == F128::ONE && b1 == F128::ONE {
                    // b(1)=1 and b(∞)=b(0)+b(1)=0 in characteristic two.
                    p1_acc ^= eq_l.mul_unreduced(a1);
                } else {
                    p1_acc ^= eq_l.mul_unreduced(a1 * b1);
                    pinf_acc ^= eq_l.mul_unreduced((a0 + a1) * (b0 + b1));
                }
            }

            a_src = a_src.add(16);
            b_src = b_src.add(16);
            a_dst = a_dst.add(2);
            b_dst = b_dst.add(2);
            eq_ptr = eq_ptr.add(1);
            pair_idx += 1;
            remaining -= 1;
        }

        (p1_acc.reduce(), pinf_acc.reduce())
    }
}

/// q-form message variant of [`round2_chunk_raw_neon`]: identical fold and
/// publication (same table lookups, same `stnp`), but the message terms are
/// computed without leaving the NEON register file — the scalar leaf above
/// extracts all four folded values to GPRs (8 `fmov`) and runs GPR-resident
/// GHASH muls. The `b ≡ 1` constant-fiber shortcut is preserved via a NEON
/// compare + `umaxv`. Output and message bits match the scalar leaf exactly.
///
/// # Safety
/// Same contract as [`round2_chunk_raw_neon`].
#[allow(clippy::too_many_arguments)]
#[cfg(target_arch = "aarch64")]
#[inline(never)]
#[target_feature(enable = "aes")]
pub(crate) unsafe fn round2_chunk_raw_neon_q(
    table_data: *const u8,
    a_packed: *const u8,
    b_packed: *const u8,
    a_out: *mut F128,
    b_out: *mut F128,
    eq_lo: *const F128,
    lo_size: usize,
    pair_idx_base: usize,
    pair_in_block_mask: usize,
    useful_pairs_inclusive: usize,
) -> (F128, F128) {
    use core::arch::aarch64::*;
    unsafe {
        let one_v = vreinterpretq_u64_u8(vec_from_f128(F128::ONE));

        let mut p1_lo = vdupq_n_u64(0);
        let mut p1_hi = vdupq_n_u64(0);
        let mut pinf_lo = vdupq_n_u64(0);
        let mut pinf_hi = vdupq_n_u64(0);

        let mut a_src = a_packed;
        let mut b_src = b_packed;
        let mut a_dst = a_out;
        let mut b_dst = b_out;
        let mut eq_ptr = eq_lo as *const u64;
        let mut pair_idx = pair_idx_base;
        let mut remaining = lo_size;

        while remaining != 0 {
            if (pair_idx & pair_in_block_mask) >= useful_pairs_inclusive {
                store_nt_f128_pair_zero(a_dst);
                store_nt_f128_pair_zero(b_dst);
            } else {
                let a0v = fold_one_row_neon_vec_8(table_data, a_src);
                let b0v = fold_one_row_neon_vec_8(table_data, b_src);
                let a1v = fold_one_row_neon_vec_8(table_data, a_src.add(8));
                let b1v = fold_one_row_neon_vec_8(table_data, b_src.add(8));

                store_nt_q_pair(a_dst, a0v, a1v);
                store_nt_q_pair(b_dst, b0v, b1v);

                let a0 = vreinterpretq_u64_u8(a0v);
                let a1 = vreinterpretq_u64_u8(a1v);
                let b0 = vreinterpretq_u64_u8(b0v);
                let b1 = vreinterpretq_u64_u8(b1v);

                let eq_l = vld1q_u64(eq_ptr);
                let eq_swap = vextq_u64::<1>(eq_l, eq_l);
                // b ≡ 1 shortcut: (b0 ^ 1) | (b1 ^ 1) == 0.
                let b_dev = vorrq_u64(veorq_u64(b0, one_v), veorq_u64(b1, one_v));
                if vmaxvq_u32(vreinterpretq_u32_u64(b_dev)) == 0 {
                    // b(1)=1 and b(∞)=b(0)+b(1)=0 in characteristic two.
                    mul_unred_acc_q(a1, eq_l, eq_swap, &mut p1_lo, &mut p1_hi);
                } else {
                    let sa = veorq_u64(a0, a1);
                    let sb = veorq_u64(b0, b1);
                    let b1_swap = vextq_u64::<1>(b1, b1);
                    let sb_swap = vextq_u64::<1>(sb, sb);
                    let (g1, g_inf) = ghash_mul2_q(a1, b1, b1_swap, sa, sb, sb_swap);
                    mul_unred_acc_q(g1, eq_l, eq_swap, &mut p1_lo, &mut p1_hi);
                    mul_unred_acc_q(g_inf, eq_l, eq_swap, &mut pinf_lo, &mut pinf_hi);
                }
            }

            a_src = a_src.add(16);
            b_src = b_src.add(16);
            a_dst = a_dst.add(2);
            b_dst = b_dst.add(2);
            eq_ptr = eq_ptr.add(2);
            pair_idx += 1;
            remaining -= 1;
        }

        let p1 = F256Unreduced {
            r0: vgetq_lane_u64::<0>(p1_lo),
            r1: vgetq_lane_u64::<1>(p1_lo),
            r2: vgetq_lane_u64::<0>(p1_hi),
            r3: vgetq_lane_u64::<1>(p1_hi),
        }
        .reduce();
        let pinf = F256Unreduced {
            r0: vgetq_lane_u64::<0>(pinf_lo),
            r1: vgetq_lane_u64::<1>(pinf_lo),
            r2: vgetq_lane_u64::<0>(pinf_hi),
            r3: vgetq_lane_u64::<1>(pinf_hi),
        }
        .reduce();
        (p1, pinf)
    }
}
