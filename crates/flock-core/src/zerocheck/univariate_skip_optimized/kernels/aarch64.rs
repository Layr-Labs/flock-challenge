use super::super::{F8, F128, InvNttTableByteSingleGf8, N_CHUNKS};

/// Four-lane convert-table fold.
///
/// Mirrors [`accumulate_convert_with_s_hat_v`]: four lanes are processed
/// together so eight independent gather/XOR chains (4 lanes × 2 banks) are in
/// flight, versus the single chain the previous one-lane-at-a-time loop
/// exposed. Each convert-table gather is an L1-resident dependent load whose
/// latency the one-chain form could not hide; the profiled gather and ALU arms
/// overlap only ~59%, so the win is overlap, not fewer operations.
///
/// Bit-exactness: each lane keeps its own accumulator and still XORs exactly
/// the same table rows in the same `b_med` order, then is scaled by the same
/// `eq_lo_val` and added into the same `partial_*` slot. Only the interleaving
/// between independent lanes changes, so every accumulator is bit-identical.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) unsafe fn accumulate_convert(
    chunk_ab_bytes: &[[u8; 64]; 16],
    chunk_c_bytes: &[[u8; 64]; 16],
    n_b_med: usize,
    convert: &[F128],
    eq_lo_val: F128,
    partial_ab: &mut [F128; 64],
    partial_c: &mut [F128; 64],
) {
    use core::arch::aarch64::*;

    // SAFETY: caller guarantees fixed input sizes and aarch64 provides NEON.
    unsafe {
        let convert_ptr = convert.as_ptr() as *const u8;
        for lane in (0..64).step_by(4) {
            // 8 live accumulator q-registers (4 lanes × 2 banks) leave ample
            // headroom in the 32-register file for the gathers in flight and
            // their address temporaries, so nothing spills.
            let mut ab0 = vdupq_n_u8(0);
            let mut ab1 = vdupq_n_u8(0);
            let mut ab2 = vdupq_n_u8(0);
            let mut ab3 = vdupq_n_u8(0);
            let mut c0 = vdupq_n_u8(0);
            let mut c1 = vdupq_n_u8(0);
            let mut c2 = vdupq_n_u8(0);
            let mut c3 = vdupq_n_u8(0);
            for pair in 0..n_b_med / 2 {
                let b0 = 2 * pair;
                let b1 = b0 + 1;
                let table0 = convert_ptr.add(b0 * 256 * 16);
                let table1 = convert_ptr.add(b1 * 256 * 16);
                let ab_word0 =
                    (chunk_ab_bytes[b0].as_ptr().add(lane) as *const u32).read_unaligned() as usize;
                let ab_word1 =
                    (chunk_ab_bytes[b1].as_ptr().add(lane) as *const u32).read_unaligned() as usize;
                let c_word0 =
                    (chunk_c_bytes[b0].as_ptr().add(lane) as *const u32).read_unaligned() as usize;
                let c_word1 =
                    (chunk_c_bytes[b1].as_ptr().add(lane) as *const u32).read_unaligned() as usize;
                ab0 = xor3_u8(
                    ab0,
                    vld1q_u8(table0.add((ab_word0 & 0xff) * 16)),
                    vld1q_u8(table1.add((ab_word1 & 0xff) * 16)),
                );
                ab1 = xor3_u8(
                    ab1,
                    vld1q_u8(table0.add(((ab_word0 >> 8) & 0xff) * 16)),
                    vld1q_u8(table1.add(((ab_word1 >> 8) & 0xff) * 16)),
                );
                ab2 = xor3_u8(
                    ab2,
                    vld1q_u8(table0.add(((ab_word0 >> 16) & 0xff) * 16)),
                    vld1q_u8(table1.add(((ab_word1 >> 16) & 0xff) * 16)),
                );
                ab3 = xor3_u8(
                    ab3,
                    vld1q_u8(table0.add((ab_word0 >> 24) * 16)),
                    vld1q_u8(table1.add((ab_word1 >> 24) * 16)),
                );
                c0 = xor3_u8(
                    c0,
                    vld1q_u8(table0.add((c_word0 & 0xff) * 16)),
                    vld1q_u8(table1.add((c_word1 & 0xff) * 16)),
                );
                c1 = xor3_u8(
                    c1,
                    vld1q_u8(table0.add(((c_word0 >> 8) & 0xff) * 16)),
                    vld1q_u8(table1.add(((c_word1 >> 8) & 0xff) * 16)),
                );
                c2 = xor3_u8(
                    c2,
                    vld1q_u8(table0.add(((c_word0 >> 16) & 0xff) * 16)),
                    vld1q_u8(table1.add(((c_word1 >> 16) & 0xff) * 16)),
                );
                c3 = xor3_u8(
                    c3,
                    vld1q_u8(table0.add((c_word0 >> 24) * 16)),
                    vld1q_u8(table1.add((c_word1 >> 24) * 16)),
                );
            }
            if n_b_med & 1 == 1 {
                let b_med = n_b_med - 1;
                let table = convert_ptr.add(b_med * 256 * 16);
                let ab_word = (chunk_ab_bytes[b_med].as_ptr().add(lane) as *const u32)
                    .read_unaligned() as usize;
                let c_word = (chunk_c_bytes[b_med].as_ptr().add(lane) as *const u32)
                    .read_unaligned() as usize;
                ab0 = veorq_u8(ab0, vld1q_u8(table.add((ab_word & 0xff) * 16)));
                ab1 = veorq_u8(ab1, vld1q_u8(table.add(((ab_word >> 8) & 0xff) * 16)));
                ab2 = veorq_u8(ab2, vld1q_u8(table.add(((ab_word >> 16) & 0xff) * 16)));
                ab3 = veorq_u8(ab3, vld1q_u8(table.add((ab_word >> 24) * 16)));
                c0 = veorq_u8(c0, vld1q_u8(table.add((c_word & 0xff) * 16)));
                c1 = veorq_u8(c1, vld1q_u8(table.add(((c_word >> 8) & 0xff) * 16)));
                c2 = veorq_u8(c2, vld1q_u8(table.add(((c_word >> 16) & 0xff) * 16)));
                c3 = veorq_u8(c3, vld1q_u8(table.add((c_word >> 24) * 16)));
            }

            macro_rules! drain_lane {
                ($offset:literal, $ab:ident, $c:ident) => {{
                    let ab = vreinterpretq_u64_u8($ab);
                    let c = vreinterpretq_u64_u8($c);
                    partial_ab[lane + $offset] += F128 {
                        lo: vgetq_lane_u64::<0>(ab),
                        hi: vgetq_lane_u64::<1>(ab),
                    } * eq_lo_val;
                    partial_c[lane + $offset] += F128 {
                        lo: vgetq_lane_u64::<0>(c),
                        hi: vgetq_lane_u64::<1>(c),
                    } * eq_lo_val;
                }};
            }
            drain_lane!(0, ab0, c0);
            drain_lane!(1, ab1, c1);
            drain_lane!(2, ab2, c2);
            drain_lane!(3, ab3, c3);
        }
    }
}

/// AB half of the eight-bank C variant.
///
/// The C side of this kernel used to run alongside the AB gathers: two masked
/// convert-table banks (`c & 0x55` / `c & 0xaa`), halved by the `b_med` pairing
/// tables. It is gone — the eight α-free single-bit-`K` banks are pure u16 bit
/// masks over the 16 `b_med` C bytes and need no table at all
/// (see [`super::accumulate_c_banks`]). What remains here is the incumbent AB
/// arm widened to 8 lanes per group, with two adjacent `b_med` folded into one
/// EOR3. Eight independent gather chains fit now that the removed C side no
/// longer shares the register file.
#[inline(always)]
pub(crate) unsafe fn accumulate_convert_ab(
    chunk_ab_bytes: &[[u8; 64]; 16],
    first_b_med: usize,
    n_b_med: usize,
    convert: &[F128],
    eq_lo_val: F128,
    partial_ab: &mut [F128; 64],
) {
    use core::arch::aarch64::*;

    debug_assert!(first_b_med <= n_b_med && n_b_med <= 16);
    debug_assert_eq!(first_b_med & 1, 0, "paired AB drain requires an even start");
    debug_assert_eq!(convert.len(), 16 * 256);

    // SAFETY: caller guarantees fixed input sizes and aarch64 provides NEON.
    // Every table offset below is a `u8` index scaled by the 16-byte row size,
    // bounded by the debug-asserted table length: `b_med < 16` selects one of
    // 16 convert blocks.
    unsafe {
        let convert_ptr = convert.as_ptr() as *const u8;
        let n_pairs = (n_b_med - first_b_med) / 2;
        for lane in (0..64).step_by(8) {
            // 8 live accumulator q-registers (8 lanes × 1 bank); the removed
            // C side leaves room for the gather temporaries.
            let mut ab0 = vdupq_n_u8(0);
            let mut ab1 = vdupq_n_u8(0);
            let mut ab2 = vdupq_n_u8(0);
            let mut ab3 = vdupq_n_u8(0);
            let mut ab4 = vdupq_n_u8(0);
            let mut ab5 = vdupq_n_u8(0);
            let mut ab6 = vdupq_n_u8(0);
            let mut ab7 = vdupq_n_u8(0);

            for p in 0..n_pairs {
                let (b_even, b_odd) = (first_b_med + 2 * p, first_b_med + 2 * p + 1);
                let t_even = convert_ptr.add(b_even * 256 * 16);
                let t_odd = convert_ptr.add(b_odd * 256 * 16);

                // One u64 load per b_med covers the 8 adjacent lanes; each
                // extracted byte addresses the same table row as the original
                // byte-load form.
                let we = (chunk_ab_bytes[b_even].as_ptr().add(lane) as *const u64).read_unaligned()
                    as usize;
                let wo = (chunk_ab_bytes[b_odd].as_ptr().add(lane) as *const u64).read_unaligned()
                    as usize;
                ab0 = xor3_u8(
                    ab0,
                    vld1q_u8(t_even.add((we & 0xff) * 16)),
                    vld1q_u8(t_odd.add((wo & 0xff) * 16)),
                );
                ab1 = xor3_u8(
                    ab1,
                    vld1q_u8(t_even.add(((we >> 8) & 0xff) * 16)),
                    vld1q_u8(t_odd.add(((wo >> 8) & 0xff) * 16)),
                );
                ab2 = xor3_u8(
                    ab2,
                    vld1q_u8(t_even.add(((we >> 16) & 0xff) * 16)),
                    vld1q_u8(t_odd.add(((wo >> 16) & 0xff) * 16)),
                );
                ab3 = xor3_u8(
                    ab3,
                    vld1q_u8(t_even.add(((we >> 24) & 0xff) * 16)),
                    vld1q_u8(t_odd.add(((wo >> 24) & 0xff) * 16)),
                );
                ab4 = xor3_u8(
                    ab4,
                    vld1q_u8(t_even.add(((we >> 32) & 0xff) * 16)),
                    vld1q_u8(t_odd.add(((wo >> 32) & 0xff) * 16)),
                );
                ab5 = xor3_u8(
                    ab5,
                    vld1q_u8(t_even.add(((we >> 40) & 0xff) * 16)),
                    vld1q_u8(t_odd.add(((wo >> 40) & 0xff) * 16)),
                );
                ab6 = xor3_u8(
                    ab6,
                    vld1q_u8(t_even.add(((we >> 48) & 0xff) * 16)),
                    vld1q_u8(t_odd.add(((wo >> 48) & 0xff) * 16)),
                );
                ab7 = xor3_u8(
                    ab7,
                    vld1q_u8(t_even.add((we >> 56) * 16)),
                    vld1q_u8(t_odd.add((wo >> 56) * 16)),
                );
            }

            // Odd trailing b_med: unpaired fallback.
            if (n_b_med - first_b_med) & 1 == 1 {
                let b_med = n_b_med - 1;
                let table = convert_ptr.add(b_med * 256 * 16);
                let wa = (chunk_ab_bytes[b_med].as_ptr().add(lane) as *const u64).read_unaligned()
                    as usize;
                ab0 = veorq_u8(ab0, vld1q_u8(table.add((wa & 0xff) * 16)));
                ab1 = veorq_u8(ab1, vld1q_u8(table.add(((wa >> 8) & 0xff) * 16)));
                ab2 = veorq_u8(ab2, vld1q_u8(table.add(((wa >> 16) & 0xff) * 16)));
                ab3 = veorq_u8(ab3, vld1q_u8(table.add(((wa >> 24) & 0xff) * 16)));
                ab4 = veorq_u8(ab4, vld1q_u8(table.add(((wa >> 32) & 0xff) * 16)));
                ab5 = veorq_u8(ab5, vld1q_u8(table.add(((wa >> 40) & 0xff) * 16)));
                ab6 = veorq_u8(ab6, vld1q_u8(table.add(((wa >> 48) & 0xff) * 16)));
                ab7 = veorq_u8(ab7, vld1q_u8(table.add((wa >> 56) * 16)));
            }

            macro_rules! drain_lane {
                ($offset:literal, $ab:ident) => {{
                    let ab = vreinterpretq_u64_u8($ab);
                    partial_ab[lane + $offset] += F128 {
                        lo: vgetq_lane_u64::<0>(ab),
                        hi: vgetq_lane_u64::<1>(ab),
                    } * eq_lo_val;
                }};
            }
            drain_lane!(0, ab0);
            drain_lane!(1, ab1);
            drain_lane!(2, ab2);
            drain_lane!(3, ab3);
            drain_lane!(4, ab4);
            drain_lane!(5, ab5);
            drain_lane!(6, ab6);
            drain_lane!(7, ab7);
        }
    }
}

/// Entry `(w >> SHIFT) & 0xff` of a 256 x 16-byte table. `U = false` is the
/// plain Rust form (`ubfx` + scaled-index load, three issue slots on Apple
/// performance cores); `U = true` forms the byte offset in one instruction
/// (`ubfiz`, or `and` with a shifted register against `0xff0`) and uses the
/// unscaled register-offset load, one slot less per lookup. Same entry.
#[inline(always)]
unsafe fn ab_ld<const U: bool, const SHIFT: u32>(
    t: *const u8,
    w: usize,
) -> core::arch::aarch64::uint8x16_t {
    use core::arch::aarch64::*;
    unsafe {
        if !U {
            return vld1q_u8(t.add(((w >> SHIFT) & 0xff) * 16));
        }
        let out: uint8x16_t;
        // SAFETY: the offset is a byte value times 16 (at most 4080), inside
        // the caller's 4 KiB table; the asm only reads that memory.
        if SHIFT == 0 {
            core::arch::asm!(
                "ubfiz {idx}, {w}, #4, #8",
                "ldr {out:q}, [{t}, {idx}]",
                w = in(reg) w,
                t = in(reg) t,
                idx = out(reg) _,
                out = lateout(vreg) out,
                options(pure, readonly, nostack, preserves_flags),
            );
        } else {
            core::arch::asm!(
                "and {idx}, {mask}, {w}, lsr #{shift}",
                "ldr {out:q}, [{t}, {idx}]",
                shift = const (SHIFT + 60) % 64,
                mask = in(reg) 0xff0usize,
                w = in(reg) w,
                t = in(reg) t,
                idx = out(reg) _,
                out = lateout(vreg) out,
                options(pure, readonly, nostack, preserves_flags),
            );
        }
        out
    }
}

/// Multiply-free twin of [`accumulate_convert_ab`] for the banked `eq_lo`
/// tensor fold.
///
/// Identical gathers, identical `b_med` pairing — only the drain changes. The
/// `eq_lo` scale no longer rides on the accumulator: its `x_lo`-high factor is
/// pre-multiplied into the convert table the caller selects
/// (`T_w[i] = convert[i] · eq_top[w]`) and its `x_lo`-low factor is applied
/// once per band to the bank this call XORs into. So the eight lanes drain as
/// a plain 128-bit EOR into memory and the kernel issues no GHASH multiply at
/// all.
///
/// Bit-exactness: F₁₂₈ multiplication is F₂-bilinear, so scaling every table
/// row by `eq_top[w]` before the XOR chain equals scaling the chain's result
/// after it, and the deferred `eq_bot[u]` fold re-associates the same product
/// terms — no term is added, dropped, or reordered within a lane.
///
#[inline(always)]
pub(crate) unsafe fn accumulate_convert_ab_nomul<const STORE: bool, const U: bool>(
    chunk_ab_bytes: &[[u8; 64]; 16],
    first_b_med: usize,
    n_b_med: usize,
    convert: &[F128],
    bank: &mut [F128; 64],
) {
    use core::arch::aarch64::*;

    debug_assert!(first_b_med <= n_b_med && n_b_med <= 16);
    debug_assert_eq!(first_b_med & 1, 0, "paired AB drain requires an even start");
    debug_assert_eq!(convert.len(), 16 * 256);

    // SAFETY: caller guarantees fixed input sizes and aarch64 provides NEON.
    // Table offsets are `u8` indices scaled by the 16-byte row size within the
    // debug-asserted table length, and every drain address is one 16-byte
    // `F128` slot of the fixed-size bank.
    unsafe {
        let convert_ptr = convert.as_ptr() as *const u8;
        let n_pairs = (n_b_med - first_b_med) / 2;
        for lane in (0..64).step_by(8) {
            let mut ab0 = vdupq_n_u8(0);
            let mut ab1 = vdupq_n_u8(0);
            let mut ab2 = vdupq_n_u8(0);
            let mut ab3 = vdupq_n_u8(0);
            let mut ab4 = vdupq_n_u8(0);
            let mut ab5 = vdupq_n_u8(0);
            let mut ab6 = vdupq_n_u8(0);
            let mut ab7 = vdupq_n_u8(0);

            for p in 0..n_pairs {
                let (b_even, b_odd) = (first_b_med + 2 * p, first_b_med + 2 * p + 1);
                let t_even = convert_ptr.add(b_even * 256 * 16);
                let t_odd = convert_ptr.add(b_odd * 256 * 16);

                let we = (chunk_ab_bytes[b_even].as_ptr().add(lane) as *const u64).read_unaligned()
                    as usize;
                let wo = (chunk_ab_bytes[b_odd].as_ptr().add(lane) as *const u64).read_unaligned()
                    as usize;
                ab0 = xor3_u8(
                    ab0,
                    ab_ld::<U, 0>(t_even, we),
                    ab_ld::<U, 0>(t_odd, wo),
                );
                ab1 = xor3_u8(
                    ab1,
                    ab_ld::<U, 8>(t_even, we),
                    ab_ld::<U, 8>(t_odd, wo),
                );
                ab2 = xor3_u8(
                    ab2,
                    ab_ld::<U, 16>(t_even, we),
                    ab_ld::<U, 16>(t_odd, wo),
                );
                ab3 = xor3_u8(
                    ab3,
                    ab_ld::<U, 24>(t_even, we),
                    ab_ld::<U, 24>(t_odd, wo),
                );
                ab4 = xor3_u8(
                    ab4,
                    ab_ld::<U, 32>(t_even, we),
                    ab_ld::<U, 32>(t_odd, wo),
                );
                ab5 = xor3_u8(
                    ab5,
                    ab_ld::<U, 40>(t_even, we),
                    ab_ld::<U, 40>(t_odd, wo),
                );
                ab6 = xor3_u8(
                    ab6,
                    ab_ld::<U, 48>(t_even, we),
                    ab_ld::<U, 48>(t_odd, wo),
                );
                ab7 = xor3_u8(
                    ab7,
                    ab_ld::<U, 56>(t_even, we),
                    ab_ld::<U, 56>(t_odd, wo),
                );
            }

            // Odd trailing b_med: unpaired fallback.
            if (n_b_med - first_b_med) & 1 == 1 {
                let b_med = n_b_med - 1;
                let table = convert_ptr.add(b_med * 256 * 16);
                let wa = (chunk_ab_bytes[b_med].as_ptr().add(lane) as *const u64).read_unaligned()
                    as usize;
                ab0 = veorq_u8(ab0, ab_ld::<U, 0>(table, wa));
                ab1 = veorq_u8(ab1, ab_ld::<U, 8>(table, wa));
                ab2 = veorq_u8(ab2, ab_ld::<U, 16>(table, wa));
                ab3 = veorq_u8(ab3, ab_ld::<U, 24>(table, wa));
                ab4 = veorq_u8(ab4, ab_ld::<U, 32>(table, wa));
                ab5 = veorq_u8(ab5, ab_ld::<U, 40>(table, wa));
                ab6 = veorq_u8(ab6, ab_ld::<U, 48>(table, wa));
                ab7 = veorq_u8(ab7, ab_ld::<U, 56>(table, wa));
            }

            macro_rules! drain_lane {
                ($offset:literal, $ab:ident) => {{
                    let slot = bank.as_mut_ptr().add(lane + $offset) as *mut u8;
                    if STORE {
                        vst1q_u8(slot, $ab);
                    } else {
                        vst1q_u8(slot, veorq_u8(vld1q_u8(slot), $ab));
                    }
                }};
            }
            drain_lane!(0, ab0);
            drain_lane!(1, ab1);
            drain_lane!(2, ab2);
            drain_lane!(3, ab3);
            drain_lane!(4, ab4);
            drain_lane!(5, ab5);
            drain_lane!(6, ab6);
            drain_lane!(7, ab7);
        }
    }
}

///
///
#[inline(always)]
pub(crate) unsafe fn accumulate_convert_ab_nomul_seeded<const U: bool>(
    chunk_ab_bytes: &[[u8; 64]; 16],
    first_b_med: usize,
    n_b_med: usize,
    convert: &[F128],
    bank: &mut [F128; 64],
) {
    use core::arch::aarch64::*;

    debug_assert!(first_b_med <= n_b_med && n_b_med <= 16);
    debug_assert_eq!(first_b_med & 1, 0, "paired AB drain requires an even start");
    debug_assert_eq!(convert.len(), 16 * 256);

    unsafe {
        let convert_ptr = convert.as_ptr() as *const u8;
        let rows = chunk_ab_bytes.as_ptr() as *const u8;
        let n_pairs = (n_b_med - first_b_med) / 2;
        let odd_tail = (n_b_med - first_b_med) & 1 == 1;
        for lane in (0..64).step_by(8) {
            let slot = bank.as_mut_ptr().add(lane) as *mut u8;
            let mut ab0 = vld1q_u8(slot);
            let mut ab1 = vld1q_u8(slot.add(16));
            let mut ab2 = vld1q_u8(slot.add(32));
            let mut ab3 = vld1q_u8(slot.add(48));
            let mut ab4 = vld1q_u8(slot.add(64));
            let mut ab5 = vld1q_u8(slot.add(80));
            let mut ab6 = vld1q_u8(slot.add(96));
            let mut ab7 = vld1q_u8(slot.add(112));

            let mut t_even = convert_ptr.add(first_b_med * 256 * 16);
            let mut row = rows.add(first_b_med * 64 + lane);
            for _ in 0..n_pairs {
                let t_odd = t_even.add(256 * 16);
                let we = (row as *const u64).read_unaligned() as usize;
                let wo = (row.add(64) as *const u64).read_unaligned() as usize;
                ab0 = xor3_u8(ab0, ab_ld::<U, 0>(t_even, we), ab_ld::<U, 0>(t_odd, wo));
                ab1 = xor3_u8(ab1, ab_ld::<U, 8>(t_even, we), ab_ld::<U, 8>(t_odd, wo));
                ab2 = xor3_u8(ab2, ab_ld::<U, 16>(t_even, we), ab_ld::<U, 16>(t_odd, wo));
                ab3 = xor3_u8(ab3, ab_ld::<U, 24>(t_even, we), ab_ld::<U, 24>(t_odd, wo));
                ab4 = xor3_u8(ab4, ab_ld::<U, 32>(t_even, we), ab_ld::<U, 32>(t_odd, wo));
                ab5 = xor3_u8(ab5, ab_ld::<U, 40>(t_even, we), ab_ld::<U, 40>(t_odd, wo));
                ab6 = xor3_u8(ab6, ab_ld::<U, 48>(t_even, we), ab_ld::<U, 48>(t_odd, wo));
                ab7 = xor3_u8(ab7, ab_ld::<U, 56>(t_even, we), ab_ld::<U, 56>(t_odd, wo));
                t_even = t_even.add(2 * 256 * 16);
                row = row.add(128);
            }

            // Odd trailing b_med: unpaired fallback.
            if odd_tail {
                let b_med = n_b_med - 1;
                let table = convert_ptr.add(b_med * 256 * 16);
                let wa = (rows.add(b_med * 64 + lane) as *const u64).read_unaligned() as usize;
                ab0 = veorq_u8(ab0, ab_ld::<U, 0>(table, wa));
                ab1 = veorq_u8(ab1, ab_ld::<U, 8>(table, wa));
                ab2 = veorq_u8(ab2, ab_ld::<U, 16>(table, wa));
                ab3 = veorq_u8(ab3, ab_ld::<U, 24>(table, wa));
                ab4 = veorq_u8(ab4, ab_ld::<U, 32>(table, wa));
                ab5 = veorq_u8(ab5, ab_ld::<U, 40>(table, wa));
                ab6 = veorq_u8(ab6, ab_ld::<U, 48>(table, wa));
                ab7 = veorq_u8(ab7, ab_ld::<U, 56>(table, wa));
            }

            vst1q_u8(slot, ab0);
            vst1q_u8(slot.add(16), ab1);
            vst1q_u8(slot.add(32), ab2);
            vst1q_u8(slot.add(48), ab3);
            vst1q_u8(slot.add(64), ab4);
            vst1q_u8(slot.add(80), ab5);
            vst1q_u8(slot.add(96), ab6);
            vst1q_u8(slot.add(112), ab7);
        }
    }
}

/// Row `B` of a fourteen-row span, 16 bytes at the pass's lane offset, as two
/// words from one paired load (`ldp`). `mid` points at row 7 of the span plus
/// the lane offset, so every row offset (`-448..=384`) fits the paired load's
/// immediate.
#[inline(always)]
unsafe fn ab_row_pair<const B: usize>(mid: *const u8) -> (usize, usize) {
    const { assert!(B < 14) };
    let (lo, hi): (usize, usize);
    // SAFETY: `mid` is row 7 of fourteen in-bounds 64-byte rows plus a lane
    // offset of at most 48, so the 16 bytes read lie inside row `B`; the asm
    // only reads that memory.
    unsafe {
        core::arch::asm!(
            "ldp {lo}, {hi}, [{p}, #{off}]",
            off = const (B as isize - 7) * 64,
            p = in(reg) mid,
            lo = lateout(reg) lo,
            hi = lateout(reg) hi,
            options(pure, readonly, nostack, preserves_flags),
        );
    }
    (lo, hi)
}

/// [`accumulate_convert_ab_nomul_seeded`] for a span of exactly fourteen rows
/// (`first_b_med..first_b_med + 14`: seven pairs, no odd row).
///
/// Sixteen lanes per pass instead of eight: sixteen accumulators, and each
/// row's 16 bytes for the pass come from one paired load, so a call issues
/// 4 x 14 row loads instead of 8 x 14. The seven pairs are unrolled and the
/// fourteen table bases are formed once per call, so no pair counter, pointer
/// bump or odd-row test remains.
///
/// Bit-exactness: every lane XORs the same `ab_ld` entries (same table, same
/// row byte) into the same bank slot, grouped in the same row pairs.
#[inline(always)]
pub(crate) unsafe fn accumulate_convert_ab_nomul_seeded_pair16<const U: bool>(
    chunk_ab_bytes: &[[u8; 64]; 16],
    first_b_med: usize,
    convert: &[F128],
    bank: &mut [F128; 64],
) {
    use core::arch::aarch64::*;

    debug_assert!(first_b_med + 14 <= 16);
    debug_assert_eq!(convert.len(), 16 * 256);

    // SAFETY: caller guarantees fixed input sizes and aarch64 provides NEON.
    // Rows `first_b_med..first_b_med + 14` and their 4 KiB tables are in
    // bounds; each pass reads 16 bytes per row at lane offset 0, 16, 32 or 48
    // and writes sixteen 16-byte bank slots.
    unsafe {
        let t0 = (convert.as_ptr() as *const u8).add(first_b_med * 256 * 16);
        let t: [*const u8; 14] = core::array::from_fn(|b| t0.add(b * 256 * 16));
        let mut mid = (chunk_ab_bytes.as_ptr() as *const u8).add((first_b_med + 7) * 64);
        let mut slot = bank.as_mut_ptr() as *mut u8;
        for _ in 0..4 {
            let mut a0 = vld1q_u8(slot);
            let mut a1 = vld1q_u8(slot.add(16));
            let mut a2 = vld1q_u8(slot.add(32));
            let mut a3 = vld1q_u8(slot.add(48));
            let mut a4 = vld1q_u8(slot.add(64));
            let mut a5 = vld1q_u8(slot.add(80));
            let mut a6 = vld1q_u8(slot.add(96));
            let mut a7 = vld1q_u8(slot.add(112));
            let mut a8 = vld1q_u8(slot.add(128));
            let mut a9 = vld1q_u8(slot.add(144));
            let mut a10 = vld1q_u8(slot.add(160));
            let mut a11 = vld1q_u8(slot.add(176));
            let mut a12 = vld1q_u8(slot.add(192));
            let mut a13 = vld1q_u8(slot.add(208));
            let mut a14 = vld1q_u8(slot.add(224));
            let mut a15 = vld1q_u8(slot.add(240));

            // One word of an (even, odd) row pair into eight accumulators.
            macro_rules! fold8 {
                ($te:expr, $to:expr, $we:expr, $wo:expr,
                 $x0:ident, $x1:ident, $x2:ident, $x3:ident,
                 $x4:ident, $x5:ident, $x6:ident, $x7:ident) => {
                    $x0 = xor3_u8($x0, ab_ld::<U, 0>($te, $we), ab_ld::<U, 0>($to, $wo));
                    $x1 = xor3_u8($x1, ab_ld::<U, 8>($te, $we), ab_ld::<U, 8>($to, $wo));
                    $x2 = xor3_u8($x2, ab_ld::<U, 16>($te, $we), ab_ld::<U, 16>($to, $wo));
                    $x3 = xor3_u8($x3, ab_ld::<U, 24>($te, $we), ab_ld::<U, 24>($to, $wo));
                    $x4 = xor3_u8($x4, ab_ld::<U, 32>($te, $we), ab_ld::<U, 32>($to, $wo));
                    $x5 = xor3_u8($x5, ab_ld::<U, 40>($te, $we), ab_ld::<U, 40>($to, $wo));
                    $x6 = xor3_u8($x6, ab_ld::<U, 48>($te, $we), ab_ld::<U, 48>($to, $wo));
                    $x7 = xor3_u8($x7, ab_ld::<U, 56>($te, $we), ab_ld::<U, 56>($to, $wo));
                };
            }
            // Rows `$e` and `$o` of the span: low word to lanes 0..8 of the
            // pass, high word to lanes 8..16.
            macro_rules! pair {
                ($e:literal, $o:literal) => {{
                    let (e0, e1) = ab_row_pair::<$e>(mid);
                    let (o0, o1) = ab_row_pair::<$o>(mid);
                    fold8!(t[$e], t[$o], e0, o0, a0, a1, a2, a3, a4, a5, a6, a7);
                    fold8!(t[$e], t[$o], e1, o1, a8, a9, a10, a11, a12, a13, a14, a15);
                }};
            }
            pair!(0, 1);
            pair!(2, 3);
            pair!(4, 5);
            pair!(6, 7);
            pair!(8, 9);
            pair!(10, 11);
            pair!(12, 13);

            vst1q_u8(slot, a0);
            vst1q_u8(slot.add(16), a1);
            vst1q_u8(slot.add(32), a2);
            vst1q_u8(slot.add(48), a3);
            vst1q_u8(slot.add(64), a4);
            vst1q_u8(slot.add(80), a5);
            vst1q_u8(slot.add(96), a6);
            vst1q_u8(slot.add(112), a7);
            vst1q_u8(slot.add(128), a8);
            vst1q_u8(slot.add(144), a9);
            vst1q_u8(slot.add(160), a10);
            vst1q_u8(slot.add(176), a11);
            vst1q_u8(slot.add(192), a12);
            vst1q_u8(slot.add(208), a13);
            vst1q_u8(slot.add(224), a14);
            vst1q_u8(slot.add(240), a15);
            mid = mid.add(16);
            slot = slot.add(256);
        }
    }
}

/// Eight α-free single-bit-`K` C banks, NEON — straight off the packed witness.
///
/// Three phases, no multiplies, no convert-table gathers, and **no per-`b_med`
/// `bit_transpose_64bytes`**:
///
/// 1. **Cross-`b_med` bit transpose, on the raw rows.** `partial_c[s][lane]`
///    wants the u16 whose bit `b` is bit `s` of `bit_transpose(raw[b])[lane]`,
///    which unfolds to *bit `t` of `raw[b][8s + g]`* for `lane = 8g + t`. So the
///    axis that has to move is `b` ↔ the bit index — a plain 8×8 bit transpose
///    across eight q-registers, sixteen byte-columns at a time, run twice (`b`
///    low half → `m_lo`, high half → `m_hi`). The butterfly is the standard
///    recursive block swap: distance 4 is a nibble exchange, one `SLI` + one
///    `SRI` per pair; distances 2 and 1 are one shift + one `BSL` per side.
///    24 vector ops per 8×8 block.
///
/// 2. **Interleaved store — this is what deletes the second permutation.** The
///    transpose leaves `P_t[j]` (bit `b` = bit `t` of `raw[b][j]`), and the
///    drain wants `m_s[8g + t] = P_t[8s + g]`. Written *interleaved*, i.e. into
///    `R[j·8 + t]`, that reindex becomes `m_s[lane] = R[64s + lane]` — the
///    identity on a `[[u8; 64]; 8]` buffer. So the byte permutation costs
///    nothing but the store order: an 8-way byte interleave of the eight
///    transpose outputs, `I8 = zip(zip(zip(r0,r4), zip(r2,r6)), zip(zip(r1,r5),
///    zip(r3,r7)))`, 24 `ZIP1`/`ZIP2` per block. Verified index-by-index by
///    `fused_c_masks_match_composed_transpose`.
///
/// 3. **Table drain.** `F128 { lo: mask } * eq_lo` is F₂-linear in the mask, so
///    it splits along the byte boundary phases 1–2 already produced:
///    `t_lo[m_lo] + t_hi[m_hi]`, folded in with one EOR3. Keeping the halves
///    separate is why nothing ever reassembles a u16.
///
/// Net vs the composed form this replaces: 16 × `bit_transpose_64bytes_neon`
/// (~72 vector ops each) plus a 320-op mask transpose, against ~512 ops here.
/// `c_block` is the 1024 raw witness bytes for this `x_outer_lo`, laid out as
/// 16 contiguous 64-byte `b_med` rows.
#[inline(always)]
pub(crate) unsafe fn accumulate_c_banks(
    c_block: &[u8; 16 * 64],
    n_b_med: usize,
    mask_tables: &[F128],
    partial_c: &mut [[F128; 64]; 8],
    drain4: bool,
) {
    use core::arch::aarch64::*;

    debug_assert!(n_b_med <= 16);
    debug_assert_eq!(mask_tables.len(), 512);

    // SAFETY: every load/store below is a fixed offset into the caller's
    // fixed-size arrays; table indices are `u8` scaled by the 16-byte row size,
    // bounded by the debug-asserted 256-row halves.
    unsafe {
        // `x[b] bit s` -> `out[s] bit b`, vectorized over the 16 byte columns.
        macro_rules! transpose8 {
            ($x:expr, $m33:expr, $m55:expr) => {{
                let x: [uint8x16_t; 8] = $x;
                // distance 4: exchange the 4x4 off-diagonal blocks.
                let mut a = [vdupq_n_u8(0); 8];
                for b in 0..4 {
                    a[b] = vsliq_n_u8::<4>(x[b], x[b + 4]);
                    a[b + 4] = vsriq_n_u8::<4>(x[b + 4], x[b]);
                }
                // distance 2, then distance 1: same swap, masked halves.
                let mut c = [vdupq_n_u8(0); 8];
                for q in 0..2 {
                    for b in 0..2 {
                        let (i, j) = (4 * q + b, 4 * q + b + 2);
                        c[i] = vbslq_u8($m33, a[i], vshlq_n_u8::<2>(a[j]));
                        c[j] = vbslq_u8($m33, vshrq_n_u8::<2>(a[i]), a[j]);
                    }
                }
                let mut out = [vdupq_n_u8(0); 8];
                for p in 0..4 {
                    let (i, j) = (2 * p, 2 * p + 1);
                    out[i] = vbslq_u8($m55, c[i], vshlq_n_u8::<1>(c[j]));
                    out[j] = vbslq_u8($m55, vshrq_n_u8::<1>(c[i]), c[j]);
                }
                out
            }};
        }

        // 8-way byte interleave: `out[8c + t] == r[t][c]` over the 128 bytes.
        macro_rules! interleave8 {
            ($r:expr) => {{
                let r: [uint8x16_t; 8] = $r;
                let (a0, a1) = (vzip1q_u8(r[0], r[4]), vzip2q_u8(r[0], r[4]));
                let (b0, b1) = (vzip1q_u8(r[2], r[6]), vzip2q_u8(r[2], r[6]));
                let (c0, c1) = (vzip1q_u8(r[1], r[5]), vzip2q_u8(r[1], r[5]));
                let (d0, d1) = (vzip1q_u8(r[3], r[7]), vzip2q_u8(r[3], r[7]));
                let (e0, e1) = (vzip1q_u8(a0, b0), vzip2q_u8(a0, b0));
                let (e2, e3) = (vzip1q_u8(a1, b1), vzip2q_u8(a1, b1));
                let (f0, f1) = (vzip1q_u8(c0, d0), vzip2q_u8(c0, d0));
                let (f2, f3) = (vzip1q_u8(c1, d1), vzip2q_u8(c1, d1));
                [
                    vzip1q_u8(e0, f0),
                    vzip2q_u8(e0, f0),
                    vzip1q_u8(e1, f1),
                    vzip2q_u8(e1, f1),
                    vzip1q_u8(e2, f2),
                    vzip2q_u8(e2, f2),
                    vzip1q_u8(e3, f3),
                    vzip2q_u8(e3, f3),
                ]
            }};
        }

        let m33 = vdupq_n_u8(0x33);
        let m55 = vdupq_n_u8(0x55);

        // Bit `b` of `m_lo[s][lane]` is bit `s` of the transposed C byte for
        // `b_med = b` (`m_hi` covers `b_med` 8..16). 1 KiB of stack, dead at
        // return; phase 2 writes it in exactly this layout.
        let mut m_lo = [[0u8; 64]; 8];
        let mut m_hi = [[0u8; 64]; 8];
        let src = c_block.as_ptr();

        // `n_b_med` is fixed for the whole call, so the padded boundary window
        // gets its own arm rather than a runtime-zeroed row array — otherwise
        // the full path pays sixteen dead vector stores per column chunk.
        // Rows past `n_b_med` must read as ZERO rather than as witness bytes:
        // the incumbent skips them outright, and matching that keeps the kernel
        // exact even on a witness whose padding is not honestly zero.
        macro_rules! build_masks {
            ($row:expr) => {
                for chunk in 0..4 {
                    let col = chunk * 16;
                    let row = |b: usize| -> uint8x16_t { $row(b, col) };
                    for (half, dst) in [m_lo.as_mut_ptr(), m_hi.as_mut_ptr()]
                        .into_iter()
                        .enumerate()
                    {
                        let base = half * 8;
                        let t = transpose8!(
                            [
                                row(base),
                                row(base + 1),
                                row(base + 2),
                                row(base + 3),
                                row(base + 4),
                                row(base + 5),
                                row(base + 6),
                                row(base + 7)
                            ],
                            m33,
                            m55
                        );
                        let g = interleave8!(t);
                        let out = dst as *mut u8;
                        for (i, value) in g.into_iter().enumerate() {
                            vst1q_u8(out.add(chunk * 128 + i * 16), value);
                        }
                    }
                }
            };
        }
        if n_b_med == 16 {
            build_masks!(|b: usize, col: usize| vld1q_u8(src.add(b * 64 + col)));
        } else {
            build_masks!(|b: usize, col: usize| if b < n_b_med {
                vld1q_u8(src.add(b * 64 + col))
            } else {
                vdupq_n_u8(0)
            });
        }

        let t_lo = mask_tables.as_ptr() as *const u8;
        let t_hi = t_lo.add(256 * 16);
        if drain4 {
            // Four independent lanes expose eight random table loads at once,
            // matching the latency-hiding shape of the AB convert kernel.
            // The old scalar drain kept only the low/high pair for one lane in
            // flight, leaving the dependent L1 gather latency on the critical
            // chain even though the register file has ample headroom.
            for s in 0..8 {
                let bank = partial_c[s].as_mut_ptr() as *mut u8;
                for lane in (0..64).step_by(4) {
                    let lo = (m_lo[s].as_ptr().add(lane) as *const u32).read_unaligned();
                    let hi = (m_hi[s].as_ptr().add(lane) as *const u32).read_unaligned();

                    let l0 = vld1q_u8(t_lo.add(usize::from((lo & 0xff) as u8) * 16));
                    let l1 = vld1q_u8(t_lo.add(usize::from(((lo >> 8) & 0xff) as u8) * 16));
                    let l2 = vld1q_u8(t_lo.add(usize::from(((lo >> 16) & 0xff) as u8) * 16));
                    let l3 = vld1q_u8(t_lo.add(usize::from((lo >> 24) as u8) * 16));
                    let h0 = vld1q_u8(t_hi.add(usize::from((hi & 0xff) as u8) * 16));
                    let h1 = vld1q_u8(t_hi.add(usize::from(((hi >> 8) & 0xff) as u8) * 16));
                    let h2 = vld1q_u8(t_hi.add(usize::from(((hi >> 16) & 0xff) as u8) * 16));
                    let h3 = vld1q_u8(t_hi.add(usize::from((hi >> 24) as u8) * 16));

                    let p0 = bank.add(lane * 16);
                    let p1 = p0.add(16);
                    let p2 = p1.add(16);
                    let p3 = p2.add(16);
                    vst1q_u8(p0, xor3_u8(vld1q_u8(p0), l0, h0));
                    vst1q_u8(p1, xor3_u8(vld1q_u8(p1), l1, h1));
                    vst1q_u8(p2, xor3_u8(vld1q_u8(p2), l2, h2));
                    vst1q_u8(p3, xor3_u8(vld1q_u8(p3), l3, h3));
                }
            }
        } else {
            for s in 0..8 {
                let bank = partial_c[s].as_mut_ptr() as *mut u8;
                for lane in 0..64 {
                    let slot = bank.add(lane * 16);
                    vst1q_u8(
                        slot,
                        xor3_u8(
                            vld1q_u8(slot),
                            vld1q_u8(t_lo.add(usize::from(m_lo[s][lane]) * 16)),
                            vld1q_u8(t_hi.add(usize::from(m_hi[s][lane]) * 16)),
                        ),
                    );
                }
            }
        }
    }
}

/// Direct-fold4 C capture, NEON — retain `q = b_med & 3` in four groups of
/// the incumbent eight alpha-free small-bit banks.
///
/// This follows the incumbent fused-mask kernel's two efficient primitives:
/// an 8x8 cross-row bit transpose directly from the packed witness, and a
/// four-lane field-table drain.  The difference is the row grouping.  For a
/// fixed q we transpose rows `[q, q+4, q+8, q+12, 0, 0, 0, 0]`; the resulting
/// low nibble is exactly the h-mask selecting gamma exponents {0,4,8,12}.
/// Thus no per-bit or per-lane scalar mask construction remains in the ranked
/// hot loop.  Four group transposes replace the incumbent two half transposes,
/// while the table shrinks from 512 to 16 F128 entries.
#[inline(always)]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) unsafe fn accumulate_c_fold4_banks(
    c_block: &[u8; 16 * 64],
    n_b_med: usize,
    mask_table: &[F128],
    partial_c: &mut [[F128; 64]; 32],
) {
    use core::arch::aarch64::*;

    debug_assert!(n_b_med <= 16);
    debug_assert_eq!(mask_table.len(), 16);

    // SAFETY: every load and store below is a fixed offset into a fixed-size
    // caller-owned array. Rows at or above n_b_med are replaced with zero, and
    // each table index is produced by a transpose with only four live rows.
    unsafe {
        macro_rules! transpose8 {
            ($x:expr, $m33:expr, $m55:expr) => {{
                let x: [uint8x16_t; 8] = $x;
                let mut a = [vdupq_n_u8(0); 8];
                for b in 0..4 {
                    a[b] = vsliq_n_u8::<4>(x[b], x[b + 4]);
                    a[b + 4] = vsriq_n_u8::<4>(x[b + 4], x[b]);
                }
                let mut c = [vdupq_n_u8(0); 8];
                for block in 0..2 {
                    for b in 0..2 {
                        let (i, j) = (4 * block + b, 4 * block + b + 2);
                        c[i] = vbslq_u8($m33, a[i], vshlq_n_u8::<2>(a[j]));
                        c[j] = vbslq_u8($m33, vshrq_n_u8::<2>(a[i]), a[j]);
                    }
                }
                let mut out = [vdupq_n_u8(0); 8];
                for pair in 0..4 {
                    let (i, j) = (2 * pair, 2 * pair + 1);
                    out[i] = vbslq_u8($m55, c[i], vshlq_n_u8::<1>(c[j]));
                    out[j] = vbslq_u8($m55, vshrq_n_u8::<1>(c[i]), c[j]);
                }
                out
            }};
        }

        macro_rules! interleave8 {
            ($r:expr) => {{
                let r: [uint8x16_t; 8] = $r;
                let (a0, a1) = (vzip1q_u8(r[0], r[4]), vzip2q_u8(r[0], r[4]));
                let (b0, b1) = (vzip1q_u8(r[2], r[6]), vzip2q_u8(r[2], r[6]));
                let (c0, c1) = (vzip1q_u8(r[1], r[5]), vzip2q_u8(r[1], r[5]));
                let (d0, d1) = (vzip1q_u8(r[3], r[7]), vzip2q_u8(r[3], r[7]));
                let (e0, e1) = (vzip1q_u8(a0, b0), vzip2q_u8(a0, b0));
                let (e2, e3) = (vzip1q_u8(a1, b1), vzip2q_u8(a1, b1));
                let (f0, f1) = (vzip1q_u8(c0, d0), vzip2q_u8(c0, d0));
                let (f2, f3) = (vzip1q_u8(c1, d1), vzip2q_u8(c1, d1));
                [
                    vzip1q_u8(e0, f0),
                    vzip2q_u8(e0, f0),
                    vzip1q_u8(e1, f1),
                    vzip2q_u8(e1, f1),
                    vzip1q_u8(e2, f2),
                    vzip2q_u8(e2, f2),
                    vzip1q_u8(e3, f3),
                    vzip2q_u8(e3, f3),
                ]
            }};
        }

        // masks[q][K][lane] is a low-nibble h mask. All bytes are overwritten
        // by the four 16-column chunks below before the drain reads them.
        let mut masks = [[[0u8; 64]; 8]; 4];
        let src = c_block.as_ptr();
        let m33 = vdupq_n_u8(0x33);
        let m55 = vdupq_n_u8(0x55);
        let zero = vdupq_n_u8(0);

        macro_rules! build_masks {
            ($row:expr) => {
                for chunk in 0..4 {
                    let col = chunk * 16;
                    let row = |b: usize| -> uint8x16_t { $row(b, col) };
                    for q in 0..4 {
                        let t = transpose8!(
                            [
                                row(q),
                                row(q + 4),
                                row(q + 8),
                                row(q + 12),
                                zero,
                                zero,
                                zero,
                                zero,
                            ],
                            m33,
                            m55
                        );
                        let interleaved = interleave8!(t);
                        let dst = masks[q].as_mut_ptr() as *mut u8;
                        for (i, value) in interleaved.into_iter().enumerate() {
                            vst1q_u8(dst.add(chunk * 128 + i * 16), value);
                        }
                    }
                }
            };
        }

        if n_b_med == 16 {
            build_masks!(|b: usize, col: usize| vld1q_u8(src.add(b * 64 + col)));
        } else {
            build_masks!(|b: usize, col: usize| if b < n_b_med {
                vld1q_u8(src.add(b * 64 + col))
            } else {
                zero
            });
        }

        let table = mask_table.as_ptr() as *const u8;
        for q in 0..4 {
            for k in 0..8 {
                let bank = partial_c[q * 8 + k].as_mut_ptr() as *mut u8;
                for lane in (0..64).step_by(4) {
                    let indices = (masks[q][k].as_ptr().add(lane) as *const u32).read_unaligned();
                    let i0 = usize::from((indices & 0xff) as u8);
                    let i1 = usize::from(((indices >> 8) & 0xff) as u8);
                    let i2 = usize::from(((indices >> 16) & 0xff) as u8);
                    let i3 = usize::from((indices >> 24) as u8);
                    debug_assert!(i0 < 16 && i1 < 16 && i2 < 16 && i3 < 16);

                    let p0 = bank.add(lane * 16);
                    let p1 = p0.add(16);
                    let p2 = p1.add(16);
                    let p3 = p2.add(16);
                    vst1q_u8(p0, veorq_u8(vld1q_u8(p0), vld1q_u8(table.add(i0 * 16))));
                    vst1q_u8(p1, veorq_u8(vld1q_u8(p1), vld1q_u8(table.add(i1 * 16))));
                    vst1q_u8(p2, veorq_u8(vld1q_u8(p2), vld1q_u8(table.add(i2 * 16))));
                    vst1q_u8(p3, veorq_u8(vld1q_u8(p3), vld1q_u8(table.add(i3 * 16))));
                }
            }
        }
    }
}

/// Pair-fused direct-fold4 C capture for one retained-medium q group.
/// Two adjacent low-coordinate blocks contribute the low and high nibble of
/// one eight-bit index into an already-composed field table.  The eight-lane
/// drain issues all independent table loads before touching the accumulator,
/// and performs one RMW for the pair instead of one per block.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn accumulate_c_fold4_q_pair_banks(
    c_block_even: &[u8; 16 * 64],
    n_b_med_even: usize,
    c_block_odd: &[u8; 16 * 64],
    n_b_med_odd: usize,
    q: usize,
    pair_mask_table: &[F128; 256],
    partial_c: &mut [[F128; 64]; 8],
) {
    use core::arch::aarch64::*;

    debug_assert!(n_b_med_even <= 16);
    debug_assert!(n_b_med_odd <= 16);
    debug_assert!(q < 4);
    debug_assert_eq!(pair_mask_table.len(), 256);

    // SAFETY: every live row is bounds-checked against its block's independent
    // padding count. Each transpose has four live rows and four zero rows, so
    // every mask byte is a nibble and their concatenation is a valid u8 index.
    unsafe {
        macro_rules! transpose8 {
            ($x:expr, $m33:expr, $m55:expr) => {{
                let x: [uint8x16_t; 8] = $x;
                let mut a = [vdupq_n_u8(0); 8];
                for b in 0..4 {
                    a[b] = vsliq_n_u8::<4>(x[b], x[b + 4]);
                    a[b + 4] = vsriq_n_u8::<4>(x[b + 4], x[b]);
                }
                let mut c = [vdupq_n_u8(0); 8];
                for block in 0..2 {
                    for b in 0..2 {
                        let (i, j) = (4 * block + b, 4 * block + b + 2);
                        c[i] = vbslq_u8($m33, a[i], vshlq_n_u8::<2>(a[j]));
                        c[j] = vbslq_u8($m33, vshrq_n_u8::<2>(a[i]), a[j]);
                    }
                }
                let mut out = [vdupq_n_u8(0); 8];
                for pair in 0..4 {
                    let (i, j) = (2 * pair, 2 * pair + 1);
                    out[i] = vbslq_u8($m55, c[i], vshlq_n_u8::<1>(c[j]));
                    out[j] = vbslq_u8($m55, vshrq_n_u8::<1>(c[i]), c[j]);
                }
                out
            }};
        }

        macro_rules! interleave8 {
            ($r:expr) => {{
                let r: [uint8x16_t; 8] = $r;
                let (a0, a1) = (vzip1q_u8(r[0], r[4]), vzip2q_u8(r[0], r[4]));
                let (b0, b1) = (vzip1q_u8(r[2], r[6]), vzip2q_u8(r[2], r[6]));
                let (c0, c1) = (vzip1q_u8(r[1], r[5]), vzip2q_u8(r[1], r[5]));
                let (d0, d1) = (vzip1q_u8(r[3], r[7]), vzip2q_u8(r[3], r[7]));
                let (e0, e1) = (vzip1q_u8(a0, b0), vzip2q_u8(a0, b0));
                let (e2, e3) = (vzip1q_u8(a1, b1), vzip2q_u8(a1, b1));
                let (f0, f1) = (vzip1q_u8(c0, d0), vzip2q_u8(c0, d0));
                let (f2, f3) = (vzip1q_u8(c1, d1), vzip2q_u8(c1, d1));
                [
                    vzip1q_u8(e0, f0),
                    vzip2q_u8(e0, f0),
                    vzip1q_u8(e1, f1),
                    vzip2q_u8(e1, f1),
                    vzip1q_u8(e2, f2),
                    vzip2q_u8(e2, f2),
                    vzip1q_u8(e3, f3),
                    vzip2q_u8(e3, f3),
                ]
            }};
        }

        // masks[side][K][lane], side 0=even and side 1=odd. All bytes are
        // overwritten by the four 16-column chunks before the drain.
        let mut masks = [[[0u8; 64]; 8]; 2];
        let sources = [c_block_even.as_ptr(), c_block_odd.as_ptr()];
        let counts = [n_b_med_even, n_b_med_odd];
        let m33 = vdupq_n_u8(0x33);
        let m55 = vdupq_n_u8(0x55);
        let zero = vdupq_n_u8(0);

        for side in 0..2 {
            let src = sources[side];
            let n_b_med = counts[side];
            for chunk in 0..4 {
                let col = chunk * 16;
                let row = |b: usize| -> uint8x16_t {
                    if b < n_b_med {
                        vld1q_u8(src.add(b * 64 + col))
                    } else {
                        zero
                    }
                };
                let t = transpose8!(
                    [
                        row(q),
                        row(q + 4),
                        row(q + 8),
                        row(q + 12),
                        zero,
                        zero,
                        zero,
                        zero,
                    ],
                    m33,
                    m55
                );
                let interleaved = interleave8!(t);
                let dst = masks[side].as_mut_ptr() as *mut u8;
                for (i, value) in interleaved.into_iter().enumerate() {
                    vst1q_u8(dst.add(chunk * 128 + i * 16), value);
                }
            }
        }

        let table = pair_mask_table.as_ptr() as *const u8;
        for k in 0..8 {
            let bank = partial_c[k].as_mut_ptr() as *mut u8;
            for lane in (0..64).step_by(8) {
                let even = (masks[0][k].as_ptr().add(lane) as *const u64).read_unaligned();
                let odd = (masks[1][k].as_ptr().add(lane) as *const u64).read_unaligned();
                // Both operands contain only low nibbles in every byte, so a
                // whole-word shift cannot carry across byte boundaries.
                let indices = even | (odd << 4);
                let i0 = usize::from((indices & 0xff) as u8);
                let i1 = usize::from(((indices >> 8) & 0xff) as u8);
                let i2 = usize::from(((indices >> 16) & 0xff) as u8);
                let i3 = usize::from(((indices >> 24) & 0xff) as u8);
                let i4 = usize::from(((indices >> 32) & 0xff) as u8);
                let i5 = usize::from(((indices >> 40) & 0xff) as u8);
                let i6 = usize::from(((indices >> 48) & 0xff) as u8);
                let i7 = usize::from((indices >> 56) as u8);

                // Pipeline the independent pair-table loads ahead of the
                // accumulator dependency chain.
                let t0 = vld1q_u8(table.add(i0 * 16));
                let t1 = vld1q_u8(table.add(i1 * 16));
                let t2 = vld1q_u8(table.add(i2 * 16));
                let t3 = vld1q_u8(table.add(i3 * 16));
                let t4 = vld1q_u8(table.add(i4 * 16));
                let t5 = vld1q_u8(table.add(i5 * 16));
                let t6 = vld1q_u8(table.add(i6 * 16));
                let t7 = vld1q_u8(table.add(i7 * 16));

                let p0 = bank.add(lane * 16);
                let p1 = p0.add(16);
                let p2 = p1.add(16);
                let p3 = p2.add(16);
                let p4 = p3.add(16);
                let p5 = p4.add(16);
                let p6 = p5.add(16);
                let p7 = p6.add(16);
                vst1q_u8(p0, veorq_u8(vld1q_u8(p0), t0));
                vst1q_u8(p1, veorq_u8(vld1q_u8(p1), t1));
                vst1q_u8(p2, veorq_u8(vld1q_u8(p2), t2));
                vst1q_u8(p3, veorq_u8(vld1q_u8(p3), t3));
                vst1q_u8(p4, veorq_u8(vld1q_u8(p4), t4));
                vst1q_u8(p5, veorq_u8(vld1q_u8(p5), t5));
                vst1q_u8(p6, veorq_u8(vld1q_u8(p6), t6));
                vst1q_u8(p7, veorq_u8(vld1q_u8(p7), t7));
            }
        }
    }
}

/// Four-block direct-fold4 drain. For each q/K/lane, blocks 0-1 form one
/// pair-table index and blocks 2-3 form another. Their two field values are
/// XORed before the sole accumulator load/store, reducing logical drain
/// traffic from 24 GiB for pair fusion to 16 GiB at the ranked shape.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn accumulate_c_fold4_four_banks(
    c_blocks: [&[u8; 16 * 64]; 4],
    n_b_med: [usize; 4],
    pair_mask_tables: [&[F128; 256]; 2],
    partial_c: &mut [[F128; 64]; 32],
) {
    use core::arch::aarch64::*;

    debug_assert!(n_b_med.into_iter().all(|count| count <= 16));

    // SAFETY: safe wrapper validates every row count. Each transpose has four
    // live rows and four zero rows, so masks remain nibbles; concatenating two
    // masks therefore indexes exactly one of 256 fixed table entries.
    unsafe {
        macro_rules! transpose8 {
            ($x:expr, $m33:expr, $m55:expr) => {{
                let x: [uint8x16_t; 8] = $x;
                let mut a = [vdupq_n_u8(0); 8];
                for b in 0..4 {
                    a[b] = vsliq_n_u8::<4>(x[b], x[b + 4]);
                    a[b + 4] = vsriq_n_u8::<4>(x[b + 4], x[b]);
                }
                let mut c = [vdupq_n_u8(0); 8];
                for block in 0..2 {
                    for b in 0..2 {
                        let (i, j) = (4 * block + b, 4 * block + b + 2);
                        c[i] = vbslq_u8($m33, a[i], vshlq_n_u8::<2>(a[j]));
                        c[j] = vbslq_u8($m33, vshrq_n_u8::<2>(a[i]), a[j]);
                    }
                }
                let mut out = [vdupq_n_u8(0); 8];
                for pair in 0..4 {
                    let (i, j) = (2 * pair, 2 * pair + 1);
                    out[i] = vbslq_u8($m55, c[i], vshlq_n_u8::<1>(c[j]));
                    out[j] = vbslq_u8($m55, vshrq_n_u8::<1>(c[i]), c[j]);
                }
                out
            }};
        }

        macro_rules! interleave8 {
            ($r:expr) => {{
                let r: [uint8x16_t; 8] = $r;
                let (a0, a1) = (vzip1q_u8(r[0], r[4]), vzip2q_u8(r[0], r[4]));
                let (b0, b1) = (vzip1q_u8(r[2], r[6]), vzip2q_u8(r[2], r[6]));
                let (c0, c1) = (vzip1q_u8(r[1], r[5]), vzip2q_u8(r[1], r[5]));
                let (d0, d1) = (vzip1q_u8(r[3], r[7]), vzip2q_u8(r[3], r[7]));
                let (e0, e1) = (vzip1q_u8(a0, b0), vzip2q_u8(a0, b0));
                let (e2, e3) = (vzip1q_u8(a1, b1), vzip2q_u8(a1, b1));
                let (f0, f1) = (vzip1q_u8(c0, d0), vzip2q_u8(c0, d0));
                let (f2, f3) = (vzip1q_u8(c1, d1), vzip2q_u8(c1, d1));
                [
                    vzip1q_u8(e0, f0),
                    vzip2q_u8(e0, f0),
                    vzip1q_u8(e1, f1),
                    vzip2q_u8(e1, f1),
                    vzip1q_u8(e2, f2),
                    vzip2q_u8(e2, f2),
                    vzip1q_u8(e3, f3),
                    vzip2q_u8(e3, f3),
                ]
            }};
        }

        // Build and consume one q at a time. This keeps mask scratch at 2 KiB
        // instead of retaining all four q groups for all four witness blocks.
        let mut masks = [[[0u8; 64]; 8]; 4];
        let sources = c_blocks.map(|block| block.as_ptr());
        let m33 = vdupq_n_u8(0x33);
        let m55 = vdupq_n_u8(0x55);
        let zero = vdupq_n_u8(0);
        let table01 = pair_mask_tables[0].as_ptr() as *const u8;
        let table23 = pair_mask_tables[1].as_ptr() as *const u8;

        for q in 0..4 {
            for side in 0..4 {
                let src = sources[side];
                let count = n_b_med[side];
                for chunk in 0..4 {
                    let col = chunk * 16;
                    let row = |b: usize| -> uint8x16_t {
                        if b < count {
                            vld1q_u8(src.add(b * 64 + col))
                        } else {
                            zero
                        }
                    };
                    let t = transpose8!(
                        [
                            row(q),
                            row(q + 4),
                            row(q + 8),
                            row(q + 12),
                            zero,
                            zero,
                            zero,
                            zero,
                        ],
                        m33,
                        m55
                    );
                    let interleaved = interleave8!(t);
                    let dst = masks[side].as_mut_ptr() as *mut u8;
                    for (i, value) in interleaved.into_iter().enumerate() {
                        vst1q_u8(dst.add(chunk * 128 + i * 16), value);
                    }
                }
            }

            for k in 0..8 {
                let bank = partial_c[q * 8 + k].as_mut_ptr() as *mut u8;
                for lane in (0..64).step_by(8) {
                    let m0 = (masks[0][k].as_ptr().add(lane) as *const u64).read_unaligned();
                    let m1 = (masks[1][k].as_ptr().add(lane) as *const u64).read_unaligned();
                    let m2 = (masks[2][k].as_ptr().add(lane) as *const u64).read_unaligned();
                    let m3 = (masks[3][k].as_ptr().add(lane) as *const u64).read_unaligned();
                    let indices01 = m0 | (m1 << 4);
                    let indices23 = m2 | (m3 << 4);

                    let i010 = usize::from((indices01 & 0xff) as u8);
                    let i011 = usize::from(((indices01 >> 8) & 0xff) as u8);
                    let i012 = usize::from(((indices01 >> 16) & 0xff) as u8);
                    let i013 = usize::from(((indices01 >> 24) & 0xff) as u8);
                    let i014 = usize::from(((indices01 >> 32) & 0xff) as u8);
                    let i015 = usize::from(((indices01 >> 40) & 0xff) as u8);
                    let i016 = usize::from(((indices01 >> 48) & 0xff) as u8);
                    let i017 = usize::from((indices01 >> 56) as u8);
                    let i230 = usize::from((indices23 & 0xff) as u8);
                    let i231 = usize::from(((indices23 >> 8) & 0xff) as u8);
                    let i232 = usize::from(((indices23 >> 16) & 0xff) as u8);
                    let i233 = usize::from(((indices23 >> 24) & 0xff) as u8);
                    let i234 = usize::from(((indices23 >> 32) & 0xff) as u8);
                    let i235 = usize::from(((indices23 >> 40) & 0xff) as u8);
                    let i236 = usize::from(((indices23 >> 48) & 0xff) as u8);
                    let i237 = usize::from((indices23 >> 56) as u8);

                    let p0 = bank.add(lane * 16);
                    let p1 = p0.add(16);
                    let p2 = p1.add(16);
                    let p3 = p2.add(16);
                    let p4 = p3.add(16);
                    let p5 = p4.add(16);
                    let p6 = p5.add(16);
                    let p7 = p6.add(16);
                    vst1q_u8(
                        p0,
                        xor3_u8(
                            vld1q_u8(p0),
                            vld1q_u8(table01.add(i010 * 16)),
                            vld1q_u8(table23.add(i230 * 16)),
                        ),
                    );
                    vst1q_u8(
                        p1,
                        xor3_u8(
                            vld1q_u8(p1),
                            vld1q_u8(table01.add(i011 * 16)),
                            vld1q_u8(table23.add(i231 * 16)),
                        ),
                    );
                    vst1q_u8(
                        p2,
                        xor3_u8(
                            vld1q_u8(p2),
                            vld1q_u8(table01.add(i012 * 16)),
                            vld1q_u8(table23.add(i232 * 16)),
                        ),
                    );
                    vst1q_u8(
                        p3,
                        xor3_u8(
                            vld1q_u8(p3),
                            vld1q_u8(table01.add(i013 * 16)),
                            vld1q_u8(table23.add(i233 * 16)),
                        ),
                    );
                    vst1q_u8(
                        p4,
                        xor3_u8(
                            vld1q_u8(p4),
                            vld1q_u8(table01.add(i014 * 16)),
                            vld1q_u8(table23.add(i234 * 16)),
                        ),
                    );
                    vst1q_u8(
                        p5,
                        xor3_u8(
                            vld1q_u8(p5),
                            vld1q_u8(table01.add(i015 * 16)),
                            vld1q_u8(table23.add(i235 * 16)),
                        ),
                    );
                    vst1q_u8(
                        p6,
                        xor3_u8(
                            vld1q_u8(p6),
                            vld1q_u8(table01.add(i016 * 16)),
                            vld1q_u8(table23.add(i236 * 16)),
                        ),
                    );
                    vst1q_u8(
                        p7,
                        xor3_u8(
                            vld1q_u8(p7),
                            vld1q_u8(table01.add(i017 * 16)),
                            vld1q_u8(table23.add(i237 * 16)),
                        ),
                    );
                }
            }
        }
    }
}

// The SHA3 extension includes EOR3; retain the two-EOR form for generic
// AArch64 builds that do not enable it.
#[cfg(target_feature = "sha3")]
#[inline(always)]
unsafe fn xor3_u64(
    a: core::arch::aarch64::uint64x2_t,
    b: core::arch::aarch64::uint64x2_t,
    c: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    unsafe { core::arch::aarch64::veor3q_u64(a, b, c) }
}

#[cfg(not(target_feature = "sha3"))]
#[inline(always)]
unsafe fn xor3_u64(
    a: core::arch::aarch64::uint64x2_t,
    b: core::arch::aarch64::uint64x2_t,
    c: core::arch::aarch64::uint64x2_t,
) -> core::arch::aarch64::uint64x2_t {
    unsafe { core::arch::aarch64::veorq_u64(a, core::arch::aarch64::veorq_u64(b, c)) }
}

// The SHA3 extension includes EOR3; retain the two-EOR form for generic
// AArch64 builds that do not enable it.
#[cfg(target_feature = "sha3")]
#[inline(always)]
unsafe fn xor3_u8(
    a: core::arch::aarch64::uint8x16_t,
    b: core::arch::aarch64::uint8x16_t,
    c: core::arch::aarch64::uint8x16_t,
) -> core::arch::aarch64::uint8x16_t {
    unsafe { core::arch::aarch64::veor3q_u8(a, b, c) }
}

#[cfg(not(target_feature = "sha3"))]
#[inline(always)]
unsafe fn xor3_u8(
    a: core::arch::aarch64::uint8x16_t,
    b: core::arch::aarch64::uint8x16_t,
    c: core::arch::aarch64::uint8x16_t,
) -> core::arch::aarch64::uint8x16_t {
    unsafe { core::arch::aarch64::veorq_u8(a, core::arch::aarch64::veorq_u8(b, c)) }
}

/// NEON 64-byte bit-transpose. Two-stage:
///   1. `vqtbl4q_u8` reorders the 64 input bytes so each 8-byte group within
///      the output is one byte-chunk's worth of `x_small=0..8` bytes.
///   2. Three rounds of bit-swap at distances 7, 14, 28 across `uint64x2_t`
///      lanes do the actual 8×8 bit transpose.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) unsafe fn bit_transpose_64bytes_neon(input: &[u8; 64], output: &mut [u8; 64]) {
    use core::arch::aarch64::*;

    unsafe {
        let in_ptr = input.as_ptr();
        let v0 = vld1q_u8(in_ptr);
        let v1 = vld1q_u8(in_ptr.add(16));
        let v2 = vld1q_u8(in_ptr.add(32));
        let v3 = vld1q_u8(in_ptr.add(48));
        let [y0, y1, y2, y3] = bit_transpose_64bytes_neon_regs(v0, v1, v2, v3);
        let out_ptr = output.as_mut_ptr();
        vst1q_u8(out_ptr, vreinterpretq_u8_u64(y0));
        vst1q_u8(out_ptr.add(16), vreinterpretq_u8_u64(y1));
        vst1q_u8(out_ptr.add(32), vreinterpretq_u8_u64(y2));
        vst1q_u8(out_ptr.add(48), vreinterpretq_u8_u64(y3));
    }
}

/// Register form of [`bit_transpose_64bytes_neon`]: `v0..v3` hold input bytes
/// 0..16, 16..32, 32..48 and 48..64; the result holds output bytes in the same
/// order. Lets a caller fill the input straight from memory lanes.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) unsafe fn bit_transpose_64bytes_neon_regs(
    v0: core::arch::aarch64::uint8x16_t,
    v1: core::arch::aarch64::uint8x16_t,
    v2: core::arch::aarch64::uint8x16_t,
    v3: core::arch::aarch64::uint8x16_t,
) -> [core::arch::aarch64::uint64x2_t; 4] {
    use core::arch::aarch64::*;

    unsafe {
        let table = uint8x16x4_t(v0, v1, v2, v3);

        // vqtbl4q indexes that bring bytes belonging to byte-chunk b ∈ 0..8
        // into contiguous 8-byte runs, packed two-chunks-per-Q-reg.
        const IDX0: [u8; 16] = [0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57];
        const IDX1: [u8; 16] = [2, 10, 18, 26, 34, 42, 50, 58, 3, 11, 19, 27, 35, 43, 51, 59];
        const IDX2: [u8; 16] = [4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53, 61];
        const IDX3: [u8; 16] = [6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63];

        let mut y0 = vreinterpretq_u64_u8(vqtbl4q_u8(table, vld1q_u8(IDX0.as_ptr())));
        let mut y1 = vreinterpretq_u64_u8(vqtbl4q_u8(table, vld1q_u8(IDX1.as_ptr())));
        let mut y2 = vreinterpretq_u64_u8(vqtbl4q_u8(table, vld1q_u8(IDX2.as_ptr())));
        let mut y3 = vreinterpretq_u64_u8(vqtbl4q_u8(table, vld1q_u8(IDX3.as_ptr())));

        let mask1 = vdupq_n_u64(0x00AA00AA00AA00AA);
        let mask2 = vdupq_n_u64(0x0000CCCC0000CCCC);
        let mask3 = vdupq_n_u64(0x00000000F0F0F0F0);

        // Round 1: distance 7.
        let t0 = vandq_u64(veorq_u64(y0, vshrq_n_u64::<7>(y0)), mask1);
        let t1 = vandq_u64(veorq_u64(y1, vshrq_n_u64::<7>(y1)), mask1);
        let t2 = vandq_u64(veorq_u64(y2, vshrq_n_u64::<7>(y2)), mask1);
        let t3 = vandq_u64(veorq_u64(y3, vshrq_n_u64::<7>(y3)), mask1);
        y0 = xor3_u64(y0, t0, vshlq_n_u64::<7>(t0));
        y1 = xor3_u64(y1, t1, vshlq_n_u64::<7>(t1));
        y2 = xor3_u64(y2, t2, vshlq_n_u64::<7>(t2));
        y3 = xor3_u64(y3, t3, vshlq_n_u64::<7>(t3));

        // Round 2: distance 14.
        let t0 = vandq_u64(veorq_u64(y0, vshrq_n_u64::<14>(y0)), mask2);
        let t1 = vandq_u64(veorq_u64(y1, vshrq_n_u64::<14>(y1)), mask2);
        let t2 = vandq_u64(veorq_u64(y2, vshrq_n_u64::<14>(y2)), mask2);
        let t3 = vandq_u64(veorq_u64(y3, vshrq_n_u64::<14>(y3)), mask2);
        y0 = xor3_u64(y0, t0, vshlq_n_u64::<14>(t0));
        y1 = xor3_u64(y1, t1, vshlq_n_u64::<14>(t1));
        y2 = xor3_u64(y2, t2, vshlq_n_u64::<14>(t2));
        y3 = xor3_u64(y3, t3, vshlq_n_u64::<14>(t3));

        // Round 3: distance 28.
        let t0 = vandq_u64(veorq_u64(y0, vshrq_n_u64::<28>(y0)), mask3);
        let t1 = vandq_u64(veorq_u64(y1, vshrq_n_u64::<28>(y1)), mask3);
        let t2 = vandq_u64(veorq_u64(y2, vshrq_n_u64::<28>(y2)), mask3);
        let t3 = vandq_u64(veorq_u64(y3, vshrq_n_u64::<28>(y3)), mask3);
        y0 = xor3_u64(y0, t0, vshlq_n_u64::<28>(t0));
        y1 = xor3_u64(y1, t1, vshlq_n_u64::<28>(t1));
        y2 = xor3_u64(y2, t2, vshlq_n_u64::<28>(t2));
        y3 = xor3_u64(y3, t3, vshlq_n_u64::<28>(t3));

        [y0, y1, y2, y3]
    }
}

// Intermediate-stage NEON kernel: scalar `inv_table.apply` writing to
// `a_col`/`b_col` Vecs, then NEON `gf8_mul_vec16` from those Vecs. Superseded
// by `shift_reduce_inner_ab_fused_neon` which keeps everything register-
// resident; kept under `#[allow(dead_code)]` as a cross-check oracle.
#[cfg(target_arch = "aarch64")]
#[allow(dead_code)]
pub(crate) fn shift_reduce_inner_ab_neon(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; 64],
    a_col: &mut [F8],
    b_col: &mut [F8],
) {
    use crate::field::gf2_8::neon::{gf8_mul_vec16, gf8_reduce_vec16};
    use core::arch::aarch64::*;

    let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;

    // Four (lo, hi) pairs of u16x8 accumulators = 64 u16 lanes total, matching
    // the 64 lanes of the inv-NTT output.
    unsafe {
        let mut acc0_lo = vdupq_n_u16(0);
        let mut acc0_hi = vdupq_n_u16(0);
        let mut acc1_lo = vdupq_n_u16(0);
        let mut acc1_hi = vdupq_n_u16(0);
        let mut acc2_lo = vdupq_n_u16(0);
        let mut acc2_hi = vdupq_n_u16(0);
        let mut acc3_lo = vdupq_n_u16(0);
        let mut acc3_hi = vdupq_n_u16(0);

        // Per-K step: scalar inv-NTT apply into a_col/b_col, then NEON load +
        // 4× gf8_mul_vec16 + 8× vshll_n_u8::<K> + 8× veorq_u16 into the accs.
        // K is `const` so vshll_n_u8 specializes per call site.
        macro_rules! step_k {
            ($k:literal) => {{
                let chunk_off = byte_base_b + $k * N_CHUNKS;
                inv_table.apply(&a_packed[chunk_off..chunk_off + N_CHUNKS], a_col);
                inv_table.apply(&b_packed[chunk_off..chunk_off + N_CHUNKS], b_col);
                let a_ptr = a_col.as_ptr() as *const u8;
                let b_ptr = b_col.as_ptr() as *const u8;
                let y0 = gf8_mul_vec16(vld1q_u8(a_ptr), vld1q_u8(b_ptr));
                let y1 = gf8_mul_vec16(vld1q_u8(a_ptr.add(16)), vld1q_u8(b_ptr.add(16)));
                let y2 = gf8_mul_vec16(vld1q_u8(a_ptr.add(32)), vld1q_u8(b_ptr.add(32)));
                let y3 = gf8_mul_vec16(vld1q_u8(a_ptr.add(48)), vld1q_u8(b_ptr.add(48)));
                acc0_lo = veorq_u16(acc0_lo, vshll_n_u8::<$k>(vget_low_u8(y0)));
                acc0_hi = veorq_u16(acc0_hi, vshll_n_u8::<$k>(vget_high_u8(y0)));
                acc1_lo = veorq_u16(acc1_lo, vshll_n_u8::<$k>(vget_low_u8(y1)));
                acc1_hi = veorq_u16(acc1_hi, vshll_n_u8::<$k>(vget_high_u8(y1)));
                acc2_lo = veorq_u16(acc2_lo, vshll_n_u8::<$k>(vget_low_u8(y2)));
                acc2_hi = veorq_u16(acc2_hi, vshll_n_u8::<$k>(vget_high_u8(y2)));
                acc3_lo = veorq_u16(acc3_lo, vshll_n_u8::<$k>(vget_low_u8(y3)));
                acc3_hi = veorq_u16(acc3_hi, vshll_n_u8::<$k>(vget_high_u8(y3)));
            }};
        }

        step_k!(0);
        step_k!(1);
        step_k!(2);
        step_k!(3);
        step_k!(4);
        step_k!(5);
        step_k!(6);
        step_k!(7);

        // Final F_8 reduction: each (acc_lo, acc_hi) pair → 16 reduced u8 values.
        let r0 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc0_lo), vreinterpretq_u8_u16(acc0_hi));
        let r1 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc1_lo), vreinterpretq_u8_u16(acc1_hi));
        let r2 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc2_lo), vreinterpretq_u8_u16(acc2_hi));
        let r3 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc3_lo), vreinterpretq_u8_u16(acc3_hi));

        let out_ptr = out.as_mut_ptr();
        vst1q_u8(out_ptr, r0);
        vst1q_u8(out_ptr.add(16), r1);
        vst1q_u8(out_ptr.add(32), r2);
        vst1q_u8(out_ptr.add(48), r3);
    }
}

// ---------------------------------------------------------------------------
// Fused NEON inner kernel: inv_NTT apply + F_8 mul + shift_reduce, all in
// NEON registers (no Vec<F8> round-trip).
//
// `xor_apply_byte_into_8_regs::<BH, ODD>` handles one byte position (b ≥ 1).
// `BH` (= b >> 1) selects which chunk-index XOR to apply; `ODD` (= b & 1)
// switches on the within-chunk half-swap. Both const-generic so the compiler
// dead-code-eliminates the if-branch and folds the chunk-index XORs.
//
// `fused_apply_one_k::<K>` runs one full K-row: the initial b=0 plain load,
// 7 calls to the byte helper for b=1..7 (with the specific protocol BH/ODD
// pattern), one 16-lane F_8 mul per output chunk, and finally widen-shift-XOR
// into the per-(K, lane) 16-bit accumulators.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn byte_from_word<const BYTE: u32>(word: u64) -> usize {
    debug_assert!(BYTE < 8);
    if BYTE == 0 || BYTE == 7 {
        return ((word >> (BYTE * 8)) & 0xff) as usize;
    }
    // Bytes 1..6 index 64-byte table rows. From the plain expression LLVM
    // folds the `* 64` into the mask, `(word >> (8*BYTE - 6)) & 0x3fc0`,
    // which loses the shifted-register add and costs a third integer op per
    // lookup. An opaque `ubfx` keeps the row address at `ubfx` + `add lsl #6`.
    let index: usize;
    // SAFETY: register-only bit-field extract; reads and writes no memory.
    unsafe {
        core::arch::asm!(
            "ubfx {index}, {word}, #{shift}, #8",
            shift = const BYTE * 8,
            word = in(reg) word,
            index = lateout(reg) index,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    index
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn xor_apply_byte_into_8_regs<const BH: usize, const ODD: bool>(
    table_base: *const u8,
    half_swapped_table_base: *const u8,
    a_byte: usize,
    b_byte: usize,
    da0: &mut core::arch::aarch64::uint8x16_t,
    da1: &mut core::arch::aarch64::uint8x16_t,
    da2: &mut core::arch::aarch64::uint8x16_t,
    da3: &mut core::arch::aarch64::uint8x16_t,
    db0: &mut core::arch::aarch64::uint8x16_t,
    db1: &mut core::arch::aarch64::uint8x16_t,
    db2: &mut core::arch::aarch64::uint8x16_t,
    db3: &mut core::arch::aarch64::uint8x16_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        let selected_table = if ODD {
            half_swapped_table_base
        } else {
            table_base
        };
        let ra = selected_table.add(a_byte * 64);
        let rb = selected_table.add(b_byte * 64);
        let va0 = vld1q_u8(ra.add((0 ^ BH) * 16));
        let va1 = vld1q_u8(ra.add((1 ^ BH) * 16));
        let va2 = vld1q_u8(ra.add((2 ^ BH) * 16));
        let va3 = vld1q_u8(ra.add((3 ^ BH) * 16));
        let vb0 = vld1q_u8(rb.add((0 ^ BH) * 16));
        let vb1 = vld1q_u8(rb.add((1 ^ BH) * 16));
        let vb2 = vld1q_u8(rb.add((2 ^ BH) * 16));
        let vb3 = vld1q_u8(rb.add((3 ^ BH) * 16));
        *da0 = veorq_u8(*da0, va0);
        *da1 = veorq_u8(*da1, va1);
        *da2 = veorq_u8(*da2, va2);
        *da3 = veorq_u8(*da3, va3);
        *db0 = veorq_u8(*db0, vb0);
        *db1 = veorq_u8(*db1, vb1);
        *db2 = veorq_u8(*db2, vb2);
        *db3 = veorq_u8(*db3, vb3);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn xor_apply_byte_pair_into_8_regs<
    const BH0: usize,
    const ODD0: bool,
    const BH1: usize,
    const ODD1: bool,
>(
    table_base: *const u8,
    half_swapped_table_base: *const u8,
    a_byte0: usize,
    b_byte0: usize,
    a_byte1: usize,
    b_byte1: usize,
    da0: &mut core::arch::aarch64::uint8x16_t,
    da1: &mut core::arch::aarch64::uint8x16_t,
    da2: &mut core::arch::aarch64::uint8x16_t,
    da3: &mut core::arch::aarch64::uint8x16_t,
    db0: &mut core::arch::aarch64::uint8x16_t,
    db1: &mut core::arch::aarch64::uint8x16_t,
    db2: &mut core::arch::aarch64::uint8x16_t,
    db3: &mut core::arch::aarch64::uint8x16_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        let table0 = if ODD0 {
            half_swapped_table_base
        } else {
            table_base
        };
        let table1 = if ODD1 {
            half_swapped_table_base
        } else {
            table_base
        };
        let ra0 = table0.add(a_byte0 * 64);
        let rb0 = table0.add(b_byte0 * 64);
        let ra1 = table1.add(a_byte1 * 64);
        let rb1 = table1.add(b_byte1 * 64);
        *da0 = xor3_u8(
            *da0,
            vld1q_u8(ra0.add((0 ^ BH0) * 16)),
            vld1q_u8(ra1.add((0 ^ BH1) * 16)),
        );
        *da1 = xor3_u8(
            *da1,
            vld1q_u8(ra0.add((1 ^ BH0) * 16)),
            vld1q_u8(ra1.add((1 ^ BH1) * 16)),
        );
        *da2 = xor3_u8(
            *da2,
            vld1q_u8(ra0.add((2 ^ BH0) * 16)),
            vld1q_u8(ra1.add((2 ^ BH1) * 16)),
        );
        *da3 = xor3_u8(
            *da3,
            vld1q_u8(ra0.add((3 ^ BH0) * 16)),
            vld1q_u8(ra1.add((3 ^ BH1) * 16)),
        );
        *db0 = xor3_u8(
            *db0,
            vld1q_u8(rb0.add((0 ^ BH0) * 16)),
            vld1q_u8(rb1.add((0 ^ BH1) * 16)),
        );
        *db1 = xor3_u8(
            *db1,
            vld1q_u8(rb0.add((1 ^ BH0) * 16)),
            vld1q_u8(rb1.add((1 ^ BH1) * 16)),
        );
        *db2 = xor3_u8(
            *db2,
            vld1q_u8(rb0.add((2 ^ BH0) * 16)),
            vld1q_u8(rb1.add((2 ^ BH1) * 16)),
        );
        *db3 = xor3_u8(
            *db3,
            vld1q_u8(rb0.add((3 ^ BH0) * 16)),
            vld1q_u8(rb1.add((3 ^ BH1) * 16)),
        );
    }
}

/// Process one K-row: 8 byte positions of `a` and `b` via the inv_NTT table,
/// F_8 multiply, widen-shift by K, XOR into the four `(acc_lo, acc_hi)` pairs.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn fused_apply_one_k<const K: i32>(
    table_base: *const u8,
    half_swapped_table_base: *const u8,
    a_row: *const u8,
    b_row: *const u8,
    acc0_lo: &mut core::arch::aarch64::uint16x8_t,
    acc0_hi: &mut core::arch::aarch64::uint16x8_t,
    acc1_lo: &mut core::arch::aarch64::uint16x8_t,
    acc1_hi: &mut core::arch::aarch64::uint16x8_t,
    acc2_lo: &mut core::arch::aarch64::uint16x8_t,
    acc2_hi: &mut core::arch::aarch64::uint16x8_t,
    acc3_lo: &mut core::arch::aarch64::uint16x8_t,
    acc3_hi: &mut core::arch::aarch64::uint16x8_t,
) {
    use crate::field::gf2_8::neon::gf8_mul_vec16;
    use core::arch::aarch64::*;
    unsafe {
        let a_word = u64::from_le(core::ptr::read_unaligned(a_row.cast::<u64>()));
        let b_word = u64::from_le(core::ptr::read_unaligned(b_row.cast::<u64>()));

        // b = 0: identity permutation — plain load of the 4 chunks.
        let ra0 = table_base.add(byte_from_word::<0>(a_word) * 64);
        let rb0 = table_base.add(byte_from_word::<0>(b_word) * 64);
        let mut da0 = vld1q_u8(ra0);
        let mut da1 = vld1q_u8(ra0.add(16));
        let mut da2 = vld1q_u8(ra0.add(32));
        let mut da3 = vld1q_u8(ra0.add(48));
        let mut db0 = vld1q_u8(rb0);
        let mut db1 = vld1q_u8(rb0.add(16));
        let mut db2 = vld1q_u8(rb0.add(32));
        let mut db3 = vld1q_u8(rb0.add(48));

        // b = 1..6: consume two table rows per EOR3, permuted per (BH, ODD).
        xor_apply_byte_pair_into_8_regs::<0, true, 1, false>(
            table_base,
            half_swapped_table_base,
            byte_from_word::<1>(a_word),
            byte_from_word::<1>(b_word),
            byte_from_word::<2>(a_word),
            byte_from_word::<2>(b_word),
            &mut da0,
            &mut da1,
            &mut da2,
            &mut da3,
            &mut db0,
            &mut db1,
            &mut db2,
            &mut db3,
        );
        xor_apply_byte_pair_into_8_regs::<1, true, 2, false>(
            table_base,
            half_swapped_table_base,
            byte_from_word::<3>(a_word),
            byte_from_word::<3>(b_word),
            byte_from_word::<4>(a_word),
            byte_from_word::<4>(b_word),
            &mut da0,
            &mut da1,
            &mut da2,
            &mut da3,
            &mut db0,
            &mut db1,
            &mut db2,
            &mut db3,
        );
        xor_apply_byte_pair_into_8_regs::<2, true, 3, false>(
            table_base,
            half_swapped_table_base,
            byte_from_word::<5>(a_word),
            byte_from_word::<5>(b_word),
            byte_from_word::<6>(a_word),
            byte_from_word::<6>(b_word),
            &mut da0,
            &mut da1,
            &mut da2,
            &mut da3,
            &mut db0,
            &mut db1,
            &mut db2,
            &mut db3,
        );
        // b = 7: one trailing row.
        xor_apply_byte_into_8_regs::<3, true>(
            table_base,
            half_swapped_table_base,
            byte_from_word::<7>(a_word),
            byte_from_word::<7>(b_word),
            &mut da0,
            &mut da1,
            &mut da2,
            &mut da3,
            &mut db0,
            &mut db1,
            &mut db2,
            &mut db3,
        );

        // F_8 multiply lane-wise (4 × 16 lanes = 64 total).
        let y0 = gf8_mul_vec16(da0, db0);
        let y1 = gf8_mul_vec16(da1, db1);
        let y2 = gf8_mul_vec16(da2, db2);
        let y3 = gf8_mul_vec16(da3, db3);

        // Widen-shift by K, XOR into the 16-bit accumulators.
        *acc0_lo = veorq_u16(*acc0_lo, vshll_n_u8::<K>(vget_low_u8(y0)));
        *acc0_hi = veorq_u16(*acc0_hi, vshll_n_u8::<K>(vget_high_u8(y0)));
        *acc1_lo = veorq_u16(*acc1_lo, vshll_n_u8::<K>(vget_low_u8(y1)));
        *acc1_hi = veorq_u16(*acc1_hi, vshll_n_u8::<K>(vget_high_u8(y1)));
        *acc2_lo = veorq_u16(*acc2_lo, vshll_n_u8::<K>(vget_low_u8(y2)));
        *acc2_hi = veorq_u16(*acc2_hi, vshll_n_u8::<K>(vget_high_u8(y2)));
        *acc3_lo = veorq_u16(*acc3_lo, vshll_n_u8::<K>(vget_low_u8(y3)));
        *acc3_hi = veorq_u16(*acc3_hi, vshll_n_u8::<K>(vget_high_u8(y3)));
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) fn shift_reduce_inner_ab_fused_neon(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; 64],
    nt_store: bool,
) {
    use crate::field::gf2_8::neon::gf8_reduce_vec16;
    use core::arch::aarch64::*;

    let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;
    let table_base = inv_table.data_ptr();
    let half_swapped_table_base = inv_table.half_swapped_data_ptr();

    unsafe {
        let mut acc0_lo = vdupq_n_u16(0);
        let mut acc0_hi = vdupq_n_u16(0);
        let mut acc1_lo = vdupq_n_u16(0);
        let mut acc1_hi = vdupq_n_u16(0);
        let mut acc2_lo = vdupq_n_u16(0);
        let mut acc2_hi = vdupq_n_u16(0);
        let mut acc3_lo = vdupq_n_u16(0);
        let mut acc3_hi = vdupq_n_u16(0);

        // 8 K-iterations — each consumes N_CHUNKS = 8 packed witness bytes
        // for `a` and `b`. K is a const generic so `vshll_n_u8::<K>` specializes.
        macro_rules! do_k {
            ($k:literal) => {{
                let off = byte_base_b + $k * N_CHUNKS;
                fused_apply_one_k::<$k>(
                    table_base,
                    half_swapped_table_base,
                    a_packed.as_ptr().add(off),
                    b_packed.as_ptr().add(off),
                    &mut acc0_lo,
                    &mut acc0_hi,
                    &mut acc1_lo,
                    &mut acc1_hi,
                    &mut acc2_lo,
                    &mut acc2_hi,
                    &mut acc3_lo,
                    &mut acc3_hi,
                );
            }};
        }
        do_k!(0);
        do_k!(1);
        do_k!(2);
        do_k!(3);
        do_k!(4);
        do_k!(5);
        do_k!(6);
        do_k!(7);

        // Reduce 16-bit accs → 16-byte F_8 results (4 × 16 lanes).
        let r0 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc0_lo), vreinterpretq_u8_u16(acc0_hi));
        let r1 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc1_lo), vreinterpretq_u8_u16(acc1_hi));
        let r2 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc2_lo), vreinterpretq_u8_u16(acc2_hi));
        let r3 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc3_lo), vreinterpretq_u8_u16(acc3_hi));

        store_row_64(out.as_mut_ptr(), nt_store, r0, r1, r2, r3);
    }
}

// ---------------------------------------------------------------------------
// Horner / scaled-table shift-reduce (compute reduction over the kernel above).
//
// The incumbent spends, per 16-lane group and per K row, two PMULLs plus a
// full `gf8_reduce_vec16` (uzp, uzp, and, ushr, tbl, tbl, eor3) plus a
// widen-shift-XOR accumulate (ushll, ushll2, eor, eor) — 13 vector ops — and
// then reduces the 16-bit accumulator a *second* time at the end. Both of
// those reductions are redundant. `reduce` is a ring homomorphism onto
// GF(2^8), so
//
//     out = reduce( Σ_K reduce(A_K·B_K)·x^K ) = reduce( Σ_K (A_K·B_K)·x^K )
//
// with `A_K·B_K` the raw 15-bit PMULL output. Raw products can be
// XOR-accumulated directly provided the `x^K` weights never push them past 16
// bits, which Horner guarantees when the carry out of bit 15 is folded back
// with `x^16 mod p`:
//
//     out = reduce( ((d3·x + d2)·x + d1)·x + d0 )
//     d_r = A_r·B_r  ⊕  x^4·A_{r+4}·B_{r+4}
//
// The `x^4` on the upper half costs nothing: the byte-collapse transform `T`
// is GF(2)-linear, so `x^4·T(w)` is `T` applied out of a pre-scaled table
// image (`InvNttTableByteSingleGf8::scaled_x4_data_ptr`).
//
// Per 16-lane group the four Horner rows cost `4·2 PMULL + 3·(shl, sshr,
// bcax, eor3) + 1 eor = 8 + 12 + 1` ops per accumulator register pair, i.e.
// 24 vector ops per 8-lane register against the incumbent's 52 — the
// multiply/reduce/accumulate block drops from 416 to 192 vector ops per
// 64-byte output block, with the table-lookup network and the single final
// `gf8_reduce_vec16` unchanged.
// ---------------------------------------------------------------------------

/// `x^16 mod (x^8 + x^4 + x^3 + x + 1)` — the weight of the bit shifted out of
/// a 16-bit lane by one Horner step. Verified against `gf8_reduce` in
/// `horner_carry_constant_matches_field`.
#[cfg(target_arch = "aarch64")]
pub(crate) const HORNER_CARRY_X16: u16 = 0x5e;

#[cfg(all(target_arch = "aarch64", target_feature = "sha3"))]
#[inline(always)]
unsafe fn bcax_u8(
    a: core::arch::aarch64::uint8x16_t,
    b: core::arch::aarch64::uint8x16_t,
    c: core::arch::aarch64::uint8x16_t,
) -> core::arch::aarch64::uint8x16_t {
    unsafe { core::arch::aarch64::vbcaxq_u8(a, b, c) }
}

#[cfg(all(target_arch = "aarch64", not(target_feature = "sha3")))]
#[inline(always)]
unsafe fn bcax_u8(
    a: core::arch::aarch64::uint8x16_t,
    b: core::arch::aarch64::uint8x16_t,
    c: core::arch::aarch64::uint8x16_t,
) -> core::arch::aarch64::uint8x16_t {
    unsafe { core::arch::aarch64::veorq_u8(a, core::arch::aarch64::vbicq_u8(b, c)) }
}

/// Carry-less 8×8→16 product of the low halves, kept in 16-bit lanes.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn pmull_lo_u16(
    a: core::arch::aarch64::uint8x16_t,
    b: core::arch::aarch64::uint8x16_t,
) -> core::arch::aarch64::uint16x8_t {
    use core::arch::aarch64::*;
    unsafe {
        vreinterpretq_u16_p16(vmull_p8(
            core::mem::transmute::<uint8x8_t, poly8x8_t>(vget_low_u8(a)),
            core::mem::transmute::<uint8x8_t, poly8x8_t>(vget_low_u8(b)),
        ))
    }
}

/// Carry-less 8×8→16 product of the high halves (selects `pmull2`).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn pmull_hi_u16(
    a: core::arch::aarch64::uint8x16_t,
    b: core::arch::aarch64::uint8x16_t,
) -> core::arch::aarch64::uint16x8_t {
    use core::arch::aarch64::*;
    unsafe {
        vreinterpretq_u16_p16(vmull_p8(
            core::mem::transmute::<uint8x8_t, poly8x8_t>(vget_high_u8(a)),
            core::mem::transmute::<uint8x8_t, poly8x8_t>(vget_high_u8(b)),
        ))
    }
}

/// One Horner step: `acc = acc·x ⊕ lo ⊕ hi`, in the 16-bit unreduced domain.
/// The bit leaving lane bit 15 re-enters as `x^16 mod p`, which keeps every
/// accumulator lane congruent to the true field value while staying 16-bit.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn horner_absorb(
    acc: &mut [core::arch::aarch64::uint16x8_t; 8],
    lo: &[core::arch::aarch64::uint16x8_t; 8],
    hi: &[core::arch::aarch64::uint16x8_t; 8],
    carry_not: core::arch::aarch64::uint8x16_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        for i in 0..8 {
            let cur = acc[i];
            let shifted = vreinterpretq_u8_u16(vshlq_n_u16::<1>(cur));
            let msb = vreinterpretq_u8_s16(vshrq_n_s16::<15>(vreinterpretq_s16_u16(cur)));
            // shifted ^ (msb & HORNER_CARRY_X16)
            let folded = bcax_u8(shifted, msb, carry_not);
            acc[i] = vreinterpretq_u16_u8(xor3_u8(
                folded,
                vreinterpretq_u8_u16(lo[i]),
                vreinterpretq_u8_u16(hi[i]),
            ));
        }
    }
}

/// First Horner step after the seed: `acc = acc·x ⊕ lo ⊕ hi` with no carry.
/// The seed is the XOR of two raw 8×8 carry-less products, so every lane has
/// degree at most 14 and bit 15 is zero; the carry term of [`horner_absorb`]
/// is identically zero there, and this form is bit-identical to it.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn horner_absorb_first(
    acc: &mut [core::arch::aarch64::uint16x8_t; 8],
    lo: &[core::arch::aarch64::uint16x8_t; 8],
    hi: &[core::arch::aarch64::uint16x8_t; 8],
) {
    use core::arch::aarch64::*;
    unsafe {
        for i in 0..8 {
            debug_assert_eq!(vmaxvq_u16(vshrq_n_u16::<15>(acc[i])), 0);
            acc[i] = vreinterpretq_u16_u8(xor3_u8(
                vreinterpretq_u8_u16(vshlq_n_u16::<1>(acc[i])),
                vreinterpretq_u8_u16(lo[i]),
                vreinterpretq_u8_u16(hi[i]),
            ));
        }
    }
}

/// Row-body variant bit: the static-B Horner rows fold the bits a Horner
/// step shifts out of the 16-bit lanes once, two degrees at a time, instead
/// of once per step. After the seed and the first absorb the accumulator is
/// `T = x·P3 + P2` (degree at most 15, see [`horner_absorb_first`]); the
/// K1 step forms `x^2·T + x·P1` as `(T << 2) ^ fold(T >> 14) ^ (P1 << 1)`
/// with `fold(c) = c0·x^16 + c1·x^17 mod p` (one `tbl`), and the K0 step is
/// a plain `eor3` with `P0`. Six plus one vector ops per lane instead of
/// four plus four; every lane stays congruent mod p to the incumbent's, so
/// `gf8_reduce_vec16` writes the same bytes.
#[cfg(target_arch = "aarch64")]
pub(crate) const ABV_HORNER2: u32 = 2048;

/// Kill switch for [`ABV_HORNER2`]: any value of `FLOCK_NO_ZC_AB_HORNER2`
/// keeps the per-step carry fold. Read once per process. Same bytes.
#[cfg(target_arch = "aarch64")]
pub(crate) fn ab_horner2_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_ZC_AB_HORNER2").is_none())
}

/// `tbl` table of [`ABV_HORNER2`]: entry `c` (bit 0 = lane bit 14, bit 1 =
/// lane bit 15 of `T`) is `c0·x^16 + c1·x^17 mod p`; entries 4..15 unused.
#[cfg(target_arch = "aarch64")]
pub(crate) const HORNER2_FOLD: [u8; 16] = {
    let x17 = (HORNER_CARRY_X16 as u8) << 1; // x^16 mod p < 0x80
    [0, HORNER_CARRY_X16 as u8, x17, (HORNER_CARRY_X16 as u8) ^ x17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
};

/// The K1 Horner step of the static-B rows in the form `V` selects:
/// [`horner_absorb`] or, under [`ABV_HORNER2`], `acc = (acc << 2) ^
/// fold(acc >> 14) ^ ((lo ^ hi) << 1)` (the accumulator then holds
/// `x^2·T + x·P1`, which [`horner_absorb_k0`] completes).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn horner_absorb_k1<const V: u32>(
    acc: &mut [core::arch::aarch64::uint16x8_t; 8],
    lo: &[core::arch::aarch64::uint16x8_t; 8],
    hi: &[core::arch::aarch64::uint16x8_t; 8],
    carry_not: core::arch::aarch64::uint8x16_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        if V & ABV_HORNER2 == 0 {
            horner_absorb(acc, lo, hi, carry_not);
            return;
        }
        let fold = vld1q_u8(HORNER2_FOLD.as_ptr());
        for i in 0..8 {
            let t = acc[i];
            let s2 = vreinterpretq_u8_u16(vshlq_n_u16::<2>(t));
            let f = vqtbl1q_u8(fold, vreinterpretq_u8_u16(vshrq_n_u16::<14>(t)));
            let p1 = vreinterpretq_u8_u16(vshlq_n_u16::<1>(veorq_u16(lo[i], hi[i])));
            acc[i] = vreinterpretq_u16_u8(xor3_u8(s2, f, p1));
        }
    }
}

/// The K0 Horner step of the static-B rows in the form `V` selects:
/// [`horner_absorb`] or, under [`ABV_HORNER2`], `acc ^= lo ^ hi`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn horner_absorb_k0<const V: u32>(
    acc: &mut [core::arch::aarch64::uint16x8_t; 8],
    lo: &[core::arch::aarch64::uint16x8_t; 8],
    hi: &[core::arch::aarch64::uint16x8_t; 8],
    carry_not: core::arch::aarch64::uint8x16_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        if V & ABV_HORNER2 == 0 {
            horner_absorb(acc, lo, hi, carry_not);
            return;
        }
        for i in 0..8 {
            acc[i] = vreinterpretq_u16_u8(xor3_u8(
                vreinterpretq_u8_u16(acc[i]),
                vreinterpretq_u8_u16(lo[i]),
                vreinterpretq_u8_u16(hi[i]),
            ));
        }
    }
}

/// Row-body variant bit (with [`ABV_HORNER2`]): the static-B Horner rows
/// accumulate every 8-bit product as an adjacent `pmull`/`eor` pair into the
/// running sum (see [`fused_pmull8_acc`]), which Apple P-cores issue as one
/// SIMD µop, instead of forming each product in its own register and
/// combining them with `eor`/`eor3`. The seed is `P3 ^ P7'`, the first
/// absorb `(acc << 1) ^ P2 ^ P6'`, the K1 step folds `P1 ^ P5'` (summed the
/// same way) as before and the K0 step adds `P0 ^ P4'`: the same XOR of the
/// same products in every lane, so `gf8_reduce_vec16` writes the same bytes.
#[cfg(target_arch = "aarch64")]
pub(crate) const ABV_FUSE: u32 = 1 << 16;


/// Kill switch for [`ABV_FUSE`] in the main-pool (P-core) window rows: any
/// value of `FLOCK_NO_ZC_AB_FUSE` keeps the separate products. Read once per
/// process. The rows written are byte-identical.
#[cfg(target_arch = "aarch64")]
pub(crate) fn ab_fuse_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_ZC_AB_FUSE").is_none())
}

/// Kill switch for [`ABV_FUSE`] in the helper-thread (half-pair z) window
/// rows: any value of `FLOCK_NO_ZC_AB_FUSE_E` keeps the separate products
/// there. Read once per process. The rows written are byte-identical.
#[cfg(target_arch = "aarch64")]
pub(crate) fn ab_fuse_e_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_ZC_AB_FUSE_E").is_none())
}

/// The operands of one K row in the h4 layout: the four A image registers
/// and the four B image registers.
#[cfg(target_arch = "aarch64")]
type AbKOperands = ([core::arch::aarch64::uint8x16_t; 4], [core::arch::aarch64::uint8x16_t; 4]);

/// The eight raw products of one K row (lane order of `prods!`).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn fuse_products(o: &AbKOperands) -> [core::arch::aarch64::uint16x8_t; 8] {
    unsafe {
        let (a, b) = o;
        [
            pmull_lo_u16(a[0], b[0]),
            pmull_hi_u16(a[0], b[0]),
            pmull_lo_u16(a[1], b[1]),
            pmull_hi_u16(a[1], b[1]),
            pmull_lo_u16(a[2], b[2]),
            pmull_hi_u16(a[2], b[2]),
            pmull_lo_u16(a[3], b[3]),
            pmull_hi_u16(a[3], b[3]),
        ]
    }
}

/// `acc[i] ^ product[i]` for the eight products of one K row, each as one
/// fused `pmull`/`eor` pair.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn fuse_acc(
    acc: &[core::arch::aarch64::uint16x8_t; 8],
    o: &AbKOperands,
) -> [core::arch::aarch64::uint16x8_t; 8] {
    unsafe {
        let (a, b) = o;
        [
            fused_pmull8_acc::<false>(acc[0], a[0], b[0]),
            fused_pmull8_acc::<true>(acc[1], a[0], b[0]),
            fused_pmull8_acc::<false>(acc[2], a[1], b[1]),
            fused_pmull8_acc::<true>(acc[3], a[1], b[1]),
            fused_pmull8_acc::<false>(acc[4], a[2], b[2]),
            fused_pmull8_acc::<true>(acc[5], a[2], b[2]),
            fused_pmull8_acc::<false>(acc[6], a[3], b[3]),
            fused_pmull8_acc::<true>(acc[7], a[3], b[3]),
        ]
    }
}

/// [`ABV_FUSE`] seed: `P3 ^ P7'`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn fuse_seed(lo: &AbKOperands, hi: &AbKOperands) -> [core::arch::aarch64::uint16x8_t; 8] {
    unsafe { fuse_acc(&fuse_products(lo), hi) }
}

/// [`ABV_FUSE`] form of [`horner_absorb_first`]: `(acc << 1) ^ P2 ^ P6'`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn fuse_absorb_first(
    acc: &mut [core::arch::aarch64::uint16x8_t; 8],
    lo: &AbKOperands,
    hi: &AbKOperands,
) {
    use core::arch::aarch64::*;
    unsafe {
        let mut s = *acc;
        for v in s.iter_mut() {
            debug_assert_eq!(vmaxvq_u16(vshrq_n_u16::<15>(*v)), 0);
            *v = vshlq_n_u16::<1>(*v);
        }
        *acc = fuse_acc(&fuse_acc(&s, lo), hi);
    }
}

/// [`ABV_FUSE`] form of the [`ABV_HORNER2`] K1 step:
/// `acc = (acc << 2) ^ fold(acc >> 14) ^ ((P1 ^ P5') << 1)`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn fuse_absorb_k1(
    acc: &mut [core::arch::aarch64::uint16x8_t; 8],
    lo: &AbKOperands,
    hi: &AbKOperands,
) {
    use core::arch::aarch64::*;
    unsafe {
        let p = fuse_acc(&fuse_products(lo), hi);
        let fold = vld1q_u8(HORNER2_FOLD.as_ptr());
        for i in 0..8 {
            let t = acc[i];
            let s2 = vreinterpretq_u8_u16(vshlq_n_u16::<2>(t));
            let f = vqtbl1q_u8(fold, vreinterpretq_u8_u16(vshrq_n_u16::<14>(t)));
            let p1 = vreinterpretq_u8_u16(vshlq_n_u16::<1>(p[i]));
            acc[i] = vreinterpretq_u16_u8(xor3_u8(s2, f, p1));
        }
    }
}

/// [`ABV_FUSE`] form of the [`ABV_HORNER2`] K0 step: `acc ^ P0 ^ P4'`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn fuse_absorb_k0(
    acc: &mut [core::arch::aarch64::uint16x8_t; 8],
    lo: &AbKOperands,
    hi: &AbKOperands,
) {
    unsafe {
        *acc = fuse_acc(&fuse_acc(acc, lo), hi);
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
#[test]
fn horner2_fold_table_matches_field() {
    let x16 = (0..16).fold(F8::ONE, |a, _| a * F8(2));
    let x17 = x16 * F8(2);
    assert_eq!(HORNER2_FOLD[..4], [0, x16.0, x17.0, x16.0 ^ x17.0]);
}

/// Kill switch for the Horner / scaled-table kernel:
/// `FLOCK_NO_FAST_SHIFT_REDUCE=1` restores [`shift_reduce_inner_ab_fused_neon`].
/// Output is bit-identical either way.
#[cfg(target_arch = "aarch64")]
pub(crate) fn fast_shift_reduce_enabled() -> bool {
    use std::sync::LazyLock;
    static FLAG: LazyLock<bool> =
        LazyLock::new(|| std::env::var_os("FLOCK_NO_FAST_SHIFT_REDUCE").is_none());
    *FLAG
}

#[cfg(target_arch = "aarch64")]
#[inline(never)]
pub(crate) fn shift_reduce_inner_ab_fused_neon_h4<const V: u32>(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; 64],
    nt_store: bool,
) {
    use crate::field::gf2_8::neon::gf8_reduce_vec16;
    use core::arch::aarch64::*;

    let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;
    let plain = inv_table.data_ptr();
    let plain_sw = inv_table.half_swapped_data_ptr();
    let scaled = inv_table.scaled_x4_data_ptr();
    let scaled_sw = inv_table.scaled_x4_half_swapped_data_ptr();
    // Karatsuba images (see `ab_kara_tables`); dangling and unused otherwise.
    let kplain = if V & ABV_KARA != 0 {
        ab_kara_tables(inv_table)
    } else {
        core::ptr::null()
    };
    let kscaled = kplain.wrapping_add(AB_KARA_IMAGE_BYTES);

    unsafe {
        let a_base = a_packed.as_ptr().add(byte_base_b);
        let b_base = b_packed.as_ptr().add(byte_base_b);

        // Raw 15-bit products of one K row. `A` comes out of `$at`, which is
        // the plain image for K < 4 and the `x^4`-scaled image for K ≥ 4; `B`
        // always comes out of the plain image.
        macro_rules! products {
            ($at:expr, $at_sw:expr, $kara:expr, $k:literal) => {{
                let aw = u64::from_le(core::ptr::read_unaligned(
                    a_base.add($k * N_CHUNKS).cast::<u64>(),
                ));
                let bw = u64::from_le(core::ptr::read_unaligned(
                    b_base.add($k * N_CHUNKS).cast::<u64>(),
                ));
                let (da0, da1, da2, da3) = apply_word_into_4_regs_v::<V, { $k >= 4 }>($at, $at_sw, $kara, aw);
                let (db0, db1, db2, db3) = apply_word_into_4_regs_v::<V, false>(plain, plain_sw, kplain, bw);
                [
                    pmull_lo_u16(da0, db0),
                    pmull_hi_u16(da0, db0),
                    pmull_lo_u16(da1, db1),
                    pmull_hi_u16(da1, db1),
                    pmull_lo_u16(da2, db2),
                    pmull_hi_u16(da2, db2),
                    pmull_lo_u16(da3, db3),
                    pmull_hi_u16(da3, db3),
                ]
            }};
        }

        // d_3 seeds the Horner chain; no shift is due before the first row.
        let lo3 = products!(plain, plain_sw, kplain, 3);
        let hi3 = products!(scaled, scaled_sw, kscaled, 7);
        let mut acc = [
            veorq_u16(lo3[0], hi3[0]),
            veorq_u16(lo3[1], hi3[1]),
            veorq_u16(lo3[2], hi3[2]),
            veorq_u16(lo3[3], hi3[3]),
            veorq_u16(lo3[4], hi3[4]),
            veorq_u16(lo3[5], hi3[5]),
            veorq_u16(lo3[6], hi3[6]),
            veorq_u16(lo3[7], hi3[7]),
        ];

        let carry_not = vreinterpretq_u8_u16(vdupq_n_u16(!HORNER_CARRY_X16));

        let lo2 = products!(plain, plain_sw, kplain, 2);
        let hi2 = products!(scaled, scaled_sw, kscaled, 6);
        horner_absorb_first(&mut acc, &lo2, &hi2);

        let lo1 = products!(plain, plain_sw, kplain, 1);
        let hi1 = products!(scaled, scaled_sw, kscaled, 5);
        horner_absorb(&mut acc, &lo1, &hi1, carry_not);

        let lo0 = products!(plain, plain_sw, kplain, 0);
        let hi0 = products!(scaled, scaled_sw, kscaled, 4);
        horner_absorb(&mut acc, &lo0, &hi0, carry_not);

        let r0 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc[0]), vreinterpretq_u8_u16(acc[1]));
        let r1 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc[2]), vreinterpretq_u8_u16(acc[3]));
        let r2 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc[4]), vreinterpretq_u8_u16(acc[5]));
        let r3 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc[6]), vreinterpretq_u8_u16(acc[7]));

        store_row_64(out.as_mut_ptr(), nt_store, r0, r1, r2, r3);
    }
}

/// Cached inverse transform for the ranked BLAKE3 `(window 0, b_med 2, K1)`
/// A word. The dispatcher exact-matches the word before using this image, so
/// arbitrary witnesses retain the generic table-lookup path.
#[cfg(target_arch = "aarch64")]
fn static_a_k1_partial(inv_table: &InvNttTableByteSingleGf8) -> &'static [u8; 64] {
    use std::sync::OnceLock;

    const STATIC_A_K1: u64 = 0x0000_0016_0000_0080;
    static PARTIAL: OnceLock<[u8; 64]> = OnceLock::new();
    PARTIAL.get_or_init(|| {
        let mut transformed = [F8::ZERO; 64];
        inv_table.apply(&STATIC_A_K1.to_le_bytes(), &mut transformed);
        core::array::from_fn(|i| transformed[i].0)
    })
}

/// Horner / scaled-table twin of [`shift_reduce_inner_mixed_const_b`]. Rows
/// selected by `ONE_MASK` have B = 1 and therefore widen the transformed A
/// bytes directly; all other rows form the same raw PMULL products as the h4
/// generic kernel. `STATIC_A` additionally replaces the exact ranked K1 A
/// transform's 32 table gathers with four contiguous vector loads and skips
/// the three statically-zero high bytes of K0.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn shift_reduce_inner_mixed_const_b_h4<
    const ONE_MASK: u8,
    const STATIC_A: bool,
    const A_LOW5_K: u8,
    const V: u32,
>(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    byte_base_b: usize,
    a_k1_partial: Option<&[u8; 64]>,
    out: &mut [u8; 64],
    nt_store: bool,
) {
    use crate::field::gf2_8::neon::gf8_reduce_vec16;
    use core::arch::aarch64::*;

    let plain = inv_table.data_ptr();
    let plain_sw = inv_table.half_swapped_data_ptr();
    let scaled = inv_table.scaled_x4_data_ptr();
    let scaled_sw = inv_table.scaled_x4_half_swapped_data_ptr();
    // Karatsuba images (see `ab_kara_tables`), or under `ABV_HZ` the sigma
    // tables (see `ab_hz_tables`); dangling and unused otherwise.
    let kplain = if V & ABV_KARA != 0 {
        ab_kara_tables(inv_table)
    } else if V & ABV_HZ != 0 {
        ab_hz_tables(inv_table)
    } else {
        core::ptr::null()
    };
    let kscaled = kplain.wrapping_add(AB_KARA_IMAGE_BYTES);

    #[inline(always)]
    unsafe fn widen_lo(v: uint8x16_t) -> uint16x8_t {
        unsafe { vmovl_u8(vget_low_u8(v)) }
    }
    #[inline(always)]
    unsafe fn widen_hi(v: uint8x16_t) -> uint16x8_t {
        unsafe { vmovl_u8(vget_high_u8(v)) }
    }

    unsafe {
        let a_base = a_packed.as_ptr().add(byte_base_b);
        let b_base = b_packed.as_ptr().add(byte_base_b);

        // The A image of K word `$k` (each form as described inline).
        macro_rules! a_img {
            ($at:expr, $at_sw:expr, $kara:expr, $k:literal) => {{
                if STATIC_A && $k == 1 {
                    let p = a_k1_partial.unwrap_unchecked().as_ptr();
                    (
                        vld1q_u8(p),
                        vld1q_u8(p.add(16)),
                        vld1q_u8(p.add(32)),
                        vld1q_u8(p.add(48)),
                    )
                } else if STATIC_A && $k == 0 {
                    let aw = u64::from_le(core::ptr::read_unaligned(a_base.cast::<u64>()));
                    if V & ABV_KARA != 0 {
                        apply_word_into_4_regs_v::<V, { $k >= 4 }>($at, $at_sw, $kara, aw)
                    } else {
                        apply_word_low5_into_4_regs($at, $at_sw, aw)
                    }
                } else if A_LOW5_K & (1 << $k) != 0 {
                    // Ranked (window 0, b_med 2): A K0 and K1 words are
                    // witness-dependent only in their low five bytes — the
                    // top three bytes are structurally zero (2048/2048
                    // measured on the scored distribution: K1 packs the
                    // fixed block_len/counter word 0x0000_0016_0000_0080,
                    // K0 a u32-magnitude value with a byte-4 carry). Skipping
                    // bytes 5-7 of each word drops twelve table loads and
                    // six XORs on every scored block; bit-identical because
                    // table row 0 is the zero vector. The caller's
                    // ((a_k0 | a_k1) & 0xffff_ff00_0000_0000) == 0 guard
                    // keeps the path witness-safe.
                    let aw = u64::from_le(core::ptr::read_unaligned(
                        a_base.add($k * N_CHUNKS).cast::<u64>(),
                    ));
                    // Table row 0 is zero, so with bytes 5..7 zero the
                    // whole-word Karatsuba apply (18 loads) is this row's
                    // five-byte apply (20 loads).
                    if V & ABV_KARA != 0 {
                        apply_word_into_4_regs_v::<V, { $k >= 4 }>($at, $at_sw, $kara, aw)
                    } else {
                        apply_word_low5_into_4_regs($at, $at_sw, aw)
                    }
                } else {
                    let aw = u64::from_le(core::ptr::read_unaligned(
                        a_base.add($k * N_CHUNKS).cast::<u64>(),
                    ));
                    apply_word_into_4_regs_v::<V, { $k >= 4 }>($at, $at_sw, $kara, aw)
                }
            }};
        }

        macro_rules! products {
            ($at:expr, $at_sw:expr, $kara:expr, $k:literal) => {{
                let (da0, da1, da2, da3) = a_img!($at, $at_sw, $kara, $k);
                if ONE_MASK & (1 << $k) != 0 {
                    [
                        widen_lo(da0),
                        widen_hi(da0),
                        widen_lo(da1),
                        widen_hi(da1),
                        widen_lo(da2),
                        widen_hi(da2),
                        widen_lo(da3),
                        widen_hi(da3),
                    ]
                } else {
                    let bw = u64::from_le(core::ptr::read_unaligned(
                        b_base.add($k * N_CHUNKS).cast::<u64>(),
                    ));
                    let (db0, db1, db2, db3) = apply_word_into_4_regs_v::<V, false>(plain, plain_sw, kplain, bw);
                    [
                        pmull_lo_u16(da0, db0),
                        pmull_hi_u16(da0, db0),
                        pmull_lo_u16(da1, db1),
                        pmull_hi_u16(da1, db1),
                        pmull_lo_u16(da2, db2),
                        pmull_hi_u16(da2, db2),
                        pmull_lo_u16(da3, db3),
                        pmull_hi_u16(da3, db3),
                    ]
                }
            }};
        }

        if V & ABV_FUSE != 0 {
            // [`ABV_FUSE`]: the window rows' fused accumulation. A
            // const-one K word's B image is the vector of ones bytes, and
            // the carry-less product of a byte with 1 is that byte widened,
            // so `pmull(A, 1)` is exactly the widened A image the other arm
            // forms; every lane then receives the XOR of the same terms.
            const { assert!(V & ABV_FUSE == 0 || (V & ABV_HORNER2 != 0 && V & ABV_HZ == 0)) };
            let ones = vdupq_n_u8(1);
            macro_rules! opnds {
                ($at:expr, $at_sw:expr, $kara:expr, $k:literal) => {{
                    let (da0, da1, da2, da3) = a_img!($at, $at_sw, $kara, $k);
                    let db = if ONE_MASK & (1 << $k) != 0 {
                        [ones; 4]
                    } else {
                        let bw = u64::from_le(core::ptr::read_unaligned(
                            b_base.add($k * N_CHUNKS).cast::<u64>(),
                        ));
                        let (db0, db1, db2, db3) = apply_word_into_4_regs_v::<V, false>(plain, plain_sw, kplain, bw);
                        [db0, db1, db2, db3]
                    };
                    ([da0, da1, da2, da3], db)
                }};
            }
            let o3 = opnds!(plain, plain_sw, kplain, 3);
            let o7 = opnds!(scaled, scaled_sw, kscaled, 7);
            let mut acc = fuse_seed(&o3, &o7);
            let o2 = opnds!(plain, plain_sw, kplain, 2);
            let o6 = opnds!(scaled, scaled_sw, kscaled, 6);
            fuse_absorb_first(&mut acc, &o2, &o6);
            let o1 = opnds!(plain, plain_sw, kplain, 1);
            let o5 = opnds!(scaled, scaled_sw, kscaled, 5);
            fuse_absorb_k1(&mut acc, &o1, &o5);
            let o0 = opnds!(plain, plain_sw, kplain, 0);
            let o4 = opnds!(scaled, scaled_sw, kscaled, 4);
            fuse_absorb_k0(&mut acc, &o0, &o4);
            let r0 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc[0]), vreinterpretq_u8_u16(acc[1]));
            let r1 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc[2]), vreinterpretq_u8_u16(acc[3]));
            let r2 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc[4]), vreinterpretq_u8_u16(acc[5]));
            let r3 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc[6]), vreinterpretq_u8_u16(acc[7]));
            store_row_64(out.as_mut_ptr(), nt_store, r0, r1, r2, r3);
            return;
        }

        let lo3 = products!(plain, plain_sw, kplain, 3);
        let hi3 = products!(scaled, scaled_sw, kscaled, 7);
        let mut acc = [
            veorq_u16(lo3[0], hi3[0]),
            veorq_u16(lo3[1], hi3[1]),
            veorq_u16(lo3[2], hi3[2]),
            veorq_u16(lo3[3], hi3[3]),
            veorq_u16(lo3[4], hi3[4]),
            veorq_u16(lo3[5], hi3[5]),
            veorq_u16(lo3[6], hi3[6]),
            veorq_u16(lo3[7], hi3[7]),
        ];

        let carry_not = vreinterpretq_u8_u16(vdupq_n_u16(!HORNER_CARRY_X16));
        if V & ABV_HZ != 0 {
            // Same terms, absorbed one image at a time: the `x^4` image's
            // products are formed after the plain image's are already in
            // `acc`, so only one set of eight products is live at a time.
            let zero = [vdupq_n_u16(0); 8];
            let lo2 = products!(plain, plain_sw, kplain, 2);
            horner_absorb_first(&mut acc, &lo2, &zero);
            let hi2 = products!(scaled, scaled_sw, kscaled, 6);
            xor_into_acc(&mut acc, &hi2);
            let lo1 = products!(plain, plain_sw, kplain, 1);
            horner_absorb_one(&mut acc, &lo1, carry_not);
            let hi1 = products!(scaled, scaled_sw, kscaled, 5);
            xor_into_acc(&mut acc, &hi1);
            let lo0 = products!(plain, plain_sw, kplain, 0);
            horner_absorb_one(&mut acc, &lo0, carry_not);
            let hi0 = products!(scaled, scaled_sw, kscaled, 4);
            xor_into_acc(&mut acc, &hi0);
        } else {
            let lo2 = products!(plain, plain_sw, kplain, 2);
            let hi2 = products!(scaled, scaled_sw, kscaled, 6);
            horner_absorb_first(&mut acc, &lo2, &hi2);
            let lo1 = products!(plain, plain_sw, kplain, 1);
            let hi1 = products!(scaled, scaled_sw, kscaled, 5);
            // `horner_absorb` itself unless `V` has `ABV_HORNER2`.
            horner_absorb_k1::<V>(&mut acc, &lo1, &hi1, carry_not);
            let lo0 = products!(plain, plain_sw, kplain, 0);
            let hi0 = products!(scaled, scaled_sw, kscaled, 4);
            horner_absorb_k0::<V>(&mut acc, &lo0, &hi0, carry_not);
        }

        let r0 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc[0]), vreinterpretq_u8_u16(acc[1]));
        let r1 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc[2]), vreinterpretq_u8_u16(acc[3]));
        let r2 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc[4]), vreinterpretq_u8_u16(acc[5]));
        let r3 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc[6]), vreinterpretq_u8_u16(acc[7]));

        store_row_64(out.as_mut_ptr(), nt_store, r0, r1, r2, r3);
    }
}

/// [`horner_absorb`] with a single product set: `acc = acc·x ⊕ lo`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn horner_absorb_one(
    acc: &mut [core::arch::aarch64::uint16x8_t; 8],
    lo: &[core::arch::aarch64::uint16x8_t; 8],
    carry_not: core::arch::aarch64::uint8x16_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        for i in 0..8 {
            let cur = acc[i];
            let shifted = vreinterpretq_u8_u16(vshlq_n_u16::<1>(cur));
            let msb = vreinterpretq_u8_s16(vshrq_n_s16::<15>(vreinterpretq_s16_u16(cur)));
            let folded = bcax_u8(shifted, msb, carry_not);
            acc[i] = vreinterpretq_u16_u8(veorq_u8(folded, vreinterpretq_u8_u16(lo[i])));
        }
    }
}

/// `acc ⊕= p`, lane by lane.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn xor_into_acc(acc: &mut [core::arch::aarch64::uint16x8_t; 8], p: &[core::arch::aarch64::uint16x8_t; 8]) {
    use core::arch::aarch64::*;
    unsafe {
        for i in 0..8 {
            acc[i] = veorq_u16(acc[i], p[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// EXPERIMENT (seam-urm): checked static-structure fast paths.
//
// BLAKE3 witness structure (measured on real witnesses, 4096 blocks):
//  * b_med blocks 0 and 1 of every first 8192-bit window have B == all-ones
//    for all 8 K rows (const-one wires of linear constraints)
//  * first-window b_med 2 has B == all-ones for K = 0..1, and second-window
//    b_med 13 has B == all-ones for K = 4..7
//  * the final processed b_med block of each witness block has B == 0 for
//    K = 1..7 (structural zero padding rows)
// These are detected at runtime with at most 8 u64 loads + compares per
// 64-byte block (vs 512 table loads), so the fast paths are witness-safe: any
// witness that does not match takes the generic path.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn xor_apply_byte_into_4_regs<const BH: usize, const ODD: bool>(
    table_base: *const u8,
    half_swapped_table_base: *const u8,
    a_byte: usize,
    da0: &mut core::arch::aarch64::uint8x16_t,
    da1: &mut core::arch::aarch64::uint8x16_t,
    da2: &mut core::arch::aarch64::uint8x16_t,
    da3: &mut core::arch::aarch64::uint8x16_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        let t = if ODD {
            half_swapped_table_base
        } else {
            table_base
        };
        let ra = t.add(a_byte * 64);
        *da0 = veorq_u8(*da0, vld1q_u8(ra.add((0 ^ BH) * 16)));
        *da1 = veorq_u8(*da1, vld1q_u8(ra.add((1 ^ BH) * 16)));
        *da2 = veorq_u8(*da2, vld1q_u8(ra.add((2 ^ BH) * 16)));
        *da3 = veorq_u8(*da3, vld1q_u8(ra.add((3 ^ BH) * 16)));
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn xor_apply_byte_pair_into_4_regs<
    const BH0: usize,
    const ODD0: bool,
    const BH1: usize,
    const ODD1: bool,
>(
    table_base: *const u8,
    half_swapped_table_base: *const u8,
    a_byte0: usize,
    a_byte1: usize,
    da0: &mut core::arch::aarch64::uint8x16_t,
    da1: &mut core::arch::aarch64::uint8x16_t,
    da2: &mut core::arch::aarch64::uint8x16_t,
    da3: &mut core::arch::aarch64::uint8x16_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        let t0 = if ODD0 {
            half_swapped_table_base
        } else {
            table_base
        };
        let t1 = if ODD1 {
            half_swapped_table_base
        } else {
            table_base
        };
        let ra0 = t0.add(a_byte0 * 64);
        let ra1 = t1.add(a_byte1 * 64);
        *da0 = xor3_u8(
            *da0,
            vld1q_u8(ra0.add((0 ^ BH0) * 16)),
            vld1q_u8(ra1.add((0 ^ BH1) * 16)),
        );
        *da1 = xor3_u8(
            *da1,
            vld1q_u8(ra0.add((1 ^ BH0) * 16)),
            vld1q_u8(ra1.add((1 ^ BH1) * 16)),
        );
        *da2 = xor3_u8(
            *da2,
            vld1q_u8(ra0.add((2 ^ BH0) * 16)),
            vld1q_u8(ra1.add((2 ^ BH1) * 16)),
        );
        *da3 = xor3_u8(
            *da3,
            vld1q_u8(ra0.add((3 ^ BH0) * 16)),
            vld1q_u8(ra1.add((3 ^ BH1) * 16)),
        );
    }
}

/// Apply the collapsed inverse-NTT table to one packed eight-byte row and
/// keep the 64 output bytes in four vector registers.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn apply_word_into_4_regs(
    table_base: *const u8,
    half_swapped_table_base: *const u8,
    word: u64,
) -> (
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        let r0 = table_base.add(byte_from_word::<0>(word) * 64);
        let mut d0 = vld1q_u8(r0);
        let mut d1 = vld1q_u8(r0.add(16));
        let mut d2 = vld1q_u8(r0.add(32));
        let mut d3 = vld1q_u8(r0.add(48));
        xor_apply_byte_pair_into_4_regs::<0, true, 1, false>(
            table_base,
            half_swapped_table_base,
            byte_from_word::<1>(word),
            byte_from_word::<2>(word),
            &mut d0,
            &mut d1,
            &mut d2,
            &mut d3,
        );
        xor_apply_byte_pair_into_4_regs::<1, true, 2, false>(
            table_base,
            half_swapped_table_base,
            byte_from_word::<3>(word),
            byte_from_word::<4>(word),
            &mut d0,
            &mut d1,
            &mut d2,
            &mut d3,
        );
        xor_apply_byte_pair_into_4_regs::<2, true, 3, false>(
            table_base,
            half_swapped_table_base,
            byte_from_word::<5>(word),
            byte_from_word::<6>(word),
            &mut d0,
            &mut d1,
            &mut d2,
            &mut d3,
        );
        xor_apply_byte_into_4_regs::<3, true>(
            table_base,
            half_swapped_table_base,
            byte_from_word::<7>(word),
            &mut d0,
            &mut d1,
            &mut d2,
            &mut d3,
        );
        (d0, d1, d2, d3)
    }
}

/// Apply a word whose bytes 5..7 are known to be zero. This is the ranked
/// `(window 0, b_med 2, K0)` A shape; the caller checks the zero mask before
/// selecting the monomorphized path.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn apply_word_low5_into_4_regs(
    table_base: *const u8,
    half_swapped_table_base: *const u8,
    word: u64,
) -> (
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        let r0 = table_base.add(byte_from_word::<0>(word) * 64);
        let mut d0 = vld1q_u8(r0);
        let mut d1 = vld1q_u8(r0.add(16));
        let mut d2 = vld1q_u8(r0.add(32));
        let mut d3 = vld1q_u8(r0.add(48));
        xor_apply_byte_pair_into_4_regs::<0, true, 1, false>(
            table_base,
            half_swapped_table_base,
            byte_from_word::<1>(word),
            byte_from_word::<2>(word),
            &mut d0,
            &mut d1,
            &mut d2,
            &mut d3,
        );
        xor_apply_byte_pair_into_4_regs::<1, true, 2, false>(
            table_base,
            half_swapped_table_base,
            byte_from_word::<3>(word),
            byte_from_word::<4>(word),
            &mut d0,
            &mut d1,
            &mut d2,
            &mut d3,
        );
        (d0, d1, d2, d3)
    }
}

/// Apply a word whose top byte is known to be zero. The final live BLAKE3
/// medium row has this A shape; the checked static-B path guards the byte
/// before selecting this helper, saving the last four table loads and XORs.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn apply_word_low7_into_4_regs(
    table_base: *const u8,
    half_swapped_table_base: *const u8,
    word: u64,
) -> (
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        let r0 = table_base.add(byte_from_word::<0>(word) * 64);
        let mut d0 = vld1q_u8(r0);
        let mut d1 = vld1q_u8(r0.add(16));
        let mut d2 = vld1q_u8(r0.add(32));
        let mut d3 = vld1q_u8(r0.add(48));
        xor_apply_byte_pair_into_4_regs::<0, true, 1, false>(
            table_base,
            half_swapped_table_base,
            byte_from_word::<1>(word),
            byte_from_word::<2>(word),
            &mut d0,
            &mut d1,
            &mut d2,
            &mut d3,
        );
        xor_apply_byte_pair_into_4_regs::<1, true, 2, false>(
            table_base,
            half_swapped_table_base,
            byte_from_word::<3>(word),
            byte_from_word::<4>(word),
            &mut d0,
            &mut d1,
            &mut d2,
            &mut d3,
        );
        xor_apply_byte_pair_into_4_regs::<2, true, 3, false>(
            table_base,
            half_swapped_table_base,
            byte_from_word::<5>(word),
            byte_from_word::<6>(word),
            &mut d0,
            &mut d1,
            &mut d2,
            &mut d3,
        );
        (d0, d1, d2, d3)
    }
}

/// Const-ones-B fast path: `out = red(sum_K x^K ntt_a[K])`. The transform
/// of the packed all-ones row is exactly the GF(2^8) multiplicative identity
/// in all 64 lanes, so the baseline's final vector multiply is redundant.
/// Caller has verified all eight B words are all-ones.
/// Store one finished 64-byte AB row: plain cached `vst1q` quads, or
/// non-temporal `stnp` pairs straight from the accumulator registers. The NT
/// flavor lets `precompute_ab_one_chunk` drop its stack bounce — the row
/// previously took four `vst1q` into a stack temporary plus an `ldp`/`stnp`
/// copy (six memory ops and a store-to-load forward per row); storing from
/// registers keeps only the two `stnp`. Bytes written are identical.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn store_row_64(
    p: *mut u8,
    nt: bool,
    v0: core::arch::aarch64::uint8x16_t,
    v1: core::arch::aarch64::uint8x16_t,
    v2: core::arch::aarch64::uint8x16_t,
    v3: core::arch::aarch64::uint8x16_t,
) {
    unsafe {
        if nt {
            core::arch::asm!(
                "stnp {v0:q}, {v1:q}, [{p}]",
                "stnp {v2:q}, {v3:q}, [{p}, #32]",
                p = in(reg) p,
                v0 = in(vreg) v0,
                v1 = in(vreg) v1,
                v2 = in(vreg) v2,
                v3 = in(vreg) v3,
                options(nostack, preserves_flags)
            );
        } else {
            use core::arch::aarch64::vst1q_u8;
            vst1q_u8(p, v0);
            vst1q_u8(p.add(16), v1);
            vst1q_u8(p.add(32), v2);
            vst1q_u8(p.add(48), v3);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn shift_reduce_inner_a_only_const_b(
    a_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    byte_base_b: usize,
    out: &mut [u8; 64],
    nt_store: bool,
) {
    use crate::field::gf2_8::neon::gf8_reduce_vec16;
    use core::arch::aarch64::*;

    let table_base = inv_table.data_ptr();
    let half_swapped_table_base = inv_table.half_swapped_data_ptr();
    unsafe {
        let mut acc0_lo = vdupq_n_u16(0);
        let mut acc0_hi = vdupq_n_u16(0);
        let mut acc1_lo = vdupq_n_u16(0);
        let mut acc1_hi = vdupq_n_u16(0);
        let mut acc2_lo = vdupq_n_u16(0);
        let mut acc2_hi = vdupq_n_u16(0);
        let mut acc3_lo = vdupq_n_u16(0);
        let mut acc3_hi = vdupq_n_u16(0);

        macro_rules! do_k {
            ($k:literal) => {{
                let off = byte_base_b + $k * N_CHUNKS;
                let word = u64::from_le(core::ptr::read_unaligned(
                    a_packed.as_ptr().add(off).cast::<u64>(),
                ));
                let (d0, d1, d2, d3) =
                    apply_word_into_4_regs(table_base, half_swapped_table_base, word);
                acc0_lo = veorq_u16(acc0_lo, vshll_n_u8::<$k>(vget_low_u8(d0)));
                acc0_hi = veorq_u16(acc0_hi, vshll_n_u8::<$k>(vget_high_u8(d0)));
                acc1_lo = veorq_u16(acc1_lo, vshll_n_u8::<$k>(vget_low_u8(d1)));
                acc1_hi = veorq_u16(acc1_hi, vshll_n_u8::<$k>(vget_high_u8(d1)));
                acc2_lo = veorq_u16(acc2_lo, vshll_n_u8::<$k>(vget_low_u8(d2)));
                acc2_hi = veorq_u16(acc2_hi, vshll_n_u8::<$k>(vget_high_u8(d2)));
                acc3_lo = veorq_u16(acc3_lo, vshll_n_u8::<$k>(vget_low_u8(d3)));
                acc3_hi = veorq_u16(acc3_hi, vshll_n_u8::<$k>(vget_high_u8(d3)));
            }};
        }
        do_k!(0);
        do_k!(1);
        do_k!(2);
        do_k!(3);
        do_k!(4);
        do_k!(5);
        do_k!(6);
        do_k!(7);

        let y0 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc0_lo), vreinterpretq_u8_u16(acc0_hi));
        let y1 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc1_lo), vreinterpretq_u8_u16(acc1_hi));
        let y2 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc2_lo), vreinterpretq_u8_u16(acc2_hi));
        let y3 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc3_lo), vreinterpretq_u8_u16(acc3_hi));

        store_row_64(out.as_mut_ptr(), nt_store, y0, y1, y2, y3);
    }
}

/// Single-live-K fast path: b rows K=1..7 are zero, so only K=0 contributes
/// and `out = ntt_a[0] . ntt_b[0]` (x^0, already reduced).
#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn shift_reduce_inner_single_k0(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    byte_base_b: usize,
    out: &mut [u8; 64],
    nt_store: bool,
) {
    use crate::field::gf2_8::neon::gf8_mul_vec16;

    let table_base = inv_table.data_ptr();
    let half_swapped_table_base = inv_table.half_swapped_data_ptr();
    unsafe {
        let a_word = u64::from_le(core::ptr::read_unaligned(
            a_packed.as_ptr().add(byte_base_b).cast::<u64>(),
        ));
        let b_word = u64::from_le(core::ptr::read_unaligned(
            b_packed.as_ptr().add(byte_base_b).cast::<u64>(),
        ));
        let (a0, a1, a2, a3) = apply_word_into_4_regs(table_base, half_swapped_table_base, a_word);
        let (b0, b1, b2, b3) = apply_word_into_4_regs(table_base, half_swapped_table_base, b_word);
        let y0 = gf8_mul_vec16(a0, b0);
        let y1 = gf8_mul_vec16(a1, b1);
        let y2 = gf8_mul_vec16(a2, b2);
        let y3 = gf8_mul_vec16(a3, b3);
        store_row_64(out.as_mut_ptr(), nt_store, y0, y1, y2, y3);
    }
}

/// Single-live-K fast path with a statically-known B word. At the ranked
/// BLAKE3 shape the only single-K row is (w1, b_med 14) = blk 30, whose B
/// word is the compile-time constant `0x0001ffffffffffff`
/// (`BSTATIC_MASKS[30][0]` pins the full word). Its inverse transform is
/// already materialized in `bstatic_partials()[240]`, so the row loads four
/// registers instead of re-deriving the transform through eight table-row
/// gathers. Caller has verified the exact B word and selects `A_TOP_ZERO`
/// only after checking A's top byte. Output is identical to
/// [`shift_reduce_inner_single_k0`].
#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn shift_reduce_inner_single_k0_static_b<const A_TOP_ZERO: bool>(
    a_word: u64,
    inv_table: &InvNttTableByteSingleGf8,
    b_partial: &[u8; 64],
    out: &mut [u8; 64],
    nt_store: bool,
) {
    use crate::field::gf2_8::neon::gf8_mul_vec16;
    use core::arch::aarch64::vld1q_u8;

    let table_base = inv_table.data_ptr();
    let half_swapped_table_base = inv_table.half_swapped_data_ptr();
    unsafe {
        let (a0, a1, a2, a3) = if A_TOP_ZERO {
            apply_word_low7_into_4_regs(table_base, half_swapped_table_base, a_word)
        } else {
            apply_word_into_4_regs(table_base, half_swapped_table_base, a_word)
        };
        let p = b_partial.as_ptr();
        let y0 = gf8_mul_vec16(a0, vld1q_u8(p));
        let y1 = gf8_mul_vec16(a1, vld1q_u8(p.add(16)));
        let y2 = gf8_mul_vec16(a2, vld1q_u8(p.add(32)));
        let y3 = gf8_mul_vec16(a3, vld1q_u8(p.add(48)));
        store_row_64(out.as_mut_ptr(), nt_store, y0, y1, y2, y3);
    }
}

/// Mixed static-one fast path. K rows selected by `ONE_MASK` have B equal to
/// the multiplicative identity in every lane, so they contribute the inverse
/// transform of A directly; all remaining K rows use the generic fused AB
/// transform and multiply.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn shift_reduce_inner_mixed_const_b<const ONE_MASK: u8>(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    byte_base_b: usize,
    out: &mut [u8; 64],
    nt_store: bool,
) {
    use crate::field::gf2_8::neon::gf8_reduce_vec16;
    use core::arch::aarch64::*;

    let table_base = inv_table.data_ptr();
    let half_swapped_table_base = inv_table.half_swapped_data_ptr();
    unsafe {
        let mut acc0_lo = vdupq_n_u16(0);
        let mut acc0_hi = vdupq_n_u16(0);
        let mut acc1_lo = vdupq_n_u16(0);
        let mut acc1_hi = vdupq_n_u16(0);
        let mut acc2_lo = vdupq_n_u16(0);
        let mut acc2_hi = vdupq_n_u16(0);
        let mut acc3_lo = vdupq_n_u16(0);
        let mut acc3_hi = vdupq_n_u16(0);

        macro_rules! do_k {
            ($k:literal) => {{
                let off = byte_base_b + $k * N_CHUNKS;
                if ONE_MASK & (1 << $k) != 0 {
                    let word = u64::from_le(core::ptr::read_unaligned(
                        a_packed.as_ptr().add(off).cast::<u64>(),
                    ));
                    let (d0, d1, d2, d3) =
                        apply_word_into_4_regs(table_base, half_swapped_table_base, word);
                    acc0_lo = veorq_u16(acc0_lo, vshll_n_u8::<$k>(vget_low_u8(d0)));
                    acc0_hi = veorq_u16(acc0_hi, vshll_n_u8::<$k>(vget_high_u8(d0)));
                    acc1_lo = veorq_u16(acc1_lo, vshll_n_u8::<$k>(vget_low_u8(d1)));
                    acc1_hi = veorq_u16(acc1_hi, vshll_n_u8::<$k>(vget_high_u8(d1)));
                    acc2_lo = veorq_u16(acc2_lo, vshll_n_u8::<$k>(vget_low_u8(d2)));
                    acc2_hi = veorq_u16(acc2_hi, vshll_n_u8::<$k>(vget_high_u8(d2)));
                    acc3_lo = veorq_u16(acc3_lo, vshll_n_u8::<$k>(vget_low_u8(d3)));
                    acc3_hi = veorq_u16(acc3_hi, vshll_n_u8::<$k>(vget_high_u8(d3)));
                } else {
                    fused_apply_one_k::<$k>(
                        table_base,
                        half_swapped_table_base,
                        a_packed.as_ptr().add(off),
                        b_packed.as_ptr().add(off),
                        &mut acc0_lo,
                        &mut acc0_hi,
                        &mut acc1_lo,
                        &mut acc1_hi,
                        &mut acc2_lo,
                        &mut acc2_hi,
                        &mut acc3_lo,
                        &mut acc3_hi,
                    );
                }
            }};
        }
        do_k!(0);
        do_k!(1);
        do_k!(2);
        do_k!(3);
        do_k!(4);
        do_k!(5);
        do_k!(6);
        do_k!(7);

        let r0 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc0_lo), vreinterpretq_u8_u16(acc0_hi));
        let r1 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc1_lo), vreinterpretq_u8_u16(acc1_hi));
        let r2 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc2_lo), vreinterpretq_u8_u16(acc2_hi));
        let r3 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc3_lo), vreinterpretq_u8_u16(acc3_hi));
        store_row_64(out.as_mut_ptr(), nt_store, r0, r1, r2, r3);
    }
}

// ---------------------------------------------------------------------------
// Static-b fast paths (generated dispatch in aarch64_bstatic_gen.rs).
//
// The round-1 b operand has byte positions the BLAKE3 circuit fixes
// independently of the inputs (census: 31.6% of b byte positions, clustered
// by (window, b_med, K)). Since the inv-NTT byte apply is linear and row 0 of
// the table is zero, each position's static contribution is precomputed once
// (`bstatic_partials`) and per block only the varying bytes are gathered.
// Every K-row is guarded by `(b_word & MASK) == EXPECTED` with a generic
// fallback, so output is bit-identical for arbitrary witnesses.
// ---------------------------------------------------------------------------

/// XOR one selected b-byte's table row into the four b-side registers. `J`
/// selects the byte; placement constants derive from `J` exactly as in the
/// generic unrolled sequence: `BH = J / 2`, odd bytes use the
/// half-swapped table.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn xor_vary_single<const J: u32>(
    table_base: *const u8,
    half_swapped_table_base: *const u8,
    b_word: u64,
    db0: &mut core::arch::aarch64::uint8x16_t,
    db1: &mut core::arch::aarch64::uint8x16_t,
    db2: &mut core::arch::aarch64::uint8x16_t,
    db3: &mut core::arch::aarch64::uint8x16_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        let bh = (J as usize) / 2;
        let t = if J % 2 == 1 {
            half_swapped_table_base
        } else {
            table_base
        };
        let r = t.add(byte_from_word::<J>(b_word) * 64);
        *db0 = veorq_u8(*db0, vld1q_u8(r.add((bh) * 16)));
        *db1 = veorq_u8(*db1, vld1q_u8(r.add((1 ^ bh) * 16)));
        *db2 = veorq_u8(*db2, vld1q_u8(r.add((2 ^ bh) * 16)));
        *db3 = veorq_u8(*db3, vld1q_u8(r.add((3 ^ bh) * 16)));
    }
}

/// Row-body variant bit: the static-B partial path swaps the halves of odd
/// bytes in registers instead of reading the half-swapped image.
#[cfg(target_arch = "aarch64")]
pub(crate) const ABV_NOSW: u32 = 64;

/// [`xor_vary_pair`] (or single with `J1 == 8`) out of the plain image only.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn xor_vary_ext<const J0: u32, const J1: u32>(
    table_base: *const u8,
    b_word: u64,
    db: [&mut core::arch::aarch64::uint8x16_t; 4],
) {
    use core::arch::aarch64::*;
    unsafe {
        let r0 = table_base.add(byte_from_word::<J0>(b_word) * 64);
        let r1 = if J1 < 8 { table_base.add(((b_word >> ((J1 & 7) * 8)) & 0xff) as usize * 64) } else { r0 };
        let fix = |v: uint8x16_t, odd: bool| if odd { vextq_u8::<8>(v, v) } else { v };
        for (q, d) in db.into_iter().enumerate() {
            let v0 = fix(vld1q_u8(r0.add((q ^ (J0 as usize / 2)) * 16)), J0 % 2 == 1);
            if J1 < 8 {
                let v1 = fix(vld1q_u8(r1.add((q ^ (J1 as usize / 2)) * 16)), J1 % 2 == 1);
                *d = xor3_u8(*d, v0, v1);
            } else {
                *d = veorq_u8(*d, v0);
            }
        }
    }
}

/// EOR3-paired form of [`xor_vary_single`] for two varying bytes.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn xor_vary_pair<const J0: u32, const J1: u32>(
    table_base: *const u8,
    half_swapped_table_base: *const u8,
    b_word: u64,
    db0: &mut core::arch::aarch64::uint8x16_t,
    db1: &mut core::arch::aarch64::uint8x16_t,
    db2: &mut core::arch::aarch64::uint8x16_t,
    db3: &mut core::arch::aarch64::uint8x16_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        let bh0 = (J0 as usize) / 2;
        let t0 = if J0 % 2 == 1 {
            half_swapped_table_base
        } else {
            table_base
        };
        let bh1 = (J1 as usize) / 2;
        let t1 = if J1 % 2 == 1 {
            half_swapped_table_base
        } else {
            table_base
        };
        let r0 = t0.add(byte_from_word::<J0>(b_word) * 64);
        let r1 = t1.add(byte_from_word::<J1>(b_word) * 64);
        *db0 = xor3_u8(*db0, vld1q_u8(r0.add(bh0 * 16)), vld1q_u8(r1.add(bh1 * 16)));
        *db1 = xor3_u8(
            *db1,
            vld1q_u8(r0.add((1 ^ bh0) * 16)),
            vld1q_u8(r1.add((1 ^ bh1) * 16)),
        );
        *db2 = xor3_u8(
            *db2,
            vld1q_u8(r0.add((2 ^ bh0) * 16)),
            vld1q_u8(r1.add((2 ^ bh1) * 16)),
        );
        *db3 = xor3_u8(
            *db3,
            vld1q_u8(r0.add((3 ^ bh0) * 16)),
            vld1q_u8(r1.add((3 ^ bh1) * 16)),
        );
    }
}

/// One K-row where the b-side NTT extension is already in registers: full
/// a-side apply, F_8 multiply, widen-shift-XOR into the accumulators. Same
/// epilogue as [`fused_apply_one_k`].
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn fused_apply_one_k_with_b<const K: i32>(
    table_base: *const u8,
    half_swapped_table_base: *const u8,
    a_row: *const u8,
    db0: core::arch::aarch64::uint8x16_t,
    db1: core::arch::aarch64::uint8x16_t,
    db2: core::arch::aarch64::uint8x16_t,
    db3: core::arch::aarch64::uint8x16_t,
    acc0_lo: &mut core::arch::aarch64::uint16x8_t,
    acc0_hi: &mut core::arch::aarch64::uint16x8_t,
    acc1_lo: &mut core::arch::aarch64::uint16x8_t,
    acc1_hi: &mut core::arch::aarch64::uint16x8_t,
    acc2_lo: &mut core::arch::aarch64::uint16x8_t,
    acc2_hi: &mut core::arch::aarch64::uint16x8_t,
    acc3_lo: &mut core::arch::aarch64::uint16x8_t,
    acc3_hi: &mut core::arch::aarch64::uint16x8_t,
) {
    use crate::field::gf2_8::neon::gf8_mul_vec16;
    use core::arch::aarch64::*;
    unsafe {
        let a_word = u64::from_le(core::ptr::read_unaligned(a_row.cast::<u64>()));
        let ra0 = table_base.add(byte_from_word::<0>(a_word) * 64);
        let mut da0 = vld1q_u8(ra0);
        let mut da1 = vld1q_u8(ra0.add(16));
        let mut da2 = vld1q_u8(ra0.add(32));
        let mut da3 = vld1q_u8(ra0.add(48));
        xor_apply_byte_pair_into_4_regs::<0, true, 1, false>(
            table_base,
            half_swapped_table_base,
            byte_from_word::<1>(a_word),
            byte_from_word::<2>(a_word),
            &mut da0,
            &mut da1,
            &mut da2,
            &mut da3,
        );
        xor_apply_byte_pair_into_4_regs::<1, true, 2, false>(
            table_base,
            half_swapped_table_base,
            byte_from_word::<3>(a_word),
            byte_from_word::<4>(a_word),
            &mut da0,
            &mut da1,
            &mut da2,
            &mut da3,
        );
        xor_apply_byte_pair_into_4_regs::<2, true, 3, false>(
            table_base,
            half_swapped_table_base,
            byte_from_word::<5>(a_word),
            byte_from_word::<6>(a_word),
            &mut da0,
            &mut da1,
            &mut da2,
            &mut da3,
        );
        xor_apply_byte_into_4_regs::<3, true>(
            table_base,
            half_swapped_table_base,
            byte_from_word::<7>(a_word),
            &mut da0,
            &mut da1,
            &mut da2,
            &mut da3,
        );
        let y0 = gf8_mul_vec16(da0, db0);
        let y1 = gf8_mul_vec16(da1, db1);
        let y2 = gf8_mul_vec16(da2, db2);
        let y3 = gf8_mul_vec16(da3, db3);
        *acc0_lo = veorq_u16(*acc0_lo, vshll_n_u8::<K>(vget_low_u8(y0)));
        *acc0_hi = veorq_u16(*acc0_hi, vshll_n_u8::<K>(vget_high_u8(y0)));
        *acc1_lo = veorq_u16(*acc1_lo, vshll_n_u8::<K>(vget_low_u8(y1)));
        *acc1_hi = veorq_u16(*acc1_hi, vshll_n_u8::<K>(vget_high_u8(y1)));
        *acc2_lo = veorq_u16(*acc2_lo, vshll_n_u8::<K>(vget_low_u8(y2)));
        *acc2_hi = veorq_u16(*acc2_hi, vshll_n_u8::<K>(vget_high_u8(y2)));
        *acc3_lo = veorq_u16(*acc3_lo, vshll_n_u8::<K>(vget_low_u8(y3)));
        *acc3_hi = veorq_u16(*acc3_hi, vshll_n_u8::<K>(vget_high_u8(y3)));
    }
}

// ---------------------------------------------------------------------------
// Low-instruction K-row epilogue for the static-B kernel.
//
// The incumbent epilogue above costs 13 vector ops per 16-lane group per K
// row: two PMULLs, a full `gf8_reduce_vec16` of the product, and a
// widen-shift-XOR into the 16-bit accumulator — after which the accumulator
// is reduced a *second* time when the block finishes. `reduce` is a ring
// homomorphism, so the first reduction is redundant:
//
//     out = reduce( Σ_K reduce(A_K·B_K)·x^K ) = reduce( Σ_K (A_K·B_K)·x^K )
//
// Raw 15-bit PMULL outputs can be XOR-accumulated directly as long as their
// `x^K` weight keeps them inside 16 bits, i.e. as long as the residual shift
// is at most 1. So `x^K` is factored as
//
//     x^K = x^(4·(K>>2)) · x^(2·((K>>1)&1)) · x^(K&1)
//
// with the `x^4` from the pre-scaled inverse-NTT table image (free — the
// byte-collapse transform is GF(2)-linear), the `x^2` from a two-lookup
// constant multiply on the four `a`-side registers, and only `x^(K&1)` left
// as an in-lane shift. Per 16-lane group per K the cost becomes
// `2 PMULL + 2 EOR` (+2 shifts for odd K, +5 for the `x^2` multiply), i.e.
// 4/6/9/11 ops for K ≡ 0/1/2/3 (mod 4) against a flat 13 — 40 fewer vector
// ops per group over the eight rows, 160 per 64-byte output block.
// ---------------------------------------------------------------------------

/// Multiply 16 GF(2^8) values by the constant `x^2` via nibble tables.
/// `LO[v] = v·x^2` and `HI[v] = (v<<4)·x^2`; verified against `F8` scalar
/// arithmetic in `mul_x2_nibble_tables_match_field`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn gf8_mul_x2_vec16(a: core::arch::aarch64::uint8x16_t) -> core::arch::aarch64::uint8x16_t {
    use core::arch::aarch64::*;
    unsafe {
        let lo = vld1q_u8(MUL_X2_LO.as_ptr());
        let hi = vld1q_u8(MUL_X2_HI.as_ptr());
        veorq_u8(
            vqtbl1q_u8(lo, vandq_u8(a, vdupq_n_u8(0x0f))),
            vqtbl1q_u8(hi, vshrq_n_u8::<4>(a)),
        )
    }
}

#[cfg(target_arch = "aarch64")]
pub(crate) const MUL_X2_LO: [u8; 16] = [
    0x00, 0x04, 0x08, 0x0c, 0x10, 0x14, 0x18, 0x1c, 0x20, 0x24, 0x28, 0x2c, 0x30, 0x34, 0x38, 0x3c,
];
#[cfg(target_arch = "aarch64")]
pub(crate) const MUL_X2_HI: [u8; 16] = [
    0x00, 0x40, 0x80, 0xc0, 0x1b, 0x5b, 0x9b, 0xdb, 0x36, 0x76, 0xb6, 0xf6, 0x2d, 0x6d, 0xad, 0xed,
];

/// XOR the raw carry-less products `da × db` into the 16-bit accumulators,
/// weighted by `x^R` with `R ∈ {0, 1}`. When `MUL_X2` the `a` side is first
/// multiplied by `x^2`; the remaining `x^4` weight, when needed, is already
/// baked into `da` by the caller's choice of table image.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn accumulate_raw_one_k<const R: i32, const MUL_X2: bool>(
    da0: core::arch::aarch64::uint8x16_t,
    da1: core::arch::aarch64::uint8x16_t,
    da2: core::arch::aarch64::uint8x16_t,
    da3: core::arch::aarch64::uint8x16_t,
    db0: core::arch::aarch64::uint8x16_t,
    db1: core::arch::aarch64::uint8x16_t,
    db2: core::arch::aarch64::uint8x16_t,
    db3: core::arch::aarch64::uint8x16_t,
    acc0_lo: &mut core::arch::aarch64::uint16x8_t,
    acc0_hi: &mut core::arch::aarch64::uint16x8_t,
    acc1_lo: &mut core::arch::aarch64::uint16x8_t,
    acc1_hi: &mut core::arch::aarch64::uint16x8_t,
    acc2_lo: &mut core::arch::aarch64::uint16x8_t,
    acc2_hi: &mut core::arch::aarch64::uint16x8_t,
    acc3_lo: &mut core::arch::aarch64::uint16x8_t,
    acc3_hi: &mut core::arch::aarch64::uint16x8_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        let (da0, da1, da2, da3) = if MUL_X2 {
            (
                gf8_mul_x2_vec16(da0),
                gf8_mul_x2_vec16(da1),
                gf8_mul_x2_vec16(da2),
                gf8_mul_x2_vec16(da3),
            )
        } else {
            (da0, da1, da2, da3)
        };
        if R == 0 {
            // One fused PMULL+EOR µop per product (see fused_pmull8_acc).
            *acc0_lo = fused_pmull8_acc::<false>(*acc0_lo, da0, db0);
            *acc0_hi = fused_pmull8_acc::<true>(*acc0_hi, da0, db0);
            *acc1_lo = fused_pmull8_acc::<false>(*acc1_lo, da1, db1);
            *acc1_hi = fused_pmull8_acc::<true>(*acc1_hi, da1, db1);
            *acc2_lo = fused_pmull8_acc::<false>(*acc2_lo, da2, db2);
            *acc2_hi = fused_pmull8_acc::<true>(*acc2_hi, da2, db2);
            *acc3_lo = fused_pmull8_acc::<false>(*acc3_lo, da3, db3);
            *acc3_hi = fused_pmull8_acc::<true>(*acc3_hi, da3, db3);
            return;
        }
        *acc0_lo = veorq_u16(*acc0_lo, vshlq_n_u16::<R>(pmull_lo_u16(da0, db0)));
        *acc0_hi = veorq_u16(*acc0_hi, vshlq_n_u16::<R>(pmull_hi_u16(da0, db0)));
        *acc1_lo = veorq_u16(*acc1_lo, vshlq_n_u16::<R>(pmull_lo_u16(da1, db1)));
        *acc1_hi = veorq_u16(*acc1_hi, vshlq_n_u16::<R>(pmull_hi_u16(da1, db1)));
        *acc2_lo = veorq_u16(*acc2_lo, vshlq_n_u16::<R>(pmull_lo_u16(da2, db2)));
        *acc2_hi = veorq_u16(*acc2_hi, vshlq_n_u16::<R>(pmull_hi_u16(da2, db2)));
        *acc3_lo = veorq_u16(*acc3_lo, vshlq_n_u16::<R>(pmull_lo_u16(da3, db3)));
        *acc3_hi = veorq_u16(*acc3_hi, vshlq_n_u16::<R>(pmull_hi_u16(da3, db3)));
    }
}

/// Low-instruction twin of [`fused_apply_one_k`]. `a_table`/`a_table_sw`
/// select the plain or the `x^4`-scaled inverse-NTT image.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn fused_apply_one_k_fast<const R: i32, const MUL_X2: bool>(
    a_table: *const u8,
    a_table_sw: *const u8,
    b_table: *const u8,
    b_table_sw: *const u8,
    a_row: *const u8,
    b_row: *const u8,
    acc0_lo: &mut core::arch::aarch64::uint16x8_t,
    acc0_hi: &mut core::arch::aarch64::uint16x8_t,
    acc1_lo: &mut core::arch::aarch64::uint16x8_t,
    acc1_hi: &mut core::arch::aarch64::uint16x8_t,
    acc2_lo: &mut core::arch::aarch64::uint16x8_t,
    acc2_hi: &mut core::arch::aarch64::uint16x8_t,
    acc3_lo: &mut core::arch::aarch64::uint16x8_t,
    acc3_hi: &mut core::arch::aarch64::uint16x8_t,
) {
    unsafe {
        let a_word = u64::from_le(core::ptr::read_unaligned(a_row.cast::<u64>()));
        let b_word = u64::from_le(core::ptr::read_unaligned(b_row.cast::<u64>()));
        let (da0, da1, da2, da3) = apply_word_into_4_regs(a_table, a_table_sw, a_word);
        let (db0, db1, db2, db3) = apply_word_into_4_regs(b_table, b_table_sw, b_word);
        accumulate_raw_one_k::<R, MUL_X2>(
            da0, da1, da2, da3, db0, db1, db2, db3, acc0_lo, acc0_hi, acc1_lo, acc1_hi, acc2_lo,
            acc2_hi, acc3_lo, acc3_hi,
        );
    }
}

/// Low-instruction twin of [`fused_apply_one_k_with_b`].
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn fused_apply_one_k_with_b_fast<const R: i32, const MUL_X2: bool>(
    a_table: *const u8,
    a_table_sw: *const u8,
    a_row: *const u8,
    db0: core::arch::aarch64::uint8x16_t,
    db1: core::arch::aarch64::uint8x16_t,
    db2: core::arch::aarch64::uint8x16_t,
    db3: core::arch::aarch64::uint8x16_t,
    acc0_lo: &mut core::arch::aarch64::uint16x8_t,
    acc0_hi: &mut core::arch::aarch64::uint16x8_t,
    acc1_lo: &mut core::arch::aarch64::uint16x8_t,
    acc1_hi: &mut core::arch::aarch64::uint16x8_t,
    acc2_lo: &mut core::arch::aarch64::uint16x8_t,
    acc2_hi: &mut core::arch::aarch64::uint16x8_t,
    acc3_lo: &mut core::arch::aarch64::uint16x8_t,
    acc3_hi: &mut core::arch::aarch64::uint16x8_t,
) {
    unsafe {
        let a_word = u64::from_le(core::ptr::read_unaligned(a_row.cast::<u64>()));
        let (da0, da1, da2, da3) = apply_word_into_4_regs(a_table, a_table_sw, a_word);
        accumulate_raw_one_k::<R, MUL_X2>(
            da0, da1, da2, da3, db0, db1, db2, db3, acc0_lo, acc0_hi, acc1_lo, acc1_hi, acc2_lo,
            acc2_hi, acc3_lo, acc3_hi,
        );
    }
}

// ---------------------------------------------------------------------------
// Karatsuba form of the collapsed inverse-NTT table apply.
//
// Byte `b` of a packed row contributes its 64-byte table row with the eight
// 8-byte blocks permuted by XOR `b` (`apply_scalar`: `out[i] += row[i ^ 8b]`),
// so applying the table to an 8-byte word is an XOR-convolution over the byte
// index. Take byte pairs `W_p = (w_2p, w_2p+1)`, `p` in `(Z/2)^2`, and the
// four 16-byte lanes `C_q` of a table row; output lane `q` is
// `sum_p U(W_p; C_{q ^ p})`, where the unit product `U(x_e, x_o; C)` is
// `C[x_e] ^ halfswap(C[x_o])`. The plain apply evaluates all 16 unit products
// (32 sixteen-byte loads). Two Karatsuba levels over the two bits of `p` need
// nine: with `P(W; C)` linear in both arguments over GF(2),
//   `(Wa, Wb) * (Ca, Cb) = (Pa ^ Pb, Pa ^ Pb ^ P(Wa ^ Wb; Ca ^ Cb))`,
// applied once over the high bit and once over the low bit of `p`. The nine
// "constants" are the lanes `C0..C3` and their sums `C0^C1`, `C2^C3`,
// `C0^C2`, `C1^C3`, `C0^C1^C2^C3`, stored per byte value in
// [`ab_kara_tables`]; the summed bytes come from three integer XORs of the
// word with itself shifted. 18 loads per word, no half-swapped image (the
// swap of the odd bytes is one `ext` per output lane). Every output bit is
// the same XOR of the same table bits; the unit test compares it with
// [`apply_word_into_4_regs`] for every byte value in every position.
//
// Why: the round-one AB rows are bound by load throughput (507 loads per
// static-B row at about 2.7 per cycle on a core that starts 3 per cycle),
// while their vector arithmetic is free at the margin.
// ---------------------------------------------------------------------------

/// Row-body variant bit: apply the table in the Karatsuba form.
#[cfg(target_arch = "aarch64")]
pub(crate) const ABV_KARA: u32 = 1;
/// Row-body variant bit: the static-B Horner rows of a window run as one
/// function body (no per-row call, saved registers or constant reloads).
#[cfg(target_arch = "aarch64")]
pub(crate) const ABV_INLINE_WINDOW: u32 = 2;
/// Row-body variant bit: form Karatsuba byte offsets before unscaled loads.
#[cfg(target_arch = "aarch64")]
pub(crate) const ABV_KARA_UNSCALED: u32 = 4;

#[cfg(target_arch = "aarch64")]
pub(crate) fn ab_kara_unscaled_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var_os("FLOCK_NO_ZC_AB_KARA_UNSCALED").is_none_or(|v| v != *"1")
    });
    *ON
}
/// The form of the static-B Horner rows when [`ab_kara_enabled`].
#[cfg(target_arch = "aarch64")]
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) const AB_ROWS_VARIANT: u32 = ABV_KARA | ABV_INLINE_WINDOW;

/// Kill switch for the Karatsuba table apply in the round-one AB rows:
/// exactly `FLOCK_NO_ZC_AB_KARA=1` restores the plain eight-row apply.
/// The rows written are byte-identical either way.
#[cfg(target_arch = "aarch64")]
pub(crate) const ENV_NO_ZC_AB_KARA: &str = "FLOCK_NO_ZC_AB_KARA";

/// 0 = not resolved yet, 1 = Karatsuba apply, 2 = plain apply.
#[cfg(target_arch = "aarch64")]
static AB_KARA_STATE: core::sync::atomic::AtomicU8 = core::sync::atomic::AtomicU8::new(0);

/// Whether the round-one AB rows on this thread use the Karatsuba apply
/// (the switch is resolved from the environment on first use; afterwards
/// one relaxed load and one thread-local read).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) fn ab_kara_enabled() -> bool {
    let on = match AB_KARA_STATE.load(core::sync::atomic::Ordering::Relaxed) {
        1 => true,
        2 => false,
        _ => ab_kara_resolve(),
    };
    on && !AB_KARA_THREAD_PLAIN.with(|plain| plain.get())
}

/// Second switch: exactly `FLOCK_NO_ZC_AB_KARA_HELPER_PLAIN=1` lets the
/// efficiency-core helper threads use the Karatsuba apply as well.
#[cfg(target_arch = "aarch64")]
pub(crate) const ENV_NO_ZC_AB_KARA_HELPER_PLAIN: &str = "FLOCK_NO_ZC_AB_KARA_HELPER_PLAIN";

#[cfg(target_arch = "aarch64")]
thread_local! {
    /// Set on threads that keep the plain apply (see [`ab_kara_note_worker`]).
    static AB_KARA_THREAD_PLAIN: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}

/// Called by every worker of an AB precompute queue before its first job.
/// Threads of the efficiency-core helper pool (`flock-ecore-*`, see
/// `epool`) keep the plain apply: the Karatsuba form trades loads for
/// instructions, which pays on a core that is bound by its three load
/// units and loses on a narrower core with two (measured: about 12% slower
/// there). The choice is per thread and both forms write the same bytes.
#[cfg(target_arch = "aarch64")]
pub(crate) fn ab_kara_note_worker() {
    static HELPER_PLAIN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let helper = std::thread::current()
        .name()
        .is_some_and(|name| name.starts_with("flock-ecore-"));
    let plain = helper
        && *HELPER_PLAIN.get_or_init(|| {
            std::env::var_os(ENV_NO_ZC_AB_KARA_HELPER_PLAIN).as_deref()
                != Some(std::ffi::OsStr::new("1"))
        });
    AB_KARA_THREAD_PLAIN.with(|cell| cell.set(plain));
}

#[cfg(target_arch = "aarch64")]
#[cold]
#[inline(never)]
fn ab_kara_resolve() -> bool {
    let on = std::env::var_os(ENV_NO_ZC_AB_KARA).as_deref() != Some(std::ffi::OsStr::new("1"));
    AB_KARA_STATE.store(if on { 1 } else { 2 }, core::sync::atomic::Ordering::Relaxed);
    on
}

/// Test and probe control of [`ab_kara_enabled`].
#[cfg(all(test, target_arch = "aarch64"))]
pub(crate) fn ab_kara_force(on: bool) {
    AB_KARA_STATE.store(if on { 1 } else { 2 }, core::sync::atomic::Ordering::Relaxed);
}

/// Bytes of one image in [`ab_kara_tables`]: 256 entries of 128 bytes (lanes
/// `C0, C1, C2, C3, C0^C1, C2^C3, C0^C2, C1^C3`), then 256 entries of 16
/// bytes (`C0^C1^C2^C3`).
#[cfg(target_arch = "aarch64")]
pub(crate) const AB_KARA_IMAGE_BYTES: usize = 256 * 128 + 256 * 16;
#[cfg(target_arch = "aarch64")]
const AB_KARA_NINTH: usize = 256 * 128;

/// Karatsuba tables of the plain image followed by those of the
/// `x^4`-scaled image, 128-byte aligned, built once per process from the
/// protocol-fixed `k_skip = 6` table (like [`bstatic_partials`]).
#[cfg(target_arch = "aarch64")]
pub(crate) fn ab_kara_tables(inv_table: &InvNttTableByteSingleGf8) -> *const u8 {
    struct Tables {
        storage: Vec<u8>,
        offset: usize,
    }
    static TABLES: std::sync::OnceLock<Tables> = std::sync::OnceLock::new();
    let tables = TABLES.get_or_init(|| {
        let mut storage = vec![0u8; 2 * AB_KARA_IMAGE_BYTES + 2 * AB_Z_IMAGE_BYTES + 127];
        let offset = storage.as_ptr().align_offset(128);
        assert!(offset < 128);
        for (image, src) in [inv_table.data_ptr(), inv_table.scaled_x4_data_ptr()]
            .into_iter()
            .enumerate()
        {
            // SAFETY: each image is 256 rows of 64 bytes.
            let rows = unsafe { core::slice::from_raw_parts(src, 256 * 64) };
            let dst = &mut storage[offset + image * AB_KARA_IMAGE_BYTES..][..AB_KARA_IMAGE_BYTES];
            for x in 0..256 {
                let row = &rows[x * 64..(x + 1) * 64];
                for i in 0..16 {
                    let (c0, c1, c2, c3) = (row[i], row[16 + i], row[32 + i], row[48 + i]);
                    let entry = &mut dst[x * 128..(x + 1) * 128];
                    entry[i] = c0;
                    entry[16 + i] = c1;
                    entry[32 + i] = c2;
                    entry[48 + i] = c3;
                    entry[64 + i] = c0 ^ c1;
                    entry[80 + i] = c2 ^ c3;
                    entry[96 + i] = c0 ^ c2;
                    entry[112 + i] = c1 ^ c3;
                    dst[AB_KARA_NINTH + x * 16 + i] = c0 ^ c1 ^ c2 ^ c3;
                }
            }
            let zdst = &mut storage[offset + 2 * AB_KARA_IMAGE_BYTES + image * AB_Z_IMAGE_BYTES..]
                [..AB_Z_IMAGE_BYTES];
            for x in 0..256 {
                ab_z_fill_entry(&rows[x * 64..(x + 1) * 64], x, zdst);
            }
        }
        Tables { storage, offset }
    });
    // SAFETY: `offset < 128` and the allocation holds `offset + 2 * AB_KARA_IMAGE_BYTES`.
    unsafe { tables.storage.as_ptr().add(tables.offset) }
}

/// One slot of a Karatsuba table, with the byte offset formed in one instruction.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn ab_kara_ld<const V: u32, const J: u32, const SLOT: usize>(
    kara: *const u8,
    word: u64,
) -> core::arch::aarch64::uint8x16_t {
    use core::arch::aarch64::*;
    unsafe {
        if V & ABV_KARA_UNSCALED == 0 {
            return vld1q_u8(kara.add(byte_from_word::<J>(word) * 128 + SLOT * 16));
        }
        let out: uint8x16_t;
        // Each slot base is invariant across words; the maximum offset is 255 * 128.
        if J == 0 {
            core::arch::asm!(
                "ubfiz {idx}, {word}, #7, #8",
                "ldr {out:q}, [{base}, {idx}]",
                word = in(reg) word,
                base = in(reg) kara.add(SLOT * 16),
                idx = out(reg) _,
                out = lateout(vreg) out,
                options(pure, readonly, nostack, preserves_flags),
            );
        } else {
            core::arch::asm!(
                "and {idx}, {mask}, {word}, lsr #{shift}",
                "ldr {out:q}, [{base}, {idx}]",
                shift = const (J * 8 + 57) % 64,
                mask = in(reg) 0x7f80usize,
                word = in(reg) word,
                base = in(reg) kara.add(SLOT * 16),
                idx = out(reg) _,
                out = lateout(vreg) out,
                options(pure, readonly, nostack, preserves_flags),
            );
        }
        out
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
#[test]
fn ab_kara_unscaled_addresses_match() {
    let data: Vec<u8> = (0..2 * AB_KARA_IMAGE_BYTES)
        .map(|i| ((i * 17) ^ (i >> 7) ^ (i >> 15)) as u8)
        .collect();
    unsafe fn check<const J: u32, const SHIFT: u32>(kara: *const u8, word: u64) {
        unsafe {
            macro_rules! slot {
                ($s:literal) => {
                    assert_eq!(
                        core::mem::transmute::<_, [u8; 16]>(ab_kara_ld::<0, J, $s>(kara, word)),
                        core::mem::transmute::<_, [u8; 16]>(ab_kara_ld::<ABV_KARA_UNSCALED, J, $s>(kara, word)),
                        "byte={J}, slot={}, word={word:016x}", $s,
                    );
                };
            }
            slot!(0); slot!(1); slot!(2); slot!(3);
            slot!(4); slot!(5); slot!(6); slot!(7);
            let ninth = kara.add(AB_KARA_NINTH);
            assert_eq!(
                core::mem::transmute::<_, [u8; 16]>(ab_ld::<false, SHIFT>(ninth, word as usize)),
                core::mem::transmute::<_, [u8; 16]>(ab_ld::<true, SHIFT>(ninth, word as usize)),
            );
        }
    }
    for image in 0..2 {
        for x in 0..256u64 {
            for background in [0, u64::MAX] {
                macro_rules! byte {
                    ($j:literal) => {
                        unsafe {
                            check::<$j, { $j * 8 }>(
                                data.as_ptr().add(image * AB_KARA_IMAGE_BYTES),
                                (background & !(0xff << ($j * 8))) | (x << ($j * 8)),
                            );
                        }
                    };
                }
                byte!(0); byte!(1); byte!(2); byte!(3);
                byte!(4); byte!(5); byte!(6); byte!(7);
            }
        }
    }
}

/// Row-body variant bit (with [`ABV_KARA_XOR16`]): the `x^4` Karatsuba
/// image's lookups form their addresses per slot pair in one asm block
/// ([`ab_kara_slot_pair`]): one `add` of the slot offset per pair, then an
/// unscaled register-offset load per byte (`and`/`ubfiz` + `ldr`), instead
/// of `ubfx` + `add (shifted)` + `ldr` per byte. Same lookups, same bytes;
/// no extra live base registers (the eight fixed slot bases of
/// [`ABV_KARA_SCALED_UNSCALED`] spill).
#[cfg(target_arch = "aarch64")]
pub(crate) const ABV_KARA_SLOT: u32 = 8192;

/// Kill switch for [`ABV_KARA_SLOT`]: any value of `FLOCK_NO_ZC_AB_KARA_SLOT`
/// keeps the incumbent address formation. Read once per process. The rows
/// written are byte-identical.
#[cfg(target_arch = "aarch64")]
pub(crate) fn ab_kara_slot_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_ZC_AB_KARA_SLOT").is_none())
}

/// Build-time choice of [`ab_kara_slot_pair`]'s form: `true` = opaque
/// per-pair `add` of the slot offset + ordinary loads (the scheduler can
/// interleave them); `false` = the whole pair in one asm block.
#[cfg(target_arch = "aarch64")]
const AB_KARA_SLOT_PLAIN_LOADS: bool = true;

/// Slot `SLOT` of the Karatsuba entries of bytes `JA` and `JB` (`JA < JB`)
/// of `word`: `(kara[byte JA][SLOT], kara[byte JB][SLOT])`, 16 bytes each.
/// One asm block: `add t, kara, #SLOT*16` (none for slot 0), then per byte
/// `ubfiz`/`and (shifted)` into a byte offset and `ldr q, [t, off]`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn ab_kara_slot_pair<const JA: u32, const JB: u32, const SLOT: usize>(
    kara: *const u8,
    word: u64,
) -> (core::arch::aarch64::uint8x16_t, core::arch::aarch64::uint8x16_t) {
    use core::arch::aarch64::*;
    const { assert!(JA < JB && JB < 8 && SLOT < 8) };
    if AB_KARA_SLOT_PLAIN_LOADS {
        // SAFETY: as below; the add is opaque and takes `word` as a dummy
        // input so the slot base is formed once per pair, not hoisted into
        // eight live registers. The loads are ordinary LLVM loads.
        unsafe {
            let t: *const u8 = if SLOT == 0 {
                kara
            } else {
                let a: usize;
                core::arch::asm!(
                    "add {a}, {b}, #{off} // {w}",
                    off = const SLOT * 16,
                    a = lateout(reg) a,
                    b = in(reg) kara,
                    w = in(reg) word,
                    options(pure, nomem, nostack, preserves_flags),
                );
                a as *const u8
            };
            return (
                vld1q_u8(t.add(ab_z_idx::<JA, 7>(word))),
                vld1q_u8(t.add(ab_z_idx::<JB, 7>(word))),
            );
        }
    }
    let (a, b): (uint8x16_t, uint8x16_t);
    // SAFETY: each offset is a byte value times 128 plus SLOT * 16, inside
    // the caller's 32 KiB image of 128-byte entries; the asm only reads it.
    unsafe {
        match (SLOT == 0, JA == 0) {
            (true, true) => core::arch::asm!(
                "ubfiz {ia}, {w}, #7, #8",
                "ldr {a:q}, [{base}, {ia}]",
                "and {ib}, {mask}, {w}, lsr #{sb}",
                "ldr {b:q}, [{base}, {ib}]",
                sb = const (JB * 8 + 57) % 64,
                mask = in(reg) 0x7f80usize,
                w = in(reg) word,
                base = in(reg) kara,
                ia = out(reg) _,
                ib = out(reg) _,
                a = lateout(vreg) a,
                b = lateout(vreg) b,
                options(pure, readonly, nostack, preserves_flags),
            ),
            (true, false) => core::arch::asm!(
                "and {ia}, {mask}, {w}, lsr #{sa}",
                "ldr {a:q}, [{base}, {ia}]",
                "and {ib}, {mask}, {w}, lsr #{sb}",
                "ldr {b:q}, [{base}, {ib}]",
                sa = const (JA * 8 + 57) % 64,
                sb = const (JB * 8 + 57) % 64,
                mask = in(reg) 0x7f80usize,
                w = in(reg) word,
                base = in(reg) kara,
                ia = out(reg) _,
                ib = out(reg) _,
                a = lateout(vreg) a,
                b = lateout(vreg) b,
                options(pure, readonly, nostack, preserves_flags),
            ),
            (false, true) => core::arch::asm!(
                "add {t}, {base}, #{off}",
                "ubfiz {ia}, {w}, #7, #8",
                "ldr {a:q}, [{t}, {ia}]",
                "and {ib}, {mask}, {w}, lsr #{sb}",
                "ldr {b:q}, [{t}, {ib}]",
                off = const SLOT * 16,
                sb = const (JB * 8 + 57) % 64,
                mask = in(reg) 0x7f80usize,
                w = in(reg) word,
                base = in(reg) kara,
                t = out(reg) _,
                ia = out(reg) _,
                ib = out(reg) _,
                a = lateout(vreg) a,
                b = lateout(vreg) b,
                options(pure, readonly, nostack, preserves_flags),
            ),
            (false, false) => core::arch::asm!(
                "add {t}, {base}, #{off}",
                "and {ia}, {mask}, {w}, lsr #{sa}",
                "ldr {a:q}, [{t}, {ia}]",
                "and {ib}, {mask}, {w}, lsr #{sb}",
                "ldr {b:q}, [{t}, {ib}]",
                off = const SLOT * 16,
                sa = const (JA * 8 + 57) % 64,
                sb = const (JB * 8 + 57) % 64,
                mask = in(reg) 0x7f80usize,
                w = in(reg) word,
                base = in(reg) kara,
                t = out(reg) _,
                ia = out(reg) _,
                ib = out(reg) _,
                a = lateout(vreg) a,
                b = lateout(vreg) b,
                options(pure, readonly, nostack, preserves_flags),
            ),
        }
    }
    (a, b)
}

/// Entries of bytes 0 and 1 of `v` in the ninth table of a Karatsuba image
/// (256 x 16 bytes at [`AB_KARA_NINTH`]): one asm block, one `add` of the
/// table offset, two unscaled loads.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn ab_kara_ninth_pair(
    kara: *const u8,
    v: u64,
) -> (core::arch::aarch64::uint8x16_t, core::arch::aarch64::uint8x16_t) {
    use core::arch::aarch64::*;
    const { assert!(AB_KARA_NINTH == 8 << 12) };
    if AB_KARA_SLOT_PLAIN_LOADS {
        // SAFETY: as below (opaque per-word add, ordinary loads).
        unsafe {
            let a: usize;
            core::arch::asm!(
                "add {a}, {b}, #8, lsl #12 // {w}",
                a = lateout(reg) a,
                b = in(reg) kara,
                w = in(reg) v,
                options(pure, nomem, nostack, preserves_flags),
            );
            let t = a as *const u8;
            return (vld1q_u8(t.add(ab_z_idx::<0, 4>(v))), vld1q_u8(t.add(ab_z_idx::<1, 4>(v))));
        }
    }
    let (a, b): (uint8x16_t, uint8x16_t);
    // SAFETY: the offsets are byte values times 16 inside the 4 KiB ninth
    // table of the image; the asm only reads that memory.
    unsafe {
        core::arch::asm!(
            "add {t}, {base}, #8, lsl #12",
            "ubfiz {ia}, {v}, #4, #8",
            "ldr {a:q}, [{t}, {ia}]",
            "and {ib}, {mask}, {v}, lsr #4",
            "ldr {b:q}, [{t}, {ib}]",
            mask = in(reg) 0xff0usize,
            v = in(reg) v,
            base = in(reg) kara,
            t = out(reg) _,
            ia = out(reg) _,
            ib = out(reg) _,
            a = lateout(vreg) a,
            b = lateout(vreg) b,
            options(pure, readonly, nostack, preserves_flags),
        );
    }
    (a, b)
}

/// [`apply_word_into_4_regs`] out of one image of [`ab_kara_tables`]
/// (`kara` points at the image's first entry).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn apply_word_kara<const V: u32, const SCALED: bool>(
    kara: *const u8,
    word: u64,
) -> (
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        // Bytes 0,1 / 4,5 of `u`: W0^W1 / W2^W3. Bytes 0..4 of `s`: W0^W2,
        // W1^W3. Bytes 0,1 of `v`: W0^W1^W2^W3.
        let u = word ^ (word >> 16);
        let s = word ^ (word >> 32);
        let v = s ^ (s >> 16);
        macro_rules! ld {
            ($src:expr, $j:literal, $slot:literal) => {
                // Two images' slot bases spill; keep only the plain image unscaled.
                if SCALED && V & ABV_KARA_SCALED_UNSCALED == 0 {
                    ab_kara_ld::<0, $j, $slot>(kara, $src)
                } else {
                    ab_kara_ld::<V, $j, $slot>(kara, $src)
                }
            };
        }
        if SCALED && V & ABV_KARA_SLOT != 0 && V & ABV_KARA_XOR16 != 0 {
            // The same eighteen lookups and the same twelve-XOR recombination
            // as the `ABV_KARA_XOR16` arm below; only the address formation
            // differs (see [`ab_kara_slot_pair`]).
            let (e0a, o0a) = ab_kara_slot_pair::<0, 1, 0>(kara, word);
            let (e0b, o0b) = ab_kara_slot_pair::<2, 3, 1>(kara, word);
            let (e1a, o1a) = ab_kara_slot_pair::<4, 5, 2>(kara, word);
            let (e1b, o1b) = ab_kara_slot_pair::<6, 7, 3>(kara, word);
            let o0 = veorq_u8(xor3_u8(o0a, o0b, o1a), o1b);
            let out0 = xor3_u8(xor3_u8(e0a, e0b, e1a), e1b, vextq_u8::<8>(o0, o0));
            let (e0s, o0s) = ab_kara_slot_pair::<0, 1, 4>(kara, u);
            let (e1s, o1s) = ab_kara_slot_pair::<4, 5, 5>(kara, u);
            let t1 = veorq_u8(o0s, o1s);
            let out1 = veorq_u8(xor3_u8(out0, e0s, e1s), vextq_u8::<8>(t1, t1));
            let (esa, osa) = ab_kara_slot_pair::<0, 1, 6>(kara, s);
            let (esb, osb) = ab_kara_slot_pair::<2, 3, 7>(kara, s);
            let t2 = veorq_u8(osa, osb);
            let d2 = xor3_u8(esa, esb, vextq_u8::<8>(t2, t2));
            let out2 = veorq_u8(out0, d2);
            let (ess, oss) = ab_kara_ninth_pair(kara, v);
            let out3 = veorq_u8(xor3_u8(out1, d2, ess), vextq_u8::<8>(oss, oss));
            return (out0, out1, out2, out3);
        }
        // First ("even") and second ("odd") byte of every unit product.
        let (e0a, o0a) = (ld!(word, 0, 0), ld!(word, 1, 0));
        let (e0b, o0b) = (ld!(word, 2, 1), ld!(word, 3, 1));
        let (e1a, o1a) = (ld!(word, 4, 2), ld!(word, 5, 2));
        let (e1b, o1b) = (ld!(word, 6, 3), ld!(word, 7, 3));
        if V & ABV_KARA_XOR16 != 0 {
            // The same eighteen lookups, recombined. The form below returns
            // `out_q = e_q ^ sw(o_q)` with `sw` the 8-byte half exchange
            // (`ext #8`), a fixed byte permutation and therefore GF(2)-linear:
            // `sw(a ^ b) = sw(a) ^ sw(b)`. Expanding its `e_q` / `o_q` chains:
            //   out0 = (e0a^e0b^e1a^e1b) ^ sw(o0a^o0b^o1a^o1b)
            //   out1 = out0 ^ e0s ^ e1s ^ sw(o0s ^ o1s)
            //   out2 = out0 ^ d2,  d2 = esa ^ esb ^ sw(osa ^ osb)
            //   out3 = out1 ^ d2 ^ ess ^ sw(oss)
            // Every output is the same XOR of the same table bytes: twelve
            // XORs and four `ext` instead of fifteen XORs and four `ext`.
            let o0 = veorq_u8(xor3_u8(o0a, o0b, o1a), o1b);
            let out0 = xor3_u8(xor3_u8(e0a, e0b, e1a), e1b, vextq_u8::<8>(o0, o0));
            let (e0s, o0s) = (ld!(u, 0, 4), ld!(u, 1, 4));
            let (e1s, o1s) = (ld!(u, 4, 5), ld!(u, 5, 5));
            let t1 = veorq_u8(o0s, o1s);
            let out1 = veorq_u8(xor3_u8(out0, e0s, e1s), vextq_u8::<8>(t1, t1));
            let (esa, osa) = (ld!(s, 0, 6), ld!(s, 1, 6));
            let (esb, osb) = (ld!(s, 2, 7), ld!(s, 3, 7));
            let t2 = veorq_u8(osa, osb);
            let d2 = xor3_u8(esa, esb, vextq_u8::<8>(t2, t2));
            let out2 = veorq_u8(out0, d2);
            let ninth = kara.add(AB_KARA_NINTH);
            let (ess, oss) = if (!SCALED || V & ABV_KARA_SCALED_UNSCALED != 0) && V & ABV_KARA_UNSCALED != 0 {
                (ab_ld::<true, 0>(ninth, v as usize), ab_ld::<true, 8>(ninth, v as usize))
            } else {
                (
                    vld1q_u8(ninth.add(byte_from_word::<0>(v) * 16)),
                    vld1q_u8(ninth.add(byte_from_word::<1>(v) * 16)),
                )
            };
            let out3 = veorq_u8(xor3_u8(out1, d2, ess), vextq_u8::<8>(oss, oss));
            return (out0, out1, out2, out3);
        }
        let e0 = veorq_u8(xor3_u8(e0a, e0b, e1a), e1b);
        let o0 = veorq_u8(xor3_u8(o0a, o0b, o1a), o1b);
        let (e0s, o0s) = (ld!(u, 0, 4), ld!(u, 1, 4));
        let (e1s, o1s) = (ld!(u, 4, 5), ld!(u, 5, 5));
        let e1 = xor3_u8(e0, e0s, e1s);
        let o1 = xor3_u8(o0, o0s, o1s);
        let (esa, osa) = (ld!(s, 0, 6), ld!(s, 1, 6));
        let (esb, osb) = (ld!(s, 2, 7), ld!(s, 3, 7));
        let te = veorq_u8(esa, esb);
        let to = veorq_u8(osa, osb);
        let e2 = veorq_u8(e0, te);
        let o2 = veorq_u8(o0, to);
        let ninth = kara.add(AB_KARA_NINTH);
        let (ess, oss) = if (!SCALED || V & ABV_KARA_SCALED_UNSCALED != 0) && V & ABV_KARA_UNSCALED != 0 {
            (ab_ld::<true, 0>(ninth, v as usize), ab_ld::<true, 8>(ninth, v as usize))
        } else {
            (
                vld1q_u8(ninth.add(byte_from_word::<0>(v) * 16)),
                vld1q_u8(ninth.add(byte_from_word::<1>(v) * 16)),
            )
        };
        let e3 = xor3_u8(e1, te, ess);
        let o3 = xor3_u8(o1, to, oss);
        (
            veorq_u8(e0, vextq_u8::<8>(o0, o0)),
            veorq_u8(e1, vextq_u8::<8>(o1, o1)),
            veorq_u8(e2, vextq_u8::<8>(o2, o2)),
            veorq_u8(e3, vextq_u8::<8>(o3, o3)),
        )
    }
}

// ---------------------------------------------------------------------------
// Moebius ("z-basis") form of the table apply: 14 loads + 8 indices per word.
//
// Output block n = 2q + h (lane q, half h) receives R_{n ^ b}(W_b) from byte
// b: a product in GF(2)[Z_2^3]. Write the word in the basis z_i = 1 + y_i:
// w_T = XOR of the bytes W_b with b a superset of T (seven shifted XORs), and
// out_n = sum_T R^T_n(w_T) with R^T_n = sum_{U subset T} R_{n ^ U}. R^T is
// constant along the bits of T, so terms without the half bit are whole lane
// values shared by 1, 2 or 4 lanes, and terms with it are 8-byte values
// duplicated over both halves of a lane, summed two lanes per register and
// spread with one zip per lane.
// ---------------------------------------------------------------------------

/// The default form of the round-one AB rows when the Karatsuba apply is on
/// (`FLOCK_NO_ZC_AB_Z=1` restores the record's `ABV_KARA | ABV_KARA_UNSCALED`
/// form): the plain image in z form out of 128-byte records, the `x^4` image
/// in Karatsuba form, and the static-B partial path without the half-swapped
/// image. Measured on an M5 Pro P-core (synthetic static-B rows, 25
/// alternated rounds): 27.36 against 29.87 ns per row at the lower quartile.
#[cfg(target_arch = "aarch64")]
pub(crate) const AB_Z_VARIANT: u32 = ABV_KARA | ABV_KARA_UNSCALED | ABV_Z | ABV_Z_REC | ABV_NOSW;

/// Kill switch: exactly `FLOCK_NO_ZC_AB_Z=1` keeps the record's Karatsuba
/// form for the plain image. Same bytes either way.
#[cfg(target_arch = "aarch64")]
pub(crate) fn ab_z_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var_os("FLOCK_NO_ZC_AB_Z").is_none_or(|v| v != *"1")
    });
    *ON
}

/// Row-body variant bit: the plain image uses the z-basis apply.
#[cfg(target_arch = "aarch64")]
pub(crate) const ABV_Z: u32 = 8;
/// Row-body variant bit: a word that keeps the Karatsuba apply (the `x^4`
/// image under [`ABV_Z`]) recombines its eighteen lookups with twelve XORs
/// and four `ext` instead of fifteen XORs and four `ext` (see
/// [`apply_word_kara`]). Same lookups, same output bytes.
#[cfg(target_arch = "aarch64")]
pub(crate) const ABV_KARA_XOR16: u32 = 512;

/// Kill switch for [`ABV_KARA_XOR16`]: any value of
/// `FLOCK_NO_ZC_AB_KARA_XOR16` restores the incumbent recombination in the
/// same binary. Read once per process. The rows written are byte-identical.
#[cfg(target_arch = "aarch64")]
pub(crate) fn ab_kara_xor16_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_ZC_AB_KARA_XOR16").is_none())
}

/// Row-body variant bit (with [`ABV_Z`] and [`ABV_Z_REC`]): a static-B word
/// whose varying bytes are exactly bytes `0..NB` (`NB` = 2, 3 or 4) takes
/// its partial image plus [`apply_word_z_low`] of the varying bytes instead
/// of the per-byte rows or the whole-word apply. Same output bytes.
#[cfg(target_arch = "aarch64")]
pub(crate) const ABV_BSTATIC_Z: u32 = 4096;

/// Kill switch for [`ABV_BSTATIC_Z`]: any value of
/// `FLOCK_NO_ZC_AB_BSTATIC_Z` keeps the incumbent static-B word forms. Read
/// once per process. The rows written are byte-identical.
#[cfg(target_arch = "aarch64")]
pub(crate) fn ab_bstatic_z_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_ZC_AB_BSTATIC_Z").is_none())
}

/// [`apply_word_z`] (record form) of the word `w` restricted to its bytes
/// `0..NB` (`2 <= NB <= 4`), the other bytes taken as zero. The apply is
/// F2-linear in each lookup index and every table's row 0 is zero, so the
/// lookups whose index involves only bytes `NB..8` are dropped: the plain
/// row, the `{1,3,5,7}` record pair, and for `NB >= 3` the `{2,3,6,7}` pair,
/// for `NB = 4` the `{3,7}` pair (8, 10 or 12 loads). `t0` is the plain
/// image, `kara` its [`ab_kara_tables`] base.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn apply_word_z_low<const NB: u32>(
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
        let v = w & ((1u64 << (8 * NB)) - 1);
        let z = kara.add(2 * AB_KARA_IMAGE_BYTES);
        let ys = v ^ (v >> 8);
        let yl1 = v ^ (v >> 16);
        let yall = ys ^ (ys >> 16);
        let ld = |base: *const u8, off: usize| vld1q_u8(base.add(off));
        let r0 = t0.add(ab_z_idx::<0, 6>(yall));
        let (l0, l1, l2, l3) = (ld(r0, 0), ld(r0, 16), ld(r0, 32), ld(r0, 48));
        let rec = |i: usize| -> *const u8 {
            let a: usize;
            core::arch::asm!("add {a}, {b}, {i}", a = lateout(reg) a, b = in(reg) z.add(AB_Z_REC), i = in(reg) i,
                options(pure, nomem, nostack, preserves_flags));
            a as *const u8
        };
        // {1,3,5,7}: bytes 1 ^ 3.
        let a = rec(ab_z_idx::<1, 7>(yl1));
        let (mut p, mut q) = (ld(a, 64), ld(a, 80));
        if NB >= 4 {
            // {3,7}: byte 3.
            let a = rec(ab_z_idx::<3, 7>(v));
            p = veorq_u8(p, ld(a, 96));
            q = veorq_u8(q, ld(a, 112));
        }
        let p = vreinterpretq_u64_u8(p);
        let q = vreinterpretq_u64_u8(q);
        let b0 = vreinterpretq_u8_u64(vzip1q_u64(p, p));
        let b1 = vreinterpretq_u8_u64(vzip2q_u64(p, p));
        let b2 = vreinterpretq_u8_u64(vzip1q_u64(q, q));
        let b3 = vreinterpretq_u8_u64(vzip2q_u64(q, q));
        if NB >= 3 {
            // {2,3,6,7}: bytes 2 ^ 3.
            let a = rec(ab_z_idx::<2, 7>(ys));
            let (m0, m2) = (ld(a, 0), ld(a, 16));
            (xor3_u8(l0, m0, b0), xor3_u8(l1, m0, b1), xor3_u8(l2, m2, b2), xor3_u8(l3, m2, b3))
        } else {
            (veorq_u8(l0, b0), veorq_u8(l1, b1), veorq_u8(l2, b2), veorq_u8(l3, b3))
        }
    }
}

/// The partial image at `pp` (64 bytes) plus [`apply_word_z_low`] of `w`.
/// Under `KF` ([`ABV_ZK`]) with `NB = 2` the partial lanes enter the same
/// `eor3` as the plain row and the duplicated half term (four `eor` fewer);
/// otherwise the incumbent form. Same bytes.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn apply_word_z_low_pp<const NB: u32, const KF: bool>(
    t0: *const u8,
    kara: *const u8,
    w: u64,
    pp: *const u8,
) -> (
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        if !(KF && NB == 2) {
            let (z0, z1, z2, z3) = apply_word_z_low::<NB>(t0, kara, w);
            return (
                veorq_u8(vld1q_u8(pp), z0),
                veorq_u8(vld1q_u8(pp.add(16)), z1),
                veorq_u8(vld1q_u8(pp.add(32)), z2),
                veorq_u8(vld1q_u8(pp.add(48)), z3),
            );
        }
        // NB = 2: the plain row of `yall` and the `{1,3,5,7}` record pair of
        // `yl1`, exactly as `apply_word_z_low::<2>`.
        let v = w & 0xffff;
        let z = kara.add(2 * AB_KARA_IMAGE_BYTES);
        let yl1 = v ^ (v >> 16);
        let ys = v ^ (v >> 8);
        let yall = ys ^ (ys >> 16);
        let ld = |base: *const u8, off: usize| vld1q_u8(base.add(off));
        let r0 = t0.add(ab_z_idx::<0, 6>(yall));
        let (l0, l1, l2, l3) = (ld(r0, 0), ld(r0, 16), ld(r0, 32), ld(r0, 48));
        let a: usize;
        core::arch::asm!("add {a}, {b}, {i}", a = lateout(reg) a, b = in(reg) z.add(AB_Z_REC), i = in(reg) ab_z_idx::<1, 7>(yl1),
            options(pure, nomem, nostack, preserves_flags));
        let a = a as *const u8;
        let p = vreinterpretq_u64_u8(ld(a, 64));
        let q = vreinterpretq_u64_u8(ld(a, 80));
        let b0 = vreinterpretq_u8_u64(vzip1q_u64(p, p));
        let b1 = vreinterpretq_u8_u64(vzip2q_u64(p, p));
        let b2 = vreinterpretq_u8_u64(vzip1q_u64(q, q));
        let b3 = vreinterpretq_u8_u64(vzip2q_u64(q, q));
        (
            xor3_u8(l0, b0, ld(pp, 0)),
            xor3_u8(l1, b1, ld(pp, 16)),
            xor3_u8(l2, b2, ld(pp, 32)),
            xor3_u8(l3, b3, ld(pp, 48)),
        )
    }
}

/// Test access: one static-B word `(blk, k)` of a plan whose varying bytes
/// are `0..NB`, through the partial path (per-byte rows) and through the
/// partial image plus [`apply_word_z_low`]; `w` must match the plan's
/// static bytes.
#[cfg(all(test, target_arch = "aarch64"))]
pub(crate) fn ab_bstatic_z_low_forms<const NB: u32>(
    inv_table: &InvNttTableByteSingleGf8,
    blk: usize,
    k: usize,
    w: u64,
) -> ([u8; 64], [u8; 64]) {
    use core::arch::aarch64::*;
    let partials = bstatic_partials(inv_table);
    let kara = ab_kara_tables(inv_table);
    let plain = inv_table.data_ptr();
    let mut out = ([0u8; 64], [0u8; 64]);
    unsafe {
        let pp = partials[blk * 8 + k].as_ptr();
        let mut d = [vld1q_u8(pp), vld1q_u8(pp.add(16)), vld1q_u8(pp.add(32)), vld1q_u8(pp.add(48))];
        {
            let [d0, d1, d2, d3] = &mut d;
            xor_vary_ext::<0, 1>(plain, w, [d0, d1, d2, d3]);
            if NB == 3 {
                xor_vary_ext::<2, 8>(plain, w, [d0, d1, d2, d3]);
            }
            if NB == 4 {
                xor_vary_ext::<2, 3>(plain, w, [d0, d1, d2, d3]);
            }
        }
        let z = apply_word_z_low::<NB>(plain, kara, w);
        let e = [veorq_u8(vld1q_u8(pp), z.0), veorq_u8(vld1q_u8(pp.add(16)), z.1), veorq_u8(vld1q_u8(pp.add(32)), z.2), veorq_u8(vld1q_u8(pp.add(48)), z.3)];
        for i in 0..4 {
            vst1q_u8(out.0.as_mut_ptr().add(16 * i), d[i]);
            vst1q_u8(out.1.as_mut_ptr().add(16 * i), e[i]);
        }
    }
    out
}

/// Row-body variant bit (with [`ABV_Z`]): the `x^4` image uses it too.
#[cfg(target_arch = "aarch64")]
pub(crate) const ABV_Z_SCALED: u32 = 16;

#[cfg(target_arch = "aarch64")]
const AB_Z_L1: usize = 0; // 256 x 32: lanes M0, M2 of R^{l1}
#[cfg(target_arch = "aarch64")]
const AB_Z_L2: usize = 8192; // 256 x 32: lanes N0, N1 of R^{l2}
#[cfg(target_arch = "aarch64")]
const AB_Z_L12: usize = 16384; // 256 x 16: lane K of R^{l1 l2}
#[cfg(target_arch = "aarch64")]
const AB_Z_S: usize = 20480; // 256 x 32: [d0|d1], [d2|d3]
#[cfg(target_arch = "aarch64")]
const AB_Z_SL1: usize = 28672; // 256 x 32: [e0|e0], [e2|e2]
#[cfg(target_arch = "aarch64")]
const AB_Z_SL2: usize = 36864; // 256 x 16: [f0|f1]
#[cfg(target_arch = "aarch64")]
const AB_Z_ALL: usize = 40960; // 256 x 16: [g|g]
/// Bytes of one image of the z-basis tables (the `T = {}` table is the
/// image itself and is not copied).
#[cfg(target_arch = "aarch64")]
pub(crate) const AB_Z_IMAGE_BYTES: usize = AB_Z_REC + 3 * 32768 + 16384;
#[cfg(target_arch = "aarch64")]
const AB_Z_REC3: usize = AB_Z_REC + 2 * 32768; // 256 x 128: [M0 M2 | N0 N1 | e0e0 e2e2 | f0f0 f1f1]
#[cfg(target_arch = "aarch64")]
const AB_Z_DDUP: usize = AB_Z_REC + 3 * 32768; // 256 x 64: d0d0 d1d1 d2d2 d3d3
/// Row-body variant bit: the `x^4` Karatsuba image also takes the
/// unscaled-index loads (for forms where the plain image is in z form).
#[cfg(target_arch = "aarch64")]
pub(crate) const ABV_KARA_SCALED_UNSCALED: u32 = 32;
#[cfg(target_arch = "aarch64")]
const AB_Z_REC2: usize = AB_Z_REC + 32768; // 256 x 128: [M0 M2 | N0 N1 | d01 d23 | e0 e2 | g | 0]
#[cfg(target_arch = "aarch64")]
// 4 KiB-aligned (2 KiB of padding after `AB_Z_ALLP`, never read): the record
// base `z + AB_Z_REC` is then one `add #imm, lsl #12` from the image base
// instead of two adds. Layout only; same table bytes.
const AB_Z_REC: usize = 53248; // 256 x 128: [M0 M2 | N0 N1 | d01 d23 | e0e0 e2e2]
#[cfg(target_arch = "aarch64")]
const _: () = assert!(AB_Z_REC % 4096 == 0 && AB_Z_ALLP + 256 * 8 <= AB_Z_REC && (2 * AB_KARA_IMAGE_BYTES) % 4096 == 0);
/// Row-body variant bit (with [`ABV_Z`]): the four two-load z tables
/// interleaved in one 128-byte record per byte value, one base register.
#[cfg(target_arch = "aarch64")]
pub(crate) const ABV_Z_REC: u32 = 128;
#[cfg(target_arch = "aarch64")]
const AB_Z_SL1P: usize = 45056; // 256 x 16: [e0|e2]
#[cfg(target_arch = "aarch64")]
const AB_Z_ALLP: usize = 49152; // 256 x 8: g
/// Row-body variant bit (with [`ABV_Z`]): packed `s l1` and `s l1 l2`
/// tables (two more zips, one 8-byte replicating load; 38 KiB per image).
#[cfg(target_arch = "aarch64")]
pub(crate) const ABV_Z_PACK: u32 = 0; // retired: slower in every measured form

#[cfg(target_arch = "aarch64")]
fn ab_z_fill_entry(row: &[u8], x: usize, dst: &mut [u8]) {
    let blk = |c: usize| -> u64 { u64::from_le_bytes(row[8 * c..8 * c + 8].try_into().unwrap()) };
    let rt = |t: usize, n: usize| -> u64 {
        let mut v = 0u64;
        for u in 0..8usize {
            if u & !t == 0 {
                v ^= blk(n ^ u);
            }
        }
        v
    };
    let mut put = |off: usize, stride: usize, vals: &[u64]| {
        for (i, v) in vals.iter().enumerate() {
            dst[off + x * stride + 8 * i..off + x * stride + 8 * i + 8].copy_from_slice(&v.to_le_bytes());
        }
    };
    put(AB_Z_L1, 32, &[rt(2, 0), rt(2, 1), rt(2, 4), rt(2, 5)]);
    put(AB_Z_L2, 32, &[rt(4, 0), rt(4, 1), rt(4, 2), rt(4, 3)]);
    put(AB_Z_L12, 16, &[rt(6, 0), rt(6, 1)]);
    put(AB_Z_S, 32, &[rt(1, 0), rt(1, 2), rt(1, 4), rt(1, 6)]);
    put(AB_Z_SL1, 32, &[rt(3, 0), rt(3, 0), rt(3, 4), rt(3, 4)]);
    put(AB_Z_SL2, 16, &[rt(5, 0), rt(5, 2)]);
    put(AB_Z_ALL, 16, &[rt(7, 0), rt(7, 0)]);
    put(AB_Z_SL1P, 16, &[rt(3, 0), rt(3, 4)]);
    put(AB_Z_ALLP, 8, &[rt(7, 0)]);
    put(AB_Z_REC, 128, &[rt(2, 0), rt(2, 1), rt(2, 4), rt(2, 5), rt(4, 0), rt(4, 1), rt(4, 2), rt(4, 3),
        rt(1, 0), rt(1, 2), rt(1, 4), rt(1, 6), rt(3, 0), rt(3, 0), rt(3, 4), rt(3, 4)]);
    put(AB_Z_REC3, 128, &[rt(2, 0), rt(2, 1), rt(2, 4), rt(2, 5), rt(4, 0), rt(4, 1), rt(4, 2), rt(4, 3),
        rt(3, 0), rt(3, 0), rt(3, 4), rt(3, 4), rt(5, 0), rt(5, 0), rt(5, 2), rt(5, 2)]);
    put(AB_Z_DDUP, 64, &[rt(1, 0), rt(1, 0), rt(1, 2), rt(1, 2), rt(1, 4), rt(1, 4), rt(1, 6), rt(1, 6)]);
    put(AB_Z_REC2, 128, &[rt(2, 0), rt(2, 1), rt(2, 4), rt(2, 5), rt(4, 0), rt(4, 1), rt(4, 2), rt(4, 3),
        rt(1, 0), rt(1, 2), rt(1, 4), rt(1, 6), rt(3, 0), rt(3, 4), rt(7, 0), 0]);
}

/// Byte offset `byte J of word << LOG` in one instruction.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn ab_z_idx<const J: u32, const LOG: u32>(word: u64) -> usize {
    let idx: usize;
    unsafe {
        if J == 0 {
            core::arch::asm!(
                "ubfiz {idx}, {word}, #{log}, #8",
                log = const LOG,
                word = in(reg) word,
                idx = lateout(reg) idx,
                options(pure, nomem, nostack, preserves_flags),
            );
        } else {
            core::arch::asm!(
                "and {idx}, {mask}, {word}, lsr #{shift}",
                shift = const (J * 8 + 64 - LOG) % 64,
                mask = in(reg) 0xffusize << LOG,
                word = in(reg) word,
                idx = lateout(reg) idx,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
    }
    idx
}

/// [`apply_word_into_4_regs`] in the z-basis form. `t0` is the table image
/// (64-byte rows), `z` its [`AB_Z_IMAGE_BYTES`] of z-basis tables. `OUT3`:
/// the outputs in the [`ABV_Z_OUT3`] form.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn apply_word_z<const PACK: bool, const REC: bool, const OUT3: bool>(
    t0: *const u8,
    z: *const u8,
    w: u64,
) -> (
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        let ys = w ^ (w >> 8);
        let yl1 = w ^ (w >> 16);
        let yl2 = w ^ (w >> 32);
        let ysl1 = ys ^ (ys >> 16);
        let ysl2 = ys ^ (ys >> 32);
        let yl12 = yl1 ^ (yl1 >> 32);
        let yall = ysl1 ^ (ysl1 >> 32);
        let ld = |base: *const u8, idx: usize| vld1q_u8(base.add(idx));
        // T = {}: the plain row.
        let i0 = ab_z_idx::<0, 6>(yall);
        let r0 = t0.add(i0);
        let (l0, l1, l2, l3) = (ld(r0, 0), ld(r0, 16), ld(r0, 32), ld(r0, 48));
        // An opaque base + offset add, so the two loads become one `ldp`.
        let rec = |i: usize| -> *const u8 {
            let a: usize;
            core::arch::asm!("add {a}, {b}, {i}", a = lateout(reg) a, b = in(reg) z.add(if PACK { AB_Z_REC2 } else { AB_Z_REC }), i = in(reg) i,
                options(pure, nomem, nostack, preserves_flags));
            a as *const u8
        };
        let (m0, m2, n0, n1, d01, d23, e0, e2, g);
        if REC {
            let a = rec(ab_z_idx::<2, 7>(ysl2));
            (m0, m2) = (ld(a, 0), ld(a, 16));
            let a = rec(ab_z_idx::<4, 7>(ysl1));
            (n0, n1) = (ld(a, 32), ld(a, 48));
            let a = rec(ab_z_idx::<1, 7>(yl12));
            (d01, d23) = (ld(a, 64), ld(a, 80));
            let a = rec(ab_z_idx::<3, 7>(yl2));
            if PACK {
                let e = vreinterpretq_u64_u8(ld(a, 96));
                e0 = vreinterpretq_u8_u64(vzip1q_u64(e, e));
                e2 = vreinterpretq_u8_u64(vzip2q_u64(e, e));
                let a = rec(ab_z_idx::<7, 7>(w));
                g = vreinterpretq_u8_u64(vld1q_dup_u64(a.add(112).cast::<u64>()));
            } else {
                (e0, e2) = (ld(a, 96), ld(a, 112));
                g = ld(z.add(AB_Z_ALL), ab_z_idx::<7, 4>(w));
            }
        } else {
            let i = ab_z_idx::<2, 5>(ysl2);
            (m0, m2) = (ld(z.add(AB_Z_L1), i), ld(z.add(AB_Z_L1 + 16), i));
            let i = ab_z_idx::<4, 5>(ysl1);
            (n0, n1) = (ld(z.add(AB_Z_L2), i), ld(z.add(AB_Z_L2 + 16), i));
            let i = ab_z_idx::<1, 5>(yl12);
            (d01, d23) = (ld(z.add(AB_Z_S), i), ld(z.add(AB_Z_S + 16), i));
            if PACK {
                let e = vreinterpretq_u64_u8(ld(z.add(AB_Z_SL1P), ab_z_idx::<3, 4>(yl2)));
                e0 = vreinterpretq_u8_u64(vzip1q_u64(e, e));
                e2 = vreinterpretq_u8_u64(vzip2q_u64(e, e));
                g = vreinterpretq_u8_u64(vld1q_dup_u64(z.add(AB_Z_ALLP + ab_z_idx::<7, 3>(w)).cast::<u64>()));
            } else {
                let i = ab_z_idx::<3, 5>(yl2);
                (e0, e2) = (ld(z.add(AB_Z_SL1), i), ld(z.add(AB_Z_SL1 + 16), i));
                g = ld(z.add(AB_Z_ALL), ab_z_idx::<7, 4>(w));
            }
        }
        let k = ld(z.add(AB_Z_L12), ab_z_idx::<6, 4>(ys));
        let f = ld(z.add(AB_Z_SL2), ab_z_idx::<5, 4>(yl1));
        let fg = veorq_u8(f, g);
        let p = vreinterpretq_u64_u8(xor3_u8(d01, e0, fg));
        let q = vreinterpretq_u64_u8(xor3_u8(d23, e2, fg));
        if OUT3 {
            // The five terms of each output below, as two three-way XORs.
            let b0 = vreinterpretq_u8_u64(vzip1q_u64(p, p));
            let b1 = vreinterpretq_u8_u64(vzip2q_u64(p, p));
            let b2 = vreinterpretq_u8_u64(vzip1q_u64(q, q));
            let b3 = vreinterpretq_u8_u64(vzip2q_u64(q, q));
            return (
                xor3_u8(xor3_u8(m0, k, n0), l0, b0),
                xor3_u8(xor3_u8(m0, k, n1), l1, b1),
                xor3_u8(xor3_u8(m2, k, n0), l2, b2),
                xor3_u8(xor3_u8(m2, k, n1), l3, b3),
            );
        }
        let mk0 = veorq_u8(m0, k);
        let mk2 = veorq_u8(m2, k);
        let b0 = vreinterpretq_u8_u64(vzip1q_u64(p, p));
        let b1 = vreinterpretq_u8_u64(vzip2q_u64(p, p));
        let b2 = vreinterpretq_u8_u64(vzip1q_u64(q, q));
        let b3 = vreinterpretq_u8_u64(vzip2q_u64(q, q));
        (
            veorq_u8(xor3_u8(l0, mk0, n0), b0),
            veorq_u8(xor3_u8(l1, mk0, n1), b1),
            veorq_u8(xor3_u8(l2, mk2, n0), b2),
            veorq_u8(xor3_u8(l3, mk2, n1), b3),
        )
    }
}

/// Row-body variant bit (with [`ABV_Z`] and [`ABV_Z_REC`]): the z-basis
/// apply forms each of its four outputs as two three-way XORs of its five
/// terms (`m`, `k`, `n`, the plain row lane and the spread `b`) instead of
/// XORing `m ^ k` once for two outputs and finishing each output with a
/// three-way and a two-way XOR: eight XOR instructions per word instead of
/// ten. Every output is the XOR of the same terms, so the bytes are the same.
#[cfg(target_arch = "aarch64")]
pub(crate) const ABV_Z_OUT3: u32 = 16384;

/// Kill switch for [`ABV_Z_OUT3`]: any value of `FLOCK_NO_ZC_AB_Z_EOR3`
/// keeps the shared `m ^ k` form. Read once per process. The rows written
/// are byte-identical.
#[cfg(target_arch = "aarch64")]
pub(crate) fn ab_z_out3_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_ZC_AB_Z_EOR3").is_none())
}

/// The [`ABV_Z_OUT3`] z-basis apply writes the shared `m ^ k` form's bytes
/// and the plain apply's, for the zero and all-ones words, every one-byte
/// word (each value at each byte) and random words.
#[cfg(all(test, target_arch = "aarch64"))]
#[test]
fn ab_z_out3_apply_matches_shared_form() {
    use crate::ntt::AdditiveNttGf8;
    use core::arch::aarch64::*;
    let table =
        InvNttTableByteSingleGf8::new(&AdditiveNttGf8::new(6, F8::ZERO), &AdditiveNttGf8::new(6, F8(1 << 6)));
    let kara = ab_kara_tables(&table);
    let (t, sw) = (table.data_ptr(), table.half_swapped_data_ptr());
    let bytes = |v: (uint8x16_t, uint8x16_t, uint8x16_t, uint8x16_t)| {
        let mut out = [0u8; 64];
        // SAFETY: four 16-byte stores into a 64-byte array.
        unsafe {
            vst1q_u8(out.as_mut_ptr(), v.0);
            vst1q_u8(out.as_mut_ptr().add(16), v.1);
            vst1q_u8(out.as_mut_ptr().add(32), v.2);
            vst1q_u8(out.as_mut_ptr().add(48), v.3);
        }
        out
    };
    let mut x = 0x2545_F491_4F6C_DD1Du64;
    for i in 0..20000u64 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let w = match i {
            0 => 0,
            1 => u64::MAX,
            2..=2049 => (i & 0xff) << (8 * ((i >> 8) & 7)),
            _ => x,
        };
        // SAFETY: the applies read inside the table images only.
        let (new, old, plain) = unsafe {
            (
                bytes(apply_word_into_4_regs_v::<{ AB_Z_VARIANT | ABV_Z_OUT3 }, false>(t, sw, kara, w)),
                bytes(apply_word_into_4_regs_v::<AB_Z_VARIANT, false>(t, sw, kara, w)),
                bytes(apply_word_into_4_regs(t, sw, w)),
            )
        };
        assert_eq!(new, old, "w={w:016x}");
        assert_eq!(new, plain, "w={w:016x}");
    }
}

/// Row-body variant bit (with [`ABV_Z`] and [`ABV_Z_REC`]): the z-basis
/// apply takes its eight lookup bytes from one word, the superset (Möbius)
/// transform of `w` (`z[T] = XOR of W_b over b ⊇ T`), formed in three
/// butterfly steps (five integer instructions, no live mask registers)
/// instead of the seven partial XOR-shift words. Same lookups, same bytes.
#[cfg(target_arch = "aarch64")]
pub(crate) const ABV_Z_MT: u32 = 1024;

/// Opt-in for [`ABV_Z_MT`]: off by default (it measured ~1.7% slower on a
/// single M5 Pro core); `FLOCK_ZC_AB_Z_MT=1` enables it. Read once per
/// process. Same bytes either way.
#[cfg(target_arch = "aarch64")]
pub(crate) fn ab_z_mt_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_ZC_AB_Z_MT").is_some())
}

/// Superset transform of the eight bytes of `w`: byte `T` of the result is
/// the XOR of the bytes `b` of `w` with `b & T == T`. Three butterflies:
/// `and` with the odd-byte / high-pair immediate mask and `eor` with the
/// shifted result, then one `eor` with a shifted register.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) fn ab_superset_bytes(w: u64) -> u64 {
    let z: u64;
    // SAFETY: register arithmetic only.
    unsafe {
        // `(w >> 8) & M == (w & (M << 8)) >> 8`: the masks are logical
        // immediates, so no mask register is live.
        core::arch::asm!(
            "and {t}, {w}, #0xff00ff00ff00ff00",
            "eor {z}, {w}, {t}, lsr #8",
            "and {t}, {z}, #0xffff0000ffff0000",
            "eor {z}, {z}, {t}, lsr #16",
            "eor {z}, {z}, {z}, lsr #32",
            w = in(reg) w,
            t = out(reg) _,
            z = out(reg) z,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    z
}

/// [`apply_word_z`] (`REC`, not `PACK`) with the lookup bytes taken from
/// [`ab_superset_bytes`] of `w`: byte 0 indexes the plain row, bytes 2, 4,
/// 1, 3 the record pairs, bytes 6, 5, 7 the `l1 l2`, `s l2` and `s l1 l2`
/// tables, exactly the bytes `apply_word_z` extracts from its partial
/// words.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn apply_word_z_mt(
    t0: *const u8,
    z: *const u8,
    w: u64,
) -> (
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        let y = ab_superset_bytes(w);
        let ld = |base: *const u8, idx: usize| vld1q_u8(base.add(idx));
        let r0 = t0.add(ab_z_idx::<0, 6>(y));
        let (l0, l1, l2, l3) = (ld(r0, 0), ld(r0, 16), ld(r0, 32), ld(r0, 48));
        let rec = |i: usize| -> *const u8 {
            let a: usize;
            core::arch::asm!("add {a}, {b}, {i}", a = lateout(reg) a, b = in(reg) z.add(AB_Z_REC), i = in(reg) i,
                options(pure, nomem, nostack, preserves_flags));
            a as *const u8
        };
        let a = rec(ab_z_idx::<2, 7>(y));
        let (m0, m2) = (ld(a, 0), ld(a, 16));
        let a = rec(ab_z_idx::<4, 7>(y));
        let (n0, n1) = (ld(a, 32), ld(a, 48));
        let a = rec(ab_z_idx::<1, 7>(y));
        let (d01, d23) = (ld(a, 64), ld(a, 80));
        let a = rec(ab_z_idx::<3, 7>(y));
        let (e0, e2) = (ld(a, 96), ld(a, 112));
        let g = ld(z.add(AB_Z_ALL), ab_z_idx::<7, 4>(y));
        let k = ld(z.add(AB_Z_L12), ab_z_idx::<6, 4>(y));
        let f = ld(z.add(AB_Z_SL2), ab_z_idx::<5, 4>(y));
        let fg = veorq_u8(f, g);
        let p = vreinterpretq_u64_u8(xor3_u8(d01, e0, fg));
        let q = vreinterpretq_u64_u8(xor3_u8(d23, e2, fg));
        let mk0 = veorq_u8(m0, k);
        let mk2 = veorq_u8(m2, k);
        let b0 = vreinterpretq_u8_u64(vzip1q_u64(p, p));
        let b1 = vreinterpretq_u8_u64(vzip2q_u64(p, p));
        let b2 = vreinterpretq_u8_u64(vzip1q_u64(q, q));
        let b3 = vreinterpretq_u8_u64(vzip2q_u64(q, q));
        (
            veorq_u8(xor3_u8(l0, mk0, n0), b0),
            veorq_u8(xor3_u8(l1, mk0, n1), b1),
            veorq_u8(xor3_u8(l2, mk2, n0), b2),
            veorq_u8(xor3_u8(l3, mk2, n1), b3),
        )
    }
}

/// Row-body variant bit (with [`ABV_Z`] and [`ABV_Z_REC`]): the z apply's
/// eight lookup bytes from four index words formed in five integer
/// instructions, depth four (masked first layer, then unmasked layers that
/// keep the previous word for the bytes they would corrupt), instead of the
/// seven partial XOR-shift words. Same lookups, same bytes. See
/// [`ab_superset_words_m1`].
#[cfg(target_arch = "aarch64")]
pub(crate) const ABV_Z_M1: u32 = 32768;

/// Kill switch for [`ABV_Z_M1`]: any value of `FLOCK_NO_ZC_AB_Z_M1` keeps
/// the seven-word index form. Read once per process. Same bytes.
#[cfg(target_arch = "aarch64")]
pub(crate) fn ab_z_m1_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_ZC_AB_Z_M1").is_none())
}

/// The superset bytes `z[T]` (XOR of the bytes `b ⊇ T` of `w`) spread over
/// four words, in five instructions:
/// - `y1 = w ^ ((w & 0xff00..ff00) >> 8)`: the complete first layer
///   (`and` with a logical immediate + `eor` with a shifted register);
/// - `y2 = y1 ^ (y1 >> 16)`: second layer, exact at bytes 0, 1, 4, 5 (at
///   bytes 2, 3, 6, 7 the second layer is `y1` itself);
/// - `y3 = y2 ^ (y2 >> 32)`: exact at bytes 0, 1;
/// - `y3b = y1 ^ (y1 >> 32)`: exact at bytes 2, 3.
/// So `z0 = y3[0]`, `z1 = y3[1]`, `z2 = y3b[2]`, `z3 = y3b[3]`, `z4 = y2[4]`,
/// `z5 = y2[5]`, `z6 = y1[6]`, `z7 = w[7]`. Returns `(y1, y2, y3, y3b)`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) fn ab_superset_words_m1(w: u64) -> (u64, u64, u64, u64) {
    // `(w & M) >> 8` with the mask applied before the shift, so it stays a
    // logical immediate (LLVM otherwise emits `lsr` + `and` + `eor`).
    let y1: u64;
    // SAFETY: register arithmetic only.
    unsafe {
        core::arch::asm!(
            "and {t}, {w}, #0xff00ff00ff00ff00",
            "eor {y}, {w}, {t}, lsr #8",
            w = in(reg) w,
            t = out(reg) _,
            y = lateout(reg) y1,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    let y2 = y1 ^ (y1 >> 16);
    let y3 = y2 ^ (y2 >> 32);
    let y3b = y1 ^ (y1 >> 32);
    (y1, y2, y3, y3b)
}

/// Row-body variant bit (with [`ABV_Z_M1`]): the static-B rows' z applies
/// XOR every lookup into the outputs with `eor3` only. The `{l1 l2}` lane
/// `K` (one register shared by all four outputs) enters each output's
/// second `eor3` next to the duplicated half term instead of being
/// pre-added to the two `M` lanes (two `eor` fewer per word: 15 vector ops
/// instead of 17); a static-B word whose varying bytes are exactly bytes 0
/// and 1 adds its partial image in the same `eor3` as the half term (four
/// `eor` fewer). Every output byte is the same XOR of the same table bytes.
#[cfg(target_arch = "aarch64")]
/// Shares its bit with [`ABV_Z_OUT3`]: the apply tests `ABV_Z_M1` first, so
/// the bit means `ABV_ZK` with `ABV_Z_M1` and `ABV_Z_OUT3` without it.
pub(crate) const ABV_ZK: u32 = 16384;

/// Kill switch for [`ABV_ZK`]: any value of `FLOCK_NO_ZC_AB_ZK` keeps the
/// incumbent XOR order. Read once per process. Same bytes.
#[cfg(target_arch = "aarch64")]
pub(crate) fn ab_zk_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_ZC_AB_ZK").is_none())
}

/// [`apply_word_z`] (`REC`, not `PACK`) with the lookup bytes taken from
/// [`ab_superset_words_m1`]: exactly the bytes `apply_word_z` extracts from
/// its seven partial words, so the same lookups. `KF` ([`ABV_ZK`]): `K`
/// enters each output's second `eor3` instead of the `M` lanes.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn apply_word_z_m1<const KF: bool>(
    t0: *const u8,
    z: *const u8,
    w: u64,
) -> (
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
) {
    use core::arch::aarch64::*;
    unsafe {
        let (y1, y2, y3, y3b) = ab_superset_words_m1(w);
        let ld = |base: *const u8, idx: usize| vld1q_u8(base.add(idx));
        let r0 = t0.add(ab_z_idx::<0, 6>(y3));
        let (l0, l1, l2, l3) = (ld(r0, 0), ld(r0, 16), ld(r0, 32), ld(r0, 48));
        let rec = |i: usize| -> *const u8 {
            let a: usize;
            core::arch::asm!("add {a}, {b}, {i}", a = lateout(reg) a, b = in(reg) z.add(AB_Z_REC), i = in(reg) i,
                options(pure, nomem, nostack, preserves_flags));
            a as *const u8
        };
        let a = rec(ab_z_idx::<2, 7>(y3b));
        let (m0, m2) = (ld(a, 0), ld(a, 16));
        let a = rec(ab_z_idx::<4, 7>(y2));
        let (n0, n1) = (ld(a, 32), ld(a, 48));
        let a = rec(ab_z_idx::<1, 7>(y3));
        let (d01, d23) = (ld(a, 64), ld(a, 80));
        let a = rec(ab_z_idx::<3, 7>(y3b));
        let (e0, e2) = (ld(a, 96), ld(a, 112));
        let g = ld(z.add(AB_Z_ALL), ab_z_idx::<7, 4>(w));
        let k = ld(z.add(AB_Z_L12), ab_z_idx::<6, 4>(y1));
        let f = ld(z.add(AB_Z_SL2), ab_z_idx::<5, 4>(y2));
        let fg = veorq_u8(f, g);
        let p = vreinterpretq_u64_u8(xor3_u8(d01, e0, fg));
        let q = vreinterpretq_u64_u8(xor3_u8(d23, e2, fg));
        let b0 = vreinterpretq_u8_u64(vzip1q_u64(p, p));
        let b1 = vreinterpretq_u8_u64(vzip2q_u64(p, p));
        let b2 = vreinterpretq_u8_u64(vzip1q_u64(q, q));
        let b3 = vreinterpretq_u8_u64(vzip2q_u64(q, q));
        if KF {
            // out_q = l_q ^ M ^ N ^ K ^ b_q: two `eor3` per output.
            return (
                xor3_u8(xor3_u8(l0, m0, n0), k, b0),
                xor3_u8(xor3_u8(l1, m0, n1), k, b1),
                xor3_u8(xor3_u8(l2, m2, n0), k, b2),
                xor3_u8(xor3_u8(l3, m2, n1), k, b3),
            );
        }
        let mk0 = veorq_u8(m0, k);
        let mk2 = veorq_u8(m2, k);
        (
            veorq_u8(xor3_u8(l0, mk0, n0), b0),
            veorq_u8(xor3_u8(l1, mk0, n1), b1),
            veorq_u8(xor3_u8(l2, mk2, n0), b2),
            veorq_u8(xor3_u8(l3, mk2, n1), b3),
        )
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
#[test]
fn ab_superset_words_m1_match_superset_bytes() {
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    for i in 0..20000u64 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let w = match i {
            0 => 0,
            1 => u64::MAX,
            2..=2049 => (i & 0xff) << (8 * ((i >> 8) & 7)),
            _ => x,
        };
        let z = ab_superset_bytes(w).to_le_bytes();
        let (y1, y2, y3, y3b) = ab_superset_words_m1(w);
        let b = |v: u64, j: usize| v.to_le_bytes()[j];
        let got = [b(y3, 0), b(y3, 1), b(y3b, 2), b(y3b, 3), b(y2, 4), b(y2, 5), b(y1, 6), b(w, 7)];
        assert_eq!(got, z, "w={w:016x}");
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
#[test]
fn ab_superset_bytes_matches_definition() {
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    for i in 0..20000u64 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let w = match i {
            0 => 0,
            1 => u64::MAX,
            2..=2049 => (i & 0xff) << (8 * ((i >> 8) & 7)),
            _ => x,
        };
        let b = w.to_le_bytes();
        let want: [u8; 8] = core::array::from_fn(|t| {
            (0..8).filter(|&j| j & t == t).fold(0u8, |acc, j| acc ^ b[j])
        });
        assert_eq!(ab_superset_bytes(w).to_le_bytes(), want, "w={w:016x}");
    }
}

// ---------------------------------------------------------------------------
// Half-pair z form ("HZ") of the table apply: 24 loads per word instead of 32,
// no half-swapped image, for the narrow efficiency cores.
//
// Bytes 2c and 2c+1 of a word (u, v) reach output block n = 2q + h through
// R_{n^2c}(u) + R_{n^2c^1}(v) = R_{n^2c}(u ^ v) + sigma_{q^c}(v), where
// sigma_q(x) = (lane q of row x, low half) ^ (high half): the z-basis
// identity on the half bit alone. The first term is an even-byte row (lane
// permutation only, never a half swap); the second is one 8-byte value per
// lane, the same in both halves. A 32-byte table per image holds
// [sigma_0 sigma_1 | sigma_2 sigma_3]; the four pairs' sigma rows are summed
// two lanes per register (two EXTs for the odd pairs) and spread with one zip
// per lane. Tables: the image itself plus 8 KiB, against the plain apply's
// image plus its 16 KiB half-swapped twin.
// ---------------------------------------------------------------------------

/// Row-body variant bit: the table apply in the half-pair z form (both
/// images), out of [`ab_hz_tables`]. Exclusive with [`ABV_KARA`].
#[cfg(target_arch = "aarch64")]
pub(crate) const ABV_HZ: u32 = 256;

/// The static-B window rows' form on the efficiency-core helper threads
/// (which keep the plain apply instead of the Karatsuba / z forms, see
/// [`ab_kara_note_worker`]): the half-pair z apply for both images, the
/// partial path without the half-swapped image, one function body per window.
/// Measured on an M5 Pro, synthetic static-B rows, lower quartile per row:
/// 84.9 against 116.3 ns on a 64 KiB-L1D core (background QoS), 32.8 against
/// 41.3 ns on a P-core.
#[cfg(target_arch = "aarch64")]
pub(crate) const AB_HZ_VARIANT: u32 = ABV_HZ | ABV_NOSW | ABV_INLINE_WINDOW;

/// Kill switch: exactly `FLOCK_NO_ZC_AB_HZ=1` keeps the plain apply on the
/// helper threads. Same bytes either way.
#[cfg(target_arch = "aarch64")]
pub(crate) const ENV_NO_ZC_AB_HZ: &str = "FLOCK_NO_ZC_AB_HZ";

/// Whether this thread's static-B rows take [`AB_HZ_VARIANT`]: a thread that
/// keeps the plain apply (helper pool or `FLOCK_NO_ZC_AB_KARA=1`), with the
/// half-pair form not switched off.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) fn ab_hz_helper_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var_os(ENV_NO_ZC_AB_HZ).is_none_or(|v| v != *"1")
    });
    *ON && !ab_kara_enabled()
}

/// Kill switch for [`ab_mixed_hz_enabled`]: any value of
/// `FLOCK_NO_ZC_AB_MIXED_HZ` keeps the plain apply in the two mixed
/// const-one rows of a thread that keeps the plain apply. Read once per
/// process. The rows written are byte-identical.
#[cfg(target_arch = "aarch64")]
pub(crate) const ENV_NO_ZC_AB_MIXED_HZ: &str = "FLOCK_NO_ZC_AB_MIXED_HZ";

/// Whether this thread's mixed const-one rows apply the table in the
/// half-pair z form (24 loads per word instead of 32): a thread that keeps
/// the plain apply and runs its static-B rows in [`AB_HZ_VARIANT`].
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) fn ab_mixed_hz_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os(ENV_NO_ZC_AB_MIXED_HZ).is_none()) && ab_hz_helper_enabled()
}

/// Test access: one mixed const-one row (`window 0, b_med 2` mask `0x03`
/// when `w0`, else `window 1, b_med 13` mask `0xf0`) through the plain apply
/// and through the half-pair z apply.
#[cfg(all(test, target_arch = "aarch64"))]
pub(crate) fn ab_mixed_row_plain_and_hz(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    byte_base_b: usize,
    w0: bool,
) -> ([u8; 64], [u8; 64]) {
    let mut out = ([0u8; 64], [0u8; 64]);
    if w0 {
        shift_reduce_inner_mixed_const_b_h4::<0x03, false, 0x03, 0>(a_packed, b_packed, inv_table, byte_base_b, None, &mut out.0, false);
        shift_reduce_inner_mixed_const_b_h4::<0x03, false, 0x03, { ABV_HZ }>(a_packed, b_packed, inv_table, byte_base_b, None, &mut out.1, false);
    } else {
        shift_reduce_inner_mixed_const_b_h4::<0xf0, false, 0, 0>(a_packed, b_packed, inv_table, byte_base_b, None, &mut out.0, false);
        shift_reduce_inner_mixed_const_b_h4::<0xf0, false, 0, { ABV_HZ }>(a_packed, b_packed, inv_table, byte_base_b, None, &mut out.1, false);
    }
    out
}

/// Test access: one mixed const-one row (as [`ab_mixed_row_plain_and_hz`])
/// through the incumbent production z form (`AB_Z_VARIANT | ABV_KARA_XOR16`)
/// and through the same form with [`ABV_KARA_SLOT`] and [`ABV_Z_MT`]. (The
/// plain form of window 0's row drops A K0/K1 bytes 5..7, which the z forms
/// keep, so it is compared only on rows of the ranked shape.)
#[cfg(all(test, target_arch = "aarch64"))]
pub(crate) fn ab_mixed_row_incumbent_and_slot_mt(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    byte_base_b: usize,
    w0: bool,
) -> ([u8; 64], [u8; 64]) {
    const V0: u32 = AB_Z_VARIANT | ABV_KARA_XOR16;
    const V: u32 = V0 | ABV_KARA_SLOT | ABV_Z_MT;
    let mut out = ([0u8; 64], [0u8; 64]);
    if w0 {
        shift_reduce_inner_mixed_const_b_h4::<0x03, false, 0x03, V0>(a_packed, b_packed, inv_table, byte_base_b, None, &mut out.0, false);
        shift_reduce_inner_mixed_const_b_h4::<0x03, false, 0x03, V>(a_packed, b_packed, inv_table, byte_base_b, None, &mut out.1, false);
    } else {
        shift_reduce_inner_mixed_const_b_h4::<0xf0, false, 0, V0>(a_packed, b_packed, inv_table, byte_base_b, None, &mut out.0, false);
        shift_reduce_inner_mixed_const_b_h4::<0xf0, false, 0, V>(a_packed, b_packed, inv_table, byte_base_b, None, &mut out.1, false);
    }
    out
}

/// Bytes of one image's sigma table (256 x 32).
#[cfg(target_arch = "aarch64")]
pub(crate) const AB_HZ_IMAGE_BYTES: usize = 256 * 32;

/// Sigma tables of the plain image followed by the `x^4`-scaled image's,
/// 128-byte aligned, built once per process.
#[cfg(target_arch = "aarch64")]
pub(crate) fn ab_hz_tables(inv_table: &InvNttTableByteSingleGf8) -> *const u8 {
    struct Tables {
        storage: Vec<u8>,
        offset: usize,
    }
    static TABLES: std::sync::OnceLock<Tables> = std::sync::OnceLock::new();
    let tables = TABLES.get_or_init(|| {
        let mut storage = vec![0u8; 2 * AB_HZ_IMAGE_BYTES + 127];
        let offset = storage.as_ptr().align_offset(128);
        assert!(offset < 128);
        for (image, src) in [inv_table.data_ptr(), inv_table.scaled_x4_data_ptr()]
            .into_iter()
            .enumerate()
        {
            // SAFETY: each image holds 256 rows of 64 bytes.
            let rows = unsafe { core::slice::from_raw_parts(src, 256 * 64) };
            let dst = &mut storage[offset + image * AB_HZ_IMAGE_BYTES..][..AB_HZ_IMAGE_BYTES];
            for x in 0..256 {
                for q in 0..4 {
                    for i in 0..8 {
                        dst[x * 32 + q * 8 + i] = rows[x * 64 + q * 16 + i] ^ rows[x * 64 + q * 16 + 8 + i];
                    }
                }
            }
        }
        Tables { storage, offset }
    });
    // SAFETY: `offset` is inside the allocation, which lives for the process.
    unsafe { tables.storage.as_ptr().add(tables.offset) }
}

/// [`apply_word_into_4_regs`] in the half-pair z form: `t0` is the table
/// image (64-byte rows), `sg` its [`AB_HZ_IMAGE_BYTES`] sigma table.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn apply_word_hz(
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
        // Even bytes: W_2c ^ W_2c+1.
        let y = w ^ (w >> 8);
        let ld = |base: *const u8, off: usize| vld1q_u8(base.add(off));
        // An opaque base + offset add, so a row's loads pair into `ldp`s.
        let at = |base: *const u8, i: usize| -> *const u8 {
            let a: usize;
            core::arch::asm!("add {a}, {b}, {i}", a = lateout(reg) a, b = in(reg) base, i = in(reg) i,
                options(pure, nomem, nostack, preserves_flags));
            a as *const u8
        };
        let r0 = at(t0, ab_z_idx::<0, 6>(y));
        let r1 = at(t0, ab_z_idx::<2, 6>(y));
        let r2 = at(t0, ab_z_idx::<4, 6>(y));
        let r3 = at(t0, ab_z_idx::<6, 6>(y));
        let s0 = at(sg, ab_z_idx::<1, 5>(w));
        let s1 = at(sg, ab_z_idx::<3, 5>(w));
        let s2 = at(sg, ab_z_idx::<5, 5>(w));
        let s3 = at(sg, ab_z_idx::<7, 5>(w));
        let (p0, q0) = (ld(s0, 0), ld(s0, 16));
        let (p1, q1) = (ld(s1, 0), ld(s1, 16));
        let (p2, q2) = (ld(s2, 0), ld(s2, 16));
        let (p3, q3) = (ld(s3, 0), ld(s3, 16));
        // Pair c adds [sigma_{0^c} sigma_{1^c}] to X and [sigma_{2^c} sigma_{3^c}] to Y.
        let xb = veorq_u8(p1, q3);
        let yb = veorq_u8(q1, p3);
        let x = vreinterpretq_u64_u8(xor3_u8(p0, q2, vextq_u8::<8>(xb, xb)));
        let yv = vreinterpretq_u64_u8(xor3_u8(q0, p2, vextq_u8::<8>(yb, yb)));
        let t0d = vreinterpretq_u8_u64(vzip1q_u64(x, x));
        let t1d = vreinterpretq_u8_u64(vzip2q_u64(x, x));
        let t2d = vreinterpretq_u8_u64(vzip1q_u64(yv, yv));
        let t3d = vreinterpretq_u8_u64(vzip2q_u64(yv, yv));
        // Lane q of pair c's row is row lane q ^ c.
        (
            xor3_u8(xor3_u8(ld(r0, 0), ld(r1, 16), ld(r2, 32)), ld(r3, 48), t0d),
            xor3_u8(xor3_u8(ld(r0, 16), ld(r1, 0), ld(r2, 48)), ld(r3, 32), t1d),
            xor3_u8(xor3_u8(ld(r0, 32), ld(r1, 48), ld(r2, 0)), ld(r3, 16), t2d),
            xor3_u8(xor3_u8(ld(r0, 48), ld(r1, 32), ld(r2, 16)), ld(r3, 0), t3d),
        )
    }
}

/// [`apply_word_into_4_regs`] in the form selected by `V`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn apply_word_into_4_regs_v<const V: u32, const SCALED: bool>(
    table_base: *const u8,
    half_swapped_table_base: *const u8,
    kara: *const u8,
    word: u64,
) -> (
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
    core::arch::aarch64::uint8x16_t,
) {
    unsafe {
        if V & ABV_Z != 0 && (!SCALED || V & ABV_Z_SCALED != 0) {
            let z = if SCALED {
                kara.add(AB_KARA_IMAGE_BYTES + AB_Z_IMAGE_BYTES)
            } else {
                kara.add(2 * AB_KARA_IMAGE_BYTES)
            };
            if V & ABV_Z_REC != 0 && V & ABV_Z_PACK != 0 {
                apply_word_z::<true, true, false>(table_base, z, word)
            } else if V & ABV_Z_REC != 0 && V & ABV_Z_MT != 0 {
                apply_word_z_mt(table_base, z, word)
            } else if V & ABV_Z_REC != 0 && V & ABV_Z_M1 != 0 {
                if V & ABV_ZK != 0 {
                    apply_word_z_m1::<true>(table_base, z, word)
                } else {
                    apply_word_z_m1::<false>(table_base, z, word)
                }
            } else if V & ABV_Z_REC != 0 && V & ABV_Z_OUT3 != 0 {
                apply_word_z::<false, true, true>(table_base, z, word)
            } else if V & ABV_Z_REC != 0 {
                apply_word_z::<false, true, false>(table_base, z, word)
            } else if V & ABV_Z_PACK != 0 {
                apply_word_z::<true, false, false>(table_base, z, word)
            } else {
                apply_word_z::<false, false, false>(table_base, z, word)
            }
        } else if V & ABV_KARA != 0 {
            apply_word_kara::<V, SCALED>(kara, word)
        } else if V & ABV_HZ != 0 {
            // `kara` is [`ab_hz_tables`] in this form.
            // Scaled callers pass the Karatsuba convention's `base + AB_KARA_IMAGE_BYTES`.
            let sg = if SCALED {
                kara.wrapping_sub(AB_KARA_IMAGE_BYTES).wrapping_add(AB_HZ_IMAGE_BYTES)
            } else {
                kara
            };
            apply_word_hz(table_base, sg, word)
        } else {
            apply_word_into_4_regs(table_base, half_swapped_table_base, word)
        }
    }
}

/// Test access: one word through the plain apply and through the Karatsuba
/// apply of the plain (`scaled = false`) or `x^4`-scaled image.
#[cfg(all(test, target_arch = "aarch64"))]
pub(crate) fn ab_apply_word_forms<const V: u32>(
    inv_table: &InvNttTableByteSingleGf8,
    scaled: bool,
    word: u64,
) -> ([u8; 64], [u8; 64]) {
    use core::arch::aarch64::*;
    let kara = ab_kara_tables(inv_table);
    let (t, s, k) = if scaled {
        (
            inv_table.scaled_x4_data_ptr(),
            inv_table.scaled_x4_half_swapped_data_ptr(),
            kara.wrapping_add(AB_KARA_IMAGE_BYTES),
        )
    } else {
        (inv_table.data_ptr(), inv_table.half_swapped_data_ptr(), kara)
    };
    let mut out = ([0u8; 64], [0u8; 64]);
    // SAFETY: both applies read inside their table images only.
    unsafe {
        let p = apply_word_into_4_regs(t, s, word);
        let q = if scaled { apply_word_kara::<V, true>(k, word) } else { apply_word_kara::<V, false>(k, word) };
        for (dst, v) in [(&mut out.0, p), (&mut out.1, q)] {
            vst1q_u8(dst.as_mut_ptr(), v.0);
            vst1q_u8(dst.as_mut_ptr().add(16), v.1);
            vst1q_u8(dst.as_mut_ptr().add(32), v.2);
            vst1q_u8(dst.as_mut_ptr().add(48), v.3);
        }
    }
    out
}

/// Test access: one word through the half-pair z apply of the plain
/// (`scaled = false`) or `x^4`-scaled image.
#[cfg(all(test, target_arch = "aarch64"))]
pub(crate) fn ab_apply_word_hz_form(inv_table: &InvNttTableByteSingleGf8, scaled: bool, word: u64) -> [u8; 64] {
    use core::arch::aarch64::*;
    let hz = ab_hz_tables(inv_table);
    let mut out = [0u8; 64];
    // SAFETY: the apply reads inside the table image and its sigma table only.
    unsafe {
        let v = if scaled {
            apply_word_hz(inv_table.scaled_x4_data_ptr(), hz.add(AB_HZ_IMAGE_BYTES), word)
        } else {
            apply_word_hz(inv_table.data_ptr(), hz, word)
        };
        vst1q_u8(out.as_mut_ptr(), v.0);
        vst1q_u8(out.as_mut_ptr().add(16), v.1);
        vst1q_u8(out.as_mut_ptr().add(32), v.2);
        vst1q_u8(out.as_mut_ptr().add(48), v.3);
    }
    out
}

/// Pre-resolved state shared by every static-B kernel call in one AB
/// precompute. The references are process-stable because the partials live in
/// `OnceLock`s; carrying them here removes both hot-path acquires from each
/// `(window, b_med)` call, including the ranked static-A arm.
#[derive(Clone, Copy)]
pub(crate) enum StaticBContext {
    Prepared {
        partials: &'static [[u8; 64]; 248],
        static_a_k1: &'static [u8; 64],
    },
    /// Exact same-binary control: retain both historical per-call OnceLock
    /// queries while leaving the generated static-B kernel enabled.
    LegacyPerCall,
}

include!("aarch64_bstatic_gen.rs");
include!("aarch64_bstatic_horner_gen.rs");

fn bstatic_legacy_enabled() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("FLOCK_ZC_BSTATIC_LEGACY").is_some())
}

#[inline]
pub(crate) fn prepare_static_b_context(
    inv_table: &InvNttTableByteSingleGf8,
    blake3_static_layout: bool,
) -> Option<StaticBContext> {
    if std::env::var_os("FLOCK_ZC_BSTATIC_CONTEXT_LEGACY").is_some() {
        return blake3_static_layout.then_some(StaticBContext::LegacyPerCall);
    }
    prepare_static_b_context_with_policy(
        inv_table,
        blake3_static_layout,
        bstatic_legacy_enabled(),
        false,
    )
}

/// Policy-injected form used by the gate/oracle tests without mutating the
/// process environment or the cached legacy switch.
#[inline]
pub(crate) fn prepare_static_b_context_with_policy(
    inv_table: &InvNttTableByteSingleGf8,
    blake3_static_layout: bool,
    legacy_enabled: bool,
    context_legacy: bool,
) -> Option<StaticBContext> {
    if !blake3_static_layout {
        None
    } else if context_legacy {
        Some(StaticBContext::LegacyPerCall)
    } else if legacy_enabled {
        None
    } else {
        Some(StaticBContext::Prepared {
            partials: bstatic_partials(inv_table),
            static_a_k1: static_a_k1_partial(inv_table),
        })
    }
}

/// Window `within_hash_outer`'s static-B Horner rows `[lo, hi)` and the
/// partials to run them with, when [`bstatic_horner_window_rows`] may replace
/// the per-row dispatch for them: the FAST policy is on, the static-B context
/// is enabled, and the ranked BLAKE3 row counts place every sniffed row
/// (all-ones, single-K0, const-one masks) outside `[lo, hi)`, so each row in
/// the run would reach the same Horner body through
/// [`shift_reduce_inner_ab_fused_neon_checked_with_fast_policy`]. The caller
/// must also have the BLAKE3 static layout. `None` keeps the per-row loop.
#[cfg(target_arch = "aarch64")]
#[inline]
pub(crate) fn bstatic_horner_window_run<const FAST_POLICY: u8>(
    static_b_context: Option<StaticBContext>,
    inv_table: &InvNttTableByteSingleGf8,
    within_hash_outer: usize,
    n_b_med: usize,
) -> Option<(usize, usize, &'static [[u8; 64]; 248])> {
    let context = static_b_context?;
    if !fast_shift_reduce_with_policy::<FAST_POLICY>()
        || (matches!(context, StaticBContext::LegacyPerCall) && bstatic_legacy_enabled())
    {
        return None;
    }
    let (lo, hi) = match (within_hash_outer, n_b_med) {
        (0, 16) => BSTATIC_HORNER_WINDOW_ROWS[0],
        (1, 15) => BSTATIC_HORNER_WINDOW_ROWS[1],
        _ => return None,
    };
    let partials = match context {
        StaticBContext::Prepared { partials, .. } => partials,
        StaticBContext::LegacyPerCall => bstatic_partials(inv_table),
    };
    Some((lo, hi, partials))
}

/// Checked static-structure dispatcher. Sniffs the 8 b-words (8 scalar loads
/// vs 512 table loads) and routes to a fast path when the block matches the
/// BLAKE3 static structure; generic otherwise. Output is bit-identical for
/// any witness.
///
/// `bstatic_w` is the BLAKE3 outer-window index (0 or 1) for the byte-level
/// static-B path, or `usize::MAX` to disable it. Block-level sniffs run first.
#[cfg(all(test, target_arch = "aarch64"))]
#[inline(always)]
pub(crate) fn shift_reduce_inner_ab_fused_neon_checked(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; 64],
    check_all_ones: bool,
    check_single_k0: bool,
    const_one_mask: u8,
    bstatic_w: usize,
    static_b_context: Option<StaticBContext>,
    nt_store: bool,
) {
    shift_reduce_inner_ab_fused_neon_checked_with_fast_policy::<{ super::AB_FAST_POLICY_PROCESS }>(
        a_packed,
        b_packed,
        inv_table,
        chunk_byte_base,
        b_med,
        out,
        check_all_ones,
        check_single_k0,
        const_one_mask,
        bstatic_w,
        static_b_context,
        nt_store,
    );
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn fast_shift_reduce_with_policy<const FAST_POLICY: u8>() -> bool {
    match FAST_POLICY {
        super::AB_FAST_POLICY_PROCESS => fast_shift_reduce_enabled(),
        super::AB_FAST_POLICY_FORCE_FAST => true,
        _ => unreachable!("unsupported AB fast-policy mode"),
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) fn shift_reduce_inner_ab_fused_neon_checked_with_fast_policy<const FAST_POLICY: u8>(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; 64],
    check_all_ones: bool,
    check_single_k0: bool,
    const_one_mask: u8,
    bstatic_w: usize,
    static_b_context: Option<StaticBContext>,
    nt_store: bool,
) {
    let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;
    let bw = |k: usize| -> u64 {
        u64::from_le(unsafe {
            core::ptr::read_unaligned(b_packed.as_ptr().add(byte_base_b + k * 8).cast::<u64>())
        })
    };
    if check_all_ones {
        let and_all = bw(0) & bw(1) & bw(2) & bw(3) & bw(4) & bw(5) & bw(6) & bw(7);
        if and_all == u64::MAX {
            shift_reduce_inner_a_only_const_b(a_packed, inv_table, byte_base_b, out, nt_store);
            return;
        }
    }
    if check_single_k0 {
        let or_tail = bw(1) | bw(2) | bw(3) | bw(4) | bw(5) | bw(6) | bw(7);
        if or_tail == 0 {
            // Ranked blk 30: the surviving K0 word is the static constant, so
            // its transform comes from the precomputed partial instead of
            // eight table-row gathers. Exact-match guard keeps the dispatch
            // bit-identical for any witness.
            if bstatic_w == 1
                && b_med == 14
                && bw(0) == 0x0001_ffff_ffff_ffff
                && let Some(context) = static_b_context
            {
                let partials = match context {
                    StaticBContext::Prepared { partials, .. } => partials,
                    StaticBContext::LegacyPerCall => bstatic_partials(inv_table),
                };
                let a_word = u64::from_le(unsafe {
                    core::ptr::read_unaligned(a_packed.as_ptr().add(byte_base_b).cast::<u64>())
                });
                if a_word & 0xff00_0000_0000_0000 == 0 {
                    shift_reduce_inner_single_k0_static_b::<true>(
                        a_word,
                        inv_table,
                        &partials[30 * 8],
                        out,
                        nt_store,
                    );
                } else {
                    shift_reduce_inner_single_k0_static_b::<false>(
                        a_word,
                        inv_table,
                        &partials[30 * 8],
                        out,
                        nt_store,
                    );
                }
                return;
            }
            shift_reduce_inner_single_k0(a_packed, b_packed, inv_table, byte_base_b, out, nt_store);
            return;
        }
    }
    match const_one_mask {
        0 => {}
        0x03 if bw(0) & bw(1) == u64::MAX => {
            if fast_shift_reduce_with_policy::<FAST_POLICY>() {
                // Ranked (window 0, b_med 2): B K0..1 are static ones and the
                // A K0/K1 words' top three bytes are structurally zero
                // (2048/2048 measured on the scored distribution), so the h4
                // kernel's K0 and K1 a-side transforms can drop bytes 5-7's
                // twelve table loads and six XORs. This fires on every scored
                // block. The guard is witness-safe: any block whose K0/K1 top
                // bytes are nonzero takes the legacy whole-word static-A arm
                // (retained as a documented fallback) or the generic h4 arm.
                let a_k0 = u64::from_le(unsafe {
                    core::ptr::read_unaligned(a_packed.as_ptr().add(byte_base_b).cast::<u64>())
                });
                let a_k1 = u64::from_le(unsafe {
                    core::ptr::read_unaligned(
                        a_packed.as_ptr().add(byte_base_b + N_CHUNKS).cast::<u64>(),
                    )
                });
                if (a_k0 | a_k1) & 0xffff_ff00_0000_0000 == 0 {
                    if ab_kara_enabled() && ab_kara_unscaled_enabled() && ab_z_enabled() && ab_kara_xor16_enabled()
                        && ab_kara_slot_enabled() && ab_z_mt_enabled()
                    {
                        shift_reduce_inner_mixed_const_b_h4::<0x03, false, 0x03, { AB_Z_VARIANT | ABV_KARA_XOR16 | ABV_KARA_SLOT | ABV_Z_MT }>(
                            a_packed,
                            b_packed,
                            inv_table,
                            byte_base_b,
                            None,
                            out,
                            nt_store,
                        );
                    } else if ab_kara_enabled() && ab_kara_unscaled_enabled() && ab_z_enabled() && ab_kara_xor16_enabled() {
                        shift_reduce_inner_mixed_const_b_h4::<0x03, false, 0x03, { AB_Z_VARIANT | ABV_KARA_XOR16 }>(
                            a_packed,
                            b_packed,
                            inv_table,
                            byte_base_b,
                            None,
                            out,
                            nt_store,
                        );
                    } else if ab_kara_enabled() && ab_kara_unscaled_enabled() && ab_z_enabled() {
                        shift_reduce_inner_mixed_const_b_h4::<0x03, false, 0x03, { AB_Z_VARIANT }>(
                            a_packed,
                            b_packed,
                            inv_table,
                            byte_base_b,
                            None,
                            out,
                            nt_store,
                        );
                    } else if ab_kara_enabled() && ab_kara_unscaled_enabled() {
                        shift_reduce_inner_mixed_const_b_h4::<0x03, false, 0x03, { ABV_KARA | ABV_KARA_UNSCALED }>(
                            a_packed,
                            b_packed,
                            inv_table,
                            byte_base_b,
                            None,
                            out,
                            nt_store,
                        );
                    } else if ab_kara_enabled() {
                        shift_reduce_inner_mixed_const_b_h4::<0x03, false, 0x03, { ABV_KARA }>(
                            a_packed,
                            b_packed,
                            inv_table,
                            byte_base_b,
                            None,
                            out,
                            nt_store,
                        );
                    } else if ab_mixed_hz_enabled() {
                        shift_reduce_inner_mixed_const_b_h4::<0x03, false, 0x03, { ABV_HZ }>(
                            a_packed,
                            b_packed,
                            inv_table,
                            byte_base_b,
                            None,
                            out,
                            nt_store,
                        );
                    } else {
                        shift_reduce_inner_mixed_const_b_h4::<0x03, false, 0x03, 0>(
                            a_packed,
                            b_packed,
                            inv_table,
                            byte_base_b,
                            None,
                            out,
                            nt_store,
                        );
                    }
                } else {
                    const STATIC_A_K1: u64 = 0x0000_0016_0000_0080;
                    const STATIC_A_K0_TOP3_MASK: u64 = 0xffff_ff00_0000_0000;
                    if a_k1 == STATIC_A_K1 && a_k0 & STATIC_A_K0_TOP3_MASK == 0 {
                        let static_a_k1 = match static_b_context {
                            Some(StaticBContext::Prepared { static_a_k1, .. }) => static_a_k1,
                            Some(StaticBContext::LegacyPerCall) | None => {
                                static_a_k1_partial(inv_table)
                            }
                        };
                        shift_reduce_inner_mixed_const_b_h4::<0x03, true, 0, 0>(
                            a_packed,
                            b_packed,
                            inv_table,
                            byte_base_b,
                            Some(static_a_k1),
                            out,
                            nt_store,
                        );
                    } else {
                        shift_reduce_inner_mixed_const_b_h4::<0x03, false, 0, 0>(
                            a_packed,
                            b_packed,
                            inv_table,
                            byte_base_b,
                            None,
                            out,
                            nt_store,
                        );
                    }
                }
            } else {
                shift_reduce_inner_mixed_const_b::<0x03>(
                    a_packed,
                    b_packed,
                    inv_table,
                    byte_base_b,
                    out,
                    nt_store,
                );
            }
            return;
        }
        0xf0 if bw(4) & bw(5) & bw(6) & bw(7) == u64::MAX => {
            if fast_shift_reduce_with_policy::<FAST_POLICY>() {
                if ab_kara_enabled() && ab_kara_unscaled_enabled() && ab_z_enabled() && ab_kara_xor16_enabled()
                    && ab_kara_slot_enabled() && ab_z_mt_enabled()
                {
                    shift_reduce_inner_mixed_const_b_h4::<0xf0, false, 0, { AB_Z_VARIANT | ABV_KARA_XOR16 | ABV_KARA_SLOT | ABV_Z_MT }>(
                        a_packed,
                        b_packed,
                        inv_table,
                        byte_base_b,
                        None,
                        out,
                        nt_store,
                    );
                } else if ab_kara_enabled() && ab_kara_unscaled_enabled() && ab_z_enabled() && ab_kara_xor16_enabled() {
                    shift_reduce_inner_mixed_const_b_h4::<0xf0, false, 0, { AB_Z_VARIANT | ABV_KARA_XOR16 }>(
                        a_packed,
                        b_packed,
                        inv_table,
                        byte_base_b,
                        None,
                        out,
                        nt_store,
                    );
                } else if ab_kara_enabled() && ab_kara_unscaled_enabled() && ab_z_enabled() {
                    shift_reduce_inner_mixed_const_b_h4::<0xf0, false, 0, { AB_Z_VARIANT }>(
                        a_packed,
                        b_packed,
                        inv_table,
                        byte_base_b,
                        None,
                        out,
                        nt_store,
                    );
                } else if ab_kara_enabled() && ab_kara_unscaled_enabled() {
                    shift_reduce_inner_mixed_const_b_h4::<0xf0, false, 0, { ABV_KARA | ABV_KARA_UNSCALED }>(
                        a_packed,
                        b_packed,
                        inv_table,
                        byte_base_b,
                        None,
                        out,
                        nt_store,
                    );
                } else if ab_kara_enabled() {
                    shift_reduce_inner_mixed_const_b_h4::<0xf0, false, 0, { ABV_KARA }>(
                        a_packed,
                        b_packed,
                        inv_table,
                        byte_base_b,
                        None,
                        out,
                        nt_store,
                    );
                } else if ab_mixed_hz_enabled() {
                    shift_reduce_inner_mixed_const_b_h4::<0xf0, false, 0, { ABV_HZ }>(
                        a_packed,
                        b_packed,
                        inv_table,
                        byte_base_b,
                        None,
                        out,
                        nt_store,
                    );
                } else {
                    shift_reduce_inner_mixed_const_b_h4::<0xf0, false, 0, 0>(
                        a_packed,
                        b_packed,
                        inv_table,
                        byte_base_b,
                        None,
                        out,
                        nt_store,
                    );
                }
            } else {
                shift_reduce_inner_mixed_const_b::<0xf0>(
                    a_packed,
                    b_packed,
                    inv_table,
                    byte_base_b,
                    out,
                    nt_store,
                );
            }
            return;
        }
        0x03 | 0xf0 => {}
        _ => unreachable!("unsupported static-one mask"),
    }
    if bstatic_w <= 1
        && let Some(context) = static_b_context
    {
        let enabled =
            !matches!(context, StaticBContext::LegacyPerCall) || !bstatic_legacy_enabled();
        // `FAST` is a const generic so the per-K table choice, the `x^2`
        // constant multiply and the residual shift all fold away at
        // monomorphization; the kill switch picks a whole kernel, never a
        // branch inside one.
        // FAST rows take the Horner-layout static-B kernel: the same plans
        // and partials as `shift_reduce_inner_ab_bstatic`, but one guard per
        // row and a straight-line body, so the skipped table loads are
        // actually saved (measured ~-10 P-cycles/row against the per-K
        // guarded kernel). Arms without a plan return `false` and take h4.
        let handled = enabled
            && if fast_shift_reduce_with_policy::<FAST_POLICY>() {
                let partials = match context {
                    StaticBContext::Prepared { partials, .. } => partials,
                    StaticBContext::LegacyPerCall => bstatic_partials(inv_table),
                };
                shift_reduce_inner_ab_bstatic_horner(
                    a_packed,
                    b_packed,
                    inv_table,
                    chunk_byte_base,
                    b_med,
                    bstatic_w,
                    partials,
                    out,
                    nt_store,
                )
            } else {
                // `FLOCK_NO_FAST_SHIFT_REDUCE`: every row takes h4 (the same
                // bytes).
                false
            };
        if handled {
            return;
        }
    }
    if fast_shift_reduce_with_policy::<FAST_POLICY>() {
        if ab_kara_enabled() {
            shift_reduce_inner_ab_fused_neon_h4::<{ ABV_KARA }>(
                a_packed,
                b_packed,
                inv_table,
                chunk_byte_base,
                b_med,
                out,
                nt_store,
            );
        } else {
            shift_reduce_inner_ab_fused_neon_h4::<0>(
                a_packed,
                b_packed,
                inv_table,
                chunk_byte_base,
                b_med,
                out,
                nt_store,
            );
        }
    } else {
        shift_reduce_inner_ab_fused_neon(
            a_packed,
            b_packed,
            inv_table,
            chunk_byte_base,
            b_med,
            out,
            nt_store,
        );
    }
}

/// Kill switch for [`ab_mixed_row_direct`]: any value of
/// `FLOCK_NO_ZC_AB_MIXED_ROW_DIRECT` sends the two mixed const-one rows of a
/// chunk through [`shift_reduce_inner_ab_fused_neon_checked_with_fast_policy`]
/// again. Read once per process. The rows written are byte-identical.
#[cfg(target_arch = "aarch64")]
pub(crate) fn ab_mixed_row_direct_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_ZC_AB_MIXED_ROW_DIRECT").is_none())
}

/// Whether the process-wide switches make the dispatcher's mixed const-one
/// arms pick the kernels of [`ab_mixed_row_guarded`] on a thread with the
/// Karatsuba apply: the fast form on (asked of both fast policies), the
/// unscaled Karatsuba z form with the twelve-XOR recombination, and not both
/// of `ABV_KARA_SLOT` and `ABV_Z_MT`. Resolved once per process.
#[cfg(target_arch = "aarch64")]
fn ab_mixed_row_direct_route() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        ab_mixed_row_direct_enabled()
            && fast_shift_reduce_enabled()
            && ab_kara_unscaled_enabled()
            && ab_z_enabled()
            && ab_kara_xor16_enabled()
            && !(ab_kara_slot_enabled() && ab_z_mt_enabled())
    })
}

/// One of the two mixed const-one rows of the BLAKE3 static layout (`w0`:
/// window 0, `b_med` 2, mask `0x03`; otherwise window 1, `b_med`
/// `n_b_med - 2`, mask `0xf0`) through the kernel the dispatcher picks for
/// it, without the dispatcher. Returns `false` with `out` untouched whenever
/// the dispatcher would pick another kernel (a thread that keeps the plain
/// apply, checked first; the switch or another process-wide form; a failed
/// guard); the caller then runs the dispatcher. As for the dispatcher, the
/// row's 64 bytes must be in bounds of both packed slices.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ab_mixed_row_direct(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    w0: bool,
    out: &mut [u8; 64],
    nt_store: bool,
) -> bool {
    ab_kara_enabled()
        && ab_mixed_row_direct_route()
        && ab_mixed_row_guarded(a_packed, b_packed, inv_table, chunk_byte_base, b_med, w0, out, nt_store)
}

/// The guards and the kernel call of [`ab_mixed_row_direct`]: the
/// dispatcher's guards for the row's mask (B K0 and K1 all ones and the top
/// three bytes of A K0 and K1 zero for `0x03`; B K4..K7 all ones for `0xf0`),
/// then the `AB_Z_VARIANT | ABV_KARA_XOR16` kernel its mask arm calls. Returns
/// `false` with `out` untouched when a guard fails.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn ab_mixed_row_guarded(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    w0: bool,
    out: &mut [u8; 64],
    nt_store: bool,
) -> bool {
    const V0: u32 = AB_Z_VARIANT | ABV_KARA_XOR16;
    // The production mixed rows: the H2 form with the fused products. Either
    // kill switch (`FLOCK_NO_ZC_AB_MIXED_H2`, `FLOCK_NO_ZC_AB_MIXED_FUSE`)
    // takes the incumbent form `V0`; the unfused H2 instance is kept for
    // the tests only, so the binary carries no extra mixed-row instance.
    const VF: u32 = AB_MIXED_H2_VARIANT | ABV_FUSE;
    let fuse = ab_mixed_h2_enabled() && ab_mixed_fuse_enabled();
    let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;
    debug_assert!(byte_base_b + N_CHUNKS * 8 <= a_packed.len().min(b_packed.len()));
    let word = |packed: &[u8], k: usize| -> u64 {
        // SAFETY: the caller keeps the row in bounds (the dispatcher reads
        // the same words).
        u64::from_le(unsafe { core::ptr::read_unaligned(packed.as_ptr().add(byte_base_b + k * 8).cast::<u64>()) })
    };
    if w0 {
        if word(b_packed, 0) & word(b_packed, 1) != u64::MAX
            || (word(a_packed, 0) | word(a_packed, 1)) & 0xffff_ff00_0000_0000 != 0
        {
            return false;
        }
        if fuse {
            shift_reduce_inner_mixed_const_b_h4::<0x03, false, 0x03, VF>(
                a_packed, b_packed, inv_table, byte_base_b, None, out, nt_store,
            );
        } else {
            shift_reduce_inner_mixed_const_b_h4::<0x03, false, 0x03, V0>(
                a_packed,
                b_packed,
                inv_table,
                byte_base_b,
                None,
                out,
                nt_store,
            );
        }
    } else {
        if word(b_packed, 4) & word(b_packed, 5) & word(b_packed, 6) & word(b_packed, 7) != u64::MAX {
            return false;
        }
        if fuse {
            shift_reduce_inner_mixed_const_b_h4::<0xf0, false, 0, VF>(
                a_packed, b_packed, inv_table, byte_base_b, None, out, nt_store,
            );
        } else {
            shift_reduce_inner_mixed_const_b_h4::<0xf0, false, 0, V0>(
                a_packed,
                b_packed,
                inv_table,
                byte_base_b,
                None,
                out,
                nt_store,
            );
        }
    }
    true
}

/// The mixed const-one rows' form on a thread with the Karatsuba apply
/// unless `FLOCK_NO_ZC_AB_MIXED_H2` is set: the window rows' production
/// applies and Horner steps ([`ABV_HORNER2`] K1/K0 steps, [`ABV_Z_OUT3`]
/// z-basis outputs, [`ABV_KARA_SLOT`] `x^4` addresses) on top of the
/// incumbent `AB_Z_VARIANT | ABV_KARA_XOR16`. Same lookups, same terms in
/// every lane, same bytes.
#[cfg(target_arch = "aarch64")]
pub(crate) const AB_MIXED_H2_VARIANT: u32 =
    AB_Z_VARIANT | ABV_KARA_XOR16 | ABV_HORNER2 | ABV_Z_OUT3 | ABV_KARA_SLOT;

/// Kill switch for [`AB_MIXED_H2_VARIANT`]: any value of
/// `FLOCK_NO_ZC_AB_MIXED_H2` keeps the incumbent mixed-row form. Read once
/// per process. The rows written are byte-identical.
#[cfg(target_arch = "aarch64")]
pub(crate) fn ab_mixed_h2_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_ZC_AB_MIXED_H2").is_none())
}

/// Kill switch for [`ABV_FUSE`] in the two mixed const-one rows: any value
/// of `FLOCK_NO_ZC_AB_MIXED_FUSE` runs the incumbent `AB_Z_VARIANT |
/// ABV_KARA_XOR16` rows instead (separate products, widened A images for the
/// const-one words; the binary keeps no unfused H2 instance). Read once per
/// process. The rows written are byte-identical.
#[cfg(target_arch = "aarch64")]
pub(crate) fn ab_mixed_fuse_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_ZC_AB_MIXED_FUSE").is_none())
}

/// Deterministic test words (splitmix64).
#[cfg(all(test, target_arch = "aarch64"))]
fn mixed_fuse_words(seed: u64) -> impl FnMut() -> u64 {
    let mut state = seed;
    move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// The fused mixed rows (`AB_MIXED_H2_VARIANT | ABV_FUSE`) write the
/// [`AB_MIXED_H2_VARIANT`] rows' and the incumbent rows' bytes: both rows,
/// both store flavours, random rows with the guard words forced, and zero
/// and all-ones A and B words (all-ones A makes every widened lane maximal).
#[cfg(all(test, target_arch = "aarch64"))]
#[test]
fn ab_mixed_fuse_matches_h2_and_incumbent() {
    use crate::ntt::AdditiveNttGf8;
    const V0: u32 = AB_Z_VARIANT | ABV_KARA_XOR16;
    const VF: u32 = AB_MIXED_H2_VARIANT | ABV_FUSE;
    let table =
        InvNttTableByteSingleGf8::new(&AdditiveNttGf8::new(6, F8::ZERO), &AdditiveNttGf8::new(6, F8(1 << 6)));
    let mut next = mixed_fuse_words(0x6d69_7865_6466_7573);
    for case in 0..2000usize {
        let mut a: Vec<u8> = (0..64).map(|_| next() as u8).collect();
        let mut b: Vec<u8> = (0..64).map(|_| next() as u8).collect();
        match case % 7 {
            3 => a.fill(0),
            4 => a.fill(0xff),
            5 => b.fill(0),
            6 => b.fill(0xff),
            _ => {}
        }
        let w0 = case % 2 == 0;
        if w0 {
            b[..16].fill(0xff);
            for k in 0..2 {
                a[k * 8 + 5..k * 8 + 8].fill(0);
            }
        } else {
            b[32..].fill(0xff);
        }
        for nt in [false, true] {
            let mut inc = [0u8; 64];
            let mut h2 = [0u8; 64];
            let mut got = [0u8; 64];
            if w0 {
                shift_reduce_inner_mixed_const_b_h4::<0x03, false, 0x03, V0>(&a, &b, &table, 0, None, &mut inc, nt);
                shift_reduce_inner_mixed_const_b_h4::<0x03, false, 0x03, AB_MIXED_H2_VARIANT>(&a, &b, &table, 0, None, &mut h2, nt);
                shift_reduce_inner_mixed_const_b_h4::<0x03, false, 0x03, VF>(&a, &b, &table, 0, None, &mut got, nt);
            } else {
                shift_reduce_inner_mixed_const_b_h4::<0xf0, false, 0, V0>(&a, &b, &table, 0, None, &mut inc, nt);
                shift_reduce_inner_mixed_const_b_h4::<0xf0, false, 0, AB_MIXED_H2_VARIANT>(&a, &b, &table, 0, None, &mut h2, nt);
                shift_reduce_inner_mixed_const_b_h4::<0xf0, false, 0, VF>(&a, &b, &table, 0, None, &mut got, nt);
            }
            assert_eq!(h2, inc, "h2 case {case} w0={w0} nt={nt}");
            assert_eq!(got, h2, "fuse case {case} w0={w0} nt={nt}");
        }
    }
}

/// The production route of the mixed rows ([`ab_mixed_row_guarded`], guards
/// included) against the H2 kernel at the ranked row positions of a chunk
/// (window 0 `b_med` 2, window 1 `b_med` 13); the guarded call takes every
/// row (returns `true`).
#[cfg(all(test, target_arch = "aarch64"))]
#[test]
fn ab_mixed_row_guarded_fuse_route_matches_h2() {
    use crate::ntt::AdditiveNttGf8;
    let table =
        InvNttTableByteSingleGf8::new(&AdditiveNttGf8::new(6, F8::ZERO), &AdditiveNttGf8::new(6, F8(1 << 6)));
    let mut next = mixed_fuse_words(0x7275_6f75_7465_6d78);
    const ROW_BYTES: usize = N_CHUNKS * 8;
    for case in 0..64usize {
        let mut a: Vec<u8> = (0..16 * ROW_BYTES).map(|_| next() as u8).collect();
        let mut b: Vec<u8> = (0..16 * ROW_BYTES).map(|_| next() as u8).collect();
        let w0 = case % 2 == 0;
        let b_med = if w0 { 2 } else { 13 };
        let base = b_med * ROW_BYTES;
        if w0 {
            b[base..base + 16].fill(0xff);
            for k in 0..2 {
                a[base + k * 8 + 5..base + k * 8 + 8].fill(0);
            }
        } else {
            b[base + 32..base + 64].fill(0xff);
        }
        for nt in [false, true] {
            let mut want = [0u8; 64];
            let mut got = [0u8; 64];
            if w0 {
                shift_reduce_inner_mixed_const_b_h4::<0x03, false, 0x03, AB_MIXED_H2_VARIANT>(&a, &b, &table, base, None, &mut want, nt);
            } else {
                shift_reduce_inner_mixed_const_b_h4::<0xf0, false, 0, AB_MIXED_H2_VARIANT>(&a, &b, &table, base, None, &mut want, nt);
            }
            assert!(ab_mixed_row_guarded(&a, &b, &table, 0, b_med, w0, &mut got, nt));
            assert_eq!(got, want, "case {case} w0={w0} nt={nt}");
        }
    }
}

/// Alternated single-thread timing of the two mixed rows, H2 against H2 +
/// [`ABV_FUSE`] (`PROBE_ROUNDS` rounds; 64 rows resident in L1, as the
/// producer unit's rows are).
#[cfg(all(test, target_arch = "aarch64"))]
#[test]
#[ignore = "timing probe"]
fn ab_mixed_fuse_timing() {
    use crate::ntt::AdditiveNttGf8;
    const VF: u32 = AB_MIXED_H2_VARIANT | ABV_FUSE;
    #[cfg(target_os = "macos")]
    unsafe {
        unsafe extern "C" {
            fn pthread_set_qos_class_self_np(qos_class: u32, priority: i32) -> i32;
        }
        pthread_set_qos_class_self_np(0x21, 0);
    }
    let table =
        InvNttTableByteSingleGf8::new(&AdditiveNttGf8::new(6, F8::ZERO), &AdditiveNttGf8::new(6, F8(1 << 6)));
    let mut next = mixed_fuse_words(0x7469_6d69_6e67_6d78);
    const ROWS: usize = 64;
    let mut a: Vec<u8> = (0..ROWS * 64).map(|_| next() as u8).collect();
    let mut b: Vec<u8> = (0..ROWS * 64).map(|_| next() as u8).collect();
    for r in 0..ROWS {
        let base = r * 64;
        if r % 2 == 0 {
            b[base..base + 16].fill(0xff);
            for k in 0..2 {
                a[base + k * 8 + 5..base + k * 8 + 8].fill(0);
            }
        } else {
            b[base + 32..base + 64].fill(0xff);
        }
    }
    let rounds: usize = std::env::var("PROBE_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(21);
    const ITERS: usize = 4000;
    let mut out = vec![[0u8; 64]; ROWS];
    let mut times = [Vec::new(), Vec::new()];
    for round in 0..rounds + 1 {
        for k in 0..2usize {
            let form = (k + round) % 2;
            let t0 = std::time::Instant::now();
            for _ in 0..ITERS {
                for r in (0..ROWS).step_by(2) {
                    let (o0, o1) = out.split_at_mut(r + 1);
                    let (o0, o1) = (&mut o0[r], &mut o1[0]);
                    if form == 0 {
                        shift_reduce_inner_mixed_const_b_h4::<0x03, false, 0x03, AB_MIXED_H2_VARIANT>(&a, &b, &table, r * 64, None, o0, true);
                        shift_reduce_inner_mixed_const_b_h4::<0xf0, false, 0, AB_MIXED_H2_VARIANT>(&a, &b, &table, (r + 1) * 64, None, o1, true);
                    } else {
                        shift_reduce_inner_mixed_const_b_h4::<0x03, false, 0x03, VF>(&a, &b, &table, r * 64, None, o0, true);
                        shift_reduce_inner_mixed_const_b_h4::<0xf0, false, 0, VF>(&a, &b, &table, (r + 1) * 64, None, o1, true);
                    }
                }
            }
            let ns = t0.elapsed().as_nanos() as f64 / (ITERS * ROWS) as f64;
            if round > 0 {
                times[form].push(ns);
            }
        }
        std::hint::black_box(&mut out);
    }
    for i in 0..rounds {
        println!("mixed-fuse round={i} h2={:.2} fuse={:.2}", times[0][i], times[1][i]);
    }
}

/// The [`AB_MIXED_H2_VARIANT`] mixed rows write the incumbent form's bytes
/// for both rows (window 0 `b_med` 2, window 1 `b_med` 13), random A and B
/// rows with the guard words forced, zero and all-ones A.
#[cfg(all(test, target_arch = "aarch64"))]
#[test]
fn ab_mixed_h2_matches_incumbent() {
    use crate::ntt::AdditiveNttGf8;
    const V0: u32 = AB_Z_VARIANT | ABV_KARA_XOR16;
    let table =
        InvNttTableByteSingleGf8::new(&AdditiveNttGf8::new(6, F8::ZERO), &AdditiveNttGf8::new(6, F8(1 << 6)));
    let mut state = 0x31CE_D00D_2468_ACE1u64;
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    for case in 0..600usize {
        let mut a: Vec<u8> = (0..64).map(|_| next() as u8).collect();
        let mut b: Vec<u8> = (0..64).map(|_| next() as u8).collect();
        if case % 5 == 3 {
            a.fill(0);
        }
        if case % 5 == 4 {
            a.fill(0xff);
        }
        let w0 = case % 2 == 0;
        if w0 {
            b[..16].fill(0xff);
            for k in 0..2 {
                a[k * 8 + 5..k * 8 + 8].fill(0);
            }
        } else {
            b[32..].fill(0xff);
        }
        for nt in [false, true] {
            let mut want = [0u8; 64];
            let mut got = [0u8; 64];
            if w0 {
                shift_reduce_inner_mixed_const_b_h4::<0x03, false, 0x03, V0>(&a, &b, &table, 0, None, &mut want, nt);
                shift_reduce_inner_mixed_const_b_h4::<0x03, false, 0x03, AB_MIXED_H2_VARIANT>(&a, &b, &table, 0, None, &mut got, nt);
            } else {
                shift_reduce_inner_mixed_const_b_h4::<0xf0, false, 0, V0>(&a, &b, &table, 0, None, &mut want, nt);
                shift_reduce_inner_mixed_const_b_h4::<0xf0, false, 0, AB_MIXED_H2_VARIANT>(&a, &b, &table, 0, None, &mut got, nt);
            }
            assert_eq!(got, want, "case {case} w0={w0} nt={nt}");
        }
    }
}

/// `acc ^ pmull(a, b)` on 8-bit polynomial lanes (`HI` selects the high
/// halves, i.e. `pmull2`), emitted as an adjacent `pmull`/`eor` pair that
/// Apple P-cores issue in one SIMD issue slot. The 8-bit PMULL fuses like the 64-bit one
/// (measured: 0.25 cycles per pair, against 0.50 when the EOR writes another
/// register). The result is bit-identical to `veorq_u16(acc, pmull_*_u16(a, b))`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn fused_pmull8_acc<const HI: bool>(
    acc: core::arch::aarch64::uint16x8_t,
    a: core::arch::aarch64::uint8x16_t,
    b: core::arch::aarch64::uint8x16_t,
) -> core::arch::aarch64::uint16x8_t {
    let out: core::arch::aarch64::uint16x8_t;
    // SAFETY: register-only polynomial multiply and XOR; `o` is an early
    // clobber so the PMULL never overwrites an input before its last read.
    unsafe {
        if HI {
            core::arch::asm!(
                "pmull2 {o:v}.8h, {a:v}.16b, {b:v}.16b",
                "eor {o:v}.16b, {o:v}.16b, {c:v}.16b",
                a = in(vreg) a,
                b = in(vreg) b,
                c = in(vreg) acc,
                o = out(vreg) out,
                options(pure, nomem, nostack, preserves_flags),
            );
        } else {
            core::arch::asm!(
                "pmull {o:v}.8h, {a:v}.8b, {b:v}.8b",
                "eor {o:v}.16b, {o:v}.16b, {c:v}.16b",
                a = in(vreg) a,
                b = in(vreg) b,
                c = in(vreg) acc,
                o = out(vreg) out,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
    }
    out
}
