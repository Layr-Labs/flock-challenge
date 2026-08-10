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
    n_b_med: usize,
    convert: &[F128],
    eq_lo_val: F128,
    partial_ab: &mut [F128; 64],
) {
    use core::arch::aarch64::*;

    debug_assert!(n_b_med <= 16);
    debug_assert_eq!(convert.len(), 16 * 256);

    // SAFETY: caller guarantees fixed input sizes and aarch64 provides NEON.
    // Every table offset below is a `u8` index scaled by the 16-byte row size,
    // bounded by the debug-asserted table length: `b_med < 16` selects one of
    // 16 convert blocks.
    unsafe {
        let convert_ptr = convert.as_ptr() as *const u8;
        let n_pairs = n_b_med / 2;
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
                let (b_even, b_odd) = (2 * p, 2 * p + 1);
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
            if n_b_med & 1 == 1 {
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
#[inline(always)]
pub(crate) unsafe fn accumulate_convert_ab_nomul(
    chunk_ab_bytes: &[[u8; 64]; 16],
    n_b_med: usize,
    convert: &[F128],
    bank: &mut [F128; 64],
) {
    use core::arch::aarch64::*;

    debug_assert!(n_b_med <= 16);
    debug_assert_eq!(convert.len(), 16 * 256);

    // SAFETY: caller guarantees fixed input sizes and aarch64 provides NEON.
    // Table offsets are `u8` indices scaled by the 16-byte row size within the
    // debug-asserted table length, and every drain address is one 16-byte
    // `F128` slot of the fixed-size bank.
    unsafe {
        let convert_ptr = convert.as_ptr() as *const u8;
        let n_pairs = n_b_med / 2;
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
                let (b_even, b_odd) = (2 * p, 2 * p + 1);
                let t_even = convert_ptr.add(b_even * 256 * 16);
                let t_odd = convert_ptr.add(b_odd * 256 * 16);

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
            if n_b_med & 1 == 1 {
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
                    let slot = bank.as_mut_ptr().add(lane + $offset) as *mut u8;
                    vst1q_u8(slot, veorq_u8(vld1q_u8(slot), $ab));
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

                    // Combine the two independent pair-table loads first, so
                    // only eight values remain live across accumulator loads.
                    let t0 = veorq_u8(
                        vld1q_u8(table01.add(i010 * 16)),
                        vld1q_u8(table23.add(i230 * 16)),
                    );
                    let t1 = veorq_u8(
                        vld1q_u8(table01.add(i011 * 16)),
                        vld1q_u8(table23.add(i231 * 16)),
                    );
                    let t2 = veorq_u8(
                        vld1q_u8(table01.add(i012 * 16)),
                        vld1q_u8(table23.add(i232 * 16)),
                    );
                    let t3 = veorq_u8(
                        vld1q_u8(table01.add(i013 * 16)),
                        vld1q_u8(table23.add(i233 * 16)),
                    );
                    let t4 = veorq_u8(
                        vld1q_u8(table01.add(i014 * 16)),
                        vld1q_u8(table23.add(i234 * 16)),
                    );
                    let t5 = veorq_u8(
                        vld1q_u8(table01.add(i015 * 16)),
                        vld1q_u8(table23.add(i235 * 16)),
                    );
                    let t6 = veorq_u8(
                        vld1q_u8(table01.add(i016 * 16)),
                        vld1q_u8(table23.add(i236 * 16)),
                    );
                    let t7 = veorq_u8(
                        vld1q_u8(table01.add(i017 * 16)),
                        vld1q_u8(table23.add(i237 * 16)),
                    );

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

        let out_ptr = output.as_mut_ptr();
        vst1q_u8(out_ptr, vreinterpretq_u8_u64(y0));
        vst1q_u8(out_ptr.add(16), vreinterpretq_u8_u64(y1));
        vst1q_u8(out_ptr.add(32), vreinterpretq_u8_u64(y2));
        vst1q_u8(out_ptr.add(48), vreinterpretq_u8_u64(y3));
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
    ((word >> (BYTE * 8)) & 0xff) as usize
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
pub(crate) fn shift_reduce_inner_ab_fused_neon_h4(
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

    unsafe {
        let a_base = a_packed.as_ptr().add(byte_base_b);
        let b_base = b_packed.as_ptr().add(byte_base_b);

        // Raw 15-bit products of one K row. `A` comes out of `$at`, which is
        // the plain image for K < 4 and the `x^4`-scaled image for K ≥ 4; `B`
        // always comes out of the plain image.
        macro_rules! products {
            ($at:expr, $at_sw:expr, $k:literal) => {{
                let aw = u64::from_le(core::ptr::read_unaligned(
                    a_base.add($k * N_CHUNKS).cast::<u64>(),
                ));
                let bw = u64::from_le(core::ptr::read_unaligned(
                    b_base.add($k * N_CHUNKS).cast::<u64>(),
                ));
                let (da0, da1, da2, da3) = apply_word_into_4_regs($at, $at_sw, aw);
                let (db0, db1, db2, db3) = apply_word_into_4_regs(plain, plain_sw, bw);
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
        let lo3 = products!(plain, plain_sw, 3);
        let hi3 = products!(scaled, scaled_sw, 7);
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

        let lo2 = products!(plain, plain_sw, 2);
        let hi2 = products!(scaled, scaled_sw, 6);
        horner_absorb(&mut acc, &lo2, &hi2, carry_not);

        let lo1 = products!(plain, plain_sw, 1);
        let hi1 = products!(scaled, scaled_sw, 5);
        horner_absorb(&mut acc, &lo1, &hi1, carry_not);

        let lo0 = products!(plain, plain_sw, 0);
        let hi0 = products!(scaled, scaled_sw, 4);
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

        macro_rules! products {
            ($at:expr, $at_sw:expr, $k:literal) => {{
                let (da0, da1, da2, da3) = if STATIC_A && $k == 1 {
                    let p = a_k1_partial.unwrap_unchecked().as_ptr();
                    (
                        vld1q_u8(p),
                        vld1q_u8(p.add(16)),
                        vld1q_u8(p.add(32)),
                        vld1q_u8(p.add(48)),
                    )
                } else if STATIC_A && $k == 0 {
                    let aw = u64::from_le(core::ptr::read_unaligned(a_base.cast::<u64>()));
                    apply_word_low5_into_4_regs($at, $at_sw, aw)
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
                    apply_word_low5_into_4_regs($at, $at_sw, aw)
                } else {
                    let aw = u64::from_le(core::ptr::read_unaligned(
                        a_base.add($k * N_CHUNKS).cast::<u64>(),
                    ));
                    apply_word_into_4_regs($at, $at_sw, aw)
                };
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
                    let (db0, db1, db2, db3) = apply_word_into_4_regs(plain, plain_sw, bw);
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

        let lo3 = products!(plain, plain_sw, 3);
        let hi3 = products!(scaled, scaled_sw, 7);
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
        let lo2 = products!(plain, plain_sw, 2);
        let hi2 = products!(scaled, scaled_sw, 6);
        horner_absorb(&mut acc, &lo2, &hi2, carry_not);
        let lo1 = products!(plain, plain_sw, 1);
        let hi1 = products!(scaled, scaled_sw, 5);
        horner_absorb(&mut acc, &lo1, &hi1, carry_not);
        let lo0 = products!(plain, plain_sw, 0);
        let hi0 = products!(scaled, scaled_sw, 4);
        horner_absorb(&mut acc, &lo0, &hi0, carry_not);

        let r0 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc[0]), vreinterpretq_u8_u16(acc[1]));
        let r1 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc[2]), vreinterpretq_u8_u16(acc[3]));
        let r2 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc[4]), vreinterpretq_u8_u16(acc[5]));
        let r3 = gf8_reduce_vec16(vreinterpretq_u8_u16(acc[6]), vreinterpretq_u8_u16(acc[7]));

        store_row_64(out.as_mut_ptr(), nt_store, r0, r1, r2, r3);
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
                    shift_reduce_inner_mixed_const_b_h4::<0x03, false, 0x03>(
                        a_packed,
                        b_packed,
                        inv_table,
                        byte_base_b,
                        None,
                        out,
                        nt_store,
                    );
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
                        shift_reduce_inner_mixed_const_b_h4::<0x03, true, 0>(
                            a_packed,
                            b_packed,
                            inv_table,
                            byte_base_b,
                            Some(static_a_k1),
                            out,
                            nt_store,
                        );
                    } else {
                        shift_reduce_inner_mixed_const_b_h4::<0x03, false, 0>(
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
                shift_reduce_inner_mixed_const_b_h4::<0xf0, false, 0>(
                    a_packed,
                    b_packed,
                    inv_table,
                    byte_base_b,
                    None,
                    out,
                    nt_store,
                );
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
        let handled = enabled
            && if fast_shift_reduce_with_policy::<FAST_POLICY>() {
                shift_reduce_inner_ab_bstatic::<true>(
                    a_packed,
                    b_packed,
                    inv_table,
                    chunk_byte_base,
                    b_med,
                    bstatic_w,
                    context,
                    out,
                    nt_store,
                )
            } else {
                shift_reduce_inner_ab_bstatic::<false>(
                    a_packed,
                    b_packed,
                    inv_table,
                    chunk_byte_base,
                    b_med,
                    bstatic_w,
                    context,
                    out,
                    nt_store,
                )
            };
        if handled {
            return;
        }
    }
    if fast_shift_reduce_with_policy::<FAST_POLICY>() {
        shift_reduce_inner_ab_fused_neon_h4(
            a_packed,
            b_packed,
            inv_table,
            chunk_byte_base,
            b_med,
            out,
            nt_store,
        );
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
