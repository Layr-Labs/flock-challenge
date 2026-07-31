use super::super::{InvNttTableByteSingleGf8, F128, F8, N_CHUNKS};

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

/// Two-bank convert fold, with the C-side gathers halved by `b_med` pairing.
///
/// The two C banks select `c & 0x55` and `c & 0xaa`, so each index carries only
/// 4 live bits while spending a full 256-row lookup — the C side runs at half
/// its addressing capacity, while the AB side consumes all 8 bits and is
/// incompressible. Pairing two adjacent `b_med` into one lookup per bank
/// therefore removes 4 of the 12 gathers per `b_med` per 4-lane group (the C
/// side goes 8 → 4) at the cost of a few non-dependent integer ops per index.
///
/// `m0` / `m1` are [`super::super::paired_c_tables`], laid out `p * 256 + i`.
/// For pair `p` covering `b_med = 2p, 2p+1`, with
/// `i0 = (c_2p & 0x55) | ((c_2p1 << 1) & 0xaa)`:
/// ```text
/// m0[p][i0] == T_2p[c_2p & 0x55] ^ T_2p1[c_2p1 & 0x55]
/// ```
/// which is exactly the two bank-0 terms the unpaired loop XORs across those
/// `b_med`, and symmetrically for bank 1. The banks never mix, so
/// `partial_c_0` and `partial_c_1` stay separate. This is an identity, not an
/// approximation: it follows from `convert` being F2-linear in its index bits
/// (verified exhaustively by `convert_table_index_linear`), so the result is
/// bit-identical to the unpaired fold.
///
/// An odd trailing `b_med` (only reachable on the padded boundary window) falls
/// back to the unpaired path.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) unsafe fn accumulate_convert_with_s_hat_v(
    chunk_ab_bytes: &[[u8; 64]; 16],
    chunk_c_bytes: &[[u8; 64]; 16],
    n_b_med: usize,
    convert: &[F128],
    m0: &[F128],
    m1: &[F128],
    eq_lo_val: F128,
    partial_ab: &mut [F128; 64],
    partial_c_0: &mut [F128; 64],
    partial_c_1: &mut [F128; 64],
) {
    use core::arch::aarch64::*;

    debug_assert!(n_b_med <= 16);
    debug_assert_eq!(convert.len(), 16 * 256);
    debug_assert_eq!(m0.len(), 8 * 256);
    debug_assert_eq!(m1.len(), 8 * 256);

    // SAFETY: caller guarantees fixed input sizes and aarch64 provides NEON.
    // Every table offset below is a `u8` index scaled by the 16-byte row size,
    // bounded by the debug-asserted table lengths: `b_med < 16` selects one of
    // 16 convert blocks, and `p < 8` one of 8 paired blocks.
    unsafe {
        let convert_ptr = convert.as_ptr() as *const u8;
        let m0_ptr = m0.as_ptr() as *const u8;
        let m1_ptr = m1.as_ptr() as *const u8;
        let n_pairs = n_b_med / 2;
        for lane in (0..64).step_by(4) {
            // 12 live accumulator q-registers (4 lanes × 3 banks); the paired
            // loop adds only address temporaries, so nothing spills.
            let mut ab0 = vdupq_n_u8(0);
            let mut ab1 = vdupq_n_u8(0);
            let mut ab2 = vdupq_n_u8(0);
            let mut ab3 = vdupq_n_u8(0);
            let mut c00 = vdupq_n_u8(0);
            let mut c01 = vdupq_n_u8(0);
            let mut c02 = vdupq_n_u8(0);
            let mut c03 = vdupq_n_u8(0);
            let mut c10 = vdupq_n_u8(0);
            let mut c11 = vdupq_n_u8(0);
            let mut c12 = vdupq_n_u8(0);
            let mut c13 = vdupq_n_u8(0);

            macro_rules! gather_pair {
                ($p:expr) => {{
                    let p = $p;
                    let (b_even, b_odd) = (2 * p, 2 * p + 1);
                    let t_even = convert_ptr.add(b_even * 256 * 16);
                    let t_odd = convert_ptr.add(b_odd * 256 * 16);
                    let q0 = m0_ptr.add(p * 256 * 16);
                    let q1 = m1_ptr.add(p * 256 * 16);

                    // One u32 load per (side, b_med) covers the 4 adjacent
                    // lanes; each extracted byte addresses the same table row
                    // as the original byte-load form.
                    let we = (chunk_ab_bytes[b_even].as_ptr().add(lane) as *const u32)
                        .read_unaligned() as usize;
                    let wo = (chunk_ab_bytes[b_odd].as_ptr().add(lane) as *const u32)
                        .read_unaligned() as usize;
                    let e0 = we & 0xff;
                    let e1 = (we >> 8) & 0xff;
                    let e2 = (we >> 16) & 0xff;
                    let e3 = we >> 24;
                    let o0 = wo & 0xff;
                    let o1 = (wo >> 8) & 0xff;
                    let o2 = (wo >> 16) & 0xff;
                    let o3 = wo >> 24;
                    ab0 = xor3_u8(
                        ab0,
                        vld1q_u8(t_even.add(e0 * 16)),
                        vld1q_u8(t_odd.add(o0 * 16)),
                    );
                    ab1 = xor3_u8(
                        ab1,
                        vld1q_u8(t_even.add(e1 * 16)),
                        vld1q_u8(t_odd.add(o1 * 16)),
                    );
                    ab2 = xor3_u8(
                        ab2,
                        vld1q_u8(t_even.add(e2 * 16)),
                        vld1q_u8(t_odd.add(o2 * 16)),
                    );
                    ab3 = xor3_u8(
                        ab3,
                        vld1q_u8(t_even.add(e3 * 16)),
                        vld1q_u8(t_odd.add(o3 * 16)),
                    );

                    // The pairing masks distribute over the word; one
                    // mask/shift sequence builds all 4 lanes' j and k.
                    let ce = (chunk_c_bytes[b_even].as_ptr().add(lane) as *const u32)
                        .read_unaligned() as usize;
                    let co = (chunk_c_bytes[b_odd].as_ptr().add(lane) as *const u32)
                        .read_unaligned() as usize;
                    let jw = (ce & 0x5555_5555) | ((co << 1) & 0xaaaa_aaaa);
                    let kw = (ce & 0xaaaa_aaaa) | ((co >> 1) & 0x5555_5555);
                    (q0, q1, jw, kw)
                }};
            }

            // Two adjacent paired-C rows feed one EOR3 per bank/lane. The AB
            // side already folds each pair's even/odd rows in one EOR3.
            let mut p = 0;
            while p + 1 < n_pairs {
                let (q00, q10, jw0, kw0) = gather_pair!(p);
                let (q01, q11, jw1, kw1) = gather_pair!(p + 1);
                c00 = xor3_u8(
                    c00,
                    vld1q_u8(q00.add((jw0 & 0xff) * 16)),
                    vld1q_u8(q01.add((jw1 & 0xff) * 16)),
                );
                c01 = xor3_u8(
                    c01,
                    vld1q_u8(q00.add(((jw0 >> 8) & 0xff) * 16)),
                    vld1q_u8(q01.add(((jw1 >> 8) & 0xff) * 16)),
                );
                c02 = xor3_u8(
                    c02,
                    vld1q_u8(q00.add(((jw0 >> 16) & 0xff) * 16)),
                    vld1q_u8(q01.add(((jw1 >> 16) & 0xff) * 16)),
                );
                c03 = xor3_u8(
                    c03,
                    vld1q_u8(q00.add((jw0 >> 24) * 16)),
                    vld1q_u8(q01.add((jw1 >> 24) * 16)),
                );
                c10 = xor3_u8(
                    c10,
                    vld1q_u8(q10.add((kw0 & 0xff) * 16)),
                    vld1q_u8(q11.add((kw1 & 0xff) * 16)),
                );
                c11 = xor3_u8(
                    c11,
                    vld1q_u8(q10.add(((kw0 >> 8) & 0xff) * 16)),
                    vld1q_u8(q11.add(((kw1 >> 8) & 0xff) * 16)),
                );
                c12 = xor3_u8(
                    c12,
                    vld1q_u8(q10.add(((kw0 >> 16) & 0xff) * 16)),
                    vld1q_u8(q11.add(((kw1 >> 16) & 0xff) * 16)),
                );
                c13 = xor3_u8(
                    c13,
                    vld1q_u8(q10.add((kw0 >> 24) * 16)),
                    vld1q_u8(q11.add((kw1 >> 24) * 16)),
                );
                p += 2;
            }
            if p < n_pairs {
                let (q0, q1, jw, kw) = gather_pair!(p);
                c00 = veorq_u8(c00, vld1q_u8(q0.add((jw & 0xff) * 16)));
                c01 = veorq_u8(c01, vld1q_u8(q0.add(((jw >> 8) & 0xff) * 16)));
                c02 = veorq_u8(c02, vld1q_u8(q0.add(((jw >> 16) & 0xff) * 16)));
                c03 = veorq_u8(c03, vld1q_u8(q0.add((jw >> 24) * 16)));
                c10 = veorq_u8(c10, vld1q_u8(q1.add((kw & 0xff) * 16)));
                c11 = veorq_u8(c11, vld1q_u8(q1.add(((kw >> 8) & 0xff) * 16)));
                c12 = veorq_u8(c12, vld1q_u8(q1.add(((kw >> 16) & 0xff) * 16)));
                c13 = veorq_u8(c13, vld1q_u8(q1.add((kw >> 24) * 16)));
            }

            // Odd trailing b_med: unpaired fallback.
            if n_b_med & 1 == 1 {
                let b_med = n_b_med - 1;
                let table = convert_ptr.add(b_med * 256 * 16);
                let wa = (chunk_ab_bytes[b_med].as_ptr().add(lane) as *const u32).read_unaligned()
                    as usize;
                let wc = (chunk_c_bytes[b_med].as_ptr().add(lane) as *const u32).read_unaligned()
                    as usize;
                let a0 = wa & 0xff;
                let a1 = (wa >> 8) & 0xff;
                let a2 = (wa >> 16) & 0xff;
                let a3 = wa >> 24;
                let j = wc & 0x5555_5555;
                let k = wc & 0xaaaa_aaaa;
                ab0 = veorq_u8(ab0, vld1q_u8(table.add(a0 * 16)));
                ab1 = veorq_u8(ab1, vld1q_u8(table.add(a1 * 16)));
                ab2 = veorq_u8(ab2, vld1q_u8(table.add(a2 * 16)));
                ab3 = veorq_u8(ab3, vld1q_u8(table.add(a3 * 16)));
                c00 = veorq_u8(c00, vld1q_u8(table.add((j & 0xff) * 16)));
                c01 = veorq_u8(c01, vld1q_u8(table.add(((j >> 8) & 0xff) * 16)));
                c02 = veorq_u8(c02, vld1q_u8(table.add(((j >> 16) & 0xff) * 16)));
                c03 = veorq_u8(c03, vld1q_u8(table.add((j >> 24) * 16)));
                c10 = veorq_u8(c10, vld1q_u8(table.add((k & 0xff) * 16)));
                c11 = veorq_u8(c11, vld1q_u8(table.add(((k >> 8) & 0xff) * 16)));
                c12 = veorq_u8(c12, vld1q_u8(table.add(((k >> 16) & 0xff) * 16)));
                c13 = veorq_u8(c13, vld1q_u8(table.add((k >> 24) * 16)));
            }

            macro_rules! drain_lane {
                ($offset:literal, $ab:ident, $c0:ident, $c1:ident) => {{
                    let ab = vreinterpretq_u64_u8($ab);
                    let c0 = vreinterpretq_u64_u8($c0);
                    let c1 = vreinterpretq_u64_u8($c1);
                    partial_ab[lane + $offset] += F128 {
                        lo: vgetq_lane_u64::<0>(ab),
                        hi: vgetq_lane_u64::<1>(ab),
                    } * eq_lo_val;
                    partial_c_0[lane + $offset] += F128 {
                        lo: vgetq_lane_u64::<0>(c0),
                        hi: vgetq_lane_u64::<1>(c0),
                    } * eq_lo_val;
                    partial_c_1[lane + $offset] += F128 {
                        lo: vgetq_lane_u64::<0>(c1),
                        hi: vgetq_lane_u64::<1>(c1),
                    } * eq_lo_val;
                }};
            }
            drain_lane!(0, ab0, c00, c10);
            drain_lane!(1, ab1, c01, c11);
            drain_lane!(2, ab2, c02, c12);
            drain_lane!(3, ab3, c03, c13);
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

        let p = out.as_mut_ptr();
        vst1q_u8(p, r0);
        vst1q_u8(p.add(16), r1);
        vst1q_u8(p.add(32), r2);
        vst1q_u8(p.add(48), r3);
    }
}

// ---------------------------------------------------------------------------
// EXPERIMENT (seam-urm): checked static-structure fast paths.
//
// BLAKE3 witness structure (measured on real witnesses, 4096 blocks):
//  * b_med blocks 0 and 1 of every first 8192-bit window have B == all-ones
//    for all 8 K rows (const-one wires of linear constraints)
//  * the final processed b_med block of each witness block has B == 0 for
//    K = 1..7 (structural zero padding rows)
// Both are detected at runtime with 8 u64 loads + compares per 64-byte block
// (vs 512 table loads), so the fast paths are witness-safe: any witness that
// does not match takes the generic path.
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

/// Const-ones-B fast path: `out = red(sum_K x^K ntt_a[K])`. The transform
/// of the packed all-ones row is exactly the GF(2^8) multiplicative identity
/// in all 64 lanes, so the baseline's final vector multiply is redundant.
/// Caller has verified all eight B words are all-ones.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
fn shift_reduce_inner_a_only_const_b(
    a_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    byte_base_b: usize,
    out: &mut [u8; 64],
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

        let p = out.as_mut_ptr();
        vst1q_u8(p, y0);
        vst1q_u8(p.add(16), y1);
        vst1q_u8(p.add(32), y2);
        vst1q_u8(p.add(48), y3);
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
) {
    use crate::field::gf2_8::neon::gf8_mul_vec16;
    use core::arch::aarch64::*;

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
        let p = out.as_mut_ptr();
        vst1q_u8(p, y0);
        vst1q_u8(p.add(16), y1);
        vst1q_u8(p.add(32), y2);
        vst1q_u8(p.add(48), y3);
    }
}

/// Checked static-structure dispatcher. Sniffs the 8 b-words (8 scalar loads
/// vs 512 table loads) and routes to a fast path when the block matches the
/// BLAKE3 static structure; generic otherwise. Output is bit-identical for
/// any witness.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) fn shift_reduce_inner_ab_fused_neon_checked(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    out: &mut [u8; 64],
) {
    let byte_base_b = chunk_byte_base + b_med * N_CHUNKS * 8;
    let bw = |k: usize| -> u64 {
        u64::from_le(unsafe {
            core::ptr::read_unaligned(b_packed.as_ptr().add(byte_base_b + k * 8).cast::<u64>())
        })
    };
    let and_all = bw(0) & bw(1) & bw(2) & bw(3) & bw(4) & bw(5) & bw(6) & bw(7);
    if and_all == u64::MAX {
        shift_reduce_inner_a_only_const_b(a_packed, inv_table, byte_base_b, out);
        return;
    }
    let or_tail = bw(1) | bw(2) | bw(3) | bw(4) | bw(5) | bw(6) | bw(7);
    if or_tail == 0 {
        shift_reduce_inner_single_k0(a_packed, b_packed, inv_table, byte_base_b, out);
        return;
    }
    shift_reduce_inner_ab_fused_neon(a_packed, b_packed, inv_table, chunk_byte_base, b_med, out);
}
