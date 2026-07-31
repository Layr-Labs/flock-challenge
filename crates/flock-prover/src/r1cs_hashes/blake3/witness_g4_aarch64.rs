//! Four-G NEON arithmetic for the ranked BLAKE3 witness builder.
//!
//! One BLAKE3 compression has four independent G functions in each column
//! or diagonal half-round.  Those four functions occupy the four `u32` lanes
//! of a NEON register here.  The canonical scalar bit writer remains the
//! output authority: vector factors are extracted in G order and appended as
//! the same contiguous 250-bit records used by the incumbent builder.

use core::arch::aarch64::*;

use super::{
    ADDS_PER_G, BLAKE3_IV, BitRecord, G_STRIDE, K, N_ROUNDS, OUT_HI_BASE, OUT_LO_BASE,
    PER_ROUND_MSG_IDX, PackedWordWriter, REC_C0, REC_C1, REC_C2, REC_C3, REC_C4, REC_C5, REC_LIN0,
    REC_LIN1, USEFUL_BITS, stream_lin_word,
};

const fn build_message_byte_indices() -> [[[[u8; 16]; 2]; 2]; N_ROUNDS] {
    let mut indices = [[[[0u8; 16]; 2]; 2]; N_ROUNDS];
    let mut round = 0;
    while round < N_ROUNDS {
        let mut half = 0;
        while half < 2 {
            let mut side = 0;
            while side < 2 {
                let mut lane = 0;
                while lane < 4 {
                    let word = PER_ROUND_MSG_IDX[round][4 * half + lane][side];
                    let mut byte = 0;
                    while byte < 4 {
                        indices[round][half][side][4 * lane + byte] = (4 * word + byte) as u8;
                        byte += 1;
                    }
                    lane += 1;
                }
                side += 1;
            }
            half += 1;
        }
        round += 1;
    }
    indices
}

const MESSAGE_BYTE_INDICES: [[[[u8; 16]; 2]; 2]; N_ROUNDS] = build_message_byte_indices();

#[derive(Clone, Copy)]
#[repr(C, align(16))]
struct AddFactors4 {
    left: [u32; 4],
    right: [u32; 4],
    carry: [u32; 4],
}

#[repr(C, align(16))]
struct HalfRoundFactors4 {
    adds: [AddFactors4; ADDS_PER_G],
    b_new: [u32; 4],
    d_new: [u32; 4],
}

impl HalfRoundFactors4 {
    #[inline(always)]
    fn new() -> Self {
        const ZERO_ADD: AddFactors4 = AddFactors4 {
            left: [0; 4],
            right: [0; 4],
            carry: [0; 4],
        };
        Self {
            adds: [ZERO_ADD; ADDS_PER_G],
            b_new: [0; 4],
            d_new: [0; 4],
        }
    }
}

#[inline(always)]
unsafe fn rotate16(x: uint32x4_t) -> uint32x4_t {
    unsafe { vreinterpretq_u32_u16(vrev32q_u16(vreinterpretq_u16_u32(x))) }
}

#[inline(always)]
unsafe fn rotate12(x: uint32x4_t) -> uint32x4_t {
    unsafe { vsriq_n_u32(vshlq_n_u32::<20>(x), x, 12) }
}

#[inline(always)]
unsafe fn rotate8(x: uint32x4_t) -> uint32x4_t {
    unsafe { vsriq_n_u32(vshlq_n_u32::<24>(x), x, 8) }
}

#[inline(always)]
unsafe fn rotate7(x: uint32x4_t) -> uint32x4_t {
    unsafe { vsriq_n_u32(vshlq_n_u32::<25>(x), x, 7) }
}

#[inline(always)]
unsafe fn add_parts4_into(
    x: uint32x4_t,
    y: uint32x4_t,
    mask_lo31: uint32x4_t,
    out: &mut AddFactors4,
) -> uint32x4_t {
    unsafe {
        let sum = vaddq_u32(x, y);
        let cin = veorq_u32(veorq_u32(sum, x), y);
        let left = vandq_u32(veorq_u32(x, cin), mask_lo31);
        let right = vandq_u32(veorq_u32(y, cin), mask_lo31);
        let carry = vandq_u32(left, right);
        vst1q_u32(out.left.as_mut_ptr(), left);
        vst1q_u32(out.right.as_mut_ptr(), right);
        vst1q_u32(out.carry.as_mut_ptr(), carry);
        sum
    }
}

#[inline(always)]
unsafe fn g4(
    a: &mut uint32x4_t,
    b: &mut uint32x4_t,
    c: &mut uint32x4_t,
    d: &mut uint32x4_t,
    mx: uint32x4_t,
    my: uint32x4_t,
    factors: &mut HalfRoundFactors4,
) {
    unsafe {
        let mask_lo31 = vdupq_n_u32(0x7fff_ffff);
        let tmp0 = add_parts4_into(*a, *b, mask_lo31, &mut factors.adds[0]);
        let a1 = add_parts4_into(tmp0, mx, mask_lo31, &mut factors.adds[1]);
        let d1 = rotate16(veorq_u32(*d, a1));
        let c1 = add_parts4_into(*c, d1, mask_lo31, &mut factors.adds[2]);
        let b1 = rotate12(veorq_u32(*b, c1));
        let tmp1 = add_parts4_into(a1, b1, mask_lo31, &mut factors.adds[3]);
        let a2 = add_parts4_into(tmp1, my, mask_lo31, &mut factors.adds[4]);
        let d2 = rotate8(veorq_u32(d1, a2));
        let c2 = add_parts4_into(c1, d2, mask_lo31, &mut factors.adds[5]);
        let b2 = rotate7(veorq_u32(b1, c2));

        vst1q_u32(factors.b_new.as_mut_ptr(), b2);
        vst1q_u32(factors.d_new.as_mut_ptr(), d2);
        *a = a2;
        *b = b2;
        *c = c2;
        *d = d2;
    }
}

#[inline(always)]
fn emit_half_round(
    factors: &HalfRoundFactors4,
    wz: &mut PackedWordWriter<'_>,
    wa: &mut PackedWordWriter<'_>,
    wb: &mut PackedWordWriter<'_>,
) {
    for lane in 0..4 {
        let mut rz = BitRecord::<4>::new();
        let mut ra = BitRecord::<4>::new();
        let mut rb = BitRecord::<4>::new();

        macro_rules! push_add {
            ($record_pos:ident, $add:expr) => {{
                rz.push::<$record_pos>(factors.adds[$add].carry[lane]);
                ra.push::<$record_pos>(factors.adds[$add].left[lane]);
                rb.push::<$record_pos>(factors.adds[$add].right[lane]);
            }};
        }
        push_add!(REC_C0, 0);
        push_add!(REC_C1, 1);
        push_add!(REC_C2, 2);
        push_add!(REC_C3, 3);
        push_add!(REC_C4, 4);
        push_add!(REC_C5, 5);
        rz.push::<REC_LIN0>(factors.b_new[lane]);
        ra.push::<REC_LIN0>(factors.b_new[lane]);
        rb.push::<REC_LIN0>(u32::MAX);
        rz.push::<REC_LIN1>(factors.d_new[lane]);
        ra.push::<REC_LIN1>(factors.d_new[lane]);
        rb.push::<REC_LIN1>(u32::MAX);

        wz.push_record(&rz, G_STRIDE);
        wa.push_record(&ra, G_STRIDE);
        wb.push_record(&rb, G_STRIDE);
    }
}

#[inline(always)]
unsafe fn load_messages4(
    message: uint8x16x4_t,
    indices: &[[u8; 16]; 2],
) -> (uint32x4_t, uint32x4_t) {
    unsafe {
        let mx = vqtbl4q_u8(message, vld1q_u8(indices[0].as_ptr()));
        let my = vqtbl4q_u8(message, vld1q_u8(indices[1].as_ptr()));
        (vreinterpretq_u32_u8(mx), vreinterpretq_u32_u8(my))
    }
}

#[inline(always)]
unsafe fn compression_rounds(
    cv: &[u32; 8],
    message: &[u32; 16],
    counter_lo: u32,
    counter_hi: u32,
    block_len: u32,
    flags: u32,
    wz: &mut PackedWordWriter<'_>,
    wa: &mut PackedWordWriter<'_>,
    wb: &mut PackedWordWriter<'_>,
) -> [u32; 16] {
    unsafe {
        let mut a = vld1q_u32(cv.as_ptr());
        let mut b = vld1q_u32(cv.as_ptr().add(4));
        let mut c = vld1q_u32(BLAKE3_IV.as_ptr());
        let d_init = [counter_lo, counter_hi, block_len, flags];
        let mut d = vld1q_u32(d_init.as_ptr());
        let message_table = uint8x16x4_t(
            vld1q_u8(message.as_ptr().cast()),
            vld1q_u8(message.as_ptr().add(4).cast()),
            vld1q_u8(message.as_ptr().add(8).cast()),
            vld1q_u8(message.as_ptr().add(12).cast()),
        );

        for indices in &MESSAGE_BYTE_INDICES {
            let (mx, my) = load_messages4(message_table, &indices[0]);
            let mut factors = HalfRoundFactors4::new();
            g4(&mut a, &mut b, &mut c, &mut d, mx, my, &mut factors);
            emit_half_round(&factors, wz, wa, wb);

            b = vextq_u32::<1>(b, b);
            c = vextq_u32::<2>(c, c);
            d = vextq_u32::<3>(d, d);
            let (mx, my) = load_messages4(message_table, &indices[1]);
            g4(&mut a, &mut b, &mut c, &mut d, mx, my, &mut factors);
            emit_half_round(&factors, wz, wa, wb);
            b = vextq_u32::<3>(b, b);
            c = vextq_u32::<2>(c, c);
            d = vextq_u32::<1>(d, d);
        }

        let mut state = [0u32; 16];
        vst1q_u32(state.as_mut_ptr(), a);
        vst1q_u32(state.as_mut_ptr().add(4), b);
        vst1q_u32(state.as_mut_ptr().add(8), c);
        vst1q_u32(state.as_mut_ptr().add(12), d);
        state
    }
}

/// Build one canonical `(z, A, B)` packed witness block with four-G NEON
/// arithmetic.  The writers and final patch mirror the scalar full-write
/// builder exactly, including explicit initialization of the padding suffix.
pub(super) fn build_block_witness_ab_stream_into(
    cv: &[u32; 8],
    message: &[u32; 16],
    counter: u64,
    block_len: u32,
    flags: u32,
    z: &mut [u64],
    a_out: &mut [u64],
    b_out: &mut [u64],
) {
    const U64_PER_BLOCK: usize = K / 64;
    debug_assert_eq!(z.len(), U64_PER_BLOCK);
    debug_assert_eq!(a_out.len(), U64_PER_BLOCK);
    debug_assert_eq!(b_out.len(), U64_PER_BLOCK);

    let mut wz = PackedWordWriter::new(z);
    let mut wa = PackedWordWriter::new(a_out);
    let mut wb = PackedWordWriter::new(b_out);

    for &value in cv {
        stream_lin_word(value, &mut wz, &mut wa, &mut wb);
    }
    for _ in 0..8 {
        wz.push(0, 32);
        wa.push(0, 32);
        wb.push(0, 32);
    }

    wz.push(1, 1);
    wa.push(1, 1);
    wb.push(1, 1);
    for &value in message {
        stream_lin_word(value, &mut wz, &mut wa, &mut wb);
    }
    let counter_lo = counter as u32;
    let counter_hi = (counter >> 32) as u32;
    stream_lin_word(counter_lo, &mut wz, &mut wa, &mut wb);
    stream_lin_word(counter_hi, &mut wz, &mut wa, &mut wb);
    stream_lin_word(block_len, &mut wz, &mut wa, &mut wb);
    stream_lin_word(flags, &mut wz, &mut wa, &mut wb);

    debug_assert_eq!(wz.position(), super::GS_BASE);
    // SAFETY: NEON is mandatory on AArch64. All vector loads read fixed-size
    // initialized arrays, and factor stores target aligned in-frame arrays.
    let state = unsafe {
        compression_rounds(
            cv, message, counter_lo, counter_hi, block_len, flags, &mut wz, &mut wa, &mut wb,
        )
    };
    debug_assert_eq!(wz.position(), OUT_HI_BASE);

    let out_lo: [u32; 8] = std::array::from_fn(|word| state[word] ^ state[word + 8]);
    for word in 0..8 {
        stream_lin_word(state[word + 8] ^ cv[word], &mut wz, &mut wa, &mut wb);
    }
    debug_assert_eq!(wz.position(), USEFUL_BITS);

    wz.finish();
    wa.finish();
    wb.finish();

    const OUT_LO_WORD: usize = OUT_LO_BASE / 64;
    debug_assert_eq!(OUT_LO_BASE % 64, 0);
    for i in 0..4 {
        let value = (out_lo[2 * i] as u64) | ((out_lo[2 * i + 1] as u64) << 32);
        z[OUT_LO_WORD + i] = value;
        a_out[OUT_LO_WORD + i] = value;
        b_out[OUT_LO_WORD + i] = u64::MAX;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::r1cs_hashes::common::add_carry_parts;

    #[test]
    fn add_parts4_matches_scalar_edges_and_random() {
        let mut seed = 0x5a17_d3c4_9b82_10efu64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed as u32
        };
        let edges = [
            0,
            1,
            u32::MAX,
            0x7fff_ffff,
            0x8000_0000,
            0xaaaa_aaaa,
            0x5555_5555,
        ];

        for iteration in 0..128 {
            let x: [u32; 4] = std::array::from_fn(|lane| {
                edges
                    .get(iteration + lane)
                    .copied()
                    .unwrap_or_else(&mut next)
            });
            let y: [u32; 4] = std::array::from_fn(|lane| {
                edges
                    .get(iteration.wrapping_mul(3) + lane)
                    .copied()
                    .unwrap_or_else(&mut next)
            });
            let mut got = AddFactors4 {
                left: [0; 4],
                right: [0; 4],
                carry: [0; 4],
            };
            let mut sums = [0u32; 4];
            unsafe {
                let xv = vld1q_u32(x.as_ptr());
                let yv = vld1q_u32(y.as_ptr());
                let mask = vdupq_n_u32(0x7fff_ffff);
                let sum = add_parts4_into(xv, yv, mask, &mut got);
                vst1q_u32(sums.as_mut_ptr(), sum);
            }
            for lane in 0..4 {
                let expected = add_carry_parts(x[lane], y[lane]);
                assert_eq!(sums[lane], expected.0);
                assert_eq!(got.left[lane], expected.1);
                assert_eq!(got.right[lane], expected.2);
                assert_eq!(got.carry[lane], expected.3);
            }
        }
    }
}
