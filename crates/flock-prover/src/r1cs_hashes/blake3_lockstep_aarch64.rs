//! Four-compression NEON witness kernel for BLAKE3's ranked row-major path.
//!
//! Arithmetic is lane-wise across four independent compression instances.
//! Output remains row-major, so completed record words are extracted and fed
//! to four independent full-word writers per A/B stream. The z stream is
//! derived only after A/B are complete.

use core::arch::aarch64::{
    uint32x4_t, vaddq_u32, vandq_u32, vdupq_n_u32, veorq_u32, vgetq_lane_u32, vorrq_u32,
    vsetq_lane_u32, vshlq_n_u32, vshrq_n_u32,
};

use super::{
    BLAKE3_IV, Compression, G_STRIDE, K, OUT_LO_BASE, REC_C0, REC_C1, REC_C2, REC_C3,
    REC_C4, REC_C5, REC_LIN0, REC_LIN1,
};

const LANES: usize = 4;
const WORDS_PER_BLOCK: usize = K / 64;
const MASK_LO31: u32 = 0x7FFF_FFFF;

#[derive(Clone, Copy)]
struct Record([u64; 4]);

impl Record {
    #[inline(always)]
    fn new() -> Self {
        Self([0; 4])
    }

    #[inline(always)]
    fn push<const POS: usize>(&mut self, value: u32) {
        let value = value as u64;
        let word = POS >> 6;
        let shift = POS & 63;
        self.0[word] |= value << shift;
        if shift > 32 {
            self.0[word + 1] |= value >> (64 - shift);
        }
    }
}

struct RawPackedWriter {
    out: *mut u64,
    word: usize,
    pending: u64,
    used: usize,
}

impl RawPackedWriter {
    #[inline(always)]
    fn new(out: *mut u64) -> Self {
        Self {
            out,
            word: 0,
            pending: 0,
            used: 0,
        }
    }

    #[inline(always)]
    unsafe fn push(&mut self, value: u64, width: usize) {
        let value = if width == 64 {
            value
        } else {
            value & ((1u64 << width) - 1)
        };
        if self.used == 0 && width == 64 {
            unsafe { self.out.add(self.word).write(value) };
            self.word += 1;
            return;
        }
        let room = 64 - self.used;
        if width < room {
            self.pending |= value << self.used;
            self.used += width;
        } else {
            unsafe {
                self.out
                    .add(self.word)
                    .write(self.pending | (value << self.used));
            }
            self.word += 1;
            if width == room {
                self.pending = 0;
                self.used = 0;
            } else {
                self.pending = value >> room;
                self.used = width - room;
            }
        }
    }

    #[inline(always)]
    unsafe fn push_record(&mut self, record: &Record) {
        unsafe {
            self.push(record.0[0], 64);
            self.push(record.0[1], 64);
            self.push(record.0[2], 64);
            self.push(record.0[3], G_STRIDE - 3 * 64);
        }
    }

    #[inline(always)]
    unsafe fn finish(mut self) {
        if self.used != 0 {
            unsafe { self.out.add(self.word).write(self.pending) };
            self.word += 1;
        }
        while self.word < WORDS_PER_BLOCK {
            unsafe { self.out.add(self.word).write(0) };
            self.word += 1;
        }
    }
}

#[inline(always)]
fn lanes(x0: u32, x1: u32, x2: u32, x3: u32) -> uint32x4_t {
    let mut out = unsafe { vdupq_n_u32(x0) };
    out = unsafe { vsetq_lane_u32::<1>(x1, out) };
    out = unsafe { vsetq_lane_u32::<2>(x2, out) };
    unsafe { vsetq_lane_u32::<3>(x3, out) }
}

#[inline(always)]
unsafe fn splat(value: u32) -> uint32x4_t {
    unsafe { vdupq_n_u32(value) }
}

#[inline(always)]
unsafe fn add_parts(x: uint32x4_t, y: uint32x4_t) -> (uint32x4_t, uint32x4_t, uint32x4_t) {
    let sum = unsafe { vaddq_u32(x, y) };
    let cin = unsafe { veorq_u32(veorq_u32(sum, x), y) };
    let mask = unsafe { vdupq_n_u32(MASK_LO31) };
    let left = unsafe { vandq_u32(veorq_u32(x, cin), mask) };
    let right = unsafe { vandq_u32(veorq_u32(y, cin), mask) };
    (sum, left, right)
}

#[inline(always)]
unsafe fn rotr<const R: i32, const L: i32>(value: uint32x4_t) -> uint32x4_t {
    unsafe { vorrq_u32(vshrq_n_u32::<R>(value), vshlq_n_u32::<L>(value)) }
}

#[inline(always)]
unsafe fn push_vector<const POS: usize>(records: &mut [Record; LANES], value: uint32x4_t) {
    records[0].push::<POS>(unsafe { vgetq_lane_u32::<0>(value) });
    records[1].push::<POS>(unsafe { vgetq_lane_u32::<1>(value) });
    records[2].push::<POS>(unsafe { vgetq_lane_u32::<2>(value) });
    records[3].push::<POS>(unsafe { vgetq_lane_u32::<3>(value) });
}

#[inline(always)]
unsafe fn push_linear(
    a: &mut [RawPackedWriter; LANES],
    b: &mut [RawPackedWriter; LANES],
    value: uint32x4_t,
) {
    unsafe {
        a[0].push(vgetq_lane_u32::<0>(value) as u64, 32);
        a[1].push(vgetq_lane_u32::<1>(value) as u64, 32);
        a[2].push(vgetq_lane_u32::<2>(value) as u64, 32);
        a[3].push(vgetq_lane_u32::<3>(value) as u64, 32);
        b[0].push(u32::MAX as u64, 32);
        b[1].push(u32::MAX as u64, 32);
        b[2].push(u32::MAX as u64, 32);
        b[3].push(u32::MAX as u64, 32);
    }
}

#[inline(always)]
unsafe fn push_zero(a: &mut [RawPackedWriter; LANES], b: &mut [RawPackedWriter; LANES]) {
    unsafe {
        a[0].push(0, 32);
        a[1].push(0, 32);
        a[2].push(0, 32);
        a[3].push(0, 32);
        b[0].push(0, 32);
        b[1].push(0, 32);
        b[2].push(0, 32);
        b[3].push(0, 32);
    }
}

#[inline(always)]
unsafe fn push_one(a: &mut [RawPackedWriter; LANES], b: &mut [RawPackedWriter; LANES]) {
    unsafe {
        a[0].push(1, 1);
        a[1].push(1, 1);
        a[2].push(1, 1);
        a[3].push(1, 1);
        b[0].push(1, 1);
        b[1].push(1, 1);
        b[2].push(1, 1);
        b[3].push(1, 1);
    }
}

#[inline(always)]
unsafe fn flush_records(
    a: &mut [RawPackedWriter; LANES],
    b: &mut [RawPackedWriter; LANES],
    ra: &[Record; LANES],
    rb: &[Record; LANES],
) {
    unsafe {
        a[0].push_record(&ra[0]);
        a[1].push_record(&ra[1]);
        a[2].push_record(&ra[2]);
        a[3].push_record(&ra[3]);
        b[0].push_record(&rb[0]);
        b[1].push_record(&rb[1]);
        b[2].push_record(&rb[2]);
        b[3].push_record(&rb[3]);
    }
}

/// Build four independent BLAKE3 witnesses with one four-lane NEON state.
///
/// # Safety
///
/// `inputs` points to four initialized [`Compression`] values. Each output
/// points to `4 * (K / 64)` writable, pairwise-disjoint `u64`s. The ranges do
/// not overlap the inputs.
#[unsafe(no_mangle)]
#[inline(never)]
pub(super) unsafe extern "C" fn flock_blake3_witness_group4_neon(
    inputs: *const Compression,
    z_out: *mut u64,
    a_out: *mut u64,
    b_out: *mut u64,
) {
    let i0 = unsafe { &*inputs.add(0) };
    let i1 = unsafe { &*inputs.add(1) };
    let i2 = unsafe { &*inputs.add(2) };
    let i3 = unsafe { &*inputs.add(3) };

    let mut wa = [
        RawPackedWriter::new(a_out),
        RawPackedWriter::new(unsafe { a_out.add(WORDS_PER_BLOCK) }),
        RawPackedWriter::new(unsafe { a_out.add(2 * WORDS_PER_BLOCK) }),
        RawPackedWriter::new(unsafe { a_out.add(3 * WORDS_PER_BLOCK) }),
    ];
    let mut wb = [
        RawPackedWriter::new(b_out),
        RawPackedWriter::new(unsafe { b_out.add(WORDS_PER_BLOCK) }),
        RawPackedWriter::new(unsafe { b_out.add(2 * WORDS_PER_BLOCK) }),
        RawPackedWriter::new(unsafe { b_out.add(3 * WORDS_PER_BLOCK) }),
    ];

    macro_rules! cv {
        ($word:literal) => {
            lanes(i0.0[$word], i1.0[$word], i2.0[$word], i3.0[$word])
        };
    }
    macro_rules! msg {
        ($word:expr) => {
            lanes(i0.1[$word], i1.1[$word], i2.1[$word], i3.1[$word])
        };
    }

    unsafe {
        push_linear(&mut wa, &mut wb, cv!(0));
        push_linear(&mut wa, &mut wb, cv!(1));
        push_linear(&mut wa, &mut wb, cv!(2));
        push_linear(&mut wa, &mut wb, cv!(3));
        push_linear(&mut wa, &mut wb, cv!(4));
        push_linear(&mut wa, &mut wb, cv!(5));
        push_linear(&mut wa, &mut wb, cv!(6));
        push_linear(&mut wa, &mut wb, cv!(7));
        for _ in 0..8 {
            push_zero(&mut wa, &mut wb);
        }
        push_one(&mut wa, &mut wb);
        for word in 0..16 {
            push_linear(&mut wa, &mut wb, msg!(word));
        }
    }

    let counter_lo = lanes(i0.2 as u32, i1.2 as u32, i2.2 as u32, i3.2 as u32);
    let counter_hi = lanes(
        (i0.2 >> 32) as u32,
        (i1.2 >> 32) as u32,
        (i2.2 >> 32) as u32,
        (i3.2 >> 32) as u32,
    );
    let block_len = lanes(i0.3, i1.3, i2.3, i3.3);
    let flags = lanes(i0.4, i1.4, i2.4, i3.4);
    unsafe {
        push_linear(&mut wa, &mut wb, counter_lo);
        push_linear(&mut wa, &mut wb, counter_hi);
        push_linear(&mut wa, &mut wb, block_len);
        push_linear(&mut wa, &mut wb, flags);
    }

    let mut state: [uint32x4_t; 16] = [
        cv!(0),
        cv!(1),
        cv!(2),
        cv!(3),
        cv!(4),
        cv!(5),
        cv!(6),
        cv!(7),
        unsafe { splat(BLAKE3_IV[0]) },
        unsafe { splat(BLAKE3_IV[1]) },
        unsafe { splat(BLAKE3_IV[2]) },
        unsafe { splat(BLAKE3_IV[3]) },
        counter_lo,
        counter_hi,
        block_len,
        flags,
    ];

    macro_rules! g {
        ($la:literal, $lb:literal, $lc:literal, $ld:literal, $mx:expr, $my:expr) => {{
            let mut ra = [Record::new(); LANES];
            let mut rb = [Record::new(); LANES];
            macro_rules! add {
                ($pos:ident, $x:expr, $y:expr) => {{
                    let (sum, left, right) = unsafe { add_parts($x, $y) };
                    unsafe {
                        push_vector::<$pos>(&mut ra, left);
                        push_vector::<$pos>(&mut rb, right);
                    }
                    sum
                }};
            }

            let mx = msg!($mx);
            let my = msg!($my);
            let tmp_0 = add!(REC_C0, state[$la], state[$lb]);
            let a_1 = add!(REC_C1, tmp_0, mx);
            let d_1 = unsafe { rotr::<16, 16>(veorq_u32(state[$ld], a_1)) };
            let c_1 = add!(REC_C2, state[$lc], d_1);
            let b_1 = unsafe { rotr::<12, 20>(veorq_u32(state[$lb], c_1)) };
            let tmp_1 = add!(REC_C3, a_1, b_1);
            let a_2 = add!(REC_C4, tmp_1, my);
            let d_2 = unsafe { rotr::<8, 24>(veorq_u32(d_1, a_2)) };
            let c_2 = add!(REC_C5, c_1, d_2);
            let b_new = unsafe { rotr::<7, 25>(veorq_u32(b_1, c_2)) };

            unsafe {
                push_vector::<REC_LIN0>(&mut ra, b_new);
                push_vector::<REC_LIN0>(&mut rb, splat(u32::MAX));
                push_vector::<REC_LIN1>(&mut ra, d_2);
                push_vector::<REC_LIN1>(&mut rb, splat(u32::MAX));
                flush_records(&mut wa, &mut wb, &ra, &rb);
            }
            state[$la] = a_2;
            state[$lb] = b_new;
            state[$lc] = c_2;
            state[$ld] = d_2;
        }};
    }

    macro_rules! round {
        ($m0:expr, $m1:expr, $m2:expr, $m3:expr, $m4:expr, $m5:expr, $m6:expr, $m7:expr,
         $m8:expr, $m9:expr, $m10:expr, $m11:expr, $m12:expr, $m13:expr, $m14:expr, $m15:expr) => {{
            g!(0, 4, 8, 12, $m0, $m1);
            g!(1, 5, 9, 13, $m2, $m3);
            g!(2, 6, 10, 14, $m4, $m5);
            g!(3, 7, 11, 15, $m6, $m7);
            g!(0, 5, 10, 15, $m8, $m9);
            g!(1, 6, 11, 12, $m10, $m11);
            g!(2, 7, 8, 13, $m12, $m13);
            g!(3, 4, 9, 14, $m14, $m15);
        }};
    }

    round!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    round!(2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8);
    round!(3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1);
    round!(10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6);
    round!(12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4);
    round!(9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7);
    round!(11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13);

    macro_rules! finish_pair {
        ($pair:literal, $w0:literal, $w1:literal) => {{
            let lo0 = unsafe { veorq_u32(state[$w0], state[$w0 + 8]) };
            let lo1 = unsafe { veorq_u32(state[$w1], state[$w1 + 8]) };
            let hi0 = unsafe { veorq_u32(state[$w0 + 8], cv!($w0)) };
            let hi1 = unsafe { veorq_u32(state[$w1 + 8], cv!($w1)) };
            unsafe {
                push_linear(&mut wa, &mut wb, hi0);
                push_linear(&mut wa, &mut wb, hi1);
            }
            let out_word = OUT_LO_BASE / 64 + $pair;
            unsafe {
                a_out.add(0 * WORDS_PER_BLOCK + out_word).write(
                    vgetq_lane_u32::<0>(lo0) as u64 | ((vgetq_lane_u32::<0>(lo1) as u64) << 32),
                );
                a_out.add(1 * WORDS_PER_BLOCK + out_word).write(
                    vgetq_lane_u32::<1>(lo0) as u64 | ((vgetq_lane_u32::<1>(lo1) as u64) << 32),
                );
                a_out.add(2 * WORDS_PER_BLOCK + out_word).write(
                    vgetq_lane_u32::<2>(lo0) as u64 | ((vgetq_lane_u32::<2>(lo1) as u64) << 32),
                );
                a_out.add(3 * WORDS_PER_BLOCK + out_word).write(
                    vgetq_lane_u32::<3>(lo0) as u64 | ((vgetq_lane_u32::<3>(lo1) as u64) << 32),
                );
                b_out.add(0 * WORDS_PER_BLOCK + out_word).write(u64::MAX);
                b_out.add(1 * WORDS_PER_BLOCK + out_word).write(u64::MAX);
                b_out.add(2 * WORDS_PER_BLOCK + out_word).write(u64::MAX);
                b_out.add(3 * WORDS_PER_BLOCK + out_word).write(u64::MAX);
            }
        }};
    }

    finish_pair!(0, 0, 1);
    finish_pair!(1, 2, 3);
    finish_pair!(2, 4, 5);
    finish_pair!(3, 6, 7);

    let [wa0, wa1, wa2, wa3] = wa;
    let [wb0, wb1, wb2, wb3] = wb;
    unsafe {
        wa0.finish();
        wa1.finish();
        wa2.finish();
        wa3.finish();
        wb0.finish();
        wb1.finish();
        wb2.finish();
        wb3.finish();
    }

    for lane in 0..LANES {
        for word in 0..WORDS_PER_BLOCK {
            let offset = lane * WORDS_PER_BLOCK + word;
            unsafe {
                z_out
                    .add(offset)
                    .write(a_out.add(offset).read() & b_out.add(offset).read())
            };
        }
    }

}
