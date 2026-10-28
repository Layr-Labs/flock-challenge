use core::arch::aarch64::*;

core::arch::global_asm!(include_str!("blake3_neon8_macos.S"), options(raw));

unsafe extern "C" {
    fn flock_blake3_hash8_neon_1024(data: *const u8, out: *mut u8, groups: usize);
}

#[inline]
pub(super) fn hash_complete_groups(data: &[u8], out: &mut [[u8; 32]]) -> usize {
    debug_assert_eq!(data.len(), out.len() * 1024);
    let groups = out.len() / 8;
    if groups == 0 {
        return 0;
    }

    unsafe {
        flock_blake3_hash8_neon_1024(data.as_ptr(), out.as_mut_ptr().cast(), groups);
    }
    groups * 8
}

const IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

const MSG_SCHEDULE: [[u8; 16]; 7] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
    [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
    [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
    [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
    [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
    [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
];

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
    const ROTATE: [u8; 16] = [1, 2, 3, 0, 5, 6, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12];
    unsafe {
        let table = vld1q_u8(ROTATE.as_ptr());
        vreinterpretq_u32_u8(vqtbl1q_u8(vreinterpretq_u8_u32(x), table))
    }
}

#[inline(always)]
unsafe fn rotate7(x: uint32x4_t) -> uint32x4_t {
    unsafe { vsriq_n_u32(vshlq_n_u32::<25>(x), x, 7) }
}

#[inline(always)]
unsafe fn transpose4(vecs: &mut [uint32x4_t; 4]) {
    unsafe {
        let rows01 = vtrnq_u32(vecs[0], vecs[1]);
        let rows23 = vtrnq_u32(vecs[2], vecs[3]);
        vecs[0] = vcombine_u32(vget_low_u32(rows01.0), vget_low_u32(rows23.0));
        vecs[1] = vcombine_u32(vget_low_u32(rows01.1), vget_low_u32(rows23.1));
        vecs[2] = vcombine_u32(vget_high_u32(rows01.0), vget_high_u32(rows23.0));
        vecs[3] = vcombine_u32(vget_high_u32(rows01.1), vget_high_u32(rows23.1));
    }
}

#[inline(always)]
unsafe fn transpose_parent4(base: *const u8, out: &mut [uint32x4_t; 16]) {
    unsafe {
        for word_vec in 0..4 {
            let word_offset = word_vec * 16;
            let mut rows = [
                vreinterpretq_u32_u8(vld1q_u8(base.add(word_offset))),
                vreinterpretq_u32_u8(vld1q_u8(base.add(64 + word_offset))),
                vreinterpretq_u32_u8(vld1q_u8(base.add(128 + word_offset))),
                vreinterpretq_u32_u8(vld1q_u8(base.add(192 + word_offset))),
            ];
            transpose4(&mut rows);
            out[word_vec * 4..word_vec * 4 + 4].copy_from_slice(&rows);
        }
    }
}

#[inline(always)]
unsafe fn store_cv4(state: &mut [uint32x4_t; 16], out: *mut u8) {
    unsafe {
        let mut low = [
            veorq_u32(state[0], state[8]),
            veorq_u32(state[1], state[9]),
            veorq_u32(state[2], state[10]),
            veorq_u32(state[3], state[11]),
        ];
        let mut high = [
            veorq_u32(state[4], state[12]),
            veorq_u32(state[5], state[13]),
            veorq_u32(state[6], state[14]),
            veorq_u32(state[7], state[15]),
        ];
        transpose4(&mut low);
        transpose4(&mut high);
        vst1q_u8(out, vreinterpretq_u8_u32(low[0]));
        vst1q_u8(out.add(16), vreinterpretq_u8_u32(high[0]));
        vst1q_u8(out.add(32), vreinterpretq_u8_u32(low[1]));
        vst1q_u8(out.add(48), vreinterpretq_u8_u32(high[1]));
        vst1q_u8(out.add(64), vreinterpretq_u8_u32(low[2]));
        vst1q_u8(out.add(80), vreinterpretq_u8_u32(high[2]));
        vst1q_u8(out.add(96), vreinterpretq_u8_u32(low[3]));
        vst1q_u8(out.add(112), vreinterpretq_u8_u32(high[3]));
    }
}

#[inline(always)]
unsafe fn round3(
    v0: &mut [uint32x4_t; 16],
    v1: &mut [uint32x4_t; 16],
    v2: &mut [uint32x4_t; 16],
    m0: &[uint32x4_t; 16],
    m1: &[uint32x4_t; 16],
    m2: &[uint32x4_t; 16],
    schedule: &[u8; 16],
) {
    macro_rules! g3 {
        ($a:literal, $b:literal, $c:literal, $d:literal, $x:expr, $y:expr) => {{
            let x = $x as usize;
            let y = $y as usize;
            unsafe {
                v0[$a] = vaddq_u32(vaddq_u32(v0[$a], v0[$b]), *m0.get_unchecked(x));
                v1[$a] = vaddq_u32(vaddq_u32(v1[$a], v1[$b]), *m1.get_unchecked(x));
                v2[$a] = vaddq_u32(vaddq_u32(v2[$a], v2[$b]), *m2.get_unchecked(x));
                v0[$d] = rotate16(veorq_u32(v0[$d], v0[$a]));
                v1[$d] = rotate16(veorq_u32(v1[$d], v1[$a]));
                v2[$d] = rotate16(veorq_u32(v2[$d], v2[$a]));
                v0[$c] = vaddq_u32(v0[$c], v0[$d]);
                v1[$c] = vaddq_u32(v1[$c], v1[$d]);
                v2[$c] = vaddq_u32(v2[$c], v2[$d]);
                v0[$b] = rotate12(veorq_u32(v0[$b], v0[$c]));
                v1[$b] = rotate12(veorq_u32(v1[$b], v1[$c]));
                v2[$b] = rotate12(veorq_u32(v2[$b], v2[$c]));
                v0[$a] = vaddq_u32(vaddq_u32(v0[$a], v0[$b]), *m0.get_unchecked(y));
                v1[$a] = vaddq_u32(vaddq_u32(v1[$a], v1[$b]), *m1.get_unchecked(y));
                v2[$a] = vaddq_u32(vaddq_u32(v2[$a], v2[$b]), *m2.get_unchecked(y));
                v0[$d] = rotate8(veorq_u32(v0[$d], v0[$a]));
                v1[$d] = rotate8(veorq_u32(v1[$d], v1[$a]));
                v2[$d] = rotate8(veorq_u32(v2[$d], v2[$a]));
                v0[$c] = vaddq_u32(v0[$c], v0[$d]);
                v1[$c] = vaddq_u32(v1[$c], v1[$d]);
                v2[$c] = vaddq_u32(v2[$c], v2[$d]);
                v0[$b] = rotate7(veorq_u32(v0[$b], v0[$c]));
                v1[$b] = rotate7(veorq_u32(v1[$b], v1[$c]));
                v2[$b] = rotate7(veorq_u32(v2[$b], v2[$c]));
            }
        }};
    }

    g3!(0, 4, 8, 12, schedule[0], schedule[1]);
    g3!(1, 5, 9, 13, schedule[2], schedule[3]);
    g3!(2, 6, 10, 14, schedule[4], schedule[5]);
    g3!(3, 7, 11, 15, schedule[6], schedule[7]);
    g3!(0, 5, 10, 15, schedule[8], schedule[9]);
    g3!(1, 6, 11, 12, schedule[10], schedule[11]);
    g3!(2, 7, 8, 13, schedule[12], schedule[13]);
    g3!(3, 4, 9, 14, schedule[14], schedule[15]);
}

#[inline]
pub(super) fn hash_complete_parent_groups(data: &[u8], out: &mut [[u8; 32]]) -> usize {
    hash_complete_groups_flags(data, out, 4)
}

pub(crate) fn hash_complete_pow_groups(data: &[u8], out: &mut [[u8; 32]]) -> usize {
    hash_complete_groups_flags(data, out, 11)
}

const POW_FLAGS: u32 = 11;

struct PowFixed {
    dw: [u32; 8],
    c: [u32; 16],
    p05: u32,
}

#[inline]
fn g_scalar(s: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, mx: u32, my: u32) {
    s[a] = s[a].wrapping_add(s[b]).wrapping_add(mx);
    s[d] = (s[d] ^ s[a]).rotate_right(16);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] = (s[b] ^ s[c]).rotate_right(12);
    s[a] = s[a].wrapping_add(s[b]).wrapping_add(my);
    s[d] = (s[d] ^ s[a]).rotate_right(8);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] = (s[b] ^ s[c]).rotate_right(7);
}

fn pow_fixed(state_digest: &[u8; 32]) -> PowFixed {
    let mut dw = [0u32; 8];
    for (i, w) in dw.iter_mut().enumerate() {
        *w = u32::from_le_bytes(state_digest[4 * i..4 * i + 4].try_into().unwrap());
    }
    let mut s = [
        IV[0], IV[1], IV[2], IV[3], IV[4], IV[5], IV[6], IV[7], IV[0], IV[1], IV[2], IV[3], 0, 0,
        64, POW_FLAGS,
    ];
    g_scalar(&mut s, 0, 4, 8, 12, dw[0], dw[1]);
    g_scalar(&mut s, 1, 5, 9, 13, dw[2], dw[3]);
    g_scalar(&mut s, 2, 6, 10, 14, dw[4], dw[5]);
    g_scalar(&mut s, 3, 7, 11, 15, dw[6], dw[7]);
    let p05 = s[0].wrapping_add(s[5]);
    g_scalar(&mut s, 1, 6, 11, 12, 0, 0);
    g_scalar(&mut s, 2, 7, 8, 13, 0, 0);
    g_scalar(&mut s, 3, 4, 9, 14, 0, 0);
    PowFixed { dw, c: s, p05 }
}

fn word0_mask(bits: u32) -> u32 {
    debug_assert!((1..=32).contains(&bits));
    let full_bytes = bits / 8;
    let extra = bits % 8;
    let mut mask: u32 = 0;
    for i in 0..full_bytes {
        mask |= 0xFF << (8 * i);
    }
    if extra > 0 {
        mask |= ((0xFFu32 << (8 - extra)) & 0xFF) << (8 * full_bytes);
    }
    mask
}

#[derive(Clone, Copy)]
enum Msg3 {
    Z,
    S(uint32x4_t),
    P([uint32x4_t; 3]),
}

macro_rules! madd3 {
    ($v0:ident, $v1:ident, $v2:ident, $a:tt, $m:expr) => {
        match $m {
            Msg3::Z => {}
            Msg3::S(m) => {
                $v0[$a] = vaddq_u32($v0[$a], m);
                $v1[$a] = vaddq_u32($v1[$a], m);
                $v2[$a] = vaddq_u32($v2[$a], m);
            }
            Msg3::P(ms) => {
                $v0[$a] = vaddq_u32($v0[$a], ms[0]);
                $v1[$a] = vaddq_u32($v1[$a], ms[1]);
                $v2[$a] = vaddq_u32($v2[$a], ms[2]);
            }
        }
    };
}

#[inline(always)]
unsafe fn g3s(
    v0: &mut [uint32x4_t; 16],
    v1: &mut [uint32x4_t; 16],
    v2: &mut [uint32x4_t; 16],
    a: usize,
    b: usize,
    c: usize,
    d: usize,
    mx: Msg3,
    my: Msg3,
) {
    unsafe {
        v0[a] = vaddq_u32(v0[a], v0[b]);
        v1[a] = vaddq_u32(v1[a], v1[b]);
        v2[a] = vaddq_u32(v2[a], v2[b]);
        madd3!(v0, v1, v2, a, mx);
        v0[d] = rotate16(veorq_u32(v0[d], v0[a]));
        v1[d] = rotate16(veorq_u32(v1[d], v1[a]));
        v2[d] = rotate16(veorq_u32(v2[d], v2[a]));
        v0[c] = vaddq_u32(v0[c], v0[d]);
        v1[c] = vaddq_u32(v1[c], v1[d]);
        v2[c] = vaddq_u32(v2[c], v2[d]);
        v0[b] = rotate12(veorq_u32(v0[b], v0[c]));
        v1[b] = rotate12(veorq_u32(v1[b], v1[c]));
        v2[b] = rotate12(veorq_u32(v2[b], v2[c]));
        v0[a] = vaddq_u32(v0[a], v0[b]);
        v1[a] = vaddq_u32(v1[a], v1[b]);
        v2[a] = vaddq_u32(v2[a], v2[b]);
        madd3!(v0, v1, v2, a, my);
        v0[d] = rotate8(veorq_u32(v0[d], v0[a]));
        v1[d] = rotate8(veorq_u32(v1[d], v1[a]));
        v2[d] = rotate8(veorq_u32(v2[d], v2[a]));
        v0[c] = vaddq_u32(v0[c], v0[d]);
        v1[c] = vaddq_u32(v1[c], v1[d]);
        v2[c] = vaddq_u32(v2[c], v2[d]);
        v0[b] = rotate7(veorq_u32(v0[b], v0[c]));
        v1[b] = rotate7(veorq_u32(v1[b], v1[c]));
        v2[b] = rotate7(veorq_u32(v2[b], v2[c]));
    }
}

#[inline(always)]
unsafe fn g3s_a_only(
    v0: &mut [uint32x4_t; 16],
    v1: &mut [uint32x4_t; 16],
    v2: &mut [uint32x4_t; 16],
    a: usize,
    b: usize,
    c: usize,
    d: usize,
    mx: Msg3,
    my: Msg3,
) {
    unsafe {
        v0[a] = vaddq_u32(v0[a], v0[b]);
        v1[a] = vaddq_u32(v1[a], v1[b]);
        v2[a] = vaddq_u32(v2[a], v2[b]);
        madd3!(v0, v1, v2, a, mx);
        v0[d] = rotate16(veorq_u32(v0[d], v0[a]));
        v1[d] = rotate16(veorq_u32(v1[d], v1[a]));
        v2[d] = rotate16(veorq_u32(v2[d], v2[a]));
        v0[c] = vaddq_u32(v0[c], v0[d]);
        v1[c] = vaddq_u32(v1[c], v1[d]);
        v2[c] = vaddq_u32(v2[c], v2[d]);
        v0[b] = rotate12(veorq_u32(v0[b], v0[c]));
        v1[b] = rotate12(veorq_u32(v1[b], v1[c]));
        v2[b] = rotate12(veorq_u32(v2[b], v2[c]));
        v0[a] = vaddq_u32(v0[a], v0[b]);
        v1[a] = vaddq_u32(v1[a], v1[b]);
        v2[a] = vaddq_u32(v2[a], v2[b]);
        madd3!(v0, v1, v2, a, my);
    }
}

#[inline(always)]
unsafe fn g3s_c_only(
    v0: &mut [uint32x4_t; 16],
    v1: &mut [uint32x4_t; 16],
    v2: &mut [uint32x4_t; 16],
    a: usize,
    b: usize,
    c: usize,
    d: usize,
    mx: Msg3,
    my: Msg3,
) {
    unsafe {
        v0[a] = vaddq_u32(v0[a], v0[b]);
        v1[a] = vaddq_u32(v1[a], v1[b]);
        v2[a] = vaddq_u32(v2[a], v2[b]);
        madd3!(v0, v1, v2, a, mx);
        v0[d] = rotate16(veorq_u32(v0[d], v0[a]));
        v1[d] = rotate16(veorq_u32(v1[d], v1[a]));
        v2[d] = rotate16(veorq_u32(v2[d], v2[a]));
        v0[c] = vaddq_u32(v0[c], v0[d]);
        v1[c] = vaddq_u32(v1[c], v1[d]);
        v2[c] = vaddq_u32(v2[c], v2[d]);
        v0[b] = rotate12(veorq_u32(v0[b], v0[c]));
        v1[b] = rotate12(veorq_u32(v1[b], v1[c]));
        v2[b] = rotate12(veorq_u32(v2[b], v2[c]));
        v0[a] = vaddq_u32(v0[a], v0[b]);
        v1[a] = vaddq_u32(v1[a], v1[b]);
        v2[a] = vaddq_u32(v2[a], v2[b]);
        madd3!(v0, v1, v2, a, my);
        v0[d] = rotate8(veorq_u32(v0[d], v0[a]));
        v1[d] = rotate8(veorq_u32(v1[d], v1[a]));
        v2[d] = rotate8(veorq_u32(v2[d], v2[a]));
        v0[c] = vaddq_u32(v0[c], v0[d]);
        v1[c] = vaddq_u32(v1[c], v1[d]);
        v2[c] = vaddq_u32(v2[c], v2[d]);
    }
}

pub(crate) fn pow_scan_reg(
    state_digest: &[u8; 32],
    start: u64,
    len: u64,
    bits: u32,
) -> Option<u64> {
    if len == 0 {
        return None;
    }
    debug_assert!((1..=32).contains(&bits));
    let fx = pow_fixed(state_digest);
    let mask = word0_mask(bits);
    unsafe { pow_scan_reg_inner(&fx, start, len, mask) }
}

unsafe fn pow_scan_reg_inner(fx: &PowFixed, start: u64, len: u64, mask: u32) -> Option<u64> {
    unsafe {
        let dv: [uint32x4_t; 8] = core::array::from_fn(|j| vdupq_n_u32(fx.dw[j]));
        let cb: [uint32x4_t; 16] = core::array::from_fn(|i| vdupq_n_u32(fx.c[i]));
        let p05 = vdupq_n_u32(fx.p05);
        let maskv = vdupq_n_u32(mask);

        let end = start.saturating_add(len);
        let mut base = start;
        while base + 12 <= end {
            #[cfg(feature = "hash-count")]
            crate::challenger::fs_count::POW_SHA256
                .fetch_add(12, std::sync::atomic::Ordering::Relaxed);
            let mut lo = [[0u32; 4]; 3];
            let mut hi = [[0u32; 4]; 3];
            for i in 0..12 {
                let n = base + i as u64;
                lo[i / 4][i % 4] = n as u32;
                hi[i / 4][i % 4] = (n >> 32) as u32;
            }
            let n8 = [
                vld1q_u32(lo[0].as_ptr()),
                vld1q_u32(lo[1].as_ptr()),
                vld1q_u32(lo[2].as_ptr()),
            ];
            let n9 = [
                vld1q_u32(hi[0].as_ptr()),
                vld1q_u32(hi[1].as_ptr()),
                vld1q_u32(hi[2].as_ptr()),
            ];

            let mut v0 = cb;
            let mut v1 = cb;
            let mut v2 = cb;
            for (k, v) in [&mut v0, &mut v1, &mut v2].into_iter().enumerate() {
                let mut a = vaddq_u32(p05, n8[k]);
                let mut d = rotate16(veorq_u32(cb[15], a));
                let mut c = vaddq_u32(cb[10], d);
                let mut b = rotate12(veorq_u32(cb[5], c));
                a = vaddq_u32(vaddq_u32(a, b), n9[k]);
                d = rotate8(veorq_u32(d, a));
                c = vaddq_u32(c, d);
                b = rotate7(veorq_u32(b, c));
                v[0] = a;
                v[5] = b;
                v[10] = c;
                v[15] = d;
            }

            let (v0, v1, v2) = (&mut v0, &mut v1, &mut v2);
            g3s(v0, v1, v2, 0, 4, 8, 12, Msg3::S(dv[2]), Msg3::S(dv[6]));
            g3s(v0, v1, v2, 1, 5, 9, 13, Msg3::S(dv[3]), Msg3::Z);
            g3s(v0, v1, v2, 2, 6, 10, 14, Msg3::S(dv[7]), Msg3::S(dv[0]));
            g3s(v0, v1, v2, 3, 7, 11, 15, Msg3::S(dv[4]), Msg3::Z);
            g3s(v0, v1, v2, 0, 5, 10, 15, Msg3::S(dv[1]), Msg3::Z);
            g3s(v0, v1, v2, 1, 6, 11, 12, Msg3::Z, Msg3::S(dv[5]));
            g3s(v0, v1, v2, 2, 7, 8, 13, Msg3::P(n9), Msg3::Z);
            g3s(v0, v1, v2, 3, 4, 9, 14, Msg3::Z, Msg3::P(n8));

            g3s(v0, v1, v2, 0, 4, 8, 12, Msg3::S(dv[3]), Msg3::S(dv[4]));
            g3s(v0, v1, v2, 1, 5, 9, 13, Msg3::Z, Msg3::Z);
            g3s(v0, v1, v2, 2, 6, 10, 14, Msg3::Z, Msg3::S(dv[2]));
            g3s(v0, v1, v2, 3, 7, 11, 15, Msg3::S(dv[7]), Msg3::Z);
            g3s(v0, v1, v2, 0, 5, 10, 15, Msg3::S(dv[6]), Msg3::S(dv[5]));
            g3s(v0, v1, v2, 1, 6, 11, 12, Msg3::P(n9), Msg3::S(dv[0]));
            g3s(v0, v1, v2, 2, 7, 8, 13, Msg3::Z, Msg3::Z);
            g3s(v0, v1, v2, 3, 4, 9, 14, Msg3::P(n8), Msg3::S(dv[1]));

            g3s(v0, v1, v2, 0, 4, 8, 12, Msg3::Z, Msg3::S(dv[7]));
            g3s(v0, v1, v2, 1, 5, 9, 13, Msg3::Z, Msg3::P(n9));
            g3s(v0, v1, v2, 2, 6, 10, 14, Msg3::Z, Msg3::S(dv[3]));
            g3s(v0, v1, v2, 3, 7, 11, 15, Msg3::Z, Msg3::Z);
            g3s(v0, v1, v2, 0, 5, 10, 15, Msg3::S(dv[4]), Msg3::S(dv[0]));
            g3s(v0, v1, v2, 1, 6, 11, 12, Msg3::Z, Msg3::S(dv[2]));
            g3s(v0, v1, v2, 2, 7, 8, 13, Msg3::S(dv[5]), Msg3::P(n8));
            g3s(v0, v1, v2, 3, 4, 9, 14, Msg3::S(dv[1]), Msg3::S(dv[6]));

            g3s(v0, v1, v2, 0, 4, 8, 12, Msg3::Z, Msg3::Z);
            g3s(v0, v1, v2, 1, 5, 9, 13, Msg3::P(n9), Msg3::Z);
            g3s(v0, v1, v2, 2, 6, 10, 14, Msg3::Z, Msg3::Z);
            g3s(v0, v1, v2, 3, 7, 11, 15, Msg3::Z, Msg3::P(n8));
            g3s(v0, v1, v2, 0, 5, 10, 15, Msg3::S(dv[7]), Msg3::S(dv[2]));
            g3s(v0, v1, v2, 1, 6, 11, 12, Msg3::S(dv[5]), Msg3::S(dv[3]));
            g3s(v0, v1, v2, 2, 7, 8, 13, Msg3::S(dv[0]), Msg3::S(dv[1]));
            g3s(v0, v1, v2, 3, 4, 9, 14, Msg3::S(dv[6]), Msg3::S(dv[4]));

            g3s(v0, v1, v2, 0, 4, 8, 12, Msg3::P(n9), Msg3::Z);
            g3s(v0, v1, v2, 1, 5, 9, 13, Msg3::Z, Msg3::S(dv[5]));
            g3s(v0, v1, v2, 2, 6, 10, 14, Msg3::P(n8), Msg3::Z);
            g3s(v0, v1, v2, 3, 7, 11, 15, Msg3::Z, Msg3::S(dv[1]));
            g3s(v0, v1, v2, 0, 5, 10, 15, Msg3::Z, Msg3::S(dv[3]));
            g3s(v0, v1, v2, 1, 6, 11, 12, Msg3::S(dv[0]), Msg3::Z);
            g3s(v0, v1, v2, 2, 7, 8, 13, Msg3::S(dv[2]), Msg3::S(dv[6]));
            g3s(v0, v1, v2, 3, 4, 9, 14, Msg3::S(dv[4]), Msg3::S(dv[7]));

            g3s(v0, v1, v2, 0, 4, 8, 12, Msg3::Z, Msg3::Z);
            g3s(v0, v1, v2, 1, 5, 9, 13, Msg3::S(dv[5]), Msg3::S(dv[0]));
            g3s(v0, v1, v2, 2, 6, 10, 14, Msg3::S(dv[1]), Msg3::P(n9));
            g3s(v0, v1, v2, 3, 7, 11, 15, Msg3::P(n8), Msg3::S(dv[6]));
            g3s_a_only(v0, v1, v2, 0, 5, 10, 15, Msg3::Z, Msg3::Z);
            g3s_c_only(v0, v1, v2, 2, 7, 8, 13, Msg3::S(dv[3]), Msg3::S(dv[4]));

            let t0 = vandq_u32(veorq_u32(v0[0], v0[8]), maskv);
            let t1 = vandq_u32(veorq_u32(v1[0], v1[8]), maskv);
            let t2 = vandq_u32(veorq_u32(v2[0], v2[8]), maskv);
            if vminvq_u32(vminq_u32(vminq_u32(t0, t1), t2)) == 0 {
                let mut words = [0u32; 12];
                vst1q_u32(words[0..4].as_mut_ptr(), t0);
                vst1q_u32(words[4..8].as_mut_ptr(), t1);
                vst1q_u32(words[8..12].as_mut_ptr(), t2);
                for (i, w) in words.iter().enumerate() {
                    if *w == 0 {
                        return Some(base + i as u64);
                    }
                }
                unreachable!("vminvq said a lane matched");
            }
            base += 12;
        }
        while base < end {
            #[cfg(feature = "hash-count")]
            crate::challenger::fs_count::POW_SHA256
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut pre = [0u8; 64];
            for (i, w) in fx.dw.iter().enumerate() {
                pre[4 * i..4 * i + 4].copy_from_slice(&w.to_le_bytes());
            }
            pre[32..40].copy_from_slice(&base.to_le_bytes());
            let h = blake3::hash(&pre);
            let w0 = u32::from_le_bytes(h.as_bytes()[..4].try_into().unwrap());
            if w0 & mask == 0 {
                return Some(base);
            }
            base += 1;
        }
        None
    }
}

fn hash_complete_groups_flags(data: &[u8], out: &mut [[u8; 32]], flags: u32) -> usize {
    debug_assert_eq!(data.len(), out.len() * 64);
    let groups = out.len() / 12;
    if groups == 0 {
        return 0;
    }

    unsafe {
        let zero = vdupq_n_u32(0);
        let iv = [
            vdupq_n_u32(IV[0]),
            vdupq_n_u32(IV[1]),
            vdupq_n_u32(IV[2]),
            vdupq_n_u32(IV[3]),
            vdupq_n_u32(IV[4]),
            vdupq_n_u32(IV[5]),
            vdupq_n_u32(IV[6]),
            vdupq_n_u32(IV[7]),
        ];
        let init = [
            iv[0],
            iv[1],
            iv[2],
            iv[3],
            iv[4],
            iv[5],
            iv[6],
            iv[7],
            iv[0],
            iv[1],
            iv[2],
            iv[3],
            zero,
            zero,
            vdupq_n_u32(64),
            vdupq_n_u32(flags),
        ];
        for group in 0..groups {
            let input = data.as_ptr().add(group * 12 * 64);
            let mut m0 = [zero; 16];
            let mut m1 = [zero; 16];
            let mut m2 = [zero; 16];
            transpose_parent4(input, &mut m0);
            transpose_parent4(input.add(4 * 64), &mut m1);
            transpose_parent4(input.add(8 * 64), &mut m2);

            let mut v0 = init;
            let mut v1 = init;
            let mut v2 = init;
            for schedule in &MSG_SCHEDULE {
                round3(&mut v0, &mut v1, &mut v2, &m0, &m1, &m2, schedule);
            }

            let output = out.as_mut_ptr().cast::<u8>().add(group * 12 * 32);
            store_cv4(&mut v0, output);
            store_cv4(&mut v1, output.add(4 * 32));
            store_cv4(&mut v2, output.add(8 * 32));
        }
    }
    groups * 12
}
