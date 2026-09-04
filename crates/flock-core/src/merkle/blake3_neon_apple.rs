//! Eight-leaf BLAKE3 chunk kernel for Apple AArch64.
//!
//! Upstream BLAKE3's NEON `hash_many` hashes four 1 KiB chunks through all
//! sixteen dependent compression blocks before starting the next four. The
//! generated kernel below keeps two independent four-lane states in flight
//! and rotates between them after every BLAKE3 round. This exposes enough
//! independent add/xor/rotate chains to fill Apple P-core execution slots
//! while fitting substantially more of the compression state in registers.
//!
//! The assembly is compiler-generated from the same BLAKE3 1.8.5 NEON
//! primitives linked by this crate, with `-O3 -mcpu=apple-m3`. It fixes the
//! exact Merkle-leaf contract used here: eight contiguous 1024-byte unkeyed
//! chunks, counter zero, `CHUNK_START | CHUNK_END`, 32 output bytes each.

use core::arch::aarch64::*;

core::arch::global_asm!(include_str!("blake3_neon8_macos.S"), options(raw));

unsafe extern "C" {
    fn flock_blake3_hash8_neon_1024(data: *const u8, out: *mut u8, groups: usize);
}

/// Hash as many complete groups of eight 1 KiB leaves as fit in `out`.
///
/// Returns the number of leaves written. The caller handles the tail through
/// upstream `hash_many`, which also makes arbitrary Rayon partition sizes
/// safe without padding or over-read.
#[inline]
pub(super) fn hash_complete_groups(data: &[u8], out: &mut [[u8; 32]]) -> usize {
    debug_assert_eq!(data.len(), out.len() * 1024);
    let groups = out.len() / 8;
    if groups == 0 {
        return 0;
    }

    // SAFETY: each group consumes exactly 8 * 1024 initialized bytes and
    // writes exactly 8 * 32 bytes. `groups` is floor(out.len() / 8), and
    // the debug assertion records the data/output correspondence established
    // by the Merkle caller. The kernel is compiled only for Apple AArch64,
    // where NEON is mandatory.
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

/// Hash complete groups of twelve contiguous 64-byte BLAKE3 parent blocks.
///
/// This intrinsic path is Apple-only through the parent module's cfg. Tails
/// remain on upstream `hash_many`, so arbitrary Rayon partitions are safe.
#[inline]
pub(super) fn hash_complete_parent_groups(data: &[u8], out: &mut [[u8; 32]]) -> usize {
    hash_complete_groups_flags(data, out, 4)
}

/// Twelve-way PoW grind blocks: 64-byte single-chunk pre-images with
/// `CHUNK_START | CHUNK_END | ROOT` (11) instead of the Merkle `PARENT`
/// flag (4) — the same layout, IV, counter-zero, and 32-byte output
/// contract as [`hash_complete_parent_groups`], so each output agrees with
/// `blake3::hash` on the same 64-byte pre-image.
pub(crate) fn hash_complete_pow_groups(data: &[u8], out: &mut [[u8; 32]]) -> usize {
    hash_complete_groups_flags(data, out, 11)
}

// ---------------------------------------------------------------------------
// Register-resident PoW grind kernel.
//
// The PoW pre-image is one 64-byte single-chunk block whose message words are
//   m[0..8]  = the grind's 32-byte transcript state digest — FIXED for the
//              entire grind (hundreds of thousands of attempts),
//   m[8..10] = the u64 nonce, the ONLY thing that varies per attempt,
//   m[10..16]= zero padding.
// The generic twelve-way kernel above treats every block as arbitrary bytes:
// it materializes each 64-byte pre-image in memory, loads and transposes all
// 16 message words per lane, runs all 112 message adds, and stores + transposes
// the full 32-byte digest, which a scalar loop then re-reads to test the
// leading-zero predicate. For the grind shape almost all of that is invariant:
//
//   * The 8 digest words are broadcast once per scan and stay in registers —
//     no pre-image buffers, no loads, no transposes.
//   * Round 1's column half uses only m[0..8] (fixed) and its diagonal G's on
//     rows (1,6,11,12), (2,7,8,13), (3,4,9,14) use only m[10..16] (zero), so
//     the whole first round except the single nonce-fed G at (0,5,10,15) is
//     precomputed scalar, once per grind.
//   * The 6 zero message words elide 6 of 16 message adds in every later round.
//   * The grind only needs the PREDICATE "hash has >= bits leading zeros", and
//     for bits <= 32 that lives entirely in output word 0 = v[0] ^ v[8]. No
//     digest is stored; the last round computes only what feeds words 0 and 8
//     (its (1,6,11,12) and (3,4,9,14) diagonal G's are dropped, the other two
//     diagonal G's are truncated to the ops on their dependency chains).
//
// Everything is bit-exact against `blake3::hash` of the 64-byte pre-image —
// `grind_reg_scan_matches_generic` in `challenger.rs` holds the two paths
// together, and the smallest-nonce selection rule is untouched (the scan
// returns the smallest in-range match, exactly like `blake3_pow_scan`).
// ---------------------------------------------------------------------------

const POW_FLAGS: u32 = 11; // CHUNK_START | CHUNK_END | ROOT

/// Fixed, nonce-independent part of the PoW compression for one grind.
struct PowFixed {
    /// Digest message words `m[0..8]`.
    dw: [u32; 8],
    /// State after round 1's column half plus its three nonce-independent
    /// diagonal G's. Entries 0, 5, 10, 15 hold the PRE-diagonal (column-half)
    /// values — the per-attempt nonce G at (0,5,10,15) starts from them.
    c: [u32; 16],
    /// `c[0] + c[5]`, the nonce G's first add, folded ahead of time.
    p05: u32,
}

/// Scalar BLAKE3 G, for the once-per-grind fixed-round precompute.
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
        IV[0], IV[1], IV[2], IV[3], IV[4], IV[5], IV[6], IV[7], // h
        IV[0], IV[1], IV[2], IV[3], // iv
        0, 0,  // counter (always 0: single chunk)
        64, // block length
        POW_FLAGS,
    ];
    // Round 1 column half: messages m[0..8] = digest words, all fixed.
    g_scalar(&mut s, 0, 4, 8, 12, dw[0], dw[1]);
    g_scalar(&mut s, 1, 5, 9, 13, dw[2], dw[3]);
    g_scalar(&mut s, 2, 6, 10, 14, dw[4], dw[5]);
    g_scalar(&mut s, 3, 7, 11, 15, dw[6], dw[7]);
    let p05 = s[0].wrapping_add(s[5]);
    // Round 1 diagonal half, minus the nonce G at (0,5,10,15): the other
    // three G's take m[10..16] = 0 and touch disjoint word sets, so they are
    // fixed and leave (0,5,10,15) at their column-half values.
    g_scalar(&mut s, 1, 6, 11, 12, 0, 0);
    g_scalar(&mut s, 2, 7, 8, 13, 0, 0);
    g_scalar(&mut s, 3, 4, 9, 14, 0, 0);
    PowFixed { dw, c: s, p05 }
}

/// Leading-zero predicate mask over output word 0 (little-endian byte order),
/// valid for `1 <= bits <= 32`: `word0 & mask == 0` iff the 32-byte hash has
/// at least `bits` leading zero bits (matching `has_leading_zero_bits`).
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

/// Message operand for one three-state G: absent (zero word), shared across
/// all lanes (digest word broadcast), or per-state (nonce lanes). Always
/// passed as a literal variant from an `#[inline(always)]` caller, so the
/// match folds away at compile time — no runtime branches in the kernel.
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

/// One G over three four-lane states with zero-elided message adds.
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

/// Truncated G whose only live output is `v[a]` (last round, feeding output
/// word 0): everything after the second `a` add is dead.
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

/// Truncated G whose only live output is `v[c]` (last round, feeding output
/// word 8): only the final `b` rotate is dead.
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

/// Smallest nonce in `start .. start + len` whose BLAKE3 PoW hash has `bits`
/// leading zero bits, or `None` — the register-resident specialization of
/// `blake3_pow_scan` for `1 <= bits <= 32`. Byte-exact against `blake3::hash`
/// of the 64-byte pre-image for every nonce.
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
        // Broadcast the grind-invariant state once; these stay live across
        // every attempt in the scan.
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
            // Nonce lanes: scalar build handles the (astronomically rare)
            // 32-bit carry between lanes for free.
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

            // Round 1 = just the nonce G at (0,5,10,15), from precomputed
            // column-half inputs; all other words come in as constants.
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
            // round 2 (schedule row 1)
            g3s(v0, v1, v2, 0, 4, 8, 12, Msg3::S(dv[2]), Msg3::S(dv[6]));
            g3s(v0, v1, v2, 1, 5, 9, 13, Msg3::S(dv[3]), Msg3::Z);
            g3s(v0, v1, v2, 2, 6, 10, 14, Msg3::S(dv[7]), Msg3::S(dv[0]));
            g3s(v0, v1, v2, 3, 7, 11, 15, Msg3::S(dv[4]), Msg3::Z);
            g3s(v0, v1, v2, 0, 5, 10, 15, Msg3::S(dv[1]), Msg3::Z);
            g3s(v0, v1, v2, 1, 6, 11, 12, Msg3::Z, Msg3::S(dv[5]));
            g3s(v0, v1, v2, 2, 7, 8, 13, Msg3::P(n9), Msg3::Z);
            g3s(v0, v1, v2, 3, 4, 9, 14, Msg3::Z, Msg3::P(n8));

            // round 3 (schedule row 2)
            g3s(v0, v1, v2, 0, 4, 8, 12, Msg3::S(dv[3]), Msg3::S(dv[4]));
            g3s(v0, v1, v2, 1, 5, 9, 13, Msg3::Z, Msg3::Z);
            g3s(v0, v1, v2, 2, 6, 10, 14, Msg3::Z, Msg3::S(dv[2]));
            g3s(v0, v1, v2, 3, 7, 11, 15, Msg3::S(dv[7]), Msg3::Z);
            g3s(v0, v1, v2, 0, 5, 10, 15, Msg3::S(dv[6]), Msg3::S(dv[5]));
            g3s(v0, v1, v2, 1, 6, 11, 12, Msg3::P(n9), Msg3::S(dv[0]));
            g3s(v0, v1, v2, 2, 7, 8, 13, Msg3::Z, Msg3::Z);
            g3s(v0, v1, v2, 3, 4, 9, 14, Msg3::P(n8), Msg3::S(dv[1]));

            // round 4 (schedule row 3)
            g3s(v0, v1, v2, 0, 4, 8, 12, Msg3::Z, Msg3::S(dv[7]));
            g3s(v0, v1, v2, 1, 5, 9, 13, Msg3::Z, Msg3::P(n9));
            g3s(v0, v1, v2, 2, 6, 10, 14, Msg3::Z, Msg3::S(dv[3]));
            g3s(v0, v1, v2, 3, 7, 11, 15, Msg3::Z, Msg3::Z);
            g3s(v0, v1, v2, 0, 5, 10, 15, Msg3::S(dv[4]), Msg3::S(dv[0]));
            g3s(v0, v1, v2, 1, 6, 11, 12, Msg3::Z, Msg3::S(dv[2]));
            g3s(v0, v1, v2, 2, 7, 8, 13, Msg3::S(dv[5]), Msg3::P(n8));
            g3s(v0, v1, v2, 3, 4, 9, 14, Msg3::S(dv[1]), Msg3::S(dv[6]));

            // round 5 (schedule row 4)
            g3s(v0, v1, v2, 0, 4, 8, 12, Msg3::Z, Msg3::Z);
            g3s(v0, v1, v2, 1, 5, 9, 13, Msg3::P(n9), Msg3::Z);
            g3s(v0, v1, v2, 2, 6, 10, 14, Msg3::Z, Msg3::Z);
            g3s(v0, v1, v2, 3, 7, 11, 15, Msg3::Z, Msg3::P(n8));
            g3s(v0, v1, v2, 0, 5, 10, 15, Msg3::S(dv[7]), Msg3::S(dv[2]));
            g3s(v0, v1, v2, 1, 6, 11, 12, Msg3::S(dv[5]), Msg3::S(dv[3]));
            g3s(v0, v1, v2, 2, 7, 8, 13, Msg3::S(dv[0]), Msg3::S(dv[1]));
            g3s(v0, v1, v2, 3, 4, 9, 14, Msg3::S(dv[6]), Msg3::S(dv[4]));

            // round 6 (schedule row 5)
            g3s(v0, v1, v2, 0, 4, 8, 12, Msg3::P(n9), Msg3::Z);
            g3s(v0, v1, v2, 1, 5, 9, 13, Msg3::Z, Msg3::S(dv[5]));
            g3s(v0, v1, v2, 2, 6, 10, 14, Msg3::P(n8), Msg3::Z);
            g3s(v0, v1, v2, 3, 7, 11, 15, Msg3::Z, Msg3::S(dv[1]));
            g3s(v0, v1, v2, 0, 5, 10, 15, Msg3::Z, Msg3::S(dv[3]));
            g3s(v0, v1, v2, 1, 6, 11, 12, Msg3::S(dv[0]), Msg3::Z);
            g3s(v0, v1, v2, 2, 7, 8, 13, Msg3::S(dv[2]), Msg3::S(dv[6]));
            g3s(v0, v1, v2, 3, 4, 9, 14, Msg3::S(dv[4]), Msg3::S(dv[7]));

            // round 7 (schedule row 6) — only output word 0 = v[0] ^ v[8] is
            // live. Column G's all feed the two diagonal G's kept below; the
            // diagonal G's at (1,6,11,12) and (3,4,9,14) touch neither word 0
            // nor word 8 and are dropped.
            g3s(v0, v1, v2, 0, 4, 8, 12, Msg3::Z, Msg3::Z);
            g3s(v0, v1, v2, 1, 5, 9, 13, Msg3::S(dv[5]), Msg3::S(dv[0]));
            g3s(v0, v1, v2, 2, 6, 10, 14, Msg3::S(dv[1]), Msg3::P(n9));
            g3s(v0, v1, v2, 3, 7, 11, 15, Msg3::P(n8), Msg3::S(dv[6]));
            g3s_a_only(v0, v1, v2, 0, 5, 10, 15, Msg3::Z, Msg3::Z);
            g3s_c_only(v0, v1, v2, 2, 7, 8, 13, Msg3::S(dv[3]), Msg3::S(dv[4]));

            // Predicate: any lane with (word0 & mask) == 0.
            let t0 = vandq_u32(veorq_u32(v0[0], v0[8]), maskv);
            let t1 = vandq_u32(veorq_u32(v1[0], v1[8]), maskv);
            let t2 = vandq_u32(veorq_u32(v2[0], v2[8]), maskv);
            if vminvq_u32(vminq_u32(vminq_u32(t0, t1), t2)) == 0 {
                // Rare path: recover the smallest matching lane in nonce order.
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
        // Tail (< 12 lanes): scalar, byte-exact by construction.
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

/// Shared driver: hash complete groups of twelve contiguous 64-byte blocks
/// with the given BLAKE3 domain `flags`. Returns the number of outputs
/// written; the caller handles the tail through upstream `hash_many`.
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
