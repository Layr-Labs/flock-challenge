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

/// [`transpose_parent4`] generalized to messages `stride` bytes apart: load
/// one 64-byte block from each of four lanes at `base`, `base + stride`,
/// `base + 2·stride`, `base + 3·stride` and transpose into sixteen
/// word-across-lanes vectors. `stride` is always a literal at the call site,
/// so the address arithmetic constant-folds exactly like the parent path.
#[inline(always)]
unsafe fn transpose_stride4(base: *const u8, stride: usize, out: &mut [uint32x4_t; 16]) {
    unsafe {
        for word_vec in 0..4 {
            let word_offset = word_vec * 16;
            let mut rows = [
                vreinterpretq_u32_u8(vld1q_u8(base.add(word_offset))),
                vreinterpretq_u32_u8(vld1q_u8(base.add(stride + word_offset))),
                vreinterpretq_u32_u8(vld1q_u8(base.add(2 * stride + word_offset))),
                vreinterpretq_u32_u8(vld1q_u8(base.add(3 * stride + word_offset))),
            ];
            transpose4(&mut rows);
            out[word_vec * 4..word_vec * 4 + 4].copy_from_slice(&rows);
        }
    }
}

/// Rebuild a compression state for the next block of the same chunk: rows
/// 0..8 take the chaining value `v[i] ^ v[i+8]` of the block just
/// compressed, rows 8..12 the IV, counter stays zero, `block_len` 64, and
/// `flags` selects the new block's domain bits.
#[inline(always)]
unsafe fn chain_state(
    v: &[uint32x4_t; 16],
    iv: &[uint32x4_t; 8],
    zero: uint32x4_t,
    flags: uint32x4_t,
) -> [uint32x4_t; 16] {
    unsafe {
        [
            veorq_u32(v[0], v[8]),
            veorq_u32(v[1], v[9]),
            veorq_u32(v[2], v[10]),
            veorq_u32(v[3], v[11]),
            veorq_u32(v[4], v[12]),
            veorq_u32(v[5], v[13]),
            veorq_u32(v[6], v[14]),
            veorq_u32(v[7], v[15]),
            iv[0],
            iv[1],
            iv[2],
            iv[3],
            zero,
            zero,
            vdupq_n_u32(64),
            flags,
        ]
    }
}

/// Hash complete groups of twelve contiguous 128-byte leaves.
///
/// Each leaf is a single two-block chunk: block 0 compressed from the IV
/// with `CHUNK_START` (1), block 1 chained through the block-0 CV with
/// `CHUNK_END` (2), counter zero throughout, 32-byte non-root CV out —
/// byte-identical to `blake3_leaf_cv` on the same 128 bytes. Twelve leaves
/// run as three interleaved four-lane states, exactly the ILP shape of
/// [`hash_complete_parent_groups`], with two sequential compressions per
/// leaf. Tails remain on upstream `hash_many`, so arbitrary Rayon
/// partitions are safe.
pub(super) fn hash_complete_leaf128_groups(data: &[u8], out: &mut [[u8; 32]]) -> usize {
    debug_assert_eq!(data.len(), out.len() * 128);
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
        // Block 0 of a fresh chunk: CV = IV, counter zero, block_len 64,
        // CHUNK_START.
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
            vdupq_n_u32(1),
        ];
        for group in 0..groups {
            let input = data.as_ptr().add(group * 12 * 128);
            let mut m0 = [zero; 16];
            let mut m1 = [zero; 16];
            let mut m2 = [zero; 16];
            transpose_stride4(input, 128, &mut m0);
            transpose_stride4(input.add(4 * 128), 128, &mut m1);
            transpose_stride4(input.add(8 * 128), 128, &mut m2);

            let mut v0 = init;
            let mut v1 = init;
            let mut v2 = init;
            for schedule in &MSG_SCHEDULE {
                round3(&mut v0, &mut v1, &mut v2, &m0, &m1, &m2, schedule);
            }

            // Chain the block-0 CVs into block 1, flagged CHUNK_END.
            let flags_end = vdupq_n_u32(2);
            let mut v0 = chain_state(&v0, &iv, zero, flags_end);
            let mut v1 = chain_state(&v1, &iv, zero, flags_end);
            let mut v2 = chain_state(&v2, &iv, zero, flags_end);

            transpose_stride4(input.add(64), 128, &mut m0);
            transpose_stride4(input.add(4 * 128 + 64), 128, &mut m1);
            transpose_stride4(input.add(8 * 128 + 64), 128, &mut m2);
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
