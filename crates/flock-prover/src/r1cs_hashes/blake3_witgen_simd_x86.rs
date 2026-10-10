//! x86_64 SSE2 port of the aarch64 NEON quad witgen builder.
//!
//! The aarch64 `witgen_simd` module (W-H2, `notes/witgen-simd.md`) runs the
//! BLAKE3 block-witness materialization as four compressions in u32-lane
//! lockstep ("quad"), which is ~33% of the x86 witgen wall time because the
//! per-block builder stays scalar on x86_64 (`#[cfg(target_arch = "aarch64")]`
//! gates every SIMD arm). This module is the 1:1 SSE2 equivalent: every NEON
//! intrinsic used by `build_quad_witness_ab_stream_neon_elide` maps to an
//! exact SSE2 twin, so the output must be bit-identical to four scalar
//! `build_block_witness_ab_stream_into` calls.
//!
//! SSE2 is the x86_64 baseline (no `target_feature` gate needed), and all
//! stores are plain (the NEON NT-store pair is a cache-policy optimization,
//! not a correctness component). Constant-region elision and the Seeded SoA
//! input path are intentionally omitted in this first port; the full-write
//! `QuadInput::Blocks` kernel is the correctness-critical core that every
//! other arm reduces to.
#![cfg(target_arch = "x86_64")]

use core::arch::x86_64::*;

use super::{
    BLAKE3_IV, Compression, G_STRIDE, GS_BASE, N_G, OUT_HI_BASE, REC_C0, REC_C1, REC_C2, REC_C3,
    REC_C4, REC_C5, REC_LIN0, REC_LIN1, USEFUL_BITS,
};

/// One 128-bit vector = one u32 word replicated across the quad's four blocks.
type V4 = __m128i;

const U32_PER_BLOCK: usize = super::K / 32; // 512
/// [`dump`] drains a block in 64 chunks of 8 u32 words (32 bytes).
const DUMP_CHUNKS: usize = U32_PER_BLOCK / 8; // 64
/// Partial final word 481 (`PackedWordWriter::finish` semantics).
const LAST_WORD: usize = (USEFUL_BITS - 1) / 32; // 481

// ---------------------------------------------------------------------------
// 1:1 NEON -> SSE2 op wrappers (bit-exact by construction).
// ---------------------------------------------------------------------------

#[inline(always)]
unsafe fn vld1q(p: *const u32) -> V4 {
    unsafe { _mm_loadu_si128(p as *const __m128i) }
}
#[inline(always)]
unsafe fn vst1q(p: *mut u32, v: V4) {
    unsafe { _mm_storeu_si128(p as *mut __m128i, v) }
}
#[inline(always)]
fn vdupq(x: u32) -> V4 {
    unsafe { _mm_set1_epi32(x as i32) }
}
#[inline(always)]
fn vaddq(a: V4, b: V4) -> V4 {
    unsafe { _mm_add_epi32(a, b) }
}
#[inline(always)]
fn veorq(a: V4, b: V4) -> V4 {
    unsafe { _mm_xor_si128(a, b) }
}
#[inline(always)]
fn vandq(a: V4, b: V4) -> V4 {
    unsafe { _mm_and_si128(a, b) }
}
#[inline(always)]
fn vorrq(a: V4, b: V4) -> V4 {
    unsafe { _mm_or_si128(a, b) }
}
#[inline(always)]
unsafe fn vshrq_n<const N: i32>(v: V4) -> V4 {
    unsafe { _mm_srli_epi32::<N>(v) }
}
#[inline(always)]
unsafe fn vshlq_n<const N: i32>(v: V4) -> V4 {
    unsafe { _mm_slli_epi32::<N>(v) }
}
/// `VSLI.32`: `(b << USED) | (a & (2^USED - 1))` — keeps the already-final low
/// USED bits of `a` (the pending lane), overwrites every following bit with
/// the shifted-in new field from `b`.
#[inline(always)]
fn vsliq_n<const USED: i32>(a: V4, b: V4) -> V4 {
    const {
        assert!(USED >= 0 && USED < 32);
    }
    if USED == 0 {
        b
    } else {
        unsafe {
            _mm_or_si128(
                _mm_slli_epi32::<USED>(b),
                _mm_and_si128(a, _mm_set1_epi32(((1u32 << USED) - 1) as i32)),
            )
        }
    }
}

/// Fixed 4x4 u32 transpose (both orientations use the same network):
/// `(word w across 4 blocks) <-> (block j's 4 consecutive words)`.
/// NEON `vtrn1q_u32` = `_mm_unpacklo_epi32`, `vtrn2q_u32` =
/// `_mm_unpackhi_epi32`; the 64-bit stage is `unpacklo/hi_epi64`.
#[inline(always)]
fn tr4(w0: V4, w1: V4, w2: V4, w3: V4) -> (V4, V4, V4, V4) {
    unsafe {
        let t0 = _mm_unpacklo_epi32(w0, w1);
        let t1 = _mm_unpackhi_epi32(w0, w1);
        let t2 = _mm_unpacklo_epi32(w2, w3);
        let t3 = _mm_unpackhi_epi32(w2, w3);
        (
            _mm_unpacklo_epi64(t0, t2),
            _mm_unpackhi_epi64(t0, t2),
            _mm_unpacklo_epi64(t1, t3),
            _mm_unpackhi_epi64(t1, t3),
        )
    }
}

/// `vld4q_u32(ptr)`: 16 consecutive u32 deinterleaved into four lane vectors —
/// exactly `load 4 V4s then tr4` (verified bit-exact against the NEON `ld4`).
#[inline(always)]
unsafe fn vld4q(ptr: *const u32) -> (V4, V4, V4, V4) {
    let v0 = unsafe { vld1q(ptr) };
    let v1 = unsafe { vld1q(ptr.add(4)) };
    let v2 = unsafe { vld1q(ptr.add(8)) };
    let v3 = unsafe { vld1q(ptr.add(12)) };
    tr4(v0, v1, v2, v3)
}

/// `(x ^ y).rotate_right(N)` — NEON has no vector ROR; shr/shl/or is exact.
/// M = 32 − N is spelled out at the call site.
#[inline(always)]
fn xor_rotr<const N: i32, const M: i32>(x: V4, y: V4) -> V4 {
    debug_assert_eq!(N + M, 32);
    unsafe {
        let v = veorq(x, y);
        vorrq(vshrq_n::<N>(v), vshlq_n::<M>(v))
    }
}

/// Lane-wise `add_carry_parts`: `(sum, left, right, carry_aux)`.
#[inline(always)]
fn add_carry_parts_v(x: V4, y: V4) -> (V4, V4, V4, V4) {
    let sum = vaddq(x, y);
    let cin = veorq(veorq(sum, x), y);
    let left = veorq(x, cin);
    let right = veorq(y, cin);
    let carry = vandq(left, right);
    (sum, left, right, carry)
}

/// u32-granular lane-wise `PackedWordWriter` — the vector analogue of the
/// scalar builder's fully-unrolled writer (see the aarch64 doc).
struct W32 {
    pending: V4,
    stage: *mut V4, // 512 block-lane words for this buffer's quad
}

impl W32 {
    #[inline(always)]
    fn at(stage: *mut V4, pending: V4) -> Self {
        Self { pending, stage }
    }

    #[inline(always)]
    unsafe fn push<const USED: i32, const WIDTH: i32, const BACK: i32, const WORD: usize>(
        &mut self,
        v: V4,
    ) {
        const {
            assert!(USED >= 0 && USED < 32);
            assert!(WIDTH == 31 || WIDTH == 32);
            assert!(BACK >= 1 && BACK < 32);
            assert!(WORD < U32_PER_BLOCK);
        }
        debug_assert!(USED + WIDTH <= 32 || BACK == 32 - USED);
        unsafe {
            if USED == 0 {
                if WIDTH == 32 {
                    vst1q(self.stage.add(WORD) as *mut u32, v);
                    self.pending = vdupq(0);
                } else {
                    self.pending = v;
                }
            } else if USED + WIDTH < 32 {
                self.pending = vsliq_n::<USED>(self.pending, v);
            } else {
                let out = vsliq_n::<USED>(self.pending, v);
                vst1q(self.stage.add(WORD) as *mut u32, out);
                if USED + WIDTH == 32 {
                    self.pending = vdupq(0);
                } else {
                    self.pending = vshrq_n::<BACK>(v);
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn finish(&mut self) {
        unsafe {
            vst1q(self.stage.add(LAST_WORD) as *mut u32, self.pending);
        }
    }
}

/// Stream-sequential field push at absolute bit position `$pos` (consts
/// monomorphized at the call site, exactly like the aarch64 `pushf!`).
macro_rules! pushf {
    ($w:ident, $pos:expr, $width:literal, $v:expr) => {{
        $w.push::<{ ($pos % 32) as i32 }, $width, {
            let u = ($pos % 32) as i32;
            if u == 0 { 1 } else { 32 - u }
        }, { $pos / 32 }>($v);
    }};
}

/// Drain a 512-word block-lane stage to the four row-major block
/// destinations (plain stores; the NEON NT-store pair is cache-policy only).
/// A dump chunk `g` covers u32 words `8g..8g+8` of every block in the quad.
#[inline(always)]
unsafe fn dump_range(stage: *const V4, dst: *mut u32, g0: usize, g1: usize) {
    unsafe {
        for g in g0..g1 {
            let w = 8 * g;
            let (x0, x1, x2, x3) = vld4q(stage.add(w) as *const u32);
            let (y0, y1, y2, y3) = vld4q(stage.add(w + 4) as *const u32);
            let p0 = dst.add(w);
            let p1 = dst.add(U32_PER_BLOCK + w);
            let p2 = dst.add(2 * U32_PER_BLOCK + w);
            let p3 = dst.add(3 * U32_PER_BLOCK + w);
            vst1q(p0, x0);
            vst1q(p0.add(4), y0);
            vst1q(p1, x1);
            vst1q(p1.add(4), y1);
            vst1q(p2, x2);
            vst1q(p2.add(4), y2);
            vst1q(p3, x3);
            vst1q(p3.add(4), y3);
        }
    }
}

/// Build the (z, a, b) blocks for FOUR compressions in u32-lane lockstep,
/// fully writing every word (the "incumbent full write": elision off, plain
/// stores). `z`/`a`/`b` point at the quad's first block; block j occupies
/// `dst + j*512 .. +512` u32 words.
///
/// Bit-exact with [`super::build_block_witness_ab_stream_into`] x4 — pinned
/// by `x86_quad_witness_matches_scalar_stream_builder`.
///
/// `#[inline(never)]`: this is a ~500-line unrolled kernel called once per
/// quad in a loop; keeping it out of the crate's inliner budget also keeps
/// unrelated codegen (and therefore stack depth elsewhere) at baseline.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn build_quad_witness_ab_stream_x86(
    inputs: [&Compression; 4],
    z: *mut u32,
    a: *mut u32,
    b: *mut u32,
) {
    unsafe {
        // ---- input gather: AoS loads + fixed 4x4 transpose networks ----
        let ptrs = [
            inputs[0].0.as_ptr(),
            inputs[1].0.as_ptr(),
            inputs[2].0.as_ptr(),
            inputs[3].0.as_ptr(),
        ];
        let (cv0, cv1, cv2, cv3) = tr4(
            vld1q(ptrs[0]),
            vld1q(ptrs[1]),
            vld1q(ptrs[2]),
            vld1q(ptrs[3]),
        );
        let (cv4, cv5, cv6, cv7) = tr4(
            vld1q(ptrs[0].add(4)),
            vld1q(ptrs[1].add(4)),
            vld1q(ptrs[2].add(4)),
            vld1q(ptrs[3].add(4)),
        );
        let cv_v = [cv0, cv1, cv2, cv3, cv4, cv5, cv6, cv7];
        let mptrs = [
            inputs[0].1.as_ptr(),
            inputs[1].1.as_ptr(),
            inputs[2].1.as_ptr(),
            inputs[3].1.as_ptr(),
        ];
        let mut m: [V4; 16] = [cv0; 16];
        for wgrp in 0..4 {
            let (m0, m1, m2, m3) = tr4(
                vld1q(mptrs[0].add(4 * wgrp)),
                vld1q(mptrs[1].add(4 * wgrp)),
                vld1q(mptrs[2].add(4 * wgrp)),
                vld1q(mptrs[3].add(4 * wgrp)),
            );
            m[4 * wgrp] = m0;
            m[4 * wgrp + 1] = m1;
            m[4 * wgrp + 2] = m2;
            m[4 * wgrp + 3] = m3;
        }
        let mut tlo_a = [0u32; 4];
        let mut thi_a = [0u32; 4];
        let mut bl_a = [0u32; 4];
        let mut fl_a = [0u32; 4];
        for j in 0..4 {
            tlo_a[j] = inputs[j].2 as u32;
            thi_a[j] = (inputs[j].2 >> 32) as u32;
            bl_a[j] = inputs[j].3;
            fl_a[j] = inputs[j].4;
        }
        let tlo = vld1q(tlo_a.as_ptr());
        let thi = vld1q(thi_a.as_ptr());
        let blen = vld1q(bl_a.as_ptr());
        let flags = vld1q(fl_a.as_ptr());

        let mut state: [V4; 16] = [
            cv_v[0],
            cv_v[1],
            cv_v[2],
            cv_v[3],
            cv_v[4],
            cv_v[5],
            cv_v[6],
            cv_v[7],
            vdupq(BLAKE3_IV[0]),
            vdupq(BLAKE3_IV[1]),
            vdupq(BLAKE3_IV[2]),
            vdupq(BLAKE3_IV[3]),
            tlo,
            thi,
            blen,
            flags,
        ];

        // ---- L1 stages (block-lane words; drained by `dump` at the end so
        // each block's 2 KiB is one ascending burst) ----
        let zero = vdupq(0);
        let mut zs = core::mem::MaybeUninit::<[V4; U32_PER_BLOCK]>::uninit();
        let mut ast = core::mem::MaybeUninit::<[V4; U32_PER_BLOCK]>::uninit();
        let mut bs = core::mem::MaybeUninit::<[V4; U32_PER_BLOCK]>::uninit();
        let zs = zs.as_mut_ptr().cast::<V4>();
        let ast = ast.as_mut_ptr().cast::<V4>();
        let bs = bs.as_mut_ptr().cast::<V4>();

        // ---- prefix (bits 0..1153), straight into the stages ----
        for w in 0..8usize {
            vst1q(zs.add(w) as *mut u32, cv_v[w]);
            vst1q(ast.add(w) as *mut u32, cv_v[w]);
        }
        let maxv = vdupq(u32::MAX);
        for w in 0..36usize {
            vst1q(bs.add(w) as *mut u32, maxv);
        }
        let one = vdupq(1);
        let chain: [V4; 20] = [
            m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13],
            m[14], m[15], tlo, thi, blen, flags,
        ];
        vst1q(zs.add(16) as *mut u32, vorrq(one, vshlq_n::<1>(chain[0])));
        for k in 1..20usize {
            let w = vorrq(vshrq_n::<31>(chain[k - 1]), vshlq_n::<1>(chain[k]));
            vst1q(zs.add(16 + k) as *mut u32, w);
        }
        for w in 16..36usize {
            let v = vld1q(zs.add(w) as *const u32);
            vst1q(ast.add(w) as *mut u32, v);
        }

        // ---- G stream (bits 1153..15409): sequential push network ----
        let pending_bit = vshrq_n::<31>(flags);
        let mut wz = W32::at(zs, pending_bit);
        let mut wa = W32::at(ast, pending_bit);
        let mut wb = W32::at(bs, one);

        macro_rules! g {
            ($g:expr, $la:literal, $lb:literal, $lc:literal, $ld:literal,
             $mx:literal, $my:literal) => {{
                let (t0, l0, r0, c0) = add_carry_parts_v(state[$la], state[$lb]);
                pushf!(wz, GS_BASE + G_STRIDE * $g + REC_C0, 31, c0);
                pushf!(wa, GS_BASE + G_STRIDE * $g + REC_C0, 31, l0);
                pushf!(wb, GS_BASE + G_STRIDE * $g + REC_C0, 31, r0);
                let (a1, l1, r1, c1) = add_carry_parts_v(t0, m[$mx]);
                pushf!(wz, GS_BASE + G_STRIDE * $g + REC_C1, 31, c1);
                pushf!(wa, GS_BASE + G_STRIDE * $g + REC_C1, 31, l1);
                pushf!(wb, GS_BASE + G_STRIDE * $g + REC_C1, 31, r1);
                let d1 = xor_rotr::<16, 16>(state[$ld], a1);
                let (c1s, l2, r2, c2) = add_carry_parts_v(state[$lc], d1);
                pushf!(wz, GS_BASE + G_STRIDE * $g + REC_C2, 31, c2);
                pushf!(wa, GS_BASE + G_STRIDE * $g + REC_C2, 31, l2);
                pushf!(wb, GS_BASE + G_STRIDE * $g + REC_C2, 31, r2);
                let b1 = xor_rotr::<12, 20>(state[$lb], c1s);
                let (t1, l3, r3, c3) = add_carry_parts_v(a1, b1);
                pushf!(wz, GS_BASE + G_STRIDE * $g + REC_C3, 31, c3);
                pushf!(wa, GS_BASE + G_STRIDE * $g + REC_C3, 31, l3);
                pushf!(wb, GS_BASE + G_STRIDE * $g + REC_C3, 31, r3);
                let (a2, l4, r4, c4) = add_carry_parts_v(t1, m[$my]);
                pushf!(wz, GS_BASE + G_STRIDE * $g + REC_C4, 31, c4);
                pushf!(wa, GS_BASE + G_STRIDE * $g + REC_C4, 31, l4);
                pushf!(wb, GS_BASE + G_STRIDE * $g + REC_C4, 31, r4);
                let d2 = xor_rotr::<8, 24>(d1, a2);
                let (c2s, l5, r5, c5) = add_carry_parts_v(c1s, d2);
                pushf!(wz, GS_BASE + G_STRIDE * $g + REC_C5, 31, c5);
                pushf!(wa, GS_BASE + G_STRIDE * $g + REC_C5, 31, l5);
                pushf!(wb, GS_BASE + G_STRIDE * $g + REC_C5, 31, r5);
                let bn = xor_rotr::<7, 25>(b1, c2s);
                pushf!(wz, GS_BASE + G_STRIDE * $g + REC_LIN0, 32, bn);
                pushf!(wa, GS_BASE + G_STRIDE * $g + REC_LIN0, 32, bn);
                pushf!(wb, GS_BASE + G_STRIDE * $g + REC_LIN0, 32, maxv);
                pushf!(wz, GS_BASE + G_STRIDE * $g + REC_LIN1, 32, d2);
                pushf!(wa, GS_BASE + G_STRIDE * $g + REC_LIN1, 32, d2);
                pushf!(wb, GS_BASE + G_STRIDE * $g + REC_LIN1, 32, maxv);
                state[$la] = a2;
                state[$lb] = bn;
                state[$lc] = c2s;
                state[$ld] = d2;
            }};
        }
        macro_rules! round {
            ($gb:literal, $m0:literal, $m1:literal, $m2:literal, $m3:literal,
             $m4:literal, $m5:literal, $m6:literal, $m7:literal,
             $m8:literal, $m9:literal, $m10:literal, $m11:literal,
             $m12:literal, $m13:literal, $m14:literal, $m15:literal) => {{
                g!($gb, 0, 4, 8, 12, $m0, $m1);
                g!($gb + 1, 1, 5, 9, 13, $m2, $m3);
                g!($gb + 2, 2, 6, 10, 14, $m4, $m5);
                g!($gb + 3, 3, 7, 11, 15, $m6, $m7);
                g!($gb + 4, 0, 5, 10, 15, $m8, $m9);
                g!($gb + 5, 1, 6, 11, 12, $m10, $m11);
                g!($gb + 6, 2, 7, 8, 13, $m12, $m13);
                g!($gb + 7, 3, 4, 9, 14, $m14, $m15);
            }};
        }
        round!(0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        round!(8, 2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8);
        round!(16, 3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1);
        round!(24, 10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6);
        round!(32, 12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4);
        round!(40, 9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7);
        round!(48, 11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13);

        // ---- out_hi (bits 15153..15409), stream-sequential ----
        const {
            assert!(OUT_HI_BASE % 32 == 17);
        }
        macro_rules! oh {
            ($w:literal) => {{
                let hv = veorq(state[$w + 8], cv_v[$w]);
                pushf!(wz, OUT_HI_BASE + 32 * $w, 32, hv);
                pushf!(wa, OUT_HI_BASE + 32 * $w, 32, hv);
                pushf!(wb, OUT_HI_BASE + 32 * $w, 32, maxv);
            }};
        }
        oh!(0);
        oh!(1);
        oh!(2);
        oh!(3);
        oh!(4);
        oh!(5);
        oh!(6);
        oh!(7);
        wz.finish();
        wa.finish();
        wb.finish();

        // ---- zero fill, words 482..512 (finish() 241..256 semantics) ----
        const ZF: usize = USEFUL_BITS.div_ceil(32); // 482
        const {
            assert!(U32_PER_BLOCK - ZF == 30);
        }
        for w in 0..30usize {
            vst1q(zs.add(ZF + w) as *mut u32, zero);
            vst1q(ast.add(ZF + w) as *mut u32, zero);
            vst1q(bs.add(ZF + w) as *mut u32, zero);
        }

        // ---- out_lo slot, words 8..16 (z/a only) ----
        for w in 0..8usize {
            let lo = veorq(state[w], state[w + 8]);
            vst1q(zs.add(8 + w) as *mut u32, lo);
            vst1q(ast.add(8 + w) as *mut u32, lo);
        }

        // ---- drain stages: per-block 2 KiB ascending bursts, plain stores ----
        dump_range(zs, z, 0, DUMP_CHUNKS);
        dump_range(ast, a, 0, DUMP_CHUNKS);
        dump_range(bs, b, 0, DUMP_CHUNKS);
    }
}

// ---------------------------------------------------------------------------
// x86 witgen driver: mirrors the scalar `drive_witness_packed_and_lincheck_impl`
// (PER_BLOCK_FULLY_WRITES, no rate-2 codeword, no Metal stream) with the SSE2
// quad kernel in place of the per-block scalar builder. Bit-exact with the
// scalar driver, incl. padding-block fill and the lincheck stripe transpose.
// ---------------------------------------------------------------------------

use flock_core::bits::transpose_8_u64s_to_64_bytes;
use flock_core::field::F128;

/// W-H2 gate with the same kill switch as the aarch64 arm:
/// `FLOCK_NO_WITGEN_SIMD=1` restores the scalar driver.
pub(crate) fn enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_WITGEN_SIMD").is_none());
    *ON
}

/// Non-streamed entry — bit-exact twin of
/// [`super::generate_witness_with_ab_packed_and_lincheck`] on x86_64.
pub(crate) fn generate(
    blocks: &[Compression],
    n_blocks_log: usize,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
    // Mirror the scalar fallback's lazy-spec handling: the QS1 speculative
    // buffer holds sentinels, so materialize an owned slice first.
    let generated = crate::seed_pipe::materialize_spec_blocks(blocks);
    let blocks = generated.as_deref().unwrap_or(blocks);
    let k = 1usize << super::K_LOG;
    let f128_per_block = k / 128;
    let u64_per_block = k / 64;
    let n_total = 1usize << n_blocks_log;
    let n_blocks = blocks.len();
    assert!(
        n_blocks <= n_total,
        "{n_blocks} blocks > 2^{n_blocks_log} = {n_total} slots"
    );
    assert!(
        n_total >= 8 && n_total.is_multiple_of(8),
        "lincheck stripe layout requires n_total ≥ 8 and divisible by 8"
    );

    let total_f128 = n_total * f128_per_block;
    // z/a/b from the recycling scratch pool (same as the scalar driver); the
    // quad kernel fully writes every word, so no zeroing pass is needed.
    let mut z = flock_core::scratch::take_f128(total_f128);
    let mut a = flock_core::scratch::take_f128(total_f128);
    let mut b = flock_core::scratch::take_f128(total_f128);
    let mut z_lincheck = flock_core::scratch::take_u8((n_total / 8) * k);

    let padding: Compression = ([0u32; 8], [0u32; 16], 0u64, 0u32, 0u32);

    let group_f128 = 8 * f128_per_block;

    #[derive(Clone, Copy)]
    struct WritePtr<T>(*mut T);
    unsafe impl<T> Send for WritePtr<T> {}
    unsafe impl<T> Sync for WritePtr<T> {}
    impl<T> WritePtr<T> {
        fn get(self) -> *mut T {
            self.0
        }
    }

    let z_base = WritePtr(z.as_mut_ptr());
    let a_base = WritePtr(a.as_mut_ptr());
    let b_base = WritePtr(b.as_mut_ptr());
    let stripe_base = WritePtr(z_lincheck.as_mut_ptr());

    let process_group = |g: usize| {
        // SAFETY: each scheduled group index occurs exactly once; every group
        // owns disjoint z/a/b ranges and one disjoint stripe.
        unsafe {
            let z_grp =
                std::slice::from_raw_parts_mut(z_base.get().add(g * group_f128), group_f128);
            let a_grp =
                std::slice::from_raw_parts_mut(a_base.get().add(g * group_f128), group_f128);
            let b_grp =
                std::slice::from_raw_parts_mut(b_base.get().add(g * group_f128), group_f128);
            let stripe = std::slice::from_raw_parts_mut(stripe_base.get().add(g * k), k);
            for half in 0..2 {
                let first = 8 * g + 4 * half;
                let base = half * 4 * f128_per_block;
                // The kernel takes `[&Compression; 4]`; out-of-range slots get
                // the padding block so every word is still fully written.
                let mut quad: [Compression; 4] = [padding; 4];
                for (j, slot) in quad.iter_mut().enumerate() {
                    let idx = first + j;
                    if idx < n_blocks {
                        *slot = blocks[idx];
                    }
                }
                build_quad_witness_ab_stream_x86(
                    [&quad[0], &quad[1], &quad[2], &quad[3]],
                    z_grp[base..].as_mut_ptr() as *mut u32,
                    a_grp[base..].as_mut_ptr() as *mut u32,
                    b_grp[base..].as_mut_ptr() as *mut u32,
                );
            }
            // Bit-transpose 8 z chunks into the lincheck stripe (identical to
            // the scalar driver; release elides the beyond-useful tail pad).
            let z_u64_all: &[u64] =
                std::slice::from_raw_parts(z_grp.as_ptr() as *const u64, z_grp.len() * 2);
            let useful_words = USEFUL_BITS.div_ceil(64);
            for i in 0..useful_words {
                let lanes: [u64; 8] = std::array::from_fn(|j| z_u64_all[j * u64_per_block + i]);
                transpose_8_u64s_to_64_bytes(&lanes, &mut stripe[i * 64..i * 64 + 64]);
            }
            // Mirrors the scalar driver: the padded fold never observes the
            // tail; the honest zero pad is test-only.
            #[cfg(test)]
            {
                stripe[useful_words * 64..].fill(0);
            }
        }
    };

    super::super::common::drain_group_jobs(n_total / 8, &process_group);

    (z, a, b, z_lincheck)
}

#[cfg(test)]
mod tests {
    use super::super::build_block_witness_ab_stream_into;
    use super::*;

    /// SplitMix64 (same as the blake3 test Rng).
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u32(&mut self) -> u32 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            (z ^ (z >> 31)) as u32
        }
    }

    /// The SSE2 quad kernel must be bit-identical to four scalar
    /// `build_block_witness_ab_stream_into` calls on all three buffers —
    /// the same pin the aarch64 NEON kernel is held to.
    #[test]
    fn x86_quad_witness_matches_scalar_stream_builder() {
        const WORDS: usize = super::super::K / 64;
        let mut rng = Rng::new(0x51D0_0F11_5EED_51AD);
        let mk = |rng: &mut Rng| -> Compression {
            let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
            let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
            let counter = ((rng.next_u32() as u64) << 32) | rng.next_u32() as u64;
            (cv, m, counter, rng.next_u32(), rng.next_u32())
        };
        for round in 0..32 {
            let inputs: [Compression; 4] = std::array::from_fn(|_| mk(&mut rng));
            let mut zq = [u64::MAX; 4 * WORDS];
            let mut aq = [u64::MAX; 4 * WORDS];
            let mut bq = [u64::MAX; 4 * WORDS];
            unsafe {
                build_quad_witness_ab_stream_x86(
                    [&inputs[0], &inputs[1], &inputs[2], &inputs[3]],
                    zq.as_mut_ptr() as *mut u32,
                    aq.as_mut_ptr() as *mut u32,
                    bq.as_mut_ptr() as *mut u32,
                );
            }
            for (j, inp) in inputs.iter().enumerate() {
                let (cv, m, t, bl, fl) = inp;
                let mut z_ref = [0u64; WORDS];
                let mut a_ref = [0u64; WORDS];
                let mut b_ref = [0u64; WORDS];
                build_block_witness_ab_stream_into(
                    cv, m, *t, *bl, *fl, &mut z_ref, &mut a_ref, &mut b_ref,
                );
                for (name, got, want) in [
                    ("z", &zq[j * WORDS..(j + 1) * WORDS], &z_ref[..]),
                    ("a", &aq[j * WORDS..(j + 1) * WORDS], &a_ref[..]),
                    ("b", &bq[j * WORDS..(j + 1) * WORDS], &b_ref[..]),
                ] {
                    if got != want {
                        let w = got.iter().zip(want).position(|(x, y)| x != y).unwrap();
                        panic!(
                            "{name} lane {j} round {round}: first diff u64 word {w} (u32 word {}):\
                             got {:#018x} want {:#018x}",
                            2 * w,
                            got[w],
                            want[w],
                        );
                    }
                }
            }
        }
    }

    /// End-to-end driver pin: `witgen_simd_x86::generate` must be
    /// byte-identical to the scalar full-write driver on z/a/b AND the
    /// lincheck stripe, including the padded non-power-of-two tail.
    #[test]
    fn x86_driver_matches_scalar_driver_end_to_end() {
        let mut rng = Rng::new(0x0F1E_2D3C_4B5A_6978);
        let mk = |rng: &mut Rng| -> Compression {
            let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
            let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
            let counter = ((rng.next_u32() as u64) << 32) | rng.next_u32() as u64;
            (cv, m, counter, rng.next_u32(), rng.next_u32())
        };
        let n_log = 8; // 256 slots
        let n_blocks = 200; // non-power-of-two → padding exercise
        let blocks: Vec<Compression> = (0..n_blocks).map(|_| mk(&mut rng)).collect();

        let (z, a, b, stripe) = super::generate(&blocks, n_log);

        let padding: Compression = ([0u32; 8], [0u32; 16], 0u64, 0u32, 0u32);
        let (zr, ar, br, striper) =
            super::super::super::common::drive_witness_packed_and_lincheck_full_write(
                &blocks,
                &padding,
                n_log,
                super::super::K_LOG,
                super::super::USEFUL_BITS,
                |block: &Compression, z_u64: &mut [u64], a_u64: &mut [u64], b_u64: &mut [u64]| {
                    let (cv, m, t, bl, fl) = block;
                    super::super::build_block_witness_ab_stream_into(
                        cv, m, *t, *bl, *fl, z_u64, a_u64, b_u64,
                    );
                },
            );
        assert_eq!(z, zr, "z differs from scalar driver");
        assert_eq!(a, ar, "a differs from scalar driver");
        assert_eq!(b, br, "b differs from scalar driver");
        assert_eq!(
            stripe, striper,
            "lincheck stripe differs from scalar driver"
        );
    }

    /// Micro-benchmark of the quad kernel vs four scalar per-block calls.
    /// Runs only when explicitly requested (ignored by default):
    /// `cargo test -p flock-prover --release -- --ignored --nocapture
    /// bench_quad_vs_scalar`.
    #[test]
    #[ignore]
    #[inline(never)]
    fn bench_quad_vs_scalar() {
        const WORDS: usize = super::super::K / 64;
        const QUADS: usize = 4096;
        let mut rng = Rng::new(0xBEEF_CAFE_2026_0814);
        let inputs: Vec<Compression> = (0..4 * QUADS)
            .map(|_| {
                let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
                let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
                let counter = ((rng.next_u32() as u64) << 32) | rng.next_u32() as u64;
                (cv, m, counter, rng.next_u32(), rng.next_u32())
            })
            .collect();

        let mut zs = vec![0u64; 4 * QUADS * WORDS];
        let mut as_ = vec![0u64; 4 * QUADS * WORDS];
        let mut bs = vec![0u64; 4 * QUADS * WORDS];

        // Warm up.
        for q in 0..QUADS {
            unsafe {
                build_quad_witness_ab_stream_x86(
                    [
                        &inputs[4 * q],
                        &inputs[4 * q + 1],
                        &inputs[4 * q + 2],
                        &inputs[4 * q + 3],
                    ],
                    zs[q * 4 * WORDS..].as_mut_ptr() as *mut u32,
                    as_[q * 4 * WORDS..].as_mut_ptr() as *mut u32,
                    bs[q * 4 * WORDS..].as_mut_ptr() as *mut u32,
                );
            }
        }
        let t0 = std::time::Instant::now();
        for q in 0..QUADS {
            unsafe {
                build_quad_witness_ab_stream_x86(
                    [
                        &inputs[4 * q],
                        &inputs[4 * q + 1],
                        &inputs[4 * q + 2],
                        &inputs[4 * q + 3],
                    ],
                    zs[q * 4 * WORDS..].as_mut_ptr() as *mut u32,
                    as_[q * 4 * WORDS..].as_mut_ptr() as *mut u32,
                    bs[q * 4 * WORDS..].as_mut_ptr() as *mut u32,
                );
            }
        }
        let quad_ns = t0.elapsed().as_nanos() as f64 / QUADS as f64;

        let t1 = std::time::Instant::now();
        for (i, inp) in inputs.iter().enumerate() {
            let (cv, m, t, bl, fl) = inp;
            let base = i * WORDS;
            build_block_witness_ab_stream_into(
                cv,
                m,
                *t,
                *bl,
                *fl,
                &mut zs[base..base + WORDS],
                &mut as_[base..base + WORDS],
                &mut bs[base..base + WORDS],
            );
        }
        let scalar_ns = t1.elapsed().as_nanos() as f64 / (4 * QUADS) as f64;

        println!(
            "quad kernel: {quad_ns:.1} ns/quad ({pb:.2} ns/block) | scalar: {scalar_ns:.1} ns/block | \
             quad/block speedup: {ratio:.2}x | per-quad wall delta: {delta:.2} ns",
            quad_ns = quad_ns,
            pb = quad_ns / 4.0,
            scalar_ns = scalar_ns,
            ratio = scalar_ns / (quad_ns / 4.0),
            delta = quad_ns - 4.0 * scalar_ns,
        );
        let m18_blocks = 1usize << 18;
        println!(
            "projected m=18 witgen kernel wall @1-thread: scalar {:.2} ms vs quad {:.2} ms (saves {:.2} ms)",
            scalar_ns * m18_blocks as f64 / 1e6,
            (quad_ns / 4.0) * m18_blocks as f64 / 1e6,
            (scalar_ns - quad_ns / 4.0) * m18_blocks as f64 / 1e6,
        );
    }

    /// How much of the full scalar witgen driver is actually the per-block
    /// kernel (vs the driver's scratch/stripe/threading machinery)? Feeds the
    /// honest projection: the SIMD kernel only saves
    /// `(1 − 1/speedup) × share` of witgen.
    #[test]
    #[ignore]
    #[inline(never)]
    fn bench_witgen_kernel_share() {
        use super::super::super::common::drive_witness_packed_and_lincheck_full_write;
        const N_LOG: usize = 16; // 65,536 blocks — 402 MiB z/a/b + 128 MiB stripe
        let n_blocks = 1usize << N_LOG;
        let mut rng = Rng::new(0x51D0_0F11_5EED_51AD);
        let blocks: Vec<Compression> = (0..n_blocks)
            .map(|_| {
                let cv: [u32; 8] = std::array::from_fn(|_| rng.next_u32());
                let m: [u32; 16] = std::array::from_fn(|_| rng.next_u32());
                let counter = ((rng.next_u32() as u64) << 32) | rng.next_u32() as u64;
                (cv, m, counter, rng.next_u32(), rng.next_u32())
            })
            .collect();
        let padding: Compression = ([0u32; 8], [0u32; 16], 0u64, 0u32, 0u32);
        let per_block =
            |block: &Compression, z_u64: &mut [u64], a_u64: &mut [u64], b_u64: &mut [u64]| {
                let (cv, m, t, bl, fl) = block;
                super::super::build_block_witness_ab_stream_into(
                    cv, m, *t, *bl, *fl, z_u64, a_u64, b_u64,
                );
            };
        // Warm-up (cold page faults + pool).
        let (_z, _a, _b, _s) = drive_witness_packed_and_lincheck_full_write(
            &blocks,
            &padding,
            N_LOG,
            super::super::K_LOG,
            super::super::USEFUL_BITS,
            per_block,
        );
        let t0 = std::time::Instant::now();
        let (_z, _a, _b, _s) = drive_witness_packed_and_lincheck_full_write(
            &blocks,
            &padding,
            N_LOG,
            super::super::K_LOG,
            super::super::USEFUL_BITS,
            per_block,
        );
        let driver_ms = t0.elapsed().as_secs_f64() * 1e3;

        // Kernel-only at the same n (4 blocks per quad call).
        let mut zq = vec![0u64; n_blocks * super::super::K / 64];
        let mut aq = vec![0u64; n_blocks * super::super::K / 64];
        let mut bq = vec![0u64; n_blocks * super::super::K / 64];
        let t1 = std::time::Instant::now();
        for q in 0..n_blocks / 4 {
            unsafe {
                build_quad_witness_ab_stream_x86(
                    [
                        &blocks[4 * q],
                        &blocks[4 * q + 1],
                        &blocks[4 * q + 2],
                        &blocks[4 * q + 3],
                    ],
                    zq[q * 1024..].as_mut_ptr() as *mut u32,
                    aq[q * 1024..].as_mut_ptr() as *mut u32,
                    bq[q * 1024..].as_mut_ptr() as *mut u32,
                );
            }
        }
        let kernel_ms = t1.elapsed().as_secs_f64() * 1e3;
        println!(
            "n={n_blocks} blocks @1-thread: full scalar witgen driver {driver_ms:.1} ms | quad kernel {kernel_ms:.1} ms | \
             kernel share of driver {share:.1}% | projected m=18 witgen {w:.0} ms @1-thread, kernel-only SIMD ceiling saves {s:.0} ms ({p:.1}%)",
            n_blocks = n_blocks,
            driver_ms = driver_ms,
            kernel_ms = kernel_ms,
            share = kernel_ms / driver_ms * 100.0,
            w = driver_ms * 4.0,
            s = kernel_ms * (1.0 - 1.0 / 1.44) * 4.0,
            p = (kernel_ms * (1.0 - 1.0 / 1.44) * 4.0) / (driver_ms * 4.0) * 100.0,
        );
    }
}
