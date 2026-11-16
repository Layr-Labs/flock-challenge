//! Zerocheck 5+6 GPU prefix arm (see `ENV_NO_ZC_T56_GPU`): geometry of the
//! ranked composed rounds-5+6 pass, the per-launch constants, and the CPU
//! finish of the kernel's raw slot banks. The Metal backend is
//! `imp::t56_metal` (macOS/aarch64 only); under `cfg(test)` the [`emu`]
//! transliteration of its kernel checks the arithmetic and the input layouts
//! against the portable reference.
//!
//! The arm's input is the K pass GPU prefix arm's output buffers, read in
//! place: the packed a layout (244 entries per 256 logical inputs) and the
//! b-one-packed b layout (56 entries per 64 logical inputs). Chunk `c` of the
//! 5+6 pass reads entries `c * IN_A..` of a and `c * IN_B..` of b.

#![cfg_attr(not(all(target_os = "macos", target_arch = "aarch64")), allow(dead_code))]

use crate::field::{F128, F256Unreduced};

/// Chunks (`hi_size`) of the ranked 5+6 pass.
pub(crate) const HI: usize = 512;
/// `lo_size` of the ranked 5+6 pass.
pub(crate) const LO: usize = 4096;
/// Folded outputs per chunk and array (`2 * lo_size`).
pub(crate) const OUT: usize = 2 * LO;
/// Packed a input entries per chunk: `4 * OUT` logical inputs, 244 per 256.
pub(crate) const IN_A: usize = 4 * OUT / 256 * 244;
/// b-one-packed b input entries per chunk: 56 per 64 logical inputs.
pub(crate) const IN_B: usize = 4 * OUT / 64 * 56;
/// Message slots per chunk.
pub(crate) const SLOTS: usize = 26;
/// Groups of eight outputs per chunk; group `y` is weighted by
/// `eq_lo[4 * y + 3]`.
pub(crate) const GROUPS: usize = OUT / 8;
/// F128 entries of the per-launch constants: the `rho_a` and `rho_b`
/// nibble tables (512 each), then `eq_odd` ([`GROUPS`]).
pub(crate) const CONSTS: usize = 1024 + GROUPS;
/// F128 entries of the flat-fold kernel's constants (see [`ENV_NO_T56_SKIP`]):
/// [`CONSTS`], then the nibble table of `rho_a * rho_b` (512).
pub(crate) const CONSTS_FLAT: usize = CONSTS + 512;

/// Kill switch for the 5+6 kernel's flat fold (`FLOCK_NO_T56_SKIP=1`,
/// exactly `1`: the record kernel text and constants). With it on, each
/// output folds its four inputs `x0..x3` as
/// `x0 + rho_a (x0 + x1) + rho_b (x0 + x2) + (rho_a rho_b) (x0 + x1 + x2 + x3)`:
/// three table products that need nothing of each other, by `rho_a`,
/// `rho_b` and a third nibble table of `rho_a * rho_b` (constants
/// `CONSTS..CONSTS_FLAT`). The record kernel folds the pairs first,
/// `f0 = x0 + rho_a (x0 + x1)`, `f1 = x2 + rho_a (x2 + x3)`, and then
/// `f0 + rho_b (f0 + f1)`, whose product waits for both pair products; the
/// flat form skips that wait. Since `f0 + f1 = (x0 + x2) + rho_a (x0 + x1 +
/// x2 + x3)`, distributivity makes the two forms the same field element, so
/// every output and slot is unchanged. Same three table products per array
/// and output. Read once per process, before the arm's pipeline build.
pub const ENV_NO_T56_SKIP: &str = "FLOCK_NO_T56_SKIP";

/// Whether the 5+6 kernel takes the flat fold (no `FLOCK_NO_T56_SKIP=1`).
pub(crate) fn flat_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os(ENV_NO_T56_SKIP).as_deref() != Some(std::ffi::OsStr::new("1")))
}

/// F128 entries of the constants the selected kernel reads.
pub(crate) fn consts_len() -> usize {
    if flat_enabled() { CONSTS_FLAT } else { CONSTS }
}

/// Raw slot-bank entries per chunk (four threadgroups of 32), each an
/// unreduced 256-bit sum `[r0, r1, r2, r3]`.
pub(crate) const BANKS: usize = 128;

/// Message slot of bank entry `e % 32 = 8 * round + lane` (rounds a, b,
/// c, d of the kernel).
pub(crate) const SLOT: [u8; 32] = [
    25, 0, 8, 1, 16, 2, 9, 3, //
    10, 4, 5, 12, 11, 6, 7, 13, //
    17, 20, 18, 21, 19, 23, 24, 22, //
    14, 14, 14, 14, 15, 15, 15, 15,
];

/// Per-launch constants: the `rho_a` then the `rho_b` nibble table
/// (bank `k` for byte `k` of the 16: `[32 k + n] = rho * (n << 8k)` and
/// `[32 k + 16 + n] = rho * ((n << 4) << 8k)`, the layout the kernel's
/// `mul_rho16` reads), then `eq_odd[y] = eq_lo[4 y + 3]`. Each entry is the
/// XOR of the `rho * x^p` of its set bits `p`, and `rho * x^p` is `rho`
/// times `x` applied `p` times, so every entry is the exact product.
pub(crate) fn consts_into(out: &mut [F128], rho_a: F128, rho_b: F128, eq_lo: &[F128]) {
    assert_eq!(out.len(), CONSTS);
    assert_eq!(eq_lo.len(), LO);
    nib_table_into(&mut out[..512], rho_a);
    nib_table_into(&mut out[512..1024], rho_b);
    for y in 0..GROUPS {
        out[1024 + y] = eq_lo[4 * y + 3];
    }
}

/// The constants of the kernel [`flat_enabled`] selects: [`consts_into`]'s,
/// then (flat fold) the nibble table of `rho_a * rho_b`. `out` holds
/// [`consts_len`] entries.
pub(crate) fn launch_consts_into(out: &mut [F128], rho_a: F128, rho_b: F128, eq_lo: &[F128]) {
    assert_eq!(out.len(), consts_len());
    consts_into(&mut out[..CONSTS], rho_a, rho_b, eq_lo);
    if out.len() == CONSTS_FLAT {
        nib_table_into(&mut out[CONSTS..], rho_a * rho_b);
    }
}

/// One challenge's nibble table, 512 entries in the layout `mul_rho16` reads.
fn nib_table_into(out: &mut [F128], rho: F128) {
    assert_eq!(out.len(), 512);
    let mut pw = [F128::ZERO; 128];
    pw[0] = rho;
    for p in 1..128 {
        pw[p] = crate::field::mul_by_x(pw[p - 1]);
    }
    let mut i = 0;
    for bank in 0..16 {
        for half in 0..2 {
            for n in 0..16usize {
                let mut v = F128::ZERO;
                for b in 0..4 {
                    if (n >> b) & 1 == 1 {
                        v += pw[8 * bank + 4 * half + b];
                    }
                }
                out[i] = v;
                i += 1;
            }
        }
    }
}

/// A chunk's 26 slots from its [`BANKS`] raw bank entries: per slot, the
/// XOR of the entries whose [`SLOT`] it is, reduced once. Reduction is
/// XOR-linear, so this is the kernel's slot sum exactly.
pub(crate) fn finish_banks(banks: &[[u64; 4]]) -> [F128; SLOTS] {
    assert_eq!(banks.len(), BANKS);
    let mut acc = [F256Unreduced::ZERO; SLOTS];
    for (e, w) in banks.iter().enumerate() {
        acc[SLOT[e & 31] as usize] ^= F256Unreduced { r0: w[0], r1: w[1], r2: w[2], r3: w[3] };
    }
    acc.map(F256Unreduced::reduce)
}

#[cfg(test)]
pub(crate) mod emu {
    //! Line-by-line transliteration of `zc_t56_fold` (one threadgroup's
    //! eight simdgroups run lane by lane), in the record form and in the flat
    //! form (`flat`, see [`ENV_NO_T56_SKIP`]; `consts` then holds
    //! [`CONSTS_FLAT`] entries).
    use super::*;
    use crate::field::F256Unreduced as U256k;

    const Z4: F128 = F128::ZERO;

    fn x4(a: F128, b: F128) -> F128 {
        F128 { lo: a.lo ^ b.lo, hi: a.hi ^ b.hi }
    }

    pub(crate) fn clmul32(a: u32, b: u32) -> u64 {
        const M: [u64; 4] = [0x1111111111111111, 0x2222222222222222, 0x4444444444444444, 0x8888888888888888];
        let am = [a & 0x11111111, a & 0x22222222, a & 0x44444444, a & 0x88888888].map(u64::from);
        let bm = [b & 0x11111111, b & 0x22222222, b & 0x44444444, b & 0x88888888].map(u64::from);
        let m = |x: u64, y: u64| x.wrapping_mul(y);
        let r0 = (m(am[0], bm[0]) ^ m(am[1], bm[3]) ^ m(am[2], bm[2]) ^ m(am[3], bm[1])) & M[0];
        let r1 = (m(am[0], bm[1]) ^ m(am[1], bm[0]) ^ m(am[2], bm[3]) ^ m(am[3], bm[2])) & M[1];
        let r2 = (m(am[0], bm[2]) ^ m(am[1], bm[1]) ^ m(am[2], bm[0]) ^ m(am[3], bm[3])) & M[2];
        let r3 = (m(am[0], bm[3]) ^ m(am[1], bm[2]) ^ m(am[2], bm[1]) ^ m(am[3], bm[0])) & M[3];
        r0 | r1 | r2 | r3
    }

    fn clmul64(a: u64, b: u64) -> (u64, u64) {
        let (al, ah) = (a as u32, (a >> 32) as u32);
        let (bl, bh) = (b as u32, (b >> 32) as u32);
        let p_lo = clmul32(al, bl);
        let p_hi = clmul32(ah, bh);
        let p_mid = clmul32(al ^ ah, bl ^ bh) ^ p_lo ^ p_hi;
        (p_lo ^ (p_mid << 32), p_hi ^ (p_mid >> 32))
    }

    fn clmul128(a: F128, b: F128) -> U256k {
        let p0 = clmul64(a.lo, b.lo);
        let p2 = clmul64(a.hi, b.hi);
        let mut pm = clmul64(a.lo ^ a.hi, b.lo ^ b.hi);
        pm.0 ^= p0.0 ^ p2.0;
        pm.1 ^= p0.1 ^ p2.1;
        U256k { r0: p0.0, r1: p0.1 ^ pm.0, r2: p2.0 ^ pm.1, r3: p2.1 }
    }

    fn gf_reduce(p: U256k) -> F128 {
        let (h0, h1) = (p.r2, p.r3);
        let mut t0 = h0 ^ (h0 << 1) ^ (h0 << 2) ^ (h0 << 7);
        let t1 = h1 ^ (h1 << 1) ^ (h1 << 2) ^ (h1 << 7) ^ (h0 >> 63) ^ (h0 >> 62) ^ (h0 >> 57);
        let ov = (h1 >> 63) ^ (h1 >> 62) ^ (h1 >> 57);
        t0 ^= ov ^ (ov << 1) ^ (ov << 2) ^ (ov << 7);
        F128 { lo: p.r0 ^ t0, hi: p.r1 ^ t1 }
    }

    fn mul_rho16(x: F128, nib: &[F128]) -> F128 {
        let mut acc = Z4;
        let w = [x.lo as u32, (x.lo >> 32) as u32, x.hi as u32, (x.hi >> 32) as u32];
        for word in 0..4 {
            for j in 0..4 {
                let b = ((w[word] >> (8 * j)) & 0xff) as usize;
                let bank = word * 4 + j;
                acc = x4(acc, x4(nib[bank * 32 + (b & 15)], nib[bank * 32 + 16 + (b >> 4)]));
            }
        }
        acc
    }

    fn xor_lanes(v: &[F128; 32], m: usize) -> [F128; 32] {
        std::array::from_fn(|lane| x4(v[lane], v[lane ^ m]))
    }

    fn xor_lanes4(v: &[U256k; 32], m: usize) -> [U256k; 32] {
        std::array::from_fn(|lane| v[lane] ^ v[lane ^ m])
    }

    /// The kernel's input loads for output `o` of a chunk: the four
    /// logical inputs of a (packed) and of b (b-one-packed).
    fn load(ai: &[F128], bi: &[F128], o: usize) -> ([F128; 4], [F128; 4]) {
        let l = o & 7;
        let y = o >> 3;
        let blk = y >> 1;
        let half = y & 1;
        let tail = half != 0 && l == 7;
        let sup = (blk >> 2) * 244;
        let a = if tail {
            [ai[sup + 240 + (blk & 3)], Z4, Z4, Z4]
        } else {
            let p = sup + 60 * (blk & 3) + (half << 5) + (l << 2);
            [ai[p], ai[p + 1], ai[p + 2], ai[p + 3]]
        };
        let bm = blk * 56;
        let b = if half == 0 && l == 0 {
            [F128::ONE; 4]
        } else if tail {
            [bi[bm + 55], Z4, Z4, Z4]
        } else {
            let q = bm + (half << 5) + (l << 2) - 4;
            let b3 = if half != 0 && l == 6 { F128::ONE } else { bi[q + 3] };
            [bi[q], bi[q + 1], bi[q + 2], b3]
        };
        (a, b)
    }

    /// `zc_t56_fold` threadgroup `4 * chunk + quarter` over one chunk's
    /// inputs (`ai`: [`IN_A`], `bi`: [`IN_B`]) and outputs ([`OUT`] each);
    /// returns its 32 bank entries.
    fn fold_threadgroup(
        quarter: usize,
        ai: &[F128],
        bi: &[F128],
        ao: &mut [F128],
        bo: &mut [F128],
        consts: &[F128],
        flat: bool,
    ) -> [U256k; 32] {
        let nib = &consts[..1024];
        let eq_odd = &consts[1024..CONSTS];
        // Flat form: the four-input fold with three independent table
        // products, the third by the `rho_a * rho_b` table.
        let fold = |x0: F128, x1: F128, x2: F128, x3: F128| -> F128 {
            if flat {
                let nab = &consts[CONSTS..CONSTS_FLAT];
                let d = x4(x4(x0, x1), x4(x2, x3));
                x4(
                    x4(x0, mul_rho16(x4(x0, x1), &nib[..512])),
                    x4(mul_rho16(x4(x0, x2), &nib[512..]), mul_rho16(d, nab)),
                )
            } else {
                let f0 = x4(x0, mul_rho16(x4(x0, x1), &nib[..512]));
                let f1 = x4(x2, mul_rho16(x4(x2, x3), &nib[..512]));
                x4(f0, mul_rho16(x4(f0, f1), &nib[512..]))
            }
        };
        let mut red = [U256k::ZERO; 256];
        for sg in 0..8 {
            let mut acc_a = [U256k::ZERO; 32];
            let mut acc_b = [U256k::ZERO; 32];
            let mut acc_c = [U256k::ZERO; 32];
            let mut acc_d = [U256k::ZERO; 32];
            let mut qa = [Z4; 32];
            let mut qb = [Z4; 32];
            for j in 0..8usize {
                let mut wa = [Z4; 32];
                let mut fb = [Z4; 32];
                for lane in 0..32 {
                    let lid = 32 * sg + lane;
                    let o = (quarter << 11) + (j << 8) + lid;
                    let ([a0, a1, a2, a3], [b0, b1, b2, b3]) = load(ai, bi, o);
                    let fa = fold(a0, a1, a2, a3);
                    let fbv = fold(b0, b1, b2, b3);
                    ao[o] = fa;
                    bo[o] = fbv;
                    wa[lane] = gf_reduce(clmul128(eq_odd[o >> 3], fa));
                    fb[lane] = fbv;
                }
                let (pa, pb) = (xor_lanes(&wa, 1), xor_lanes(&fb, 1));
                let (ea, eb) = (xor_lanes(&wa, 2), xor_lanes(&fb, 2));
                let (qsa, qsb) = (xor_lanes(&ea, 1), xor_lanes(&eb, 1));
                let (ha, hb) = (xor_lanes(&wa, 4), xor_lanes(&fb, 4));
                let (h1a, h1b) = (xor_lanes(&ha, 1), xor_lanes(&hb, 1));
                let (h2a, h2b) = (xor_lanes(&ha, 2), xor_lanes(&hb, 2));
                let (hha, hhb) = (xor_lanes(&h1a, 2), xor_lanes(&h1b, 2));
                for lane in 0..32 {
                    let l = lane & 7;
                    let sel_b = ((l ^ (l >> 1)) & 1) != 0;
                    let (x, y) = if sel_b { (pa[lane], pb[lane]) } else { (ea[lane], eb[lane]) };
                    acc_b[lane] ^= clmul128(x, y);
                    if (l & 3) == (j & 3) {
                        qa[lane] = qsa[lane];
                        qb[lane] = qsb[lane];
                    }
                    let ca = if l < 4 { ha[lane] } else if sel_b { h1a[lane] } else { h2a[lane] };
                    let cb = if l < 4 { hb[lane] } else if sel_b { h1b[lane] } else { h2b[lane] };
                    acc_c[lane] ^= clmul128(ca, cb);
                    let (x, y) = if l == 0 { (hha[lane], hhb[lane]) } else { (wa[lane], fb[lane]) };
                    acc_a[lane] ^= clmul128(x, y);
                    if (j & 3) == 3 {
                        acc_d[lane] ^= clmul128(qa[lane], qb[lane]);
                    }
                }
            }
            for m in [8, 16] {
                acc_a = xor_lanes4(&acc_a, m);
                acc_b = xor_lanes4(&acc_b, m);
                acc_c = xor_lanes4(&acc_c, m);
                acc_d = xor_lanes4(&acc_d, m);
            }
            for lane in 0..8 {
                let base = sg * 32 + lane;
                red[base] = acc_a[lane];
                red[base + 8] = acc_b[lane];
                red[base + 16] = acc_c[lane];
                red[base + 24] = acc_d[lane];
            }
        }
        std::array::from_fn(|lid| {
            let mut s = red[lid];
            for k in 1..8 {
                s ^= red[(k << 5) + lid];
            }
            s
        })
    }

    /// The kernel over one chunk (its four threadgroups) plus the CPU's
    /// bank finish: outputs into `ao`, `bo`; returns the chunk's slots.
    /// `flat`: the flat-fold kernel (`consts` of [`CONSTS_FLAT`] entries).
    pub(crate) fn emulate_chunk(
        ai: &[F128],
        bi: &[F128],
        ao: &mut [F128],
        bo: &mut [F128],
        consts: &[F128],
        flat: bool,
    ) -> [F128; SLOTS] {
        assert_eq!((ai.len(), bi.len(), ao.len(), bo.len()), (IN_A, IN_B, OUT, OUT));
        assert_eq!(consts.len(), if flat { CONSTS_FLAT } else { CONSTS });
        let mut banks = Vec::with_capacity(BANKS);
        for quarter in 0..4 {
            let tg = fold_threadgroup(quarter, ai, bi, ao, bo, consts, flat);
            banks.extend(tg.iter().map(|u| [u.r0, u.r1, u.r2, u.r3]));
        }
        finish_banks(&banks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand(state: &mut u64) -> F128 {
        let mut next = || {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            *state
        };
        F128::new(next(), next())
    }

    /// One chunk's logical inputs in the K pass's output conventions
    /// (inputs 61..63 of every 64 zero; b inputs 0..3 and 59 one), packed
    /// into the arm's input layouts.
    fn chunk_inputs(seed: u64) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<F128>) {
        let mut s = seed;
        let n = 4 * OUT;
        let mut a = vec![F128::ZERO; n];
        let mut b = vec![F128::ZERO; n];
        for i in 0..n {
            let k = i % 64;
            if k < 61 {
                a[i] = rand(&mut s);
                b[i] = if k < 4 || k == 59 { F128::ONE } else { rand(&mut s) };
            }
        }
        let mut pa = Vec::with_capacity(IN_A);
        for sup in a.chunks_exact(256) {
            for blk in sup.chunks_exact(64) {
                pa.extend_from_slice(&blk[..60]);
            }
            pa.extend(sup.chunks_exact(64).map(|blk| blk[60]));
        }
        let mut pb = Vec::with_capacity(IN_B);
        for blk in b.chunks_exact(64) {
            pb.extend_from_slice(&blk[4..59]);
            pb.push(blk[60]);
        }
        assert_eq!((pa.len(), pb.len()), (IN_A, IN_B));
        (a, b, pa, pb)
    }

    #[test]
    fn t56_emulator_matches_cpu_tri_chunk() {
        for seed in [0x9e37_79b9_7f4a_7c15u64, 0x0123_4567_89ab_cdef] {
            let (a, b, pa, pb) = chunk_inputs(seed);
            let mut s = seed.rotate_left(17) | 1;
            let rhos = [rand(&mut s), rand(&mut s)];
            let eq_lo: Vec<F128> = (0..LO).map(|_| rand(&mut s)).collect();
            let (mut ca, mut cb) = (vec![F128::ZERO; OUT], vec![F128::ZERO; OUT]);
            let want = crate::zerocheck::multilinear::tri_chunk_scalar(&a, &b, &mut ca, &mut cb, &rhos, &eq_lo);
            // The record form, then the flat fold (its constants extend the
            // record's: same first `CONSTS` entries, then the `rho_a rho_b`
            // table).
            let mut consts = vec![F128::ZERO; CONSTS_FLAT];
            consts_into(&mut consts[..CONSTS], rhos[0], rhos[1], &eq_lo);
            let record = consts[..CONSTS].to_vec();
            nib_table_into(&mut consts[CONSTS..], rhos[0] * rhos[1]);
            assert_eq!(&consts[..CONSTS], &record[..]);
            for (flat, cs) in [(false, &record[..]), (true, &consts[..])] {
                let (mut ga, mut gb) = (vec![F128::ZERO; OUT], vec![F128::ZERO; OUT]);
                let slots = emu::emulate_chunk(&pa, &pb, &mut ga, &mut gb, cs, flat);
                assert!(ga == ca && gb == cb, "outputs differ (flat {flat})");
                assert_eq!(slots, want, "slots differ (flat {flat})");
            }
        }
    }

    /// The flat fold equals the pairwise fold for every input pattern the
    /// layouts produce (ones, zeros, a lone input) and for random inputs,
    /// and the `rho_a rho_b` table is that product's table.
    #[test]
    fn t56_flat_fold_matches_pairwise_fold() {
        let mut s = 0x5eed_7a56_u64;
        let (ra, rb) = (rand(&mut s), rand(&mut s));
        let mut t = vec![F128::ZERO; 1536];
        nib_table_into(&mut t[..512], ra);
        nib_table_into(&mut t[512..1024], rb);
        nib_table_into(&mut t[1024..], ra * rb);
        let tab = |x: F128, k: usize| -> F128 {
            let mut v = F128::ZERO;
            for (i, byte) in x.lo.to_le_bytes().into_iter().chain(x.hi.to_le_bytes()).enumerate() {
                v += t[512 * k + 32 * i + (byte & 15) as usize] + t[512 * k + 32 * i + 16 + (byte >> 4) as usize];
            }
            v
        };
        for _ in 0..64 {
            let x = rand(&mut s);
            assert_eq!(tab(x, 0), ra * x);
            assert_eq!(tab(x, 1), rb * x);
            assert_eq!(tab(x, 2), ra * rb * x);
        }
        let (o, z) = (F128::ONE, F128::ZERO);
        let r = |s: &mut u64| rand(s);
        let mut cases = vec![[o, o, o, o], [r(&mut s), z, z, z], [r(&mut s), r(&mut s), r(&mut s), o], [z, z, z, z]];
        for _ in 0..64 {
            cases.push([r(&mut s), r(&mut s), r(&mut s), r(&mut s)]);
        }
        for [x0, x1, x2, x3] in cases {
            let f0 = x0 + tab(x0 + x1, 0);
            let f1 = x2 + tab(x2 + x3, 0);
            let pair = f0 + tab(f0 + f1, 1);
            let flat = x0 + tab(x0 + x1, 0) + tab(x0 + x2, 1) + tab(x0 + x1 + x2 + x3, 2);
            assert_eq!(flat, pair);
        }
    }

    #[test]
    fn t56_slot_map_covers_every_slot() {
        let mut seen = [0usize; SLOTS];
        for &s in &SLOT {
            seen[s as usize] += 1;
        }
        assert!(seen.iter().all(|&c| c > 0));
    }
}
