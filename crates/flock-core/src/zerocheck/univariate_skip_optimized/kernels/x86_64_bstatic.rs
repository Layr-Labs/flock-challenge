use core::arch::x86_64::*;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::super::{F8, InvNttTableByteSingleGf8, N_CHUNKS};
use super::x86_64_bstatic_plan::{
    BSTATIC_BLOCKS, BSTATIC_PLAN, BstaticRow, ROW_GENERIC, ROW_STATIC, ROW_ZERO,
};

#[repr(C, align(64))]
pub(crate) struct BstaticPartials {
    rows: [[[u8; 64]; 8]; BSTATIC_BLOCKS],

    img2: bool,
}

struct Cached {
    fingerprint: [u8; 8 * 64],
    partials: Box<BstaticPartials>,
}

static CACHE: OnceLock<Cached> = OnceLock::new();

static VERIFIED_PTR: AtomicUsize = AtomicUsize::new(0);

fn fingerprint_of(inv_table: &InvNttTableByteSingleGf8) -> [u8; 8 * 64] {
    let mut fp = [0u8; 8 * 64];
    let base = inv_table.data_ptr();
    for t in 0..8 {
        let row = unsafe { core::slice::from_raw_parts(base.add((1usize << t) * 64), 64) };
        fp[t * 64..(t + 1) * 64].copy_from_slice(row);
    }
    fp
}

fn build_partials(inv_table: &InvNttTableByteSingleGf8) -> Box<BstaticPartials> {
    let mut out = Box::new(BstaticPartials {
        rows: [[[0u8; 64]; 8]; BSTATIC_BLOCKS],
        img2: apply_2img_for(inv_table),
    });
    let mut f = [F8::ZERO; 64];
    for blk in 0..BSTATIC_BLOCKS {
        for k in 0..8 {
            let p = BSTATIC_PLAN[blk][k];
            if p.kind != ROW_STATIC {
                continue;
            }
            inv_table.apply(&p.expected.to_le_bytes(), &mut f);
            let scale = if p.vary == 0 { F8(1u8 << k) } else { F8::ONE };
            for i in 0..64 {
                out.rows[blk][k][i] = (f[i] * scale).0;
            }
        }
    }
    out
}

pub(crate) fn prepare_bstatic(
    inv_table: &InvNttTableByteSingleGf8,
) -> Option<&'static BstaticPartials> {
    if !fast_shift_reduce_enabled()
        || inv_table.k != 6
        || inv_table.ell != 64
        || inv_table.n_chunks != 8
    {
        return None;
    }
    let ptr = inv_table.data_ptr() as usize;
    if VERIFIED_PTR.load(Ordering::Acquire) == ptr {
        return CACHE.get().map(|c| &*c.partials);
    }
    let cached = CACHE.get_or_init(|| Cached {
        fingerprint: fingerprint_of(inv_table),
        partials: build_partials(inv_table),
    });
    if cached.fingerprint == fingerprint_of(inv_table)
        && cached.partials.img2 == apply_2img_for(inv_table)
    {
        VERIFIED_PTR.store(ptr, Ordering::Release);
        Some(&*cached.partials)
    } else {
        None
    }
}

fn apply_2img_for(inv_table: &InvNttTableByteSingleGf8) -> bool {
    bstatic_apply_2img_enabled() && inv_table.has_second_image()
}

pub(crate) fn bstatic_apply_2img_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_BSTATIC_APPLY_2IMG").is_none())
}

pub(crate) fn fast_shift_reduce_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_FAST_SHIFT_REDUCE").is_none())
}

fn bstatic_28_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_BSTATIC_28").is_none())
}

#[cfg(test)]
pub(crate) const BSTATIC_LIVE: [bool; BSTATIC_BLOCKS] = {
    let mut t = [false; BSTATIC_BLOCKS];
    t[0] = true;
    t[1] = true;
    t[28] = true;
    t[30] = true;
    t[31] = true;
    t
};

#[repr(C, align(64))]
struct PermIdx([[u64; 8]; 8]);
static PERM_IDX: PermIdx = {
    let mut t = [[0u64; 8]; 8];
    let mut j = 0;
    while j < 8 {
        let mut q = 0;
        while q < 8 {
            t[j][q] = (q ^ j) as u64;
            q += 1;
        }
        j += 1;
    }
    PermIdx(t)
};

#[inline(always)]
unsafe fn perm_row(v: __m512i, j: usize) -> __m512i {
    if j == 0 {
        v
    } else {
        unsafe {
            _mm512_permutexvar_epi64(
                _mm512_load_si512(PERM_IDX.0[j].as_ptr() as *const __m512i),
                v,
            )
        }
    }
}

#[inline(always)]
unsafe fn apply_full(table: *const u8, bytes: *const u8) -> __m512i {
    unsafe {
        let mut acc = _mm512_loadu_si512(table.add(*bytes as usize * 64) as *const __m512i);
        for j in 1..8 {
            let row = _mm512_loadu_si512(table.add(*bytes.add(j) as usize * 64) as *const __m512i);
            acc = _mm512_xor_si512(acc, perm_row(row, j));
        }
        acc
    }
}

#[inline(always)]
unsafe fn apply_full_2img(table: *const u8, table8: *const u8, bytes: *const u8) -> __m512i {
    unsafe {
        let row = |img: *const u8, b: usize| {
            _mm512_loadu_si512(img.add(*bytes.add(b) as usize * 64) as *const __m512i)
        };
        let u0 = _mm512_xor_si512(row(table, 0), row(table8, 1));
        let u1 = _mm512_xor_si512(row(table, 2), row(table8, 3));
        let u2 = _mm512_xor_si512(row(table, 4), row(table8, 5));
        let u3 = _mm512_xor_si512(row(table, 6), row(table8, 7));
        let even = _mm512_xor_si512(u0, _mm512_shuffle_i64x2::<0x4E>(u2, u2));
        let odd = _mm512_xor_si512(u1, _mm512_shuffle_i64x2::<0x4E>(u3, u3));
        _mm512_xor_si512(even, _mm512_shuffle_i64x2::<0xB1>(odd, odd))
    }
}

#[inline(always)]
unsafe fn all_static_acc<const IMG2: bool>(
    table: *const u8,
    table8: *const u8,
    a_base: *const u8,
    parts: *const u8,
) -> __m512i {
    unsafe {
        let mut acc = _mm512_setzero_si512();
        for k in 0..8usize {
            let av = if IMG2 {
                apply_full_2img(table, table8, a_base.add(k * N_CHUNKS))
            } else {
                apply_full(table, a_base.add(k * N_CHUNKS))
            };
            let part = _mm512_load_si512(parts.add(k * 64) as *const __m512i);
            acc = _mm512_xor_si512(acc, _mm512_gf2p8mul_epi8(av, part));
        }
        acc
    }
}

#[inline(always)]
unsafe fn kernel<const BLK: usize>(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    byte_base_b: usize,
    partials: &BstaticPartials,
    out: &mut [u8; 64],
    nt: u8,
) -> bool {
    let table = inv_table.data_ptr();

    let img2 = partials.img2;
    let table8 = if img2 {
        inv_table.half_swapped_data_ptr()
    } else {
        table
    };

    unsafe {
        let a_base = a_packed.as_ptr().add(byte_base_b);
        let b_base = b_packed.as_ptr().add(byte_base_b);

        let mut b_words = [0u64; 8];
        let mut hit = true;
        macro_rules! sniff {
            ($k:literal) => {{
                let p: BstaticRow = BSTATIC_PLAN[BLK][$k];
                if p.kind != ROW_GENERIC {
                    let w = u64::from_le(core::ptr::read_unaligned(
                        b_base.add($k * N_CHUNKS) as *const u64
                    ));
                    b_words[$k] = w;
                    hit &= (w & p.mask) == p.expected;
                }
            }};
        }
        sniff!(0);
        sniff!(1);
        sniff!(2);
        sniff!(3);
        sniff!(4);
        sniff!(5);
        sniff!(6);
        sniff!(7);
        if !hit {
            return false;
        }

        const ALL_FULLY_STATIC: [bool; 33] = {
            let mut t = [false; 33];
            let mut b = 0;
            while b < 33 {
                let mut all = true;
                let mut k = 0;
                while k < 8 {
                    let p = BSTATIC_PLAN[b][k];
                    if p.kind != ROW_STATIC || p.vary != 0 {
                        all = false;
                    }
                    k += 1;
                }
                t[b] = all;
                b += 1;
            }
            t
        };
        if ALL_FULLY_STATIC[BLK] {
            let parts = partials.rows[BLK].as_ptr() as *const u8;
            let acc = if img2 {
                all_static_acc::<true>(table, table8, a_base, parts)
            } else {
                all_static_acc::<false>(table, table8, a_base, parts)
            };
            super::x86_64::store_out64(out, acc, nt);
            return true;
        }

        let mut acc = _mm512_setzero_si512();
        macro_rules! row {
            ($k:literal) => {{
                let p: BstaticRow = BSTATIC_PLAN[BLK][$k];
                let a_ptr = a_base.add($k * N_CHUNKS);
                if p.kind == ROW_ZERO {
                } else if p.kind == ROW_STATIC {
                    let av = if img2 {
                        apply_full_2img(table, table8, a_ptr)
                    } else {
                        apply_full(table, a_ptr)
                    };
                    let part = _mm512_load_si512(partials.rows[BLK][$k].as_ptr() as *const __m512i);
                    if p.vary == 0 {
                        acc = _mm512_xor_si512(acc, _mm512_gf2p8mul_epi8(av, part));
                    } else {
                        let b_word = b_words[$k];
                        let mut bv = part;
                        let mut v = p.vary;
                        while v != 0 {
                            let j = v.trailing_zeros() as usize;
                            let byte = (b_word >> (8 * j)) as u8 as usize;
                            let r = _mm512_loadu_si512(table.add(byte * 64) as *const __m512i);
                            bv = _mm512_xor_si512(bv, perm_row(r, j));
                            v &= v - 1;
                        }
                        let prod = _mm512_gf2p8mul_epi8(av, bv);
                        let scaled = if $k == 0 {
                            prod
                        } else {
                            _mm512_gf2p8mul_epi8(prod, _mm512_set1_epi8((1u8 << $k) as i8))
                        };
                        acc = _mm512_xor_si512(acc, scaled);
                    }
                } else {
                    let av = apply_full(table, a_ptr);
                    let bv = apply_full(table, b_base.add($k * N_CHUNKS));
                    let prod = _mm512_gf2p8mul_epi8(av, bv);
                    let scaled = if $k == 0 {
                        prod
                    } else {
                        _mm512_gf2p8mul_epi8(prod, _mm512_set1_epi8((1u8 << $k) as i8))
                    };
                    acc = _mm512_xor_si512(acc, scaled);
                }
            }};
        }
        row!(0);
        row!(1);
        row!(2);
        row!(3);
        row!(4);
        row!(5);
        row!(6);
        row!(7);
        super::x86_64::store_out64(out, acc, nt);
        true
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn shift_reduce_inner_ab_x86_avx512_bstatic(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    chunk_byte_base: usize,
    b_med: usize,
    w: usize,
    partials: &BstaticPartials,
    out: &mut [u8; 64],
    nt: u8,
) -> bool {
    if w > 1 || b_med >= 16 {
        return false;
    }

    unsafe {
        shift_reduce_inner_ab_x86_avx512_bstatic_at(
            a_packed,
            b_packed,
            inv_table,
            chunk_byte_base + b_med * N_CHUNKS * 8,
            w * 16 + b_med,
            partials,
            out,
            nt,
        )
    }
}

#[inline(never)]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn shift_reduce_inner_ab_x86_avx512_bstatic_at(
    a_packed: &[u8],
    b_packed: &[u8],
    inv_table: &InvNttTableByteSingleGf8,
    byte_base_b: usize,
    blk: usize,
    partials: &BstaticPartials,
    out: &mut [u8; 64],
    nt: u8,
) -> bool {
    unsafe {
        if blk == 31 {
            kernel::<31>(
                a_packed,
                b_packed,
                inv_table,
                byte_base_b,
                partials,
                out,
                nt,
            )
        } else if blk == 30 {
            kernel::<30>(
                a_packed,
                b_packed,
                inv_table,
                byte_base_b,
                partials,
                out,
                nt,
            )
        } else if blk <= 1 {
            kernel::<0>(
                a_packed,
                b_packed,
                inv_table,
                byte_base_b,
                partials,
                out,
                nt,
            )
        } else if blk == 28 && bstatic_28_enabled() {
            kernel::<28>(
                a_packed,
                b_packed,
                inv_table,
                byte_base_b,
                partials,
                out,
                nt,
            )
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::x86_64::shift_reduce_inner_ab_x86_avx512;
    use super::super::x86_64_bstatic_plan::BSTATIC_GENERIC_PLAN;
    use super::*;
    use crate::ntt::AdditiveNttGf8;
    use crate::zerocheck::univariate_skip_optimized::K_SKIP;

    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
    }

    fn standard_table() -> InvNttTableByteSingleGf8 {
        let ntt_s = AdditiveNttGf8::new(K_SKIP, F8::ZERO);
        let ntt_l = AdditiveNttGf8::new(K_SKIP, F8(1u8 << K_SKIP));
        InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l)
    }

    #[test]
    fn bstatic_matches_incumbent_all_plans() {
        let inv_table = standard_table();
        let partials = build_partials(&inv_table);
        let mut rng = Rng(0xB57A_71C0);

        const OUTER: usize = 1024;
        let n_windows = 4;
        for plan in 0..=BSTATIC_GENERIC_PLAN {
            let (w, b_med) = if plan == BSTATIC_GENERIC_PLAN {
                (0, 3)
            } else {
                (plan / 16, plan % 16)
            };
            for mode in 0..4 {
                let mut a = vec![0u8; OUTER * n_windows];
                let mut b = vec![0u8; OUTER * n_windows];
                for x in 0..n_windows {
                    for bm in 0..16 {
                        for k in 0..8 {
                            let off = x * OUTER + bm * 64 + k * 8;
                            let a_word = rng.next_u64();
                            let p = BSTATIC_PLAN[plan.min(BSTATIC_BLOCKS - 1)][k];
                            let b_word = match mode {
                                0 => (p.expected & p.mask) | (rng.next_u64() & !p.mask),
                                1 => rng.next_u64(),
                                2 => 0,
                                _ => u64::MAX,
                            };
                            a[off..off + 8].copy_from_slice(&a_word.to_le_bytes());
                            b[off..off + 8].copy_from_slice(&b_word.to_le_bytes());
                        }
                    }
                }
                for x in 0..n_windows {
                    let mut got = [0u8; 64];
                    let mut want = [0u8; 64];

                    unsafe {
                        let byte_base_b = x * OUTER + b_med * N_CHUNKS * 8;
                        macro_rules! run {
                            ($($i:literal),*) => {
                                match plan {
                                    $($i => kernel::<$i>(&a, &b, &inv_table, byte_base_b, &partials, &mut got, 0),)*
                                    _ => unreachable!(),
                                }
                            };
                        }
                        let ran = run!(
                            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                            20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
                        );
                        shift_reduce_inner_ab_x86_avx512(
                            &a,
                            &b,
                            &inv_table,
                            x * OUTER,
                            b_med,
                            &mut want,
                            0,
                        );

                        if mode == 0 || plan == BSTATIC_GENERIC_PLAN {
                            assert!(
                                ran,
                                "plan {plan} mode {mode} window {x} unexpectedly missed"
                            );
                        }
                        if !ran {
                            got = want;
                        }

                        if plan != BSTATIC_GENERIC_PLAN {
                            let mut got2 = [0u8; 64];
                            let ran2 = shift_reduce_inner_ab_x86_avx512_bstatic(
                                &a,
                                &b,
                                &inv_table,
                                x * OUTER,
                                b_med,
                                w,
                                &partials,
                                &mut got2,
                                0,
                            );
                            let live = BSTATIC_LIVE[plan];
                            assert_eq!(
                                ran2,
                                ran && live,
                                "dispatcher/kernel disagree on plan {plan} mode {mode}"
                            );
                            if ran2 {
                                assert_eq!(
                                    got2, want,
                                    "dispatcher plan {plan} mode {mode} window {x}"
                                );
                            }
                        }
                    }
                    assert_eq!(got, want, "plan {plan} mode {mode} window {x}");
                }
            }
        }
    }

    #[test]
    fn partials_shape() {
        let inv_table = standard_table();
        let partials = build_partials(&inv_table);
        let mut f = [F8::ZERO; 64];
        for blk in 0..BSTATIC_BLOCKS {
            for k in 0..8 {
                let p = BSTATIC_PLAN[blk][k];
                if p.kind != ROW_STATIC {
                    assert_eq!(partials.rows[blk][k], [0u8; 64]);
                    continue;
                }
                inv_table.apply(&p.expected.to_le_bytes(), &mut f);
                for i in 0..64 {
                    let want = if p.vary == 0 {
                        f[i] * F8(1u8 << k)
                    } else {
                        f[i]
                    };
                    assert_eq!(partials.rows[blk][k][i], want.0, "blk {blk} k {k} lane {i}");
                }
            }
        }
    }

    #[test]
    fn prepare_bstatic_fingerprint() {
        if !fast_shift_reduce_enabled() {
            return;
        }
        let t1 = standard_table();
        let t2 = standard_table();
        assert!(prepare_bstatic(&t1).is_some());
        assert!(prepare_bstatic(&t2).is_some());

        let ntt_s = AdditiveNttGf8::new(K_SKIP, F8::ZERO);
        let ntt_l = AdditiveNttGf8::new(K_SKIP, F8(2u8 << K_SKIP));
        let t3 = InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);
        assert!(prepare_bstatic(&t3).is_none());

        assert!(prepare_bstatic(&t1).is_some());
    }

    #[test]
    fn perm_row_matches_incumbent_apply() {
        let inv_table = standard_table();
        let mut rng = Rng(0x9E4D);
        for _ in 0..256 {
            let word = rng.next_u64().to_le_bytes();
            let mut want = [F8::ZERO; 64];
            inv_table.apply(&word, &mut want);

            let got = unsafe {
                let v = apply_full(inv_table.data_ptr(), word.as_ptr());
                let mut o = [0u8; 64];
                _mm512_storeu_si512(o.as_mut_ptr() as *mut __m512i, v);
                o
            };
            let want: [u8; 64] = core::array::from_fn(|i| want[i].0);
            assert_eq!(got, want);
        }
    }
}

#[cfg(test)]
mod microbench {

    use super::super::x86_64::shift_reduce_inner_ab_x86_avx512;
    use super::*;
    use crate::ntt::AdditiveNttGf8;
    use crate::zerocheck::univariate_skip_optimized::K_SKIP;

    #[repr(C, align(64))]
    struct XkTable([[u8; 64]; 8]);
    static XK: XkTable = {
        let mut t = [[0u8; 64]; 8];
        let mut k = 0;
        while k < 8 {
            let mut i = 0;
            while i < 64 {
                t[k][i] = 1u8 << k;
                i += 1;
            }
            k += 1;
        }
        XkTable(t)
    };

    #[target_feature(enable = "gfni,avx512f,avx512bw")]
    unsafe fn kernel_dyn(
        a_packed: &[u8],
        b_packed: &[u8],
        inv_table: &InvNttTableByteSingleGf8,
        byte_base_b: usize,
        plan: &[BstaticRow; 8],
        parts: &[[u8; 64]; 8],
        out: &mut [u8; 64],
    ) -> bool {
        let table = inv_table.data_ptr();

        unsafe {
            let a_base = a_packed.as_ptr().add(byte_base_b);
            let b_base = b_packed.as_ptr().add(byte_base_b);
            let mut b_words = [0u64; 8];
            let mut hit = true;
            for k in 0..8 {
                let p = plan[k];
                if p.kind != ROW_GENERIC {
                    let w = u64::from_le(core::ptr::read_unaligned(
                        b_base.add(k * N_CHUNKS) as *const u64
                    ));
                    b_words[k] = w;
                    hit &= (w & p.mask) == p.expected;
                }
            }
            if !hit {
                return false;
            }
            let mut acc = _mm512_setzero_si512();

            for k in 0..core::hint::black_box(8usize) {
                let p = plan[k];
                if p.kind == ROW_ZERO {
                    continue;
                }
                let av = apply_full(table, a_base.add(k * N_CHUNKS));
                if p.kind == ROW_STATIC {
                    let mut bv = _mm512_load_si512(parts[k].as_ptr() as *const __m512i);
                    if p.vary == 0 {
                        acc = _mm512_xor_si512(acc, _mm512_gf2p8mul_epi8(av, bv));
                        continue;
                    }
                    let b_word = b_words[k];
                    let mut v = p.vary;
                    while v != 0 {
                        let j = v.trailing_zeros() as usize;
                        let byte = (b_word >> (8 * j)) as u8 as usize;
                        let r = _mm512_loadu_si512(table.add(byte * 64) as *const __m512i);
                        bv = _mm512_xor_si512(bv, perm_row(r, j));
                        v &= v - 1;
                    }
                    let prod = _mm512_gf2p8mul_epi8(av, bv);
                    let scaled = if k == 0 {
                        prod
                    } else {
                        _mm512_gf2p8mul_epi8(
                            prod,
                            _mm512_load_si512(XK.0[k].as_ptr() as *const __m512i),
                        )
                    };
                    acc = _mm512_xor_si512(acc, scaled);
                } else {
                    let bv = apply_full(table, b_base.add(k * N_CHUNKS));
                    let prod = _mm512_gf2p8mul_epi8(av, bv);
                    let scaled = if k == 0 {
                        prod
                    } else {
                        _mm512_gf2p8mul_epi8(
                            prod,
                            _mm512_load_si512(XK.0[k].as_ptr() as *const __m512i),
                        )
                    };
                    acc = _mm512_xor_si512(acc, scaled);
                }
            }
            _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, acc);
            true
        }
    }

    #[target_feature(enable = "gfni,avx512f,avx512bw")]
    unsafe fn kernel_rolled<const BLK: usize>(
        a_packed: &[u8],
        b_packed: &[u8],
        inv_table: &InvNttTableByteSingleGf8,
        byte_base_b: usize,
        partials: &BstaticPartials,
        out: &mut [u8; 64],
    ) -> bool {
        unsafe {
            kernel_dyn(
                a_packed,
                b_packed,
                inv_table,
                byte_base_b,
                &BSTATIC_PLAN[BLK],
                &partials.rows[BLK],
                out,
            )
        }
    }

    #[test]
    #[ignore]
    fn bstatic_cycle_bench() {
        let ntt_s = AdditiveNttGf8::new(K_SKIP, F8::ZERO);
        let ntt_l = AdditiveNttGf8::new(K_SKIP, F8(1u8 << K_SKIP));
        let inv_table = InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);
        let partials: &'static BstaticPartials = Box::leak(build_partials(&inv_table));
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut rnd = || {
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };

        const BLOCK: usize = 2048;
        let n_blocks = 32;
        let mut a = vec![0u8; BLOCK * n_blocks];
        let mut b = vec![0u8; BLOCK * n_blocks];
        for blk in 0..n_blocks {
            for w in 0..2 {
                for bm in 0..16 {
                    for k in 0..8 {
                        let off = blk * BLOCK + w * 1024 + bm * 64 + k * 8;
                        let p = BSTATIC_PLAN[w * 16 + bm][k];
                        a[off..off + 8].copy_from_slice(&rnd().to_le_bytes());
                        b[off..off + 8].copy_from_slice(
                            &((p.expected & p.mask) | (rnd() & !p.mask)).to_le_bytes(),
                        );
                    }
                }
            }
        }
        let iters = 20_000usize;
        let mut out = [0u8; 64];
        let mut a_col = [F8::ZERO; 64];
        let mut b_col = [F8::ZERO; 64];
        let mut sink = 0u64;
        for round in 0..3 {
            let t = std::time::Instant::now();
            for i in 0..iters {
                let base = (i % n_blocks) * BLOCK;
                for w in 0..2 {
                    for bm in 0..16 {
                        super::super::shift_reduce_inner_ab(
                            &a,
                            &b,
                            &inv_table,
                            base + w * 1024,
                            bm,
                            &mut out,
                            &mut a_col,
                            &mut b_col,
                            None,
                            0,
                        );
                        sink ^= out[0] as u64;
                    }
                }
            }
            let off_ns = t.elapsed().as_secs_f64() * 1e9 / (iters * 32) as f64;
            let t = std::time::Instant::now();
            for i in 0..iters {
                let base = (i % n_blocks) * BLOCK;
                for w in 0..2 {
                    for bm in 0..16 {
                        super::super::shift_reduce_inner_ab(
                            &a,
                            &b,
                            &inv_table,
                            base + w * 1024,
                            bm,
                            &mut out,
                            &mut a_col,
                            &mut b_col,
                            Some((w, partials)),
                            0,
                        );
                        sink ^= out[0] as u64;
                    }
                }
            }
            let live_ns = t.elapsed().as_secs_f64() * 1e9 / (iters * 32) as f64;
            let t = std::time::Instant::now();
            for i in 0..iters {
                let base = (i % n_blocks) * BLOCK;
                for w in 0..2 {
                    for bm in 0..16 {
                        let blk = w * 16 + bm;
                        let ok = unsafe {
                            kernel_dyn(
                                &a,
                                &b,
                                &inv_table,
                                base + w * 1024 + bm * 64,
                                &BSTATIC_PLAN[blk],
                                &partials.rows[blk],
                                &mut out,
                            )
                        };
                        assert!(ok);
                        sink ^= out[0] as u64;
                    }
                }
            }
            let dyn_ns = t.elapsed().as_secs_f64() * 1e9 / (iters * 32) as f64;
            let t = std::time::Instant::now();
            for i in 0..iters {
                let base = (i % n_blocks) * BLOCK;
                for w in 0..2 {
                    for bm in 0..16 {
                        let blk = w * 16 + bm;
                        let bb = base + w * 1024 + bm * 64;
                        macro_rules! run { ($($i:literal),*) => { match blk { $($i => unsafe { kernel_rolled::<$i>(&a, &b, &inv_table, bb, partials, &mut out) },)* _ => unreachable!() } }; }
                        let ok = run!(
                            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                            20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
                        );
                        assert!(ok);
                        sink ^= out[0] as u64;
                    }
                }
            }
            let rolled_ns = t.elapsed().as_secs_f64() * 1e9 / (iters * 32) as f64;
            println!(
                "cycle round {round}: incumbent {off_ns:6.2} ns/call  live-plan {live_ns:6.2} ({:.3})  dyn-all-blocks {dyn_ns:6.2} ({:.3})  rolled-per-blk-all {rolled_ns:6.2} ({:.3})  sink {sink}",
                live_ns / off_ns,
                dyn_ns / off_ns,
                rolled_ns / off_ns
            );
        }

        for blk in 0..n_blocks {
            let base = blk * BLOCK;
            for w in 0..2 {
                for bm in 0..16 {
                    let plan_i = w * 16 + bm;
                    let mut want = [0u8; 64];
                    let mut got = [0u8; 64];
                    unsafe {
                        shift_reduce_inner_ab_x86_avx512(
                            &a,
                            &b,
                            &inv_table,
                            base + w * 1024,
                            bm,
                            &mut want,
                            0,
                        );
                        assert!(kernel_dyn(
                            &a,
                            &b,
                            &inv_table,
                            base + w * 1024 + bm * 64,
                            &BSTATIC_PLAN[plan_i],
                            &partials.rows[plan_i],
                            &mut got
                        ));
                    }
                    assert_eq!(got, want, "kernel_dyn mismatch blk {blk} w {w} bm {bm}");
                }
            }
        }
    }

    #[test]
    #[ignore]
    fn bstatic_microbench() {
        let ntt_s = AdditiveNttGf8::new(K_SKIP, F8::ZERO);
        let ntt_l = AdditiveNttGf8::new(K_SKIP, F8(1u8 << K_SKIP));
        let inv_table = InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);
        let partials: &'static BstaticPartials = Box::leak(build_partials(&inv_table));
        let mut seed = 0x1234_5678_9abc_def0u64;
        let mut rnd = || {
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };

        const OUTER: usize = 1024;
        let n_windows = 64;
        let iters = 200_000usize;
        let sweep: Vec<(usize, bool)> = if std::env::var_os("BSTATIC_SWEEP_ALL").is_some() {
            (0..32).map(|b| (b, true)).collect()
        } else {
            vec![
                (0usize, true),
                (1, true),
                (3, true),
                (12, true),
                (20, true),
                (30, true),
                (31, true),
                (3, false),
                (12, false),
            ]
        };
        for &(plan, hits) in &sweep {
            let (w, b_med) = (plan / 16, plan % 16);
            let mut a = vec![0u8; OUTER * n_windows];
            let mut b = vec![0u8; OUTER * n_windows];
            for x in 0..n_windows {
                for bm in 0..16 {
                    for k in 0..8 {
                        let off = x * OUTER + bm * 64 + k * 8;
                        let p = BSTATIC_PLAN[plan][k];
                        let a_word = rnd();
                        let b_word = if hits {
                            (p.expected & p.mask) | (rnd() & !p.mask)
                        } else {
                            rnd()
                        };
                        a[off..off + 8].copy_from_slice(&a_word.to_le_bytes());
                        b[off..off + 8].copy_from_slice(&b_word.to_le_bytes());
                    }
                }
            }
            let mut out = [0u8; 64];
            let mut sink = 0u64;
            let mut a_col = [F8::ZERO; 64];
            let mut b_col = [F8::ZERO; 64];

            let t = std::time::Instant::now();
            for i in 0..iters {
                let x = i % n_windows;
                super::super::shift_reduce_inner_ab(
                    &a,
                    &b,
                    &inv_table,
                    x * OUTER,
                    b_med,
                    &mut out,
                    &mut a_col,
                    &mut b_col,
                    None,
                    0,
                );
                sink ^= out[0] as u64 ^ (out[63] as u64) << 8;
            }
            let prod_off = t.elapsed().as_secs_f64() * 1e9 / iters as f64;
            let t = std::time::Instant::now();
            for i in 0..iters {
                let x = i % n_windows;
                super::super::shift_reduce_inner_ab(
                    &a,
                    &b,
                    &inv_table,
                    x * OUTER,
                    b_med,
                    &mut out,
                    &mut a_col,
                    &mut b_col,
                    Some((w, partials)),
                    0,
                );
                sink ^= out[0] as u64 ^ (out[63] as u64) << 8;
            }
            let prod_on = t.elapsed().as_secs_f64() * 1e9 / iters as f64;
            println!(
                "plan {plan:2} hits={hits:5}: PRODUCTION entry hint-off {prod_off:6.1}  hint-on {prod_on:6.1} ns/call  ratio {:.2}",
                prod_on / prod_off
            );

            let t = std::time::Instant::now();
            for i in 0..iters {
                let x = i % n_windows;
                unsafe {
                    shift_reduce_inner_ab_x86_avx512(
                        &a,
                        &b,
                        &inv_table,
                        x * OUTER,
                        b_med,
                        &mut out,
                        0,
                    );
                }
                sink ^= out[0] as u64 ^ (out[63] as u64) << 8;
            }
            let inc = t.elapsed().as_secs_f64() * 1e9 / iters as f64;

            let t = std::time::Instant::now();
            for i in 0..iters {
                let x = i % n_windows;
                unsafe {
                    let ok = shift_reduce_inner_ab_x86_avx512_bstatic(
                        &a,
                        &b,
                        &inv_table,
                        x * OUTER,
                        b_med,
                        w,
                        &partials,
                        &mut out,
                        0,
                    );
                    if !ok {
                        shift_reduce_inner_ab_x86_avx512(
                            &a,
                            &b,
                            &inv_table,
                            x * OUTER,
                            b_med,
                            &mut out,
                            0,
                        );
                    }
                }
                sink ^= out[0] as u64 ^ (out[63] as u64) << 8;
            }
            let spec = t.elapsed().as_secs_f64() * 1e9 / iters as f64;

            let t = std::time::Instant::now();
            for i in 0..iters {
                let x = i % n_windows;
                unsafe {
                    let byte_base_b = x * OUTER + b_med * N_CHUNKS * 8;
                    let ok = kernel::<32>(&a, &b, &inv_table, byte_base_b, &partials, &mut out, 0);
                    assert!(ok);
                }
                sink ^= out[0] as u64 ^ (out[63] as u64) << 8;
            }
            let gen32 = t.elapsed().as_secs_f64() * 1e9 / iters as f64;

            let t = std::time::Instant::now();
            for i in 0..iters {
                let x = i % n_windows;
                unsafe {
                    let byte_base_b = x * OUTER + b_med * N_CHUNKS * 8;
                    macro_rules! run {
                        ($($i:literal),*) => { match plan { $($i => kernel::<$i>(&a, &b, &inv_table, byte_base_b, &partials, &mut out, 0),)* _ => unreachable!() } };
                    }
                    let ok = run!(
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
                    );
                    if !ok {
                        shift_reduce_inner_ab_x86_avx512(
                            &a,
                            &b,
                            &inv_table,
                            x * OUTER,
                            b_med,
                            &mut out,
                            0,
                        );
                    }
                }
                sink ^= out[0] as u64 ^ (out[63] as u64) << 8;
            }
            let direct = t.elapsed().as_secs_f64() * 1e9 / iters as f64;
            println!(
                "plan {plan:2} hits={hits:5}: incumbent {inc:6.1}  dispatch+spec {spec:6.1}  direct-spec {direct:6.1}  direct-generic32 {gen32:6.1} ns/call  (sink {sink})"
            );
        }
    }
}
