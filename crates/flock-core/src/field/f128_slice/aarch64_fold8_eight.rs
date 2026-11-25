use super::*;

const BANKS: usize = 64;
const SLOTS: usize = 8;
const TILE: usize = BANKS * SLOTS;
type Tile = [F128; TILE];
type Output = [F128; SLOTS];

#[derive(Clone, Copy)]
struct Acc {
    ll: uint64x2_t,
    cross: uint64x2_t,
    hh: uint64x2_t,
}

// Same schoolbook products and reduction as the four-slot leaf. Scan each
// bank pair once for eight slots. Finalize odd slots before the even-only
// tail so those accumulators do not stay live through the tail.
#[inline(never)]
#[target_feature(enable = "aes")]
pub(in crate::field::f128_slice) unsafe fn one_scan<const ODD_LIVE: usize, const LOW_ONLY: bool>(
    weights: &[F128; BANKS],
    tile: &Tile,
    out: &mut Output,
) {
    const { assert!(ODD_LIVE <= BANKS) };
    const fn low_only(live: usize, low: bool) -> bool {
        low && live < BANKS && live % 2 == 1
    }
    let low_tail = low_only(ODD_LIVE, LOW_ONLY);
    let split = if low_tail {
        ODD_LIVE - 1
    } else {
        (ODD_LIVE + 1) & !1
    };
    let zero = vdupq_n_u64(0);
    let mut acc = [Acc {
        ll: zero,
        cross: zero,
        hh: zero,
    }; SLOTS];
    unsafe {
        let w = weights.as_ptr();
        let x = tile.as_ptr();
        macro_rules! load_weights {
            ($bank:expr) => {{
                let w0 = vld1q_u64(w.add($bank).cast::<u64>());
                let w1 = vld1q_u64(w.add($bank + 1).cast::<u64>());
                (w0, swap_halves(w0), w1, swap_halves(w1))
            }};
        }
        macro_rules! accumulate {
            ($bank:expr, $pair:ident, $($slot:literal),+) => {{
                let (w0, w0_sw, w1, w1_sw) = $pair;
                $(
                    let row = x.add($slot * BANKS + $bank);
                    let Acc { ll, cross, hh } = &mut acc[$slot];
                    tri_acc_two_banks(
                        ll, cross, hh, load_whole(row), w0, w0_sw,
                        load_whole(row.add(1)), w1, w1_sw,
                    );
                )+
            }};
        }
        for bank in (0..split).step_by(2) {
            let pair = load_weights!(bank);
            accumulate!(bank, pair, 0, 1, 2, 3, 4, 5, 6, 7);
        }
        if low_tail {
            let pair = load_weights!(split);
            accumulate!(split, pair, 0, 2, 4, 6);
            let (w0, w0_sw, _, _) = pair;
            for slot in [1, 3, 5, 7] {
                let xl = vgetq_lane_u64::<0>(load_whole(x.add(slot * BANKS + split)));
                let a = &mut acc[slot];
                a.ll = veorq_u64(a.ll, pmull(xl, vgetq_lane_u64::<0>(w0)));
                a.cross = veorq_u64(a.cross, pmull(xl, vgetq_lane_u64::<0>(w0_sw)));
            }
        }
        let wide = |a: Acc| WideNeon {
            lo: veorq_u64(a.ll, vextq_u64::<1>(zero, a.cross)),
            hi: veorq_u64(a.hh, vextq_u64::<1>(a.cross, zero)),
        };
        macro_rules! finish {
            ($a:literal, $b:literal) => {{
                let [first, second] = reduce_wide_pair(wide(acc[$a]), wide(acc[$b]));
                out[$a] = transmute::<uint64x2_t, F128>(first);
                out[$b] = transmute::<uint64x2_t, F128>(second);
            }};
        }
        finish!(1, 3);
        finish!(5, 7);
        let even_tail = split + if low_tail { 2 } else { 0 };
        for bank in (even_tail..BANKS).step_by(2) {
            let pair = load_weights!(bank);
            accumulate!(bank, pair, 0, 2, 4, 6);
        }
        finish!(0, 2);
        finish!(4, 6);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::gf2_128::software::ghash_mul;
    use std::hint::black_box;

    #[inline(never)]
    #[target_feature(enable = "aes")]
    unsafe fn two_scans<const ODD_LIVE: usize, const LOW_ONLY: bool>(
        weights: &[F128; BANKS],
        tile: &Tile,
        out: &mut Output,
    ) {
        let (first, second) = out.split_at_mut(4);
        let (a, b) = tile.split_at(TILE / 2);
        unsafe {
            fold_banked_slots4_whole_into::<BANKS>(weights, a, first, ODD_LIVE, LOW_ONLY);
            fold_banked_slots4_whole_into::<BANKS>(weights, b, second, ODD_LIVE, LOW_ONLY);
        }
    }

    fn rng(state: &mut u64) -> F128 {
        let mut next = || {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            *state
        };
        F128::new(next(), next())
    }

    fn prepare_tile(tile: &mut Tile, live: usize, low: bool, state: &mut u64) {
        for (slot, row) in tile.chunks_exact_mut(BANKS).enumerate() {
            row.fill_with(|| rng(state));
            if slot % 2 == 1 {
                row[live..].fill(F128::ZERO);
                if live % 2 == 1 && live < BANKS && low {
                    row[live - 1].hi = 0;
                }
            }
        }
    }

    #[repr(align(128))]
    struct AlignedInput([F128; 2 * TILE + 8]);

    #[test]
    fn eight_slot_fold_matches_scalar_and_two_scans() {
        const SENTINEL: F128 = F128 {
            lo: 0x64be_cdc9_04aa_e6cb,
            hi: 0xdd38_5ba2_1fa3_78c6,
        };
        let mut seed = 0xd009_547c_7ef1_bdac;
        let mut input = Box::new(AlignedInput([F128::ZERO; 2 * TILE + 8]));
        let mut weight_buf = [F128::ZERO; BANKS + 1];
        macro_rules! check {
            ($live:literal, $low:literal) => {{
                for (offset, weight_offset) in [(0, 0), (1, 1), (7, 0)] {
                    let weights: &mut [F128; BANKS] = (&mut weight_buf
                        [weight_offset..weight_offset + BANKS])
                        .try_into()
                        .unwrap();
                    weights.fill_with(|| rng(&mut seed));
                    for chunk in input.0[offset..offset + 2 * TILE].chunks_exact_mut(TILE) {
                        let tile: &mut Tile = chunk.try_into().unwrap();
                        prepare_tile(tile, $live, $low, &mut seed);
                        let mut exact = [SENTINEL; SLOTS + 2];
                        let mut reference = [F128::ZERO; SLOTS];
                        unsafe {
                            one_scan::<$live, $low>(
                                weights,
                                tile,
                                (&mut exact[1..1 + SLOTS]).try_into().unwrap(),
                            );
                            two_scans::<$live, $low>(weights, tile, &mut reference);
                        }
                        for (k, row) in tile.chunks_exact(BANKS).enumerate() {
                            let scalar = row
                                .iter()
                                .zip(weights.iter())
                                .fold(F128::ZERO, |sum, (&x, &w)| sum + ghash_mul(x, w));
                            assert_eq!(
                                exact[1 + k],
                                scalar,
                                "live={} low={} slot={k}",
                                $live,
                                $low
                            );
                        }
                        assert_eq!(exact[1..1 + SLOTS], reference);
                        assert_eq!(exact[0], SENTINEL);
                        assert_eq!(exact[SLOTS + 1], SENTINEL);
                    }
                }
            }};
        }
        check!(0, false);
        check!(1, true);
        check!(19, false);
        check!(56, true);
        check!(57, false);
        check!(57, true);
        check!(63, true);
        check!(64, false);
    }

    type Kernel = unsafe fn(&[F128; BANKS], &Tile, &mut Output);

    #[test]
    #[ignore = "isolated native timing"]
    fn eight_slot_fold_component_time() {
        let mut seed = 0x74c9_113d_8d50_72ae;
        let weights = std::array::from_fn(|_| rng(&mut seed));
        let mut tile = Box::new([F128::ZERO; TILE]);
        prepare_tile(&mut tile, 57, true, &mut seed);
        let kernels: [Kernel; 2] = [two_scans::<57, true>, one_scan::<57, true>];
        let bout = |kernel: Kernel, out: &mut Output| {
            let start = std::time::Instant::now();
            for _ in 0..2048 {
                unsafe { kernel(black_box(&weights), black_box(&tile), out) };
                black_box(&*out);
            }
            start.elapsed().as_nanos()
        };
        let mut out = [F128::ZERO; SLOTS];
        for kernel in kernels {
            black_box(bout(kernel, &mut out));
        }
        for pair in 0..24 {
            let first = pair % 2;
            let mut times = [0; 2];
            times[first] = bout(kernels[first], &mut out);
            times[1 - first] = bout(kernels[1 - first], &mut out);
            eprintln!(
                "fold8-eight pair={pair} eight_first={} two_ns={} one_ns={}",
                first == 1,
                times[0],
                times[1]
            );
        }
    }
}
