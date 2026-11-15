use super::*;
use core::sync::atomic::AtomicBool;

#[derive(Clone)]
#[repr(align(128))]
struct Line([u8; 128]);

fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Compare every output byte, including poisoned gaps and allocation guards,
/// and all eight message slots for each independent memory option.
#[test]
fn r2_memory_matches_incumbent() {
    let mut state = 0x7864_5475_3412_8976;
    for size in [128usize, 256, 1024] {
        let mut a = vec![0u8; size * 16];
        let mut b = vec![0u8; size * 16];
        let eq: Vec<F128> = (0..size)
            .map(|_| F128::new(next(&mut state), next(&mut state)))
            .collect();
        for input in 0..3 {
            for byte in a.iter_mut().chain(b.iter_mut()) {
                *byte = match input {
                    0 => 0,
                    1 => 0xff,
                    _ => next(&mut state) as u8,
                };
            }
            // The compact head stores require constant-one rows in the head.
            for block in 0..size / 128 {
                b[block * 2048..block * 2048 + 128].fill(0xff);
            }
            for table_case in 0..3 {
                let mut table: Vec<F128> = (0..8 * 256)
                    .map(|_| {
                        if table_case == 0 {
                            F128::ZERO
                        } else {
                            F128::new(next(&mut state), next(&mut state))
                        }
                    })
                    .collect();
                if table_case == 2 {
                    let rest = (0..7).fold(F128::ZERO, |sum, c| sum + table[c * 256 + 255]);
                    table[7 * 256 + 255] = F128::ONE + rest;
                }
                for align in (0..128).step_by(16) {
                    for (mask, useful, base, skip, tail, head, degen) in [
                        (127, 121, 0, true, false, false, false),
                        (127, 121, 128, true, true, false, true),
                        (127, 121, 0, true, true, true, true),
                        (127, 121, 0, false, false, false, true),
                        (127, 121, 64, false, false, false, true),
                        (usize::MAX, usize::MAX, 0, false, false, false, false),
                    ] {
                        if head && table_case != 2 {
                            continue;
                        }
                        let run = |variant| {
                            let poison = F128::new(0x1234_5678_90ab_cdef, 0xfedc_ba09_8765_4321);
                            let mut anchors = vec![poison; 2 * size + 16];
                            let mut backing = vec![Line([0xa5; 128]); (16 * size + 384) / 128];
                            let mut out = [poison; 8];
                            let fail = AtomicBool::new(false);
                            let kernel = match variant {
                                0 => fold_round2_compact_chunk_neon_prescaled_8_impl::<true, true, true, true>,
                                1 => candidate::<true, false>,
                                2 => candidate::<false, true>,
                                _ => candidate::<true, true>,
                            };
                            unsafe {
                                kernel(
                                    table.as_ptr().cast::<u8>(),
                                    a.as_ptr(),
                                    b.as_ptr(),
                                    anchors.as_mut_ptr().add(8),
                                    backing.as_mut_ptr().cast::<u8>().add(128 + align),
                                    eq.as_ptr(),
                                    size,
                                    base,
                                    mask,
                                    useful,
                                    degen,
                                    skip,
                                    tail,
                                    if head { &fail } else { core::ptr::null() },
                                    out.as_mut_ptr(),
                                );
                            }
                            let bytes = backing.iter().flat_map(|line| line.0).collect::<Vec<_>>();
                            (anchors, bytes, out, fail.load(Ordering::Relaxed))
                        };
                        let want = run(0);
                        for variant in 1..=3 {
                            let got = run(variant);
                            let at = format!("size {size}, input {input}, table {table_case}, align {align}, mask {mask}, base {base}, skip {skip}, tail {tail}, head {head}, degen {degen}, variant {variant}");
                            assert_eq!(got, want, "{at}");
                        }
                    }
                }
            }
        }
    }
}
