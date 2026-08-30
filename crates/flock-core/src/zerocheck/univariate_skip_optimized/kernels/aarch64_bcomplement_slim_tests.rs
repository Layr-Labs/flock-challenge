//! Oracles for the fourteen formerly-generic seven-vary positions only.
//! The other eighty interior partial rows retain the incumbent global cache.
//! Full-window tests therefore use the standard domain. Other domains test
//! one new position at a time, with all old partial guards forced to miss.

use super::{
    BSTATIC_MASKS, StaticBContext, b_complement_table_is_one,
    prepare_static_b_context_with_complement_policy, shift_reduce_inner_ab_bstatic,
};
use core::mem::{align_of, size_of};
use crate::field::F8;
use crate::ntt::{AdditiveNttGf8, InvNttTableByteSingleGf8};

const BLOCK_BYTES: usize = 2048;
// (window, K, known-ff byte). Fourteen positions, but only two byte masks.
const SITES: [(usize, usize, usize); 14] = [
    (3, 0, 7),
    (6, 5, 0),
    (8, 3, 7),
    (11, 4, 0),
    (12, 0, 0),
    (13, 2, 7),
    (13, 6, 7),
    (16, 7, 0),
    (18, 5, 7),
    (22, 2, 0),
    (24, 0, 7),
    (27, 1, 0),
    (27, 5, 0),
    (28, 7, 7),
];

fn one_bit(bit: usize) -> bool {
    bit < 1153
        || (0..56).any(|g| {
            let start = 1153 + 250 * g;
            (start + 186..start + 250).contains(&bit)
        })
        || (15153..15409).contains(&bit)
}

fn independent_mask(blk: usize, k: usize) -> u64 {
    let mut mask = 0u64;
    for byte in 0..8 {
        let first = blk * 512 + k * 64 + byte * 8;
        if (first..first + 8).all(one_bit) {
            mask |= 0xffu64 << (8 * byte);
        }
    }
    mask
}

fn make_table(k: usize, beta: u8) -> InvNttTableByteSingleGf8 {
    InvNttTableByteSingleGf8::new(
        &AdditiveNttGf8::new(k, F8::ZERO),
        &AdditiveNttGf8::new(k, F8(beta)),
    )
}

fn prepared(table: &InvNttTableByteSingleGf8) -> StaticBContext {
    // The old partial caches have one process-wide table. Initialize them
    // from the standard domain before preparing any isolated nonstandard
    // test. requested=false would select Legacy and would NOT initialize them.
    let standard = InvNttTableByteSingleGf8::cached_standard_k6();
    let first =
        prepare_static_b_context_with_complement_policy(standard, true, false, false, true)
            .expect("standard context");
    assert!(matches!(first, StaticBContext::Prepared { .. }));
    let context =
        prepare_static_b_context_with_complement_policy(table, true, false, false, true)
            .expect("certified table");
    assert!(matches!(context, StaticBContext::Prepared { .. }));
    context
}

fn inputs(pattern: u8, prefix: usize) -> (Vec<u8>, Vec<u8>) {
    let mut state = 0xbc00_1eaf_5eed_7701u64 ^ u64::from(pattern);
    let mut a = vec![0xc3; prefix + BLOCK_BYTES];
    let mut b = vec![0x69; prefix + BLOCK_BYTES];
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 23) as u8
    };
    for byte in prefix..prefix + BLOCK_BYTES {
        a[byte] = match pattern {
            0 => 0,
            1 => 0xff,
            2 => 0x55,
            _ => next(),
        };
        b[byte] = match pattern {
            0 => 0,
            1 => 0xff,
            2 => 0xaa,
            _ => next(),
        };
    }
    for bit in 0..BLOCK_BYTES * 8 {
        if one_bit(bit) {
            b[prefix + bit / 8] |= 1u8 << (bit % 8);
        } else if bit >= 15409 {
            b[prefix + bit / 8] &= !(1u8 << (bit % 8));
        }
    }
    (a, b)
}

fn scalar(
    a: &[u8],
    b: &[u8],
    table: &InvNttTableByteSingleGf8,
    prefix: usize,
    blk: usize,
) -> [u8; 64] {
    let mut result = [F8::ZERO; 64];
    for k in 0..8 {
        let off = prefix + blk * 64 + k * 8;
        let mut a_row = [F8::ZERO; 64];
        let mut b_row = [F8::ZERO; 64];
        table.apply_scalar(&a[off..off + 8], &mut a_row);
        table.apply_scalar(&b[off..off + 8], &mut b_row);
        for lane in 0..64 {
            result[lane] += a_row[lane] * b_row[lane] * F8(1u8 << k);
        }
    }
    result.map(|value| value.0)
}

#[repr(align(64))]
struct AlignedRow([u8; 64]);

fn kernel(
    a: &[u8],
    b: &[u8],
    table: &InvNttTableByteSingleGf8,
    prefix: usize,
    blk: usize,
    context: StaticBContext,
    fast: bool,
    nt: bool,
) -> [u8; 64] {
    let mut out = AlignedRow([0x96; 64]);
    let w = blk / 16;
    let b_med = blk % 16;
    let base = prefix + w * 1024;
    let handled = if fast {
        shift_reduce_inner_ab_bstatic::<true>(
            a, b, table, base, b_med, w, context, &mut out.0, nt,
        )
    } else {
        shift_reduce_inner_ab_bstatic::<false>(
            a, b, table, base, b_med, w, context, &mut out.0, nt,
        )
    };
    assert!(handled, "live window {blk}");
    if nt {
        // Test-only readback fence; the production store/fence policy stays
        // with the incumbent caller and is not changed by this candidate.
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    }
    out.0
}

fn with_large_stack(f: impl FnOnce() + Send + 'static) {
    // Like the incumbent generated-kernel oracles: debug builds can retain
    // stack slots from every expanded match arm. This is test-only storage.
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(f)
        .expect("spawn slim complement oracle")
        .join()
        .expect("slim complement oracle");
}

#[test]
fn slim_sites_are_fourteen_positions_with_two_masks() {
    let mut found = Vec::new();
    let mut old_partials = 0;
    let mut shapes = [0usize; 8];
    for blk in 3..29 {
        for k in 0..8 {
            let mask = independent_mask(blk, k);
            assert_eq!(BSTATIC_MASKS[blk][k], (mask, mask));
            if mask.count_ones() == 8 {
                let byte = mask.trailing_zeros() as usize / 8;
                assert_eq!(mask, 0xffu64 << (8 * byte));
                found.push((blk, k, byte));
                shapes[byte] += 1;
            } else if mask != 0 {
                old_partials += 1;
            }
        }
    }
    assert_eq!(found.as_slice(), SITES.as_slice());
    assert_eq!(old_partials, 80);
    assert_eq!(shapes, [7, 0, 0, 0, 0, 0, 0, 7]);
}

#[test]
fn slim_context_keeps_the_incumbent_payload_shape() {
    #[allow(dead_code)]
    enum IncumbentContext {
        Prepared {
            partials: &'static [[u8; 64]; 248],
            static_a_k1: &'static [u8; 64],
        },
        LegacyPerCall,
    }
    // Do not assume or report any concrete ABI size. Compare the original
    // type shape on the target that actually runs this test.
    assert_eq!(size_of::<StaticBContext>(), size_of::<IncumbentContext>());
    assert_eq!(
        size_of::<Option<StaticBContext>>(),
        size_of::<Option<IncumbentContext>>()
    );
    assert_eq!(align_of::<StaticBContext>(), align_of::<IncumbentContext>());
}

#[test]
fn slim_prepared_requires_the_actual_table_certificate() {
    let standard = InvNttTableByteSingleGf8::cached_standard_k6();
    prepared(standard);
    for beta in [64, 128, 192] {
        let table = make_table(6, beta);
        let mut folded = [F8::ZERO; 64];
        table.apply_scalar(&[0xff; 8], &mut folded);
        assert_eq!(folded, [F8::ONE; 64]);
        assert!(b_complement_table_is_one(&table));
        prepared(&table);
    }
    for k in [5, 7] {
        let table = make_table(k, 1u8 << k);
        assert!(!b_complement_table_is_one(&table));
        assert!(
            prepare_static_b_context_with_complement_policy(&table, true, false, false, true)
                .is_none()
        );
    }
    for (layout, legacy, context_legacy, requested, prepared_expected, legacy_expected) in [
        (false, false, false, true, false, false),
        (true, true, false, true, false, false),
        (true, false, true, true, false, true),
        (true, false, false, false, false, true),
        (true, false, false, true, true, false),
    ] {
        let context = prepare_static_b_context_with_complement_policy(
            standard,
            layout,
            legacy,
            context_legacy,
            requested,
        );
        assert_eq!(
            matches!(context, Some(StaticBContext::Prepared { .. })),
            prepared_expected
        );
        assert_eq!(
            matches!(context, Some(StaticBContext::LegacyPerCall)),
            legacy_expected
        );
        assert_eq!(context.is_none(), !prepared_expected && !legacy_expected);
    }
}

#[test]
fn slim_standard_windows_match_legacy_and_scalar() {
    with_large_stack(|| {
        let table = InvNttTableByteSingleGf8::cached_standard_k6();
        let on = prepared(table);
        let off =
            prepare_static_b_context_with_complement_policy(table, true, false, false, false)
                .expect("legacy rollback");
        for prefix in [0, 64] {
            for pattern in 0..4 {
                let (a, b) = inputs(pattern, prefix);
                let saved_a = a.clone();
                let saved_b = b.clone();
                for blk in 0..31 {
                    let want = scalar(&a, &b, table, prefix, blk);
                    for fast in [false, true] {
                        for nt in [false, true] {
                            for context in [on, off] {
                                assert_eq!(
                                    kernel(&a, &b, table, prefix, blk, context, fast, nt),
                                    want,
                                    "p={prefix} pattern={pattern} blk={blk} fast={fast} nt={nt}"
                                );
                            }
                        }
                    }
                }
                assert_eq!(a, saved_a);
                assert_eq!(b, saved_b);
            }
        }
    });
}

#[test]
fn slim_other_domains_isolate_each_new_row_and_its_guard_fallback() {
    with_large_stack(|| {
        for beta in [128, 192] {
            let table = make_table(6, beta);
            let on = prepared(&table);
            for &(blk, k, known_byte) in &SITES {
                const PREFIX: usize = 64;
                let mut a = vec![0; PREFIX + BLOCK_BYTES];
                let mut b = vec![0; PREFIX + BLOCK_BYTES];
                let off = PREFIX + blk * 64 + k * 8;
                a[off..off + 8].fill(0xff);
                // Other B rows stay zero. Every old partial at this interior
                // window has a nonzero ff expectation, so its guard misses:
                // this does NOT test or claim cross-domain partial caching.
                for other in 0..8 {
                    if other != k && BSTATIC_MASKS[blk][other].0 != 0 {
                        assert_ne!(BSTATIC_MASKS[blk][other].1, 0);
                    }
                }
                // Zero complement, 56 varying basis bits, and all eight
                // single-bit failures of the one required ff byte.
                for bit in core::iter::once(None).chain((0..64).map(Some)) {
                    let word = bit.map_or(u64::MAX, |bit| u64::MAX ^ (1u64 << bit));
                    b[off..off + 8].copy_from_slice(&word.to_le_bytes());
                    let mask = 0xffu64 << (8 * known_byte);
                    let expected_hit = bit.is_none_or(|bit| bit / 8 != known_byte);
                    assert_eq!((word & mask) == mask, expected_hit);
                    let want = scalar(&a, &b, &table, PREFIX, blk);
                    for fast in [false, true] {
                        for context in [on, StaticBContext::LegacyPerCall] {
                            assert_eq!(
                                kernel(&a, &b, &table, PREFIX, blk, context, fast, false),
                                want,
                                "beta={beta} blk={blk} k={k} bit={bit:?} fast={fast}"
                            );
                        }
                    }
                }
            }
        }
    });
}

#[test]
fn slim_unsupported_windows_do_not_write() {
    with_large_stack(|| {
        let table = InvNttTableByteSingleGf8::cached_standard_k6();
        let context = prepared(table);
        for (w, b_med) in [(2, 0), (1, 15), (usize::MAX, 0)] {
            let mut out = AlignedRow([0x6d; 64]);
            assert!(!shift_reduce_inner_ab_bstatic::<true>(
                &[], &[], table, 0, b_med, w, context, &mut out.0, true,
            ));
            assert_eq!(out.0, [0x6d; 64]);
        }
    });
}
