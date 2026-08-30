//! Source oracles for the ARM-only canonical B seed. No producer masks are
//! imported when constructing inputs: the one bits follow the circuit's
//! 1153-bit prefix, 56 independent 250-bit G records, and final output band.

use super::{
    BSTATIC_MASKS, StaticBContext, b_complement_table_is_one,
    prepare_static_b_context_with_complement_policy, shift_reduce_inner_ab_bstatic,
};
use crate::field::F8;
use crate::ntt::{AdditiveNttGf8, InvNttTableByteSingleGf8};
use std::collections::BTreeMap;
use std::sync::OnceLock;

const BLOCK_BYTES: usize = 2048;

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

fn prepared(table: &InvNttTableByteSingleGf8, requested: bool) -> StaticBContext {
    // The incumbent partial caches are process-global and standard-domain.
    // Initialize them from that domain before testing other extension
    // shifts; canonical hits below must not depend on their contents.
    let standard = InvNttTableByteSingleGf8::cached_standard_k6();
    prepare_static_b_context_with_complement_policy(standard, true, false, false, false)
        .expect("standard partial cache");
    prepare_static_b_context_with_complement_policy(table, true, false, false, requested)
        .expect("BLAKE3 static context")
}

fn poisoned_partials(table: &InvNttTableByteSingleGf8) -> StaticBContext {
    static POISON: OnceLock<[[u8; 64]; 248]> = OnceLock::new();
    let StaticBContext::Prepared {
        static_a_k1,
        b_complement,
        ..
    } = prepared(table, true)
    else {
        unreachable!("prepared context")
    };
    assert!(b_complement);
    StaticBContext::Prepared {
        partials: POISON.get_or_init(|| [[0xa5; 64]; 248]),
        static_a_k1,
        b_complement,
    }
}

fn next_byte(state: &mut u64) -> u8 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state >> 23) as u8
}

fn inputs(pattern: u8, prefix: usize) -> (Vec<u8>, Vec<u8>) {
    let mut state = 0xbc00_1eaf_5eed_7701u64 ^ u64::from(pattern);
    let mut a = vec![0xc3; prefix + BLOCK_BYTES];
    let mut b = vec![0x69; prefix + BLOCK_BYTES];
    for byte in prefix..prefix + BLOCK_BYTES {
        a[byte] = match pattern {
            0 => 0,
            1 => 0xff,
            2 => 0x55,
            _ => next_byte(&mut state),
        };
        b[byte] = match pattern {
            0 => 0,
            1 => 0xff,
            2 => 0xaa,
            _ => next_byte(&mut state),
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
    core::array::from_fn(|lane| result[lane].0)
}

#[repr(align(64))]
struct AlignedRow([u8; 64]);

#[allow(clippy::too_many_arguments)]
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
        // Test-only readback barrier. Production storage and fence policy
        // remain in the existing caller; no kernel barrier is added.
        unsafe { core::arch::asm!("dmb ish", options(nostack, preserves_flags)) };
    }
    out.0
}

fn with_large_stack(f: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(f)
        .expect("spawn complement oracle")
        .join()
        .expect("complement oracle");
}

#[test]
fn complement_geometry_and_actual_table_gate() {
    let mut representatives = BTreeMap::new();
    let mut partial = 0;
    let mut one_known = 0;
    let mut known_bytes = 0;
    for blk in 3..29 {
        for k in 0..8 {
            let mask = independent_mask(blk, k);
            assert_eq!(BSTATIC_MASKS[blk][k], (mask, mask));
            if mask != 0 {
                assert_ne!(mask, u64::MAX);
                let known = mask.count_ones() / 8;
                partial += 1;
                one_known += usize::from(known == 1);
                known_bytes += known;
                representatives.entry(mask).or_insert((blk, k));
            }
        }
    }
    assert_eq!((partial, one_known, known_bytes), (94, 14, 372));
    assert_eq!(representatives.len(), 14);
    assert_eq!(F8::ONE.0, 1);
    for beta in [64, 128, 192] {
        let table = make_table(6, beta);
        assert!(b_complement_table_is_one(&table));
        assert!(matches!(
            prepared(&table, true),
            StaticBContext::Prepared {
                b_complement: true,
                ..
            }
        ));
        assert!(matches!(
            prepared(&table, false),
            StaticBContext::Prepared {
                b_complement: false,
                ..
            }
        ));
    }
    for k in [5, 7] {
        assert!(!b_complement_table_is_one(&make_table(k, 1u8 << k)));
    }
    let table = make_table(6, 64);
    assert!(
        prepare_static_b_context_with_complement_policy(&table, false, false, false, true)
            .is_none()
    );
    assert!(
        prepare_static_b_context_with_complement_policy(&table, true, true, false, true)
            .is_none()
    );
    assert!(matches!(
        prepare_static_b_context_with_complement_policy(&table, true, false, true, true),
        Some(StaticBContext::LegacyPerCall)
    ));
}

#[test]
fn complement_all_windows_match_scalar_without_partial_values() {
    with_large_stack(|| {
        for beta in [64, 128, 192] {
            let table = make_table(6, beta);
            let context = poisoned_partials(&table);
            for prefix in [0, 64] {
                for pattern in 0..4 {
                    let (a, b) = inputs(pattern, prefix);
                    for blk in 3..29 {
                        let want = scalar(&a, &b, &table, prefix, blk);
                        for fast in [false, true] {
                            for nt in [false, true] {
                                assert_eq!(
                                    kernel(&a, &b, &table, prefix, blk, context, fast, nt),
                                    want,
                                    "beta={beta} prefix={prefix} pattern={pattern} blk={blk} fast={fast} nt={nt}"
                                );
                            }
                        }
                    }
                }
            }
        }
        // These partial rows are initialized poison, not uninitialized
        // memory. Equality proves value-independence, not absence of loads.
    });
}

#[test]
fn complement_fourteen_masks_cover_varying_basis_vectors() {
    with_large_stack(|| {
        let mut representatives = BTreeMap::new();
        for blk in 3..29 {
            for k in 0..8 {
                let mask = independent_mask(blk, k);
                if mask != 0 {
                    representatives.entry(mask).or_insert((blk, k));
                }
            }
        }
        assert_eq!(representatives.len(), 14);
        for beta in [64, 128, 192] {
            let table = make_table(6, beta);
            let context = poisoned_partials(&table);
            for (&mask, &(blk, k)) in &representatives {
                let mut a = vec![0; BLOCK_BYTES];
                let mut b = vec![0xff; BLOCK_BYTES];
                let off = blk * 64 + k * 8;
                a[off..off + 8].fill(0xff);
                // Zero complement plus every varying input basis vector.
                let words = core::iter::once(u64::MAX).chain(
                    (0..64)
                        .filter(|&bit| mask & (1u64 << bit) == 0)
                        .map(|bit| u64::MAX ^ (1u64 << bit)),
                );
                for word in words {
                    b[off..off + 8].copy_from_slice(&word.to_le_bytes());
                    let want = scalar(&a, &b, &table, 0, blk);
                    for fast in [false, true] {
                        assert_eq!(
                            kernel(&a, &b, &table, 0, blk, context, fast, false),
                            want,
                            "beta={beta} mask={mask:#018x} word={word:#018x} fast={fast}"
                        );
                    }
                }
            }
        }
    });
}

#[test]
fn complement_each_guard_byte_falls_back_to_original_row() {
    with_large_stack(|| {
        for beta in [64, 128, 192] {
            let table = make_table(6, beta);
            let context = poisoned_partials(&table);
            let (a, mut b) = inputs(3, 0);
            let mut misses = 0;
            for blk in 3..29 {
                for k in 0..8 {
                    let mask = independent_mask(blk, k);
                    for byte in 0..8 {
                        if (mask >> (8 * byte)) & 0xff != 0xff {
                            continue;
                        }
                        let off = blk * 64 + k * 8 + byte;
                        assert_eq!(b[off], 0xff);
                        b[off] ^= 1;
                        let want = scalar(&a, &b, &table, 0, blk);
                        for fast in [false, true] {
                            assert_eq!(
                                kernel(&a, &b, &table, 0, blk, context, fast, byte % 2 == 0),
                                want,
                                "beta={beta} blk={blk} k={k} byte={byte} fast={fast}"
                            );
                        }
                        b[off] ^= 1;
                        misses += 1;
                    }
                }
            }
            assert_eq!(misses, 372);
        }
    });
}

#[test]
fn complement_disabled_and_boundary_windows_retain_original_semantics() {
    with_large_stack(|| {
        let table = make_table(6, 64);
        let on = prepared(&table, true);
        let off = prepared(&table, false);
        let (a, b) = inputs(3, 64);
        for blk in 0..31 {
            let want = scalar(&a, &b, &table, 64, blk);
            for context in [on, off, StaticBContext::LegacyPerCall] {
                for fast in [false, true] {
                    assert_eq!(
                        kernel(&a, &b, &table, 64, blk, context, fast, true),
                        want,
                        "standard domain blk={blk} fast={fast}"
                    );
                }
            }
        }
        for (w, b_med) in [(2, 0), (1, 15), (usize::MAX, 0)] {
            let mut out = AlignedRow([0x6d; 64]);
            assert!(!shift_reduce_inner_ab_bstatic::<true>(
                &[], &[], &table, 0, b_med, w, on, &mut out.0, true,
            ));
            assert_eq!(out.0, [0x6d; 64]);
        }
    });
}
