//! GENERATED static-B plan for the x86 AVX-512/GFNI round-1 shift-reduce
//! kernel — the same census data as the mac track's `aarch64_bstatic_gen.rs`
//! (blocks 0..=30 are byte-identical to that file's `BSTATIC_MASKS`), plus
//! block 31 (window 1, `b_med = 15`: the all-zero padding window that the
//! streaming witness seam still computes) and one all-generic control plan
//! (index 32) for same-binary A/B.
//!
//! Produced by `flock_prover::r1cs_hashes::blake3::tests::bstatic_census`
//! (2^12 ranked-distribution BLAKE3 blocks: `block_len = 64`, `flags = 11`,
//! u32 counter — the harness generator's exact shape) at the ranked layout
//! `k_log = 14`, `useful_bits = 15409`. The `(mask, expected)` pairs describe
//! bytes of the round-1 **b operand** that the BLAKE3 circuit fixes
//! independently of the block inputs (const-one wires and structural zeros).
//!
//! They are ONLY a performance hint: every planned K-row checks
//! `(b_word & mask) == expected` and falls back to the generic row when the
//! witness disagrees, so the kernel is bit-exact for any witness and any
//! layout. `kind` / `vary` are derived from `mask` at generation time:
//!   * `ROW_ZERO`    — mask covers all 64 bits and expected == 0 ⇒ the b row is
//!                     structurally zero ⇒ the whole K-row contributes nothing.
//!   * `ROW_STATIC`  — ≤ 6 varying bytes: `T(b) = T(expected) ⊕ Σ_{j∈vary} π_j(T₀[b_j])`.
//!                     `vary` is the bitmask of varying byte positions.
//!   * `ROW_GENERIC` — no useful static structure (or > 6 varying bytes).

/// One planned K-row.
#[derive(Clone, Copy)]
pub(crate) struct BstaticRow {
    pub(crate) mask: u64,
    pub(crate) expected: u64,
    pub(crate) kind: u8,
    /// Bit `j` set ⇒ byte `j` of the b word varies (only meaningful for `ROW_STATIC`).
    pub(crate) vary: u8,
}

pub(crate) const ROW_GENERIC: u8 = 0;
pub(crate) const ROW_STATIC: u8 = 1;
pub(crate) const ROW_ZERO: u8 = 2;

/// Number of BLAKE3 `(w, b_med)` blocks with a plan (`blk = w * 16 + b_med`).
pub(crate) const BSTATIC_BLOCKS: usize = 32;
/// Index of the all-generic control plan (tests / microbenchmarks only).
#[cfg(test)]
pub(crate) const BSTATIC_GENERIC_PLAN: usize = 32;

const fn row(mask: u64, expected: u64) -> BstaticRow {
    if mask == 0 {
        return BstaticRow {
            mask,
            expected,
            kind: ROW_GENERIC,
            vary: 0,
        };
    }
    if mask == u64::MAX && expected == 0 {
        return BstaticRow {
            mask,
            expected,
            kind: ROW_ZERO,
            vary: 0,
        };
    }
    let mut vary = 0u8;
    let mut j = 0;
    while j < 8 {
        if (mask >> (8 * j)) & 0xff == 0 {
            vary |= 1 << j;
        }
        j += 1;
    }
    if vary.count_ones() <= 6 {
        BstaticRow {
            mask,
            expected,
            kind: ROW_STATIC,
            vary,
        }
    } else {
        BstaticRow {
            mask,
            expected,
            kind: ROW_GENERIC,
            vary: 0,
        }
    }
}

const G: BstaticRow = BstaticRow {
    mask: 0,
    expected: 0,
    kind: ROW_GENERIC,
    vary: 0,
};

#[rustfmt::skip]
pub(crate) const BSTATIC_PLAN: [[BstaticRow; 8]; 33] = [
    [ // blk 0: w = 0, b_med = 0
        row(0xffffffffffffffff, 0xffffffffffffffff),
        row(0xffffffffffffffff, 0xffffffffffffffff),
        row(0xffffffffffffffff, 0xffffffffffffffff),
        row(0xffffffffffffffff, 0xffffffffffffffff),
        row(0xffffffffffffffff, 0xffffffffffffffff),
        row(0xffffffffffffffff, 0xffffffffffffffff),
        row(0xffffffffffffffff, 0xffffffffffffffff),
        row(0xffffffffffffffff, 0xffffffffffffffff),
    ],
    [ // blk 1: w = 0, b_med = 1
        row(0xffffffffffffffff, 0xffffffffffffffff),
        row(0xffffffffffffffff, 0xffffffffffffffff),
        row(0xffffffffffffffff, 0xffffffffffffffff),
        row(0xffffffffffffffff, 0xffffffffffffffff),
        row(0xffffffffffffffff, 0xffffffffffffffff),
        row(0xffffffffffffffff, 0xffffffffffffffff),
        row(0xffffffffffffffff, 0xffffffffffffffff),
        row(0xffffffffffffffff, 0xffffffffffffffff),
    ],
    [ // blk 2: w = 0, b_med = 2
        row(0xffffffffffffffff, 0xffffffffffffffff),
        row(0xffffffffffffffff, 0xffffffffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x00ffffffffffffff, 0x00ffffffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
    ],
    [ // blk 3: w = 0, b_med = 3
        row(0xff00000000000000, 0xff00000000000000),
        row(0x0000ffffffffffff, 0x0000ffffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffff000000000000, 0xffff000000000000),
        row(0x000000ffffffffff, 0x000000ffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
    ],
    [ // blk 4: w = 0, b_med = 4
        row(0xffff000000000000, 0xffff000000000000),
        row(0x000000ffffffffff, 0x000000ffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffff0000000000, 0xffffff0000000000),
        row(0x00000000ffffffff, 0x00000000ffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
    ],
    [ // blk 5: w = 0, b_med = 5
        row(0xffffffff00000000, 0xffffffff00000000),
        row(0x0000000000ffffff, 0x0000000000ffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffffffff000000, 0xffffffffff000000),
        row(0x000000000000ffff, 0x000000000000ffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
    ],
    [ // blk 6: w = 0, b_med = 6
        row(0xffffffffff000000, 0xffffffffff000000),
        row(0x000000000000ffff, 0x000000000000ffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffffffffff0000, 0xffffffffffff0000),
        row(0x00000000000000ff, 0x00000000000000ff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
    ],
    [ // blk 7: w = 0, b_med = 7
        row(0xffffffffffffff00, 0xffffffffffffff00),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x00ffffffffffffff, 0x00ffffffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
    ],
    [ // blk 8: w = 0, b_med = 8
        row(0x00ffffffffffffff, 0x00ffffffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xff00000000000000, 0xff00000000000000),
        row(0x0000ffffffffffff, 0x0000ffffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffff000000000000, 0xffff000000000000),
    ],
    [ // blk 9: w = 0, b_med = 9
        row(0x000000ffffffffff, 0x000000ffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffff0000000000, 0xffffff0000000000),
        row(0x00000000ffffffff, 0x00000000ffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffff0000000000, 0xffffff0000000000),
    ],
    [ // blk 10: w = 0, b_med = 10
        row(0x00000000ffffffff, 0x00000000ffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffffff00000000, 0xffffffff00000000),
        row(0x0000000000ffffff, 0x0000000000ffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffffffff000000, 0xffffffffff000000),
    ],
    [ // blk 11: w = 0, b_med = 11
        row(0x000000000000ffff, 0x000000000000ffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffffffffff0000, 0xffffffffffff0000),
        row(0x00000000000000ff, 0x00000000000000ff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffffffffff0000, 0xffffffffffff0000),
    ],
    [ // blk 12: w = 0, b_med = 12
        row(0x00000000000000ff, 0x00000000000000ff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffffffffffff00, 0xffffffffffffff00),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x00ffffffffffffff, 0x00ffffffffffffff),
    ],
    [ // blk 13: w = 0, b_med = 13
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xff00000000000000, 0xff00000000000000),
        row(0x0000ffffffffffff, 0x0000ffffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xff00000000000000, 0xff00000000000000),
        row(0x0000ffffffffffff, 0x0000ffffffffffff),
    ],
    [ // blk 14: w = 0, b_med = 14
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffff000000000000, 0xffff000000000000),
        row(0x000000ffffffffff, 0x000000ffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffff0000000000, 0xffffff0000000000),
        row(0x00000000ffffffff, 0x00000000ffffffff),
    ],
    [ // blk 15: w = 0, b_med = 15
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffffff00000000, 0xffffffff00000000),
        row(0x0000000000ffffff, 0x0000000000ffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffffff00000000, 0xffffffff00000000),
        row(0x0000000000ffffff, 0x0000000000ffffff),
    ],
    [ // blk 16: w = 1, b_med = 0
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffffffff000000, 0xffffffffff000000),
        row(0x000000000000ffff, 0x000000000000ffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffffffffff0000, 0xffffffffffff0000),
        row(0x00000000000000ff, 0x00000000000000ff),
    ],
    [ // blk 17: w = 1, b_med = 1
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffffffffffff00, 0xffffffffffffff00),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffffffffffff00, 0xffffffffffffff00),
        row(0x0000000000000000, 0x0000000000000000),
    ],
    [ // blk 18: w = 1, b_med = 2
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x00ffffffffffffff, 0x00ffffffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xff00000000000000, 0xff00000000000000),
        row(0x0000ffffffffffff, 0x0000ffffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
    ],
    [ // blk 19: w = 1, b_med = 3
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffff000000000000, 0xffff000000000000),
        row(0x000000ffffffffff, 0x000000ffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffff000000000000, 0xffff000000000000),
        row(0x000000ffffffffff, 0x000000ffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
    ],
    [ // blk 20: w = 1, b_med = 4
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffff0000000000, 0xffffff0000000000),
        row(0x00000000ffffffff, 0x00000000ffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffffff00000000, 0xffffffff00000000),
        row(0x0000000000ffffff, 0x0000000000ffffff),
        row(0x0000000000000000, 0x0000000000000000),
    ],
    [ // blk 21: w = 1, b_med = 5
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffffffff000000, 0xffffffffff000000),
        row(0x000000000000ffff, 0x000000000000ffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffffffff000000, 0xffffffffff000000),
        row(0x000000000000ffff, 0x000000000000ffff),
        row(0x0000000000000000, 0x0000000000000000),
    ],
    [ // blk 22: w = 1, b_med = 6
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffffffffff0000, 0xffffffffffff0000),
        row(0x00000000000000ff, 0x00000000000000ff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffffffffffff00, 0xffffffffffffff00),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
    ],
    [ // blk 23: w = 1, b_med = 7
        row(0x0000000000000000, 0x0000000000000000),
        row(0x00ffffffffffffff, 0x00ffffffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x00ffffffffffffff, 0x00ffffffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
    ],
    [ // blk 24: w = 1, b_med = 8
        row(0xff00000000000000, 0xff00000000000000),
        row(0x0000ffffffffffff, 0x0000ffffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffff000000000000, 0xffff000000000000),
        row(0x000000ffffffffff, 0x000000ffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
    ],
    [ // blk 25: w = 1, b_med = 9
        row(0xffffff0000000000, 0xffffff0000000000),
        row(0x00000000ffffffff, 0x00000000ffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffff0000000000, 0xffffff0000000000),
        row(0x00000000ffffffff, 0x00000000ffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
    ],
    [ // blk 26: w = 1, b_med = 10
        row(0xffffffff00000000, 0xffffffff00000000),
        row(0x0000000000ffffff, 0x0000000000ffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffffffff000000, 0xffffffffff000000),
        row(0x000000000000ffff, 0x000000000000ffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
    ],
    [ // blk 27: w = 1, b_med = 11
        row(0xffffffffffff0000, 0xffffffffffff0000),
        row(0x00000000000000ff, 0x00000000000000ff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffffffffff0000, 0xffffffffffff0000),
        row(0x00000000000000ff, 0x00000000000000ff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
    ],
    [ // blk 28: w = 1, b_med = 12
        row(0xffffffffffffff00, 0xffffffffffffff00),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x00ffffffffffffff, 0x00ffffffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xff00000000000000, 0xff00000000000000),
    ],
    [ // blk 29: w = 1, b_med = 13
        row(0x0000ffffffffffff, 0x0000ffffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xff00000000000000, 0xff00000000000000),
        row(0xffffffffffffffff, 0xffffffffffffffff),
        row(0xffffffffffffffff, 0xffffffffffffffff),
        row(0xffffffffffffffff, 0xffffffffffffffff),
        row(0xffffffffffffffff, 0xffffffffffffffff),
    ],
    [ // blk 30: w = 1, b_med = 14
        row(0xffffffffffffffff, 0x0001ffffffffffff),
        row(0xffffffffffffffff, 0x0000000000000000),
        row(0xffffffffffffffff, 0x0000000000000000),
        row(0xffffffffffffffff, 0x0000000000000000),
        row(0xffffffffffffffff, 0x0000000000000000),
        row(0xffffffffffffffff, 0x0000000000000000),
        row(0xffffffffffffffff, 0x0000000000000000),
        row(0xffffffffffffffff, 0x0000000000000000),
    ],
    [ // blk 31: w = 1, b_med = 15
        row(0xffffffffffffffff, 0x0000000000000000),
        row(0xffffffffffffffff, 0x0000000000000000),
        row(0xffffffffffffffff, 0x0000000000000000),
        row(0xffffffffffffffff, 0x0000000000000000),
        row(0xffffffffffffffff, 0x0000000000000000),
        row(0xffffffffffffffff, 0x0000000000000000),
        row(0xffffffffffffffff, 0x0000000000000000),
        row(0xffffffffffffffff, 0x0000000000000000),
    ],
    [G, G, G, G, G, G, G, G], // all-generic control plan
];
