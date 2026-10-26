#[derive(Clone, Copy)]
pub(crate) struct BstaticRow {
    pub(crate) mask: u64,
    pub(crate) expected: u64,
    pub(crate) kind: u8,

    pub(crate) vary: u8,
}

pub(crate) const ROW_GENERIC: u8 = 0;
pub(crate) const ROW_STATIC: u8 = 1;
pub(crate) const ROW_ZERO: u8 = 2;

pub(crate) const BSTATIC_BLOCKS: usize = 32;

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
    [
        row(0xffffffffffffffff, 0xffffffffffffffff),
        row(0xffffffffffffffff, 0xffffffffffffffff),
        row(0xffffffffffffffff, 0xffffffffffffffff),
        row(0xffffffffffffffff, 0xffffffffffffffff),
        row(0xffffffffffffffff, 0xffffffffffffffff),
        row(0xffffffffffffffff, 0xffffffffffffffff),
        row(0xffffffffffffffff, 0xffffffffffffffff),
        row(0xffffffffffffffff, 0xffffffffffffffff),
    ],
    [
        row(0xffffffffffffffff, 0xffffffffffffffff),
        row(0xffffffffffffffff, 0xffffffffffffffff),
        row(0xffffffffffffffff, 0xffffffffffffffff),
        row(0xffffffffffffffff, 0xffffffffffffffff),
        row(0xffffffffffffffff, 0xffffffffffffffff),
        row(0xffffffffffffffff, 0xffffffffffffffff),
        row(0xffffffffffffffff, 0xffffffffffffffff),
        row(0xffffffffffffffff, 0xffffffffffffffff),
    ],
    [
        row(0xffffffffffffffff, 0xffffffffffffffff),
        row(0xffffffffffffffff, 0xffffffffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x00ffffffffffffff, 0x00ffffffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
    ],
    [
        row(0xff00000000000000, 0xff00000000000000),
        row(0x0000ffffffffffff, 0x0000ffffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffff000000000000, 0xffff000000000000),
        row(0x000000ffffffffff, 0x000000ffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
    ],
    [
        row(0xffff000000000000, 0xffff000000000000),
        row(0x000000ffffffffff, 0x000000ffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffff0000000000, 0xffffff0000000000),
        row(0x00000000ffffffff, 0x00000000ffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
    ],
    [
        row(0xffffffff00000000, 0xffffffff00000000),
        row(0x0000000000ffffff, 0x0000000000ffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffffffff000000, 0xffffffffff000000),
        row(0x000000000000ffff, 0x000000000000ffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
    ],
    [
        row(0xffffffffff000000, 0xffffffffff000000),
        row(0x000000000000ffff, 0x000000000000ffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffffffffff0000, 0xffffffffffff0000),
        row(0x00000000000000ff, 0x00000000000000ff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
    ],
    [
        row(0xffffffffffffff00, 0xffffffffffffff00),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x00ffffffffffffff, 0x00ffffffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
    ],
    [
        row(0x00ffffffffffffff, 0x00ffffffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xff00000000000000, 0xff00000000000000),
        row(0x0000ffffffffffff, 0x0000ffffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffff000000000000, 0xffff000000000000),
    ],
    [
        row(0x000000ffffffffff, 0x000000ffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffff0000000000, 0xffffff0000000000),
        row(0x00000000ffffffff, 0x00000000ffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffff0000000000, 0xffffff0000000000),
    ],
    [
        row(0x00000000ffffffff, 0x00000000ffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffffff00000000, 0xffffffff00000000),
        row(0x0000000000ffffff, 0x0000000000ffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffffffff000000, 0xffffffffff000000),
    ],
    [
        row(0x000000000000ffff, 0x000000000000ffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffffffffff0000, 0xffffffffffff0000),
        row(0x00000000000000ff, 0x00000000000000ff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffffffffff0000, 0xffffffffffff0000),
    ],
    [
        row(0x00000000000000ff, 0x00000000000000ff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffffffffffff00, 0xffffffffffffff00),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x00ffffffffffffff, 0x00ffffffffffffff),
    ],
    [
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xff00000000000000, 0xff00000000000000),
        row(0x0000ffffffffffff, 0x0000ffffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xff00000000000000, 0xff00000000000000),
        row(0x0000ffffffffffff, 0x0000ffffffffffff),
    ],
    [
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffff000000000000, 0xffff000000000000),
        row(0x000000ffffffffff, 0x000000ffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffff0000000000, 0xffffff0000000000),
        row(0x00000000ffffffff, 0x00000000ffffffff),
    ],
    [
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffffff00000000, 0xffffffff00000000),
        row(0x0000000000ffffff, 0x0000000000ffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffffff00000000, 0xffffffff00000000),
        row(0x0000000000ffffff, 0x0000000000ffffff),
    ],
    [
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffffffff000000, 0xffffffffff000000),
        row(0x000000000000ffff, 0x000000000000ffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffffffffff0000, 0xffffffffffff0000),
        row(0x00000000000000ff, 0x00000000000000ff),
    ],
    [
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffffffffffff00, 0xffffffffffffff00),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffffffffffff00, 0xffffffffffffff00),
        row(0x0000000000000000, 0x0000000000000000),
    ],
    [
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x00ffffffffffffff, 0x00ffffffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xff00000000000000, 0xff00000000000000),
        row(0x0000ffffffffffff, 0x0000ffffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
    ],
    [
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffff000000000000, 0xffff000000000000),
        row(0x000000ffffffffff, 0x000000ffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffff000000000000, 0xffff000000000000),
        row(0x000000ffffffffff, 0x000000ffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
    ],
    [
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffff0000000000, 0xffffff0000000000),
        row(0x00000000ffffffff, 0x00000000ffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffffff00000000, 0xffffffff00000000),
        row(0x0000000000ffffff, 0x0000000000ffffff),
        row(0x0000000000000000, 0x0000000000000000),
    ],
    [
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffffffff000000, 0xffffffffff000000),
        row(0x000000000000ffff, 0x000000000000ffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffffffff000000, 0xffffffffff000000),
        row(0x000000000000ffff, 0x000000000000ffff),
        row(0x0000000000000000, 0x0000000000000000),
    ],
    [
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffffffffff0000, 0xffffffffffff0000),
        row(0x00000000000000ff, 0x00000000000000ff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffffffffffff00, 0xffffffffffffff00),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
    ],
    [
        row(0x0000000000000000, 0x0000000000000000),
        row(0x00ffffffffffffff, 0x00ffffffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x00ffffffffffffff, 0x00ffffffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
    ],
    [
        row(0xff00000000000000, 0xff00000000000000),
        row(0x0000ffffffffffff, 0x0000ffffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffff000000000000, 0xffff000000000000),
        row(0x000000ffffffffff, 0x000000ffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
    ],
    [
        row(0xffffff0000000000, 0xffffff0000000000),
        row(0x00000000ffffffff, 0x00000000ffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffff0000000000, 0xffffff0000000000),
        row(0x00000000ffffffff, 0x00000000ffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
    ],
    [
        row(0xffffffff00000000, 0xffffffff00000000),
        row(0x0000000000ffffff, 0x0000000000ffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffffffff000000, 0xffffffffff000000),
        row(0x000000000000ffff, 0x000000000000ffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
    ],
    [
        row(0xffffffffffff0000, 0xffffffffffff0000),
        row(0x00000000000000ff, 0x00000000000000ff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xffffffffffff0000, 0xffffffffffff0000),
        row(0x00000000000000ff, 0x00000000000000ff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
    ],
    [
        row(0xffffffffffffff00, 0xffffffffffffff00),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x00ffffffffffffff, 0x00ffffffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xff00000000000000, 0xff00000000000000),
    ],
    [
        row(0x0000ffffffffffff, 0x0000ffffffffffff),
        row(0x0000000000000000, 0x0000000000000000),
        row(0x0000000000000000, 0x0000000000000000),
        row(0xff00000000000000, 0xff00000000000000),
        row(0xffffffffffffffff, 0xffffffffffffffff),
        row(0xffffffffffffffff, 0xffffffffffffffff),
        row(0xffffffffffffffff, 0xffffffffffffffff),
        row(0xffffffffffffffff, 0xffffffffffffffff),
    ],
    [
        row(0xffffffffffffffff, 0x0001ffffffffffff),
        row(0xffffffffffffffff, 0x0000000000000000),
        row(0xffffffffffffffff, 0x0000000000000000),
        row(0xffffffffffffffff, 0x0000000000000000),
        row(0xffffffffffffffff, 0x0000000000000000),
        row(0xffffffffffffffff, 0x0000000000000000),
        row(0xffffffffffffffff, 0x0000000000000000),
        row(0xffffffffffffffff, 0x0000000000000000),
    ],
    [
        row(0xffffffffffffffff, 0x0000000000000000),
        row(0xffffffffffffffff, 0x0000000000000000),
        row(0xffffffffffffffff, 0x0000000000000000),
        row(0xffffffffffffffff, 0x0000000000000000),
        row(0xffffffffffffffff, 0x0000000000000000),
        row(0xffffffffffffffff, 0x0000000000000000),
        row(0xffffffffffffffff, 0x0000000000000000),
        row(0xffffffffffffffff, 0x0000000000000000),
    ],
    [G, G, G, G, G, G, G, G],
];
