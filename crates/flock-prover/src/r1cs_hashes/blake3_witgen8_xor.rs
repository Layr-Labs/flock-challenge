use super::*;
use flock_core::zerocheck::univariate_skip_optimized::round1_ranked_bcomplement_mode as mode;
#[repr(align(64))]
pub(super) struct Tables([u64; 8 * 128]);
impl Tables {
    pub(super) fn new(table: &InvNttTableByteSingleGf8) -> Self {
        assert_eq!(table.ell, 64);

        let mut m = [0u64; 8 * 128];
        for k in 0..8 {
            let bank = &mut m[k * 128..(k + 1) * 128];
            for i in 0..64 {
                for outbit in 0..8 {
                    let mut row = 0u8;
                    for bit in 0..8 {
                        let c = unsafe { *table.data_ptr().add((1 << bit) * 64 + i) };
                        let scaled = flock_core::field::F8(c) * flock_core::field::F8(1u8 << k);
                        row |= ((scaled.0 >> outbit) & 1) << bit;
                    }
                    bank[i] |= (row as u64) << (8 * (7 - outbit));
                }
            }
            for subset in 0..8 {
                for c in 0..8 {
                    let mut v = 0;
                    for t in 0..8 {
                        if t & subset == subset {
                            v ^= bank[t * 8 + c];
                        }
                    }
                    bank[64 + subset * 8 + c] = v;
                }
            }
        }
        Self(m)
    }
}

#[inline(always)]
const fn known(mode: u8) -> u8 {
    if mode == 15 {
        255
    } else if mode == 0 {
        0
    } else if mode <= 7 {
        (1u16 << mode) as u8 - 1
    } else {
        (255u16 << (15 - mode)) as u8
    }
}

#[inline(always)]
unsafe fn affine(p: *const u64, m: *const u64, mode_value: u8) -> [__m512i; 8] {
    unsafe {
        let z = _mm512_setzero_si512();
        let mut v = [z; 8];
        macro_rules! add {
            ($j:expr) => {
                if known(mode_value) & (1 << $j) == 0 {
                    let w = *p.add($j);
                    let w = if mode_value == 0 { w } else { !w };
                    let x = _mm512_set1_epi64(w as i64);
                    v[0] = _mm512_xor_si512(
                        v[0],
                        _mm512_gf2p8affine_epi64_epi8::<0>(
                            x,
                            _mm512_loadu_si512(m.add(($j ^ 0) * 8).cast()),
                        ),
                    );
                    v[1] = _mm512_xor_si512(
                        v[1],
                        _mm512_gf2p8affine_epi64_epi8::<0>(
                            x,
                            _mm512_loadu_si512(m.add(($j ^ 1) * 8).cast()),
                        ),
                    );
                    v[2] = _mm512_xor_si512(
                        v[2],
                        _mm512_gf2p8affine_epi64_epi8::<0>(
                            x,
                            _mm512_loadu_si512(m.add(($j ^ 2) * 8).cast()),
                        ),
                    );
                    v[3] = _mm512_xor_si512(
                        v[3],
                        _mm512_gf2p8affine_epi64_epi8::<0>(
                            x,
                            _mm512_loadu_si512(m.add(($j ^ 3) * 8).cast()),
                        ),
                    );
                    v[4] = _mm512_xor_si512(
                        v[4],
                        _mm512_gf2p8affine_epi64_epi8::<0>(
                            x,
                            _mm512_loadu_si512(m.add(($j ^ 4) * 8).cast()),
                        ),
                    );
                    v[5] = _mm512_xor_si512(
                        v[5],
                        _mm512_gf2p8affine_epi64_epi8::<0>(
                            x,
                            _mm512_loadu_si512(m.add(($j ^ 5) * 8).cast()),
                        ),
                    );
                    v[6] = _mm512_xor_si512(
                        v[6],
                        _mm512_gf2p8affine_epi64_epi8::<0>(
                            x,
                            _mm512_loadu_si512(m.add(($j ^ 6) * 8).cast()),
                        ),
                    );
                    v[7] = _mm512_xor_si512(
                        v[7],
                        _mm512_gf2p8affine_epi64_epi8::<0>(
                            x,
                            _mm512_loadu_si512(m.add(($j ^ 7) * 8).cast()),
                        ),
                    );
                }
            };
        }
        add!(0);
        add!(1);
        add!(2);
        add!(3);
        add!(4);
        add!(5);
        add!(6);
        add!(7);
        v
    }
}

#[inline(always)]
unsafe fn affine_conv(p: *const u64, m: *const u64, mode_value: u8) -> [__m512i; 8] {
    unsafe {
        if known(mode_value).count_ones() >= 5 {
            return affine(p, m, mode_value);
        }
        let mut x: [u64; 8] = [
            if known(mode_value) & (1 << 0) != 0 {
                0
            } else if mode_value == 0 {
                *p.add(0)
            } else {
                !*p.add(0)
            },
            if known(mode_value) & (1 << 1) != 0 {
                0
            } else if mode_value == 0 {
                *p.add(1)
            } else {
                !*p.add(1)
            },
            if known(mode_value) & (1 << 2) != 0 {
                0
            } else if mode_value == 0 {
                *p.add(2)
            } else {
                !*p.add(2)
            },
            if known(mode_value) & (1 << 3) != 0 {
                0
            } else if mode_value == 0 {
                *p.add(3)
            } else {
                !*p.add(3)
            },
            if known(mode_value) & (1 << 4) != 0 {
                0
            } else if mode_value == 0 {
                *p.add(4)
            } else {
                !*p.add(4)
            },
            if known(mode_value) & (1 << 5) != 0 {
                0
            } else if mode_value == 0 {
                *p.add(5)
            } else {
                !*p.add(5)
            },
            if known(mode_value) & (1 << 6) != 0 {
                0
            } else if mode_value == 0 {
                *p.add(6)
            } else {
                !*p.add(6)
            },
            if known(mode_value) & (1 << 7) != 0 {
                0
            } else if mode_value == 0 {
                *p.add(7)
            } else {
                !*p.add(7)
            },
        ];
        x[0] ^= x[1];
        x[2] ^= x[3];
        x[4] ^= x[5];
        x[6] ^= x[7];
        x[0] ^= x[2];
        x[1] ^= x[3];
        x[4] ^= x[6];
        x[5] ^= x[7];
        x[0] ^= x[4];
        x[1] ^= x[5];
        x[2] ^= x[6];
        x[3] ^= x[7];
        let mut c = [_mm512_setzero_si512(); 8];
        c[0] = _mm512_gf2p8affine_epi64_epi8::<0>(
            _mm512_set1_epi64(x[0] as i64),
            _mm512_loadu_si512(m.add(64).cast()),
        );
        c[1] = _mm512_xor_si512(
            _mm512_gf2p8affine_epi64_epi8::<0>(
                _mm512_set1_epi64(x[0] as i64),
                _mm512_loadu_si512(m.add(72).cast()),
            ),
            _mm512_gf2p8affine_epi64_epi8::<0>(
                _mm512_set1_epi64(x[1] as i64),
                _mm512_loadu_si512(m.add(64).cast()),
            ),
        );
        c[2] = _mm512_xor_si512(
            _mm512_gf2p8affine_epi64_epi8::<0>(
                _mm512_set1_epi64(x[0] as i64),
                _mm512_loadu_si512(m.add(80).cast()),
            ),
            _mm512_gf2p8affine_epi64_epi8::<0>(
                _mm512_set1_epi64(x[2] as i64),
                _mm512_loadu_si512(m.add(64).cast()),
            ),
        );
        c[3] = _mm512_xor_si512(
            _mm512_ternarylogic_epi64::<0x96>(
                _mm512_gf2p8affine_epi64_epi8::<0>(
                    _mm512_set1_epi64(x[0] as i64),
                    _mm512_loadu_si512(m.add(88).cast()),
                ),
                _mm512_gf2p8affine_epi64_epi8::<0>(
                    _mm512_set1_epi64(x[1] as i64),
                    _mm512_loadu_si512(m.add(80).cast()),
                ),
                _mm512_gf2p8affine_epi64_epi8::<0>(
                    _mm512_set1_epi64(x[2] as i64),
                    _mm512_loadu_si512(m.add(72).cast()),
                ),
            ),
            _mm512_gf2p8affine_epi64_epi8::<0>(
                _mm512_set1_epi64(x[3] as i64),
                _mm512_loadu_si512(m.add(64).cast()),
            ),
        );
        c[4] = _mm512_xor_si512(
            _mm512_gf2p8affine_epi64_epi8::<0>(
                _mm512_set1_epi64(x[0] as i64),
                _mm512_loadu_si512(m.add(96).cast()),
            ),
            _mm512_gf2p8affine_epi64_epi8::<0>(
                _mm512_set1_epi64(x[4] as i64),
                _mm512_loadu_si512(m.add(64).cast()),
            ),
        );
        c[5] = _mm512_xor_si512(
            _mm512_ternarylogic_epi64::<0x96>(
                _mm512_gf2p8affine_epi64_epi8::<0>(
                    _mm512_set1_epi64(x[0] as i64),
                    _mm512_loadu_si512(m.add(104).cast()),
                ),
                _mm512_gf2p8affine_epi64_epi8::<0>(
                    _mm512_set1_epi64(x[1] as i64),
                    _mm512_loadu_si512(m.add(96).cast()),
                ),
                _mm512_gf2p8affine_epi64_epi8::<0>(
                    _mm512_set1_epi64(x[4] as i64),
                    _mm512_loadu_si512(m.add(72).cast()),
                ),
            ),
            _mm512_gf2p8affine_epi64_epi8::<0>(
                _mm512_set1_epi64(x[5] as i64),
                _mm512_loadu_si512(m.add(64).cast()),
            ),
        );
        c[6] = _mm512_xor_si512(
            _mm512_ternarylogic_epi64::<0x96>(
                _mm512_gf2p8affine_epi64_epi8::<0>(
                    _mm512_set1_epi64(x[0] as i64),
                    _mm512_loadu_si512(m.add(112).cast()),
                ),
                _mm512_gf2p8affine_epi64_epi8::<0>(
                    _mm512_set1_epi64(x[2] as i64),
                    _mm512_loadu_si512(m.add(96).cast()),
                ),
                _mm512_gf2p8affine_epi64_epi8::<0>(
                    _mm512_set1_epi64(x[4] as i64),
                    _mm512_loadu_si512(m.add(80).cast()),
                ),
            ),
            _mm512_gf2p8affine_epi64_epi8::<0>(
                _mm512_set1_epi64(x[6] as i64),
                _mm512_loadu_si512(m.add(64).cast()),
            ),
        );
        c[7] = _mm512_xor_si512(
            _mm512_ternarylogic_epi64::<0x96>(
                _mm512_ternarylogic_epi64::<0x96>(
                    _mm512_ternarylogic_epi64::<0x96>(
                        _mm512_gf2p8affine_epi64_epi8::<0>(
                            _mm512_set1_epi64(x[0] as i64),
                            _mm512_loadu_si512(m.add(120).cast()),
                        ),
                        _mm512_gf2p8affine_epi64_epi8::<0>(
                            _mm512_set1_epi64(x[1] as i64),
                            _mm512_loadu_si512(m.add(112).cast()),
                        ),
                        _mm512_gf2p8affine_epi64_epi8::<0>(
                            _mm512_set1_epi64(x[2] as i64),
                            _mm512_loadu_si512(m.add(104).cast()),
                        ),
                    ),
                    _mm512_gf2p8affine_epi64_epi8::<0>(
                        _mm512_set1_epi64(x[3] as i64),
                        _mm512_loadu_si512(m.add(96).cast()),
                    ),
                    _mm512_gf2p8affine_epi64_epi8::<0>(
                        _mm512_set1_epi64(x[4] as i64),
                        _mm512_loadu_si512(m.add(88).cast()),
                    ),
                ),
                _mm512_gf2p8affine_epi64_epi8::<0>(
                    _mm512_set1_epi64(x[5] as i64),
                    _mm512_loadu_si512(m.add(80).cast()),
                ),
                _mm512_gf2p8affine_epi64_epi8::<0>(
                    _mm512_set1_epi64(x[6] as i64),
                    _mm512_loadu_si512(m.add(72).cast()),
                ),
            ),
            _mm512_gf2p8affine_epi64_epi8::<0>(
                _mm512_set1_epi64(x[7] as i64),
                _mm512_loadu_si512(m.add(64).cast()),
            ),
        );
        c[0] = _mm512_xor_si512(c[0], c[1]);
        c[2] = _mm512_xor_si512(c[2], c[3]);
        c[4] = _mm512_xor_si512(c[4], c[5]);
        c[6] = _mm512_xor_si512(c[6], c[7]);
        c[0] = _mm512_xor_si512(c[0], c[2]);
        c[1] = _mm512_xor_si512(c[1], c[3]);
        c[4] = _mm512_xor_si512(c[4], c[6]);
        c[5] = _mm512_xor_si512(c[5], c[7]);
        c[0] = _mm512_xor_si512(c[0], c[4]);
        c[1] = _mm512_xor_si512(c[1], c[5]);
        c[2] = _mm512_xor_si512(c[2], c[6]);
        c[3] = _mm512_xor_si512(c[3], c[7]);
        c
    }
}

#[inline(always)]
unsafe fn affine_prepared(p: *const u64, m: *const u64) -> [__m512i; 8] {
    unsafe {
        let x = core::ptr::read_unaligned(p.cast::<[u64; 8]>());
        let mut c = [_mm512_setzero_si512(); 8];
        c[0] = _mm512_gf2p8affine_epi64_epi8::<0>(
            _mm512_set1_epi64(x[0] as i64),
            _mm512_loadu_si512(m.add(64).cast()),
        );
        c[1] = _mm512_xor_si512(
            _mm512_gf2p8affine_epi64_epi8::<0>(
                _mm512_set1_epi64(x[0] as i64),
                _mm512_loadu_si512(m.add(72).cast()),
            ),
            _mm512_gf2p8affine_epi64_epi8::<0>(
                _mm512_set1_epi64(x[1] as i64),
                _mm512_loadu_si512(m.add(64).cast()),
            ),
        );
        c[2] = _mm512_xor_si512(
            _mm512_gf2p8affine_epi64_epi8::<0>(
                _mm512_set1_epi64(x[0] as i64),
                _mm512_loadu_si512(m.add(80).cast()),
            ),
            _mm512_gf2p8affine_epi64_epi8::<0>(
                _mm512_set1_epi64(x[2] as i64),
                _mm512_loadu_si512(m.add(64).cast()),
            ),
        );
        c[3] = _mm512_xor_si512(
            _mm512_ternarylogic_epi64::<0x96>(
                _mm512_gf2p8affine_epi64_epi8::<0>(
                    _mm512_set1_epi64(x[0] as i64),
                    _mm512_loadu_si512(m.add(88).cast()),
                ),
                _mm512_gf2p8affine_epi64_epi8::<0>(
                    _mm512_set1_epi64(x[1] as i64),
                    _mm512_loadu_si512(m.add(80).cast()),
                ),
                _mm512_gf2p8affine_epi64_epi8::<0>(
                    _mm512_set1_epi64(x[2] as i64),
                    _mm512_loadu_si512(m.add(72).cast()),
                ),
            ),
            _mm512_gf2p8affine_epi64_epi8::<0>(
                _mm512_set1_epi64(x[3] as i64),
                _mm512_loadu_si512(m.add(64).cast()),
            ),
        );
        c[4] = _mm512_xor_si512(
            _mm512_gf2p8affine_epi64_epi8::<0>(
                _mm512_set1_epi64(x[0] as i64),
                _mm512_loadu_si512(m.add(96).cast()),
            ),
            _mm512_gf2p8affine_epi64_epi8::<0>(
                _mm512_set1_epi64(x[4] as i64),
                _mm512_loadu_si512(m.add(64).cast()),
            ),
        );
        c[5] = _mm512_xor_si512(
            _mm512_ternarylogic_epi64::<0x96>(
                _mm512_gf2p8affine_epi64_epi8::<0>(
                    _mm512_set1_epi64(x[0] as i64),
                    _mm512_loadu_si512(m.add(104).cast()),
                ),
                _mm512_gf2p8affine_epi64_epi8::<0>(
                    _mm512_set1_epi64(x[1] as i64),
                    _mm512_loadu_si512(m.add(96).cast()),
                ),
                _mm512_gf2p8affine_epi64_epi8::<0>(
                    _mm512_set1_epi64(x[4] as i64),
                    _mm512_loadu_si512(m.add(72).cast()),
                ),
            ),
            _mm512_gf2p8affine_epi64_epi8::<0>(
                _mm512_set1_epi64(x[5] as i64),
                _mm512_loadu_si512(m.add(64).cast()),
            ),
        );
        c[6] = _mm512_xor_si512(
            _mm512_ternarylogic_epi64::<0x96>(
                _mm512_gf2p8affine_epi64_epi8::<0>(
                    _mm512_set1_epi64(x[0] as i64),
                    _mm512_loadu_si512(m.add(112).cast()),
                ),
                _mm512_gf2p8affine_epi64_epi8::<0>(
                    _mm512_set1_epi64(x[2] as i64),
                    _mm512_loadu_si512(m.add(96).cast()),
                ),
                _mm512_gf2p8affine_epi64_epi8::<0>(
                    _mm512_set1_epi64(x[4] as i64),
                    _mm512_loadu_si512(m.add(80).cast()),
                ),
            ),
            _mm512_gf2p8affine_epi64_epi8::<0>(
                _mm512_set1_epi64(x[6] as i64),
                _mm512_loadu_si512(m.add(64).cast()),
            ),
        );
        c[7] = _mm512_xor_si512(
            _mm512_ternarylogic_epi64::<0x96>(
                _mm512_ternarylogic_epi64::<0x96>(
                    _mm512_ternarylogic_epi64::<0x96>(
                        _mm512_gf2p8affine_epi64_epi8::<0>(
                            _mm512_set1_epi64(x[0] as i64),
                            _mm512_loadu_si512(m.add(120).cast()),
                        ),
                        _mm512_gf2p8affine_epi64_epi8::<0>(
                            _mm512_set1_epi64(x[1] as i64),
                            _mm512_loadu_si512(m.add(112).cast()),
                        ),
                        _mm512_gf2p8affine_epi64_epi8::<0>(
                            _mm512_set1_epi64(x[2] as i64),
                            _mm512_loadu_si512(m.add(104).cast()),
                        ),
                    ),
                    _mm512_gf2p8affine_epi64_epi8::<0>(
                        _mm512_set1_epi64(x[3] as i64),
                        _mm512_loadu_si512(m.add(96).cast()),
                    ),
                    _mm512_gf2p8affine_epi64_epi8::<0>(
                        _mm512_set1_epi64(x[4] as i64),
                        _mm512_loadu_si512(m.add(88).cast()),
                    ),
                ),
                _mm512_gf2p8affine_epi64_epi8::<0>(
                    _mm512_set1_epi64(x[5] as i64),
                    _mm512_loadu_si512(m.add(80).cast()),
                ),
                _mm512_gf2p8affine_epi64_epi8::<0>(
                    _mm512_set1_epi64(x[6] as i64),
                    _mm512_loadu_si512(m.add(72).cast()),
                ),
            ),
            _mm512_gf2p8affine_epi64_epi8::<0>(
                _mm512_set1_epi64(x[7] as i64),
                _mm512_loadu_si512(m.add(64).cast()),
            ),
        );
        c[0] = _mm512_xor_si512(c[0], c[1]);
        c[2] = _mm512_xor_si512(c[2], c[3]);
        c[4] = _mm512_xor_si512(c[4], c[5]);
        c[6] = _mm512_xor_si512(c[6], c[7]);
        c[0] = _mm512_xor_si512(c[0], c[2]);
        c[1] = _mm512_xor_si512(c[1], c[3]);
        c[4] = _mm512_xor_si512(c[4], c[6]);
        c[5] = _mm512_xor_si512(c[5], c[7]);
        c[0] = _mm512_xor_si512(c[0], c[4]);
        c[1] = _mm512_xor_si512(c[1], c[5]);
        c[2] = _mm512_xor_si512(c[2], c[6]);
        c[3] = _mm512_xor_si512(c[3], c[7]);
        c
    }
}

#[inline(always)]
unsafe fn trq(x: [__m512i; 8]) -> [__m512i; 8] {
    unsafe {
        let lo1 = _mm512_set_epi64(11, 3, 10, 2, 9, 1, 8, 0);
        let hi1 = _mm512_set_epi64(15, 7, 14, 6, 13, 5, 12, 4);
        let lo2 = _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0);
        let hi2 = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4);
        let lo4 = _mm512_set_epi64(11, 10, 9, 8, 3, 2, 1, 0);
        let hi4 = _mm512_set_epi64(15, 14, 13, 12, 7, 6, 5, 4);
        let t0 = _mm512_permutex2var_epi64(x[0], lo1, x[1]);
        let t1 = _mm512_permutex2var_epi64(x[0], hi1, x[1]);
        let t2 = _mm512_permutex2var_epi64(x[2], lo1, x[3]);
        let t3 = _mm512_permutex2var_epi64(x[2], hi1, x[3]);
        let t4 = _mm512_permutex2var_epi64(x[4], lo1, x[5]);
        let t5 = _mm512_permutex2var_epi64(x[4], hi1, x[5]);
        let t6 = _mm512_permutex2var_epi64(x[6], lo1, x[7]);
        let t7 = _mm512_permutex2var_epi64(x[6], hi1, x[7]);
        let u0 = _mm512_permutex2var_epi64(t0, lo2, t2);
        let u1 = _mm512_permutex2var_epi64(t0, hi2, t2);
        let u2 = _mm512_permutex2var_epi64(t1, lo2, t3);
        let u3 = _mm512_permutex2var_epi64(t1, hi2, t3);
        let u4 = _mm512_permutex2var_epi64(t4, lo2, t6);
        let u5 = _mm512_permutex2var_epi64(t4, hi2, t6);
        let u6 = _mm512_permutex2var_epi64(t5, lo2, t7);
        let u7 = _mm512_permutex2var_epi64(t5, hi2, t7);
        [
            _mm512_permutex2var_epi64(u0, lo4, u4),
            _mm512_permutex2var_epi64(u0, hi4, u4),
            _mm512_permutex2var_epi64(u1, lo4, u5),
            _mm512_permutex2var_epi64(u1, hi4, u5),
            _mm512_permutex2var_epi64(u2, lo4, u6),
            _mm512_permutex2var_epi64(u2, hi4, u6),
            _mm512_permutex2var_epi64(u3, lo4, u7),
            _mm512_permutex2var_epi64(u3, hi4, u7),
        ]
    }
}
const BYTE_TR: [u8; 64] = [
    0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57, 2, 10, 18, 26, 34, 42, 50, 58, 3,
    11, 19, 27, 35, 43, 51, 59, 4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53, 61, 6,
    14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63,
];

#[inline(always)]
unsafe fn ring_prepared_pack(v: __m512i) -> __m512i {
    const ROW: [u8; 64] = [
        0, 1, 2, 3, 32, 33, 34, 35, 4, 5, 6, 7, 36, 37, 38, 39, 8, 9, 10, 11, 40, 41, 42, 43, 12,
        13, 14, 15, 44, 45, 46, 47, 16, 17, 18, 19, 48, 49, 50, 51, 20, 21, 22, 23, 52, 53, 54, 55,
        24, 25, 26, 27, 56, 57, 58, 59, 28, 29, 30, 31, 60, 61, 62, 63,
    ];
    const COLUMN: [u8; 64] = [
        0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57, 2, 10, 18, 26, 34, 42, 50, 58,
        3, 11, 19, 27, 35, 43, 51, 59, 4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53,
        61, 6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63,
    ];
    let row = _mm512_permutexvar_epi8(_mm512_loadu_si512(ROW.as_ptr().cast()), v);

    let basis = _mm512_set1_epi64(0x0102040810204080u64 as i64);
    let transpose = _mm512_gf2p8affine_epi64_epi8::<0>(basis, row);
    let subsets = _mm512_set1_epi64(0x0103050f113355ffu64 as i64);
    let prepared = _mm512_gf2p8affine_epi64_epi8::<0>(subsets, transpose);
    _mm512_permutexvar_epi8(_mm512_loadu_si512(COLUMN.as_ptr().cast()), prepared)
}

#[inline(always)]
unsafe fn pretranspose_ring<const BLK: usize, const PREPARE_A: bool>(
    ring: *const V8,
    rw: usize,
) -> [u64; 64] {
    unsafe {
        const INDEX: [u8; 64] = [
            0, 4, 8, 12, 16, 20, 24, 28, 1, 5, 9, 13, 17, 21, 25, 29, 2, 6, 10, 14, 18, 22, 26, 30,
            3, 7, 11, 15, 19, 23, 27, 31, 32, 36, 40, 44, 48, 52, 56, 60, 33, 37, 41, 45, 49, 53,
            57, 61, 34, 38, 42, 46, 50, 54, 58, 62, 35, 39, 43, 47, 51, 55, 59, 63,
        ];
        let ix = _mm512_loadu_si512(INDEX.as_ptr().cast());
        let mut out = core::mem::MaybeUninit::<[u64; 64]>::uninit();
        let p = out.as_mut_ptr().cast::<u64>();
        let v = _mm512_loadu_si512(ring.add(rw + 0).cast());
        let v = if PREPARE_A || mode(BLK, 0) == 0 {
            ring_prepared_pack(v)
        } else {
            _mm512_permutexvar_epi8(ix, v)
        };
        _mm512_storeu_si512(p.add(0).cast(), v);
        let v = _mm512_loadu_si512(ring.add(rw + 2).cast());
        let v = if PREPARE_A || mode(BLK, 1) == 0 {
            ring_prepared_pack(v)
        } else {
            _mm512_permutexvar_epi8(ix, v)
        };
        _mm512_storeu_si512(p.add(8).cast(), v);
        let v = _mm512_loadu_si512(ring.add(rw + 4).cast());
        let v = if PREPARE_A || mode(BLK, 2) == 0 {
            ring_prepared_pack(v)
        } else {
            _mm512_permutexvar_epi8(ix, v)
        };
        _mm512_storeu_si512(p.add(16).cast(), v);
        let v = _mm512_loadu_si512(ring.add(rw + 6).cast());
        let v = if PREPARE_A || mode(BLK, 3) == 0 {
            ring_prepared_pack(v)
        } else {
            _mm512_permutexvar_epi8(ix, v)
        };
        _mm512_storeu_si512(p.add(24).cast(), v);
        let v = _mm512_loadu_si512(ring.add(rw + 8).cast());
        let v = if PREPARE_A || mode(BLK, 4) == 0 {
            ring_prepared_pack(v)
        } else {
            _mm512_permutexvar_epi8(ix, v)
        };
        _mm512_storeu_si512(p.add(32).cast(), v);
        let v = _mm512_loadu_si512(ring.add(rw + 10).cast());
        let v = if PREPARE_A || mode(BLK, 5) == 0 {
            ring_prepared_pack(v)
        } else {
            _mm512_permutexvar_epi8(ix, v)
        };
        _mm512_storeu_si512(p.add(40).cast(), v);
        let v = _mm512_loadu_si512(ring.add(rw + 12).cast());
        let v = if PREPARE_A || mode(BLK, 6) == 0 {
            ring_prepared_pack(v)
        } else {
            _mm512_permutexvar_epi8(ix, v)
        };
        _mm512_storeu_si512(p.add(48).cast(), v);
        let v = _mm512_loadu_si512(ring.add(rw + 14).cast());
        let v = if PREPARE_A || mode(BLK, 7) == 0 {
            ring_prepared_pack(v)
        } else {
            _mm512_permutexvar_epi8(ix, v)
        };
        _mm512_storeu_si512(p.add(56).cast(), v);
        out.assume_init()
    }
}

#[inline(always)]
unsafe fn row_product<const MODE: u8, const FIRST: bool>(
    ap: *const u64,
    bp: *const u64,
    ma: *const u64,
    mb: *const u64,
    acc: &mut [__m512i; 8],
) {
    unsafe {
        let acc0 = acc[0];
        acc[0] = row_product_vector_abi::<MODE, FIRST>(ap, bp, ma, mb, acc, acc0);
    }
}

#[inline(never)]
#[allow(improper_ctypes_definitions)]
unsafe extern "C" fn row_product_vector_abi<const MODE: u8, const FIRST: bool>(
    ap: *const u64,
    bp: *const u64,
    ma: *const u64,
    mb: *const u64,
    acc: &mut [__m512i; 8],
    acc0: __m512i,
) -> __m512i {
    unsafe {
        let av = affine_prepared(ap, ma);
        let bv = if MODE == 0 {
            affine_prepared(bp, mb)
        } else {
            affine_conv(bp, mb, MODE)
        };
        let mut result = _mm512_setzero_si512();
        for i in 0..8 {
            let value = if MODE == 15 {
                av[i]
            } else {
                let value = _mm512_gf2p8mul_epi8(av[i], bv[i]);
                if MODE == 0 {
                    value
                } else {
                    _mm512_xor_si512(value, av[i])
                }
            };
            let previous = if i == 0 { acc0 } else { acc[i] };
            let value = if FIRST {
                value
            } else {
                _mm512_xor_si512(previous, value)
            };
            if i == 0 {
                result = value;
            } else {
                acc[i] = value;
            }
        }
        result
    }
}
#[inline(always)]
const fn residual_row_live(blk: usize, k: usize) -> bool {
    (blk != 2 || k >= 2) && (blk != 29 || k <= 3)
}
#[inline(always)]
unsafe fn project_const<const BLK: usize>(
    proj: &StreamProj<'_>,
    rows: RankedRows,
    a: [__m512i; 8],
    b: [__m512i; 8],
    ap: &[u64; 64],
    br: *const V8,
    rw: usize,
    m: *const u64,
) {
    unsafe {
        let bp = pretranspose_ring::<BLK, false>(br, rw);
        let mut acc = [_mm512_setzero_si512(); 8];
        rows.publish_dense_values(0, a[0], b[0]);
        if residual_row_live(BLK, 7) {
            match mode(BLK, 7) {
                0 => row_product::<0, true>(
                    ap.as_ptr().add(7 * 8),
                    bp.as_ptr().add(7 * 8),
                    m.add(7 * 128),
                    m,
                    &mut acc,
                ),
                1 => row_product::<1, true>(
                    ap.as_ptr().add(7 * 8),
                    bp.as_ptr().add(7 * 8),
                    m.add(7 * 128),
                    m,
                    &mut acc,
                ),
                2 => row_product::<2, true>(
                    ap.as_ptr().add(7 * 8),
                    bp.as_ptr().add(7 * 8),
                    m.add(7 * 128),
                    m,
                    &mut acc,
                ),
                3 => row_product::<3, true>(
                    ap.as_ptr().add(7 * 8),
                    bp.as_ptr().add(7 * 8),
                    m.add(7 * 128),
                    m,
                    &mut acc,
                ),
                4 => row_product::<4, true>(
                    ap.as_ptr().add(7 * 8),
                    bp.as_ptr().add(7 * 8),
                    m.add(7 * 128),
                    m,
                    &mut acc,
                ),
                5 => row_product::<5, true>(
                    ap.as_ptr().add(7 * 8),
                    bp.as_ptr().add(7 * 8),
                    m.add(7 * 128),
                    m,
                    &mut acc,
                ),
                6 => row_product::<6, true>(
                    ap.as_ptr().add(7 * 8),
                    bp.as_ptr().add(7 * 8),
                    m.add(7 * 128),
                    m,
                    &mut acc,
                ),
                7 => row_product::<7, true>(
                    ap.as_ptr().add(7 * 8),
                    bp.as_ptr().add(7 * 8),
                    m.add(7 * 128),
                    m,
                    &mut acc,
                ),
                8 => row_product::<8, true>(
                    ap.as_ptr().add(7 * 8),
                    bp.as_ptr().add(7 * 8),
                    m.add(7 * 128),
                    m,
                    &mut acc,
                ),
                9 => row_product::<9, true>(
                    ap.as_ptr().add(7 * 8),
                    bp.as_ptr().add(7 * 8),
                    m.add(7 * 128),
                    m,
                    &mut acc,
                ),
                10 => row_product::<10, true>(
                    ap.as_ptr().add(7 * 8),
                    bp.as_ptr().add(7 * 8),
                    m.add(7 * 128),
                    m,
                    &mut acc,
                ),
                11 => row_product::<11, true>(
                    ap.as_ptr().add(7 * 8),
                    bp.as_ptr().add(7 * 8),
                    m.add(7 * 128),
                    m,
                    &mut acc,
                ),
                12 => row_product::<12, true>(
                    ap.as_ptr().add(7 * 8),
                    bp.as_ptr().add(7 * 8),
                    m.add(7 * 128),
                    m,
                    &mut acc,
                ),
                13 => row_product::<13, true>(
                    ap.as_ptr().add(7 * 8),
                    bp.as_ptr().add(7 * 8),
                    m.add(7 * 128),
                    m,
                    &mut acc,
                ),
                14 => row_product::<14, true>(
                    ap.as_ptr().add(7 * 8),
                    bp.as_ptr().add(7 * 8),
                    m.add(7 * 128),
                    m,
                    &mut acc,
                ),
                15 => row_product::<15, true>(
                    ap.as_ptr().add(7 * 8),
                    bp.as_ptr().add(7 * 8),
                    m.add(7 * 128),
                    m,
                    &mut acc,
                ),
                _ => unreachable!(),
            }
        }

        core::arch::asm!(
            "/* Horner NT publication boundary */",
            options(nostack, preserves_flags)
        );

        rows.publish_dense_values(1, a[1], b[1]);
        if residual_row_live(BLK, 6) {
            match mode(BLK, 6) {
                0 => row_product::<0, false>(
                    ap.as_ptr().add(6 * 8),
                    bp.as_ptr().add(6 * 8),
                    m.add(6 * 128),
                    m,
                    &mut acc,
                ),
                1 => row_product::<1, false>(
                    ap.as_ptr().add(6 * 8),
                    bp.as_ptr().add(6 * 8),
                    m.add(6 * 128),
                    m,
                    &mut acc,
                ),
                2 => row_product::<2, false>(
                    ap.as_ptr().add(6 * 8),
                    bp.as_ptr().add(6 * 8),
                    m.add(6 * 128),
                    m,
                    &mut acc,
                ),
                3 => row_product::<3, false>(
                    ap.as_ptr().add(6 * 8),
                    bp.as_ptr().add(6 * 8),
                    m.add(6 * 128),
                    m,
                    &mut acc,
                ),
                4 => row_product::<4, false>(
                    ap.as_ptr().add(6 * 8),
                    bp.as_ptr().add(6 * 8),
                    m.add(6 * 128),
                    m,
                    &mut acc,
                ),
                5 => row_product::<5, false>(
                    ap.as_ptr().add(6 * 8),
                    bp.as_ptr().add(6 * 8),
                    m.add(6 * 128),
                    m,
                    &mut acc,
                ),
                6 => row_product::<6, false>(
                    ap.as_ptr().add(6 * 8),
                    bp.as_ptr().add(6 * 8),
                    m.add(6 * 128),
                    m,
                    &mut acc,
                ),
                7 => row_product::<7, false>(
                    ap.as_ptr().add(6 * 8),
                    bp.as_ptr().add(6 * 8),
                    m.add(6 * 128),
                    m,
                    &mut acc,
                ),
                8 => row_product::<8, false>(
                    ap.as_ptr().add(6 * 8),
                    bp.as_ptr().add(6 * 8),
                    m.add(6 * 128),
                    m,
                    &mut acc,
                ),
                9 => row_product::<9, false>(
                    ap.as_ptr().add(6 * 8),
                    bp.as_ptr().add(6 * 8),
                    m.add(6 * 128),
                    m,
                    &mut acc,
                ),
                10 => row_product::<10, false>(
                    ap.as_ptr().add(6 * 8),
                    bp.as_ptr().add(6 * 8),
                    m.add(6 * 128),
                    m,
                    &mut acc,
                ),
                11 => row_product::<11, false>(
                    ap.as_ptr().add(6 * 8),
                    bp.as_ptr().add(6 * 8),
                    m.add(6 * 128),
                    m,
                    &mut acc,
                ),
                12 => row_product::<12, false>(
                    ap.as_ptr().add(6 * 8),
                    bp.as_ptr().add(6 * 8),
                    m.add(6 * 128),
                    m,
                    &mut acc,
                ),
                13 => row_product::<13, false>(
                    ap.as_ptr().add(6 * 8),
                    bp.as_ptr().add(6 * 8),
                    m.add(6 * 128),
                    m,
                    &mut acc,
                ),
                14 => row_product::<14, false>(
                    ap.as_ptr().add(6 * 8),
                    bp.as_ptr().add(6 * 8),
                    m.add(6 * 128),
                    m,
                    &mut acc,
                ),
                15 => row_product::<15, false>(
                    ap.as_ptr().add(6 * 8),
                    bp.as_ptr().add(6 * 8),
                    m.add(6 * 128),
                    m,
                    &mut acc,
                ),
                _ => unreachable!(),
            }
        }

        core::arch::asm!(
            "/* Horner NT publication boundary */",
            options(nostack, preserves_flags)
        );

        rows.publish_dense_values(2, a[2], b[2]);
        if residual_row_live(BLK, 5) {
            match mode(BLK, 5) {
                0 => row_product::<0, false>(
                    ap.as_ptr().add(5 * 8),
                    bp.as_ptr().add(5 * 8),
                    m.add(5 * 128),
                    m,
                    &mut acc,
                ),
                1 => row_product::<1, false>(
                    ap.as_ptr().add(5 * 8),
                    bp.as_ptr().add(5 * 8),
                    m.add(5 * 128),
                    m,
                    &mut acc,
                ),
                2 => row_product::<2, false>(
                    ap.as_ptr().add(5 * 8),
                    bp.as_ptr().add(5 * 8),
                    m.add(5 * 128),
                    m,
                    &mut acc,
                ),
                3 => row_product::<3, false>(
                    ap.as_ptr().add(5 * 8),
                    bp.as_ptr().add(5 * 8),
                    m.add(5 * 128),
                    m,
                    &mut acc,
                ),
                4 => row_product::<4, false>(
                    ap.as_ptr().add(5 * 8),
                    bp.as_ptr().add(5 * 8),
                    m.add(5 * 128),
                    m,
                    &mut acc,
                ),
                5 => row_product::<5, false>(
                    ap.as_ptr().add(5 * 8),
                    bp.as_ptr().add(5 * 8),
                    m.add(5 * 128),
                    m,
                    &mut acc,
                ),
                6 => row_product::<6, false>(
                    ap.as_ptr().add(5 * 8),
                    bp.as_ptr().add(5 * 8),
                    m.add(5 * 128),
                    m,
                    &mut acc,
                ),
                7 => row_product::<7, false>(
                    ap.as_ptr().add(5 * 8),
                    bp.as_ptr().add(5 * 8),
                    m.add(5 * 128),
                    m,
                    &mut acc,
                ),
                8 => row_product::<8, false>(
                    ap.as_ptr().add(5 * 8),
                    bp.as_ptr().add(5 * 8),
                    m.add(5 * 128),
                    m,
                    &mut acc,
                ),
                9 => row_product::<9, false>(
                    ap.as_ptr().add(5 * 8),
                    bp.as_ptr().add(5 * 8),
                    m.add(5 * 128),
                    m,
                    &mut acc,
                ),
                10 => row_product::<10, false>(
                    ap.as_ptr().add(5 * 8),
                    bp.as_ptr().add(5 * 8),
                    m.add(5 * 128),
                    m,
                    &mut acc,
                ),
                11 => row_product::<11, false>(
                    ap.as_ptr().add(5 * 8),
                    bp.as_ptr().add(5 * 8),
                    m.add(5 * 128),
                    m,
                    &mut acc,
                ),
                12 => row_product::<12, false>(
                    ap.as_ptr().add(5 * 8),
                    bp.as_ptr().add(5 * 8),
                    m.add(5 * 128),
                    m,
                    &mut acc,
                ),
                13 => row_product::<13, false>(
                    ap.as_ptr().add(5 * 8),
                    bp.as_ptr().add(5 * 8),
                    m.add(5 * 128),
                    m,
                    &mut acc,
                ),
                14 => row_product::<14, false>(
                    ap.as_ptr().add(5 * 8),
                    bp.as_ptr().add(5 * 8),
                    m.add(5 * 128),
                    m,
                    &mut acc,
                ),
                15 => row_product::<15, false>(
                    ap.as_ptr().add(5 * 8),
                    bp.as_ptr().add(5 * 8),
                    m.add(5 * 128),
                    m,
                    &mut acc,
                ),
                _ => unreachable!(),
            }
        }

        core::arch::asm!(
            "/* Horner NT publication boundary */",
            options(nostack, preserves_flags)
        );

        rows.publish_dense_values(3, a[3], b[3]);
        if residual_row_live(BLK, 4) {
            match mode(BLK, 4) {
                0 => row_product::<0, false>(
                    ap.as_ptr().add(4 * 8),
                    bp.as_ptr().add(4 * 8),
                    m.add(4 * 128),
                    m,
                    &mut acc,
                ),
                1 => row_product::<1, false>(
                    ap.as_ptr().add(4 * 8),
                    bp.as_ptr().add(4 * 8),
                    m.add(4 * 128),
                    m,
                    &mut acc,
                ),
                2 => row_product::<2, false>(
                    ap.as_ptr().add(4 * 8),
                    bp.as_ptr().add(4 * 8),
                    m.add(4 * 128),
                    m,
                    &mut acc,
                ),
                3 => row_product::<3, false>(
                    ap.as_ptr().add(4 * 8),
                    bp.as_ptr().add(4 * 8),
                    m.add(4 * 128),
                    m,
                    &mut acc,
                ),
                4 => row_product::<4, false>(
                    ap.as_ptr().add(4 * 8),
                    bp.as_ptr().add(4 * 8),
                    m.add(4 * 128),
                    m,
                    &mut acc,
                ),
                5 => row_product::<5, false>(
                    ap.as_ptr().add(4 * 8),
                    bp.as_ptr().add(4 * 8),
                    m.add(4 * 128),
                    m,
                    &mut acc,
                ),
                6 => row_product::<6, false>(
                    ap.as_ptr().add(4 * 8),
                    bp.as_ptr().add(4 * 8),
                    m.add(4 * 128),
                    m,
                    &mut acc,
                ),
                7 => row_product::<7, false>(
                    ap.as_ptr().add(4 * 8),
                    bp.as_ptr().add(4 * 8),
                    m.add(4 * 128),
                    m,
                    &mut acc,
                ),
                8 => row_product::<8, false>(
                    ap.as_ptr().add(4 * 8),
                    bp.as_ptr().add(4 * 8),
                    m.add(4 * 128),
                    m,
                    &mut acc,
                ),
                9 => row_product::<9, false>(
                    ap.as_ptr().add(4 * 8),
                    bp.as_ptr().add(4 * 8),
                    m.add(4 * 128),
                    m,
                    &mut acc,
                ),
                10 => row_product::<10, false>(
                    ap.as_ptr().add(4 * 8),
                    bp.as_ptr().add(4 * 8),
                    m.add(4 * 128),
                    m,
                    &mut acc,
                ),
                11 => row_product::<11, false>(
                    ap.as_ptr().add(4 * 8),
                    bp.as_ptr().add(4 * 8),
                    m.add(4 * 128),
                    m,
                    &mut acc,
                ),
                12 => row_product::<12, false>(
                    ap.as_ptr().add(4 * 8),
                    bp.as_ptr().add(4 * 8),
                    m.add(4 * 128),
                    m,
                    &mut acc,
                ),
                13 => row_product::<13, false>(
                    ap.as_ptr().add(4 * 8),
                    bp.as_ptr().add(4 * 8),
                    m.add(4 * 128),
                    m,
                    &mut acc,
                ),
                14 => row_product::<14, false>(
                    ap.as_ptr().add(4 * 8),
                    bp.as_ptr().add(4 * 8),
                    m.add(4 * 128),
                    m,
                    &mut acc,
                ),
                15 => row_product::<15, false>(
                    ap.as_ptr().add(4 * 8),
                    bp.as_ptr().add(4 * 8),
                    m.add(4 * 128),
                    m,
                    &mut acc,
                ),
                _ => unreachable!(),
            }
        }

        core::arch::asm!(
            "/* Horner NT publication boundary */",
            options(nostack, preserves_flags)
        );

        rows.publish_dense_values(4, a[4], b[4]);
        if BLK == 29 {
            match mode(BLK, 3) {
                0 => row_product::<0, true>(
                    ap.as_ptr().add(3 * 8),
                    bp.as_ptr().add(3 * 8),
                    m.add(3 * 128),
                    m,
                    &mut acc,
                ),
                1 => row_product::<1, true>(
                    ap.as_ptr().add(3 * 8),
                    bp.as_ptr().add(3 * 8),
                    m.add(3 * 128),
                    m,
                    &mut acc,
                ),
                2 => row_product::<2, true>(
                    ap.as_ptr().add(3 * 8),
                    bp.as_ptr().add(3 * 8),
                    m.add(3 * 128),
                    m,
                    &mut acc,
                ),
                3 => row_product::<3, true>(
                    ap.as_ptr().add(3 * 8),
                    bp.as_ptr().add(3 * 8),
                    m.add(3 * 128),
                    m,
                    &mut acc,
                ),
                4 => row_product::<4, true>(
                    ap.as_ptr().add(3 * 8),
                    bp.as_ptr().add(3 * 8),
                    m.add(3 * 128),
                    m,
                    &mut acc,
                ),
                5 => row_product::<5, true>(
                    ap.as_ptr().add(3 * 8),
                    bp.as_ptr().add(3 * 8),
                    m.add(3 * 128),
                    m,
                    &mut acc,
                ),
                6 => row_product::<6, true>(
                    ap.as_ptr().add(3 * 8),
                    bp.as_ptr().add(3 * 8),
                    m.add(3 * 128),
                    m,
                    &mut acc,
                ),
                7 => row_product::<7, true>(
                    ap.as_ptr().add(3 * 8),
                    bp.as_ptr().add(3 * 8),
                    m.add(3 * 128),
                    m,
                    &mut acc,
                ),
                8 => row_product::<8, true>(
                    ap.as_ptr().add(3 * 8),
                    bp.as_ptr().add(3 * 8),
                    m.add(3 * 128),
                    m,
                    &mut acc,
                ),
                9 => row_product::<9, true>(
                    ap.as_ptr().add(3 * 8),
                    bp.as_ptr().add(3 * 8),
                    m.add(3 * 128),
                    m,
                    &mut acc,
                ),
                10 => row_product::<10, true>(
                    ap.as_ptr().add(3 * 8),
                    bp.as_ptr().add(3 * 8),
                    m.add(3 * 128),
                    m,
                    &mut acc,
                ),
                11 => row_product::<11, true>(
                    ap.as_ptr().add(3 * 8),
                    bp.as_ptr().add(3 * 8),
                    m.add(3 * 128),
                    m,
                    &mut acc,
                ),
                12 => row_product::<12, true>(
                    ap.as_ptr().add(3 * 8),
                    bp.as_ptr().add(3 * 8),
                    m.add(3 * 128),
                    m,
                    &mut acc,
                ),
                13 => row_product::<13, true>(
                    ap.as_ptr().add(3 * 8),
                    bp.as_ptr().add(3 * 8),
                    m.add(3 * 128),
                    m,
                    &mut acc,
                ),
                14 => row_product::<14, true>(
                    ap.as_ptr().add(3 * 8),
                    bp.as_ptr().add(3 * 8),
                    m.add(3 * 128),
                    m,
                    &mut acc,
                ),
                15 => row_product::<15, true>(
                    ap.as_ptr().add(3 * 8),
                    bp.as_ptr().add(3 * 8),
                    m.add(3 * 128),
                    m,
                    &mut acc,
                ),
                _ => unreachable!(),
            }
        } else {
            match mode(BLK, 3) {
                0 => row_product::<0, false>(
                    ap.as_ptr().add(3 * 8),
                    bp.as_ptr().add(3 * 8),
                    m.add(3 * 128),
                    m,
                    &mut acc,
                ),
                1 => row_product::<1, false>(
                    ap.as_ptr().add(3 * 8),
                    bp.as_ptr().add(3 * 8),
                    m.add(3 * 128),
                    m,
                    &mut acc,
                ),
                2 => row_product::<2, false>(
                    ap.as_ptr().add(3 * 8),
                    bp.as_ptr().add(3 * 8),
                    m.add(3 * 128),
                    m,
                    &mut acc,
                ),
                3 => row_product::<3, false>(
                    ap.as_ptr().add(3 * 8),
                    bp.as_ptr().add(3 * 8),
                    m.add(3 * 128),
                    m,
                    &mut acc,
                ),
                4 => row_product::<4, false>(
                    ap.as_ptr().add(3 * 8),
                    bp.as_ptr().add(3 * 8),
                    m.add(3 * 128),
                    m,
                    &mut acc,
                ),
                5 => row_product::<5, false>(
                    ap.as_ptr().add(3 * 8),
                    bp.as_ptr().add(3 * 8),
                    m.add(3 * 128),
                    m,
                    &mut acc,
                ),
                6 => row_product::<6, false>(
                    ap.as_ptr().add(3 * 8),
                    bp.as_ptr().add(3 * 8),
                    m.add(3 * 128),
                    m,
                    &mut acc,
                ),
                7 => row_product::<7, false>(
                    ap.as_ptr().add(3 * 8),
                    bp.as_ptr().add(3 * 8),
                    m.add(3 * 128),
                    m,
                    &mut acc,
                ),
                8 => row_product::<8, false>(
                    ap.as_ptr().add(3 * 8),
                    bp.as_ptr().add(3 * 8),
                    m.add(3 * 128),
                    m,
                    &mut acc,
                ),
                9 => row_product::<9, false>(
                    ap.as_ptr().add(3 * 8),
                    bp.as_ptr().add(3 * 8),
                    m.add(3 * 128),
                    m,
                    &mut acc,
                ),
                10 => row_product::<10, false>(
                    ap.as_ptr().add(3 * 8),
                    bp.as_ptr().add(3 * 8),
                    m.add(3 * 128),
                    m,
                    &mut acc,
                ),
                11 => row_product::<11, false>(
                    ap.as_ptr().add(3 * 8),
                    bp.as_ptr().add(3 * 8),
                    m.add(3 * 128),
                    m,
                    &mut acc,
                ),
                12 => row_product::<12, false>(
                    ap.as_ptr().add(3 * 8),
                    bp.as_ptr().add(3 * 8),
                    m.add(3 * 128),
                    m,
                    &mut acc,
                ),
                13 => row_product::<13, false>(
                    ap.as_ptr().add(3 * 8),
                    bp.as_ptr().add(3 * 8),
                    m.add(3 * 128),
                    m,
                    &mut acc,
                ),
                14 => row_product::<14, false>(
                    ap.as_ptr().add(3 * 8),
                    bp.as_ptr().add(3 * 8),
                    m.add(3 * 128),
                    m,
                    &mut acc,
                ),
                15 => row_product::<15, false>(
                    ap.as_ptr().add(3 * 8),
                    bp.as_ptr().add(3 * 8),
                    m.add(3 * 128),
                    m,
                    &mut acc,
                ),
                _ => unreachable!(),
            }
        }

        core::arch::asm!(
            "/* Horner NT publication boundary */",
            options(nostack, preserves_flags)
        );

        rows.publish_dense_values(5, a[5], b[5]);
        if residual_row_live(BLK, 2) {
            match mode(BLK, 2) {
                0 => row_product::<0, false>(
                    ap.as_ptr().add(2 * 8),
                    bp.as_ptr().add(2 * 8),
                    m.add(2 * 128),
                    m,
                    &mut acc,
                ),
                1 => row_product::<1, false>(
                    ap.as_ptr().add(2 * 8),
                    bp.as_ptr().add(2 * 8),
                    m.add(2 * 128),
                    m,
                    &mut acc,
                ),
                2 => row_product::<2, false>(
                    ap.as_ptr().add(2 * 8),
                    bp.as_ptr().add(2 * 8),
                    m.add(2 * 128),
                    m,
                    &mut acc,
                ),
                3 => row_product::<3, false>(
                    ap.as_ptr().add(2 * 8),
                    bp.as_ptr().add(2 * 8),
                    m.add(2 * 128),
                    m,
                    &mut acc,
                ),
                4 => row_product::<4, false>(
                    ap.as_ptr().add(2 * 8),
                    bp.as_ptr().add(2 * 8),
                    m.add(2 * 128),
                    m,
                    &mut acc,
                ),
                5 => row_product::<5, false>(
                    ap.as_ptr().add(2 * 8),
                    bp.as_ptr().add(2 * 8),
                    m.add(2 * 128),
                    m,
                    &mut acc,
                ),
                6 => row_product::<6, false>(
                    ap.as_ptr().add(2 * 8),
                    bp.as_ptr().add(2 * 8),
                    m.add(2 * 128),
                    m,
                    &mut acc,
                ),
                7 => row_product::<7, false>(
                    ap.as_ptr().add(2 * 8),
                    bp.as_ptr().add(2 * 8),
                    m.add(2 * 128),
                    m,
                    &mut acc,
                ),
                8 => row_product::<8, false>(
                    ap.as_ptr().add(2 * 8),
                    bp.as_ptr().add(2 * 8),
                    m.add(2 * 128),
                    m,
                    &mut acc,
                ),
                9 => row_product::<9, false>(
                    ap.as_ptr().add(2 * 8),
                    bp.as_ptr().add(2 * 8),
                    m.add(2 * 128),
                    m,
                    &mut acc,
                ),
                10 => row_product::<10, false>(
                    ap.as_ptr().add(2 * 8),
                    bp.as_ptr().add(2 * 8),
                    m.add(2 * 128),
                    m,
                    &mut acc,
                ),
                11 => row_product::<11, false>(
                    ap.as_ptr().add(2 * 8),
                    bp.as_ptr().add(2 * 8),
                    m.add(2 * 128),
                    m,
                    &mut acc,
                ),
                12 => row_product::<12, false>(
                    ap.as_ptr().add(2 * 8),
                    bp.as_ptr().add(2 * 8),
                    m.add(2 * 128),
                    m,
                    &mut acc,
                ),
                13 => row_product::<13, false>(
                    ap.as_ptr().add(2 * 8),
                    bp.as_ptr().add(2 * 8),
                    m.add(2 * 128),
                    m,
                    &mut acc,
                ),
                14 => row_product::<14, false>(
                    ap.as_ptr().add(2 * 8),
                    bp.as_ptr().add(2 * 8),
                    m.add(2 * 128),
                    m,
                    &mut acc,
                ),
                15 => row_product::<15, false>(
                    ap.as_ptr().add(2 * 8),
                    bp.as_ptr().add(2 * 8),
                    m.add(2 * 128),
                    m,
                    &mut acc,
                ),
                _ => unreachable!(),
            }
        }

        core::arch::asm!(
            "/* Horner NT publication boundary */",
            options(nostack, preserves_flags)
        );

        rows.publish_dense_values(6, a[6], b[6]);
        if residual_row_live(BLK, 1) {
            match mode(BLK, 1) {
                0 => row_product::<0, false>(
                    ap.as_ptr().add(1 * 8),
                    bp.as_ptr().add(1 * 8),
                    m.add(1 * 128),
                    m,
                    &mut acc,
                ),
                1 => row_product::<1, false>(
                    ap.as_ptr().add(1 * 8),
                    bp.as_ptr().add(1 * 8),
                    m.add(1 * 128),
                    m,
                    &mut acc,
                ),
                2 => row_product::<2, false>(
                    ap.as_ptr().add(1 * 8),
                    bp.as_ptr().add(1 * 8),
                    m.add(1 * 128),
                    m,
                    &mut acc,
                ),
                3 => row_product::<3, false>(
                    ap.as_ptr().add(1 * 8),
                    bp.as_ptr().add(1 * 8),
                    m.add(1 * 128),
                    m,
                    &mut acc,
                ),
                4 => row_product::<4, false>(
                    ap.as_ptr().add(1 * 8),
                    bp.as_ptr().add(1 * 8),
                    m.add(1 * 128),
                    m,
                    &mut acc,
                ),
                5 => row_product::<5, false>(
                    ap.as_ptr().add(1 * 8),
                    bp.as_ptr().add(1 * 8),
                    m.add(1 * 128),
                    m,
                    &mut acc,
                ),
                6 => row_product::<6, false>(
                    ap.as_ptr().add(1 * 8),
                    bp.as_ptr().add(1 * 8),
                    m.add(1 * 128),
                    m,
                    &mut acc,
                ),
                7 => row_product::<7, false>(
                    ap.as_ptr().add(1 * 8),
                    bp.as_ptr().add(1 * 8),
                    m.add(1 * 128),
                    m,
                    &mut acc,
                ),
                8 => row_product::<8, false>(
                    ap.as_ptr().add(1 * 8),
                    bp.as_ptr().add(1 * 8),
                    m.add(1 * 128),
                    m,
                    &mut acc,
                ),
                9 => row_product::<9, false>(
                    ap.as_ptr().add(1 * 8),
                    bp.as_ptr().add(1 * 8),
                    m.add(1 * 128),
                    m,
                    &mut acc,
                ),
                10 => row_product::<10, false>(
                    ap.as_ptr().add(1 * 8),
                    bp.as_ptr().add(1 * 8),
                    m.add(1 * 128),
                    m,
                    &mut acc,
                ),
                11 => row_product::<11, false>(
                    ap.as_ptr().add(1 * 8),
                    bp.as_ptr().add(1 * 8),
                    m.add(1 * 128),
                    m,
                    &mut acc,
                ),
                12 => row_product::<12, false>(
                    ap.as_ptr().add(1 * 8),
                    bp.as_ptr().add(1 * 8),
                    m.add(1 * 128),
                    m,
                    &mut acc,
                ),
                13 => row_product::<13, false>(
                    ap.as_ptr().add(1 * 8),
                    bp.as_ptr().add(1 * 8),
                    m.add(1 * 128),
                    m,
                    &mut acc,
                ),
                14 => row_product::<14, false>(
                    ap.as_ptr().add(1 * 8),
                    bp.as_ptr().add(1 * 8),
                    m.add(1 * 128),
                    m,
                    &mut acc,
                ),
                15 => row_product::<15, false>(
                    ap.as_ptr().add(1 * 8),
                    bp.as_ptr().add(1 * 8),
                    m.add(1 * 128),
                    m,
                    &mut acc,
                ),
                _ => unreachable!(),
            }
        }

        core::arch::asm!(
            "/* Horner NT publication boundary */",
            options(nostack, preserves_flags)
        );

        rows.publish_dense_values(7, a[7], b[7]);
        if residual_row_live(BLK, 0) {
            match mode(BLK, 0) {
                0 => row_product::<0, false>(
                    ap.as_ptr().add(0 * 8),
                    bp.as_ptr().add(0 * 8),
                    m.add(0 * 128),
                    m,
                    &mut acc,
                ),
                1 => row_product::<1, false>(
                    ap.as_ptr().add(0 * 8),
                    bp.as_ptr().add(0 * 8),
                    m.add(0 * 128),
                    m,
                    &mut acc,
                ),
                2 => row_product::<2, false>(
                    ap.as_ptr().add(0 * 8),
                    bp.as_ptr().add(0 * 8),
                    m.add(0 * 128),
                    m,
                    &mut acc,
                ),
                3 => row_product::<3, false>(
                    ap.as_ptr().add(0 * 8),
                    bp.as_ptr().add(0 * 8),
                    m.add(0 * 128),
                    m,
                    &mut acc,
                ),
                4 => row_product::<4, false>(
                    ap.as_ptr().add(0 * 8),
                    bp.as_ptr().add(0 * 8),
                    m.add(0 * 128),
                    m,
                    &mut acc,
                ),
                5 => row_product::<5, false>(
                    ap.as_ptr().add(0 * 8),
                    bp.as_ptr().add(0 * 8),
                    m.add(0 * 128),
                    m,
                    &mut acc,
                ),
                6 => row_product::<6, false>(
                    ap.as_ptr().add(0 * 8),
                    bp.as_ptr().add(0 * 8),
                    m.add(0 * 128),
                    m,
                    &mut acc,
                ),
                7 => row_product::<7, false>(
                    ap.as_ptr().add(0 * 8),
                    bp.as_ptr().add(0 * 8),
                    m.add(0 * 128),
                    m,
                    &mut acc,
                ),
                8 => row_product::<8, false>(
                    ap.as_ptr().add(0 * 8),
                    bp.as_ptr().add(0 * 8),
                    m.add(0 * 128),
                    m,
                    &mut acc,
                ),
                9 => row_product::<9, false>(
                    ap.as_ptr().add(0 * 8),
                    bp.as_ptr().add(0 * 8),
                    m.add(0 * 128),
                    m,
                    &mut acc,
                ),
                10 => row_product::<10, false>(
                    ap.as_ptr().add(0 * 8),
                    bp.as_ptr().add(0 * 8),
                    m.add(0 * 128),
                    m,
                    &mut acc,
                ),
                11 => row_product::<11, false>(
                    ap.as_ptr().add(0 * 8),
                    bp.as_ptr().add(0 * 8),
                    m.add(0 * 128),
                    m,
                    &mut acc,
                ),
                12 => row_product::<12, false>(
                    ap.as_ptr().add(0 * 8),
                    bp.as_ptr().add(0 * 8),
                    m.add(0 * 128),
                    m,
                    &mut acc,
                ),
                13 => row_product::<13, false>(
                    ap.as_ptr().add(0 * 8),
                    bp.as_ptr().add(0 * 8),
                    m.add(0 * 128),
                    m,
                    &mut acc,
                ),
                14 => row_product::<14, false>(
                    ap.as_ptr().add(0 * 8),
                    bp.as_ptr().add(0 * 8),
                    m.add(0 * 128),
                    m,
                    &mut acc,
                ),
                15 => row_product::<15, false>(
                    ap.as_ptr().add(0 * 8),
                    bp.as_ptr().add(0 * 8),
                    m.add(0 * 128),
                    m,
                    &mut acc,
                ),
                _ => unreachable!(),
            }
        }

        core::arch::asm!(
            "/* Horner NT publication boundary */",
            options(nostack, preserves_flags)
        );

        let ix = _mm512_loadu_si512(BYTE_TR.as_ptr().cast());
        let out = trq([
            _mm512_permutexvar_epi8(ix, acc[0]),
            _mm512_permutexvar_epi8(ix, acc[1]),
            _mm512_permutexvar_epi8(ix, acc[2]),
            _mm512_permutexvar_epi8(ix, acc[3]),
            _mm512_permutexvar_epi8(ix, acc[4]),
            _mm512_permutexvar_epi8(ix, acc[5]),
            _mm512_permutexvar_epi8(ix, acc[6]),
            _mm512_permutexvar_epi8(ix, acc[7]),
        ]);
        _mm512_stream_si512(
            proj.out
                .add(0 * proj.out_stride + BLK * 64 - proj.out_bias)
                .cast(),
            out[0],
        );
        _mm512_stream_si512(
            proj.out
                .add(1 * proj.out_stride + BLK * 64 - proj.out_bias)
                .cast(),
            out[1],
        );
        _mm512_stream_si512(
            proj.out
                .add(2 * proj.out_stride + BLK * 64 - proj.out_bias)
                .cast(),
            out[2],
        );
        _mm512_stream_si512(
            proj.out
                .add(3 * proj.out_stride + BLK * 64 - proj.out_bias)
                .cast(),
            out[3],
        );
        _mm512_stream_si512(
            proj.out
                .add(4 * proj.out_stride + BLK * 64 - proj.out_bias)
                .cast(),
            out[4],
        );
        _mm512_stream_si512(
            proj.out
                .add(5 * proj.out_stride + BLK * 64 - proj.out_bias)
                .cast(),
            out[5],
        );
        _mm512_stream_si512(
            proj.out
                .add(6 * proj.out_stride + BLK * 64 - proj.out_bias)
                .cast(),
            out[6],
        );
        _mm512_stream_si512(
            proj.out
                .add(7 * proj.out_stride + BLK * 64 - proj.out_bias)
                .cast(),
            out[7],
        );
    }
}

#[inline(never)]
pub(super) unsafe fn project(
    proj: &StreamProj<'_>,
    blk: usize,
    rows: RankedRows,
    ar: *const V8,
    br: *const V8,
    rw: usize,
    t: &Tables,
) {
    unsafe {
        let ap = pretranspose_ring::<0, true>(ar, rw);
        let a = tr8x16_zmm(ar, rw);
        let b = tr8x16_zmm(br, rw);
        match blk {
            2 => project_const::<2>(proj, rows, a, b, &ap, br, rw, t.0.as_ptr()),
            3 => project_const::<3>(proj, rows, a, b, &ap, br, rw, t.0.as_ptr()),
            4 => project_const::<4>(proj, rows, a, b, &ap, br, rw, t.0.as_ptr()),
            5 => project_const::<5>(proj, rows, a, b, &ap, br, rw, t.0.as_ptr()),
            6 => project_const::<6>(proj, rows, a, b, &ap, br, rw, t.0.as_ptr()),
            7 => project_const::<7>(proj, rows, a, b, &ap, br, rw, t.0.as_ptr()),
            8 => project_const::<8>(proj, rows, a, b, &ap, br, rw, t.0.as_ptr()),
            9 => project_const::<9>(proj, rows, a, b, &ap, br, rw, t.0.as_ptr()),
            10 => project_const::<10>(proj, rows, a, b, &ap, br, rw, t.0.as_ptr()),
            11 => project_const::<11>(proj, rows, a, b, &ap, br, rw, t.0.as_ptr()),
            12 => project_const::<12>(proj, rows, a, b, &ap, br, rw, t.0.as_ptr()),
            13 => project_const::<13>(proj, rows, a, b, &ap, br, rw, t.0.as_ptr()),
            14 => project_const::<14>(proj, rows, a, b, &ap, br, rw, t.0.as_ptr()),
            15 => project_const::<15>(proj, rows, a, b, &ap, br, rw, t.0.as_ptr()),
            16 => project_const::<16>(proj, rows, a, b, &ap, br, rw, t.0.as_ptr()),
            17 => project_const::<17>(proj, rows, a, b, &ap, br, rw, t.0.as_ptr()),
            18 => project_const::<18>(proj, rows, a, b, &ap, br, rw, t.0.as_ptr()),
            19 => project_const::<19>(proj, rows, a, b, &ap, br, rw, t.0.as_ptr()),
            20 => project_const::<20>(proj, rows, a, b, &ap, br, rw, t.0.as_ptr()),
            21 => project_const::<21>(proj, rows, a, b, &ap, br, rw, t.0.as_ptr()),
            22 => project_const::<22>(proj, rows, a, b, &ap, br, rw, t.0.as_ptr()),
            23 => project_const::<23>(proj, rows, a, b, &ap, br, rw, t.0.as_ptr()),
            24 => project_const::<24>(proj, rows, a, b, &ap, br, rw, t.0.as_ptr()),
            25 => project_const::<25>(proj, rows, a, b, &ap, br, rw, t.0.as_ptr()),
            26 => project_const::<26>(proj, rows, a, b, &ap, br, rw, t.0.as_ptr()),
            27 => project_const::<27>(proj, rows, a, b, &ap, br, rw, t.0.as_ptr()),
            28 => project_const::<28>(proj, rows, a, b, &ap, br, rw, t.0.as_ptr()),
            29 => project_const::<29>(proj, rows, a, b, &ap, br, rw, t.0.as_ptr()),
            _ => unreachable!("only ranked mixed windows 2 through 29 reach XOR projection"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flock_core::{field::F8, ntt::AdditiveNttGf8};
    #[repr(align(64))]
    struct Buf([u8; 8 * BYTES_PER_BLOCK]);
    #[repr(align(32))]
    struct Ring([[u32; 8]; 32]);

    #[test]
    fn prepared_pack_matches_scalar_subset_transform() {
        let mut rng = 0x7841_43cb_c50d_779bu64;
        for case in 0..1538 {
            let mut input = [0u8; 64];
            if case < 512 {
                input[case / 8] = 1 << (case % 8);
            } else if case == 513 {
                input.fill(255);
            } else if case > 513 {
                for byte in &mut input {
                    rng ^= rng << 13;
                    rng ^= rng >> 7;
                    rng ^= rng << 17;
                    *byte = rng as u8;
                }
            }
            let mut expected = [0u8; 64];
            for i in 0..8 {
                for row in 0..8 {
                    for j in 0..8 {
                        if i & j == i {
                            expected[8 * i + row] ^= input[32 * (j / 4) + 4 * row + j % 4];
                        }
                    }
                }
            }
            let mut actual = [0u8; 64];
            unsafe {
                let value = _mm512_loadu_si512(input.as_ptr().cast());
                _mm512_storeu_si512(actual.as_mut_ptr().cast(), ring_prepared_pack(value));
            }
            assert_eq!(actual, expected, "prepared input case {case}");
        }
    }

    #[test]
    fn ranked_inverse_table_is_dense_not_two_sparse() {
        let ns = AdditiveNttGf8::new(6, F8::ZERO);
        let nl = AdditiveNttGf8::new(6, F8(64));
        let table = InvNttTableByteSingleGf8::new(&ns, &nl);
        for t in 0..64 {
            let mut unit = [F8::ZERO; 64];
            unit[t] = F8::ONE;
            ns.inverse(&mut unit);
            nl.forward(&mut unit);
            assert_eq!(unit.iter().filter(|v| **v != F8::ZERO).count(), 64);
            let mut input = [0u8; 8];
            input[t / 8] = 1 << (t % 8);
            let mut actual = [F8::ZERO; 64];
            table.apply_scalar(&input, &mut actual);
            assert_eq!(actual, unit);
            if t == 0 {
                assert_eq!(actual[0], F8(0x76));
            }
        }
        println!("PASS 64 actual NTT unit columns are dense; sparse/involution claim refuted");
    }
    #[test]
    fn xor_projection_publishes_and_matches_scalar() {
        unsafe {
            let ns = AdditiveNttGf8::new(6, F8::ZERO);
            let nl = AdditiveNttGf8::new(6, F8(64));
            let table = InvNttTableByteSingleGf8::new(&ns, &nl);
            let t = Tables::new(&table);
            let mut seed = 1357u64;
            let mut next = || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed as u8
            };
            for rw in [0, 16] {
                for case in 0..5 {
                    for blk in 2..30 {
                        let keep: u8 = match blk {
                            2 => 0xfc,
                            29 => 0x0f,
                            _ => 0xff,
                        };
                        for k in 0..8 {
                            if keep & (1 << k) == 0 {
                                assert_eq!(known(mode(blk, k)), 255);
                            }
                        }
                        let mut aa = [[0u8; 64]; 8];
                        let mut bb = [[0u8; 64]; 8];
                        let mut ar = Ring([[0xdeadbeef; 8]; 32]);
                        let mut br = Ring([[0xdeadbeef; 8]; 32]);
                        for row in 0..8 {
                            for i in 0..64 {
                                aa[row][i] = if case == 0 {
                                    0
                                } else if case == 1 {
                                    255
                                } else {
                                    next()
                                };
                                bb[row][i] = if known(mode(blk, i / 8)) & (1 << (i % 8)) != 0 {
                                    255
                                } else if case == 0 {
                                    0
                                } else if case == 1 {
                                    255
                                } else {
                                    next()
                                };
                            }
                            for w in 0..16 {
                                ar.0[rw + w][row] = u32::from_le_bytes(
                                    aa[row][w * 4..w * 4 + 4].try_into().unwrap(),
                                );
                                br.0[rw + w][row] = u32::from_le_bytes(
                                    bb[row][w * 4..w * 4 + 4].try_into().unwrap(),
                                );
                            }
                        }
                        let mut a = Buf([0xa5; 8 * BYTES_PER_BLOCK]);
                        let mut b = Buf([0xa5; 8 * BYTES_PER_BLOCK]);
                        let mut z = Buf([0xa5; 8 * BYTES_PER_BLOCK]);
                        let mut output = Buf([0xa5; 8 * BYTES_PER_BLOCK]);
                        let stride = 29 * 64;
                        let bias = 128;
                        let plan=flock_core::zerocheck::univariate_skip_optimized::prepare_round1_ab_window_plan(&table,&output.0,true);
                        assert!(plan.bcomplement_static_eligible());
                        let proj = StreamProj {
                            stage: core::ptr::null_mut(),
                            out: output.0.as_mut_ptr(),
                            out_stride: stride,
                            out_bias: bias,
                            inv_table: &table,
                            plan,
                            one_rows_elided: true,
                            w30_elided: true,
                        };
                        let offset = blk * 64;
                        let rows = RankedRows::new(
                            z.0.as_mut_ptr().add(offset).cast(),
                            a.0.as_mut_ptr().add(offset).cast(),
                            b.0.as_mut_ptr().add(offset).cast(),
                        );
                        project(
                            &proj,
                            blk,
                            rows,
                            ar.0.as_ptr().cast(),
                            br.0.as_ptr().cast(),
                            rw,
                            &t,
                        );
                        _mm_sfence();
                        let mut want = Buf([0xa5; 8 * BYTES_PER_BLOCK]);
                        let mut wa = Buf([0xa5; 8 * BYTES_PER_BLOCK]);
                        let mut wb = Buf([0xa5; 8 * BYTES_PER_BLOCK]);
                        let mut wz = Buf([0xa5; 8 * BYTES_PER_BLOCK]);
                        for row in 0..8 {
                            let start = row * BYTES_PER_BLOCK + offset;
                            wa.0[start..start + 64].copy_from_slice(&aa[row]);
                            wb.0[start..start + 64].copy_from_slice(&bb[row]);
                            for i in 0..64 {
                                wz.0[start + i] = aa[row][i] & bb[row][i];
                            }
                            let mut acc = [F8::ZERO; 64];
                            for k in (0..8).rev() {
                                let mut av = [F8::ZERO; 64];
                                let mut bv = [F8::ZERO; 64];
                                table.apply_scalar(&aa[row][k * 8..k * 8 + 8], &mut av);
                                table.apply_scalar(&bb[row][k * 8..k * 8 + 8], &mut bv);
                                for i in 0..64 {
                                    acc[i] = acc[i] * F8(2);
                                    if keep & (1 << k) != 0 {
                                        acc[i] += av[i] * bv[i];
                                    }
                                }
                            }
                            for i in 0..64 {
                                want.0[row * stride + offset - bias + i] = acc[i].0;
                            }
                        }
                        assert_eq!(output.0, want.0, "projection rw={rw} case={case} blk={blk}");
                        assert_eq!(a.0, wa.0);
                        assert_eq!(b.0, wb.0);
                        assert_eq!(z.0, wz.0);
                    }
                }
            }
            println!(
                "280 complete ring-to-published-window cases, including A/B/Z, compact output stride, both ring halves and untouched bytes"
            );
        }
    }
}
