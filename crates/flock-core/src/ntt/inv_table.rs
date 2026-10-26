use crate::field::F8;
use crate::ntt::AdditiveNttGf8;

#[inline]
fn gf8_to_composite_u8(x: u8) -> u8 {
    unsafe { crate::gf2_kernels::gf2_kern_gf8_to_composite(x as u64) as u8 }
}

#[derive(Clone, Debug)]
pub struct InvNttTableByteSingleGf8 {
    pub k: usize,
    pub ell: usize,
    pub n_chunks: usize,

    data: Vec<F8>,

    data_offset: usize,

    composite_data: Vec<u8>,
    composite_offset: usize,
}

impl InvNttTableByteSingleGf8 {
    pub fn new(ntt_s: &AdditiveNttGf8, ntt_l: &AdditiveNttGf8) -> Self {
        assert_eq!(ntt_s.k(), ntt_l.k(), "ntt_S and ntt_L must share k");
        let k = ntt_s.k();
        let ell = 1usize << k;
        assert!(ell >= 8, "ell must be ≥ 8 so n_chunks ≥ 1");
        let n_chunks = ell / 8;
        assert!(
            n_chunks <= 16,
            "n_chunks must fit the i'/chunk XOR encoding"
        );

        const TABLE_ALIGNMENT: usize = 64;
        let table_len = 256 * ell;
        let mut data = vec![F8::ZERO; table_len + TABLE_ALIGNMENT - 1];
        let data_offset = (TABLE_ALIGNMENT - (data.as_ptr() as usize & (TABLE_ALIGNMENT - 1)))
            & (TABLE_ALIGNMENT - 1);
        let table = &mut data[data_offset..data_offset + table_len];

        let mut tmp = vec![F8::ZERO; ell];
        let mut cols: Vec<Vec<F8>> = Vec::with_capacity(8);
        for t in 0..8 {
            tmp.iter_mut().for_each(|x| *x = F8::ZERO);
            tmp[t] = F8::ONE;
            ntt_s.inverse(&mut tmp);
            ntt_l.forward(&mut tmp);
            cols.push(tmp.clone());
        }

        for t in 0..8 {
            let entry_start = (1usize << t) * ell;
            table[entry_start..entry_start + ell].copy_from_slice(&cols[t]);
        }
        for w in 3usize..256 {
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            let (lo_bit, parent) = unsafe {
                (
                    crate::gf2_kernels::gf2_kern_isolate_lowest_one(w as u64) as usize,
                    crate::gf2_kernels::gf2_kern_clear_lowest_one(w as u64) as usize,
                )
            };
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            let (lo_bit, parent) = (w.isolate_lowest_one(), w & (w - 1));
            if parent == 0 {
                continue;
            }

            let (parent_off, bit_off, entry_off) = (parent * ell, lo_bit * ell, w * ell);
            for i in 0..ell {
                let v = table[parent_off + i] + table[bit_off + i];
                table[entry_off + i] = v;
            }
        }

        let mut composite_data = vec![0u8; table_len + TABLE_ALIGNMENT - 1];
        let composite_offset = (TABLE_ALIGNMENT
            - (composite_data.as_ptr() as usize & (TABLE_ALIGNMENT - 1)))
            & (TABLE_ALIGNMENT - 1);
        let comp_table = &mut composite_data[composite_offset..composite_offset + table_len];
        for i in 0..table_len {
            comp_table[i] = gf8_to_composite_u8(table[i].0);
        }

        Self {
            k,
            ell,
            n_chunks,
            data,
            data_offset,
            composite_data,
            composite_offset,
        }
    }

    #[inline]
    pub fn data_ptr(&self) -> *const u8 {
        unsafe { self.data.as_ptr().add(self.data_offset) as *const u8 }
    }

    #[inline]
    pub fn composite_data_ptr(&self) -> *const u8 {
        unsafe { self.composite_data.as_ptr().add(self.composite_offset) }
    }

    #[inline]
    fn table(&self) -> &[F8] {
        &self.data[self.data_offset..self.data_offset + 256 * self.ell]
    }

    #[inline]
    pub fn apply(&self, bytes: &[u8], out: &mut [F8]) {
        #[cfg(target_arch = "aarch64")]
        if self.ell >= 16 {
            unsafe { self.apply_neon_unchecked(bytes, out) };
            return;
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
        if self.ell == 64 {
            unsafe { self.apply_x86_avx512_unchecked(bytes, out) };
            return;
        }
        #[cfg(target_arch = "x86_64")]
        if self.ell >= 16 {
            unsafe { self.apply_x86_unchecked(bytes, out) };
            return;
        }
        self.apply_scalar(bytes, out);
    }

    pub fn apply_scalar(&self, bytes: &[u8], out: &mut [F8]) {
        assert_eq!(bytes.len(), self.n_chunks);
        assert_eq!(out.len(), self.ell);
        out.iter_mut().for_each(|x| *x = F8::ZERO);
        let table = self.table();
        for (b, &byte_b) in bytes.iter().enumerate() {
            let row_off = byte_b as usize * self.ell;
            let row = &table[row_off..row_off + self.ell];
            let shift = 8 * b;
            for i in 0..self.ell {
                out[i] += row[i ^ shift];
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub unsafe fn apply_neon_unchecked(&self, bytes: &[u8], out: &mut [F8]) {
        use core::arch::aarch64::*;
        assert_eq!(bytes.len(), self.n_chunks);
        assert_eq!(out.len(), self.ell);
        let n128 = self.ell / 16;
        let base = self.data_ptr();
        let out_ptr = out.as_mut_ptr() as *mut u8;

        unsafe {
            let row0 = base.add(bytes[0] as usize * self.ell);
            for c in 0..n128 {
                vst1q_u8(out_ptr.add(c * 16), vld1q_u8(row0.add(c * 16)));
            }

            for b in 1..self.n_chunks {
                let b_high = b >> 1;
                let b_odd = (b & 1) != 0;
                let row_b = base.add(bytes[b] as usize * self.ell);
                if b_odd {
                    for c in 0..n128 {
                        let sc = c ^ b_high;
                        let v = vld1q_u8(row_b.add(sc * 16));
                        let v_swapped = vextq_u8::<8>(v, v);
                        let dst = out_ptr.add(c * 16);
                        vst1q_u8(dst, veorq_u8(vld1q_u8(dst), v_swapped));
                    }
                } else {
                    for c in 0..n128 {
                        let sc = c ^ b_high;
                        let v = vld1q_u8(row_b.add(sc * 16));
                        let dst = out_ptr.add(c * 16);
                        vst1q_u8(dst, veorq_u8(vld1q_u8(dst), v));
                    }
                }
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[inline(always)]
    pub(crate) unsafe fn apply_x86_avx2_pair_unchecked(
        &self,
        bytes: *const u8,
    ) -> (core::arch::x86_64::__m256i, core::arch::x86_64::__m256i) {
        use core::arch::x86_64::*;
        debug_assert_eq!(self.ell, 64);
        debug_assert_eq!(self.n_chunks, 8);
        let base = self.data_ptr();
        unsafe {
            let w = (bytes as *const u64).read_unaligned() as usize;
            let row0 = base.add((w & 0xFF) << 6);
            let row4 = base.add(((w >> 32) & 0xFF) << 6);
            let r04_lo = _mm256_xor_si256(
                _mm256_load_si256(row0 as *const __m256i),
                _mm256_load_si256(row4.add(32) as *const __m256i),
            );
            let r04_hi = _mm256_xor_si256(
                _mm256_load_si256(row0.add(32) as *const __m256i),
                _mm256_load_si256(row4 as *const __m256i),
            );

            let row1 = base.add(((w >> 8) & 0xFF) << 6);
            let row5 = base.add(((w >> 40) & 0xFF) << 6);
            let r15_lo = _mm256_xor_si256(
                _mm256_load_si256(row1 as *const __m256i),
                _mm256_load_si256(row5.add(32) as *const __m256i),
            );
            let r15_hi = _mm256_xor_si256(
                _mm256_load_si256(row1.add(32) as *const __m256i),
                _mm256_load_si256(row5 as *const __m256i),
            );
            let even_lo = _mm256_xor_si256(r04_lo, _mm256_shuffle_epi32::<0x4E>(r15_lo));
            let even_hi = _mm256_xor_si256(r04_hi, _mm256_shuffle_epi32::<0x4E>(r15_hi));

            let row2 = base.add(((w >> 16) & 0xFF) << 6);
            let row6 = base.add(((w >> 48) & 0xFF) << 6);
            let r26_lo = _mm256_xor_si256(
                _mm256_load_si256(row2 as *const __m256i),
                _mm256_load_si256(row6.add(32) as *const __m256i),
            );
            let r26_hi = _mm256_xor_si256(
                _mm256_load_si256(row2.add(32) as *const __m256i),
                _mm256_load_si256(row6 as *const __m256i),
            );

            let row3 = base.add(((w >> 24) & 0xFF) << 6);
            let row7 = base.add((w >> 56) << 6);
            let r37_lo = _mm256_xor_si256(
                _mm256_load_si256(row3 as *const __m256i),
                _mm256_load_si256(row7.add(32) as *const __m256i),
            );
            let r37_hi = _mm256_xor_si256(
                _mm256_load_si256(row3.add(32) as *const __m256i),
                _mm256_load_si256(row7 as *const __m256i),
            );
            let odd_lo = _mm256_xor_si256(r26_lo, _mm256_shuffle_epi32::<0x4E>(r37_lo));
            let odd_hi = _mm256_xor_si256(r26_hi, _mm256_shuffle_epi32::<0x4E>(r37_hi));
            let lo = _mm256_xor_si256(even_lo, _mm256_permute4x64_epi64::<0x4E>(odd_lo));
            let hi = _mm256_xor_si256(even_hi, _mm256_permute4x64_epi64::<0x4E>(odd_hi));

            (lo, hi)
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    #[inline(always)]
    pub(crate) unsafe fn apply_x86_avx2_pair_composite_unchecked(
        &self,
        bytes: *const u8,
    ) -> (core::arch::x86_64::__m256i, core::arch::x86_64::__m256i) {
        use core::arch::x86_64::*;
        debug_assert_eq!(self.ell, 64);
        debug_assert_eq!(self.n_chunks, 8);
        let base = self.composite_data_ptr();
        unsafe {
            debug_assert_eq!(crate::gf2_kernels::gf2_kern_inv_ntt_cyclic_offset_lo(6), 64);
            debug_assert_eq!(crate::gf2_kernels::gf2_kern_inv_ntt_cyclic_offset_hi(6), 16);
            let w = (bytes as *const u64).read_unaligned() as usize;
            let row0 = base.add((w & 0xFF) << 6);
            let row4 = base.add(((w >> 32) & 0xFF) << 6);
            let r04_lo = _mm256_xor_si256(
                _mm256_load_si256(row0 as *const __m256i),
                _mm256_load_si256(row4.add(32) as *const __m256i),
            );
            let r04_hi = _mm256_xor_si256(
                _mm256_load_si256(row0.add(32) as *const __m256i),
                _mm256_load_si256(row4 as *const __m256i),
            );

            let row1 = base.add(((w >> 8) & 0xFF) << 6);
            let row5 = base.add(((w >> 40) & 0xFF) << 6);
            let r15_lo = _mm256_xor_si256(
                _mm256_load_si256(row1 as *const __m256i),
                _mm256_load_si256(row5.add(32) as *const __m256i),
            );
            let r15_hi = _mm256_xor_si256(
                _mm256_load_si256(row1.add(32) as *const __m256i),
                _mm256_load_si256(row5 as *const __m256i),
            );
            let even_lo = _mm256_xor_si256(r04_lo, _mm256_shuffle_epi32::<0x4E>(r15_lo));
            let even_hi = _mm256_xor_si256(r04_hi, _mm256_shuffle_epi32::<0x4E>(r15_hi));

            let row2 = base.add(((w >> 16) & 0xFF) << 6);
            let row6 = base.add(((w >> 48) & 0xFF) << 6);
            let r26_lo = _mm256_xor_si256(
                _mm256_load_si256(row2 as *const __m256i),
                _mm256_load_si256(row6.add(32) as *const __m256i),
            );
            let r26_hi = _mm256_xor_si256(
                _mm256_load_si256(row2.add(32) as *const __m256i),
                _mm256_load_si256(row6 as *const __m256i),
            );

            let row3 = base.add(((w >> 24) & 0xFF) << 6);
            let row7 = base.add((w >> 56) << 6);
            let r37_lo = _mm256_xor_si256(
                _mm256_load_si256(row3 as *const __m256i),
                _mm256_load_si256(row7.add(32) as *const __m256i),
            );
            let r37_hi = _mm256_xor_si256(
                _mm256_load_si256(row3.add(32) as *const __m256i),
                _mm256_load_si256(row7 as *const __m256i),
            );
            let odd_lo = _mm256_xor_si256(r26_lo, _mm256_shuffle_epi32::<0x4E>(r37_lo));
            let odd_hi = _mm256_xor_si256(r26_hi, _mm256_shuffle_epi32::<0x4E>(r37_hi));
            let lo = _mm256_xor_si256(even_lo, _mm256_permute4x64_epi64::<0x4E>(odd_lo));
            let hi = _mm256_xor_si256(even_hi, _mm256_permute4x64_epi64::<0x4E>(odd_hi));

            (lo, hi)
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    pub(crate) unsafe fn apply_x86_u64x8_unchecked(&self, bytes: *const u8) -> [u64; 8] {
        use core::arch::x86_64::*;
        debug_assert_eq!(self.ell, 64);
        debug_assert_eq!(self.n_chunks, 8);
        let base = self.data_ptr();

        #[cfg(target_feature = "avx2")]
        unsafe {
            let _ = base;
            let (lo, hi) = self.apply_x86_avx2_pair_unchecked(bytes);
            core::mem::transmute::<[__m256i; 2], [u64; 8]>([lo, hi])
        }

        #[cfg(not(target_feature = "avx2"))]
        unsafe {
            let row0 = base.add(*bytes as usize * 64);
            let mut d0 = _mm_loadu_si128(row0 as *const __m128i);
            let mut d1 = _mm_loadu_si128(row0.add(16) as *const __m128i);
            let mut d2 = _mm_loadu_si128(row0.add(32) as *const __m128i);
            let mut d3 = _mm_loadu_si128(row0.add(48) as *const __m128i);

            macro_rules! step_b {
                ($b:literal, $bh:literal, true) => {{
                    let row_b = base.add(*bytes.add($b) as usize * 64);
                    d0 = _mm_xor_si128(
                        d0,
                        _mm_shuffle_epi32::<0x4E>(_mm_loadu_si128(
                            row_b.add((0 ^ $bh) * 16) as *const __m128i
                        )),
                    );
                    d1 = _mm_xor_si128(
                        d1,
                        _mm_shuffle_epi32::<0x4E>(_mm_loadu_si128(
                            row_b.add((1 ^ $bh) * 16) as *const __m128i
                        )),
                    );
                    d2 = _mm_xor_si128(
                        d2,
                        _mm_shuffle_epi32::<0x4E>(_mm_loadu_si128(
                            row_b.add((2 ^ $bh) * 16) as *const __m128i
                        )),
                    );
                    d3 = _mm_xor_si128(
                        d3,
                        _mm_shuffle_epi32::<0x4E>(_mm_loadu_si128(
                            row_b.add((3 ^ $bh) * 16) as *const __m128i
                        )),
                    );
                }};
                ($b:literal, $bh:literal, false) => {{
                    let row_b = base.add(*bytes.add($b) as usize * 64);
                    d0 = _mm_xor_si128(
                        d0,
                        _mm_loadu_si128(row_b.add((0 ^ $bh) * 16) as *const __m128i),
                    );
                    d1 = _mm_xor_si128(
                        d1,
                        _mm_loadu_si128(row_b.add((1 ^ $bh) * 16) as *const __m128i),
                    );
                    d2 = _mm_xor_si128(
                        d2,
                        _mm_loadu_si128(row_b.add((2 ^ $bh) * 16) as *const __m128i),
                    );
                    d3 = _mm_xor_si128(
                        d3,
                        _mm_loadu_si128(row_b.add((3 ^ $bh) * 16) as *const __m128i),
                    );
                }};
            }

            step_b!(1, 0, true);
            step_b!(2, 1, false);
            step_b!(3, 1, true);
            step_b!(4, 2, false);
            step_b!(5, 2, true);
            step_b!(6, 3, false);
            step_b!(7, 3, true);

            core::mem::transmute::<[__m128i; 4], [u64; 8]>([d0, d1, d2, d3])
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "sse2")]
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn apply_x86_unchecked(&self, bytes: &[u8], out: &mut [F8]) {
        use core::arch::x86_64::*;
        assert_eq!(bytes.len(), self.n_chunks);
        assert_eq!(out.len(), self.ell);
        let base = self.data_ptr();
        let out_ptr = out.as_mut_ptr() as *mut u8;

        if self.ell == 64 && self.n_chunks == 8 {
            unsafe {
                let words = self.apply_x86_u64x8_unchecked(bytes.as_ptr());
                core::ptr::copy_nonoverlapping(words.as_ptr() as *const u8, out_ptr, 64);
            }
            return;
        }

        let n128 = self.ell / 16;
        unsafe {
            let row0 = base.add(bytes[0] as usize * self.ell);
            for c in 0..n128 {
                let v = _mm_loadu_si128(row0.add(c * 16) as *const __m128i);
                _mm_storeu_si128(out_ptr.add(c * 16) as *mut __m128i, v);
            }

            for b in 1..self.n_chunks {
                let b_high = b >> 1;
                let b_odd = (b & 1) != 0;
                let row_b = base.add(bytes[b] as usize * self.ell);
                if b_odd {
                    for c in 0..n128 {
                        let sc = c ^ b_high;
                        let v = _mm_loadu_si128(row_b.add(sc * 16) as *const __m128i);

                        let v_swapped = _mm_shuffle_epi32::<0x4E>(v);
                        let dst = out_ptr.add(c * 16) as *mut __m128i;
                        _mm_storeu_si128(dst, _mm_xor_si128(_mm_loadu_si128(dst), v_swapped));
                    }
                } else {
                    for c in 0..n128 {
                        let sc = c ^ b_high;
                        let v = _mm_loadu_si128(row_b.add(sc * 16) as *const __m128i);
                        let dst = out_ptr.add(c * 16) as *mut __m128i;
                        _mm_storeu_si128(dst, _mm_xor_si128(_mm_loadu_si128(dst), v));
                    }
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f")]
    pub unsafe fn apply_x86_avx512_unchecked(&self, bytes: &[u8], out: &mut [F8]) {
        assert_eq!(bytes.len(), self.n_chunks);
        assert_eq!(out.len(), self.ell);
        debug_assert_eq!(self.ell, 64, "avx512 apply is specialized for ell = 64");

        unsafe {
            use core::arch::x86_64::*;
            let acc = self.apply_x86_avx512_register_unchecked(bytes.as_ptr());
            _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, acc);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx512f")]
    pub(crate) unsafe fn apply_x86_avx512_register_unchecked(
        &self,
        bytes: *const u8,
    ) -> core::arch::x86_64::__m512i {
        use core::arch::x86_64::*;
        debug_assert_eq!(self.ell, 64, "avx512 apply is specialized for ell = 64");
        debug_assert_eq!(self.n_chunks, 8, "avx512 apply expects eight input bytes");
        let base = self.data_ptr();

        unsafe {
            let row0 = base.add(*bytes as usize * self.ell);
            let mut acc = _mm512_loadu_si512(row0 as *const __m512i);
            for b in 1..8 {
                let row_b = base.add(*bytes.add(b) as usize * self.ell);
                let v = _mm512_loadu_si512(row_b as *const __m512i);

                let perm = match b >> 1 {
                    0 => v,
                    1 => _mm512_shuffle_i64x2::<0xB1>(v, v),
                    2 => _mm512_shuffle_i64x2::<0x4E>(v, v),
                    _ => _mm512_shuffle_i64x2::<0x1B>(v, v),
                };

                let perm = if b & 1 != 0 {
                    _mm512_shuffle_epi32::<0x4E>(perm)
                } else {
                    perm
                };
                acc = _mm512_xor_si512(acc, perm);
            }
            acc
        }
    }

    pub fn apply_triple(
        &self,
        a_bytes: &[u8],
        a_out: &mut [F8],
        b_bytes: &[u8],
        b_out: &mut [F8],
        c_bytes: &[u8],
        c_out: &mut [F8],
    ) {
        self.apply(a_bytes, a_out);
        self.apply(b_bytes, b_out);
        self.apply(c_bytes, c_out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
    }

    fn naive_apply(ntt_s: &AdditiveNttGf8, ntt_l: &AdditiveNttGf8, bytes: &[u8]) -> Vec<F8> {
        let ell = 1usize << ntt_s.k();
        assert_eq!(bytes.len(), ell / 8);
        let mut v = vec![F8::ZERO; ell];
        for (b, &byte) in bytes.iter().enumerate() {
            for t in 0..8 {
                if (byte >> t) & 1 != 0 {
                    v[8 * b + t] = F8::ONE;
                }
            }
        }
        ntt_s.inverse(&mut v);
        ntt_l.forward(&mut v);
        v
    }

    #[test]
    fn table_storage_is_cache_line_aligned() {
        for k in 3..=7 {
            let ntt_s = AdditiveNttGf8::new(k, F8::ZERO);
            let ntt_l = AdditiveNttGf8::new(k, F8(1u8 << k));
            let table = InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);
            assert_eq!(table.data_ptr() as usize % 64, 0, "k={k}");
        }
    }

    #[test]
    fn matches_naive_k3() {
        let ntt_s = AdditiveNttGf8::new(3, F8::ZERO);
        let ntt_l = AdditiveNttGf8::new(3, F8(1 << 3));
        let table = InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);
        assert_eq!(table.ell, 8);
        assert_eq!(table.n_chunks, 1);

        let mut rng = Rng::new(100);
        let mut out = vec![F8::ZERO; 8];
        for _ in 0..64 {
            let bytes = [(rng.next_u64() & 0xff) as u8];
            table.apply(&bytes, &mut out);
            let expected = naive_apply(&ntt_s, &ntt_l, &bytes);
            assert_eq!(out, expected, "byte={:02x}", bytes[0]);
        }
    }

    #[test]
    fn matches_naive_k4() {
        let ntt_s = AdditiveNttGf8::new(4, F8::ZERO);
        let ntt_l = AdditiveNttGf8::new(4, F8(1 << 4));
        let table = InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);
        assert_eq!(table.ell, 16);
        assert_eq!(table.n_chunks, 2);

        let mut rng = Rng::new(101);
        let mut out = vec![F8::ZERO; 16];
        for _ in 0..64 {
            let bytes: [u8; 2] = [(rng.next_u64() & 0xff) as u8, (rng.next_u64() & 0xff) as u8];
            table.apply(&bytes, &mut out);
            let expected = naive_apply(&ntt_s, &ntt_l, &bytes);
            assert_eq!(out, expected, "bytes={:02x?}", bytes);
        }
    }

    #[test]
    fn matches_naive_k6_protocol_size() {
        let ntt_s = AdditiveNttGf8::new(6, F8::ZERO);
        let ntt_l = AdditiveNttGf8::new(6, F8(1 << 6));
        let table = InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);
        assert_eq!(table.ell, 64);
        assert_eq!(table.n_chunks, 8);

        let mut rng = Rng::new(102);
        let mut out = vec![F8::ZERO; 64];
        for _ in 0..16 {
            let bytes: Vec<u8> = (0..8).map(|_| (rng.next_u64() & 0xff) as u8).collect();
            table.apply(&bytes, &mut out);
            let expected = naive_apply(&ntt_s, &ntt_l, &bytes);
            assert_eq!(out, expected, "bytes={:02x?}", bytes);
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn apply_neon_matches_apply_scalar() {
        for &k in &[4usize, 5, 6] {
            let ntt_s = AdditiveNttGf8::new(k, F8::ZERO);
            let ntt_l = AdditiveNttGf8::new(k, F8(1u8 << k));
            let table = InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);
            let n_chunks = table.n_chunks;
            let ell = table.ell;

            let mut rng = Rng::new(100 + k as u64);
            for _ in 0..32 {
                let bytes: Vec<u8> = (0..n_chunks)
                    .map(|_| (rng.next_u64() & 0xff) as u8)
                    .collect();
                let mut out_scalar = vec![F8::ZERO; ell];
                let mut out_neon = vec![F8::ZERO; ell];
                table.apply_scalar(&bytes, &mut out_scalar);

                unsafe { table.apply_neon_unchecked(&bytes, &mut out_neon) };
                assert_eq!(
                    out_scalar, out_neon,
                    "scalar/neon apply disagree at k={k}, bytes={:02x?}",
                    bytes
                );
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn apply_x86_matches_apply_scalar() {
        for &k in &[4usize, 5, 6] {
            let ntt_s = AdditiveNttGf8::new(k, F8::ZERO);
            let ntt_l = AdditiveNttGf8::new(k, F8(1u8 << k));
            let table = InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);
            let n_chunks = table.n_chunks;
            let ell = table.ell;

            let mut rng = Rng::new(100 + k as u64);
            for _ in 0..32 {
                let bytes: Vec<u8> = (0..n_chunks)
                    .map(|_| (rng.next_u64() & 0xff) as u8)
                    .collect();
                let mut out_scalar = vec![F8::ZERO; ell];
                let mut out_x86 = vec![F8::ZERO; ell];
                table.apply_scalar(&bytes, &mut out_scalar);

                unsafe { table.apply_x86_unchecked(&bytes, &mut out_x86) };
                assert_eq!(
                    out_scalar, out_x86,
                    "scalar/x86 apply disagree at k={k}, bytes={:02x?}",
                    bytes
                );

                #[cfg(target_feature = "avx512f")]
                if ell == 64 {
                    let mut out_avx = vec![F8::ZERO; ell];

                    unsafe { table.apply_x86_avx512_unchecked(&bytes, &mut out_avx) };
                    assert_eq!(
                        out_scalar, out_avx,
                        "scalar/avx512 apply disagree at k={k}, bytes={:02x?}",
                        bytes
                    );
                }
            }
        }
    }

    #[test]
    fn apply_triple_matches_three_singles() {
        let ntt_s = AdditiveNttGf8::new(5, F8::ZERO);
        let ntt_l = AdditiveNttGf8::new(5, F8(1 << 5));
        let table = InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l);

        let mut rng = Rng::new(103);
        let nc = table.n_chunks;
        let ell = table.ell;
        let ab: Vec<u8> = (0..nc).map(|_| (rng.next_u64() & 0xff) as u8).collect();
        let bb: Vec<u8> = (0..nc).map(|_| (rng.next_u64() & 0xff) as u8).collect();
        let cb: Vec<u8> = (0..nc).map(|_| (rng.next_u64() & 0xff) as u8).collect();

        let mut a1 = vec![F8::ZERO; ell];
        let mut b1 = vec![F8::ZERO; ell];
        let mut c1 = vec![F8::ZERO; ell];
        table.apply(&ab, &mut a1);
        table.apply(&bb, &mut b1);
        table.apply(&cb, &mut c1);

        let mut a2 = vec![F8::ZERO; ell];
        let mut b2 = vec![F8::ZERO; ell];
        let mut c2 = vec![F8::ZERO; ell];
        table.apply_triple(&ab, &mut a2, &bb, &mut b2, &cb, &mut c2);

        assert_eq!(a1, a2);
        assert_eq!(b1, b2);
        assert_eq!(c1, c2);
    }
}
