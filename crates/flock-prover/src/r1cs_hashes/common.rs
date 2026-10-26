use std::sync::OnceLock;

use flock_core::bits::transpose_8_u64s_to_64_bytes;
use flock_core::field::F128;
use flock_core::r1cs::{BlockR1cs, SparseBinaryMatrix, WitnessLayout};

#[inline(always)]
pub(crate) fn or_u32_at_bit(buf: &mut [u64], bit_off: usize, val: u32) {
    let u64_idx = bit_off >> 6;
    let shift = (bit_off & 63) as u64;
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    unsafe {
        buf[u64_idx] |= flock_core::gf2_kernels::gf2_kern_or_u32_lo(val as u64, shift);
        if shift > 32 {
            buf[u64_idx + 1] |= flock_core::gf2_kernels::gf2_kern_or_u32_hi(val as u64, shift);
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        buf[u64_idx] |= (val as u64) << shift;
        if shift > 32 {
            buf[u64_idx + 1] |= (val as u64) >> (64 - shift);
        }
    }
}

#[inline(always)]
pub(crate) fn or_bit_at(buf: &mut [u64], bit_off: usize) {
    buf[bit_off >> 6] |= 1u64 << (bit_off & 63);
}

pub(crate) struct BitRecord<const NW: usize> {
    w: [u64; NW],
}

impl<const NW: usize> BitRecord<NW> {
    #[inline(always)]
    pub(crate) fn new() -> Self {
        Self { w: [0u64; NW] }
    }

    #[inline(always)]
    pub(crate) fn push<const POS: usize>(&mut self, val: u32) {
        let v = val as u64;
        let idx = POS >> 6;
        let s = POS & 63;
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        unsafe {
            self.w[idx] |= flock_core::gf2_kernels::gf2_kern_or_u32_lo(v, s as u64);
            if s > 32 {
                self.w[idx + 1] |= flock_core::gf2_kernels::gf2_kern_or_u32_hi(v, s as u64);
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            self.w[idx] |= v << s;
            if s > 32 {
                self.w[idx + 1] |= v >> (64 - s);
            }
        }
    }

    #[inline(always)]
    pub(crate) fn flush(&self, buf: &mut [u64], base_bit: usize) {
        let bi = base_bit >> 6;
        let s = base_bit & 63;
        let mut spill = 0u64;
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        unsafe {
            let su = s as u64;
            for j in 0..NW {
                let wj = self.w[j];
                buf[bi + j] |=
                    flock_core::gf2_kernels::gf2_kern_bitrecord_flush_word(wj, spill, su);
                spill = flock_core::gf2_kernels::gf2_kern_bitrecord_flush_spill(wj, su);
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            for j in 0..NW {
                buf[bi + j] |= (self.w[j] << s) | spill;

                spill = (self.w[j] >> 1) >> (63 - s);
            }
        }
        buf[bi + NW] |= spill;
    }
}

#[inline(always)]
pub(crate) fn add_carry_parts(x: u32, y: u32) -> (u32, u32, u32, u32) {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    unsafe {
        let xu = x as u64;
        let yu = y as u64;
        let sc = flock_core::gf2_kernels::gf2_kern_blake3_add_sum_carry(xu, yu);
        let lr = flock_core::gf2_kernels::gf2_kern_blake3_add_lr(xu, yu);
        (sc as u32, lr as u32, (lr >> 32) as u32, (sc >> 32) as u32)
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let sum = x.wrapping_add(y);
        let cin = sum ^ x ^ y;
        const MASK_LO31: u32 = 0x7FFF_FFFF;
        let left = (x ^ cin) & MASK_LO31;
        let right = (y ^ cin) & MASK_LO31;
        let carry_aux = left & right;
        (sum, left, right, carry_aux)
    }
}

pub(crate) fn empty_matrix(k: usize) -> SparseBinaryMatrix {
    SparseBinaryMatrix {
        num_rows: k,
        num_cols: k,
        rows: vec![Vec::new(); k],
    }
}

pub(crate) fn identity(k: usize) -> SparseBinaryMatrix {
    SparseBinaryMatrix {
        num_rows: k,
        num_cols: k,
        rows: (0..k).map(|i| vec![i]).collect(),
    }
}

pub(crate) fn build_block_r1cs_empty_stub(
    n_blocks_log: usize,
    k_log: usize,
    k_skip: usize,
    useful_bits: usize,
) -> BlockR1cs {
    let k = 1usize << k_log;

    build_block_r1cs_with_matrices(
        n_blocks_log,
        k_log,
        k_skip,
        useful_bits,
        empty_matrix(k),
        empty_matrix(k),
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_block_r1cs_with_matrices(
    n_blocks_log: usize,
    k_log: usize,
    k_skip: usize,
    useful_bits: usize,
    a_0: SparseBinaryMatrix,
    b_0: SparseBinaryMatrix,
    const_pin: Option<usize>,
) -> BlockR1cs {
    assert!(
        n_blocks_log >= 3,
        "lincheck needs n_outer ≥ 8 — pick n_blocks_log ≥ 3"
    );
    let k = 1usize << k_log;
    assert!(
        useful_bits <= k,
        "useful_bits ({useful_bits}) must be ≤ 2^k_log ({k})"
    );
    BlockR1cs {
        m: k_log + n_blocks_log,
        k_log,
        k_skip,
        useful_bits,
        a_0,
        b_0,
        c_0: identity(k),
        layout: WitnessLayout::RowMajor,
        const_pin,
        digest_cache: OnceLock::new(),
        csc_cache: OnceLock::new(),
    }
}

pub(crate) fn drive_witness_packed_and_lincheck<S: Sync, F>(
    initial_states: &[S],
    padding: Option<&S>,
    n_blocks_log: usize,
    k_log: usize,
    per_block: F,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>)
where
    F: Fn(&S, &mut [u64], &mut [u64], &mut [u64]) + Sync,
{
    drive_witness_packed_and_lincheck_padded(
        initial_states,
        padding,
        n_blocks_log,
        k_log,
        1usize << k_log,
        per_block,
    )
}

pub(crate) fn drive_witness_packed_and_lincheck_padded<S: Sync, F>(
    initial_states: &[S],
    padding: Option<&S>,
    n_blocks_log: usize,
    k_log: usize,
    useful_bits: usize,
    per_block: F,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>)
where
    F: Fn(&S, &mut [u64], &mut [u64], &mut [u64]) + Sync,
{
    drive_witness_packed_and_lincheck_padded_impl::<true, S, F>(
        initial_states,
        padding,
        n_blocks_log,
        k_log,
        useful_bits,
        per_block,
    )
}

pub(crate) fn drive_witness_packed_and_lincheck_overwrite<S: Sync, F>(
    initial_states: &[S],
    padding: Option<&S>,
    n_blocks_log: usize,
    k_log: usize,
    useful_bits: usize,
    per_block: F,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>)
where
    F: Fn(&S, &mut [u64], &mut [u64], &mut [u64]) + Sync,
{
    drive_witness_packed_and_lincheck_padded_impl::<false, S, F>(
        initial_states,
        padding,
        n_blocks_log,
        k_log,
        useful_bits,
        per_block,
    )
}

pub(crate) fn drive_witness_packed_and_lincheck_overwrite_by_index<F, P>(
    n_blocks: usize,
    n_blocks_log: usize,
    k_log: usize,
    useful_bits: usize,
    per_block_idx: F,
    per_padding: Option<P>,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>)
where
    F: Fn(usize, &mut [u64], &mut [u64], &mut [u64]) + Sync,
    P: Fn(&mut [u64], &mut [u64], &mut [u64]) + Sync,
{
    use rayon::prelude::*;

    let k = 1usize << k_log;
    let f128_per_block = k / 128;
    let u64_per_block = k / 64;
    let useful_words = useful_bits.div_ceil(64).min(u64_per_block);
    let useful_f128 = useful_words.div_ceil(2).min(f128_per_block);
    let n_total = 1usize << n_blocks_log;
    assert!(
        n_blocks <= n_total,
        "{n_blocks} blocks > 2^{n_blocks_log} = {n_total} slots"
    );
    assert!(
        n_total >= 8 && n_total.is_multiple_of(8),
        "lincheck stripe layout requires n_total >= 8 and divisible by 8"
    );

    let total_f128 = n_total * f128_per_block;
    let mut z = flock_core::scratch::take_f128(total_f128);
    let mut a = flock_core::scratch::take_f128(total_f128);
    let mut b = flock_core::scratch::take_f128(total_f128);
    let mut z_lincheck = flock_core::scratch::take_u8((n_total / 8) * k);

    #[cfg(target_arch = "x86_64")]
    if total_f128 >= (1 << 20) {
        let z_stride = u64_per_block + 8;
        z.par_chunks_mut(8 * f128_per_block)
            .zip(a.par_chunks_mut(8 * f128_per_block))
            .zip(b.par_chunks_mut(8 * f128_per_block))
            .zip(z_lincheck.par_chunks_mut(k))
            .enumerate()
            .for_each_init(
                || {
                    (
                        vec![0u64; 8 * z_stride],
                        vec![0u64; u64_per_block],
                        vec![0u64; u64_per_block],
                    )
                },
                |(z_local, a_local, b_local), (g, (((z_grp, a_grp), b_grp), stripe))| {
                    for k_in in 0..8 {
                        let global_idx = 8 * g + k_in;
                        let z_slice =
                            &mut z_local[k_in * z_stride..k_in * z_stride + u64_per_block];
                        let z_dst = &mut z_grp[k_in * f128_per_block..(k_in + 1) * f128_per_block];
                        let a_dst = &mut a_grp[k_in * f128_per_block..(k_in + 1) * f128_per_block];
                        let b_dst = &mut b_grp[k_in * f128_per_block..(k_in + 1) * f128_per_block];

                        let wrote = if global_idx < n_blocks {
                            per_block_idx(global_idx, z_slice, a_local, b_local);
                            true
                        } else if let Some(ref pad_fn) = per_padding {
                            pad_fn(z_slice, a_local, b_local);
                            true
                        } else {
                            z_slice[..useful_words].fill(0);
                            false
                        };
                        if wrote {
                            unsafe {
                                use core::arch::x86_64::*;
                                let zp = z_dst.as_mut_ptr() as *mut __m128i;
                                let ap = a_dst.as_mut_ptr() as *mut __m128i;
                                let bp = b_dst.as_mut_ptr() as *mut __m128i;
                                let zs = z_slice.as_ptr() as *const __m128i;
                                let as_ = a_local.as_ptr() as *const __m128i;
                                let bs = b_local.as_ptr() as *const __m128i;
                                for w in 0..useful_f128 {
                                    _mm_stream_si128(zp.add(w), _mm_loadu_si128(zs.add(w)));
                                    _mm_stream_si128(ap.add(w), _mm_loadu_si128(as_.add(w)));
                                    _mm_stream_si128(bp.add(w), _mm_loadu_si128(bs.add(w)));
                                }
                                let zero = _mm_setzero_si128();
                                for w in useful_f128..f128_per_block {
                                    _mm_stream_si128(zp.add(w), zero);
                                    _mm_stream_si128(ap.add(w), zero);
                                    _mm_stream_si128(bp.add(w), zero);
                                }
                            }
                        } else {
                            unsafe {
                                use core::arch::x86_64::*;
                                let zero = _mm_setzero_si128();
                                let zp = z_dst.as_mut_ptr() as *mut __m128i;
                                let ap = a_dst.as_mut_ptr() as *mut __m128i;
                                let bp = b_dst.as_mut_ptr() as *mut __m128i;
                                for w in 0..f128_per_block {
                                    _mm_stream_si128(zp.add(w), zero);
                                    _mm_stream_si128(ap.add(w), zero);
                                    _mm_stream_si128(bp.add(w), zero);
                                }
                            }
                        }
                    }

                    if (stripe.as_ptr() as usize) & 15 == 0 {
                        #[cfg(target_feature = "avx2")]
                        unsafe {
                            use core::arch::x86_64::*;
                            let sp = stripe.as_mut_ptr() as *mut __m128i;
                            let m1 = _mm256_set1_epi64x(0x00AA00AA00AA00AAu64 as i64);
                            let m2 = _mm256_set1_epi64x(0x0000CCCC0000CCCCu64 as i64);
                            let m3 = _mm256_set1_epi64x(0x00000000F0F0F0F0u64 as i64);
                            for i in 0..useful_words {
                                let p01 = _mm_unpacklo_epi8(
                                    _mm_cvtsi64_si128(z_local[i] as i64),
                                    _mm_cvtsi64_si128(z_local[z_stride + i] as i64),
                                );
                                let p23 = _mm_unpacklo_epi8(
                                    _mm_cvtsi64_si128(z_local[2 * z_stride + i] as i64),
                                    _mm_cvtsi64_si128(z_local[3 * z_stride + i] as i64),
                                );
                                let p45 = _mm_unpacklo_epi8(
                                    _mm_cvtsi64_si128(z_local[4 * z_stride + i] as i64),
                                    _mm_cvtsi64_si128(z_local[5 * z_stride + i] as i64),
                                );
                                let p67 = _mm_unpacklo_epi8(
                                    _mm_cvtsi64_si128(z_local[6 * z_stride + i] as i64),
                                    _mm_cvtsi64_si128(z_local[7 * z_stride + i] as i64),
                                );
                                let q0123_lo = _mm_unpacklo_epi16(p01, p23);
                                let q0123_hi = _mm_unpackhi_epi16(p01, p23);
                                let q4567_lo = _mm_unpacklo_epi16(p45, p67);
                                let q4567_hi = _mm_unpackhi_epi16(p45, p67);
                                let c01 = _mm_unpacklo_epi32(q0123_lo, q4567_lo);
                                let c23 = _mm_unpackhi_epi32(q0123_lo, q4567_lo);
                                let c45 = _mm_unpacklo_epi32(q0123_hi, q4567_hi);
                                let c67 = _mm_unpackhi_epi32(q0123_hi, q4567_hi);
                                let mut y0 = _mm256_set_m128i(c23, c01);
                                let mut y1 = _mm256_set_m128i(c67, c45);
                                let t0 = _mm256_and_si256(
                                    _mm256_xor_si256(y0, _mm256_srli_epi64::<7>(y0)),
                                    m1,
                                );
                                let t1 = _mm256_and_si256(
                                    _mm256_xor_si256(y1, _mm256_srli_epi64::<7>(y1)),
                                    m1,
                                );
                                y0 = _mm256_xor_si256(
                                    y0,
                                    _mm256_xor_si256(t0, _mm256_slli_epi64::<7>(t0)),
                                );
                                y1 = _mm256_xor_si256(
                                    y1,
                                    _mm256_xor_si256(t1, _mm256_slli_epi64::<7>(t1)),
                                );
                                let t0 = _mm256_and_si256(
                                    _mm256_xor_si256(y0, _mm256_srli_epi64::<14>(y0)),
                                    m2,
                                );
                                let t1 = _mm256_and_si256(
                                    _mm256_xor_si256(y1, _mm256_srli_epi64::<14>(y1)),
                                    m2,
                                );
                                y0 = _mm256_xor_si256(
                                    y0,
                                    _mm256_xor_si256(t0, _mm256_slli_epi64::<14>(t0)),
                                );
                                y1 = _mm256_xor_si256(
                                    y1,
                                    _mm256_xor_si256(t1, _mm256_slli_epi64::<14>(t1)),
                                );
                                let t0 = _mm256_and_si256(
                                    _mm256_xor_si256(y0, _mm256_srli_epi64::<28>(y0)),
                                    m3,
                                );
                                let t1 = _mm256_and_si256(
                                    _mm256_xor_si256(y1, _mm256_srli_epi64::<28>(y1)),
                                    m3,
                                );
                                y0 = _mm256_xor_si256(
                                    y0,
                                    _mm256_xor_si256(t0, _mm256_slli_epi64::<28>(t0)),
                                );
                                y1 = _mm256_xor_si256(
                                    y1,
                                    _mm256_xor_si256(t1, _mm256_slli_epi64::<28>(t1)),
                                );
                                let dst = sp.add(i * 4);
                                _mm_stream_si128(dst, _mm256_castsi256_si128(y0));
                                _mm_stream_si128(dst.add(1), _mm256_extracti128_si256::<1>(y0));
                                _mm_stream_si128(dst.add(2), _mm256_castsi256_si128(y1));
                                _mm_stream_si128(dst.add(3), _mm256_extracti128_si256::<1>(y1));
                            }
                            let zero = _mm_setzero_si128();
                            for i in useful_words..u64_per_block {
                                let dst = sp.add(i * 4);
                                _mm_stream_si128(dst, zero);
                                _mm_stream_si128(dst.add(1), zero);
                                _mm_stream_si128(dst.add(2), zero);
                                _mm_stream_si128(dst.add(3), zero);
                            }
                            _mm_sfence();
                        }
                        #[cfg(not(target_feature = "avx2"))]
                        {
                            for i in 0..useful_words {
                                let lanes: [u64; 8] = [
                                    z_local[i],
                                    z_local[z_stride + i],
                                    z_local[2 * z_stride + i],
                                    z_local[3 * z_stride + i],
                                    z_local[4 * z_stride + i],
                                    z_local[5 * z_stride + i],
                                    z_local[6 * z_stride + i],
                                    z_local[7 * z_stride + i],
                                ];
                                transpose_8_u64s_to_64_bytes(
                                    &lanes,
                                    &mut stripe[i * 64..i * 64 + 64],
                                );
                            }
                            if useful_words * 64 < stripe.len() {
                                stripe[useful_words * 64..].fill(0);
                            }
                        }
                    } else {
                        for i in 0..useful_words {
                            let lanes: [u64; 8] = [
                                z_local[i],
                                z_local[z_stride + i],
                                z_local[2 * z_stride + i],
                                z_local[3 * z_stride + i],
                                z_local[4 * z_stride + i],
                                z_local[5 * z_stride + i],
                                z_local[6 * z_stride + i],
                                z_local[7 * z_stride + i],
                            ];
                            transpose_8_u64s_to_64_bytes(&lanes, &mut stripe[i * 64..i * 64 + 64]);
                        }
                        if useful_words * 64 < stripe.len() {
                            stripe[useful_words * 64..].fill(0);
                        }
                    }
                },
            );
        return (z, a, b, z_lincheck);
    }

    z.par_chunks_mut(8 * f128_per_block)
        .zip(a.par_chunks_mut(8 * f128_per_block))
        .zip(b.par_chunks_mut(8 * f128_per_block))
        .zip(z_lincheck.par_chunks_mut(k))
        .enumerate()
        .for_each(|(g, (((z_grp, a_grp), b_grp), stripe))| {
            for k_in in 0..8 {
                let global_idx = 8 * g + k_in;
                let z_chunk = &mut z_grp[k_in * f128_per_block..(k_in + 1) * f128_per_block];
                let a_chunk = &mut a_grp[k_in * f128_per_block..(k_in + 1) * f128_per_block];
                let b_chunk = &mut b_grp[k_in * f128_per_block..(k_in + 1) * f128_per_block];
                if useful_f128 < f128_per_block {
                    z_chunk[useful_f128..].fill(F128::ZERO);
                    a_chunk[useful_f128..].fill(F128::ZERO);
                    b_chunk[useful_f128..].fill(F128::ZERO);
                }
                let z_u64: &mut [u64] = unsafe {
                    std::slice::from_raw_parts_mut(
                        z_chunk.as_mut_ptr() as *mut u64,
                        z_chunk.len() * 2,
                    )
                };
                let a_u64: &mut [u64] = unsafe {
                    std::slice::from_raw_parts_mut(
                        a_chunk.as_mut_ptr() as *mut u64,
                        a_chunk.len() * 2,
                    )
                };
                let b_u64: &mut [u64] = unsafe {
                    std::slice::from_raw_parts_mut(
                        b_chunk.as_mut_ptr() as *mut u64,
                        b_chunk.len() * 2,
                    )
                };
                if global_idx < n_blocks {
                    per_block_idx(global_idx, z_u64, a_u64, b_u64);
                } else if let Some(ref pad_fn) = per_padding {
                    pad_fn(z_u64, a_u64, b_u64);
                } else {
                    z_chunk.fill(F128::ZERO);
                    a_chunk.fill(F128::ZERO);
                    b_chunk.fill(F128::ZERO);
                }
            }

            let z_u64_all: &[u64] = unsafe {
                std::slice::from_raw_parts(z_grp.as_ptr() as *const u64, z_grp.len() * 2)
            };
            for i in 0..useful_words {
                let lanes: [u64; 8] = [
                    z_u64_all[0 * u64_per_block + i],
                    z_u64_all[u64_per_block + i],
                    z_u64_all[2 * u64_per_block + i],
                    z_u64_all[3 * u64_per_block + i],
                    z_u64_all[4 * u64_per_block + i],
                    z_u64_all[5 * u64_per_block + i],
                    z_u64_all[6 * u64_per_block + i],
                    z_u64_all[7 * u64_per_block + i],
                ];
                transpose_8_u64s_to_64_bytes(&lanes, &mut stripe[i * 64..i * 64 + 64]);
            }
            if useful_words * 64 < stripe.len() {
                stripe[useful_words * 64..].fill(0);
            }
        });

    (z, a, b, z_lincheck)
}

fn drive_witness_packed_and_lincheck_padded_impl<const ZERO_INIT: bool, S: Sync, F>(
    initial_states: &[S],
    padding: Option<&S>,
    n_blocks_log: usize,
    k_log: usize,
    useful_bits: usize,
    per_block: F,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>)
where
    F: Fn(&S, &mut [u64], &mut [u64], &mut [u64]) + Sync,
{
    use rayon::prelude::*;

    let k = 1usize << k_log;
    let f128_per_block = k / 128;
    let u64_per_block = k / 64;
    let useful_words = useful_bits.div_ceil(64).min(u64_per_block);
    let useful_f128 = useful_words.div_ceil(2).min(f128_per_block);
    let n_total = 1usize << n_blocks_log;
    let n_blocks = initial_states.len();
    assert!(
        n_blocks <= n_total,
        "{n_blocks} blocks > 2^{n_blocks_log} = {n_total} slots"
    );
    assert!(
        n_total >= 8 && n_total.is_multiple_of(8),
        "lincheck stripe layout requires n_total ≥ 8 and divisible by 8"
    );

    let total_f128 = n_total * f128_per_block;
    let mut z = flock_core::scratch::take_f128(total_f128);
    let mut a = flock_core::scratch::take_f128(total_f128);
    let mut b = flock_core::scratch::take_f128(total_f128);
    let mut z_lincheck = flock_core::scratch::take_u8((n_total / 8) * k);

    #[cfg(target_arch = "x86_64")]
    if total_f128 >= (1 << 20) {
        let z_stride = u64_per_block + 8;
        z.par_chunks_mut(8 * f128_per_block)
            .zip(a.par_chunks_mut(8 * f128_per_block))
            .zip(b.par_chunks_mut(8 * f128_per_block))
            .zip(z_lincheck.par_chunks_mut(k))
            .enumerate()
            .for_each_init(
                || {
                    (
                        vec![0u64; 8 * z_stride],
                        vec![0u64; u64_per_block],
                        vec![0u64; u64_per_block],
                    )
                },
                |(z_local, a_local, b_local), (g, (((z_grp, a_grp), b_grp), stripe))| {
                    for k_in in 0..8 {
                        let global_idx = 8 * g + k_in;
                        let z_slice =
                            &mut z_local[k_in * z_stride..k_in * z_stride + u64_per_block];
                        let z_dst = &mut z_grp[k_in * f128_per_block..(k_in + 1) * f128_per_block];
                        let a_dst = &mut a_grp[k_in * f128_per_block..(k_in + 1) * f128_per_block];
                        let b_dst = &mut b_grp[k_in * f128_per_block..(k_in + 1) * f128_per_block];

                        let init_opt: Option<&S> = if global_idx < n_blocks {
                            Some(&initial_states[global_idx])
                        } else {
                            padding
                        };
                        if let Some(init) = init_opt {
                            if ZERO_INIT {
                                z_slice[..useful_words].fill(0);
                                a_local[..useful_words].fill(0);
                                b_local[..useful_words].fill(0);
                            }
                            per_block(init, z_slice, a_local, b_local);
                            unsafe {
                                use core::arch::x86_64::*;
                                let zp = z_dst.as_mut_ptr() as *mut __m128i;
                                let ap = a_dst.as_mut_ptr() as *mut __m128i;
                                let bp = b_dst.as_mut_ptr() as *mut __m128i;
                                let zs = z_slice.as_ptr() as *const __m128i;
                                let as_ = a_local.as_ptr() as *const __m128i;
                                let bs = b_local.as_ptr() as *const __m128i;
                                for w in 0..useful_f128 {
                                    _mm_stream_si128(zp.add(w), _mm_loadu_si128(zs.add(w)));
                                    _mm_stream_si128(ap.add(w), _mm_loadu_si128(as_.add(w)));
                                    _mm_stream_si128(bp.add(w), _mm_loadu_si128(bs.add(w)));
                                }
                                let zero = _mm_setzero_si128();
                                for w in useful_f128..f128_per_block {
                                    _mm_stream_si128(zp.add(w), zero);
                                    _mm_stream_si128(ap.add(w), zero);
                                    _mm_stream_si128(bp.add(w), zero);
                                }
                            }
                        } else {
                            unsafe {
                                use core::arch::x86_64::*;
                                let zero = _mm_setzero_si128();
                                let zp = z_dst.as_mut_ptr() as *mut __m128i;
                                let ap = a_dst.as_mut_ptr() as *mut __m128i;
                                let bp = b_dst.as_mut_ptr() as *mut __m128i;
                                for w in 0..f128_per_block {
                                    _mm_stream_si128(zp.add(w), zero);
                                    _mm_stream_si128(ap.add(w), zero);
                                    _mm_stream_si128(bp.add(w), zero);
                                }
                            }
                        }
                    }

                    if (stripe.as_ptr() as usize) & 15 == 0 {
                        #[cfg(target_feature = "avx2")]
                        unsafe {
                            use core::arch::x86_64::*;
                            let sp = stripe.as_mut_ptr() as *mut __m128i;
                            let m1 = _mm256_set1_epi64x(0x00AA00AA00AA00AAu64 as i64);
                            let m2 = _mm256_set1_epi64x(0x0000CCCC0000CCCCu64 as i64);
                            let m3 = _mm256_set1_epi64x(0x00000000F0F0F0F0u64 as i64);
                            for i in 0..useful_words {
                                let p01 = _mm_unpacklo_epi8(
                                    _mm_cvtsi64_si128(z_local[i] as i64),
                                    _mm_cvtsi64_si128(z_local[z_stride + i] as i64),
                                );
                                let p23 = _mm_unpacklo_epi8(
                                    _mm_cvtsi64_si128(z_local[2 * z_stride + i] as i64),
                                    _mm_cvtsi64_si128(z_local[3 * z_stride + i] as i64),
                                );
                                let p45 = _mm_unpacklo_epi8(
                                    _mm_cvtsi64_si128(z_local[4 * z_stride + i] as i64),
                                    _mm_cvtsi64_si128(z_local[5 * z_stride + i] as i64),
                                );
                                let p67 = _mm_unpacklo_epi8(
                                    _mm_cvtsi64_si128(z_local[6 * z_stride + i] as i64),
                                    _mm_cvtsi64_si128(z_local[7 * z_stride + i] as i64),
                                );
                                let q0123_lo = _mm_unpacklo_epi16(p01, p23);
                                let q0123_hi = _mm_unpackhi_epi16(p01, p23);
                                let q4567_lo = _mm_unpacklo_epi16(p45, p67);
                                let q4567_hi = _mm_unpackhi_epi16(p45, p67);
                                let c01 = _mm_unpacklo_epi32(q0123_lo, q4567_lo);
                                let c23 = _mm_unpackhi_epi32(q0123_lo, q4567_lo);
                                let c45 = _mm_unpacklo_epi32(q0123_hi, q4567_hi);
                                let c67 = _mm_unpackhi_epi32(q0123_hi, q4567_hi);
                                let mut y0 = _mm256_set_m128i(c23, c01);
                                let mut y1 = _mm256_set_m128i(c67, c45);
                                let t0 = _mm256_and_si256(
                                    _mm256_xor_si256(y0, _mm256_srli_epi64::<7>(y0)),
                                    m1,
                                );
                                let t1 = _mm256_and_si256(
                                    _mm256_xor_si256(y1, _mm256_srli_epi64::<7>(y1)),
                                    m1,
                                );
                                y0 = _mm256_xor_si256(
                                    y0,
                                    _mm256_xor_si256(t0, _mm256_slli_epi64::<7>(t0)),
                                );
                                y1 = _mm256_xor_si256(
                                    y1,
                                    _mm256_xor_si256(t1, _mm256_slli_epi64::<7>(t1)),
                                );
                                let t0 = _mm256_and_si256(
                                    _mm256_xor_si256(y0, _mm256_srli_epi64::<14>(y0)),
                                    m2,
                                );
                                let t1 = _mm256_and_si256(
                                    _mm256_xor_si256(y1, _mm256_srli_epi64::<14>(y1)),
                                    m2,
                                );
                                y0 = _mm256_xor_si256(
                                    y0,
                                    _mm256_xor_si256(t0, _mm256_slli_epi64::<14>(t0)),
                                );
                                y1 = _mm256_xor_si256(
                                    y1,
                                    _mm256_xor_si256(t1, _mm256_slli_epi64::<14>(t1)),
                                );
                                let t0 = _mm256_and_si256(
                                    _mm256_xor_si256(y0, _mm256_srli_epi64::<28>(y0)),
                                    m3,
                                );
                                let t1 = _mm256_and_si256(
                                    _mm256_xor_si256(y1, _mm256_srli_epi64::<28>(y1)),
                                    m3,
                                );
                                y0 = _mm256_xor_si256(
                                    y0,
                                    _mm256_xor_si256(t0, _mm256_slli_epi64::<28>(t0)),
                                );
                                y1 = _mm256_xor_si256(
                                    y1,
                                    _mm256_xor_si256(t1, _mm256_slli_epi64::<28>(t1)),
                                );
                                let dst = sp.add(i * 4);
                                _mm_stream_si128(dst, _mm256_castsi256_si128(y0));
                                _mm_stream_si128(dst.add(1), _mm256_extracti128_si256::<1>(y0));
                                _mm_stream_si128(dst.add(2), _mm256_castsi256_si128(y1));
                                _mm_stream_si128(dst.add(3), _mm256_extracti128_si256::<1>(y1));
                            }
                            let zero = _mm_setzero_si128();
                            for i in useful_words..u64_per_block {
                                let dst = sp.add(i * 4);
                                _mm_stream_si128(dst, zero);
                                _mm_stream_si128(dst.add(1), zero);
                                _mm_stream_si128(dst.add(2), zero);
                                _mm_stream_si128(dst.add(3), zero);
                            }
                            _mm_sfence();
                        }
                        #[cfg(not(target_feature = "avx2"))]
                        {
                            let mut stripe_local = [0u8; 64];
                            unsafe {
                                use core::arch::x86_64::*;
                                let sp = stripe.as_mut_ptr() as *mut __m128i;
                                for i in 0..useful_words {
                                    let lanes: [u64; 8] = [
                                        z_local[i],
                                        z_local[z_stride + i],
                                        z_local[2 * z_stride + i],
                                        z_local[3 * z_stride + i],
                                        z_local[4 * z_stride + i],
                                        z_local[5 * z_stride + i],
                                        z_local[6 * z_stride + i],
                                        z_local[7 * z_stride + i],
                                    ];
                                    transpose_8_u64s_to_64_bytes(&lanes, &mut stripe_local);
                                    let sl = stripe_local.as_ptr() as *const __m128i;
                                    let dst = sp.add(i * 4);
                                    _mm_stream_si128(dst, _mm_loadu_si128(sl));
                                    _mm_stream_si128(dst.add(1), _mm_loadu_si128(sl.add(1)));
                                    _mm_stream_si128(dst.add(2), _mm_loadu_si128(sl.add(2)));
                                    _mm_stream_si128(dst.add(3), _mm_loadu_si128(sl.add(3)));
                                }
                                let zero = _mm_setzero_si128();
                                for i in useful_words..u64_per_block {
                                    let dst = sp.add(i * 4);
                                    _mm_stream_si128(dst, zero);
                                    _mm_stream_si128(dst.add(1), zero);
                                    _mm_stream_si128(dst.add(2), zero);
                                    _mm_stream_si128(dst.add(3), zero);
                                }
                                _mm_sfence();
                            }
                        }
                    } else {
                        for i in 0..useful_words {
                            let lanes: [u64; 8] = [
                                z_local[i],
                                z_local[z_stride + i],
                                z_local[2 * z_stride + i],
                                z_local[3 * z_stride + i],
                                z_local[4 * z_stride + i],
                                z_local[5 * z_stride + i],
                                z_local[6 * z_stride + i],
                                z_local[7 * z_stride + i],
                            ];
                            transpose_8_u64s_to_64_bytes(&lanes, &mut stripe[i * 64..i * 64 + 64]);
                        }
                        if useful_words * 64 < stripe.len() {
                            stripe[useful_words * 64..].fill(0);
                        }
                    }
                },
            );
        return (z, a, b, z_lincheck);
    }

    z.par_chunks_mut(8 * f128_per_block)
        .zip(a.par_chunks_mut(8 * f128_per_block))
        .zip(b.par_chunks_mut(8 * f128_per_block))
        .zip(z_lincheck.par_chunks_mut(k))
        .enumerate()
        .for_each(|(g, (((z_grp, a_grp), b_grp), stripe))| {
            if ZERO_INIT {
                unsafe {
                    std::ptr::write_bytes(z_grp.as_mut_ptr(), 0, z_grp.len());
                    std::ptr::write_bytes(a_grp.as_mut_ptr(), 0, a_grp.len());
                    std::ptr::write_bytes(b_grp.as_mut_ptr(), 0, b_grp.len());
                }
            }
            for k_in in 0..8 {
                let global_idx = 8 * g + k_in;
                let z_chunk = &mut z_grp[k_in * f128_per_block..(k_in + 1) * f128_per_block];
                let a_chunk = &mut a_grp[k_in * f128_per_block..(k_in + 1) * f128_per_block];
                let b_chunk = &mut b_grp[k_in * f128_per_block..(k_in + 1) * f128_per_block];
                let init: &S = if global_idx < n_blocks {
                    &initial_states[global_idx]
                } else if let Some(p) = padding {
                    p
                } else {
                    if !ZERO_INIT {
                        z_chunk.fill(F128::ZERO);
                        a_chunk.fill(F128::ZERO);
                        b_chunk.fill(F128::ZERO);
                    }
                    continue;
                };
                if !ZERO_INIT && useful_f128 < f128_per_block {
                    z_chunk[useful_f128..].fill(F128::ZERO);
                    a_chunk[useful_f128..].fill(F128::ZERO);
                    b_chunk[useful_f128..].fill(F128::ZERO);
                }
                let z_u64: &mut [u64] = unsafe {
                    std::slice::from_raw_parts_mut(
                        z_chunk.as_mut_ptr() as *mut u64,
                        z_chunk.len() * 2,
                    )
                };
                let a_u64: &mut [u64] = unsafe {
                    std::slice::from_raw_parts_mut(
                        a_chunk.as_mut_ptr() as *mut u64,
                        a_chunk.len() * 2,
                    )
                };
                let b_u64: &mut [u64] = unsafe {
                    std::slice::from_raw_parts_mut(
                        b_chunk.as_mut_ptr() as *mut u64,
                        b_chunk.len() * 2,
                    )
                };
                per_block(init, z_u64, a_u64, b_u64);
            }

            let z_u64_all: &[u64] = unsafe {
                std::slice::from_raw_parts(z_grp.as_ptr() as *const u64, z_grp.len() * 2)
            };
            for i in 0..useful_words {
                let lanes: [u64; 8] = [
                    z_u64_all[0 * u64_per_block + i],
                    z_u64_all[u64_per_block + i],
                    z_u64_all[2 * u64_per_block + i],
                    z_u64_all[3 * u64_per_block + i],
                    z_u64_all[4 * u64_per_block + i],
                    z_u64_all[5 * u64_per_block + i],
                    z_u64_all[6 * u64_per_block + i],
                    z_u64_all[7 * u64_per_block + i],
                ];
                transpose_8_u64s_to_64_bytes(&lanes, &mut stripe[i * 64..i * 64 + 64]);
            }
            if useful_words * 64 < stripe.len() {
                stripe[useful_words * 64..].fill(0);
            }
        });

    (z, a, b, z_lincheck)
}

pub(crate) fn xor_dedup(mut v: Vec<usize>) -> Vec<usize> {
    v.sort();
    let mut out = Vec::with_capacity(v.len());
    let mut i = 0;
    while i < v.len() {
        let val = v[i];
        let mut count = 0;
        while i < v.len() && v[i] == val {
            count += 1;
            i += 1;
        }
        if count % 2 == 1 {
            out.push(val);
        }
    }
    out
}

pub(crate) const BM_V: usize = 8;

pub(crate) type BmRow = [u64; BM_V];

#[derive(Copy, Clone)]
pub(crate) struct SendPtr(pub *mut u64);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    pub(crate) fn get(self) -> *mut u64 {
        self.0
    }
}

#[inline(always)]
pub(crate) fn or_u32_row(rows: &mut [BmRow], bit: usize, vals: &[u32; BM_V]) {
    let w = bit >> 6;
    let s = bit & 63;
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    unsafe {
        let su = s as u64;
        for j in 0..BM_V {
            rows[w][j] |= flock_core::gf2_kernels::gf2_kern_or_u32_lo(vals[j] as u64, su);
        }
        if s > 32 {
            for j in 0..BM_V {
                rows[w + 1][j] |= flock_core::gf2_kernels::gf2_kern_or_u32_hi(vals[j] as u64, su);
            }
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        for j in 0..BM_V {
            rows[w][j] |= (vals[j] as u64) << s;
        }
        if s > 32 {
            for j in 0..BM_V {
                rows[w + 1][j] |= (vals[j] as u64) >> (64 - s);
            }
        }
    }
}

#[inline(always)]
pub(crate) fn or_bit_row(rows: &mut [BmRow], bit: usize) {
    let w = bit >> 6;
    let s = bit & 63;
    for j in 0..BM_V {
        rows[w][j] |= 1u64 << s;
    }
}

#[inline(always)]
pub(crate) unsafe fn nt_store_row(src: *const u64, dst: *mut u64) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        std::arch::asm!(
            "ldp {t0:q}, {t1:q}, [{s}]",
            "stnp {t0:q}, {t1:q}, [{d}]",
            "ldp {t0:q}, {t1:q}, [{s}, #32]",
            "stnp {t0:q}, {t1:q}, [{d}, #32]",
            "ldp {t0:q}, {t1:q}, [{s}, #64]",
            "stnp {t0:q}, {t1:q}, [{d}, #64]",
            "ldp {t0:q}, {t1:q}, [{s}, #96]",
            "stnp {t0:q}, {t1:q}, [{d}, #96]",
            s = in(reg) src, d = in(reg) dst,
            t0 = out(vreg) _, t1 = out(vreg) _,
            options(nostack),
        );
    }
    #[cfg(not(target_arch = "aarch64"))]
    unsafe {
        std::ptr::copy_nonoverlapping(src, dst, 2 * BM_V);
    }
}

#[inline]
pub(crate) unsafe fn flush_rows_nt(
    rows: &[BmRow],
    dest: *mut u64,
    o0: usize,
    n_log: usize,
    useful_chunks: usize,
) {
    debug_assert!(2 * useful_chunks <= rows.len());
    for c in 0..useful_chunks {
        let even = &rows[2 * c];
        let odd = &rows[2 * c + 1];
        let mut buf = [0u64; 2 * BM_V];
        for j in 0..BM_V {
            buf[2 * j] = even[j];
            buf[2 * j + 1] = odd[j];
        }
        unsafe {
            nt_store_row(buf.as_ptr(), dest.add(((c << n_log) + o0) * 2));
        }
    }
}

#[inline]
pub(crate) unsafe fn stripe_from_rows(
    rows: &[BmRow],
    stripe: *mut u8,
    o0: usize,
    u64_per_block: usize,
    useful_words: usize,
) {
    let base = (o0 / 8) * u64_per_block * 64;
    for (w, row) in rows.iter().enumerate().take(useful_words) {
        let out = unsafe { std::slice::from_raw_parts_mut(stripe.add(base + w * 64), 64) };
        transpose_8_u64s_to_64_bytes(row, out);
    }
    if useful_words < u64_per_block {
        let tail_len = (u64_per_block - useful_words) * 64;
        unsafe {
            std::ptr::write_bytes(stripe.add(base + useful_words * 64), 0, tail_len);
        }
    }
}

#[inline(always)]
pub(crate) fn add_carry_parts_v(
    x: &[u32; BM_V],
    y: &[u32; BM_V],
) -> ([u32; BM_V], [u32; BM_V], [u32; BM_V], [u32; BM_V]) {
    let mut sum = [0u32; BM_V];
    let mut left = [0u32; BM_V];
    let mut right = [0u32; BM_V];
    let mut carry = [0u32; BM_V];
    for j in 0..BM_V {
        let (s, l, r, c) = add_carry_parts(x[j], y[j]);
        sum[j] = s;
        left[j] = l;
        right[j] = r;
        carry[j] = c;
    }
    (sum, left, right, carry)
}

pub(crate) fn drive_witness_batch_major<S: Sync, F>(
    inputs: &[S],
    padding: &S,
    n_blocks_log: usize,
    k_log: usize,
    useful_bits: usize,
    per_group: F,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>)
where
    F: Fn([&S; BM_V], &mut [BmRow], &mut [BmRow], &mut [BmRow]) + Sync + Send,
{
    use rayon::prelude::*;

    let n_total = 1usize << n_blocks_log;
    assert!(inputs.len() <= n_total);
    assert!(n_total >= BM_V);
    let u64_per_block = (1usize << k_log) / 64;
    let useful_chunks = useful_bits.div_ceil(128);
    let useful_words = useful_bits.div_ceil(64);
    let total_f128 = n_total * (u64_per_block / 2);

    let mut z = flock_core::scratch::take_f128(total_f128);
    let mut a = flock_core::scratch::take_f128(total_f128);
    let mut b = flock_core::scratch::take_f128(total_f128);
    let stripe = flock_core::scratch::take_u8(n_total * u64_per_block * 8);

    let tail = useful_chunks << n_blocks_log;
    for buf in [&mut z, &mut a, &mut b] {
        buf[tail..]
            .par_chunks_mut(1 << 16)
            .for_each(|c| c.fill(F128::ZERO));
    }

    let (zp, ap, bp) = (
        SendPtr(z.as_mut_ptr() as *mut u64),
        SendPtr(a.as_mut_ptr() as *mut u64),
        SendPtr(b.as_mut_ptr() as *mut u64),
    );
    let sp = SendPtr(stripe.as_ptr() as *mut u64);
    let inputs_ref = inputs;

    (0..n_total / BM_V).into_par_iter().for_each_init(
        || {
            (
                vec![[0u64; BM_V]; u64_per_block],
                vec![[0u64; BM_V]; u64_per_block],
                vec![[0u64; BM_V]; u64_per_block],
            )
        },
        move |(rz, ra, rb), g| {
            rz[..useful_words].fill([0u64; BM_V]);
            ra[..useful_words].fill([0u64; BM_V]);
            rb[..useful_words].fill([0u64; BM_V]);
            let o0 = g * BM_V;
            let group: [&S; BM_V] =
                std::array::from_fn(|j| inputs_ref.get(o0 + j).unwrap_or(padding));
            per_group(group, rz, ra, rb);

            unsafe {
                flush_rows_nt(rz, zp.get(), o0, n_blocks_log, useful_chunks);
                flush_rows_nt(ra, ap.get(), o0, n_blocks_log, useful_chunks);
                flush_rows_nt(rb, bp.get(), o0, n_blocks_log, useful_chunks);
                stripe_from_rows(rz, sp.get() as *mut u8, o0, u64_per_block, useful_words);
            }
        },
    );

    (z, a, b, stripe)
}
