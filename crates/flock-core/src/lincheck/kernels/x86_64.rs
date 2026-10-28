use super::super::{F128, build_sum_table};
use std::sync::Mutex;

static PARTIALS_POOL: Mutex<Vec<F128>> = Mutex::new(Vec::new());

#[cfg(target_arch = "x86_64")]
#[allow(dead_code)]
#[inline(always)]
unsafe fn process_block_x86_4<const INIT: bool>(
    tile_bytes_ptr: *const u8,
    k: usize,
    bs: usize,
    tables_ptr: *const F128,
    out_ptr: *mut F128,
) {
    use core::arch::x86_64::*;
    const SUB_T: usize = 4;
    unsafe {
        let base_ptr = tile_bytes_ptr.add(bs);
        let w0 = (base_ptr as *const u64).read_unaligned() as usize;
        let ta0 = tables_ptr;
        let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5, mut a6, mut a7) = if INIT {
            (
                _mm_load_si128(ta0.add(w0 & 0xFF) as *const __m128i),
                _mm_load_si128(ta0.add((w0 >> 8) & 0xFF) as *const __m128i),
                _mm_load_si128(ta0.add((w0 >> 16) & 0xFF) as *const __m128i),
                _mm_load_si128(ta0.add((w0 >> 24) & 0xFF) as *const __m128i),
                _mm_load_si128(ta0.add((w0 >> 32) & 0xFF) as *const __m128i),
                _mm_load_si128(ta0.add((w0 >> 40) & 0xFF) as *const __m128i),
                _mm_load_si128(ta0.add((w0 >> 48) & 0xFF) as *const __m128i),
                _mm_load_si128(ta0.add(w0 >> 56) as *const __m128i),
            )
        } else {
            (
                _mm_xor_si128(
                    _mm_load_si128(out_ptr as *const __m128i),
                    _mm_load_si128(ta0.add(w0 & 0xFF) as *const __m128i),
                ),
                _mm_xor_si128(
                    _mm_load_si128(out_ptr.add(1) as *const __m128i),
                    _mm_load_si128(ta0.add((w0 >> 8) & 0xFF) as *const __m128i),
                ),
                _mm_xor_si128(
                    _mm_load_si128(out_ptr.add(2) as *const __m128i),
                    _mm_load_si128(ta0.add((w0 >> 16) & 0xFF) as *const __m128i),
                ),
                _mm_xor_si128(
                    _mm_load_si128(out_ptr.add(3) as *const __m128i),
                    _mm_load_si128(ta0.add((w0 >> 24) & 0xFF) as *const __m128i),
                ),
                _mm_xor_si128(
                    _mm_load_si128(out_ptr.add(4) as *const __m128i),
                    _mm_load_si128(ta0.add((w0 >> 32) & 0xFF) as *const __m128i),
                ),
                _mm_xor_si128(
                    _mm_load_si128(out_ptr.add(5) as *const __m128i),
                    _mm_load_si128(ta0.add((w0 >> 40) & 0xFF) as *const __m128i),
                ),
                _mm_xor_si128(
                    _mm_load_si128(out_ptr.add(6) as *const __m128i),
                    _mm_load_si128(ta0.add((w0 >> 48) & 0xFF) as *const __m128i),
                ),
                _mm_xor_si128(
                    _mm_load_si128(out_ptr.add(7) as *const __m128i),
                    _mm_load_si128(ta0.add(w0 >> 56) as *const __m128i),
                ),
            )
        };
        for t in 1..SUB_T {
            let w = (base_ptr.add(t * k) as *const u64).read_unaligned() as usize;
            let ta = tables_ptr.add(t * 256);
            a0 = _mm_xor_si128(a0, _mm_load_si128(ta.add(w & 0xFF) as *const __m128i));
            a1 = _mm_xor_si128(
                a1,
                _mm_load_si128(ta.add((w >> 8) & 0xFF) as *const __m128i),
            );
            a2 = _mm_xor_si128(
                a2,
                _mm_load_si128(ta.add((w >> 16) & 0xFF) as *const __m128i),
            );
            a3 = _mm_xor_si128(
                a3,
                _mm_load_si128(ta.add((w >> 24) & 0xFF) as *const __m128i),
            );
            a4 = _mm_xor_si128(
                a4,
                _mm_load_si128(ta.add((w >> 32) & 0xFF) as *const __m128i),
            );
            a5 = _mm_xor_si128(
                a5,
                _mm_load_si128(ta.add((w >> 40) & 0xFF) as *const __m128i),
            );
            a6 = _mm_xor_si128(
                a6,
                _mm_load_si128(ta.add((w >> 48) & 0xFF) as *const __m128i),
            );
            a7 = _mm_xor_si128(a7, _mm_load_si128(ta.add(w >> 56) as *const __m128i));
        }
        _mm_store_si128(out_ptr as *mut __m128i, a0);
        _mm_store_si128(out_ptr.add(1) as *mut __m128i, a1);
        _mm_store_si128(out_ptr.add(2) as *mut __m128i, a2);
        _mm_store_si128(out_ptr.add(3) as *mut __m128i, a3);
        _mm_store_si128(out_ptr.add(4) as *mut __m128i, a4);
        _mm_store_si128(out_ptr.add(5) as *mut __m128i, a5);
        _mm_store_si128(out_ptr.add(6) as *mut __m128i, a6);
        _mm_store_si128(out_ptr.add(7) as *mut __m128i, a7);
    }
}

#[cfg(target_arch = "x86_64")]
pub fn partial_fold_packed_z_x86_tiled_padded(
    z_packed: &[u8],
    m: usize,
    k_log: usize,
    useful_bits: usize,
    eq_outer: &[F128],
) -> Vec<F128> {
    use rayon::prelude::*;

    const TILE_T: usize = 8;
    const SUB_T: usize = 4;
    const BLOCK_K: usize = 8;

    let n_log = m - k_log;
    let k = 1usize << k_log;
    let n_outer = 1usize << n_log;
    assert_eq!(z_packed.len(), (1usize << m) / 8);
    assert_eq!(eq_outer.len(), n_outer);
    assert!(n_log >= 6, "need n_outer ≥ 64 for tile of 8 stripes");
    assert!(k_log >= 3, "need k ≥ 8");
    assert!(useful_bits <= k);
    let n_stripes = n_outer / 8;
    assert_eq!(n_stripes % TILE_T, 0);
    assert_eq!(k % BLOCK_K, 0);
    let n_sub_tiles = n_stripes / SUB_T;

    let useful = (useful_bits.div_ceil(BLOCK_K) * BLOCK_K).min(k);
    if useful == 0 {
        return vec![F128::ZERO; k];
    }

    let p = rayon::current_num_threads().max(1);
    let sub_tiles_per_worker = n_sub_tiles.div_ceil(p);
    let n_workers = n_sub_tiles.div_ceil(sub_tiles_per_worker);

    if n_workers <= 1 {
        let mut out = vec![F128::ZERO; k];
        let mut tables = [F128::ZERO; SUB_T * 256];
        for sub_tile in 0..n_sub_tiles {
            let stripe_base = sub_tile * SUB_T;
            for t in 0..SUB_T {
                let eq_off = 8 * (stripe_base + t);
                build_sum_table(
                    &eq_outer[eq_off..eq_off + 8],
                    &mut tables[t * 256..(t + 1) * 256],
                );
            }
            let z_base = unsafe { z_packed.as_ptr().add(stripe_base * k) };
            unsafe {
                crate::gf2_kernels::k0_f128_partial_fold_tile4_slice(
                    z_base,
                    k,
                    0,
                    useful,
                    tables.as_ptr() as *const u64,
                    out.as_mut_ptr() as *mut u64,
                    u8::from(sub_tile == 0),
                );
            }
        }
        return out;
    }

    #[repr(C, align(64))]
    struct AlignedTables([F128; SUB_T * 256]);

    let mut out = crate::alloc_uninit_f128_vec(k);
    out[useful..k].fill(F128::ZERO);

    let rem_workers = n_workers - 1;
    let total_len = rem_workers * k;
    let mut partials = {
        let mut guard = PARTIALS_POOL.lock().unwrap();
        let mut v = std::mem::take(&mut *guard);
        if v.capacity() < total_len {
            v = crate::alloc_uninit_f128_vec(total_len);
        } else {
            unsafe { v.set_len(total_len) };
        }
        v
    };

    let out_base_addr = out.as_mut_ptr() as usize;
    let partials_base_addr = partials.as_mut_ptr() as usize;

    (0..n_workers).into_par_iter().for_each(|w| {
        let sub_lo = w * sub_tiles_per_worker;
        let sub_hi = ((w + 1) * sub_tiles_per_worker).min(n_sub_tiles);
        let dst_ptr = unsafe {
            if w == 0 {
                out_base_addr as *mut u64
            } else {
                (partials_base_addr as *mut F128).add((w - 1) * k) as *mut u64
            }
        };
        let mut tables = AlignedTables([F128::ZERO; SUB_T * 256]);
        for sub_tile in sub_lo..sub_hi {
            let stripe_base = sub_tile * SUB_T;
            for t in 0..SUB_T {
                let eq_off = 8 * (stripe_base + t);
                build_sum_table(
                    &eq_outer[eq_off..eq_off + 8],
                    &mut tables.0[t * 256..(t + 1) * 256],
                );
            }
            let z_base = unsafe { z_packed.as_ptr().add(stripe_base * k) };
            unsafe {
                crate::gf2_kernels::k0_f128_partial_fold_tile4_slice(
                    z_base,
                    k,
                    0,
                    useful,
                    tables.0.as_ptr() as *const u64,
                    dst_ptr,
                    u8::from(sub_tile == sub_lo),
                );
            }
        }
    });

    out[..useful]
        .par_chunks_mut(256)
        .enumerate()
        .for_each(|(ci, out_chunk)| {
            let col_off = ci * 256;
            let chunk_len = out_chunk.len();
            for w in 0..rem_workers {
                let src = &partials[w * k + col_off..w * k + col_off + chunk_len];
                crate::field::f128_slice::xor_assign(out_chunk, src);
            }
        });
    *PARTIALS_POOL.lock().unwrap() = partials;
    out
}
