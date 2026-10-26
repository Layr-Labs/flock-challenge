use super::*;
use core::arch::x86_64::*;
use rayon::prelude::*;

#[repr(C, align(64))]
pub(super) struct Block([u8; 512]);

struct PooledStorage(Vec<F128>);

impl Drop for PooledStorage {
    fn drop(&mut self) {
        crate::scratch::give_f128(std::mem::take(&mut self.0));
    }
}

pub struct RootTileCache {
    storage: PooledStorage,
    offset: usize,
    tiles: usize,
    blocks: usize,
    word_major: bool,
    fixed_roots: bool,
}

#[derive(Clone, Copy)]
pub(super) struct Capture {
    ptr: *mut Block,
    tiles: usize,
    blocks: usize,
    word_major: bool,
}

const WORD_TO_COLUMNS: [u8; 64] = {
    let mut a = [0; 64];
    let mut i = 0;
    while i < 64 {
        let byte = i / 8;
        let lane = 7 - i % 8;
        a[i] = (if byte < 4 {
            lane * 4 + byte
        } else {
            32 + lane * 4 + byte - 4
        }) as u8;
        i += 1;
    }
    a
};

#[inline(always)]
unsafe fn raw_words(row: __m512i) -> __m512i {
    const REVERSE: [u8; 64] = {
        let mut a = [0; 64];
        let mut i = 0;
        while i < 64 {
            a[i] = ((i & 8) + 7 - i % 8) as u8;
            i += 1;
        }
        a
    };
    unsafe {
        let row = _mm512_shuffle_epi8(row, _mm512_loadu_si512(REVERSE.as_ptr().cast()));
        let matrix = _mm512_gf2p8affine_epi64_epi8::<0>(
            _mm512_set1_epi64(0x0102040810204080u64 as i64),
            row,
        );
        const COLUMNS_TO_WORD: [u8; 64] = {
            let mut a = [0; 64];
            let mut i = 0;
            while i < 64 {
                a[WORD_TO_COLUMNS[i] as usize] = i as u8;
                i += 1;
            }
            a
        };
        _mm512_permutexvar_epi8(_mm512_loadu_si512(COLUMNS_TO_WORD.as_ptr().cast()), matrix)
    }
}

unsafe impl Send for Capture {}
unsafe impl Sync for Capture {}

impl Capture {
    #[inline]
    pub(super) unsafe fn publish(&self, tile: usize, block: usize, rows: &[__m512i; 8]) {
        debug_assert!(tile < self.tiles && block < self.blocks);

        unsafe {
            let dst = self.ptr.add(tile * self.blocks + block).cast::<__m512i>();
            for (i, value) in rows.iter().enumerate() {
                let value = if self.word_major {
                    raw_words(*value)
                } else {
                    *value
                };
                _mm512_stream_si512(dst.add(i), value);
            }
        }
    }
}

fn prepare(
    splice: &RootSplicePlan,
    z: &[F128],
    k: usize,
    eq8_at: &(impl Fn(usize) -> [F128; 8] + Sync),
    word_major: bool,
) -> (RootTileCache, Vec<F128>) {
    let tiles = z.len() / (k / 128) / (S0_TILE_STRIPES * 8);
    let blocks = splice.root_bits.div_ceil(64);
    assert!(tiles > 0 && blocks > 0);
    assert_eq!(z.len(), tiles * (k / 128) * S0_TILE_STRIPES * 8);

    let fields = tiles.checked_mul(blocks).unwrap().checked_mul(32).unwrap();
    let mut storage = PooledStorage(crate::scratch::take_f128(fields.checked_add(3).unwrap()));
    let offset = storage.0.as_ptr().align_offset(64);
    assert!(offset <= 3);

    let aligned = unsafe { storage.0.as_mut_ptr().add(offset).cast::<Block>() };
    let mut writer = splice.clone();
    writer.capture = Some(Capture {
        ptr: aligned,
        tiles,
        blocks,
        word_major,
    });
    let folded = writer.fold(z, k, eq8_at);

    (
        RootTileCache {
            storage,
            offset,
            tiles,
            blocks,
            word_major,
            fixed_roots: false,
        },
        folded,
    )
}

pub(super) fn fold(
    source: &RootFoldSource,
    splice: &RootSplicePlan,
    z: &[F128],
    k: usize,
    eq8_at: &(impl Fn(usize) -> [F128; 8] + Sync),
) -> Vec<F128> {
    let mut first_fold = None;
    let cache = source.cached_tiles.get_or_init(|| {
        let (cache, folded) = prepare(splice, z, k, eq8_at, false);
        first_fold = Some(folded);
        cache
    });
    if let Some(folded) = first_fold {
        folded
    } else {
        cache.fold(eq8_at)
    }
}

impl RootTileCache {
    pub unsafe fn from_witness_storage(storage: Vec<F128>, compressions: usize) -> Self {
        let storage = PooledStorage(storage);
        assert!(compressions > 0 && compressions.is_multiple_of(64));
        let tiles = compressions / 64;
        let blocks = 176;
        let fields = tiles.checked_mul(blocks).unwrap().checked_mul(32).unwrap();
        let offset = storage.0.as_ptr().align_offset(64);
        assert!(offset <= 3 && offset.checked_add(fields).unwrap() <= storage.0.len());
        Self {
            storage,
            offset,
            tiles,
            blocks,
            word_major: true,
            fixed_roots: true,
        }
    }

    pub(super) fn matches_geometry(&self, compressions: usize, root_bits: usize) -> bool {
        let blocks = if self.fixed_roots { 176 } else { 177 };
        root_bits == 11313
            && self.blocks == blocks
            && self.tiles.checked_mul(64) == Some(compressions)
    }

    pub fn assert_matches_witness(&self, z: &[F128], k_log: usize, plan: &LinearFoldPlan) {
        let splice = RootSplicePlan::new(plan, 1 << k_log).expect("root splice geometry");
        let (reference, _) = prepare(
            &splice,
            z,
            1 << k_log,
            &|_| [F128::ZERO; 8],
            self.word_major,
        );
        assert_eq!(self.tiles, reference.tiles);
        assert_eq!(
            self.blocks + usize::from(self.fixed_roots),
            reference.blocks
        );
        for tile in 0..self.tiles {
            for block in 0..self.blocks {
                let source = block + usize::from(self.fixed_roots && block >= 14);
                let mut expected = reference.block(tile * reference.blocks + source).0;
                if self.fixed_roots && block == 13 {
                    assert!(self.word_major);
                    expected = reference.block(tile * reference.blocks + 14).0;
                    let counter = &reference.block(tile * reference.blocks + 13).0;

                    for stripe in 0..8 {
                        for lane in 0..8 {
                            let byte = stripe * 64 + lane * 4;
                            expected[byte] = (expected[byte] & !1) | (counter[byte] & 1);
                        }
                    }
                }
                assert_eq!(
                    self.block(tile * self.blocks + block).0,
                    expected,
                    "root cache tile {tile}, block {block}"
                );
            }
        }
    }

    #[inline(always)]
    fn block(&self, index: usize) -> &Block {
        assert!(index < self.tiles * self.blocks);

        unsafe {
            &*self
                .storage
                .0
                .as_ptr()
                .add(self.offset)
                .cast::<Block>()
                .add(index)
        }
    }

    fn fold(&self, eq8_at: &(impl Fn(usize) -> [F128; 8] + Sync)) -> Vec<F128> {
        static BATCH: std::sync::LazyLock<bool> =
            std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_ROOT_CACHE_BATCH4").is_none());
        self.fold_with_mode(eq8_at, *BATCH)
    }

    fn fold_with_mode(
        &self,
        eq8_at: &(impl Fn(usize) -> [F128; 8] + Sync),
        batch_word_tiles: bool,
    ) -> Vec<F128> {
        if batch_word_tiles && self.word_major && self.fixed_roots {
            self.fold_impl::<true>(eq8_at)
        } else {
            self.fold_impl::<false>(eq8_at)
        }
    }

    fn fold_impl<const BATCH_WORD_TILES: bool>(
        &self,
        eq8_at: &(impl Fn(usize) -> [F128; 8] + Sync),
    ) -> Vec<F128> {
        let tiles_per_worker = self.tiles.div_ceil(rayon::current_num_threads().max(1));
        let workers = self.tiles.div_ceil(tiles_per_worker);
        let stride = self.blocks * 1024;
        #[cfg(not(test))]
        let mut planes = crate::alloc_uninit_vec::<core::mem::MaybeUninit<u8>>(workers * stride);

        #[cfg(test)]
        let mut plane_storage =
            crate::alloc_uninit_vec::<core::mem::MaybeUninit<u8>>(workers * stride + 63);
        #[cfg(test)]
        let plane_offset = plane_storage.as_ptr().align_offset(64);
        #[cfg(test)]
        let mut planes = &mut plane_storage[plane_offset..plane_offset + workers * stride];
        let mut active = vec![0u8; workers];
        let next = std::sync::atomic::AtomicUsize::new(0);
        let dynamic = dynamic_tiles_enabled();
        planes
            .par_chunks_mut(stride)
            .zip(active.par_iter_mut())
            .enumerate()
            .for_each(|(worker, (dst, active))| {
                let mut mats = [0u64; 128];

                let mut batch_mats = BATCH_WORD_TILES.then(|| [[0u64; 128]; 4]);
                let mut first = true;
                let (mut lo, mut hi) = if dynamic {
                    (0, 0)
                } else {
                    (
                        worker * tiles_per_worker,
                        ((worker + 1) * tiles_per_worker).min(self.tiles),
                    )
                };
                loop {
                    if lo == hi {
                        if !dynamic {
                            break;
                        }
                        lo = next.fetch_add(4, std::sync::atomic::Ordering::Relaxed);
                        if lo >= self.tiles {
                            break;
                        }
                        hi = (lo + 4).min(self.tiles);
                    }
                    if BATCH_WORD_TILES {
                        let tile = lo;
                        let count = (hi - lo).min(4);
                        lo += count;
                        let matrices = batch_mats.as_mut().expect("selected-word batch matrices");
                        for (slot, matrix) in matrices[..count].iter_mut().enumerate() {
                            for t in 0..S0_TILE_STRIPES {
                                let eq8 = eq8_at(8 * ((tile + slot) * S0_TILE_STRIPES + t));
                                kernels::fold_mats_from_basis(
                                    &eq8,
                                    &mut matrix[t * 16..(t + 1) * 16],
                                );
                            }
                        }
                        for block in 0..self.blocks {
                            unsafe {
                                kernels::gfni_fold_word_tiles4(
                                    self.storage
                                        .0
                                        .as_ptr()
                                        .add(self.offset)
                                        .cast::<u8>()
                                        .add((tile * self.blocks + block) * 512),
                                    matrices.as_ptr().cast::<u64>(),
                                    dst.as_mut_ptr().cast::<u8>().add(block * 1024),
                                    self.blocks * 512,
                                    count,
                                    first,
                                    WORD_TO_COLUMNS.as_ptr(),
                                );
                            }
                        }
                        first = false;
                        continue;
                    }
                    let tile = lo;
                    lo += 1;
                    for t in 0..S0_TILE_STRIPES {
                        let eq8 = eq8_at(8 * (tile * S0_TILE_STRIPES + t));
                        kernels::fold_mats_from_basis(&eq8, &mut mats[t * 16..(t + 1) * 16]);
                    }
                    for block in 0..self.blocks {
                        unsafe {
                            let src = self
                                .block(tile * self.blocks + block)
                                .0
                                .as_ptr()
                                .cast::<__m512i>();
                            let rows = core::array::from_fn(|i| {
                                let row = _mm512_load_si512(src.add(i));
                                if self.word_major {
                                    let matrix = _mm512_permutexvar_epi8(
                                        _mm512_loadu_si512(WORD_TO_COLUMNS.as_ptr().cast()),
                                        row,
                                    );
                                    _mm512_gf2p8affine_epi64_epi8::<0>(
                                        _mm512_set1_epi64(0x8040201008040201u64 as i64),
                                        matrix,
                                    )
                                } else {
                                    row
                                }
                            });
                            kernels::gfni_fold_rows(
                                &rows,
                                &mats,
                                dst.as_mut_ptr().cast::<u8>().add(block * 1024),
                                first,
                            );
                        }
                    }
                    first = false;
                }
                *active = u8::from(!first);
            });
        let active_workers: Vec<usize> = active
            .into_iter()
            .enumerate()
            .filter_map(|(i, v)| (v != 0).then_some(i))
            .collect();
        let output_blocks = self.blocks + usize::from(self.fixed_roots);
        let mut out = vec![F128::ZERO; output_blocks * 64];
        out.par_chunks_mut(64).enumerate().for_each(|(block, dst)| {
            if self.fixed_roots && block == 14 {
                return;
            }
            let source = block - usize::from(self.fixed_roots && block > 14);

            unsafe {
                reduce_worker_plane_block(&planes, stride, &active_workers, source, dst);
            }
        });
        if self.fixed_roots {
            out.copy_within(833..896, 897);
            out[833..897].fill(F128::ZERO);

            let one = out[256];
            for column in [839, 865, 866, 868] {
                out[column] = one;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_word_fold4_matches_byte_plane_oracle() {
        const STRIDE: usize = 176 * 512;
        let mut source = vec![0xa7u8; 4 * STRIDE];
        for group in [13usize, 175] {
            let mut mats = [[0u64; 128]; 4];
            let mut oracle_tiles = [[0u8; 1024]; 4];
            for tile in 0..4 {
                for stripe in 0..8 {
                    let weights = core::array::from_fn::<_, 8, _>(|lane| {
                        let n = (tile * 64 + stripe * 8 + lane + 31) as u64;
                        F128::new(
                            n.wrapping_mul(0x9e3779b97f4a7c15),
                            n.rotate_left(29).wrapping_mul(0xbf58476d1ce4e5b9),
                        )
                    });
                    kernels::fold_mats_from_basis(
                        &weights,
                        &mut mats[tile][stripe * 16..(stripe + 1) * 16],
                    );
                    for lane in 0..8 {
                        let n = (tile * 64 + stripe * 8 + lane + 91) as u64;
                        let mut bits = n.wrapping_mul(0xd6e8feb86659fd93)
                            ^ n.rotate_left(37).wrapping_mul(0xa0761d6478bd642f);
                        if group == 175 {
                            bits &= (1u64 << 49) - 1;
                        }
                        let row = tile * STRIDE + group * 512 + stripe * 64;
                        source[row + lane * 4..row + lane * 4 + 4]
                            .copy_from_slice(&(bits as u32).to_le_bytes());
                        source[row + 32 + lane * 4..row + 36 + lane * 4]
                            .copy_from_slice(&((bits >> 32) as u32).to_le_bytes());
                        let mut bytes = [0u8; 16];
                        bytes[..8].copy_from_slice(&weights[lane].lo.to_le_bytes());
                        bytes[8..].copy_from_slice(&weights[lane].hi.to_le_bytes());
                        for bit in 0..64 {
                            if bits >> bit & 1 != 0 {
                                for plane in 0..16 {
                                    oracle_tiles[tile][plane * 64 + bit] ^= bytes[plane];
                                }
                            }
                        }
                    }
                }
            }
            for count in 1..=4 {
                for seed_zero in [true, false] {
                    let mut expected = [0u8; 1024];
                    let mut old = vec![0x5du8; 64 + 1024 + 64];
                    let mut new = old.clone();
                    for i in 0..1024 {
                        old[64 + i] = (i as u8).wrapping_mul(73) ^ 0xc3;
                        new[64 + i] = old[64 + i];
                        expected[i] = if seed_zero { 0 } else { old[64 + i] };
                        for tile in 0..count {
                            expected[i] ^= oracle_tiles[tile][i];
                        }
                    }
                    unsafe {
                        for tile in 0..count {
                            let src = source.as_ptr().add(tile * STRIDE + group * 512);
                            let rows = core::array::from_fn(|stripe| {
                                let row = _mm512_loadu_si512(src.add(stripe * 64).cast());
                                let permuted = _mm512_permutexvar_epi8(
                                    _mm512_loadu_si512(WORD_TO_COLUMNS.as_ptr().cast()),
                                    row,
                                );
                                _mm512_gf2p8affine_epi64_epi8::<0>(
                                    _mm512_set1_epi64(0x8040201008040201u64 as i64),
                                    permuted,
                                )
                            });
                            kernels::gfni_fold_rows(
                                &rows,
                                &mats[tile],
                                old.as_mut_ptr().add(64),
                                seed_zero && tile == 0,
                            );
                        }
                        kernels::gfni_fold_word_tiles4(
                            source.as_ptr().add(group * 512),
                            mats.as_ptr().cast(),
                            new.as_mut_ptr().add(64),
                            STRIDE,
                            count,
                            seed_zero,
                            WORD_TO_COLUMNS.as_ptr(),
                        );
                    }
                    assert_eq!(
                        &old[64..1088],
                        &expected,
                        "old oracle, group={group}, count={count}, first={seed_zero}"
                    );
                    assert_eq!(
                        &new[64..1088],
                        &expected,
                        "fold4 oracle, group={group}, count={count}, first={seed_zero}"
                    );
                    assert!(
                        new[..64]
                            .iter()
                            .chain(new[1088..].iter())
                            .all(|&b| b == 0x5d)
                    );
                    assert!(
                        old[..64]
                            .iter()
                            .chain(old[1088..].iter())
                            .all(|&b| b == 0x5d)
                    );
                }
            }
        }
    }

    fn selected_word_cache_fixture(tiles: usize) -> RootTileCache {
        let blocks = 176;
        let mut storage = crate::scratch::take_f128(tiles * blocks * 32 + 3);
        storage.fill(F128::ZERO);
        let offset = storage.as_ptr().align_offset(64);
        let mut state = 0x953a81c7u64;
        for tile in 0..tiles {
            for block in 0..blocks {
                for stripe in 0..8 {
                    for lane in 0..8 {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        let mut bits = state;
                        if block == 4 {
                            bits |= 1;
                        }
                        if block + 1 == blocks {
                            bits &= (1u64 << 49) - 1;
                        }
                        unsafe {
                            let dst = storage
                                .as_mut_ptr()
                                .add(offset)
                                .cast::<u8>()
                                .add((tile * blocks + block) * 512 + stripe * 64)
                                .cast::<u32>();
                            dst.add(lane).write(bits as u32);
                            dst.add(8 + lane).write((bits >> 32) as u32);
                        }
                    }
                }
            }
        }
        unsafe { RootTileCache::from_witness_storage(storage, tiles * 64) }
    }

    #[test]
    fn selected_word_fold4_cache_ranges_and_worker_reduction() {
        let weights = |i: usize| {
            core::array::from_fn(|lane| {
                let n = (i + lane + 997) as u64;
                F128::new(
                    n.rotate_left(17).wrapping_mul(0x9e3779b97f4a7c15),
                    n.wrapping_mul(0xbf58476d1ce4e5b9),
                )
            })
        };
        for threads in [9, 7] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                for tiles in [1, 3, 4, 5, 17, 29, 33, 43] {
                    let cache = selected_word_cache_fixture(tiles);
                    let expected = cache.fold_with_mode(&weights, false);
                    let actual = cache.fold_with_mode(&weights, true);
                    assert_eq!(actual, expected, "tiles={tiles}, threads={threads}");
                    let (a, b) = rayon::join(
                        || cache.fold_with_mode(&weights, false),
                        || cache.fold_with_mode(&weights, true),
                    );
                    assert_eq!(a, expected);
                    assert_eq!(b, expected);
                }
            });
        }
    }

    #[test]
    #[ignore = "large complete-reader premise; run explicitly in release mode"]
    fn selected_word_fold4_full_reader_premise() {
        use std::time::Instant;
        let cache = selected_word_cache_fixture(4096);
        let weights = |i: usize| {
            core::array::from_fn(|lane| {
                let n = (i + lane + 31) as u64;
                F128::new(
                    n.wrapping_mul(0x9e3779b97f4a7c15),
                    n.rotate_left(29).wrapping_mul(0xbf58476d1ce4e5b9),
                )
            })
        };
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(16)
            .build()
            .unwrap();
        pool.install(|| {
            let expected = cache.fold_with_mode(&weights, false);
            assert_eq!(cache.fold_with_mode(&weights, true), expected);
            let mut samples = [Vec::new(), Vec::new()];
            for _ in 0..4 {
                for batch in [false, true, true, false] {
                    let start = Instant::now();
                    let output = cache.fold_with_mode(&weights, batch);
                    let elapsed = start.elapsed().as_secs_f64() * 1000.;
                    assert_eq!(output, expected, "complete-reader batch={batch}");
                    samples[usize::from(batch)].push(elapsed);
                }
            }
            for times in &mut samples {
                times.sort_by(f64::total_cmp);
            }
            let median = |v: &Vec<f64>| (v[v.len() / 2 - 1] + v[v.len() / 2]) * 0.5;
            let old_ms = median(&samples[0]);
            let new_ms = median(&samples[1]);
            eprintln!(
                "selected-word complete-reader ABBA: old={old_ms:.3}ms \
                batch4={new_ms:.3}ms improvement={:.2}% old_samples={:?} batch4_samples={:?}",
                100. * (old_ms - new_ms) / old_ms,
                samples[0],
                samples[1]
            );
        });
    }

    #[test]
    fn compact_fixed_roots_reweight_arbitrary_basis_and_recycle() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        pool.install(|| {
            let eq = |i: usize, salt: u64| {
                core::array::from_fn::<_, 8, _>(|j| {
                    let x = (i as u64 + j as u64 + salt).wrapping_mul(0x9e3779b97f4a7c15);
                    F128::new(
                        x.rotate_left(13) ^ (x >> 7),
                        (x ^ (x >> 31)).wrapping_mul(0xbf58476d1ce4e5b9),
                    )
                })
            };
            let eq1 = |i| eq(i, 31);
            let eq2 = |i| eq(i, 997);
            for pass in 0..2 {
                let tiles = 3;
                let blocks = 176;
                let mut storage = crate::scratch::take_f128(tiles * blocks * 32 + 3);
                storage.fill(F128::new(u64::MAX, u64::MAX));
                let offset = storage.as_ptr().align_offset(64);
                let mut expected1 = vec![F128::ZERO; 177 * 64];
                let mut expected2 = vec![F128::ZERO; 177 * 64];
                let mut state = 0x953a81c7u64 + pass;
                for tile in 0..tiles {
                    for block in 0..blocks {
                        for stripe in 0..8 {
                            let w1 = eq1(tile * 64 + stripe * 8);
                            let w2 = eq2(tile * 64 + stripe * 8);
                            for lane in 0..8 {
                                state ^= state << 13;
                                state ^= state >> 7;
                                state ^= state << 17;
                                let mut bits = state;
                                if block == 4 {
                                    bits |= 1;
                                }
                                if block + 1 == blocks {
                                    bits &= (1u64 << 49) - 1;
                                }

                                unsafe {
                                    let dst = storage
                                        .as_mut_ptr()
                                        .add(offset)
                                        .cast::<u8>()
                                        .add((tile * blocks + block) * 512 + stripe * 64)
                                        .cast::<u32>();
                                    dst.add(lane).write(bits as u32);
                                    dst.add(8 + lane).write((bits >> 32) as u32);
                                }
                                for bit in 0..64 {
                                    if (bits >> bit) & 1 != 0 {
                                        let compact = block * 64 + bit;
                                        let column = compact + if compact >= 833 { 64 } else { 0 };
                                        expected1[column] += w1[lane];
                                        expected2[column] += w2[lane];
                                    }
                                }
                            }
                        }
                    }
                }
                for expected in [&mut expected1, &mut expected2] {
                    let one = expected[256];
                    for column in [839, 865, 866, 868] {
                        expected[column] = one;
                    }
                }
                let cache = unsafe { RootTileCache::from_witness_storage(storage, tiles * 64) };
                assert!(cache.matches_geometry(tiles * 64, 11313));
                assert!(!cache.matches_geometry(tiles * 64, 11249));
                for batch in [false, true] {
                    assert_eq!(
                        cache.fold_with_mode(&eq1, batch),
                        expected1,
                        "pass={pass}, first basis, batch={batch}"
                    );
                    assert_eq!(
                        cache.fold_with_mode(&eq2, batch),
                        expected2,
                        "pass={pass}, second basis, batch={batch}"
                    );
                    let (a, b) = rayon::join(
                        || cache.fold_with_mode(&eq1, batch),
                        || cache.fold_with_mode(&eq2, batch),
                    );
                    assert_eq!(a, expected1);
                    assert_eq!(b, expected2);
                }

                drop(cache);
            }
        });
    }

    #[test]
    fn root_tile_cache_reweights_and_masks_tail() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        pool.install(|| {
            let roots: Vec<usize> = (0..64)
                .chain((0..64).map(|i| 64 + i + if i >= 17 { 64 } else { 0 }))
                .chain(192..256)
                .chain(256..305)
                .collect();
            let plan = RootSplicePlan {
                actions: vec![
                    RootSpliceAction::Full(0),
                    RootSpliceAction::Start,
                    RootSpliceAction::End {
                        out: 1,
                        low_mask: (1 << 17) - 1,
                    },
                    RootSpliceAction::Full(2),
                    RootSpliceAction::Tail {
                        out: 3,
                        mask: (1 << 49) - 1,
                    },
                    RootSpliceAction::Skip,
                ]
                .into_boxed_slice(),
                source_bits: 305,
                root_bits: 241,
                capture: None,
            };
            plan.check_columns(&roots);
            let mut seed = 0x6182ab49u64;
            let mut rng = || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed
            };
            let z: Vec<F128> = (0..9 * 64 * 3)
                .map(|_| F128 {
                    lo: rng(),
                    hi: rng(),
                })
                .collect();
            let eq1 = |i: usize| {
                core::array::from_fn(|j| F128 {
                    lo: (i + j + 1) as u64 * 0x21931,
                    hi: ((i + j + 5) as u64).rotate_left(29),
                })
            };
            let eq2 = |i: usize| {
                core::array::from_fn(|j| F128 {
                    lo: ((i + j + 13) as u64).rotate_left(37),
                    hi: (i + j + 1) as u64 * 0x821117,
                })
            };
            let reference1 = plan.fold(&z, 384, &eq1);
            let reference2 = plan.fold(&z, 384, &eq2);
            assert_ne!(reference1, reference2);
            let (cache, produced) = prepare(&plan, &z, 384, &eq1, false);
            assert_eq!(produced, reference1);
            assert_eq!(cache.fold(&eq1), reference1);
            assert_eq!(cache.fold(&eq2), reference2);
            let (a, b) = rayon::join(|| cache.fold(&eq1), || cache.fold(&eq2));
            assert_eq!(a, reference1);
            assert_eq!(b, reference2);
            let (raw_cache, raw_first) = prepare(&plan, &z, 384, &eq1, true);
            assert_eq!(raw_first, reference1);
            assert_eq!(raw_cache.fold(&eq1), reference1);
            assert_eq!(raw_cache.fold(&eq2), reference2);
            assert_eq!(raw_cache.fold_with_mode(&eq1, true), reference1);
            assert_eq!(raw_cache.fold_with_mode(&eq2, true), reference2);
            for tile in 0..cache.tiles {
                let tail = &cache.block(tile * cache.blocks + 3).0;
                for row in 0..8 {
                    assert!(tail[row * 64 + 49..row * 64 + 64].iter().all(|&v| v == 0));
                }
            }
            assert_eq!(cache.block(0) as *const Block as usize % 64, 0);
            drop(cache);
            let mut fresh_z = z.clone();
            fresh_z[0].lo ^= 1;
            let fresh_reference = plan.fold(&fresh_z, 384, &eq1);
            assert_ne!(fresh_reference, reference1);
            let (fresh_cache, fresh_first) = prepare(&plan, &fresh_z, 384, &eq1, false);
            assert_eq!(fresh_cache.block(0) as *const Block as usize % 64, 0);
            assert_eq!(fresh_first, fresh_reference);
            assert_eq!(fresh_cache.fold(&eq1), fresh_reference);
            assert_eq!(fresh_cache.fold(&eq2), plan.fold(&fresh_z, 384, &eq2));
        });
    }
}
