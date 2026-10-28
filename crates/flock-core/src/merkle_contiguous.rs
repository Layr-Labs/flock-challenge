use super::{BLAKE3_IV, Hash};
core::arch::global_asm!(include_str!("merkle_contiguous.S"));
unsafe extern "C" {
    fn flock_blake3_contiguous_64(
        data: *const u8,
        count: usize,
        blocks: usize,
        key: *const u32,
        counter: u64,
        increment: bool,
        flags: u8,
        start: u8,
        end: u8,
        out: *mut u8,
    );
    fn flock_blake3_contiguous_128(
        data: *const u8,
        count: usize,
        blocks: usize,
        key: *const u32,
        counter: u64,
        increment: bool,
        flags: u8,
        start: u8,
        end: u8,
        out: *mut u8,
    );
    fn flock_blake3_contiguous_256(
        data: *const u8,
        count: usize,
        blocks: usize,
        key: *const u32,
        counter: u64,
        increment: bool,
        flags: u8,
        start: u8,
        end: u8,
        out: *mut u8,
    );
    fn flock_blake3_contiguous_512(
        data: *const u8,
        count: usize,
        blocks: usize,
        key: *const u32,
        counter: u64,
        increment: bool,
        flags: u8,
        start: u8,
        end: u8,
        out: *mut u8,
    );
    fn flock_blake3_contiguous_1024(
        data: *const u8,
        count: usize,
        blocks: usize,
        key: *const u32,
        counter: u64,
        increment: bool,
        flags: u8,
        start: u8,
        end: u8,
        out: *mut u8,
    );
}
type Kernel =
    unsafe extern "C" fn(*const u8, usize, usize, *const u32, u64, bool, u8, u8, u8, *mut u8);

#[inline]
pub(super) fn hash_many<const N: usize>(
    data: &[u8],
    out: &mut [Hash],
    flags: u8,
    start: u8,
    end: u8,
) {
    let kernel: Kernel = match N {
        64 => flock_blake3_contiguous_64,
        128 => flock_blake3_contiguous_128,
        256 => flock_blake3_contiguous_256,
        512 => flock_blake3_contiguous_512,
        1024 => flock_blake3_contiguous_1024,
        _ => unreachable!("unsupported contiguous BLAKE3 size"),
    };
    assert!(!out.is_empty() && out.len() % 16 == 0);
    assert_eq!(data.len(), out.len() * N);
    unsafe {
        kernel(
            data.as_ptr(),
            out.len(),
            N / 64,
            BLAKE3_IV.as_ptr(),
            0,
            false,
            flags,
            start,
            end,
            out.as_mut_ptr().cast(),
        );
    }
}

unsafe extern "C" {
    fn flock_blake3_planes_leaf_64(
        data: *const u8,
        count: usize,
        blocks: usize,
        key: *const u32,
        counter: u64,
        increment: bool,
        flags: u8,
        start: u8,
        end: u8,
        out: *mut u8,
    );
    fn flock_blake3_planes_leaf_128(
        data: *const u8,
        count: usize,
        blocks: usize,
        key: *const u32,
        counter: u64,
        increment: bool,
        flags: u8,
        start: u8,
        end: u8,
        out: *mut u8,
    );
    fn flock_blake3_planes_leaf_256(
        data: *const u8,
        count: usize,
        blocks: usize,
        key: *const u32,
        counter: u64,
        increment: bool,
        flags: u8,
        start: u8,
        end: u8,
        out: *mut u8,
    );
    fn flock_blake3_planes_leaf_512(
        data: *const u8,
        count: usize,
        blocks: usize,
        key: *const u32,
        counter: u64,
        increment: bool,
        flags: u8,
        start: u8,
        end: u8,
        out: *mut u8,
    );
    fn flock_blake3_planes_leaf_1024(
        data: *const u8,
        count: usize,
        blocks: usize,
        key: *const u32,
        counter: u64,
        increment: bool,
        flags: u8,
        start: u8,
        end: u8,
        out: *mut u8,
    );
    fn flock_blake3_planes_parent(
        data: *const u8,
        count: usize,
        blocks: usize,
        key: *const u32,
        counter: u64,
        increment: bool,
        flags: u8,
        start: u8,
        end: u8,
        out: *mut u8,
    );
}

#[allow(dead_code)]
#[inline]
pub(super) fn hash_leaf_pairs<const N: usize>(data: &[u8], out: &mut [Hash]) {
    assert!(!out.is_empty() && out.len() % 16 == 0);
    assert_eq!(data.len(), 2 * out.len() * N);
    let leaf: Kernel = match N {
        64 => flock_blake3_planes_leaf_64,
        128 => flock_blake3_planes_leaf_128,
        256 => flock_blake3_planes_leaf_256,
        512 => flock_blake3_planes_leaf_512,
        1024 => flock_blake3_planes_leaf_1024,
        _ => unreachable!("unsupported plane leaf size"),
    };
    let mut scratch = core::mem::MaybeUninit::<[u8; 128 * 32]>::uninit();
    for (parents, leaves) in out.chunks_mut(64).zip(data.chunks(128 * N)) {
        let planes = scratch.as_mut_ptr().cast::<u8>();
        unsafe {
            leaf(
                leaves.as_ptr(),
                2 * parents.len(),
                N / 64,
                BLAKE3_IV.as_ptr(),
                0,
                false,
                0,
                super::BLAKE3_CHUNK_START,
                super::BLAKE3_CHUNK_END,
                planes,
            );
            flock_blake3_planes_parent(
                planes,
                parents.len(),
                1,
                BLAKE3_IV.as_ptr(),
                0,
                false,
                super::BLAKE3_PARENT,
                0,
                0,
                parents.as_mut_ptr().cast(),
            );
        }
    }
}

#[cfg(test)]
mod plane_tests {
    use super::*;
    #[test]
    fn direct_leaf_planes_match_hazmat_words() {
        use blake3::hazmat::HasherExt;
        for size in [64usize, 128, 256, 512, 1024] {
            let leaf: Kernel = match size {
                64 => flock_blake3_planes_leaf_64,
                128 => flock_blake3_planes_leaf_128,
                256 => flock_blake3_planes_leaf_256,
                512 => flock_blake3_planes_leaf_512,
                1024 => flock_blake3_planes_leaf_1024,
                _ => unreachable!(),
            };
            for count in [16usize, 32, 64, 128] {
                for offset in [0usize, 1, 31, 63] {
                    let mut seed = 0x9e3779b97f4a7c15u64;
                    let mut storage = vec![0u8; offset + count * size];
                    for x in &mut storage {
                        seed ^= seed << 13;
                        seed ^= seed >> 7;
                        seed ^= seed << 17;
                        *x = seed as u8;
                    }
                    let data = &storage[offset..];
                    let mut plane_bytes = vec![0xa5u8; count * 32 + 128];
                    unsafe {
                        leaf(
                            data.as_ptr(),
                            count,
                            size / 64,
                            BLAKE3_IV.as_ptr(),
                            0,
                            false,
                            0,
                            super::super::BLAKE3_CHUNK_START,
                            super::super::BLAKE3_CHUNK_END,
                            plane_bytes.as_mut_ptr().add(64),
                        );
                    }
                    for i in 0..count {
                        let cv = blake3::Hasher::new()
                            .update(&data[i * size..(i + 1) * size])
                            .finalize_non_root();
                        for word in 0..8 {
                            let p = 64 + (i / 16) * 512 + word * 64 + (i % 16) * 4;
                            assert_eq!(
                                plane_bytes[p..p + 4],
                                cv[word * 4..word * 4 + 4],
                                "size={size} count={count} offset={offset} leaf={i} word={word}"
                            );
                        }
                    }
                    assert!(plane_bytes[..64].iter().all(|&x| x == 0xa5));
                    assert!(plane_bytes[count * 32 + 64..].iter().all(|&x| x == 0xa5));
                }
            }
        }
    }
}
