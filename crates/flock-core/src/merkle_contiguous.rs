//! adjacent license. It preserves all seven compression rounds and the

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
