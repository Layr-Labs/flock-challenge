//! Twelve-leaf BLAKE3 chunk kernel for Apple AArch64.
//!
//! Upstream BLAKE3's NEON `hash_many` hashes four 1 KiB chunks through all
//! sixteen dependent compression blocks before starting the next four. The
//! generated kernel below keeps three independent four-lane states in flight
//! and rotates between them after every BLAKE3 round. This exposes enough
//! independent add/xor/rotate chains to fill Apple P-core execution slots.
//!
//! The assembly is compiler-generated from the same BLAKE3 1.8.5 NEON
//! primitives linked by this crate, with `-O3 -mcpu=apple-m3`. It fixes the
//! exact Merkle-leaf contract used here: twelve contiguous 1024-byte unkeyed
//! chunks, counter zero, `CHUNK_START | CHUNK_END`, 32 output bytes each.

core::arch::global_asm!(include_str!("blake3_neon12_macos.S"), options(raw));

unsafe extern "C" {
    fn flock_blake3_hash12_neon_1024(data: *const u8, out: *mut u8, groups: usize);
}

/// Hash as many complete groups of twelve 1 KiB leaves as fit in `out`.
///
/// Returns the number of leaves written. The caller handles the tail through
/// upstream `hash_many`, which also makes arbitrary Rayon partition sizes
/// safe without padding or over-read.
#[inline]
pub(super) fn hash_complete_groups(data: &[u8], out: &mut [[u8; 32]]) -> usize {
    debug_assert_eq!(data.len(), out.len() * 1024);
    let groups = out.len() / 12;
    if groups == 0 {
        return 0;
    }

    // SAFETY: each group consumes exactly 12 * 1024 initialized bytes and
    // writes exactly 12 * 32 bytes. `groups` is floor(out.len() / 12), and
    // the debug assertion records the data/output correspondence established
    // by the Merkle caller. The kernel is compiled only for Apple AArch64,
    // where NEON is mandatory.
    unsafe {
        flock_blake3_hash12_neon_1024(data.as_ptr(), out.as_mut_ptr().cast(), groups);
    }
    groups * 12
}
