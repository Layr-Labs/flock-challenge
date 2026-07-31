#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "aarch64")]
pub use aarch64::{
    partial_fold_packed_z_neon_iblock_padded, partial_fold_packed_z_neon_oblock_padded,
    partial_fold_packed_z_neon_single, partial_fold_packed_z_neon_single_padded,
};
/// Raw const-generic entry point — tests instantiate it at several `TILE_T`
/// to check every tiling factor against the scalar reference, and to A/B them.
/// Production always goes through `partial_fold_packed_z_neon_oblock_padded`.
#[cfg(all(target_arch = "aarch64", test))]
pub(crate) use aarch64::oblock_padded_tiled;

#[cfg(target_arch = "x86_64")]
mod x86_64;
#[cfg(target_arch = "x86_64")]
pub use x86_64::partial_fold_packed_z_x86_tiled_padded;
