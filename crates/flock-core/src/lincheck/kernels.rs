#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "aarch64")]
pub use aarch64::{
    fold_block_major_chunk_neon_x2, gather_transpose_tile_neon, lincheck_qform_enabled,
    partial_fold_packed_z_neon_iblock_padded, partial_fold_packed_z_neon_oblock_padded,
    partial_fold_packed_z_neon_single, partial_fold_packed_z_neon_single_padded,
};

#[cfg(target_arch = "x86_64")]
mod x86_64;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "gfni"
))]
pub use x86_64::partial_fold_packed_z_x86_gfni_padded;
#[cfg(target_arch = "x86_64")]
pub use x86_64::partial_fold_packed_z_x86_tiled_padded;
#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
pub(crate) use x86_64::{NibbleTables, build_nibble_tables as build_nibble_tables_portable};
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw"
))]
pub(crate) use x86_64::{build_nibble_tables, fold_block_major_chunk_x86_avx512};
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "gfni"
))]
pub(crate) use x86_64::{fold_mats_from_basis, gfni_fold_tile, xor_bytes_avx512};
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512vbmi",
    target_feature = "gfni"
))]
pub(crate) use x86_64::{gather_transpose_stripe_x86, gather_transpose_stripe4_x86};
