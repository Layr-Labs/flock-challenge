#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "aarch64")]
pub(crate) use aarch64::{
    oblock_claim_count, oblock_claim_stripe_base, partial_fold_packed_z_neon_oblock_padded_range,
    partial_fold_packed_z_neon_oblock_padded_suffix,
};
#[cfg(target_arch = "aarch64")]
pub use aarch64::{
    partial_fold_packed_z_neon_iblock_padded, partial_fold_packed_z_neon_oblock_padded,
    partial_fold_packed_z_neon_oblock16_padded, partial_fold_packed_z_neon_single,
    partial_fold_packed_z_neon_single_padded,
};

#[cfg(target_arch = "x86_64")]
mod x86_64;
#[cfg(target_arch = "x86_64")]
pub use x86_64::partial_fold_packed_z_x86_tiled_padded;
