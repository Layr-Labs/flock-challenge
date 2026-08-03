//! r296 board marker — zero-byte lane-rent probe on true tip `81acf4f`.
//!
//! Purpose: re-anchor the post-rewrite floor with a byte-identical binary.
//! This module is deliberately NOT referenced from `lib.rs`, so it is inert:
//! cargo never compiles it, codegen is unchanged, and the timed region sees
//! zero difference. The scored value is a floor sample + window thermometer
//! after `5c7b437` (rejected, −0.13%, inside the same-codegen envelope) and
//! before any content mint.
//!
//! Marker family history (all zero-byte tree diffs):
//! - `31a936e` promoted +0.19% true ≈ +109 µs — became the standing bar
//!   `1544427.235` (median 0.169735 s).
//! - `5c7b437` rejected −0.13% ≈ −691 pts — inside floor σ (~130–210 µs).
//!
//! This sample prices the current window on the −33,844-line post-rewrite
//! tip (81c064 → 81acf4f era) and confirms the lane is uncontended before
//! the α-rebalance content port (foreign `1c49661b`: ZC_R2_ALPHA 0.55→0.20,
//! g 1920→1495, FLOCK_ZC_R2_LEGACY_ALPHA restore) is staged.

/// Inert probe tag; never referenced from lib.rs.
#[allow(dead_code)]
pub const R296_MARKER_TAG: &str = "r296-zero-byte-lane-rent-81acf4f";
