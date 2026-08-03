//! r284 lane-rent marker on post-variant-K tip 474b720.
//!
//! Dead code on every prove path. Exists only to keep the account's
//! one-in-flight submission lane occupied after noskillcoding's variant-K
//! promotion (+41262 absolute / new bar 1538710.78) while residual levers
//! after the 2 GiB T3+i1 elimination are authored.
//!
//! CRITICAL packaging rule (validated by 19dca4d's -9.25%): after a
//! promotion, a marker built on the pre-promotion parent is scored as the
//! pre-promotion tree and is measured against the new bar. Parent MUST be
//! origin/main == 474b720 (variant K intact). Never package from local main
//! 972420b or a8729f3.

/// Lane-rent constant. Never read on any timed or untimed path.
pub const R284_LANE_RENT: u64 = 0x284_474b_7200_0001;
