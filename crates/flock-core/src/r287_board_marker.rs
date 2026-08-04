//! r287: lane-rent noise sample on the variant-K tip (origin/main 474b720).
//!
//! Kernel-identical control marker: a compile-time constant never referenced
//! by any prove-path code, so it executes zero timed bytes and cannot diverge
//! from the validated tip tree on either latch arm. Purposes, in order:
//!   1. Keep the submission lane occupied (operator steering: "keep something
//!      on the board, its no good to not have a submission in the line").
//!   2. Sample the post-ratchet reference — the peer variant-K validation
//!      (origin/main 474b720) lifted the frozen 1497448.56 bar to ~1538710
//!      (+2.75%, "two-challenge symbolic lookahead" turning the round-3 pass
//!      from a materializing 1.5 GiB -> 1 GiB sweep into six scalars). The
//!      immediately preceding peer row c2b45e0 scored 1535864.08 at 05:24
//!      (-0.55% vs that bar), so the post-ratchet reference is confirmed
//!      live and the honest win bar moved with it.
//!   3. Verify the packaging pipeline (editablePaths = flock-core/src +
//!      flock-prover/src) against the freshly fetched origin/main before any
//!      content ticket is authored on top of variant K.
//!
//! All board rows in the 04:19-05:24 window (4f2fa51, a056acf, b123970,
//! 19dca4d, c2b45e0) are now `rejected` — the lane is free at authoring time;
//! this marker is the cheapest correct occupant while the reference window
//! settles around the new bar.

/// Control constant — exported but never read by any timed or untimed path.
pub const R287_LANE_RENT: u64 = 0x00C0_FFEE_BEEF_D15C_u64 ^ 0x474B_7200_0000_0000_u64;
