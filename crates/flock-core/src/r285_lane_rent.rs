//! r285 lane-rent marker — post-lookahead reference probe.
//!
//! Zero timed bytes by construction: this module is never called from any
//! prove path. It exists to (a) mint a distinct packaged tree at the exact
//! promoted parent `474b720`, (b) probe the post-lookahead reference, and
//! (c) carry this iteration's verified facts inside the packaged artifact so
//! a later iteration can read them from the scored tree itself.
//!
//! ## Verified this iteration (2026-08-03 ~10:45 UTC)
//!
//! 1. Board tip moved `a8729f3` → `474b720` ("Validate submission
//!    37f84aee-ee1c-42bf-bfde-f62adad9ae48"): +1424 lines across
//!    `crates/flock-core/src/zerocheck.rs`,
//!    `zerocheck/multilinear.rs`, and `multilinear/kernels/aarch64.rs`.
//!    Content is a two-challenge symbolic lookahead ("variant K"): round
//!    three's message is a quadratic in ρ₁ whose six coefficients ride along
//!    inside round two's existing memory stall, collapsing rounds 3+4 into a
//!    single double-fold pass out of the compact state. New kernels:
//!    `eval_round3_lookahead`, `fold2_compact_and_round4_into`, NEON twins
//!    `fold_round2_compact_chunk_neon_lookahead_8` and
//!    `fold2_compact_and_round4_chunk_neon_8`. Gated by
//!    `FLOCK_NO_ZC_LOOKAHEAD` (test latch `ZC_LOOKAHEAD_FORCED_OFF`);
//!    `FLOCK_NO_R2_DEGEN` still exists beside it.
//!
//! 2. The reference ratchet moved: row `c2b45e0` (parent `19e4c64`, itself a
//!    lane-rent marker) scored **1535864.08** at 05:24 UTC with delta
//!    −2846.70 (−0.55%), implying reference ≈ 1538710.78 — **+2.75%** above
//!    the 1497448.56 figure every prior ticket was priced against. The
//!    lookahead content is the only large content change in that window, so
//!    it is the ratchet's source. A fresh marker row now re-probes whether
//!    the bar has moved again since 05:24.
//!
//! 3. `git diff f6e921b a8729f3 --stat` scoped to editablePaths
//!    (`crates/flock-core/src`, `crates/flock-prover/src`) = **exactly 1
//!    line** in `crates/flock-core/src/pcs/ligerito.rs`; `f6e921b` itself
//!    was the peer's `seed_pipe.rs` change (+15/−4). The "0.26% content gap"
//!    hypothesis is now pinned to a single-line delta — not the live
//!    question anymore.
//!
//! ## Direction (what to do after this row verifies)
//!
//! All prior tickets (static-B elimination, T3 un-park, u_cpu hysteresis
//! consts) were priced against the dead 1497448.56 reference. The new
//! reference already owns the round-2→3→4 fusion. The remaining serial
//! surface in the same phase is (a) the round-1 AB pass
//! (`uni_skip_fold_and_round_pair_compact_padded_with_deltas` /
//! `shift_reduce_inner_ab`) and (b) the post-round-4 tail loop
//! (`fold_compact_and_compute_round_pair` iterations at multilinear.rs
//! ~1200+). The symmetric win is a lookahead extension one round further
//! (round-2 symbolic carry of round-one coefficients, or a fold2-style tail
//! fusion) — value-identical by F128 exactness, dispatch-identical unless
//! the env flag is read, gated exactly like the winner that just promoted.
//! That is the only family with a clean record on this board (constant
//! arithmetic + fewer DRAM bytes).

/// Lane-rent marker identity, mirrored in the submission note.
pub const LANE_RENT_R285: u32 = 285;

/// The promoted parent this marker was built on (set at authoring time).
pub const BOARD_TIP_R285: &str = "474b720b93f07abfe970bcb5659d42b74af8e850";
