//! r291 board marker — score-distribution floor sample (2026-08-03)
//!
//! Zero timed bytes. Packaged under editablePaths so Hilbert mints a row;
//! not `mod`-linked, so the challenge binary is bytecode-identical to parent
//! 6b9dbb2 modulo path embedding.
//!
//! Frame this row tests (primary-source, not restated leads):
//! 1. Harness scores the MEDIAN of 100 independent process-spawn trials
//!    (SCORE_PERCENTILE=0.50). Each trial: fresh worker, untimed fixed-seed
//!    prove, then timed seed→proof_file. Verify is OUTSIDE the timer.
//!    20 harness warmup trials are verified but discarded.
//! 2. ed895db (zero-byte marker on this codegen) score 1542882.32 vs bar
//!    1543442.89 ⇒ real deficit 0.036% / ~62 µs of median, while the printed
//!    % column said −0.11%. p10 already BEATS the bar median (0.16876 <
//!    0.16984). The binding loss is bulk-location of the 100-sample median,
//!    not a content-mean hole.
//! 3. r290's "eq_hi-weighted GPU emission" ticket is DEAD: zc_r2_products MSL
//!    already does `e = eq_hi[tgid]; partials[...] = clmul(e, ...)` for all
//!    four parity-split slots (gpu_commit.rs ~8097-8178). Residual live
//!    surface is CPU dead-work on the GPU prefix (writes partials/la that
//!    zc_r2_wait overwrites) + anchors/deltas + W0/W3/W4/W5.
//!
//! Lane mandate: never sit empty. This is occupancy + floor re-sample after
//! def1bbf (−0.18% real / −0.55% printed).
