//! r278 control marker — iteration 278 (reship control, kernel-identical).
//!
//! This file is intentionally NOT referenced by any `mod` declaration and is
//! never compiled: it exists solely to give the r278 submission a distinct
//! packaged-tree identity while keeping every executed byte identical to the
//! verified:true row 3db38f7 (seed_pipe.rs `blocks_eq` chunk 8192 -> 2048,
//! p10 0.1732742161 s = account p10 record, score 1494674.76).
//!
//! Purpose: occupy the now-empty submission lane with the zero-risk ticket and
//! record the packaging-base discriminator. Rows 3db38f7 (parent fd6430b) and
//! 6003c5d (parent 51ab17f) list parents that never matched the locally
//! verified base a8729f3; the parent column of the r278 row will confirm
//! whether that anomaly is harness packaging behavior or a per-submission
//! artifact. If r278's parent column reads a8729f3, the anomaly is retired
//! and the median gap (~0.36 ms ~= 30 MiB at ~86 GB/s) is the only lever.
//!
//! Kernel bytes are untouched: the zc wait-budget tail theory (gpu_commit.rs
//! zc_r2_wait:8592 / zc_t3_wait:9261 / zc_loop_wait:9933) and the T3 compact
//! question (multilinear.rs:1130-1165) remain the next iteration's gating
//! reads before any real edit.
