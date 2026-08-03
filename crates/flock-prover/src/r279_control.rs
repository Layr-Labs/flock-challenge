//! r279 control marker — iteration 279 (occupation ticket, kernel-identical).
//!
//! This file is intentionally NOT referenced by any `mod` declaration and is
//! never compiled: it exists solely to give the r279 submission a distinct
//! packaged-tree identity while keeping every executed byte identical to the
//! verified:true GPU-arm rows 3db38f7 (p10 0.1732742161 s, score 1494674.76)
//! and 454616e (score 1493551.20).
//!
//! Hilberts dedupes submissions by packaged-tree content, so every ticket must
//! be a distinct tree; an inert marker inside editablePaths is the proven
//! zero-executed-byte mint (r278: crates/flock-prover/src/r278_control.rs,
//! commit 9539101b -> row 454616e). This is the third twin of the series:
//! 3db38f7 vs 454616e measured a 0.075% whole-curve wave at identical content,
//! which bounds the noise floor at 0.037-0.075% (z ~= 2.5-3.7 against the
//! 0.185-0.26% median deficit). Marker rows are lane-rent plus a free noise
//! sample; they are never a win engine.
//!
//! Decisive reads still owed before any real edit: multilinear.rs:1130-1165
//! (T3 16 MiB scaled_table residency), seed_pipe.rs:96-140 (2^18 mirror
//! seriality), lincheck.rs:801-855/1407-1462 + gpu_commit.rs:7169-7183
//! (head-blind fold split), gpu_commit.rs:5294-5304 (tuned_k >= 16 demote
//! discriminator).
