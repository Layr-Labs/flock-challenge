//! r281 control marker — kernel-identical reship identity (control).
//!
//! This file is UNREFERENCED and NEVER COMPILED into the crate. It exists
//! solely to give this submission a distinct packaged-tree hash so Hilbert's
//! content dedupe (observed: byte-identical reship of 3db38f7 as 847b847
//! returned "Submission already exists" and minted NO row) accepts the
//! archive. Runtime bytes are therefore byte-identical to the verified r279
//! tree (4f2fa51, score 1494687.54, median ≈ 0.175339 s), so this ticket
//! carries zero demote risk: the warmup cache admission (gpu_commit.rs
//! tuned_k >= 16 outright rejection) sees the same kernel, the same consts,
//! the same latch, and the same DRAM traffic.
//!
//! What this ticket buys (per the standing corpus analysis):
//!   1. Lane occupancy — the submission slot must never sit empty; the
//!      account's EV policy is ~20 tickets at P(win)≈0.15-0.2 each.
//!   2. A free noise sample of the scored seconds channel at this
//!      time-of-day window (lead 22: time-of-day is the only never-controlled
//!      covariate; every GPU-era row clusters 00:54-04:10 and trends
//!      monotone-worse — 4f2fa51 at 04:19 scored 1494687.54).
//!   3. A base-drift control: parent column recorded as a8729f3; if the
//!      reference (1497448.56 frozen >= 4 windows) re-anchors while this
//!      ticket is in flight, the delta column quantifies the ratchet
//!      (lead 21: 1492530.06 -> 1494635.17 -> 1496555.39 -> 1497448.56,
//!      +0.33% in ~1.5h, rises coincided with foreign submissions/*).
//!
//! Do NOT delete this file's siblings (r278_control.rs, r279_control.rs):
//! each minted row verified against the same a8729f3 base and the parent
//! column is the post-hoc bot commit, so every row measured the exact tree
//! it verified (lead 14 retired the packaging-parent anomaly).
//!
//! Next real candidate (do not ship before reading, in order):
//!   (a) `git log -S ZC_T3_INTEGRATION_PARKED -- crates/flock-core/src/zerocheck/multilinear.rs`
//!       — the const at multilinear.rs:1063 is `true` ("UN-PARKED (v11)"
//!       comment contradicts its own const); the T3 CPU sweep at
//!       multilinear.rs:1087 is a `run_hetero_chunks` joined against the
//!       (currently None) GPU job drained at :1171 — so an un-park is a
//!       true GPU/CPU overlap edit, not a serial spine edit, and the flip
//!       must be gated on the static latch being On (lead 20) to keep
//!       P(demote) ≈ 0.
//!   (b) multilinear.rs:1864-1873 — the LOOP arm const, same machinery.
//!   (c) gpu_commit.rs ~5250-5340 — warmup-cache snapshot overlap with ZC
//!       phases decides the demote register.
//!
//! r281 is a lane-rent ticket; treat its score as a noise sample, never as
//! evidence about a kernel edit.
