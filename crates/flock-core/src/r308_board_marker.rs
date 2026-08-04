//! r308 board marker — STRUCTURAL PIVOT: the sticky latch-off may be a
//! repo-shipped config delta, not machine state. One diff settles it.
//!
//! This module is a durable board-state record (zero timed bytes, doc-only,
//! ships because `crates/flock-core/src` is in `editablePaths`). It exists
//! to preserve the pivot reasoning across the operator hold and to pin the
//! gating read for the next iteration.

//! ## Why the machine-state story is now suspect
//!
//! The harness spawns the worker under `.env_clear()` and re-injects ONLY
//! `RAYON_NUM_THREADS` + `TMPDIR` (benchmark-tools/harness/src/main.rs:157-161),
//! and the GPU keeper `commit_l0_or_fallback` requires "more than one rayon
//! thread" as part of `is_ranked_gpu_shape` (crates/flock-core/src/gpu_commit.rs).
//! So a `RAYON_NUM_THREADS=1` at the workflow level is a SUFFICIENT cause for
//! a clean, uniform, sticky latch-off — no thermal story needed.
//!
//! The flip happened between 15:57Z (fbaa9b5, GPU world, p10 0.16893 s) and
//! 16:24Z (60f15d7, CPU world, p10 0.27839 s). In that exact window
//! `origin/main` moved 81acf4f -> 972420b -> cb4f607 (bot validation pushes
//! carry the FULL candidate tree; only crates/flock-core/src and
//! crates/flock-prover/src are *packaged*, so a peer commit could smuggle a
//! `.github/workflows/benchmark-blake3-mac.yml` or harness change through the
//! same push). The r302-r307 "machine-side event" conclusions all assumed the
//! workflow file was frozen — but that file is NOT inside the SHA256SUMS-gated
//! `benchmark-tools/trusted/` dir, so it was never verified frozen.
//!
//! ## What each outcome means for the 10-wins game
//!
//! - Workflow diff non-empty (RAYON_NUM_THREADS=1 or GPU gate added): the
//!   regime is a CONFIG fact. The GPU world will not return while that commit
//!   stands; marker thermometry is the wrong instrument; the game reduces to
//!   out-engineering the CPU floor (~0.282 s, ~40% below the frozen bar) —
//!   which the byte ledger says is bandwidth-bound and unwinnable by content
//!   — or to reverting/overriding the config path from inside flock-core
//!   (the only shippable surface; e.g. making the ranked path ignore
//!   RAYON_NUM_THREADS or latch on a fixed thread count).
//! - Workflow diff empty and RAYON_NUM_THREADS >= 2: machine-state story
//!   stands; marker cadence resumes as a pure latch thermometer and the next
//!   content mint waits for a GPU-world row.
//!
//! ## Gating read (one command, run when the hold lifts)
//!
//! ```bash
//! git -C /home/frosty/eth/flock-challenge diff 81acf4f cb4f607 -- \
//!   .github/workflows/benchmark-blake3-mac.yml \
//!   benchmark-tools/harness/src/main.rs
//! # then: grep the workflow blob for RAYON_NUM_THREADS / *_ON / gpu
//! ```
//!
//! If the diff is empty, the alternative one-read check is the worker-side
//! thread count at warmup: `RAYON_NUM_THREADS` unset would leave rayon's
//! default (= logical cores), so the latch would need a different cause.
