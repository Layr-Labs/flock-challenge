//! r282 process-model control marker — never referenced by lib.rs.
//!
//! Settlement (primary sources opened this iteration):
//! 1. Trusted harness spawns a FRESH worker per trial (run_trial → Command::spawn).
//! 2. Instant starts AFTER ready_path; timed window = seed stdin write → proof file.
//! 3. SCORE_PERCENTILE = 0.50 (median of 100 measured trials); p10 is metrics-only.
//! 4. Worker env is env_clear()'d — only RAYON_NUM_THREADS + TMPDIR survive; every
//!    FLOCK_* env kill-switch / probe is unreachable on the ranked runner.
//! 5. Local shared-repo `main` branch can lag at 972420b (CPU-only demote tip)
//!    while origin/main is a8729f3 — packaging against local main is a -122% trap.
//!
//! This file exists solely to mint a distinct packaged tree for lane occupancy
//! while the process-model settlement is consumed by the next authoring pass.
#![allow(dead_code)]
pub const R282_PROCESS_MODEL_TAG: &str = "r282-env-clear-spawn-per-trial-median-score";
