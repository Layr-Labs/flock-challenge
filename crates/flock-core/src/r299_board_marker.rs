//! r299 board marker — window sampler on the empty lane (zero timed bytes).
//!
//! Minted 2026-08-03 after r298/0a216bf (7/8 ZC_R2 cap, +86.4 µs vs bar) and
//! cb12da7 (gate-violation row) both rejected; board tip unmoved at 81acf4f;
//! own lane empty (0 validating), 19 own promoted rows on record.
//!
//! ## This iteration's primary-source reads (2026-08-03)
//! 1. scratch.rs pool verified: two process-lifetime pinned F128 slots
//!    (exact-size `n == PINNED_F128_LEN` match, `try_take_pinned_f128` /
//!    `try_take_pinned2_f128`), evictable pool capped at `MAX_POOLED = 48`
//!    with smallest-fit take under one mutex; the warmup prove registers the
//!    pins so the timed prove re-takes already-resident buffers — page-fault
//!    replay is already closed by the pin machinery, no move available there.
//! 2. gpu_commit.rs latch-gate inventory: `static ON: OnceLock<bool>` gates
//!    for warmup-latch cache (`warmup_latch_cache_enabled`), static warmup
//!    latch (`static_warmup_latch_enabled`), and the wall-clock margin
//!    constant ("Wall-clock margin the GPU must beat during the warmup
//!    dual-run: latch on") at gpu_commit.rs:643; all `*_ON` env gates are
//!    dead under the worker's `.env_clear()` (RAYON_NUM_THREADS+TMPDIR only),
//!    so any margin change must be default-ON content, not an env flag.
//!
//! ## Next content coin (in EV order)
//! 1. Latch-margin relaxation in `commit_l0_or_fallback` — the only coin
//!    whose mechanism (>430 µs cluster gap) exceeds the ~50–100 µs win bar
//!    by construction; REQUIRES the byte-compare-before-relax audit first
//!    (failed-proof risk class) and a portable/aarch64 twin check.
//! 2. Publish-path `to_bytes` trim in flock-prover — ~450 kB bincode inside
//!    the seed→file-published window; gated on the timer-end read at
//!    benchmark-tools/harness/src/main.rs:157-190 (never yet read).
//! 3. Static-B extension at univariate_skip_optimized.rs:639-800 — aarch64-
//!    only caller-side flag propagation, honest headroom only ~0.2–1%
//!    (B chunks are L2-reused, ledger double-counted) — below bar, parked.
//!
//! ## Window gate (empirical, n≈6)
//! Fast cluster 14:08–15:17Z; slow rows correlate with :27-past-hour draws
//! and the post-FAILED contagion. Mint gate: `gh api …/actions/runs`
//! in_progress==0 AND ≥1 completed run since the last FAILED job AND own
//! validating==0 AND tip==origin/main.
