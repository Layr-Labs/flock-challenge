//! r300 board marker — window sampler + record-bar re-anchor (2026-08-03)
//!
//! Zero timed bytes. Packaged under editablePaths so Hilbert mints a row;
//! not `mod`-linked, so the challenge binary is bytecode-identical to parent
//! 81acf4f modulo path embedding.
//!
//! What this row tests (primary-source this cycle, not restated leads):
//! 1. The board's best score 1544427.23507854 IS a marker row: parent commit
//!    81acf4f = "Validate submission 31a936e5" whose only diff is
//!    crates/flock-core/src/r291_board_marker.rs (+24 lines, doc-only). The
//!    all-time median record was set by a zero-timed-bytes mint in a
//!    fast-cluster window, not by any kernel change.
//! 2. r298 (0a216bf, cap 15/16->7/8 at zc_r2_gate_share, gpu_commit.rs:8293)
//!    rejected 1543640.38 -> median 0.169796 s = fast-cluster draw (+61 us vs
//!    record bar; band 0.169736-0.169811). The cap family is priced null
//!    alone on the post-rewrite codegen; the promoted-era mechanism (ratio
//!    0.125 -> formula g=3562 -> cap binds at 1920/1792) did not transfer.
//! 3. Depth-n deferred-evaluation is CLOSED at primary source: the fold is a
//!    chain - round i draws its own rho (zerocheck.rs mlv_rhos[i], five push
//!    sites 655/682/714/737/799) and round i's message is computed FROM the
//!    folded state of round i-1 (fold_and_compute_round_pair_into), so the
//!    coefficient tensor of the "one-pass Horner" form does not exist.
//! 4. Thread-reuse is embedded at tip (challenger.rs grind_worker + OnceLock,
//!    FLOCK_NO_GRIND_WORKER default-ON under the worker's .env_clear()); the
//!    0be0dfc/4fd6d50 port is a byte-no-op.
//!
//! Lane mandate: never sit empty. The record bar is a fast-cluster low draw;
//! P(any content mint beats it) is governed by cluster placement, so this row
//! re-samples the window at minimum risk and re-anchors the reference for the
//! next content coin.
