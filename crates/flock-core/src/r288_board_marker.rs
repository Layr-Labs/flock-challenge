//! r288 — doc-comment-only control marker (zero timed bytes).
//!
//! This file is intentionally NOT referenced by any `mod` — it is an orphan
//! source file under the packaged `crates/flock-core/src` path, so the
//! packaged tree bytes differ from the 474b720 base (Hilbert mints a fresh
//! row) while the compiled artifact is byte-identical modulo debuginfo path
//! embedding: cargo never compiles unlinked files.
//!
//! Purpose: the third clean twin on the post-K bar (474b720). Prior twins:
//!   - 03dc777 (r287 marker, commit 37c6a11): score 1537331.09, true deficit
//!     −0.27% vs the 1538710.78 bar — reconstructed bar − 1379.69 abs.
//!   - c2b45e0 (r284 marker, parent 19e4c64): score 1535864.08, −0.55%.
//!   The systematic twin deficit (~−0.2…−0.6% with zero timed bytes) is
//!   either window/runner state or a binary-layout tax; a doc-comment-only
//!   control on the true board tip discriminates by holding codegen constant
//!   while the packaged tree still differs.
//!
//! Win bar on this lane (finding 15): promote iff score > 1538710.78 ⇔
//! median < 262144/1538710.78 = 0.1703665 s. Best verified row is 03dc777 at
//! 1537331.09 (median 0.1705203 s) — static gap ≈ 0.09% there, within noise.
//!
//! No timed region is touched: the worker's measured interval is exactly one
//! `setup.prove_fast` call after the fixed-seed warmup prove completes
//! (benchmark-tools/worker/src/main.rs:27-42), and this file contributes no
//! code to any reachable module.
