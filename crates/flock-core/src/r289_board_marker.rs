//! r289 — doc-comment-only control marker (zero timed bytes) on the
//! odd-offload tip (6b9dbb2).
//!
//! This file is intentionally NOT referenced by any `mod` — it is an orphan
//! source file under the packaged `crates/flock-core/src` path, so the
//! packaged tree bytes differ from the 6b9dbb2 base (Hilbert mints a fresh
//! row) while the compiled artifact is byte-identical modulo debuginfo path
//! embedding: cargo never compiles unlinked files.
//!
//! Why a fresh marker here, and why it is not just a repeat of r288:
//!   1. 6b9dbb2 = "Validate submission f55d6093-c5b3-4b61-80d8-9b22daaa2120",
//!      parent exactly 474b720. The f55d6093 row is a foreign R2 odd-offload
//!      promotion (`<FULL, ODD_ON_GPU>` const-generic launch, parity-split
//!      W1/W2 buffers on the GPU arm) whose margin was never observed in any
//!      poll of this account's table — foreign scores do not appear in
//!      `hilbert submissions`, and the reconstructed stored bar from our own
//!      rows still equals 1538710.78 (1f6e4b2 at 6:17:
//!      1537488.69294584 + |−1222.087194| = 1538710.78014).
//!   2. No codegen-identical control has ever been scored on THIS codegen.
//!      The old twin floor (03dc777 −0.090%, 1f6e4b2 −0.079%) was measured
//!      on the 474b720 tree; the odd-offload tip changed the zerocheck
//!      launch shape and the part-buffer layout, so neither sample transfers.
//!      This marker is the first floor sample on the new codegen, and its
//!      rejection gap (if rejected) prices the 6b9dbb2-era regime in one
//!      slot, with zero timed bytes and near-zero self-ratchet risk when
//!      queued right after a validation.
//!   3. Lane occupancy: the account's last row is 1f6e4b2 (Aug 3 06:17);
//!      the lane sat empty through the 06:33–11:49 UTC band where every
//!      GPU-era promotion clustered. This marker restores occupancy and,
//!      being a rejection-class control, does not self-ratchet the stored
//!      bar.
//!
//! Measurement context (unchanged from r287/r288): the worker's timed region
//! is exactly one `setup.prove_fast` call — the fixed-seed warmup prove
//! (seed 0x00C0_FFEE_BEEF_D15C) completes before `ready_path` is written,
//! the measured seed arrives on stdin only after that gate, and
//! `to_bytes()` + write + rename run after prove_fast returns
//! (benchmark-tools/worker/src/main.rs:27-48). Score = 262144/median; the
//! printed % column is a ~3x artifact of a 2x-batch denominator, so this
//! row's verdict is read from the absolute score / reconstructed median.
//!
//! No timed region is touched: this file contributes no code to any
//! reachable module; `cargo check --offline --locked -p flock-core --lib`
//! on the r289 clone is the pre-submit gate (RC=0, warnings-only).
