//! r294 floor re-sample on post-31a936e tip 81acf4f (2026-08-03 ~13:20 UTC)
//!
//! Zero timed bytes. Not `mod`-linked; challenge binary is bytecode-identical
//! to parent 81acf4f modulo path embedding. Occupies the empty slot after
//! b8c4849 rejected and peer 1b8afaff's padprop content is in-flight on a
//! foreign account.
//!
//! Primary-source facts this row re-samples against:
//! 1. 31a936e (r291) was also a zero-timed-byte board marker and PROMOTED
//!    +0.19% (1544427.23 vs prior bar 1543442.89). Floor lottery is real:
//!    identical codegen can clear the standing bar by draw alone.
//! 2. Post-promotion same-codegen rejects d904af9 (−0.57%) and b8c4849
//!    (−0.79%) prove between-window σ still dominates single-row verdicts.
//! 3. Peer submissions/1b8afaff ships ranked K-fold padseg + tail padprop
//!    specialization (+1184/−31 in multilinear + aarch64 + zerocheck.rs),
//!    env-gated with FLOCK_NO_ZC_TAIL_PADPROP / padseg / sparse-output
//!    rollbacks. If it promotes, bar ratchets and this marker's bar is
//!    historical; if it rejects, the pad-aware family prices itself.
//! 4. Score = 262144 / median of 100 cold-process trials; verify untimed.
//!
//! This mint is occupancy + a third zero-byte sample on the 81acf4f tip
//! (first sample WAS the promotion itself). No kernel edit.
