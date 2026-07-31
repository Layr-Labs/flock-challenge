//! `flock-prover`: the Apple-silicon-optimized end-to-end Flock prover.
//!
//! Builds on [`flock_core`] (the protocol library + verifier) with the
//! top-level prove orchestration ([`prover`]), the monolithic hash R1CS
//! encoders ([`r1cs_hashes`]), and the hash-chain / Merkle-path statement
//! builders ([`chain`], [`merkle_path`], [`proof_io`]).
//!
//! For convenience, the entire `flock_core` API is re-exported here, so code
//! depending on `flock-prover` can reach `field`, `pcs`, `verifier`, etc.
//! through this crate.
//!
//! Workspace-wide Clippy `allow`s for the hand-tuned numeric kernels are
//! declared in `[workspace.lints.clippy]` at the repo root.

pub use flock_core::*;

pub mod chain;
pub mod merkle_path;
pub mod proof_io;
pub mod prover;
pub mod r1cs_hashes;
// NOTE: a recycling #[global_allocator] (recycle_alloc.rs) was part of
// submission 6c3a0ffe, which regressed -4.7% vs its base on the ranked
// runner while the NTT fusion alone measured +6% locally; the allocator is
// unwired pending an isolated ranked measurement.
