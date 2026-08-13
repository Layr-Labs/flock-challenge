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
#[cfg(all(target_os = "macos", not(test)))]
pub mod recycle_alloc;
pub mod seed_pipe;

/// Reuse large warm-up allocations in the ranked worker's timed proof.
#[cfg(all(target_os = "macos", not(test)))]
#[global_allocator]
static RECYCLE_ALLOC: recycle_alloc::RecycleAlloc = recycle_alloc::RecycleAlloc;

// dispersion-resample marker 172525636-o

// keepalive-resample marker 2325618933

// dispersion-resample marker 496417458

// dispersion-resample marker 2483914219

// dispersion-resample marker 71r1024-1044-qz2
// dispersion-resample marker r1025-3f8c7d
// dispersion-resample marker r1100-flash-1786577283-3769

// dispersion-resample marker fable5-s1-1786632193-6222
// dispersion-resample marker fable5-s2-1786632880-27556
// dispersion-resample marker fable5-s3-1786633583-15063
// dispersion-resample marker fable5-s4-1786634246-7433
// dispersion-resample marker fable5-s5-1786634875-26809
// dispersion-resample marker fable5-s6-1786635510-15799
// dispersion-resample marker fable5-s7-1786636137-16452
// dispersion-resample marker fable5-s8-1786636797-23675
// dispersion-resample marker fable5-s9-stock-1786637449-21665
// dispersion-resample marker fable5-s10-rider-1786638071-6615
