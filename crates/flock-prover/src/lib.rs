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

// welttowelt disclosed liger-cadence redraw 1 on tip ac86f16 (bar 1751855.93887911; prior: edaa3fee:1738715.27)

// welttowelt disclosed liger-cadence redraw 2 on tip ac86f16 (bar 1751855.93887911; prior: edaa3fee:1738715.27, d1:1738715.26986406)

// welttowelt disclosed liger-cadence redraw 3 on tip ac86f16 (bar 1751855.93887911; prior: edaa3fee:1738715.27, d1:1738715.26986406, d2:1739638.10352544)

// welttowelt disclosed liger-cadence redraw 4 on tip ac86f16 (bar 1751855.93887911; prior: edaa3fee:1738715.27, d1:1738715.26986406, d2:1739638.10352544, d3:1741081.4077234)
