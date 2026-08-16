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
// dispersion-resample marker fable5-s11-stock-1786638725-22305
// dispersion-resample marker fable5-s12-stock-1786639407-3317
// dispersion-resample marker fable5-s13-stock-1786640154-7266
// dispersion-resample marker fable5-s13-stock-1786640201-20309
// dispersion-resample marker fable5-s14-stock-1786640820-13162
// dispersion-resample marker fable5-s15-stock-1786641443-13895
// dispersion-resample marker fable5-s16-stock-1786642072-3822
// dispersion-resample marker fable5-s17-stock-1786642682-22589
// dispersion-resample marker fable5-s18-stock-1786643300-17569
// dispersion-resample marker fable5-s19-stock-1786643900-21318
// dispersion-resample marker fable5-s20-stock-1786644488-25482
// dispersion-resample marker fable5-s21-stock-1786645084-7290
// dispersion-resample marker fable5-s22-stock-1786645675-24825
// dispersion-resample marker fable5-s23-stock-1786646280-6536
// dispersion-resample marker fable5-s24-stock-1786646871-22680
// dispersion-resample marker fable5-s25-stock-1786647478-12144
// dispersion-resample marker fable5-s26-stock-1786648067-31685
// dispersion-resample marker fable5-s27-stock-1786659216-10352
// dispersion-resample marker fable5-s28-stock-1786659815-9870
// dispersion-resample marker fable5-s29-stock-1786660433-31261
// dispersion-resample marker fable5-s30-stock-1786661040-18344
// dispersion-resample marker fable5-s31-stock-1786661652-27962
// dispersion-resample marker fable5-s32-stock-1786662240-18063
// dispersion-resample marker fable5-s33-stock-1786662839-16995
// dispersion-resample marker fable5-s34-stock-1786663417-1627
// dispersion-resample marker sample-141-20260815-1806

// dispersion-resample marker 65865230
// dispersion-resample marker sample-146-20260815-1907
// dispersion-resample marker sample-147-20260815-1920
// dispersion-resample marker sample-148-20260815-1935
// dispersion-resample marker sample-149-20260815-1940
// dispersion-resample marker sample-150-20260815-1950
// dispersion-resample marker sample-153-20260815-2012

// dispersion-resample marker sample-154-20260815-2030

// dispersion-resample marker sample-155-20260815-2035

// dispersion-resample marker sample-156-20260815-2045

// dispersion-resample marker sample-157-20260815-2052

// dispersion-resample marker sample-158-20260815-2102

// dispersion-resample marker sample-159-20260815-2113

// dispersion-resample marker sample-160-20260815-2130

// dispersion-resample marker sample-161-20260815-2142

// dispersion-resample marker sample-162-20260815-2159

// dispersion-resample marker sample-163-20260815-2210

// dispersion-resample marker sample-164-20260815-2225

// dispersion-resample marker sample-176-20260816-0036

// dispersion-resample marker sample-177-20260816-0057

// dispersion-resample marker sample-178-20260816-0113

// dispersion-resample marker sample-179-20260816-0131

// dispersion-resample marker sample-180-20260816-0146

// dispersion-resample marker sample-181-20260816-0159

// dispersion-resample marker sample-182-20260816-0338

// dispersion-resample marker sample-183-20260816-0412

// dispersion-resample marker sample-184-20260816-0415

// dispersion-resample marker sample-185-20260816-0430

// dispersion-resample marker sample-186-20260816-0443

// dispersion-resample marker sample-187-20260816-0454

// welttowelt census ticket r95 — 2026-08-16T23:12:57Z
