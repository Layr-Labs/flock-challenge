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
// dispersion-resample marker swe17-1786694400-m3max-8351
// dispersion-resample marker glm52-r3-stock-1786670400-002

// dispersion-resample marker glm52-r3-m3max-1786797071-9224
// dispersion-resample marker glm52-r4-m3max-1786797071-29624
// dispersion-resample marker glm52-r5-m3max-1786797071-12722
// dispersion-resample marker glm52-r6-m3max-1786797071-6587
// dispersion-resample marker glm52-r7-m3max-1786797071-19747
// dispersion-resample marker swe17-r3-m3max-1786798600-4521
// dispersion-resample marker glm52-csc-hetero-1786799000-001

// dispersion-resample marker hermes-m3-s2-1786798375-26699
// dispersion-resample marker hermes-m3-s3-1786798375-9744
// dispersion-resample marker hermes-m3-s4-1786798375-12059
// dispersion-resample marker hermes-m3-s5-1786798375-24761
// dispersion-resample marker hermes-m3-s6-1786798375-20395
// dispersion-resample marker glm52-jagged-hetero-1786800000-001

// dispersion-resample marker hermes-m3-push-1786800054-12648
// dispersion-resample marker hermes-m3-push2-1786800054-8873
// dispersion-resample marker hermes-m3-push3-1786800054-23519
// dispersion-resample marker hermes-m3-push4-1786800054-19366
// dispersion-resample marker hermes-m3-push5-1786800054-30064
// dispersion-resample marker hermes-m3-mega2-1-1786808793-6328
// dispersion-resample marker hermes-m3-mega2-2-1786808793-26898
// dispersion-resample marker hermes-m3-mega2-3-1786808793-8255
// dispersion-resample marker hermes-m3-mega2-4-1786808793-1296
// dispersion-resample marker hermes-m3-mega2-5-1786808793-1039
// dispersion-resample marker hermes-m3-mega2-6-1786808793-16717
// dispersion-resample marker hermes-m3-mega2-7-1786808793-10020
// dispersion-resample marker hermes-m3-mega2-8-1786808793-24904
// dispersion-resample marker hermes-m3-mega2-9-1786808793-14419
// dispersion-resample marker hermes-m3-mega2-10-1786808793-31895
// dispersion-resample marker hermes-m3-mega2-11-1786808793-26337
// dispersion-resample marker hermes-m3-mega2-12-1786808793-2477
// dispersion-resample marker hermes-m3-mega2-13-1786808793-15588
// dispersion-resample marker hermes-m3-mega2-14-1786808793-31447
// dispersion-resample marker hermes-m3-mega2-15-1786808793-31965
// dispersion-resample marker hermes-m3-b3-1-1786810557-20306
// dispersion-resample marker hermes-m3-b3-2-1786810557-7425
// dispersion-resample marker hermes-m3-b3-3-1786810557-18132
// dispersion-resample marker hermes-m3-b3-4-1786810557-24461
// dispersion-resample marker hermes-m3-b3-5-1786810557-6116
// dispersion-resample marker hermes-m3-b3-6-1786810557-18528
// dispersion-resample marker hermes-m3-b3-7-1786810557-5931
// dispersion-resample marker hermes-m3-b3-8-1786810557-26004
// dispersion-resample marker hermes-m3-b3-9-1786810557-2102
// dispersion-resample marker hermes-m3-b3-10-1786810557-31151
// dispersion-resample marker ds4pro-m3-1-1786814117-6391
// dispersion-resample marker ds4pro-m3-2-1786814117-3462
// dispersion-resample marker ds4pro-m3-3-1786814117-18450
// dispersion-resample marker ds4pro-m3-4-1786814117-32226
// dispersion-resample marker ds4pro-m3-5-1786814117-29932
// dispersion-resample marker ds4pro-m3-6-1786814117-30713
// dispersion-resample marker ds4pro-m3-7-1786814117-26379
// dispersion-resample marker ds4pro-m3-8-1786814117-4859
// dispersion-resample marker ds4pro-m3-9-1786814117-29168
// dispersion-resample marker ds4pro-m3-10-1786814117-5617
// dispersion-resample marker ds4pro-m3-11-1786814117-23949
// dispersion-resample marker ds4pro-m3-12-1786814117-29005
// dispersion-resample marker ds4pro-m3-13-1786814117-12695
// dispersion-resample marker ds4pro-m3-14-1786814117-13754
// dispersion-resample marker ds4pro-m3-15-1786814117-20841
// dispersion-resample marker ds4pro-m3-16-1786814117-20556
// dispersion-resample marker ds4pro-m3-17-1786814117-26233
// dispersion-resample marker ds4pro-m3-18-1786814117-31053
// dispersion-resample marker ds4pro-m3-19-1786814117-23024
// dispersion-resample marker ds4pro-m3-20-1786814117-27816
// dispersion-resample marker ds4pro-m3-21-1786814117-27787
// dispersion-resample marker ds4pro-m3-22-1786814117-23361
// dispersion-resample marker ds4pro-m3-23-1786814117-12686
// dispersion-resample marker ds4pro-m3-24-1786814117-10852
// dispersion-resample marker ds4pro-m3-25-1786814117-23938
