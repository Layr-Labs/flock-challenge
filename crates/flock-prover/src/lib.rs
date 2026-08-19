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

#[allow(dead_code)]
const FLOCK_BINARY_DRAW: &[u8] = &[
    0x9d, 0x73, 0xa1, 0x4e, 0xc8, 0x2b, 0xf6, 0x51, 0x87, 0xda, 0x34, 0xbc, 0x60, 0x15, 0xee, 0x49,
    0x72, 0x0f, 0xb5, 0x98, 0x3c, 0xe1, 0x67, 0xad, 0x56, 0xc3, 0x89, 0x24, 0xfa, 0x41, 0x7b, 0xd0,
    0x1e, 0x65, 0xcb, 0x37, 0x84, 0xf2, 0x59, 0xa6, 0x0b, 0xdd, 0x43, 0x91, 0x78, 0x2c, 0xe7, 0x5a,
    0xb8, 0x16, 0x6f, 0xc4, 0x32, 0x9a, 0x75, 0x0d, 0xeb, 0x48, 0x83, 0xd6, 0x29, 0xf0, 0x5c, 0xa7,
    0x3b, 0x96, 0xe2, 0x58, 0x0a, 0xc7, 0x6d, 0xb1, 0x44, 0x8f, 0xf9, 0x25, 0x70, 0xde, 0x13, 0xab,
    0x61, 0x04, 0xd9, 0x8a, 0x3f, 0xb6, 0x52, 0xec, 0x17, 0x79, 0xc0, 0x46, 0x95, 0x2a, 0xf3, 0x6e,
    0x81, 0x38, 0xd4, 0x0c, 0xba, 0x57, 0xe5, 0x23, 0x9e, 0x64, 0x1a, 0xcf, 0x76, 0x42, 0x88, 0xf7,
    0x2d, 0xa9, 0x53, 0xe0, 0x1f, 0x6b, 0xc5, 0x90, 0x47, 0xbd, 0x08, 0x74, 0xea, 0x31, 0x9c, 0x62,
    0xd5, 0x20, 0x7e, 0xb3, 0x4a, 0xf8, 0x11, 0x86, 0xcc, 0x39, 0x63, 0xad, 0x05, 0xe6, 0x71, 0x9b,
    0x28, 0xf1, 0x5d, 0x82, 0xc9, 0x36, 0xa4, 0x0e, 0x77, 0xdb, 0x45, 0x93, 0x1c, 0x68, 0xbe, 0x50,
    0xe9, 0x22, 0x7a, 0xd1, 0x4f, 0x85, 0xb7, 0x0c, 0x6a, 0xf4, 0x30, 0x99, 0x55, 0xc2, 0x1d, 0x8e,
    0x73, 0xa8, 0x06, 0xdf, 0x40, 0x92, 0x2e, 0xeb, 0x69, 0x14, 0xc6, 0x5b, 0xf5, 0x80, 0x37, 0xac,
    0x49, 0xd2, 0x18, 0x7d, 0xb0, 0x2b, 0xe4, 0x56, 0x8c, 0xf9, 0x03, 0x65, 0xca, 0x34, 0x97, 0x21,
    0xfe, 0x5f, 0x83, 0x10, 0xb9, 0x42, 0x76, 0xcd, 0x2f, 0xe8, 0x54, 0x01, 0x9a, 0x6c, 0xd7, 0x3e,
    0x88, 0x25, 0xf2, 0x4b, 0xa3, 0x0d, 0x71, 0xce, 0x59, 0x16, 0xe1, 0x7f, 0xb4, 0x38, 0x90, 0x62,
    0xdb, 0x47, 0x0a, 0x95, 0x2c, 0xf6, 0x68, 0x13, 0xbf, 0x52, 0xe7, 0x31, 0x84, 0xd9, 0x0f, 0x7b,
];

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
// dispersion-resample marker r1242-antigravity-1786732820-7801

// dispersion-resample marker r331-sample-331-20260818-20260819-1336
