#![allow(
    clippy::needless_return,
    clippy::missing_safety_doc,
    clippy::redundant_closure,
    clippy::collapsible_if,
    clippy::manual_range_contains,
    clippy::manual_is_multiple_of,
    clippy::manual_slice_size_calculation,
    clippy::manual_div_ceil,
    clippy::let_and_return,
    clippy::needless_borrow,
    clippy::needless_borrows_for_generic_args,
    clippy::unnecessary_cast,
    clippy::items_after_test_module
)]

pub use flock_core::*;

pub mod chain;
pub mod merkle_path;
pub mod proof_io;
pub mod prover;
pub mod r1cs_hashes;
pub mod seed_pipe;

pub mod recycle_alloc;

#[cfg(any(
    target_arch = "aarch64",
    all(target_arch = "x86_64", target_feature = "avx512f")
))]
#[global_allocator]
static RECYCLE_ALLOC: recycle_alloc::RecycleAlloc = recycle_alloc::RecycleAlloc;
