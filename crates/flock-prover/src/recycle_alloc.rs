//! Recycling global allocator for the prover process.
//!
//! Blocks at least 32 KiB are parked on exact-size freelists rather than
//! returned to libmalloc. The ranked worker performs an untimed warm proof
//! with the same allocation pattern, so the timed proof reuses resident pages
//! for large allocations not already handled by the typed scratch pools.
//!
//! That guarantee only holds while a size can still claim a class slot. See
//! [`MAX_CLASSES`]: at the historical width the table saturated during the
//! warm proof and the timed proof re-faulted 126.6 MiB inside the scored
//! window; with the table wide enough not to saturate the same proof faults
//! 1.8 MiB (the residual is the query-shaped, seed-dependent sizes, which
//! differ between the two proofs by construction and cannot be pre-warmed).

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{
    AtomicUsize,
    Ordering::{Acquire, Release},
};

const RECYCLE_MIN: usize = 32 * 1024;
const MAX_ALIGN: usize = 16;
/// Exact-size class slots. A traced ranked worker process (untimed warm proof
/// + one timed proof at m = 32) requests **more than 512 distinct recyclable
/// byte sizes**: every `Vec` growth rung, every per-level Merkle/eq/densify
/// buffer, and the query-shaped opening vectors each burn one. At 512 the
/// table saturated partway through the *warm* proof, after which `find_class`
/// returned `None` for every unseen size and `dealloc` fell through to
/// `System.dealloc` — measured 1,799 failed parks totalling 245.6 MiB, which
/// the timed proof then had to re-`mmap` and re-fault **inside the scored
/// window** (133.8 MiB of fresh first-touch across 171 allocations, including
/// 3 × 16 MiB densify arenas, the 16/4/1 MiB recursion-level Merkle trees and
/// the 8/4 MiB eq tables). Sizing the table so it cannot saturate keeps those
/// pages resident from the warm proof, which is the whole point of this
/// allocator. Retention stays bounded by peak concurrent live bytes per size
/// (a parked block is popped by the next same-size request), so a wider table
/// costs BSS, not RSS.
const MAX_CLASSES: usize = 4096;
const CLASS_SHIFT: u32 = usize::BITS - MAX_CLASSES.trailing_zeros();

struct Class {
    size: AtomicUsize,
    head: Mutex<usize>,
}

#[allow(clippy::declare_interior_mutable_const)]
const EMPTY: Class = Class {
    size: AtomicUsize::new(0),
    head: Mutex::new(0),
};
static CLASSES: [Class; MAX_CLASSES] = [EMPTY; MAX_CLASSES];

/// Number of occupied class slots. Read before probing for an insert so a
/// saturated table costs one atomic load instead of a full `MAX_CLASSES`
/// scan, and so the kill switch can pin the effective width.
static OCCUPIED: AtomicUsize = AtomicUsize::new(0);

/// Effective class-table width (kill switch, see [`init_from_env`]). Slots
/// beyond this are never *inserted* into, but the full table is always
/// searched, so nothing already parked can be orphaned.
static CLASS_BUDGET: AtomicUsize = AtomicUsize::new(MAX_CLASSES);

/// Hard ceiling on bytes held on the freelists. Beyond it `dealloc` returns
/// the block to the system exactly as before, so retention can never trade
/// resident memory for page residency: a traced ranked worker parks 174 MiB
/// (up from 22 MiB at the old 512-class width) against a ~6.5 GiB prover
/// working set, so this never binds at the ranked shape — it exists so the
/// bound is a property of the code rather than of the measured shape.
const RETENTION_BUDGET: usize = 768 * 1024 * 1024;
static PARKED_BYTES: AtomicUsize = AtomicUsize::new(0);

/// `FLOCK_NO_RECYCLE_WIDE=1` pins the table to the historical 512 classes.
/// Called once from the prover Setup constructor (never from inside the
/// allocator: reading the environment allocates).
pub fn init_from_env() {
    if std::env::var_os("FLOCK_NO_RECYCLE_WIDE").is_some() {
        CLASS_BUDGET.store(512, Release);
    }
}

#[inline]
fn class_slot(size: usize) -> usize {
    (size.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> CLASS_SHIFT) % MAX_CLASSES
}

#[inline]
fn find_class(size: usize, insert: bool) -> Option<usize> {
    let insert = insert && OCCUPIED.load(Acquire) < CLASS_BUDGET.load(Acquire);
    let start = class_slot(size);
    for probe in 0..MAX_CLASSES {
        let i = (start + probe) % MAX_CLASSES;
        let s = CLASSES[i].size.load(Acquire);
        if s == size {
            return Some(i);
        }
        if s == 0 {
            if !insert {
                return None;
            }
            if CLASSES[i]
                .size
                .compare_exchange(0, size, Release, Acquire)
                .is_ok()
            {
                OCCUPIED.fetch_add(1, Release);
                return Some(i);
            }
            if CLASSES[i].size.load(Acquire) == size {
                return Some(i);
            }
        }
    }
    None
}

#[inline]
fn recyclable(layout: &Layout) -> bool {
    layout.size() >= RECYCLE_MIN && layout.align() <= MAX_ALIGN
}

#[inline]
fn pop(size: usize) -> *mut u8 {
    let Some(i) = find_class(size, false) else {
        return core::ptr::null_mut();
    };
    let mut head = CLASSES[i].head.lock().unwrap();
    let top = *head;
    if top == 0 {
        return core::ptr::null_mut();
    }
    *head = unsafe { *(top as *const usize) };
    drop(head);
    PARKED_BYTES.fetch_sub(size, Release);
    top as *mut u8
}

#[inline]
fn push(ptr: *mut u8, size: usize) -> bool {
    if PARKED_BYTES.load(Acquire) + size > RETENTION_BUDGET {
        return false;
    }
    let Some(i) = find_class(size, true) else {
        return false;
    };
    let mut head = CLASSES[i].head.lock().unwrap();
    unsafe { *(ptr as *mut usize) = *head };
    *head = ptr as usize;
    drop(head);
    PARKED_BYTES.fetch_add(size, Release);
    true
}

pub struct RecycleAlloc;

// SAFETY: every recycled block came from System with the exact same size.
// macOS libmalloc provides at least 16-byte alignment at these sizes, and
// layouts requiring larger alignment bypass the recycler.
unsafe impl GlobalAlloc for RecycleAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if recyclable(&layout) {
            let p = pop(layout.size());
            if !p.is_null() {
                return p;
            }
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if recyclable(&layout) {
            let p = pop(layout.size());
            if !p.is_null() {
                unsafe { core::ptr::write_bytes(p, 0, layout.size()) };
                return p;
            }
        }
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if recyclable(&layout) && push(ptr, layout.size()) {
            return;
        }
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The class-slot hash must span the whole table. The historical form
    /// shifted by a constant 55, which yields only 512 distinct slots no
    /// matter how many entries the table has — widening `MAX_CLASSES` without
    /// widening the shift would leave every extra slot unreachable and keep
    /// the saturation bug.
    #[test]
    fn class_slot_spans_the_whole_table() {
        assert!(MAX_CLASSES.is_power_of_two());
        assert_eq!(CLASS_SHIFT, usize::BITS - MAX_CLASSES.trailing_zeros());
        let mut seen = vec![false; MAX_CLASSES];
        for k in 0..(64 * MAX_CLASSES) {
            let slot = class_slot(RECYCLE_MIN + k * 16);
            assert!(slot < MAX_CLASSES);
            seen[slot] = true;
        }
        let covered = seen.iter().filter(|s| **s).count();
        assert!(
            covered > MAX_CLASSES * 3 / 4,
            "class_slot only reached {covered}/{MAX_CLASSES} slots"
        );
    }

    /// A ranked worker process needs well over 512 distinct recyclable sizes;
    /// the table must have room for them with load factor to spare.
    #[test]
    fn table_is_wider_than_a_measured_ranked_process() {
        // Measured on a traced m = 32 worker: 1,408 distinct recyclable sizes
        // across the warm proof plus one timed proof.
        assert!(MAX_CLASSES >= 2 * 1408);
    }
}
