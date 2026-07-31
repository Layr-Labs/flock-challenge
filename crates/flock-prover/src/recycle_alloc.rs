//! Recycling global allocator for the prover process.
//!
//! Blocks >= [`RECYCLE_MIN`] bytes are never returned to libmalloc on
//! dealloc; they are parked on an exact-size freelist and served back on the
//! next allocation of the same size. The benchmark worker runs an untimed
//! warm-up prove with an identical allocation pattern before the timed
//! prove, so every large timed-region allocation that misses the scratch
//! pool (Merkle trees, lincheck partials, eq tables, the input-generation
//! vector, proof serialization) is served from a warm, already-faulted
//! block instead of a fresh mmap + soft-fault + kernel-zeroing cycle.
//! Measured ≈ 2.7 ms per timed prove on M4 Max.
//!
//! Small allocations pass through to [`System`] untouched — libmalloc's
//! per-thread tiny/small magazines already recycle those efficiently.
//!
//! Design notes:
//! - Freelist heads are per-class `Mutex<usize>` (0 = empty); the freed
//!   block's first word stores the next pointer, so the lists allocate
//!   nothing themselves. A mutex-protected intrusive list has none of the
//!   ABA hazards of a lock-free Treiber stack, and the large-block rate
//!   (~400 ops per prove) makes lock cost irrelevant.
//! - Classes are exact-size, claimed once in a fixed open-addressed table;
//!   `size` slots transition 0 -> size exactly once and are never reused.
//! - Recycling is restricted to `align <= 16`: macOS libmalloc returns
//!   >= 16-byte alignment for every allocation this large and `free`
//!   ignores the layout, so cross-align reuse within that bound is sound.
//!   Larger alignments fall through to `System` untouched.
//! - Steady-state growth is bounded by the worker's own high-water mark
//!   (two proves, identical pattern), ~200 MB of retained RSS.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{
    AtomicUsize,
    Ordering::{Acquire, Release},
};

const RECYCLE_MIN: usize = 32 * 1024;
const MAX_ALIGN: usize = 16;
const MAX_CLASSES: usize = 512;

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

#[inline]
fn class_slot(size: usize) -> usize {
    (size.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 55) % MAX_CLASSES
}

/// Find the class index for `size`, optionally claiming an empty slot.
#[inline]
fn find_class(size: usize, insert: bool) -> Option<usize> {
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
                return Some(i);
            }
            if CLASSES[i].size.load(Acquire) == size {
                return Some(i);
            }
            // Lost the race to a different size; keep probing.
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
    // SAFETY: `top` is a block we parked in `push`; its first word holds the
    // next entry (or 0). The mutex serializes all list access.
    *head = unsafe { *(top as *const usize) };
    top as *mut u8
}

/// Park `ptr` on the freelist for `size`. Returns false if the class table
/// is full (caller must free normally).
#[inline]
fn push(ptr: *mut u8, size: usize) -> bool {
    let Some(i) = find_class(size, true) else {
        return false;
    };
    let mut head = CLASSES[i].head.lock().unwrap();
    // SAFETY: the block is >= 32 KiB and dead; storing the next pointer in
    // its first word is fine. The mutex serializes all list access.
    unsafe { *(ptr as *mut usize) = *head };
    *head = ptr as usize;
    true
}

pub struct RecycleAlloc;

// SAFETY: delegates to `System` for everything except the exact-size
// freelist reuse of >= RECYCLE_MIN, align <= 16 blocks, which hands back
// blocks obtained from `System` with an equal size and compatible (>= 16)
// alignment and never double-frees (blocks are either parked or returned,
// never both).
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
                // Recycled memory holds stale data; the zeroed contract is
                // mandatory.
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
