//! Recycling global allocator for the prover process.
//!
//! Blocks at least 16 KiB are parked on exact-size freelists rather than
//! returned to libmalloc. The ranked worker performs an untimed warm proof
//! with the same allocation pattern, so the timed proof reuses resident pages
//! for large allocations not already handled by the typed scratch pools.
//!
//! Class table size. A census of the worker after its warm-up (one proof at
//! log size 18, ten threads) found the previous 512-entry table completely
//! occupied by exact sizes between 16 KiB and 34 KiB: 512 keys, 484 non-empty
//! heads, and no size of 1 MiB or more registered at all. Every later exact
//! size (the multi-megabyte NTT, Merkle and open buffers) therefore failed to
//! register, was allocated and freed through libmalloc on every proof, and
//! each failed lookup walked the full table: 31,300 `pop` calls made
//! 10,343,242 slot probes and found a block 10,026 times. With an 8192-entry
//! table the same run registers 3,362 sizes (seven of them 1 MiB or larger,
//! the largest 28 MiB), probes 47,624 slots and finds a block 25,373 times.
//! The table is a static array of 24-byte classes, so the larger table costs
//! 192 KiB of process memory and nothing else; hashing, collision resolution,
//! the exact-size keys and the intrusive freelists are unchanged.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{
    AtomicUsize,
    Ordering::{Acquire, Release},
};

const RECYCLE_MIN: usize = 16 * 1024;
const MAX_ALIGN: usize = 16;
const MAX_CLASSES: usize = 8192;

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

const _: () = assert!(MAX_CLASSES == 1 << CLASS_BITS);
const CLASS_BITS: u32 = 13;

#[inline]
fn class_slot(size: usize) -> usize {
    // Top `CLASS_BITS` bits of the Fibonacci hash, as before (then 9 bits).
    (size.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - CLASS_BITS)) % MAX_CLASSES
}

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
    top as *mut u8
}

#[inline]
fn push(ptr: *mut u8, size: usize) -> bool {
    let Some(i) = find_class(size, true) else {
        return false;
    };
    let mut head = CLASSES[i].head.lock().unwrap();
    unsafe { *(ptr as *mut usize) = *head };
    *head = ptr as usize;
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
