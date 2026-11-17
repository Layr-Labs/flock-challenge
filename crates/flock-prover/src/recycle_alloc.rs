//! Recycling global allocator for the prover process.
//!
//! Blocks at least 32 KiB are parked on exact-size freelists rather than
//! returned to libmalloc. The ranked worker performs an untimed warm proof
//! with the same allocation pattern, so the timed proof reuses resident pages
//! for large allocations not already handled by the typed scratch pools.
//!
//! Size classes are registered only from the process's first prove on
//! ([`open_classes`], kill switch [`ENV_NO_RECYCLE_SETUP_GATE`]): the setup's
//! R1CS construction alone frees blocks of more than 512 distinct sizes
//! between 16 and 32 KiB, which used to take every class slot before the
//! first prove, so no prove-time size ever got a class.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{
    AtomicU8, AtomicUsize,
    Ordering::{Acquire, Relaxed, Release},
};

const RECYCLE_MIN: usize = 16 * 1024;
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

/// Kill switch for the setup class gate (`FLOCK_NO_RECYCLE_SETUP_GATE`, any
/// value: every freed size registers a class from process start, as before).
/// With the switch unset, no class is registered until the first prove of the
/// process calls [`open_classes`]: the setup's transient row vectors (more
/// than 512 distinct sizes from 16 to 32 KiB, `xor_dedup`) then no longer
/// fill the class table, so the warm-up prove's freed blocks get classes and
/// the timed prove reuses them instead of taking ~60 blocks of 16 KiB-512 KiB
/// (~7 MiB) from libmalloc. Blocks of a size that already has a class are
/// handled exactly as before. Read once per process with `getenv` (no
/// allocation: this runs inside the allocator).
pub const ENV_NO_RECYCLE_SETUP_GATE: &str = "FLOCK_NO_RECYCLE_SETUP_GATE";

const GATE_UNKNOWN: u8 = 0;
const GATE_OPEN: u8 = 1;
const GATE_SETUP: u8 = 2;
static GATE: AtomicU8 = AtomicU8::new(GATE_UNKNOWN);

/// Whether a new size class may be registered now.
#[inline]
fn classes_open() -> bool {
    match GATE.load(Relaxed) {
        GATE_OPEN => true,
        GATE_SETUP => false,
        _ => {
            unsafe extern "C" {
                fn getenv(name: *const core::ffi::c_char) -> *const core::ffi::c_char;
            }
            let killed = !unsafe { getenv(c"FLOCK_NO_RECYCLE_SETUP_GATE".as_ptr()) }.is_null();
            let resolved = if killed { GATE_OPEN } else { GATE_SETUP };
            let _ = GATE.compare_exchange(GATE_UNKNOWN, resolved, Relaxed, Relaxed);
            GATE.load(Relaxed) == GATE_OPEN
        }
    }
}

/// Open class registration for the rest of the process (idempotent; the
/// prover calls it at the start of every prove, so the first one opens it).
pub fn open_classes() {
    if !classes_open() {
        GATE.store(GATE_OPEN, Relaxed);
    }
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
            if !insert || !classes_open() {
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
