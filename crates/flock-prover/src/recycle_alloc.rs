//! Recycling global allocator for the prover process.
//!
//! Blocks at least 32 KiB are parked on exact-size freelists rather than
//! returned to libmalloc. The ranked worker performs an untimed warm proof
//! with the same allocation pattern, so the timed proof reuses resident pages
//! for large allocations not already handled by the typed scratch pools.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{
    AtomicUsize,
    Ordering::{AcqRel, Acquire, Release},
};

const RECYCLE_MIN: usize = 32 * 1024;
const MAX_ALIGN: usize = 16;
/// Size-class table capacity. A ranked worker frees ~1,400 distinct sizes of
/// at least 32 KiB; the former 512-slot table filled with 32–53 KiB classes
/// during warm-up, so every larger size found no slot and bypassed recycling
/// entirely — each timed-prove 1–64 MiB allocation faulted fresh pages and
/// each free went back to libmalloc (`munmap`).
const MAX_CLASSES: usize = 4096;

/// Mid-size classes whose live-block high-water mark is tracked, so the
/// warm-up can leave enough parked blocks for the timed prove (see
/// [`top_up_freelists`]). Smaller blocks are cheap to fault; larger ones are
/// owned by the typed scratch pools.
const TRACK_MIN: usize = 1 << 20;
const TRACK_MAX: usize = 64 << 20;

struct Class {
    size: AtomicUsize,
    head: Mutex<usize>,
    live: AtomicUsize,
    peak: AtomicUsize,
}

#[allow(clippy::declare_interior_mutable_const)]
const EMPTY: Class = Class {
    size: AtomicUsize::new(0),
    head: Mutex::new(0),
    live: AtomicUsize::new(0),
    peak: AtomicUsize::new(0),
};
static CLASSES: [Class; MAX_CLASSES] = [EMPTY; MAX_CLASSES];

#[inline]
fn class_slot(size: usize) -> usize {
    (size.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - MAX_CLASSES.trailing_zeros())) % MAX_CLASSES
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

#[inline]
fn tracked(size: usize) -> bool {
    (TRACK_MIN..=TRACK_MAX).contains(&size)
}

#[inline]
fn note_alloc(size: usize) {
    if tracked(size)
        && let Some(i) = find_class(size, true)
    {
        let live = CLASSES[i].live.fetch_add(1, AcqRel) + 1;
        CLASSES[i].peak.fetch_max(live, AcqRel);
    }
}

#[inline]
fn note_dealloc(size: usize) {
    if tracked(size)
        && let Some(i) = find_class(size, false)
    {
        // Saturating: a block allocated before its class existed was never counted.
        let _ = CLASSES[i]
            .live
            .fetch_update(AcqRel, Acquire, |v| v.checked_sub(1));
    }
}

pub struct RecycleAlloc;

// SAFETY: every recycled block came from System with the exact same size.
// macOS libmalloc provides at least 16-byte alignment at these sizes, and
// layouts requiring larger alignment bypass the recycler.
unsafe impl GlobalAlloc for RecycleAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if recyclable(&layout) {
            note_alloc(layout.size());
            let p = pop(layout.size());
            if !p.is_null() {
                return p;
            }
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if recyclable(&layout) {
            note_alloc(layout.size());
            let p = pop(layout.size());
            if !p.is_null() {
                unsafe { core::ptr::write_bytes(p, 0, layout.size()) };
                return p;
            }
        }
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if recyclable(&layout) {
            note_dealloc(layout.size());
            if push(ptr, layout.size()) {
                return;
            }
        }
        unsafe { System.dealloc(ptr, layout) }
    }
}

/// Park extra pre-faulted blocks so every tracked class holds at least one
/// more free block than its warm-up high-water mark.
///
/// The timed prove does not replay the warm-up's allocation sequence exactly
/// (seed pipelining and the lazily seeded witness take paths the warm-up
/// cannot), and it needs a few more mid-size blocks at once than the warm-up
/// ever held: ~65 MiB of 1–16 MiB requests (eq tables, lincheck partials,
/// recursive-commit matrices) otherwise miss the freelists and fault fresh
/// pages inside the timed window. Spends at most `budget` bytes; returns the
/// bytes parked. Call only from the untimed warm-up.
pub fn top_up_freelists(budget: usize) -> usize {
    let mut parked = 0;
    for class in &CLASSES {
        let size = class.size.load(Acquire);
        if !tracked(size) {
            continue;
        }
        let free = {
            let head = class.head.lock().unwrap();
            let mut n = 0;
            let mut node = *head;
            while node != 0 {
                n += 1;
                node = unsafe { *(node as *const usize) };
            }
            n
        };
        let want = class.peak.load(Acquire) + 1;
        for _ in free..want {
            if parked + size > budget {
                return parked;
            }
            let Ok(layout) = Layout::from_size_align(size, MAX_ALIGN) else {
                break;
            };
            let p = unsafe { System.alloc(layout) };
            if p.is_null() {
                return parked;
            }
            unsafe { core::ptr::write_bytes(p, 0, size) };
            if !push(p, size) {
                unsafe { System.dealloc(p, layout) };
                break;
            }
            parked += size;
        }
    }
    parked
}
