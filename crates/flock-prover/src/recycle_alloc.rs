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
    Ordering::{Acquire, Release},
};

const RECYCLE_MIN: usize = 32 * 1024;
const MAX_ALIGN: usize = 16;
const LARGE_MIN: usize = 1024 * 1024;
const CLASSES_PER_TIER: usize = 256;

struct Class {
    size: AtomicUsize,
    head: Mutex<usize>,
}

#[allow(clippy::declare_interior_mutable_const)]
const EMPTY: Class = Class {
    size: AtomicUsize::new(0),
    head: Mutex::new(0),
};

// Small, exact-size proof allocations can fill the table before the first
// large working buffer is freed. Reserve half of the existing 512 slots for
// sizes >= 1 MiB, so small-size churn cannot evict those buffers from reuse.
// Each tier still falls back to System when full; no sizes are rounded and no
// memory is allocated merely to populate a freelist.
struct ClassTables {
    small: [Class; CLASSES_PER_TIER],
    large: [Class; CLASSES_PER_TIER],
}

impl ClassTables {
    const fn new() -> Self {
        Self {
            small: [EMPTY; CLASSES_PER_TIER],
            large: [EMPTY; CLASSES_PER_TIER],
        }
    }

    #[inline]
    fn find_class(&self, size: usize, insert: bool) -> Option<&Class> {
        let classes = if size >= LARGE_MIN {
            &self.large
        } else {
            &self.small
        };
        let start = class_slot(size);
        for probe in 0..CLASSES_PER_TIER {
            let class = &classes[(start + probe) % CLASSES_PER_TIER];
            let s = class.size.load(Acquire);
            if s == size {
                return Some(class);
            }
            if s == 0 {
                if !insert {
                    return None;
                }
                if class
                    .size
                    .compare_exchange(0, size, Release, Acquire)
                    .is_ok()
                {
                    return Some(class);
                }
                if class.size.load(Acquire) == size {
                    return Some(class);
                }
            }
        }
        None
    }
}

static CLASSES: ClassTables = ClassTables::new();

#[inline]
fn class_slot(size: usize) -> usize {
    size.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (usize::BITS - CLASSES_PER_TIER.trailing_zeros())
}

#[inline]
fn recyclable(layout: &Layout) -> bool {
    layout.size() >= RECYCLE_MIN && layout.align() <= MAX_ALIGN
}

#[inline]
fn pop(size: usize) -> *mut u8 {
    let Some(class) = CLASSES.find_class(size, false) else {
        return core::ptr::null_mut();
    };
    let mut head = class.head.lock().unwrap();
    let top = *head;
    if top == 0 {
        return core::ptr::null_mut();
    }
    *head = unsafe { *(top as *const usize) };
    top as *mut u8
}

#[inline]
fn push(ptr: *mut u8, size: usize) -> bool {
    let Some(class) = CLASSES.find_class(size, true) else {
        return false;
    };
    let mut head = class.head.lock().unwrap();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_class_saturation_preserves_large_slots() {
        let tables = ClassTables::new();
        for i in 0..CLASSES_PER_TIER {
            assert!(tables.find_class(RECYCLE_MIN + 16 * i, true).is_some());
        }
        assert!(tables.find_class(LARGE_MIN - 16, true).is_none());
        for size in [LARGE_MIN, 4 << 20, 16 << 20, 64 << 20] {
            let inserted = tables.find_class(size, true).unwrap();
            assert_eq!(inserted.size.load(Acquire), size);
            assert!(std::ptr::eq(
                inserted,
                tables.find_class(size, false).unwrap()
            ));
        }
        for i in 0..CLASSES_PER_TIER {
            assert!(tables.find_class(RECYCLE_MIN + 16 * i, false).is_some());
        }
    }

    #[test]
    fn large_class_saturation_preserves_small_slots() {
        let tables = ClassTables::new();
        for i in 0..CLASSES_PER_TIER {
            assert!(tables.find_class(LARGE_MIN + 16 * i, true).is_some());
        }
        assert!(tables.find_class(64 << 20, true).is_none());
        for size in [RECYCLE_MIN, LARGE_MIN - 16, LARGE_MIN - 1] {
            assert!(tables.find_class(size, false).is_none());
            let class = tables.find_class(size, true).unwrap();
            assert_eq!(class.size.load(Acquire), size);
        }
        // Layouts that cannot safely be recycled keep using System.
        assert!(!recyclable(
            &Layout::from_size_align(RECYCLE_MIN - 1, 16).unwrap()
        ));
        assert!(!recyclable(
            &Layout::from_size_align(LARGE_MIN, 32).unwrap()
        ));
        assert!(recyclable(&Layout::from_size_align(LARGE_MIN, 16).unwrap()));
    }

    #[test]
    fn concurrent_insert_uses_one_class_per_exact_size() {
        let tables = ClassTables::new();
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    let tables = &tables;
                    scope.spawn(move || {
                        for i in 0..128 {
                            tables.find_class(LARGE_MIN + 16 * i, true).unwrap();
                        }
                        tables.find_class(LARGE_MIN, false).unwrap() as *const Class as usize
                    })
                })
                .collect();
            let expected = handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>();
            assert!(expected.iter().all(|address| *address == expected[0]));
        });
        assert_eq!(
            tables
                .large
                .iter()
                .filter(|c| c.size.load(Acquire) != 0)
                .count(),
            128
        );
    }

    #[test]
    fn recycled_large_block_is_zeroed_and_aligned() {
        let layout = Layout::from_size_align(LARGE_MIN + 16, MAX_ALIGN).unwrap();
        // The test harness retains System as its global allocator. Exercise
        // RecycleAlloc directly, then return the popped block to System.
        unsafe {
            let original = RecycleAlloc.alloc(layout);
            assert!(!original.is_null());
            core::ptr::write_bytes(original, 0xa5, layout.size());
            RecycleAlloc.dealloc(original, layout);
            let recycled = RecycleAlloc.alloc_zeroed(layout);
            assert_eq!(original, recycled);
            assert_eq!(recycled as usize % MAX_ALIGN, 0);
            assert!(
                std::slice::from_raw_parts(recycled, layout.size())
                    .iter()
                    .all(|b| *b == 0)
            );
            System.dealloc(recycled, layout);
        }
    }
}
