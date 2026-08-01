//! Recycling global allocator for the prover process.
//!
//! Blocks at least 32 KiB are parked on exact-size freelists rather than
//! returned to libmalloc. The ranked worker performs an untimed warm proof
//! with the same allocation pattern, so the timed proof reuses resident pages
//! for large allocations not already handled by the typed scratch pools.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{
    AtomicBool, AtomicUsize,
    Ordering::{AcqRel, Acquire, Release},
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

/// One-shot escape hatch for an owned warmup-only allocation. The exact
/// pointer, rather than its size class, identifies the sole deallocation that
/// must bypass the recycler. Unrelated concurrent frees keep their ordinary
/// behavior, and the Vec's normal Drop supplies its original allocation
/// layout to `GlobalAlloc::dealloc`.
static RELEASE_TO_SYSTEM_PTR: AtomicUsize = AtomicUsize::new(0);
/// A ranked worker has one warmup boundary, so the uncached release is a
/// process-wide one-shot. CAS makes accidental/concurrent rearming fail before
/// it can interfere with the pointer token's consume assertion.
static RELEASE_ATTEMPTED: AtomicBool = AtomicBool::new(false);

#[inline]
fn class_slot(size: usize) -> usize {
    (size.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 55) % MAX_CLASSES
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

fn with_one_shot_system_release(ptr: usize, deallocate: impl FnOnce()) {
    assert_ne!(ptr, 0, "cannot arm a null allocator-release token");
    RELEASE_ATTEMPTED
        .compare_exchange(false, true, AcqRel, Acquire)
        .expect("allocator-release one-shot already used");
    RELEASE_TO_SYSTEM_PTR
        .compare_exchange(0, ptr, AcqRel, Acquire)
        .expect("allocator-release token already armed");
    deallocate();
    assert_eq!(
        RELEASE_TO_SYSTEM_PTR.load(Acquire),
        0,
        "allocator-release token was not consumed by the target Vec drop"
    );
}

/// Return one owned F128 allocation all the way to `System` instead of
/// parking it on this allocator's size-only freelist. The ordinary Vec drop is
/// essential: it reaches [`GlobalAlloc::dealloc`] with the Vec's actual
/// allocation layout. This escape hatch is macOS-only together with the
/// process global allocator defined in `lib.rs`.
pub(crate) fn release_f128_to_system(v: Vec<flock_core::field::F128>) {
    if v.capacity() == 0 {
        return;
    }
    let ptr = v.as_ptr() as usize;
    with_one_shot_system_release(ptr, || drop(v));
}

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
        if RELEASE_TO_SYSTEM_PTR
            .compare_exchange(ptr as usize, 0, AcqRel, Acquire)
            .is_ok()
        {
            unsafe { System.dealloc(ptr, layout) };
            return;
        }
        if recyclable(&layout) && push(ptr, layout.size()) {
            return;
        }
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn freelist_len(size: usize) -> usize {
        let Some(i) = find_class(size, false) else {
            return 0;
        };
        let head = CLASSES[i].head.lock().unwrap();
        let mut next = *head;
        let mut len = 0usize;
        while next != 0 {
            len += 1;
            // SAFETY: every freelist node is a live recycled allocation whose
            // first word stores the next node, protected by this class lock.
            next = unsafe { *(next as *const usize) };
        }
        len
    }

    #[test]
    fn ordinary_vec_drop_consumes_one_shot_without_freelist_push() {
        // Odd exact size avoids unrelated libtest allocations sharing this
        // recycler class. The crate's test binary installs the same global
        // RecycleAlloc as production, so `drop(v)` exercises the real path.
        let v = vec![flock_core::field::F128::ZERO; 4_099];
        let size = v.capacity() * core::mem::size_of::<flock_core::field::F128>();
        assert!(size >= RECYCLE_MIN);
        let before = freelist_len(size);

        release_f128_to_system(v);

        assert_eq!(RELEASE_TO_SYSTEM_PTR.load(Acquire), 0);
        assert!(RELEASE_ATTEMPTED.load(Acquire));
        assert_eq!(freelist_len(size), before);
    }
}
