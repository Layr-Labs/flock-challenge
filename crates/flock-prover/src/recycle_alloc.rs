//! Recycling global allocator for the prover process.
//!
//! Blocks at least 32 KiB are parked on exact-size freelists rather than
//! returned to the system allocator. The ranked worker performs an untimed warm proof
//! with the same allocation pattern, so the timed proof reuses resident pages
//! for large allocations not already handled by the typed scratch pools.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{
    AtomicUsize,
    Ordering::{Acquire, Release},
};

const RECYCLE_MIN: usize = 32 * 1024; // ranked value (arm A/B/C comparisons pin their own builds)
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
    // MEASUREMENT-ONLY latch: FLOCK_NO_RECYCLE=1 forces every block to System.
    // Order matters: size/align are tested FIRST so env reads (whose own small
    // allocations land here) can never re-enter the latch.
    if layout.size() < RECYCLE_MIN || layout.align() > MAX_ALIGN {
        return false;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_RECYCLE").is_none())
}

/// `FLOCK_NO_ALIGN64=1` restores raw System pointers for the recyclable
/// class (exact same-binary A/B). Latched once, and only ever initialized
/// from a RECYCLABLE allocation — `var_os`'s own small allocations take the
/// non-recyclable branch straight to System, so initialization cannot
/// re-enter this latch.
#[inline]
fn align64_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FLOCK_NO_ALIGN64").is_none())
}

/// Every large prover buffer historically landed `16 mod 64` (glibc mmap
/// chunk header), which the hot kernels pay for three ways: every wide
/// load/store on these buffers is a cache-line split, non-temporal 64-byte
/// writes straddle two lines (partial write-combining flushes are DRAM
/// read-modify-writes), and single-uop ZMM streaming stores are illegal.
/// For the recyclable class, over-allocate by `ALIGN_SLACK` and return
/// `round_up(base + 8, 64)`: a 64-aligned pointer with room for the
/// back-offset word stored at `aligned - 8` — BEFORE the block, so it can
/// never collide with the freelist link word at `aligned + 0`.
const ALIGN_SLACK: usize = 64;

#[inline]
fn adjusted(layout: &Layout) -> Layout {
    // SAFETY of unwrap: size + 64 cannot overflow isize for any layout the
    // caller could have constructed, and 16 is a power of two.
    Layout::from_size_align(layout.size() + ALIGN_SLACK, MAX_ALIGN).unwrap()
}

#[inline]
unsafe fn align_up(base: *mut u8) -> *mut u8 {
    if base.is_null() {
        return base;
    }
    let aligned = ((base as usize + 8 + 63) & !63) as *mut u8;
    // SAFETY: aligned - base is in [8, 64] ⊂ the ALIGN_SLACK the caller
    // over-allocated, and aligned - 8 >= base.
    unsafe { *(aligned.sub(8) as *mut usize) = aligned as usize - base as usize };
    aligned
}

#[inline]
unsafe fn align_base(ptr: *mut u8) -> *mut u8 {
    // SAFETY: ptr was produced by align_up, so the offset word at ptr - 8 is
    // intact (freelist links live at ptr + 0 and never touch it).
    let off = unsafe { *(ptr.sub(8) as *const usize) };
    debug_assert!((8..=ALIGN_SLACK).contains(&off));
    unsafe { ptr.sub(off) }
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

// SAFETY: every recycled block came from this allocator with the exact same
// size class. With align64 on, every recyclable-class block System sees uses
// the `adjusted` layout on both alloc and dealloc, and the user pointer is
// recovered to its System base via the offset word `align_up` stored.
// glibc/mimalloc and the macOS allocator provide at least 16-byte alignment
// at these sizes; layouts requiring larger alignment bypass the recycler.
unsafe impl GlobalAlloc for RecycleAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if recyclable(&layout) {
            let p = pop(layout.size());
            if !p.is_null() {
                return p;
            }
            if align64_enabled() {
                // SAFETY: adjusted() reserves the slack align_up consumes.
                return unsafe { align_up(System.alloc(adjusted(&layout))) };
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
            if align64_enabled() {
                // The user range [aligned, aligned + size) is inside the
                // zeroed System block; the offset word sits before it.
                // SAFETY: as for alloc.
                return unsafe { align_up(System.alloc_zeroed(adjusted(&layout))) };
            }
        }
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if recyclable(&layout) {
            if push(ptr, layout.size()) {
                return;
            }
            if align64_enabled() {
                // SAFETY: every recyclable-class pointer this allocator
                // handed out with align64 on came from align_up; recover the
                // System base and the adjusted layout it was allocated with.
                unsafe {
                    return System.dealloc(align_base(ptr), adjusted(&layout));
                }
            }
        }
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[cfg(test)]
mod tests {
    /// With `FLOCK_NO_ALIGN64` unset (the ranked worker's cleared env), every
    /// recyclable-class allocation — fresh from System, recycled off the
    /// freelist, and grown through realloc — returns a 64-aligned pointer
    /// with intact contents. (The switch-off arm can't be covered in the
    /// same process: the latch resolves once.)
    #[test]
    fn recyclable_class_is_64_aligned_and_roundtrips() {
        for i in 0..4usize {
            let n = 32 * 1024 + 4096 * i + 128;
            let v = vec![7u8; n];
            assert_eq!(v.as_ptr() as usize % 64, 0, "fresh alloc n={n}");
            drop(v);
            let v2 = vec![9u8; n];
            assert_eq!(v2.as_ptr() as usize % 64, 0, "recycled alloc n={n}");
            assert!(v2.iter().all(|&b| b == 9), "contents survive recycle n={n}");
            let z = vec![0u8; n + 64];
            assert_eq!(z.as_ptr() as usize % 64, 0, "alloc_zeroed n={}", n + 64);
            assert!(z.iter().all(|&b| b == 0), "zeroed contents n={}", n + 64);
        }
        // Growth path: realloc = alloc + copy + dealloc through this
        // allocator; contents must survive the move between size classes.
        let mut g: Vec<u8> = Vec::with_capacity(48 * 1024);
        g.extend(std::iter::repeat_n(0xA5u8, 48 * 1024));
        g.extend(std::iter::repeat_n(0x5Au8, 128 * 1024));
        assert_eq!(g.as_ptr() as usize % 64, 0, "grown alloc");
        assert!(g[..48 * 1024].iter().all(|&b| b == 0xA5));
        assert!(g[48 * 1024..].iter().all(|&b| b == 0x5A));
    }
}
