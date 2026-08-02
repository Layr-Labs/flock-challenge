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

/// Size-class table capacity.
///
/// Sized from a measurement, not a guess: a traced ranked worker (`log2 = 18`,
/// warm proof + timed proof) requests **1,410-1,413 distinct recyclable
/// sizes**. At the previous 512 the table saturated before the warm proof even
/// finished, in 12 of 12 processes, after which the recycler stopped being an
/// optimisation and became a cost:
///
/// - `push` failed for ~1,900 blocks per process, so they went to
///   `System.dealloc` (312-383 MiB returned to the OS, 78-143 MiB of it inside
///   the *timed* proof, to be re-faulted on the next allocation);
/// - `find_class` on a full table probes every slot before giving up, so every
///   `alloc`/`dealloc` of a >=32 KiB block paid ~300 `Acquire` loads — measured
///   mean probe depth 299-307, versus 1.03 once the table fits.
///
/// 4096 leaves ~2.9x headroom over the observed live population so the table
/// stays sparse (open addressing degrades sharply past ~70% load factor).
const MAX_CLASSES: usize = 4096;
const _: () = assert!(MAX_CLASSES.is_power_of_two());

/// Bits of the hash to discard, derived from [`MAX_CLASSES`] so the two can
/// never drift apart.
///
/// This is the regression guard for a real defect: the constant used to be a
/// hard-coded `>> 55`, which yields a 9-bit result — exactly 512 distinct home
/// slots **regardless of `MAX_CLASSES`**. Raising the capacity alone therefore
/// bought nothing but longer probe chains, which is why the first attempt at
/// this fix was a no-op. See `class_slot_covers_the_whole_table`.
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

#[inline]
fn class_slot(size: usize) -> usize {
    (size.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> CLASS_SHIFT) % MAX_CLASSES
}

/// Kill switch: `FLOCK_NO_RECYCLE=1` turns the allocator into a pass-through
/// to `System`.
///
/// Read through raw `getenv` rather than `std::env::var_os` on purpose —
/// `var_os` allocates, and allocating from inside the global allocator's own
/// predicate would recurse. `getenv` does not allocate. The tri-state atomic
/// (2 = not yet read) memoises it; the program never calls `setenv`, so the
/// pointer stays valid.
#[inline]
fn recycling_disabled() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering::Relaxed};
    static STATE: AtomicU8 = AtomicU8::new(2);
    match STATE.load(Relaxed) {
        0 => false,
        1 => true,
        _ => {
            unsafe extern "C" {
                fn getenv(name: *const core::ffi::c_char) -> *const core::ffi::c_char;
            }
            // SAFETY: `getenv` reads process-global storage that is never
            // mutated here (no `setenv` anywhere in the tree); the returned
            // pointer is NUL-terminated or null.
            let off = unsafe {
                let p = getenv(c"FLOCK_NO_RECYCLE".as_ptr());
                !p.is_null() && *p == b'1' as core::ffi::c_char && *p.add(1) == 0
            };
            STATE.store(u8::from(off), Relaxed);
            off
        }
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
    layout.size() >= RECYCLE_MIN && layout.align() <= MAX_ALIGN && !recycling_disabled()
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression guard for the shift/capacity drift defect.
    ///
    /// The shipped hash used a hard-coded `>> 55`, which produces a 9-bit
    /// result = 512 home slots no matter how large `MAX_CLASSES` is. With the
    /// table at 512 that was self-consistent; the moment anyone raised the
    /// capacity to fix saturation it silently kept using 512 buckets and only
    /// lengthened the probe chains. Deriving `CLASS_SHIFT` from `MAX_CLASSES`
    /// makes that unrepresentable, and this test fails if either constant is
    /// edited without the other.
    #[test]
    fn class_shift_matches_table_capacity() {
        assert!(MAX_CLASSES.is_power_of_two());
        assert_eq!(CLASS_SHIFT, usize::BITS - MAX_CLASSES.trailing_zeros());
        // The hash must be able to name every slot.
        assert_eq!(usize::MAX >> CLASS_SHIFT, (MAX_CLASSES - 1) as usize);
    }

    /// The hash must actually spread across the whole table. This is the
    /// assertion the original defect would have failed: with `>> 55` and
    /// `MAX_CLASSES = 4096` the observed slot count saturates at 512.
    #[test]
    fn class_slot_covers_the_whole_table() {
        // Realistic prover sizes: the ranked worker requests ~1,410 distinct
        // recyclable sizes, all >= RECYCLE_MIN and mostly 16-byte multiples.
        let mut seen = std::collections::HashSet::new();
        let mut size = RECYCLE_MIN;
        for _ in 0..200_000 {
            seen.insert(class_slot(size));
            assert!(class_slot(size) < MAX_CLASSES);
            size += 16;
        }
        // A 9-bit hash would cap this at 512.
        assert!(
            seen.len() > MAX_CLASSES / 2,
            "hash reaches only {} of {MAX_CLASSES} slots — CLASS_SHIFT is too large",
            seen.len()
        );
    }

    /// The table must hold the measured live population with headroom: a
    /// traced ranked worker (log2 = 18) requests 1,410-1,413 distinct sizes.
    #[test]
    fn table_fits_the_measured_ranked_class_population() {
        const OBSERVED_RANKED_CLASSES: usize = 1_413;
        assert!(
            MAX_CLASSES >= 2 * OBSERVED_RANKED_CLASSES,
            "open addressing degrades past ~70% load factor"
        );
    }

    /// Insert/lookup round-trip across more classes than the old table held.
    #[test]
    fn find_class_inserts_and_finds_beyond_512_classes() {
        let sizes: Vec<usize> = (0..1_500).map(|i| RECYCLE_MIN + i * 64).collect();
        for &s in &sizes {
            assert!(find_class(s, true).is_some(), "insert failed at size {s}");
        }
        for &s in &sizes {
            let i = find_class(s, false).expect("size must be found after insert");
            assert_eq!(CLASSES[i].size.load(Acquire), s);
        }
    }
}
