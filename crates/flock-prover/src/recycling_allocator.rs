//! Warm-proof recycling allocator for medium transient allocations.
//!
//! The ranked worker performs an untimed proof and then a timed proof in the
//! same process.  Flock's explicit scratch pools retain the very large field
//! buffers, but the prover still creates many 4 KiB--16 MiB temporary vectors
//! outside those pools.  Keeping a small, exact size-class freelist lets the
//! timed proof reuse the warm proof's mappings without changing any caller.

use core::alloc::{GlobalAlloc, Layout};
use core::ffi::c_char;
use core::ptr;
use std::alloc::System;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, Ordering};

const MIN_SHIFT: usize = 12; // 4 KiB: smaller allocations are cheap in malloc.
const MAX_SHIFT: usize = 24; // 16 MiB: larger F128 buffers use scratch.rs.
const N_BINS: usize = MAX_SHIFT - MIN_SHIFT + 1;
const MAX_PER_BIN: usize = 8;
const CACHE_ALIGN: usize = 64;

#[derive(Clone, Copy)]
struct BinState {
    head: usize,
    len: usize,
}

struct Bin(Mutex<BinState>);

static BINS: [Bin; N_BINS] = [const { Bin(Mutex::new(BinState { head: 0, len: 0 })) }; N_BINS];

const MODE_UNKNOWN: u8 = 0;
const MODE_ENABLED: u8 = 1;
const MODE_DISABLED: u8 = 2;
static MODE: AtomicU8 = AtomicU8::new(MODE_UNKNOWN);

unsafe extern "C" {
    fn getenv(name: *const c_char) -> *mut c_char;
}

/// Resolve the process-level gate without allocating. Calling
/// `std::env::var_os` from `GlobalAlloc` would recurse into this allocator.
#[inline]
fn enabled() -> bool {
    match MODE.load(Ordering::Acquire) {
        MODE_ENABLED => true,
        MODE_DISABLED => false,
        _ => {
            // SAFETY: the name is static and NUL-terminated. The benchmark
            // does not mutate its environment after process launch.
            let disabled = unsafe { !getenv(c"FLOCK_NO_RECYCLING_ALLOCATOR".as_ptr()).is_null() };
            let mode = if disabled {
                MODE_DISABLED
            } else {
                MODE_ENABLED
            };
            MODE.store(mode, Ordering::Release);
            mode == MODE_ENABLED
        }
    }
}

pub(crate) struct RecyclingAllocator;

#[inline]
fn class(layout: Layout) -> Option<(usize, Layout)> {
    let size = layout.size().max(1);
    let class_size = size.checked_next_power_of_two()?;
    let shift = class_size.trailing_zeros() as usize;
    if !(MIN_SHIFT..=MAX_SHIFT).contains(&shift) {
        return None;
    }
    // Every allocation in a size bin uses one common alignment.  Keying only
    // by size while preserving the caller's alignment would be unsound: a
    // later over-aligned request could otherwise receive an under-aligned
    // cached block.
    if layout.align() > CACHE_ALIGN {
        return None;
    }
    let class_layout = Layout::from_size_align(class_size, CACHE_ALIGN).ok()?;
    Some((shift - MIN_SHIFT, class_layout))
}

impl RecyclingAllocator {
    #[inline]
    unsafe fn alloc_inner(&self, layout: Layout) -> *mut u8 {
        if enabled()
            && let Some((bin_index, class_layout)) = class(layout)
        {
            let mut state = match BINS[bin_index].0.lock() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            if state.head != 0 {
                let out = state.head as *mut u8;
                // SAFETY: cached blocks are at least 4 KiB and suitably
                // aligned; dealloc_inner stored the next pointer here while
                // holding this bin's mutex.
                state.head = unsafe { out.cast::<usize>().read() };
                state.len -= 1;
                return out;
            }
            drop(state);
            // SAFETY: class_layout is valid and belongs to System.
            return unsafe { System.alloc(class_layout) };
        }
        // SAFETY: caller supplied a valid GlobalAlloc layout.
        unsafe { System.alloc(layout) }
    }

    #[inline]
    unsafe fn dealloc_inner(&self, ptr: *mut u8, layout: Layout) {
        if enabled()
            && let Some((bin_index, class_layout)) = class(layout)
        {
            let mut state = match BINS[bin_index].0.lock() {
                Ok(guard) => guard,
                Err(poisoned) => poisoned.into_inner(),
            };
            if state.len < MAX_PER_BIN {
                // SAFETY: the allocation uses class_layout, is at least one
                // usize long, and is exclusively owned at deallocation.
                unsafe { ptr.cast::<usize>().write(state.head) };
                state.head = ptr as usize;
                state.len += 1;
                return;
            }
            drop(state);
            // SAFETY: alloc_inner used this same derived layout.
            unsafe { System.dealloc(ptr, class_layout) };
            return;
        }
        // SAFETY: non-recycled allocations were allocated with this layout.
        unsafe { System.dealloc(ptr, layout) }
    }
}

// SAFETY: every cached allocation is stored in exactly one mutex-protected
// size class and is returned only for the identical derived System layout.
unsafe impl GlobalAlloc for RecyclingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarded GlobalAlloc contract.
        unsafe { self.alloc_inner(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarded GlobalAlloc contract.
        let out = unsafe { self.alloc_inner(layout) };
        if !out.is_null() {
            // SAFETY: out covers at least layout.size() bytes.
            unsafe { ptr::write_bytes(out, 0, layout.size()) };
        }
        out
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarded GlobalAlloc contract.
        unsafe { self.dealloc_inner(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, old: Layout, new_size: usize) -> *mut u8 {
        let Ok(new_layout) = Layout::from_size_align(new_size, old.align()) else {
            return ptr::null_mut();
        };
        // SAFETY: forwarded GlobalAlloc contract.
        let new_ptr = unsafe { self.alloc_inner(new_layout) };
        if new_ptr.is_null() {
            return ptr::null_mut();
        }
        // SAFETY: both allocations are valid for the copied minimum length
        // and GlobalAlloc::realloc requires them not to overlap.
        unsafe { ptr::copy_nonoverlapping(ptr, new_ptr, old.size().min(new_size)) };
        // SAFETY: the old allocation is no longer used after the copy.
        unsafe { self.dealloc_inner(ptr, old) };
        new_ptr
    }
}
