//! Packed even rows and adjacent-row XORs, produced before challenges exist.
use crate::field::F128;
use std::sync::{Arc, Mutex, OnceLock};

pub fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("FLOCK_NO_DELTA_PLANES").as_deref()
            != Some(std::ffi::OsStr::new("1"))
    })
}

/// A separate reservation class: these bytes are not ordinary B constants.
pub fn b_storage_tag(block_log: usize) -> u64 {
    0xD317_A000_0000_0000 | block_log as u64
}

#[inline(always)]
pub(crate) fn decode<const PLANES: bool>(x0: u64, x1: u64, y0: u64, y1: u64)
    -> (u64, u64, u64, u64) {
    if PLANES {
        (x0, x0 ^ y0, x1, x1 ^ y1)
    } else {
        (x0, x1, y0, y1)
    }
}

/// Publish two fully initialized, densely interleaved planes.
///
/// # Safety
/// All four ranges contain `len` bytes, `len` is a multiple of 16, and the
/// two destinations are exclusive and disjoint from both initialized inputs.
#[inline]
pub unsafe fn publish(a: *const u8, b: *const u8, even: *mut u8, delta: *mut u8, len: usize) {
    assert!(len.is_multiple_of(16));
    unsafe {
        for i in (0..len).step_by(16) {
            let a0 = a.add(i).cast::<u64>().read_unaligned();
            let a1 = a.add(i + 8).cast::<u64>().read_unaligned();
            let b0 = b.add(i).cast::<u64>().read_unaligned();
            let b1 = b.add(i + 8).cast::<u64>().read_unaligned();
            #[cfg(target_arch = "aarch64")]
            core::arch::asm!(
                "stnp {a}, {b}, [{e}]", "stnp {da}, {db}, [{d}]",
                a=in(reg) a0, b=in(reg) b0, da=in(reg) (a0 ^ a1), db=in(reg) (b0 ^ b1),
                e=in(reg) even.add(i), d=in(reg) delta.add(i),
                options(nostack, preserves_flags));
            #[cfg(not(target_arch = "aarch64"))]
            {
                even.add(i).cast::<u64>().write_unaligned(a0);
                even.add(i+8).cast::<u64>().write_unaligned(b0);
                delta.add(i).cast::<u64>().write_unaligned(a0 ^ a1);
                delta.add(i+8).cast::<u64>().write_unaligned(b0 ^ b1);
            }
        }
    }
}

/// The packed owner returns to its caller after the compact consumer joins.
/// The caller then releases it at the same point as its ordinary packed B.
pub struct Owner {
    pub(crate) storage: Vec<F128>,
    returned: Arc<Mutex<Option<Vec<F128>>>>,
}
pub struct Return(Arc<Mutex<Option<Vec<F128>>>>);
impl Owner {
    pub fn new(storage: Vec<F128>) -> (Self, Return) {
        let returned = Arc::new(Mutex::new(None));
        (Self { storage, returned: returned.clone() }, Return(returned))
    }
    pub(crate) fn recycle(self) {
        *self.returned.lock().unwrap() = Some(self.storage);
    }
}
impl Return {
    pub fn take(self) -> Vec<F128> {
        self.0.lock().unwrap().take().expect("compact delta reader has not joined")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn delta_planes_layout_basis_and_roundtrip() {
        for bit in 0..256 {
            let mut a = [0u64; 2];
            let mut b = [0u64; 2];
            if bit < 128 { a[bit / 64] = 1 << (bit % 64); }
            else { b[(bit - 128) / 64] = 1 << (bit % 64); }
            let mut e = [0u64; 2];
            let mut d = [0u64; 2];
            unsafe { publish(a.as_ptr().cast(), b.as_ptr().cast(), e.as_mut_ptr().cast(), d.as_mut_ptr().cast(), 16); }
            assert_eq!(decode::<true>(e[0], e[1], d[0], d[1]), (a[0], a[1], b[0], b[1]));
        }
    }
}

/// Aggregate CPU wall/chunk for full work and for GPU-owned products.
/// Balance g*q = (N-g)*f + g*l through the actual heterogeneous queue.
pub(crate) fn gate_share(q: f64, f: f64, l: f64, n: usize) -> usize {
    if !q.is_finite() || q <= 0.0 || !f.is_finite() || f <= 0.0
        || !l.is_finite() || l < 0.0 || l >= f
    {
        return 0;
    }
    ((n as f64 * f / (q + f - l)).round() as usize).min(n)
}
