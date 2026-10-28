use crate::field::F128;

pub const LOG_PACKING: usize = 7;

pub const PACKING_WIDTH: usize = 1 << LOG_PACKING;

pub fn pack_witness(z: &[bool], m: usize) -> Vec<F128> {
    use rayon::prelude::*;
    assert_eq!(z.len(), 1usize << m, "z length must be 2^m");
    assert!(
        m >= LOG_PACKING,
        "witness too small to pack: m = {m} < LOG_PACKING = {LOG_PACKING}",
    );
    let n_packed = 1usize << (m - LOG_PACKING);

    let bytes: &[u8] = unsafe { core::slice::from_raw_parts(z.as_ptr() as *const u8, z.len()) };
    #[inline]
    fn pack64(b: &[u8]) -> u64 {
        let mut w = 0u64;
        for (i, ch) in b.as_chunks::<8>().0.iter().enumerate() {
            let x = u64::from_le_bytes(*ch);
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            let packed_byte = unsafe { crate::gf2_kernels::gf2_kern_pack8_bool_bytes(x) };
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            let packed_byte = x.wrapping_mul(0x0102_0408_1020_4080) >> 56;
            w |= packed_byte << (8 * i);
        }
        w
    }
    let one = |i_rest: usize| {
        let base = i_rest << LOG_PACKING;
        F128 {
            lo: pack64(&bytes[base..base + 64]),
            hi: pack64(&bytes[base + 64..base + 128]),
        }
    };
    if n_packed >= (1 << 12) {
        (0..n_packed).into_par_iter().map(one).collect()
    } else {
        (0..n_packed).map(one).collect()
    }
}

pub fn unpack_witness(packed: &[F128], m: usize) -> Vec<bool> {
    let n_packed = 1usize << (m - LOG_PACKING);
    assert_eq!(
        packed.len(),
        n_packed,
        "packed length must be 2^(m - LOG_PACKING)"
    );
    let mut out = vec![false; 1usize << m];
    for (i_rest, elem) in packed.iter().enumerate() {
        let base = i_rest << LOG_PACKING;
        for b in 0..8 {
            let byte_lo = (elem.lo >> (8 * b)) & 0xFF;
            let byte_hi = (elem.hi >> (8 * b)) & 0xFF;
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            let (u_lo, u_hi) = unsafe {
                (
                    crate::gf2_kernels::gf2_kern_unpack8_bool_bytes(byte_lo),
                    crate::gf2_kernels::gf2_kern_unpack8_bool_bytes(byte_hi),
                )
            };
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            let (u_lo, u_hi) = (
                (((byte_lo & 0x7F).wrapping_mul(0x0000_0408_1020_4081)) | ((byte_lo & 0x80) << 49))
                    & 0x0101_0101_0101_0101,
                (((byte_hi & 0x7F).wrapping_mul(0x0000_0408_1020_4081)) | ((byte_hi & 0x80) << 49))
                    & 0x0101_0101_0101_0101,
            );
            unsafe {
                let dst = out.as_mut_ptr().cast::<u8>();
                core::ptr::copy_nonoverlapping(
                    u_lo.to_le_bytes().as_ptr(),
                    dst.add(base + 8 * b),
                    8,
                );
                core::ptr::copy_nonoverlapping(
                    u_hi.to_le_bytes().as_ptr(),
                    dst.add(base + 64 + 8 * b),
                    8,
                );
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn bits(&mut self, n: usize) -> Vec<bool> {
            (0..n).map(|_| self.next_u64() & 1 == 1).collect()
        }
    }

    #[test]
    fn pack_unpack_roundtrip() {
        let mut rng = Rng::new(0xC0FFEE);
        for m in [7usize, 8, 10, 12, 14] {
            let z = rng.bits(1 << m);
            let packed = pack_witness(&z, m);
            assert_eq!(packed.len(), 1 << (m - LOG_PACKING));
            let z_back = unpack_witness(&packed, m);
            assert_eq!(z, z_back, "roundtrip failed at m={m}");
        }
    }

    #[test]
    fn pack_layout_matches_natural_bit_order() {
        let mut z = vec![false; 128];
        for &i in &[0usize, 1, 5, 63, 64, 127] {
            z[i] = true;
        }
        let packed = pack_witness(&z, LOG_PACKING);
        assert_eq!(packed.len(), 1);
        let expected = F128 {
            lo: (1u64 << 0) | (1u64 << 1) | (1u64 << 5) | (1u64 << 63),
            hi: (1u64 << 0) | (1u64 << 63),
        };
        assert_eq!(packed[0], expected);
    }

    #[test]
    fn pack_independent_chunks() {
        let z = vec![true; 256];
        let packed = pack_witness(&z, 8);
        assert_eq!(packed.len(), 2);
        assert_eq!(
            packed[0],
            F128 {
                lo: u64::MAX,
                hi: u64::MAX
            }
        );
        assert_eq!(
            packed[1],
            F128 {
                lo: u64::MAX,
                hi: u64::MAX
            }
        );
    }

    #[test]
    #[should_panic(expected = "witness too small")]
    fn rejects_undersized_witness() {
        let z = vec![false; 64];
        let _ = pack_witness(&z, 6);
    }
}
