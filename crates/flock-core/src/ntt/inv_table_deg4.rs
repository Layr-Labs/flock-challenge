use crate::field::F8;
use crate::ntt::AdditiveNttGf8;

#[derive(Clone, Debug)]
pub struct InvNttTableSToV8Gf8 {
    pub k_in: usize,

    pub k_out: usize,

    pub ell_in: usize,

    pub ell_out: usize,

    pub n_chunks: usize,

    data: Vec<F8>,
}

impl InvNttTableSToV8Gf8 {
    pub fn new(ntt_s: &AdditiveNttGf8, ntt_v8: &AdditiveNttGf8) -> Self {
        let k_in = ntt_s.k();
        let k_out = ntt_v8.k();
        assert!(k_in <= k_out, "input k must be ≤ output k");
        assert!(k_in >= 3, "k_in must be ≥ 3 so n_chunks ≥ 1");
        let ell_in = 1usize << k_in;
        let ell_out = 1usize << k_out;
        let n_chunks = ell_in / 8;
        assert!(
            n_chunks <= 16,
            "n_chunks must fit the i'/chunk XOR encoding"
        );

        let mut data = vec![F8::ZERO; 256 * ell_out];

        let mut tmp_in = vec![F8::ZERO; ell_in];
        let mut tmp_out = vec![F8::ZERO; ell_out];
        let mut cols: Vec<Vec<F8>> = Vec::with_capacity(8);
        for t in 0..8 {
            tmp_in.iter_mut().for_each(|x| *x = F8::ZERO);
            tmp_in[t] = F8::ONE;
            ntt_s.inverse(&mut tmp_in);

            tmp_out.iter_mut().for_each(|x| *x = F8::ZERO);
            tmp_out[..ell_in].copy_from_slice(&tmp_in);
            ntt_v8.forward(&mut tmp_out);
            cols.push(tmp_out.clone());
        }

        for t in 0..8 {
            let entry_start = (1usize << t) * ell_out;
            data[entry_start..entry_start + ell_out].copy_from_slice(&cols[t]);
        }
        for w in 3usize..256 {
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            let (lo_bit, parent) = unsafe {
                let lo = crate::gf2_kernels::gf2_kern_isolate_lowest_one(w as u64) as usize;
                let par = crate::gf2_kernels::gf2_kern_clear_lowest_one(w as u64) as usize;
                (lo, par)
            };
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            let (lo_bit, parent) = {
                let lo = w.isolate_lowest_one();
                (lo, w ^ lo)
            };
            if parent == 0 {
                continue;
            }
            let (parent_off, bit_off, entry_off) =
                (parent * ell_out, lo_bit * ell_out, w * ell_out);
            for i in 0..ell_out {
                let v = data[parent_off + i] + data[bit_off + i];
                data[entry_off + i] = v;
            }
        }

        Self {
            k_in,
            k_out,
            ell_in,
            ell_out,
            n_chunks,
            data,
        }
    }

    #[inline]
    pub fn data_ptr(&self) -> *const u8 {
        self.data.as_ptr() as *const u8
    }

    #[inline]
    pub fn apply_scalar(&self, bytes: &[u8], out: &mut [F8]) {
        assert_eq!(bytes.len(), self.n_chunks);
        assert_eq!(out.len(), self.ell_out);
        out.iter_mut().for_each(|x| *x = F8::ZERO);
        for (b, &byte_b) in bytes.iter().enumerate() {
            let row_off = byte_b as usize * self.ell_out;
            let row = &self.data[row_off..row_off + self.ell_out];
            let shift = 8 * b;
            for i in 0..self.ell_out {
                out[i] += row[i ^ shift];
            }
        }
    }

    #[inline]
    pub fn apply(&self, bytes: &[u8], out: &mut [F8]) {
        #[cfg(target_arch = "aarch64")]
        if self.ell_out >= 16 {
            unsafe { self.apply_neon_unchecked(bytes, out) };
            return;
        }
        self.apply_scalar(bytes, out);
    }

    #[cfg(target_arch = "aarch64")]
    pub unsafe fn apply_neon_unchecked(&self, bytes: &[u8], out: &mut [F8]) {
        use core::arch::aarch64::*;
        assert_eq!(bytes.len(), self.n_chunks);
        assert_eq!(out.len(), self.ell_out);
        let n128 = self.ell_out / 16;
        let base = self.data.as_ptr() as *const u8;
        let out_ptr = out.as_mut_ptr() as *mut u8;

        unsafe {
            let row0 = base.add(bytes[0] as usize * self.ell_out);
            for c in 0..n128 {
                vst1q_u8(out_ptr.add(c * 16), vld1q_u8(row0.add(c * 16)));
            }

            for b in 1..self.n_chunks {
                let b_high = b >> 1;
                let b_odd = (b & 1) != 0;
                let row_b = base.add(bytes[b] as usize * self.ell_out);
                if b_odd {
                    for c in 0..n128 {
                        let sc = c ^ b_high;
                        let v = vld1q_u8(row_b.add(sc * 16));
                        let v_swapped = vextq_u8::<8>(v, v);
                        let dst = out_ptr.add(c * 16);
                        vst1q_u8(dst, veorq_u8(vld1q_u8(dst), v_swapped));
                    }
                } else {
                    for c in 0..n128 {
                        let sc = c ^ b_high;
                        let v = vld1q_u8(row_b.add(sc * 16));
                        let dst = out_ptr.add(c * 16);
                        vst1q_u8(dst, veorq_u8(vld1q_u8(dst), v));
                    }
                }
            }
        }
    }
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
    }

    fn naive_extend(bytes: &[u8], ntt_s: &AdditiveNttGf8, ntt_v8: &AdditiveNttGf8) -> Vec<F8> {
        let ell_in = 1usize << ntt_s.k();
        let ell_out = 1usize << ntt_v8.k();
        let mut buf = vec![F8::ZERO; ell_out];
        for s in 0..ell_in {
            let bit = (bytes[s / 8] >> (s % 8)) & 1;
            buf[s] = F8(bit);
        }
        ntt_s.inverse(&mut buf[..ell_in]);

        ntt_v8.forward(&mut buf);
        buf
    }

    #[test]
    fn apply_matches_naive_random() {
        let ntt_s = AdditiveNttGf8::new(6, F8::ZERO);
        let ntt_v8 = AdditiveNttGf8::new(8, F8::ZERO);
        let table = InvNttTableSToV8Gf8::new(&ntt_s, &ntt_v8);

        let mut rng = Rng::new(0xC0FFEE);
        for _trial in 0..16 {
            let bytes: [u8; 8] = [
                rng.next_u64() as u8,
                rng.next_u64() as u8,
                rng.next_u64() as u8,
                rng.next_u64() as u8,
                rng.next_u64() as u8,
                rng.next_u64() as u8,
                rng.next_u64() as u8,
                rng.next_u64() as u8,
            ];
            let naive = naive_extend(&bytes, &ntt_s, &ntt_v8);
            let mut got = vec![F8::ZERO; 256];
            table.apply(&bytes, &mut got);
            assert_eq!(got, naive, "table.apply ≠ direct NTT on bytes {bytes:?}");
        }
    }

    #[test]
    fn apply_matches_apply_scalar() {
        let ntt_s = AdditiveNttGf8::new(6, F8::ZERO);
        let ntt_v8 = AdditiveNttGf8::new(8, F8::ZERO);
        let table = InvNttTableSToV8Gf8::new(&ntt_s, &ntt_v8);

        let mut rng = Rng::new(0xBADCAFE);
        for _trial in 0..16 {
            let bytes: [u8; 8] = [
                rng.next_u64() as u8,
                rng.next_u64() as u8,
                rng.next_u64() as u8,
                rng.next_u64() as u8,
                rng.next_u64() as u8,
                rng.next_u64() as u8,
                rng.next_u64() as u8,
                rng.next_u64() as u8,
            ];
            let mut got_scalar = vec![F8::ZERO; 256];
            let mut got_dispatch = vec![F8::ZERO; 256];
            table.apply_scalar(&bytes, &mut got_scalar);
            table.apply(&bytes, &mut got_dispatch);
            assert_eq!(
                got_scalar, got_dispatch,
                "apply (NEON or scalar) ≠ apply_scalar on bytes {bytes:?}"
            );
        }
    }

    #[test]
    fn first_64_lanes_match_input_bits() {
        let ntt_s = AdditiveNttGf8::new(6, F8::ZERO);
        let ntt_v8 = AdditiveNttGf8::new(8, F8::ZERO);
        let table = InvNttTableSToV8Gf8::new(&ntt_s, &ntt_v8);

        let mut rng = Rng::new(0xA17);
        for _trial in 0..8 {
            let bytes: [u8; 8] = [
                rng.next_u64() as u8,
                rng.next_u64() as u8,
                rng.next_u64() as u8,
                rng.next_u64() as u8,
                rng.next_u64() as u8,
                rng.next_u64() as u8,
                rng.next_u64() as u8,
                rng.next_u64() as u8,
            ];
            let mut got = vec![F8::ZERO; 256];
            table.apply(&bytes, &mut got);
            for s in 0..64 {
                let expected_bit = (bytes[s / 8] >> (s % 8)) & 1;
                assert_eq!(got[s], F8(expected_bit), "input bit mismatch at s={s}");
            }
        }
    }
}
