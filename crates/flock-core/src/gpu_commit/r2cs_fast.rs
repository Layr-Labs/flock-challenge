mod r2cs_fast {
    use std::borrow::Cow;
    use std::ffi::OsStr;

    const KERNEL: &str = include_str!("r2cs_fast.metal");
    const KERNEL2: &str = include_str!("r2cs_fast2.metal");
    const ENTRY: &str = "kernel void zc_r2_products_horner_cs(";

    /// The R2-CS library's own source: the original's helpers, then the lean
    /// kernel's product helpers and the lean kernel with its b == 1 step
    /// (`r2cs_fast2.metal`), under the same entry point. `FLOCK_NO_R2CS_FAST2`
    /// exactly `1` keeps the lean kernel of `r2cs_fast.metal` (that text
    /// exactly); `FLOCK_NO_R2CS_FAST` exactly `1` the original text. Both are
    /// read once, during the R2-CS pipeline build.
    pub(super) fn source(original: &'static str) -> Cow<'static, str> {
        source_for(
            original,
            std::env::var_os("FLOCK_NO_R2CS_FAST").as_deref(),
            std::env::var_os("FLOCK_NO_R2CS_FAST2").as_deref(),
        )
    }

    /// Whether [`source`] selects the b == 1 kernel of `r2cs_fast2.metal`
    /// (the only text that reads the R2 C capture flag), by the same reads.
    pub(super) fn fast2_selected() -> bool {
        std::env::var_os("FLOCK_NO_R2CS_FAST").as_deref() != Some(OsStr::new("1"))
            && std::env::var_os("FLOCK_NO_R2CS_FAST2").as_deref() != Some(OsStr::new("1"))
            && KERNEL.contains(ENTRY)
    }

    pub(super) fn source_for(original: &'static str, control: Option<&OsStr>, control2: Option<&OsStr>) -> Cow<'static, str> {
        if control == Some(OsStr::new("1")) {
            return Cow::Borrowed(original);
        }
        // The entry point is the last item of the original source.
        let Some(start) = original.find(ENTRY) else {
            return Cow::Borrowed(original);
        };
        // The lean kernel's own helpers end where its entry point starts.
        let lean_helpers = KERNEL.find(ENTRY);
        let parts: [&str; 2] = match lean_helpers {
            Some(k) if control2 != Some(OsStr::new("1")) => [&KERNEL[..k], KERNEL2],
            _ => [KERNEL, ""],
        };
        let mut source = String::with_capacity(start + parts[0].len() + parts[1].len());
        source.push_str(&original[..start]);
        source.push_str(parts[0]);
        source.push_str(parts[1]);
        Cow::Owned(source)
    }

    #[cfg(test)]
    mod tests {
        use super::super::*;
        use super::*;

        #[test]
        fn r2cs_fast_source_selection() {
            let off = source_for(ZC_R2_CS_MSL_SOURCE, Some(OsStr::new("1")), None);
            assert!(matches!(off, Cow::Borrowed(_)));
            assert_eq!(off.as_ref(), ZC_R2_CS_MSL_SOURCE);
            for control2 in [None, Some("1"), Some("0")] {
                let off = source_for(ZC_R2_CS_MSL_SOURCE, Some(OsStr::new("1")), control2.map(OsStr::new));
                assert_eq!(off.as_ref(), ZC_R2_CS_MSL_SOURCE);
            }
            let start = ZC_R2_CS_MSL_SOURCE.find(ENTRY).unwrap();
            let lean = format!("{}{KERNEL}", &ZC_R2_CS_MSL_SOURCE[..start]);
            let k = KERNEL.find(ENTRY).unwrap();
            let fast2 = format!("{}{}{KERNEL2}", &ZC_R2_CS_MSL_SOURCE[..start], &KERNEL[..k]);
            assert_eq!(KERNEL2.matches(ENTRY).count(), 1);
            for control in [None, Some(""), Some("0"), Some("01"), Some("true")] {
                // FLOCK_NO_R2CS_FAST2=1: the lean kernel's text exactly.
                let on = source_for(ZC_R2_CS_MSL_SOURCE, control.map(OsStr::new), Some(OsStr::new("1")));
                assert_eq!(on.as_ref(), lean);
                for control2 in [None, Some(""), Some("0"), Some("01"), Some("true")] {
                    let on = source_for(ZC_R2_CS_MSL_SOURCE, control.map(OsStr::new), control2.map(OsStr::new));
                    assert_eq!(on.as_ref(), fast2);
                    assert!(on.ends_with(KERNEL2));
                    assert_eq!(on.matches("kernel void ").count(), 1);
                    assert_eq!(on.matches(ENTRY).count(), 1);
                }
            }
        }

        /// The original kernel, the lean kernel and the lean kernel with its
        /// b == 1 step store the same bytes and return the same partials and
        /// head flag, over several padding patterns (including padded steps
        /// between live ones) and both compact layouts, with a random nibble
        /// table and with a partition-of-unity one (the all-ones row folds to
        /// ONE, so the b == 1 step runs wherever the b rows allow it).
        #[test]
        fn r2cs_fast_matches_original_kernel() {
            const CHUNKS: usize = 3;
            const LO: usize = 16_384;
            let gpu = gpu().expect("R2-CS kernel oracle requires Metal");
            let mut seed = 0x2545_f491_4f6c_dd1du64;
            let mut next = move || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed
            };
            let mut words = |n: usize| -> Vec<u32> {
                (0..n).map(|_| next() as u32).collect()
            };
            let a_in = words(CHUNKS * LO * 4);
            let b_rand = words(CHUNKS * LO * 4);
            let nib_rand = words(256 * 4);
            // A linear nibble table (entry n of nibble position j is the sum of
            // the basis rows of the set bits of n) whose 64 basis rows sum to
            // ONE: the all-ones row folds to ONE.
            let basis = words(64 * 4);
            let mut unity_basis = basis.clone();
            let mut sum = [1u32, 0, 0, 0];
            for row in basis.chunks(4).take(63) {
                for (s, w) in sum.iter_mut().zip(row) {
                    *s ^= w;
                }
            }
            unity_basis[63 * 4..].copy_from_slice(&sum);
            let mut nib_unity = vec![0u32; 256 * 4];
            for pos in 0..16 {
                for n in 0..16 {
                    for bit in 0..4 {
                        if n >> bit & 1 == 1 {
                            for w in 0..4 {
                                nib_unity[(pos * 16 + n) * 4 + w] ^= unity_basis[(pos * 4 + bit) * 4 + w];
                            }
                        }
                    }
                }
            }
            let eq_lo = words(1024 * 4);
            let eq_hi = words(CHUNKS * 4);
            // b rows all ones: the head pairs (0..8 of each 128-pair block);
            // the b == 1 mass of the shipped layout (the head pairs and pairs
            // 118..121, so pair 120's partner is padded under the shipped
            // padding); every pair.
            let ones_where = |f: &dyn Fn(usize) -> bool| -> Vec<u32> {
                let mut b = b_rand.clone();
                for pair in 0..CHUNKS * LO {
                    if f(pair % 128) {
                        b[pair * 4..pair * 4 + 4].fill(u32::MAX);
                    }
                }
                b
            };
            let b_head = ones_where(&|q| q < 8);
            let b_mass = ones_where(&|q| q < 8 || (118..121).contains(&q));
            let b_all = ones_where(&|_| true);
            let b_sets: [(&str, &Vec<u32>); 4] =
                [("random", &b_rand), ("head ones", &b_head), ("b == 1 mass", &b_mass), ("all ones", &b_all)];
            let mut base = zc_r2_msl_source(false).into_owned();
            let mut fast = base.clone();
            let mut fast2 = base.clone();
            base.push_str(&source_for(ZC_R2_CS_MSL_SOURCE, Some(OsStr::new("1")), None));
            fast.push_str(&source_for(ZC_R2_CS_MSL_SOURCE, None, Some(OsStr::new("1"))));
            fast2.push_str(&source_for(ZC_R2_CS_MSL_SOURCE, None, None));
            unsafe {
                let pool = gpu.pool_push();
                let upload = |v: &[u32]| -> Id {
                    let buf = gpu.new_buffer(v.len() * 4).unwrap();
                    std::ptr::copy_nonoverlapping(v.as_ptr().cast::<u8>(), gpu.buffer_contents(buf), v.len() * 4);
                    buf
                };
                let psos = [
                    compile_supplemental_pipeline(gpu, &base, "zc_r2_products_horner_cs").unwrap(),
                    compile_supplemental_pipeline(gpu, &fast, "zc_r2_products_horner_cs").unwrap(),
                    compile_supplemental_pipeline(gpu, &fast2, "zc_r2_products_horner_cs").unwrap(),
                ];
                let width = gpu.pipeline_resources(psos[0]).1;
                for &pso in &psos[1..] {
                    assert_eq!(width, gpu.pipeline_resources(pso).1, "SIMD width");
                    assert!(gpu.pipeline_resources(pso).2 >= 256, "threadgroup capacity");
                }
                let fast2_tg = gpu.pipeline_resources(psos[2]).0;
                eprintln!("r2cs_fast2 staticThreadgroupMemoryLength = {fast2_tg} B");
                assert_eq!(fast2_tg, 29_184, "r2cs_fast2 TG SRAM must stay 29,184 B (got {fast2_tg})");
                let a_buf = upload(&a_in);
                let eq_lo_buf = upload(&eq_lo);
                let eq_hi_buf = upload(&eq_hi);
                let blocks = CHUNKS * LO / 128;
                // (mask, useful): the shipped padding, three padded medium
                // steps plus a partial one, none, and every other medium step.
                let pads = [(127u32, 121u32), (63, 50), (127, 128), (4, 4)];
                // (hc, a_base, a_stride, d_base, d_stride) and the padding
                // patterns whose live pairs fit that layout's blocks.
                let layouts = [([1u32, 2568, 236, 16_448, 1920], &pads[..2]), ([0, 0, 256, 0, 2048], &pads[..])];
                let mut runs = 0;
                for (nib_name, nib) in [("random", &nib_rand), ("unity", &nib_unity)] {
                    let nib_buf = upload(nib);
                    for (li, (l, pads)) in layouts.iter().enumerate() {
                        let a_bytes = 16 * (l[1] as usize + blocks * l[2] as usize);
                        let d_bytes = l[3] as usize + blocks * l[4] as usize;
                        let anchors = gpu.new_buffer(a_bytes).unwrap();
                        let deltas = gpu.new_buffer(d_bytes).unwrap();
                        let parts = gpu.new_buffer(CHUNKS * 4 * 8 * 16).unwrap();
                        let flag = gpu.new_buffer(16).unwrap();
                        for &(mask, useful) in pads.iter() {
                            for (b_name, b_in) in b_sets {
                                let b_buf = upload(b_in);
                                let params: [u32; 4] =
                                    [LO as u32, 16 | if width == 32 { 0x8000_0000 } else { 0 }, mask, useful];
                                let lp: [u32; 8] = [l[0], l[1], l[2], l[3], l[4], 0, 0, 0];
                                let mut out: Vec<Vec<u8>> = Vec::new();
                                for &pso in &psos {
                                    std::ptr::write_bytes(gpu.buffer_contents(anchors), 0x5a, a_bytes);
                                    std::ptr::write_bytes(gpu.buffer_contents(deltas), 0xc3, d_bytes);
                                    std::ptr::write_bytes(gpu.buffer_contents(parts), 0x3c, CHUNKS * 4 * 8 * 16);
                                    std::ptr::write_bytes(gpu.buffer_contents(flag), 0, 16);
                                    let cb = gpu.command_buffer().unwrap();
                                    let enc = gpu.compute_encoder(cb).unwrap();
                                    gpu.set_pipeline(enc, pso);
                                    for (index, buffer) in [a_buf, b_buf, eq_lo_buf, eq_hi_buf, nib_buf, parts].iter().enumerate() {
                                        gpu.set_buffer(enc, *buffer, 0, index);
                                    }
                                    gpu.set_bytes(enc, core::slice::from_raw_parts(params.as_ptr().cast::<u8>(), 16), 6);
                                    gpu.set_buffer(enc, anchors, 0, 7);
                                    gpu.set_buffer(enc, deltas, 0, 8);
                                    gpu.set_bytes(enc, core::slice::from_raw_parts(lp.as_ptr().cast::<u8>(), 32), 9);
                                    gpu.set_buffer(enc, flag, 0, 10);
                                    gpu.dispatch(enc, (CHUNKS * 4) as u64, 256);
                                    gpu.end_encoding(enc);
                                    gpu.commit_and_wait(cb).unwrap();
                                    let mut bytes = Vec::new();
                                    for (buf, len) in [(parts, CHUNKS * 4 * 8 * 16), (anchors, a_bytes), (deltas, d_bytes), (flag, 4)] {
                                        bytes.extend_from_slice(core::slice::from_raw_parts(gpu.buffer_contents(buf), len));
                                    }
                                    out.push(bytes);
                                }
                                for (kernel, bytes) in ["lean", "lean b == 1"].iter().zip(&out[1..]) {
                                    assert!(
                                        out[0] == *bytes,
                                        "{kernel}: nib {nib_name} layout {li} mask {mask} useful {useful} b rows {b_name}"
                                    );
                                }
                                gpu.release(b_buf);
                                runs += 1;
                            }
                        }
                        for buffer in [anchors, deltas, parts, flag] {
                            gpu.release(buffer);
                        }
                    }
                    gpu.release(nib_buf);
                }
                eprintln!("[r2cs-fast-oracle] {runs} launches of {CHUNKS} chunks equal in three kernels (SIMD width {width})");
                for buffer in [a_buf, eq_lo_buf, eq_hi_buf] {
                    gpu.release(buffer);
                }
                for pso in psos {
                    gpu.release(pso);
                }
                gpu.pool_pop(pool);
            }
        }
    }
}
