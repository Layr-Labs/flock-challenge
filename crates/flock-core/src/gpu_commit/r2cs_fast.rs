mod r2cs_fast {
    use std::borrow::Cow;
    use std::ffi::OsStr;

    const KERNEL: &str = include_str!("r2cs_fast.metal");
    const ENTRY: &str = "kernel void zc_r2_products_horner_cs(";

    /// The R2-CS library's own source: the original's helpers, then the
    /// lean kernel under the same entry point, unless `FLOCK_NO_R2CS_FAST`
    /// is exactly `1` (the original text). Chosen once, during the R2-CS
    /// pipeline build.
    pub(super) fn source(original: &'static str) -> Cow<'static, str> {
        source_for(original, std::env::var_os("FLOCK_NO_R2CS_FAST").as_deref())
    }

    fn source_for(original: &'static str, control: Option<&OsStr>) -> Cow<'static, str> {
        if control == Some(OsStr::new("1")) {
            return Cow::Borrowed(original);
        }
        // The entry point is the last item of the original source.
        let Some(start) = original.find(ENTRY) else {
            return Cow::Borrowed(original);
        };
        let mut source = String::with_capacity(start + KERNEL.len());
        source.push_str(&original[..start]);
        source.push_str(KERNEL);
        Cow::Owned(source)
    }

    #[cfg(test)]
    mod tests {
        use super::super::*;
        use super::*;

        #[test]
        fn r2cs_fast_source_selection() {
            let off = source_for(ZC_R2_CS_MSL_SOURCE, Some(OsStr::new("1")));
            assert!(matches!(off, Cow::Borrowed(_)));
            assert_eq!(off.as_ref(), ZC_R2_CS_MSL_SOURCE);
            let start = ZC_R2_CS_MSL_SOURCE.find(ENTRY).unwrap();
            for control in [None, Some(""), Some("0"), Some("01"), Some("true")] {
                let on = source_for(ZC_R2_CS_MSL_SOURCE, control.map(OsStr::new));
                assert!(on.starts_with(&ZC_R2_CS_MSL_SOURCE[..start]));
                assert!(on.ends_with(KERNEL));
                assert_eq!(on.matches("kernel void ").count(), 1);
                assert_eq!(on.matches(ENTRY).count(), 1);
            }
        }

        /// The lean kernel and the original store the same bytes and return
        /// the same partials and head flag, over several padding patterns
        /// (including padded steps between live ones) and both compact
        /// layouts.
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
            let nib = words(256 * 4);
            let eq_lo = words(1024 * 4);
            let eq_hi = words(CHUNKS * 4);
            // Head pairs (0..8 of each 128-pair block) with all-ones b rows.
            let mut b_ones = b_rand.clone();
            for pair in 0..CHUNKS * LO {
                if pair % 128 < 8 {
                    b_ones[pair * 4..pair * 4 + 4].fill(u32::MAX);
                }
            }
            let mut base = zc_r2_msl_source(false).into_owned();
            let mut fast = base.clone();
            base.push_str(&source_for(ZC_R2_CS_MSL_SOURCE, Some(OsStr::new("1"))));
            fast.push_str(&source_for(ZC_R2_CS_MSL_SOURCE, None));
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
                ];
                let width = gpu.pipeline_resources(psos[0]).1;
                assert_eq!(width, gpu.pipeline_resources(psos[1]).1, "SIMD width");
                assert!(gpu.pipeline_resources(psos[1]).2 >= 256, "threadgroup capacity");
                let a_buf = upload(&a_in);
                let nib_buf = upload(&nib);
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
                for (li, (l, pads)) in layouts.iter().enumerate() {
                    let a_bytes = 16 * (l[1] as usize + blocks * l[2] as usize);
                    let d_bytes = l[3] as usize + blocks * l[4] as usize;
                    let anchors = gpu.new_buffer(a_bytes).unwrap();
                    let deltas = gpu.new_buffer(d_bytes).unwrap();
                    let parts = gpu.new_buffer(CHUNKS * 4 * 8 * 16).unwrap();
                    let flag = gpu.new_buffer(16).unwrap();
                    for &(mask, useful) in pads.iter() {
                        for b_in in [&b_rand, &b_ones] {
                            let b_buf = upload(b_in);
                            let params: [u32; 4] = [LO as u32, 16 | if width == 32 { 0x8000_0000 } else { 0 }, mask, useful];
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
                            assert!(
                                out[0] == out[1],
                                "layout {li} mask {mask} useful {useful} head ones {}",
                                core::ptr::eq(b_in, &b_ones)
                            );
                            gpu.release(b_buf);
                            runs += 1;
                        }
                    }
                    for buffer in [anchors, deltas, parts, flag] {
                        gpu.release(buffer);
                    }
                }
                eprintln!("[r2cs-fast-oracle] {runs} launches of {CHUNKS} chunks equal (SIMD width {width})");
                for buffer in [a_buf, nib_buf, eq_lo_buf, eq_hi_buf] {
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
