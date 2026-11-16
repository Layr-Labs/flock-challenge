mod r2steal {
    /// The R2-CS kernel's entry, its last buffer, its first barrier and its
    /// step loop, as all three kernel texts (`ZC_R2_CS_MSL_SOURCE`,
    /// `r2cs_fast.metal`, `r2cs_fast2.metal`) write them.
    const ENTRY: &str = "kernel void zc_r2_products_horner_cs(";
    const FLAG: &str = "    device atomic_uint* cs_flag     [[buffer(10)]],\n";
    const HEAD: &str = "    nib[lid] = nib_tab_dev[lid];\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    const LOOP: &str = "    for (uint step = 0u; step < (p.xpt & 0x7fffffffu); step++) {\n";

    /// The steal source (`super::super::ENV_NO_R2_STEAL`): `source` (the
    /// assembled R2-CS library) with the kernel's claim word, buffer 11, read
    /// through a volatile system-coherent pointer (so every read sees the
    /// CPU's latest store, not a cached copy; the internals pragma enables
    /// the qualifier, as for the race grind's tail kernel).
    /// - Thread 0 reads the word into threadgroup memory before the kernel's
    ///   first barrier; after that barrier the whole threadgroup returns if
    ///   its chunk (`tgid >> 2`) is at or above it.
    /// - At steps 4, 8 and 12 of the step loop each SIMD group reads it again
    ///   and leaves the loop, all lanes alike (`simd_any`), once the chunk is
    ///   at or above it; the steps it walked made their stores, and the rest of
    ///   the kernel runs on its partial sums, whose slots are the CPU's chunk's
    ///   and never read.
    ///
    /// Nothing else changes. `None` = the text is not the expected one (the
    /// steal stays off).
    pub(super) fn source(source: &str) -> Option<String> {
        let start = source.find(ENTRY)?;
        if source[start + 1..].contains(ENTRY) {
            return None;
        }
        let kernel = &source[start..];
        if [FLAG, HEAD, LOOP].iter().any(|t| kernel.matches(t).count() != 1) {
            return None;
        }
        let flag = start + kernel.find(FLAG)? + FLAG.len();
        let head = start + kernel.find(HEAD)?;
        let step = start + kernel.find(LOOP)? + LOOP.len();
        if !(flag <= head && head + HEAD.len() <= step) {
            return None;
        }
        let mut out = String::with_capacity(source.len() + 512);
        out.push_str(&source[..start]);
        out.push_str("#pragma METAL internals : enable\n");
        out.push_str(&source[start..flag]);
        out.push_str("    volatile coherent(system) device const uint* cs_cut [[buffer(11)]],\n");
        out.push_str(&source[flag..head]);
        out.push_str("    threadgroup uint cs_cut_tg;\n    if (lid == 0u) { cs_cut_tg = cs_cut[0]; }\n");
        out.push_str(HEAD);
        out.push_str("    if ((tgid >> 2) >= cs_cut_tg) { return; }\n");
        out.push_str(&source[head + HEAD.len()..step]);
        out.push_str("        if (step != 0u && (step & 3u) == 0u && simd_any((tgid >> 2) >= cs_cut[0])) { break; }\n");
        out.push_str(&source[step..]);
        Some(out)
    }

    #[cfg(test)]
    mod tests {
        use super::super::*;
        use std::ffi::OsStr;

        /// Each of the three R2-CS kernels with the claim word (its read at
        /// the threadgroup's start and its step-loop poll): with the word
        /// above every chunk it stores the same bytes and returns the same
        /// partials and head flag as the kernel without it; with the word at
        /// chunk 1, chunk 0 is unchanged and the threadgroups of chunks 1 and
        /// 2 return without a store.
        #[test]
        fn r2steal_kernel_claim_word() {
            const CHUNKS: usize = 3;
            const LO: usize = 16_384;
            let gpu = gpu().expect("R2-CS steal check requires Metal");
            let mut seed = 0x9e37_79b9_7f4a_7c15u64;
            let mut words = |n: usize| -> Vec<u32> {
                (0..n)
                    .map(|_| {
                        seed ^= seed << 13;
                        seed ^= seed >> 7;
                        seed ^= seed << 17;
                        seed as u32
                    })
                    .collect()
            };
            let a_in = words(CHUNKS * LO * 4);
            let mut b_in = words(CHUNKS * LO * 4);
            // Head pairs with all-ones b rows (the b == 1 step), one with not.
            for pair in 0..CHUNKS * LO {
                if pair % 128 < 8 && pair != 128 * 5 + 3 {
                    b_in[pair * 4..pair * 4 + 4].fill(u32::MAX);
                }
            }
            let nib = words(256 * 4);
            let eq_lo = words(1024 * 4);
            let eq_hi = words(CHUNKS * 4);
            let base = zc_r2_msl_source(false).into_owned();
            let variants = [
                ("original", Some("1"), None),
                ("lean", None, Some("1")),
                ("lean b == 1", None, None),
            ];
            unsafe {
                let pool = gpu.pool_push();
                let upload = |v: &[u32]| -> Id {
                    let buf = gpu.new_buffer(v.len() * 4).unwrap();
                    std::ptr::copy_nonoverlapping(v.as_ptr().cast::<u8>(), gpu.buffer_contents(buf), v.len() * 4);
                    buf
                };
                let ins = [upload(&a_in), upload(&b_in), upload(&eq_lo), upload(&eq_hi), upload(&nib)];
                // The head-compacted layout of the ranked walk.
                let l: [u32; 8] = [1, 2568, 236, 16_448, 1920, 0, 0, 0];
                let blocks = CHUNKS * LO / 128;
                let a_bytes = 16 * (l[1] as usize + blocks * l[2] as usize);
                let d_bytes = l[3] as usize + blocks * l[4] as usize;
                let p_bytes = CHUNKS * 4 * 8 * 16;
                let anchors = gpu.new_buffer(a_bytes).unwrap();
                let deltas = gpu.new_buffer(d_bytes).unwrap();
                let parts = gpu.new_buffer(p_bytes).unwrap();
                let flag = gpu.new_buffer(16).unwrap();
                let cut = gpu.new_buffer(16).unwrap();
                for (name, c1, c2) in variants {
                    let plain = format!(
                        "{base}{}",
                        r2cs_fast::source_for(ZC_R2_CS_MSL_SOURCE, c1.map(OsStr::new), c2.map(OsStr::new))
                    );
                    let steal = super::source(&plain).unwrap_or_else(|| panic!("{name}: steal source"));
                    assert_eq!(steal.matches("cs_cut_tg").count(), 3, "{name}");
                    assert_eq!(steal.matches("simd_any((tgid >> 2) >= cs_cut[0])").count(), 1, "{name}");
                    let psos = [
                        compile_supplemental_pipeline(gpu, &plain, "zc_r2_products_horner_cs").unwrap(),
                        compile_supplemental_pipeline(gpu, &steal, "zc_r2_products_horner_cs").unwrap(),
                    ];
                    let width = gpu.pipeline_resources(psos[0]).1;
                    assert_eq!(width, gpu.pipeline_resources(psos[1]).1, "{name}: SIMD width");
                    let params: [u32; 4] = [LO as u32, 16 | if width == 32 { 0x8000_0000 } else { 0 }, 127, 121];
                    // (pipeline, claim word) -> parts, anchors, deltas, flag.
                    let run = |pso: Id, word: u32| -> [Vec<u8>; 4] {
                        std::ptr::write_bytes(gpu.buffer_contents(anchors), 0x5a, a_bytes);
                        std::ptr::write_bytes(gpu.buffer_contents(deltas), 0xc3, d_bytes);
                        std::ptr::write_bytes(gpu.buffer_contents(parts), 0x3c, p_bytes);
                        std::ptr::write_bytes(gpu.buffer_contents(flag), 0, 16);
                        gpu.buffer_contents(cut).cast::<u32>().write(word);
                        let cb = gpu.command_buffer().unwrap();
                        let enc = gpu.compute_encoder(cb).unwrap();
                        gpu.set_pipeline(enc, pso);
                        for (index, buffer) in ins.iter().enumerate() {
                            gpu.set_buffer(enc, *buffer, 0, index);
                        }
                        gpu.set_buffer(enc, parts, 0, 5);
                        gpu.set_bytes(enc, core::slice::from_raw_parts(params.as_ptr().cast::<u8>(), 16), 6);
                        gpu.set_buffer(enc, anchors, 0, 7);
                        gpu.set_buffer(enc, deltas, 0, 8);
                        gpu.set_bytes(enc, core::slice::from_raw_parts(l.as_ptr().cast::<u8>(), 32), 9);
                        gpu.set_buffer(enc, flag, 0, 10);
                        gpu.set_buffer(enc, cut, 0, 11);
                        gpu.dispatch(enc, (CHUNKS * 4) as u64, 256);
                        gpu.end_encoding(enc);
                        gpu.commit_and_wait(cb).unwrap();
                        [(parts, p_bytes), (anchors, a_bytes), (deltas, d_bytes), (flag, 4)]
                            .map(|(buf, len)| core::slice::from_raw_parts(gpu.buffer_contents(buf), len).to_vec())
                    };
                    let want = run(psos[0], 0);
                    assert_ne!(want[3], [0u8; 4], "{name}: the head flag case runs");
                    assert_eq!(run(psos[1], u32::MAX), want, "{name}: word above every chunk");
                    assert_eq!(run(psos[1], CHUNKS as u32), want, "{name}: word at the prefix end");
                    let one = run(psos[1], 1);
                    // Chunk 0: its four groups' partials and its blocks'
                    // compact bytes; chunks 1 and 2: nothing stored.
                    let (pa, aa, da) = (p_bytes / CHUNKS, 16 * (l[1] as usize + LO / 128 * l[2] as usize), l[3] as usize + LO / 128 * l[4] as usize);
                    assert_eq!(one[0][..pa], want[0][..pa], "{name}: chunk 0 partials");
                    assert!(one[0][pa..].iter().all(|&b| b == 0x3c), "{name}: skipped partials");
                    assert_eq!(one[1][..aa], want[1][..aa], "{name}: chunk 0 anchors");
                    assert!(one[1][aa..].iter().all(|&b| b == 0x5a), "{name}: skipped anchors");
                    assert_eq!(one[2][..da], want[2][..da], "{name}: chunk 0 deltas");
                    assert!(one[2][da..].iter().all(|&b| b == 0xc3), "{name}: skipped deltas");
                    for pso in psos {
                        gpu.release(pso);
                    }
                }
                for buffer in ins.into_iter().chain([anchors, deltas, parts, flag, cut]) {
                    gpu.release(buffer);
                }
                gpu.pool_pop(pool);
            }
        }
    }
}
