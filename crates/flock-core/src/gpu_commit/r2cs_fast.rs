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

        /// Group 60 (pair 120 = row 240 with fixed 49-bit B mask and row 241
        /// zero; pair 121 padded) under R2 C capture (`cl.pad0 = 1`) in
        /// `r2cs_fast2.metal`:
        /// - compact anchors and deltas match the original kernel (`cl.pad0 = 0`)
        ///   byte-for-byte;
        /// - adding the exact algebraic group 0..4, 59, and 60 capture terms to
        ///   `r2cs_fast2`'s partials reproduces the original kernel's partials
        ///   bit-for-bit across all 8 output slots;
        /// - corrupting row 240/241 (high bit in a0, nonzero a1/b1, or wrong b0)
        ///   sets `cs_flag != 0` so the host falls back cleanly.
        #[test]
        fn r2cs_fast2_group60_capture_exact() {
            const CHUNKS: usize = 2;
            const LO: usize = 16_384;
            const B240: u64 = 0x0001_ffff_ffff_ffff;
            let gpu = gpu().expect("R2-CS group 60 oracle requires Metal");
            let mut seed = 0x60c0_cafe_1234_5678u64;
            let mut next = move || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed
            };
            // Build structured eq_lo and eq_hi from random MLV challenges so we
            // can compare against exact algebraic capture terms.
            let mut mlv = Vec::with_capacity(26);
            for _ in 0..26 {
                mlv.push(F128::new(next(), next()));
            }
            let med = crate::zerocheck::univariate_skip_optimized::medium_challenges_ghash();
            mlv[3..7].copy_from_slice(&med);
            let eq_lo_full = crate::zerocheck::univariate_skip::build_eq(&mlv[1..15]);
            let f128_words = |v: F128| -> [u32; 4] {
                [v.lo as u32, (v.lo >> 32) as u32, v.hi as u32, (v.hi >> 32) as u32]
            };
            let mut eq_lo_compact = vec![0u32; 1024 * 4];
            for window in 0..256 {
                for class in 0..4 {
                    let w = f128_words(eq_lo_full[(window << 6) | class]);
                    eq_lo_compact[(window * 4 + class) * 4..(window * 4 + class + 1) * 4].copy_from_slice(&w);
                }
            }
            let eq_hi_f: Vec<F128> = (0..CHUNKS).map(|_| F128::new(next(), next())).collect();
            let mut eq_hi = vec![0u32; CHUNKS * 4];
            for (i, &v) in eq_hi_f.iter().enumerate() {
                eq_hi[i * 4..i * 4 + 4].copy_from_slice(&f128_words(v));
            }
            // Partition-of-unity linear nibble table (all-ones row folds to ONE).
            let mut singles: Vec<F128> = (0..64).map(|_| F128::new(next(), next())).collect();
            let rest = singles[..63].iter().fold(F128::ZERO, |acc, s| acc + *s);
            singles[63] = F128::ONE + rest;
            let mut nib_unity = vec![0u32; 256 * 4];
            for pos in 0..16 {
                for n in 0..16 {
                    let mut v = F128::ZERO;
                    for bit in 0..4 {
                        if (n >> bit) & 1 == 1 {
                            v += singles[pos * 4 + bit];
                        }
                    }
                    nib_unity[(pos * 16 + n) * 4..(pos * 16 + n + 1) * 4].copy_from_slice(&f128_words(v));
                }
            }
            let fold_row = |code: u64| -> F128 {
                let mut acc = F128::ZERO;
                for s in 0..64 {
                    if (code >> s) & 1 == 1 {
                        acc += singles[s];
                    }
                }
                acc
            };
            let mut a_in = vec![0u32; CHUNKS * LO * 4];
            let mut b_in = vec![0u32; CHUNKS * LO * 4];
            for v in a_in.iter_mut().chain(b_in.iter_mut()) {
                *v = next() as u32;
            }
            let blocks = CHUNKS * LO / 128;
            // BLAKE3-shaped witness: groups 0..4 and 59 all-ones B, pair 8 all-ones B,
            // group 60 pair 120 has B_240 = 0x0001_ffff_ffff_ffff, A_240 top 15 bits 0,
            // row 241 all zero.
            for pair in 0..CHUNKS * LO {
                let pib = pair % 128;
                if pib < 8 || pib == 8 || pib == 118 || pib == 119 {
                    b_in[pair * 4..pair * 4 + 4].fill(u32::MAX);
                } else if pib == 120 {
                    a_in[pair * 4 + 1] &= 0x0001_ffffu32;
                    a_in[pair * 4 + 2] = 0;
                    a_in[pair * 4 + 3] = 0;
                    b_in[pair * 4] = 0xffff_ffffu32;
                    b_in[pair * 4 + 1] = 0x0001_ffffu32;
                    b_in[pair * 4 + 2] = 0;
                    b_in[pair * 4 + 3] = 0;
                }
            }
            // Compute exact algebraic capture delta across the CHUNKS chunks.
            let mut eq_blk = vec![F128::ZERO; blocks];
            for ch in 0..CHUNKS {
                for b_in_ch in 0..(LO / 128) {
                    // eq_lo_full index for block b_in_ch, group 0, pair 0 is b_in_ch * 128;
                    // factor out (1+d)*eq(r[2..8], 0) to get pure block weight.
                    let g0p0 = eq_lo_full[b_in_ch * 128];
                    let base0 = (0..7).fold(F128::ONE, |acc, j| acc * (F128::ONE + mlv[1 + j]));
                    eq_blk[ch * (LO / 128) + b_in_ch] = eq_hi_f[ch] * g0p0 * base0.inv();
                }
            }
            let d = mlv[1];
            let b240_f = fold_row(B240);
            let mut cap_p1_e = F128::ZERO;
            let mut cap_pinf_e = F128::ZERO;
            let mut cap_p1_o = F128::ZERO;
            let mut cap_w0 = F128::ZERO;
            let mut cap_w3 = F128::ZERO;
            let mut cap_w5 = F128::ZERO;
            let h_of = |g: usize| -> F128 {
                (0..6).fold(F128::ONE, |h, bit| {
                    let c = mlv[2 + bit];
                    h * if (g >> bit) & 1 == 1 { c } else { F128::ONE + c }
                })
            };
            for blk in 0..blocks {
                let eb = eq_blk[blk];
                for g in [0usize, 1, 2, 3, 59] {
                    let h = h_of(g) * eb;
                    let p0 = blk * 128 + 2 * g;
                    let p1 = p0 + 1;
                    let a1 = fold_row((a_in[p0 * 4 + 2] as u64) | ((a_in[p0 * 4 + 3] as u64) << 32));
                    let a2 = fold_row((a_in[p1 * 4] as u64) | ((a_in[p1 * 4 + 1] as u64) << 32));
                    let a3 = fold_row((a_in[p1 * 4 + 2] as u64) | ((a_in[p1 * 4 + 3] as u64) << 32));
                    cap_p1_e += h * (F128::ONE + d) * a1;
                    cap_p1_o += h * d * a3;
                    cap_w0 += h * d * a2;
                }
                let h60 = h_of(60) * eb;
                let p120 = blk * 128 + 120;
                let a240 = fold_row((a_in[p120 * 4] as u64) | ((a_in[p120 * 4 + 1] as u64) << 32));
                let p60 = h60 * a240 * b240_f;
                cap_pinf_e += (F128::ONE + d) * p60;
                cap_w3 += d * p60;
                cap_w5 += d * p60;
            }
            let cap_slots = [cap_p1_e, cap_pinf_e, cap_p1_o, F128::ZERO, cap_w0, cap_w3, F128::ZERO, cap_w5];

            let mut base = zc_r2_msl_source(false).into_owned();
            let mut fast2 = base.clone();
            base.push_str(&source_for(ZC_R2_CS_MSL_SOURCE, Some(OsStr::new("1")), None));
            fast2.push_str(&source_for(ZC_R2_CS_MSL_SOURCE, None, None));
            unsafe {
                let pool = gpu.pool_push();
                let upload = |v: &[u32]| -> Id {
                    let buf = gpu.new_buffer(v.len() * 4).unwrap();
                    std::ptr::copy_nonoverlapping(v.as_ptr().cast::<u8>(), gpu.buffer_contents(buf), v.len() * 4);
                    buf
                };
                let pso_orig = compile_supplemental_pipeline(gpu, &base, "zc_r2_products_horner_cs").unwrap();
                let pso_fast2 = compile_supplemental_pipeline(gpu, &fast2, "zc_r2_products_horner_cs").unwrap();
                let width = gpu.pipeline_resources(pso_orig).1;
                let eq_lo_buf = upload(&eq_lo_compact);
                let eq_hi_buf = upload(&eq_hi);
                let nib_buf = upload(&nib_unity);
                let l: [u32; 5] = [1, 2568, 236, 16_448, 1920];
                let a_bytes = 16 * (l[1] as usize + blocks * l[2] as usize);
                let d_bytes = l[3] as usize + blocks * l[4] as usize;
                let p_bytes = CHUNKS * 4 * 8 * 16;
                let anchors = gpu.new_buffer(a_bytes).unwrap();
                let deltas = gpu.new_buffer(d_bytes).unwrap();
                let parts = gpu.new_buffer(p_bytes).unwrap();
                let flag = gpu.new_buffer(16).unwrap();
                let params: [u32; 4] = [LO as u32, 16 | if width == 32 { 0x8000_0000 } else { 0 }, 127, 121];
                let run = |pso: Id, a_v: &[u32], b_v: &[u32], pad0: u32| -> ([F128; 8], Vec<u8>, Vec<u8>, u32) {
                    let a_buf = upload(a_v);
                    let b_buf = upload(b_v);
                    let lp: [u32; 8] = [l[0], l[1], l[2], l[3], l[4], pad0, 0, 0];
                    std::ptr::write_bytes(gpu.buffer_contents(anchors), 0x5a, a_bytes);
                    std::ptr::write_bytes(gpu.buffer_contents(deltas), 0xc3, d_bytes);
                    std::ptr::write_bytes(gpu.buffer_contents(parts), 0, p_bytes);
                    std::ptr::write_bytes(gpu.buffer_contents(flag), 0, 16);
                    let cb = gpu.command_buffer().unwrap();
                    let enc = gpu.compute_encoder(cb).unwrap();
                    gpu.set_pipeline(enc, pso);
                    for (idx, buf) in [a_buf, b_buf, eq_lo_buf, eq_hi_buf, nib_buf, parts].iter().enumerate() {
                        gpu.set_buffer(enc, *buf, 0, idx);
                    }
                    gpu.set_bytes(enc, core::slice::from_raw_parts(params.as_ptr().cast::<u8>(), 16), 6);
                    gpu.set_buffer(enc, anchors, 0, 7);
                    gpu.set_buffer(enc, deltas, 0, 8);
                    gpu.set_bytes(enc, core::slice::from_raw_parts(lp.as_ptr().cast::<u8>(), 32), 9);
                    gpu.set_buffer(enc, flag, 0, 10);
                    gpu.dispatch(enc, (CHUNKS * 4) as u64, 256);
                    gpu.end_encoding(enc);
                    gpu.commit_and_wait(cb).unwrap();
                    let raw_parts = core::slice::from_raw_parts(gpu.buffer_contents(parts).cast::<F128>(), CHUNKS * 4 * 8);
                    let mut sums = [F128::ZERO; 8];
                    for tg in 0..(CHUNKS * 4) {
                        for s in 0..8 {
                            sums[s] += raw_parts[tg * 8 + s];
                        }
                    }
                    let anc = core::slice::from_raw_parts(gpu.buffer_contents(anchors), a_bytes).to_vec();
                    let del = core::slice::from_raw_parts(gpu.buffer_contents(deltas), d_bytes).to_vec();
                    let fl = *gpu.buffer_contents(flag).cast::<u32>();
                    gpu.release(a_buf);
                    gpu.release(b_buf);
                    (sums, anc, del, fl)
                };
                let (want_sums, want_anc, want_del, want_fl) = run(pso_orig, &a_in, &b_in, 0);
                let (got_sums, got_anc, got_del, got_fl) = run(pso_fast2, &a_in, &b_in, 1);
                assert_eq!(want_fl, 0);
                assert_eq!(got_fl, 0, "valid BLAKE3-shaped group 60 must not set head_bad");
                assert_eq!(got_anc, want_anc, "group 60 capture anchors mismatch");
                assert_eq!(got_del, want_del, "group 60 capture deltas mismatch");
                for s in 0..8 {
                    assert_eq!(
                        got_sums[s] + cap_slots[s],
                        want_sums[s],
                        "group 60 capture slot {s} mismatch"
                    );
                }
                // Bad group 60 cases must all set cs_flag != 0.
                let bad_pair = 128 * 3 + 120;
                for (name, mutate) in [
                    ("a0 high bit", Box::new(|a: &mut [u32], _b: &mut [u32]| a[bad_pair * 4 + 1] |= 0x0002_0000) as Box<dyn Fn(&mut [u32], &mut [u32])>),
                    ("a1 nonzero", Box::new(|a: &mut [u32], _b: &mut [u32]| a[bad_pair * 4 + 2] = 1)),
                    ("b0 wrong", Box::new(|_a: &mut [u32], b: &mut [u32]| b[bad_pair * 4 + 1] = 0x0003_ffff)),
                    ("b1 nonzero", Box::new(|_a: &mut [u32], b: &mut [u32]| b[bad_pair * 4 + 3] = 1)),
                ] {
                    let mut a_bad = a_in.clone();
                    let mut b_bad = b_in.clone();
                    mutate(&mut a_bad, &mut b_bad);
                    let (_, _, _, fl) = run(pso_fast2, &a_bad, &b_bad, 1);
                    assert_ne!(fl, 0, "bad group 60 ({name}) must flag cs_flag");
                }
                for buf in [eq_lo_buf, eq_hi_buf, nib_buf, anchors, deltas, parts, flag] {
                    gpu.release(buf);
                }
                gpu.release(pso_orig);
                gpu.release(pso_fast2);
                gpu.pool_pop(pool);
            }
        }
    }
}
