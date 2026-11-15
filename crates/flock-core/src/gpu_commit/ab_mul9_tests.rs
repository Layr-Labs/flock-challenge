#[test]
fn ab_nine_product_clmul_matches_cpu() {
    let gpu = gpu().expect("AB arithmetic oracle requires Metal");
    let mut seed = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        F128::new(seed, seed.rotate_left(29) ^ 0xd1b5_4a32_d192_ed03)
    };
    let basis: Vec<F128> = (0..128)
        .map(|bit| if bit < 64 { F128::new(1 << bit, 0) } else { F128::new(0, 1 << (bit - 64)) })
        .collect();
    let mut pairs = Vec::new();
    for &a in &basis {
        for &b in &basis {
            pairs.push((a, b));
        }
    }
    let mut edges = vec![F128::ZERO, F128::ONE];
    for word in [u64::MAX, 0xaaaa_aaaa_aaaa_aaaa, 0x5555_5555_5555_5555,
                 0x1111_1111_1111_1111, 0x8888_8888_8888_8888,
                 0xffff_ffff, 0xffff_ffff_0000_0000] {
        edges.extend([F128::new(word, 0), F128::new(0, word), F128::new(word, word)]);
    }
    for &a in &edges {
        for &b in &edges {
            pairs.push((a, b));
        }
    }
    for _ in 0..4096 {
        pairs.push((next(), next()));
    }
    while pairs.len() % 64 != 0 {
        pairs.push((F128::ZERO, F128::ZERO));
    }
    let a: Vec<F128> = pairs.iter().map(|v| v.0).collect();
    let b: Vec<F128> = pairs.iter().map(|v| v.1).collect();
    let expected: Vec<u64> = pairs.iter().flat_map(|&(a, b)| {
        let mut p32 = 0u64;
        for bit in 0..32 {
            if (b.lo >> bit) & 1 != 0 {
                p32 ^= u64::from(a.lo as u32) << bit;
            }
        }
        let p = a.mul_unreduced(b);
        let reduced = p.reduce();
        [p32, p.r0, p.r1, p.r2, p.r3, reduced.lo, reduced.hi]
    }).collect();

    unsafe {
        let pool = gpu.pool_push();
        let a_buf = ab_upload(gpu, &a);
        let b_buf = ab_upload(gpu, &b);
        let out_buf = gpu.new_buffer(core::mem::size_of_val(expected.as_slice())).unwrap();
        let mut previous_resources: Option<Vec<(u64, u64, u64)>> = None;
        for control in [Some(OsStr::new("1")), None] {
            let base = source_for(ZC_AB_MSL_SOURCE, control);
            let mut arithmetic = base.to_string();
            arithmetic.push_str(r#"
kernel void ab_arithmetic_oracle(
    device const uint4* a [[buffer(0)]], device const uint4* b [[buffer(1)]],
    device ulong* out [[buffer(2)]], uint id [[thread_position_in_grid]])
{
    U256k p = clmul128(a[id], b[id]);
    uint4 r = gf_reduce(p);
    out[id * 7u] = clmul32(a[id].x, b[id].x);
    out[id * 7u + 1u] = p.r0; out[id * 7u + 2u] = p.r1;
    out[id * 7u + 3u] = p.r2; out[id * 7u + 4u] = p.r3;
    out[id * 7u + 5u] = (ulong(r.y) << 32) | r.x;
    out[id * 7u + 6u] = (ulong(r.w) << 32) | r.z;
}
"#);
            let pso = compile_supplemental_pipeline(gpu, &arithmetic, "ab_arithmetic_oracle").unwrap();
            std::ptr::write_bytes(gpu.buffer_contents(out_buf), 0xa5, expected.len() * 8);
            let cb = gpu.command_buffer().unwrap();
            let enc = gpu.compute_encoder(cb).unwrap();
            gpu.set_pipeline(enc, pso);
            gpu.set_buffer(enc, a_buf, 0, 0);
            gpu.set_buffer(enc, b_buf, 0, 1);
            gpu.set_buffer(enc, out_buf, 0, 2);
            gpu.dispatch(enc, (pairs.len() / 64) as u64, 64);
            gpu.end_encoding(enc);
            gpu.commit_and_wait(cb).unwrap();
            let actual = core::slice::from_raw_parts(gpu.buffer_contents(out_buf).cast::<u64>(), expected.len());
            assert_eq!(actual, expected.as_slice(), "arithmetic control={control:?}");
            gpu.release(pso);

            let mut psos = Vec::new();
            let mut resources = Vec::new();
            for (name, width) in [("zc_ab_tables", 256), ("zc_ab_gather", 64), ("zc_ab_finish", 64)] {
                let pso = compile_supplemental_pipeline(gpu, &base, name).unwrap();
                let usage = gpu.pipeline_resources(pso);
                assert!(usage.2 >= width, "{name} threadgroup capacity");
                resources.push(usage);
                psos.push(pso);
            }
            if let Some(previous) = &previous_resources {
                for (old, new) in previous.iter().zip(&resources) {
                    assert_eq!(old.0, new.0, "shared bytes changed");
                    assert_eq!(old.1, new.1, "SIMD width changed");
                }
            }
            eprintln!("[ab-multiply-oracle] control={control:?} pairs={} resources={resources:?}", pairs.len());
            previous_resources = Some(resources);
            ab_chunk_oracle(gpu, &psos, control.is_none(), &mut next);
            for pso in psos {
                gpu.release(pso);
            }
        }
        for buffer in [a_buf, b_buf, out_buf] {
            gpu.release(buffer);
        }
        gpu.pool_pop(pool);
    }
}

unsafe fn ab_upload<T: Copy>(gpu: &Gpu, values: &[T]) -> Id {
    unsafe {
        let bytes = core::mem::size_of_val(values);
        let buffer = gpu.new_buffer(bytes).unwrap();
        std::ptr::copy_nonoverlapping(values.as_ptr().cast::<u8>(), gpu.buffer_contents(buffer), bytes);
        buffer
    }
}

unsafe fn ab_chunk_oracle(gpu: &Gpu, psos: &[Id], mul9: bool, next: &mut impl FnMut() -> F128) {
    const CHUNKS: usize = 2;
    let mut basis = [F128::ZERO; 128];
    for k in 0..8 {
        basis[k] = next();
    }
    for b in 1..16 {
        for k in 0..8 {
            basis[b * 8 + k] = crate::field::mul_by_x(basis[(b - 1) * 8 + k]);
        }
    }
    let mut convert = vec![F128::ZERO; 16 * 256];
    for b in 0..16 {
        for v in 1usize..256 {
            for k in 0..8 {
                if (v >> k) & 1 != 0 {
                    convert[b * 256 + v] += basis[b * 8 + k];
                }
            }
        }
    }
    let top: Vec<F128> = (0..256).map(|_| next()).collect();
    let bot: Vec<F128> = (0..16).map(|_| next()).collect();
    let hi = [F128::ONE, next()];
    let bmask = [0xffff, 0x5555, 0xaaaa, 0, 0x8001, 0x6996, 0xffff, 0x5555,
                 0xaaaa, 0, 0x8001, 0x6996, 0xffff, 0x5555, 0xaaaa, 0xffff];
    let ab: Vec<u8> = (0..(CHUNKS << ZC_AB_CHUNK_SHIFT)).map(|i| {
        (i as u64).wrapping_mul(0x9e37_79b9).rotate_left((i & 31) as u32) as u8
    }).collect();
    let weights: Vec<F128> = (0..4096).map(|lo| top[lo >> 4] * bot[lo & 15]).collect();
    let mut expected = [[F128::ZERO; 64]; CHUNKS];
    for chunk in 0..CHUNKS {
        for lo in 0..4096 {
            for lam in 0..64 {
                let mut row = F128::ZERO;
                for b in 0..16 {
                    if (bmask[b] >> (lo & 15)) & 1 != 0 {
                        let v = ab[(chunk << ZC_AB_CHUNK_SHIFT) + lo * 1024 + b * 64 + lam] as usize;
                        row += convert[b * 256 + v];
                    }
                }
                expected[chunk][lam] += weights[lo] * row;
            }
        }
        for lane in &mut expected[chunk] {
            *lane *= hi[chunk];
        }
    }
    unsafe {
        let basis_buf = ab_upload(gpu, &basis);
        let top_buf = ab_upload(gpu, &top);
        let bot_buf = ab_upload(gpu, &bot);
        let hi_buf = ab_upload(gpu, &hi);
        let ab_buf = ab_upload(gpu, &ab);
        let tables = gpu.new_buffer(ZC_AB_TOPS * 16 * 256 * 16).unwrap();
        let banks = gpu.new_buffer(CHUNKS * 32 * 16 * 64 * 16).unwrap();
        let out = gpu.new_buffer(CHUNKS * 64 * 16).unwrap();
        for tops in [8, zc_ab_tops_per_group(32)] {
            #[repr(C)]
            struct Params { tops: u32, groups: u32, shift: u32, pad: u32, bmask: [u32; 16] }
            let p = Params { tops: tops as u32, groups: (256 / tops) as u32, shift: ZC_AB_CHUNK_SHIFT, pad: 0, bmask };
            let bytes = core::slice::from_raw_parts((&raw const p).cast::<u8>(), core::mem::size_of::<Params>());
            std::ptr::write_bytes(gpu.buffer_contents(out), 0xa5, CHUNKS * 64 * 16);
            let mut walls = Vec::new();
            for pass in 0..3 {
                let cb = gpu.command_buffer().unwrap();
                let enc = gpu.compute_encoder(cb).unwrap();
                gpu.set_pipeline(enc, psos[pass]);
                let bindings = match pass {
                    0 => vec![basis_buf, top_buf, tables],
                    1 => vec![ab_buf, tables, banks],
                    _ => vec![banks, bot_buf, hi_buf, out],
                };
                for (index, buffer) in bindings.iter().enumerate() {
                    gpu.set_buffer(enc, *buffer, 0, index);
                }
                if pass != 0 {
                    gpu.set_bytes(enc, bytes, bindings.len());
                }
                let groups = match pass { 0 => 256, 1 => CHUNKS * 256 / tops, _ => CHUNKS };
                gpu.dispatch(enc, groups as u64, if pass == 0 { 256 } else { 64 });
                gpu.end_encoding(enc);
                gpu.commit_and_wait(cb).unwrap();
                walls.push(zc_fold_gpu_wall_ms(gpu, cb));
            }
            let actual = core::slice::from_raw_parts(gpu.buffer_contents(out).cast::<[F128; 64]>(), CHUNKS);
            assert_eq!(actual, expected.as_slice(), "per-chunk CPU oracle mul9={mul9} tops={tops}");
            eprintln!("[ab-multiply-oracle] mul9={mul9} chunks={CHUNKS} tops={tops} tables/gather/finish-ms={walls:?}");
        }
        for buffer in [basis_buf, top_buf, bot_buf, hi_buf, ab_buf, tables, banks, out] {
            gpu.release(buffer);
        }
    }
}
