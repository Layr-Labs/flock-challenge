    use super::*;
    #[test]
    fn delta_planes_producer_keeps_original_scratch_and_fallback() {
        const WORDS: usize = DeferredAb::UNIT_BYTES / 4;
        let blocks: Vec<Compression> = (0..12u32).map(|i| (
            std::array::from_fn(|j| i.wrapping_mul(0x1234567).wrapping_add(j as u32)),
            std::array::from_fn(|j| (i ^ (j as u32).wrapping_mul(0x9e3779b9)).rotate_left(j as u32)),
            (i as u64) << 37, i.wrapping_mul(13), i ^ 0xa5,
        )).collect();
        let mut e = vec![0xa5a5a5a5u32; 3 * WORDS];
        let mut d = e.clone();
        let mut sa = vec![0x5a5a5a5au32; WORDS];
        let mut sb = sa.clone();
        let mut ordinary_a = vec![0u32; WORDS];
        let mut ordinary_b = ordinary_a.clone();
        let mut expect_a = ordinary_a.clone();
        let mut expect_b = ordinary_a.clone();
        let handle = DeferredAb { delta_planes: true, blocks: &blocks, spec_init: None,
            gen_ilp: true, a: e.as_mut_ptr(), b: d.as_mut_ptr(), n_quads: 3,
            primed_scratch: true, copy_drain: true, input_mem: true, seed_in_place: true };
        for q in 0..3 {
            unsafe {
                witgen_simd::build_quad_split_neon_mem::<false>(
                    witgen_simd::QuadInput::Blocks(std::array::from_fn(|j| &blocks[4*q+j])),
                    ordinary_a.as_mut_ptr(), ordinary_b.as_mut_ptr(), expect_a.as_mut_ptr(), expect_b.as_mut_ptr());
            }
            handle.fill_quad(q, sa.as_mut_ptr().cast(), sb.as_mut_ptr().cast(), q != 0);
            assert_eq!(sa, expect_a);
            assert_eq!(sb, expect_b);
            let mut expected_e = vec![0u32; WORDS];
            let mut expected_d = expected_e.clone();
            unsafe { flock_core::zerocheck::delta_planes::publish(
                sa.as_ptr().cast(), sb.as_ptr().cast(), expected_e.as_mut_ptr().cast(), expected_d.as_mut_ptr().cast(), DeferredAb::UNIT_BYTES); }
            assert_eq!(&e[q*WORDS..(q+1)*WORDS], expected_e);
            assert_eq!(&d[q*WORDS..(q+1)*WORDS], expected_d);
            handle.fill_quad(q, core::ptr::null_mut(), core::ptr::null_mut(), false);
            assert_eq!(&e[q*WORDS..(q+1)*WORDS], expect_a);
            assert_eq!(&d[q*WORDS..(q+1)*WORDS], expect_b);
        }
    }
