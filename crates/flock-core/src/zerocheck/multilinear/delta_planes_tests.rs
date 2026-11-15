
fn delta_planes_assert_state(a: &UniSkipCompactFold, b: &UniSkipCompactFold) {
    assert_eq!(a.len(), b.len());
    for p in 0..a.len() {
        if (a.dead_tail || b.dead_tail) && (p & 127) >= 121 { continue; }
        let read = |c: &UniSkipCompactFold| {
            let i = c.anchor_index(p);
            let h = c.head_compact && (p & 127) < 8;
            let d = c.delta_offset(p);
            let ac = c.anchors[i];
            let bc = if h { c.head_b } else { c.anchors[i + 1] };
            let da = u64::from_le_bytes(c.deltas[d..d+8].try_into().unwrap());
            let db = if h && !c.deltas.is_preproduced() { 0 } else {
                u64::from_le_bytes(c.deltas[d+8..d+16].try_into().unwrap())
            };
            (ac, bc, da, db)
        };
        assert_eq!(read(a), read(b), "logical compact pair {p}");
    }
}

#[test]
fn delta_planes_round2_k_and_owner_receipts() {
    use crate::zerocheck::delta_planes;
    const TAG: u64 = 0xD317_7E57_0000_0001;
    for (m, ones_head) in [(16, false), (25, false), (25, true)] {
        let mut f = la_fixture(m, 0xD317_0001, true);
        if ones_head {
            for block in f.b_packed.chunks_exact_mut(2048) { block[..128].fill(255); }
        }
        let mut e = vec![0u8; f.a_packed.len()];
        let mut d = vec![F128::ZERO; e.len() / 16];
        unsafe { delta_planes::publish(f.a_packed.as_ptr(), f.b_packed.as_ptr(), e.as_mut_ptr(), d.as_mut_ptr().cast(), e.len()); }
        for z in [F128::ZERO, F128::ONE, f.rho_grid[3]] {
            f.table = UniSkipFoldTable::new(6, z);
            for k_only in [false, true] {
                let ordinary = ScratchBytes::from_initialized_f128(vec![LA_POISON; d.len()]);
                let (expected, m1, mi, la) = uni_skip_fold_and_round_pair_compact_padded_lookahead_for(
                    &f.a_packed, &f.b_packed, f.m, 6, &f.table, &f.mlv, &f.padding, Some(ordinary), k_only);
                let donor = vec![LA_POISON; d.len()];
                let donor_ptr = donor.as_ptr();
                let reservation = ScratchBytes::from_initialized_f128_for_full_overwrite(donor, TAG);
                let (owner, returned) = delta_planes::Owner::new(d.clone());
                let delta_ptr = owner.storage.as_ptr();
                let adopted = reservation.adopt_produced(owner);
                let bytes = unsafe { core::slice::from_raw_parts(delta_ptr.cast::<u8>(), e.len()) };
                let (actual, a1, ai, ala) = uni_skip_fold_and_round_pair_compact_padded_lookahead_for(
                    &e, bytes, f.m, 6, &f.table, &f.mlv, &f.padding, Some(adopted), k_only);
                assert_eq!((a1, ai, ala), (m1, mi, la));
                assert_eq!(actual.dead_tail, expected.dead_tail);
                assert_eq!(actual.deltas.as_ptr(), delta_ptr.cast::<u8>());
                delta_planes_assert_state(&actual, &expected);

                for (r1, r2) in [(F128::ZERO,F128::ONE), (f.rho_grid[3],f.rho_grid[2])] {
                    let n = actual.len()/2;
                    let mut ea=vec![LA_POISON;n]; let mut eb=ea.clone();
                    let mut aa=ea.clone(); let mut ab=ea.clone();
                    let em=fold2_compact_and_round45_into_packed(&expected,&f.table,r1,r2,&la_r_next4(&f),&mut ea,&mut eb,true);
                    let am=fold2_compact_and_round45_into_packed(&actual,&f.table,r1,r2,&la_r_next4(&f),&mut aa,&mut ab,true);
                    assert_eq!(am,em); assert_eq!(aa,ea); assert_eq!(ab,eb);
                }
                expected.recycle(); actual.recycle();
                let held=returned.take(); assert_eq!(held.as_ptr(),delta_ptr); assert_eq!(held,d);
                let (old,hit)=crate::scratch::take_f128_with_token(d.len(),TAG);
                assert!(hit); assert_eq!(old.as_ptr(),donor_ptr); assert!(old.iter().all(|x| *x==LA_POISON));
                crate::scratch::give_f128(old);
            }
            // The exceptional no-lookahead route consumes the same E/D owner.
            f.mlv[1]=F128::ZERO;
            let (expected,m1,mi)=uni_skip_fold_and_round_pair_compact_padded(&f.a_packed,&f.b_packed,m,6,&f.table,&f.mlv,&f.padding);
            let (owner,returned)=delta_planes::Owner::new(d.clone());
            let ptr=owner.storage.as_ptr();
            let adopted=ScratchBytes::from_initialized_f128(vec![LA_POISON;d.len()]).adopt_produced(owner);
            let bytes=unsafe{core::slice::from_raw_parts(ptr.cast::<u8>(),e.len())};
            let (actual,a1,ai)=uni_skip_fold_and_round_pair_compact_padded_with_deltas(&e,bytes,m,6,&f.table,&f.mlv,&f.padding,Some(adopted));
            assert_eq!((a1,ai),(m1,mi)); assert_eq!(actual.anchors,expected.anchors); assert_eq!(&*actual.deltas,&*expected.deltas);
            expected.recycle();actual.recycle();assert_eq!(returned.take(),d);
            f.mlv[1]=F128::ONE;
        }
    }
}

#[cfg(target_os = "macos")]
#[test]
fn delta_planes_gpu_failure_redo_matches_cpu() {
    use crate::{gpu_commit as gpu, zerocheck::delta_planes};
    let f=la_fixture(26,0xD317_FA11,true);
    let mut e=vec![0u8;f.a_packed.len()];let mut d=vec![F128::ZERO;e.len()/16];
    unsafe { delta_planes::publish(f.a_packed.as_ptr(),f.b_packed.as_ptr(),e.as_mut_ptr(),d.as_mut_ptr().cast(),e.len()); }
    gpu::zc_r2_test_reset();gpu::zc_r2_test_set_share(0);
    let (expected,m1,mi,la)=uni_skip_fold_and_round_pair_compact_padded_lookahead_for(
        &f.a_packed,&f.b_packed,f.m,6,&f.table,&f.mlv,&f.padding,
        Some(ScratchBytes::from_initialized_f128(vec![LA_POISON;d.len()])),true);
    for lookahead in [false,true] {
        let (owner,returned)=delta_planes::Owner::new(d.clone());let ptr=owner.storage.as_ptr();
        let adopted=ScratchBytes::from_initialized_f128(vec![LA_POISON;d.len()]).adopt_produced(owner);
        let bytes=unsafe{core::slice::from_raw_parts(ptr.cast::<u8>(),e.len())};
        gpu::zc_r2_test_reset();gpu::zc_r2_test_set_share(8);gpu::zc_r2_test_fail_next();
        let (actual,a1,ai)=if lookahead {
            let (actual,a1,ai,ala)=uni_skip_fold_and_round_pair_compact_padded_lookahead_for(
                &e,bytes,f.m,6,&f.table,&f.mlv,&f.padding,Some(adopted),true);
            assert_eq!(ala,la);(actual,a1,ai)
        } else {
            uni_skip_fold_and_round_pair_compact_padded_with_deltas(
                &e,bytes,f.m,6,&f.table,&f.mlv,&f.padding,Some(adopted))
        };
        assert!(!gpu::zc_r2_test_failure_pending(),"the GPU prefix must actually run");
        assert_eq!((a1,ai),(m1,mi));
        delta_planes_assert_state(&actual, &expected);
        actual.recycle();assert_eq!(returned.take(),d);
    }
    expected.recycle();gpu::zc_r2_test_reset();
}

#[cfg(target_os = "macos")]
#[test]
fn delta_planes_even_plane_and_joined_deltas() {
    use crate::zerocheck::delta_planes;
    let f = la_fixture(16, 0xD317_C01D, true);
    let inputs = Round2PlaneProbe::new(&f.a_packed, &f.b_packed, 8, false);
    assert!(inputs.dense_deltas.is_none());
    let e = &inputs.even;
    let mut expected_e = vec![0u8; e.len()];
    let mut expected_d = vec![0u8; e.len()];
    unsafe { delta_planes::publish(f.a_packed.as_ptr(), f.b_packed.as_ptr(),
        expected_e.as_mut_ptr(), expected_d.as_mut_ptr(), e.len()); }
    assert_eq!(*e, expected_e);
    let compacted = Round2PlaneProbe::new(&f.a_packed, &f.b_packed, 8, true);
    assert_eq!(compacted.even, expected_e);
    assert_eq!(compacted.dense_deltas.as_deref(), Some(expected_d.as_slice()));
    for lookahead in [false, true] {
        let (expected, p1, pi) = if lookahead {
            let (c, p1, pi, _) = uni_skip_fold_and_round_pair_compact_padded_lookahead_for(
                &f.a_packed, &f.b_packed, f.m, 6, &f.table, &f.mlv, &f.padding, None, false);
            (c, p1, pi)
        } else {
            uni_skip_fold_and_round_pair_compact_padded(&f.a_packed, &f.b_packed,
                f.m, 6, &f.table, &f.mlv, &f.padding)
        };
        assert_eq!(&*expected.deltas, expected_d.as_slice());
        let mut storage = vec![F128::ZERO; e.len() / 16];
        unsafe { core::ptr::copy_nonoverlapping(expected.deltas.as_ptr(),
            storage.as_mut_ptr().cast::<u8>(), e.len()); }
        let (owner, returned) = delta_planes::Owner::new(storage);
        let ptr = owner.storage.as_ptr();
        let adopted = ScratchBytes::from_initialized_f128(vec![LA_POISON; e.len()/16]).adopt_produced(owner);
        let bytes = unsafe { core::slice::from_raw_parts(ptr.cast::<u8>(), e.len()) };
        let (actual, a1, ai) = if lookahead {
            let (c, p1, pi, _) = uni_skip_fold_and_round_pair_compact_padded_lookahead_for(
                &e, bytes, f.m, 6, &f.table, &f.mlv, &f.padding, Some(adopted), false);
            (c, p1, pi)
        } else {
            uni_skip_fold_and_round_pair_compact_padded_with_deltas(&e, bytes, f.m, 6,
                &f.table, &f.mlv, &f.padding, Some(adopted))
        };
        assert_eq!((a1, ai), (p1, pi));
        assert_eq!(actual.anchors, expected.anchors);
        assert_eq!(&*actual.deltas, &*expected.deltas);
        expected.recycle(); actual.recycle();
        let _ = returned.take();
    }
}

#[cfg(all(target_os = "macos", target_feature = "sha3"))]
#[test]
fn delta_planes_warmup_probe_preserves_compacted_output() {
    use kernels::aarch64::{HC_ANCHORS_PER_BLOCK, TC_ANCHORS_PER_BLOCK, TC_ANCHOR_GAP};
    for ones_head in [false, true] {
        let mut f = la_fixture(25, 0xD317_C01E, true);
        if ones_head {
            for block in f.b_packed.chunks_exact_mut(2048) { block[..128].fill(255); }
        }
        let (mut compact, p1, pi, _) = uni_skip_fold_and_round_pair_compact_padded_lookahead_for(
            &f.a_packed, &f.b_packed, f.m, 6, &f.table, &f.mlv, &f.padding,
            Some(ScratchBytes::from_initialized_f128(vec![LA_POISON; f.a_packed.len()/16])), true);
        assert!(compact.tail_compact);
        let saved_anchors = compact.anchors.clone();
        let saved_deltas = compact.deltas.to_vec();
        let inputs = Round2PlaneProbe::new(&f.a_packed, &f.b_packed, 8, true);
        let delta = inputs.dense_deltas.as_ref().unwrap();
        let eq = SplitEqGhash::with_n_hi(&f.mlv[1..], lookahead_n_hi(f.mlv.len()-1));
        let lo = 1usize << eq.n_lo;
        let hi = 1usize << eq.n_hi;
        let prep = kernels::aarch64::round2_eq_prep(&eq.lo);
        let (mask, useful) = round2_pair_skip(&f.padding, 6);
        let (kappa, _) = lookahead_kappa_and_retained_inv(f.mlv[1], true);
        let mut sums = vec![(F128::ZERO, F128::ZERO); hi];
        let mut la = vec![[F128::ZERO; 6]; hi];
        let failed = std::sync::atomic::AtomicBool::new(false);
        let tail_base = compact.tail_base_blocks * TC_ANCHOR_GAP;
        let tail = crate::epool::SyncPtr(unsafe { compact.anchors.as_mut_ptr().add(tail_base) });
        let sums_ptr = crate::epool::SyncPtr(sums.as_mut_ptr());
        let la_ptr = crate::epool::SyncPtr(la.as_mut_ptr());
        // A non-Horner light sweep uses tail packing even if the ordinary
        // full sweep allowed head packing. Price it, then restore the proof.
        for prefix in [hi, 0] {
            round2_lookahead_cpu_sweep::<true>(
                &inputs.even, delta, &f.table, &eq.lo, &eq.hi, 8, lo, hi,
                lo / 128 * TC_ANCHORS_PER_BLOCK, lo * 16, mask, useful, true,
                tail, crate::epool::SyncPtr(delta.as_ptr().cast_mut()), sums_ptr, la_ptr,
                prep.as_ptr() as usize, kappa, true, true, true, 0, false, false, false,
                prefix, false);
        }
        let sum = sums.iter().fold((F128::ZERO, F128::ZERO), |(a,b), &(c,d)| (a+c,b+d));
        assert_eq!((f.mlv[0]*sum.0, sum.1), (p1,pi));
        assert_eq!(&*compact.deltas, saved_deltas.as_slice());
        let stride = lo / 128 * if compact.head_compact { HC_ANCHORS_PER_BLOCK } else { TC_ANCHORS_PER_BLOCK };
        let abase = compact.anchor_index(0);
        let dbase = compact.delta_offset(0);
        let dstride = lo / 128 * if compact.head_compact {
            kernels::aarch64::HC_DELTA_BYTES_PER_BLOCK
        } else { kernels::aarch64::TC_DELTA_BYTES_PER_BLOCK };
        round2_lookahead_cpu_sweep::<false>(
            &f.a_packed, &f.b_packed, &f.table, &eq.lo, &eq.hi, 8, lo, hi,
            stride, dstride, mask, useful, true,
            crate::epool::SyncPtr(unsafe { compact.anchors.as_mut_ptr().add(abase) }),
            crate::epool::SyncPtr(unsafe { compact.deltas.full_overwrite_ptr().add(dbase) }), sums_ptr, la_ptr,
            prep.as_ptr() as usize, kappa, true, true, true,
            if compact.head_compact { &failed as *const _ as usize } else { 0 },
            false, false, false, 0, false);
        for pair in 0..compact.len() {
            if pair & 127 >= 121 { continue; }
            let i = compact.anchor_index(pair);
            assert_eq!(compact.anchors[i], saved_anchors[i]);
            if !compact.head_compact || pair & 127 >= 8 {
                assert_eq!(compact.anchors[i+1], saved_anchors[i+1]);
            }
        }
        assert!(!failed.load(std::sync::atomic::Ordering::Relaxed));
        assert_eq!(&*compact.deltas, saved_deltas.as_slice());
        compact.recycle();
    }
}
