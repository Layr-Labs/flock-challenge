
#[test]
fn delta_planes_all_cpu_round2_variants() {
    use crate::zerocheck::multilinear::UniSkipFoldTable;
    use crate::zerocheck::univariate_skip::build_eq;
    use crate::zerocheck::univariate_skip_optimized::medium_challenges_ghash;
    const LO: usize = 512;
    let mut rng = 0xD317_C0DE_1234u64;
    let mut a = vec![0u8;16*LO]; let mut b=a.clone();
    for x in a.iter_mut().chain(b.iter_mut()) { *x=splitmix64(&mut rng) as u8; }
    for block in 0..LO/128 {
        b[block*2048..block*2048+4*16].fill(255);
        b[block*2048+5*16..block*2048+6*16].fill(255);
        a[block*2048+121*16..(block+1)*2048].fill(0);
        b[block*2048+121*16..(block+1)*2048].fill(0);
    }
    let mut e=vec![0u8;a.len()];let mut d=e.clone();
    unsafe { crate::zerocheck::delta_planes::publish(a.as_ptr(),b.as_ptr(),e.as_mut_ptr(),d.as_mut_ptr(),a.len()); }
    let mut r=vec![F128::ONE;9];
    r[2..6].copy_from_slice(&medium_challenges_ghash());
    let eq=build_eq(&r);
    #[cfg(target_feature="sha3")]
    let prep=round2_eq_prep(&eq);
    #[cfg(not(target_feature="sha3"))]
    let prep=Vec::<F128>::new();
    let pp=if prep.is_empty(){core::ptr::null()}else{prep.as_ptr()};
    for z in [F128::ZERO,F128::ONE,F128::new(12345,98765)] {
        let table=UniSkipFoldTable::new(6,z);
        for degen in [false,true] { for skip in [false,true] { for periodic in [false,true] {
            let poison=F128::new(u64::MAX,0xabad1dea);
            macro_rules! run {
                ($planes:literal,$kind:expr) => {{
                    let (ap,bp)=if $planes {(e.as_ptr(),d.as_ptr())}else{(a.as_ptr(),b.as_ptr())};
                    let mut anchors=vec![poison;2*LO];let mut dst=vec![0x5au8;16*LO];let mut out=[F128::ZERO;8];
                    let t=table.data.as_ptr().cast();let an=anchors.as_mut_ptr();let dd=dst.as_mut_ptr();let ep=eq.as_ptr();let o=out.as_mut_ptr();
                    unsafe { match $kind {
                        0 => {let v=fold_round2_compact_chunk_neon_unchecked_8::<$planes>(t,ap,bp,an,dd,ep,LO,0,127,121,degen);out[0]=v.0;out[1]=v.1;},
                        1 => fold_round2_compact_chunk_neon_anchors_only_8::<$planes>(t,ap,bp,an,dd,LO,0,127,121),
                        2 => fold_round2_compact_chunk_neon_lookahead_8_impl::<true,false,false,$planes>(t,ap,bp,an,dd,ep,pp,LO,0,127,121,degen,periodic,skip,false,core::ptr::null(),o),
                        3 => fold_round2_compact_chunk_neon_lookahead_8_impl::<false,false,false,$planes>(t,ap,bp,an,dd,ep,pp,LO,0,127,121,degen,periodic,skip,false,core::ptr::null(),o),
                        4 => fold_round2_compact_chunk_neon_lookahead_8_impl::<false,true,false,$planes>(t,ap,bp,an,dd,ep,pp,LO,0,127,121,degen,periodic,skip,false,core::ptr::null(),o),
                        5 => fold_round2_compact_chunk_neon_horner_8_impl::<false,false,$planes>(t,ap,bp,an,dd,ep,LO,0,127,121,degen,skip,false,core::ptr::null(),o),
                        6 => fold_round2_compact_chunk_neon_horner_8_impl::<true,true,$planes>(t,ap,bp,an,dd,ep,LO,0,127,121,degen,skip,false,core::ptr::null(),o),
                        #[cfg(target_feature="sha3")]
                        7 => fold_round2_compact_chunk_neon_lookahead_8_impl::<true,false,true,$planes>(t,ap,bp,an,dd,ep,pp,LO,0,127,121,degen,periodic,skip,false,core::ptr::null(),o),
                        #[cfg(target_feature="sha3")]
                        8 => fold_round2_compact_chunk_neon_prescaled_8_impl::<false,false,false,false,$planes>(t,ap,bp,an,dd,ep,LO,0,127,121,degen,skip,false,core::ptr::null(),o),
                        #[cfg(target_feature="sha3")]
                        9 => fold_round2_compact_chunk_neon_prescaled_8_impl::<true,true,true,true,$planes>(t,ap,bp,an,dd,ep,LO,0,127,121,degen,skip,false,core::ptr::null(),o),
                        _ => unreachable!(),
                    }}
                    (anchors,dst,out)
                }};
            }
            let kinds=if cfg!(target_feature="sha3"){10}else{7};
            for kind in 0..kinds {
                let old=run!(false,kind);let new=run!(true,kind);
                assert_eq!(new.0,old.0,"anchors kind={kind} degen={degen} skip={skip}");
                assert_eq!(new.2,old.2,"arithmetic kind={kind}");
                assert!(new.1.iter().all(|x|*x==0x5a),"a delta store survived in kind={kind}");
            }
        }}}
    }
}

/// Exercise the compact head helpers directly, including their failure flag.
/// Compare against the same ordinary-row kernels and poison all delta stores.
#[cfg(target_feature = "sha3")]
#[test]
fn delta_planes_compacted_cpu_walks() {
    use crate::zerocheck::{multilinear::UniSkipFoldTable, univariate_skip::build_eq,
        univariate_skip_optimized::medium_challenges_ghash};
    const LO: usize = 512;
    let mut seed = 0xD317_2233_u64;
    let mut a = vec![0u8; LO * 16];
    let mut b = a.clone();
    for x in a.iter_mut().chain(b.iter_mut()) { *x = splitmix64(&mut seed) as u8; }
    let mut r = vec![F128::ONE; 9];
    r[2..6].copy_from_slice(&medium_challenges_ghash());
    let eq = build_eq(&r);
    let prep = round2_eq_prep(&eq);
    for valid_head in [false, true] {
        for blk in 0..LO/128 {
            b[blk*2048..blk*2048+128].fill(if valid_head { 255 } else { 123 });
            a[blk*2048+121*16..(blk+1)*2048].fill(0);
            b[blk*2048+121*16..(blk+1)*2048].fill(0);
        }
        let mut e = vec![0u8; a.len()]; let mut d = e.clone();
        unsafe { crate::zerocheck::delta_planes::publish(a.as_ptr(), b.as_ptr(), e.as_mut_ptr(), d.as_mut_ptr(), a.len()); }
        for head in [false, true] {
            // Head compaction is supported by the full prepared-offset and
            // ranked prescaled walks. Other light routes use tail-only.
            for kind in 0..6 {
                if head && kind != 0 && kind != 1 { continue; }
                let table = UniSkipFoldTable::new(6, F128::new(789, 123));
                assert_eq!(table.fold_one_row(&[255;8]), F128::ONE);
                let (ab, db) = if head {
                    (LO/128*HC_ANCHOR_GAP + HC_ANCHOR_SHIFT, LO/128*HC_DELTA_GAP + HC_DELTA_SHIFT)
                } else { (LO/128*TC_ANCHOR_GAP, LO/128*TC_DELTA_GAP) };
                macro_rules! run {
                    ($planes:literal) => {{
                        let (ap,bp) = if $planes { (e.as_ptr(),d.as_ptr()) } else { (a.as_ptr(),b.as_ptr()) };
                        let mut anchors = vec![F128::new(23,45); 2*LO];
                        let mut deltas = vec![0x5au8; 16*LO];
                        let mut out = [F128::ZERO;8];
                        let failed = std::sync::atomic::AtomicBool::new(false);
                        let fail = if head { &failed as *const _ } else { core::ptr::null() };
                        unsafe {
                            let an = anchors.as_mut_ptr().add(ab);
                            let dd = deltas.as_mut_ptr().add(if $planes {0} else {db});
                            let t = table.data.as_ptr().cast();
                            let ep = eq.as_ptr(); let pp = prep.as_ptr(); let op = out.as_mut_ptr();
                            match kind {
                                0 => fold_round2_compact_chunk_neon_lookahead_8_impl::<true,false,true,$planes>(t,ap,bp,an,dd,ep,pp,LO,0,127,121,true,true,true,true,fail,op),
                                1 => fold_round2_compact_chunk_neon_prescaled_8_impl::<true,true,true,true,$planes>(t,ap,bp,an,dd,ep,LO,0,127,121,true,true,true,fail,op),
                                2 => fold_round2_compact_chunk_neon_lookahead_8_impl::<false,true,false,$planes>(t,ap,bp,an,dd,ep,pp,LO,0,127,121,true,true,true,true,fail,op),
                                3 => fold_round2_compact_chunk_neon_lookahead_8_impl::<false,false,false,$planes>(t,ap,bp,an,dd,ep,pp,LO,0,127,121,true,true,true,true,fail,op),
                                4 => fold_round2_compact_chunk_neon_horner_8_impl::<true,true,$planes>(t,ap,bp,an,dd,ep,LO,0,127,121,true,true,true,fail,op),
                                5 => fold_round2_compact_chunk_neon_prescaled_8_impl::<false,false,false,false,$planes>(t,ap,bp,an,dd,ep,LO,0,127,121,true,true,true,fail,op),
                                _ => unreachable!(),
                            }
                        }
                        (anchors, deltas, out, failed.into_inner())
                    }};
                }
                let ordinary = run!(false); let planes = run!(true);
                assert_eq!(planes.0, ordinary.0, "anchors head={head} valid={valid_head} kind={kind}");
                assert_eq!(planes.2, ordinary.2, "products head={head} valid={valid_head} kind={kind}");
                assert_eq!(planes.3, ordinary.3);
                assert_eq!(planes.3, head && !valid_head);
                assert!(planes.1.iter().all(|&v| v == 0x5a), "delta store head={head} kind={kind}");
            }
        }
    }
}
