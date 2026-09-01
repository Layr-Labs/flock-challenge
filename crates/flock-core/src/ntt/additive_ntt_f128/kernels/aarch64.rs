use crate::field::F128;

/// Process two butterflies at a time within a block sharing one twiddle.
///
/// # Safety
/// Requires the `aes` target feature.
#[target_feature(enable = "aes")]
pub(super) unsafe fn butterfly_block(chunk: &mut [F128], twiddle: F128, half: usize) {
    use crate::field::gf2_128::aarch64::ghash_mul_vec2_neon;

    debug_assert!(half >= 2);
    debug_assert_eq!(chunk.len(), 2 * half);
    let mut idx0 = 0;
    while idx0 < half {
        let idx1 = idx0 + half;
        let u_a = chunk[idx0];
        let v_a = chunk[idx1];
        let u_b = chunk[idx0 + 1];
        let v_b = chunk[idx1 + 1];

        // SAFETY: caller guarantees the aes target feature.
        let product = unsafe { ghash_mul_vec2_neon([twiddle, twiddle], [v_a, v_b]) };
        let new_u_a = F128 {
            lo: u_a.lo ^ product[0].lo,
            hi: u_a.hi ^ product[0].hi,
        };
        let new_u_b = F128 {
            lo: u_b.lo ^ product[1].lo,
            hi: u_b.hi ^ product[1].hi,
        };
        let new_v_a = F128 {
            lo: v_a.lo ^ new_u_a.lo,
            hi: v_a.hi ^ new_u_a.hi,
        };
        let new_v_b = F128 {
            lo: v_b.lo ^ new_u_b.lo,
            hi: v_b.hi ^ new_u_b.hi,
        };

        chunk[idx0] = new_u_a;
        chunk[idx1] = new_v_a;
        chunk[idx0 + 1] = new_u_b;
        chunk[idx1 + 1] = new_v_b;
        idx0 += 2;
    }
}

/// Process the single pair in each of two adjacent blocks with distinct
/// twiddles.
///
/// # Safety
/// Requires the `aes` target feature.
#[target_feature(enable = "aes")]
pub(super) unsafe fn butterfly_block_pair(chunk: &mut [F128], t_a: F128, t_b: F128) {
    use crate::field::gf2_128::aarch64::ghash_mul_vec2_neon;

    debug_assert_eq!(chunk.len(), 4);
    let u_a = chunk[0];
    let v_a = chunk[1];
    let u_b = chunk[2];
    let v_b = chunk[3];

    // SAFETY: caller guarantees the aes target feature.
    let product = unsafe { ghash_mul_vec2_neon([t_a, t_b], [v_a, v_b]) };
    let new_u_a = F128 {
        lo: u_a.lo ^ product[0].lo,
        hi: u_a.hi ^ product[0].hi,
    };
    let new_u_b = F128 {
        lo: u_b.lo ^ product[1].lo,
        hi: u_b.hi ^ product[1].hi,
    };
    let new_v_a = F128 {
        lo: v_a.lo ^ new_u_a.lo,
        hi: v_a.hi ^ new_u_a.hi,
    };
    let new_v_b = F128 {
        lo: v_b.lo ^ new_u_b.lo,
        hi: v_b.hi ^ new_u_b.hi,
    };

    chunk[0] = new_u_a;
    chunk[1] = new_v_a;
    chunk[2] = new_u_b;
    chunk[3] = new_v_b;
}

/// Non-temporal store of two adjacent F128s (32 B) from NEON registers.
///
/// Same helper shape as the promoted zerocheck `store_nt_q_pair`: q-form
/// `stnp` bypasses write-allocate so a cold destination line costs one DRAM
/// write instead of a read-for-ownership plus a write. No Rust intrinsic
/// emits `stnp`; raw `asm!`.
///
/// # Safety
/// `dst` must be valid for 32 bytes of writes and 16-byte aligned.
#[inline(always)]
unsafe fn store_nt_q_pair(
    dst: *mut F128,
    v0: core::arch::aarch64::uint8x16_t,
    v1: core::arch::aarch64::uint8x16_t,
) {
    unsafe {
        core::arch::asm!(
            "stnp {a:q}, {b:q}, [{p}]",
            a = in(vreg) v0,
            b = in(vreg) v1,
            p = in(reg) dst,
            options(nostack, preserves_flags)
        );
    }
}

/// Flush one staged row (`num_ntts` F128s) to `dst` with 32 B non-temporal
/// pairs. The row is L1-resident (just produced), so the loads are free; the
/// destination is a single sequential burst — one active DRAM write stream.
///
/// # Safety
/// `row` must be valid for `num_ntts` reads, `dst` for `num_ntts` writes,
/// both 16-byte aligned, and `num_ntts` must be even.
#[inline(always)]
unsafe fn flush_row_nt(row: *const F128, dst: *mut F128, num_ntts: usize) {
    use core::arch::aarch64::vld1q_u8;
    debug_assert_eq!(num_ntts % 2, 0);
    unsafe {
        let mut i = 0;
        while i < num_ntts {
            let v0 = vld1q_u8(row.add(i) as *const u8);
            let v1 = vld1q_u8(row.add(i + 1) as *const u8);
            store_nt_q_pair(dst.add(i), v0, v1);
            i += 2;
        }
    }
}

/// One rate-1/2 seed row group with a staged non-temporal flush.
///
/// Computes the same eight output rows as the unstaged pair
/// `butterfly_fused_2layer_row_from_sparse` (first codeword half) followed by
/// `butterfly_fused_2layer_row_from` (second half) — identical field-op
/// sequences, so the published bytes are identical — but stages them in an
/// 8-row stack block and publishes each row with `stnp q,q` 32 B pairs at
/// full-128 B-line granularity. The destinations are cold and next read only
/// by the layer-3 sweep (far beyond cache for the 1 GiB ranked codeword), so
/// bypassing write-allocate skips one DRAM read-for-ownership per destination
/// line. Streams per thread: four sequential source reads during compute
/// (stack writes stay in L1), then eight sequential row bursts flushed one at
/// a time — never more than the promoted ≤4 concurrent DRAM streams.
///
/// # Safety
/// Same contract as the unstaged kernel pair: the four source rows and all
/// eight destination rows selected by `r` must be valid, `src` must not
/// overlap `dst`, and concurrent calls must write disjoint destination row
/// groups. Additionally `num_ntts` must be a multiple of 8 and at most
/// [`super::SEED_NT_MAX_NTTS`], and `dst`, `dst + half_len` must be 128-byte
/// aligned (full-line non-temporal coverage).
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn seed_fused_2layer_row_group_nt(
    src: *const F128,
    dst: *mut F128,
    quarter: usize,
    num_ntts: usize,
    half_len: usize,
    r: usize,
    right_twiddle: F128,
    twiddles: &[F128; 3],
) {
    use core::mem::MaybeUninit;

    debug_assert!(num_ntts % 8 == 0 && num_ntts <= super::SEED_NT_MAX_NTTS);
    debug_assert_eq!(dst as usize % 128, 0);
    debug_assert_eq!((half_len * core::mem::size_of::<F128>()) % 128, 0);

    let [t_outer, t_inner_a, t_inner_b] = *twiddles;

    // Uninitialized staging block (8 KiB max): every slot below
    // `8 * num_ntts` is written before the flush reads it.
    let mut stage = MaybeUninit::<[F128; 8 * super::SEED_NT_MAX_NTTS]>::uninit();
    let sp = stage.as_mut_ptr() as *mut F128;

    unsafe {
        for lane in 0..num_ntts {
            let a = *src.add(r * num_ntts + lane);
            let b = *src.add((quarter + r) * num_ntts + lane);
            let c = *src.add((2 * quarter + r) * num_ntts + lane);
            let d = *src.add((3 * quarter + r) * num_ntts + lane);

            // First half: layer 1 and the left layer-2 butterfly have zero
            // twiddle (op-for-op the `_sparse` kernel).
            let mut b0 = b;
            let mut c0 = c;
            let mut d0 = d;
            c0 += a;
            d0 += b;
            b0 += a;
            let new_c = c0 + d0 * right_twiddle;
            d0 += new_c;
            c0 = new_c;
            *sp.add(lane) = a;
            *sp.add(num_ntts + lane) = b0;
            *sp.add(2 * num_ntts + lane) = c0;
            *sp.add(3 * num_ntts + lane) = d0;

            // Second half: full fused-2 twiddle tree (op-for-op
            // `butterfly_fused_2layer_row_from`).
            let mut a1 = a;
            let mut b1 = b;
            let mut c1 = c;
            let mut d1 = d;
            let new_a = a1 + c1 * t_outer;
            c1 += new_a;
            a1 = new_a;
            let new_b = b1 + d1 * t_outer;
            d1 += new_b;
            b1 = new_b;
            let new_a = a1 + b1 * t_inner_a;
            b1 += new_a;
            a1 = new_a;
            let new_c = c1 + d1 * t_inner_b;
            d1 += new_c;
            c1 = new_c;
            *sp.add(4 * num_ntts + lane) = a1;
            *sp.add(5 * num_ntts + lane) = b1;
            *sp.add(6 * num_ntts + lane) = c1;
            *sp.add(7 * num_ntts + lane) = d1;
        }

        // Flush: eight sequential 1 KiB (at ranked shape) non-temporal row
        // bursts, one active write stream at a time.
        for i in 0..4 {
            flush_row_nt(
                sp.add(i * num_ntts),
                dst.add((i * quarter + r) * num_ntts),
                num_ntts,
            );
        }
        let dst_hi = dst.add(half_len);
        for i in 0..4 {
            flush_row_nt(
                sp.add((4 + i) * num_ntts),
                dst_hi.add((i * quarter + r) * num_ntts),
                num_ntts,
            );
        }
    }
}
