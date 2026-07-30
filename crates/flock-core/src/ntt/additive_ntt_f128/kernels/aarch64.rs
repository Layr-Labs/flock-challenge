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

// ---------------------------------------------------------------------------
// Fused four-layer (16-point) top-layer kernel.
//
// The interleaved top layers are pure DRAM streaming: at the ranked shape the
// codeword is 1 GiB, so each layer sweep costs a read + a write of the whole
// buffer. Fusing four layers into one pass cuts that traffic 4×.
//
// Two things make the naive shape lose the traffic it saves:
//
//  * The 16 contributing rows sit `sixteenth` positions apart — a power of two,
//    up to 64 MiB at the top layer — so they collide in the same cache sets and
//    cannot all stay resident while the network runs.
//  * Holding all 16 values of one lane live across the 32 multiplies overruns
//    the NEON register file once the multiply temporaries are counted.
//
// So this kernel stages the row group into a small contiguous tile first. The
// tile is L1-resident and conflict-free, the 16 gathers/scatters against it are
// each a sequential run the prefetcher handles, and the network then runs
// lane-inner over the tile — the same shape as the fused-2 top-layer kernel,
// which is the fastest form measured here (see `butterfly_interleaved_block`
// on why explicit lane batching loses to per-lane ILP).
// ---------------------------------------------------------------------------

/// Lanes staged per tile pass. 16 rows × 64 lanes × 16 B = 16 KiB, comfortably
/// L1-resident, and 64 covers the protocol's `num_ntts` in a single pass so
/// each row's gather is one 1 KiB sequential run.
const FUSED4_TILE_LANES: usize = 64;

/// Process one fused-four-layer row group across every interleaved NTT lane.
///
/// Reads each of the 16 rows once and writes it once, applying layers
/// `L..L+4` in between. `twiddles` is laid out as `[0] = layer L`,
/// `[1..3] = L+1`, `[3..7] = L+2`, `[7..15] = L+3` (see the caller).
///
/// The network decomposes into eight 4-row fused-2 butterflies, so it reuses
/// [`super::portable::butterfly_fused_2layer`] — the same leaf the fused-2 top
/// layers run — with the row indices regrouped:
///
///  * layers `L, L+1` pair rows `(i, i+4, i+8, i+12)` for `i ∈ 0..4`, since
///    layer `L` butterflies `(i, i+8)` and layer `L+1` butterflies `(i, i+4)`;
///  * layers `L+2, L+3` pair rows `(4g, 4g+1, 4g+2, 4g+3)` for `g ∈ 0..4`,
///    since layer `L+2` butterflies `(4g, 4g+2)` and `L+3` butterflies
///    `(4g, 4g+1)`.
///
/// # Safety
/// Requires the `aes` target feature. The caller must ensure the 16 row
/// slices selected by `r` are valid and disjoint from any row group being
/// processed concurrently.
#[target_feature(enable = "aes")]
pub(super) unsafe fn butterfly_fused_4layer_row(
    ptr: *mut F128,
    sixteenth: usize,
    num_ntts: usize,
    r: usize,
    twiddles: &[F128; 15],
) {
    use super::portable::butterfly_fused_2layer;

    // SAFETY: caller supplies the pointer geometry and disjointness contract.
    unsafe {
        let row = |i: usize| ptr.add((i * sixteenth + r) * num_ntts);

        let mut tile = [F128::ZERO; 16 * FUSED4_TILE_LANES];
        let mut lane0 = 0usize;
        while lane0 < num_ntts {
            let n = FUSED4_TILE_LANES.min(num_ntts - lane0);
            let tile_ptr = tile.as_mut_ptr();
            // Tile row `i` holds lanes `lane0..lane0 + n` of buffer row `i`.
            let trow = |i: usize| tile_ptr.add(i * n);

            for i in 0..16 {
                core::ptr::copy_nonoverlapping(row(i).add(lane0), trow(i), n);
            }

            // Four disjoint tile rows. Distinct indices ⇒ disjoint `n`-element
            // ranges of `tile`, so the four slices never alias.
            //
            // The leaf is the *portable* fused-2, not a hand-vectorised one: a
            // 2-lane `ghash_mul_vec2_neon`-style variant of it was tried here
            // and lost by ~45 % (57 ms vs 39 ms on the top-layer sweep), the
            // same way the one on `butterfly_interleaved_block` did. Pairing
            // caps the multiply ILP at 2 where the compiler otherwise keeps
            // many independent scalar muls in flight across the tile row.
            let fuse2 = |a: usize, b: usize, c: usize, d: usize, t0: F128, t1: F128, t2: F128| {
                butterfly_fused_2layer(
                    core::slice::from_raw_parts_mut(trow(a), n),
                    core::slice::from_raw_parts_mut(trow(b), n),
                    core::slice::from_raw_parts_mut(trow(c), n),
                    core::slice::from_raw_parts_mut(trow(d), n),
                    t0,
                    t1,
                    t2,
                );
            };

            for i in 0..4 {
                fuse2(i, i + 4, i + 8, i + 12, twiddles[0], twiddles[1], twiddles[2]);
            }
            for g in 0..4 {
                fuse2(
                    4 * g,
                    4 * g + 1,
                    4 * g + 2,
                    4 * g + 3,
                    twiddles[3 + g],
                    twiddles[7 + 2 * g],
                    twiddles[8 + 2 * g],
                );
            }

            for i in 0..16 {
                core::ptr::copy_nonoverlapping(trow(i), row(i).add(lane0), n);
            }
            lane0 += n;
        }
    }
}
