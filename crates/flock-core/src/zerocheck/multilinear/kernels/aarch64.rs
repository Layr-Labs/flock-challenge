use crate::field::{F128, F256Unreduced};

#[cfg(target_os = "macos")]
core::arch::global_asm!(include_str!("sme_pmull32_tail_x8.S"), options(raw));

#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
#[repr(C)]
struct SmePmull32Twiddle {
    lo: u64,
    lo_swap32: u64,
    hi: u64,
    hi_swap32: u64,
    cross: u64,
    cross_swap32: u64,
}

#[cfg(target_os = "macos")]
impl SmePmull32Twiddle {
    #[inline]
    fn new(value: F128) -> Self {
        let cross = value.lo ^ value.hi;
        Self {
            lo: value.lo,
            lo_swap32: value.lo.rotate_left(32),
            hi: value.hi,
            hi_swap32: value.hi.rotate_left(32),
            cross,
            cross_swap32: cross.rotate_left(32),
        }
    }
}

#[cfg(target_os = "macos")]
unsafe extern "C" {
    fn flock_sme_pmull32_tail_svl_bytes() -> usize;
    fn flock_sme_pmull32_tail_x8(
        a_out: *mut F128,
        b_out: *mut F128,
        a_in: *const F128,
        b_in: *const F128,
        eq_lo: *const F128,
        lo_size: usize,
        rho: *const SmePmull32Twiddle,
        messages: *mut u64,
    );
}

#[cfg(target_os = "macos")]
fn apple_arm_capability(bit: u32) -> bool {
    use core::ffi::{c_char, c_void};

    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            old_value: *mut c_void,
            old_len: *mut usize,
            new_value: *mut c_void,
            new_len: usize,
        ) -> i32;
    }

    // XNU's public arm/cpu_capabilities_public.h defines the stable bit ABI
    // for this aggregate sysctl. FEAT_SME2 is bit 41.
    let mut caps = [0_u64; 2];
    let mut caps_len = core::mem::size_of_val(&caps);
    // SAFETY: the key is static and NUL-terminated, the destination is live,
    // and this is a read-only query with no replacement value.
    let result = unsafe {
        sysctlbyname(
            c"hw.optional.arm.caps".as_ptr(),
            caps.as_mut_ptr().cast(),
            &mut caps_len,
            core::ptr::null_mut(),
            0,
        )
    };
    result == 0
        && caps_len >= core::mem::size_of::<u64>()
        && (caps[(bit / 64) as usize] & (1_u64 << (bit % 64))) != 0
}

#[cfg(target_os = "macos")]
fn sme2_pmull32_tail_available() -> bool {
    use std::sync::OnceLock;

    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        const CAP_BIT_FEAT_SME2: u32 = 41;
        if !apple_arm_capability(CAP_BIT_FEAT_SME2) {
            return false;
        }
        // SAFETY: the capability bit makes entering streaming mode legal.
        unsafe { flock_sme_pmull32_tail_svl_bytes() == 64 }
    })
}

#[cfg(target_os = "macos")]
unsafe fn fold_and_message_sme2(
    a_in: &[F128],
    b_in: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    r_fold: F128,
    eq_lo: &[F128],
) -> (F128, F128) {
    debug_assert_eq!(a_in.len(), b_in.len());
    debug_assert_eq!(a_out.len(), b_out.len());
    debug_assert_eq!(a_in.len(), 2 * a_out.len());
    debug_assert_eq!(a_out.len(), 2 * eq_lo.len());
    debug_assert_eq!(eq_lo.len() & 7, 0);

    let twiddle = SmePmull32Twiddle::new(r_fold);
    // Words 4..11 seed the deferred accumulators. Production starts at zero;
    // retaining the input/output form lets the semantic gate poison them.
    let mut messages = [0_u64; 12];
    // SAFETY: the slice geometry above exactly matches the assembly contract;
    // runtime capability and SVL checks are made by the dispatching wrapper.
    unsafe {
        flock_sme_pmull32_tail_x8(
            a_out.as_mut_ptr(),
            b_out.as_mut_ptr(),
            a_in.as_ptr(),
            b_in.as_ptr(),
            eq_lo.as_ptr(),
            eq_lo.len(),
            &twiddle,
            messages.as_mut_ptr(),
        );
    }
    (
        F128::new(messages[0], messages[1]),
        F128::new(messages[2], messages[3]),
    )
}

/// Fold two adjacent output values while keeping both results in registers.
/// `out_base` is measured in folded/output elements, so the four source
/// elements begin at `2 * out_base`.
///
/// # Safety
/// Requires the `aes` target feature and four in-bounds source elements.
#[inline]
#[target_feature(enable = "aes")]
unsafe fn fold_two(src: &[F128], out_base: usize, r: F128) -> [F128; 2] {
    use crate::field::gf2_128::aarch64::ghash_mul_vec2_neon;

    let s = 2 * out_base;
    let e0 = src[s];
    let o0 = src[s + 1];
    let e1 = src[s + 2];
    let o1 = src[s + 3];
    // SAFETY: this helper carries the required target feature.
    let prod = unsafe { ghash_mul_vec2_neon([r, r], [e0 + o0, e1 + o1]) };
    [e0 + prod[0], e1 + prod[1]]
}

/// ARM fused tail kernel: fold `a_in`/`b_in` and form the next sumcheck
/// message without rereading the freshly written output arrays.
///
/// The generic ARM path first writes all folded values, then streams both
/// output arrays again. At the benchmark's first tail round that is 64 MiB of
/// avoidable reads. This kernel uses the same two-output PMULL fold primitive,
/// but consumes each pair directly from registers before moving on.
///
/// # Safety
/// Requires the `aes` target feature. The caller must provide
/// `a_in.len() == b_in.len() == 2 * a_out.len()`, equal output lengths, and
/// `eq_lo.len() * 2 == a_out.len()` with an even `eq_lo.len()`.
#[target_feature(enable = "aes")]
pub(crate) unsafe fn fold_and_message_neon(
    a_in: &[F128],
    b_in: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    r_fold: F128,
    eq_lo: &[F128],
) -> (F128, F128) {
    #[cfg(target_os = "macos")]
    if eq_lo.len() >= 128 && sme2_pmull32_tail_available() {
        // SAFETY: this wrapper has checked SME2, SVL=64, and the assembly
        // threshold; the caller supplies the same geometry as the NEON leaf.
        return unsafe { fold_and_message_sme2(a_in, b_in, a_out, b_out, r_fold, eq_lo) };
    }

    // SAFETY: this wrapper carries AES and forwards the caller's geometry.
    unsafe { fold_and_message_neon_impl(a_in, b_in, a_out, b_out, r_fold, eq_lo) }
}

#[target_feature(enable = "aes")]
unsafe fn fold_and_message_neon_impl(
    a_in: &[F128],
    b_in: &[F128],
    a_out: &mut [F128],
    b_out: &mut [F128],
    r_fold: F128,
    eq_lo: &[F128],
) -> (F128, F128) {
    debug_assert_eq!(a_in.len(), b_in.len());
    debug_assert_eq!(a_out.len(), b_out.len());
    debug_assert_eq!(a_in.len(), 2 * a_out.len());
    debug_assert_eq!(a_out.len(), 2 * eq_lo.len());
    debug_assert_eq!(eq_lo.len() & 1, 0);

    let mut p1_acc = F256Unreduced::ZERO;
    let mut pinf_acc = F256Unreduced::ZERO;
    let mut x_lo = 0;
    while x_lo < eq_lo.len() {
        let o = 2 * x_lo;
        // Two x_lo points = four adjacent folded values per witness. Keeping
        // this at two points limits register pressure while exposing four
        // independent two-lane fold products to the out-of-order engine.
        let aa = unsafe { fold_two(a_in, o, r_fold) };
        let ab = unsafe { fold_two(a_in, o + 2, r_fold) };
        let ba = unsafe { fold_two(b_in, o, r_fold) };
        let bb = unsafe { fold_two(b_in, o + 2, r_fold) };

        a_out[o] = aa[0];
        a_out[o + 1] = aa[1];
        a_out[o + 2] = ab[0];
        a_out[o + 3] = ab[1];
        b_out[o] = ba[0];
        b_out[o + 1] = ba[1];
        b_out[o + 2] = bb[0];
        b_out[o + 3] = bb[1];

        let g1_a = aa[1] * ba[1];
        let g1_b = ab[1] * bb[1];
        let g_inf_a = (aa[0] + aa[1]) * (ba[0] + ba[1]);
        let g_inf_b = (ab[0] + ab[1]) * (bb[0] + bb[1]);
        p1_acc ^= eq_lo[x_lo].mul_unreduced(g1_a);
        p1_acc ^= eq_lo[x_lo + 1].mul_unreduced(g1_b);
        pinf_acc ^= eq_lo[x_lo].mul_unreduced(g_inf_a);
        pinf_acc ^= eq_lo[x_lo + 1].mul_unreduced(g_inf_b);

        x_lo += 2;
    }

    (p1_acc.reduce(), pinf_acc.reduce())
}

/// Raw-pointer leaf for one round-2 `x_hi` chunk at the protocol-fixed
/// `k_skip = 6` / eight-table-chunk geometry.
///
/// The caller advances every pointer to the beginning of one Rayon chunk.
/// Keeping this loop in a noinline leaf prevents the closure's slice and
/// capture state from occupying registers across the table-lookup and GHASH
/// dependency chains.
///
/// # Safety
/// Requires the `aes` target feature and all of the following:
///
/// - `table_data` addresses at least `8 * 256 * size_of::<F128>()` bytes;
/// - `a_packed` and `b_packed` each address `2 * lo_size * 8` bytes;
/// - `a_out` and `b_out` each address `2 * lo_size` writable `F128`s;
/// - `eq_lo` addresses `lo_size` initialized `F128`s;
/// - `pair_idx_base + lo_size` does not overflow `usize`.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
#[target_feature(enable = "aes")]
pub(crate) unsafe fn round2_chunk_raw_neon(
    table_data: *const u8,
    a_packed: *const u8,
    b_packed: *const u8,
    a_out: *mut F128,
    b_out: *mut F128,
    eq_lo: *const F128,
    lo_size: usize,
    pair_idx_base: usize,
    pair_in_block_mask: usize,
    useful_pairs_inclusive: usize,
) -> (F128, F128) {
    unsafe {
        let mut p1_acc = F256Unreduced::ZERO;
        let mut pinf_acc = F256Unreduced::ZERO;

        let mut a_src = a_packed;
        let mut b_src = b_packed;
        let mut a_dst = a_out;
        let mut b_dst = b_out;
        let mut eq_ptr = eq_lo;
        let mut pair_idx = pair_idx_base;
        let mut remaining = lo_size;

        while remaining != 0 {
            if (pair_idx & pair_in_block_mask) >= useful_pairs_inclusive {
                a_dst.write(F128::ZERO);
                a_dst.add(1).write(F128::ZERO);
                b_dst.write(F128::ZERO);
                b_dst.add(1).write(F128::ZERO);
            } else {
                let a0 = fold_one_row_neon_unchecked_8(table_data, a_src);
                let b0 = fold_one_row_neon_unchecked_8(table_data, b_src);
                let a1 = fold_one_row_neon_unchecked_8(table_data, a_src.add(8));
                let b1 = fold_one_row_neon_unchecked_8(table_data, b_src.add(8));

                a_dst.write(a0);
                a_dst.add(1).write(a1);
                b_dst.write(b0);
                b_dst.add(1).write(b1);

                let eq_l = eq_ptr.read();
                let g1 = a1 * b1;
                p1_acc ^= eq_l.mul_unreduced(g1);
                let g_inf = (a0 + a1) * (b0 + b1);
                pinf_acc ^= eq_l.mul_unreduced(g_inf);
            }

            a_src = a_src.add(16);
            b_src = b_src.add(16);
            a_dst = a_dst.add(2);
            b_dst = b_dst.add(2);
            eq_ptr = eq_ptr.add(1);
            pair_idx += 1;
            remaining -= 1;
        }

        (p1_acc.reduce(), pinf_acc.reduce())
    }
}

/// NEON one-row fold: 8 aligned 16-byte loads + 8 XORs, hand-unrolled for
/// `n_chunks = 8` (the k_skip=6 protocol size). Returns the folded F128.
///
/// The table is `Vec<F128>` with each entry 16-byte aligned (F128 is
/// `repr(C, align(16))`), so every `vld1q_u8` lands on an aligned address.
///
/// # Safety
/// Caller must guarantee `table_data` points to ≥ 8 × 256 × 16 valid bytes
/// (an `n_chunks ≥ 8` table) and `bytes_ptr` to ≥ 8 valid bytes.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) unsafe fn fold_one_row_neon_unchecked_8(
    table_data: *const u8,
    bytes_ptr: *const u8,
) -> F128 {
    use core::arch::aarch64::*;
    unsafe {
        const STRIDE: usize = 256 * 16;
        let mut acc = vld1q_u8(table_data.add((*bytes_ptr) as usize * 16));
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(1 * STRIDE + (*bytes_ptr.add(1)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(2 * STRIDE + (*bytes_ptr.add(2)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(3 * STRIDE + (*bytes_ptr.add(3)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(4 * STRIDE + (*bytes_ptr.add(4)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(5 * STRIDE + (*bytes_ptr.add(5)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(6 * STRIDE + (*bytes_ptr.add(6)) as usize * 16)),
        );
        acc = veorq_u8(
            acc,
            vld1q_u8(table_data.add(7 * STRIDE + (*bytes_ptr.add(7)) as usize * 16)),
        );
        let acc_u64 = vreinterpretq_u64_u8(acc);
        F128 {
            lo: vgetq_lane_u64::<0>(acc_u64),
            hi: vgetq_lane_u64::<1>(acc_u64),
        }
    }
}

#[cfg(all(test, target_os = "macos"))]
mod sme2_tests {
    use super::*;

    #[derive(Clone, Copy)]
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        fn f128(&mut self) -> F128 {
            F128::new(self.next(), self.next())
        }
    }

    #[test]
    fn pmull32_tail_matches_untouched_neon_at_required_sizes() {
        if !sme2_pmull32_tail_available() {
            eprintln!("SME2 tail differential skipped: runtime capability/SVL unavailable");
            return;
        }

        let mut rng = Rng(0x534d_4532_5441_494c);
        for lo_size in [8_usize, 64, 128, 256, 1_024, 65_536, 131_072] {
            let a_in: Vec<_> = (0..4 * lo_size).map(|_| rng.f128()).collect();
            let b_in: Vec<_> = (0..4 * lo_size).map(|_| rng.f128()).collect();
            let eq_lo: Vec<_> = (0..lo_size).map(|_| rng.f128()).collect();
            let rho = rng.f128();
            let mut a_neon = vec![F128::ZERO; 2 * lo_size];
            let mut b_neon = vec![F128::ZERO; 2 * lo_size];
            let mut a_sme = vec![F128::ZERO; 2 * lo_size];
            let mut b_sme = vec![F128::ZERO; 2 * lo_size];

            // SAFETY: all arrays have the documented geometry and this test
            // has checked the SME2/SVL runtime gate.
            let neon = unsafe {
                fold_and_message_neon_impl(&a_in, &b_in, &mut a_neon, &mut b_neon, rho, &eq_lo)
            };
            let sme =
                unsafe { fold_and_message_sme2(&a_in, &b_in, &mut a_sme, &mut b_sme, rho, &eq_lo) };

            assert_eq!(a_sme, a_neon, "A mismatch at lo_size={lo_size}");
            assert_eq!(b_sme, b_neon, "B mismatch at lo_size={lo_size}");
            assert_eq!(sme, neon, "message mismatch at lo_size={lo_size}");
        }
    }

    #[test]
    fn pmull32_dispatch_threshold_matches_neon() {
        let mut rng = Rng(0x5448_5245_5348_4f4c);
        for lo_size in [64_usize, 128] {
            let a_in: Vec<_> = (0..4 * lo_size).map(|_| rng.f128()).collect();
            let b_in: Vec<_> = (0..4 * lo_size).map(|_| rng.f128()).collect();
            let eq_lo: Vec<_> = (0..lo_size).map(|_| rng.f128()).collect();
            let rho = rng.f128();
            let mut a_expected = vec![F128::ZERO; 2 * lo_size];
            let mut b_expected = vec![F128::ZERO; 2 * lo_size];
            let mut a_actual = vec![F128::ZERO; 2 * lo_size];
            let mut b_actual = vec![F128::ZERO; 2 * lo_size];

            // SAFETY: all arrays have the documented geometry.
            let expected = unsafe {
                fold_and_message_neon_impl(
                    &a_in,
                    &b_in,
                    &mut a_expected,
                    &mut b_expected,
                    rho,
                    &eq_lo,
                )
            };
            let actual = unsafe {
                fold_and_message_neon(&a_in, &b_in, &mut a_actual, &mut b_actual, rho, &eq_lo)
            };
            assert_eq!(a_actual, a_expected, "A mismatch at threshold {lo_size}");
            assert_eq!(b_actual, b_expected, "B mismatch at threshold {lo_size}");
            assert_eq!(actual, expected, "message mismatch at threshold {lo_size}");
        }
    }

    #[test]
    #[ignore = "diagnostic timing; run explicitly with --release --ignored --nocapture"]
    fn pmull32_tail_diagnostic_timing() {
        use std::time::Instant;

        if !sme2_pmull32_tail_available() {
            eprintln!("SME2 tail timing skipped: runtime capability/SVL unavailable");
            return;
        }

        let mut rng = Rng(0x5449_4d45_504d_3332);
        for lo_size in [128_usize, 1_024, 131_072] {
            let a_in: Vec<_> = (0..4 * lo_size).map(|_| rng.f128()).collect();
            let b_in: Vec<_> = (0..4 * lo_size).map(|_| rng.f128()).collect();
            let eq_lo: Vec<_> = (0..lo_size).map(|_| rng.f128()).collect();
            let rho = rng.f128();
            let mut a_neon = vec![F128::ZERO; 2 * lo_size];
            let mut b_neon = vec![F128::ZERO; 2 * lo_size];
            let mut a_sme = vec![F128::ZERO; 2 * lo_size];
            let mut b_sme = vec![F128::ZERO; 2 * lo_size];
            let mut candidate_wins = 0_usize;

            for pair in 0..12 {
                let candidate_first = pair % 2 == 1;
                let (neon_elapsed, neon_message, sme_elapsed, sme_message) = if candidate_first {
                    let start = Instant::now();
                    // SAFETY: geometry and SME2/SVL were checked above.
                    let sme = unsafe {
                        fold_and_message_sme2(&a_in, &b_in, &mut a_sme, &mut b_sme, rho, &eq_lo)
                    };
                    let sme_elapsed = start.elapsed();
                    let start = Instant::now();
                    // SAFETY: geometry is exact and this test is AArch64/AES.
                    let neon = unsafe {
                        fold_and_message_neon_impl(
                            &a_in,
                            &b_in,
                            &mut a_neon,
                            &mut b_neon,
                            rho,
                            &eq_lo,
                        )
                    };
                    (start.elapsed(), neon, sme_elapsed, sme)
                } else {
                    let start = Instant::now();
                    // SAFETY: geometry is exact and this test is AArch64/AES.
                    let neon = unsafe {
                        fold_and_message_neon_impl(
                            &a_in,
                            &b_in,
                            &mut a_neon,
                            &mut b_neon,
                            rho,
                            &eq_lo,
                        )
                    };
                    let neon_elapsed = start.elapsed();
                    let start = Instant::now();
                    // SAFETY: geometry and SME2/SVL were checked above.
                    let sme = unsafe {
                        fold_and_message_sme2(&a_in, &b_in, &mut a_sme, &mut b_sme, rho, &eq_lo)
                    };
                    (neon_elapsed, neon, start.elapsed(), sme)
                };

                // Comparison is deliberately outside both timing intervals.
                assert_eq!(
                    a_sme, a_neon,
                    "A mismatch at lo_size={lo_size}, pair={pair}"
                );
                assert_eq!(
                    b_sme, b_neon,
                    "B mismatch at lo_size={lo_size}, pair={pair}"
                );
                assert_eq!(
                    sme_message, neon_message,
                    "message mismatch at lo_size={lo_size}, pair={pair}"
                );
                candidate_wins += usize::from(sme_elapsed < neon_elapsed);
                eprintln!(
                    "PMULL32_TAIL_TIMING lo_size={lo_size} pair={pair} order={} neon_ns={} sme_ns={} winner={}",
                    if candidate_first {
                        "sme-neon"
                    } else {
                        "neon-sme"
                    },
                    neon_elapsed.as_nanos(),
                    sme_elapsed.as_nanos(),
                    if sme_elapsed < neon_elapsed {
                        "sme"
                    } else {
                        "neon"
                    },
                );
            }
            eprintln!(
                "PMULL32_TAIL_TIMING_SUMMARY lo_size={lo_size} candidate_wins={candidate_wins}/12"
            );
        }
    }
}
