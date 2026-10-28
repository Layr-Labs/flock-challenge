use crate::field::F128;

#[inline]
pub(super) fn butterfly_row_pair(top: &mut [F128], bot: &mut [F128], twiddle: F128) {
    let n = top.len();
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    unsafe {
        let pair_n = n & !1;
        if pair_n > 0 {
            crate::gf2_kernels::k0_ntt_butterfly_row_pair_slice(
                top.as_mut_ptr() as *mut u64,
                bot.as_mut_ptr() as *mut u64,
                pair_n,
                twiddle.lo,
                twiddle.hi,
            );
        }
        if pair_n < n {
            let u = top[pair_n];
            let v = bot[pair_n];
            let prod = v * twiddle;
            let u_lo = crate::gf2_kernels::gf2_kern_ntt_btf_u_lo(u.lo, prod.lo);
            let u_hi = crate::gf2_kernels::gf2_kern_ntt_btf_u_lo(u.hi, prod.hi);
            let v_lo = crate::gf2_kernels::gf2_kern_ntt_btf_v_lo(u.lo, v.lo, prod.lo);
            let v_hi = crate::gf2_kernels::gf2_kern_ntt_btf_v_lo(u.hi, v.hi, prod.hi);
            top[pair_n] = F128 { lo: u_lo, hi: u_hi };
            bot[pair_n] = F128 { lo: v_lo, hi: v_hi };
        }
        return;
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    for lane in 0..n {
        let u = top[lane];
        let v = bot[lane];
        let prod = v * twiddle;
        let new_u = u + prod;
        top[lane] = new_u;
        bot[lane] = v + new_u;
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn butterfly_fused_2layer(
    a: &mut [F128],
    b: &mut [F128],
    c: &mut [F128],
    d: &mut [F128],
    t_outer: F128,
    t_inner_a: F128,
    t_inner_b: F128,
) {
    let n = a.len();
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    unsafe {
        let pair_n = n & !1;
        if pair_n > 0 {
            crate::gf2_kernels::k0_ntt_butterfly_fused_2layer_slice(
                a.as_mut_ptr() as *mut u64,
                b.as_mut_ptr() as *mut u64,
                c.as_mut_ptr() as *mut u64,
                d.as_mut_ptr() as *mut u64,
                pair_n,
                t_outer.lo,
                t_outer.hi,
                t_inner_a.lo,
                t_inner_a.hi,
                t_inner_b.lo,
                t_inner_b.hi,
            );
        }
        if pair_n < n {
            let xa = a[pair_n];
            let xb = b[pair_n];
            let xc = c[pair_n];
            let xd = d[pair_n];
            let p_out_a = xc * t_outer;
            let p_out_b = xd * t_outer;
            let b1 = xb + p_out_b;
            let d1 = xd + b1;
            let p_in_a = b1 * t_inner_a;
            let p_in_b = d1 * t_inner_b;
            a[pair_n] = F128 {
                lo: crate::gf2_kernels::gf2_kern_ntt_radix4_comb_a(xa.lo, p_out_a.lo, p_in_a.lo),
                hi: crate::gf2_kernels::gf2_kern_ntt_radix4_comb_a(xa.hi, p_out_a.hi, p_in_a.hi),
            };
            b[pair_n] = F128 {
                lo: crate::gf2_kernels::gf2_kern_ntt_radix4_comb_b(
                    xa.lo, xb.lo, p_out_a.lo, p_out_b.lo, p_in_a.lo,
                ),
                hi: crate::gf2_kernels::gf2_kern_ntt_radix4_comb_b(
                    xa.hi, xb.hi, p_out_a.hi, p_out_b.hi, p_in_a.hi,
                ),
            };
            c[pair_n] = F128 {
                lo: crate::gf2_kernels::gf2_kern_ntt_radix4_comb_c(
                    xa.lo, xc.lo, p_out_a.lo, p_in_b.lo,
                ),
                hi: crate::gf2_kernels::gf2_kern_ntt_radix4_comb_c(
                    xa.hi, xc.hi, p_out_a.hi, p_in_b.hi,
                ),
            };
            d[pair_n] = F128 {
                lo: crate::gf2_kernels::gf2_kern_ntt_radix4_comb_d(
                    xa.lo, xb.lo, xc.lo, xd.lo, p_out_a.lo, p_out_b.lo, p_in_b.lo,
                ),
                hi: crate::gf2_kernels::gf2_kern_ntt_radix4_comb_d(
                    xa.hi, xb.hi, xc.hi, xd.hi, p_out_a.hi, p_out_b.hi, p_in_b.hi,
                ),
            };
        }
        return;
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    for lane in 0..n {
        let mut xa = a[lane];
        let mut xb = b[lane];
        let mut xc = c[lane];
        let mut xd = d[lane];
        let na = xa + xc * t_outer;
        xc += na;
        xa = na;
        let nb = xb + xd * t_outer;
        xd += nb;
        xb = nb;
        let na2 = xa + xb * t_inner_a;
        xb += na2;
        xa = na2;
        let nc2 = xc + xd * t_inner_b;
        xd += nc2;
        xc = nc2;
        a[lane] = xa;
        b[lane] = xb;
        c[lane] = xc;
        d[lane] = xd;
    }
}

#[allow(dead_code)]
#[inline]
pub(super) fn butterfly_fused_4layer(values: &mut [F128; 16], twiddles: &[F128; 15]) {
    #[inline(always)]
    fn butterfly(values: &mut [F128; 16], u: usize, v: usize, twiddle: F128) {
        let new_u = values[u] + values[v] * twiddle;
        values[v] += new_u;
        values[u] = new_u;
    }

    for i in 0..8 {
        butterfly(values, i, i + 8, twiddles[0]);
    }
    for s in 0..2 {
        for i in 0..4 {
            butterfly(values, 8 * s + i, 8 * s + i + 4, twiddles[1 + s]);
        }
    }
    for s in 0..4 {
        for i in 0..2 {
            butterfly(values, 4 * s + i, 4 * s + i + 2, twiddles[3 + s]);
        }
    }
    for s in 0..8 {
        butterfly(values, 2 * s, 2 * s + 1, twiddles[7 + s]);
    }
}

#[allow(dead_code)]
#[inline]
pub(super) fn butterfly_fused_3layer(values: &mut [F128; 8], twiddles: &[F128; 7]) {
    #[inline(always)]
    fn butterfly(values: &mut [F128; 8], u: usize, v: usize, twiddle: F128) {
        let new_u = values[u] + values[v] * twiddle;
        values[v] += new_u;
        values[u] = new_u;
    }

    for i in 0..4 {
        butterfly(values, i, i + 4, twiddles[0]);
    }
    for s in 0..2 {
        for i in 0..2 {
            butterfly(values, 4 * s + i, 4 * s + i + 2, twiddles[1 + s]);
        }
    }
    for s in 0..4 {
        butterfly(values, 2 * s, 2 * s + 1, twiddles[3 + s]);
    }
}

#[inline]
pub(super) unsafe fn butterfly_fused_3layer_row(
    ptr: *mut F128,
    eighth: usize,
    num_ntts: usize,
    r: usize,
    twiddles: &[F128; 7],
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    unsafe {
        use core::arch::x86_64::*;
        let tw_ptr = twiddles.as_ptr() as *const u64;
        let u64_ptr = ptr as *mut u64;
        let stride = 2 * eighth * num_ntts;
        let base_r = u64_ptr.add(2 * r * num_ntts);
        if (twiddles[0].lo
            | twiddles[0].hi
            | twiddles[1].lo
            | twiddles[1].hi
            | twiddles[3].lo
            | twiddles[3].hi)
            == 0
        {
            let mut tw_r0 = [_mm_setzero_si128(); 7];
            let mut tw_r1 = [_mm_setzero_si128(); 7];
            for i in [2usize, 4, 5, 6] {
                let t = _mm_loadu_si128(tw_ptr.add(2 * i) as *const __m128i);
                let (r0, r1) = crate::gf2_kernels::k0_f128_pre_reduce_const_m128i(t);
                tw_r0[i] = r0;
                tw_r1[i] = r1;
            }
            for lane in 0..num_ntts {
                let p_lane = base_r.add(2 * lane);
                let v0 = _mm_loadu_si128(p_lane as *const __m128i);
                let mut v1 = _mm_loadu_si128(p_lane.add(stride) as *const __m128i);
                let mut v2 = _mm_loadu_si128(p_lane.add(2 * stride) as *const __m128i);
                let mut v3 = _mm_loadu_si128(p_lane.add(3 * stride) as *const __m128i);
                let mut v4 = _mm_loadu_si128(p_lane.add(4 * stride) as *const __m128i);
                let mut v5 = _mm_loadu_si128(p_lane.add(5 * stride) as *const __m128i);
                let mut v6 = _mm_loadu_si128(p_lane.add(6 * stride) as *const __m128i);
                let mut v7 = _mm_loadu_si128(p_lane.add(7 * stride) as *const __m128i);
                v4 = _mm_xor_si128(v4, v0);
                v5 = _mm_xor_si128(v5, v1);
                v6 = _mm_xor_si128(v6, v2);
                v7 = _mm_xor_si128(v7, v3);
                v2 = _mm_xor_si128(v2, v0);
                v3 = _mm_xor_si128(v3, v1);
                let (p0, p1) = crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                    v6, v7, tw_r0[2], tw_r1[2], tw_r0[2], tw_r1[2],
                );
                let nu4 = _mm_xor_si128(v4, p0);
                v6 = _mm_xor_si128(v6, nu4);
                v4 = nu4;
                let nu5 = _mm_xor_si128(v5, p1);
                v7 = _mm_xor_si128(v7, nu5);
                v5 = nu5;
                v1 = _mm_xor_si128(v1, v0);
                let p3 = crate::gf2_kernels::k0_f128_mul_const_m128i(v3, tw_r0[4], tw_r1[4]);
                let nu2 = _mm_xor_si128(v2, p3);
                v3 = _mm_xor_si128(v3, nu2);
                v2 = nu2;
                let (p5, p7) = crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                    v5, v7, tw_r0[5], tw_r1[5], tw_r0[6], tw_r1[6],
                );
                let nu4b = _mm_xor_si128(v4, p5);
                v5 = _mm_xor_si128(v5, nu4b);
                v4 = nu4b;
                let nu6b = _mm_xor_si128(v6, p7);
                v7 = _mm_xor_si128(v7, nu6b);
                v6 = nu6b;
                _mm_storeu_si128(p_lane as *mut __m128i, v0);
                _mm_storeu_si128(p_lane.add(stride) as *mut __m128i, v1);
                _mm_storeu_si128(p_lane.add(2 * stride) as *mut __m128i, v2);
                _mm_storeu_si128(p_lane.add(3 * stride) as *mut __m128i, v3);
                _mm_storeu_si128(p_lane.add(4 * stride) as *mut __m128i, v4);
                _mm_storeu_si128(p_lane.add(5 * stride) as *mut __m128i, v5);
                _mm_storeu_si128(p_lane.add(6 * stride) as *mut __m128i, v6);
                _mm_storeu_si128(p_lane.add(7 * stride) as *mut __m128i, v7);
            }
            return;
        }
        let mut tw_r0 = [_mm_setzero_si128(); 7];
        let mut tw_r1 = [_mm_setzero_si128(); 7];
        for i in 0..7 {
            let t = _mm_loadu_si128(tw_ptr.add(2 * i) as *const __m128i);
            let (r0, r1) = crate::gf2_kernels::k0_f128_pre_reduce_const_m128i(t);
            tw_r0[i] = r0;
            tw_r1[i] = r1;
        }
        let poly = _mm_set1_epi64x(0x87);
        macro_rules! btf2_3layer_x86 {
            ($u0:ident, $v0:ident, $tr0a:expr, $tr1a:expr, $u1:ident, $v1:ident, $tr0b:expr, $tr1b:expr) => {{
                let t0a = $tr0a;
                let t0b = $tr0b;
                let l0 = _mm_clmulepi64_si128::<0x00>($v0, t0a);
                let l1 = _mm_clmulepi64_si128::<0x00>($v1, t0b);
                let l0 = _mm_xor_si128(l0, _mm_clmulepi64_si128::<0x11>($v0, t0a));
                let l1 = _mm_xor_si128(l1, _mm_clmulepi64_si128::<0x11>($v1, t0b));
                $u0 = _mm_xor_si128($u0, l0);
                $u1 = _mm_xor_si128($u1, l1);
                let t1a = $tr1a;
                let t1b = $tr1b;
                let h0 = _mm_clmulepi64_si128::<0x00>($v0, t1a);
                let h1 = _mm_clmulepi64_si128::<0x00>($v1, t1b);
                let h0 = _mm_xor_si128(h0, _mm_clmulepi64_si128::<0x11>($v0, t1a));
                let h1 = _mm_xor_si128(h1, _mm_clmulepi64_si128::<0x11>($v1, t1b));
                let red0 = _mm_clmulepi64_si128::<0x01>(h0, poly);
                let red1 = _mm_clmulepi64_si128::<0x01>(h1, poly);
                $u0 = _mm_xor_si128($u0, _mm_xor_si128(_mm_bslli_si128::<8>(h0), red0));
                $v0 = _mm_xor_si128($v0, $u0);
                $u1 = _mm_xor_si128($u1, _mm_xor_si128(_mm_bslli_si128::<8>(h1), red1));
                $v1 = _mm_xor_si128($v1, $u1);
            }};
        }
        for lane in 0..num_ntts {
            let p_lane = base_r.add(2 * lane);
            let mut v0 = _mm_loadu_si128(p_lane as *const __m128i);
            let mut v1 = _mm_loadu_si128(p_lane.add(stride) as *const __m128i);
            let mut v2 = _mm_loadu_si128(p_lane.add(2 * stride) as *const __m128i);
            let mut v3 = _mm_loadu_si128(p_lane.add(3 * stride) as *const __m128i);
            let mut v4 = _mm_loadu_si128(p_lane.add(4 * stride) as *const __m128i);
            let mut v5 = _mm_loadu_si128(p_lane.add(5 * stride) as *const __m128i);
            let mut v6 = _mm_loadu_si128(p_lane.add(6 * stride) as *const __m128i);
            let mut v7 = _mm_loadu_si128(p_lane.add(7 * stride) as *const __m128i);

            btf2_3layer_x86!(v0, v4, tw_r0[0], tw_r1[0], v1, v5, tw_r0[0], tw_r1[0]);
            btf2_3layer_x86!(v2, v6, tw_r0[0], tw_r1[0], v3, v7, tw_r0[0], tw_r1[0]);
            btf2_3layer_x86!(v0, v2, tw_r0[1], tw_r1[1], v1, v3, tw_r0[1], tw_r1[1]);
            btf2_3layer_x86!(v4, v6, tw_r0[2], tw_r1[2], v5, v7, tw_r0[2], tw_r1[2]);
            btf2_3layer_x86!(v0, v1, tw_r0[3], tw_r1[3], v2, v3, tw_r0[4], tw_r1[4]);
            btf2_3layer_x86!(v4, v5, tw_r0[5], tw_r1[5], v6, v7, tw_r0[6], tw_r1[6]);

            _mm_storeu_si128(p_lane as *mut __m128i, v0);
            _mm_storeu_si128(p_lane.add(stride) as *mut __m128i, v1);
            _mm_storeu_si128(p_lane.add(2 * stride) as *mut __m128i, v2);
            _mm_storeu_si128(p_lane.add(3 * stride) as *mut __m128i, v3);
            _mm_storeu_si128(p_lane.add(4 * stride) as *mut __m128i, v4);
            _mm_storeu_si128(p_lane.add(5 * stride) as *mut __m128i, v5);
            _mm_storeu_si128(p_lane.add(6 * stride) as *mut __m128i, v6);
            _mm_storeu_si128(p_lane.add(7 * stride) as *mut __m128i, v7);
        }
        return;
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        use core::arch::aarch64::*;
        let z = vdupq_n_u64(0);
        let tw_ptr = twiddles.as_ptr() as *const u64;
        if (twiddles[0].lo
            | twiddles[0].hi
            | twiddles[1].lo
            | twiddles[1].hi
            | twiddles[3].lo
            | twiddles[3].hi)
            == 0
        {
            let mut tw_r0 = [z; 7];
            let mut tw_r1 = [z; 7];
            for i in [2usize, 4, 5, 6] {
                let t = vld1q_u64(tw_ptr.add(2 * i));
                let (r0, r1) = crate::gf2_kernels::k0_f128_pre_reduce_const_neon(t);
                tw_r0[i] = r0;
                tw_r1[i] = r1;
            }
            let u64_ptr = ptr as *mut u64;
            for lane in 0..num_ntts {
                let mut v = [z; 8];
                for i in 0..8 {
                    let off = 2 * ((i * eighth + r) * num_ntts + lane);
                    v[i] = vld1q_u64(u64_ptr.add(off));
                }
                v[4] = veorq_u64(v[4], v[0]);
                v[5] = veorq_u64(v[5], v[1]);
                v[6] = veorq_u64(v[6], v[2]);
                v[7] = veorq_u64(v[7], v[3]);
                v[2] = veorq_u64(v[2], v[0]);
                v[3] = veorq_u64(v[3], v[1]);
                let (p0, p1) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                    v[6], v[7], tw_r0[2], tw_r1[2], tw_r0[2], tw_r1[2],
                );
                let nu4 = veorq_u64(v[4], p0);
                v[6] = veorq_u64(v[6], nu4);
                v[4] = nu4;
                let nu5 = veorq_u64(v[5], p1);
                v[7] = veorq_u64(v[7], nu5);
                v[5] = nu5;
                v[1] = veorq_u64(v[1], v[0]);
                let p3 = crate::gf2_kernels::k0_f128_mul_const_neon(v[3], tw_r0[4], tw_r1[4]);
                let nu2 = veorq_u64(v[2], p3);
                v[3] = veorq_u64(v[3], nu2);
                v[2] = nu2;
                let (p5, p7) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                    v[5], v[7], tw_r0[5], tw_r1[5], tw_r0[6], tw_r1[6],
                );
                let nu4b = veorq_u64(v[4], p5);
                v[5] = veorq_u64(v[5], nu4b);
                v[4] = nu4b;
                let nu6b = veorq_u64(v[6], p7);
                v[7] = veorq_u64(v[7], nu6b);
                v[6] = nu6b;
                for i in 0..8 {
                    let off = 2 * ((i * eighth + r) * num_ntts + lane);
                    vst1q_u64(u64_ptr.add(off), v[i]);
                }
            }
            return;
        }
        let mut tw_r0 = [z; 7];
        let mut tw_r1 = [z; 7];
        for i in 0..7 {
            let t = vld1q_u64(tw_ptr.add(2 * i));
            let (r0, r1) = crate::gf2_kernels::k0_f128_pre_reduce_const_neon(t);
            tw_r0[i] = r0;
            tw_r1[i] = r1;
        }
        let u64_ptr = ptr as *mut u64;
        let stride = 2 * eighth * num_ntts;
        let base_r = u64_ptr.add(2 * r * num_ntts);
        macro_rules! btf2_neon {
            ($u0:ident, $v0:ident, $tr0a:expr, $tr1a:expr, $u1:ident, $v1:ident, $tr0b:expr, $tr1b:expr) => {{
                let (p0, p1) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                    $v0, $v1, $tr0a, $tr1a, $tr0b, $tr1b,
                );
                $u0 = veorq_u64($u0, p0);
                $v0 = veorq_u64($v0, $u0);
                $u1 = veorq_u64($u1, p1);
                $v1 = veorq_u64($v1, $u1);
            }};
        }
        for lane in 0..num_ntts {
            let p_lane = base_r.add(2 * lane);
            let mut v0 = vld1q_u64(p_lane);
            let mut v1 = vld1q_u64(p_lane.add(stride));
            let mut v2 = vld1q_u64(p_lane.add(2 * stride));
            let mut v3 = vld1q_u64(p_lane.add(3 * stride));
            let mut v4 = vld1q_u64(p_lane.add(4 * stride));
            let mut v5 = vld1q_u64(p_lane.add(5 * stride));
            let mut v6 = vld1q_u64(p_lane.add(6 * stride));
            let mut v7 = vld1q_u64(p_lane.add(7 * stride));

            btf2_neon!(v0, v4, tw_r0[0], tw_r1[0], v1, v5, tw_r0[0], tw_r1[0]);
            btf2_neon!(v2, v6, tw_r0[0], tw_r1[0], v3, v7, tw_r0[0], tw_r1[0]);
            btf2_neon!(v0, v2, tw_r0[1], tw_r1[1], v1, v3, tw_r0[1], tw_r1[1]);
            btf2_neon!(v4, v6, tw_r0[2], tw_r1[2], v5, v7, tw_r0[2], tw_r1[2]);
            btf2_neon!(v0, v1, tw_r0[3], tw_r1[3], v2, v3, tw_r0[4], tw_r1[4]);
            btf2_neon!(v4, v5, tw_r0[5], tw_r1[5], v6, v7, tw_r0[6], tw_r1[6]);

            vst1q_u64(p_lane, v0);
            vst1q_u64(p_lane.add(stride), v1);
            vst1q_u64(p_lane.add(2 * stride), v2);
            vst1q_u64(p_lane.add(3 * stride), v3);
            vst1q_u64(p_lane.add(4 * stride), v4);
            vst1q_u64(p_lane.add(5 * stride), v5);
            vst1q_u64(p_lane.add(6 * stride), v6);
            vst1q_u64(p_lane.add(7 * stride), v7);
        }
        return;
    }
    #[cfg(all(
        any(target_arch = "x86_64", target_arch = "aarch64"),
        not(all(target_arch = "x86_64", target_feature = "pclmulqdq"))
    ))]
    unsafe {
        crate::gf2_kernels::k0_ntt_butterfly_fused_3layer_row_slice(
            ptr as *mut u64,
            eighth,
            num_ntts,
            r,
            twiddles.as_ptr() as *const u64,
        );
        return;
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    unsafe {
        for lane in 0..num_ntts {
            let mut values = [F128::ZERO; 8];
            for (i, value) in values.iter_mut().enumerate() {
                *value = *ptr.add((i * eighth + r) * num_ntts + lane);
            }
            butterfly_fused_3layer(&mut values, twiddles);
            for (i, value) in values.iter().enumerate() {
                *ptr.add((i * eighth + r) * num_ntts + lane) = *value;
            }
        }
    }
}

#[allow(dead_code)]
#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
)))]
pub(super) unsafe fn butterfly_fused_4layer_row(
    ptr: *mut F128,
    sixteenth: usize,
    num_ntts: usize,
    r: usize,
    twiddles: &[F128; 15],
) {
    unsafe {
        butterfly_fused_4layer_rows(ptr, sixteenth, num_ntts, r, 1, twiddles);
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
#[target_feature(enable = "pclmulqdq,sse2")]
#[inline]
unsafe fn butterfly_fused_4layer_rows_x86_staged(
    ptr: *mut F128,
    sixteenth: usize,
    num_ntts: usize,
    r_start: usize,
    r_count: usize,
    twiddles: &[F128; 15],
) {
    use core::arch::x86_64::*;
    unsafe {
        let tw_ptr = twiddles.as_ptr() as *const u64;
        let mut tw_r0 = [_mm_setzero_si128(); 15];
        let mut tw_r1 = [_mm_setzero_si128(); 15];
        for i in 0..15 {
            let t = _mm_loadu_si128(tw_ptr.add(2 * i) as *const __m128i);
            let (r0, r1) = crate::gf2_kernels::k0_f128_pre_reduce_const_m128i(t);
            tw_r0[i] = r0;
            tw_r1[i] = r1;
        }
        let zero_tw = (twiddles[0].lo
            | twiddles[0].hi
            | twiddles[1].lo
            | twiddles[1].hi
            | twiddles[3].lo
            | twiddles[3].hi
            | twiddles[7].lo
            | twiddles[7].hi)
            == 0;

        #[repr(align(64))]
        struct Stage {
            _buf: [__m128i; 16 * 16],
        }
        let mut stage = core::mem::MaybeUninit::<Stage>::uninit();
        let st_ptr = stage.as_mut_ptr() as *mut __m128i;
        let u64_ptr = ptr as *mut u64;

        let poly = _mm_set1_epi64x(0x87);
        macro_rules! btf_pair {
            ($u0:ident, $u1:ident, $v0:ident, $v1:ident, $tr0:expr, $tr1:expr) => {{
                let l0 = _mm_xor_si128(
                    _mm_clmulepi64_si128::<0x00>($v0, $tr0),
                    _mm_clmulepi64_si128::<0x11>($v0, $tr0),
                );
                let l1 = _mm_xor_si128(
                    _mm_clmulepi64_si128::<0x00>($v1, $tr0),
                    _mm_clmulepi64_si128::<0x11>($v1, $tr0),
                );
                $u0 = _mm_xor_si128($u0, l0);
                $u1 = _mm_xor_si128($u1, l1);
                let h0 = _mm_xor_si128(
                    _mm_clmulepi64_si128::<0x00>($v0, $tr1),
                    _mm_clmulepi64_si128::<0x11>($v0, $tr1),
                );
                let h1 = _mm_xor_si128(
                    _mm_clmulepi64_si128::<0x00>($v1, $tr1),
                    _mm_clmulepi64_si128::<0x11>($v1, $tr1),
                );
                let r0 = _mm_xor_si128(
                    _mm_bslli_si128::<8>(h0),
                    _mm_clmulepi64_si128::<0x01>(h0, poly),
                );
                let r1 = _mm_xor_si128(
                    _mm_bslli_si128::<8>(h1),
                    _mm_clmulepi64_si128::<0x01>(h1, poly),
                );
                $u0 = _mm_xor_si128($u0, r0);
                $u1 = _mm_xor_si128($u1, r1);
                $v0 = _mm_xor_si128($v0, $u0);
                $v1 = _mm_xor_si128($v1, $u1);
            }};
        }
        macro_rules! radix4_1lane {
            ($a:ident, $b:ident, $c:ident, $d:ident, $to0:expr, $to1:expr, $ti00:expr, $ti01:expr, $ti10:expr, $ti11:expr) => {{
                let l0 = _mm_xor_si128(
                    _mm_clmulepi64_si128::<0x00>($c, $to0),
                    _mm_clmulepi64_si128::<0x11>($c, $to0),
                );
                let l1 = _mm_xor_si128(
                    _mm_clmulepi64_si128::<0x00>($d, $to0),
                    _mm_clmulepi64_si128::<0x11>($d, $to0),
                );
                $a = _mm_xor_si128($a, l0);
                $b = _mm_xor_si128($b, l1);
                let h0 = _mm_xor_si128(
                    _mm_clmulepi64_si128::<0x00>($c, $to1),
                    _mm_clmulepi64_si128::<0x11>($c, $to1),
                );
                let h1 = _mm_xor_si128(
                    _mm_clmulepi64_si128::<0x00>($d, $to1),
                    _mm_clmulepi64_si128::<0x11>($d, $to1),
                );
                let r0 = _mm_xor_si128(
                    _mm_bslli_si128::<8>(h0),
                    _mm_clmulepi64_si128::<0x01>(h0, poly),
                );
                let r1 = _mm_xor_si128(
                    _mm_bslli_si128::<8>(h1),
                    _mm_clmulepi64_si128::<0x01>(h1, poly),
                );
                $a = _mm_xor_si128($a, r0);
                $b = _mm_xor_si128($b, r1);
                $c = _mm_xor_si128($c, $a);
                $d = _mm_xor_si128($d, $b);

                let l0 = _mm_xor_si128(
                    _mm_clmulepi64_si128::<0x00>($b, $ti00),
                    _mm_clmulepi64_si128::<0x11>($b, $ti00),
                );
                let l1 = _mm_xor_si128(
                    _mm_clmulepi64_si128::<0x00>($d, $ti10),
                    _mm_clmulepi64_si128::<0x11>($d, $ti10),
                );
                $a = _mm_xor_si128($a, l0);
                $c = _mm_xor_si128($c, l1);
                let h0 = _mm_xor_si128(
                    _mm_clmulepi64_si128::<0x00>($b, $ti01),
                    _mm_clmulepi64_si128::<0x11>($b, $ti01),
                );
                let h1 = _mm_xor_si128(
                    _mm_clmulepi64_si128::<0x00>($d, $ti11),
                    _mm_clmulepi64_si128::<0x11>($d, $ti11),
                );
                let r0 = _mm_xor_si128(
                    _mm_bslli_si128::<8>(h0),
                    _mm_clmulepi64_si128::<0x01>(h0, poly),
                );
                let r1 = _mm_xor_si128(
                    _mm_bslli_si128::<8>(h1),
                    _mm_clmulepi64_si128::<0x01>(h1, poly),
                );
                $a = _mm_xor_si128($a, r0);
                $c = _mm_xor_si128($c, r1);
                $b = _mm_xor_si128($b, $a);
                $d = _mm_xor_si128($d, $c);
            }};
        }

        let t0_r0 = tw_r0[0];
        let t0_r1 = tw_r1[0];
        let t1_r0 = tw_r0[1];
        let t1_r1 = tw_r1[1];
        let t2_r0 = tw_r0[2];
        let t2_r1 = tw_r1[2];

        for r in r_start..(r_start + r_count) {
            let mut lane_base = 0usize;
            while lane_base < num_ntts {
                let tile = (num_ntts - lane_base).min(16);
                for q in 0..4 {
                    let row_a = (u64_ptr.add(2 * ((q * sixteenth + r) * num_ntts))
                        as *const __m128i)
                        .add(lane_base);
                    let row_b = (u64_ptr.add(2 * (((q + 4) * sixteenth + r) * num_ntts))
                        as *const __m128i)
                        .add(lane_base);
                    let row_c = (u64_ptr.add(2 * (((q + 8) * sixteenth + r) * num_ntts))
                        as *const __m128i)
                        .add(lane_base);
                    let row_d = (u64_ptr.add(2 * (((q + 12) * sixteenth + r) * num_ntts))
                        as *const __m128i)
                        .add(lane_base);
                    let dst_a = st_ptr.add(q * 16);
                    let dst_b = st_ptr.add((q + 4) * 16);
                    let dst_c = st_ptr.add((q + 8) * 16);
                    let dst_d = st_ptr.add((q + 12) * 16);
                    if zero_tw {
                        let mut lane = 0usize;
                        while lane < tile {
                            let a0 = _mm_loadu_si128(row_a.add(lane));
                            let a1 = _mm_loadu_si128(row_a.add(lane + 1));
                            let mut b0 = _mm_loadu_si128(row_b.add(lane));
                            let mut b1 = _mm_loadu_si128(row_b.add(lane + 1));
                            let mut c0 = _mm_loadu_si128(row_c.add(lane));
                            let mut c1 = _mm_loadu_si128(row_c.add(lane + 1));
                            let mut d0 = _mm_loadu_si128(row_d.add(lane));
                            let mut d1 = _mm_loadu_si128(row_d.add(lane + 1));
                            c0 = _mm_xor_si128(c0, a0);
                            c1 = _mm_xor_si128(c1, a1);
                            d0 = _mm_xor_si128(d0, b0);
                            d1 = _mm_xor_si128(d1, b1);
                            b0 = _mm_xor_si128(b0, a0);
                            b1 = _mm_xor_si128(b1, a1);
                            btf_pair!(c0, c1, d0, d1, t2_r0, t2_r1);
                            _mm_store_si128(dst_a.add(lane), a0);
                            _mm_store_si128(dst_a.add(lane + 1), a1);
                            _mm_store_si128(dst_b.add(lane), b0);
                            _mm_store_si128(dst_b.add(lane + 1), b1);
                            _mm_store_si128(dst_c.add(lane), c0);
                            _mm_store_si128(dst_c.add(lane + 1), c1);
                            _mm_store_si128(dst_d.add(lane), d0);
                            _mm_store_si128(dst_d.add(lane + 1), d1);
                            lane += 2;
                        }
                    } else {
                        for lane in 0..tile {
                            let mut a = _mm_loadu_si128(row_a.add(lane));
                            let mut b = _mm_loadu_si128(row_b.add(lane));
                            let mut c = _mm_loadu_si128(row_c.add(lane));
                            let mut d = _mm_loadu_si128(row_d.add(lane));
                            radix4_1lane!(a, b, c, d, t0_r0, t0_r1, t1_r0, t1_r1, t2_r0, t2_r1);
                            _mm_store_si128(dst_a.add(lane), a);
                            _mm_store_si128(dst_b.add(lane), b);
                            _mm_store_si128(dst_c.add(lane), c);
                            _mm_store_si128(dst_d.add(lane), d);
                        }
                    }
                }

                for s in 0..4 {
                    let to_r0 = tw_r0[3 + s];
                    let to_r1 = tw_r1[3 + s];
                    let ti0_r0 = tw_r0[7 + 2 * s];
                    let ti0_r1 = tw_r1[7 + 2 * s];
                    let ti1_r0 = tw_r0[8 + 2 * s];
                    let ti1_r1 = tw_r1[8 + 2 * s];
                    let src_a = st_ptr.add((4 * s) * 16);
                    let src_b = st_ptr.add((4 * s + 1) * 16);
                    let src_c = st_ptr.add((4 * s + 2) * 16);
                    let src_d = st_ptr.add((4 * s + 3) * 16);
                    let row_a = (u64_ptr.add(2 * (((4 * s) * sixteenth + r) * num_ntts))
                        as *mut __m128i)
                        .add(lane_base);
                    let row_b = (u64_ptr.add(2 * (((4 * s + 1) * sixteenth + r) * num_ntts))
                        as *mut __m128i)
                        .add(lane_base);
                    let row_c = (u64_ptr.add(2 * (((4 * s + 2) * sixteenth + r) * num_ntts))
                        as *mut __m128i)
                        .add(lane_base);
                    let row_d = (u64_ptr.add(2 * (((4 * s + 3) * sixteenth + r) * num_ntts))
                        as *mut __m128i)
                        .add(lane_base);
                    if zero_tw && s == 0 {
                        let mut lane = 0usize;
                        while lane < tile {
                            let a0 = _mm_load_si128(src_a.add(lane));
                            let a1 = _mm_load_si128(src_a.add(lane + 1));
                            let mut b0 = _mm_load_si128(src_b.add(lane));
                            let mut b1 = _mm_load_si128(src_b.add(lane + 1));
                            let mut c0 = _mm_load_si128(src_c.add(lane));
                            let mut c1 = _mm_load_si128(src_c.add(lane + 1));
                            let mut d0 = _mm_load_si128(src_d.add(lane));
                            let mut d1 = _mm_load_si128(src_d.add(lane + 1));
                            c0 = _mm_xor_si128(c0, a0);
                            c1 = _mm_xor_si128(c1, a1);
                            d0 = _mm_xor_si128(d0, b0);
                            d1 = _mm_xor_si128(d1, b1);
                            b0 = _mm_xor_si128(b0, a0);
                            b1 = _mm_xor_si128(b1, a1);
                            btf_pair!(c0, c1, d0, d1, ti1_r0, ti1_r1);
                            _mm_storeu_si128(row_a.add(lane), a0);
                            _mm_storeu_si128(row_a.add(lane + 1), a1);
                            _mm_storeu_si128(row_b.add(lane), b0);
                            _mm_storeu_si128(row_b.add(lane + 1), b1);
                            _mm_storeu_si128(row_c.add(lane), c0);
                            _mm_storeu_si128(row_c.add(lane + 1), c1);
                            _mm_storeu_si128(row_d.add(lane), d0);
                            _mm_storeu_si128(row_d.add(lane + 1), d1);
                            lane += 2;
                        }
                    } else {
                        for lane in 0..tile {
                            let mut a = _mm_load_si128(src_a.add(lane));
                            let mut b = _mm_load_si128(src_b.add(lane));
                            let mut c = _mm_load_si128(src_c.add(lane));
                            let mut d = _mm_load_si128(src_d.add(lane));
                            radix4_1lane!(a, b, c, d, to_r0, to_r1, ti0_r0, ti0_r1, ti1_r0, ti1_r1);
                            _mm_storeu_si128(row_a.add(lane), a);
                            _mm_storeu_si128(row_b.add(lane), b);
                            _mm_storeu_si128(row_c.add(lane), c);
                            _mm_storeu_si128(row_d.add(lane), d);
                        }
                    }
                }
                lane_base += tile;
            }
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[target_feature(enable = "neon,aes")]
#[inline]
unsafe fn butterfly_fused_4layer_rows_neon_staged(
    ptr: *mut F128,
    sixteenth: usize,
    num_ntts: usize,
    r_start: usize,
    r_count: usize,
    twiddles: &[F128; 15],
) {
    use core::arch::aarch64::*;
    unsafe {
        let z = vdupq_n_u64(0);
        let tw_ptr = twiddles.as_ptr() as *const u64;
        let mut tw_r0 = [z; 15];
        let mut tw_r1 = [z; 15];
        for i in 0..15 {
            let t = vld1q_u64(tw_ptr.add(2 * i));
            let (r0, r1) = crate::gf2_kernels::k0_f128_pre_reduce_const_neon(t);
            tw_r0[i] = r0;
            tw_r1[i] = r1;
        }
        let is_zero_tw = (twiddles[0].lo
            | twiddles[0].hi
            | twiddles[1].lo
            | twiddles[1].hi
            | twiddles[3].lo
            | twiddles[3].hi
            | twiddles[7].lo
            | twiddles[7].hi)
            == 0;
        let u64_ptr = ptr as *mut u64;
        #[repr(align(64))]
        struct StageNeon {
            _buf: [uint64x2_t; 16 * 16],
        }
        let mut stage = core::mem::MaybeUninit::<StageNeon>::uninit();
        let stage_ptr = stage.as_mut_ptr() as *mut u64;

        macro_rules! btf_pair_neon {
            ($u0:ident, $u1:ident, $v0:ident, $v1:ident, $tr0:expr, $tr1:expr) => {{
                let (p0, p1) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                    $v0, $v1, $tr0, $tr1, $tr0, $tr1,
                );
                $u0 = veorq_u64($u0, p0);
                $v0 = veorq_u64($v0, $u0);
                $u1 = veorq_u64($u1, p1);
                $v1 = veorq_u64($v1, $u1);
            }};
        }

        let t0_r0 = tw_r0[0];
        let t0_r1 = tw_r1[0];
        let t1_r0 = tw_r0[1];
        let t1_r1 = tw_r1[1];
        let t2_r0 = tw_r0[2];
        let t2_r1 = tw_r1[2];

        for r in r_start..(r_start + r_count) {
            let mut lane_base = 0usize;
            while lane_base < num_ntts {
                let tile = (num_ntts - lane_base).min(16);
                for q in 0..4 {
                    let row_a = u64_ptr.add(2 * ((q * sixteenth + r) * num_ntts + lane_base));
                    let row_b = u64_ptr.add(2 * (((q + 4) * sixteenth + r) * num_ntts + lane_base));
                    let row_c = u64_ptr.add(2 * (((q + 8) * sixteenth + r) * num_ntts + lane_base));
                    let row_d =
                        u64_ptr.add(2 * (((q + 12) * sixteenth + r) * num_ntts + lane_base));
                    let dst_a = stage_ptr.add(2 * (q * 16));
                    let dst_b = stage_ptr.add(2 * ((q + 4) * 16));
                    let dst_c = stage_ptr.add(2 * ((q + 8) * 16));
                    let dst_d = stage_ptr.add(2 * ((q + 12) * 16));
                    let mut lane = 0usize;
                    if is_zero_tw {
                        while lane < tile {
                            let a0 = vld1q_u64(row_a.add(2 * lane));
                            let a1 = vld1q_u64(row_a.add(2 * (lane + 1)));
                            let mut b0 = vld1q_u64(row_b.add(2 * lane));
                            let mut b1 = vld1q_u64(row_b.add(2 * (lane + 1)));
                            let mut c0 = vld1q_u64(row_c.add(2 * lane));
                            let mut c1 = vld1q_u64(row_c.add(2 * (lane + 1)));
                            let mut d0 = vld1q_u64(row_d.add(2 * lane));
                            let mut d1 = vld1q_u64(row_d.add(2 * (lane + 1)));
                            c0 = veorq_u64(c0, a0);
                            c1 = veorq_u64(c1, a1);
                            d0 = veorq_u64(d0, b0);
                            d1 = veorq_u64(d1, b1);
                            b0 = veorq_u64(b0, a0);
                            b1 = veorq_u64(b1, a1);
                            btf_pair_neon!(c0, c1, d0, d1, t2_r0, t2_r1);
                            vst1q_u64(dst_a.add(2 * lane), a0);
                            vst1q_u64(dst_a.add(2 * (lane + 1)), a1);
                            vst1q_u64(dst_b.add(2 * lane), b0);
                            vst1q_u64(dst_b.add(2 * (lane + 1)), b1);
                            vst1q_u64(dst_c.add(2 * lane), c0);
                            vst1q_u64(dst_c.add(2 * (lane + 1)), c1);
                            vst1q_u64(dst_d.add(2 * lane), d0);
                            vst1q_u64(dst_d.add(2 * (lane + 1)), d1);
                            lane += 2;
                        }
                    } else {
                        while lane < tile {
                            let mut a0 = vld1q_u64(row_a.add(2 * lane));
                            let mut a1 = vld1q_u64(row_a.add(2 * (lane + 1)));
                            let mut b0 = vld1q_u64(row_b.add(2 * lane));
                            let mut b1 = vld1q_u64(row_b.add(2 * (lane + 1)));
                            let mut c0 = vld1q_u64(row_c.add(2 * lane));
                            let mut c1 = vld1q_u64(row_c.add(2 * (lane + 1)));
                            let mut d0 = vld1q_u64(row_d.add(2 * lane));
                            let mut d1 = vld1q_u64(row_d.add(2 * (lane + 1)));
                            btf_pair_neon!(a0, a1, c0, c1, t0_r0, t0_r1);
                            btf_pair_neon!(b0, b1, d0, d1, t0_r0, t0_r1);
                            btf_pair_neon!(a0, a1, b0, b1, t1_r0, t1_r1);
                            btf_pair_neon!(c0, c1, d0, d1, t2_r0, t2_r1);
                            vst1q_u64(dst_a.add(2 * lane), a0);
                            vst1q_u64(dst_a.add(2 * (lane + 1)), a1);
                            vst1q_u64(dst_b.add(2 * lane), b0);
                            vst1q_u64(dst_b.add(2 * (lane + 1)), b1);
                            vst1q_u64(dst_c.add(2 * lane), c0);
                            vst1q_u64(dst_c.add(2 * (lane + 1)), c1);
                            vst1q_u64(dst_d.add(2 * lane), d0);
                            vst1q_u64(dst_d.add(2 * (lane + 1)), d1);
                            lane += 2;
                        }
                    }
                }
                for s in 0..4 {
                    let to_r0 = tw_r0[3 + s];
                    let to_r1 = tw_r1[3 + s];
                    let ti0_r0 = tw_r0[7 + 2 * s];
                    let ti0_r1 = tw_r1[7 + 2 * s];
                    let ti1_r0 = tw_r0[8 + 2 * s];
                    let ti1_r1 = tw_r1[8 + 2 * s];
                    let src_a = stage_ptr.add(2 * ((4 * s) * 16));
                    let src_b = stage_ptr.add(2 * ((4 * s + 1) * 16));
                    let src_c = stage_ptr.add(2 * ((4 * s + 2) * 16));
                    let src_d = stage_ptr.add(2 * ((4 * s + 3) * 16));
                    let row_a = u64_ptr.add(2 * (((4 * s) * sixteenth + r) * num_ntts + lane_base));
                    let row_b =
                        u64_ptr.add(2 * (((4 * s + 1) * sixteenth + r) * num_ntts + lane_base));
                    let row_c =
                        u64_ptr.add(2 * (((4 * s + 2) * sixteenth + r) * num_ntts + lane_base));
                    let row_d =
                        u64_ptr.add(2 * (((4 * s + 3) * sixteenth + r) * num_ntts + lane_base));
                    let mut lane = 0usize;
                    if is_zero_tw && s == 0 {
                        while lane < tile {
                            let a0 = vld1q_u64(src_a.add(2 * lane));
                            let a1 = vld1q_u64(src_a.add(2 * (lane + 1)));
                            let mut b0 = vld1q_u64(src_b.add(2 * lane));
                            let mut b1 = vld1q_u64(src_b.add(2 * (lane + 1)));
                            let mut c0 = vld1q_u64(src_c.add(2 * lane));
                            let mut c1 = vld1q_u64(src_c.add(2 * (lane + 1)));
                            let mut d0 = vld1q_u64(src_d.add(2 * lane));
                            let mut d1 = vld1q_u64(src_d.add(2 * (lane + 1)));
                            c0 = veorq_u64(c0, a0);
                            c1 = veorq_u64(c1, a1);
                            d0 = veorq_u64(d0, b0);
                            d1 = veorq_u64(d1, b1);
                            b0 = veorq_u64(b0, a0);
                            b1 = veorq_u64(b1, a1);
                            btf_pair_neon!(c0, c1, d0, d1, ti1_r0, ti1_r1);
                            vst1q_u64(row_a.add(2 * lane), a0);
                            vst1q_u64(row_a.add(2 * (lane + 1)), a1);
                            vst1q_u64(row_b.add(2 * lane), b0);
                            vst1q_u64(row_b.add(2 * (lane + 1)), b1);
                            vst1q_u64(row_c.add(2 * lane), c0);
                            vst1q_u64(row_c.add(2 * (lane + 1)), c1);
                            vst1q_u64(row_d.add(2 * lane), d0);
                            vst1q_u64(row_d.add(2 * (lane + 1)), d1);
                            lane += 2;
                        }
                    } else {
                        while lane < tile {
                            let mut a0 = vld1q_u64(src_a.add(2 * lane));
                            let mut a1 = vld1q_u64(src_a.add(2 * (lane + 1)));
                            let mut b0 = vld1q_u64(src_b.add(2 * lane));
                            let mut b1 = vld1q_u64(src_b.add(2 * (lane + 1)));
                            let mut c0 = vld1q_u64(src_c.add(2 * lane));
                            let mut c1 = vld1q_u64(src_c.add(2 * (lane + 1)));
                            let mut d0 = vld1q_u64(src_d.add(2 * lane));
                            let mut d1 = vld1q_u64(src_d.add(2 * (lane + 1)));
                            btf_pair_neon!(a0, a1, c0, c1, to_r0, to_r1);
                            btf_pair_neon!(b0, b1, d0, d1, to_r0, to_r1);
                            btf_pair_neon!(a0, a1, b0, b1, ti0_r0, ti0_r1);
                            btf_pair_neon!(c0, c1, d0, d1, ti1_r0, ti1_r1);
                            vst1q_u64(row_a.add(2 * lane), a0);
                            vst1q_u64(row_a.add(2 * (lane + 1)), a1);
                            vst1q_u64(row_b.add(2 * lane), b0);
                            vst1q_u64(row_b.add(2 * (lane + 1)), b1);
                            vst1q_u64(row_c.add(2 * lane), c0);
                            vst1q_u64(row_c.add(2 * (lane + 1)), c1);
                            vst1q_u64(row_d.add(2 * lane), d0);
                            vst1q_u64(row_d.add(2 * (lane + 1)), d1);
                            lane += 2;
                        }
                    }
                }
                lane_base += tile;
            }
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[target_feature(enable = "neon,aes")]
#[inline]
unsafe fn butterfly_fused_4layer_rows_neon_zero_tw0(
    ptr: *mut F128,
    sixteenth: usize,
    num_ntts: usize,
    r_start: usize,
    r_count: usize,
    twiddles: &[F128; 15],
) {
    use core::arch::aarch64::*;
    unsafe {
        let z = vdupq_n_u64(0);
        let tw_ptr = twiddles.as_ptr() as *const u64;
        let mut tw_r0 = [z; 15];
        let mut tw_r1 = [z; 15];
        for i in 0..15 {
            let t = vld1q_u64(tw_ptr.add(2 * i));
            let (r0, r1) = crate::gf2_kernels::k0_f128_pre_reduce_const_neon(t);
            tw_r0[i] = r0;
            tw_r1[i] = r1;
        }
        macro_rules! btf2_neon {
            ($v:ident, $tw0:ident, $tw1:ident, $u0:expr, $v0:expr, $t0:expr, $u1:expr, $v1:expr, $t1:expr) => {{
                let (p0, p1) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                    $v[$v0], $v[$v1], $tw0[$t0], $tw1[$t0], $tw0[$t1], $tw1[$t1],
                );
                let nu0 = veorq_u64($v[$u0], p0);
                $v[$v0] = veorq_u64($v[$v0], nu0);
                $v[$u0] = nu0;
                let nu1 = veorq_u64($v[$u1], p1);
                $v[$v1] = veorq_u64($v[$v1], nu1);
                $v[$u1] = nu1;
            }};
        }
        macro_rules! btf1_zero_left_neon {
            ($v:ident, $tw0:ident, $tw1:ident, $u0:expr, $v0:expr, $u1:expr, $v1:expr, $t1:expr) => {{
                $v[$v0] = veorq_u64($v[$v0], $v[$u0]);
                let p1 = crate::gf2_kernels::k0_f128_mul_const_neon($v[$v1], $tw0[$t1], $tw1[$t1]);
                let nu1 = veorq_u64($v[$u1], p1);
                $v[$v1] = veorq_u64($v[$v1], nu1);
                $v[$u1] = nu1;
            }};
        }
        let u64_ptr = ptr as *mut u64;
        for r in r_start..(r_start + r_count) {
            for lane in 0..num_ntts {
                let mut v = [z; 16];
                for i in 0..16 {
                    let off = 2 * ((i * sixteenth + r) * num_ntts + lane);
                    v[i] = vld1q_u64(u64_ptr.add(off));
                }
                v[8] = veorq_u64(v[8], v[0]);
                v[9] = veorq_u64(v[9], v[1]);
                v[10] = veorq_u64(v[10], v[2]);
                v[11] = veorq_u64(v[11], v[3]);
                v[12] = veorq_u64(v[12], v[4]);
                v[13] = veorq_u64(v[13], v[5]);
                v[14] = veorq_u64(v[14], v[6]);
                v[15] = veorq_u64(v[15], v[7]);
                v[4] = veorq_u64(v[4], v[0]);
                v[5] = veorq_u64(v[5], v[1]);
                v[6] = veorq_u64(v[6], v[2]);
                v[7] = veorq_u64(v[7], v[3]);
                btf2_neon!(v, tw_r0, tw_r1, 8, 12, 2, 9, 13, 2);
                btf2_neon!(v, tw_r0, tw_r1, 10, 14, 2, 11, 15, 2);
                v[2] = veorq_u64(v[2], v[0]);
                v[3] = veorq_u64(v[3], v[1]);
                btf2_neon!(v, tw_r0, tw_r1, 4, 6, 4, 5, 7, 4);
                btf2_neon!(v, tw_r0, tw_r1, 8, 10, 5, 9, 11, 5);
                btf2_neon!(v, tw_r0, tw_r1, 12, 14, 6, 13, 15, 6);
                btf1_zero_left_neon!(v, tw_r0, tw_r1, 0, 1, 2, 3, 8);
                btf2_neon!(v, tw_r0, tw_r1, 4, 5, 9, 6, 7, 10);
                btf2_neon!(v, tw_r0, tw_r1, 8, 9, 11, 10, 11, 12);
                btf2_neon!(v, tw_r0, tw_r1, 12, 13, 13, 14, 15, 14);
                for i in 0..16 {
                    let off = 2 * ((i * sixteenth + r) * num_ntts + lane);
                    vst1q_u64(u64_ptr.add(off), v[i]);
                }
            }
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
#[target_feature(enable = "pclmulqdq,sse2")]
#[inline]
unsafe fn butterfly_fused_4layer_rows_x86_zero_tw0(
    ptr: *mut F128,
    sixteenth: usize,
    num_ntts: usize,
    r_start: usize,
    r_count: usize,
    twiddles: &[F128; 15],
) {
    use core::arch::x86_64::*;
    unsafe {
        let tw_ptr = twiddles.as_ptr() as *const u64;
        let mut tw_r0 = [_mm_setzero_si128(); 15];
        let mut tw_r1 = [_mm_setzero_si128(); 15];
        for i in 0..15 {
            let t = _mm_loadu_si128(tw_ptr.add(2 * i) as *const __m128i);
            let (r0, r1) = crate::gf2_kernels::k0_f128_pre_reduce_const_m128i(t);
            tw_r0[i] = r0;
            tw_r1[i] = r1;
        }
        macro_rules! btf2_x86 {
            ($v:ident, $tw0:ident, $tw1:ident, $u0:expr, $v0:expr, $t0:expr, $u1:expr, $v1:expr, $t1:expr) => {{
                let (p0, p1) = crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                    $v[$v0], $v[$v1], $tw0[$t0], $tw1[$t0], $tw0[$t1], $tw1[$t1],
                );
                let nu0 = _mm_xor_si128($v[$u0], p0);
                $v[$v0] = _mm_xor_si128($v[$v0], nu0);
                $v[$u0] = nu0;
                let nu1 = _mm_xor_si128($v[$u1], p1);
                $v[$v1] = _mm_xor_si128($v[$v1], nu1);
                $v[$u1] = nu1;
            }};
        }
        macro_rules! btf1_zero_left {
            ($v:ident, $tw0:ident, $tw1:ident, $u0:expr, $v0:expr, $u1:expr, $v1:expr, $t1:expr) => {{
                $v[$v0] = _mm_xor_si128($v[$v0], $v[$u0]);
                let p1 = crate::gf2_kernels::k0_f128_mul_const_m128i($v[$v1], $tw0[$t1], $tw1[$t1]);
                let nu1 = _mm_xor_si128($v[$u1], p1);
                $v[$v1] = _mm_xor_si128($v[$v1], nu1);
                $v[$u1] = nu1;
            }};
        }
        let u64_ptr = ptr as *mut u64;
        for r in r_start..(r_start + r_count) {
            for lane in 0..num_ntts {
                let mut v = [_mm_setzero_si128(); 16];
                for i in 0..16 {
                    let off = 2 * ((i * sixteenth + r) * num_ntts + lane);
                    v[i] = _mm_loadu_si128(u64_ptr.add(off) as *const __m128i);
                }
                v[8] = _mm_xor_si128(v[8], v[0]);
                v[9] = _mm_xor_si128(v[9], v[1]);
                v[10] = _mm_xor_si128(v[10], v[2]);
                v[11] = _mm_xor_si128(v[11], v[3]);
                v[12] = _mm_xor_si128(v[12], v[4]);
                v[13] = _mm_xor_si128(v[13], v[5]);
                v[14] = _mm_xor_si128(v[14], v[6]);
                v[15] = _mm_xor_si128(v[15], v[7]);
                v[4] = _mm_xor_si128(v[4], v[0]);
                v[5] = _mm_xor_si128(v[5], v[1]);
                v[6] = _mm_xor_si128(v[6], v[2]);
                v[7] = _mm_xor_si128(v[7], v[3]);
                btf2_x86!(v, tw_r0, tw_r1, 8, 12, 2, 9, 13, 2);
                btf2_x86!(v, tw_r0, tw_r1, 10, 14, 2, 11, 15, 2);
                v[2] = _mm_xor_si128(v[2], v[0]);
                v[3] = _mm_xor_si128(v[3], v[1]);
                btf2_x86!(v, tw_r0, tw_r1, 4, 6, 4, 5, 7, 4);
                btf2_x86!(v, tw_r0, tw_r1, 8, 10, 5, 9, 11, 5);
                btf2_x86!(v, tw_r0, tw_r1, 12, 14, 6, 13, 15, 6);
                btf1_zero_left!(v, tw_r0, tw_r1, 0, 1, 2, 3, 8);
                btf2_x86!(v, tw_r0, tw_r1, 4, 5, 9, 6, 7, 10);
                btf2_x86!(v, tw_r0, tw_r1, 8, 9, 11, 10, 11, 12);
                btf2_x86!(v, tw_r0, tw_r1, 12, 13, 13, 14, 15, 14);
                for i in 0..16 {
                    let off = 2 * ((i * sixteenth + r) * num_ntts + lane);
                    _mm_storeu_si128(u64_ptr.add(off) as *mut __m128i, v[i]);
                }
            }
        }
    }
}

#[cfg(not(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
)))]
#[inline]
pub(super) unsafe fn butterfly_fused_4layer_rows(
    ptr: *mut F128,
    sixteenth: usize,
    num_ntts: usize,
    r_start: usize,
    r_count: usize,
    twiddles: &[F128; 15],
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    unsafe {
        if num_ntts >= 2 && num_ntts <= 64 && (num_ntts & 1) == 0 {
            butterfly_fused_4layer_rows_x86_staged(
                ptr, sixteenth, num_ntts, r_start, r_count, twiddles,
            );
            return;
        }
        if (twiddles[0].lo
            | twiddles[0].hi
            | twiddles[1].lo
            | twiddles[1].hi
            | twiddles[3].lo
            | twiddles[3].hi
            | twiddles[7].lo
            | twiddles[7].hi)
            == 0
        {
            butterfly_fused_4layer_rows_x86_zero_tw0(
                ptr, sixteenth, num_ntts, r_start, r_count, twiddles,
            );
            return;
        }
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        if num_ntts >= 2 && num_ntts <= 64 && (num_ntts & 1) == 0 {
            butterfly_fused_4layer_rows_neon_staged(
                ptr, sixteenth, num_ntts, r_start, r_count, twiddles,
            );
            return;
        }
        if (twiddles[0].lo
            | twiddles[0].hi
            | twiddles[1].lo
            | twiddles[1].hi
            | twiddles[3].lo
            | twiddles[3].hi
            | twiddles[7].lo
            | twiddles[7].hi)
            == 0
        {
            butterfly_fused_4layer_rows_neon_zero_tw0(
                ptr, sixteenth, num_ntts, r_start, r_count, twiddles,
            );
            return;
        }
    }
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    unsafe {
        crate::gf2_kernels::k0_ntt_butterfly_fused_4layer_rows_slice(
            ptr as *mut u64,
            sixteenth,
            num_ntts,
            r_start,
            r_count,
            twiddles.as_ptr() as *const u64,
        );
        return;
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    unsafe {
        for r in r_start..(r_start + r_count) {
            for lane in 0..num_ntts {
                let mut values = [F128::ZERO; 16];
                for (i, value) in values.iter_mut().enumerate() {
                    *value = *ptr.add((i * sixteenth + r) * num_ntts + lane);
                }
                butterfly_fused_4layer(&mut values, twiddles);
                for (i, value) in values.iter().enumerate() {
                    *ptr.add((i * sixteenth + r) * num_ntts + lane) = *value;
                }
            }
        }
    }
}

#[allow(dead_code, clippy::too_many_arguments)]
#[inline]
pub(super) unsafe fn butterfly_fused_4layer_dual_from_src_row(
    src_ptr: *const F128,
    dst0_ptr: *mut F128,
    dst1_ptr: *mut F128,
    sixteenth: usize,
    num_ntts: usize,
    r: usize,
    tw0: &[F128; 15],
    tw1: &[F128; 15],
) {
    unsafe {
        butterfly_fused_4layer_dual_from_src_rows(
            src_ptr, dst0_ptr, dst1_ptr, sixteenth, num_ntts, r, 1, tw0, tw1,
        );
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
#[target_feature(enable = "pclmulqdq,sse2")]
#[inline]
unsafe fn butterfly_fused_4layer_dual_from_src_rows_x86(
    src_ptr: *const F128,
    dst0_ptr: *mut F128,
    dst1_ptr: *mut F128,
    sixteenth: usize,
    num_ntts: usize,
    r_start: usize,
    r_count: usize,
    tw0: &[F128; 15],
    tw1: &[F128; 15],
) {
    use core::arch::x86_64::*;
    unsafe {
        let tw0_ptr = tw0.as_ptr() as *const u64;
        let tw1_ptr = tw1.as_ptr() as *const u64;
        let mut tw0_r0 = [_mm_setzero_si128(); 15];
        let mut tw0_r1 = [_mm_setzero_si128(); 15];
        let mut tw1_r0 = [_mm_setzero_si128(); 15];
        let mut tw1_r1 = [_mm_setzero_si128(); 15];
        for i in 0..15 {
            let t0 = _mm_loadu_si128(tw0_ptr.add(2 * i) as *const __m128i);
            let (r00, r01) = crate::gf2_kernels::k0_f128_pre_reduce_const_m128i(t0);
            tw0_r0[i] = r00;
            tw0_r1[i] = r01;
            let t1 = _mm_loadu_si128(tw1_ptr.add(2 * i) as *const __m128i);
            let (r10, r11) = crate::gf2_kernels::k0_f128_pre_reduce_const_m128i(t1);
            tw1_r0[i] = r10;
            tw1_r1[i] = r11;
        }
        let tw0_is_zero = (tw0[0].lo
            | tw0[0].hi
            | tw0[1].lo
            | tw0[1].hi
            | tw0[3].lo
            | tw0[3].hi
            | tw0[7].lo
            | tw0[7].hi)
            == 0;
        let s_ptr = src_ptr as *const u64;
        let d0_ptr = dst0_ptr as *mut u64;
        let d1_ptr = dst1_ptr as *mut u64;

        if num_ntts >= 2 && num_ntts <= 64 && (num_ntts & 1) == 0 {
            #[repr(align(64))]
            struct Stage([__m128i; 16 * 16]);
            let mut stage = core::mem::MaybeUninit::<Stage>::uninit();
            let st_ptr = (*stage.as_mut_ptr()).0.as_mut_ptr();

            let poly = _mm_set1_epi64x(0x87);
            macro_rules! btf_pair {
                ($u0:ident, $u1:ident, $v0:ident, $v1:ident, $tr0:expr, $tr1:expr) => {{
                    let l0 = _mm_xor_si128(
                        _mm_clmulepi64_si128::<0x00>($v0, $tr0),
                        _mm_clmulepi64_si128::<0x11>($v0, $tr0),
                    );
                    let l1 = _mm_xor_si128(
                        _mm_clmulepi64_si128::<0x00>($v1, $tr0),
                        _mm_clmulepi64_si128::<0x11>($v1, $tr0),
                    );
                    $u0 = _mm_xor_si128($u0, l0);
                    $u1 = _mm_xor_si128($u1, l1);
                    let h0 = _mm_xor_si128(
                        _mm_clmulepi64_si128::<0x00>($v0, $tr1),
                        _mm_clmulepi64_si128::<0x11>($v0, $tr1),
                    );
                    let h1 = _mm_xor_si128(
                        _mm_clmulepi64_si128::<0x00>($v1, $tr1),
                        _mm_clmulepi64_si128::<0x11>($v1, $tr1),
                    );
                    let r0 = _mm_xor_si128(
                        _mm_bslli_si128::<8>(h0),
                        _mm_clmulepi64_si128::<0x01>(h0, poly),
                    );
                    let r1 = _mm_xor_si128(
                        _mm_bslli_si128::<8>(h1),
                        _mm_clmulepi64_si128::<0x01>(h1, poly),
                    );
                    $u0 = _mm_xor_si128($u0, r0);
                    $u1 = _mm_xor_si128($u1, r1);
                    $v0 = _mm_xor_si128($v0, $u0);
                    $v1 = _mm_xor_si128($v1, $u1);
                }};
            }
            macro_rules! radix4_1lane {
                ($a:ident, $b:ident, $c:ident, $d:ident, $to0:expr, $to1:expr, $ti00:expr, $ti01:expr, $ti10:expr, $ti11:expr) => {{
                    let l0 = _mm_xor_si128(
                        _mm_clmulepi64_si128::<0x00>($c, $to0),
                        _mm_clmulepi64_si128::<0x11>($c, $to0),
                    );
                    let l1 = _mm_xor_si128(
                        _mm_clmulepi64_si128::<0x00>($d, $to0),
                        _mm_clmulepi64_si128::<0x11>($d, $to0),
                    );
                    $a = _mm_xor_si128($a, l0);
                    $b = _mm_xor_si128($b, l1);
                    let h0 = _mm_xor_si128(
                        _mm_clmulepi64_si128::<0x00>($c, $to1),
                        _mm_clmulepi64_si128::<0x11>($c, $to1),
                    );
                    let h1 = _mm_xor_si128(
                        _mm_clmulepi64_si128::<0x00>($d, $to1),
                        _mm_clmulepi64_si128::<0x11>($d, $to1),
                    );
                    let r0 = _mm_xor_si128(
                        _mm_bslli_si128::<8>(h0),
                        _mm_clmulepi64_si128::<0x01>(h0, poly),
                    );
                    let r1 = _mm_xor_si128(
                        _mm_bslli_si128::<8>(h1),
                        _mm_clmulepi64_si128::<0x01>(h1, poly),
                    );
                    $a = _mm_xor_si128($a, r0);
                    $b = _mm_xor_si128($b, r1);
                    $c = _mm_xor_si128($c, $a);
                    $d = _mm_xor_si128($d, $b);

                    let l0 = _mm_xor_si128(
                        _mm_clmulepi64_si128::<0x00>($b, $ti00),
                        _mm_clmulepi64_si128::<0x11>($b, $ti00),
                    );
                    let l1 = _mm_xor_si128(
                        _mm_clmulepi64_si128::<0x00>($d, $ti10),
                        _mm_clmulepi64_si128::<0x11>($d, $ti10),
                    );
                    $a = _mm_xor_si128($a, l0);
                    $c = _mm_xor_si128($c, l1);
                    let h0 = _mm_xor_si128(
                        _mm_clmulepi64_si128::<0x00>($b, $ti01),
                        _mm_clmulepi64_si128::<0x11>($b, $ti01),
                    );
                    let h1 = _mm_xor_si128(
                        _mm_clmulepi64_si128::<0x00>($d, $ti11),
                        _mm_clmulepi64_si128::<0x11>($d, $ti11),
                    );
                    let r0 = _mm_xor_si128(
                        _mm_bslli_si128::<8>(h0),
                        _mm_clmulepi64_si128::<0x01>(h0, poly),
                    );
                    let r1 = _mm_xor_si128(
                        _mm_bslli_si128::<8>(h1),
                        _mm_clmulepi64_si128::<0x01>(h1, poly),
                    );
                    $a = _mm_xor_si128($a, r0);
                    $c = _mm_xor_si128($c, r1);
                    $b = _mm_xor_si128($b, $a);
                    $d = _mm_xor_si128($d, $c);
                }};
            }

            let t00_r0 = tw0_r0[0];
            let t00_r1 = tw0_r1[0];
            let t01_r0 = tw0_r0[1];
            let t01_r1 = tw0_r1[1];
            let t02_r0 = tw0_r0[2];
            let t02_r1 = tw0_r1[2];

            let t10_r0 = tw1_r0[0];
            let t10_r1 = tw1_r1[0];
            let t11_r0 = tw1_r0[1];
            let t11_r1 = tw1_r1[1];
            let t12_r0 = tw1_r0[2];
            let t12_r1 = tw1_r1[2];

            for r in r_start..(r_start + r_count) {
                let mut lane_base = 0usize;
                while lane_base < num_ntts {
                    let tile = (num_ntts - lane_base).min(16);

                    if tw0_is_zero {
                        for q in 0..4 {
                            let row_a = (s_ptr.add(2 * ((q * sixteenth + r) * num_ntts))
                                as *const __m128i)
                                .add(lane_base);
                            let row_b = (s_ptr.add(2 * (((q + 4) * sixteenth + r) * num_ntts))
                                as *const __m128i)
                                .add(lane_base);
                            let row_c = (s_ptr.add(2 * (((q + 8) * sixteenth + r) * num_ntts))
                                as *const __m128i)
                                .add(lane_base);
                            let row_d = (s_ptr.add(2 * (((q + 12) * sixteenth + r) * num_ntts))
                                as *const __m128i)
                                .add(lane_base);
                            let dst_a = st_ptr.add(q * 16);
                            let dst_b = st_ptr.add((q + 4) * 16);
                            let dst_c = st_ptr.add((q + 8) * 16);
                            let dst_d = st_ptr.add((q + 12) * 16);
                            let mut lane = 0usize;
                            while lane < tile {
                                let a0 = _mm_loadu_si128(row_a.add(lane));
                                let a1 = _mm_loadu_si128(row_a.add(lane + 1));
                                let mut b0 = _mm_loadu_si128(row_b.add(lane));
                                let mut b1 = _mm_loadu_si128(row_b.add(lane + 1));
                                let mut c0 = _mm_loadu_si128(row_c.add(lane));
                                let mut c1 = _mm_loadu_si128(row_c.add(lane + 1));
                                let mut d0 = _mm_loadu_si128(row_d.add(lane));
                                let mut d1 = _mm_loadu_si128(row_d.add(lane + 1));
                                c0 = _mm_xor_si128(c0, a0);
                                c1 = _mm_xor_si128(c1, a1);
                                d0 = _mm_xor_si128(d0, b0);
                                d1 = _mm_xor_si128(d1, b1);
                                b0 = _mm_xor_si128(b0, a0);
                                b1 = _mm_xor_si128(b1, a1);
                                btf_pair!(c0, c1, d0, d1, t02_r0, t02_r1);
                                _mm_store_si128(dst_a.add(lane), a0);
                                _mm_store_si128(dst_a.add(lane + 1), a1);
                                _mm_store_si128(dst_b.add(lane), b0);
                                _mm_store_si128(dst_b.add(lane + 1), b1);
                                _mm_store_si128(dst_c.add(lane), c0);
                                _mm_store_si128(dst_c.add(lane + 1), c1);
                                _mm_store_si128(dst_d.add(lane), d0);
                                _mm_store_si128(dst_d.add(lane + 1), d1);
                                lane += 2;
                            }
                        }
                    } else {
                        for q in 0..4 {
                            let row_a = (s_ptr.add(2 * ((q * sixteenth + r) * num_ntts))
                                as *const __m128i)
                                .add(lane_base);
                            let row_b = (s_ptr.add(2 * (((q + 4) * sixteenth + r) * num_ntts))
                                as *const __m128i)
                                .add(lane_base);
                            let row_c = (s_ptr.add(2 * (((q + 8) * sixteenth + r) * num_ntts))
                                as *const __m128i)
                                .add(lane_base);
                            let row_d = (s_ptr.add(2 * (((q + 12) * sixteenth + r) * num_ntts))
                                as *const __m128i)
                                .add(lane_base);
                            let dst_a = st_ptr.add(q * 16);
                            let dst_b = st_ptr.add((q + 4) * 16);
                            let dst_c = st_ptr.add((q + 8) * 16);
                            let dst_d = st_ptr.add((q + 12) * 16);
                            for lane in 0..tile {
                                let mut a = _mm_loadu_si128(row_a.add(lane));
                                let mut b = _mm_loadu_si128(row_b.add(lane));
                                let mut c = _mm_loadu_si128(row_c.add(lane));
                                let mut d = _mm_loadu_si128(row_d.add(lane));
                                radix4_1lane!(
                                    a, b, c, d, t00_r0, t00_r1, t01_r0, t01_r1, t02_r0, t02_r1
                                );
                                _mm_store_si128(dst_a.add(lane), a);
                                _mm_store_si128(dst_b.add(lane), b);
                                _mm_store_si128(dst_c.add(lane), c);
                                _mm_store_si128(dst_d.add(lane), d);
                            }
                        }
                    }

                    for s in 0..4 {
                        let to_r0 = tw0_r0[3 + s];
                        let to_r1 = tw0_r1[3 + s];
                        let ti0_r0 = tw0_r0[7 + 2 * s];
                        let ti0_r1 = tw0_r1[7 + 2 * s];
                        let ti1_r0 = tw0_r0[8 + 2 * s];
                        let ti1_r1 = tw0_r1[8 + 2 * s];
                        let src_a = st_ptr.add((4 * s) * 16);
                        let src_b = st_ptr.add((4 * s + 1) * 16);
                        let src_c = st_ptr.add((4 * s + 2) * 16);
                        let src_d = st_ptr.add((4 * s + 3) * 16);
                        let row_a = (d0_ptr.add(2 * (((4 * s) * sixteenth + r) * num_ntts))
                            as *mut __m128i)
                            .add(lane_base);
                        let row_b = (d0_ptr.add(2 * (((4 * s + 1) * sixteenth + r) * num_ntts))
                            as *mut __m128i)
                            .add(lane_base);
                        let row_c = (d0_ptr.add(2 * (((4 * s + 2) * sixteenth + r) * num_ntts))
                            as *mut __m128i)
                            .add(lane_base);
                        let row_d = (d0_ptr.add(2 * (((4 * s + 3) * sixteenth + r) * num_ntts))
                            as *mut __m128i)
                            .add(lane_base);
                        if tw0_is_zero && s == 0 {
                            let mut lane = 0usize;
                            while lane < tile {
                                let a0 = _mm_load_si128(src_a.add(lane));
                                let a1 = _mm_load_si128(src_a.add(lane + 1));
                                let mut b0 = _mm_load_si128(src_b.add(lane));
                                let mut b1 = _mm_load_si128(src_b.add(lane + 1));
                                let mut c0 = _mm_load_si128(src_c.add(lane));
                                let mut c1 = _mm_load_si128(src_c.add(lane + 1));
                                let mut d0 = _mm_load_si128(src_d.add(lane));
                                let mut d1 = _mm_load_si128(src_d.add(lane + 1));
                                c0 = _mm_xor_si128(c0, a0);
                                c1 = _mm_xor_si128(c1, a1);
                                d0 = _mm_xor_si128(d0, b0);
                                d1 = _mm_xor_si128(d1, b1);
                                b0 = _mm_xor_si128(b0, a0);
                                b1 = _mm_xor_si128(b1, a1);
                                btf_pair!(c0, c1, d0, d1, ti1_r0, ti1_r1);
                                _mm_storeu_si128(row_a.add(lane), a0);
                                _mm_storeu_si128(row_a.add(lane + 1), a1);
                                _mm_storeu_si128(row_b.add(lane), b0);
                                _mm_storeu_si128(row_b.add(lane + 1), b1);
                                _mm_storeu_si128(row_c.add(lane), c0);
                                _mm_storeu_si128(row_c.add(lane + 1), c1);
                                _mm_storeu_si128(row_d.add(lane), d0);
                                _mm_storeu_si128(row_d.add(lane + 1), d1);
                                lane += 2;
                            }
                        } else {
                            for lane in 0..tile {
                                let mut a = _mm_load_si128(src_a.add(lane));
                                let mut b = _mm_load_si128(src_b.add(lane));
                                let mut c = _mm_load_si128(src_c.add(lane));
                                let mut d = _mm_load_si128(src_d.add(lane));
                                radix4_1lane!(
                                    a, b, c, d, to_r0, to_r1, ti0_r0, ti0_r1, ti1_r0, ti1_r1
                                );
                                _mm_storeu_si128(row_a.add(lane), a);
                                _mm_storeu_si128(row_b.add(lane), b);
                                _mm_storeu_si128(row_c.add(lane), c);
                                _mm_storeu_si128(row_d.add(lane), d);
                            }
                        }
                    }

                    for q in 0..4 {
                        let row_a = (s_ptr.add(2 * ((q * sixteenth + r) * num_ntts))
                            as *const __m128i)
                            .add(lane_base);
                        let row_b = (s_ptr.add(2 * (((q + 4) * sixteenth + r) * num_ntts))
                            as *const __m128i)
                            .add(lane_base);
                        let row_c = (s_ptr.add(2 * (((q + 8) * sixteenth + r) * num_ntts))
                            as *const __m128i)
                            .add(lane_base);
                        let row_d = (s_ptr.add(2 * (((q + 12) * sixteenth + r) * num_ntts))
                            as *const __m128i)
                            .add(lane_base);
                        let dst_a = st_ptr.add(q * 16);
                        let dst_b = st_ptr.add((q + 4) * 16);
                        let dst_c = st_ptr.add((q + 8) * 16);
                        let dst_d = st_ptr.add((q + 12) * 16);
                        for lane in 0..tile {
                            let mut a = _mm_loadu_si128(row_a.add(lane));
                            let mut b = _mm_loadu_si128(row_b.add(lane));
                            let mut c = _mm_loadu_si128(row_c.add(lane));
                            let mut d = _mm_loadu_si128(row_d.add(lane));
                            radix4_1lane!(
                                a, b, c, d, t10_r0, t10_r1, t11_r0, t11_r1, t12_r0, t12_r1
                            );
                            _mm_store_si128(dst_a.add(lane), a);
                            _mm_store_si128(dst_b.add(lane), b);
                            _mm_store_si128(dst_c.add(lane), c);
                            _mm_store_si128(dst_d.add(lane), d);
                        }
                    }

                    for s in 0..4 {
                        let to_r0 = tw1_r0[3 + s];
                        let to_r1 = tw1_r1[3 + s];
                        let ti0_r0 = tw1_r0[7 + 2 * s];
                        let ti0_r1 = tw1_r1[7 + 2 * s];
                        let ti1_r0 = tw1_r0[8 + 2 * s];
                        let ti1_r1 = tw1_r1[8 + 2 * s];
                        let src_a = st_ptr.add((4 * s) * 16);
                        let src_b = st_ptr.add((4 * s + 1) * 16);
                        let src_c = st_ptr.add((4 * s + 2) * 16);
                        let src_d = st_ptr.add((4 * s + 3) * 16);
                        let row_a = (d1_ptr.add(2 * (((4 * s) * sixteenth + r) * num_ntts))
                            as *mut __m128i)
                            .add(lane_base);
                        let row_b = (d1_ptr.add(2 * (((4 * s + 1) * sixteenth + r) * num_ntts))
                            as *mut __m128i)
                            .add(lane_base);
                        let row_c = (d1_ptr.add(2 * (((4 * s + 2) * sixteenth + r) * num_ntts))
                            as *mut __m128i)
                            .add(lane_base);
                        let row_d = (d1_ptr.add(2 * (((4 * s + 3) * sixteenth + r) * num_ntts))
                            as *mut __m128i)
                            .add(lane_base);
                        for lane in 0..tile {
                            let mut a = _mm_load_si128(src_a.add(lane));
                            let mut b = _mm_load_si128(src_b.add(lane));
                            let mut c = _mm_load_si128(src_c.add(lane));
                            let mut d = _mm_load_si128(src_d.add(lane));
                            radix4_1lane!(a, b, c, d, to_r0, to_r1, ti0_r0, ti0_r1, ti1_r0, ti1_r1);
                            _mm_storeu_si128(row_a.add(lane), a);
                            _mm_storeu_si128(row_b.add(lane), b);
                            _mm_storeu_si128(row_c.add(lane), c);
                            _mm_storeu_si128(row_d.add(lane), d);
                        }
                    }
                    lane_base += tile;
                }
            }
            return;
        }
        macro_rules! btf2_x86 {
            ($v:ident, $tw0:ident, $tw1:ident, $u0:expr, $v0:expr, $t0:expr, $u1:expr, $v1:expr, $t1:expr) => {{
                let (p0, p1) = crate::gf2_kernels::k0_f128_mul_const_vec2_m128i(
                    $v[$v0], $v[$v1], $tw0[$t0], $tw1[$t0], $tw0[$t1], $tw1[$t1],
                );
                let nu0 = _mm_xor_si128($v[$u0], p0);
                $v[$v0] = _mm_xor_si128($v[$v0], nu0);
                $v[$u0] = nu0;
                let nu1 = _mm_xor_si128($v[$u1], p1);
                $v[$v1] = _mm_xor_si128($v[$v1], nu1);
                $v[$u1] = nu1;
            }};
        }
        macro_rules! btf1_zero_left {
            ($v:ident, $tw0:ident, $tw1:ident, $u0:expr, $v0:expr, $u1:expr, $v1:expr, $t1:expr) => {{
                $v[$v0] = _mm_xor_si128($v[$v0], $v[$u0]);
                let p1 = crate::gf2_kernels::k0_f128_mul_const_m128i($v[$v1], $tw0[$t1], $tw1[$t1]);
                let nu1 = _mm_xor_si128($v[$u1], p1);
                $v[$v1] = _mm_xor_si128($v[$v1], nu1);
                $v[$u1] = nu1;
            }};
        }

        for r in r_start..(r_start + r_count) {
            for lane in 0..num_ntts {
                let mut v = [_mm_setzero_si128(); 16];
                for i in 0..16 {
                    let off = 2 * ((i * sixteenth + r) * num_ntts + lane);
                    v[i] = _mm_loadu_si128(s_ptr.add(off) as *const __m128i);
                }
                if tw0_is_zero {
                    v[8] = _mm_xor_si128(v[8], v[0]);
                    v[9] = _mm_xor_si128(v[9], v[1]);
                    v[10] = _mm_xor_si128(v[10], v[2]);
                    v[11] = _mm_xor_si128(v[11], v[3]);
                    v[12] = _mm_xor_si128(v[12], v[4]);
                    v[13] = _mm_xor_si128(v[13], v[5]);
                    v[14] = _mm_xor_si128(v[14], v[6]);
                    v[15] = _mm_xor_si128(v[15], v[7]);
                    v[4] = _mm_xor_si128(v[4], v[0]);
                    v[5] = _mm_xor_si128(v[5], v[1]);
                    v[6] = _mm_xor_si128(v[6], v[2]);
                    v[7] = _mm_xor_si128(v[7], v[3]);
                    btf2_x86!(v, tw0_r0, tw0_r1, 8, 12, 2, 9, 13, 2);
                    btf2_x86!(v, tw0_r0, tw0_r1, 10, 14, 2, 11, 15, 2);
                    v[2] = _mm_xor_si128(v[2], v[0]);
                    v[3] = _mm_xor_si128(v[3], v[1]);
                    btf2_x86!(v, tw0_r0, tw0_r1, 4, 6, 4, 5, 7, 4);
                    btf2_x86!(v, tw0_r0, tw0_r1, 8, 10, 5, 9, 11, 5);
                    btf2_x86!(v, tw0_r0, tw0_r1, 12, 14, 6, 13, 15, 6);
                    btf1_zero_left!(v, tw0_r0, tw0_r1, 0, 1, 2, 3, 8);
                    btf2_x86!(v, tw0_r0, tw0_r1, 4, 5, 9, 6, 7, 10);
                    btf2_x86!(v, tw0_r0, tw0_r1, 8, 9, 11, 10, 11, 12);
                    btf2_x86!(v, tw0_r0, tw0_r1, 12, 13, 13, 14, 15, 14);
                } else {
                    btf2_x86!(v, tw0_r0, tw0_r1, 0, 8, 0, 1, 9, 0);
                    btf2_x86!(v, tw0_r0, tw0_r1, 2, 10, 0, 3, 11, 0);
                    btf2_x86!(v, tw0_r0, tw0_r1, 4, 12, 0, 5, 13, 0);
                    btf2_x86!(v, tw0_r0, tw0_r1, 6, 14, 0, 7, 15, 0);
                    btf2_x86!(v, tw0_r0, tw0_r1, 0, 4, 1, 1, 5, 1);
                    btf2_x86!(v, tw0_r0, tw0_r1, 2, 6, 1, 3, 7, 1);
                    btf2_x86!(v, tw0_r0, tw0_r1, 8, 12, 2, 9, 13, 2);
                    btf2_x86!(v, tw0_r0, tw0_r1, 10, 14, 2, 11, 15, 2);
                    btf2_x86!(v, tw0_r0, tw0_r1, 0, 2, 3, 1, 3, 3);
                    btf2_x86!(v, tw0_r0, tw0_r1, 4, 6, 4, 5, 7, 4);
                    btf2_x86!(v, tw0_r0, tw0_r1, 8, 10, 5, 9, 11, 5);
                    btf2_x86!(v, tw0_r0, tw0_r1, 12, 14, 6, 13, 15, 6);
                    btf2_x86!(v, tw0_r0, tw0_r1, 0, 1, 7, 2, 3, 8);
                    btf2_x86!(v, tw0_r0, tw0_r1, 4, 5, 9, 6, 7, 10);
                    btf2_x86!(v, tw0_r0, tw0_r1, 8, 9, 11, 10, 11, 12);
                    btf2_x86!(v, tw0_r0, tw0_r1, 12, 13, 13, 14, 15, 14);
                }
                for i in 0..16 {
                    let off = 2 * ((i * sixteenth + r) * num_ntts + lane);
                    _mm_storeu_si128(d0_ptr.add(off) as *mut __m128i, v[i]);
                }

                for i in 0..16 {
                    let off = 2 * ((i * sixteenth + r) * num_ntts + lane);
                    v[i] = _mm_loadu_si128(s_ptr.add(off) as *const __m128i);
                }
                btf2_x86!(v, tw1_r0, tw1_r1, 0, 8, 0, 1, 9, 0);
                btf2_x86!(v, tw1_r0, tw1_r1, 2, 10, 0, 3, 11, 0);
                btf2_x86!(v, tw1_r0, tw1_r1, 4, 12, 0, 5, 13, 0);
                btf2_x86!(v, tw1_r0, tw1_r1, 6, 14, 0, 7, 15, 0);
                btf2_x86!(v, tw1_r0, tw1_r1, 0, 4, 1, 1, 5, 1);
                btf2_x86!(v, tw1_r0, tw1_r1, 2, 6, 1, 3, 7, 1);
                btf2_x86!(v, tw1_r0, tw1_r1, 8, 12, 2, 9, 13, 2);
                btf2_x86!(v, tw1_r0, tw1_r1, 10, 14, 2, 11, 15, 2);
                btf2_x86!(v, tw1_r0, tw1_r1, 0, 2, 3, 1, 3, 3);
                btf2_x86!(v, tw1_r0, tw1_r1, 4, 6, 4, 5, 7, 4);
                btf2_x86!(v, tw1_r0, tw1_r1, 8, 10, 5, 9, 11, 5);
                btf2_x86!(v, tw1_r0, tw1_r1, 12, 14, 6, 13, 15, 6);
                btf2_x86!(v, tw1_r0, tw1_r1, 0, 1, 7, 2, 3, 8);
                btf2_x86!(v, tw1_r0, tw1_r1, 4, 5, 9, 6, 7, 10);
                btf2_x86!(v, tw1_r0, tw1_r1, 8, 9, 11, 10, 11, 12);
                btf2_x86!(v, tw1_r0, tw1_r1, 12, 13, 13, 14, 15, 14);
                for i in 0..16 {
                    let off = 2 * ((i * sixteenth + r) * num_ntts + lane);
                    _mm_storeu_si128(d1_ptr.add(off) as *mut __m128i, v[i]);
                }
            }
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
#[target_feature(enable = "neon,aes")]
#[inline]
unsafe fn butterfly_fused_4layer_dual_from_src_rows_neon(
    src_ptr: *const F128,
    dst0_ptr: *mut F128,
    dst1_ptr: *mut F128,
    sixteenth: usize,
    num_ntts: usize,
    r_start: usize,
    r_count: usize,
    tw0: &[F128; 15],
    tw1: &[F128; 15],
) {
    use core::arch::aarch64::*;
    unsafe {
        let z = vdupq_n_u64(0);
        let tw0_ptr = tw0.as_ptr() as *const u64;
        let tw1_ptr = tw1.as_ptr() as *const u64;
        let mut tw0_r0 = [z; 15];
        let mut tw0_r1 = [z; 15];
        let mut tw1_r0 = [z; 15];
        let mut tw1_r1 = [z; 15];
        for i in 0..15 {
            let t0 = vld1q_u64(tw0_ptr.add(2 * i));
            let (r00, r01) = crate::gf2_kernels::k0_f128_pre_reduce_const_neon(t0);
            tw0_r0[i] = r00;
            tw0_r1[i] = r01;
            let t1 = vld1q_u64(tw1_ptr.add(2 * i));
            let (r10, r11) = crate::gf2_kernels::k0_f128_pre_reduce_const_neon(t1);
            tw1_r0[i] = r10;
            tw1_r1[i] = r11;
        }
        macro_rules! btf2_neon {
            ($v:ident, $tw0:ident, $tw1:ident, $u0:expr, $v0:expr, $t0:expr, $u1:expr, $v1:expr, $t1:expr) => {{
                let (p0, p1) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                    $v[$v0], $v[$v1], $tw0[$t0], $tw1[$t0], $tw0[$t1], $tw1[$t1],
                );
                let nu0 = veorq_u64($v[$u0], p0);
                $v[$v0] = veorq_u64($v[$v0], nu0);
                $v[$u0] = nu0;
                let nu1 = veorq_u64($v[$u1], p1);
                $v[$v1] = veorq_u64($v[$v1], nu1);
                $v[$u1] = nu1;
            }};
        }
        macro_rules! btf1_zero_left_neon {
            ($v:ident, $tw0:ident, $tw1:ident, $u0:expr, $v0:expr, $u1:expr, $v1:expr, $t1:expr) => {{
                $v[$v0] = veorq_u64($v[$v0], $v[$u0]);
                let p1 = crate::gf2_kernels::k0_f128_mul_const_neon($v[$v1], $tw0[$t1], $tw1[$t1]);
                let nu1 = veorq_u64($v[$u1], p1);
                $v[$v1] = veorq_u64($v[$v1], nu1);
                $v[$u1] = nu1;
            }};
        }
        let tw0_is_zero = (tw0[0].lo
            | tw0[0].hi
            | tw0[1].lo
            | tw0[1].hi
            | tw0[3].lo
            | tw0[3].hi
            | tw0[7].lo
            | tw0[7].hi)
            == 0;
        let s_ptr = src_ptr as *const u64;
        let d0_ptr = dst0_ptr as *mut u64;
        let d1_ptr = dst1_ptr as *mut u64;

        if num_ntts >= 2 && num_ntts <= 64 && (num_ntts & 1) == 0 {
            #[repr(align(64))]
            struct DualStageNeon {
                s0: [uint64x2_t; 16 * 16],
                s1: [uint64x2_t; 16 * 16],
            }
            let mut stage = core::mem::MaybeUninit::<DualStageNeon>::uninit();
            let st0_ptr = (*stage.as_mut_ptr()).s0.as_mut_ptr() as *mut u64;
            let st1_ptr = (*stage.as_mut_ptr()).s1.as_mut_ptr() as *mut u64;

            macro_rules! btf_pair_neon {
                ($u0:ident, $u1:ident, $v0:ident, $v1:ident, $tr0:expr, $tr1:expr) => {{
                    let (p0, p1) = crate::gf2_kernels::k0_f128_mul_const_vec2_neon(
                        $v0, $v1, $tr0, $tr1, $tr0, $tr1,
                    );
                    $u0 = veorq_u64($u0, p0);
                    $v0 = veorq_u64($v0, $u0);
                    $u1 = veorq_u64($u1, p1);
                    $v1 = veorq_u64($v1, $u1);
                }};
            }

            let t00_r0 = tw0_r0[0];
            let t00_r1 = tw0_r1[0];
            let t01_r0 = tw0_r0[1];
            let t01_r1 = tw0_r1[1];
            let t02_r0 = tw0_r0[2];
            let t02_r1 = tw0_r1[2];

            let t10_r0 = tw1_r0[0];
            let t10_r1 = tw1_r1[0];
            let t11_r0 = tw1_r0[1];
            let t11_r1 = tw1_r1[1];
            let t12_r0 = tw1_r0[2];
            let t12_r1 = tw1_r1[2];

            for r in r_start..(r_start + r_count) {
                let mut lane_base = 0usize;
                while lane_base < num_ntts {
                    let tile = (num_ntts - lane_base).min(16);
                    for q in 0..4 {
                        let row_a = s_ptr.add(2 * ((q * sixteenth + r) * num_ntts + lane_base));
                        let row_b =
                            s_ptr.add(2 * (((q + 4) * sixteenth + r) * num_ntts + lane_base));
                        let row_c =
                            s_ptr.add(2 * (((q + 8) * sixteenth + r) * num_ntts + lane_base));
                        let row_d =
                            s_ptr.add(2 * (((q + 12) * sixteenth + r) * num_ntts + lane_base));
                        let dst0_a = st0_ptr.add(2 * (q * 16));
                        let dst0_b = st0_ptr.add(2 * ((q + 4) * 16));
                        let dst0_c = st0_ptr.add(2 * ((q + 8) * 16));
                        let dst0_d = st0_ptr.add(2 * ((q + 12) * 16));
                        let dst1_a = st1_ptr.add(2 * (q * 16));
                        let dst1_b = st1_ptr.add(2 * ((q + 4) * 16));
                        let dst1_c = st1_ptr.add(2 * ((q + 8) * 16));
                        let dst1_d = st1_ptr.add(2 * ((q + 12) * 16));
                        let mut lane = 0usize;
                        while lane < tile {
                            let raw_a0 = vld1q_u64(row_a.add(2 * lane));
                            let raw_a1 = vld1q_u64(row_a.add(2 * (lane + 1)));
                            let raw_b0 = vld1q_u64(row_b.add(2 * lane));
                            let raw_b1 = vld1q_u64(row_b.add(2 * (lane + 1)));
                            let raw_c0 = vld1q_u64(row_c.add(2 * lane));
                            let raw_c1 = vld1q_u64(row_c.add(2 * (lane + 1)));
                            let raw_d0 = vld1q_u64(row_d.add(2 * lane));
                            let raw_d1 = vld1q_u64(row_d.add(2 * (lane + 1)));

                            let mut a0 = raw_a0;
                            let mut a1 = raw_a1;
                            let mut b0 = raw_b0;
                            let mut b1 = raw_b1;
                            let mut c0 = raw_c0;
                            let mut c1 = raw_c1;
                            let mut d0 = raw_d0;
                            let mut d1 = raw_d1;
                            if tw0_is_zero {
                                c0 = veorq_u64(c0, a0);
                                c1 = veorq_u64(c1, a1);
                                d0 = veorq_u64(d0, b0);
                                d1 = veorq_u64(d1, b1);
                                b0 = veorq_u64(b0, a0);
                                b1 = veorq_u64(b1, a1);
                                btf_pair_neon!(c0, c1, d0, d1, t02_r0, t02_r1);
                            } else {
                                btf_pair_neon!(a0, a1, c0, c1, t00_r0, t00_r1);
                                btf_pair_neon!(b0, b1, d0, d1, t00_r0, t00_r1);
                                btf_pair_neon!(a0, a1, b0, b1, t01_r0, t01_r1);
                                btf_pair_neon!(c0, c1, d0, d1, t02_r0, t02_r1);
                            }
                            vst1q_u64(dst0_a.add(2 * lane), a0);
                            vst1q_u64(dst0_a.add(2 * (lane + 1)), a1);
                            vst1q_u64(dst0_b.add(2 * lane), b0);
                            vst1q_u64(dst0_b.add(2 * (lane + 1)), b1);
                            vst1q_u64(dst0_c.add(2 * lane), c0);
                            vst1q_u64(dst0_c.add(2 * (lane + 1)), c1);
                            vst1q_u64(dst0_d.add(2 * lane), d0);
                            vst1q_u64(dst0_d.add(2 * (lane + 1)), d1);

                            let mut a0 = raw_a0;
                            let mut a1 = raw_a1;
                            let mut b0 = raw_b0;
                            let mut b1 = raw_b1;
                            let mut c0 = raw_c0;
                            let mut c1 = raw_c1;
                            let mut d0 = raw_d0;
                            let mut d1 = raw_d1;
                            btf_pair_neon!(a0, a1, c0, c1, t10_r0, t10_r1);
                            btf_pair_neon!(b0, b1, d0, d1, t10_r0, t10_r1);
                            btf_pair_neon!(a0, a1, b0, b1, t11_r0, t11_r1);
                            btf_pair_neon!(c0, c1, d0, d1, t12_r0, t12_r1);
                            vst1q_u64(dst1_a.add(2 * lane), a0);
                            vst1q_u64(dst1_a.add(2 * (lane + 1)), a1);
                            vst1q_u64(dst1_b.add(2 * lane), b0);
                            vst1q_u64(dst1_b.add(2 * (lane + 1)), b1);
                            vst1q_u64(dst1_c.add(2 * lane), c0);
                            vst1q_u64(dst1_c.add(2 * (lane + 1)), c1);
                            vst1q_u64(dst1_d.add(2 * lane), d0);
                            vst1q_u64(dst1_d.add(2 * (lane + 1)), d1);
                            lane += 2;
                        }
                    }

                    for s in 0..4 {
                        let to_r0 = tw0_r0[3 + s];
                        let to_r1 = tw0_r1[3 + s];
                        let ti0_r0 = tw0_r0[7 + 2 * s];
                        let ti0_r1 = tw0_r1[7 + 2 * s];
                        let ti1_r0 = tw0_r0[8 + 2 * s];
                        let ti1_r1 = tw0_r1[8 + 2 * s];
                        let src_a = st0_ptr.add(2 * ((4 * s) * 16));
                        let src_b = st0_ptr.add(2 * ((4 * s + 1) * 16));
                        let src_c = st0_ptr.add(2 * ((4 * s + 2) * 16));
                        let src_d = st0_ptr.add(2 * ((4 * s + 3) * 16));
                        let row_a =
                            d0_ptr.add(2 * (((4 * s) * sixteenth + r) * num_ntts + lane_base));
                        let row_b =
                            d0_ptr.add(2 * (((4 * s + 1) * sixteenth + r) * num_ntts + lane_base));
                        let row_c =
                            d0_ptr.add(2 * (((4 * s + 2) * sixteenth + r) * num_ntts + lane_base));
                        let row_d =
                            d0_ptr.add(2 * (((4 * s + 3) * sixteenth + r) * num_ntts + lane_base));
                        let mut lane = 0usize;
                        if tw0_is_zero && s == 0 {
                            while lane < tile {
                                let a0 = vld1q_u64(src_a.add(2 * lane));
                                let a1 = vld1q_u64(src_a.add(2 * (lane + 1)));
                                let mut b0 = vld1q_u64(src_b.add(2 * lane));
                                let mut b1 = vld1q_u64(src_b.add(2 * (lane + 1)));
                                let mut c0 = vld1q_u64(src_c.add(2 * lane));
                                let mut c1 = vld1q_u64(src_c.add(2 * (lane + 1)));
                                let mut d0 = vld1q_u64(src_d.add(2 * lane));
                                let mut d1 = vld1q_u64(src_d.add(2 * (lane + 1)));
                                c0 = veorq_u64(c0, a0);
                                c1 = veorq_u64(c1, a1);
                                d0 = veorq_u64(d0, b0);
                                d1 = veorq_u64(d1, b1);
                                b0 = veorq_u64(b0, a0);
                                b1 = veorq_u64(b1, a1);
                                btf_pair_neon!(c0, c1, d0, d1, ti1_r0, ti1_r1);
                                vst1q_u64(row_a.add(2 * lane), a0);
                                vst1q_u64(row_a.add(2 * (lane + 1)), a1);
                                vst1q_u64(row_b.add(2 * lane), b0);
                                vst1q_u64(row_b.add(2 * (lane + 1)), b1);
                                vst1q_u64(row_c.add(2 * lane), c0);
                                vst1q_u64(row_c.add(2 * (lane + 1)), c1);
                                vst1q_u64(row_d.add(2 * lane), d0);
                                vst1q_u64(row_d.add(2 * (lane + 1)), d1);
                                lane += 2;
                            }
                        } else {
                            while lane < tile {
                                let mut a0 = vld1q_u64(src_a.add(2 * lane));
                                let mut a1 = vld1q_u64(src_a.add(2 * (lane + 1)));
                                let mut b0 = vld1q_u64(src_b.add(2 * lane));
                                let mut b1 = vld1q_u64(src_b.add(2 * (lane + 1)));
                                let mut c0 = vld1q_u64(src_c.add(2 * lane));
                                let mut c1 = vld1q_u64(src_c.add(2 * (lane + 1)));
                                let mut d0 = vld1q_u64(src_d.add(2 * lane));
                                let mut d1 = vld1q_u64(src_d.add(2 * (lane + 1)));
                                btf_pair_neon!(a0, a1, c0, c1, to_r0, to_r1);
                                btf_pair_neon!(b0, b1, d0, d1, to_r0, to_r1);
                                btf_pair_neon!(a0, a1, b0, b1, ti0_r0, ti0_r1);
                                btf_pair_neon!(c0, c1, d0, d1, ti1_r0, ti1_r1);
                                vst1q_u64(row_a.add(2 * lane), a0);
                                vst1q_u64(row_a.add(2 * (lane + 1)), a1);
                                vst1q_u64(row_b.add(2 * lane), b0);
                                vst1q_u64(row_b.add(2 * (lane + 1)), b1);
                                vst1q_u64(row_c.add(2 * lane), c0);
                                vst1q_u64(row_c.add(2 * (lane + 1)), c1);
                                vst1q_u64(row_d.add(2 * lane), d0);
                                vst1q_u64(row_d.add(2 * (lane + 1)), d1);
                                lane += 2;
                            }
                        }
                    }

                    for s in 0..4 {
                        let to_r0 = tw1_r0[3 + s];
                        let to_r1 = tw1_r1[3 + s];
                        let ti0_r0 = tw1_r0[7 + 2 * s];
                        let ti0_r1 = tw1_r1[7 + 2 * s];
                        let ti1_r0 = tw1_r0[8 + 2 * s];
                        let ti1_r1 = tw1_r1[8 + 2 * s];
                        let src_a = st1_ptr.add(2 * ((4 * s) * 16));
                        let src_b = st1_ptr.add(2 * ((4 * s + 1) * 16));
                        let src_c = st1_ptr.add(2 * ((4 * s + 2) * 16));
                        let src_d = st1_ptr.add(2 * ((4 * s + 3) * 16));
                        let row_a =
                            d1_ptr.add(2 * (((4 * s) * sixteenth + r) * num_ntts + lane_base));
                        let row_b =
                            d1_ptr.add(2 * (((4 * s + 1) * sixteenth + r) * num_ntts + lane_base));
                        let row_c =
                            d1_ptr.add(2 * (((4 * s + 2) * sixteenth + r) * num_ntts + lane_base));
                        let row_d =
                            d1_ptr.add(2 * (((4 * s + 3) * sixteenth + r) * num_ntts + lane_base));
                        let mut lane = 0usize;
                        while lane < tile {
                            let mut a0 = vld1q_u64(src_a.add(2 * lane));
                            let mut a1 = vld1q_u64(src_a.add(2 * (lane + 1)));
                            let mut b0 = vld1q_u64(src_b.add(2 * lane));
                            let mut b1 = vld1q_u64(src_b.add(2 * (lane + 1)));
                            let mut c0 = vld1q_u64(src_c.add(2 * lane));
                            let mut c1 = vld1q_u64(src_c.add(2 * (lane + 1)));
                            let mut d0 = vld1q_u64(src_d.add(2 * lane));
                            let mut d1 = vld1q_u64(src_d.add(2 * (lane + 1)));
                            btf_pair_neon!(a0, a1, c0, c1, to_r0, to_r1);
                            btf_pair_neon!(b0, b1, d0, d1, to_r0, to_r1);
                            btf_pair_neon!(a0, a1, b0, b1, ti0_r0, ti0_r1);
                            btf_pair_neon!(c0, c1, d0, d1, ti1_r0, ti1_r1);
                            vst1q_u64(row_a.add(2 * lane), a0);
                            vst1q_u64(row_a.add(2 * (lane + 1)), a1);
                            vst1q_u64(row_b.add(2 * lane), b0);
                            vst1q_u64(row_b.add(2 * (lane + 1)), b1);
                            vst1q_u64(row_c.add(2 * lane), c0);
                            vst1q_u64(row_c.add(2 * (lane + 1)), c1);
                            vst1q_u64(row_d.add(2 * lane), d0);
                            vst1q_u64(row_d.add(2 * (lane + 1)), d1);
                            lane += 2;
                        }
                    }
                    lane_base += tile;
                }
            }
            return;
        }

        for r in r_start..(r_start + r_count) {
            for lane in 0..num_ntts {
                let mut v = [z; 16];
                for i in 0..16 {
                    let off = 2 * ((i * sixteenth + r) * num_ntts + lane);
                    v[i] = vld1q_u64(s_ptr.add(off));
                }
                if tw0_is_zero {
                    v[8] = veorq_u64(v[8], v[0]);
                    v[9] = veorq_u64(v[9], v[1]);
                    v[10] = veorq_u64(v[10], v[2]);
                    v[11] = veorq_u64(v[11], v[3]);
                    v[12] = veorq_u64(v[12], v[4]);
                    v[13] = veorq_u64(v[13], v[5]);
                    v[14] = veorq_u64(v[14], v[6]);
                    v[15] = veorq_u64(v[15], v[7]);
                    v[4] = veorq_u64(v[4], v[0]);
                    v[5] = veorq_u64(v[5], v[1]);
                    v[6] = veorq_u64(v[6], v[2]);
                    v[7] = veorq_u64(v[7], v[3]);
                    btf2_neon!(v, tw0_r0, tw0_r1, 8, 12, 2, 9, 13, 2);
                    btf2_neon!(v, tw0_r0, tw0_r1, 10, 14, 2, 11, 15, 2);
                    v[2] = veorq_u64(v[2], v[0]);
                    v[3] = veorq_u64(v[3], v[1]);
                    btf2_neon!(v, tw0_r0, tw0_r1, 4, 6, 4, 5, 7, 4);
                    btf2_neon!(v, tw0_r0, tw0_r1, 8, 10, 5, 9, 11, 5);
                    btf2_neon!(v, tw0_r0, tw0_r1, 12, 14, 6, 13, 15, 6);
                    btf1_zero_left_neon!(v, tw0_r0, tw0_r1, 0, 1, 2, 3, 8);
                    btf2_neon!(v, tw0_r0, tw0_r1, 4, 5, 9, 6, 7, 10);
                    btf2_neon!(v, tw0_r0, tw0_r1, 8, 9, 11, 10, 11, 12);
                    btf2_neon!(v, tw0_r0, tw0_r1, 12, 13, 13, 14, 15, 14);
                } else {
                    btf2_neon!(v, tw0_r0, tw0_r1, 0, 8, 0, 1, 9, 0);
                    btf2_neon!(v, tw0_r0, tw0_r1, 2, 10, 0, 3, 11, 0);
                    btf2_neon!(v, tw0_r0, tw0_r1, 4, 12, 0, 5, 13, 0);
                    btf2_neon!(v, tw0_r0, tw0_r1, 6, 14, 0, 7, 15, 0);
                    btf2_neon!(v, tw0_r0, tw0_r1, 0, 4, 1, 1, 5, 1);
                    btf2_neon!(v, tw0_r0, tw0_r1, 2, 6, 1, 3, 7, 1);
                    btf2_neon!(v, tw0_r0, tw0_r1, 8, 12, 2, 9, 13, 2);
                    btf2_neon!(v, tw0_r0, tw0_r1, 10, 14, 2, 11, 15, 2);
                    btf2_neon!(v, tw0_r0, tw0_r1, 0, 2, 3, 1, 3, 3);
                    btf2_neon!(v, tw0_r0, tw0_r1, 4, 6, 4, 5, 7, 4);
                    btf2_neon!(v, tw0_r0, tw0_r1, 8, 10, 5, 9, 11, 5);
                    btf2_neon!(v, tw0_r0, tw0_r1, 12, 14, 6, 13, 15, 6);
                    btf2_neon!(v, tw0_r0, tw0_r1, 0, 1, 7, 2, 3, 8);
                    btf2_neon!(v, tw0_r0, tw0_r1, 4, 5, 9, 6, 7, 10);
                    btf2_neon!(v, tw0_r0, tw0_r1, 8, 9, 11, 10, 11, 12);
                    btf2_neon!(v, tw0_r0, tw0_r1, 12, 13, 13, 14, 15, 14);
                }
                for i in 0..16 {
                    let off = 2 * ((i * sixteenth + r) * num_ntts + lane);
                    vst1q_u64(d0_ptr.add(off), v[i]);
                }

                for i in 0..16 {
                    let off = 2 * ((i * sixteenth + r) * num_ntts + lane);
                    v[i] = vld1q_u64(s_ptr.add(off));
                }
                btf2_neon!(v, tw1_r0, tw1_r1, 0, 8, 0, 1, 9, 0);
                btf2_neon!(v, tw1_r0, tw1_r1, 2, 10, 0, 3, 11, 0);
                btf2_neon!(v, tw1_r0, tw1_r1, 4, 12, 0, 5, 13, 0);
                btf2_neon!(v, tw1_r0, tw1_r1, 6, 14, 0, 7, 15, 0);
                btf2_neon!(v, tw1_r0, tw1_r1, 0, 4, 1, 1, 5, 1);
                btf2_neon!(v, tw1_r0, tw1_r1, 2, 6, 1, 3, 7, 1);
                btf2_neon!(v, tw1_r0, tw1_r1, 8, 12, 2, 9, 13, 2);
                btf2_neon!(v, tw1_r0, tw1_r1, 10, 14, 2, 11, 15, 2);
                btf2_neon!(v, tw1_r0, tw1_r1, 0, 2, 3, 1, 3, 3);
                btf2_neon!(v, tw1_r0, tw1_r1, 4, 6, 4, 5, 7, 4);
                btf2_neon!(v, tw1_r0, tw1_r1, 8, 10, 5, 9, 11, 5);
                btf2_neon!(v, tw1_r0, tw1_r1, 12, 14, 6, 13, 15, 6);
                btf2_neon!(v, tw1_r0, tw1_r1, 0, 1, 7, 2, 3, 8);
                btf2_neon!(v, tw1_r0, tw1_r1, 4, 5, 9, 6, 7, 10);
                btf2_neon!(v, tw1_r0, tw1_r1, 8, 9, 11, 10, 11, 12);
                btf2_neon!(v, tw1_r0, tw1_r1, 12, 13, 13, 14, 15, 14);
                for i in 0..16 {
                    let off = 2 * ((i * sixteenth + r) * num_ntts + lane);
                    vst1q_u64(d1_ptr.add(off), v[i]);
                }
            }
        }
    }
}

#[allow(dead_code, clippy::too_many_arguments)]
#[inline]
pub(super) unsafe fn butterfly_fused_4layer_dual_from_src_rows(
    src_ptr: *const F128,
    dst0_ptr: *mut F128,
    dst1_ptr: *mut F128,
    sixteenth: usize,
    num_ntts: usize,
    r_start: usize,
    r_count: usize,
    tw0: &[F128; 15],
    tw1: &[F128; 15],
) {
    #[cfg(all(target_arch = "x86_64", target_feature = "pclmulqdq"))]
    unsafe {
        butterfly_fused_4layer_dual_from_src_rows_x86(
            src_ptr, dst0_ptr, dst1_ptr, sixteenth, num_ntts, r_start, r_count, tw0, tw1,
        );
        return;
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "aes"))]
    unsafe {
        butterfly_fused_4layer_dual_from_src_rows_neon(
            src_ptr, dst0_ptr, dst1_ptr, sixteenth, num_ntts, r_start, r_count, tw0, tw1,
        );
        return;
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    unsafe {
        for r in r_start..(r_start + r_count) {
            for lane in 0..num_ntts {
                let mut v0 = [F128::ZERO; 16];
                for (i, value) in v0.iter_mut().enumerate() {
                    *value = *src_ptr.add((i * sixteenth + r) * num_ntts + lane);
                }
                let mut v1 = v0;
                butterfly_fused_4layer(&mut v0, tw0);
                butterfly_fused_4layer(&mut v1, tw1);
                for (i, value) in v0.iter().enumerate() {
                    *dst0_ptr.add((i * sixteenth + r) * num_ntts + lane) = *value;
                }
                for (i, value) in v1.iter().enumerate() {
                    *dst1_ptr.add((i * sixteenth + r) * num_ntts + lane) = *value;
                }
            }
        }
    }
}

#[allow(dead_code, clippy::too_many_arguments)]
pub(super) unsafe fn butterfly_fused_4layer_row_from_replicated_msg(
    msg_ptr: *const F128,
    msg_len: usize,
    dst_off: usize,
    dst_ptr: *mut F128,
    sixteenth: usize,
    num_ntts: usize,
    r: usize,
    twiddles: &[F128; 15],
) {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    unsafe {
        if (dst_off & (msg_len - 1)) == 0 && 16 * sixteenth * num_ntts <= msg_len {
            let step = sixteenth * num_ntts;
            let base_r = r * num_ntts;
            for i in 0..16 {
                let off = i * step + base_r;
                core::ptr::copy_nonoverlapping(msg_ptr.add(off), dst_ptr.add(off), num_ntts);
            }
            crate::gf2_kernels::k0_ntt_butterfly_fused_4layer_row_slice(
                dst_ptr as *mut u64,
                sixteenth,
                num_ntts,
                r,
                twiddles.as_ptr() as *const u64,
            );
            return;
        }
    }
    unsafe {
        let log_msg_len = msg_len.trailing_zeros() as u64;
        for lane in 0..num_ntts {
            let mut values = [F128::ZERO; 16];
            for (i, value) in values.iter_mut().enumerate() {
                let idx = crate::gf2_kernels::gf2_kern_ntt_replicated_msg_idx(
                    (dst_off + (i * sixteenth + r) * num_ntts + lane) as u64,
                    log_msg_len,
                ) as usize;
                *value = *msg_ptr.add(idx);
            }
            butterfly_fused_4layer(&mut values, twiddles);
            for (i, value) in values.iter().enumerate() {
                *dst_ptr.add((i * sixteenth + r) * num_ntts + lane) = *value;
            }
        }
    }
}
