//! Pair transpose with a structural constant-one input specialization.

use core::arch::aarch64::*;

/// Transpose two words of eight lanes into two 64-byte rows.
/// `B_ONE` omits the B loads and ANDs; all exchanges and stores are shared.
///
/// # Safety
/// Each A lane must allow a 16-byte read at `OFF`. Unless `B_ONE`, each B
/// lane must allow the same read. The output must allow 128 bytes at `OUT`.
/// With `B_ONE`, the omitted B words must be all ones.
#[inline(always)]
pub(super) unsafe fn pair_op<const B_ONE: bool, const OFF: usize, const OUT: usize>(
    pa: *const u8,
    pb: *const u8,
    out: *mut u8,
    m33: uint8x16_t,
    m55: uint8x16_t,
    idx: [uint8x16_t; 4],
) {
    const LANE: usize = super::AB_STRIPE_WORDS * 8;
    macro_rules! emit {
        ([$($load:literal),* $(,)?], [$($extra:tt)*]) => {
            core::arch::asm!(
                $($load,)*
                "mov v24.16b, v16.16b",
                "sli v24.16b, v20.16b, #4",
                "sri v20.16b, v16.16b, #4",
                "mov v25.16b, v17.16b",
                "sli v25.16b, v21.16b, #4",
                "sri v21.16b, v17.16b, #4",
                "mov v26.16b, v18.16b",
                "sli v26.16b, v22.16b, #4",
                "sri v22.16b, v18.16b, #4",
                "mov v27.16b, v19.16b",
                "sli v27.16b, v23.16b, #4",
                "sri v23.16b, v19.16b, #4",
                "shl v16.16b, v26.16b, #2",
                "bit v16.16b, v24.16b, {m33:v}.16b",
                "ushr v24.16b, v24.16b, #2",
                "bif v24.16b, v26.16b, {m33:v}.16b",
                "shl v17.16b, v27.16b, #2",
                "bit v17.16b, v25.16b, {m33:v}.16b",
                "ushr v25.16b, v25.16b, #2",
                "bif v25.16b, v27.16b, {m33:v}.16b",
                "shl v18.16b, v22.16b, #2",
                "bit v18.16b, v20.16b, {m33:v}.16b",
                "ushr v20.16b, v20.16b, #2",
                "bif v20.16b, v22.16b, {m33:v}.16b",
                "shl v19.16b, v23.16b, #2",
                "bit v19.16b, v21.16b, {m33:v}.16b",
                "ushr v21.16b, v21.16b, #2",
                "bif v21.16b, v23.16b, {m33:v}.16b",
                "shl v26.16b, v17.16b, #1",
                "bit v26.16b, v16.16b, {m55:v}.16b",
                "ushr v16.16b, v16.16b, #1",
                "bif v16.16b, v17.16b, {m55:v}.16b",
                "shl v27.16b, v25.16b, #1",
                "bit v27.16b, v24.16b, {m55:v}.16b",
                "ushr v24.16b, v24.16b, #1",
                "bif v24.16b, v25.16b, {m55:v}.16b",
                "shl v22.16b, v19.16b, #1",
                "bit v22.16b, v18.16b, {m55:v}.16b",
                "ushr v18.16b, v18.16b, #1",
                "bif v18.16b, v19.16b, {m55:v}.16b",
                "shl v23.16b, v21.16b, #1",
                "bit v23.16b, v20.16b, {m55:v}.16b",
                "ushr v20.16b, v20.16b, #1",
                "bif v20.16b, v21.16b, {m55:v}.16b",
                "zip1 v28.2d, v26.2d, v16.2d",
                "zip1 v29.2d, v27.2d, v24.2d",
                "zip1 v30.2d, v22.2d, v18.2d",
                "zip1 v31.2d, v23.2d, v20.2d",
                "tbl v17.16b, {{v28.16b, v29.16b, v30.16b, v31.16b}}, {i0:v}.16b",
                "tbl v25.16b, {{v28.16b, v29.16b, v30.16b, v31.16b}}, {i1:v}.16b",
                "tbl v19.16b, {{v28.16b, v29.16b, v30.16b, v31.16b}}, {i2:v}.16b",
                "tbl v21.16b, {{v28.16b, v29.16b, v30.16b, v31.16b}}, {i3:v}.16b",
                "stnp q17, q25, [{out}, #{s0}]",
                "stnp q19, q21, [{out}, #{s1}]",
                "zip2 v28.2d, v26.2d, v16.2d",
                "zip2 v29.2d, v27.2d, v24.2d",
                "zip2 v30.2d, v22.2d, v18.2d",
                "zip2 v31.2d, v23.2d, v20.2d",
                "tbl v17.16b, {{v28.16b, v29.16b, v30.16b, v31.16b}}, {i0:v}.16b",
                "tbl v25.16b, {{v28.16b, v29.16b, v30.16b, v31.16b}}, {i1:v}.16b",
                "tbl v19.16b, {{v28.16b, v29.16b, v30.16b, v31.16b}}, {i2:v}.16b",
                "tbl v21.16b, {{v28.16b, v29.16b, v30.16b, v31.16b}}, {i3:v}.16b",
                "stnp q17, q25, [{out}, #{s2}]",
                "stnp q19, q21, [{out}, #{s3}]",
                pa = in(reg) pa,
                $($extra)*
                out = in(reg) out,
                o0 = const OFF + 0 * LANE,
                o1 = const OFF + 1 * LANE,
                o2 = const OFF + 2 * LANE,
                o3 = const OFF + 3 * LANE,
                o4 = const OFF + 4 * LANE,
                o5 = const OFF + 5 * LANE,
                o6 = const OFF + 6 * LANE,
                o7 = const OFF + 7 * LANE,
                s0 = const OUT,
                s1 = const OUT + 32,
                s2 = const OUT + 64,
                s3 = const OUT + 96,
                m33 = in(vreg) m33,
                m55 = in(vreg) m55,
                i0 = in(vreg) idx[0],
                i1 = in(vreg) idx[1],
                i2 = in(vreg) idx[2],
                i3 = in(vreg) idx[3],
                out("v16") _,
                out("v17") _,
                out("v18") _,
                out("v19") _,
                out("v20") _,
                out("v21") _,
                out("v22") _,
                out("v23") _,
                out("v24") _,
                out("v25") _,
                out("v26") _,
                out("v27") _,
                out("v28") _,
                out("v29") _,
                out("v30") _,
                out("v31") _,
                options(nostack, preserves_flags),
            )
        };
    }
    // The constant branch is removed during monomorphization.
    unsafe {
        if B_ONE {
            emit!([
                "ldr q16, [{pa}, #{o0}]",
                "ldr q17, [{pa}, #{o1}]",
                "ldr q18, [{pa}, #{o2}]",
                "ldr q19, [{pa}, #{o3}]",
                "ldr q20, [{pa}, #{o4}]",
                "ldr q21, [{pa}, #{o5}]",
                "ldr q22, [{pa}, #{o6}]",
                "ldr q23, [{pa}, #{o7}]",
            ], []);
        } else {
            emit!([
                "ldr q16, [{pa}, #{o0}]",
                "ldr q24, [{pb}, #{o0}]",
                "ldr q17, [{pa}, #{o1}]",
                "ldr q25, [{pb}, #{o1}]",
                "ldr q18, [{pa}, #{o2}]",
                "ldr q26, [{pb}, #{o2}]",
                "ldr q19, [{pa}, #{o3}]",
                "ldr q27, [{pb}, #{o3}]",
                "ldr q20, [{pa}, #{o4}]",
                "ldr q28, [{pb}, #{o4}]",
                "ldr q21, [{pa}, #{o5}]",
                "ldr q29, [{pb}, #{o5}]",
                "ldr q22, [{pa}, #{o6}]",
                "ldr q30, [{pb}, #{o6}]",
                "ldr q23, [{pa}, #{o7}]",
                "ldr q31, [{pb}, #{o7}]",
                "and v16.16b, v16.16b, v24.16b",
                "and v17.16b, v17.16b, v25.16b",
                "and v18.16b, v18.16b, v26.16b",
                "and v19.16b, v19.16b, v27.16b",
                "and v20.16b, v20.16b, v28.16b",
                "and v21.16b, v21.16b, v29.16b",
                "and v22.16b, v22.16b, v30.16b",
                "and v23.16b, v23.16b, v31.16b",
            ], [pb = in(reg) pb,]);
        }
    }
}
