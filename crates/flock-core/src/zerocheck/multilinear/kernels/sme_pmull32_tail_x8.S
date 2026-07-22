	.arch armv9.2-a+sme2+aes
	.section	__TEXT,__text,regular,pure_instructions

/* Four 32x32 carry-less products reconstruct one 64x64 product. */
	.macro SB a, b, bs, olo, ohi, t0, t1
	pmullb	\olo\().d, \a\().s, \b\().s
	pmullt	\ohi\().d, \a\().s, \b\().s
	pmullb	\t0\().d, \a\().s, \bs\().s
	pmullt	\t1\().d, \a\().s, \bs\().s
	eor	\t0\().d, \t0\().d, \t1\().d
	lsl	\t1\().d, \t0\().d, #32
	lsr	\t0\().d, \t0\().d, #32
	eor	\olo\().d, \olo\().d, \t1\().d
	eor	\ohi\().d, \ohi\().d, \t0\().d
	.endm

/* Carry-less multiply by the GHASH low polynomial 0x87. */
	.macro M87 a, olo, ohi, tmp
	pmullb	\olo\().d, \a\().s, z17.s
	pmullt	\ohi\().d, \a\().s, z17.s
	lsl	\tmp\().d, \ohi\().d, #32
	eor	\olo\().d, \olo\().d, \tmp\().d
	lsr	\ohi\().d, \ohi\().d, #32
	.endm

/*
 * Fold one LD4D tuple with rho.  Input is z0..z3 = even.lo, even.hi,
 * odd.lo, odd.hi.  The canonical result is left in z11,z12.  Seven scratch
 * registers suffice while all six prepared rho registers remain live.
 */
	.macro FOLD0
	eor	z2.d, z2.d, z0.d
	eor	z3.d, z3.d, z1.d
	SB	z2, z18, z19, z11, z12, z4, z5
	eor	z2.d, z2.d, z3.d
	SB	z3, z20, z21, z6, z7, z4, z5
	SB	z2, z22, z23, z8, z9, z4, z5
	eor3	z8.d, z8.d, z11.d, z6.d
	eor3	z9.d, z9.d, z12.d, z7.d
	eor	z12.d, z12.d, z8.d
	eor	z6.d, z6.d, z9.d
	M87	z7, z8, z9, z4
	M87	z6, z2, z3, z4
	pmullb	z4.d, z9.s, z17.s
	eor3	z11.d, z11.d, z2.d, z4.d
	eor3	z12.d, z12.d, z3.d, z8.d
	eor	z11.d, z11.d, z0.d
	eor	z12.d, z12.d, z1.d
	.endm

/* Same fold graph, with z9,z10 as output while z11,z12 stay live. */
	.macro FOLD1_KEEP_RHO
	eor	z2.d, z2.d, z0.d
	eor	z3.d, z3.d, z1.d
	SB	z2, z18, z19, z9, z10, z4, z5
	eor	z2.d, z2.d, z3.d
	SB	z3, z20, z21, z6, z7, z4, z5
	SB	z2, z22, z23, z8, z3, z4, z5
	eor3	z8.d, z8.d, z9.d, z6.d
	eor3	z3.d, z3.d, z10.d, z7.d
	eor	z10.d, z10.d, z8.d
	eor	z6.d, z6.d, z3.d
	M87	z7, z8, z3, z4
	M87	z6, z2, z5, z4
	pmullb	z4.d, z3.s, z17.s
	eor3	z9.d, z9.d, z2.d, z4.d
	eor3	z10.d, z10.d, z5.d, z8.d
	eor	z9.d, z9.d, z0.d
	eor	z10.d, z10.d, z1.d
	.endm

/*
 * Last rho fold in the packet.  Reuse rho lo/swap after d0 and rho hi/swap
 * after d2.  z22,z23 remain intact; z18..z21 are restored after messages.
 */
	.macro FOLD1_LAST_RHO
	eor	z2.d, z2.d, z0.d
	eor	z3.d, z3.d, z1.d
	SB	z2, z18, z19, z9, z10, z4, z5
	eor	z2.d, z2.d, z3.d
	SB	z3, z20, z21, z18, z19, z4, z5
	SB	z2, z22, z23, z20, z21, z4, z5
	eor3	z20.d, z20.d, z9.d, z18.d
	eor3	z21.d, z21.d, z10.d, z19.d
	eor	z10.d, z10.d, z20.d
	eor	z18.d, z18.d, z21.d
	M87	z19, z20, z21, z6
	M87	z18, z2, z3, z6
	pmullb	z6.d, z21.s, z17.s
	eor3	z9.d, z9.d, z2.d, z6.d
	eor3	z10.d, z10.d, z3.d, z20.d
	eor	z9.d, z9.d, z0.d
	eor	z10.d, z10.d, z1.d
	.endm

/* Reduced g1 = Aodd*Bodd.  Inputs z15,z16 and z2,z3; output z8,z9. */
	.macro MUL_G1
	revw	z4.d, p0/m, z2.d
	revw	z5.d, p0/m, z3.d
	eor	z6.d, z2.d, z3.d
	revw	z7.d, p0/m, z6.d
	SB	z15, z2, z4, z8, z9, z10, z11
	eor	z15.d, z15.d, z16.d
	SB	z16, z3, z5, z12, z18, z10, z11
	SB	z15, z6, z7, z19, z16, z10, z11
	eor3	z19.d, z19.d, z8.d, z12.d
	eor3	z16.d, z16.d, z9.d, z18.d
	eor	z9.d, z9.d, z19.d
	eor	z12.d, z12.d, z16.d
	M87	z18, z19, z16, z10
	M87	z12, z15, z11, z10
	pmullb	z10.d, z16.s, z17.s
	eor3	z8.d, z8.d, z15.d, z10.d
	eor3	z9.d, z9.d, z11.d, z19.d
	.endm

/* Unreduced g1*eq into p1 z24..z27; shared eq prep is z2..z7. */
	.macro WEIGHT_G1
	SB	z8, z2, z4, z10, z11, z12, z15
	eor	z8.d, z8.d, z9.d
	SB	z9, z3, z5, z16, z18, z12, z15
	SB	z8, z6, z7, z19, z9, z12, z15
	eor3	z19.d, z19.d, z10.d, z16.d
	eor3	z9.d, z9.d, z11.d, z18.d
	eor	z24.d, z24.d, z10.d
	eor3	z25.d, z25.d, z11.d, z19.d
	eor3	z26.d, z26.d, z16.d, z9.d
	eor	z27.d, z27.d, z18.d
	.endm

/* Reduced ginf = (A0+A1)*(B0+B1); output z12,z15. */
	.macro MUL_GINF
	revw	z8.d, p0/m, z0.d
	revw	z9.d, p0/m, z1.d
	eor	z10.d, z0.d, z1.d
	revw	z11.d, p0/m, z10.d
	SB	z13, z0, z8, z12, z15, z16, z18
	eor	z13.d, z13.d, z14.d
	SB	z14, z1, z9, z19, z20, z16, z18
	SB	z13, z10, z11, z21, z14, z16, z18
	eor3	z21.d, z21.d, z12.d, z19.d
	eor3	z14.d, z14.d, z15.d, z20.d
	eor	z15.d, z15.d, z21.d
	eor	z19.d, z19.d, z14.d
	M87	z20, z21, z14, z16
	M87	z19, z13, z18, z16
	pmullb	z16.d, z14.s, z17.s
	eor3	z12.d, z12.d, z13.d, z16.d
	eor3	z15.d, z15.d, z18.d, z21.d
	.endm

/* Unreduced ginf*eq into pinf z28..z31; shared eq prep is z2..z7. */
	.macro WEIGHT_GINF
	SB	z12, z2, z4, z8, z9, z10, z11
	eor	z12.d, z12.d, z15.d
	SB	z15, z3, z5, z13, z14, z10, z11
	SB	z12, z6, z7, z16, z15, z10, z11
	eor3	z16.d, z16.d, z8.d, z13.d
	eor3	z15.d, z15.d, z9.d, z14.d
	eor	z28.d, z28.d, z8.d
	eor3	z29.d, z29.d, z9.d, z16.d
	eor3	z30.d, z30.d, z13.d, z15.d
	eor	z31.d, z31.d, z14.d
	.endm

	.globl	_flock_sme_pmull32_tail_svl_bytes
	.p2align	2
_flock_sme_pmull32_tail_svl_bytes:
	stp	d15, d14, [sp, #-64]!
	stp	d13, d12, [sp, #16]
	stp	d11, d10, [sp, #32]
	stp	d9, d8, [sp, #48]
	smstart	sm
	rdvl	x0, #1
	smstop	sm
	ldp	d9, d8, [sp, #48]
	ldp	d11, d10, [sp, #32]
	ldp	d13, d12, [sp, #16]
	ldp	d15, d14, [sp], #64
	ret

	.globl	_flock_sme_pmull32_tail_x8
	.p2align	2
_flock_sme_pmull32_tail_x8:
	stp	d15, d14, [sp, #-64]!
	stp	d13, d12, [sp, #16]
	stp	d11, d10, [sp, #32]
	stp	d9, d8, [sp, #48]
	smstart	sm
	ptrue	p0.d
	ld1rd	{ z18.d }, p0/z, [x6]
	ld1rd	{ z19.d }, p0/z, [x6, #8]
	ld1rd	{ z20.d }, p0/z, [x6, #16]
	ld1rd	{ z21.d }, p0/z, [x6, #24]
	ld1rd	{ z22.d }, p0/z, [x6, #32]
	ld1rd	{ z23.d }, p0/z, [x6, #40]
	ptrue	p1.d, vl1
	add	x8, x7, #32
	ld1d	{ z24.d }, p1/z, [x8]
	add	x8, x8, #8
	ld1d	{ z25.d }, p1/z, [x8]
	add	x8, x8, #8
	ld1d	{ z26.d }, p1/z, [x8]
	add	x8, x8, #8
	ld1d	{ z27.d }, p1/z, [x8]
	add	x8, x8, #8
	ld1d	{ z28.d }, p1/z, [x8]
	add	x8, x8, #8
	ld1d	{ z29.d }, p1/z, [x8]
	add	x8, x8, #8
	ld1d	{ z30.d }, p1/z, [x8]
	add	x8, x8, #8
	ld1d	{ z31.d }, p1/z, [x8]
	mov	w8, #135
	mov	z17.s, w8

Lflock_sme_pmull32_tail_x8_loop:
	ld4d	{ z0.d - z3.d }, p0/z, [x2]
	FOLD0
	ld4d	{ z0.d - z3.d }, p0/z, [x2, #4, mul vl]
	FOLD1_KEEP_RHO
	st2d	{ z11.d, z12.d }, p0, [x0]
	st2d	{ z9.d, z10.d }, p0, [x0, #2, mul vl]
	uzp1	z13.d, z11.d, z9.d
	uzp1	z14.d, z12.d, z10.d
	uzp2	z15.d, z11.d, z9.d
	uzp2	z16.d, z12.d, z10.d

	ld4d	{ z0.d - z3.d }, p0/z, [x3]
	FOLD0
	ld4d	{ z0.d - z3.d }, p0/z, [x3, #4, mul vl]
	FOLD1_LAST_RHO
	st2d	{ z11.d, z12.d }, p0, [x1]
	st2d	{ z9.d, z10.d }, p0, [x1, #2, mul vl]
	uzp1	z0.d, z11.d, z9.d
	uzp1	z1.d, z12.d, z10.d
	uzp2	z2.d, z11.d, z9.d
	uzp2	z3.d, z12.d, z10.d

	/* Convert endpoint pairs to the two message multiplicands. */
	eor	z13.d, z13.d, z15.d
	eor	z14.d, z14.d, z16.d
	eor	z0.d, z0.d, z2.d
	eor	z1.d, z1.d, z3.d
	MUL_G1

	ld2d	{ z2.d, z3.d }, p0/z, [x4]
	revw	z4.d, p0/m, z2.d
	revw	z5.d, p0/m, z3.d
	eor	z6.d, z2.d, z3.d
	revw	z7.d, p0/m, z6.d
	WEIGHT_G1
	MUL_GINF
	WEIGHT_GINF

	/* The last fold borrowed rho lo/hi and their REVW forms; restore them. */
	ld1rd	{ z18.d }, p0/z, [x6]
	ld1rd	{ z19.d }, p0/z, [x6, #8]
	ld1rd	{ z20.d }, p0/z, [x6, #16]
	ld1rd	{ z21.d }, p0/z, [x6, #24]
	add	x2, x2, #512
	add	x3, x3, #512
	add	x4, x4, #128
	add	x0, x0, #256
	add	x1, x1, #256
	subs	x5, x5, #8
	b.ne	Lflock_sme_pmull32_tail_x8_loop

	/* Preserve the exact deferred state before the two linear reductions. */
	eorv	d0, p0, z24.d
	eorv	d1, p0, z25.d
	stp	d0, d1, [x7, #32]
	eorv	d0, p0, z26.d
	eorv	d1, p0, z27.d
	stp	d0, d1, [x7, #48]
	eorv	d0, p0, z28.d
	eorv	d1, p0, z29.d
	stp	d0, d1, [x7, #64]
	eorv	d0, p0, z30.d
	eorv	d1, p0, z31.d
	stp	d0, d1, [x7, #80]

	M87	z27, z0, z1, z2
	M87	z26, z3, z4, z2
	pmullb	z2.d, z1.s, z17.s
	eor3	z24.d, z24.d, z3.d, z2.d
	eor3	z25.d, z25.d, z4.d, z0.d
	eorv	d0, p0, z24.d
	eorv	d1, p0, z25.d
	stp	d0, d1, [x7]

	M87	z31, z0, z1, z2
	M87	z30, z3, z4, z2
	pmullb	z2.d, z1.s, z17.s
	eor3	z28.d, z28.d, z3.d, z2.d
	eor3	z29.d, z29.d, z4.d, z0.d
	eorv	d0, p0, z28.d
	eorv	d1, p0, z29.d
	stp	d0, d1, [x7, #16]

	smstop	sm
	ldp	d9, d8, [sp, #48]
	ldp	d11, d10, [sp, #32]
	ldp	d13, d12, [sp, #16]
	ldp	d15, d14, [sp], #64
	ret

