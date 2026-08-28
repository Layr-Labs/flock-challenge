	.section	__TEXT,__text,regular,pure_instructions
	.private_extern	_flock_blake3_hash8_neon_1024 ; -- Begin function flock_blake3_hash8_neon_1024
	.globl	_flock_blake3_hash8_neon_1024
	.p2align	2
_flock_blake3_hash8_neon_1024:          ; @flock_blake3_hash8_neon_1024
	.cfi_startproc
; %bb.0:
	cbz	x2, LBB0_8
; %bb.1:
	sub	sp, sp, #448
	stp	d15, d14, [sp, #384]            ; 16-byte Folded Spill
	stp	d13, d12, [sp, #400]            ; 16-byte Folded Spill
	stp	d11, d10, [sp, #416]            ; 16-byte Folded Spill
	stp	d9, d8, [sp, #432]              ; 16-byte Folded Spill
	.cfi_def_cfa_offset 448
	.cfi_offset b8, -8
	.cfi_offset b9, -16
	.cfi_offset b10, -24
	.cfi_offset b11, -32
	.cfi_offset b12, -40
	.cfi_offset b13, -48
	.cfi_offset b14, -56
	.cfi_offset b15, -64
	mov	x12, x0
	mov	w16, #2                         ; =0x2
	mov	w9, #52505                      ; =0xcd19
	movk	w9, #23520, lsl #16
	dup.4s	v1, w9
	mov	w9, #55723                      ; =0xd9ab
	movk	w9, #8067, lsl #16
	dup.4s	v0, w9
	stp	q0, q1, [sp, #32]               ; 32-byte Folded Spill
	mov	w9, #26764                      ; =0x688c
	movk	w9, #39685, lsl #16
	dup.4s	v1, w9
	mov	w9, #21119                      ; =0x527f
	movk	w9, #20750, lsl #16
	dup.4s	v0, w9
	stp	q0, q1, [sp]                    ; 32-byte Folded Spill
	mov	w9, #62778                      ; =0xf53a
	movk	w9, #42319, lsl #16
	dup.4s	v18, w9
	mov	w9, #62322                      ; =0xf372
	movk	w9, #15470, lsl #16
	dup.4s	v17, w9
	mov	w9, #44677                      ; =0xae85
	movk	w9, #47975, lsl #16
	dup.4s	v8, w9
	mov	w9, #58983                      ; =0xe667
	movk	w9, #27145, lsl #16
	dup.4s	v7, w9
	stp	q8, q17, [sp, #64]              ; 32-byte Folded Spill
	stp	q18, q7, [sp, #96]              ; 32-byte Folded Spill
LBB0_2:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_3 Depth 2
                                        ;       Child Loop BB0_4 Depth 3
	mov	w11, #2                         ; =0x2
LBB0_3:                                 ;   Parent Loop BB0_2 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_4 Depth 3
	mov	w14, #16                        ; =0x10
	mov.16b	v15, v8
	mov.16b	v12, v17
	ldp	q14, q13, [sp]                  ; 32-byte Folded Reload
	ldp	q11, q9, [sp, #32]              ; 32-byte Folded Reload
LBB0_4:                                 ;   Parent Loop BB0_2 Depth=1
                                        ;     Parent Loop BB0_3 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	mov.16b	v20, v17
	ldr	q0, [x12, #1024]
	ldr	q1, [x12, #2048]
	ldr	q2, [x12, #3072]
	ldp	q3, q4, [x12]
	trn1.4s	v26, v3, v0
	trn2.4s	v25, v3, v0
	trn1.4s	v0, v1, v2
	trn2.4s	v1, v1, v2
	zip2.2d	v10, v26, v0
	zip2.2d	v17, v25, v1
	str	q17, [sp, #352]                 ; 16-byte Folded Spill
	ldr	q2, [x12, #1040]
	ldr	q3, [x12, #2064]
	ldr	q5, [x12, #3088]
	trn1.4s	v31, v4, v2
	trn2.4s	v27, v4, v2
	trn1.4s	v2, v3, v5
	trn2.4s	v3, v3, v5
	zip2.2d	v22, v31, v2
	zip2.2d	v19, v27, v3
	ldr	q4, [x12, #1056]
	ldr	q5, [x12, #2080]
	ldr	q6, [x12, #3104]
	mov.d	v26[1], v0[0]
	str	q26, [sp, #240]                 ; 16-byte Folded Spill
	ldp	q0, q16, [x12, #32]
	trn1.4s	v28, v0, v4
	trn2.4s	v29, v0, v4
	mov.d	v25[1], v1[0]
	trn1.4s	v0, v5, v6
	trn2.4s	v1, v5, v6
	zip2.2d	v23, v28, v0
	mov.d	v31[1], v2[0]
	str	q31, [sp, #336]                 ; 16-byte Folded Spill
	zip2.2d	v30, v29, v1
	ldr	q2, [x12, #1072]
	ldr	q4, [x12, #2096]
	mov.d	v27[1], v3[0]
	ldr	q3, [x12, #3120]
	trn1.4s	v24, v16, v2
	trn2.4s	v21, v16, v2
	trn1.4s	v2, v4, v3
	mov.d	v28[1], v0[0]
	trn2.4s	v3, v4, v3
	zip2.2d	v0, v24, v2
	stp	q0, q25, [sp, #288]             ; 32-byte Folded Spill
	zip2.2d	v0, v21, v3
	str	q0, [sp, #192]                  ; 16-byte Folded Spill
	mov.d	v29[1], v1[0]
	str	q29, [sp, #256]                 ; 16-byte Folded Spill
	subs	x14, x14, #1
	csel	w15, w16, wzr, eq
	cmp	x14, #15
	mov.d	v24[1], v2[0]
	cinc	w15, w15, eq
	dup.4s	v0, w15
	mov.d	v21[1], v3[0]
	add.4s	v1, v7, v14
	add.4s	v1, v1, v26
	rev32.8h	v2, v1
	ldr	q3, [sp, #112]                  ; 16-byte Folded Reload
	add.4s	v4, v2, v3
	eor.16b	v3, v4, v14
	shl.4s	v5, v3, #20
	sri.4s	v5, v3, #12
	add.4s	v1, v1, v25
	add.4s	v1, v1, v5
	add.4s	v3, v15, v13
	add.4s	v6, v3, v10
	mov.16b	v25, v10
	eor.16b	v2, v1, v2
	rev32.8h	v7, v6
	add.4s	v14, v7, v8
	eor.16b	v3, v14, v13
	shl.4s	v10, v3, #20
	sri.4s	v10, v3, #12
	shl.4s	v3, v2, #24
	add.4s	v6, v6, v17
	add.4s	v16, v6, v10
	eor.16b	v6, v16, v7
	shl.4s	v15, v6, #24
	sri.4s	v15, v6, #8
	sri.4s	v3, v2, #8
	add.4s	v13, v15, v14
	eor.16b	v2, v13, v10
	shl.4s	v6, v2, #25
	add.4s	v7, v12, v11
	add.4s	v7, v7, v31
	sri.4s	v6, v2, #7
	movi.4s	v2, #64
	eor.16b	v2, v7, v2
	rev32.8h	v10, v2
	add.4s	v12, v10, v20
	eor.16b	v11, v12, v11
	shl.4s	v17, v11, #20
	add.4s	v2, v4, v3
	sri.4s	v17, v11, #12
	add.4s	v4, v7, v27
	mov.16b	v8, v27
	add.4s	v4, v4, v17
	eor.16b	v7, v4, v10
	shl.4s	v11, v7, #24
	eor.16b	v5, v2, v5
	sri.4s	v11, v7, #8
	shl.4s	v14, v5, #25
	add.4s	v7, v18, v9
	mov.16b	v27, v22
	str	q22, [sp, #176]                 ; 16-byte Folded Spill
	add.4s	v7, v7, v22
	eor.16b	v0, v7, v0
	sri.4s	v14, v5, #7
	rev32.8h	v0, v0
	ldr	q5, [sp, #96]                   ; 16-byte Folded Reload
	add.4s	v5, v0, v5
	eor.16b	v18, v5, v9
	shl.4s	v10, v18, #20
	sri.4s	v10, v18, #12
	add.4s	v7, v7, v19
	mov.16b	v31, v19
	str	q19, [sp, #272]                 ; 16-byte Folded Spill
	add.4s	v9, v7, v10
	eor.16b	v0, v9, v0
	shl.4s	v7, v0, #24
	sri.4s	v7, v0, #8
	add.4s	v0, v7, v5
	eor.16b	v5, v0, v10
	shl.4s	v18, v5, #25
	sri.4s	v18, v5, #7
	add.4s	v5, v11, v12
	eor.16b	v17, v5, v17
	shl.4s	v12, v17, #25
	sri.4s	v12, v17, #7
	mov.16b	v26, v28
	str	q28, [sp, #128]                 ; 16-byte Folded Spill
	add.4s	v1, v1, v28
	add.4s	v1, v1, v6
	eor.16b	v7, v7, v1
	rev32.8h	v7, v7
	add.4s	v17, v5, v7
	eor.16b	v5, v17, v6
	shl.4s	v6, v5, #20
	sri.4s	v6, v5, #12
	add.4s	v1, v1, v29
	add.4s	v5, v1, v6
	eor.16b	v1, v5, v7
	shl.4s	v7, v1, #24
	sri.4s	v7, v1, #8
	add.4s	v10, v17, v7
	eor.16b	v6, v10, v6
	shl.4s	v1, v6, #25
	sri.4s	v1, v6, #7
	mov.16b	v22, v23
	stp	q30, q23, [sp, #208]            ; 32-byte Folded Spill
	add.4s	v6, v16, v23
	add.4s	v6, v6, v12
	eor.16b	v3, v6, v3
	rev32.8h	v3, v3
	add.4s	v0, v0, v3
	eor.16b	v16, v0, v12
	shl.4s	v17, v16, #20
	sri.4s	v17, v16, #12
	mov.16b	v23, v30
	add.4s	v6, v6, v30
	add.4s	v6, v6, v17
	eor.16b	v3, v6, v3
	shl.4s	v16, v3, #24
	sri.4s	v16, v3, #8
	add.4s	v3, v4, v24
	add.4s	v4, v3, v18
	eor.16b	v3, v4, v15
	rev32.8h	v15, v3
	add.4s	v2, v2, v15
	eor.16b	v3, v2, v18
	shl.4s	v18, v3, #20
	sri.4s	v18, v3, #12
	add.4s	v12, v16, v0
	eor.16b	v0, v12, v17
	shl.4s	v3, v0, #25
	sri.4s	v3, v0, #7
	add.4s	v0, v4, v21
	add.4s	v19, v0, v18
	eor.16b	v4, v19, v15
	shl.4s	v17, v4, #24
	sri.4s	v17, v4, #8
	add.4s	v4, v17, v2
	eor.16b	v18, v4, v18
	shl.4s	v2, v18, #25
	sri.4s	v2, v18, #7
	ldr	q28, [sp, #288]                 ; 16-byte Folded Reload
	add.4s	v18, v28, v9
	add.4s	v18, v14, v18
	eor.16b	v9, v18, v11
	rev32.8h	v11, v9
	add.4s	v13, v13, v11
	eor.16b	v9, v13, v14
	shl.4s	v14, v9, #20
	sri.4s	v14, v9, #12
	ldr	q30, [sp, #192]                 ; 16-byte Folded Reload
	add.4s	v18, v18, v30
	add.4s	v9, v18, v14
	eor.16b	v18, v9, v11
	shl.4s	v0, v18, #24
	sri.4s	v0, v18, #8
	add.4s	v18, v0, v13
	eor.16b	v11, v18, v14
	shl.4s	v13, v11, #25
	sri.4s	v13, v11, #7
	mov.16b	v20, v25
	str	q25, [sp, #160]                 ; 16-byte Folded Spill
	add.4s	v5, v5, v25
	add.4s	v5, v5, v13
	eor.16b	v16, v5, v16
	rev32.8h	v11, v16
	add.4s	v4, v4, v11
	eor.16b	v16, v4, v13
	shl.4s	v13, v16, #20
	sri.4s	v13, v16, #12
	add.4s	v5, v27, v5
	add.4s	v16, v13, v5
	eor.16b	v5, v16, v11
	shl.4s	v14, v5, #24
	sri.4s	v14, v5, #8
	ldr	q25, [sp, #352]                 ; 16-byte Folded Reload
	add.4s	v5, v25, v6
	add.4s	v5, v1, v5
	eor.16b	v6, v17, v5
	rev32.8h	v17, v6
	add.4s	v18, v18, v17
	eor.16b	v1, v18, v1
	shl.4s	v11, v1, #20
	sri.4s	v11, v1, #12
	add.4s	v1, v5, v22
	add.4s	v6, v1, v11
	eor.16b	v1, v6, v17
	shl.4s	v15, v1, #24
	sri.4s	v15, v1, #8
	add.4s	v5, v15, v18
	eor.16b	v1, v5, v11
	shl.4s	v17, v1, #25
	sri.4s	v17, v1, #7
	add.4s	v1, v4, v14
	eor.16b	v4, v1, v13
	shl.4s	v11, v4, #25
	sri.4s	v11, v4, #7
	add.4s	v4, v31, v19
	add.4s	v4, v3, v4
	eor.16b	v0, v0, v4
	rev32.8h	v18, v0
	add.4s	v19, v10, v18
	eor.16b	v0, v19, v3
	shl.4s	v3, v0, #20
	sri.4s	v3, v0, #12
	ldr	q31, [sp, #240]                 ; 16-byte Folded Reload
	add.4s	v0, v4, v31
	add.4s	v0, v0, v3
	eor.16b	v4, v0, v18
	shl.4s	v13, v4, #24
	sri.4s	v13, v4, #8
	ldr	q22, [sp, #336]                 ; 16-byte Folded Reload
	add.4s	v4, v22, v9
	add.4s	v4, v2, v4
	eor.16b	v7, v4, v7
	rev32.8h	v7, v7
	add.4s	v18, v12, v7
	eor.16b	v2, v18, v2
	shl.4s	v9, v2, #20
	sri.4s	v9, v2, #12
	add.4s	v2, v4, v21
	str	q21, [sp, #368]                 ; 16-byte Folded Spill
	add.4s	v2, v2, v9
	eor.16b	v4, v2, v7
	shl.4s	v7, v4, #24
	sri.4s	v7, v4, #8
	add.4s	v4, v13, v19
	eor.16b	v3, v4, v3
	shl.4s	v19, v3, #25
	sri.4s	v19, v3, #7
	add.4s	v18, v7, v18
	eor.16b	v3, v18, v9
	shl.4s	v9, v3, #25
	sri.4s	v9, v3, #7
	ldr	q3, [sp, #304]                  ; 16-byte Folded Reload
	add.4s	v3, v16, v3
	add.4s	v3, v3, v17
	eor.16b	v7, v7, v3
	rev32.8h	v7, v7
	add.4s	v4, v4, v7
	eor.16b	v16, v4, v17
	shl.4s	v17, v16, #20
	sri.4s	v17, v16, #12
	add.4s	v3, v3, v23
	add.4s	v3, v3, v17
	eor.16b	v16, v3, v7
	shl.4s	v7, v16, #24
	sri.4s	v7, v16, #8
	add.4s	v10, v4, v7
	eor.16b	v4, v10, v17
	shl.4s	v12, v4, #25
	sri.4s	v12, v4, #7
	mov.16b	v29, v24
	add.4s	v4, v6, v24
	add.4s	v4, v4, v19
	eor.16b	v6, v4, v14
	rev32.8h	v6, v6
	add.4s	v16, v18, v6
	eor.16b	v17, v16, v19
	shl.4s	v18, v17, #20
	sri.4s	v18, v17, #12
	ldr	q27, [sp, #256]                 ; 16-byte Folded Reload
	add.4s	v0, v0, v27
	add.4s	v0, v0, v9
	eor.16b	v17, v0, v15
	rev32.8h	v17, v17
	add.4s	v1, v1, v17
	eor.16b	v19, v1, v9
	shl.4s	v9, v19, #20
	sri.4s	v9, v19, #12
	mov.16b	v24, v8
	str	q8, [sp, #144]                  ; 16-byte Folded Spill
	add.4s	v4, v4, v8
	add.4s	v4, v4, v18
	eor.16b	v6, v4, v6
	shl.4s	v19, v6, #24
	sri.4s	v19, v6, #8
	add.4s	v0, v0, v28
	add.4s	v0, v0, v9
	eor.16b	v6, v0, v17
	shl.4s	v17, v6, #24
	sri.4s	v17, v6, #8
	add.4s	v6, v17, v1
	eor.16b	v1, v6, v9
	shl.4s	v14, v1, #25
	sri.4s	v14, v1, #7
	add.4s	v1, v30, v2
	mov.16b	v8, v30
	add.4s	v1, v11, v1
	eor.16b	v2, v1, v13
	rev32.8h	v2, v2
	add.4s	v5, v5, v2
	eor.16b	v9, v5, v11
	shl.4s	v11, v9, #20
	sri.4s	v11, v9, #12
	add.4s	v1, v1, v26
	add.4s	v1, v1, v11
	eor.16b	v2, v1, v2
	shl.4s	v15, v2, #24
	sri.4s	v15, v2, #8
	add.4s	v2, v15, v5
	eor.16b	v5, v2, v11
	shl.4s	v9, v5, #25
	sri.4s	v9, v5, #7
	add.4s	v5, v19, v16
	eor.16b	v16, v5, v18
	shl.4s	v18, v16, #25
	sri.4s	v18, v16, #7
	add.4s	v3, v3, v25
	mov.16b	v26, v25
	add.4s	v3, v3, v9
	eor.16b	v16, v3, v19
	rev32.8h	v16, v16
	add.4s	v6, v6, v16
	eor.16b	v19, v6, v9
	shl.4s	v11, v19, #20
	sri.4s	v11, v19, #12
	add.4s	v3, v22, v3
	add.4s	v3, v11, v3
	eor.16b	v16, v3, v16
	shl.4s	v13, v16, #24
	sri.4s	v13, v16, #8
	ldr	q23, [sp, #224]                 ; 16-byte Folded Reload
	add.4s	v4, v23, v4
	add.4s	v4, v12, v4
	eor.16b	v16, v17, v4
	rev32.8h	v16, v16
	add.4s	v2, v2, v16
	eor.16b	v17, v2, v12
	shl.4s	v19, v17, #20
	sri.4s	v19, v17, #12
	add.4s	v4, v4, v29
	add.4s	v4, v4, v19
	eor.16b	v16, v4, v16
	shl.4s	v12, v16, #24
	sri.4s	v12, v16, #8
	add.4s	v9, v12, v2
	eor.16b	v2, v9, v19
	shl.4s	v16, v2, #25
	sri.4s	v16, v2, #7
	add.4s	v0, v21, v0
	add.4s	v0, v18, v0
	eor.16b	v2, v15, v0
	rev32.8h	v17, v2
	add.4s	v19, v10, v17
	eor.16b	v2, v19, v18
	shl.4s	v18, v2, #20
	sri.4s	v18, v2, #12
	add.4s	v2, v6, v13
	eor.16b	v6, v2, v11
	shl.4s	v10, v6, #25
	sri.4s	v10, v6, #7
	add.4s	v0, v0, v20
	add.4s	v6, v0, v18
	eor.16b	v0, v6, v17
	shl.4s	v15, v0, #24
	sri.4s	v15, v0, #8
	add.4s	v0, v15, v19
	eor.16b	v17, v0, v18
	shl.4s	v18, v17, #25
	sri.4s	v18, v17, #7
	ldr	q21, [sp, #272]                 ; 16-byte Folded Reload
	add.4s	v1, v21, v1
	add.4s	v1, v14, v1
	eor.16b	v7, v1, v7
	rev32.8h	v7, v7
	add.4s	v17, v5, v7
	eor.16b	v5, v17, v14
	shl.4s	v19, v5, #20
	sri.4s	v19, v5, #12
	add.4s	v1, v1, v28
	add.4s	v5, v1, v19
	eor.16b	v1, v5, v7
	shl.4s	v7, v1, #24
	sri.4s	v7, v1, #8
	add.4s	v17, v7, v17
	eor.16b	v1, v17, v19
	shl.4s	v19, v1, #25
	sri.4s	v19, v1, #7
	ldr	q30, [sp, #176]                 ; 16-byte Folded Reload
	add.4s	v1, v3, v30
	add.4s	v1, v1, v16
	eor.16b	v3, v7, v1
	rev32.8h	v3, v3
	add.4s	v11, v0, v3
	eor.16b	v0, v11, v16
	shl.4s	v16, v0, #20
	sri.4s	v16, v0, #12
	add.4s	v0, v1, v24
	add.4s	v0, v0, v16
	eor.16b	v1, v0, v3
	shl.4s	v7, v1, #24
	sri.4s	v7, v1, #8
	add.4s	v11, v11, v7
	eor.16b	v3, v11, v16
	shl.4s	v1, v3, #25
	sri.4s	v1, v3, #7
	add.4s	v3, v4, v27
	add.4s	v3, v3, v18
	eor.16b	v4, v3, v13
	rev32.8h	v4, v4
	add.4s	v16, v17, v4
	eor.16b	v17, v16, v18
	shl.4s	v18, v17, #20
	sri.4s	v18, v17, #12
	mov.16b	v24, v31
	add.4s	v3, v3, v31
	add.4s	v3, v3, v18
	eor.16b	v4, v3, v4
	shl.4s	v17, v4, #24
	sri.4s	v17, v4, #8
	ldr	q25, [sp, #208]                 ; 16-byte Folded Reload
	add.4s	v4, v6, v25
	add.4s	v4, v4, v19
	eor.16b	v6, v4, v12
	rev32.8h	v6, v6
	add.4s	v2, v2, v6
	eor.16b	v19, v2, v19
	shl.4s	v12, v19, #20
	sri.4s	v12, v19, #12
	add.4s	v4, v4, v8
	add.4s	v4, v4, v12
	eor.16b	v6, v4, v6
	shl.4s	v19, v6, #24
	sri.4s	v19, v6, #8
	add.4s	v6, v19, v2
	eor.16b	v12, v6, v12
	shl.4s	v2, v12, #25
	sri.4s	v2, v12, #7
	ldr	q22, [sp, #128]                 ; 16-byte Folded Reload
	add.4s	v5, v22, v5
	add.4s	v5, v10, v5
	eor.16b	v12, v5, v15
	rev32.8h	v12, v12
	add.4s	v9, v9, v12
	eor.16b	v10, v9, v10
	shl.4s	v13, v10, #20
	sri.4s	v13, v10, #12
	ldr	q20, [sp, #304]                 ; 16-byte Folded Reload
	add.4s	v5, v5, v20
	add.4s	v5, v5, v13
	eor.16b	v10, v5, v12
	shl.4s	v14, v10, #24
	sri.4s	v14, v10, #8
	add.4s	v16, v17, v16
	eor.16b	v18, v16, v18
	shl.4s	v10, v18, #25
	sri.4s	v10, v18, #7
	add.4s	v18, v14, v9
	eor.16b	v9, v18, v13
	shl.4s	v12, v9, #25
	sri.4s	v12, v9, #7
	add.4s	v0, v0, v23
	add.4s	v0, v0, v12
	eor.16b	v17, v0, v17
	rev32.8h	v17, v17
	add.4s	v6, v6, v17
	eor.16b	v9, v6, v12
	shl.4s	v15, v9, #20
	sri.4s	v15, v9, #12
	add.4s	v0, v21, v0
	add.4s	v0, v15, v0
	eor.16b	v17, v0, v17
	shl.4s	v13, v17, #24
	sri.4s	v13, v17, #8
	add.4s	v12, v6, v13
	eor.16b	v6, v12, v15
	shl.4s	v9, v6, #25
	sri.4s	v9, v6, #7
	str	q29, [sp, #320]                 ; 16-byte Folded Spill
	add.4s	v3, v29, v3
	add.4s	v3, v1, v3
	eor.16b	v6, v19, v3
	rev32.8h	v6, v6
	add.4s	v17, v18, v6
	eor.16b	v1, v17, v1
	shl.4s	v18, v1, #20
	sri.4s	v18, v1, #12
	add.4s	v1, v3, v27
	mov.16b	v23, v27
	add.4s	v3, v1, v18
	eor.16b	v1, v3, v6
	shl.4s	v15, v1, #24
	sri.4s	v15, v1, #8
	add.4s	v17, v15, v17
	eor.16b	v1, v17, v18
	shl.4s	v18, v1, #25
	sri.4s	v18, v1, #7
	add.4s	v1, v28, v4
	mov.16b	v21, v28
	add.4s	v1, v10, v1
	eor.16b	v4, v14, v1
	rev32.8h	v4, v4
	add.4s	v19, v11, v4
	eor.16b	v6, v19, v10
	shl.4s	v10, v6, #20
	sri.4s	v10, v6, #12
	add.4s	v1, v1, v26
	add.4s	v6, v1, v10
	eor.16b	v1, v6, v4
	shl.4s	v14, v1, #24
	sri.4s	v14, v1, #8
	ldr	q31, [sp, #368]                 ; 16-byte Folded Reload
	add.4s	v1, v31, v5
	add.4s	v1, v2, v1
	eor.16b	v4, v1, v7
	rev32.8h	v5, v4
	add.4s	v7, v16, v5
	eor.16b	v2, v7, v2
	shl.4s	v16, v2, #20
	sri.4s	v16, v2, #12
	add.4s	v1, v1, v8
	mov.16b	v26, v8
	add.4s	v4, v1, v16
	eor.16b	v1, v4, v5
	shl.4s	v2, v1, #24
	sri.4s	v2, v1, #8
	add.4s	v5, v2, v7
	eor.16b	v1, v5, v16
	shl.4s	v16, v1, #25
	sri.4s	v16, v1, #7
	add.4s	v1, v14, v19
	eor.16b	v7, v1, v10
	shl.4s	v19, v7, #25
	sri.4s	v19, v7, #7
	ldr	q7, [sp, #336]                  ; 16-byte Folded Reload
	add.4s	v0, v0, v7
	add.4s	v0, v0, v18
	eor.16b	v2, v2, v0
	rev32.8h	v2, v2
	add.4s	v1, v1, v2
	eor.16b	v7, v1, v18
	shl.4s	v18, v7, #20
	sri.4s	v18, v7, #12
	add.4s	v0, v0, v24
	add.4s	v0, v0, v18
	eor.16b	v2, v0, v2
	shl.4s	v7, v2, #24
	sri.4s	v7, v2, #8
	add.4s	v11, v1, v7
	eor.16b	v2, v11, v18
	shl.4s	v1, v2, #25
	sri.4s	v1, v2, #7
	add.4s	v2, v3, v25
	add.4s	v2, v2, v19
	eor.16b	v3, v2, v13
	rev32.8h	v3, v3
	add.4s	v18, v5, v3
	eor.16b	v5, v18, v19
	shl.4s	v19, v5, #20
	sri.4s	v19, v5, #12
	ldp	q24, q27, [sp, #144]            ; 32-byte Folded Reload
	add.4s	v2, v2, v27
	add.4s	v5, v2, v19
	eor.16b	v2, v5, v3
	shl.4s	v13, v2, #24
	sri.4s	v13, v2, #8
	add.4s	v2, v6, v24
	add.4s	v2, v2, v16
	eor.16b	v3, v2, v15
	rev32.8h	v10, v3
	add.4s	v12, v12, v10
	eor.16b	v3, v12, v16
	shl.4s	v16, v3, #20
	sri.4s	v16, v3, #12
	add.4s	v18, v13, v18
	eor.16b	v6, v18, v19
	shl.4s	v3, v6, #25
	sri.4s	v3, v6, #7
	add.4s	v2, v2, v22
	add.4s	v6, v2, v16
	eor.16b	v2, v6, v10
	shl.4s	v19, v2, #24
	sri.4s	v19, v2, #8
	add.4s	v12, v19, v12
	eor.16b	v16, v12, v16
	shl.4s	v2, v16, #25
	sri.4s	v2, v16, #7
	add.4s	v4, v20, v4
	mov.16b	v28, v20
	add.4s	v4, v9, v4
	eor.16b	v16, v4, v14
	rev32.8h	v16, v16
	add.4s	v17, v17, v16
	eor.16b	v9, v17, v9
	shl.4s	v14, v9, #20
	sri.4s	v14, v9, #12
	mov.16b	v20, v30
	add.4s	v4, v4, v30
	add.4s	v10, v4, v14
	eor.16b	v4, v10, v16
	shl.4s	v9, v4, #24
	sri.4s	v9, v4, #8
	add.4s	v16, v9, v17
	eor.16b	v4, v16, v14
	shl.4s	v17, v4, #25
	sri.4s	v17, v4, #7
	add.4s	v0, v0, v29
	add.4s	v0, v0, v17
	eor.16b	v4, v0, v13
	rev32.8h	v13, v4
	add.4s	v12, v12, v13
	eor.16b	v4, v12, v17
	shl.4s	v17, v4, #20
	sri.4s	v17, v4, #12
	add.4s	v0, v31, v0
	add.4s	v4, v17, v0
	eor.16b	v0, v4, v13
	shl.4s	v14, v0, #24
	sri.4s	v14, v0, #8
	mov.16b	v8, v23
	add.4s	v0, v23, v5
	add.4s	v0, v1, v0
	eor.16b	v5, v19, v0
	rev32.8h	v5, v5
	add.4s	v19, v16, v5
	eor.16b	v1, v19, v1
	shl.4s	v13, v1, #20
	sri.4s	v13, v1, #12
	add.4s	v0, v0, v25
	add.4s	v16, v0, v13
	eor.16b	v0, v16, v5
	shl.4s	v15, v0, #24
	sri.4s	v15, v0, #8
	add.4s	v5, v15, v19
	eor.16b	v0, v5, v13
	shl.4s	v19, v0, #25
	sri.4s	v19, v0, #7
	add.4s	v1, v12, v14
	eor.16b	v0, v1, v17
	shl.4s	v12, v0, #25
	sri.4s	v12, v0, #7
	add.4s	v0, v26, v6
	add.4s	v0, v3, v0
	eor.16b	v6, v9, v0
	rev32.8h	v6, v6
	add.4s	v17, v11, v6
	eor.16b	v3, v17, v3
	shl.4s	v9, v3, #20
	sri.4s	v9, v3, #12
	ldr	q29, [sp, #224]                 ; 16-byte Folded Reload
	add.4s	v0, v0, v29
	add.4s	v0, v0, v9
	eor.16b	v3, v0, v6
	shl.4s	v11, v3, #24
	sri.4s	v11, v3, #8
	mov.16b	v30, v21
	add.4s	v3, v21, v10
	add.4s	v3, v2, v3
	eor.16b	v6, v3, v7
	rev32.8h	v6, v6
	add.4s	v7, v18, v6
	eor.16b	v2, v7, v2
	shl.4s	v18, v2, #20
	sri.4s	v18, v2, #12
	add.4s	v2, v3, v22
	add.4s	v2, v2, v18
	eor.16b	v3, v2, v6
	shl.4s	v6, v3, #24
	sri.4s	v6, v3, #8
	add.4s	v3, v11, v17
	eor.16b	v17, v3, v9
	shl.4s	v9, v17, #25
	sri.4s	v9, v17, #7
	add.4s	v17, v6, v7
	eor.16b	v7, v17, v18
	shl.4s	v18, v7, #25
	sri.4s	v18, v7, #7
	ldr	q26, [sp, #272]                 ; 16-byte Folded Reload
	add.4s	v4, v4, v26
	add.4s	v4, v4, v19
	eor.16b	v6, v6, v4
	rev32.8h	v6, v6
	add.4s	v10, v3, v6
	eor.16b	v3, v10, v19
	shl.4s	v19, v3, #20
	sri.4s	v19, v3, #12
	add.4s	v3, v4, v27
	add.4s	v3, v3, v19
	eor.16b	v4, v3, v6
	shl.4s	v7, v4, #24
	sri.4s	v7, v4, #8
	add.4s	v10, v10, v7
	eor.16b	v4, v10, v19
	shl.4s	v13, v4, #25
	sri.4s	v13, v4, #7
	mov.16b	v25, v24
	add.4s	v4, v16, v24
	add.4s	v4, v4, v9
	eor.16b	v6, v4, v14
	rev32.8h	v6, v6
	add.4s	v16, v17, v6
	eor.16b	v17, v16, v9
	shl.4s	v19, v17, #20
	sri.4s	v19, v17, #12
	ldr	q21, [sp, #240]                 ; 16-byte Folded Reload
	add.4s	v0, v0, v21
	add.4s	v0, v0, v18
	eor.16b	v17, v0, v15
	rev32.8h	v17, v17
	add.4s	v1, v1, v17
	eor.16b	v18, v1, v18
	shl.4s	v9, v18, #20
	sri.4s	v9, v18, #12
	ldr	q23, [sp, #352]                 ; 16-byte Folded Reload
	add.4s	v4, v4, v23
	add.4s	v18, v4, v19
	eor.16b	v4, v18, v6
	shl.4s	v6, v4, #24
	sri.4s	v6, v4, #8
	add.4s	v0, v0, v28
	add.4s	v0, v0, v9
	eor.16b	v4, v0, v17
	shl.4s	v17, v4, #24
	sri.4s	v17, v4, #8
	add.4s	v1, v17, v1
	eor.16b	v4, v1, v9
	shl.4s	v14, v4, #25
	sri.4s	v14, v4, #7
	add.4s	v2, v20, v2
	mov.16b	v24, v20
	add.4s	v2, v12, v2
	eor.16b	v4, v2, v11
	rev32.8h	v9, v4
	add.4s	v5, v5, v9
	eor.16b	v4, v5, v12
	shl.4s	v11, v4, #20
	sri.4s	v11, v4, #12
	ldr	q31, [sp, #336]                 ; 16-byte Folded Reload
	add.4s	v2, v2, v31
	add.4s	v4, v2, v11
	eor.16b	v2, v4, v9
	shl.4s	v9, v2, #24
	sri.4s	v9, v2, #8
	add.4s	v2, v9, v5
	eor.16b	v5, v2, v11
	shl.4s	v11, v5, #25
	sri.4s	v11, v5, #7
	add.4s	v16, v6, v16
	eor.16b	v5, v16, v19
	shl.4s	v19, v5, #25
	sri.4s	v19, v5, #7
	add.4s	v3, v3, v8
	add.4s	v3, v3, v11
	eor.16b	v5, v3, v6
	rev32.8h	v6, v5
	add.4s	v12, v1, v6
	eor.16b	v1, v12, v11
	shl.4s	v11, v1, #20
	sri.4s	v11, v1, #12
	add.4s	v1, v30, v3
	add.4s	v5, v11, v1
	eor.16b	v1, v5, v6
	shl.4s	v15, v1, #24
	sri.4s	v15, v1, #8
	ldr	q8, [sp, #208]                  ; 16-byte Folded Reload
	add.4s	v1, v8, v18
	add.4s	v1, v13, v1
	eor.16b	v3, v17, v1
	rev32.8h	v3, v3
	add.4s	v2, v2, v3
	eor.16b	v6, v2, v13
	shl.4s	v18, v6, #20
	sri.4s	v18, v6, #12
	add.4s	v1, v1, v25
	add.4s	v6, v1, v18
	eor.16b	v3, v6, v3
	shl.4s	v1, v3, #24
	sri.4s	v1, v3, #8
	add.4s	v17, v1, v2
	eor.16b	v2, v17, v18
	shl.4s	v18, v2, #25
	sri.4s	v18, v2, #7
	add.4s	v0, v22, v0
	add.4s	v0, v19, v0
	eor.16b	v2, v9, v0
	rev32.8h	v2, v2
	add.4s	v9, v10, v2
	eor.16b	v3, v9, v19
	shl.4s	v19, v3, #20
	sri.4s	v19, v3, #12
	add.4s	v3, v12, v15
	eor.16b	v10, v3, v11
	shl.4s	v12, v10, #25
	sri.4s	v12, v10, #7
	ldr	q20, [sp, #320]                 ; 16-byte Folded Reload
	add.4s	v0, v0, v20
	add.4s	v10, v0, v19
	eor.16b	v0, v10, v2
	shl.4s	v2, v0, #24
	sri.4s	v2, v0, #8
	add.4s	v0, v2, v9
	eor.16b	v19, v0, v19
	shl.4s	v11, v19, #25
	sri.4s	v11, v19, #7
	ldr	q30, [sp, #192]                 ; 16-byte Folded Reload
	add.4s	v4, v30, v4
	add.4s	v4, v14, v4
	eor.16b	v7, v4, v7
	rev32.8h	v7, v7
	add.4s	v19, v16, v7
	eor.16b	v16, v19, v14
	shl.4s	v9, v16, #20
	sri.4s	v9, v16, #12
	add.4s	v4, v4, v28
	add.4s	v16, v4, v9
	eor.16b	v4, v16, v7
	shl.4s	v7, v4, #24
	sri.4s	v7, v4, #8
	add.4s	v4, v7, v19
	eor.16b	v19, v4, v9
	shl.4s	v14, v19, #25
	sri.4s	v14, v19, #7
	ldr	q19, [sp, #368]                 ; 16-byte Folded Reload
	add.4s	v5, v5, v19
	add.4s	v5, v5, v18
	eor.16b	v7, v7, v5
	rev32.8h	v7, v7
	add.4s	v0, v0, v7
	eor.16b	v18, v0, v18
	shl.4s	v19, v18, #20
	sri.4s	v19, v18, #12
	add.4s	v5, v5, v23
	add.4s	v5, v5, v19
	eor.16b	v18, v5, v7
	shl.4s	v7, v18, #24
	sri.4s	v7, v18, #8
	add.4s	v9, v0, v7
	eor.16b	v0, v9, v19
	shl.4s	v13, v0, #25
	sri.4s	v13, v0, #7
	mov.16b	v20, v21
	add.4s	v0, v6, v21
	add.4s	v0, v0, v11
	eor.16b	v6, v0, v15
	rev32.8h	v6, v6
	add.4s	v18, v4, v6
	eor.16b	v4, v18, v11
	shl.4s	v19, v4, #20
	sri.4s	v19, v4, #12
	add.4s	v0, v0, v29
	mov.16b	v23, v29
	add.4s	v4, v0, v19
	eor.16b	v6, v4, v6
	shl.4s	v0, v6, #24
	sri.4s	v0, v6, #8
	add.4s	v6, v10, v27
	add.4s	v6, v6, v14
	eor.16b	v1, v6, v1
	rev32.8h	v1, v1
	add.4s	v10, v3, v1
	eor.16b	v3, v10, v14
	shl.4s	v11, v3, #20
	sri.4s	v11, v3, #12
	add.4s	v3, v6, v24
	add.4s	v3, v3, v11
	eor.16b	v1, v3, v1
	shl.4s	v14, v1, #24
	sri.4s	v14, v1, #8
	add.4s	v15, v14, v10
	eor.16b	v1, v15, v11
	shl.4s	v10, v1, #25
	sri.4s	v10, v1, #7
	mov.16b	v21, v31
	add.4s	v1, v31, v16
	add.4s	v1, v12, v1
	eor.16b	v2, v1, v2
	rev32.8h	v2, v2
	add.4s	v16, v17, v2
	eor.16b	v6, v16, v12
	shl.4s	v17, v6, #20
	sri.4s	v17, v6, #12
	add.4s	v1, v1, v26
	mov.16b	v12, v26
	add.4s	v1, v1, v17
	eor.16b	v2, v1, v2
	shl.4s	v6, v2, #24
	sri.4s	v6, v2, #8
	add.4s	v2, v6, v16
	eor.16b	v16, v2, v17
	shl.4s	v17, v16, #25
	sri.4s	v17, v16, #7
	add.4s	v5, v5, v8
	add.4s	v8, v0, v18
	eor.16b	v16, v8, v19
	add.4s	v4, v25, v4
	shl.4s	v18, v16, #25
	sri.4s	v18, v16, #7
	add.4s	v5, v5, v17
	eor.16b	v0, v5, v0
	add.4s	v4, v13, v4
	eor.16b	v16, v14, v4
	rev32.8h	v16, v16
	add.4s	v5, v30, v5
	add.4s	v19, v2, v16
	eor.16b	v2, v19, v13
	shl.4s	v30, v2, #20
	sri.4s	v30, v2, #12
	rev32.8h	v0, v0
	add.4s	v31, v15, v0
	eor.16b	v17, v31, v17
	add.4s	v2, v4, v20
	shl.4s	v11, v17, #20
	add.4s	v2, v2, v30
	eor.16b	v4, v2, v16
	sri.4s	v11, v17, #12
	shl.4s	v26, v4, #24
	sri.4s	v26, v4, #8
	add.4s	v5, v11, v5
	eor.16b	v0, v5, v0
	add.4s	v3, v28, v3
	shl.4s	v4, v0, #24
	add.4s	v25, v26, v19
	eor.16b	v16, v25, v30
	add.4s	v3, v18, v3
	eor.16b	v6, v6, v3
	rev32.8h	v6, v6
	sri.4s	v4, v0, #8
	add.4s	v17, v9, v6
	eor.16b	v0, v17, v18
	shl.4s	v18, v0, #20
	sri.4s	v18, v0, #12
	shl.4s	v19, v16, #25
	sri.4s	v19, v16, #7
	ldr	q0, [sp, #256]                  ; 16-byte Folded Reload
	add.4s	v0, v3, v0
	add.4s	v3, v0, v18
	eor.16b	v0, v3, v6
	add.4s	v1, v22, v1
	shl.4s	v28, v0, #24
	sri.4s	v28, v0, #8
	add.4s	v1, v10, v1
	eor.16b	v0, v1, v7
	rev32.8h	v6, v0
	add.4s	v16, v8, v6
	eor.16b	v0, v16, v10
	ldr	q8, [sp, #64]                   ; 16-byte Folded Reload
	shl.4s	v29, v0, #20
	sri.4s	v29, v0, #12
	add.4s	v0, v31, v4
	add.4s	v1, v1, v24
	add.4s	v7, v1, v29
	eor.16b	v1, v7, v6
	eor.16b	v6, v0, v11
	shl.4s	v22, v1, #24
	sri.4s	v22, v1, #8
	shl.4s	v1, v6, #25
	add.4s	v17, v28, v17
	eor.16b	v18, v17, v18
	sri.4s	v1, v6, #7
	add.4s	v6, v22, v16
	eor.16b	v16, v6, v29
	shl.4s	v29, v16, #25
	sri.4s	v29, v16, #7
	shl.4s	v16, v18, #25
	sri.4s	v16, v18, #7
	ldr	q18, [sp, #288]                 ; 16-byte Folded Reload
	add.4s	v5, v5, v18
	add.4s	v5, v5, v19
	eor.16b	v18, v22, v5
	rev32.8h	v18, v18
	add.4s	v17, v17, v18
	eor.16b	v19, v17, v19
	shl.4s	v22, v19, #20
	sri.4s	v22, v19, #12
	add.4s	v5, v5, v23
	add.4s	v2, v2, v27
	add.4s	v2, v2, v16
	eor.16b	v4, v2, v4
	ldr	q19, [sp, #320]                 ; 16-byte Folded Reload
	add.4s	v19, v2, v19
	add.4s	v2, v5, v22
	eor.16b	v5, v2, v18
	rev32.8h	v4, v4
	ldr	q18, [sp, #352]                 ; 16-byte Folded Reload
	add.4s	v18, v3, v18
	shl.4s	v3, v5, #24
	add.4s	v6, v6, v4
	eor.16b	v16, v6, v16
	add.4s	v18, v18, v29
	eor.16b	v23, v18, v26
	rev32.8h	v23, v23
	sri.4s	v3, v5, #8
	add.4s	v0, v0, v23
	eor.16b	v5, v0, v29
	shl.4s	v24, v5, #20
	sri.4s	v24, v5, #12
	shl.4s	v5, v16, #20
	sri.4s	v5, v16, #12
	add.4s	v16, v18, v21
	add.4s	v16, v16, v24
	eor.16b	v18, v16, v23
	shl.4s	v20, v18, #24
	sri.4s	v20, v18, #8
	add.4s	v17, v17, v3
	eor.16b	v18, v17, v22
	add.4s	v7, v12, v7
	shl.4s	v22, v18, #25
	add.4s	v19, v19, v5
	eor.16b	v4, v19, v4
	sri.4s	v22, v18, #7
	shl.4s	v23, v4, #24
	add.4s	v7, v1, v7
	eor.16b	v18, v7, v28
	rev32.8h	v18, v18
	add.4s	v25, v25, v18
	sri.4s	v23, v4, #8
	eor.16b	v1, v25, v1
	shl.4s	v4, v1, #20
	sri.4s	v4, v1, #12
	add.4s	v1, v23, v6
	add.4s	v0, v20, v0
	eor.16b	v5, v1, v5
	eor.16b	v6, v0, v24
	ldr	q21, [sp, #368]                 ; 16-byte Folded Reload
	add.4s	v7, v7, v21
	shl.4s	v21, v6, #25
	add.4s	v24, v7, v4
	eor.16b	v7, v24, v18
	sri.4s	v21, v6, #7
	shl.4s	v6, v7, #24
	sri.4s	v6, v7, #8
	shl.4s	v26, v5, #25
	add.4s	v18, v6, v25
	eor.16b	v4, v18, v4
	sri.4s	v26, v5, #7
	shl.4s	v5, v4, #25
	sri.4s	v5, v4, #7
	eor.16b	v7, v0, v2
	eor.16b	v15, v18, v19
	eor.16b	v12, v16, v17
	ldr	q17, [sp, #80]                  ; 16-byte Folded Reload
	eor.16b	v18, v24, v1
	eor.16b	v14, v5, v23
	eor.16b	v13, v20, v22
	eor.16b	v11, v6, v26
	eor.16b	v9, v21, v3
	add	x12, x12, #64
	cbnz	x14, LBB0_4
; %bb.5:                                ;   in Loop: Header=BB0_3 Depth=2
	trn1.4s	v0, v7, v15
	trn2.4s	v1, v7, v15
	trn1.4s	v2, v12, v18
	trn2.4s	v3, v12, v18
	zip2.2d	v4, v0, v2
	mov.d	v0[1], v2[0]
	zip2.2d	v2, v1, v3
	mov.d	v1[1], v3[0]
	trn1.4s	v3, v14, v13
	trn2.4s	v5, v14, v13
	trn1.4s	v6, v11, v9
	trn2.4s	v7, v11, v9
	zip2.2d	v16, v3, v6
	mov.d	v3[1], v6[0]
	zip2.2d	v6, v5, v7
	mov.d	v5[1], v7[0]
	stp	q0, q3, [x1], #128
	stp	q1, q5, [x1, #-96]
	stp	q4, q16, [x1, #-64]
	stp	q2, q6, [x1, #-32]
	ldp	q18, q7, [sp, #96]              ; 32-byte Folded Reload
	add	x12, x12, #3072
	subs	w11, w11, #1
	b.ne	LBB0_3
; %bb.6:                                ;   in Loop: Header=BB0_2 Depth=1
	subs	x2, x2, #1
	b.ne	LBB0_2
; %bb.7:
	ldp	d9, d8, [sp, #432]              ; 16-byte Folded Reload
	ldp	d11, d10, [sp, #416]            ; 16-byte Folded Reload
	ldp	d13, d12, [sp, #400]            ; 16-byte Folded Reload
	ldp	d15, d14, [sp, #384]            ; 16-byte Folded Reload
	add	sp, sp, #448
LBB0_8:
	ret
	.cfi_endproc
                                        ; -- End function
	.private_extern	_flock_blake3_hash8_neon_1024_parent1 ; -- Begin function flock_blake3_hash8_neon_1024_parent1
	.globl	_flock_blake3_hash8_neon_1024_parent1
	.p2align	2
_flock_blake3_hash8_neon_1024_parent1:  ; @flock_blake3_hash8_neon_1024_parent1
	.cfi_startproc
; %bb.0:
	cbz	x3, LBB1_8
; %bb.1:
	stp	d15, d14, [sp, #-96]!           ; 16-byte Folded Spill
	stp	d13, d12, [sp, #16]             ; 16-byte Folded Spill
	stp	d11, d10, [sp, #32]             ; 16-byte Folded Spill
	stp	d9, d8, [sp, #48]               ; 16-byte Folded Spill
	stp	x22, x21, [sp, #64]             ; 16-byte Folded Spill
	stp	x20, x19, [sp, #80]             ; 16-byte Folded Spill
	sub	sp, sp, #608
	.cfi_def_cfa_offset 704
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
	.cfi_offset w22, -32
	.cfi_offset b8, -40
	.cfi_offset b9, -48
	.cfi_offset b10, -56
	.cfi_offset b11, -64
	.cfi_offset b12, -72
	.cfi_offset b13, -80
	.cfi_offset b14, -88
	.cfi_offset b15, -96
	mov	x8, #0                          ; =0x0
	mov	w9, #52505                      ; =0xcd19
	movk	w9, #23520, lsl #16
	dup.4s	v1, w9
	mov	w10, #55723                     ; =0xd9ab
	movk	w10, #8067, lsl #16
	dup.4s	v0, w10
	stp	q0, q1, [sp, #32]               ; 32-byte Folded Spill
	mov	w11, #26764                     ; =0x688c
	movk	w11, #39685, lsl #16
	dup.4s	v1, w11
	mov	w12, #21119                     ; =0x527f
	movk	w12, #20750, lsl #16
	dup.4s	v0, w12
	stp	q0, q1, [sp]                    ; 32-byte Folded Spill
	mov	w13, #62778                     ; =0xf53a
	movk	w13, #42319, lsl #16
	dup.4s	v29, w13
	mov	w14, #62322                     ; =0xf372
	movk	w14, #15470, lsl #16
	dup.4s	v1, w14
	mov	w15, #44677                     ; =0xae85
	movk	w15, #47975, lsl #16
	dup.4s	v2, w15
	mov	w16, #58983                     ; =0xe667
	movk	w16, #27145, lsl #16
	dup.4s	v3, w16
	mov	w17, #14566                     ; =0x38e6
	movk	w17, #47896, lsl #16
	mov	w4, #5905                       ; =0x1711
	movk	w4, #22125, lsl #16
	mov	w5, #52509                      ; =0xcd1d
	movk	w5, #23538, lsl #16
	mov	w6, #49747                      ; =0xc253
	movk	w6, #304, lsl #16
	stp	q1, q29, [sp, #320]             ; 32-byte Folded Spill
	stp	q3, q2, [sp, #288]              ; 32-byte Folded Spill
LBB1_2:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB1_3 Depth 2
                                        ;     Child Loop BB1_5 Depth 2
	mov	x7, x0
	mov	w20, #16                        ; =0x10
	mov.16b	v9, v3
	mov.16b	v23, v2
	mov.16b	v13, v1
	ldp	q24, q8, [sp]                   ; 32-byte Folded Reload
	ldp	q11, q15, [sp, #32]             ; 32-byte Folded Reload
LBB1_3:                                 ;   Parent Loop BB1_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	ldr	q0, [x7, #1024]
	ldr	q1, [x7, #2048]
	ldr	q2, [x7, #3072]
	ldp	q3, q5, [x7]
	trn1.4s	v26, v3, v0
	trn2.4s	v3, v3, v0
	trn1.4s	v0, v1, v2
	trn2.4s	v2, v1, v2
	zip2.2d	v25, v26, v0
	zip2.2d	v12, v3, v2
	ldr	q1, [x7, #1040]
	ldr	q6, [x7, #2064]
	ldr	q7, [x7, #3088]
	trn1.4s	v28, v5, v1
	mov.16b	v4, v3
	trn2.4s	v16, v5, v1
	trn1.4s	v5, v6, v7
	trn2.4s	v18, v6, v7
	zip2.2d	v1, v28, v5
	zip2.2d	v3, v16, v18
	mov.16b	v22, v16
	ldr	q6, [x7, #1056]
	ldr	q7, [x7, #2080]
	ldr	q17, [x7, #3104]
	mov.d	v26[1], v0[0]
	ldp	q0, q19, [x7, #32]
	trn1.4s	v20, v0, v6
	trn2.4s	v0, v0, v6
	mov.16b	v6, v4
	mov.d	v6[1], v2[0]
	str	q6, [sp, #560]                  ; 16-byte Folded Spill
	trn1.4s	v2, v7, v17
	trn2.4s	v21, v7, v17
	zip2.2d	v4, v20, v2
	mov.16b	v7, v20
	mov.d	v28[1], v5[0]
	mov.16b	v14, v28
	str	q28, [sp, #496]                 ; 16-byte Folded Spill
	zip2.2d	v16, v0, v21
	mov.16b	v17, v0
	ldr	q0, [x7, #1072]
	ldr	q5, [x7, #2096]
	mov.d	v22[1], v18[0]
	mov.16b	v27, v22
	str	q22, [sp, #528]                 ; 16-byte Folded Spill
	ldr	q18, [x7, #3120]
	trn1.4s	v20, v19, v0
	trn2.4s	v30, v19, v0
	trn1.4s	v19, v5, v18
	mov.d	v7[1], v2[0]
	str	q7, [sp, #416]                  ; 16-byte Folded Spill
	trn2.4s	v22, v5, v18
	zip2.2d	v28, v20, v19
	str	q28, [sp, #576]                 ; 16-byte Folded Spill
	zip2.2d	v31, v30, v22
	mov.d	v17[1], v21[0]
	mov.16b	v2, v17
	str	q17, [sp, #480]                 ; 16-byte Folded Spill
	subs	x19, x20, #1
	cset	w21, eq
	cmp	x20, #16
	mov.d	v20[1], v19[0]
	mov.16b	v17, v20
	cset	w20, eq
	orr	w20, w20, w21, lsl #1
	dup.4s	v18, w20
	mov.d	v30[1], v22[0]
	add.4s	v19, v9, v24
	add.4s	v19, v19, v26
	str	q26, [sp, #384]                 ; 16-byte Folded Spill
	rev32.8h	v20, v19
	ldp	q5, q0, [sp, #288]              ; 32-byte Folded Reload
	add.4s	v21, v20, v5
	eor.16b	v22, v21, v24
	shl.4s	v24, v22, #20
	sri.4s	v24, v22, #12
	add.4s	v19, v19, v6
	add.4s	v19, v19, v24
	add.4s	v22, v23, v8
	add.4s	v22, v22, v25
	mov.16b	v6, v25
	eor.16b	v20, v19, v20
	rev32.8h	v23, v22
	add.4s	v25, v23, v0
	eor.16b	v8, v25, v8
	shl.4s	v10, v8, #20
	sri.4s	v10, v8, #12
	shl.4s	v8, v20, #24
	add.4s	v22, v22, v12
	mov.16b	v5, v12
	add.4s	v9, v22, v10
	eor.16b	v22, v9, v23
	shl.4s	v23, v22, #24
	sri.4s	v23, v22, #8
	sri.4s	v8, v20, #8
	add.4s	v22, v23, v25
	eor.16b	v20, v22, v10
	shl.4s	v12, v20, #25
	add.4s	v25, v13, v11
	add.4s	v10, v25, v14
	sri.4s	v12, v20, #7
	movi.4s	v0, #64
	eor.16b	v20, v10, v0
	rev32.8h	v20, v20
	ldr	q0, [sp, #320]                  ; 16-byte Folded Reload
	add.4s	v13, v20, v0
	eor.16b	v11, v13, v11
	shl.4s	v14, v11, #20
	add.4s	v25, v21, v8
	sri.4s	v14, v11, #12
	add.4s	v21, v10, v27
	add.4s	v10, v21, v14
	eor.16b	v20, v10, v20
	shl.4s	v21, v20, #24
	eor.16b	v11, v25, v24
	sri.4s	v21, v20, #8
	shl.4s	v24, v11, #25
	add.4s	v20, v29, v15
	mov.16b	v27, v1
	add.4s	v20, v20, v1
	eor.16b	v18, v20, v18
	sri.4s	v24, v11, #7
	rev32.8h	v18, v18
	mov.16b	v0, v15
	ldr	q1, [sp, #336]                  ; 16-byte Folded Reload
	add.4s	v15, v18, v1
	eor.16b	v11, v15, v0
	shl.4s	v0, v11, #20
	sri.4s	v0, v11, #12
	str	q3, [sp, #512]                  ; 16-byte Folded Spill
	add.4s	v20, v20, v3
	add.4s	v11, v20, v0
	eor.16b	v18, v11, v18
	shl.4s	v20, v18, #24
	sri.4s	v20, v18, #8
	add.4s	v15, v20, v15
	eor.16b	v0, v15, v0
	shl.4s	v1, v0, #25
	sri.4s	v1, v0, #7
	add.4s	v0, v21, v13
	eor.16b	v18, v0, v14
	shl.4s	v13, v18, #25
	sri.4s	v13, v18, #7
	add.4s	v18, v19, v7
	add.4s	v18, v18, v12
	eor.16b	v19, v20, v18
	rev32.8h	v19, v19
	add.4s	v0, v0, v19
	eor.16b	v20, v0, v12
	shl.4s	v14, v20, #20
	sri.4s	v14, v20, #12
	add.4s	v18, v18, v2
	add.4s	v12, v18, v14
	eor.16b	v19, v12, v19
	shl.4s	v18, v19, #24
	sri.4s	v18, v19, #8
	add.4s	v20, v0, v18
	eor.16b	v0, v20, v14
	shl.4s	v19, v0, #25
	sri.4s	v19, v0, #7
	add.4s	v0, v9, v4
	add.4s	v0, v0, v13
	eor.16b	v8, v0, v8
	rev32.8h	v8, v8
	add.4s	v14, v15, v8
	eor.16b	v9, v14, v13
	shl.4s	v13, v9, #20
	sri.4s	v13, v9, #12
	str	q16, [sp, #448]                 ; 16-byte Folded Spill
	add.4s	v0, v0, v16
	add.4s	v9, v0, v13
	eor.16b	v0, v9, v8
	shl.4s	v15, v0, #24
	sri.4s	v15, v0, #8
	mov.16b	v7, v17
	add.4s	v0, v10, v17
	add.4s	v0, v0, v1
	eor.16b	v23, v0, v23
	rev32.8h	v8, v23
	add.4s	v25, v25, v8
	eor.16b	v1, v25, v1
	shl.4s	v23, v1, #20
	sri.4s	v23, v1, #12
	add.4s	v2, v15, v14
	eor.16b	v13, v2, v13
	shl.4s	v10, v13, #25
	sri.4s	v10, v13, #7
	add.4s	v0, v0, v30
	add.4s	v13, v0, v23
	eor.16b	v0, v13, v8
	shl.4s	v1, v0, #24
	sri.4s	v1, v0, #8
	add.4s	v0, v1, v25
	eor.16b	v23, v0, v23
	shl.4s	v8, v23, #25
	sri.4s	v8, v23, #7
	add.4s	v23, v28, v11
	add.4s	v23, v24, v23
	eor.16b	v21, v23, v21
	rev32.8h	v21, v21
	add.4s	v22, v22, v21
	eor.16b	v24, v22, v24
	shl.4s	v25, v24, #20
	sri.4s	v25, v24, #12
	add.4s	v23, v23, v31
	add.4s	v14, v23, v25
	eor.16b	v21, v14, v21
	shl.4s	v23, v21, #24
	sri.4s	v23, v21, #8
	add.4s	v21, v23, v22
	eor.16b	v22, v21, v25
	shl.4s	v24, v22, #25
	sri.4s	v24, v22, #7
	add.4s	v22, v12, v6
	str	q6, [sp, #464]                  ; 16-byte Folded Spill
	add.4s	v22, v22, v24
	eor.16b	v25, v22, v15
	rev32.8h	v25, v25
	add.4s	v0, v0, v25
	eor.16b	v24, v0, v24
	shl.4s	v15, v24, #20
	sri.4s	v15, v24, #12
	add.4s	v22, v27, v22
	str	q27, [sp, #400]                 ; 16-byte Folded Spill
	add.4s	v11, v15, v22
	eor.16b	v22, v11, v25
	shl.4s	v24, v22, #24
	sri.4s	v24, v22, #8
	add.4s	v22, v5, v9
	add.4s	v22, v19, v22
	eor.16b	v1, v1, v22
	rev32.8h	v1, v1
	add.4s	v21, v21, v1
	eor.16b	v19, v21, v19
	shl.4s	v9, v19, #20
	sri.4s	v9, v19, #12
	add.4s	v19, v22, v4
	mov.16b	v29, v4
	add.4s	v12, v19, v9
	eor.16b	v1, v12, v1
	shl.4s	v25, v1, #24
	sri.4s	v25, v1, #8
	add.4s	v19, v25, v21
	eor.16b	v1, v19, v9
	shl.4s	v9, v1, #25
	sri.4s	v9, v1, #7
	add.4s	v0, v0, v24
	eor.16b	v1, v0, v15
	shl.4s	v21, v1, #25
	sri.4s	v21, v1, #7
	add.4s	v1, v3, v13
	add.4s	v1, v10, v1
	eor.16b	v22, v23, v1
	rev32.8h	v22, v22
	add.4s	v20, v20, v22
	eor.16b	v23, v20, v10
	shl.4s	v10, v23, #20
	sri.4s	v10, v23, #12
	add.4s	v1, v1, v26
	add.4s	v13, v1, v10
	eor.16b	v1, v13, v22
	shl.4s	v22, v1, #24
	sri.4s	v22, v1, #8
	ldr	q3, [sp, #496]                  ; 16-byte Folded Reload
	add.4s	v1, v3, v14
	add.4s	v1, v8, v1
	eor.16b	v18, v1, v18
	rev32.8h	v18, v18
	add.4s	v2, v2, v18
	eor.16b	v23, v2, v8
	shl.4s	v14, v23, #20
	sri.4s	v14, v23, #12
	mov.16b	v4, v30
	str	q30, [sp, #592]                 ; 16-byte Folded Spill
	add.4s	v1, v1, v30
	add.4s	v8, v1, v14
	eor.16b	v1, v8, v18
	shl.4s	v18, v1, #24
	sri.4s	v18, v1, #8
	add.4s	v1, v22, v20
	eor.16b	v20, v1, v10
	shl.4s	v15, v20, #25
	sri.4s	v15, v20, #7
	add.4s	v2, v18, v2
	eor.16b	v20, v2, v14
	shl.4s	v14, v20, #25
	sri.4s	v14, v20, #7
	ldr	q17, [sp, #560]                 ; 16-byte Folded Reload
	add.4s	v20, v11, v17
	add.4s	v20, v20, v9
	eor.16b	v18, v18, v20
	rev32.8h	v18, v18
	add.4s	v1, v1, v18
	eor.16b	v23, v1, v9
	shl.4s	v9, v23, #20
	sri.4s	v9, v23, #12
	add.4s	v20, v20, v16
	add.4s	v10, v20, v9
	eor.16b	v20, v10, v18
	shl.4s	v18, v20, #24
	sri.4s	v18, v20, #8
	add.4s	v20, v1, v18
	eor.16b	v1, v20, v9
	shl.4s	v23, v1, #25
	sri.4s	v23, v1, #7
	add.4s	v1, v12, v7
	add.4s	v1, v1, v15
	eor.16b	v24, v1, v24
	rev32.8h	v24, v24
	add.4s	v2, v2, v24
	eor.16b	v9, v2, v15
	shl.4s	v12, v9, #20
	sri.4s	v12, v9, #12
	ldr	q28, [sp, #480]                 ; 16-byte Folded Reload
	add.4s	v9, v13, v28
	add.4s	v9, v9, v14
	eor.16b	v25, v9, v25
	rev32.8h	v11, v25
	add.4s	v0, v0, v11
	eor.16b	v25, v0, v14
	shl.4s	v13, v25, #20
	sri.4s	v13, v25, #12
	ldr	q26, [sp, #528]                 ; 16-byte Folded Reload
	add.4s	v1, v1, v26
	add.4s	v1, v1, v12
	eor.16b	v24, v1, v24
	shl.4s	v14, v24, #24
	sri.4s	v14, v24, #8
	ldr	q17, [sp, #576]                 ; 16-byte Folded Reload
	add.4s	v24, v9, v17
	add.4s	v25, v24, v13
	eor.16b	v24, v25, v11
	shl.4s	v15, v24, #24
	sri.4s	v15, v24, #8
	add.4s	v0, v15, v0
	eor.16b	v9, v0, v13
	shl.4s	v24, v9, #25
	sri.4s	v24, v9, #7
	add.4s	v8, v31, v8
	add.4s	v8, v21, v8
	eor.16b	v22, v8, v22
	rev32.8h	v22, v22
	add.4s	v19, v19, v22
	eor.16b	v21, v19, v21
	shl.4s	v9, v21, #20
	sri.4s	v9, v21, #12
	ldr	q30, [sp, #416]                 ; 16-byte Folded Reload
	add.4s	v21, v8, v30
	add.4s	v21, v21, v9
	eor.16b	v22, v21, v22
	shl.4s	v8, v22, #24
	sri.4s	v8, v22, #8
	add.4s	v19, v8, v19
	eor.16b	v22, v19, v9
	shl.4s	v9, v22, #25
	sri.4s	v9, v22, #7
	add.4s	v11, v14, v2
	eor.16b	v2, v11, v12
	shl.4s	v12, v2, #25
	sri.4s	v12, v2, #7
	str	q5, [sp, #432]                  ; 16-byte Folded Spill
	add.4s	v2, v10, v5
	add.4s	v2, v2, v9
	eor.16b	v22, v2, v14
	rev32.8h	v22, v22
	add.4s	v0, v0, v22
	eor.16b	v9, v0, v9
	shl.4s	v13, v9, #20
	sri.4s	v13, v9, #12
	add.4s	v2, v3, v2
	add.4s	v9, v13, v2
	eor.16b	v2, v9, v22
	shl.4s	v22, v2, #24
	sri.4s	v22, v2, #8
	mov.16b	v3, v29
	str	q29, [sp, #368]                 ; 16-byte Folded Spill
	add.4s	v1, v29, v1
	add.4s	v1, v23, v1
	eor.16b	v2, v15, v1
	rev32.8h	v2, v2
	add.4s	v19, v19, v2
	eor.16b	v23, v19, v23
	shl.4s	v14, v23, #20
	sri.4s	v14, v23, #12
	add.4s	v1, v1, v7
	mov.16b	v16, v7
	add.4s	v10, v1, v14
	eor.16b	v1, v10, v2
	shl.4s	v23, v1, #24
	sri.4s	v23, v1, #8
	add.4s	v19, v23, v19
	eor.16b	v1, v19, v14
	shl.4s	v2, v1, #25
	sri.4s	v2, v1, #7
	add.4s	v1, v4, v25
	add.4s	v1, v12, v1
	eor.16b	v25, v8, v1
	rev32.8h	v25, v25
	add.4s	v14, v20, v25
	eor.16b	v20, v14, v12
	shl.4s	v15, v20, #20
	sri.4s	v15, v20, #12
	add.4s	v8, v0, v22
	eor.16b	v0, v8, v13
	shl.4s	v20, v0, #25
	sri.4s	v20, v0, #7
	add.4s	v0, v1, v6
	add.4s	v12, v0, v15
	eor.16b	v0, v12, v25
	shl.4s	v25, v0, #24
	sri.4s	v25, v0, #8
	add.4s	v0, v25, v14
	eor.16b	v1, v0, v15
	shl.4s	v14, v1, #25
	sri.4s	v14, v1, #7
	ldr	q29, [sp, #512]                 ; 16-byte Folded Reload
	add.4s	v1, v29, v21
	add.4s	v1, v24, v1
	eor.16b	v18, v1, v18
	rev32.8h	v18, v18
	add.4s	v21, v11, v18
	eor.16b	v24, v21, v24
	shl.4s	v11, v24, #20
	sri.4s	v11, v24, #12
	add.4s	v1, v1, v17
	add.4s	v13, v1, v11
	eor.16b	v1, v13, v18
	shl.4s	v18, v1, #24
	sri.4s	v18, v1, #8
	add.4s	v1, v18, v21
	eor.16b	v21, v1, v11
	shl.4s	v11, v21, #25
	sri.4s	v11, v21, #7
	add.4s	v21, v9, v27
	add.4s	v21, v21, v2
	eor.16b	v18, v18, v21
	rev32.8h	v18, v18
	add.4s	v0, v0, v18
	eor.16b	v2, v0, v2
	shl.4s	v24, v2, #20
	sri.4s	v24, v2, #12
	add.4s	v2, v21, v26
	add.4s	v9, v2, v24
	eor.16b	v2, v9, v18
	shl.4s	v18, v2, #24
	sri.4s	v18, v2, #8
	add.4s	v21, v0, v18
	eor.16b	v0, v21, v24
	shl.4s	v24, v0, #25
	sri.4s	v24, v0, #7
	add.4s	v0, v10, v28
	add.4s	v0, v0, v14
	eor.16b	v2, v0, v22
	rev32.8h	v2, v2
	add.4s	v1, v1, v2
	eor.16b	v22, v1, v14
	shl.4s	v15, v22, #20
	sri.4s	v15, v22, #12
	ldr	q26, [sp, #384]                 ; 16-byte Folded Reload
	add.4s	v0, v0, v26
	add.4s	v10, v0, v15
	eor.16b	v0, v10, v2
	shl.4s	v2, v0, #24
	sri.4s	v2, v0, #8
	ldr	q6, [sp, #448]                  ; 16-byte Folded Reload
	add.4s	v0, v12, v6
	add.4s	v0, v0, v11
	eor.16b	v22, v0, v23
	rev32.8h	v22, v22
	add.4s	v23, v8, v22
	eor.16b	v8, v23, v11
	shl.4s	v12, v8, #20
	sri.4s	v12, v8, #12
	add.4s	v0, v0, v31
	add.4s	v11, v0, v12
	eor.16b	v0, v11, v22
	shl.4s	v22, v0, #24
	sri.4s	v22, v0, #8
	add.4s	v0, v22, v23
	eor.16b	v23, v0, v12
	shl.4s	v8, v23, #25
	sri.4s	v8, v23, #7
	add.4s	v23, v30, v13
	add.4s	v23, v20, v23
	eor.16b	v25, v23, v25
	rev32.8h	v25, v25
	add.4s	v19, v19, v25
	eor.16b	v20, v19, v20
	shl.4s	v13, v20, #20
	sri.4s	v13, v20, #12
	ldr	q7, [sp, #560]                  ; 16-byte Folded Reload
	add.4s	v20, v23, v7
	add.4s	v12, v20, v13
	eor.16b	v20, v12, v25
	shl.4s	v14, v20, #24
	sri.4s	v14, v20, #8
	add.4s	v1, v2, v1
	eor.16b	v20, v1, v15
	shl.4s	v15, v20, #25
	sri.4s	v15, v20, #7
	add.4s	v20, v14, v19
	eor.16b	v19, v20, v13
	shl.4s	v23, v19, #25
	sri.4s	v23, v19, #7
	add.4s	v19, v9, v3
	add.4s	v19, v19, v23
	eor.16b	v2, v19, v2
	rev32.8h	v2, v2
	add.4s	v0, v0, v2
	eor.16b	v23, v0, v23
	shl.4s	v25, v23, #20
	sri.4s	v25, v23, #12
	add.4s	v19, v29, v19
	add.4s	v9, v25, v19
	eor.16b	v2, v9, v2
	shl.4s	v23, v2, #24
	sri.4s	v23, v2, #8
	add.4s	v0, v0, v23
	eor.16b	v2, v0, v25
	shl.4s	v19, v2, #25
	sri.4s	v19, v2, #7
	str	q16, [sp, #544]                 ; 16-byte Folded Spill
	add.4s	v2, v16, v10
	add.4s	v2, v24, v2
	eor.16b	v22, v22, v2
	rev32.8h	v22, v22
	add.4s	v20, v20, v22
	eor.16b	v24, v20, v24
	shl.4s	v13, v24, #20
	sri.4s	v13, v24, #12
	add.4s	v2, v2, v28
	mov.16b	v29, v28
	add.4s	v10, v2, v13
	eor.16b	v2, v10, v22
	shl.4s	v25, v2, #24
	sri.4s	v25, v2, #8
	add.4s	v20, v25, v20
	eor.16b	v2, v20, v13
	shl.4s	v22, v2, #25
	sri.4s	v22, v2, #7
	add.4s	v2, v17, v11
	add.4s	v2, v15, v2
	eor.16b	v24, v14, v2
	rev32.8h	v24, v24
	add.4s	v21, v21, v24
	eor.16b	v11, v21, v15
	shl.4s	v14, v11, #20
	sri.4s	v14, v11, #12
	add.4s	v2, v2, v5
	add.4s	v13, v2, v14
	eor.16b	v2, v13, v24
	shl.4s	v24, v2, #24
	sri.4s	v24, v2, #8
	ldr	q3, [sp, #592]                  ; 16-byte Folded Reload
	add.4s	v2, v3, v12
	add.4s	v2, v8, v2
	eor.16b	v18, v2, v18
	rev32.8h	v18, v18
	add.4s	v1, v1, v18
	eor.16b	v8, v1, v8
	shl.4s	v12, v8, #20
	sri.4s	v12, v8, #12
	add.4s	v2, v2, v31
	mov.16b	v27, v31
	str	q31, [sp, #352]                 ; 16-byte Folded Spill
	add.4s	v11, v2, v12
	eor.16b	v2, v11, v18
	shl.4s	v18, v2, #24
	sri.4s	v18, v2, #8
	add.4s	v1, v18, v1
	eor.16b	v2, v1, v12
	shl.4s	v15, v2, #25
	sri.4s	v15, v2, #7
	add.4s	v2, v24, v21
	eor.16b	v21, v2, v14
	shl.4s	v14, v21, #25
	sri.4s	v14, v21, #7
	ldr	q31, [sp, #496]                 ; 16-byte Folded Reload
	add.4s	v21, v9, v31
	add.4s	v21, v21, v22
	eor.16b	v18, v18, v21
	rev32.8h	v18, v18
	add.4s	v2, v2, v18
	eor.16b	v22, v2, v22
	shl.4s	v8, v22, #20
	sri.4s	v8, v22, #12
	add.4s	v21, v21, v26
	add.4s	v12, v21, v8
	eor.16b	v21, v12, v18
	shl.4s	v18, v21, #24
	sri.4s	v18, v21, #8
	add.4s	v21, v2, v18
	eor.16b	v2, v21, v8
	shl.4s	v8, v2, #25
	sri.4s	v8, v2, #7
	add.4s	v2, v10, v6
	add.4s	v2, v2, v14
	eor.16b	v22, v2, v23
	rev32.8h	v22, v22
	add.4s	v1, v1, v22
	eor.16b	v23, v1, v14
	shl.4s	v9, v23, #20
	sri.4s	v9, v23, #12
	ldr	q17, [sp, #464]                 ; 16-byte Folded Reload
	add.4s	v2, v2, v17
	add.4s	v14, v2, v9
	eor.16b	v2, v14, v22
	shl.4s	v22, v2, #24
	sri.4s	v22, v2, #8
	ldr	q28, [sp, #528]                 ; 16-byte Folded Reload
	add.4s	v2, v13, v28
	add.4s	v2, v2, v15
	eor.16b	v23, v2, v25
	rev32.8h	v25, v23
	add.4s	v0, v0, v25
	eor.16b	v23, v0, v15
	shl.4s	v15, v23, #20
	sri.4s	v15, v23, #12
	add.4s	v23, v22, v1
	eor.16b	v1, v23, v9
	shl.4s	v10, v1, #25
	sri.4s	v10, v1, #7
	add.4s	v1, v2, v30
	add.4s	v13, v1, v15
	eor.16b	v1, v13, v25
	shl.4s	v2, v1, #24
	sri.4s	v2, v1, #8
	add.4s	v0, v2, v0
	eor.16b	v1, v0, v15
	shl.4s	v9, v1, #25
	sri.4s	v9, v1, #7
	add.4s	v1, v7, v11
	add.4s	v1, v19, v1
	eor.16b	v24, v1, v24
	rev32.8h	v24, v24
	add.4s	v20, v20, v24
	eor.16b	v19, v20, v19
	shl.4s	v25, v19, #20
	sri.4s	v25, v19, #12
	ldr	q5, [sp, #400]                  ; 16-byte Folded Reload
	add.4s	v1, v1, v5
	add.4s	v15, v1, v25
	eor.16b	v1, v15, v24
	shl.4s	v19, v1, #24
	sri.4s	v19, v1, #8
	add.4s	v1, v19, v20
	eor.16b	v20, v1, v25
	shl.4s	v24, v20, #25
	sri.4s	v24, v20, #7
	add.4s	v20, v12, v16
	add.4s	v20, v20, v24
	eor.16b	v22, v20, v22
	rev32.8h	v22, v22
	add.4s	v12, v0, v22
	eor.16b	v0, v12, v24
	shl.4s	v11, v0, #20
	sri.4s	v11, v0, #12
	add.4s	v0, v3, v20
	add.4s	v20, v11, v0
	eor.16b	v0, v20, v22
	shl.4s	v24, v0, #24
	sri.4s	v24, v0, #8
	add.4s	v0, v29, v14
	add.4s	v0, v8, v0
	eor.16b	v2, v2, v0
	rev32.8h	v2, v2
	add.4s	v22, v1, v2
	eor.16b	v1, v22, v8
	shl.4s	v8, v1, #20
	sri.4s	v8, v1, #12
	add.4s	v0, v0, v6
	add.4s	v1, v0, v8
	eor.16b	v0, v1, v2
	shl.4s	v25, v0, #24
	sri.4s	v25, v0, #8
	add.4s	v3, v25, v22
	eor.16b	v2, v3, v8
	shl.4s	v14, v2, #25
	sri.4s	v14, v2, #7
	add.4s	v8, v12, v24
	eor.16b	v2, v8, v11
	shl.4s	v22, v2, #25
	sri.4s	v22, v2, #7
	add.4s	v2, v27, v13
	add.4s	v2, v10, v2
	eor.16b	v19, v19, v2
	rev32.8h	v19, v19
	add.4s	v11, v21, v19
	eor.16b	v21, v11, v10
	shl.4s	v10, v21, #20
	sri.4s	v10, v21, #12
	ldr	q27, [sp, #368]                 ; 16-byte Folded Reload
	add.4s	v2, v2, v27
	add.4s	v12, v2, v10
	eor.16b	v2, v12, v19
	shl.4s	v21, v2, #24
	sri.4s	v21, v2, #8
	ldr	q16, [sp, #576]                 ; 16-byte Folded Reload
	add.4s	v2, v16, v15
	add.4s	v2, v9, v2
	eor.16b	v18, v2, v18
	rev32.8h	v18, v18
	add.4s	v19, v23, v18
	eor.16b	v23, v19, v9
	shl.4s	v13, v23, #20
	sri.4s	v13, v23, #12
	add.4s	v2, v2, v30
	mov.16b	v6, v30
	add.4s	v9, v2, v13
	eor.16b	v2, v9, v18
	shl.4s	v18, v2, #24
	sri.4s	v18, v2, #8
	add.4s	v2, v21, v11
	eor.16b	v23, v2, v10
	shl.4s	v11, v23, #25
	sri.4s	v11, v23, #7
	add.4s	v19, v18, v19
	eor.16b	v23, v19, v13
	shl.4s	v13, v23, #25
	sri.4s	v13, v23, #7
	ldr	q30, [sp, #512]                 ; 16-byte Folded Reload
	add.4s	v20, v20, v30
	add.4s	v20, v20, v14
	eor.16b	v18, v18, v20
	rev32.8h	v18, v18
	add.4s	v2, v2, v18
	eor.16b	v23, v2, v14
	shl.4s	v14, v23, #20
	sri.4s	v14, v23, #12
	add.4s	v20, v20, v17
	add.4s	v10, v20, v14
	eor.16b	v20, v10, v18
	shl.4s	v18, v20, #24
	sri.4s	v18, v20, #8
	add.4s	v20, v2, v18
	eor.16b	v2, v20, v14
	shl.4s	v23, v2, #25
	sri.4s	v23, v2, #7
	mov.16b	v4, v28
	add.4s	v1, v1, v28
	add.4s	v1, v1, v11
	eor.16b	v2, v1, v24
	rev32.8h	v2, v2
	add.4s	v14, v19, v2
	eor.16b	v19, v14, v11
	shl.4s	v15, v19, #20
	sri.4s	v15, v19, #12
	add.4s	v19, v12, v26
	add.4s	v19, v19, v13
	eor.16b	v24, v19, v25
	rev32.8h	v24, v24
	add.4s	v25, v8, v24
	eor.16b	v8, v25, v13
	shl.4s	v12, v8, #20
	sri.4s	v12, v8, #12
	ldr	q28, [sp, #432]                 ; 16-byte Folded Reload
	add.4s	v1, v1, v28
	add.4s	v1, v1, v15
	eor.16b	v2, v1, v2
	shl.4s	v8, v2, #24
	sri.4s	v8, v2, #8
	add.4s	v2, v19, v7
	add.4s	v11, v2, v12
	eor.16b	v2, v11, v24
	shl.4s	v0, v2, #24
	sri.4s	v0, v2, #8
	add.4s	v2, v0, v25
	eor.16b	v19, v2, v12
	shl.4s	v24, v19, #25
	sri.4s	v24, v19, #7
	add.4s	v19, v5, v9
	mov.16b	v17, v5
	add.4s	v19, v22, v19
	eor.16b	v21, v19, v21
	rev32.8h	v21, v21
	add.4s	v3, v3, v21
	eor.16b	v22, v3, v22
	shl.4s	v25, v22, #20
	sri.4s	v25, v22, #12
	add.4s	v19, v19, v31
	add.4s	v19, v19, v25
	eor.16b	v21, v19, v21
	shl.4s	v22, v21, #24
	sri.4s	v22, v21, #8
	add.4s	v3, v22, v3
	eor.16b	v21, v3, v25
	shl.4s	v25, v21, #25
	sri.4s	v25, v21, #7
	add.4s	v13, v8, v14
	eor.16b	v21, v13, v15
	shl.4s	v9, v21, #25
	sri.4s	v9, v21, #7
	add.4s	v21, v10, v29
	add.4s	v21, v21, v25
	eor.16b	v8, v21, v8
	rev32.8h	v8, v8
	add.4s	v2, v2, v8
	eor.16b	v25, v2, v25
	shl.4s	v14, v25, #20
	sri.4s	v14, v25, #12
	add.4s	v21, v16, v21
	add.4s	v10, v14, v21
	eor.16b	v21, v10, v8
	shl.4s	v25, v21, #24
	sri.4s	v25, v21, #8
	ldr	q5, [sp, #448]                  ; 16-byte Folded Reload
	add.4s	v1, v5, v1
	add.4s	v1, v23, v1
	eor.16b	v0, v0, v1
	rev32.8h	v0, v0
	add.4s	v3, v3, v0
	eor.16b	v21, v3, v23
	shl.4s	v23, v21, #20
	sri.4s	v23, v21, #12
	add.4s	v1, v1, v4
	mov.16b	v29, v4
	add.4s	v12, v1, v23
	eor.16b	v0, v12, v0
	shl.4s	v8, v0, #24
	sri.4s	v8, v0, #8
	add.4s	v21, v8, v3
	eor.16b	v0, v21, v23
	shl.4s	v1, v0, #25
	sri.4s	v1, v0, #7
	add.4s	v0, v6, v11
	add.4s	v0, v9, v0
	eor.16b	v3, v22, v0
	rev32.8h	v3, v3
	add.4s	v23, v20, v3
	eor.16b	v20, v23, v9
	shl.4s	v15, v20, #20
	sri.4s	v15, v20, #12
	add.4s	v20, v2, v25
	eor.16b	v2, v20, v14
	shl.4s	v22, v2, #25
	sri.4s	v22, v2, #7
	ldr	q2, [sp, #544]                  ; 16-byte Folded Reload
	add.4s	v0, v0, v2
	add.4s	v11, v0, v15
	eor.16b	v0, v11, v3
	shl.4s	v9, v0, #24
	sri.4s	v9, v0, #8
	add.4s	v0, v9, v23
	eor.16b	v2, v0, v15
	shl.4s	v3, v2, #25
	sri.4s	v3, v2, #7
	ldr	q4, [sp, #352]                  ; 16-byte Folded Reload
	add.4s	v2, v4, v19
	add.4s	v2, v24, v2
	eor.16b	v18, v2, v18
	rev32.8h	v18, v18
	add.4s	v19, v13, v18
	eor.16b	v23, v19, v24
	shl.4s	v24, v23, #20
	sri.4s	v24, v23, #12
	add.4s	v2, v2, v7
	add.4s	v13, v2, v24
	eor.16b	v2, v13, v18
	shl.4s	v18, v2, #24
	sri.4s	v18, v2, #8
	add.4s	v2, v18, v19
	eor.16b	v19, v2, v24
	shl.4s	v24, v19, #25
	sri.4s	v24, v19, #7
	ldr	q16, [sp, #592]                 ; 16-byte Folded Reload
	add.4s	v19, v10, v16
	add.4s	v19, v19, v1
	eor.16b	v18, v18, v19
	rev32.8h	v18, v18
	add.4s	v0, v0, v18
	eor.16b	v1, v0, v1
	shl.4s	v23, v1, #20
	sri.4s	v23, v1, #12
	add.4s	v1, v19, v28
	add.4s	v14, v1, v23
	eor.16b	v1, v14, v18
	shl.4s	v18, v1, #24
	sri.4s	v18, v1, #8
	add.4s	v19, v0, v18
	eor.16b	v0, v19, v23
	shl.4s	v23, v0, #25
	sri.4s	v23, v0, #7
	add.4s	v0, v12, v26
	add.4s	v0, v0, v3
	eor.16b	v1, v0, v25
	rev32.8h	v1, v1
	add.4s	v2, v2, v1
	eor.16b	v3, v2, v3
	shl.4s	v12, v3, #20
	sri.4s	v12, v3, #12
	mov.16b	v15, v27
	add.4s	v0, v0, v27
	add.4s	v10, v0, v12
	eor.16b	v0, v10, v1
	shl.4s	v25, v0, #24
	sri.4s	v25, v0, #8
	ldr	q27, [sp, #464]                 ; 16-byte Folded Reload
	add.4s	v0, v11, v27
	add.4s	v0, v0, v24
	eor.16b	v1, v0, v8
	rev32.8h	v1, v1
	add.4s	v3, v20, v1
	eor.16b	v20, v3, v24
	shl.4s	v8, v20, #20
	sri.4s	v8, v20, #12
	add.4s	v0, v0, v17
	add.4s	v24, v0, v8
	eor.16b	v0, v24, v1
	shl.4s	v1, v0, #24
	sri.4s	v1, v0, #8
	add.4s	v0, v1, v3
	eor.16b	v3, v0, v8
	shl.4s	v20, v3, #25
	sri.4s	v20, v3, #7
	add.4s	v3, v31, v13
	mov.16b	v13, v31
	add.4s	v3, v22, v3
	eor.16b	v8, v3, v9
	rev32.8h	v8, v8
	add.4s	v9, v21, v8
	eor.16b	v21, v9, v22
	shl.4s	v22, v21, #20
	sri.4s	v22, v21, #12
	mov.16b	v11, v30
	add.4s	v3, v3, v30
	add.4s	v21, v3, v22
	eor.16b	v3, v21, v8
	shl.4s	v8, v3, #24
	sri.4s	v8, v3, #8
	add.4s	v3, v8, v9
	eor.16b	v22, v3, v22
	shl.4s	v9, v22, #25
	sri.4s	v9, v22, #7
	add.4s	v31, v14, v5
	add.4s	v22, v25, v2
	eor.16b	v2, v22, v12
	add.4s	v16, v29, v10
	shl.4s	v10, v2, #25
	sri.4s	v10, v2, #7
	add.4s	v2, v31, v9
	eor.16b	v25, v2, v25
	add.4s	v16, v23, v16
	eor.16b	v1, v1, v16
	rev32.8h	v1, v1
	add.4s	v2, v4, v2
	add.4s	v3, v3, v1
	eor.16b	v23, v3, v23
	shl.4s	v30, v23, #20
	sri.4s	v30, v23, #12
	rev32.8h	v23, v25
	add.4s	v0, v0, v23
	eor.16b	v25, v0, v9
	add.4s	v4, v16, v26
	shl.4s	v31, v25, #20
	add.4s	v16, v4, v30
	eor.16b	v1, v16, v1
	sri.4s	v31, v25, #12
	shl.4s	v4, v1, #24
	sri.4s	v4, v1, #8
	add.4s	v2, v31, v2
	eor.16b	v1, v2, v23
	add.4s	v24, v7, v24
	shl.4s	v23, v1, #24
	add.4s	v3, v4, v3
	eor.16b	v25, v3, v30
	add.4s	v24, v10, v24
	eor.16b	v26, v8, v24
	rev32.8h	v26, v26
	sri.4s	v23, v1, #8
	add.4s	v30, v19, v26
	eor.16b	v1, v30, v10
	shl.4s	v8, v1, #20
	sri.4s	v8, v1, #12
	shl.4s	v9, v25, #25
	sri.4s	v9, v25, #7
	ldr	q1, [sp, #480]                  ; 16-byte Folded Reload
	add.4s	v1, v24, v1
	add.4s	v19, v1, v8
	eor.16b	v1, v19, v26
	add.4s	v7, v6, v21
	shl.4s	v6, v1, #24
	sri.4s	v6, v1, #8
	add.4s	v1, v20, v7
	eor.16b	v7, v1, v18
	rev32.8h	v21, v7
	add.4s	v22, v22, v21
	eor.16b	v7, v22, v20
	shl.4s	v20, v7, #20
	sri.4s	v20, v7, #12
	add.4s	v18, v0, v23
	add.4s	v0, v1, v17
	add.4s	v7, v0, v20
	eor.16b	v0, v7, v21
	eor.16b	v21, v18, v31
	shl.4s	v24, v0, #24
	sri.4s	v24, v0, #8
	shl.4s	v1, v21, #25
	add.4s	v0, v6, v30
	eor.16b	v25, v0, v8
	sri.4s	v1, v21, #7
	add.4s	v21, v24, v22
	eor.16b	v20, v21, v20
	shl.4s	v22, v20, #25
	sri.4s	v22, v20, #7
	shl.4s	v20, v25, #25
	sri.4s	v20, v25, #7
	ldr	q5, [sp, #576]                  ; 16-byte Folded Reload
	add.4s	v2, v2, v5
	add.4s	v2, v2, v9
	eor.16b	v24, v24, v2
	rev32.8h	v24, v24
	add.4s	v0, v0, v24
	eor.16b	v25, v0, v9
	shl.4s	v26, v25, #20
	sri.4s	v26, v25, #12
	add.4s	v2, v2, v15
	add.4s	v16, v16, v27
	add.4s	v16, v16, v20
	eor.16b	v23, v16, v23
	ldr	q5, [sp, #544]                  ; 16-byte Folded Reload
	add.4s	v16, v16, v5
	add.4s	v2, v2, v26
	eor.16b	v24, v2, v24
	rev32.8h	v23, v23
	add.4s	v19, v19, v28
	shl.4s	v5, v24, #24
	add.4s	v21, v21, v23
	eor.16b	v20, v21, v20
	add.4s	v19, v19, v22
	eor.16b	v4, v19, v4
	rev32.8h	v4, v4
	sri.4s	v5, v24, #8
	add.4s	v18, v18, v4
	eor.16b	v22, v18, v22
	shl.4s	v24, v22, #20
	sri.4s	v24, v22, #12
	shl.4s	v22, v20, #20
	sri.4s	v22, v20, #12
	add.4s	v19, v19, v13
	add.4s	v19, v19, v24
	eor.16b	v4, v19, v4
	shl.4s	v20, v4, #24
	sri.4s	v20, v4, #8
	add.4s	v0, v0, v5
	eor.16b	v4, v0, v26
	add.4s	v7, v11, v7
	shl.4s	v25, v4, #25
	add.4s	v16, v16, v22
	eor.16b	v23, v16, v23
	sri.4s	v25, v4, #7
	shl.4s	v4, v23, #24
	add.4s	v7, v1, v7
	eor.16b	v6, v7, v6
	rev32.8h	v6, v6
	add.4s	v3, v3, v6
	sri.4s	v4, v23, #8
	eor.16b	v1, v3, v1
	shl.4s	v23, v1, #20
	sri.4s	v23, v1, #12
	add.4s	v1, v4, v21
	add.4s	v18, v20, v18
	eor.16b	v21, v1, v22
	eor.16b	v22, v18, v24
	ldr	q17, [sp, #592]                 ; 16-byte Folded Reload
	add.4s	v7, v7, v17
	shl.4s	v17, v22, #25
	add.4s	v7, v7, v23
	eor.16b	v6, v7, v6
	sri.4s	v17, v22, #7
	shl.4s	v22, v6, #24
	sri.4s	v22, v6, #8
	shl.4s	v6, v21, #25
	add.4s	v3, v22, v3
	eor.16b	v23, v3, v23
	sri.4s	v6, v21, #7
	shl.4s	v21, v23, #25
	sri.4s	v21, v23, #7
	eor.16b	v9, v18, v2
	eor.16b	v23, v3, v16
	eor.16b	v13, v19, v0
	eor.16b	v29, v7, v1
	eor.16b	v24, v21, v4
	eor.16b	v8, v20, v25
	eor.16b	v11, v22, v6
	eor.16b	v15, v17, v5
	add	x7, x7, #64
	mov	x20, x19
	cbnz	x19, LBB1_3
; %bb.4:                                ;   in Loop: Header=BB1_2 Depth=1
	stp	q9, q23, [sp, #96]              ; 32-byte Folded Spill
	stp	q24, q29, [sp, #192]            ; 32-byte Folded Spill
	stp	q8, q13, [sp, #128]             ; 32-byte Folded Spill
	stp	q11, q15, [sp, #160]            ; 32-byte Folded Spill
	mov	x7, #0                          ; =0x0
	dup.4s	v23, w9
	dup.4s	v27, w10
	dup.4s	v30, w11
	dup.4s	v31, w12
	dup.4s	v15, w13
	dup.4s	v29, w14
	dup.4s	v8, w15
	dup.4s	v9, w16
	stp	q9, q8, [sp, #224]              ; 32-byte Folded Spill
	stp	q29, q15, [sp, #256]            ; 32-byte Folded Spill
LBB1_5:                                 ;   Parent Loop BB1_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	add	x20, x0, x7
	cmp	x7, #960
	cset	w19, eq
	ldr	q0, [x20, #4096]
	ldr	q1, [x20, #5120]
	ldr	q2, [x20, #6144]
	ldr	q3, [x20, #7168]
	trn1.4s	v5, v0, v1
	trn2.4s	v4, v0, v1
	trn1.4s	v1, v2, v3
	trn2.4s	v2, v2, v3
	zip2.2d	v25, v5, v1
	zip2.2d	v26, v4, v2
	ldr	q3, [x20, #4112]
	ldr	q6, [x20, #5136]
	ldr	q7, [x20, #6160]
	ldr	q16, [x20, #7184]
	trn1.4s	v17, v3, v6
	trn2.4s	v11, v3, v6
	trn1.4s	v6, v7, v16
	trn2.4s	v16, v7, v16
	zip2.2d	v13, v17, v6
	zip2.2d	v12, v11, v16
	ldr	q7, [x20, #4128]
	mov.d	v5[1], v1[0]
	str	q5, [sp, #576]                  ; 16-byte Folded Spill
	ldr	q1, [x20, #5152]
	ldr	q18, [x20, #6176]
	ldr	q19, [x20, #7200]
	trn1.4s	v3, v7, v1
	mov.d	v4[1], v2[0]
	trn2.4s	v10, v7, v1
	trn1.4s	v1, v18, v19
	trn2.4s	v18, v18, v19
	zip2.2d	v7, v3, v1
	mov.d	v17[1], v6[0]
	zip2.2d	v2, v10, v18
	ldr	q6, [x20, #4144]
	mov.d	v11[1], v16[0]
	mov.16b	v14, v11
	stp	q4, q11, [sp, #480]             ; 32-byte Folded Spill
	ldr	q16, [x20, #5168]
	ldr	q19, [x20, #6192]
	ldr	q20, [x20, #7216]
	trn1.4s	v21, v6, v16
	mov.d	v3[1], v1[0]
	str	q3, [sp, #448]                  ; 16-byte Folded Spill
	trn2.4s	v11, v6, v16
	trn1.4s	v22, v19, v20
	trn2.4s	v19, v19, v20
	zip2.2d	v6, v21, v22
	stp	q2, q6, [sp, #544]              ; 32-byte Folded Spill
	mov.d	v10[1], v18[0]
	mov.16b	v16, v21
	zip2.2d	v0, v11, v19
	str	q0, [sp, #384]                  ; 16-byte Folded Spill
	cmp	x7, #0
	mov.d	v16[1], v22[0]
	cset	w20, eq
	orr	w19, w20, w19, lsl #1
	dup.4s	v20, w19
	mov.d	v11[1], v19[0]
	add.4s	v18, v9, v31
	add.4s	v18, v18, v5
	rev32.8h	v21, v18
	ldp	q1, q0, [sp, #224]              ; 32-byte Folded Reload
	add.4s	v22, v21, v1
	eor.16b	v19, v22, v31
	shl.4s	v24, v19, #20
	sri.4s	v24, v19, #12
	add.4s	v18, v18, v4
	add.4s	v19, v18, v24
	add.4s	v18, v8, v30
	add.4s	v18, v18, v25
	mov.16b	v4, v25
	eor.16b	v21, v19, v21
	rev32.8h	v25, v18
	add.4s	v9, v25, v0
	mov.16b	v2, v23
	eor.16b	v23, v9, v30
	shl.4s	v28, v23, #20
	sri.4s	v28, v23, #12
	shl.4s	v8, v21, #24
	add.4s	v18, v18, v26
	mov.16b	v5, v26
	add.4s	v23, v18, v28
	eor.16b	v18, v23, v25
	shl.4s	v31, v18, #24
	sri.4s	v31, v18, #8
	sri.4s	v8, v21, #8
	add.4s	v30, v31, v9
	eor.16b	v18, v30, v28
	shl.4s	v25, v18, #25
	add.4s	v21, v29, v27
	add.4s	v28, v21, v17
	sri.4s	v25, v18, #7
	movi.4s	v0, #64
	eor.16b	v18, v28, v0
	rev32.8h	v29, v18
	ldr	q0, [sp, #256]                  ; 16-byte Folded Reload
	add.4s	v26, v29, v0
	eor.16b	v18, v26, v27
	shl.4s	v1, v18, #20
	add.4s	v21, v22, v8
	sri.4s	v1, v18, #12
	add.4s	v18, v28, v14
	add.4s	v18, v18, v1
	eor.16b	v22, v18, v29
	shl.4s	v27, v22, #24
	eor.16b	v24, v21, v24
	sri.4s	v27, v22, #8
	shl.4s	v29, v24, #25
	add.4s	v22, v15, v2
	str	q13, [sp, #528]                 ; 16-byte Folded Spill
	add.4s	v22, v22, v13
	eor.16b	v20, v22, v20
	sri.4s	v29, v24, #7
	rev32.8h	v20, v20
	ldr	q0, [sp, #272]                  ; 16-byte Folded Reload
	add.4s	v24, v20, v0
	eor.16b	v28, v24, v2
	shl.4s	v0, v28, #20
	sri.4s	v0, v28, #12
	str	q12, [sp, #464]                 ; 16-byte Folded Spill
	add.4s	v22, v22, v12
	add.4s	v9, v22, v0
	eor.16b	v20, v9, v20
	shl.4s	v22, v20, #24
	sri.4s	v22, v20, #8
	add.4s	v20, v22, v24
	eor.16b	v0, v20, v0
	shl.4s	v24, v0, #25
	sri.4s	v24, v0, #7
	add.4s	v0, v27, v26
	eor.16b	v1, v0, v1
	shl.4s	v28, v1, #25
	sri.4s	v28, v1, #7
	add.4s	v1, v19, v3
	add.4s	v1, v1, v25
	eor.16b	v19, v22, v1
	rev32.8h	v19, v19
	add.4s	v0, v0, v19
	eor.16b	v22, v0, v25
	shl.4s	v25, v22, #20
	sri.4s	v25, v22, #12
	add.4s	v1, v1, v10
	mov.16b	v15, v10
	add.4s	v1, v1, v25
	eor.16b	v19, v1, v19
	shl.4s	v26, v19, #24
	sri.4s	v26, v19, #8
	add.4s	v2, v0, v26
	eor.16b	v22, v2, v25
	shl.4s	v19, v22, #25
	sri.4s	v19, v22, #7
	mov.16b	v3, v7
	str	q7, [sp, #432]                  ; 16-byte Folded Spill
	add.4s	v22, v23, v7
	add.4s	v22, v22, v28
	eor.16b	v23, v22, v8
	rev32.8h	v23, v23
	add.4s	v25, v20, v23
	eor.16b	v20, v25, v28
	shl.4s	v28, v20, #20
	sri.4s	v28, v20, #12
	ldr	q7, [sp, #544]                  ; 16-byte Folded Reload
	add.4s	v20, v22, v7
	add.4s	v20, v20, v28
	eor.16b	v22, v20, v23
	shl.4s	v8, v22, #24
	sri.4s	v8, v22, #8
	mov.16b	v14, v16
	add.4s	v18, v18, v16
	add.4s	v18, v18, v24
	eor.16b	v22, v18, v31
	rev32.8h	v0, v22
	add.4s	v21, v21, v0
	eor.16b	v22, v21, v24
	shl.4s	v24, v22, #20
	sri.4s	v24, v22, #12
	add.4s	v31, v8, v25
	eor.16b	v22, v31, v28
	shl.4s	v23, v22, #25
	sri.4s	v23, v22, #7
	mov.16b	v16, v11
	add.4s	v18, v18, v11
	add.4s	v22, v18, v24
	eor.16b	v0, v22, v0
	shl.4s	v18, v0, #24
	sri.4s	v18, v0, #8
	add.4s	v0, v18, v21
	eor.16b	v24, v0, v24
	shl.4s	v21, v24, #25
	sri.4s	v21, v24, #7
	add.4s	v24, v6, v9
	add.4s	v24, v29, v24
	eor.16b	v25, v24, v27
	rev32.8h	v25, v25
	add.4s	v27, v30, v25
	eor.16b	v28, v27, v29
	shl.4s	v29, v28, #20
	sri.4s	v29, v28, #12
	ldr	q11, [sp, #384]                 ; 16-byte Folded Reload
	add.4s	v24, v24, v11
	add.4s	v24, v24, v29
	eor.16b	v25, v24, v25
	shl.4s	v28, v25, #24
	sri.4s	v28, v25, #8
	add.4s	v27, v28, v27
	eor.16b	v25, v27, v29
	shl.4s	v29, v25, #25
	sri.4s	v29, v25, #7
	add.4s	v1, v1, v4
	str	q4, [sp, #400]                  ; 16-byte Folded Spill
	add.4s	v1, v1, v29
	eor.16b	v25, v1, v8
	rev32.8h	v30, v25
	add.4s	v0, v0, v30
	eor.16b	v25, v0, v29
	shl.4s	v29, v25, #20
	sri.4s	v29, v25, #12
	add.4s	v1, v13, v1
	add.4s	v25, v29, v1
	eor.16b	v1, v25, v30
	shl.4s	v8, v1, #24
	sri.4s	v8, v1, #8
	mov.16b	v13, v5
	add.4s	v1, v5, v20
	add.4s	v1, v19, v1
	eor.16b	v18, v18, v1
	rev32.8h	v20, v18
	add.4s	v27, v27, v20
	eor.16b	v18, v27, v19
	shl.4s	v19, v18, #20
	sri.4s	v19, v18, #12
	add.4s	v1, v1, v3
	add.4s	v18, v1, v19
	eor.16b	v1, v18, v20
	shl.4s	v9, v1, #24
	sri.4s	v9, v1, #8
	add.4s	v27, v9, v27
	eor.16b	v1, v27, v19
	shl.4s	v19, v1, #25
	sri.4s	v19, v1, #7
	add.4s	v0, v0, v8
	eor.16b	v1, v0, v29
	shl.4s	v29, v1, #25
	sri.4s	v29, v1, #7
	add.4s	v1, v12, v22
	add.4s	v1, v23, v1
	eor.16b	v20, v28, v1
	rev32.8h	v22, v20
	add.4s	v2, v2, v22
	eor.16b	v20, v2, v23
	shl.4s	v23, v20, #20
	sri.4s	v23, v20, #12
	ldr	q3, [sp, #576]                  ; 16-byte Folded Reload
	add.4s	v1, v1, v3
	add.4s	v20, v1, v23
	eor.16b	v1, v20, v22
	shl.4s	v30, v1, #24
	sri.4s	v30, v1, #8
	str	q17, [sp, #352]                 ; 16-byte Folded Spill
	add.4s	v1, v17, v24
	add.4s	v1, v21, v1
	eor.16b	v22, v1, v26
	rev32.8h	v22, v22
	add.4s	v24, v31, v22
	eor.16b	v21, v24, v21
	shl.4s	v26, v21, #20
	sri.4s	v26, v21, #12
	str	q16, [sp, #592]                 ; 16-byte Folded Spill
	add.4s	v1, v1, v16
	add.4s	v21, v1, v26
	eor.16b	v1, v21, v22
	shl.4s	v22, v1, #24
	sri.4s	v22, v1, #8
	add.4s	v1, v30, v2
	eor.16b	v2, v1, v23
	shl.4s	v28, v2, #25
	sri.4s	v28, v2, #7
	add.4s	v2, v22, v24
	eor.16b	v23, v2, v26
	shl.4s	v24, v23, #25
	sri.4s	v24, v23, #7
	ldr	q10, [sp, #480]                 ; 16-byte Folded Reload
	add.4s	v23, v25, v10
	add.4s	v23, v23, v19
	eor.16b	v22, v22, v23
	rev32.8h	v22, v22
	add.4s	v1, v1, v22
	eor.16b	v19, v1, v19
	shl.4s	v25, v19, #20
	sri.4s	v25, v19, #12
	add.4s	v19, v23, v7
	add.4s	v23, v19, v25
	eor.16b	v19, v23, v22
	shl.4s	v26, v19, #24
	sri.4s	v26, v19, #8
	add.4s	v1, v1, v26
	eor.16b	v19, v1, v25
	shl.4s	v31, v19, #25
	sri.4s	v31, v19, #7
	mov.16b	v7, v14
	add.4s	v18, v18, v14
	add.4s	v18, v18, v28
	eor.16b	v19, v18, v8
	rev32.8h	v19, v19
	add.4s	v2, v2, v19
	eor.16b	v22, v2, v28
	shl.4s	v25, v22, #20
	sri.4s	v25, v22, #12
	add.4s	v20, v20, v15
	mov.16b	v14, v15
	add.4s	v20, v20, v24
	eor.16b	v22, v20, v9
	rev32.8h	v22, v22
	add.4s	v0, v0, v22
	eor.16b	v24, v0, v24
	shl.4s	v28, v24, #20
	sri.4s	v28, v24, #12
	ldr	q15, [sp, #496]                 ; 16-byte Folded Reload
	add.4s	v18, v18, v15
	add.4s	v9, v18, v25
	eor.16b	v18, v9, v19
	shl.4s	v24, v18, #24
	sri.4s	v24, v18, #8
	ldr	q5, [sp, #560]                  ; 16-byte Folded Reload
	add.4s	v18, v20, v5
	add.4s	v19, v18, v28
	eor.16b	v18, v19, v22
	shl.4s	v22, v18, #24
	sri.4s	v22, v18, #8
	add.4s	v20, v22, v0
	eor.16b	v0, v20, v28
	shl.4s	v8, v0, #25
	sri.4s	v8, v0, #7
	add.4s	v0, v11, v21
	mov.16b	v3, v11
	add.4s	v0, v29, v0
	eor.16b	v18, v0, v30
	rev32.8h	v21, v18
	add.4s	v27, v27, v21
	eor.16b	v18, v27, v29
	shl.4s	v28, v18, #20
	sri.4s	v28, v18, #12
	ldr	q11, [sp, #448]                 ; 16-byte Folded Reload
	add.4s	v0, v0, v11
	add.4s	v18, v0, v28
	eor.16b	v0, v18, v21
	shl.4s	v21, v0, #24
	sri.4s	v21, v0, #8
	add.4s	v27, v21, v27
	eor.16b	v0, v27, v28
	shl.4s	v28, v0, #25
	sri.4s	v28, v0, #7
	add.4s	v0, v24, v2
	eor.16b	v2, v0, v25
	shl.4s	v25, v2, #25
	sri.4s	v25, v2, #7
	add.4s	v2, v23, v13
	str	q13, [sp, #368]                 ; 16-byte Folded Spill
	add.4s	v2, v2, v28
	eor.16b	v23, v2, v24
	rev32.8h	v23, v23
	add.4s	v24, v20, v23
	eor.16b	v20, v24, v28
	shl.4s	v28, v20, #20
	sri.4s	v28, v20, #12
	add.4s	v2, v17, v2
	add.4s	v20, v28, v2
	eor.16b	v2, v20, v23
	shl.4s	v30, v2, #24
	sri.4s	v30, v2, #8
	ldr	q12, [sp, #432]                 ; 16-byte Folded Reload
	add.4s	v2, v12, v9
	add.4s	v2, v31, v2
	eor.16b	v22, v22, v2
	rev32.8h	v23, v22
	add.4s	v27, v27, v23
	eor.16b	v22, v27, v31
	shl.4s	v29, v22, #20
	sri.4s	v29, v22, #12
	add.4s	v2, v2, v7
	mov.16b	v17, v7
	add.4s	v22, v2, v29
	eor.16b	v2, v22, v23
	shl.4s	v31, v2, #24
	sri.4s	v31, v2, #8
	add.4s	v27, v31, v27
	eor.16b	v2, v27, v29
	shl.4s	v29, v2, #25
	sri.4s	v29, v2, #7
	add.4s	v2, v16, v19
	add.4s	v2, v25, v2
	eor.16b	v19, v21, v2
	rev32.8h	v19, v19
	add.4s	v1, v1, v19
	eor.16b	v21, v1, v25
	shl.4s	v25, v21, #20
	sri.4s	v25, v21, #12
	add.4s	v21, v24, v30
	eor.16b	v23, v21, v28
	shl.4s	v28, v23, #25
	sri.4s	v28, v23, #7
	add.4s	v2, v2, v4
	add.4s	v23, v2, v25
	eor.16b	v2, v23, v19
	shl.4s	v9, v2, #24
	sri.4s	v9, v2, #8
	add.4s	v1, v9, v1
	eor.16b	v2, v1, v25
	shl.4s	v25, v2, #25
	sri.4s	v25, v2, #7
	ldr	q4, [sp, #464]                  ; 16-byte Folded Reload
	add.4s	v2, v4, v18
	add.4s	v2, v8, v2
	eor.16b	v18, v2, v26
	rev32.8h	v18, v18
	add.4s	v0, v0, v18
	eor.16b	v19, v0, v8
	shl.4s	v26, v19, #20
	sri.4s	v26, v19, #12
	add.4s	v2, v2, v5
	add.4s	v24, v2, v26
	eor.16b	v2, v24, v18
	shl.4s	v18, v2, #24
	sri.4s	v18, v2, #8
	add.4s	v0, v18, v0
	eor.16b	v2, v0, v26
	shl.4s	v8, v2, #25
	sri.4s	v8, v2, #7
	ldr	q2, [sp, #528]                  ; 16-byte Folded Reload
	add.4s	v2, v20, v2
	add.4s	v2, v2, v29
	eor.16b	v18, v18, v2
	rev32.8h	v19, v18
	add.4s	v1, v1, v19
	eor.16b	v18, v1, v29
	shl.4s	v20, v18, #20
	sri.4s	v20, v18, #12
	add.4s	v2, v2, v15
	add.4s	v18, v2, v20
	eor.16b	v2, v18, v19
	shl.4s	v26, v2, #24
	sri.4s	v26, v2, #8
	add.4s	v29, v1, v26
	eor.16b	v1, v29, v20
	shl.4s	v19, v1, #25
	sri.4s	v19, v1, #7
	str	q14, [sp, #416]                 ; 16-byte Folded Spill
	add.4s	v1, v22, v14
	add.4s	v1, v1, v25
	eor.16b	v2, v1, v30
	rev32.8h	v2, v2
	add.4s	v0, v0, v2
	eor.16b	v20, v0, v25
	shl.4s	v30, v20, #20
	sri.4s	v30, v20, #12
	ldr	q16, [sp, #576]                 ; 16-byte Folded Reload
	add.4s	v1, v1, v16
	add.4s	v20, v1, v30
	eor.16b	v1, v20, v2
	shl.4s	v2, v1, #24
	sri.4s	v2, v1, #8
	ldr	q6, [sp, #544]                  ; 16-byte Folded Reload
	add.4s	v1, v23, v6
	add.4s	v1, v1, v8
	eor.16b	v22, v1, v31
	rev32.8h	v23, v22
	add.4s	v21, v21, v23
	eor.16b	v22, v21, v8
	shl.4s	v25, v22, #20
	sri.4s	v25, v22, #12
	mov.16b	v15, v3
	add.4s	v1, v1, v3
	add.4s	v22, v1, v25
	eor.16b	v1, v22, v23
	shl.4s	v8, v1, #24
	sri.4s	v8, v1, #8
	add.4s	v1, v8, v21
	eor.16b	v23, v1, v25
	shl.4s	v21, v23, #25
	sri.4s	v21, v23, #7
	add.4s	v23, v11, v24
	mov.16b	v7, v11
	add.4s	v23, v28, v23
	eor.16b	v24, v23, v9
	rev32.8h	v24, v24
	add.4s	v27, v27, v24
	eor.16b	v25, v27, v28
	shl.4s	v28, v25, #20
	sri.4s	v28, v25, #12
	add.4s	v23, v23, v10
	add.4s	v23, v23, v28
	eor.16b	v24, v23, v24
	shl.4s	v25, v24, #24
	sri.4s	v25, v24, #8
	add.4s	v0, v2, v0
	eor.16b	v24, v0, v30
	shl.4s	v30, v24, #25
	sri.4s	v30, v24, #7
	add.4s	v9, v25, v27
	eor.16b	v24, v9, v28
	shl.4s	v27, v24, #25
	sri.4s	v27, v24, #7
	add.4s	v18, v18, v12
	mov.16b	v11, v12
	add.4s	v18, v18, v27
	eor.16b	v2, v18, v2
	rev32.8h	v2, v2
	add.4s	v1, v1, v2
	eor.16b	v24, v1, v27
	shl.4s	v27, v24, #20
	sri.4s	v27, v24, #12
	add.4s	v18, v4, v18
	add.4s	v24, v27, v18
	eor.16b	v2, v24, v2
	shl.4s	v31, v2, #24
	sri.4s	v31, v2, #8
	add.4s	v1, v1, v31
	eor.16b	v2, v1, v27
	shl.4s	v27, v2, #25
	sri.4s	v27, v2, #7
	mov.16b	v12, v17
	str	q17, [sp, #512]                 ; 16-byte Folded Spill
	add.4s	v2, v17, v20
	add.4s	v2, v19, v2
	eor.16b	v18, v8, v2
	rev32.8h	v18, v18
	add.4s	v28, v9, v18
	eor.16b	v19, v28, v19
	shl.4s	v8, v19, #20
	sri.4s	v8, v19, #12
	add.4s	v2, v2, v14
	add.4s	v20, v2, v8
	eor.16b	v2, v20, v18
	shl.4s	v9, v2, #24
	sri.4s	v9, v2, #8
	add.4s	v28, v9, v28
	eor.16b	v2, v28, v8
	shl.4s	v19, v2, #25
	sri.4s	v19, v2, #7
	add.4s	v2, v5, v22
	add.4s	v2, v30, v2
	eor.16b	v18, v25, v2
	rev32.8h	v18, v18
	add.4s	v25, v29, v18
	eor.16b	v22, v25, v30
	shl.4s	v29, v22, #20
	sri.4s	v29, v22, #12
	add.4s	v2, v2, v13
	add.4s	v22, v2, v29
	eor.16b	v2, v22, v18
	shl.4s	v8, v2, #24
	sri.4s	v8, v2, #8
	ldr	q13, [sp, #592]                 ; 16-byte Folded Reload
	add.4s	v2, v13, v23
	add.4s	v2, v21, v2
	eor.16b	v18, v2, v26
	rev32.8h	v23, v18
	add.4s	v0, v0, v23
	eor.16b	v18, v0, v21
	shl.4s	v21, v18, #20
	sri.4s	v21, v18, #12
	add.4s	v2, v2, v3
	add.4s	v18, v2, v21
	eor.16b	v2, v18, v23
	shl.4s	v23, v2, #24
	sri.4s	v23, v2, #8
	add.4s	v0, v23, v0
	eor.16b	v2, v0, v21
	shl.4s	v21, v2, #25
	sri.4s	v21, v2, #7
	add.4s	v2, v8, v25
	eor.16b	v25, v2, v29
	shl.4s	v30, v25, #25
	sri.4s	v30, v25, #7
	ldr	q10, [sp, #352]                 ; 16-byte Folded Reload
	add.4s	v24, v24, v10
	add.4s	v24, v24, v19
	eor.16b	v23, v23, v24
	rev32.8h	v23, v23
	add.4s	v2, v2, v23
	eor.16b	v19, v2, v19
	shl.4s	v25, v19, #20
	sri.4s	v25, v19, #12
	add.4s	v19, v24, v16
	add.4s	v24, v19, v25
	eor.16b	v19, v24, v23
	shl.4s	v26, v19, #24
	sri.4s	v26, v19, #8
	add.4s	v29, v2, v26
	eor.16b	v2, v29, v25
	shl.4s	v19, v2, #25
	sri.4s	v19, v2, #7
	add.4s	v2, v20, v6
	add.4s	v2, v2, v30
	eor.16b	v20, v2, v31
	rev32.8h	v23, v20
	add.4s	v0, v0, v23
	eor.16b	v20, v0, v30
	shl.4s	v25, v20, #20
	sri.4s	v25, v20, #12
	ldr	q16, [sp, #400]                 ; 16-byte Folded Reload
	add.4s	v2, v2, v16
	add.4s	v20, v2, v25
	eor.16b	v2, v20, v23
	shl.4s	v30, v2, #24
	sri.4s	v30, v2, #8
	ldr	q17, [sp, #496]                 ; 16-byte Folded Reload
	add.4s	v2, v22, v17
	add.4s	v2, v2, v21
	eor.16b	v22, v2, v9
	rev32.8h	v9, v22
	add.4s	v1, v1, v9
	eor.16b	v21, v1, v21
	shl.4s	v31, v21, #20
	sri.4s	v31, v21, #12
	add.4s	v0, v30, v0
	eor.16b	v21, v0, v25
	shl.4s	v23, v21, #25
	sri.4s	v23, v21, #7
	mov.16b	v14, v7
	add.4s	v2, v2, v7
	add.4s	v22, v2, v31
	eor.16b	v2, v22, v9
	shl.4s	v25, v2, #24
	sri.4s	v25, v2, #8
	add.4s	v1, v25, v1
	eor.16b	v2, v1, v31
	shl.4s	v21, v2, #25
	sri.4s	v21, v2, #7
	ldr	q4, [sp, #480]                  ; 16-byte Folded Reload
	add.4s	v2, v4, v18
	add.4s	v2, v27, v2
	eor.16b	v18, v2, v8
	rev32.8h	v18, v18
	add.4s	v31, v28, v18
	eor.16b	v27, v31, v27
	shl.4s	v8, v27, #20
	sri.4s	v8, v27, #12
	ldr	q7, [sp, #528]                  ; 16-byte Folded Reload
	add.4s	v2, v2, v7
	add.4s	v28, v2, v8
	eor.16b	v2, v28, v18
	shl.4s	v27, v2, #24
	sri.4s	v27, v2, #8
	add.4s	v2, v27, v31
	eor.16b	v18, v2, v8
	shl.4s	v31, v18, #25
	sri.4s	v31, v18, #7
	add.4s	v18, v24, v12
	add.4s	v18, v18, v31
	eor.16b	v24, v18, v30
	rev32.8h	v24, v24
	add.4s	v30, v1, v24
	eor.16b	v1, v30, v31
	shl.4s	v31, v1, #20
	sri.4s	v31, v1, #12
	add.4s	v1, v13, v18
	add.4s	v18, v31, v1
	eor.16b	v1, v18, v24
	shl.4s	v8, v1, #24
	sri.4s	v8, v1, #8
	ldr	q3, [sp, #416]                  ; 16-byte Folded Reload
	add.4s	v1, v3, v20
	add.4s	v1, v19, v1
	eor.16b	v20, v25, v1
	rev32.8h	v20, v20
	add.4s	v2, v2, v20
	eor.16b	v19, v2, v19
	shl.4s	v24, v19, #20
	sri.4s	v24, v19, #12
	add.4s	v1, v1, v6
	add.4s	v25, v1, v24
	eor.16b	v1, v25, v20
	shl.4s	v9, v1, #24
	sri.4s	v9, v1, #8
	add.4s	v1, v9, v2
	eor.16b	v2, v1, v24
	shl.4s	v24, v2, #25
	sri.4s	v24, v2, #7
	add.4s	v19, v30, v8
	eor.16b	v2, v19, v31
	shl.4s	v30, v2, #25
	sri.4s	v30, v2, #7
	add.4s	v2, v15, v22
	add.4s	v2, v23, v2
	eor.16b	v20, v27, v2
	rev32.8h	v22, v20
	add.4s	v27, v29, v22
	eor.16b	v20, v27, v23
	shl.4s	v23, v20, #20
	sri.4s	v23, v20, #12
	add.4s	v2, v2, v11
	add.4s	v20, v2, v23
	eor.16b	v2, v20, v22
	shl.4s	v29, v2, #24
	sri.4s	v29, v2, #8
	ldr	q6, [sp, #560]                  ; 16-byte Folded Reload
	add.4s	v2, v6, v28
	add.4s	v2, v21, v2
	eor.16b	v22, v2, v26
	rev32.8h	v22, v22
	add.4s	v0, v0, v22
	eor.16b	v21, v0, v21
	shl.4s	v26, v21, #20
	sri.4s	v26, v21, #12
	add.4s	v2, v2, v14
	add.4s	v21, v2, v26
	eor.16b	v2, v21, v22
	shl.4s	v22, v2, #24
	sri.4s	v22, v2, #8
	add.4s	v2, v29, v27
	eor.16b	v23, v2, v23
	shl.4s	v27, v23, #25
	sri.4s	v27, v23, #7
	add.4s	v0, v22, v0
	eor.16b	v23, v0, v26
	shl.4s	v28, v23, #25
	sri.4s	v28, v23, #7
	ldr	q13, [sp, #464]                 ; 16-byte Folded Reload
	add.4s	v18, v18, v13
	add.4s	v18, v18, v24
	eor.16b	v22, v22, v18
	rev32.8h	v22, v22
	add.4s	v2, v2, v22
	eor.16b	v23, v2, v24
	shl.4s	v24, v23, #20
	sri.4s	v24, v23, #12
	add.4s	v18, v18, v16
	add.4s	v23, v18, v24
	eor.16b	v18, v23, v22
	shl.4s	v26, v18, #24
	sri.4s	v26, v18, #8
	add.4s	v2, v2, v26
	eor.16b	v18, v2, v24
	shl.4s	v31, v18, #25
	sri.4s	v31, v18, #7
	mov.16b	v15, v17
	add.4s	v18, v25, v17
	add.4s	v18, v18, v27
	eor.16b	v22, v18, v8
	rev32.8h	v22, v22
	add.4s	v0, v0, v22
	eor.16b	v24, v0, v27
	shl.4s	v27, v24, #20
	sri.4s	v27, v24, #12
	ldr	q5, [sp, #576]                  ; 16-byte Folded Reload
	add.4s	v20, v20, v5
	add.4s	v20, v20, v28
	eor.16b	v24, v20, v9
	rev32.8h	v24, v24
	add.4s	v19, v19, v24
	eor.16b	v25, v19, v28
	shl.4s	v28, v25, #20
	sri.4s	v28, v25, #12
	ldr	q12, [sp, #368]                 ; 16-byte Folded Reload
	add.4s	v18, v18, v12
	add.4s	v18, v18, v27
	eor.16b	v22, v18, v22
	shl.4s	v9, v22, #24
	sri.4s	v9, v22, #8
	add.4s	v20, v20, v4
	add.4s	v20, v20, v28
	eor.16b	v22, v20, v24
	shl.4s	v24, v22, #24
	sri.4s	v24, v22, #8
	add.4s	v19, v24, v19
	eor.16b	v22, v19, v28
	shl.4s	v8, v22, #25
	sri.4s	v8, v22, #7
	add.4s	v21, v7, v21
	add.4s	v21, v30, v21
	eor.16b	v22, v21, v29
	rev32.8h	v22, v22
	add.4s	v25, v1, v22
	eor.16b	v1, v25, v30
	shl.4s	v28, v1, #20
	sri.4s	v28, v1, #12
	add.4s	v1, v21, v10
	add.4s	v1, v1, v28
	eor.16b	v21, v1, v22
	shl.4s	v30, v21, #24
	sri.4s	v30, v21, #8
	add.4s	v21, v30, v25
	eor.16b	v22, v21, v28
	shl.4s	v28, v22, #25
	sri.4s	v28, v22, #7
	add.4s	v25, v9, v0
	eor.16b	v0, v25, v27
	shl.4s	v27, v0, #25
	sri.4s	v27, v0, #7
	add.4s	v0, v23, v3
	add.4s	v0, v0, v28
	eor.16b	v22, v0, v9
	rev32.8h	v23, v22
	add.4s	v29, v19, v23
	eor.16b	v19, v29, v28
	shl.4s	v28, v19, #20
	sri.4s	v28, v19, #12
	add.4s	v0, v6, v0
	add.4s	v22, v28, v0
	eor.16b	v0, v22, v23
	shl.4s	v9, v0, #24
	sri.4s	v9, v0, #8
	ldr	q3, [sp, #544]                  ; 16-byte Folded Reload
	add.4s	v0, v3, v18
	add.4s	v0, v31, v0
	eor.16b	v18, v24, v0
	rev32.8h	v18, v18
	add.4s	v21, v21, v18
	eor.16b	v19, v21, v31
	shl.4s	v23, v19, #20
	sri.4s	v23, v19, #12
	add.4s	v0, v0, v17
	add.4s	v24, v0, v23
	eor.16b	v0, v24, v18
	shl.4s	v19, v0, #24
	sri.4s	v19, v0, #8
	add.4s	v0, v19, v21
	eor.16b	v18, v0, v23
	shl.4s	v31, v18, #25
	sri.4s	v31, v18, #7
	add.4s	v18, v14, v20
	add.4s	v18, v27, v18
	eor.16b	v20, v30, v18
	rev32.8h	v20, v20
	add.4s	v2, v2, v20
	eor.16b	v21, v2, v27
	shl.4s	v27, v21, #20
	sri.4s	v27, v21, #12
	add.4s	v23, v29, v9
	eor.16b	v21, v23, v28
	shl.4s	v30, v21, #25
	sri.4s	v30, v21, #7
	ldr	q6, [sp, #512]                  ; 16-byte Folded Reload
	add.4s	v18, v18, v6
	add.4s	v28, v18, v27
	eor.16b	v18, v28, v20
	shl.4s	v21, v18, #24
	sri.4s	v21, v18, #8
	add.4s	v2, v21, v2
	eor.16b	v18, v2, v27
	shl.4s	v20, v18, #25
	sri.4s	v20, v18, #7
	ldr	q11, [sp, #384]                 ; 16-byte Folded Reload
	add.4s	v1, v11, v1
	add.4s	v1, v8, v1
	eor.16b	v18, v1, v26
	rev32.8h	v18, v18
	add.4s	v26, v25, v18
	eor.16b	v25, v26, v8
	shl.4s	v27, v25, #20
	sri.4s	v27, v25, #12
	add.4s	v1, v1, v4
	add.4s	v25, v1, v27
	eor.16b	v1, v25, v18
	shl.4s	v18, v1, #24
	sri.4s	v18, v1, #8
	add.4s	v1, v18, v26
	eor.16b	v26, v1, v27
	shl.4s	v29, v26, #25
	sri.4s	v29, v26, #7
	ldr	q6, [sp, #592]                  ; 16-byte Folded Reload
	add.4s	v22, v22, v6
	add.4s	v22, v22, v31
	eor.16b	v18, v18, v22
	rev32.8h	v18, v18
	add.4s	v2, v2, v18
	eor.16b	v26, v2, v31
	shl.4s	v31, v26, #20
	sri.4s	v31, v26, #12
	add.4s	v22, v22, v12
	add.4s	v22, v22, v31
	eor.16b	v18, v22, v18
	shl.4s	v26, v18, #24
	sri.4s	v26, v18, #8
	add.4s	v27, v2, v26
	eor.16b	v2, v27, v31
	shl.4s	v31, v2, #25
	sri.4s	v31, v2, #7
	add.4s	v2, v24, v5
	add.4s	v2, v2, v20
	eor.16b	v18, v2, v9
	rev32.8h	v18, v18
	add.4s	v1, v1, v18
	eor.16b	v20, v1, v20
	shl.4s	v24, v20, #20
	sri.4s	v24, v20, #12
	ldr	q17, [sp, #432]                 ; 16-byte Folded Reload
	add.4s	v2, v2, v17
	add.4s	v20, v2, v24
	eor.16b	v2, v20, v18
	shl.4s	v18, v2, #24
	sri.4s	v18, v2, #8
	ldr	q14, [sp, #400]                 ; 16-byte Folded Reload
	add.4s	v2, v28, v14
	add.4s	v2, v2, v29
	eor.16b	v19, v2, v19
	rev32.8h	v19, v19
	add.4s	v23, v23, v19
	eor.16b	v28, v23, v29
	shl.4s	v29, v28, #20
	sri.4s	v29, v28, #12
	add.4s	v2, v2, v7
	mov.16b	v16, v7
	add.4s	v8, v2, v29
	eor.16b	v2, v8, v19
	shl.4s	v9, v2, #24
	sri.4s	v9, v2, #8
	add.4s	v2, v9, v23
	eor.16b	v19, v2, v29
	shl.4s	v28, v19, #25
	sri.4s	v28, v19, #7
	add.4s	v19, v10, v25
	add.4s	v19, v30, v19
	eor.16b	v21, v19, v21
	rev32.8h	v21, v21
	add.4s	v0, v0, v21
	eor.16b	v23, v0, v30
	shl.4s	v25, v23, #20
	sri.4s	v25, v23, #12
	add.4s	v19, v19, v13
	add.4s	v19, v19, v25
	eor.16b	v21, v19, v21
	shl.4s	v23, v21, #24
	sri.4s	v23, v21, #8
	add.4s	v0, v23, v0
	eor.16b	v21, v0, v25
	shl.4s	v25, v21, #25
	sri.4s	v25, v21, #7
	add.4s	v21, v22, v3
	add.4s	v29, v18, v1
	eor.16b	v1, v29, v24
	add.4s	v20, v15, v20
	shl.4s	v22, v1, #25
	sri.4s	v22, v1, #7
	add.4s	v1, v21, v25
	eor.16b	v18, v1, v18
	add.4s	v20, v31, v20
	eor.16b	v21, v9, v20
	rev32.8h	v24, v21
	add.4s	v1, v11, v1
	add.4s	v0, v0, v24
	eor.16b	v3, v0, v31
	shl.4s	v31, v3, #20
	sri.4s	v31, v3, #12
	rev32.8h	v3, v18
	add.4s	v2, v2, v3
	eor.16b	v18, v2, v25
	add.4s	v5, v20, v5
	shl.4s	v25, v18, #20
	add.4s	v21, v5, v31
	eor.16b	v20, v21, v24
	sri.4s	v25, v18, #12
	shl.4s	v5, v20, #24
	sri.4s	v5, v20, #8
	add.4s	v1, v25, v1
	eor.16b	v3, v1, v3
	add.4s	v18, v4, v8
	shl.4s	v30, v3, #24
	add.4s	v4, v5, v0
	eor.16b	v0, v4, v31
	add.4s	v18, v22, v18
	eor.16b	v20, v23, v18
	rev32.8h	v20, v20
	sri.4s	v30, v3, #8
	add.4s	v24, v27, v20
	eor.16b	v3, v24, v22
	shl.4s	v22, v3, #20
	sri.4s	v22, v3, #12
	shl.4s	v27, v0, #25
	sri.4s	v27, v0, #7
	ldr	q0, [sp, #416]                  ; 16-byte Folded Reload
	add.4s	v0, v18, v0
	add.4s	v23, v0, v22
	eor.16b	v0, v23, v20
	ldr	q3, [sp, #448]                  ; 16-byte Folded Reload
	add.4s	v3, v3, v19
	shl.4s	v7, v0, #24
	sri.4s	v7, v0, #8
	add.4s	v0, v28, v3
	eor.16b	v3, v0, v26
	rev32.8h	v3, v3
	add.4s	v18, v29, v3
	eor.16b	v19, v18, v28
	shl.4s	v26, v19, #20
	sri.4s	v26, v19, #12
	add.4s	v20, v2, v30
	add.4s	v0, v0, v16
	add.4s	v19, v0, v26
	eor.16b	v0, v19, v3
	eor.16b	v2, v20, v25
	shl.4s	v25, v0, #24
	sri.4s	v25, v0, #8
	shl.4s	v3, v2, #25
	add.4s	v0, v7, v24
	eor.16b	v22, v0, v22
	sri.4s	v3, v2, #7
	add.4s	v24, v25, v18
	eor.16b	v2, v24, v26
	shl.4s	v26, v2, #25
	sri.4s	v26, v2, #7
	shl.4s	v28, v22, #25
	sri.4s	v28, v22, #7
	ldr	q2, [sp, #560]                  ; 16-byte Folded Reload
	add.4s	v1, v1, v2
	add.4s	v1, v1, v27
	eor.16b	v2, v25, v1
	rev32.8h	v2, v2
	add.4s	v0, v0, v2
	eor.16b	v6, v0, v27
	shl.4s	v22, v6, #20
	sri.4s	v22, v6, #12
	add.4s	v1, v1, v17
	add.4s	v6, v1, v22
	eor.16b	v1, v6, v2
	shl.4s	v2, v1, #24
	sri.4s	v2, v1, #8
	add.4s	v18, v0, v2
	eor.16b	v1, v18, v22
	add.4s	v21, v21, v14
	shl.4s	v0, v1, #25
	sri.4s	v0, v1, #7
	add.4s	v1, v21, v28
	eor.16b	v21, v1, v30
	rev32.8h	v21, v21
	add.4s	v22, v24, v21
	eor.16b	v24, v22, v28
	shl.4s	v25, v24, #20
	sri.4s	v25, v24, #12
	ldr	q16, [sp, #512]                 ; 16-byte Folded Reload
	add.4s	v1, v1, v16
	add.4s	v1, v1, v25
	eor.16b	v16, v1, v21
	add.4s	v21, v23, v12
	shl.4s	v23, v16, #24
	add.4s	v21, v21, v26
	eor.16b	v5, v21, v5
	rev32.8h	v5, v5
	add.4s	v20, v20, v5
	eor.16b	v24, v20, v26
	sri.4s	v23, v16, #8
	shl.4s	v16, v24, #20
	sri.4s	v16, v24, #12
	add.4s	v21, v21, v10
	add.4s	v21, v21, v16
	eor.16b	v5, v21, v5
	shl.4s	v24, v5, #24
	sri.4s	v24, v5, #8
	add.4s	v5, v24, v20
	eor.16b	v16, v5, v16
	add.4s	v19, v13, v19
	shl.4s	v20, v16, #25
	sri.4s	v20, v16, #7
	add.4s	v16, v3, v19
	eor.16b	v7, v16, v7
	rev32.8h	v7, v7
	add.4s	v4, v4, v7
	eor.16b	v3, v4, v3
	shl.4s	v19, v3, #20
	sri.4s	v19, v3, #12
	add.4s	v3, v23, v22
	ldr	q17, [sp, #592]                 ; 16-byte Folded Reload
	add.4s	v16, v16, v17
	add.4s	v16, v16, v19
	eor.16b	v7, v16, v7
	eor.16b	v17, v3, v25
	shl.4s	v22, v7, #24
	sri.4s	v22, v7, #8
	shl.4s	v7, v17, #25
	add.4s	v4, v22, v4
	eor.16b	v19, v4, v19
	sri.4s	v7, v17, #7
	shl.4s	v17, v19, #25
	sri.4s	v17, v19, #7
	eor.16b	v9, v5, v6
	eor.16b	v8, v4, v1
	eor.16b	v29, v21, v18
	eor.16b	v15, v16, v3
	eor.16b	v31, v17, v23
	eor.16b	v30, v24, v0
	eor.16b	v27, v22, v7
	eor.16b	v23, v20, v2
	add	x7, x7, #64
	cmp	x7, #1024
	b.ne	LBB1_5
; %bb.6:                                ;   in Loop: Header=BB1_2 Depth=1
	ldr	q0, [sp, #96]                   ; 16-byte Folded Reload
	uzp1.4s	v1, v0, v9
	mov.16b	v26, v0
	ldr	q0, [sp, #112]                  ; 16-byte Folded Reload
	uzp1.4s	v3, v0, v8
	str	q3, [sp, #576]                  ; 16-byte Folded Spill
	mov.16b	v14, v0
	ldr	q25, [sp, #144]                 ; 16-byte Folded Reload
	uzp1.4s	v7, v25, v29
	str	q7, [sp, #368]                  ; 16-byte Folded Spill
	ldp	q0, q2, [sp, #192]              ; 32-byte Folded Reload
	uzp1.4s	v6, v2, v15
	str	q6, [sp, #464]                  ; 16-byte Folded Spill
	uzp1.4s	v16, v0, v31
	stp	q1, q16, [sp, #432]             ; 32-byte Folded Spill
	dup.4s	v0, w17
	add.4s	v0, v1, v0
	mov.16b	v24, v15
	str	q15, [sp, #80]                  ; 16-byte Folded Spill
	rev32.8h	v1, v0
	dup.4s	v2, w16
	add.4s	v4, v1, v2
	dup.4s	v2, w12
	eor.16b	v2, v4, v2
	shl.4s	v5, v2, #20
	sri.4s	v5, v2, #12
	add.4s	v0, v0, v3
	add.4s	v3, v0, v5
	eor.16b	v0, v3, v1
	shl.4s	v2, v0, #24
	sri.4s	v2, v0, #8
	add.4s	v13, v4, v2
	eor.16b	v0, v13, v5
	shl.4s	v15, v0, #25
	dup.4s	v1, w4
	sri.4s	v15, v0, #7
	add.4s	v0, v7, v1
	rev32.8h	v1, v0
	dup.4s	v4, w15
	add.4s	v5, v1, v4
	dup.4s	v4, w11
	eor.16b	v4, v5, v4
	shl.4s	v7, v4, #20
	sri.4s	v7, v4, #12
	add.4s	v0, v0, v6
	add.4s	v4, v0, v7
	eor.16b	v0, v4, v1
	shl.4s	v6, v0, #24
	sri.4s	v6, v0, #8
	add.4s	v0, v5, v6
	str	q0, [sp, #496]                  ; 16-byte Folded Spill
	eor.16b	v0, v0, v7
	shl.4s	v7, v0, #25
	sri.4s	v7, v0, #7
	dup.4s	v0, w5
	add.4s	v0, v16, v0
	movi.4s	v5, #64
	eor.16b	v5, v0, v5
	rev32.8h	v16, v5
	dup.4s	v5, w14
	add.4s	v17, v16, v5
	dup.4s	v5, w10
	eor.16b	v5, v17, v5
	shl.4s	v18, v5, #20
	sri.4s	v18, v5, #12
	ldr	q1, [sp, #128]                  ; 16-byte Folded Reload
	uzp1.4s	v19, v1, v30
	str	q19, [sp, #512]                 ; 16-byte Folded Spill
	add.4s	v0, v0, v19
	add.4s	v5, v0, v18
	eor.16b	v16, v5, v16
	shl.4s	v0, v16, #24
	sri.4s	v0, v16, #8
	add.4s	v16, v17, v0
	eor.16b	v17, v16, v18
	shl.4s	v20, v17, #25
	sri.4s	v20, v17, #7
	ldp	q11, q10, [sp, #160]            ; 32-byte Folded Reload
	uzp1.4s	v12, v11, v27
	dup.4s	v17, w6
	add.4s	v17, v12, v17
	movi.4s	v18, #4
	eor.16b	v18, v17, v18
	rev32.8h	v18, v18
	dup.4s	v19, w13
	add.4s	v19, v18, v19
	dup.4s	v21, w9
	eor.16b	v21, v19, v21
	shl.4s	v22, v21, #20
	sri.4s	v22, v21, #12
	uzp1.4s	v28, v10, v23
	stp	q12, q28, [sp, #544]            ; 32-byte Folded Spill
	add.4s	v17, v17, v28
	add.4s	v17, v17, v22
	eor.16b	v18, v17, v18
	shl.4s	v21, v18, #24
	sri.4s	v21, v18, #8
	mov.16b	v28, v23
	str	q23, [sp, #64]                  ; 16-byte Folded Spill
	add.4s	v23, v19, v21
	eor.16b	v18, v23, v22
	shl.4s	v22, v18, #25
	sri.4s	v22, v18, #7
	uzp2.4s	v18, v26, v9
	str	q18, [sp, #592]                 ; 16-byte Folded Spill
	add.4s	v3, v3, v18
	add.4s	v3, v7, v3
	eor.16b	v18, v3, v21
	rev32.8h	v19, v18
	add.4s	v16, v16, v19
	eor.16b	v7, v16, v7
	shl.4s	v21, v7, #20
	sri.4s	v21, v7, #12
	uzp2.4s	v7, v14, v8
	str	q7, [sp, #528]                  ; 16-byte Folded Spill
	add.4s	v3, v3, v7
	add.4s	v18, v3, v21
	eor.16b	v3, v18, v19
	shl.4s	v7, v3, #24
	sri.4s	v7, v3, #8
	add.4s	v3, v16, v7
	eor.16b	v16, v3, v21
	shl.4s	v19, v16, #25
	sri.4s	v19, v16, #7
	uzp2.4s	v14, v25, v29
	add.4s	v4, v4, v14
	add.4s	v4, v20, v4
	eor.16b	v2, v4, v2
	rev32.8h	v2, v2
	add.4s	v16, v23, v2
	eor.16b	v20, v16, v20
	shl.4s	v21, v20, #20
	sri.4s	v21, v20, #12
	ldr	q20, [sp, #208]                 ; 16-byte Folded Reload
	uzp2.4s	v26, v20, v24
	add.4s	v4, v4, v26
	str	q26, [sp, #352]                 ; 16-byte Folded Spill
	add.4s	v20, v4, v21
	eor.16b	v2, v20, v2
	shl.4s	v4, v2, #24
	sri.4s	v4, v2, #8
	add.4s	v16, v16, v4
	eor.16b	v2, v16, v21
	shl.4s	v21, v2, #25
	sri.4s	v21, v2, #7
	ldr	q2, [sp, #192]                  ; 16-byte Folded Reload
	uzp2.4s	v2, v2, v31
	str	q2, [sp, #480]                  ; 16-byte Folded Spill
	add.4s	v2, v5, v2
	add.4s	v2, v22, v2
	eor.16b	v5, v2, v6
	rev32.8h	v5, v5
	add.4s	v6, v13, v5
	eor.16b	v22, v6, v22
	shl.4s	v23, v22, #20
	sri.4s	v23, v22, #12
	uzp2.4s	v13, v1, v30
	add.4s	v2, v2, v13
	add.4s	v22, v2, v23
	eor.16b	v2, v22, v5
	shl.4s	v24, v2, #24
	sri.4s	v24, v2, #8
	add.4s	v5, v6, v24
	eor.16b	v6, v5, v23
	shl.4s	v2, v6, #25
	sri.4s	v2, v6, #7
	uzp2.4s	v1, v11, v27
	str	q1, [sp, #384]                  ; 16-byte Folded Spill
	add.4s	v6, v17, v1
	add.4s	v6, v6, v15
	eor.16b	v0, v6, v0
	rev32.8h	v17, v0
	ldr	q0, [sp, #496]                  ; 16-byte Folded Reload
	add.4s	v1, v0, v17
	eor.16b	v0, v1, v15
	shl.4s	v23, v0, #20
	sri.4s	v23, v0, #12
	uzp2.4s	v0, v10, v28
	str	q0, [sp, #496]                  ; 16-byte Folded Spill
	add.4s	v0, v6, v0
	add.4s	v0, v0, v23
	eor.16b	v6, v0, v17
	shl.4s	v25, v6, #24
	sri.4s	v25, v6, #8
	add.4s	v1, v1, v25
	eor.16b	v6, v1, v23
	shl.4s	v17, v6, #25
	sri.4s	v17, v6, #7
	ldr	q6, [sp, #368]                  ; 16-byte Folded Reload
	add.4s	v6, v18, v6
	add.4s	v6, v6, v17
	eor.16b	v4, v6, v4
	rev32.8h	v4, v4
	add.4s	v5, v5, v4
	eor.16b	v17, v5, v17
	shl.4s	v18, v17, #20
	sri.4s	v18, v17, #12
	add.4s	v6, v12, v6
	add.4s	v17, v18, v6
	eor.16b	v6, v17, v4
	shl.4s	v4, v6, #24
	sri.4s	v4, v6, #8
	add.4s	v6, v5, v4
	eor.16b	v18, v6, v18
	shl.4s	v5, v18, #25
	sri.4s	v5, v18, #7
	ldr	q12, [sp, #464]                 ; 16-byte Folded Reload
	add.4s	v18, v12, v20
	add.4s	v18, v19, v18
	eor.16b	v20, v24, v18
	rev32.8h	v20, v20
	add.4s	v23, v1, v20
	eor.16b	v1, v23, v19
	shl.4s	v24, v1, #20
	sri.4s	v24, v1, #12
	mov.16b	v10, v14
	str	q14, [sp, #400]                 ; 16-byte Folded Spill
	add.4s	v1, v18, v14
	add.4s	v18, v1, v24
	eor.16b	v19, v18, v20
	shl.4s	v1, v19, #24
	sri.4s	v1, v19, #8
	add.4s	v19, v1, v23
	eor.16b	v20, v19, v24
	shl.4s	v23, v20, #25
	sri.4s	v23, v20, #7
	ldr	q20, [sp, #560]                 ; 16-byte Folded Reload
	add.4s	v20, v20, v22
	add.4s	v20, v21, v20
	eor.16b	v22, v25, v20
	rev32.8h	v22, v22
	add.4s	v24, v3, v22
	eor.16b	v3, v24, v21
	shl.4s	v21, v3, #20
	sri.4s	v21, v3, #12
	ldp	q3, q28, [sp, #432]             ; 32-byte Folded Reload
	add.4s	v3, v20, v3
	add.4s	v20, v3, v21
	eor.16b	v22, v20, v22
	shl.4s	v3, v22, #24
	sri.4s	v3, v22, #8
	add.4s	v22, v3, v24
	eor.16b	v21, v22, v21
	shl.4s	v24, v21, #25
	sri.4s	v24, v21, #7
	add.4s	v0, v28, v0
	add.4s	v0, v2, v0
	eor.16b	v7, v0, v7
	rev32.8h	v7, v7
	add.4s	v16, v16, v7
	eor.16b	v2, v16, v2
	shl.4s	v21, v2, #20
	sri.4s	v21, v2, #12
	mov.16b	v11, v13
	str	q13, [sp, #416]                 ; 16-byte Folded Spill
	add.4s	v0, v0, v13
	add.4s	v2, v0, v21
	eor.16b	v0, v2, v7
	shl.4s	v7, v0, #24
	sri.4s	v7, v0, #8
	add.4s	v16, v7, v16
	eor.16b	v0, v16, v21
	shl.4s	v25, v0, #25
	sri.4s	v25, v0, #7
	ldr	q0, [sp, #576]                  ; 16-byte Folded Reload
	add.4s	v0, v17, v0
	add.4s	v0, v0, v23
	eor.16b	v7, v7, v0
	rev32.8h	v7, v7
	add.4s	v22, v22, v7
	eor.16b	v17, v22, v23
	shl.4s	v23, v17, #20
	sri.4s	v23, v17, #12
	add.4s	v0, v0, v26
	add.4s	v17, v0, v23
	eor.16b	v0, v17, v7
	shl.4s	v21, v0, #24
	sri.4s	v21, v0, #8
	add.4s	v7, v22, v21
	eor.16b	v22, v7, v23
	shl.4s	v0, v22, #25
	sri.4s	v0, v22, #7
	ldr	q14, [sp, #480]                 ; 16-byte Folded Reload
	add.4s	v18, v18, v14
	add.4s	v18, v18, v24
	eor.16b	v4, v18, v4
	rev32.8h	v4, v4
	add.4s	v16, v16, v4
	eor.16b	v22, v16, v24
	shl.4s	v23, v22, #20
	sri.4s	v23, v22, #12
	ldr	q15, [sp, #512]                 ; 16-byte Folded Reload
	add.4s	v18, v18, v15
	add.4s	v18, v18, v23
	eor.16b	v4, v18, v4
	shl.4s	v22, v4, #24
	sri.4s	v22, v4, #8
	add.4s	v16, v22, v16
	eor.16b	v23, v16, v23
	shl.4s	v4, v23, #25
	sri.4s	v4, v23, #7
	ldr	q23, [sp, #528]                 ; 16-byte Folded Reload
	add.4s	v20, v20, v23
	add.4s	v20, v20, v25
	eor.16b	v1, v20, v1
	rev32.8h	v1, v1
	add.4s	v6, v6, v1
	eor.16b	v23, v6, v25
	shl.4s	v24, v23, #20
	sri.4s	v24, v23, #12
	ldr	q13, [sp, #384]                 ; 16-byte Folded Reload
	add.4s	v20, v20, v13
	add.4s	v20, v20, v24
	eor.16b	v1, v20, v1
	shl.4s	v23, v1, #24
	sri.4s	v23, v1, #8
	add.4s	v6, v23, v6
	eor.16b	v24, v6, v24
	shl.4s	v1, v24, #25
	sri.4s	v1, v24, #7
	ldr	q24, [sp, #496]                 ; 16-byte Folded Reload
	add.4s	v2, v24, v2
	add.4s	v2, v5, v2
	eor.16b	v3, v2, v3
	rev32.8h	v3, v3
	add.4s	v19, v19, v3
	eor.16b	v5, v19, v5
	shl.4s	v24, v5, #20
	sri.4s	v24, v5, #12
	ldr	q5, [sp, #592]                  ; 16-byte Folded Reload
	add.4s	v2, v2, v5
	add.4s	v5, v2, v24
	eor.16b	v2, v5, v3
	shl.4s	v25, v2, #24
	sri.4s	v25, v2, #8
	add.4s	v19, v25, v19
	eor.16b	v2, v19, v24
	shl.4s	v3, v2, #25
	sri.4s	v3, v2, #7
	add.4s	v2, v17, v12
	add.4s	v2, v2, v3
	eor.16b	v17, v2, v22
	rev32.8h	v17, v17
	add.4s	v24, v6, v17
	eor.16b	v3, v24, v3
	shl.4s	v26, v3, #20
	sri.4s	v26, v3, #12
	add.4s	v2, v28, v2
	add.4s	v22, v26, v2
	eor.16b	v2, v22, v17
	shl.4s	v6, v2, #24
	sri.4s	v6, v2, #8
	add.4s	v2, v24, v6
	eor.16b	v17, v2, v26
	shl.4s	v3, v17, #25
	sri.4s	v3, v17, #7
	add.4s	v17, v10, v18
	add.4s	v17, v0, v17
	eor.16b	v18, v23, v17
	rev32.8h	v18, v18
	add.4s	v23, v19, v18
	eor.16b	v0, v23, v0
	shl.4s	v24, v0, #20
	sri.4s	v24, v0, #12
	add.4s	v0, v17, v14
	add.4s	v0, v0, v24
	eor.16b	v17, v0, v18
	shl.4s	v19, v17, #24
	sri.4s	v19, v17, #8
	add.4s	v17, v19, v23
	eor.16b	v18, v17, v24
	shl.4s	v23, v18, #25
	sri.4s	v23, v18, #7
	add.4s	v18, v11, v20
	add.4s	v18, v4, v18
	eor.16b	v20, v25, v18
	rev32.8h	v20, v20
	add.4s	v24, v7, v20
	eor.16b	v4, v24, v4
	shl.4s	v25, v4, #20
	sri.4s	v25, v4, #12
	ldr	q4, [sp, #368]                  ; 16-byte Folded Reload
	add.4s	v4, v18, v4
	add.4s	v18, v4, v25
	eor.16b	v4, v18, v20
	shl.4s	v7, v4, #24
	sri.4s	v7, v4, #8
	add.4s	v4, v7, v24
	eor.16b	v20, v4, v25
	shl.4s	v24, v20, #25
	sri.4s	v24, v20, #7
	ldr	q10, [sp, #560]                 ; 16-byte Folded Reload
	add.4s	v5, v10, v5
	add.4s	v5, v1, v5
	eor.16b	v20, v5, v21
	rev32.8h	v21, v20
	add.4s	v16, v16, v21
	eor.16b	v1, v16, v1
	shl.4s	v25, v1, #20
	sri.4s	v25, v1, #12
	add.4s	v1, v5, v13
	mov.16b	v12, v13
	add.4s	v20, v1, v25
	eor.16b	v1, v20, v21
	shl.4s	v5, v1, #24
	sri.4s	v5, v1, #8
	add.4s	v16, v5, v16
	eor.16b	v1, v16, v25
	shl.4s	v25, v1, #25
	sri.4s	v25, v1, #7
	ldp	q11, q1, [sp, #528]             ; 32-byte Folded Reload
	add.4s	v1, v22, v1
	add.4s	v1, v1, v23
	eor.16b	v5, v5, v1
	rev32.8h	v5, v5
	add.4s	v4, v4, v5
	eor.16b	v21, v4, v23
	shl.4s	v23, v21, #20
	sri.4s	v23, v21, #12
	add.4s	v1, v1, v15
	add.4s	v22, v1, v23
	eor.16b	v5, v22, v5
	shl.4s	v1, v5, #24
	sri.4s	v1, v5, #8
	add.4s	v21, v4, v1
	eor.16b	v5, v21, v23
	shl.4s	v4, v5, #25
	sri.4s	v4, v5, #7
	add.4s	v0, v0, v11
	add.4s	v0, v0, v24
	eor.16b	v5, v0, v6
	rev32.8h	v5, v5
	add.4s	v6, v16, v5
	eor.16b	v16, v6, v24
	shl.4s	v24, v16, #20
	sri.4s	v24, v16, #12
	ldr	q28, [sp, #432]                 ; 16-byte Folded Reload
	add.4s	v0, v0, v28
	add.4s	v23, v0, v24
	eor.16b	v0, v23, v5
	shl.4s	v26, v0, #24
	sri.4s	v26, v0, #8
	add.4s	v5, v26, v6
	eor.16b	v0, v5, v24
	shl.4s	v16, v0, #25
	sri.4s	v16, v0, #7
	ldr	q15, [sp, #352]                 ; 16-byte Folded Reload
	add.4s	v0, v18, v15
	add.4s	v0, v0, v25
	eor.16b	v6, v0, v19
	rev32.8h	v6, v6
	add.4s	v2, v2, v6
	eor.16b	v18, v2, v25
	shl.4s	v19, v18, #20
	sri.4s	v19, v18, #12
	ldr	q13, [sp, #496]                 ; 16-byte Folded Reload
	add.4s	v0, v0, v13
	add.4s	v18, v0, v19
	eor.16b	v0, v18, v6
	shl.4s	v24, v0, #24
	sri.4s	v24, v0, #8
	add.4s	v2, v24, v2
	eor.16b	v6, v2, v19
	shl.4s	v0, v6, #25
	sri.4s	v0, v6, #7
	ldp	q14, q6, [sp, #576]             ; 32-byte Folded Reload
	add.4s	v6, v6, v20
	add.4s	v6, v3, v6
	eor.16b	v7, v6, v7
	rev32.8h	v7, v7
	add.4s	v19, v17, v7
	eor.16b	v3, v19, v3
	shl.4s	v20, v3, #20
	sri.4s	v20, v3, #12
	add.4s	v3, v6, v14
	add.4s	v17, v3, v20
	eor.16b	v3, v17, v7
	shl.4s	v25, v3, #24
	sri.4s	v25, v3, #8
	add.4s	v7, v25, v19
	eor.16b	v3, v7, v20
	shl.4s	v6, v3, #25
	sri.4s	v6, v3, #7
	ldr	q3, [sp, #400]                  ; 16-byte Folded Reload
	add.4s	v3, v22, v3
	add.4s	v3, v3, v6
	eor.16b	v19, v3, v26
	rev32.8h	v19, v19
	add.4s	v2, v2, v19
	eor.16b	v6, v2, v6
	shl.4s	v22, v6, #20
	sri.4s	v22, v6, #12
	add.4s	v3, v10, v3
	add.4s	v20, v22, v3
	eor.16b	v3, v20, v19
	shl.4s	v6, v3, #24
	sri.4s	v6, v3, #8
	add.4s	v2, v2, v6
	eor.16b	v19, v2, v22
	shl.4s	v3, v19, #25
	sri.4s	v3, v19, #7
	ldr	q10, [sp, #480]                 ; 16-byte Folded Reload
	add.4s	v19, v10, v23
	add.4s	v19, v4, v19
	eor.16b	v22, v24, v19
	rev32.8h	v23, v22
	add.4s	v24, v7, v23
	eor.16b	v4, v24, v4
	shl.4s	v26, v4, #20
	sri.4s	v26, v4, #12
	add.4s	v4, v19, v11
	add.4s	v22, v4, v26
	eor.16b	v4, v22, v23
	shl.4s	v7, v4, #24
	sri.4s	v7, v4, #8
	add.4s	v4, v7, v24
	eor.16b	v19, v4, v26
	shl.4s	v24, v19, #25
	sri.4s	v24, v19, #7
	add.4s	v18, v12, v18
	add.4s	v18, v16, v18
	eor.16b	v19, v25, v18
	rev32.8h	v19, v19
	add.4s	v21, v21, v19
	eor.16b	v16, v21, v16
	shl.4s	v23, v16, #20
	sri.4s	v23, v16, #12
	ldr	q16, [sp, #464]                 ; 16-byte Folded Reload
	add.4s	v16, v18, v16
	add.4s	v18, v16, v23
	eor.16b	v16, v18, v19
	shl.4s	v19, v16, #24
	sri.4s	v19, v16, #8
	add.4s	v16, v19, v21
	eor.16b	v21, v16, v23
	shl.4s	v25, v21, #25
	sri.4s	v25, v21, #7
	ldr	q11, [sp, #416]                 ; 16-byte Folded Reload
	add.4s	v17, v11, v17
	add.4s	v17, v0, v17
	eor.16b	v1, v17, v1
	rev32.8h	v1, v1
	add.4s	v5, v5, v1
	eor.16b	v0, v5, v0
	shl.4s	v21, v0, #20
	sri.4s	v21, v0, #12
	add.4s	v0, v17, v13
	add.4s	v23, v0, v21
	eor.16b	v0, v23, v1
	shl.4s	v1, v0, #24
	sri.4s	v1, v0, #8
	add.4s	v0, v1, v5
	eor.16b	v5, v0, v21
	shl.4s	v26, v5, #25
	sri.4s	v26, v5, #7
	ldr	q5, [sp, #448]                  ; 16-byte Folded Reload
	add.4s	v5, v20, v5
	add.4s	v5, v5, v24
	eor.16b	v1, v1, v5
	rev32.8h	v1, v1
	add.4s	v16, v16, v1
	eor.16b	v17, v16, v24
	shl.4s	v21, v17, #20
	sri.4s	v21, v17, #12
	add.4s	v5, v5, v28
	add.4s	v20, v5, v21
	eor.16b	v5, v20, v1
	shl.4s	v1, v5, #24
	sri.4s	v1, v5, #8
	add.4s	v16, v16, v1
	eor.16b	v5, v16, v21
	shl.4s	v17, v5, #25
	sri.4s	v17, v5, #7
	mov.16b	v13, v15
	add.4s	v5, v22, v15
	add.4s	v5, v5, v25
	eor.16b	v6, v5, v6
	rev32.8h	v6, v6
	add.4s	v0, v0, v6
	eor.16b	v21, v0, v25
	shl.4s	v24, v21, #20
	sri.4s	v24, v21, #12
	ldr	q28, [sp, #368]                 ; 16-byte Folded Reload
	add.4s	v5, v5, v28
	add.4s	v22, v5, v24
	eor.16b	v5, v22, v6
	shl.4s	v6, v5, #24
	sri.4s	v6, v5, #8
	add.4s	v5, v6, v0
	eor.16b	v0, v5, v24
	shl.4s	v21, v0, #25
	sri.4s	v21, v0, #7
	ldr	q0, [sp, #512]                  ; 16-byte Folded Reload
	add.4s	v0, v18, v0
	add.4s	v0, v0, v26
	eor.16b	v7, v0, v7
	rev32.8h	v7, v7
	add.4s	v2, v2, v7
	eor.16b	v18, v2, v26
	shl.4s	v24, v18, #20
	sri.4s	v24, v18, #12
	ldr	q12, [sp, #592]                 ; 16-byte Folded Reload
	add.4s	v0, v0, v12
	add.4s	v18, v0, v24
	eor.16b	v0, v18, v7
	shl.4s	v7, v0, #24
	sri.4s	v7, v0, #8
	add.4s	v2, v7, v2
	eor.16b	v24, v2, v24
	shl.4s	v0, v24, #25
	sri.4s	v0, v24, #7
	add.4s	v23, v14, v23
	add.4s	v23, v3, v23
	eor.16b	v19, v23, v19
	rev32.8h	v19, v19
	add.4s	v4, v4, v19
	eor.16b	v3, v4, v3
	shl.4s	v24, v3, #20
	sri.4s	v24, v3, #12
	ldr	q3, [sp, #544]                  ; 16-byte Folded Reload
	add.4s	v3, v23, v3
	add.4s	v23, v3, v24
	eor.16b	v3, v23, v19
	shl.4s	v19, v3, #24
	sri.4s	v19, v3, #8
	add.4s	v4, v19, v4
	eor.16b	v3, v4, v24
	shl.4s	v24, v3, #25
	sri.4s	v24, v3, #7
	add.4s	v3, v20, v10
	add.4s	v3, v3, v24
	eor.16b	v6, v3, v6
	rev32.8h	v6, v6
	add.4s	v2, v2, v6
	eor.16b	v20, v2, v24
	shl.4s	v24, v20, #20
	sri.4s	v24, v20, #12
	add.4s	v3, v11, v3
	add.4s	v20, v24, v3
	eor.16b	v3, v20, v6
	shl.4s	v6, v3, #24
	sri.4s	v6, v3, #8
	add.4s	v2, v2, v6
	eor.16b	v24, v2, v24
	shl.4s	v3, v24, #25
	sri.4s	v3, v24, #7
	ldr	q15, [sp, #528]                 ; 16-byte Folded Reload
	add.4s	v22, v15, v22
	add.4s	v22, v17, v22
	eor.16b	v7, v7, v22
	rev32.8h	v7, v7
	add.4s	v4, v4, v7
	eor.16b	v17, v4, v17
	shl.4s	v24, v17, #20
	sri.4s	v24, v17, #12
	add.4s	v17, v22, v13
	add.4s	v22, v17, v24
	eor.16b	v17, v22, v7
	shl.4s	v7, v17, #24
	sri.4s	v7, v17, #8
	add.4s	v4, v7, v4
	eor.16b	v17, v4, v24
	shl.4s	v24, v17, #25
	sri.4s	v24, v17, #7
	ldr	q17, [sp, #496]                 ; 16-byte Folded Reload
	add.4s	v17, v17, v18
	add.4s	v17, v21, v17
	eor.16b	v18, v19, v17
	rev32.8h	v19, v18
	add.4s	v16, v16, v19
	eor.16b	v18, v16, v21
	shl.4s	v21, v18, #20
	sri.4s	v21, v18, #12
	ldp	q26, q18, [sp, #384]            ; 32-byte Folded Reload
	add.4s	v17, v17, v18
	add.4s	v18, v17, v21
	eor.16b	v17, v18, v19
	shl.4s	v19, v17, #24
	sri.4s	v19, v17, #8
	add.4s	v16, v19, v16
	eor.16b	v17, v16, v21
	shl.4s	v21, v17, #25
	sri.4s	v21, v17, #7
	add.4s	v17, v26, v23
	add.4s	v17, v0, v17
	eor.16b	v1, v17, v1
	rev32.8h	v1, v1
	add.4s	v5, v5, v1
	eor.16b	v0, v5, v0
	shl.4s	v25, v0, #20
	sri.4s	v25, v0, #12
	add.4s	v0, v17, v12
	add.4s	v23, v0, v25
	eor.16b	v0, v23, v1
	shl.4s	v1, v0, #24
	sri.4s	v1, v0, #8
	add.4s	v0, v1, v5
	eor.16b	v5, v0, v25
	shl.4s	v25, v5, #25
	sri.4s	v25, v5, #7
	ldr	q5, [sp, #560]                  ; 16-byte Folded Reload
	add.4s	v5, v20, v5
	add.4s	v5, v5, v24
	eor.16b	v1, v1, v5
	rev32.8h	v1, v1
	add.4s	v16, v16, v1
	eor.16b	v17, v16, v24
	shl.4s	v24, v17, #20
	sri.4s	v24, v17, #12
	mov.16b	v11, v28
	add.4s	v5, v5, v28
	add.4s	v20, v5, v24
	eor.16b	v5, v20, v1
	shl.4s	v1, v5, #24
	sri.4s	v1, v5, #8
	add.4s	v16, v16, v1
	eor.16b	v5, v16, v24
	shl.4s	v17, v5, #25
	sri.4s	v17, v5, #7
	ldr	q14, [sp, #512]                 ; 16-byte Folded Reload
	add.4s	v5, v22, v14
	add.4s	v5, v5, v21
	eor.16b	v6, v5, v6
	rev32.8h	v6, v6
	add.4s	v0, v0, v6
	eor.16b	v21, v0, v21
	shl.4s	v24, v21, #20
	sri.4s	v24, v21, #12
	ldr	q12, [sp, #464]                 ; 16-byte Folded Reload
	add.4s	v5, v5, v12
	add.4s	v22, v5, v24
	eor.16b	v5, v22, v6
	shl.4s	v6, v5, #24
	sri.4s	v6, v5, #8
	add.4s	v5, v6, v0
	eor.16b	v0, v5, v24
	shl.4s	v21, v0, #25
	sri.4s	v21, v0, #7
	ldr	q13, [sp, #432]                 ; 16-byte Folded Reload
	add.4s	v0, v18, v13
	add.4s	v0, v0, v25
	eor.16b	v7, v0, v7
	rev32.8h	v7, v7
	add.4s	v2, v2, v7
	eor.16b	v18, v2, v25
	shl.4s	v24, v18, #20
	sri.4s	v24, v18, #12
	ldr	q25, [sp, #576]                 ; 16-byte Folded Reload
	add.4s	v0, v0, v25
	add.4s	v18, v0, v24
	eor.16b	v0, v18, v7
	shl.4s	v7, v0, #24
	sri.4s	v7, v0, #8
	add.4s	v2, v7, v2
	eor.16b	v24, v2, v24
	shl.4s	v0, v24, #25
	sri.4s	v0, v24, #7
	ldr	q24, [sp, #544]                 ; 16-byte Folded Reload
	add.4s	v23, v24, v23
	add.4s	v23, v3, v23
	eor.16b	v19, v23, v19
	rev32.8h	v19, v19
	add.4s	v4, v4, v19
	eor.16b	v3, v4, v3
	shl.4s	v24, v3, #20
	sri.4s	v24, v3, #12
	ldr	q10, [sp, #448]                 ; 16-byte Folded Reload
	add.4s	v3, v23, v10
	add.4s	v23, v3, v24
	eor.16b	v3, v23, v19
	shl.4s	v19, v3, #24
	sri.4s	v19, v3, #8
	add.4s	v4, v19, v4
	eor.16b	v3, v4, v24
	shl.4s	v24, v3, #25
	sri.4s	v24, v3, #7
	add.4s	v3, v20, v15
	add.4s	v3, v3, v24
	eor.16b	v6, v3, v6
	rev32.8h	v6, v6
	add.4s	v2, v2, v6
	eor.16b	v20, v2, v24
	shl.4s	v24, v20, #20
	sri.4s	v24, v20, #12
	add.4s	v3, v26, v3
	add.4s	v20, v24, v3
	eor.16b	v3, v20, v6
	shl.4s	v6, v3, #24
	sri.4s	v6, v3, #8
	add.4s	v2, v2, v6
	eor.16b	v24, v2, v24
	shl.4s	v3, v24, #25
	sri.4s	v3, v24, #7
	ldr	q15, [sp, #352]                 ; 16-byte Folded Reload
	add.4s	v22, v15, v22
	add.4s	v22, v17, v22
	eor.16b	v7, v7, v22
	rev32.8h	v7, v7
	add.4s	v4, v4, v7
	eor.16b	v17, v4, v17
	shl.4s	v24, v17, #20
	sri.4s	v24, v17, #12
	add.4s	v17, v22, v14
	add.4s	v22, v17, v24
	eor.16b	v17, v22, v7
	shl.4s	v7, v17, #24
	sri.4s	v7, v17, #8
	add.4s	v4, v7, v4
	eor.16b	v17, v4, v24
	shl.4s	v24, v17, #25
	sri.4s	v24, v17, #7
	ldr	q17, [sp, #592]                 ; 16-byte Folded Reload
	add.4s	v17, v17, v18
	add.4s	v17, v21, v17
	eor.16b	v18, v19, v17
	rev32.8h	v19, v18
	add.4s	v16, v16, v19
	eor.16b	v18, v16, v21
	shl.4s	v21, v18, #20
	sri.4s	v21, v18, #12
	ldp	q18, q14, [sp, #480]            ; 32-byte Folded Reload
	add.4s	v17, v17, v18
	add.4s	v18, v17, v21
	eor.16b	v17, v18, v19
	shl.4s	v19, v17, #24
	sri.4s	v19, v17, #8
	add.4s	v16, v19, v16
	eor.16b	v17, v16, v21
	shl.4s	v21, v17, #25
	sri.4s	v21, v17, #7
	add.4s	v17, v14, v23
	add.4s	v17, v0, v17
	eor.16b	v1, v17, v1
	rev32.8h	v1, v1
	add.4s	v5, v5, v1
	eor.16b	v0, v5, v0
	shl.4s	v23, v0, #20
	sri.4s	v23, v0, #12
	add.4s	v0, v17, v25
	add.4s	v0, v0, v23
	eor.16b	v1, v0, v1
	shl.4s	v17, v1, #24
	sri.4s	v17, v1, #8
	add.4s	v5, v17, v5
	eor.16b	v1, v5, v23
	shl.4s	v23, v1, #25
	sri.4s	v23, v1, #7
	ldr	q1, [sp, #416]                  ; 16-byte Folded Reload
	add.4s	v1, v20, v1
	add.4s	v1, v1, v24
	eor.16b	v17, v17, v1
	rev32.8h	v17, v17
	add.4s	v16, v16, v17
	eor.16b	v20, v16, v24
	shl.4s	v24, v20, #20
	sri.4s	v24, v20, #12
	add.4s	v1, v1, v12
	add.4s	v20, v1, v24
	eor.16b	v17, v20, v17
	shl.4s	v1, v17, #24
	sri.4s	v1, v17, #8
	add.4s	v26, v16, v1
	eor.16b	v24, v26, v24
	shl.4s	v17, v24, #25
	sri.4s	v17, v24, #7
	mov.16b	v16, v13
	add.4s	v22, v22, v13
	add.4s	v22, v22, v21
	eor.16b	v6, v22, v6
	rev32.8h	v6, v6
	add.4s	v5, v5, v6
	eor.16b	v21, v5, v21
	shl.4s	v24, v21, #20
	sri.4s	v24, v21, #12
	ldr	q28, [sp, #400]                 ; 16-byte Folded Reload
	add.4s	v21, v22, v28
	add.4s	v22, v21, v24
	eor.16b	v6, v22, v6
	shl.4s	v25, v6, #24
	sri.4s	v25, v6, #8
	add.4s	v5, v25, v5
	eor.16b	v6, v5, v24
	shl.4s	v21, v6, #25
	sri.4s	v21, v6, #7
	add.4s	v6, v18, v11
	add.4s	v6, v6, v23
	eor.16b	v7, v6, v7
	rev32.8h	v7, v7
	add.4s	v2, v2, v7
	eor.16b	v18, v2, v23
	shl.4s	v23, v18, #20
	sri.4s	v23, v18, #12
	ldr	q13, [sp, #544]                 ; 16-byte Folded Reload
	add.4s	v6, v6, v13
	add.4s	v18, v6, v23
	eor.16b	v6, v18, v7
	shl.4s	v7, v6, #24
	sri.4s	v7, v6, #8
	add.4s	v6, v7, v2
	eor.16b	v23, v6, v23
	shl.4s	v2, v23, #25
	sri.4s	v2, v23, #7
	add.4s	v0, v10, v0
	mov.16b	v12, v10
	add.4s	v0, v3, v0
	eor.16b	v19, v0, v19
	rev32.8h	v19, v19
	add.4s	v4, v4, v19
	eor.16b	v3, v4, v3
	shl.4s	v23, v3, #20
	sri.4s	v23, v3, #12
	ldr	q10, [sp, #560]                 ; 16-byte Folded Reload
	add.4s	v0, v0, v10
	add.4s	v3, v0, v23
	eor.16b	v0, v3, v19
	shl.4s	v19, v0, #24
	sri.4s	v19, v0, #8
	add.4s	v4, v19, v4
	eor.16b	v0, v4, v23
	shl.4s	v23, v0, #25
	sri.4s	v23, v0, #7
	add.4s	v0, v20, v15
	add.4s	v0, v0, v23
	eor.16b	v20, v0, v25
	rev32.8h	v24, v20
	add.4s	v25, v6, v24
	eor.16b	v6, v25, v23
	shl.4s	v23, v6, #20
	sri.4s	v23, v6, #12
	add.4s	v0, v14, v0
	add.4s	v20, v23, v0
	eor.16b	v0, v20, v24
	shl.4s	v6, v0, #24
	sri.4s	v6, v0, #8
	add.4s	v14, v25, v6
	eor.16b	v23, v14, v23
	shl.4s	v0, v23, #25
	sri.4s	v0, v23, #7
	ldr	q23, [sp, #512]                 ; 16-byte Folded Reload
	add.4s	v22, v23, v22
	add.4s	v22, v17, v22
	eor.16b	v7, v7, v22
	rev32.8h	v7, v7
	add.4s	v4, v4, v7
	eor.16b	v17, v4, v17
	shl.4s	v23, v17, #20
	sri.4s	v23, v17, #12
	add.4s	v17, v22, v16
	add.4s	v17, v17, v23
	eor.16b	v22, v17, v7
	shl.4s	v7, v22, #24
	sri.4s	v7, v22, #8
	add.4s	v4, v7, v4
	eor.16b	v22, v4, v23
	shl.4s	v23, v22, #25
	sri.4s	v23, v22, #7
	ldr	q16, [sp, #576]                 ; 16-byte Folded Reload
	add.4s	v18, v16, v18
	add.4s	v18, v21, v18
	eor.16b	v19, v19, v18
	rev32.8h	v19, v19
	add.4s	v22, v26, v19
	eor.16b	v16, v22, v21
	shl.4s	v21, v16, #20
	sri.4s	v21, v16, #12
	ldr	q16, [sp, #528]                 ; 16-byte Folded Reload
	add.4s	v16, v18, v16
	add.4s	v16, v16, v21
	eor.16b	v18, v16, v19
	shl.4s	v19, v18, #24
	sri.4s	v19, v18, #8
	add.4s	v22, v19, v22
	eor.16b	v18, v22, v21
	shl.4s	v24, v18, #25
	sri.4s	v24, v18, #7
	ldr	q18, [sp, #592]                 ; 16-byte Folded Reload
	add.4s	v3, v18, v3
	add.4s	v3, v2, v3
	eor.16b	v1, v3, v1
	rev32.8h	v1, v1
	add.4s	v5, v5, v1
	eor.16b	v2, v5, v2
	shl.4s	v21, v2, #20
	sri.4s	v21, v2, #12
	add.4s	v2, v3, v13
	add.4s	v18, v2, v21
	eor.16b	v1, v18, v1
	shl.4s	v2, v1, #24
	sri.4s	v2, v1, #8
	add.4s	v5, v2, v5
	eor.16b	v1, v5, v21
	shl.4s	v25, v1, #25
	sri.4s	v25, v1, #7
	ldr	q1, [sp, #384]                  ; 16-byte Folded Reload
	add.4s	v1, v20, v1
	add.4s	v1, v1, v23
	eor.16b	v2, v2, v1
	rev32.8h	v2, v2
	add.4s	v3, v22, v2
	eor.16b	v20, v3, v23
	shl.4s	v22, v20, #20
	sri.4s	v22, v20, #12
	add.4s	v1, v1, v28
	add.4s	v20, v1, v22
	eor.16b	v2, v20, v2
	shl.4s	v1, v2, #24
	sri.4s	v1, v2, #8
	add.4s	v21, v3, v1
	eor.16b	v2, v21, v22
	shl.4s	v3, v2, #25
	sri.4s	v3, v2, #7
	add.4s	v2, v17, v11
	add.4s	v2, v2, v24
	eor.16b	v6, v2, v6
	rev32.8h	v6, v6
	add.4s	v5, v5, v6
	eor.16b	v17, v5, v24
	shl.4s	v22, v17, #20
	sri.4s	v22, v17, #12
	ldr	q17, [sp, #480]                 ; 16-byte Folded Reload
	add.4s	v2, v2, v17
	add.4s	v17, v2, v22
	eor.16b	v6, v17, v6
	shl.4s	v2, v6, #24
	sri.4s	v2, v6, #8
	add.4s	v23, v2, v5
	eor.16b	v6, v23, v22
	shl.4s	v5, v6, #25
	sri.4s	v5, v6, #7
	ldr	q6, [sp, #464]                  ; 16-byte Folded Reload
	add.4s	v6, v16, v6
	add.4s	v6, v6, v25
	eor.16b	v7, v6, v7
	rev32.8h	v7, v7
	add.4s	v16, v14, v7
	eor.16b	v22, v16, v25
	shl.4s	v24, v22, #20
	sri.4s	v24, v22, #12
	add.4s	v6, v6, v12
	add.4s	v22, v6, v24
	eor.16b	v7, v22, v7
	shl.4s	v6, v7, #24
	sri.4s	v6, v7, #8
	add.4s	v7, v6, v16
	eor.16b	v16, v7, v24
	shl.4s	v24, v16, #25
	sri.4s	v24, v16, #7
	add.4s	v16, v10, v18
	add.4s	v16, v0, v16
	eor.16b	v18, v16, v19
	rev32.8h	v18, v18
	add.4s	v4, v4, v18
	eor.16b	v0, v4, v0
	shl.4s	v19, v0, #20
	sri.4s	v19, v0, #12
	ldr	q0, [sp, #416]                  ; 16-byte Folded Reload
	add.4s	v0, v16, v0
	add.4s	v0, v0, v19
	eor.16b	v16, v0, v18
	shl.4s	v18, v16, #24
	sri.4s	v18, v16, #8
	eor.16b	v7, v7, v20
	eor.16b	v16, v22, v21
	add.4s	v4, v18, v4
	eor.16b	v17, v4, v17
	eor.16b	v0, v0, v23
	trn1.4s	v20, v7, v17
	trn2.4s	v7, v7, v17
	trn1.4s	v17, v16, v0
	trn2.4s	v0, v16, v0
	zip2.2d	v16, v20, v17
	mov.d	v20[1], v17[0]
	zip2.2d	v17, v7, v0
	mov.d	v7[1], v0[0]
	eor.16b	v0, v4, v19
	shl.4s	v4, v0, #25
	sri.4s	v4, v0, #7
	eor.16b	v0, v4, v2
	eor.16b	v2, v6, v3
	eor.16b	v3, v18, v5
	eor.16b	v1, v24, v1
	trn1.4s	v4, v0, v2
	trn1.4s	v5, v3, v1
	mov.16b	v6, v4
	mov.d	v6[1], v5[0]
	zip2.2d	v4, v4, v5
	trn2.4s	v0, v0, v2
	trn2.4s	v1, v3, v1
	mov.16b	v2, v0
	mov.d	v2[1], v1[0]
	zip2.2d	v0, v0, v1
	add	x7, x2, x8, lsl #7
	stp	q20, q6, [x7]
	stp	q7, q2, [x7, #32]
	stp	q16, q4, [x7, #64]
	stp	q17, q0, [x7, #96]
	ldp	q4, q5, [sp, #96]               ; 32-byte Folded Reload
	trn1.4s	v0, v4, v5
	ldr	q6, [sp, #144]                  ; 16-byte Folded Reload
	ldr	q7, [sp, #208]                  ; 16-byte Folded Reload
	trn1.4s	v1, v6, v7
	zip2.2d	v2, v0, v1
	mov.16b	v3, v0
	mov.d	v3[1], v1[0]
	trn2.4s	v1, v4, v5
	trn2.4s	v4, v6, v7
	zip2.2d	v0, v1, v4
	mov.d	v1[1], v4[0]
	ldp	q16, q5, [sp, #176]             ; 32-byte Folded Reload
	ldr	q6, [sp, #128]                  ; 16-byte Folded Reload
	trn1.4s	v4, v5, v6
	trn2.4s	v5, v5, v6
	ldr	q7, [sp, #160]                  ; 16-byte Folded Reload
	trn1.4s	v6, v7, v16
	trn2.4s	v7, v7, v16
	zip2.2d	v16, v4, v6
	mov.d	v4[1], v6[0]
	zip2.2d	v6, v5, v7
	mov.d	v5[1], v7[0]
	add	x7, x1, x8, lsl #8
	stp	q3, q4, [x7]
	stp	q1, q5, [x7, #32]
	stp	q2, q16, [x7, #64]
	trn1.4s	v1, v9, v8
	trn2.4s	v2, v9, v8
	ldp	q18, q5, [sp, #64]              ; 32-byte Folded Reload
	trn1.4s	v3, v29, v5
	zip2.2d	v4, v1, v3
	mov.d	v1[1], v3[0]
	trn2.4s	v3, v29, v5
	zip2.2d	v5, v2, v3
	mov.d	v2[1], v3[0]
	trn1.4s	v3, v31, v30
	trn2.4s	v7, v31, v30
	trn1.4s	v16, v27, v18
	zip2.2d	v17, v3, v16
	mov.d	v3[1], v16[0]
	trn2.4s	v16, v27, v18
	zip2.2d	v18, v7, v16
	mov.d	v7[1], v16[0]
	stp	q0, q6, [x7, #96]
	stp	q1, q3, [x7, #128]
	stp	q2, q7, [x7, #160]
	stp	q4, q17, [x7, #192]
	stp	q5, q18, [x7, #224]
	add	x8, x8, #1
	add	x0, x0, #2, lsl #12             ; =8192
	cmp	x8, x3
	ldp	q1, q29, [sp, #320]             ; 32-byte Folded Reload
	ldp	q3, q2, [sp, #288]              ; 32-byte Folded Reload
	b.ne	LBB1_2
; %bb.7:
	add	sp, sp, #608
	ldp	x20, x19, [sp, #80]             ; 16-byte Folded Reload
	ldp	x22, x21, [sp, #64]             ; 16-byte Folded Reload
	ldp	d9, d8, [sp, #48]               ; 16-byte Folded Reload
	ldp	d11, d10, [sp, #32]             ; 16-byte Folded Reload
	ldp	d13, d12, [sp, #16]             ; 16-byte Folded Reload
	ldp	d15, d14, [sp], #96             ; 16-byte Folded Reload
LBB1_8:
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols
