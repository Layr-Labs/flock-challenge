	.section	__TEXT,__text,regular,pure_instructions
	.private_extern	_flock_blake3_hash8_neon_1024 ; -- Begin function flock_blake3_hash8_neon_1024
	.globl	_flock_blake3_hash8_neon_1024
	.p2align	2
_flock_blake3_hash8_neon_1024:          ; @flock_blake3_hash8_neon_1024
	.cfi_startproc
; %bb.0:
	cbz	x2, LBB0_6
; %bb.1:
	stp	d15, d14, [sp, #-64]!           ; 16-byte Folded Spill
	stp	d13, d12, [sp, #16]             ; 16-byte Folded Spill
	stp	d11, d10, [sp, #32]             ; 16-byte Folded Spill
	stp	d9, d8, [sp, #48]               ; 16-byte Folded Spill
	sub	sp, sp, #832
	.cfi_def_cfa_offset 896
	.cfi_offset b8, -8
	.cfi_offset b9, -16
	.cfi_offset b10, -24
	.cfi_offset b11, -32
	.cfi_offset b12, -40
	.cfi_offset b13, -48
	.cfi_offset b14, -56
	.cfi_offset b15, -64
	mov	w13, #2                         ; =0x2
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
	dup.4s	v2, w9
	mov	w9, #62322                      ; =0xf372
	movk	w9, #15470, lsl #16
	dup.4s	v22, w9
	mov	w9, #44677                      ; =0xae85
	movk	w9, #47975, lsl #16
	dup.4s	v8, w9
	mov	w9, #58983                      ; =0xe667
	movk	w9, #27145, lsl #16
	dup.4s	v30, w9
	stp	q8, q22, [sp, #80]              ; 32-byte Folded Spill
	str	q30, [sp, #64]                  ; 16-byte Folded Spill
	str	q2, [sp, #112]                  ; 16-byte Folded Spill
LBB0_2:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_3 Depth 2
	mov	x9, x0
	mov	w11, #16                        ; =0x10
	stp	q30, q8, [sp, #288]             ; 32-byte Folded Spill
	mov.16b	v5, v22
	mov.16b	v3, v2
	ldp	q13, q0, [sp]                   ; 32-byte Folded Reload
	mov.16b	v6, v13
	mov.16b	v31, v0
	ldp	q21, q1, [sp, #32]              ; 32-byte Folded Reload
	mov.16b	v28, v21
	mov.16b	v24, v1
	mov.16b	v14, v30
	mov.16b	v9, v8
	mov.16b	v29, v22
	mov.16b	v20, v2
	mov.16b	v19, v1
LBB0_3:                                 ;   Parent Loop BB0_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	str	q29, [sp, #128]                 ; 16-byte Folded Spill
	str	q19, [sp, #208]                 ; 16-byte Folded Spill
	stp	q5, q20, [sp, #240]             ; 32-byte Folded Spill
	str	q3, [sp, #272]                  ; 16-byte Folded Spill
	ldr	q1, [x9, #1024]
	ldp	q2, q3, [x9]
	trn1.4s	v5, v2, v1
	ldr	q17, [x9, #2048]
	ldr	q18, [x9, #3072]
	trn2.4s	v1, v2, v1
	trn1.4s	v2, v17, v18
	trn2.4s	v17, v17, v18
	zip2.2d	v16, v5, v2
	str	q16, [sp, #560]                 ; 16-byte Folded Spill
	ldr	q18, [x9, #1040]
	trn1.4s	v20, v3, v18
	zip2.2d	v12, v1, v17
	str	q12, [sp, #608]                 ; 16-byte Folded Spill
	trn2.4s	v10, v3, v18
	ldr	q18, [x9, #2064]
	ldr	q25, [x9, #3088]
	trn1.4s	v26, v18, v25
	trn2.4s	v18, v18, v25
	zip2.2d	v4, v20, v26
	str	q4, [sp, #768]                  ; 16-byte Folded Spill
	zip2.2d	v4, v10, v18
	str	q4, [sp, #816]                  ; 16-byte Folded Spill
	zip1.2d	v4, v5, v2
	ldr	q5, [x9, #1056]
	mov.16b	v7, v4
	str	q4, [sp, #496]                  ; 16-byte Folded Spill
	ldp	q25, q2, [x9, #32]
	trn1.4s	v27, v25, v5
	trn2.4s	v5, v25, v5
	zip1.2d	v15, v1, v17
	str	q15, [sp, #528]                 ; 16-byte Folded Spill
	ldr	q1, [x9, #2080]
	ldr	q17, [x9, #3104]
	trn1.4s	v25, v1, v17
	mov.d	v20[1], v26[0]
	str	q20, [sp, #624]                 ; 16-byte Folded Spill
	trn2.4s	v1, v1, v17
	zip2.2d	v4, v27, v25
	str	q4, [sp, #464]                  ; 16-byte Folded Spill
	mov.d	v10[1], v18[0]
	str	q10, [sp, #320]                 ; 16-byte Folded Spill
	mov.d	v27[1], v25[0]
	zip2.2d	v3, v5, v1
	str	q3, [sp, #400]                  ; 16-byte Folded Spill
	mov.d	v5[1], v1[0]
	str	q5, [sp, #368]                  ; 16-byte Folded Spill
	ldr	q1, [x9, #1072]
	trn1.4s	v3, v2, v1
	trn2.4s	v1, v2, v1
	ldr	q2, [x9, #2096]
	ldr	q5, [x9, #3120]
	trn1.4s	v17, v2, v5
	trn2.4s	v2, v2, v5
	zip2.2d	v4, v3, v17
	stp	q27, q4, [sp, #736]             ; 32-byte Folded Spill
	mov.d	v3[1], v17[0]
	str	q3, [sp, #432]                  ; 16-byte Folded Spill
	zip2.2d	v3, v1, v2
	str	q3, [sp, #672]                  ; 16-byte Folded Spill
	mov.d	v1[1], v2[0]
	str	q1, [sp, #800]                  ; 16-byte Folded Spill
	ldr	q1, [x9, #4096]
	ldr	q2, [x9, #5120]
	trn1.4s	v3, v1, v2
	trn2.4s	v1, v1, v2
	ldr	q2, [x9, #6144]
	ldr	q5, [x9, #7168]
	trn1.4s	v17, v2, v5
	trn2.4s	v2, v2, v5
	zip2.2d	v26, v3, v17
	str	q26, [sp, #544]                 ; 16-byte Folded Spill
	zip1.2d	v18, v3, v17
	str	q18, [sp, #480]                 ; 16-byte Folded Spill
	zip1.2d	v25, v1, v2
	str	q25, [sp, #512]                 ; 16-byte Folded Spill
	zip2.2d	v11, v1, v2
	ldr	q1, [x9, #4112]
	ldr	q2, [x9, #5136]
	trn1.4s	v3, v1, v2
	trn2.4s	v19, v1, v2
	ldr	q2, [x9, #6160]
	ldr	q5, [x9, #7184]
	trn1.4s	v17, v2, v5
	zip2.2d	v4, v3, v17
	str	q4, [sp, #656]                  ; 16-byte Folded Spill
	mov.16b	v29, v24
	zip1.2d	v24, v3, v17
	stp	q11, q24, [sp, #576]            ; 32-byte Folded Spill
	trn2.4s	v2, v2, v5
	zip2.2d	v3, v19, v2
	str	q3, [sp, #704]                  ; 16-byte Folded Spill
	mov.d	v19[1], v2[0]
	ldr	q1, [x9, #4128]
	ldr	q2, [x9, #5152]
	trn1.4s	v3, v1, v2
	trn2.4s	v1, v1, v2
	ldr	q2, [x9, #6176]
	ldr	q5, [x9, #7200]
	trn1.4s	v17, v2, v5
	trn2.4s	v2, v2, v5
	zip2.2d	v4, v3, v17
	str	q4, [sp, #448]                  ; 16-byte Folded Spill
	mov.d	v3[1], v17[0]
	str	q3, [sp, #688]                  ; 16-byte Folded Spill
	zip2.2d	v3, v1, v2
	str	q3, [sp, #384]                  ; 16-byte Folded Spill
	mov.d	v1[1], v2[0]
	stp	q19, q1, [sp, #336]             ; 32-byte Folded Spill
	ldr	q1, [x9, #4144]
	ldr	q2, [x9, #5168]
	trn1.4s	v3, v1, v2
	trn2.4s	v1, v1, v2
	ldr	q2, [x9, #6192]
	ldr	q5, [x9, #7216]
	trn1.4s	v17, v2, v5
	trn2.4s	v2, v2, v5
	zip2.2d	v4, v3, v17
	str	q4, [sp, #720]                  ; 16-byte Folded Spill
	mov.d	v3[1], v17[0]
	str	q3, [sp, #416]                  ; 16-byte Folded Spill
	zip1.2d	v3, v1, v2
	str	q3, [sp, #784]                  ; 16-byte Folded Spill
	zip2.2d	v1, v1, v2
	str	q1, [sp, #640]                  ; 16-byte Folded Spill
	ldr	q1, [sp, #288]                  ; 16-byte Folded Reload
	add.4s	v1, v1, v6
	add.4s	v1, v1, v7
	rev32.8h	v2, v1
	add.4s	v5, v2, v30
	eor.16b	v3, v5, v6
	shl.4s	v6, v3, #20
	sri.4s	v6, v3, #12
	add.4s	v3, v14, v13
	add.4s	v4, v3, v18
	rev32.8h	v7, v4
	add.4s	v17, v7, v30
	eor.16b	v3, v17, v13
	shl.4s	v18, v3, #20
	sri.4s	v18, v3, #12
	add.4s	v1, v1, v15
	add.4s	v3, v1, v6
	eor.16b	v1, v3, v2
	shl.4s	v23, v1, #24
	sri.4s	v23, v1, #8
	add.4s	v1, v4, v25
	add.4s	v25, v1, v18
	eor.16b	v2, v25, v7
	shl.4s	v27, v2, #24
	sri.4s	v27, v2, #8
	add.4s	v1, v23, v5
	str	q1, [sp, #160]                  ; 16-byte Folded Spill
	eor.16b	v2, v1, v6
	shl.4s	v13, v2, #25
	sri.4s	v13, v2, #7
	add.4s	v1, v27, v17
	str	q1, [sp, #176]                  ; 16-byte Folded Spill
	eor.16b	v2, v1, v18
	shl.4s	v15, v2, #25
	sri.4s	v15, v2, #7
	ldr	q1, [sp, #304]                  ; 16-byte Folded Reload
	add.4s	v2, v1, v31
	add.4s	v2, v2, v16
	rev32.8h	v4, v2
	add.4s	v5, v4, v8
	eor.16b	v6, v5, v31
	shl.4s	v16, v6, #20
	sri.4s	v16, v6, #12
	add.4s	v6, v9, v0
	add.4s	v6, v6, v26
	rev32.8h	v18, v6
	add.4s	v31, v18, v8
	eor.16b	v26, v31, v0
	shl.4s	v8, v26, #20
	sri.4s	v8, v26, #12
	add.4s	v2, v2, v12
	add.4s	v12, v2, v16
	eor.16b	v2, v12, v4
	shl.4s	v14, v2, #24
	sri.4s	v14, v2, #8
	add.4s	v2, v6, v11
	add.4s	v0, v2, v8
	str	q0, [sp, #144]                  ; 16-byte Folded Spill
	eor.16b	v2, v0, v18
	shl.4s	v9, v2, #24
	sri.4s	v9, v2, #8
	add.4s	v0, v14, v5
	str	q0, [sp, #192]                  ; 16-byte Folded Spill
	eor.16b	v2, v0, v16
	shl.4s	v6, v2, #25
	sri.4s	v6, v2, #7
	add.4s	v0, v9, v31
	str	q0, [sp, #224]                  ; 16-byte Folded Spill
	eor.16b	v2, v0, v8
	shl.4s	v4, v2, #25
	sri.4s	v4, v2, #7
	ldr	q0, [sp, #240]                  ; 16-byte Folded Reload
	add.4s	v2, v0, v28
	add.4s	v5, v2, v20
	movi.4s	v0, #64
	eor.16b	v2, v5, v0
	rev32.8h	v16, v2
	add.4s	v18, v16, v22
	eor.16b	v2, v18, v28
	shl.4s	v30, v2, #20
	sri.4s	v30, v2, #12
	ldr	q1, [sp, #128]                  ; 16-byte Folded Reload
	add.4s	v2, v1, v21
	add.4s	v20, v2, v24
	eor.16b	v2, v20, v0
	rev32.8h	v31, v2
	add.4s	v2, v31, v22
	eor.16b	v22, v2, v21
	shl.4s	v8, v22, #20
	sri.4s	v8, v22, #12
	add.4s	v5, v5, v10
	add.4s	v28, v5, v30
	eor.16b	v5, v28, v16
	shl.4s	v22, v5, #24
	sri.4s	v22, v5, #8
	add.4s	v5, v20, v19
	add.4s	v17, v5, v8
	eor.16b	v16, v17, v31
	shl.4s	v20, v16, #24
	sri.4s	v20, v16, #8
	add.4s	v31, v22, v18
	eor.16b	v18, v31, v30
	shl.4s	v16, v18, #25
	sri.4s	v16, v18, #7
	subs	x11, x11, #1
	csel	w12, w13, wzr, eq
	cmp	x11, #15
	cinc	w12, w12, eq
	ldp	q1, q0, [sp, #256]              ; 32-byte Folded Reload
	add.4s	v18, v0, v29
	ldr	q0, [sp, #768]                  ; 16-byte Folded Reload
	add.4s	v18, v18, v0
	dup.4s	v24, w12
	eor.16b	v30, v18, v24
	rev32.8h	v10, v30
	ldr	q0, [sp, #112]                  ; 16-byte Folded Reload
	add.4s	v11, v10, v0
	eor.16b	v21, v11, v29
	shl.4s	v19, v21, #20
	sri.4s	v19, v21, #12
	ldr	q5, [sp, #208]                  ; 16-byte Folded Reload
	add.4s	v21, v1, v5
	ldr	q1, [sp, #656]                  ; 16-byte Folded Reload
	add.4s	v21, v21, v1
	eor.16b	v24, v21, v24
	rev32.8h	v24, v24
	add.4s	v1, v24, v0
	eor.16b	v30, v1, v5
	shl.4s	v0, v30, #20
	sri.4s	v0, v30, #12
	ldr	q5, [sp, #816]                  ; 16-byte Folded Reload
	add.4s	v18, v18, v5
	add.4s	v26, v18, v19
	eor.16b	v18, v26, v10
	shl.4s	v10, v18, #24
	sri.4s	v10, v18, #8
	ldr	q5, [sp, #704]                  ; 16-byte Folded Reload
	add.4s	v18, v21, v5
	add.4s	v18, v18, v0
	eor.16b	v21, v18, v24
	shl.4s	v24, v21, #24
	sri.4s	v24, v21, #8
	add.4s	v21, v20, v2
	eor.16b	v2, v21, v8
	shl.4s	v8, v2, #25
	sri.4s	v8, v2, #7
	add.4s	v11, v10, v11
	eor.16b	v19, v11, v19
	shl.4s	v2, v19, #25
	sri.4s	v2, v19, #7
	add.4s	v7, v24, v1
	eor.16b	v19, v7, v0
	shl.4s	v5, v19, #25
	sri.4s	v5, v19, #7
	ldr	q0, [sp, #736]                  ; 16-byte Folded Reload
	add.4s	v3, v3, v0
	add.4s	v3, v3, v6
	eor.16b	v19, v10, v3
	rev32.8h	v10, v19
	add.4s	v31, v31, v10
	eor.16b	v6, v31, v6
	shl.4s	v1, v6, #20
	sri.4s	v1, v6, #12
	ldr	q0, [sp, #688]                  ; 16-byte Folded Reload
	add.4s	v6, v25, v0
	add.4s	v25, v6, v4
	eor.16b	v6, v24, v25
	rev32.8h	v24, v6
	add.4s	v21, v21, v24
	eor.16b	v4, v21, v4
	shl.4s	v0, v4, #20
	sri.4s	v0, v4, #12
	ldr	q4, [sp, #368]                  ; 16-byte Folded Reload
	add.4s	v3, v3, v4
	add.4s	v19, v3, v1
	eor.16b	v3, v19, v10
	shl.4s	v6, v3, #24
	sri.4s	v6, v3, #8
	ldr	q3, [sp, #352]                  ; 16-byte Folded Reload
	add.4s	v3, v25, v3
	add.4s	v3, v3, v0
	stp	q3, q6, [sp, #272]              ; 32-byte Folded Spill
	eor.16b	v3, v3, v24
	shl.4s	v4, v3, #24
	sri.4s	v4, v3, #8
	str	q4, [sp, #304]                  ; 16-byte Folded Spill
	add.4s	v3, v6, v31
	str	q3, [sp, #208]                  ; 16-byte Folded Spill
	eor.16b	v1, v3, v1
	shl.4s	v24, v1, #25
	sri.4s	v24, v1, #7
	add.4s	v1, v4, v21
	str	q1, [sp, #240]                  ; 16-byte Folded Spill
	eor.16b	v0, v1, v0
	shl.4s	v21, v0, #25
	sri.4s	v21, v0, #7
	ldp	q29, q30, [sp, #448]            ; 32-byte Folded Reload
	add.4s	v0, v12, v30
	add.4s	v0, v0, v16
	eor.16b	v1, v0, v23
	rev32.8h	v1, v1
	add.4s	v11, v11, v1
	eor.16b	v3, v11, v16
	shl.4s	v12, v3, #20
	sri.4s	v12, v3, #12
	ldr	q3, [sp, #144]                  ; 16-byte Folded Reload
	add.4s	v3, v3, v29
	add.4s	v3, v3, v8
	eor.16b	v16, v3, v27
	rev32.8h	v16, v16
	add.4s	v31, v7, v16
	eor.16b	v7, v31, v8
	shl.4s	v8, v7, #20
	sri.4s	v8, v7, #12
	ldr	q4, [sp, #400]                  ; 16-byte Folded Reload
	add.4s	v0, v0, v4
	add.4s	v0, v0, v12
	str	q0, [sp, #256]                  ; 16-byte Folded Spill
	eor.16b	v0, v0, v1
	shl.4s	v10, v0, #24
	sri.4s	v10, v0, #8
	ldr	q0, [sp, #432]                  ; 16-byte Folded Reload
	add.4s	v0, v28, v0
	add.4s	v0, v0, v2
	eor.16b	v1, v0, v14
	rev32.8h	v1, v1
	ldp	q6, q4, [sp, #160]              ; 32-byte Folded Reload
	add.4s	v7, v6, v1
	eor.16b	v2, v7, v2
	shl.4s	v23, v2, #20
	sri.4s	v23, v2, #12
	ldr	q2, [sp, #416]                  ; 16-byte Folded Reload
	add.4s	v2, v17, v2
	add.4s	v2, v2, v5
	eor.16b	v17, v2, v9
	rev32.8h	v27, v17
	add.4s	v9, v4, v27
	eor.16b	v5, v9, v5
	shl.4s	v14, v5, #20
	sri.4s	v14, v5, #12
	ldr	q4, [sp, #384]                  ; 16-byte Folded Reload
	add.4s	v3, v3, v4
	add.4s	v25, v3, v8
	eor.16b	v3, v25, v16
	shl.4s	v17, v3, #24
	sri.4s	v17, v3, #8
	ldr	q3, [sp, #800]                  ; 16-byte Folded Reload
	add.4s	v0, v0, v3
	add.4s	v0, v0, v23
	str	q0, [sp, #144]                  ; 16-byte Folded Spill
	eor.16b	v1, v0, v1
	shl.4s	v16, v1, #24
	sri.4s	v16, v1, #8
	ldr	q0, [sp, #784]                  ; 16-byte Folded Reload
	add.4s	v1, v2, v0
	add.4s	v0, v1, v14
	str	q0, [sp, #176]                  ; 16-byte Folded Spill
	eor.16b	v1, v0, v27
	shl.4s	v28, v1, #24
	sri.4s	v28, v1, #8
	add.4s	v3, v16, v7
	eor.16b	v2, v3, v23
	shl.4s	v23, v2, #25
	sri.4s	v23, v2, #7
	add.4s	v2, v28, v9
	eor.16b	v5, v2, v14
	shl.4s	v27, v5, #25
	sri.4s	v27, v5, #7
	ldr	q0, [sp, #752]                  ; 16-byte Folded Reload
	add.4s	v5, v0, v26
	add.4s	v5, v13, v5
	eor.16b	v7, v5, v22
	rev32.8h	v7, v7
	ldr	q0, [sp, #192]                  ; 16-byte Folded Reload
	add.4s	v14, v0, v7
	eor.16b	v22, v14, v13
	shl.4s	v13, v22, #20
	sri.4s	v13, v22, #12
	ldr	q0, [sp, #720]                  ; 16-byte Folded Reload
	add.4s	v18, v0, v18
	add.4s	v18, v15, v18
	eor.16b	v20, v18, v20
	rev32.8h	v20, v20
	ldr	q0, [sp, #224]                  ; 16-byte Folded Reload
	add.4s	v1, v0, v20
	eor.16b	v22, v1, v15
	shl.4s	v0, v22, #20
	sri.4s	v0, v22, #12
	ldr	q4, [sp, #672]                  ; 16-byte Folded Reload
	add.4s	v5, v5, v4
	add.4s	v4, v5, v13
	str	q4, [sp, #160]                  ; 16-byte Folded Spill
	eor.16b	v5, v4, v7
	shl.4s	v9, v5, #24
	sri.4s	v9, v5, #8
	ldr	q4, [sp, #640]                  ; 16-byte Folded Reload
	add.4s	v5, v18, v4
	add.4s	v4, v5, v0
	str	q4, [sp, #224]                  ; 16-byte Folded Spill
	eor.16b	v5, v4, v20
	shl.4s	v22, v5, #24
	sri.4s	v22, v5, #8
	add.4s	v7, v10, v11
	eor.16b	v5, v7, v12
	shl.4s	v15, v5, #25
	sri.4s	v15, v5, #7
	add.4s	v11, v9, v14
	eor.16b	v5, v11, v13
	shl.4s	v20, v5, #25
	sri.4s	v20, v5, #7
	add.4s	v18, v22, v1
	eor.16b	v0, v18, v0
	shl.4s	v13, v0, #25
	sri.4s	v13, v0, #7
	add.4s	v12, v17, v31
	eor.16b	v0, v12, v8
	shl.4s	v5, v0, #25
	sri.4s	v5, v0, #7
	ldr	q0, [sp, #560]                  ; 16-byte Folded Reload
	add.4s	v0, v19, v0
	add.4s	v0, v0, v20
	eor.16b	v1, v0, v10
	rev32.8h	v19, v1
	add.4s	v14, v3, v19
	eor.16b	v3, v14, v20
	shl.4s	v6, v3, #20
	sri.4s	v6, v3, #12
	ldr	q1, [sp, #544]                  ; 16-byte Folded Reload
	ldr	q3, [sp, #272]                  ; 16-byte Folded Reload
	add.4s	v3, v3, v1
	add.4s	v4, v3, v13
	eor.16b	v17, v4, v17
	rev32.8h	v17, v17
	add.4s	v31, v2, v17
	eor.16b	v2, v31, v13
	shl.4s	v13, v2, #20
	sri.4s	v13, v2, #12
	ldr	q1, [sp, #768]                  ; 16-byte Folded Reload
	add.4s	v0, v1, v0
	add.4s	v10, v6, v0
	eor.16b	v0, v10, v19
	shl.4s	v20, v0, #24
	sri.4s	v20, v0, #8
	ldr	q0, [sp, #608]                  ; 16-byte Folded Reload
	ldr	q1, [sp, #256]                  ; 16-byte Folded Reload
	add.4s	v0, v0, v1
	add.4s	v1, v24, v0
	eor.16b	v2, v16, v1
	rev32.8h	v2, v2
	add.4s	v16, v11, v2
	eor.16b	v19, v16, v24
	shl.4s	v8, v19, #20
	sri.4s	v8, v19, #12
	ldr	q0, [sp, #576]                  ; 16-byte Folded Reload
	add.4s	v19, v0, v25
	add.4s	v11, v21, v19
	eor.16b	v19, v28, v11
	rev32.8h	v28, v19
	add.4s	v3, v18, v28
	eor.16b	v18, v3, v21
	shl.4s	v0, v18, #20
	sri.4s	v0, v18, #12
	ldr	q18, [sp, #656]                 ; 16-byte Folded Reload
	add.4s	v4, v18, v4
	add.4s	v26, v13, v4
	eor.16b	v4, v26, v17
	shl.4s	v24, v4, #24
	sri.4s	v24, v4, #8
	add.4s	v1, v1, v30
	add.4s	v19, v1, v8
	eor.16b	v1, v19, v2
	shl.4s	v25, v1, #24
	sri.4s	v25, v1, #8
	add.4s	v1, v11, v29
	add.4s	v1, v1, v0
	str	q1, [sp, #192]                  ; 16-byte Folded Spill
	eor.16b	v1, v1, v28
	shl.4s	v21, v1, #24
	sri.4s	v21, v1, #8
	add.4s	v1, v25, v16
	str	q1, [sp, #256]                  ; 16-byte Folded Spill
	eor.16b	v1, v1, v8
	shl.4s	v18, v1, #25
	sri.4s	v18, v1, #7
	add.4s	v1, v21, v3
	str	q1, [sp, #272]                  ; 16-byte Folded Spill
	eor.16b	v0, v1, v0
	shl.4s	v8, v0, #25
	sri.4s	v8, v0, #7
	ldr	q0, [sp, #816]                  ; 16-byte Folded Reload
	ldr	q1, [sp, #144]                  ; 16-byte Folded Reload
	add.4s	v0, v0, v1
	add.4s	v1, v15, v0
	eor.16b	v0, v9, v1
	rev32.8h	v2, v0
	ldr	q0, [sp, #208]                  ; 16-byte Folded Reload
	add.4s	v16, v0, v2
	eor.16b	v3, v16, v15
	shl.4s	v0, v3, #20
	sri.4s	v0, v3, #12
	ldr	q3, [sp, #704]                  ; 16-byte Folded Reload
	ldr	q4, [sp, #176]                  ; 16-byte Folded Reload
	add.4s	v3, v3, v4
	add.4s	v4, v5, v3
	eor.16b	v3, v22, v4
	rev32.8h	v22, v3
	ldr	q3, [sp, #240]                  ; 16-byte Folded Reload
	add.4s	v3, v3, v22
	eor.16b	v5, v3, v5
	shl.4s	v11, v5, #20
	sri.4s	v11, v5, #12
	ldr	q5, [sp, #496]                  ; 16-byte Folded Reload
	add.4s	v1, v1, v5
	add.4s	v1, v1, v0
	str	q1, [sp, #176]                  ; 16-byte Folded Spill
	eor.16b	v1, v1, v2
	shl.4s	v17, v1, #24
	sri.4s	v17, v1, #8
	ldr	q1, [sp, #480]                  ; 16-byte Folded Reload
	add.4s	v1, v4, v1
	add.4s	v1, v1, v11
	str	q1, [sp, #240]                  ; 16-byte Folded Spill
	eor.16b	v1, v1, v22
	shl.4s	v15, v1, #24
	sri.4s	v15, v1, #8
	add.4s	v28, v14, v20
	eor.16b	v1, v28, v6
	shl.4s	v22, v1, #25
	sri.4s	v22, v1, #7
	ldr	q1, [sp, #624]                  ; 16-byte Folded Reload
	ldr	q2, [sp, #160]                  ; 16-byte Folded Reload
	add.4s	v1, v1, v2
	add.4s	v1, v23, v1
	ldr	q2, [sp, #288]                  ; 16-byte Folded Reload
	eor.16b	v4, v1, v2
	rev32.8h	v4, v4
	add.4s	v29, v7, v4
	eor.16b	v5, v29, v23
	shl.4s	v30, v5, #20
	sri.4s	v30, v5, #12
	add.4s	v7, v31, v24
	eor.16b	v6, v7, v13
	shl.4s	v23, v6, #25
	sri.4s	v23, v6, #7
	ldr	q2, [sp, #592]                  ; 16-byte Folded Reload
	ldr	q5, [sp, #224]                  ; 16-byte Folded Reload
	add.4s	v6, v2, v5
	add.4s	v31, v27, v6
	ldr	q2, [sp, #304]                  ; 16-byte Folded Reload
	eor.16b	v6, v31, v2
	rev32.8h	v14, v6
	add.4s	v12, v12, v14
	eor.16b	v6, v12, v27
	shl.4s	v27, v6, #20
	sri.4s	v27, v6, #12
	add.4s	v16, v17, v16
	eor.16b	v0, v16, v0
	shl.4s	v6, v0, #25
	sri.4s	v6, v0, #7
	ldr	q0, [sp, #800]                  ; 16-byte Folded Reload
	add.4s	v0, v1, v0
	add.4s	v9, v0, v30
	eor.16b	v0, v9, v4
	shl.4s	v4, v0, #24
	sri.4s	v4, v0, #8
	ldr	q0, [sp, #784]                  ; 16-byte Folded Reload
	add.4s	v0, v31, v0
	add.4s	v13, v0, v27
	eor.16b	v0, v13, v14
	shl.4s	v31, v0, #24
	sri.4s	v31, v0, #8
	add.4s	v5, v4, v29
	eor.16b	v29, v5, v30
	shl.4s	v2, v29, #25
	sri.4s	v2, v29, #7
	add.4s	v29, v15, v3
	eor.16b	v30, v29, v11
	shl.4s	v3, v30, #25
	sri.4s	v3, v30, #7
	add.4s	v30, v31, v12
	eor.16b	v11, v30, v27
	shl.4s	v27, v11, #25
	sri.4s	v27, v11, #7
	ldp	q0, q1, [sp, #512]              ; 32-byte Folded Reload
	add.4s	v10, v10, v1
	add.4s	v10, v10, v18
	eor.16b	v4, v4, v10
	rev32.8h	v4, v4
	add.4s	v12, v16, v4
	eor.16b	v16, v12, v18
	shl.4s	v18, v16, #20
	sri.4s	v18, v16, #12
	add.4s	v16, v26, v0
	add.4s	v26, v16, v8
	eor.16b	v16, v31, v26
	rev32.8h	v31, v16
	add.4s	v1, v29, v31
	eor.16b	v16, v1, v8
	shl.4s	v0, v16, #20
	sri.4s	v0, v16, #12
	ldr	q16, [sp, #400]                 ; 16-byte Folded Reload
	add.4s	v16, v10, v16
	add.4s	v16, v16, v18
	eor.16b	v4, v16, v4
	shl.4s	v29, v4, #24
	sri.4s	v29, v4, #8
	ldr	q4, [sp, #384]                  ; 16-byte Folded Reload
	add.4s	v4, v26, v4
	add.4s	v14, v4, v0
	eor.16b	v4, v14, v31
	shl.4s	v26, v4, #24
	sri.4s	v26, v4, #8
	stp	q26, q29, [sp, #288]            ; 32-byte Folded Spill
	ldr	q11, [sp, #432]                 ; 16-byte Folded Reload
	add.4s	v4, v19, v11
	add.4s	v19, v4, v6
	eor.16b	v4, v19, v20
	rev32.8h	v10, v4
	add.4s	v4, v5, v10
	eor.16b	v5, v4, v6
	shl.4s	v31, v5, #20
	sri.4s	v31, v5, #12
	add.4s	v5, v29, v12
	str	q5, [sp, #224]                  ; 16-byte Folded Spill
	eor.16b	v5, v5, v18
	shl.4s	v29, v5, #25
	sri.4s	v29, v5, #7
	ldr	q20, [sp, #416]                 ; 16-byte Folded Reload
	ldr	q5, [sp, #192]                  ; 16-byte Folded Reload
	add.4s	v5, v5, v20
	add.4s	v18, v5, v3
	eor.16b	v5, v18, v24
	rev32.8h	v24, v5
	add.4s	v6, v30, v24
	eor.16b	v3, v6, v3
	shl.4s	v8, v3, #20
	sri.4s	v8, v3, #12
	add.4s	v1, v26, v1
	str	q1, [sp, #208]                  ; 16-byte Folded Spill
	eor.16b	v0, v1, v0
	shl.4s	v30, v0, #25
	sri.4s	v30, v0, #7
	ldr	q0, [sp, #320]                  ; 16-byte Folded Reload
	add.4s	v0, v19, v0
	add.4s	v26, v0, v31
	eor.16b	v0, v26, v10
	shl.4s	v12, v0, #24
	sri.4s	v12, v0, #8
	ldr	q0, [sp, #368]                  ; 16-byte Folded Reload
	ldr	q1, [sp, #176]                  ; 16-byte Folded Reload
	add.4s	v0, v1, v0
	add.4s	v0, v0, v2
	eor.16b	v1, v0, v25
	rev32.8h	v1, v1
	add.4s	v3, v28, v1
	eor.16b	v2, v3, v2
	shl.4s	v28, v2, #20
	sri.4s	v28, v2, #12
	ldr	q2, [sp, #352]                  ; 16-byte Folded Reload
	ldr	q5, [sp, #240]                  ; 16-byte Folded Reload
	add.4s	v2, v5, v2
	add.4s	v19, v2, v27
	eor.16b	v2, v19, v21
	rev32.8h	v21, v2
	add.4s	v7, v7, v21
	eor.16b	v2, v7, v27
	shl.4s	v27, v2, #20
	sri.4s	v27, v2, #12
	ldr	q2, [sp, #752]                  ; 16-byte Folded Reload
	add.4s	v0, v0, v2
	add.4s	v0, v0, v28
	str	q0, [sp, #144]                  ; 16-byte Folded Spill
	eor.16b	v0, v0, v1
	shl.4s	v2, v0, #24
	sri.4s	v2, v0, #8
	ldr	q0, [sp, #720]                  ; 16-byte Folded Reload
	add.4s	v0, v19, v0
	add.4s	v0, v0, v27
	str	q0, [sp, #160]                  ; 16-byte Folded Spill
	eor.16b	v0, v0, v21
	shl.4s	v25, v0, #24
	sri.4s	v25, v0, #8
	ldr	q0, [sp, #336]                  ; 16-byte Folded Reload
	add.4s	v0, v18, v0
	add.4s	v10, v0, v8
	eor.16b	v0, v10, v24
	shl.4s	v21, v0, #24
	sri.4s	v21, v0, #8
	add.4s	v19, v2, v3
	eor.16b	v1, v19, v28
	shl.4s	v28, v1, #25
	sri.4s	v28, v1, #7
	add.4s	v5, v25, v7
	eor.16b	v1, v5, v27
	shl.4s	v27, v1, #25
	sri.4s	v27, v1, #7
	add.4s	v0, v12, v4
	str	q0, [sp, #192]                  ; 16-byte Folded Spill
	eor.16b	v1, v0, v31
	shl.4s	v31, v1, #25
	sri.4s	v31, v1, #7
	ldr	q0, [sp, #672]                  ; 16-byte Folded Reload
	add.4s	v1, v0, v9
	add.4s	v1, v22, v1
	eor.16b	v4, v1, v17
	rev32.8h	v4, v4
	ldr	q0, [sp, #256]                  ; 16-byte Folded Reload
	add.4s	v17, v0, v4
	eor.16b	v7, v17, v22
	shl.4s	v22, v7, #20
	sri.4s	v22, v7, #12
	ldr	q0, [sp, #640]                  ; 16-byte Folded Reload
	add.4s	v7, v0, v13
	add.4s	v18, v23, v7
	eor.16b	v7, v18, v15
	rev32.8h	v9, v7
	ldr	q0, [sp, #272]                  ; 16-byte Folded Reload
	add.4s	v13, v0, v9
	eor.16b	v7, v13, v23
	shl.4s	v15, v7, #20
	sri.4s	v15, v7, #12
	ldr	q0, [sp, #736]                  ; 16-byte Folded Reload
	add.4s	v1, v1, v0
	add.4s	v7, v1, v22
	eor.16b	v4, v7, v4
	shl.4s	v3, v4, #24
	sri.4s	v3, v4, #8
	ldr	q0, [sp, #688]                  ; 16-byte Folded Reload
	add.4s	v4, v18, v0
	add.4s	v24, v4, v15
	eor.16b	v4, v24, v9
	shl.4s	v23, v4, #24
	sri.4s	v23, v4, #8
	add.4s	v18, v21, v6
	eor.16b	v8, v18, v8
	shl.4s	v6, v8, #25
	sri.4s	v6, v8, #7
	add.4s	v9, v3, v17
	eor.16b	v17, v9, v22
	shl.4s	v8, v17, #25
	sri.4s	v8, v17, #7
	add.4s	v22, v23, v13
	eor.16b	v17, v22, v15
	shl.4s	v13, v17, #25
	sri.4s	v13, v17, #7
	ldr	q0, [sp, #608]                  ; 16-byte Folded Reload
	add.4s	v16, v16, v0
	add.4s	v0, v16, v8
	eor.16b	v1, v0, v12
	rev32.8h	v15, v1
	add.4s	v17, v19, v15
	eor.16b	v4, v17, v8
	shl.4s	v1, v4, #20
	sri.4s	v1, v4, #12
	ldr	q4, [sp, #576]                  ; 16-byte Folded Reload
	add.4s	v4, v14, v4
	add.4s	v16, v4, v13
	eor.16b	v4, v16, v21
	rev32.8h	v12, v4
	add.4s	v5, v5, v12
	eor.16b	v4, v5, v13
	shl.4s	v8, v4, #20
	sri.4s	v8, v4, #12
	ldr	q4, [sp, #624]                  ; 16-byte Folded Reload
	add.4s	v0, v4, v0
	add.4s	v14, v1, v0
	eor.16b	v0, v14, v15
	shl.4s	v21, v0, #24
	sri.4s	v21, v0, #8
	ldp	q4, q0, [sp, #448]              ; 32-byte Folded Reload
	add.4s	v0, v0, v26
	add.4s	v0, v29, v0
	eor.16b	v2, v2, v0
	rev32.8h	v2, v2
	add.4s	v19, v9, v2
	eor.16b	v26, v19, v29
	shl.4s	v29, v26, #20
	sri.4s	v29, v26, #12
	add.4s	v26, v4, v10
	add.4s	v26, v30, v26
	eor.16b	v25, v25, v26
	rev32.8h	v9, v25
	add.4s	v15, v22, v9
	eor.16b	v22, v15, v30
	shl.4s	v30, v22, #20
	sri.4s	v30, v22, #12
	add.4s	v0, v0, v11
	add.4s	v0, v0, v29
	str	q0, [sp, #128]                  ; 16-byte Folded Spill
	eor.16b	v0, v0, v2
	shl.4s	v25, v0, #24
	sri.4s	v25, v0, #8
	add.4s	v0, v26, v20
	add.4s	v0, v0, v30
	str	q0, [sp, #176]                  ; 16-byte Folded Spill
	eor.16b	v0, v0, v9
	shl.4s	v22, v0, #24
	sri.4s	v22, v0, #8
	ldr	q0, [sp, #592]                  ; 16-byte Folded Reload
	add.4s	v0, v0, v16
	add.4s	v26, v8, v0
	eor.16b	v0, v26, v12
	shl.4s	v13, v0, #24
	sri.4s	v13, v0, #8
	add.4s	v0, v25, v19
	str	q0, [sp, #240]                  ; 16-byte Folded Spill
	eor.16b	v0, v0, v29
	shl.4s	v10, v0, #25
	sri.4s	v10, v0, #7
	add.4s	v0, v22, v15
	str	q0, [sp, #272]                  ; 16-byte Folded Spill
	eor.16b	v0, v0, v30
	shl.4s	v29, v0, #25
	sri.4s	v29, v0, #7
	add.4s	v12, v17, v21
	eor.16b	v1, v12, v1
	shl.4s	v17, v1, #25
	sri.4s	v17, v1, #7
	ldr	q0, [sp, #800]                  ; 16-byte Folded Reload
	ldr	q1, [sp, #144]                  ; 16-byte Folded Reload
	add.4s	v1, v0, v1
	add.4s	v19, v31, v1
	eor.16b	v1, v3, v19
	rev32.8h	v3, v1
	ldr	q0, [sp, #224]                  ; 16-byte Folded Reload
	add.4s	v4, v0, v3
	eor.16b	v1, v4, v31
	shl.4s	v16, v1, #20
	sri.4s	v16, v1, #12
	ldr	q0, [sp, #784]                  ; 16-byte Folded Reload
	ldr	q1, [sp, #160]                  ; 16-byte Folded Reload
	add.4s	v1, v0, v1
	add.4s	v20, v6, v1
	eor.16b	v1, v23, v20
	rev32.8h	v30, v1
	ldr	q0, [sp, #208]                  ; 16-byte Folded Reload
	add.4s	v31, v0, v30
	eor.16b	v6, v31, v6
	shl.4s	v2, v6, #20
	sri.4s	v2, v6, #12
	ldp	q0, q1, [sp, #544]              ; 32-byte Folded Reload
	add.4s	v6, v19, v1
	add.4s	v9, v6, v16
	eor.16b	v3, v9, v3
	shl.4s	v23, v3, #24
	sri.4s	v23, v3, #8
	add.4s	v3, v20, v0
	add.4s	v6, v3, v2
	eor.16b	v3, v6, v30
	shl.4s	v19, v3, #24
	sri.4s	v19, v3, #8
	add.4s	v5, v5, v13
	eor.16b	v3, v5, v8
	shl.4s	v20, v3, #25
	sri.4s	v20, v3, #7
	ldr	q0, [sp, #816]                  ; 16-byte Folded Reload
	add.4s	v3, v0, v7
	add.4s	v3, v28, v3
	ldr	q0, [sp, #304]                  ; 16-byte Folded Reload
	eor.16b	v7, v3, v0
	rev32.8h	v7, v7
	ldr	q0, [sp, #192]                  ; 16-byte Folded Reload
	add.4s	v8, v0, v7
	eor.16b	v28, v8, v28
	shl.4s	v15, v28, #20
	sri.4s	v15, v28, #12
	ldr	q0, [sp, #704]                  ; 16-byte Folded Reload
	add.4s	v24, v0, v24
	add.4s	v24, v27, v24
	ldr	q0, [sp, #288]                  ; 16-byte Folded Reload
	eor.16b	v28, v24, v0
	rev32.8h	v28, v28
	add.4s	v1, v18, v28
	eor.16b	v18, v1, v27
	shl.4s	v0, v18, #20
	sri.4s	v0, v18, #12
	ldr	q18, [sp, #752]                 ; 16-byte Folded Reload
	add.4s	v3, v3, v18
	add.4s	v11, v3, v15
	eor.16b	v3, v11, v7
	shl.4s	v7, v3, #24
	sri.4s	v7, v3, #8
	ldr	q3, [sp, #720]                  ; 16-byte Folded Reload
	add.4s	v3, v24, v3
	add.4s	v3, v3, v0
	str	q3, [sp, #208]                  ; 16-byte Folded Spill
	eor.16b	v3, v3, v28
	shl.4s	v24, v3, #24
	sri.4s	v24, v3, #8
	add.4s	v3, v23, v4
	eor.16b	v4, v3, v16
	shl.4s	v18, v4, #25
	sri.4s	v18, v4, #7
	add.4s	v28, v7, v8
	eor.16b	v4, v28, v15
	shl.4s	v27, v4, #25
	sri.4s	v27, v4, #7
	add.4s	v16, v24, v1
	eor.16b	v1, v16, v0
	shl.4s	v0, v1, #25
	sri.4s	v0, v1, #7
	add.4s	v4, v19, v31
	eor.16b	v2, v4, v2
	shl.4s	v1, v2, #25
	sri.4s	v1, v2, #7
	ldr	q2, [sp, #768]                  ; 16-byte Folded Reload
	add.4s	v2, v14, v2
	add.4s	v2, v2, v10
	eor.16b	v7, v7, v2
	rev32.8h	v31, v7
	add.4s	v3, v3, v31
	eor.16b	v7, v3, v10
	shl.4s	v8, v7, #20
	sri.4s	v8, v7, #12
	ldr	q7, [sp, #656]                  ; 16-byte Folded Reload
	add.4s	v7, v26, v7
	add.4s	v26, v7, v29
	eor.16b	v7, v24, v26
	rev32.8h	v24, v7
	add.4s	v4, v4, v24
	eor.16b	v7, v4, v29
	shl.4s	v29, v7, #20
	sri.4s	v29, v7, #12
	ldr	q7, [sp, #320]                  ; 16-byte Folded Reload
	add.4s	v2, v2, v7
	add.4s	v7, v2, v8
	eor.16b	v2, v7, v31
	shl.4s	v31, v2, #24
	sri.4s	v31, v2, #8
	ldr	q2, [sp, #336]                  ; 16-byte Folded Reload
	add.4s	v2, v26, v2
	add.4s	v10, v2, v29
	eor.16b	v2, v10, v24
	shl.4s	v24, v2, #24
	sri.4s	v24, v2, #8
	mov.16b	v30, v24
	stp	q24, q31, [sp, #288]            ; 32-byte Folded Spill
	add.4s	v2, v31, v3
	str	q2, [sp, #256]                  ; 16-byte Folded Spill
	eor.16b	v2, v2, v8
	shl.4s	v24, v2, #25
	sri.4s	v24, v2, #7
	ldr	q31, [sp, #368]                 ; 16-byte Folded Reload
	ldr	q2, [sp, #128]                  ; 16-byte Folded Reload
	add.4s	v2, v2, v31
	add.4s	v2, v2, v18
	eor.16b	v3, v2, v21
	rev32.8h	v8, v3
	add.4s	v3, v28, v8
	eor.16b	v18, v3, v18
	shl.4s	v26, v18, #20
	sri.4s	v26, v18, #12
	add.4s	v4, v30, v4
	str	q4, [sp, #224]                  ; 16-byte Folded Spill
	eor.16b	v4, v4, v29
	shl.4s	v28, v4, #25
	sri.4s	v28, v4, #7
	ldr	q30, [sp, #352]                 ; 16-byte Folded Reload
	ldr	q4, [sp, #176]                  ; 16-byte Folded Reload
	add.4s	v4, v4, v30
	add.4s	v4, v4, v1
	eor.16b	v18, v4, v13
	rev32.8h	v29, v18
	add.4s	v18, v16, v29
	eor.16b	v1, v18, v1
	shl.4s	v15, v1, #20
	sri.4s	v15, v1, #12
	ldr	q1, [sp, #496]                  ; 16-byte Folded Reload
	add.4s	v1, v2, v1
	add.4s	v21, v1, v26
	eor.16b	v1, v21, v8
	shl.4s	v13, v1, #24
	sri.4s	v13, v1, #8
	ldr	q1, [sp, #480]                  ; 16-byte Folded Reload
	add.4s	v1, v4, v1
	add.4s	v14, v1, v15
	eor.16b	v1, v14, v29
	shl.4s	v8, v1, #24
	sri.4s	v8, v1, #8
	ldr	q1, [sp, #400]                  ; 16-byte Folded Reload
	add.4s	v1, v9, v1
	add.4s	v1, v1, v27
	eor.16b	v2, v1, v25
	rev32.8h	v2, v2
	add.4s	v4, v12, v2
	eor.16b	v16, v4, v27
	shl.4s	v27, v16, #20
	sri.4s	v27, v16, #12
	ldr	q16, [sp, #384]                 ; 16-byte Folded Reload
	add.4s	v6, v6, v16
	add.4s	v16, v6, v0
	eor.16b	v6, v16, v22
	rev32.8h	v22, v6
	add.4s	v5, v5, v22
	eor.16b	v0, v5, v0
	shl.4s	v29, v0, #20
	sri.4s	v29, v0, #12
	ldr	q0, [sp, #672]                  ; 16-byte Folded Reload
	add.4s	v0, v1, v0
	add.4s	v6, v0, v27
	eor.16b	v0, v6, v2
	shl.4s	v25, v0, #24
	sri.4s	v25, v0, #8
	ldr	q0, [sp, #640]                  ; 16-byte Folded Reload
	add.4s	v0, v16, v0
	add.4s	v16, v0, v29
	eor.16b	v0, v16, v22
	shl.4s	v22, v0, #24
	sri.4s	v22, v0, #8
	add.4s	v1, v25, v4
	eor.16b	v0, v1, v27
	shl.4s	v27, v0, #25
	sri.4s	v27, v0, #7
	ldr	q0, [sp, #736]                  ; 16-byte Folded Reload
	add.4s	v0, v0, v11
	add.4s	v0, v17, v0
	eor.16b	v2, v0, v23
	rev32.8h	v2, v2
	ldr	q4, [sp, #240]                  ; 16-byte Folded Reload
	add.4s	v23, v4, v2
	eor.16b	v4, v23, v17
	shl.4s	v9, v4, #20
	sri.4s	v9, v4, #12
	add.4s	v12, v22, v5
	eor.16b	v4, v12, v29
	shl.4s	v17, v4, #25
	sri.4s	v17, v4, #7
	ldr	q4, [sp, #688]                  ; 16-byte Folded Reload
	ldr	q5, [sp, #208]                  ; 16-byte Folded Reload
	add.4s	v4, v4, v5
	add.4s	v5, v20, v4
	eor.16b	v4, v5, v19
	rev32.8h	v19, v4
	ldr	q4, [sp, #272]                  ; 16-byte Folded Reload
	add.4s	v29, v4, v19
	eor.16b	v4, v29, v20
	shl.4s	v20, v4, #20
	sri.4s	v20, v4, #12
	add.4s	v3, v13, v3
	str	q3, [sp, #240]                  ; 16-byte Folded Spill
	eor.16b	v3, v3, v26
	shl.4s	v4, v3, #25
	sri.4s	v4, v3, #7
	ldr	q3, [sp, #528]                  ; 16-byte Folded Reload
	add.4s	v0, v0, v3
	add.4s	v11, v0, v9
	eor.16b	v0, v11, v2
	shl.4s	v2, v0, #24
	sri.4s	v2, v0, #8
	ldr	q0, [sp, #512]                  ; 16-byte Folded Reload
	add.4s	v0, v5, v0
	add.4s	v0, v0, v20
	str	q0, [sp, #160]                  ; 16-byte Folded Spill
	eor.16b	v0, v0, v19
	shl.4s	v3, v0, #24
	sri.4s	v3, v0, #8
	add.4s	v5, v2, v23
	eor.16b	v0, v5, v9
	shl.4s	v23, v0, #25
	sri.4s	v23, v0, #7
	add.4s	v26, v8, v18
	eor.16b	v18, v26, v15
	shl.4s	v19, v18, #25
	sri.4s	v19, v18, #7
	add.4s	v9, v3, v29
	eor.16b	v18, v9, v20
	shl.4s	v20, v18, #25
	sri.4s	v20, v18, #7
	ldr	q0, [sp, #464]                  ; 16-byte Folded Reload
	add.4s	v7, v7, v0
	add.4s	v15, v7, v23
	eor.16b	v7, v15, v13
	rev32.8h	v0, v7
	add.4s	v1, v1, v0
	eor.16b	v7, v1, v23
	shl.4s	v29, v7, #20
	sri.4s	v29, v7, #12
	ldr	q7, [sp, #448]                  ; 16-byte Folded Reload
	add.4s	v7, v10, v7
	add.4s	v10, v7, v20
	eor.16b	v7, v10, v8
	rev32.8h	v8, v7
	add.4s	v7, v12, v8
	eor.16b	v20, v7, v20
	shl.4s	v18, v20, #20
	sri.4s	v18, v20, #12
	ldr	q20, [sp, #816]                 ; 16-byte Folded Reload
	add.4s	v20, v20, v15
	add.4s	v13, v29, v20
	eor.16b	v0, v13, v0
	shl.4s	v23, v0, #24
	sri.4s	v23, v0, #8
	ldr	q0, [sp, #704]                  ; 16-byte Folded Reload
	add.4s	v0, v0, v10
	add.4s	v12, v18, v0
	eor.16b	v0, v12, v8
	shl.4s	v10, v0, #24
	sri.4s	v10, v0, #8
	ldr	q0, [sp, #432]                  ; 16-byte Folded Reload
	add.4s	v0, v0, v21
	add.4s	v0, v24, v0
	eor.16b	v21, v25, v0
	rev32.8h	v21, v21
	add.4s	v8, v5, v21
	eor.16b	v5, v8, v24
	shl.4s	v24, v5, #20
	sri.4s	v24, v5, #12
	ldr	q5, [sp, #416]                  ; 16-byte Folded Reload
	add.4s	v5, v5, v14
	add.4s	v5, v28, v5
	eor.16b	v22, v22, v5
	rev32.8h	v22, v22
	add.4s	v9, v9, v22
	eor.16b	v25, v9, v28
	shl.4s	v15, v25, #20
	sri.4s	v15, v25, #12
	add.4s	v0, v0, v31
	add.4s	v14, v0, v24
	eor.16b	v0, v14, v21
	shl.4s	v28, v0, #24
	sri.4s	v28, v0, #8
	add.4s	v0, v5, v30
	add.4s	v0, v0, v15
	str	q0, [sp, #176]                  ; 16-byte Folded Spill
	eor.16b	v0, v0, v22
	shl.4s	v25, v0, #24
	sri.4s	v25, v0, #8
	add.4s	v0, v1, v23
	str	q0, [sp, #192]                  ; 16-byte Folded Spill
	eor.16b	v0, v0, v29
	shl.4s	v29, v0, #25
	sri.4s	v29, v0, #7
	add.4s	v0, v28, v8
	str	q0, [sp, #208]                  ; 16-byte Folded Spill
	eor.16b	v0, v0, v24
	shl.4s	v20, v0, #25
	sri.4s	v20, v0, #7
	add.4s	v0, v25, v9
	str	q0, [sp, #272]                  ; 16-byte Folded Spill
	eor.16b	v0, v0, v15
	shl.4s	v15, v0, #25
	sri.4s	v15, v0, #7
	add.4s	v9, v7, v10
	eor.16b	v0, v9, v18
	shl.4s	v21, v0, #25
	sri.4s	v21, v0, #7
	ldr	q0, [sp, #752]                  ; 16-byte Folded Reload
	add.4s	v0, v0, v6
	add.4s	v6, v4, v0
	eor.16b	v2, v2, v6
	rev32.8h	v2, v2
	ldr	q0, [sp, #256]                  ; 16-byte Folded Reload
	add.4s	v18, v0, v2
	eor.16b	v4, v18, v4
	shl.4s	v7, v4, #20
	sri.4s	v7, v4, #12
	ldr	q0, [sp, #720]                  ; 16-byte Folded Reload
	add.4s	v4, v0, v16
	add.4s	v16, v19, v4
	eor.16b	v3, v3, v16
	rev32.8h	v0, v3
	ldr	q1, [sp, #224]                  ; 16-byte Folded Reload
	add.4s	v4, v1, v0
	eor.16b	v3, v4, v19
	shl.4s	v19, v3, #20
	sri.4s	v19, v3, #12
	ldr	q1, [sp, #608]                  ; 16-byte Folded Reload
	add.4s	v3, v6, v1
	add.4s	v22, v3, v7
	eor.16b	v2, v22, v2
	shl.4s	v24, v2, #24
	sri.4s	v24, v2, #8
	ldr	q1, [sp, #576]                  ; 16-byte Folded Reload
	add.4s	v2, v16, v1
	add.4s	v8, v2, v19
	eor.16b	v0, v8, v0
	shl.4s	v16, v0, #24
	sri.4s	v16, v0, #8
	add.4s	v18, v24, v18
	eor.16b	v0, v18, v7
	shl.4s	v5, v0, #25
	sri.4s	v5, v0, #7
	ldr	q0, [sp, #800]                  ; 16-byte Folded Reload
	add.4s	v0, v0, v11
	add.4s	v2, v27, v0
	ldr	q0, [sp, #304]                  ; 16-byte Folded Reload
	eor.16b	v7, v2, v0
	rev32.8h	v7, v7
	ldr	q0, [sp, #240]                  ; 16-byte Folded Reload
	add.4s	v11, v0, v7
	eor.16b	v27, v11, v27
	shl.4s	v30, v27, #20
	sri.4s	v30, v27, #12
	ldr	q0, [sp, #784]                  ; 16-byte Folded Reload
	ldr	q1, [sp, #160]                  ; 16-byte Folded Reload
	add.4s	v27, v0, v1
	add.4s	v27, v17, v27
	ldr	q0, [sp, #288]                  ; 16-byte Folded Reload
	eor.16b	v31, v27, v0
	rev32.8h	v1, v31
	add.4s	v3, v26, v1
	eor.16b	v17, v3, v17
	shl.4s	v0, v17, #20
	sri.4s	v0, v17, #12
	ldr	q6, [sp, #672]                  ; 16-byte Folded Reload
	add.4s	v2, v2, v6
	add.4s	v17, v2, v30
	eor.16b	v2, v17, v7
	shl.4s	v7, v2, #24
	sri.4s	v7, v2, #8
	ldr	q2, [sp, #640]                  ; 16-byte Folded Reload
	add.4s	v2, v27, v2
	add.4s	v31, v2, v0
	eor.16b	v1, v31, v1
	shl.4s	v27, v1, #24
	sri.4s	v27, v1, #8
	add.4s	v1, v16, v4
	eor.16b	v2, v1, v19
	shl.4s	v26, v2, #25
	sri.4s	v26, v2, #7
	add.4s	v11, v7, v11
	eor.16b	v2, v11, v30
	shl.4s	v4, v2, #25
	sri.4s	v4, v2, #7
	add.4s	v6, v27, v3
	eor.16b	v3, v6, v0
	shl.4s	v2, v3, #25
	sri.4s	v2, v3, #7
	ldr	q0, [sp, #624]                  ; 16-byte Folded Reload
	add.4s	v3, v13, v0
	add.4s	v3, v3, v20
	eor.16b	v7, v7, v3
	rev32.8h	v7, v7
	add.4s	v18, v18, v7
	eor.16b	v19, v18, v20
	shl.4s	v20, v19, #20
	sri.4s	v20, v19, #12
	ldr	q0, [sp, #592]                  ; 16-byte Folded Reload
	add.4s	v19, v12, v0
	add.4s	v30, v19, v15
	eor.16b	v19, v27, v30
	rev32.8h	v27, v19
	add.4s	v1, v1, v27
	eor.16b	v19, v1, v15
	shl.4s	v0, v19, #20
	sri.4s	v0, v19, #12
	ldr	q19, [sp, #496]                 ; 16-byte Folded Reload
	add.4s	v3, v3, v19
	add.4s	v19, v3, v20
	eor.16b	v3, v19, v7
	shl.4s	v12, v3, #24
	sri.4s	v12, v3, #8
	ldr	q3, [sp, #480]                  ; 16-byte Folded Reload
	add.4s	v3, v30, v3
	add.4s	v15, v3, v0
	eor.16b	v3, v15, v27
	shl.4s	v7, v3, #24
	sri.4s	v7, v3, #8
	stp	q12, q7, [sp, #288]             ; 32-byte Folded Spill
	add.4s	v3, v12, v18
	str	q3, [sp, #224]                  ; 16-byte Folded Spill
	eor.16b	v3, v3, v20
	shl.4s	v30, v3, #25
	sri.4s	v30, v3, #7
	add.4s	v1, v7, v1
	str	q1, [sp, #240]                  ; 16-byte Folded Spill
	eor.16b	v0, v1, v0
	shl.4s	v27, v0, #25
	sri.4s	v27, v0, #7
	ldp	q12, q13, [sp, #384]            ; 32-byte Folded Reload
	add.4s	v0, v14, v13
	add.4s	v0, v0, v5
	eor.16b	v1, v0, v23
	rev32.8h	v3, v1
	add.4s	v18, v11, v3
	eor.16b	v5, v18, v5
	shl.4s	v1, v5, #20
	sri.4s	v1, v5, #12
	ldr	q5, [sp, #176]                  ; 16-byte Folded Reload
	add.4s	v5, v5, v12
	add.4s	v5, v5, v26
	eor.16b	v7, v5, v10
	rev32.8h	v10, v7
	add.4s	v14, v6, v10
	eor.16b	v6, v14, v26
	shl.4s	v7, v6, #20
	sri.4s	v7, v6, #12
	ldr	q6, [sp, #560]                  ; 16-byte Folded Reload
	add.4s	v0, v0, v6
	add.4s	v26, v0, v1
	eor.16b	v0, v26, v3
	shl.4s	v11, v0, #24
	sri.4s	v11, v0, #8
	ldr	q0, [sp, #320]                  ; 16-byte Folded Reload
	add.4s	v0, v22, v0
	add.4s	v0, v0, v4
	eor.16b	v3, v0, v28
	rev32.8h	v3, v3
	ldr	q6, [sp, #192]                  ; 16-byte Folded Reload
	add.4s	v23, v6, v3
	eor.16b	v4, v23, v4
	shl.4s	v28, v4, #20
	sri.4s	v28, v4, #12
	ldr	q4, [sp, #336]                  ; 16-byte Folded Reload
	add.4s	v4, v8, v4
	add.4s	v4, v4, v2
	eor.16b	v6, v4, v25
	rev32.8h	v6, v6
	add.4s	v8, v9, v6
	eor.16b	v2, v8, v2
	shl.4s	v20, v2, #20
	sri.4s	v20, v2, #12
	ldr	q2, [sp, #544]                  ; 16-byte Folded Reload
	add.4s	v2, v5, v2
	add.4s	v22, v2, v7
	eor.16b	v2, v22, v10
	shl.4s	v9, v2, #24
	sri.4s	v9, v2, #8
	ldr	q2, [sp, #736]                  ; 16-byte Folded Reload
	add.4s	v0, v0, v2
	add.4s	v10, v0, v28
	eor.16b	v0, v10, v3
	shl.4s	v25, v0, #24
	sri.4s	v25, v0, #8
	ldr	q0, [sp, #688]                  ; 16-byte Folded Reload
	add.4s	v0, v4, v0
	add.4s	v0, v0, v20
	str	q0, [sp, #176]                  ; 16-byte Folded Spill
	eor.16b	v0, v0, v6
	shl.4s	v6, v0, #24
	sri.4s	v6, v0, #8
	add.4s	v3, v25, v23
	eor.16b	v0, v3, v28
	shl.4s	v23, v0, #25
	sri.4s	v23, v0, #7
	add.4s	v5, v6, v8
	eor.16b	v0, v5, v20
	shl.4s	v20, v0, #25
	sri.4s	v20, v0, #7
	ldr	q0, [sp, #528]                  ; 16-byte Folded Reload
	add.4s	v0, v0, v17
	add.4s	v28, v29, v0
	eor.16b	v2, v28, v24
	rev32.8h	v2, v2
	ldr	q0, [sp, #208]                  ; 16-byte Folded Reload
	add.4s	v17, v0, v2
	eor.16b	v4, v17, v29
	shl.4s	v24, v4, #20
	sri.4s	v24, v4, #12
	ldr	q0, [sp, #512]                  ; 16-byte Folded Reload
	add.4s	v4, v0, v31
	add.4s	v29, v21, v4
	eor.16b	v4, v29, v16
	rev32.8h	v16, v4
	ldr	q0, [sp, #272]                  ; 16-byte Folded Reload
	add.4s	v31, v0, v16
	eor.16b	v4, v31, v21
	shl.4s	v0, v4, #20
	sri.4s	v0, v4, #12
	ldr	q4, [sp, #768]                  ; 16-byte Folded Reload
	add.4s	v4, v28, v4
	add.4s	v28, v4, v24
	eor.16b	v2, v28, v2
	shl.4s	v4, v2, #24
	sri.4s	v4, v2, #8
	ldr	q2, [sp, #656]                  ; 16-byte Folded Reload
	add.4s	v2, v29, v2
	add.4s	v2, v2, v0
	str	q2, [sp, #192]                  ; 16-byte Folded Spill
	eor.16b	v2, v2, v16
	shl.4s	v8, v2, #24
	sri.4s	v8, v2, #8
	add.4s	v18, v11, v18
	eor.16b	v1, v18, v1
	shl.4s	v2, v1, #25
	sri.4s	v2, v1, #7
	add.4s	v21, v4, v17
	eor.16b	v1, v21, v24
	shl.4s	v24, v1, #25
	sri.4s	v24, v1, #7
	add.4s	v17, v8, v31
	eor.16b	v0, v17, v0
	shl.4s	v31, v0, #25
	sri.4s	v31, v0, #7
	add.4s	v14, v9, v14
	eor.16b	v0, v14, v7
	shl.4s	v16, v0, #25
	sri.4s	v16, v0, #7
	ldr	q0, [sp, #432]                  ; 16-byte Folded Reload
	add.4s	v0, v19, v0
	add.4s	v7, v0, v24
	eor.16b	v0, v7, v11
	rev32.8h	v19, v0
	add.4s	v0, v3, v19
	eor.16b	v3, v0, v24
	shl.4s	v1, v3, #20
	sri.4s	v1, v3, #12
	ldr	q3, [sp, #416]                  ; 16-byte Folded Reload
	add.4s	v3, v15, v3
	add.4s	v24, v3, v31
	eor.16b	v3, v24, v9
	rev32.8h	v9, v3
	add.4s	v29, v5, v9
	eor.16b	v5, v29, v31
	shl.4s	v3, v5, #20
	sri.4s	v3, v5, #12
	ldr	q5, [sp, #800]                  ; 16-byte Folded Reload
	add.4s	v5, v5, v7
	add.4s	v5, v1, v5
	eor.16b	v7, v5, v19
	shl.4s	v19, v7, #24
	sri.4s	v19, v7, #8
	ldr	q7, [sp, #368]                  ; 16-byte Folded Reload
	add.4s	v7, v7, v26
	add.4s	v7, v30, v7
	eor.16b	v25, v25, v7
	rev32.8h	v25, v25
	add.4s	v21, v21, v25
	eor.16b	v26, v21, v30
	shl.4s	v30, v26, #20
	sri.4s	v30, v26, #12
	ldr	q26, [sp, #352]                 ; 16-byte Folded Reload
	add.4s	v22, v26, v22
	add.4s	v22, v27, v22
	eor.16b	v6, v6, v22
	rev32.8h	v31, v6
	add.4s	v17, v17, v31
	eor.16b	v6, v17, v27
	shl.4s	v27, v6, #20
	sri.4s	v27, v6, #12
	ldr	q6, [sp, #784]                  ; 16-byte Folded Reload
	add.4s	v6, v6, v24
	add.4s	v6, v3, v6
	eor.16b	v26, v6, v9
	shl.4s	v24, v26, #24
	sri.4s	v24, v26, #8
	add.4s	v7, v7, v13
	add.4s	v9, v7, v30
	eor.16b	v7, v9, v25
	shl.4s	v25, v7, #24
	sri.4s	v25, v7, #8
	add.4s	v7, v22, v12
	add.4s	v7, v7, v27
	str	q7, [sp, #208]                  ; 16-byte Folded Spill
	eor.16b	v7, v7, v31
	shl.4s	v22, v7, #24
	sri.4s	v22, v7, #8
	add.4s	v7, v25, v21
	str	q7, [sp, #256]                  ; 16-byte Folded Spill
	eor.16b	v7, v7, v30
	shl.4s	v11, v7, #25
	sri.4s	v11, v7, #7
	add.4s	v7, v22, v17
	str	q7, [sp, #272]                  ; 16-byte Folded Spill
	eor.16b	v7, v7, v27
	shl.4s	v27, v7, #25
	sri.4s	v27, v7, #7
	ldr	q7, [sp, #672]                  ; 16-byte Folded Reload
	add.4s	v7, v7, v10
	add.4s	v17, v2, v7
	eor.16b	v4, v4, v17
	rev32.8h	v21, v4
	ldr	q4, [sp, #224]                  ; 16-byte Folded Reload
	add.4s	v7, v4, v21
	eor.16b	v2, v7, v2
	shl.4s	v31, v2, #20
	sri.4s	v31, v2, #12
	ldr	q2, [sp, #640]                  ; 16-byte Folded Reload
	ldr	q4, [sp, #176]                  ; 16-byte Folded Reload
	add.4s	v2, v2, v4
	add.4s	v2, v16, v2
	eor.16b	v4, v8, v2
	rev32.8h	v30, v4
	ldr	q4, [sp, #240]                  ; 16-byte Folded Reload
	add.4s	v4, v4, v30
	eor.16b	v8, v4, v16
	shl.4s	v16, v8, #20
	sri.4s	v16, v8, #12
	ldr	q26, [sp, #464]                 ; 16-byte Folded Reload
	add.4s	v17, v17, v26
	add.4s	v17, v17, v31
	str	q17, [sp, #160]                 ; 16-byte Folded Spill
	eor.16b	v21, v17, v21
	shl.4s	v26, v21, #24
	sri.4s	v26, v21, #8
	ldr	q17, [sp, #448]                 ; 16-byte Folded Reload
	add.4s	v2, v2, v17
	add.4s	v2, v2, v16
	str	q2, [sp, #240]                  ; 16-byte Folded Spill
	eor.16b	v2, v2, v30
	shl.4s	v8, v2, #24
	sri.4s	v8, v2, #8
	add.4s	v10, v0, v19
	eor.16b	v0, v10, v1
	shl.4s	v21, v0, #25
	sri.4s	v21, v0, #7
	ldr	q0, [sp, #752]                  ; 16-byte Folded Reload
	add.4s	v0, v0, v28
	add.4s	v0, v23, v0
	ldr	q1, [sp, #288]                  ; 16-byte Folded Reload
	eor.16b	v1, v0, v1
	rev32.8h	v1, v1
	add.4s	v2, v18, v1
	eor.16b	v18, v2, v23
	shl.4s	v12, v18, #20
	sri.4s	v12, v18, #12
	add.4s	v30, v29, v24
	eor.16b	v3, v30, v3
	shl.4s	v23, v3, #25
	sri.4s	v23, v3, #7
	ldr	q3, [sp, #720]                  ; 16-byte Folded Reload
	ldr	q17, [sp, #192]                 ; 16-byte Folded Reload
	add.4s	v3, v3, v17
	add.4s	v28, v20, v3
	ldr	q3, [sp, #304]                  ; 16-byte Folded Reload
	eor.16b	v3, v28, v3
	rev32.8h	v29, v3
	add.4s	v14, v14, v29
	eor.16b	v3, v14, v20
	shl.4s	v20, v3, #20
	sri.4s	v20, v3, #12
	add.4s	v3, v26, v7
	eor.16b	v31, v3, v31
	shl.4s	v7, v31, #25
	sri.4s	v7, v31, #7
	ldr	q17, [sp, #736]                 ; 16-byte Folded Reload
	add.4s	v0, v0, v17
	add.4s	v13, v0, v12
	eor.16b	v0, v13, v1
	shl.4s	v31, v0, #24
	sri.4s	v31, v0, #8
	ldr	q0, [sp, #688]                  ; 16-byte Folded Reload
	add.4s	v0, v28, v0
	add.4s	v0, v0, v20
	str	q0, [sp, #176]                  ; 16-byte Folded Spill
	eor.16b	v0, v0, v29
	shl.4s	v29, v0, #24
	sri.4s	v29, v0, #8
	add.4s	v28, v31, v2
	eor.16b	v0, v28, v12
	shl.4s	v17, v0, #25
	sri.4s	v17, v0, #7
	add.4s	v4, v8, v4
	eor.16b	v0, v4, v16
	shl.4s	v16, v0, #25
	sri.4s	v16, v0, #7
	add.4s	v18, v29, v14
	eor.16b	v20, v18, v20
	shl.4s	v2, v20, #25
	sri.4s	v2, v20, #7
	ldr	q0, [sp, #816]                  ; 16-byte Folded Reload
	add.4s	v5, v5, v0
	add.4s	v5, v5, v11
	eor.16b	v20, v31, v5
	rev32.8h	v20, v20
	add.4s	v3, v3, v20
	eor.16b	v31, v3, v11
	shl.4s	v11, v31, #20
	sri.4s	v11, v31, #12
	ldr	q0, [sp, #704]                  ; 16-byte Folded Reload
	add.4s	v6, v6, v0
	add.4s	v6, v6, v27
	eor.16b	v29, v29, v6
	rev32.8h	v29, v29
	add.4s	v1, v4, v29
	eor.16b	v4, v1, v27
	shl.4s	v27, v4, #20
	sri.4s	v27, v4, #12
	ldr	q4, [sp, #560]                  ; 16-byte Folded Reload
	add.4s	v4, v5, v4
	add.4s	v12, v4, v11
	eor.16b	v4, v12, v20
	shl.4s	v0, v4, #24
	sri.4s	v0, v4, #8
	mov.16b	v20, v0
	str	q0, [sp, #304]                  ; 16-byte Folded Spill
	ldr	q4, [sp, #544]                  ; 16-byte Folded Reload
	add.4s	v4, v6, v4
	add.4s	v14, v4, v27
	eor.16b	v4, v14, v29
	shl.4s	v0, v4, #24
	sri.4s	v0, v4, #8
	str	q0, [sp, #288]                  ; 16-byte Folded Spill
	ldr	q15, [sp, #320]                 ; 16-byte Folded Reload
	add.4s	v4, v9, v15
	add.4s	v5, v4, v7
	eor.16b	v4, v5, v19
	rev32.8h	v6, v4
	add.4s	v4, v28, v6
	eor.16b	v7, v4, v7
	shl.4s	v31, v7, #20
	sri.4s	v31, v7, #12
	add.4s	v3, v20, v3
	str	q3, [sp, #224]                  ; 16-byte Folded Spill
	eor.16b	v3, v3, v11
	shl.4s	v28, v3, #25
	sri.4s	v28, v3, #7
	ldr	q20, [sp, #336]                 ; 16-byte Folded Reload
	ldr	q3, [sp, #208]                  ; 16-byte Folded Reload
	add.4s	v3, v3, v20
	add.4s	v19, v3, v16
	eor.16b	v3, v19, v24
	rev32.8h	v29, v3
	add.4s	v7, v18, v29
	eor.16b	v3, v7, v16
	shl.4s	v24, v3, #20
	sri.4s	v24, v3, #12
	add.4s	v0, v0, v1
	str	q0, [sp, #208]                  ; 16-byte Folded Spill
	eor.16b	v0, v0, v27
	shl.4s	v9, v0, #25
	sri.4s	v9, v0, #7
	ldr	q0, [sp, #608]                  ; 16-byte Folded Reload
	add.4s	v0, v5, v0
	add.4s	v5, v0, v31
	eor.16b	v0, v5, v6
	shl.4s	v27, v0, #24
	sri.4s	v27, v0, #8
	ldr	q0, [sp, #496]                  ; 16-byte Folded Reload
	ldr	q1, [sp, #160]                  ; 16-byte Folded Reload
	add.4s	v0, v1, v0
	add.4s	v0, v0, v17
	eor.16b	v3, v0, v25
	rev32.8h	v3, v3
	add.4s	v6, v10, v3
	eor.16b	v1, v6, v17
	shl.4s	v16, v1, #20
	sri.4s	v16, v1, #12
	ldr	q1, [sp, #480]                  ; 16-byte Folded Reload
	ldr	q17, [sp, #240]                 ; 16-byte Folded Reload
	add.4s	v1, v17, v1
	add.4s	v1, v1, v2
	eor.16b	v17, v1, v22
	rev32.8h	v17, v17
	add.4s	v18, v30, v17
	eor.16b	v2, v18, v2
	shl.4s	v11, v2, #20
	sri.4s	v11, v2, #12
	ldr	q2, [sp, #528]                  ; 16-byte Folded Reload
	add.4s	v0, v0, v2
	add.4s	v25, v0, v16
	eor.16b	v0, v25, v3
	shl.4s	v10, v0, #24
	sri.4s	v10, v0, #8
	ldr	q0, [sp, #512]                  ; 16-byte Folded Reload
	add.4s	v0, v1, v0
	add.4s	v0, v0, v11
	str	q0, [sp, #144]                  ; 16-byte Folded Spill
	eor.16b	v0, v0, v17
	shl.4s	v22, v0, #24
	sri.4s	v22, v0, #8
	ldr	q0, [sp, #576]                  ; 16-byte Folded Reload
	add.4s	v0, v19, v0
	add.4s	v19, v0, v24
	eor.16b	v0, v19, v29
	shl.4s	v1, v0, #24
	sri.4s	v1, v0, #8
	add.4s	v0, v10, v6
	eor.16b	v2, v0, v16
	shl.4s	v30, v2, #25
	sri.4s	v30, v2, #7
	add.4s	v2, v22, v18
	eor.16b	v3, v2, v11
	shl.4s	v29, v3, #25
	sri.4s	v29, v3, #7
	add.4s	v3, v27, v4
	str	q3, [sp, #192]                  ; 16-byte Folded Spill
	eor.16b	v3, v3, v31
	shl.4s	v31, v3, #25
	sri.4s	v31, v3, #7
	ldr	q3, [sp, #768]                  ; 16-byte Folded Reload
	add.4s	v3, v3, v13
	add.4s	v3, v21, v3
	eor.16b	v4, v3, v26
	rev32.8h	v4, v4
	ldr	q6, [sp, #256]                  ; 16-byte Folded Reload
	add.4s	v6, v6, v4
	eor.16b	v16, v6, v21
	shl.4s	v17, v16, #20
	sri.4s	v17, v16, #12
	ldr	q16, [sp, #656]                 ; 16-byte Folded Reload
	ldr	q18, [sp, #176]                 ; 16-byte Folded Reload
	add.4s	v16, v16, v18
	add.4s	v16, v23, v16
	eor.16b	v18, v16, v8
	rev32.8h	v21, v18
	ldr	q18, [sp, #272]                 ; 16-byte Folded Reload
	add.4s	v26, v18, v21
	eor.16b	v18, v26, v23
	shl.4s	v13, v18, #20
	sri.4s	v13, v18, #12
	ldr	q18, [sp, #624]                 ; 16-byte Folded Reload
	add.4s	v3, v3, v18
	add.4s	v18, v3, v17
	eor.16b	v3, v18, v4
	shl.4s	v11, v3, #24
	sri.4s	v11, v3, #8
	ldr	q3, [sp, #592]                  ; 16-byte Folded Reload
	add.4s	v3, v16, v3
	add.4s	v16, v3, v13
	eor.16b	v3, v16, v21
	shl.4s	v23, v3, #24
	sri.4s	v23, v3, #8
	add.4s	v7, v1, v7
	eor.16b	v3, v7, v24
	shl.4s	v8, v3, #25
	sri.4s	v8, v3, #7
	add.4s	v21, v11, v6
	eor.16b	v3, v21, v17
	shl.4s	v6, v3, #25
	sri.4s	v6, v3, #7
	add.4s	v4, v23, v26
	eor.16b	v3, v4, v13
	shl.4s	v26, v3, #25
	sri.4s	v26, v3, #7
	ldr	q3, [sp, #368]                  ; 16-byte Folded Reload
	add.4s	v3, v12, v3
	add.4s	v12, v3, v6
	eor.16b	v3, v12, v27
	rev32.8h	v27, v3
	add.4s	v3, v0, v27
	eor.16b	v6, v3, v6
	shl.4s	v0, v6, #20
	sri.4s	v0, v6, #12
	ldr	q6, [sp, #352]                  ; 16-byte Folded Reload
	add.4s	v6, v14, v6
	add.4s	v17, v6, v26
	eor.16b	v1, v17, v1
	rev32.8h	v14, v1
	add.4s	v24, v2, v14
	eor.16b	v2, v24, v26
	shl.4s	v1, v2, #20
	sri.4s	v1, v2, #12
	ldr	q2, [sp, #752]                  ; 16-byte Folded Reload
	add.4s	v2, v2, v12
	add.4s	v6, v0, v2
	eor.16b	v2, v6, v27
	shl.4s	v27, v2, #24
	sri.4s	v27, v2, #8
	ldr	q2, [sp, #400]                  ; 16-byte Folded Reload
	add.4s	v2, v2, v5
	add.4s	v2, v28, v2
	eor.16b	v5, v10, v2
	rev32.8h	v5, v5
	add.4s	v10, v21, v5
	eor.16b	v21, v10, v28
	shl.4s	v28, v21, #20
	sri.4s	v28, v21, #12
	ldr	q21, [sp, #384]                 ; 16-byte Folded Reload
	add.4s	v19, v21, v19
	add.4s	v19, v9, v19
	eor.16b	v21, v22, v19
	rev32.8h	v21, v21
	add.4s	v12, v4, v21
	eor.16b	v4, v12, v9
	shl.4s	v9, v4, #20
	sri.4s	v9, v4, #12
	add.4s	v2, v2, v15
	add.4s	v2, v2, v28
	str	q2, [sp, #128]                  ; 16-byte Folded Spill
	eor.16b	v2, v2, v5
	shl.4s	v22, v2, #24
	sri.4s	v22, v2, #8
	add.4s	v2, v19, v20
	add.4s	v2, v2, v9
	str	q2, [sp, #160]                  ; 16-byte Folded Spill
	eor.16b	v2, v2, v21
	shl.4s	v21, v2, #24
	sri.4s	v21, v2, #8
	ldr	q2, [sp, #720]                  ; 16-byte Folded Reload
	add.4s	v2, v2, v17
	add.4s	v13, v1, v2
	eor.16b	v2, v13, v14
	shl.4s	v14, v2, #24
	sri.4s	v14, v2, #8
	add.4s	v2, v22, v10
	str	q2, [sp, #240]                  ; 16-byte Folded Spill
	eor.16b	v2, v2, v28
	shl.4s	v10, v2, #25
	sri.4s	v10, v2, #7
	add.4s	v2, v21, v12
	str	q2, [sp, #256]                  ; 16-byte Folded Spill
	eor.16b	v2, v2, v9
	shl.4s	v19, v2, #25
	sri.4s	v19, v2, #7
	add.4s	v2, v3, v27
	str	q2, [sp, #176]                  ; 16-byte Folded Spill
	eor.16b	v0, v2, v0
	shl.4s	v26, v0, #25
	sri.4s	v26, v0, #7
	ldr	q0, [sp, #736]                  ; 16-byte Folded Reload
	add.4s	v0, v0, v25
	add.4s	v5, v31, v0
	eor.16b	v0, v11, v5
	rev32.8h	v17, v0
	ldr	q0, [sp, #224]                  ; 16-byte Folded Reload
	add.4s	v0, v0, v17
	eor.16b	v20, v0, v31
	shl.4s	v2, v20, #20
	sri.4s	v2, v20, #12
	ldr	q3, [sp, #688]                  ; 16-byte Folded Reload
	ldr	q4, [sp, #144]                  ; 16-byte Folded Reload
	add.4s	v20, v3, v4
	add.4s	v20, v8, v20
	eor.16b	v23, v23, v20
	rev32.8h	v25, v23
	ldr	q3, [sp, #208]                  ; 16-byte Folded Reload
	add.4s	v31, v3, v25
	eor.16b	v23, v31, v8
	shl.4s	v8, v23, #20
	sri.4s	v8, v23, #12
	ldp	q3, q4, [sp, #416]              ; 32-byte Folded Reload
	add.4s	v5, v5, v4
	add.4s	v12, v5, v2
	eor.16b	v5, v12, v17
	shl.4s	v23, v5, #24
	sri.4s	v23, v5, #8
	add.4s	v5, v20, v3
	add.4s	v17, v5, v8
	eor.16b	v5, v17, v25
	shl.4s	v25, v5, #24
	sri.4s	v25, v5, #8
	add.4s	v20, v24, v14
	eor.16b	v1, v20, v1
	shl.4s	v28, v1, #25
	sri.4s	v28, v1, #7
	ldr	q1, [sp, #672]                  ; 16-byte Folded Reload
	add.4s	v1, v1, v18
	add.4s	v1, v30, v1
	ldr	q3, [sp, #304]                  ; 16-byte Folded Reload
	eor.16b	v5, v1, v3
	rev32.8h	v5, v5
	ldr	q3, [sp, #192]                  ; 16-byte Folded Reload
	add.4s	v24, v3, v5
	eor.16b	v18, v24, v30
	shl.4s	v11, v18, #20
	sri.4s	v11, v18, #12
	ldr	q3, [sp, #640]                  ; 16-byte Folded Reload
	add.4s	v16, v3, v16
	add.4s	v16, v29, v16
	ldr	q3, [sp, #288]                  ; 16-byte Folded Reload
	eor.16b	v18, v16, v3
	rev32.8h	v15, v18
	add.4s	v7, v7, v15
	eor.16b	v18, v7, v29
	shl.4s	v29, v18, #20
	sri.4s	v29, v18, #12
	ldr	q3, [sp, #528]                  ; 16-byte Folded Reload
	add.4s	v1, v1, v3
	add.4s	v18, v1, v11
	eor.16b	v1, v18, v5
	shl.4s	v5, v1, #24
	sri.4s	v5, v1, #8
	ldr	q1, [sp, #512]                  ; 16-byte Folded Reload
	add.4s	v1, v16, v1
	add.4s	v30, v1, v29
	eor.16b	v1, v30, v15
	shl.4s	v15, v1, #24
	sri.4s	v15, v1, #8
	add.4s	v0, v23, v0
	eor.16b	v1, v0, v2
	shl.4s	v2, v1, #25
	sri.4s	v2, v1, #7
	add.4s	v24, v5, v24
	eor.16b	v3, v24, v11
	shl.4s	v4, v3, #25
	sri.4s	v4, v3, #7
	add.4s	v3, v15, v7
	eor.16b	v7, v3, v29
	shl.4s	v11, v7, #25
	sri.4s	v11, v7, #7
	add.4s	v7, v25, v31
	eor.16b	v29, v7, v8
	shl.4s	v16, v29, #25
	sri.4s	v16, v29, #7
	ldr	q1, [sp, #800]                  ; 16-byte Folded Reload
	add.4s	v6, v6, v1
	add.4s	v6, v6, v10
	eor.16b	v5, v5, v6
	rev32.8h	v5, v5
	add.4s	v1, v0, v5
	eor.16b	v29, v1, v10
	shl.4s	v31, v29, #20
	sri.4s	v31, v29, #12
	ldr	q0, [sp, #784]                  ; 16-byte Folded Reload
	add.4s	v29, v13, v0
	add.4s	v29, v29, v19
	eor.16b	v8, v15, v29
	rev32.8h	v8, v8
	add.4s	v10, v7, v8
	eor.16b	v7, v10, v19
	shl.4s	v0, v7, #20
	sri.4s	v0, v7, #12
	ldr	q7, [sp, #608]                  ; 16-byte Folded Reload
	add.4s	v6, v6, v7
	add.4s	v19, v6, v31
	eor.16b	v5, v19, v5
	shl.4s	v7, v5, #24
	sri.4s	v7, v5, #8
	ldr	q5, [sp, #576]                  ; 16-byte Folded Reload
	add.4s	v5, v29, v5
	add.4s	v13, v5, v0
	eor.16b	v5, v13, v8
	shl.4s	v6, v5, #24
	sri.4s	v6, v5, #8
	stp	q6, q7, [sp, #288]              ; 32-byte Folded Spill
	add.4s	v1, v7, v1
	str	q1, [sp, #272]                  ; 16-byte Folded Spill
	eor.16b	v1, v1, v31
	shl.4s	v29, v1, #25
	sri.4s	v29, v1, #7
	ldr	q1, [sp, #496]                  ; 16-byte Folded Reload
	ldr	q5, [sp, #128]                  ; 16-byte Folded Reload
	add.4s	v1, v5, v1
	add.4s	v1, v1, v2
	eor.16b	v5, v1, v27
	rev32.8h	v5, v5
	add.4s	v31, v24, v5
	eor.16b	v2, v31, v2
	shl.4s	v27, v2, #20
	sri.4s	v27, v2, #12
	add.4s	v2, v6, v10
	str	q2, [sp, #224]                  ; 16-byte Folded Spill
	eor.16b	v0, v2, v0
	shl.4s	v24, v0, #25
	sri.4s	v24, v0, #7
	ldr	q0, [sp, #480]                  ; 16-byte Folded Reload
	ldr	q2, [sp, #160]                  ; 16-byte Folded Reload
	add.4s	v0, v2, v0
	add.4s	v0, v0, v16
	eor.16b	v2, v0, v14
	rev32.8h	v6, v2
	add.4s	v8, v3, v6
	eor.16b	v3, v8, v16
	shl.4s	v2, v3, #20
	sri.4s	v2, v3, #12
	ldr	q3, [sp, #464]                  ; 16-byte Folded Reload
	add.4s	v1, v1, v3
	add.4s	v9, v1, v27
	eor.16b	v1, v9, v5
	shl.4s	v10, v1, #24
	sri.4s	v10, v1, #8
	ldr	q1, [sp, #448]                  ; 16-byte Folded Reload
	add.4s	v0, v0, v1
	add.4s	v5, v0, v2
	eor.16b	v0, v5, v6
	shl.4s	v14, v0, #24
	sri.4s	v14, v0, #8
	ldr	q0, [sp, #560]                  ; 16-byte Folded Reload
	add.4s	v0, v12, v0
	add.4s	v0, v0, v4
	eor.16b	v1, v0, v22
	rev32.8h	v1, v1
	ldr	q3, [sp, #176]                  ; 16-byte Folded Reload
	add.4s	v3, v3, v1
	eor.16b	v4, v3, v4
	shl.4s	v22, v4, #20
	sri.4s	v22, v4, #12
	ldr	q4, [sp, #544]                  ; 16-byte Folded Reload
	add.4s	v4, v17, v4
	add.4s	v4, v4, v11
	eor.16b	v6, v4, v21
	rev32.8h	v16, v6
	add.4s	v6, v20, v16
	eor.16b	v17, v6, v11
	shl.4s	v11, v17, #20
	sri.4s	v11, v17, #12
	ldr	q7, [sp, #768]                  ; 16-byte Folded Reload
	add.4s	v0, v0, v7
	add.4s	v12, v0, v22
	eor.16b	v0, v12, v1
	shl.4s	v21, v0, #24
	sri.4s	v21, v0, #8
	ldr	q0, [sp, #656]                  ; 16-byte Folded Reload
	add.4s	v0, v4, v0
	add.4s	v15, v0, v11
	eor.16b	v0, v15, v16
	shl.4s	v1, v0, #24
	sri.4s	v1, v0, #8
	add.4s	v16, v21, v3
	eor.16b	v0, v16, v22
	shl.4s	v20, v0, #25
	sri.4s	v20, v0, #7
	ldr	q0, [sp, #624]                  ; 16-byte Folded Reload
	add.4s	v0, v0, v18
	add.4s	v0, v26, v0
	eor.16b	v3, v0, v23
	rev32.8h	v3, v3
	ldr	q4, [sp, #240]                  ; 16-byte Folded Reload
	add.4s	v4, v4, v3
	eor.16b	v17, v4, v26
	shl.4s	v18, v17, #20
	sri.4s	v18, v17, #12
	ldr	q7, [sp, #592]                  ; 16-byte Folded Reload
	add.4s	v17, v7, v30
	add.4s	v17, v28, v17
	eor.16b	v22, v17, v25
	rev32.8h	v22, v22
	ldr	q7, [sp, #256]                  ; 16-byte Folded Reload
	add.4s	v26, v7, v22
	eor.16b	v23, v26, v28
	shl.4s	v28, v23, #20
	sri.4s	v28, v23, #12
	ldr	q7, [sp, #816]                  ; 16-byte Folded Reload
	add.4s	v0, v0, v7
	add.4s	v0, v0, v18
	str	q0, [sp, #256]                  ; 16-byte Folded Spill
	eor.16b	v0, v0, v3
	shl.4s	v30, v0, #24
	sri.4s	v30, v0, #8
	ldr	q0, [sp, #704]                  ; 16-byte Folded Reload
	add.4s	v0, v17, v0
	add.4s	v0, v0, v28
	str	q0, [sp, #240]                  ; 16-byte Folded Spill
	eor.16b	v3, v0, v22
	shl.4s	v7, v3, #24
	sri.4s	v7, v3, #8
	add.4s	v22, v1, v6
	eor.16b	v3, v22, v11
	shl.4s	v17, v3, #25
	sri.4s	v17, v3, #7
	add.4s	v6, v30, v4
	eor.16b	v3, v6, v18
	shl.4s	v4, v3, #25
	sri.4s	v4, v3, #7
	add.4s	v3, v7, v26
	eor.16b	v18, v3, v28
	shl.4s	v26, v18, #25
	sri.4s	v26, v18, #7
	add.4s	v23, v10, v31
	eor.16b	v27, v23, v27
	shl.4s	v11, v27, #25
	sri.4s	v11, v27, #7
	ldr	q0, [sp, #400]                  ; 16-byte Folded Reload
	add.4s	v19, v19, v0
	ldr	q0, [sp, #384]                  ; 16-byte Folded Reload
	add.4s	v31, v13, v0
	add.4s	v28, v14, v8
	eor.16b	v2, v28, v2
	shl.4s	v27, v2, #25
	sri.4s	v27, v2, #7
	add.4s	v2, v19, v4
	eor.16b	v19, v2, v10
	ldr	q0, [sp, #672]                  ; 16-byte Folded Reload
	add.4s	v8, v0, v2
	add.4s	v2, v31, v26
	eor.16b	v31, v2, v14
	ldr	q0, [sp, #640]                  ; 16-byte Folded Reload
	add.4s	v14, v0, v2
	ldr	q0, [sp, #320]                  ; 16-byte Folded Reload
	add.4s	v10, v0, v9
	rev32.8h	v19, v19
	add.4s	v18, v16, v19
	eor.16b	v4, v18, v4
	shl.4s	v2, v4, #20
	sri.4s	v2, v4, #12
	ldr	q0, [sp, #336]                  ; 16-byte Folded Reload
	add.4s	v5, v0, v5
	rev32.8h	v16, v31
	add.4s	v25, v22, v16
	eor.16b	v22, v25, v26
	shl.4s	v13, v22, #20
	sri.4s	v13, v22, #12
	add.4s	v26, v29, v10
	eor.16b	v21, v21, v26
	rev32.8h	v10, v21
	add.4s	v6, v6, v10
	eor.16b	v21, v6, v29
	shl.4s	v29, v21, #20
	sri.4s	v29, v21, #12
	add.4s	v5, v24, v5
	eor.16b	v1, v1, v5
	rev32.8h	v0, v1
	add.4s	v3, v3, v0
	eor.16b	v1, v3, v24
	shl.4s	v9, v1, #20
	sri.4s	v9, v1, #12
	add.4s	v31, v2, v8
	eor.16b	v1, v31, v19
	shl.4s	v22, v1, #24
	sri.4s	v22, v1, #8
	ldp	q4, q1, [sp, #480]              ; 32-byte Folded Reload
	add.4s	v1, v26, v1
	add.4s	v5, v5, v4
	add.4s	v8, v13, v14
	eor.16b	v16, v8, v16
	shl.4s	v21, v16, #24
	sri.4s	v21, v16, #8
	add.4s	v1, v1, v29
	str	q1, [sp, #496]                  ; 16-byte Folded Spill
	eor.16b	v1, v1, v10
	shl.4s	v16, v1, #24
	sri.4s	v16, v1, #8
	add.4s	v24, v5, v9
	eor.16b	v0, v24, v0
	shl.4s	v5, v0, #24
	sri.4s	v5, v0, #8
	add.4s	v0, v16, v6
	str	q0, [sp, #672]                  ; 16-byte Folded Spill
	eor.16b	v0, v0, v29
	shl.4s	v14, v0, #25
	sri.4s	v14, v0, #7
	ldp	q1, q0, [sp, #512]              ; 32-byte Folded Reload
	add.4s	v0, v0, v12
	add.4s	v6, v1, v15
	add.4s	v1, v5, v3
	str	q1, [sp, #640]                  ; 16-byte Folded Spill
	eor.16b	v3, v1, v9
	shl.4s	v9, v3, #25
	sri.4s	v9, v3, #7
	add.4s	v3, v11, v0
	eor.16b	v0, v30, v3
	rev32.8h	v19, v0
	ldr	q0, [sp, #272]                  ; 16-byte Folded Reload
	add.4s	v30, v0, v19
	eor.16b	v0, v30, v11
	shl.4s	v11, v0, #20
	sri.4s	v11, v0, #12
	add.4s	v26, v27, v6
	eor.16b	v0, v7, v26
	rev32.8h	v7, v0
	ldr	q0, [sp, #224]                  ; 16-byte Folded Reload
	add.4s	v4, v0, v7
	eor.16b	v27, v4, v27
	shl.4s	v6, v27, #20
	sri.4s	v6, v27, #12
	ldp	q0, q1, [sp, #352]              ; 32-byte Folded Reload
	add.4s	v3, v3, v1
	add.4s	v26, v26, v0
	add.4s	v27, v3, v11
	eor.16b	v19, v27, v19
	shl.4s	v29, v19, #24
	sri.4s	v29, v19, #8
	add.4s	v0, v26, v6
	str	q0, [sp, #528]                  ; 16-byte Folded Spill
	eor.16b	v7, v0, v7
	shl.4s	v12, v7, #24
	sri.4s	v12, v7, #8
	ldr	q0, [sp, #736]                  ; 16-byte Folded Reload
	ldp	q1, q3, [sp, #240]              ; 32-byte Folded Reload
	add.4s	v19, v0, v3
	ldr	q0, [sp, #688]                  ; 16-byte Folded Reload
	add.4s	v10, v0, v1
	add.4s	v26, v18, v22
	eor.16b	v2, v26, v2
	shl.4s	v7, v2, #25
	sri.4s	v7, v2, #7
	add.4s	v2, v20, v19
	ldp	q0, q1, [sp, #288]              ; 32-byte Folded Reload
	eor.16b	v18, v2, v1
	rev32.8h	v18, v18
	add.4s	v23, v23, v18
	eor.16b	v19, v23, v20
	shl.4s	v20, v19, #20
	sri.4s	v20, v19, #12
	add.4s	v19, v17, v10
	eor.16b	v10, v19, v0
	rev32.8h	v15, v10
	add.4s	v1, v28, v15
	eor.16b	v17, v1, v17
	shl.4s	v0, v17, #20
	sri.4s	v0, v17, #12
	add.4s	v28, v25, v21
	eor.16b	v17, v28, v13
	shl.4s	v3, v17, #25
	sri.4s	v3, v17, #7
	ldr	q17, [sp, #768]                 ; 16-byte Folded Reload
	add.4s	v2, v2, v17
	ldr	q17, [sp, #656]                 ; 16-byte Folded Reload
	add.4s	v17, v19, v17
	add.4s	v25, v29, v30
	eor.16b	v19, v25, v11
	shl.4s	v10, v19, #25
	sri.4s	v10, v19, #7
	add.4s	v2, v2, v20
	str	q2, [sp, #736]                  ; 16-byte Folded Spill
	eor.16b	v2, v2, v18
	shl.4s	v18, v2, #24
	sri.4s	v18, v2, #8
	add.4s	v2, v17, v0
	str	q2, [sp, #688]                  ; 16-byte Folded Spill
	eor.16b	v2, v2, v15
	shl.4s	v17, v2, #24
	sri.4s	v17, v2, #8
	add.4s	v4, v12, v4
	eor.16b	v6, v4, v6
	shl.4s	v2, v6, #25
	sri.4s	v2, v6, #7
	add.4s	v6, v18, v23
	eor.16b	v20, v6, v20
	shl.4s	v11, v20, #25
	sri.4s	v11, v20, #7
	add.4s	v15, v17, v1
	eor.16b	v0, v15, v0
	shl.4s	v30, v0, #25
	sri.4s	v30, v0, #7
	ldr	q0, [sp, #752]                  ; 16-byte Folded Reload
	add.4s	v0, v31, v0
	ldr	q1, [sp, #720]                  ; 16-byte Folded Reload
	add.4s	v1, v8, v1
	add.4s	v0, v0, v14
	eor.16b	v18, v18, v0
	rev32.8h	v18, v18
	add.4s	v31, v25, v18
	eor.16b	v20, v31, v14
	shl.4s	v13, v20, #20
	sri.4s	v13, v20, #12
	add.4s	v1, v1, v9
	eor.16b	v17, v17, v1
	rev32.8h	v23, v17
	add.4s	v8, v4, v23
	eor.16b	v4, v8, v9
	shl.4s	v9, v4, #20
	sri.4s	v9, v4, #12
	ldr	q4, [sp, #464]                  ; 16-byte Folded Reload
	add.4s	v0, v0, v4
	ldr	q4, [sp, #448]                  ; 16-byte Folded Reload
	add.4s	v4, v1, v4
	add.4s	v0, v0, v13
	str	q0, [sp, #768]                  ; 16-byte Folded Spill
	eor.16b	v0, v0, v18
	shl.4s	v20, v0, #24
	sri.4s	v20, v0, #8
	ldr	q0, [sp, #560]                  ; 16-byte Folded Reload
	ldr	q1, [sp, #496]                  ; 16-byte Folded Reload
	add.4s	v0, v1, v0
	ldr	q1, [sp, #544]                  ; 16-byte Folded Reload
	add.4s	v18, v24, v1
	add.4s	v0, v0, v10
	eor.16b	v22, v0, v22
	ldr	q1, [sp, #432]                  ; 16-byte Folded Reload
	add.4s	v19, v0, v1
	add.4s	v0, v18, v2
	eor.16b	v21, v0, v21
	ldr	q1, [sp, #416]                  ; 16-byte Folded Reload
	add.4s	v18, v0, v1
	add.4s	v14, v4, v9
	eor.16b	v0, v14, v23
	shl.4s	v23, v0, #24
	sri.4s	v23, v0, #8
	ldr	q0, [sp, #608]                  ; 16-byte Folded Reload
	add.4s	v4, v27, v0
	rev32.8h	v25, v22
	add.4s	v24, v6, v25
	eor.16b	v0, v24, v10
	shl.4s	v27, v0, #20
	sri.4s	v27, v0, #12
	ldr	q0, [sp, #576]                  ; 16-byte Folded Reload
	ldr	q1, [sp, #528]                  ; 16-byte Folded Reload
	add.4s	v6, v1, v0
	rev32.8h	v1, v21
	add.4s	v21, v15, v1
	eor.16b	v2, v21, v2
	shl.4s	v22, v2, #20
	sri.4s	v22, v2, #12
	add.4s	v2, v4, v11
	eor.16b	v4, v2, v16
	rev32.8h	v4, v4
	add.4s	v16, v26, v4
	eor.16b	v26, v16, v11
	shl.4s	v11, v26, #20
	sri.4s	v11, v26, #12
	add.4s	v26, v6, v30
	eor.16b	v5, v26, v5
	rev32.8h	v0, v5
	add.4s	v10, v28, v0
	eor.16b	v5, v10, v30
	shl.4s	v15, v5, #20
	sri.4s	v15, v5, #12
	add.4s	v6, v20, v31
	eor.16b	v5, v6, v13
	shl.4s	v13, v5, #25
	sri.4s	v13, v5, #7
	ldr	q5, [sp, #624]                  ; 16-byte Folded Reload
	add.4s	v28, v2, v5
	ldr	q2, [sp, #592]                  ; 16-byte Folded Reload
	add.4s	v26, v26, v2
	add.4s	v17, v23, v8
	eor.16b	v30, v17, v9
	shl.4s	v2, v30, #25
	sri.4s	v2, v30, #7
	add.4s	v30, v28, v11
	eor.16b	v4, v30, v4
	shl.4s	v31, v4, #24
	sri.4s	v31, v4, #8
	add.4s	v28, v26, v15
	eor.16b	v0, v28, v0
	shl.4s	v26, v0, #24
	sri.4s	v26, v0, #8
	add.4s	v5, v19, v27
	eor.16b	v4, v5, v25
	shl.4s	v8, v4, #24
	sri.4s	v8, v4, #8
	ldr	q0, [sp, #816]                  ; 16-byte Folded Reload
	ldr	q4, [sp, #736]                  ; 16-byte Folded Reload
	add.4s	v4, v0, v4
	ldp	q19, q0, [sp, #688]             ; 32-byte Folded Reload
	add.4s	v25, v0, v19
	add.4s	v19, v18, v22
	eor.16b	v1, v19, v1
	shl.4s	v18, v1, #24
	sri.4s	v18, v1, #8
	add.4s	v1, v7, v4
	eor.16b	v4, v1, v29
	rev32.8h	v4, v4
	ldr	q0, [sp, #672]                  ; 16-byte Folded Reload
	add.4s	v29, v0, v4
	eor.16b	v7, v29, v7
	shl.4s	v9, v7, #20
	sri.4s	v9, v7, #12
	add.4s	v7, v3, v25
	eor.16b	v25, v7, v12
	rev32.8h	v25, v25
	ldr	q0, [sp, #640]                  ; 16-byte Folded Reload
	add.4s	v12, v0, v25
	eor.16b	v3, v12, v3
	shl.4s	v0, v3, #20
	sri.4s	v0, v3, #12
	add.4s	v3, v31, v16
	eor.16b	v16, v3, v11
	shl.4s	v11, v16, #25
	sri.4s	v11, v16, #7
	ldr	q16, [sp, #800]                 ; 16-byte Folded Reload
	add.4s	v1, v1, v16
	ldr	q16, [sp, #784]                 ; 16-byte Folded Reload
	add.4s	v7, v7, v16
	add.4s	v10, v26, v10
	eor.16b	v16, v10, v15
	shl.4s	v15, v16, #25
	sri.4s	v15, v16, #7
	add.4s	v1, v1, v9
	eor.16b	v4, v1, v4
	shl.4s	v16, v4, #24
	sri.4s	v16, v4, #8
	add.4s	v4, v7, v0
	eor.16b	v7, v4, v25
	shl.4s	v25, v7, #24
	sri.4s	v25, v7, #8
	add.4s	v24, v8, v24
	eor.16b	v7, v24, v27
	shl.4s	v27, v7, #25
	sri.4s	v27, v7, #7
	add.4s	v29, v16, v29
	eor.16b	v7, v29, v9
	shl.4s	v9, v7, #25
	sri.4s	v9, v7, #7
	add.4s	v12, v25, v12
	eor.16b	v0, v12, v0
	shl.4s	v7, v0, #25
	sri.4s	v7, v0, #7
	add.4s	v0, v18, v21
	eor.16b	v21, v0, v22
	shl.4s	v22, v21, #25
	sri.4s	v22, v21, #7
	ldr	q21, [sp, #768]                 ; 16-byte Folded Reload
	eor.16b	v3, v3, v21
	str	q3, [sp, #288]                  ; 16-byte Folded Spill
	eor.16b	v3, v29, v5
	str	q3, [sp, #304]                  ; 16-byte Folded Spill
	eor.16b	v5, v30, v6
	eor.16b	v3, v1, v24
	eor.16b	v31, v31, v13
	eor.16b	v6, v9, v8
	ldp	q30, q8, [sp, #64]              ; 32-byte Folded Reload
	eor.16b	v24, v11, v20
	eor.16b	v14, v10, v14
	eor.16b	v9, v12, v19
	eor.16b	v29, v28, v17
	eor.16b	v20, v4, v0
	eor.16b	v28, v16, v27
	eor.16b	v13, v7, v18
	eor.16b	v0, v26, v2
	eor.16b	v21, v25, v22
	ldr	q22, [sp, #96]                  ; 16-byte Folded Reload
	eor.16b	v19, v15, v23
	add	x9, x9, #64
	cbnz	x11, LBB0_3
; %bb.4:                                ;   in Loop: Header=BB0_2 Depth=1
	ldp	q17, q16, [sp, #288]            ; 32-byte Folded Reload
	trn1.4s	v1, v17, v16
	trn1.4s	v2, v5, v3
	trn2.4s	v7, v5, v3
	zip2.2d	v3, v1, v2
	zip1.2d	v5, v1, v2
	trn2.4s	v2, v17, v16
	zip2.2d	v1, v2, v7
	mov.d	v2[1], v7[0]
	trn1.4s	v7, v6, v31
	trn2.4s	v6, v6, v31
	trn1.4s	v16, v28, v24
	zip2.2d	v17, v7, v16
	mov.d	v7[1], v16[0]
	trn2.4s	v16, v28, v24
	zip2.2d	v18, v6, v16
	mov.d	v6[1], v16[0]
	stp	q5, q7, [x1], #256
	stp	q2, q6, [x1, #-224]
	stp	q3, q17, [x1, #-192]
	trn1.4s	v2, v14, v9
	trn2.4s	v3, v14, v9
	trn1.4s	v4, v29, v20
	zip2.2d	v5, v2, v4
	mov.d	v2[1], v4[0]
	trn2.4s	v4, v29, v20
	zip2.2d	v6, v3, v4
	mov.d	v3[1], v4[0]
	trn1.4s	v4, v13, v0
	trn2.4s	v7, v13, v0
	trn1.4s	v16, v21, v19
	zip2.2d	v17, v4, v16
	mov.d	v4[1], v16[0]
	trn2.4s	v0, v21, v19
	zip2.2d	v16, v7, v0
	mov.d	v7[1], v0[0]
	stp	q1, q18, [x1, #-160]
	stp	q2, q4, [x1, #-128]
	stp	q3, q7, [x1, #-96]
	stp	q5, q17, [x1, #-64]
	add	x0, x0, #2, lsl #12             ; =8192
	stp	q6, q16, [x1, #-32]
	subs	x2, x2, #1
	ldr	q2, [sp, #112]                  ; 16-byte Folded Reload
	b.ne	LBB0_2
; %bb.5:
	add	sp, sp, #832
	ldp	d9, d8, [sp, #48]               ; 16-byte Folded Reload
	ldp	d11, d10, [sp, #32]             ; 16-byte Folded Reload
	ldp	d13, d12, [sp, #16]             ; 16-byte Folded Reload
	ldp	d15, d14, [sp], #64             ; 16-byte Folded Reload
LBB0_6:
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols
