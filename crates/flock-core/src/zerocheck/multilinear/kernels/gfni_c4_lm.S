// SPDX-License-Identifier: Apache-2.0 OR MIT
// Exact composed GFNI fold, lane-major output, with a private register ABI.
// Inputs: RDI = 512 input bytes, RSI = CFoldMats.1, RDX = 256 output bytes.
// Clobbers: RAX, flags, ZMM0..11. Preserves every other GPR and ZMM12..31.
// Called ONLY by the matching inline-asm wrapper, never through the C ABI.
// No vzeroupper: the upper lanes of ZMM12..15 belong to the caller.
.text
.p2align 6
.globl flock_gfni_c4_lm_preserving
.hidden flock_gfni_c4_lm_preserving
.type flock_gfni_c4_lm_preserving,@function
flock_gfni_c4_lm_preserving:
    mov rax, rsp
    and rsp, -64
    sub rsp, 512
    vmovdqa64 zmm10, [rip + .Lflock_c4_g1_lo]
    vmovdqa64 zmm11, [rip + .Lflock_c4_g1_hi]
    vmovdqa64 zmm8, [rip + .Lflock_c4_oct_even]
    vmovdqa64 zmm9, [rip + .Lflock_c4_oct_odd]

// Materialize both passes before publishing anything, just as in Rust.
.macro flock_c4_gather p
    vmovdqu64 zmm0, [rdi + 64*\p]
    vmovdqu64 zmm1, [rdi + 64*\p + 128]
    vmovdqu64 zmm2, [rdi + 64*\p + 256]
    vmovdqu64 zmm3, [rdi + 64*\p + 384]
    vmovdqa64 zmm4, zmm0
    vpermt2q zmm4, zmm10, zmm1
    vpermt2q zmm0, zmm11, zmm1
    vmovdqa64 zmm5, zmm2
    vpermt2q zmm5, zmm10, zmm3
    vpermt2q zmm2, zmm11, zmm3
    vmovdqa64 zmm6, zmm4
    vpermt2b zmm6, zmm8, zmm5
    vpermt2b zmm4, zmm9, zmm5
    vmovdqa64 [rsp + 256*\p], zmm6
    vmovdqa64 [rsp + 256*\p + 64], zmm4
    vmovdqa64 zmm6, zmm0
    vpermt2b zmm6, zmm8, zmm2
    vpermt2b zmm0, zmm9, zmm2
    vmovdqa64 [rsp + 256*\p + 128], zmm6
    vmovdqa64 [rsp + 256*\p + 192], zmm0
.endm
    flock_c4_gather 0
    flock_c4_gather 1

// One octet contributes to both halves of the output field element. Keeping
// the matrices as memory operands bounds the entire body to twelve ZMMs.
.macro flock_c4_pair lo, hi, j, a, p
    vpbroadcastq zmm\lo, [rsp + 256*\p + 64*\a + 8*\j]
    vgf2p8affineqb zmm\hi, zmm\lo, [rsi + 2048 + 512*\a + 64*\j], 0
    vgf2p8affineqb zmm\lo, zmm\lo, [rsi + 512*\a + 64*\j], 0
.endm
.macro flock_c4_residue a, p
    flock_c4_pair 0, 4, 0, \a, \p
    flock_c4_pair 1, 5, 1, \a, \p
    flock_c4_pair 2, 6, 2, \a, \p
    vpternlogq zmm0, zmm1, zmm2, 0x96
    vpternlogq zmm4, zmm5, zmm6, 0x96
    flock_c4_pair 1, 5, 3, \a, \p
    flock_c4_pair 2, 6, 4, \a, \p
    flock_c4_pair 3, 7, 5, \a, \p
    vpternlogq zmm1, zmm2, zmm3, 0x96
    vpternlogq zmm5, zmm6, zmm7, 0x96
    flock_c4_pair 2, 6, 6, \a, \p
    flock_c4_pair 3, 7, 7, \a, \p
    vpternlogq zmm2, zmm3, zmm0, 0x96
    vpternlogq zmm6, zmm7, zmm4, 0x96
.if \a == 0
    vmovdqa64 zmm8, zmm1
    vmovdqa64 zmm9, zmm2
    vmovdqa64 zmm10, zmm5
    vmovdqa64 zmm11, zmm6
.else
    vpternlogq zmm8, zmm9, zmm1, 0x96
    vmovdqa64 zmm9, zmm2
    vpternlogq zmm10, zmm11, zmm5, 0x96
    vmovdqa64 zmm11, zmm6
.endif
.endm
.macro flock_c4_pass p
    flock_c4_residue 0, \p
    flock_c4_residue 1, \p
    flock_c4_residue 2, \p
    flock_c4_residue 3, \p
    vpxorq zmm8, zmm8, zmm9
    vpxorq zmm10, zmm10, zmm11
    // The XOR halves are dead now: reuse them for the output indices.
    vmovdqa64 zmm9, [rip + .Lflock_c4_out_even]
    vmovdqa64 zmm11, [rip + .Lflock_c4_out_odd]
    vmovdqa64 zmm0, zmm8
    vpermt2b zmm0, zmm9, zmm10
    vpermt2b zmm8, zmm11, zmm10
    vmovdqu64 [rdx + 128*\p], zmm0
    vmovdqu64 [rdx + 128*\p + 64], zmm8
.endm
    flock_c4_pass 0
    flock_c4_pass 1
    mov rsp, rax
    ret
.size flock_gfni_c4_lm_preserving, .-flock_gfni_c4_lm_preserving
.purgem flock_c4_gather
.purgem flock_c4_pair
.purgem flock_c4_residue
.purgem flock_c4_pass

.section .rodata

.p2align 6
.Lflock_c4_g1_lo:
    .quad 0, 4, 8, 12, 1, 5, 9, 13

.p2align 6
.Lflock_c4_g1_hi:
    .quad 2, 6, 10, 14, 3, 7, 11, 15

.p2align 6
.Lflock_c4_oct_even:
    .byte 0, 8, 16, 24, 64, 72, 80, 88, 1, 9, 17, 25, 65, 73, 81, 89, 2, 10, 18, 26, 66, 74, 82, 90, 3, 11, 19, 27, 67, 75, 83, 91, 4, 12, 20, 28, 68, 76, 84, 92, 5, 13, 21, 29, 69, 77, 85, 93, 6, 14, 22, 30, 70, 78, 86, 94, 7, 15, 23, 31, 71, 79, 87, 95

.p2align 6
.Lflock_c4_oct_odd:
    .byte 32, 40, 48, 56, 96, 104, 112, 120, 33, 41, 49, 57, 97, 105, 113, 121, 34, 42, 50, 58, 98, 106, 114, 122, 35, 43, 51, 59, 99, 107, 115, 123, 36, 44, 52, 60, 100, 108, 116, 124, 37, 45, 53, 61, 101, 109, 117, 125, 38, 46, 54, 62, 102, 110, 118, 126, 39, 47, 55, 63, 103, 111, 119, 127

.p2align 6
.Lflock_c4_out_even:
    .byte 0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 2, 10, 18, 26, 34, 42, 50, 58, 66, 74, 82, 90, 98, 106, 114, 122, 4, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 124, 6, 14, 22, 30, 38, 46, 54, 62, 70, 78, 86, 94, 102, 110, 118, 126

.p2align 6
.Lflock_c4_out_odd:
    .byte 1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121, 3, 11, 19, 27, 35, 43, 51, 59, 67, 75, 83, 91, 99, 107, 115, 123, 5, 13, 21, 29, 37, 45, 53, 61, 69, 77, 85, 93, 101, 109, 117, 125, 7, 15, 23, 31, 39, 47, 55, 63, 71, 79, 87, 95, 103, 111, 119, 127
