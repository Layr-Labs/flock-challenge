// SPDX-License-Identifier: Apache-2.0 OR MIT
// Exact composed GFNI fold, lane-major output, with a private register ABI.
// Inputs: RDI = 512 input bytes, RSI = CFoldMats.1. Returns ZMM8..11 in lane-major order.
// Clobbers: RAX, flags, ZMM0..11. Preserves every other GPR and ZMM12..31.
// Called ONLY by the matching inline-asm wrapper, never through the C ABI.
// No vzeroupper: the upper lanes of ZMM12..15 belong to the caller.
.text
.p2align 6
.globl flock_gfni_c4_lm_register_result
.hidden flock_gfni_c4_lm_register_result
.type flock_gfni_c4_lm_register_result,@function
flock_gfni_c4_lm_register_result:
    mov rax, rsp
    and rsp, -64
    sub rsp, 512
    vmovdqa64 zmm8, [rip + .Lflock_c4_result_oct_even]
    vmovdqa64 zmm9, [rip + .Lflock_c4_result_oct_odd]

// Gather both passes into the original initialized scratch before reducing.
.macro flock_c4_result_gather p
    vmovdqu64 zmm0, [rdi + 64*\p]
    vmovdqu64 zmm1, [rdi + 64*\p + 128]
    vmovdqu64 zmm2, [rdi + 64*\p + 256]
    vmovdqu64 zmm3, [rdi + 64*\p + 384]
    vshufi64x2 zmm4, zmm0, zmm1, 0x88
    vshufi64x2 zmm0, zmm0, zmm1, 0xdd
    vshufi64x2 zmm5, zmm2, zmm3, 0x88
    vshufi64x2 zmm2, zmm2, zmm3, 0xdd
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
    flock_c4_result_gather 0
    flock_c4_result_gather 1

    vpbroadcastq zmm0, [rsp + 0]
    vgf2p8affineqb zmm1, zmm0, [rsi + 2048], 0
    vgf2p8affineqb zmm0, zmm0, [rsi + 0], 0
    vpbroadcastq zmm2, [rsp + 8]
    vgf2p8affineqb zmm3, zmm2, [rsi + 2112], 0
    vgf2p8affineqb zmm2, zmm2, [rsi + 64], 0
    vpbroadcastq zmm4, [rsp + 16]
    vgf2p8affineqb zmm5, zmm4, [rsi + 2176], 0
    vgf2p8affineqb zmm4, zmm4, [rsi + 128], 0
    vpternlogq zmm0, zmm2, zmm4, 0x96
    vpternlogq zmm1, zmm3, zmm5, 0x96
    vpbroadcastq zmm2, [rsp + 24]
    vgf2p8affineqb zmm3, zmm2, [rsi + 2240], 0
    vgf2p8affineqb zmm2, zmm2, [rsi + 192], 0
    vpbroadcastq zmm4, [rsp + 32]
    vgf2p8affineqb zmm5, zmm4, [rsi + 2304], 0
    vgf2p8affineqb zmm4, zmm4, [rsi + 256], 0
    vpternlogq zmm0, zmm2, zmm4, 0x96
    vpternlogq zmm1, zmm3, zmm5, 0x96
    vpbroadcastq zmm2, [rsp + 40]
    vgf2p8affineqb zmm3, zmm2, [rsi + 2368], 0
    vgf2p8affineqb zmm2, zmm2, [rsi + 320], 0
    vpbroadcastq zmm4, [rsp + 48]
    vgf2p8affineqb zmm5, zmm4, [rsi + 2432], 0
    vgf2p8affineqb zmm4, zmm4, [rsi + 384], 0
    vpternlogq zmm0, zmm2, zmm4, 0x96
    vpternlogq zmm1, zmm3, zmm5, 0x96
    vpbroadcastq zmm2, [rsp + 56]
    vgf2p8affineqb zmm3, zmm2, [rsi + 2496], 0
    vgf2p8affineqb zmm2, zmm2, [rsi + 448], 0
    vpbroadcastq zmm4, [rsp + 64]
    vgf2p8affineqb zmm5, zmm4, [rsi + 2560], 0
    vgf2p8affineqb zmm4, zmm4, [rsi + 512], 0
    vpternlogq zmm0, zmm2, zmm4, 0x96
    vpternlogq zmm1, zmm3, zmm5, 0x96
    vpbroadcastq zmm2, [rsp + 72]
    vgf2p8affineqb zmm3, zmm2, [rsi + 2624], 0
    vgf2p8affineqb zmm2, zmm2, [rsi + 576], 0
    vpbroadcastq zmm4, [rsp + 80]
    vgf2p8affineqb zmm5, zmm4, [rsi + 2688], 0
    vgf2p8affineqb zmm4, zmm4, [rsi + 640], 0
    vpternlogq zmm0, zmm2, zmm4, 0x96
    vpternlogq zmm1, zmm3, zmm5, 0x96
    vpbroadcastq zmm2, [rsp + 88]
    vgf2p8affineqb zmm3, zmm2, [rsi + 2752], 0
    vgf2p8affineqb zmm2, zmm2, [rsi + 704], 0
    vpbroadcastq zmm4, [rsp + 96]
    vgf2p8affineqb zmm5, zmm4, [rsi + 2816], 0
    vgf2p8affineqb zmm4, zmm4, [rsi + 768], 0
    vpternlogq zmm0, zmm2, zmm4, 0x96
    vpternlogq zmm1, zmm3, zmm5, 0x96
    vpbroadcastq zmm2, [rsp + 104]
    vgf2p8affineqb zmm3, zmm2, [rsi + 2880], 0
    vgf2p8affineqb zmm2, zmm2, [rsi + 832], 0
    vpbroadcastq zmm4, [rsp + 112]
    vgf2p8affineqb zmm5, zmm4, [rsi + 2944], 0
    vgf2p8affineqb zmm4, zmm4, [rsi + 896], 0
    vpternlogq zmm0, zmm2, zmm4, 0x96
    vpternlogq zmm1, zmm3, zmm5, 0x96
    vpbroadcastq zmm2, [rsp + 120]
    vgf2p8affineqb zmm3, zmm2, [rsi + 3008], 0
    vgf2p8affineqb zmm2, zmm2, [rsi + 960], 0
    vpbroadcastq zmm4, [rsp + 128]
    vgf2p8affineqb zmm5, zmm4, [rsi + 3072], 0
    vgf2p8affineqb zmm4, zmm4, [rsi + 1024], 0
    vpternlogq zmm0, zmm2, zmm4, 0x96
    vpternlogq zmm1, zmm3, zmm5, 0x96
    vpbroadcastq zmm2, [rsp + 136]
    vgf2p8affineqb zmm3, zmm2, [rsi + 3136], 0
    vgf2p8affineqb zmm2, zmm2, [rsi + 1088], 0
    vpbroadcastq zmm4, [rsp + 144]
    vgf2p8affineqb zmm5, zmm4, [rsi + 3200], 0
    vgf2p8affineqb zmm4, zmm4, [rsi + 1152], 0
    vpternlogq zmm0, zmm2, zmm4, 0x96
    vpternlogq zmm1, zmm3, zmm5, 0x96
    vpbroadcastq zmm2, [rsp + 152]
    vgf2p8affineqb zmm3, zmm2, [rsi + 3264], 0
    vgf2p8affineqb zmm2, zmm2, [rsi + 1216], 0
    vpbroadcastq zmm4, [rsp + 160]
    vgf2p8affineqb zmm5, zmm4, [rsi + 3328], 0
    vgf2p8affineqb zmm4, zmm4, [rsi + 1280], 0
    vpternlogq zmm0, zmm2, zmm4, 0x96
    vpternlogq zmm1, zmm3, zmm5, 0x96
    vpbroadcastq zmm2, [rsp + 168]
    vgf2p8affineqb zmm3, zmm2, [rsi + 3392], 0
    vgf2p8affineqb zmm2, zmm2, [rsi + 1344], 0
    vpbroadcastq zmm4, [rsp + 176]
    vgf2p8affineqb zmm5, zmm4, [rsi + 3456], 0
    vgf2p8affineqb zmm4, zmm4, [rsi + 1408], 0
    vpternlogq zmm0, zmm2, zmm4, 0x96
    vpternlogq zmm1, zmm3, zmm5, 0x96
    vpbroadcastq zmm2, [rsp + 184]
    vgf2p8affineqb zmm3, zmm2, [rsi + 3520], 0
    vgf2p8affineqb zmm2, zmm2, [rsi + 1472], 0
    vpbroadcastq zmm4, [rsp + 192]
    vgf2p8affineqb zmm5, zmm4, [rsi + 3584], 0
    vgf2p8affineqb zmm4, zmm4, [rsi + 1536], 0
    vpternlogq zmm0, zmm2, zmm4, 0x96
    vpternlogq zmm1, zmm3, zmm5, 0x96
    vpbroadcastq zmm2, [rsp + 200]
    vgf2p8affineqb zmm3, zmm2, [rsi + 3648], 0
    vgf2p8affineqb zmm2, zmm2, [rsi + 1600], 0
    vpbroadcastq zmm4, [rsp + 208]
    vgf2p8affineqb zmm5, zmm4, [rsi + 3712], 0
    vgf2p8affineqb zmm4, zmm4, [rsi + 1664], 0
    vpternlogq zmm0, zmm2, zmm4, 0x96
    vpternlogq zmm1, zmm3, zmm5, 0x96
    vpbroadcastq zmm2, [rsp + 216]
    vgf2p8affineqb zmm3, zmm2, [rsi + 3776], 0
    vgf2p8affineqb zmm2, zmm2, [rsi + 1728], 0
    vpbroadcastq zmm4, [rsp + 224]
    vgf2p8affineqb zmm5, zmm4, [rsi + 3840], 0
    vgf2p8affineqb zmm4, zmm4, [rsi + 1792], 0
    vpternlogq zmm0, zmm2, zmm4, 0x96
    vpternlogq zmm1, zmm3, zmm5, 0x96
    vpbroadcastq zmm2, [rsp + 232]
    vgf2p8affineqb zmm3, zmm2, [rsi + 3904], 0
    vgf2p8affineqb zmm2, zmm2, [rsi + 1856], 0
    vpbroadcastq zmm4, [rsp + 240]
    vgf2p8affineqb zmm5, zmm4, [rsi + 3968], 0
    vgf2p8affineqb zmm4, zmm4, [rsi + 1920], 0
    vpternlogq zmm0, zmm2, zmm4, 0x96
    vpternlogq zmm1, zmm3, zmm5, 0x96
    vpbroadcastq zmm2, [rsp + 248]
    vgf2p8affineqb zmm3, zmm2, [rsi + 4032], 0
    vgf2p8affineqb zmm2, zmm2, [rsi + 1984], 0
    vpxorq zmm0, zmm0, zmm2
    vpxorq zmm1, zmm1, zmm3
    vmovdqa64 zmm6, [rip + .Lflock_c4_result_out_even]
    vmovdqa64 zmm7, [rip + .Lflock_c4_result_out_odd]
    vmovdqa64 zmm8, zmm0
    vmovdqa64 zmm9, zmm0
    vpermt2b zmm8, zmm6, zmm1
    vpermt2b zmm9, zmm7, zmm1
    vpbroadcastq zmm0, [rsp + 256]
    vgf2p8affineqb zmm1, zmm0, [rsi + 2048], 0
    vgf2p8affineqb zmm0, zmm0, [rsi + 0], 0
    vpbroadcastq zmm2, [rsp + 264]
    vgf2p8affineqb zmm3, zmm2, [rsi + 2112], 0
    vgf2p8affineqb zmm2, zmm2, [rsi + 64], 0
    vpbroadcastq zmm4, [rsp + 272]
    vgf2p8affineqb zmm5, zmm4, [rsi + 2176], 0
    vgf2p8affineqb zmm4, zmm4, [rsi + 128], 0
    vpternlogq zmm0, zmm2, zmm4, 0x96
    vpternlogq zmm1, zmm3, zmm5, 0x96
    vpbroadcastq zmm2, [rsp + 280]
    vgf2p8affineqb zmm3, zmm2, [rsi + 2240], 0
    vgf2p8affineqb zmm2, zmm2, [rsi + 192], 0
    vpbroadcastq zmm4, [rsp + 288]
    vgf2p8affineqb zmm5, zmm4, [rsi + 2304], 0
    vgf2p8affineqb zmm4, zmm4, [rsi + 256], 0
    vpternlogq zmm0, zmm2, zmm4, 0x96
    vpternlogq zmm1, zmm3, zmm5, 0x96
    vpbroadcastq zmm2, [rsp + 296]
    vgf2p8affineqb zmm3, zmm2, [rsi + 2368], 0
    vgf2p8affineqb zmm2, zmm2, [rsi + 320], 0
    vpbroadcastq zmm4, [rsp + 304]
    vgf2p8affineqb zmm5, zmm4, [rsi + 2432], 0
    vgf2p8affineqb zmm4, zmm4, [rsi + 384], 0
    vpternlogq zmm0, zmm2, zmm4, 0x96
    vpternlogq zmm1, zmm3, zmm5, 0x96
    vpbroadcastq zmm2, [rsp + 312]
    vgf2p8affineqb zmm3, zmm2, [rsi + 2496], 0
    vgf2p8affineqb zmm2, zmm2, [rsi + 448], 0
    vpbroadcastq zmm4, [rsp + 320]
    vgf2p8affineqb zmm5, zmm4, [rsi + 2560], 0
    vgf2p8affineqb zmm4, zmm4, [rsi + 512], 0
    vpternlogq zmm0, zmm2, zmm4, 0x96
    vpternlogq zmm1, zmm3, zmm5, 0x96
    vpbroadcastq zmm2, [rsp + 328]
    vgf2p8affineqb zmm3, zmm2, [rsi + 2624], 0
    vgf2p8affineqb zmm2, zmm2, [rsi + 576], 0
    vpbroadcastq zmm4, [rsp + 336]
    vgf2p8affineqb zmm5, zmm4, [rsi + 2688], 0
    vgf2p8affineqb zmm4, zmm4, [rsi + 640], 0
    vpternlogq zmm0, zmm2, zmm4, 0x96
    vpternlogq zmm1, zmm3, zmm5, 0x96
    vpbroadcastq zmm2, [rsp + 344]
    vgf2p8affineqb zmm3, zmm2, [rsi + 2752], 0
    vgf2p8affineqb zmm2, zmm2, [rsi + 704], 0
    vpbroadcastq zmm4, [rsp + 352]
    vgf2p8affineqb zmm5, zmm4, [rsi + 2816], 0
    vgf2p8affineqb zmm4, zmm4, [rsi + 768], 0
    vpternlogq zmm0, zmm2, zmm4, 0x96
    vpternlogq zmm1, zmm3, zmm5, 0x96
    vpbroadcastq zmm2, [rsp + 360]
    vgf2p8affineqb zmm3, zmm2, [rsi + 2880], 0
    vgf2p8affineqb zmm2, zmm2, [rsi + 832], 0
    vpbroadcastq zmm4, [rsp + 368]
    vgf2p8affineqb zmm5, zmm4, [rsi + 2944], 0
    vgf2p8affineqb zmm4, zmm4, [rsi + 896], 0
    vpternlogq zmm0, zmm2, zmm4, 0x96
    vpternlogq zmm1, zmm3, zmm5, 0x96
    vpbroadcastq zmm2, [rsp + 376]
    vgf2p8affineqb zmm3, zmm2, [rsi + 3008], 0
    vgf2p8affineqb zmm2, zmm2, [rsi + 960], 0
    vpbroadcastq zmm4, [rsp + 384]
    vgf2p8affineqb zmm5, zmm4, [rsi + 3072], 0
    vgf2p8affineqb zmm4, zmm4, [rsi + 1024], 0
    vpternlogq zmm0, zmm2, zmm4, 0x96
    vpternlogq zmm1, zmm3, zmm5, 0x96
    vpbroadcastq zmm2, [rsp + 392]
    vgf2p8affineqb zmm3, zmm2, [rsi + 3136], 0
    vgf2p8affineqb zmm2, zmm2, [rsi + 1088], 0
    vpbroadcastq zmm4, [rsp + 400]
    vgf2p8affineqb zmm5, zmm4, [rsi + 3200], 0
    vgf2p8affineqb zmm4, zmm4, [rsi + 1152], 0
    vpternlogq zmm0, zmm2, zmm4, 0x96
    vpternlogq zmm1, zmm3, zmm5, 0x96
    vpbroadcastq zmm2, [rsp + 408]
    vgf2p8affineqb zmm3, zmm2, [rsi + 3264], 0
    vgf2p8affineqb zmm2, zmm2, [rsi + 1216], 0
    vpbroadcastq zmm4, [rsp + 416]
    vgf2p8affineqb zmm5, zmm4, [rsi + 3328], 0
    vgf2p8affineqb zmm4, zmm4, [rsi + 1280], 0
    vpternlogq zmm0, zmm2, zmm4, 0x96
    vpternlogq zmm1, zmm3, zmm5, 0x96
    vpbroadcastq zmm2, [rsp + 424]
    vgf2p8affineqb zmm3, zmm2, [rsi + 3392], 0
    vgf2p8affineqb zmm2, zmm2, [rsi + 1344], 0
    vpbroadcastq zmm4, [rsp + 432]
    vgf2p8affineqb zmm5, zmm4, [rsi + 3456], 0
    vgf2p8affineqb zmm4, zmm4, [rsi + 1408], 0
    vpternlogq zmm0, zmm2, zmm4, 0x96
    vpternlogq zmm1, zmm3, zmm5, 0x96
    vpbroadcastq zmm2, [rsp + 440]
    vgf2p8affineqb zmm3, zmm2, [rsi + 3520], 0
    vgf2p8affineqb zmm2, zmm2, [rsi + 1472], 0
    vpbroadcastq zmm4, [rsp + 448]
    vgf2p8affineqb zmm5, zmm4, [rsi + 3584], 0
    vgf2p8affineqb zmm4, zmm4, [rsi + 1536], 0
    vpternlogq zmm0, zmm2, zmm4, 0x96
    vpternlogq zmm1, zmm3, zmm5, 0x96
    vpbroadcastq zmm2, [rsp + 456]
    vgf2p8affineqb zmm3, zmm2, [rsi + 3648], 0
    vgf2p8affineqb zmm2, zmm2, [rsi + 1600], 0
    vpbroadcastq zmm4, [rsp + 464]
    vgf2p8affineqb zmm5, zmm4, [rsi + 3712], 0
    vgf2p8affineqb zmm4, zmm4, [rsi + 1664], 0
    vpternlogq zmm0, zmm2, zmm4, 0x96
    vpternlogq zmm1, zmm3, zmm5, 0x96
    vpbroadcastq zmm2, [rsp + 472]
    vgf2p8affineqb zmm3, zmm2, [rsi + 3776], 0
    vgf2p8affineqb zmm2, zmm2, [rsi + 1728], 0
    vpbroadcastq zmm4, [rsp + 480]
    vgf2p8affineqb zmm5, zmm4, [rsi + 3840], 0
    vgf2p8affineqb zmm4, zmm4, [rsi + 1792], 0
    vpternlogq zmm0, zmm2, zmm4, 0x96
    vpternlogq zmm1, zmm3, zmm5, 0x96
    vpbroadcastq zmm2, [rsp + 488]
    vgf2p8affineqb zmm3, zmm2, [rsi + 3904], 0
    vgf2p8affineqb zmm2, zmm2, [rsi + 1856], 0
    vpbroadcastq zmm4, [rsp + 496]
    vgf2p8affineqb zmm5, zmm4, [rsi + 3968], 0
    vgf2p8affineqb zmm4, zmm4, [rsi + 1920], 0
    vpternlogq zmm0, zmm2, zmm4, 0x96
    vpternlogq zmm1, zmm3, zmm5, 0x96
    vpbroadcastq zmm2, [rsp + 504]
    vgf2p8affineqb zmm3, zmm2, [rsi + 4032], 0
    vgf2p8affineqb zmm2, zmm2, [rsi + 1984], 0
    vpxorq zmm0, zmm0, zmm2
    vpxorq zmm1, zmm1, zmm3
    vmovdqa64 zmm6, [rip + .Lflock_c4_result_out_even]
    vmovdqa64 zmm7, [rip + .Lflock_c4_result_out_odd]
    vmovdqa64 zmm10, zmm0
    vmovdqa64 zmm11, zmm0
    vpermt2b zmm10, zmm6, zmm1
    vpermt2b zmm11, zmm7, zmm1
    mov rsp, rax
    ret
.size flock_gfni_c4_lm_register_result, .-flock_gfni_c4_lm_register_result
.purgem flock_c4_result_gather
.section .rodata

.p2align 6
.Lflock_c4_result_oct_even:
    .byte 0, 16, 32, 48, 64, 80, 96, 112, 1, 17, 33, 49, 65, 81, 97, 113, 2, 18, 34, 50, 66, 82, 98, 114, 3, 19, 35, 51, 67, 83, 99, 115, 4, 20, 36, 52, 68, 84, 100, 116, 5, 21, 37, 53, 69, 85, 101, 117, 6, 22, 38, 54, 70, 86, 102, 118, 7, 23, 39, 55, 71, 87, 103, 119

.p2align 6
.Lflock_c4_result_oct_odd:
    .byte 8, 24, 40, 56, 72, 88, 104, 120, 9, 25, 41, 57, 73, 89, 105, 121, 10, 26, 42, 58, 74, 90, 106, 122, 11, 27, 43, 59, 75, 91, 107, 123, 12, 28, 44, 60, 76, 92, 108, 124, 13, 29, 45, 61, 77, 93, 109, 125, 14, 30, 46, 62, 78, 94, 110, 126, 15, 31, 47, 63, 79, 95, 111, 127

.p2align 6
.Lflock_c4_result_out_even:
    .byte 0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 2, 10, 18, 26, 34, 42, 50, 58, 66, 74, 82, 90, 98, 106, 114, 122, 4, 12, 20, 28, 36, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 124, 6, 14, 22, 30, 38, 46, 54, 62, 70, 78, 86, 94, 102, 110, 118, 126

.p2align 6
.Lflock_c4_result_out_odd:
    .byte 1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121, 3, 11, 19, 27, 35, 43, 51, 59, 67, 75, 83, 91, 99, 107, 115, 123, 5, 13, 21, 29, 37, 45, 53, 61, 69, 77, 85, 93, 101, 109, 117, 125, 7, 15, 23, 31, 39, 47, 55, 63, 71, 79, 87, 95, 103, 111, 119, 127
