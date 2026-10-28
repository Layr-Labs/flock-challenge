// Derived from BLAKE3 1.8.6 c/blake3_avx512_x86-64_unix.S, under CC0-1.0.
// Changed: contiguous fixed strides, no input pointer array, zero counter, full 16-input groups only.
.text
// R10 is call-clobbered and unused throughout the incumbent leaf body.
.p2align 6
.global flock_blake3_planes_leaf_64
.hidden flock_blake3_planes_leaf_64
.type flock_blake3_planes_leaf_64, @function
flock_blake3_planes_leaf_64:
        mov     r10d, 1
        jmp     .Lflock_leaf_shared_64
.p2align 6
.global flock_blake3_contiguous_64
.hidden flock_blake3_contiguous_64
.type flock_blake3_contiguous_64, @function
flock_blake3_contiguous_64:
        xor     r10d, r10d
.Lflock_leaf_shared_64:
        push    r15
        push    r14
        push    r13
        push    r12
        push    rbx
        push    rbp
        mov     rbp, rsp
        sub     rsp, 144
        and     rsp, 0xFFFFFFFFFFFFFFC0
2:
        vpbroadcastd zmm0, dword ptr [rcx]
        vpbroadcastd zmm1, dword ptr [rcx+0x1*0x4]
        vpbroadcastd zmm2, dword ptr [rcx+0x2*0x4]
        vpbroadcastd zmm3, dword ptr [rcx+0x3*0x4]
        vpbroadcastd zmm4, dword ptr [rcx+0x4*0x4]
        vpbroadcastd zmm5, dword ptr [rcx+0x5*0x4]
        vpbroadcastd zmm6, dword ptr [rcx+0x6*0x4]
        vpbroadcastd zmm7, dword ptr [rcx+0x7*0x4]
        movzx   eax, byte ptr [rbp+0x38]
        movzx   ebx, byte ptr [rbp+0x40]
        or      eax, ebx
        xor     edx, edx
.p2align 5
9:
        movzx   ebx, byte ptr [rbp+0x48]
        or      ebx, eax
        add     rdx, 64
        cmp     rdx, 64
        cmove   eax, ebx
        mov     dword ptr [rsp+0x88], eax
        vmovdqu32 ymm16, ymmword ptr [rdi+rdx+-64]
        vinserti64x4 zmm16, zmm16, ymmword ptr [rdi+rdx+448], 0x01
        vmovdqu32 ymm17, ymmword ptr [rdi+rdx+0]
        vinserti64x4 zmm17, zmm17, ymmword ptr [rdi+rdx+512], 0x01
        vpunpcklqdq zmm8, zmm16, zmm17
        vpunpckhqdq zmm9, zmm16, zmm17
        vmovdqu32 ymm18, ymmword ptr [rdi+rdx+64]
        vinserti64x4 zmm18, zmm18, ymmword ptr [rdi+rdx+576], 0x01
        vmovdqu32 ymm19, ymmword ptr [rdi+rdx+128]
        vinserti64x4 zmm19, zmm19, ymmword ptr [rdi+rdx+640], 0x01
        vpunpcklqdq zmm10, zmm18, zmm19
        vpunpckhqdq zmm11, zmm18, zmm19
        vmovdqu32 ymm16, ymmword ptr [rdi+rdx+192]
        vinserti64x4 zmm16, zmm16, ymmword ptr [rdi+rdx+704], 0x01
        vmovdqu32 ymm17, ymmword ptr [rdi+rdx+256]
        vinserti64x4 zmm17, zmm17, ymmword ptr [rdi+rdx+768], 0x01
        vpunpcklqdq zmm12, zmm16, zmm17
        vpunpckhqdq zmm13, zmm16, zmm17
        vmovdqu32 ymm18, ymmword ptr [rdi+rdx+320]
        vinserti64x4 zmm18, zmm18, ymmword ptr [rdi+rdx+832], 0x01
        vmovdqu32 ymm19, ymmword ptr [rdi+rdx+384]
        vinserti64x4 zmm19, zmm19, ymmword ptr [rdi+rdx+896], 0x01
        vpunpcklqdq zmm14, zmm18, zmm19
        vpunpckhqdq zmm15, zmm18, zmm19
        vmovdqa32 zmm27, zmmword ptr [.Lflock_contiguous_INDEX0+rip]
        vmovdqa32 zmm31, zmmword ptr [.Lflock_contiguous_INDEX1+rip]
        vshufps zmm16, zmm8, zmm10, 136
        vshufps zmm17, zmm12, zmm14, 136
        vmovdqa32 zmm20, zmm16
        vpermt2d zmm16, zmm27, zmm17
        vpermt2d zmm20, zmm31, zmm17
        vshufps zmm17, zmm8, zmm10, 221
        vshufps zmm30, zmm12, zmm14, 221
        vmovdqa32 zmm21, zmm17
        vpermt2d zmm17, zmm27, zmm30
        vpermt2d zmm21, zmm31, zmm30
        vshufps zmm18, zmm9, zmm11, 136
        vshufps zmm8, zmm13, zmm15, 136
        vmovdqa32 zmm22, zmm18
        vpermt2d zmm18, zmm27, zmm8
        vpermt2d zmm22, zmm31, zmm8
        vshufps zmm19, zmm9, zmm11, 221
        vshufps zmm8, zmm13, zmm15, 221
        vmovdqa32 zmm23, zmm19
        vpermt2d zmm19, zmm27, zmm8
        vpermt2d zmm23, zmm31, zmm8
        vmovdqu32 ymm24, ymmword ptr [rdi+rdx+-32]
        vinserti64x4 zmm24, zmm24, ymmword ptr [rdi+rdx+480], 0x01
        vmovdqu32 ymm25, ymmword ptr [rdi+rdx+32]
        vinserti64x4 zmm25, zmm25, ymmword ptr [rdi+rdx+544], 0x01
        vpunpcklqdq zmm8, zmm24, zmm25
        vpunpckhqdq zmm9, zmm24, zmm25
        vmovdqu32 ymm24, ymmword ptr [rdi+rdx+96]
        vinserti64x4 zmm24, zmm24, ymmword ptr [rdi+rdx+608], 0x01
        vmovdqu32 ymm25, ymmword ptr [rdi+rdx+160]
        vinserti64x4 zmm25, zmm25, ymmword ptr [rdi+rdx+672], 0x01
        vpunpcklqdq zmm10, zmm24, zmm25
        vpunpckhqdq zmm11, zmm24, zmm25
        prefetcht0 [rdi+rdx+128]
        prefetcht0 [rdi+rdx+640]
        prefetcht0 [rdi+rdx+192]
        prefetcht0 [rdi+rdx+704]
        prefetcht0 [rdi+rdx+256]
        prefetcht0 [rdi+rdx+768]
        prefetcht0 [rdi+rdx+320]
        prefetcht0 [rdi+rdx+832]
        vmovdqu32 ymm24, ymmword ptr [rdi+rdx+224]
        vinserti64x4 zmm24, zmm24, ymmword ptr [rdi+rdx+736], 0x01
        vmovdqu32 ymm25, ymmword ptr [rdi+rdx+288]
        vinserti64x4 zmm25, zmm25, ymmword ptr [rdi+rdx+800], 0x01
        vpunpcklqdq zmm12, zmm24, zmm25
        vpunpckhqdq zmm13, zmm24, zmm25
        vmovdqu32 ymm24, ymmword ptr [rdi+rdx+352]
        vinserti64x4 zmm24, zmm24, ymmword ptr [rdi+rdx+864], 0x01
        vmovdqu32 ymm25, ymmword ptr [rdi+rdx+416]
        vinserti64x4 zmm25, zmm25, ymmword ptr [rdi+rdx+928], 0x01
        vpunpcklqdq zmm14, zmm24, zmm25
        vpunpckhqdq zmm15, zmm24, zmm25
        prefetcht0 [rdi+rdx+384]
        prefetcht0 [rdi+rdx+896]
        prefetcht0 [rdi+rdx+448]
        prefetcht0 [rdi+rdx+960]
        prefetcht0 [rdi+rdx+512]
        prefetcht0 [rdi+rdx+1024]
        prefetcht0 [rdi+rdx+576]
        prefetcht0 [rdi+rdx+1088]
        vshufps zmm24, zmm8, zmm10, 136
        vshufps zmm30, zmm12, zmm14, 136
        vmovdqa32 zmm28, zmm24
        vpermt2d zmm24, zmm27, zmm30
        vpermt2d zmm28, zmm31, zmm30
        vshufps zmm25, zmm8, zmm10, 221
        vshufps zmm30, zmm12, zmm14, 221
        vmovdqa32 zmm29, zmm25
        vpermt2d zmm25, zmm27, zmm30
        vpermt2d zmm29, zmm31, zmm30
        vshufps zmm26, zmm9, zmm11, 136
        vshufps zmm8, zmm13, zmm15, 136
        vmovdqa32 zmm30, zmm26
        vpermt2d zmm26, zmm27, zmm8
        vpermt2d zmm30, zmm31, zmm8
        vshufps zmm8, zmm9, zmm11, 221
        vshufps zmm10, zmm13, zmm15, 221
        vpermi2d zmm27, zmm8, zmm10
        vpermi2d zmm31, zmm8, zmm10
        vpbroadcastd zmm8, dword ptr [.Lflock_contiguous_BLAKE3_IV_0+rip]
        vpbroadcastd zmm9, dword ptr [.Lflock_contiguous_BLAKE3_IV_1+rip]
        vpbroadcastd zmm10, dword ptr [.Lflock_contiguous_BLAKE3_IV_2+rip]
        vpbroadcastd zmm11, dword ptr [.Lflock_contiguous_BLAKE3_IV_3+rip]
        vpbroadcastd zmm14, dword ptr [.Lflock_contiguous_BLAKE3_BLOCK_LEN+rip]
        vpbroadcastd zmm15, dword ptr [rsp+0x22*0x4]
        vpaddd  zmm0, zmm0, zmm16
        vpaddd  zmm1, zmm1, zmm18
        vpaddd  zmm2, zmm2, zmm20
        vpaddd  zmm3, zmm3, zmm22
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm0, 16
        vprord  zmm13, zmm1, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm17
        vpaddd  zmm1, zmm1, zmm19
        vpaddd  zmm2, zmm2, zmm21
        vpaddd  zmm3, zmm3, zmm23
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm24
        vpaddd  zmm1, zmm1, zmm26
        vpaddd  zmm2, zmm2, zmm28
        vpaddd  zmm3, zmm3, zmm30
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm25
        vpaddd  zmm1, zmm1, zmm27
        vpaddd  zmm2, zmm2, zmm29
        vpaddd  zmm3, zmm3, zmm31
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpaddd  zmm0, zmm0, zmm18
        vpaddd  zmm1, zmm1, zmm19
        vpaddd  zmm2, zmm2, zmm23
        vpaddd  zmm3, zmm3, zmm20
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm22
        vpaddd  zmm1, zmm1, zmm26
        vpaddd  zmm2, zmm2, zmm16
        vpaddd  zmm3, zmm3, zmm29
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm17
        vpaddd  zmm1, zmm1, zmm28
        vpaddd  zmm2, zmm2, zmm25
        vpaddd  zmm3, zmm3, zmm31
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm27
        vpaddd  zmm1, zmm1, zmm21
        vpaddd  zmm2, zmm2, zmm30
        vpaddd  zmm3, zmm3, zmm24
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpaddd  zmm0, zmm0, zmm19
        vpaddd  zmm1, zmm1, zmm26
        vpaddd  zmm2, zmm2, zmm29
        vpaddd  zmm3, zmm3, zmm23
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm20
        vpaddd  zmm1, zmm1, zmm28
        vpaddd  zmm2, zmm2, zmm18
        vpaddd  zmm3, zmm3, zmm30
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm22
        vpaddd  zmm1, zmm1, zmm25
        vpaddd  zmm2, zmm2, zmm27
        vpaddd  zmm3, zmm3, zmm24
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm21
        vpaddd  zmm1, zmm1, zmm16
        vpaddd  zmm2, zmm2, zmm31
        vpaddd  zmm3, zmm3, zmm17
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpaddd  zmm0, zmm0, zmm26
        vpaddd  zmm1, zmm1, zmm28
        vpaddd  zmm2, zmm2, zmm30
        vpaddd  zmm3, zmm3, zmm29
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm23
        vpaddd  zmm1, zmm1, zmm25
        vpaddd  zmm2, zmm2, zmm19
        vpaddd  zmm3, zmm3, zmm31
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm20
        vpaddd  zmm1, zmm1, zmm27
        vpaddd  zmm2, zmm2, zmm21
        vpaddd  zmm3, zmm3, zmm17
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm16
        vpaddd  zmm1, zmm1, zmm18
        vpaddd  zmm2, zmm2, zmm24
        vpaddd  zmm3, zmm3, zmm22
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpaddd  zmm0, zmm0, zmm28
        vpaddd  zmm1, zmm1, zmm25
        vpaddd  zmm2, zmm2, zmm31
        vpaddd  zmm3, zmm3, zmm30
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm29
        vpaddd  zmm1, zmm1, zmm27
        vpaddd  zmm2, zmm2, zmm26
        vpaddd  zmm3, zmm3, zmm24
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm23
        vpaddd  zmm1, zmm1, zmm21
        vpaddd  zmm2, zmm2, zmm16
        vpaddd  zmm3, zmm3, zmm22
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm18
        vpaddd  zmm1, zmm1, zmm19
        vpaddd  zmm2, zmm2, zmm17
        vpaddd  zmm3, zmm3, zmm20
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpaddd  zmm0, zmm0, zmm25
        vpaddd  zmm1, zmm1, zmm27
        vpaddd  zmm2, zmm2, zmm24
        vpaddd  zmm3, zmm3, zmm31
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm30
        vpaddd  zmm1, zmm1, zmm21
        vpaddd  zmm2, zmm2, zmm28
        vpaddd  zmm3, zmm3, zmm17
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm29
        vpaddd  zmm1, zmm1, zmm16
        vpaddd  zmm2, zmm2, zmm18
        vpaddd  zmm3, zmm3, zmm20
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm19
        vpaddd  zmm1, zmm1, zmm26
        vpaddd  zmm2, zmm2, zmm22
        vpaddd  zmm3, zmm3, zmm23
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpaddd  zmm0, zmm0, zmm27
        vpaddd  zmm1, zmm1, zmm21
        vpaddd  zmm2, zmm2, zmm17
        vpaddd  zmm3, zmm3, zmm24
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm31
        vpaddd  zmm1, zmm1, zmm16
        vpaddd  zmm2, zmm2, zmm25
        vpaddd  zmm3, zmm3, zmm22
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm30
        vpaddd  zmm1, zmm1, zmm18
        vpaddd  zmm2, zmm2, zmm19
        vpaddd  zmm3, zmm3, zmm23
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm26
        vpaddd  zmm1, zmm1, zmm28
        vpaddd  zmm2, zmm2, zmm20
        vpaddd  zmm3, zmm3, zmm29
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpxord  zmm0, zmm0, zmm8
        vpxord  zmm1, zmm1, zmm9
        vpxord  zmm2, zmm2, zmm10
        vpxord  zmm3, zmm3, zmm11
        vpxord  zmm4, zmm4, zmm12
        vpxord  zmm5, zmm5, zmm13
        vpxord  zmm6, zmm6, zmm14
        vpxord  zmm7, zmm7, zmm15
        movzx   eax, byte ptr [rbp+0x38]
        jne     9b
        mov     rbx, qword ptr [rbp+0x50]
        test    r10d, r10d
        jnz     .Lflock_leaf_store_64
        vpunpckldq zmm16, zmm0, zmm1
        vpunpckhdq zmm17, zmm0, zmm1
        vpunpckldq zmm18, zmm2, zmm3
        vpunpckhdq zmm19, zmm2, zmm3
        vpunpckldq zmm20, zmm4, zmm5
        vpunpckhdq zmm21, zmm4, zmm5
        vpunpckldq zmm22, zmm6, zmm7
        vpunpckhdq zmm23, zmm6, zmm7
        vpunpcklqdq zmm0, zmm16, zmm18
        vpunpckhqdq zmm1, zmm16, zmm18
        vpunpcklqdq zmm2, zmm17, zmm19
        vpunpckhqdq zmm3, zmm17, zmm19
        vpunpcklqdq zmm4, zmm20, zmm22
        vpunpckhqdq zmm5, zmm20, zmm22
        vpunpcklqdq zmm6, zmm21, zmm23
        vpunpckhqdq zmm7, zmm21, zmm23
        vshufi32x4 zmm16, zmm0, zmm4, 0x88
        vshufi32x4 zmm17, zmm1, zmm5, 0x88
        vshufi32x4 zmm18, zmm2, zmm6, 0x88
        vshufi32x4 zmm19, zmm3, zmm7, 0x88
        vshufi32x4 zmm20, zmm0, zmm4, 0xDD
        vshufi32x4 zmm21, zmm1, zmm5, 0xDD
        vshufi32x4 zmm22, zmm2, zmm6, 0xDD
        vshufi32x4 zmm23, zmm3, zmm7, 0xDD
        vshufi32x4 zmm0, zmm16, zmm17, 0x88
        vshufi32x4 zmm1, zmm18, zmm19, 0x88
        vshufi32x4 zmm2, zmm20, zmm21, 0x88
        vshufi32x4 zmm3, zmm22, zmm23, 0x88
        vshufi32x4 zmm4, zmm16, zmm17, 0xDD
        vshufi32x4 zmm5, zmm18, zmm19, 0xDD
        vshufi32x4 zmm6, zmm20, zmm21, 0xDD
        vshufi32x4 zmm7, zmm22, zmm23, 0xDD
.Lflock_leaf_store_64:
        vmovdqu32 zmmword ptr [rbx], zmm0
        vmovdqu32 zmmword ptr [rbx+0x1*0x40], zmm1
        vmovdqu32 zmmword ptr [rbx+0x2*0x40], zmm2
        vmovdqu32 zmmword ptr [rbx+0x3*0x40], zmm3
        vmovdqu32 zmmword ptr [rbx+0x4*0x40], zmm4
        vmovdqu32 zmmword ptr [rbx+0x5*0x40], zmm5
        vmovdqu32 zmmword ptr [rbx+0x6*0x40], zmm6
        vmovdqu32 zmmword ptr [rbx+0x7*0x40], zmm7
        add     rdi, 1024
        add     rbx, 512
        mov     qword ptr [rbp+0x50], rbx
        sub     rsi, 16
        cmp     rsi, 16
        jnc     2b
4:
        vzeroupper
        mov     rsp, rbp
        pop     rbp
        pop     rbx
        pop     r12
        pop     r13
        pop     r14
        pop     r15
        ret
.size flock_blake3_contiguous_64, .-flock_blake3_contiguous_64
.size flock_blake3_planes_leaf_64, .-flock_blake3_planes_leaf_64
// R10 is call-clobbered and unused throughout the incumbent leaf body.
.p2align 6
.global flock_blake3_planes_leaf_128
.hidden flock_blake3_planes_leaf_128
.type flock_blake3_planes_leaf_128, @function
flock_blake3_planes_leaf_128:
        mov     r10d, 1
        jmp     .Lflock_leaf_shared_128
.p2align 6
.global flock_blake3_contiguous_128
.hidden flock_blake3_contiguous_128
.type flock_blake3_contiguous_128, @function
flock_blake3_contiguous_128:
        xor     r10d, r10d
.Lflock_leaf_shared_128:
        push    r15
        push    r14
        push    r13
        push    r12
        push    rbx
        push    rbp
        mov     rbp, rsp
        sub     rsp, 144
        and     rsp, 0xFFFFFFFFFFFFFFC0
2:
        vpbroadcastd zmm0, dword ptr [rcx]
        vpbroadcastd zmm1, dword ptr [rcx+0x1*0x4]
        vpbroadcastd zmm2, dword ptr [rcx+0x2*0x4]
        vpbroadcastd zmm3, dword ptr [rcx+0x3*0x4]
        vpbroadcastd zmm4, dword ptr [rcx+0x4*0x4]
        vpbroadcastd zmm5, dword ptr [rcx+0x5*0x4]
        vpbroadcastd zmm6, dword ptr [rcx+0x6*0x4]
        vpbroadcastd zmm7, dword ptr [rcx+0x7*0x4]
        movzx   eax, byte ptr [rbp+0x38]
        movzx   ebx, byte ptr [rbp+0x40]
        or      eax, ebx
        xor     edx, edx
.p2align 5
9:
        movzx   ebx, byte ptr [rbp+0x48]
        or      ebx, eax
        add     rdx, 64
        cmp     rdx, 128
        cmove   eax, ebx
        mov     dword ptr [rsp+0x88], eax
        vmovdqu32 ymm16, ymmword ptr [rdi+rdx+-64]
        vinserti64x4 zmm16, zmm16, ymmword ptr [rdi+rdx+960], 0x01
        vmovdqu32 ymm17, ymmword ptr [rdi+rdx+64]
        vinserti64x4 zmm17, zmm17, ymmword ptr [rdi+rdx+1088], 0x01
        vpunpcklqdq zmm8, zmm16, zmm17
        vpunpckhqdq zmm9, zmm16, zmm17
        vmovdqu32 ymm18, ymmword ptr [rdi+rdx+192]
        vinserti64x4 zmm18, zmm18, ymmword ptr [rdi+rdx+1216], 0x01
        vmovdqu32 ymm19, ymmword ptr [rdi+rdx+320]
        vinserti64x4 zmm19, zmm19, ymmword ptr [rdi+rdx+1344], 0x01
        vpunpcklqdq zmm10, zmm18, zmm19
        vpunpckhqdq zmm11, zmm18, zmm19
        vmovdqu32 ymm16, ymmword ptr [rdi+rdx+448]
        vinserti64x4 zmm16, zmm16, ymmword ptr [rdi+rdx+1472], 0x01
        vmovdqu32 ymm17, ymmword ptr [rdi+rdx+576]
        vinserti64x4 zmm17, zmm17, ymmword ptr [rdi+rdx+1600], 0x01
        vpunpcklqdq zmm12, zmm16, zmm17
        vpunpckhqdq zmm13, zmm16, zmm17
        vmovdqu32 ymm18, ymmword ptr [rdi+rdx+704]
        vinserti64x4 zmm18, zmm18, ymmword ptr [rdi+rdx+1728], 0x01
        vmovdqu32 ymm19, ymmword ptr [rdi+rdx+832]
        vinserti64x4 zmm19, zmm19, ymmword ptr [rdi+rdx+1856], 0x01
        vpunpcklqdq zmm14, zmm18, zmm19
        vpunpckhqdq zmm15, zmm18, zmm19
        vmovdqa32 zmm27, zmmword ptr [.Lflock_contiguous_INDEX0+rip]
        vmovdqa32 zmm31, zmmword ptr [.Lflock_contiguous_INDEX1+rip]
        vshufps zmm16, zmm8, zmm10, 136
        vshufps zmm17, zmm12, zmm14, 136
        vmovdqa32 zmm20, zmm16
        vpermt2d zmm16, zmm27, zmm17
        vpermt2d zmm20, zmm31, zmm17
        vshufps zmm17, zmm8, zmm10, 221
        vshufps zmm30, zmm12, zmm14, 221
        vmovdqa32 zmm21, zmm17
        vpermt2d zmm17, zmm27, zmm30
        vpermt2d zmm21, zmm31, zmm30
        vshufps zmm18, zmm9, zmm11, 136
        vshufps zmm8, zmm13, zmm15, 136
        vmovdqa32 zmm22, zmm18
        vpermt2d zmm18, zmm27, zmm8
        vpermt2d zmm22, zmm31, zmm8
        vshufps zmm19, zmm9, zmm11, 221
        vshufps zmm8, zmm13, zmm15, 221
        vmovdqa32 zmm23, zmm19
        vpermt2d zmm19, zmm27, zmm8
        vpermt2d zmm23, zmm31, zmm8
        vmovdqu32 ymm24, ymmword ptr [rdi+rdx+-32]
        vinserti64x4 zmm24, zmm24, ymmword ptr [rdi+rdx+992], 0x01
        vmovdqu32 ymm25, ymmword ptr [rdi+rdx+96]
        vinserti64x4 zmm25, zmm25, ymmword ptr [rdi+rdx+1120], 0x01
        vpunpcklqdq zmm8, zmm24, zmm25
        vpunpckhqdq zmm9, zmm24, zmm25
        vmovdqu32 ymm24, ymmword ptr [rdi+rdx+224]
        vinserti64x4 zmm24, zmm24, ymmword ptr [rdi+rdx+1248], 0x01
        vmovdqu32 ymm25, ymmword ptr [rdi+rdx+352]
        vinserti64x4 zmm25, zmm25, ymmword ptr [rdi+rdx+1376], 0x01
        vpunpcklqdq zmm10, zmm24, zmm25
        vpunpckhqdq zmm11, zmm24, zmm25
        prefetcht0 [rdi+rdx+128]
        prefetcht0 [rdi+rdx+1152]
        prefetcht0 [rdi+rdx+256]
        prefetcht0 [rdi+rdx+1280]
        prefetcht0 [rdi+rdx+384]
        prefetcht0 [rdi+rdx+1408]
        prefetcht0 [rdi+rdx+512]
        prefetcht0 [rdi+rdx+1536]
        vmovdqu32 ymm24, ymmword ptr [rdi+rdx+480]
        vinserti64x4 zmm24, zmm24, ymmword ptr [rdi+rdx+1504], 0x01
        vmovdqu32 ymm25, ymmword ptr [rdi+rdx+608]
        vinserti64x4 zmm25, zmm25, ymmword ptr [rdi+rdx+1632], 0x01
        vpunpcklqdq zmm12, zmm24, zmm25
        vpunpckhqdq zmm13, zmm24, zmm25
        vmovdqu32 ymm24, ymmword ptr [rdi+rdx+736]
        vinserti64x4 zmm24, zmm24, ymmword ptr [rdi+rdx+1760], 0x01
        vmovdqu32 ymm25, ymmword ptr [rdi+rdx+864]
        vinserti64x4 zmm25, zmm25, ymmword ptr [rdi+rdx+1888], 0x01
        vpunpcklqdq zmm14, zmm24, zmm25
        vpunpckhqdq zmm15, zmm24, zmm25
        prefetcht0 [rdi+rdx+640]
        prefetcht0 [rdi+rdx+1664]
        prefetcht0 [rdi+rdx+768]
        prefetcht0 [rdi+rdx+1792]
        prefetcht0 [rdi+rdx+896]
        prefetcht0 [rdi+rdx+1920]
        prefetcht0 [rdi+rdx+1024]
        prefetcht0 [rdi+rdx+2048]
        vshufps zmm24, zmm8, zmm10, 136
        vshufps zmm30, zmm12, zmm14, 136
        vmovdqa32 zmm28, zmm24
        vpermt2d zmm24, zmm27, zmm30
        vpermt2d zmm28, zmm31, zmm30
        vshufps zmm25, zmm8, zmm10, 221
        vshufps zmm30, zmm12, zmm14, 221
        vmovdqa32 zmm29, zmm25
        vpermt2d zmm25, zmm27, zmm30
        vpermt2d zmm29, zmm31, zmm30
        vshufps zmm26, zmm9, zmm11, 136
        vshufps zmm8, zmm13, zmm15, 136
        vmovdqa32 zmm30, zmm26
        vpermt2d zmm26, zmm27, zmm8
        vpermt2d zmm30, zmm31, zmm8
        vshufps zmm8, zmm9, zmm11, 221
        vshufps zmm10, zmm13, zmm15, 221
        vpermi2d zmm27, zmm8, zmm10
        vpermi2d zmm31, zmm8, zmm10
        vpbroadcastd zmm8, dword ptr [.Lflock_contiguous_BLAKE3_IV_0+rip]
        vpbroadcastd zmm9, dword ptr [.Lflock_contiguous_BLAKE3_IV_1+rip]
        vpbroadcastd zmm10, dword ptr [.Lflock_contiguous_BLAKE3_IV_2+rip]
        vpbroadcastd zmm11, dword ptr [.Lflock_contiguous_BLAKE3_IV_3+rip]
        vpbroadcastd zmm14, dword ptr [.Lflock_contiguous_BLAKE3_BLOCK_LEN+rip]
        vpbroadcastd zmm15, dword ptr [rsp+0x22*0x4]
        vpaddd  zmm0, zmm0, zmm16
        vpaddd  zmm1, zmm1, zmm18
        vpaddd  zmm2, zmm2, zmm20
        vpaddd  zmm3, zmm3, zmm22
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm0, 16
        vprord  zmm13, zmm1, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm17
        vpaddd  zmm1, zmm1, zmm19
        vpaddd  zmm2, zmm2, zmm21
        vpaddd  zmm3, zmm3, zmm23
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm24
        vpaddd  zmm1, zmm1, zmm26
        vpaddd  zmm2, zmm2, zmm28
        vpaddd  zmm3, zmm3, zmm30
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm25
        vpaddd  zmm1, zmm1, zmm27
        vpaddd  zmm2, zmm2, zmm29
        vpaddd  zmm3, zmm3, zmm31
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpaddd  zmm0, zmm0, zmm18
        vpaddd  zmm1, zmm1, zmm19
        vpaddd  zmm2, zmm2, zmm23
        vpaddd  zmm3, zmm3, zmm20
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm22
        vpaddd  zmm1, zmm1, zmm26
        vpaddd  zmm2, zmm2, zmm16
        vpaddd  zmm3, zmm3, zmm29
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm17
        vpaddd  zmm1, zmm1, zmm28
        vpaddd  zmm2, zmm2, zmm25
        vpaddd  zmm3, zmm3, zmm31
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm27
        vpaddd  zmm1, zmm1, zmm21
        vpaddd  zmm2, zmm2, zmm30
        vpaddd  zmm3, zmm3, zmm24
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpaddd  zmm0, zmm0, zmm19
        vpaddd  zmm1, zmm1, zmm26
        vpaddd  zmm2, zmm2, zmm29
        vpaddd  zmm3, zmm3, zmm23
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm20
        vpaddd  zmm1, zmm1, zmm28
        vpaddd  zmm2, zmm2, zmm18
        vpaddd  zmm3, zmm3, zmm30
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm22
        vpaddd  zmm1, zmm1, zmm25
        vpaddd  zmm2, zmm2, zmm27
        vpaddd  zmm3, zmm3, zmm24
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm21
        vpaddd  zmm1, zmm1, zmm16
        vpaddd  zmm2, zmm2, zmm31
        vpaddd  zmm3, zmm3, zmm17
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpaddd  zmm0, zmm0, zmm26
        vpaddd  zmm1, zmm1, zmm28
        vpaddd  zmm2, zmm2, zmm30
        vpaddd  zmm3, zmm3, zmm29
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm23
        vpaddd  zmm1, zmm1, zmm25
        vpaddd  zmm2, zmm2, zmm19
        vpaddd  zmm3, zmm3, zmm31
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm20
        vpaddd  zmm1, zmm1, zmm27
        vpaddd  zmm2, zmm2, zmm21
        vpaddd  zmm3, zmm3, zmm17
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm16
        vpaddd  zmm1, zmm1, zmm18
        vpaddd  zmm2, zmm2, zmm24
        vpaddd  zmm3, zmm3, zmm22
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpaddd  zmm0, zmm0, zmm28
        vpaddd  zmm1, zmm1, zmm25
        vpaddd  zmm2, zmm2, zmm31
        vpaddd  zmm3, zmm3, zmm30
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm29
        vpaddd  zmm1, zmm1, zmm27
        vpaddd  zmm2, zmm2, zmm26
        vpaddd  zmm3, zmm3, zmm24
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm23
        vpaddd  zmm1, zmm1, zmm21
        vpaddd  zmm2, zmm2, zmm16
        vpaddd  zmm3, zmm3, zmm22
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm18
        vpaddd  zmm1, zmm1, zmm19
        vpaddd  zmm2, zmm2, zmm17
        vpaddd  zmm3, zmm3, zmm20
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpaddd  zmm0, zmm0, zmm25
        vpaddd  zmm1, zmm1, zmm27
        vpaddd  zmm2, zmm2, zmm24
        vpaddd  zmm3, zmm3, zmm31
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm30
        vpaddd  zmm1, zmm1, zmm21
        vpaddd  zmm2, zmm2, zmm28
        vpaddd  zmm3, zmm3, zmm17
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm29
        vpaddd  zmm1, zmm1, zmm16
        vpaddd  zmm2, zmm2, zmm18
        vpaddd  zmm3, zmm3, zmm20
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm19
        vpaddd  zmm1, zmm1, zmm26
        vpaddd  zmm2, zmm2, zmm22
        vpaddd  zmm3, zmm3, zmm23
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpaddd  zmm0, zmm0, zmm27
        vpaddd  zmm1, zmm1, zmm21
        vpaddd  zmm2, zmm2, zmm17
        vpaddd  zmm3, zmm3, zmm24
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm31
        vpaddd  zmm1, zmm1, zmm16
        vpaddd  zmm2, zmm2, zmm25
        vpaddd  zmm3, zmm3, zmm22
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm30
        vpaddd  zmm1, zmm1, zmm18
        vpaddd  zmm2, zmm2, zmm19
        vpaddd  zmm3, zmm3, zmm23
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm26
        vpaddd  zmm1, zmm1, zmm28
        vpaddd  zmm2, zmm2, zmm20
        vpaddd  zmm3, zmm3, zmm29
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpxord  zmm0, zmm0, zmm8
        vpxord  zmm1, zmm1, zmm9
        vpxord  zmm2, zmm2, zmm10
        vpxord  zmm3, zmm3, zmm11
        vpxord  zmm4, zmm4, zmm12
        vpxord  zmm5, zmm5, zmm13
        vpxord  zmm6, zmm6, zmm14
        vpxord  zmm7, zmm7, zmm15
        movzx   eax, byte ptr [rbp+0x38]
        jne     9b
        mov     rbx, qword ptr [rbp+0x50]
        test    r10d, r10d
        jnz     .Lflock_leaf_store_128
        vpunpckldq zmm16, zmm0, zmm1
        vpunpckhdq zmm17, zmm0, zmm1
        vpunpckldq zmm18, zmm2, zmm3
        vpunpckhdq zmm19, zmm2, zmm3
        vpunpckldq zmm20, zmm4, zmm5
        vpunpckhdq zmm21, zmm4, zmm5
        vpunpckldq zmm22, zmm6, zmm7
        vpunpckhdq zmm23, zmm6, zmm7
        vpunpcklqdq zmm0, zmm16, zmm18
        vpunpckhqdq zmm1, zmm16, zmm18
        vpunpcklqdq zmm2, zmm17, zmm19
        vpunpckhqdq zmm3, zmm17, zmm19
        vpunpcklqdq zmm4, zmm20, zmm22
        vpunpckhqdq zmm5, zmm20, zmm22
        vpunpcklqdq zmm6, zmm21, zmm23
        vpunpckhqdq zmm7, zmm21, zmm23
        vshufi32x4 zmm16, zmm0, zmm4, 0x88
        vshufi32x4 zmm17, zmm1, zmm5, 0x88
        vshufi32x4 zmm18, zmm2, zmm6, 0x88
        vshufi32x4 zmm19, zmm3, zmm7, 0x88
        vshufi32x4 zmm20, zmm0, zmm4, 0xDD
        vshufi32x4 zmm21, zmm1, zmm5, 0xDD
        vshufi32x4 zmm22, zmm2, zmm6, 0xDD
        vshufi32x4 zmm23, zmm3, zmm7, 0xDD
        vshufi32x4 zmm0, zmm16, zmm17, 0x88
        vshufi32x4 zmm1, zmm18, zmm19, 0x88
        vshufi32x4 zmm2, zmm20, zmm21, 0x88
        vshufi32x4 zmm3, zmm22, zmm23, 0x88
        vshufi32x4 zmm4, zmm16, zmm17, 0xDD
        vshufi32x4 zmm5, zmm18, zmm19, 0xDD
        vshufi32x4 zmm6, zmm20, zmm21, 0xDD
        vshufi32x4 zmm7, zmm22, zmm23, 0xDD
.Lflock_leaf_store_128:
        vmovdqu32 zmmword ptr [rbx], zmm0
        vmovdqu32 zmmword ptr [rbx+0x1*0x40], zmm1
        vmovdqu32 zmmword ptr [rbx+0x2*0x40], zmm2
        vmovdqu32 zmmword ptr [rbx+0x3*0x40], zmm3
        vmovdqu32 zmmword ptr [rbx+0x4*0x40], zmm4
        vmovdqu32 zmmword ptr [rbx+0x5*0x40], zmm5
        vmovdqu32 zmmword ptr [rbx+0x6*0x40], zmm6
        vmovdqu32 zmmword ptr [rbx+0x7*0x40], zmm7
        add     rdi, 2048
        add     rbx, 512
        mov     qword ptr [rbp+0x50], rbx
        sub     rsi, 16
        cmp     rsi, 16
        jnc     2b
4:
        vzeroupper
        mov     rsp, rbp
        pop     rbp
        pop     rbx
        pop     r12
        pop     r13
        pop     r14
        pop     r15
        ret
.size flock_blake3_contiguous_128, .-flock_blake3_contiguous_128
.size flock_blake3_planes_leaf_128, .-flock_blake3_planes_leaf_128
// R10 is call-clobbered and unused throughout the incumbent leaf body.
.p2align 6
.global flock_blake3_planes_leaf_256
.hidden flock_blake3_planes_leaf_256
.type flock_blake3_planes_leaf_256, @function
flock_blake3_planes_leaf_256:
        mov     r10d, 1
        jmp     .Lflock_leaf_shared_256
.p2align 6
.global flock_blake3_contiguous_256
.hidden flock_blake3_contiguous_256
.type flock_blake3_contiguous_256, @function
flock_blake3_contiguous_256:
        xor     r10d, r10d
.Lflock_leaf_shared_256:
        push    r15
        push    r14
        push    r13
        push    r12
        push    rbx
        push    rbp
        mov     rbp, rsp
        sub     rsp, 144
        and     rsp, 0xFFFFFFFFFFFFFFC0
2:
        vpbroadcastd zmm0, dword ptr [rcx]
        vpbroadcastd zmm1, dword ptr [rcx+0x1*0x4]
        vpbroadcastd zmm2, dword ptr [rcx+0x2*0x4]
        vpbroadcastd zmm3, dword ptr [rcx+0x3*0x4]
        vpbroadcastd zmm4, dword ptr [rcx+0x4*0x4]
        vpbroadcastd zmm5, dword ptr [rcx+0x5*0x4]
        vpbroadcastd zmm6, dword ptr [rcx+0x6*0x4]
        vpbroadcastd zmm7, dword ptr [rcx+0x7*0x4]
        movzx   eax, byte ptr [rbp+0x38]
        movzx   ebx, byte ptr [rbp+0x40]
        or      eax, ebx
        xor     edx, edx
.p2align 5
9:
        movzx   ebx, byte ptr [rbp+0x48]
        or      ebx, eax
        add     rdx, 64
        cmp     rdx, 256
        cmove   eax, ebx
        mov     dword ptr [rsp+0x88], eax
        vmovdqu32 ymm16, ymmword ptr [rdi+rdx+-64]
        vinserti64x4 zmm16, zmm16, ymmword ptr [rdi+rdx+1984], 0x01
        vmovdqu32 ymm17, ymmword ptr [rdi+rdx+192]
        vinserti64x4 zmm17, zmm17, ymmword ptr [rdi+rdx+2240], 0x01
        vpunpcklqdq zmm8, zmm16, zmm17
        vpunpckhqdq zmm9, zmm16, zmm17
        vmovdqu32 ymm18, ymmword ptr [rdi+rdx+448]
        vinserti64x4 zmm18, zmm18, ymmword ptr [rdi+rdx+2496], 0x01
        vmovdqu32 ymm19, ymmword ptr [rdi+rdx+704]
        vinserti64x4 zmm19, zmm19, ymmword ptr [rdi+rdx+2752], 0x01
        vpunpcklqdq zmm10, zmm18, zmm19
        vpunpckhqdq zmm11, zmm18, zmm19
        vmovdqu32 ymm16, ymmword ptr [rdi+rdx+960]
        vinserti64x4 zmm16, zmm16, ymmword ptr [rdi+rdx+3008], 0x01
        vmovdqu32 ymm17, ymmword ptr [rdi+rdx+1216]
        vinserti64x4 zmm17, zmm17, ymmword ptr [rdi+rdx+3264], 0x01
        vpunpcklqdq zmm12, zmm16, zmm17
        vpunpckhqdq zmm13, zmm16, zmm17
        vmovdqu32 ymm18, ymmword ptr [rdi+rdx+1472]
        vinserti64x4 zmm18, zmm18, ymmword ptr [rdi+rdx+3520], 0x01
        vmovdqu32 ymm19, ymmword ptr [rdi+rdx+1728]
        vinserti64x4 zmm19, zmm19, ymmword ptr [rdi+rdx+3776], 0x01
        vpunpcklqdq zmm14, zmm18, zmm19
        vpunpckhqdq zmm15, zmm18, zmm19
        vmovdqa32 zmm27, zmmword ptr [.Lflock_contiguous_INDEX0+rip]
        vmovdqa32 zmm31, zmmword ptr [.Lflock_contiguous_INDEX1+rip]
        vshufps zmm16, zmm8, zmm10, 136
        vshufps zmm17, zmm12, zmm14, 136
        vmovdqa32 zmm20, zmm16
        vpermt2d zmm16, zmm27, zmm17
        vpermt2d zmm20, zmm31, zmm17
        vshufps zmm17, zmm8, zmm10, 221
        vshufps zmm30, zmm12, zmm14, 221
        vmovdqa32 zmm21, zmm17
        vpermt2d zmm17, zmm27, zmm30
        vpermt2d zmm21, zmm31, zmm30
        vshufps zmm18, zmm9, zmm11, 136
        vshufps zmm8, zmm13, zmm15, 136
        vmovdqa32 zmm22, zmm18
        vpermt2d zmm18, zmm27, zmm8
        vpermt2d zmm22, zmm31, zmm8
        vshufps zmm19, zmm9, zmm11, 221
        vshufps zmm8, zmm13, zmm15, 221
        vmovdqa32 zmm23, zmm19
        vpermt2d zmm19, zmm27, zmm8
        vpermt2d zmm23, zmm31, zmm8
        vmovdqu32 ymm24, ymmword ptr [rdi+rdx+-32]
        vinserti64x4 zmm24, zmm24, ymmword ptr [rdi+rdx+2016], 0x01
        vmovdqu32 ymm25, ymmword ptr [rdi+rdx+224]
        vinserti64x4 zmm25, zmm25, ymmword ptr [rdi+rdx+2272], 0x01
        vpunpcklqdq zmm8, zmm24, zmm25
        vpunpckhqdq zmm9, zmm24, zmm25
        vmovdqu32 ymm24, ymmword ptr [rdi+rdx+480]
        vinserti64x4 zmm24, zmm24, ymmword ptr [rdi+rdx+2528], 0x01
        vmovdqu32 ymm25, ymmword ptr [rdi+rdx+736]
        vinserti64x4 zmm25, zmm25, ymmword ptr [rdi+rdx+2784], 0x01
        vpunpcklqdq zmm10, zmm24, zmm25
        vpunpckhqdq zmm11, zmm24, zmm25
        prefetcht0 [rdi+rdx+128]
        prefetcht0 [rdi+rdx+2176]
        prefetcht0 [rdi+rdx+384]
        prefetcht0 [rdi+rdx+2432]
        prefetcht0 [rdi+rdx+640]
        prefetcht0 [rdi+rdx+2688]
        prefetcht0 [rdi+rdx+896]
        prefetcht0 [rdi+rdx+2944]
        vmovdqu32 ymm24, ymmword ptr [rdi+rdx+992]
        vinserti64x4 zmm24, zmm24, ymmword ptr [rdi+rdx+3040], 0x01
        vmovdqu32 ymm25, ymmword ptr [rdi+rdx+1248]
        vinserti64x4 zmm25, zmm25, ymmword ptr [rdi+rdx+3296], 0x01
        vpunpcklqdq zmm12, zmm24, zmm25
        vpunpckhqdq zmm13, zmm24, zmm25
        vmovdqu32 ymm24, ymmword ptr [rdi+rdx+1504]
        vinserti64x4 zmm24, zmm24, ymmword ptr [rdi+rdx+3552], 0x01
        vmovdqu32 ymm25, ymmword ptr [rdi+rdx+1760]
        vinserti64x4 zmm25, zmm25, ymmword ptr [rdi+rdx+3808], 0x01
        vpunpcklqdq zmm14, zmm24, zmm25
        vpunpckhqdq zmm15, zmm24, zmm25
        prefetcht0 [rdi+rdx+1152]
        prefetcht0 [rdi+rdx+3200]
        prefetcht0 [rdi+rdx+1408]
        prefetcht0 [rdi+rdx+3456]
        prefetcht0 [rdi+rdx+1664]
        prefetcht0 [rdi+rdx+3712]
        prefetcht0 [rdi+rdx+1920]
        prefetcht0 [rdi+rdx+3968]
        vshufps zmm24, zmm8, zmm10, 136
        vshufps zmm30, zmm12, zmm14, 136
        vmovdqa32 zmm28, zmm24
        vpermt2d zmm24, zmm27, zmm30
        vpermt2d zmm28, zmm31, zmm30
        vshufps zmm25, zmm8, zmm10, 221
        vshufps zmm30, zmm12, zmm14, 221
        vmovdqa32 zmm29, zmm25
        vpermt2d zmm25, zmm27, zmm30
        vpermt2d zmm29, zmm31, zmm30
        vshufps zmm26, zmm9, zmm11, 136
        vshufps zmm8, zmm13, zmm15, 136
        vmovdqa32 zmm30, zmm26
        vpermt2d zmm26, zmm27, zmm8
        vpermt2d zmm30, zmm31, zmm8
        vshufps zmm8, zmm9, zmm11, 221
        vshufps zmm10, zmm13, zmm15, 221
        vpermi2d zmm27, zmm8, zmm10
        vpermi2d zmm31, zmm8, zmm10
        vpbroadcastd zmm8, dword ptr [.Lflock_contiguous_BLAKE3_IV_0+rip]
        vpbroadcastd zmm9, dword ptr [.Lflock_contiguous_BLAKE3_IV_1+rip]
        vpbroadcastd zmm10, dword ptr [.Lflock_contiguous_BLAKE3_IV_2+rip]
        vpbroadcastd zmm11, dword ptr [.Lflock_contiguous_BLAKE3_IV_3+rip]
        vpbroadcastd zmm14, dword ptr [.Lflock_contiguous_BLAKE3_BLOCK_LEN+rip]
        vpbroadcastd zmm15, dword ptr [rsp+0x22*0x4]
        vpaddd  zmm0, zmm0, zmm16
        vpaddd  zmm1, zmm1, zmm18
        vpaddd  zmm2, zmm2, zmm20
        vpaddd  zmm3, zmm3, zmm22
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm0, 16
        vprord  zmm13, zmm1, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm17
        vpaddd  zmm1, zmm1, zmm19
        vpaddd  zmm2, zmm2, zmm21
        vpaddd  zmm3, zmm3, zmm23
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm24
        vpaddd  zmm1, zmm1, zmm26
        vpaddd  zmm2, zmm2, zmm28
        vpaddd  zmm3, zmm3, zmm30
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm25
        vpaddd  zmm1, zmm1, zmm27
        vpaddd  zmm2, zmm2, zmm29
        vpaddd  zmm3, zmm3, zmm31
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpaddd  zmm0, zmm0, zmm18
        vpaddd  zmm1, zmm1, zmm19
        vpaddd  zmm2, zmm2, zmm23
        vpaddd  zmm3, zmm3, zmm20
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm22
        vpaddd  zmm1, zmm1, zmm26
        vpaddd  zmm2, zmm2, zmm16
        vpaddd  zmm3, zmm3, zmm29
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm17
        vpaddd  zmm1, zmm1, zmm28
        vpaddd  zmm2, zmm2, zmm25
        vpaddd  zmm3, zmm3, zmm31
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm27
        vpaddd  zmm1, zmm1, zmm21
        vpaddd  zmm2, zmm2, zmm30
        vpaddd  zmm3, zmm3, zmm24
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpaddd  zmm0, zmm0, zmm19
        vpaddd  zmm1, zmm1, zmm26
        vpaddd  zmm2, zmm2, zmm29
        vpaddd  zmm3, zmm3, zmm23
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm20
        vpaddd  zmm1, zmm1, zmm28
        vpaddd  zmm2, zmm2, zmm18
        vpaddd  zmm3, zmm3, zmm30
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm22
        vpaddd  zmm1, zmm1, zmm25
        vpaddd  zmm2, zmm2, zmm27
        vpaddd  zmm3, zmm3, zmm24
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm21
        vpaddd  zmm1, zmm1, zmm16
        vpaddd  zmm2, zmm2, zmm31
        vpaddd  zmm3, zmm3, zmm17
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpaddd  zmm0, zmm0, zmm26
        vpaddd  zmm1, zmm1, zmm28
        vpaddd  zmm2, zmm2, zmm30
        vpaddd  zmm3, zmm3, zmm29
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm23
        vpaddd  zmm1, zmm1, zmm25
        vpaddd  zmm2, zmm2, zmm19
        vpaddd  zmm3, zmm3, zmm31
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm20
        vpaddd  zmm1, zmm1, zmm27
        vpaddd  zmm2, zmm2, zmm21
        vpaddd  zmm3, zmm3, zmm17
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm16
        vpaddd  zmm1, zmm1, zmm18
        vpaddd  zmm2, zmm2, zmm24
        vpaddd  zmm3, zmm3, zmm22
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpaddd  zmm0, zmm0, zmm28
        vpaddd  zmm1, zmm1, zmm25
        vpaddd  zmm2, zmm2, zmm31
        vpaddd  zmm3, zmm3, zmm30
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm29
        vpaddd  zmm1, zmm1, zmm27
        vpaddd  zmm2, zmm2, zmm26
        vpaddd  zmm3, zmm3, zmm24
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm23
        vpaddd  zmm1, zmm1, zmm21
        vpaddd  zmm2, zmm2, zmm16
        vpaddd  zmm3, zmm3, zmm22
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm18
        vpaddd  zmm1, zmm1, zmm19
        vpaddd  zmm2, zmm2, zmm17
        vpaddd  zmm3, zmm3, zmm20
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpaddd  zmm0, zmm0, zmm25
        vpaddd  zmm1, zmm1, zmm27
        vpaddd  zmm2, zmm2, zmm24
        vpaddd  zmm3, zmm3, zmm31
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm30
        vpaddd  zmm1, zmm1, zmm21
        vpaddd  zmm2, zmm2, zmm28
        vpaddd  zmm3, zmm3, zmm17
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm29
        vpaddd  zmm1, zmm1, zmm16
        vpaddd  zmm2, zmm2, zmm18
        vpaddd  zmm3, zmm3, zmm20
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm19
        vpaddd  zmm1, zmm1, zmm26
        vpaddd  zmm2, zmm2, zmm22
        vpaddd  zmm3, zmm3, zmm23
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpaddd  zmm0, zmm0, zmm27
        vpaddd  zmm1, zmm1, zmm21
        vpaddd  zmm2, zmm2, zmm17
        vpaddd  zmm3, zmm3, zmm24
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm31
        vpaddd  zmm1, zmm1, zmm16
        vpaddd  zmm2, zmm2, zmm25
        vpaddd  zmm3, zmm3, zmm22
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm30
        vpaddd  zmm1, zmm1, zmm18
        vpaddd  zmm2, zmm2, zmm19
        vpaddd  zmm3, zmm3, zmm23
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm26
        vpaddd  zmm1, zmm1, zmm28
        vpaddd  zmm2, zmm2, zmm20
        vpaddd  zmm3, zmm3, zmm29
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpxord  zmm0, zmm0, zmm8
        vpxord  zmm1, zmm1, zmm9
        vpxord  zmm2, zmm2, zmm10
        vpxord  zmm3, zmm3, zmm11
        vpxord  zmm4, zmm4, zmm12
        vpxord  zmm5, zmm5, zmm13
        vpxord  zmm6, zmm6, zmm14
        vpxord  zmm7, zmm7, zmm15
        movzx   eax, byte ptr [rbp+0x38]
        jne     9b
        mov     rbx, qword ptr [rbp+0x50]
        test    r10d, r10d
        jnz     .Lflock_leaf_store_256
        vpunpckldq zmm16, zmm0, zmm1
        vpunpckhdq zmm17, zmm0, zmm1
        vpunpckldq zmm18, zmm2, zmm3
        vpunpckhdq zmm19, zmm2, zmm3
        vpunpckldq zmm20, zmm4, zmm5
        vpunpckhdq zmm21, zmm4, zmm5
        vpunpckldq zmm22, zmm6, zmm7
        vpunpckhdq zmm23, zmm6, zmm7
        vpunpcklqdq zmm0, zmm16, zmm18
        vpunpckhqdq zmm1, zmm16, zmm18
        vpunpcklqdq zmm2, zmm17, zmm19
        vpunpckhqdq zmm3, zmm17, zmm19
        vpunpcklqdq zmm4, zmm20, zmm22
        vpunpckhqdq zmm5, zmm20, zmm22
        vpunpcklqdq zmm6, zmm21, zmm23
        vpunpckhqdq zmm7, zmm21, zmm23
        vshufi32x4 zmm16, zmm0, zmm4, 0x88
        vshufi32x4 zmm17, zmm1, zmm5, 0x88
        vshufi32x4 zmm18, zmm2, zmm6, 0x88
        vshufi32x4 zmm19, zmm3, zmm7, 0x88
        vshufi32x4 zmm20, zmm0, zmm4, 0xDD
        vshufi32x4 zmm21, zmm1, zmm5, 0xDD
        vshufi32x4 zmm22, zmm2, zmm6, 0xDD
        vshufi32x4 zmm23, zmm3, zmm7, 0xDD
        vshufi32x4 zmm0, zmm16, zmm17, 0x88
        vshufi32x4 zmm1, zmm18, zmm19, 0x88
        vshufi32x4 zmm2, zmm20, zmm21, 0x88
        vshufi32x4 zmm3, zmm22, zmm23, 0x88
        vshufi32x4 zmm4, zmm16, zmm17, 0xDD
        vshufi32x4 zmm5, zmm18, zmm19, 0xDD
        vshufi32x4 zmm6, zmm20, zmm21, 0xDD
        vshufi32x4 zmm7, zmm22, zmm23, 0xDD
.Lflock_leaf_store_256:
        vmovdqu32 zmmword ptr [rbx], zmm0
        vmovdqu32 zmmword ptr [rbx+0x1*0x40], zmm1
        vmovdqu32 zmmword ptr [rbx+0x2*0x40], zmm2
        vmovdqu32 zmmword ptr [rbx+0x3*0x40], zmm3
        vmovdqu32 zmmword ptr [rbx+0x4*0x40], zmm4
        vmovdqu32 zmmword ptr [rbx+0x5*0x40], zmm5
        vmovdqu32 zmmword ptr [rbx+0x6*0x40], zmm6
        vmovdqu32 zmmword ptr [rbx+0x7*0x40], zmm7
        add     rdi, 4096
        add     rbx, 512
        mov     qword ptr [rbp+0x50], rbx
        sub     rsi, 16
        cmp     rsi, 16
        jnc     2b
4:
        vzeroupper
        mov     rsp, rbp
        pop     rbp
        pop     rbx
        pop     r12
        pop     r13
        pop     r14
        pop     r15
        ret
.size flock_blake3_contiguous_256, .-flock_blake3_contiguous_256
.size flock_blake3_planes_leaf_256, .-flock_blake3_planes_leaf_256
// R10 is call-clobbered and unused throughout the incumbent leaf body.
.p2align 6
.global flock_blake3_planes_leaf_512
.hidden flock_blake3_planes_leaf_512
.type flock_blake3_planes_leaf_512, @function
flock_blake3_planes_leaf_512:
        mov     r10d, 1
        jmp     .Lflock_leaf_shared_512
.p2align 6
.global flock_blake3_contiguous_512
.hidden flock_blake3_contiguous_512
.type flock_blake3_contiguous_512, @function
flock_blake3_contiguous_512:
        xor     r10d, r10d
.Lflock_leaf_shared_512:
        push    r15
        push    r14
        push    r13
        push    r12
        push    rbx
        push    rbp
        mov     rbp, rsp
        sub     rsp, 144
        and     rsp, 0xFFFFFFFFFFFFFFC0
2:
        vpbroadcastd zmm0, dword ptr [rcx]
        vpbroadcastd zmm1, dword ptr [rcx+0x1*0x4]
        vpbroadcastd zmm2, dword ptr [rcx+0x2*0x4]
        vpbroadcastd zmm3, dword ptr [rcx+0x3*0x4]
        vpbroadcastd zmm4, dword ptr [rcx+0x4*0x4]
        vpbroadcastd zmm5, dword ptr [rcx+0x5*0x4]
        vpbroadcastd zmm6, dword ptr [rcx+0x6*0x4]
        vpbroadcastd zmm7, dword ptr [rcx+0x7*0x4]
        movzx   eax, byte ptr [rbp+0x38]
        movzx   ebx, byte ptr [rbp+0x40]
        or      eax, ebx
        xor     edx, edx
.p2align 5
9:
        movzx   ebx, byte ptr [rbp+0x48]
        or      ebx, eax
        add     rdx, 64
        cmp     rdx, 512
        cmove   eax, ebx
        mov     dword ptr [rsp+0x88], eax
        vmovdqu32 ymm16, ymmword ptr [rdi+rdx+-64]
        vinserti64x4 zmm16, zmm16, ymmword ptr [rdi+rdx+4032], 0x01
        vmovdqu32 ymm17, ymmword ptr [rdi+rdx+448]
        vinserti64x4 zmm17, zmm17, ymmword ptr [rdi+rdx+4544], 0x01
        vpunpcklqdq zmm8, zmm16, zmm17
        vpunpckhqdq zmm9, zmm16, zmm17
        vmovdqu32 ymm18, ymmword ptr [rdi+rdx+960]
        vinserti64x4 zmm18, zmm18, ymmword ptr [rdi+rdx+5056], 0x01
        vmovdqu32 ymm19, ymmword ptr [rdi+rdx+1472]
        vinserti64x4 zmm19, zmm19, ymmword ptr [rdi+rdx+5568], 0x01
        vpunpcklqdq zmm10, zmm18, zmm19
        vpunpckhqdq zmm11, zmm18, zmm19
        vmovdqu32 ymm16, ymmword ptr [rdi+rdx+1984]
        vinserti64x4 zmm16, zmm16, ymmword ptr [rdi+rdx+6080], 0x01
        vmovdqu32 ymm17, ymmword ptr [rdi+rdx+2496]
        vinserti64x4 zmm17, zmm17, ymmword ptr [rdi+rdx+6592], 0x01
        vpunpcklqdq zmm12, zmm16, zmm17
        vpunpckhqdq zmm13, zmm16, zmm17
        vmovdqu32 ymm18, ymmword ptr [rdi+rdx+3008]
        vinserti64x4 zmm18, zmm18, ymmword ptr [rdi+rdx+7104], 0x01
        vmovdqu32 ymm19, ymmword ptr [rdi+rdx+3520]
        vinserti64x4 zmm19, zmm19, ymmword ptr [rdi+rdx+7616], 0x01
        vpunpcklqdq zmm14, zmm18, zmm19
        vpunpckhqdq zmm15, zmm18, zmm19
        vmovdqa32 zmm27, zmmword ptr [.Lflock_contiguous_INDEX0+rip]
        vmovdqa32 zmm31, zmmword ptr [.Lflock_contiguous_INDEX1+rip]
        vshufps zmm16, zmm8, zmm10, 136
        vshufps zmm17, zmm12, zmm14, 136
        vmovdqa32 zmm20, zmm16
        vpermt2d zmm16, zmm27, zmm17
        vpermt2d zmm20, zmm31, zmm17
        vshufps zmm17, zmm8, zmm10, 221
        vshufps zmm30, zmm12, zmm14, 221
        vmovdqa32 zmm21, zmm17
        vpermt2d zmm17, zmm27, zmm30
        vpermt2d zmm21, zmm31, zmm30
        vshufps zmm18, zmm9, zmm11, 136
        vshufps zmm8, zmm13, zmm15, 136
        vmovdqa32 zmm22, zmm18
        vpermt2d zmm18, zmm27, zmm8
        vpermt2d zmm22, zmm31, zmm8
        vshufps zmm19, zmm9, zmm11, 221
        vshufps zmm8, zmm13, zmm15, 221
        vmovdqa32 zmm23, zmm19
        vpermt2d zmm19, zmm27, zmm8
        vpermt2d zmm23, zmm31, zmm8
        vmovdqu32 ymm24, ymmword ptr [rdi+rdx+-32]
        vinserti64x4 zmm24, zmm24, ymmword ptr [rdi+rdx+4064], 0x01
        vmovdqu32 ymm25, ymmword ptr [rdi+rdx+480]
        vinserti64x4 zmm25, zmm25, ymmword ptr [rdi+rdx+4576], 0x01
        vpunpcklqdq zmm8, zmm24, zmm25
        vpunpckhqdq zmm9, zmm24, zmm25
        vmovdqu32 ymm24, ymmword ptr [rdi+rdx+992]
        vinserti64x4 zmm24, zmm24, ymmword ptr [rdi+rdx+5088], 0x01
        vmovdqu32 ymm25, ymmword ptr [rdi+rdx+1504]
        vinserti64x4 zmm25, zmm25, ymmword ptr [rdi+rdx+5600], 0x01
        vpunpcklqdq zmm10, zmm24, zmm25
        vpunpckhqdq zmm11, zmm24, zmm25
        prefetcht0 [rdi+rdx+128]
        prefetcht0 [rdi+rdx+4224]
        prefetcht0 [rdi+rdx+640]
        prefetcht0 [rdi+rdx+4736]
        prefetcht0 [rdi+rdx+1152]
        prefetcht0 [rdi+rdx+5248]
        prefetcht0 [rdi+rdx+1664]
        prefetcht0 [rdi+rdx+5760]
        vmovdqu32 ymm24, ymmword ptr [rdi+rdx+2016]
        vinserti64x4 zmm24, zmm24, ymmword ptr [rdi+rdx+6112], 0x01
        vmovdqu32 ymm25, ymmword ptr [rdi+rdx+2528]
        vinserti64x4 zmm25, zmm25, ymmword ptr [rdi+rdx+6624], 0x01
        vpunpcklqdq zmm12, zmm24, zmm25
        vpunpckhqdq zmm13, zmm24, zmm25
        vmovdqu32 ymm24, ymmword ptr [rdi+rdx+3040]
        vinserti64x4 zmm24, zmm24, ymmword ptr [rdi+rdx+7136], 0x01
        vmovdqu32 ymm25, ymmword ptr [rdi+rdx+3552]
        vinserti64x4 zmm25, zmm25, ymmword ptr [rdi+rdx+7648], 0x01
        vpunpcklqdq zmm14, zmm24, zmm25
        vpunpckhqdq zmm15, zmm24, zmm25
        prefetcht0 [rdi+rdx+2176]
        prefetcht0 [rdi+rdx+6272]
        prefetcht0 [rdi+rdx+2688]
        prefetcht0 [rdi+rdx+6784]
        prefetcht0 [rdi+rdx+3200]
        prefetcht0 [rdi+rdx+7296]
        prefetcht0 [rdi+rdx+3712]
        prefetcht0 [rdi+rdx+7808]
        vshufps zmm24, zmm8, zmm10, 136
        vshufps zmm30, zmm12, zmm14, 136
        vmovdqa32 zmm28, zmm24
        vpermt2d zmm24, zmm27, zmm30
        vpermt2d zmm28, zmm31, zmm30
        vshufps zmm25, zmm8, zmm10, 221
        vshufps zmm30, zmm12, zmm14, 221
        vmovdqa32 zmm29, zmm25
        vpermt2d zmm25, zmm27, zmm30
        vpermt2d zmm29, zmm31, zmm30
        vshufps zmm26, zmm9, zmm11, 136
        vshufps zmm8, zmm13, zmm15, 136
        vmovdqa32 zmm30, zmm26
        vpermt2d zmm26, zmm27, zmm8
        vpermt2d zmm30, zmm31, zmm8
        vshufps zmm8, zmm9, zmm11, 221
        vshufps zmm10, zmm13, zmm15, 221
        vpermi2d zmm27, zmm8, zmm10
        vpermi2d zmm31, zmm8, zmm10
        vpbroadcastd zmm8, dword ptr [.Lflock_contiguous_BLAKE3_IV_0+rip]
        vpbroadcastd zmm9, dword ptr [.Lflock_contiguous_BLAKE3_IV_1+rip]
        vpbroadcastd zmm10, dword ptr [.Lflock_contiguous_BLAKE3_IV_2+rip]
        vpbroadcastd zmm11, dword ptr [.Lflock_contiguous_BLAKE3_IV_3+rip]
        vpbroadcastd zmm14, dword ptr [.Lflock_contiguous_BLAKE3_BLOCK_LEN+rip]
        vpbroadcastd zmm15, dword ptr [rsp+0x22*0x4]
        vpaddd  zmm0, zmm0, zmm16
        vpaddd  zmm1, zmm1, zmm18
        vpaddd  zmm2, zmm2, zmm20
        vpaddd  zmm3, zmm3, zmm22
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm0, 16
        vprord  zmm13, zmm1, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm17
        vpaddd  zmm1, zmm1, zmm19
        vpaddd  zmm2, zmm2, zmm21
        vpaddd  zmm3, zmm3, zmm23
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm24
        vpaddd  zmm1, zmm1, zmm26
        vpaddd  zmm2, zmm2, zmm28
        vpaddd  zmm3, zmm3, zmm30
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm25
        vpaddd  zmm1, zmm1, zmm27
        vpaddd  zmm2, zmm2, zmm29
        vpaddd  zmm3, zmm3, zmm31
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpaddd  zmm0, zmm0, zmm18
        vpaddd  zmm1, zmm1, zmm19
        vpaddd  zmm2, zmm2, zmm23
        vpaddd  zmm3, zmm3, zmm20
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm22
        vpaddd  zmm1, zmm1, zmm26
        vpaddd  zmm2, zmm2, zmm16
        vpaddd  zmm3, zmm3, zmm29
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm17
        vpaddd  zmm1, zmm1, zmm28
        vpaddd  zmm2, zmm2, zmm25
        vpaddd  zmm3, zmm3, zmm31
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm27
        vpaddd  zmm1, zmm1, zmm21
        vpaddd  zmm2, zmm2, zmm30
        vpaddd  zmm3, zmm3, zmm24
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpaddd  zmm0, zmm0, zmm19
        vpaddd  zmm1, zmm1, zmm26
        vpaddd  zmm2, zmm2, zmm29
        vpaddd  zmm3, zmm3, zmm23
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm20
        vpaddd  zmm1, zmm1, zmm28
        vpaddd  zmm2, zmm2, zmm18
        vpaddd  zmm3, zmm3, zmm30
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm22
        vpaddd  zmm1, zmm1, zmm25
        vpaddd  zmm2, zmm2, zmm27
        vpaddd  zmm3, zmm3, zmm24
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm21
        vpaddd  zmm1, zmm1, zmm16
        vpaddd  zmm2, zmm2, zmm31
        vpaddd  zmm3, zmm3, zmm17
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpaddd  zmm0, zmm0, zmm26
        vpaddd  zmm1, zmm1, zmm28
        vpaddd  zmm2, zmm2, zmm30
        vpaddd  zmm3, zmm3, zmm29
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm23
        vpaddd  zmm1, zmm1, zmm25
        vpaddd  zmm2, zmm2, zmm19
        vpaddd  zmm3, zmm3, zmm31
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm20
        vpaddd  zmm1, zmm1, zmm27
        vpaddd  zmm2, zmm2, zmm21
        vpaddd  zmm3, zmm3, zmm17
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm16
        vpaddd  zmm1, zmm1, zmm18
        vpaddd  zmm2, zmm2, zmm24
        vpaddd  zmm3, zmm3, zmm22
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpaddd  zmm0, zmm0, zmm28
        vpaddd  zmm1, zmm1, zmm25
        vpaddd  zmm2, zmm2, zmm31
        vpaddd  zmm3, zmm3, zmm30
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm29
        vpaddd  zmm1, zmm1, zmm27
        vpaddd  zmm2, zmm2, zmm26
        vpaddd  zmm3, zmm3, zmm24
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm23
        vpaddd  zmm1, zmm1, zmm21
        vpaddd  zmm2, zmm2, zmm16
        vpaddd  zmm3, zmm3, zmm22
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm18
        vpaddd  zmm1, zmm1, zmm19
        vpaddd  zmm2, zmm2, zmm17
        vpaddd  zmm3, zmm3, zmm20
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpaddd  zmm0, zmm0, zmm25
        vpaddd  zmm1, zmm1, zmm27
        vpaddd  zmm2, zmm2, zmm24
        vpaddd  zmm3, zmm3, zmm31
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm30
        vpaddd  zmm1, zmm1, zmm21
        vpaddd  zmm2, zmm2, zmm28
        vpaddd  zmm3, zmm3, zmm17
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm29
        vpaddd  zmm1, zmm1, zmm16
        vpaddd  zmm2, zmm2, zmm18
        vpaddd  zmm3, zmm3, zmm20
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm19
        vpaddd  zmm1, zmm1, zmm26
        vpaddd  zmm2, zmm2, zmm22
        vpaddd  zmm3, zmm3, zmm23
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpaddd  zmm0, zmm0, zmm27
        vpaddd  zmm1, zmm1, zmm21
        vpaddd  zmm2, zmm2, zmm17
        vpaddd  zmm3, zmm3, zmm24
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm31
        vpaddd  zmm1, zmm1, zmm16
        vpaddd  zmm2, zmm2, zmm25
        vpaddd  zmm3, zmm3, zmm22
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm30
        vpaddd  zmm1, zmm1, zmm18
        vpaddd  zmm2, zmm2, zmm19
        vpaddd  zmm3, zmm3, zmm23
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm26
        vpaddd  zmm1, zmm1, zmm28
        vpaddd  zmm2, zmm2, zmm20
        vpaddd  zmm3, zmm3, zmm29
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpxord  zmm0, zmm0, zmm8
        vpxord  zmm1, zmm1, zmm9
        vpxord  zmm2, zmm2, zmm10
        vpxord  zmm3, zmm3, zmm11
        vpxord  zmm4, zmm4, zmm12
        vpxord  zmm5, zmm5, zmm13
        vpxord  zmm6, zmm6, zmm14
        vpxord  zmm7, zmm7, zmm15
        movzx   eax, byte ptr [rbp+0x38]
        jne     9b
        mov     rbx, qword ptr [rbp+0x50]
        test    r10d, r10d
        jnz     .Lflock_leaf_store_512
        vpunpckldq zmm16, zmm0, zmm1
        vpunpckhdq zmm17, zmm0, zmm1
        vpunpckldq zmm18, zmm2, zmm3
        vpunpckhdq zmm19, zmm2, zmm3
        vpunpckldq zmm20, zmm4, zmm5
        vpunpckhdq zmm21, zmm4, zmm5
        vpunpckldq zmm22, zmm6, zmm7
        vpunpckhdq zmm23, zmm6, zmm7
        vpunpcklqdq zmm0, zmm16, zmm18
        vpunpckhqdq zmm1, zmm16, zmm18
        vpunpcklqdq zmm2, zmm17, zmm19
        vpunpckhqdq zmm3, zmm17, zmm19
        vpunpcklqdq zmm4, zmm20, zmm22
        vpunpckhqdq zmm5, zmm20, zmm22
        vpunpcklqdq zmm6, zmm21, zmm23
        vpunpckhqdq zmm7, zmm21, zmm23
        vshufi32x4 zmm16, zmm0, zmm4, 0x88
        vshufi32x4 zmm17, zmm1, zmm5, 0x88
        vshufi32x4 zmm18, zmm2, zmm6, 0x88
        vshufi32x4 zmm19, zmm3, zmm7, 0x88
        vshufi32x4 zmm20, zmm0, zmm4, 0xDD
        vshufi32x4 zmm21, zmm1, zmm5, 0xDD
        vshufi32x4 zmm22, zmm2, zmm6, 0xDD
        vshufi32x4 zmm23, zmm3, zmm7, 0xDD
        vshufi32x4 zmm0, zmm16, zmm17, 0x88
        vshufi32x4 zmm1, zmm18, zmm19, 0x88
        vshufi32x4 zmm2, zmm20, zmm21, 0x88
        vshufi32x4 zmm3, zmm22, zmm23, 0x88
        vshufi32x4 zmm4, zmm16, zmm17, 0xDD
        vshufi32x4 zmm5, zmm18, zmm19, 0xDD
        vshufi32x4 zmm6, zmm20, zmm21, 0xDD
        vshufi32x4 zmm7, zmm22, zmm23, 0xDD
.Lflock_leaf_store_512:
        vmovdqu32 zmmword ptr [rbx], zmm0
        vmovdqu32 zmmword ptr [rbx+0x1*0x40], zmm1
        vmovdqu32 zmmword ptr [rbx+0x2*0x40], zmm2
        vmovdqu32 zmmword ptr [rbx+0x3*0x40], zmm3
        vmovdqu32 zmmword ptr [rbx+0x4*0x40], zmm4
        vmovdqu32 zmmword ptr [rbx+0x5*0x40], zmm5
        vmovdqu32 zmmword ptr [rbx+0x6*0x40], zmm6
        vmovdqu32 zmmword ptr [rbx+0x7*0x40], zmm7
        add     rdi, 8192
        add     rbx, 512
        mov     qword ptr [rbp+0x50], rbx
        sub     rsi, 16
        cmp     rsi, 16
        jnc     2b
4:
        vzeroupper
        mov     rsp, rbp
        pop     rbp
        pop     rbx
        pop     r12
        pop     r13
        pop     r14
        pop     r15
        ret
.size flock_blake3_contiguous_512, .-flock_blake3_contiguous_512
.size flock_blake3_planes_leaf_512, .-flock_blake3_planes_leaf_512
// R10 is call-clobbered and unused throughout the incumbent leaf body.
.p2align 6
.global flock_blake3_planes_leaf_1024
.hidden flock_blake3_planes_leaf_1024
.type flock_blake3_planes_leaf_1024, @function
flock_blake3_planes_leaf_1024:
        mov     r10d, 1
        jmp     .Lflock_leaf_shared_1024
.p2align 6
.global flock_blake3_contiguous_1024
.hidden flock_blake3_contiguous_1024
.type flock_blake3_contiguous_1024, @function
flock_blake3_contiguous_1024:
        xor     r10d, r10d
.Lflock_leaf_shared_1024:
        push    r15
        push    r14
        push    r13
        push    r12
        push    rbx
        push    rbp
        mov     rbp, rsp
        sub     rsp, 144
        and     rsp, 0xFFFFFFFFFFFFFFC0
2:
        vpbroadcastd zmm0, dword ptr [rcx]
        vpbroadcastd zmm1, dword ptr [rcx+0x1*0x4]
        vpbroadcastd zmm2, dword ptr [rcx+0x2*0x4]
        vpbroadcastd zmm3, dword ptr [rcx+0x3*0x4]
        vpbroadcastd zmm4, dword ptr [rcx+0x4*0x4]
        vpbroadcastd zmm5, dword ptr [rcx+0x5*0x4]
        vpbroadcastd zmm6, dword ptr [rcx+0x6*0x4]
        vpbroadcastd zmm7, dword ptr [rcx+0x7*0x4]
        movzx   eax, byte ptr [rbp+0x38]
        movzx   ebx, byte ptr [rbp+0x40]
        or      eax, ebx
        xor     edx, edx
.p2align 5
9:
        movzx   ebx, byte ptr [rbp+0x48]
        or      ebx, eax
        add     rdx, 64
        cmp     rdx, 1024
        cmove   eax, ebx
        mov     dword ptr [rsp+0x88], eax
        vmovdqu32 ymm16, ymmword ptr [rdi+rdx+-64]
        vinserti64x4 zmm16, zmm16, ymmword ptr [rdi+rdx+8128], 0x01
        vmovdqu32 ymm17, ymmword ptr [rdi+rdx+960]
        vinserti64x4 zmm17, zmm17, ymmword ptr [rdi+rdx+9152], 0x01
        vpunpcklqdq zmm8, zmm16, zmm17
        vpunpckhqdq zmm9, zmm16, zmm17
        vmovdqu32 ymm18, ymmword ptr [rdi+rdx+1984]
        vinserti64x4 zmm18, zmm18, ymmword ptr [rdi+rdx+10176], 0x01
        vmovdqu32 ymm19, ymmword ptr [rdi+rdx+3008]
        vinserti64x4 zmm19, zmm19, ymmword ptr [rdi+rdx+11200], 0x01
        vpunpcklqdq zmm10, zmm18, zmm19
        vpunpckhqdq zmm11, zmm18, zmm19
        vmovdqu32 ymm16, ymmword ptr [rdi+rdx+4032]
        vinserti64x4 zmm16, zmm16, ymmword ptr [rdi+rdx+12224], 0x01
        vmovdqu32 ymm17, ymmword ptr [rdi+rdx+5056]
        vinserti64x4 zmm17, zmm17, ymmword ptr [rdi+rdx+13248], 0x01
        vpunpcklqdq zmm12, zmm16, zmm17
        vpunpckhqdq zmm13, zmm16, zmm17
        vmovdqu32 ymm18, ymmword ptr [rdi+rdx+6080]
        vinserti64x4 zmm18, zmm18, ymmword ptr [rdi+rdx+14272], 0x01
        vmovdqu32 ymm19, ymmword ptr [rdi+rdx+7104]
        vinserti64x4 zmm19, zmm19, ymmword ptr [rdi+rdx+15296], 0x01
        vpunpcklqdq zmm14, zmm18, zmm19
        vpunpckhqdq zmm15, zmm18, zmm19
        vmovdqa32 zmm27, zmmword ptr [.Lflock_contiguous_INDEX0+rip]
        vmovdqa32 zmm31, zmmword ptr [.Lflock_contiguous_INDEX1+rip]
        vshufps zmm16, zmm8, zmm10, 136
        vshufps zmm17, zmm12, zmm14, 136
        vmovdqa32 zmm20, zmm16
        vpermt2d zmm16, zmm27, zmm17
        vpermt2d zmm20, zmm31, zmm17
        vshufps zmm17, zmm8, zmm10, 221
        vshufps zmm30, zmm12, zmm14, 221
        vmovdqa32 zmm21, zmm17
        vpermt2d zmm17, zmm27, zmm30
        vpermt2d zmm21, zmm31, zmm30
        vshufps zmm18, zmm9, zmm11, 136
        vshufps zmm8, zmm13, zmm15, 136
        vmovdqa32 zmm22, zmm18
        vpermt2d zmm18, zmm27, zmm8
        vpermt2d zmm22, zmm31, zmm8
        vshufps zmm19, zmm9, zmm11, 221
        vshufps zmm8, zmm13, zmm15, 221
        vmovdqa32 zmm23, zmm19
        vpermt2d zmm19, zmm27, zmm8
        vpermt2d zmm23, zmm31, zmm8
        vmovdqu32 ymm24, ymmword ptr [rdi+rdx+-32]
        vinserti64x4 zmm24, zmm24, ymmword ptr [rdi+rdx+8160], 0x01
        vmovdqu32 ymm25, ymmword ptr [rdi+rdx+992]
        vinserti64x4 zmm25, zmm25, ymmword ptr [rdi+rdx+9184], 0x01
        vpunpcklqdq zmm8, zmm24, zmm25
        vpunpckhqdq zmm9, zmm24, zmm25
        vmovdqu32 ymm24, ymmword ptr [rdi+rdx+2016]
        vinserti64x4 zmm24, zmm24, ymmword ptr [rdi+rdx+10208], 0x01
        vmovdqu32 ymm25, ymmword ptr [rdi+rdx+3040]
        vinserti64x4 zmm25, zmm25, ymmword ptr [rdi+rdx+11232], 0x01
        vpunpcklqdq zmm10, zmm24, zmm25
        vpunpckhqdq zmm11, zmm24, zmm25
        prefetcht0 [rdi+rdx+128]
        prefetcht0 [rdi+rdx+8320]
        prefetcht0 [rdi+rdx+1152]
        prefetcht0 [rdi+rdx+9344]
        prefetcht0 [rdi+rdx+2176]
        prefetcht0 [rdi+rdx+10368]
        prefetcht0 [rdi+rdx+3200]
        prefetcht0 [rdi+rdx+11392]
        vmovdqu32 ymm24, ymmword ptr [rdi+rdx+4064]
        vinserti64x4 zmm24, zmm24, ymmword ptr [rdi+rdx+12256], 0x01
        vmovdqu32 ymm25, ymmword ptr [rdi+rdx+5088]
        vinserti64x4 zmm25, zmm25, ymmword ptr [rdi+rdx+13280], 0x01
        vpunpcklqdq zmm12, zmm24, zmm25
        vpunpckhqdq zmm13, zmm24, zmm25
        vmovdqu32 ymm24, ymmword ptr [rdi+rdx+6112]
        vinserti64x4 zmm24, zmm24, ymmword ptr [rdi+rdx+14304], 0x01
        vmovdqu32 ymm25, ymmword ptr [rdi+rdx+7136]
        vinserti64x4 zmm25, zmm25, ymmword ptr [rdi+rdx+15328], 0x01
        vpunpcklqdq zmm14, zmm24, zmm25
        vpunpckhqdq zmm15, zmm24, zmm25
        prefetcht0 [rdi+rdx+4224]
        prefetcht0 [rdi+rdx+12416]
        prefetcht0 [rdi+rdx+5248]
        prefetcht0 [rdi+rdx+13440]
        prefetcht0 [rdi+rdx+6272]
        prefetcht0 [rdi+rdx+14464]
        prefetcht0 [rdi+rdx+7296]
        prefetcht0 [rdi+rdx+15488]
        vshufps zmm24, zmm8, zmm10, 136
        vshufps zmm30, zmm12, zmm14, 136
        vmovdqa32 zmm28, zmm24
        vpermt2d zmm24, zmm27, zmm30
        vpermt2d zmm28, zmm31, zmm30
        vshufps zmm25, zmm8, zmm10, 221
        vshufps zmm30, zmm12, zmm14, 221
        vmovdqa32 zmm29, zmm25
        vpermt2d zmm25, zmm27, zmm30
        vpermt2d zmm29, zmm31, zmm30
        vshufps zmm26, zmm9, zmm11, 136
        vshufps zmm8, zmm13, zmm15, 136
        vmovdqa32 zmm30, zmm26
        vpermt2d zmm26, zmm27, zmm8
        vpermt2d zmm30, zmm31, zmm8
        vshufps zmm8, zmm9, zmm11, 221
        vshufps zmm10, zmm13, zmm15, 221
        vpermi2d zmm27, zmm8, zmm10
        vpermi2d zmm31, zmm8, zmm10
        vpbroadcastd zmm8, dword ptr [.Lflock_contiguous_BLAKE3_IV_0+rip]
        vpbroadcastd zmm9, dword ptr [.Lflock_contiguous_BLAKE3_IV_1+rip]
        vpbroadcastd zmm10, dword ptr [.Lflock_contiguous_BLAKE3_IV_2+rip]
        vpbroadcastd zmm11, dword ptr [.Lflock_contiguous_BLAKE3_IV_3+rip]
        vpbroadcastd zmm14, dword ptr [.Lflock_contiguous_BLAKE3_BLOCK_LEN+rip]
        vpbroadcastd zmm15, dword ptr [rsp+0x22*0x4]
        vpaddd  zmm0, zmm0, zmm16
        vpaddd  zmm1, zmm1, zmm18
        vpaddd  zmm2, zmm2, zmm20
        vpaddd  zmm3, zmm3, zmm22
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm0, 16
        vprord  zmm13, zmm1, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm17
        vpaddd  zmm1, zmm1, zmm19
        vpaddd  zmm2, zmm2, zmm21
        vpaddd  zmm3, zmm3, zmm23
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm24
        vpaddd  zmm1, zmm1, zmm26
        vpaddd  zmm2, zmm2, zmm28
        vpaddd  zmm3, zmm3, zmm30
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm25
        vpaddd  zmm1, zmm1, zmm27
        vpaddd  zmm2, zmm2, zmm29
        vpaddd  zmm3, zmm3, zmm31
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpaddd  zmm0, zmm0, zmm18
        vpaddd  zmm1, zmm1, zmm19
        vpaddd  zmm2, zmm2, zmm23
        vpaddd  zmm3, zmm3, zmm20
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm22
        vpaddd  zmm1, zmm1, zmm26
        vpaddd  zmm2, zmm2, zmm16
        vpaddd  zmm3, zmm3, zmm29
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm17
        vpaddd  zmm1, zmm1, zmm28
        vpaddd  zmm2, zmm2, zmm25
        vpaddd  zmm3, zmm3, zmm31
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm27
        vpaddd  zmm1, zmm1, zmm21
        vpaddd  zmm2, zmm2, zmm30
        vpaddd  zmm3, zmm3, zmm24
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpaddd  zmm0, zmm0, zmm19
        vpaddd  zmm1, zmm1, zmm26
        vpaddd  zmm2, zmm2, zmm29
        vpaddd  zmm3, zmm3, zmm23
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm20
        vpaddd  zmm1, zmm1, zmm28
        vpaddd  zmm2, zmm2, zmm18
        vpaddd  zmm3, zmm3, zmm30
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm22
        vpaddd  zmm1, zmm1, zmm25
        vpaddd  zmm2, zmm2, zmm27
        vpaddd  zmm3, zmm3, zmm24
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm21
        vpaddd  zmm1, zmm1, zmm16
        vpaddd  zmm2, zmm2, zmm31
        vpaddd  zmm3, zmm3, zmm17
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpaddd  zmm0, zmm0, zmm26
        vpaddd  zmm1, zmm1, zmm28
        vpaddd  zmm2, zmm2, zmm30
        vpaddd  zmm3, zmm3, zmm29
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm23
        vpaddd  zmm1, zmm1, zmm25
        vpaddd  zmm2, zmm2, zmm19
        vpaddd  zmm3, zmm3, zmm31
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm20
        vpaddd  zmm1, zmm1, zmm27
        vpaddd  zmm2, zmm2, zmm21
        vpaddd  zmm3, zmm3, zmm17
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm16
        vpaddd  zmm1, zmm1, zmm18
        vpaddd  zmm2, zmm2, zmm24
        vpaddd  zmm3, zmm3, zmm22
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpaddd  zmm0, zmm0, zmm28
        vpaddd  zmm1, zmm1, zmm25
        vpaddd  zmm2, zmm2, zmm31
        vpaddd  zmm3, zmm3, zmm30
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm29
        vpaddd  zmm1, zmm1, zmm27
        vpaddd  zmm2, zmm2, zmm26
        vpaddd  zmm3, zmm3, zmm24
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm23
        vpaddd  zmm1, zmm1, zmm21
        vpaddd  zmm2, zmm2, zmm16
        vpaddd  zmm3, zmm3, zmm22
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm18
        vpaddd  zmm1, zmm1, zmm19
        vpaddd  zmm2, zmm2, zmm17
        vpaddd  zmm3, zmm3, zmm20
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpaddd  zmm0, zmm0, zmm25
        vpaddd  zmm1, zmm1, zmm27
        vpaddd  zmm2, zmm2, zmm24
        vpaddd  zmm3, zmm3, zmm31
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm30
        vpaddd  zmm1, zmm1, zmm21
        vpaddd  zmm2, zmm2, zmm28
        vpaddd  zmm3, zmm3, zmm17
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm29
        vpaddd  zmm1, zmm1, zmm16
        vpaddd  zmm2, zmm2, zmm18
        vpaddd  zmm3, zmm3, zmm20
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm19
        vpaddd  zmm1, zmm1, zmm26
        vpaddd  zmm2, zmm2, zmm22
        vpaddd  zmm3, zmm3, zmm23
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpaddd  zmm0, zmm0, zmm27
        vpaddd  zmm1, zmm1, zmm21
        vpaddd  zmm2, zmm2, zmm17
        vpaddd  zmm3, zmm3, zmm24
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm31
        vpaddd  zmm1, zmm1, zmm16
        vpaddd  zmm2, zmm2, zmm25
        vpaddd  zmm3, zmm3, zmm22
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm30
        vpaddd  zmm1, zmm1, zmm18
        vpaddd  zmm2, zmm2, zmm19
        vpaddd  zmm3, zmm3, zmm23
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm26
        vpaddd  zmm1, zmm1, zmm28
        vpaddd  zmm2, zmm2, zmm20
        vpaddd  zmm3, zmm3, zmm29
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpxord  zmm0, zmm0, zmm8
        vpxord  zmm1, zmm1, zmm9
        vpxord  zmm2, zmm2, zmm10
        vpxord  zmm3, zmm3, zmm11
        vpxord  zmm4, zmm4, zmm12
        vpxord  zmm5, zmm5, zmm13
        vpxord  zmm6, zmm6, zmm14
        vpxord  zmm7, zmm7, zmm15
        movzx   eax, byte ptr [rbp+0x38]
        jne     9b
        mov     rbx, qword ptr [rbp+0x50]
        test    r10d, r10d
        jnz     .Lflock_leaf_store_1024
        vpunpckldq zmm16, zmm0, zmm1
        vpunpckhdq zmm17, zmm0, zmm1
        vpunpckldq zmm18, zmm2, zmm3
        vpunpckhdq zmm19, zmm2, zmm3
        vpunpckldq zmm20, zmm4, zmm5
        vpunpckhdq zmm21, zmm4, zmm5
        vpunpckldq zmm22, zmm6, zmm7
        vpunpckhdq zmm23, zmm6, zmm7
        vpunpcklqdq zmm0, zmm16, zmm18
        vpunpckhqdq zmm1, zmm16, zmm18
        vpunpcklqdq zmm2, zmm17, zmm19
        vpunpckhqdq zmm3, zmm17, zmm19
        vpunpcklqdq zmm4, zmm20, zmm22
        vpunpckhqdq zmm5, zmm20, zmm22
        vpunpcklqdq zmm6, zmm21, zmm23
        vpunpckhqdq zmm7, zmm21, zmm23
        vshufi32x4 zmm16, zmm0, zmm4, 0x88
        vshufi32x4 zmm17, zmm1, zmm5, 0x88
        vshufi32x4 zmm18, zmm2, zmm6, 0x88
        vshufi32x4 zmm19, zmm3, zmm7, 0x88
        vshufi32x4 zmm20, zmm0, zmm4, 0xDD
        vshufi32x4 zmm21, zmm1, zmm5, 0xDD
        vshufi32x4 zmm22, zmm2, zmm6, 0xDD
        vshufi32x4 zmm23, zmm3, zmm7, 0xDD
        vshufi32x4 zmm0, zmm16, zmm17, 0x88
        vshufi32x4 zmm1, zmm18, zmm19, 0x88
        vshufi32x4 zmm2, zmm20, zmm21, 0x88
        vshufi32x4 zmm3, zmm22, zmm23, 0x88
        vshufi32x4 zmm4, zmm16, zmm17, 0xDD
        vshufi32x4 zmm5, zmm18, zmm19, 0xDD
        vshufi32x4 zmm6, zmm20, zmm21, 0xDD
        vshufi32x4 zmm7, zmm22, zmm23, 0xDD
.Lflock_leaf_store_1024:
        vmovdqu32 zmmword ptr [rbx], zmm0
        vmovdqu32 zmmword ptr [rbx+0x1*0x40], zmm1
        vmovdqu32 zmmword ptr [rbx+0x2*0x40], zmm2
        vmovdqu32 zmmword ptr [rbx+0x3*0x40], zmm3
        vmovdqu32 zmmword ptr [rbx+0x4*0x40], zmm4
        vmovdqu32 zmmword ptr [rbx+0x5*0x40], zmm5
        vmovdqu32 zmmword ptr [rbx+0x6*0x40], zmm6
        vmovdqu32 zmmword ptr [rbx+0x7*0x40], zmm7
        add     rdi, 16384
        add     rbx, 512
        mov     qword ptr [rbp+0x50], rbx
        sub     rsi, 16
        cmp     rsi, 16
        jnc     2b
4:
        vzeroupper
        mov     rsp, rbp
        pop     rbp
        pop     rbx
        pop     r12
        pop     r13
        pop     r14
        pop     r15
        ret
.size flock_blake3_contiguous_1024, .-flock_blake3_contiguous_1024
.size flock_blake3_planes_leaf_1024, .-flock_blake3_planes_leaf_1024

.p2align 6
.global flock_blake3_planes_parent
.hidden flock_blake3_planes_parent
.type flock_blake3_planes_parent, @function
flock_blake3_planes_parent:
        push    r15
        push    r14
        push    r13
        push    r12
        push    rbx
        push    rbp
        mov     rbp, rsp
        sub     rsp, 144
        and     rsp, 0xFFFFFFFFFFFFFFC0
2:
        vpbroadcastd zmm0, dword ptr [rcx]
        vpbroadcastd zmm1, dword ptr [rcx+0x1*0x4]
        vpbroadcastd zmm2, dword ptr [rcx+0x2*0x4]
        vpbroadcastd zmm3, dword ptr [rcx+0x3*0x4]
        vpbroadcastd zmm4, dword ptr [rcx+0x4*0x4]
        vpbroadcastd zmm5, dword ptr [rcx+0x5*0x4]
        vpbroadcastd zmm6, dword ptr [rcx+0x6*0x4]
        vpbroadcastd zmm7, dword ptr [rcx+0x7*0x4]
        movzx   eax, byte ptr [rbp+0x38]
        movzx   ebx, byte ptr [rbp+0x40]
        or      eax, ebx
        xor     edx, edx
.p2align 5
9:
        movzx   ebx, byte ptr [rbp+0x48]
        or      ebx, eax
        add     rdx, 64
        cmp     rdx, 64
        cmove   eax, ebx
        mov     dword ptr [rsp+0x88], eax
        vmovdqa32 zmm8, zmmword ptr [.Lflock_planes_EVEN+rip]
        vmovdqa32 zmm9, zmmword ptr [.Lflock_planes_ODD+rip]
        vmovdqu32 zmm16, zmmword ptr [rdi+0]
        vmovdqu32 zmm10, zmmword ptr [rdi+512]
        vmovdqa32 zmm24, zmm16
        vpermt2d zmm16, zmm8, zmm10
        vpermt2d zmm24, zmm9, zmm10
        vmovdqu32 zmm17, zmmword ptr [rdi+64]
        vmovdqu32 zmm10, zmmword ptr [rdi+576]
        vmovdqa32 zmm25, zmm17
        vpermt2d zmm17, zmm8, zmm10
        vpermt2d zmm25, zmm9, zmm10
        vmovdqu32 zmm18, zmmword ptr [rdi+128]
        vmovdqu32 zmm10, zmmword ptr [rdi+640]
        vmovdqa32 zmm26, zmm18
        vpermt2d zmm18, zmm8, zmm10
        vpermt2d zmm26, zmm9, zmm10
        vmovdqu32 zmm19, zmmword ptr [rdi+192]
        vmovdqu32 zmm10, zmmword ptr [rdi+704]
        vmovdqa32 zmm27, zmm19
        vpermt2d zmm19, zmm8, zmm10
        vpermt2d zmm27, zmm9, zmm10
        vmovdqu32 zmm20, zmmword ptr [rdi+256]
        vmovdqu32 zmm10, zmmword ptr [rdi+768]
        vmovdqa32 zmm28, zmm20
        vpermt2d zmm20, zmm8, zmm10
        vpermt2d zmm28, zmm9, zmm10
        vmovdqu32 zmm21, zmmword ptr [rdi+320]
        vmovdqu32 zmm10, zmmword ptr [rdi+832]
        vmovdqa32 zmm29, zmm21
        vpermt2d zmm21, zmm8, zmm10
        vpermt2d zmm29, zmm9, zmm10
        vmovdqu32 zmm22, zmmword ptr [rdi+384]
        vmovdqu32 zmm10, zmmword ptr [rdi+896]
        vmovdqa32 zmm30, zmm22
        vpermt2d zmm22, zmm8, zmm10
        vpermt2d zmm30, zmm9, zmm10
        vmovdqu32 zmm23, zmmword ptr [rdi+448]
        vmovdqu32 zmm10, zmmword ptr [rdi+960]
        vmovdqa32 zmm31, zmm23
        vpermt2d zmm23, zmm8, zmm10
        vpermt2d zmm31, zmm9, zmm10
        vpbroadcastd zmm8, dword ptr [.Lflock_contiguous_BLAKE3_IV_0+rip]
        vpbroadcastd zmm9, dword ptr [.Lflock_contiguous_BLAKE3_IV_1+rip]
        vpbroadcastd zmm10, dword ptr [.Lflock_contiguous_BLAKE3_IV_2+rip]
        vpbroadcastd zmm11, dword ptr [.Lflock_contiguous_BLAKE3_IV_3+rip]
        vpbroadcastd zmm14, dword ptr [.Lflock_contiguous_BLAKE3_BLOCK_LEN+rip]
        vpbroadcastd zmm15, dword ptr [rsp+0x22*0x4]
        vpaddd  zmm0, zmm0, zmm16
        vpaddd  zmm1, zmm1, zmm18
        vpaddd  zmm2, zmm2, zmm20
        vpaddd  zmm3, zmm3, zmm22
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm0, 16
        vprord  zmm13, zmm1, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm17
        vpaddd  zmm1, zmm1, zmm19
        vpaddd  zmm2, zmm2, zmm21
        vpaddd  zmm3, zmm3, zmm23
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm24
        vpaddd  zmm1, zmm1, zmm26
        vpaddd  zmm2, zmm2, zmm28
        vpaddd  zmm3, zmm3, zmm30
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm25
        vpaddd  zmm1, zmm1, zmm27
        vpaddd  zmm2, zmm2, zmm29
        vpaddd  zmm3, zmm3, zmm31
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpaddd  zmm0, zmm0, zmm18
        vpaddd  zmm1, zmm1, zmm19
        vpaddd  zmm2, zmm2, zmm23
        vpaddd  zmm3, zmm3, zmm20
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm22
        vpaddd  zmm1, zmm1, zmm26
        vpaddd  zmm2, zmm2, zmm16
        vpaddd  zmm3, zmm3, zmm29
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm17
        vpaddd  zmm1, zmm1, zmm28
        vpaddd  zmm2, zmm2, zmm25
        vpaddd  zmm3, zmm3, zmm31
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm27
        vpaddd  zmm1, zmm1, zmm21
        vpaddd  zmm2, zmm2, zmm30
        vpaddd  zmm3, zmm3, zmm24
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpaddd  zmm0, zmm0, zmm19
        vpaddd  zmm1, zmm1, zmm26
        vpaddd  zmm2, zmm2, zmm29
        vpaddd  zmm3, zmm3, zmm23
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm20
        vpaddd  zmm1, zmm1, zmm28
        vpaddd  zmm2, zmm2, zmm18
        vpaddd  zmm3, zmm3, zmm30
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm22
        vpaddd  zmm1, zmm1, zmm25
        vpaddd  zmm2, zmm2, zmm27
        vpaddd  zmm3, zmm3, zmm24
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm21
        vpaddd  zmm1, zmm1, zmm16
        vpaddd  zmm2, zmm2, zmm31
        vpaddd  zmm3, zmm3, zmm17
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpaddd  zmm0, zmm0, zmm26
        vpaddd  zmm1, zmm1, zmm28
        vpaddd  zmm2, zmm2, zmm30
        vpaddd  zmm3, zmm3, zmm29
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm23
        vpaddd  zmm1, zmm1, zmm25
        vpaddd  zmm2, zmm2, zmm19
        vpaddd  zmm3, zmm3, zmm31
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm20
        vpaddd  zmm1, zmm1, zmm27
        vpaddd  zmm2, zmm2, zmm21
        vpaddd  zmm3, zmm3, zmm17
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm16
        vpaddd  zmm1, zmm1, zmm18
        vpaddd  zmm2, zmm2, zmm24
        vpaddd  zmm3, zmm3, zmm22
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpaddd  zmm0, zmm0, zmm28
        vpaddd  zmm1, zmm1, zmm25
        vpaddd  zmm2, zmm2, zmm31
        vpaddd  zmm3, zmm3, zmm30
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm29
        vpaddd  zmm1, zmm1, zmm27
        vpaddd  zmm2, zmm2, zmm26
        vpaddd  zmm3, zmm3, zmm24
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm23
        vpaddd  zmm1, zmm1, zmm21
        vpaddd  zmm2, zmm2, zmm16
        vpaddd  zmm3, zmm3, zmm22
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm18
        vpaddd  zmm1, zmm1, zmm19
        vpaddd  zmm2, zmm2, zmm17
        vpaddd  zmm3, zmm3, zmm20
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpaddd  zmm0, zmm0, zmm25
        vpaddd  zmm1, zmm1, zmm27
        vpaddd  zmm2, zmm2, zmm24
        vpaddd  zmm3, zmm3, zmm31
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm30
        vpaddd  zmm1, zmm1, zmm21
        vpaddd  zmm2, zmm2, zmm28
        vpaddd  zmm3, zmm3, zmm17
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm29
        vpaddd  zmm1, zmm1, zmm16
        vpaddd  zmm2, zmm2, zmm18
        vpaddd  zmm3, zmm3, zmm20
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm19
        vpaddd  zmm1, zmm1, zmm26
        vpaddd  zmm2, zmm2, zmm22
        vpaddd  zmm3, zmm3, zmm23
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpaddd  zmm0, zmm0, zmm27
        vpaddd  zmm1, zmm1, zmm21
        vpaddd  zmm2, zmm2, zmm17
        vpaddd  zmm3, zmm3, zmm24
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vprord  zmm15, zmm15, 16
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 12
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vpaddd  zmm0, zmm0, zmm31
        vpaddd  zmm1, zmm1, zmm16
        vpaddd  zmm2, zmm2, zmm25
        vpaddd  zmm3, zmm3, zmm22
        vpaddd  zmm0, zmm0, zmm4
        vpaddd  zmm1, zmm1, zmm5
        vpaddd  zmm2, zmm2, zmm6
        vpaddd  zmm3, zmm3, zmm7
        vpxord  zmm12, zmm12, zmm0
        vpxord  zmm13, zmm13, zmm1
        vpxord  zmm14, zmm14, zmm2
        vpxord  zmm15, zmm15, zmm3
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vprord  zmm15, zmm15, 8
        vpaddd  zmm8, zmm8, zmm12
        vpaddd  zmm9, zmm9, zmm13
        vpaddd  zmm10, zmm10, zmm14
        vpaddd  zmm11, zmm11, zmm15
        vpxord  zmm4, zmm4, zmm8
        vpxord  zmm5, zmm5, zmm9
        vpxord  zmm6, zmm6, zmm10
        vpxord  zmm7, zmm7, zmm11
        vprord  zmm4, zmm4, 7
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vpaddd  zmm0, zmm0, zmm30
        vpaddd  zmm1, zmm1, zmm18
        vpaddd  zmm2, zmm2, zmm19
        vpaddd  zmm3, zmm3, zmm23
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 16
        vprord  zmm12, zmm12, 16
        vprord  zmm13, zmm13, 16
        vprord  zmm14, zmm14, 16
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 12
        vprord  zmm6, zmm6, 12
        vprord  zmm7, zmm7, 12
        vprord  zmm4, zmm4, 12
        vpaddd  zmm0, zmm0, zmm26
        vpaddd  zmm1, zmm1, zmm28
        vpaddd  zmm2, zmm2, zmm20
        vpaddd  zmm3, zmm3, zmm29
        vpaddd  zmm0, zmm0, zmm5
        vpaddd  zmm1, zmm1, zmm6
        vpaddd  zmm2, zmm2, zmm7
        vpaddd  zmm3, zmm3, zmm4
        vpxord  zmm15, zmm15, zmm0
        vpxord  zmm12, zmm12, zmm1
        vpxord  zmm13, zmm13, zmm2
        vpxord  zmm14, zmm14, zmm3
        vprord  zmm15, zmm15, 8
        vprord  zmm12, zmm12, 8
        vprord  zmm13, zmm13, 8
        vprord  zmm14, zmm14, 8
        vpaddd  zmm10, zmm10, zmm15
        vpaddd  zmm11, zmm11, zmm12
        vpaddd  zmm8, zmm8, zmm13
        vpaddd  zmm9, zmm9, zmm14
        vpxord  zmm5, zmm5, zmm10
        vpxord  zmm6, zmm6, zmm11
        vpxord  zmm7, zmm7, zmm8
        vpxord  zmm4, zmm4, zmm9
        vprord  zmm5, zmm5, 7
        vprord  zmm6, zmm6, 7
        vprord  zmm7, zmm7, 7
        vprord  zmm4, zmm4, 7
        vpxord  zmm0, zmm0, zmm8
        vpxord  zmm1, zmm1, zmm9
        vpxord  zmm2, zmm2, zmm10
        vpxord  zmm3, zmm3, zmm11
        vpxord  zmm4, zmm4, zmm12
        vpxord  zmm5, zmm5, zmm13
        vpxord  zmm6, zmm6, zmm14
        vpxord  zmm7, zmm7, zmm15
        movzx   eax, byte ptr [rbp+0x38]
        jne     9b
        mov     rbx, qword ptr [rbp+0x50]
        vpunpckldq zmm16, zmm0, zmm1
        vpunpckhdq zmm17, zmm0, zmm1
        vpunpckldq zmm18, zmm2, zmm3
        vpunpckhdq zmm19, zmm2, zmm3
        vpunpckldq zmm20, zmm4, zmm5
        vpunpckhdq zmm21, zmm4, zmm5
        vpunpckldq zmm22, zmm6, zmm7
        vpunpckhdq zmm23, zmm6, zmm7
        vpunpcklqdq zmm0, zmm16, zmm18
        vpunpckhqdq zmm1, zmm16, zmm18
        vpunpcklqdq zmm2, zmm17, zmm19
        vpunpckhqdq zmm3, zmm17, zmm19
        vpunpcklqdq zmm4, zmm20, zmm22
        vpunpckhqdq zmm5, zmm20, zmm22
        vpunpcklqdq zmm6, zmm21, zmm23
        vpunpckhqdq zmm7, zmm21, zmm23
        vshufi32x4 zmm16, zmm0, zmm4, 0x88
        vshufi32x4 zmm17, zmm1, zmm5, 0x88
        vshufi32x4 zmm18, zmm2, zmm6, 0x88
        vshufi32x4 zmm19, zmm3, zmm7, 0x88
        vshufi32x4 zmm20, zmm0, zmm4, 0xDD
        vshufi32x4 zmm21, zmm1, zmm5, 0xDD
        vshufi32x4 zmm22, zmm2, zmm6, 0xDD
        vshufi32x4 zmm23, zmm3, zmm7, 0xDD
        vshufi32x4 zmm0, zmm16, zmm17, 0x88
        vshufi32x4 zmm1, zmm18, zmm19, 0x88
        vshufi32x4 zmm2, zmm20, zmm21, 0x88
        vshufi32x4 zmm3, zmm22, zmm23, 0x88
        vshufi32x4 zmm4, zmm16, zmm17, 0xDD
        vshufi32x4 zmm5, zmm18, zmm19, 0xDD
        vshufi32x4 zmm6, zmm20, zmm21, 0xDD
        vshufi32x4 zmm7, zmm22, zmm23, 0xDD
        vmovdqu32 zmmword ptr [rbx], zmm0
        vmovdqu32 zmmword ptr [rbx+0x1*0x40], zmm1
        vmovdqu32 zmmword ptr [rbx+0x2*0x40], zmm2
        vmovdqu32 zmmword ptr [rbx+0x3*0x40], zmm3
        vmovdqu32 zmmword ptr [rbx+0x4*0x40], zmm4
        vmovdqu32 zmmword ptr [rbx+0x5*0x40], zmm5
        vmovdqu32 zmmword ptr [rbx+0x6*0x40], zmm6
        vmovdqu32 zmmword ptr [rbx+0x7*0x40], zmm7
        add     rdi, 1024
        add     rbx, 512
        mov     qword ptr [rbp+0x50], rbx
        sub     rsi, 16
        cmp     rsi, 16
        jnc     2b
4:
        vzeroupper
        mov     rsp, rbp
        pop     rbp
        pop     rbx
        pop     r12
        pop     r13
        pop     r14
        pop     r15
        ret
.size flock_blake3_planes_parent, .-flock_blake3_planes_parent

.section .rodata
.p2align 6
.Lflock_planes_EVEN:
        .long 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30
.Lflock_planes_ODD:
        .long 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31
.Lflock_contiguous_INDEX0:
        .long    0,  1,  2,  3, 16, 17, 18, 19
        .long    8,  9, 10, 11, 24, 25, 26, 27
.Lflock_contiguous_INDEX1:
        .long    4,  5,  6,  7, 20, 21, 22, 23
        .long   12, 13, 14, 15, 28, 29, 30, 31
.Lflock_contiguous_BLAKE3_BLOCK_LEN: .long 64
.Lflock_contiguous_BLAKE3_IV_0: .long 0x6A09E667
.Lflock_contiguous_BLAKE3_IV_1: .long 0xBB67AE85
.Lflock_contiguous_BLAKE3_IV_2: .long 0x3C6EF372
.Lflock_contiguous_BLAKE3_IV_3: .long 0xA54FF53A

.section .note.GNU-stack,"",@progbits
