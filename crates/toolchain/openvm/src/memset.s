// Original source: https://elixir.bootlin.com/glibc/glibc-2.42.9000/source/string/memset.c
//
// This was compiled into assembly with:
//
// clang-14 -target riscv64 -march=rv64im -O3 -S memcpy.c -nostdlib -fno-builtin -funroll-loops
//
// and labels manually updated to not conflict.
	.text
	.attribute	4, 16
	.attribute	5, "rv64im"
	.file	"glibc_memset_no_includes.c"
	.globl	memset
	.p2align	2
	.type	memset,@function
memset:
	li	a3, 8
	bltu	a2, a3, .LBBmemset0_5
	andi	a3, a1, 255
	lui	a4, 4112
	addiw	a4, a4, 257
	mul	a4, a3, a4
	andi	a3, a0, 7
	slli	a5, a4, 32
	beqz	a3, .LBBmemset0_6
	mv	a3, a0
.LBBmemset0_3:
	sb	a1, 0(a3)
	addi	a3, a3, 1
	andi	a6, a3, 7
	addi	a2, a2, -1
	bnez	a6, .LBBmemset0_3
	li	a6, 64
	or	a4, a5, a4
	bgeu	a2, a6, .LBBmemset0_7
	j	.LBBmemset0_9
.LBBmemset0_5:
	mv	a3, a0
	bnez	a2, .LBBmemset0_12
	j	.LBBmemset0_13
.LBBmemset0_6:
	mv	a3, a0
	li	a6, 64
	or	a4, a5, a4
	bltu	a2, a6, .LBBmemset0_9
.LBBmemset0_7:
	srli	a5, a2, 6
.LBBmemset0_8:
	sd	a4, 0(a3)
	sd	a4, 8(a3)
	sd	a4, 16(a3)
	sd	a4, 24(a3)
	sd	a4, 32(a3)
	sd	a4, 40(a3)
	sd	a4, 48(a3)
	sd	a4, 56(a3)
	addi	a5, a5, -1
	addi	a3, a3, 64
	bnez	a5, .LBBmemset0_8
.LBBmemset0_9:
	srli	a5, a2, 3
	andi	a5, a5, 7
	beqz	a5, .LBBmemset0_11
.LBBmemset0_10:
	sd	a4, 0(a3)
	addi	a5, a5, -1
	addi	a3, a3, 8
	bnez	a5, .LBBmemset0_10
.LBBmemset0_11:
	andi	a2, a2, 7
	beqz	a2, .LBBmemset0_13
.LBBmemset0_12:
	sb	a1, 0(a3)
	addi	a2, a2, -1
	addi	a3, a3, 1
	bnez	a2, .LBBmemset0_12
.LBBmemset0_13:
	ret
.Lmemsetfunc_end0:
	.size	memset, .Lmemsetfunc_end0-memset

	.ident	"Homebrew clang version 14.0.6"
	.section	".note.GNU-stack","",@progbits
	.addrsig
