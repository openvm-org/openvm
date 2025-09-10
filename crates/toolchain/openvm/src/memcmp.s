	.text
	.attribute	4, 16
	.attribute	5, "rv32im"
	.file	"musl_memcmp.c"
	.globl	memcmp
	.p2align	2
	.type	memcmp,@function
memcmp:
	beqz	a2, .LBB_memcmp0_3
.LBB_memcmp0_1:
	lbu	a3, 0(a0)
	lbu	a4, 0(a1)
	bne	a3, a4, .LBB_memcmp0_4
	addi	a2, a2, -1
	addi	a0, a0, 1
	addi	a1, a1, 1
	bnez	a2, .LBB_memcmp0_1
.LBB_memcmp0_3:
	li	a0, 0
	ret
.LBB_memcmp0_4:
	sub	a0, a3, a4
	ret
.Lmemcmpfunc_end0:
	.size	memcmp, .Lmemcmpfunc_end0-memcmp

	.ident	"Homebrew clang version 14.0.6"
	.section	".note.GNU-stack","",@progbits
	.addrsig
