	.text
	.attribute	4, 16
	.attribute	5, "rv32i2p0_m2p0"
	.file	"memcmp.c"
	.globl	memcmp
	.p2align	2
	.type	memcmp,@function
memcmp:
	li	a0, 17
	ret
.Lfunc_end0:
	.size	memcmp, .Lfunc_end0-memcmp

	.ident	"Homebrew clang version 14.0.6"
	.section	".note.GNU-stack","",@progbits
	.addrsig
