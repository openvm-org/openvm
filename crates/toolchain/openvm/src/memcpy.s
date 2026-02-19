// Original source: https://elixir.bootlin.com/glibc/glibc-2.42.9000/source/string/memcpy.c
//
// This was compiled into assembly with:
//
// clang-14 -target riscv64 -march=rv64im -O3 -S memcpy.c -nostdlib -fno-builtin -funroll-loops
//
// and labels manually updated to not conflict.
	.text
	.attribute	4, 16
	.attribute	5, "rv64im"
	.file	"glibc_memcpy_no_includes.c"
	.globl	memcpy
	.p2align	2
	.type	memcpy,@function
memcpy:
	li	a3, 16
	bltu	a2, a3, .LBBmemcpy0_8
	negw	a3, a0
	andi	a4, a3, 7
	sub	a2, a2, a4
	mv	a3, a0
	beqz	a4, .LBBmemcpy0_4
	mv	a3, a0
.LBBmemcpy0_3:
	lb	a5, 0(a1)
	addi	a1, a1, 1
	addi	a4, a4, -1
	sb	a5, 0(a3)
	addi	a3, a3, 1
	bnez	a4, .LBBmemcpy0_3
.LBBmemcpy0_4:
	andi	a5, a1, 7
	srli	a4, a2, 3
	beqz	a5, .LBBmemcpy0_9
	slliw	a5, a1, 3
	andi	a6, a5, 56
	li	a5, 64
	sub	a7, a5, a6
	andi	a5, a4, 3
	li	t1, 1
	andi	t0, a1, -8
	blt	t1, a5, .LBBmemcpy0_11
	bnez	a5, .LBBmemcpy0_13
	ld	t1, 0(t0)
	ld	t3, 8(t0)
	addi	t0, t0, 8
	addi	a5, a3, -8
	j	.LBBmemcpy0_33
.LBBmemcpy0_8:
	mv	a3, a0
	bnez	a2, .LBBmemcpy0_38
	j	.LBBmemcpy0_39
.LBBmemcpy0_9:
	andi	a5, a4, 7
	slli	a5, a5, 2
	lui	a6, %hi(.LJTI0_0)
	addi	a6, a6, %lo(.LJTI0_0)
	add	a5, a5, a6
	lw	a5, 0(a5)
	jr	a5
.LBBmemcpy0_10:
	ld	t0, 0(a1)
	addi	a5, a3, -8
	mv	a6, a1
	j	.LBBmemcpy0_25
.LBBmemcpy0_11:
	li	t1, 3
	bne	a5, t1, .LBBmemcpy0_15
	ld	t3, 0(t0)
	ld	t4, 8(t0)
	addi	a5, a3, -16
	addi	a4, a4, 1
	j	.LBBmemcpy0_34
.LBBmemcpy0_13:
	ld	t2, 0(t0)
	ld	t1, 8(t0)
	addi	a4, a4, -1
	beqz	a4, .LBBmemcpy0_16
	addi	t0, t0, 16
	mv	a5, a3
	j	.LBBmemcpy0_32
.LBBmemcpy0_15:
	ld	t4, 0(t0)
	ld	t2, 8(t0)
	addi	t0, t0, -8
	addi	a5, a3, -24
	addi	a4, a4, 2
	j	.LBBmemcpy0_35
.LBBmemcpy0_16:
	mv	a5, a3
	j	.LBBmemcpy0_36
.LBBmemcpy0_17:
	ld	a7, 0(a1)
	addi	a4, a4, -1
	addi	a6, a1, 8
	mv	a5, a3
	bnez	a4, .LBBmemcpy0_24
	j	.LBBmemcpy0_37
.LBBmemcpy0_18:
	ld	t0, 0(a1)
	addi	a6, a1, -48
	addi	a5, a3, -56
	addi	a4, a4, 6
	j	.LBBmemcpy0_31
.LBBmemcpy0_19:
	ld	a7, 0(a1)
	addi	a6, a1, -40
	addi	a5, a3, -48
	addi	a4, a4, 5
	j	.LBBmemcpy0_30
.LBBmemcpy0_20:
	ld	t0, 0(a1)
	addi	a6, a1, -32
	addi	a5, a3, -40
	addi	a4, a4, 4
	j	.LBBmemcpy0_29
.LBBmemcpy0_21:
	ld	a7, 0(a1)
	addi	a6, a1, -24
	addi	a5, a3, -32
	addi	a4, a4, 3
	j	.LBBmemcpy0_28
.LBBmemcpy0_22:
	ld	t0, 0(a1)
	addi	a6, a1, -16
	addi	a5, a3, -24
	addi	a4, a4, 2
	j	.LBBmemcpy0_27
.LBBmemcpy0_23:
	ld	a7, 0(a1)
	addi	a6, a1, -8
	addi	a5, a3, -16
	addi	a4, a4, 1
	j	.LBBmemcpy0_26
.LBBmemcpy0_24:
	ld	t0, 0(a6)
	sd	a7, 0(a5)
.LBBmemcpy0_25:
	ld	a7, 8(a6)
	sd	t0, 8(a5)
.LBBmemcpy0_26:
	ld	t0, 16(a6)
	sd	a7, 16(a5)
.LBBmemcpy0_27:
	ld	a7, 24(a6)
	sd	t0, 24(a5)
.LBBmemcpy0_28:
	ld	t0, 32(a6)
	sd	a7, 32(a5)
.LBBmemcpy0_29:
	ld	a7, 40(a6)
	sd	t0, 40(a5)
.LBBmemcpy0_30:
	ld	t0, 48(a6)
	sd	a7, 48(a5)
.LBBmemcpy0_31:
	ld	a7, 56(a6)
	sd	t0, 56(a5)
	addi	a6, a6, 64
	addi	a4, a4, -8
	addi	a5, a5, 64
	bnez	a4, .LBBmemcpy0_24
	j	.LBBmemcpy0_37
.LBBmemcpy0_32:
	ld	t3, 0(t0)
	srl	t2, t2, a6
	sll	t4, t1, a7
	or	t2, t4, t2
	sd	t2, 0(a5)
.LBBmemcpy0_33:
	ld	t4, 8(t0)
	srl	t1, t1, a6
	sll	t2, t3, a7
	or	t1, t1, t2
	sd	t1, 8(a5)
.LBBmemcpy0_34:
	ld	t2, 16(t0)
	srl	t1, t3, a6
	sll	t3, t4, a7
	or	t1, t3, t1
	sd	t1, 16(a5)
.LBBmemcpy0_35:
	ld	t1, 24(t0)
	srl	t3, t4, a6
	sll	t4, t2, a7
	or	t3, t4, t3
	sd	t3, 24(a5)
	addi	t0, t0, 32
	addi	a4, a4, -4
	addi	a5, a5, 32
	bnez	a4, .LBBmemcpy0_32
.LBBmemcpy0_36:
	srl	a4, t2, a6
	sll	a6, t1, a7
	or	a7, a6, a4
.LBBmemcpy0_37:
	sd	a7, 0(a5)
	andi	a4, a2, -8
	add	a1, a1, a4
	add	a3, a3, a4
	andi	a2, a2, 7
	beqz	a2, .LBBmemcpy0_39
.LBBmemcpy0_38:
	lb	a4, 0(a1)
	addi	a1, a1, 1
	addi	a2, a2, -1
	sb	a4, 0(a3)
	addi	a3, a3, 1
	bnez	a2, .LBBmemcpy0_38
.LBBmemcpy0_39:
	ret
.Lmemcpyfunc_end0:
	.size	memcpy, .Lmemcpyfunc_end0-memcpy
	.section	.rodata,"a",@progbits
	.p2align	2
.LJTI0_0:
	.word	.LBBmemcpy0_10
	.word	.LBBmemcpy0_17
	.word	.LBBmemcpy0_18
	.word	.LBBmemcpy0_19
	.word	.LBBmemcpy0_20
	.word	.LBBmemcpy0_21
	.word	.LBBmemcpy0_22
	.word	.LBBmemcpy0_23

	.ident	"Homebrew clang version 14.0.6"
	.section	".note.GNU-stack","",@progbits
	.addrsig
