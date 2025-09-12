// This was compiled into assembly with from glibc's memcmp implementation at https://elixir.bootlin.com/glibc/glibc-2.42.9000/source/string/memcmp.c: 
// 
// clang-14 -target riscv32 -march=rv32im -O2 -fno-builtin -ffreestanding \
//  	-D"size_t=unsigned" \
//  	-D"__THROW=" \
//		-D"op_t=unsigned long" \
//		-D"OPSIZ=4" \
//		-D"OP_T_THRES=16" \
// 		-D"libc_hidden_builtin_def(x)=" \
//		-D"libc_hidden_def(x)=" \
//		-D"strong_alias(x,y)=" \
//		-D"weak_alias(x,y)=" \
//		-D"__BYTE_ORDER=1234" \
//		-D"__LITTLE_ENDIAN=1234" \
//		-S glibc_memcmp_no_includes.c -o src/memcmp.s
// 
// Then labels were renamed to not conflict with these steps:
// * Replace "rv32i2p0_m2p0" to rv32im
// * Replace all .LBB0_X instances and replace to .LBBmemcmp0_X
// * Replace .Lfunc_end0 to .Lmemcmpfunc_end0
// * Replace .Lfunc_end0-memcmp to .Lmemcmpfunc_end0-memcmp
	.text
	.attribute	4, 16
	.attribute	5, "rv32im"
	.file	"glibc_memcmp_no_includes.c"
	.globl	memcmp
	.p2align	2
	.type	memcmp,@function
memcmp:
	addi	sp, sp, -16
	li	a3, 16
	bltu	a2, a3, .LBBmemcmp0_53
	andi	a3, a1, 3
	beqz	a3, .LBBmemcmp0_4
.LBBmemcmp0_2:
	lbu	a3, 0(a0)
	lbu	a4, 0(a1)
	sub	a3, a3, a4
	bnez	a3, .LBBmemcmp0_56
	addi	a1, a1, 1
	addi	a0, a0, 1
	andi	a3, a1, 3
	addi	a2, a2, -1
	bnez	a3, .LBBmemcmp0_2
.LBBmemcmp0_4:
	andi	a4, a0, 3
	srli	a3, a2, 2
	andi	a7, a3, 3
	beqz	a4, .LBBmemcmp0_8
	slli	a4, a0, 3
	andi	a4, a4, 24
	li	a5, 32
	sub	a5, a5, a4
	li	t0, 1
	andi	a6, a0, -4
	blt	t0, a7, .LBBmemcmp0_11
	bnez	a7, .LBBmemcmp0_15
	lw	t0, 0(a6)
	addi	a6, a6, 4
	mv	a7, a1
	j	.LBBmemcmp0_35
.LBBmemcmp0_8:
	li	a4, 1
	blt	a4, a7, .LBBmemcmp0_13
	bnez	a7, .LBBmemcmp0_17
	mv	a4, a0
	mv	a5, a1
	j	.LBBmemcmp0_19
.LBBmemcmp0_11:
	li	t0, 2
	bne	a7, t0, .LBBmemcmp0_16
	lw	t3, 0(a6)
	addi	t0, a6, 4
	addi	a6, a6, -4
	addi	a7, a1, -8
	addi	a3, a3, 2
	mv	t1, a1
	j	.LBBmemcmp0_37
.LBBmemcmp0_13:
	li	a4, 2
	bne	a7, a4, .LBBmemcmp0_18
	addi	a4, a0, -8
	addi	a5, a1, -8
	addi	a3, a3, 2
	mv	a6, a0
	mv	a7, a1
	j	.LBBmemcmp0_21
.LBBmemcmp0_15:
	lw	t2, 0(a6)
	lw	t0, 4(a6)
	lw	t1, 0(a1)
	addi	a6, a6, 8
	addi	a7, a1, 4
	addi	a3, a3, -1
	j	.LBBmemcmp0_39
.LBBmemcmp0_16:
	lw	t1, 0(a6)
	addi	a7, a1, -4
	addi	a3, a3, 1
	mv	t0, a1
	j	.LBBmemcmp0_36
.LBBmemcmp0_17:
	lw	a6, 0(a0)
	lw	a7, 0(a1)
	addi	a4, a0, 4
	addi	a5, a1, 4
	addi	a3, a3, -1
	j	.LBBmemcmp0_23
.LBBmemcmp0_18:
	addi	a4, a0, -4
	addi	a5, a1, -4
	addi	a3, a3, 1
	mv	a6, a0
	mv	a7, a1
	j	.LBBmemcmp0_20
.LBBmemcmp0_19:
	lw	t0, 0(a5)
	lw	t1, 0(a4)
	addi	a6, a4, 4
	addi	a7, a5, 4
	bne	t1, t0, .LBBmemcmp0_26
.LBBmemcmp0_20:
	lw	t0, 0(a7)
	lw	t1, 0(a6)
	addi	a6, a4, 8
	addi	a7, a5, 8
	bne	t1, t0, .LBBmemcmp0_28
.LBBmemcmp0_21:
	lw	t0, 0(a7)
	lw	t1, 0(a6)
	lw	a6, 12(a4)
	lw	a7, 12(a5)
	bne	t1, t0, .LBBmemcmp0_30
	addi	a4, a4, 16
	addi	a3, a3, -4
	addi	a5, a5, 16
	beqz	a3, .LBBmemcmp0_32
.LBBmemcmp0_23:
	beq	a6, a7, .LBBmemcmp0_19
	sw	a6, 12(sp)
	sw	a7, 8(sp)
	addi	a3, sp, 12
	addi	a4, sp, 8
.LBBmemcmp0_25:
	lbu	a5, 0(a3)
	lbu	a6, 0(a4)
	addi	a3, a3, 1
	addi	a4, a4, 1
	beq	a5, a6, .LBBmemcmp0_25
	j	.LBBmemcmp0_51
.LBBmemcmp0_26:
	sw	t1, 12(sp)
	sw	t0, 8(sp)
	addi	a3, sp, 12
	addi	a4, sp, 8
.LBBmemcmp0_27:
	lbu	a5, 0(a3)
	lbu	a6, 0(a4)
	addi	a3, a3, 1
	addi	a4, a4, 1
	beq	a5, a6, .LBBmemcmp0_27
	j	.LBBmemcmp0_51
.LBBmemcmp0_28:
	sw	t1, 12(sp)
	sw	t0, 8(sp)
	addi	a3, sp, 12
	addi	a4, sp, 8
.LBBmemcmp0_29:
	lbu	a5, 0(a3)
	lbu	a6, 0(a4)
	addi	a3, a3, 1
	addi	a4, a4, 1
	beq	a5, a6, .LBBmemcmp0_29
	j	.LBBmemcmp0_51
.LBBmemcmp0_30:
	sw	t1, 12(sp)
	sw	t0, 8(sp)
	addi	a3, sp, 12
	addi	a4, sp, 8
.LBBmemcmp0_31:
	lbu	a5, 0(a3)
	lbu	a6, 0(a4)
	addi	a3, a3, 1
	addi	a4, a4, 1
	beq	a5, a6, .LBBmemcmp0_31
	j	.LBBmemcmp0_51
.LBBmemcmp0_32:
	beq	a6, a7, .LBBmemcmp0_52
	sw	a6, 12(sp)
	sw	a7, 8(sp)
	addi	a3, sp, 12
	addi	a4, sp, 8
.LBBmemcmp0_34:
	lbu	a5, 0(a3)
	lbu	a6, 0(a4)
	addi	a3, a3, 1
	addi	a4, a4, 1
	beq	a5, a6, .LBBmemcmp0_34
	j	.LBBmemcmp0_51
.LBBmemcmp0_35:
	lw	t1, 0(a6)
	lw	t2, 0(a7)
	srl	t0, t0, a4
	sll	t3, t1, a5
	or	t3, t3, t0
	addi	t0, a7, 4
	bne	t3, t2, .LBBmemcmp0_42
.LBBmemcmp0_36:
	lw	t3, 4(a6)
	lw	t2, 0(t0)
	addi	t0, a6, 8
	srl	t1, t1, a4
	sll	t4, t3, a5
	or	t4, t4, t1
	addi	t1, a7, 8
	bne	t4, t2, .LBBmemcmp0_44
.LBBmemcmp0_37:
	lw	t4, 0(t1)
	lw	t2, 0(t0)
	lw	t0, 12(a6)
	lw	t1, 12(a7)
	srl	t3, t3, a4
	sll	t5, t2, a5
	or	t3, t5, t3
	bne	t3, t4, .LBBmemcmp0_46
	addi	a6, a6, 16
	addi	a3, a3, -4
	addi	a7, a7, 16
	beqz	a3, .LBBmemcmp0_48
.LBBmemcmp0_39:
	srl	t2, t2, a4
	sll	t3, t0, a5
	or	t2, t3, t2
	beq	t2, t1, .LBBmemcmp0_35
	sw	t2, 12(sp)
	sw	t1, 8(sp)
	addi	a3, sp, 12
	addi	a4, sp, 8
.LBBmemcmp0_41:
	lbu	a5, 0(a3)
	lbu	a6, 0(a4)
	addi	a3, a3, 1
	addi	a4, a4, 1
	beq	a5, a6, .LBBmemcmp0_41
	j	.LBBmemcmp0_51
.LBBmemcmp0_42:
	sw	t3, 12(sp)
	sw	t2, 8(sp)
	addi	a3, sp, 12
	addi	a4, sp, 8
.LBBmemcmp0_43:
	lbu	a5, 0(a3)
	lbu	a6, 0(a4)
	addi	a3, a3, 1
	addi	a4, a4, 1
	beq	a5, a6, .LBBmemcmp0_43
	j	.LBBmemcmp0_51
.LBBmemcmp0_44:
	sw	t4, 12(sp)
	sw	t2, 8(sp)
	addi	a3, sp, 12
	addi	a4, sp, 8
.LBBmemcmp0_45:
	lbu	a5, 0(a3)
	lbu	a6, 0(a4)
	addi	a3, a3, 1
	addi	a4, a4, 1
	beq	a5, a6, .LBBmemcmp0_45
	j	.LBBmemcmp0_51
.LBBmemcmp0_46:
	sw	t3, 12(sp)
	sw	t4, 8(sp)
	addi	a3, sp, 12
	addi	a4, sp, 8
.LBBmemcmp0_47:
	lbu	a5, 0(a3)
	lbu	a6, 0(a4)
	addi	a3, a3, 1
	addi	a4, a4, 1
	beq	a5, a6, .LBBmemcmp0_47
	j	.LBBmemcmp0_51
.LBBmemcmp0_48:
	srl	a3, t2, a4
	sll	a4, t0, a5
	or	a3, a4, a3
	beq	a3, t1, .LBBmemcmp0_52
	sw	a3, 12(sp)
	sw	t1, 8(sp)
	addi	a3, sp, 12
	addi	a4, sp, 8
.LBBmemcmp0_50:
	lbu	a5, 0(a3)
	lbu	a6, 0(a4)
	addi	a3, a3, 1
	addi	a4, a4, 1
	beq	a5, a6, .LBBmemcmp0_50
.LBBmemcmp0_51:
	sub	a3, a5, a6
	bnez	a3, .LBBmemcmp0_56
.LBBmemcmp0_52:
	andi	a3, a2, -4
	add	a0, a0, a3
	add	a1, a1, a3
	andi	a2, a2, 3
.LBBmemcmp0_53:
	li	a3, 0
	beqz	a2, .LBBmemcmp0_56
.LBBmemcmp0_54:
	lbu	a3, 0(a0)
	lbu	a4, 0(a1)
	sub	a3, a3, a4
	bnez	a3, .LBBmemcmp0_56
	li	a3, 0
	addi	a1, a1, 1
	addi	a2, a2, -1
	addi	a0, a0, 1
	bnez	a2, .LBBmemcmp0_54
.LBBmemcmp0_56:
	mv	a0, a3
	addi	sp, sp, 16
	ret
.Lmemcmpfunc_end0:
	.size	memcmp, .Lmemcmpfunc_end0-memcmp

	.ident	"Homebrew clang version 14.0.6"
	.section	".note.GNU-stack","",@progbits
	.addrsig