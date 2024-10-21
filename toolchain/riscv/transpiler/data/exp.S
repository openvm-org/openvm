.global _start

_start:
	li a0, 57
	li a2, 10007
	addi a1, a2, -2
	li a4, 1

loop:
	beqz a1, finish
	andi t3, a1, 1
	srli a1, a1, 1

	beqz t3, tmp
	mul a4, a4, a0
	rem a4, a4, a2
tmp:
	mul a0, a0, a0
	rem a0, a0, a2
	jal t4, loop

finish:
	li a0, 57
	mul a0, a0, a4
	rem a0, a0, a2
	li a1, 1
	bne a0, a1, 228
	ecall
