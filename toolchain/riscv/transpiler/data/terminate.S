#define CUSTOM_0 0x0b
#define CUSTOM_1 0x2b

.macro addmod_1 rd, rs1, rs2
	.insn r CUSTOM_1, 0, 0, \rd, \rs1, \rs2
.endm

.macro addmod_2 rd, rs1, rs2
	.insn r CUSTOM_1, 0, 4, \rd, \rs1, \rs2
.endm

.macro terminate ec
	.insn r CUSTOM_0, 0, \ec, x0, x0, x0
.endm

.global _start

_start:
	li zero, 1
	add a0, a0, zero
	bne a0, a1, 8
	terminate 0
	terminate 1
