#define CUSTOM_0 0x0b
#define CUSTOM_1 0x2b

.macro addmod_1 rd, rs1, rs2
	.insn r CUSTOM_1, 0, 0, \rd, \rs1, \rs2
.endm

.macro submod_1 rd, rs1, rs2
	.insn r CUSTOM_1, 0, 1, \rd, \rs1, \rs2
.endm

.macro mulmod_1 rd, rs1, rs2
	.insn r CUSTOM_1, 0, 2, \rd, \rs1, \rs2
.endm

.macro iseqmod_1 rd, rs1, rs2
	.insn r CUSTOM_1, 0, 4, \rd, \rs1, \rs2
.endm

.macro addmod_2 rd, rs1, rs2
	.insn r CUSTOM_1, 0, 8, \rd, \rs1, \rs2
.endm


.global _start

_start:
	addmod_1 a2, a0, a1
	addmod_2 a2, a0, a1
	submod_1 a3, a0, a1
	mulmod_1 a4, a0, a1
	iseqmod_1 a6, a0, a1
    .insn i 0x0b, 0, x0, x0, 0
