#define CUSTOM_1 0x2b

.macro addmod_1 rd, rs1, rs2
	.insn r CUSTOM_1, 0, 0, \rd, \rs1, \rs2
.endm

.macro addmod_2 rd, rs1, rs2
	.insn r CUSTOM_1, 0, 4, \rd, \rs1, \rs2
.endm

.global _start

_start:
	addmod_1 a2, a0, a1
	addmod_2 a3, a0, a1
    ecall
