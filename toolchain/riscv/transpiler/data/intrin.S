#define CUSTOM_0 0x0b
#define CUSTOM_1 0x2b

.macro addmod_1 rd, rs1, rs2
	.insn r CUSTOM_1, 0, 0, \rd, \rs1, \rs2
.endm

.global _start

_start:
	addmod_1 a2, a0, a1
    .insn i 0x0b, 0, x0, x0, 0
