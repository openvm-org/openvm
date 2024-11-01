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

.macro divmod_1 rd, rs1, rs2
	.insn r CUSTOM_1, 0, 3, \rd, \rs1, \rs2
.endm

.macro iseqmod_1 rd, rs1, rs2
	.insn r CUSTOM_1, 0, 4, \rd, \rs1, \rs2
.endm

.macro addmod_2 rd, rs1, rs2
	.insn r CUSTOM_1, 0, 8, \rd, \rs1, \rs2
.endm

.macro add256 rd, rs1, rs2
	.insn r CUSTOM_0, 5, 0, \rd, \rs1, \rs2
.endm

.macro sub256 rd, rs1, rs2
	.insn r CUSTOM_0, 5, 0, \rd, \rs1, \rs2
.endm

.macro xor256 rd, rs1, rs2
	.insn r CUSTOM_0, 5, 0, \rd, \rs1, \rs2
.endm

.macro or256 rd, rs1, rs2
	.insn r CUSTOM_0, 5, 0, \rd, \rs1, \rs2
.endm

.macro and256 rd, rs1, rs2
	.insn r CUSTOM_0, 5, 0, \rd, \rs1, \rs2
.endm

.macro sll256 rd, rs1, rs2
	.insn r CUSTOM_0, 5, 0, \rd, \rs1, \rs2
.endm

.macro srl256 rd, rs1, rs2
	.insn r CUSTOM_0, 5, 0, \rd, \rs1, \rs2
.endm

.macro sra256 rd, rs1, rs2
	.insn r CUSTOM_0, 5, 0, \rd, \rs1, \rs2
.endm

.macro slt256 rd, rs1, rs2
	.insn r CUSTOM_0, 5, 0, \rd, \rs1, \rs2
.endm

.macro sltu256 rd, rs1, rs2
	.insn r CUSTOM_0, 5, 0, \rd, \rs1, \rs2
.endm

.macro mul256 rd, rs1, rs2
	.insn r CUSTOM_0, 5, 0, \rd, \rs1, \rs2
.endm


.global _start

_start:
	li a0, 32
	li a1, 64
	addmod_1 a2, a0, a1
	submod_1 a3, a0, a1
	mulmod_1 a4, a0, a1
	divmod_1 a5, a0, a1
	#iseqmod_1 a6, a0, a1
	add256 a2, a0, a1
	sub256 a3, a0, a1
	xor256 a4, a0, a1
	or256 a5, a0, a1
	and256 a6, a0, a1
	sll256 a2, a0, a1
	srl256 a3, a0, a1
	sra256 a4, a0, a1
	slt256 a5, a0, a1
	sltu256 a6, a0, a1
	mul256 a2, a0, a1
    .insn i 0x0b, 0, x0, x0, 0
