# RV64 intrinsic test for custom OpenVM opcodes.
# Exercises: TERMINATE, PHANTOM (HintInput, PrintStr), HINT_STORED, HINT_BUFFER.

#define CUSTOM_0 0x0b

# funct3 values for CUSTOM_0
#define TERMINATE_FUNCT3 0
#define HINT_FUNCT3      1
#define PHANTOM_FUNCT3   3

# PHANTOM sub-opcodes (imm field)
#define PHANTOM_HINT_INPUT 0
#define PHANTOM_PRINT_STR  1

# HINT sub-opcodes (imm field)
#define HINT_STORED_IMM 0
#define HINT_BUFFER_IMM 1

.macro terminate ec
    .insn i CUSTOM_0, TERMINATE_FUNCT3, x0, x0, \ec
.endm

.macro phantom_hint_input
    .insn i CUSTOM_0, PHANTOM_FUNCT3, x0, x0, PHANTOM_HINT_INPUT
.endm

.macro phantom_print_str rd, rs1
    .insn i CUSTOM_0, PHANTOM_FUNCT3, \rd, \rs1, PHANTOM_PRINT_STR
.endm

.macro hint_stored rd
    .insn i CUSTOM_0, HINT_FUNCT3, \rd, x0, HINT_STORED_IMM
.endm

.macro hint_buffer rd, rs1
    .insn i CUSTOM_0, HINT_FUNCT3, \rd, \rs1, HINT_BUFFER_IMM
.endm

.text
.global _start
_start:

# ===================================================================
# 1. PHANTOM(HintInput) — request hint data from host
# ===================================================================
    phantom_hint_input

# ===================================================================
# 2. PHANTOM(PrintStr) — tell host to print string at (ptr, len)
# ===================================================================
    li   a0, 0x300000           # dummy pointer
    li   a1, 18                 # dummy length
    phantom_print_str a0, a1

# ===================================================================
# 3. HINT_STORED — store one dword from hint stream
# ===================================================================
    li   a0, 0x300100           # dummy destination address
    hint_stored a0

# ===================================================================
# 4. HINT_BUFFER — store multiple dwords from hint stream
# ===================================================================
    li   a0, 0x300200           # dummy buffer address
    li   a1, 8                  # 8 dwords
    hint_buffer a0, a1

# ===================================================================
# 5. TERMINATE with exit code 0 (success)
# ===================================================================
    terminate 0
