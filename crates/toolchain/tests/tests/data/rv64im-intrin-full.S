# Comprehensive RV64 intrinsic test for custom OpenVM opcodes.
# Exercises and VERIFIES: TERMINATE, PHANTOM (HintInput, PrintStr),
# HINT_STORED, HINT_BUFFER.
#
# This test expects TWO input vectors:
#   Input 0: 24 field elements [0x10, 0x11, ..., 0x27]
#   Input 1: 16 field elements [0xA0, 0xA1, ..., 0xAF]
#
# Build:
#   riscv64-unknown-elf-gcc -march=rv64im -mabi=lp64 -nostartfiles \
#       -e _start -Ttext 0x00200800 -Wl,-N rv64im-intrin-full.S -o rv64im-intrin-full

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

# ---------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------
.macro assert_eq reg, imm
    li   t6, \imm
    bne  \reg, t6, fail
.endm

.text
.global _start
_start:

# ===================================================================
# ROUND 1: 24-element input [0x10 .. 0x27]
# ===================================================================
# After phantom_hint_input, hint_stream contains:
#   dword 0: length = 24 as u64 LE  (0x18, 0, 0, 0, 0, 0, 0, 0)
#   dwords 1-3: data bytes [0x10..0x27]
# ===================================================================

    # 1a. Request hint data (pops input 0 into hint_stream)
    phantom_hint_input

    # 1b. HINT_STORED: read 1 dword (the length prefix) into 0x300100
    li   a0, 0x300100
    hint_stored a0

    # 1c. Verify length dword: should be 24 (0x18) as u64 LE
    ld   t0, 0(a0)                 # load the full 8-byte dword
    assert_eq t0, 24               # 24 as u64

    # 1d. HINT_BUFFER: read 3 dwords of data into 0x300200
    li   a0, 0x300200
    li   a1, 3                     # 3 dwords = 24 bytes
    hint_buffer a0, a1

    # 1e. Verify data bytes via byte loads
    # dword 0 at 0x300200: bytes [0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17]
    lbu  t0, 0(a0)
    assert_eq t0, 0x10
    lbu  t0, 1(a0)
    assert_eq t0, 0x11
    lbu  t0, 7(a0)
    assert_eq t0, 0x17

    # dword 1 at 0x300208: bytes [0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F]
    lbu  t0, 8(a0)
    assert_eq t0, 0x18
    lbu  t0, 15(a0)
    assert_eq t0, 0x1F

    # dword 2 at 0x300210: bytes [0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27]
    lbu  t0, 16(a0)
    assert_eq t0, 0x20
    lbu  t0, 23(a0)
    assert_eq t0, 0x27

    # 1f. Also verify full dword load of first data dword
    ld   t0, 0(a0)
    li   t6, 0x1716151413121110
    bne  t0, t6, fail

# ===================================================================
# ROUND 2: 16-element input [0xA0 .. 0xAF]
# ===================================================================
# After phantom_hint_input, hint_stream contains:
#   dword 0: length = 16 as u64 LE  (0x10, 0, 0, 0, 0, 0, 0, 0)
#   dwords 1-2: data bytes [0xA0..0xAF]
# ===================================================================

    # 2a. Request second hint input
    phantom_hint_input

    # 2b. HINT_STORED: read length into 0x300300
    li   a0, 0x300300
    hint_stored a0

    # 2c. Verify length dword: should be 16 (0x10) as u64 LE
    ld   t0, 0(a0)
    assert_eq t0, 16

    # 2d. HINT_BUFFER: read 2 dwords of data into 0x300400
    li   a0, 0x300400
    li   a1, 2                     # 2 dwords = 16 bytes
    hint_buffer a0, a1

    # 2e. Verify data bytes
    # dword 0 at 0x300400: bytes [0xA0, 0xA1, ..., 0xA7]
    lbu  t0, 0(a0)
    assert_eq t0, 0xA0
    lbu  t0, 7(a0)
    assert_eq t0, 0xA7

    # dword 1 at 0x300408: bytes [0xA8, 0xA9, ..., 0xAF]
    lbu  t0, 8(a0)
    assert_eq t0, 0xA8
    lbu  t0, 15(a0)
    assert_eq t0, 0xAF

    # Full dword verification
    ld   t0, 0(a0)
    li   t6, 0xA7A6A5A4A3A2A1A0
    bne  t0, t6, fail

# ===================================================================
# 3. PHANTOM(PrintStr) — quick smoke test
# ===================================================================
    li   a0, 0x300000
    li   a1, 5
    phantom_print_str a0, a1

# ===================================================================
# SUCCESS: terminate with exit code 0
# ===================================================================
    terminate 0

# ===================================================================
# FAILURE: terminate with exit code 1
# ===================================================================
fail:
    terminate 1
