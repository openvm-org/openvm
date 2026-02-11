# RV64IM comprehensive stress test.
# Exercises every instruction the rv64im transpiler handles.
# Terminates with exit-code 0 on success, 1 on failure.

#define CUSTOM_0 0x0b

.macro terminate ec
    .insn i CUSTOM_0, 0, x0, x0, \ec
.endm

.macro assert_eq reg, imm, fail_label
    li   t6, \imm
    bne  \reg, t6, \fail_label
.endm

.text
.global _start
_start:
    # Initialise the stack pointer to a valid address within POINTER_MAX_BITS (29).
    # STACK_TOP = 0x200400, same as the OpenVM platform constant.
    li   sp, 0x200400

# ===================================================================
# 1. Base ALU (R-type): add, sub, xor, or, and
# ===================================================================
    li   a0, 10
    li   a1, 32
    add  a2, a0, a1        # 42
    assert_eq a2, 42, fail

    li   a0, 100
    li   a1, 58
    sub  a2, a0, a1        # 42
    assert_eq a2, 42, fail

    li   a0, 0xFF00
    li   a1, 0xF0F0
    xor  a2, a0, a1        # 0x0FF0
    assert_eq a2, 0x0FF0, fail

    li   a0, 0xF000
    li   a1, 0x0F00
    or   a2, a0, a1        # 0xFF00
    assert_eq a2, 0xFF00, fail

    li   a0, 0xFF00
    li   a1, 0x0F00
    and  a2, a0, a1        # 0x0F00
    assert_eq a2, 0x0F00, fail

# ===================================================================
# 2. Shifts (R-type): sll, srl, sra
# ===================================================================
    li   a0, 1
    li   a1, 10
    sll  a2, a0, a1        # 1024
    assert_eq a2, 1024, fail

    li   a0, 1024
    li   a1, 5
    srl  a2, a0, a1        # 32
    assert_eq a2, 32, fail

    li   a0, -16
    li   a1, 2
    sra  a2, a0, a1        # -4
    assert_eq a2, -4, fail

# ===================================================================
# 3. Comparisons (R-type): slt, sltu
# ===================================================================
    li   a0, -1
    li   a1, 1
    slt  a2, a0, a1        # 1 (signed: -1 < 1)
    assert_eq a2, 1, fail

    slt  a2, a1, a0        # 0
    assert_eq a2, 0, fail

    li   a0, 1
    li   a1, 2
    sltu a2, a0, a1        # 1
    assert_eq a2, 1, fail

    sltu a2, a1, a0        # 0
    assert_eq a2, 0, fail

# ===================================================================
# 4. ALU immediate (I-type): addi, xori, ori, andi, slti, sltiu
# ===================================================================
    li   a0, 100
    addi a2, a0, 42        # 142
    assert_eq a2, 142, fail

    li   a0, 0xFF
    xori a2, a0, 0xFF      # 0
    assert_eq a2, 0, fail

    li   a0, 0x0F
    ori  a2, a0, 0xF0      # 0xFF
    assert_eq a2, 0xFF, fail

    li   a0, 0x1234
    andi a2, a0, 0xFF      # 0x34
    assert_eq a2, 0x34, fail

    li   a0, 5
    slti a2, a0, 10        # 1
    assert_eq a2, 1, fail

    li   a0, 15
    slti a2, a0, 10        # 0
    assert_eq a2, 0, fail

    li   a0, 5
    sltiu a2, a0, 10       # 1
    assert_eq a2, 1, fail

    li   a0, 15
    sltiu a2, a0, 10       # 0
    assert_eq a2, 0, fail

# ===================================================================
# 5. Shift immediate with 6-bit shamt: slli, srli, srai
# ===================================================================
    # Small shamt (< 32)
    li   a0, 1
    slli a2, a0, 8         # 256
    assert_eq a2, 256, fail

    li   a0, 256
    srli a2, a0, 4         # 16
    assert_eq a2, 16, fail

    li   a0, -256
    srai a2, a0, 4         # -16
    assert_eq a2, -16, fail

    # Large shamt (>= 32) — exercises the 6-bit interception path
    li   a0, 1
    slli a2, a0, 32        # 1 << 32 = 0x100000000
    li   t6, 1
    slli t6, t6, 32
    bne  a2, t6, fail

    li   a0, 1
    slli a2, a0, 63        # 1 << 63 = 0x8000000000000000
    li   t6, 1
    slli t6, t6, 63
    bne  a2, t6, fail

    # srli with shamt=63
    li   a0, 1
    slli a0, a0, 63        # 0x8000000000000000
    srli a2, a0, 63        # 1
    assert_eq a2, 1, fail

    # srli with shamt=32
    li   a0, -1             # 0xFFFFFFFFFFFFFFFF
    srli a2, a0, 32        # 0xFFFFFFFF
    li   t6, 1
    slli t6, t6, 32
    addi t6, t6, -1        # 0xFFFFFFFF
    bne  a2, t6, fail

    # srai with shamt=63 (sign bit propagation)
    li   a0, 1
    slli a0, a0, 63        # 0x8000000000000000 (i64::MIN)
    srai a2, a0, 63        # -1
    assert_eq a2, -1, fail

    # srai with shamt=32
    li   a0, 1
    slli a0, a0, 63        # i64::MIN
    srai a2, a0, 32        # -2147483648
    li   t6, 1
    slli t6, t6, 31
    neg  t6, t6             # -2147483648
    bne  a2, t6, fail

# ===================================================================
# 6. W-type ALU (RV64-only R-type): addw, subw
# ===================================================================
    # addw: 0x7FFFFFFF + 1 = 0x80000000, sign-extended to -2147483648
    li   a0, 0x7FFFFFFF
    li   a1, 1
    addw a2, a0, a1
    li   t6, 1
    slli t6, t6, 31
    neg  t6, t6             # -2147483648
    bne  a2, t6, fail

    # subw: 10 - 25 = -15 (sign-extended)
    li   a0, 10
    li   a1, 25
    subw a2, a0, a1
    assert_eq a2, -15, fail

# ===================================================================
# 7. W-type shifts (RV64-only R-type): sllw, srlw, sraw
# ===================================================================
    li   a0, 0xFF
    li   a1, 8
    sllw a2, a0, a1         # 0xFF00
    assert_eq a2, 0xFF00, fail

    li   a0, 0xFF00
    li   a1, 4
    srlw a2, a0, a1         # 0x0FF0
    assert_eq a2, 0x0FF0, fail

    # sraw: 0xFFFFFF00 (as 32-bit = -256), >> 4 = -16
    li   a0, -256
    li   a1, 4
    sraw a2, a0, a1
    assert_eq a2, -16, fail

# ===================================================================
# 8. W-type immediate: addiw, slliw, srliw, sraiw
# ===================================================================
    li   a0, 100
    addiw a2, a0, 42       # 142
    assert_eq a2, 142, fail

    li   a0, 0xFF
    slliw a2, a0, 16       # 0x00FF0000
    assert_eq a2, 0x00FF0000, fail

    li   a0, 0x00FF0000
    srliw a2, a0, 16       # 0xFF
    assert_eq a2, 0xFF, fail

    # sraiw: -256 (0xFFFFFF00 in 32-bit) >> 16 = -1
    li   a0, -256
    sraiw a2, a0, 16
    assert_eq a2, -1, fail

# ===================================================================
# 9. Loads and stores: sb, sh, sw, sd, lb, lbu, lh, lhu, lw, lwu, ld
# ===================================================================
    # Use stack area as scratch buffer
    addi sp, sp, -32

    # sd / ld
    li   a0, 0xDEADBEEFCAFEBABE
    sd   a0, 0(sp)
    ld   a2, 0(sp)
    bne  a2, a0, fail

    # sw / lw (sign-extending) / lwu (zero-extending)
    li   a0, 0x81234567
    sw   a0, 0(sp)
    lw   a2, 0(sp)         # sign-extended: negative
    lwu  a3, 0(sp)         # zero-extended: 0x81234567

    # Check lw result is negative (sign-extended)
    slt  t5, a2, zero
    assert_eq t5, 1, fail
    # Check lwu result matches original (zero-extended)
    bne  a3, a0, fail

    # sh / lh (sign-extending) / lhu (zero-extending)
    li   a0, 0xFEDC
    sh   a0, 0(sp)
    lh   a2, 0(sp)         # sign-extended: -292
    lhu  a3, 0(sp)         # zero-extended: 0xFEDC
    assert_eq a2, -292, fail
    assert_eq a3, 0xFEDC, fail

    # sb / lb (sign-extending) / lbu (zero-extending)
    li   a0, 0x80
    sb   a0, 0(sp)
    lb   a2, 0(sp)         # sign-extended: -128
    lbu  a3, 0(sp)         # zero-extended: 0x80
    assert_eq a2, -128, fail
    assert_eq a3, 0x80, fail

    addi sp, sp, 32

# ===================================================================
# 10. Branches: beq, bne, blt, bge, bltu, bgeu
# ===================================================================
    # beq: taken
    li   a0, 42
    li   a1, 42
    beq  a0, a1, beq_ok
    j    fail
beq_ok:

    # bne: taken
    li   a0, 1
    li   a1, 2
    bne  a0, a1, bne_ok
    j    fail
bne_ok:

    # blt: signed, taken (-1 < 1)
    li   a0, -1
    li   a1, 1
    blt  a0, a1, blt_ok
    j    fail
blt_ok:

    # bge: signed, taken (1 >= -1)
    li   a0, 1
    li   a1, -1
    bge  a0, a1, bge_ok
    j    fail
bge_ok:

    # bltu: unsigned, taken (1 < 2)
    li   a0, 1
    li   a1, 2
    bltu a0, a1, bltu_ok
    j    fail
bltu_ok:

    # bgeu: unsigned, taken (2 >= 1)
    li   a0, 2
    li   a1, 1
    bgeu a0, a1, bgeu_ok
    j    fail
bgeu_ok:

# ===================================================================
# 11. JAL / JALR
# ===================================================================
    jal  ra, jal_target
    j    jal_return
jal_target:
    jalr zero, ra, 0       # return via jalr
jal_return:

# ===================================================================
# 12. LUI / AUIPC
# ===================================================================
    lui  a0, 0x12345        # a0 = 0x12345000
    srli a2, a0, 12
    assert_eq a2, 0x12345, fail

    auipc a0, 0             # a0 = current PC (just check non-zero)
    beqz a0, fail

# ===================================================================
# 13. M-extension (64-bit): mul, mulh, mulhu, mulhsu, div, divu, rem, remu
# ===================================================================
    li   a0, 7
    li   a1, 6
    mul  a2, a0, a1         # 42
    assert_eq a2, 42, fail

    # mulh: (-1) * 2, signed high half = -1
    li   a0, -1
    li   a1, 2
    mulh a2, a0, a1
    assert_eq a2, -1, fail

    # mulhu: (2^63) * 2, unsigned high half = 1
    li   a0, 1
    slli a0, a0, 63
    li   a1, 2
    mulhu a2, a0, a1
    assert_eq a2, 1, fail

    # mulhsu: (-1) * 1, signed*unsigned high half = -1
    li   a0, -1
    li   a1, 1
    mulhsu a2, a0, a1
    assert_eq a2, -1, fail

    # div: -100 / 7 = -14 (signed)
    li   a0, -100
    li   a1, 7
    div  a2, a0, a1
    assert_eq a2, -14, fail

    # divu: 100 / 7 = 14 (unsigned)
    li   a0, 100
    li   a1, 7
    divu a2, a0, a1
    assert_eq a2, 14, fail

    # rem: -100 % 7 = -2 (signed)
    li   a0, -100
    li   a1, 7
    rem  a2, a0, a1
    assert_eq a2, -2, fail

    # remu: 100 % 7 = 2 (unsigned)
    li   a0, 100
    li   a1, 7
    remu a2, a0, a1
    assert_eq a2, 2, fail

# ===================================================================
# 14. M-extension W (32-bit on RV64): mulw, divw, divuw, remw, remuw
# ===================================================================
    # mulw: 100000 * 40000 = 4000000000 → wraps 32-bit → sign-extends
    li   a0, 100000
    li   a1, 40000
    mulw a2, a0, a1
    # 4000000000 as u32 = 0xEE6B2800, sign-extended = -294967296
    li   t6, -294967296
    bne  a2, t6, fail

    # divw: -100 / 7 = -14 (signed 32-bit)
    li   a0, -100
    li   a1, 7
    divw a2, a0, a1
    assert_eq a2, -14, fail

    # divuw: 0xFFFFFFFE / 3 = 0x55555554 (unsigned 32-bit)
    li   a0, 0xFFFFFFFE
    li   a1, 3
    divuw a2, a0, a1
    assert_eq a2, 0x55555554, fail

    # remw: -100 % 7 = -2 (signed 32-bit)
    li   a0, -100
    li   a1, 7
    remw a2, a0, a1
    assert_eq a2, -2, fail

    # remuw: 0xFFFFFFFE % 3 = 2 (unsigned 32-bit)
    li   a0, 0xFFFFFFFE
    li   a1, 3
    remuw a2, a0, a1
    assert_eq a2, 2, fail

# ===================================================================
# 15. Fibonacci (integration test exercising loops, branches, add)
# ===================================================================
    li   a0, 15             # n
    li   a1, 0              # fib(0)
    li   a2, 1              # fib(1)
fib_loop:
    beqz a0, fib_done
    addi a0, a0, -1
    add  a3, a1, a2
    mv   a1, a2
    mv   a2, a3
    j    fib_loop
fib_done:
    # fib(15) = 610
    assert_eq a1, 610, fail

# ===================================================================
# SUCCESS
# ===================================================================
    terminate 0

# ===================================================================
# FAILURE
# ===================================================================
fail:
    terminate 1
