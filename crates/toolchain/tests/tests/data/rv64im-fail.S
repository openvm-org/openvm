# Minimal RV64 program that always fails.
# Used to verify the test harness detects assertion failures.

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
    # Intentionally fail: assert_eq(0, 1) should branch to fail.
    li   a0, 0
    assert_eq a0, 1, fail

    # Should never reach here.
    terminate 0

fail:
    terminate 1
