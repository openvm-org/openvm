# Minimal OpenVM program that immediately terminates with exit code 0
.global _start
_start:
    .insn i 0x0b, 0, x0, x0, 0
