.intel_syntax noprefix
.code64
.section .text
.extern TEST_FN
.extern ADD_RV32
.global asm_run_internal

asm_run_internal:
    mov rbx, rdi
    sub rsp, 8
    call TEST_FN

pc_0:
    mov rdi, rbx
    mov rsi, 4
    mov rdx, 4
    mov rcx, 4
    mov r8, 1
    mov r9, 0
    call ADD_RV32

asm_run_end:
    add rsp, 8
    xor eax, eax
    ret
