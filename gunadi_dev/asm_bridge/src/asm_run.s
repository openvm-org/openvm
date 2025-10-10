.intel_syntax noprefix
.code64
.section .text
.extern extern_handler
.global asm_run_internal

asm_run_internal:
    mov rbx, rdi
    mov rbp, rsi
    mov r12, rdx
    sub rsp, 8
pc_0:
    mov rdi, rbx
    mov rsi, rbp
    mov r12, rdx
    call extern_handler

pc_4:
    mov rdi, rbx
    mov rsi, rbp
    mov r12, rdx
    call extern_handler

asm_run_end:
    add rsp, 8
    xor eax, eax
    ret
