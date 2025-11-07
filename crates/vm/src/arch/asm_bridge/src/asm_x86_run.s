.intel_syntax noprefix
.code64
.section .text
.global asm_run_internal

asm_run_internal:
    push rbp
    push rbx
    push r13
    mov  rbx, rdi      ; state
    mov  rbp, rsi      ; ctx
    mov  r13, rdx      ; pc

asm_execute:
    mov  rdi, rbx
    call should_suspend
    cmp  rax, 1
    je   asm_run_end

    mov  rdi, rbx
    mov  rsi, rbp
    mov  rdx, r13
    call metered_extern_handler
    cmp  rax, 1
    je   asm_run_end
    mov  r13, rax
    jmp  asm_execute

asm_run_end:
    mov  rdi, rbx
    mov  rsi, rbp
    mov  rdx, r13
    call metered_set_pc
    xor  rax, rax
    pop  r13
    pop  rbx
    pop  rbp
    ret