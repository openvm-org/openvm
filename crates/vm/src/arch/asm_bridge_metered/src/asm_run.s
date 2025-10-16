.code64
.section .text
.global asm_run_internal

asm_run_internal:
    push rbp
    push rbx
    push r12
    push r13
    push r14
    mov rbx, rdi
    mov rbp, rsi
    mov r13, rdx
    mov r14, rcx

asm_execute:
    mov rdi, r14 
    mov rsi, rbp
    mov rdx, rbx 
    call should_suspend
    cmp rax, 1
    je asm_run_end
    mov rdi, rbx
    mov rsi, rbp
    mov rdx, r13
    mov rcx, r14
    call metered_extern_handler
    add r14, 1
    cmp rax, 1
    je asm_run_end
    mov r13, rax
    jmp asm_execute

asm_run_end:
    mov rdi, rbx
    mov rsi, r14
    mov rdx, r13
    call metered_set_instret_and_pc
    xor eax, eax
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
