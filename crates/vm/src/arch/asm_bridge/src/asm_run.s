.intel_syntax noprefix
.code64
.section .text
.extern extern_handler
.global asm_run_internal

asm_run_internal:
    push rbp
    push rbx
    push r12
    mov rbx, rdi
    mov rbp, rsi
    xor r13, r13
    lea r10, [rip + map_pc_base]
    lea r12, [rip + map_pc_end]
    sub r12, r10
    shr r12, 2
pc_0:
    mov rdi, rbx
    mov rsi, rbp
    mov rdx, r13
    call extern_handler
    mov r13, rax
    shr eax, 2
    cmp rax, r12
    jae asm_run_end
    lea r10, [rip + map_pc_base]
    movsxd  r11, dword ptr [r10 + rax*4]
    add r11, r10
    jmp r11

pc_4:
    mov rdi, rbx
    mov rsi, rbp
    mov rdx, r13
    call extern_handler
    mov r13, rax
    shr eax, 2
    cmp rax, r12
    jae asm_run_end
    lea r10, [rip + map_pc_base]
    movsxd  r11, dword ptr [r10 + rax*4]
    add r11, r10
    jmp r11

pc_8:
    mov rdi, rbx
    mov rsi, rbp
    mov rdx, r13
    call extern_handler
    mov r13, rax
    shr eax, 2
    cmp rax, r12
    jae asm_run_end
    lea r10, [rip + map_pc_base]
    movsxd  r11, dword ptr [r10 + rax*4]
    add r11, r10
    jmp r11

pc_c:
    mov rdi, rbx
    mov rsi, rbp
    mov rdx, r13
    call extern_handler
    mov r13, rax
    shr eax, 2
    cmp rax, r12
    jae asm_run_end
    lea r10, [rip + map_pc_base]
    movsxd  r11, dword ptr [r10 + rax*4]
    add r11, r10
    jmp r11

pc_10:
    mov rdi, rbx
    mov rsi, rbp
    mov rdx, r13
    call extern_handler
    mov r13, rax
    shr eax, 2
    cmp rax, r12
    jae asm_run_end
    lea r10, [rip + map_pc_base]
    movsxd  r11, dword ptr [r10 + rax*4]
    add r11, r10
    jmp r11

pc_14:
    mov rdi, rbx
    mov rsi, rbp
    mov rdx, r13
    call extern_handler
    mov r13, rax
    shr eax, 2
    cmp rax, r12
    jae asm_run_end
    lea r10, [rip + map_pc_base]
    movsxd  r11, dword ptr [r10 + rax*4]
    add r11, r10
    jmp r11

pc_18:
    mov rdi, rbx
    mov rsi, rbp
    mov rdx, r13
    call extern_handler
    mov r13, rax
    shr eax, 2
    cmp rax, r12
    jae asm_run_end
    lea r10, [rip + map_pc_base]
    movsxd  r11, dword ptr [r10 + rax*4]
    add r11, r10
    jmp r11

pc_1c:
    mov rdi, rbx
    mov rsi, rbp
    mov rdx, r13
    call extern_handler
    mov r13, rax
    shr eax, 2
    cmp rax, r12
    jae asm_run_end
    lea r10, [rip + map_pc_base]
    movsxd  r11, dword ptr [r10 + rax*4]
    add r11, r10
    jmp r11

pc_20:
    mov rdi, rbx
    mov rsi, rbp
    mov rdx, r13
    call extern_handler
    mov r13, rax
    shr eax, 2
    cmp rax, r12
    jae asm_run_end
    lea r10, [rip + map_pc_base]
    movsxd  r11, dword ptr [r10 + rax*4]
    add r11, r10
    jmp r11

pc_24:
    mov rdi, rbx
    mov rsi, rbp
    mov rdx, r13
    call extern_handler
    mov r13, rax
    shr eax, 2
    cmp rax, r12
    jae asm_run_end
    lea r10, [rip + map_pc_base]
    movsxd  r11, dword ptr [r10 + rax*4]
    add r11, r10
    jmp r11

pc_28:
    jmp asm_run_end

asm_run_end:
    xor eax, eax
    pop r12
    pop rbx
    pop rbp
    ret
.section .rodata,"a",@progbits
.p2align 4
map_pc_base:
    .long (pc_0 - map_pc_base)
    .long (pc_4 - map_pc_base)
    .long (pc_8 - map_pc_base)
    .long (pc_c - map_pc_base)
    .long (pc_10 - map_pc_base)
    .long (pc_14 - map_pc_base)
    .long (pc_18 - map_pc_base)
    .long (pc_1c - map_pc_base)
    .long (pc_20 - map_pc_base)
    .long (pc_24 - map_pc_base)
    .long (pc_28 - map_pc_base)
map_pc_end:

