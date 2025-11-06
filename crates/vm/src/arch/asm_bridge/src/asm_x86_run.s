.intel_syntax noprefix
.code64
.section .text
.global asm_run_internal

/*
rbx = vm_exec_state_ptr 
rbp = pre_compute_insns_ptr
r13 = cur_pc 
r12 is currently unused 

asm_run_internal:
    push rbp        ; push callee saved registers
    push rbx
    push r12
    push r13
    push r13        ; A dummy push to ensure the stack is 16 bytes aligned
    mov rbx, rdi    ; rbx = rdi = vm_exec_state_ptr 
    mov rbp, rsi    ; rbp = rsi = pre_compute_insns_ptr 
    mov r13, rdx    ; r13 = rdx = from_state_pc 

asm_execute:
    mov rdi, rbx        ; rdi = vm_exec_state_ptr 
    call should_suspend ; should_suspend(vm_exec_state_ptr)
    cmp rax, 1          ; if return value of should_suspend is 1 
    je asm_run_end      ; jump to asm_run_end
    mov rdi, rbx        ; rdi = vm_exec_state_ptr
    mov rsi, rbp        ; rsi = pre_compute_insns_ptr
    mov rdx, r13        ; rdx = cur_pc 
    call extern_handler ; extern_handler(vm_exec_state_ptr, pre_compute_insns_ptr, cur_pc)
    cmp rax, 1          ; if return value of extern_handler is 1
    je asm_run_end      ; jump to asm_run_end 
    mov r13, rax        ; cur_pc = return value of extern_handler 
    jmp asm_execute     ; jump to asm_execute 

asm_run_end:
    mov rdi, rbx        ; rdi = vm_exec_state_ptr
    mov rsi, rbp        ; rsi = pre_compute_insns_ptr
    mov rdx, r13        ; rdx = cur_pc 
    call set_pc ; set_pc(vm_exec_state_ptr, pre_compute_insns_ptr, cur_pc)
    xor rax, rax        ; set return value to 0
    pop r13             ; There was a dummy push to ensure the stack is 16 bytes aligned
    pop r13             
    pop r12
    pop rbx
    pop rbp
    ret
*/

asm_run_internal:
    push rbp        
    push rbx
    push r12
    push r13
    push r13        
    mov rbx, rdi    
    mov rbp, rsi    
    mov r13, rdx

asm_execute:
    mov rsi, r13     
    mov rdx, rbx    
    call should_suspend 
    cmp rax, 1     
    je asm_run_end   
    mov rdi, rbx    
    mov rsi, rbp    
    mov rdx, r13
    call extern_handler
    cmp rax, 1
    je asm_run_end
    mov r13, rax
    jmp asm_execute

asm_run_end:
    mov rdi, rbx
    mov rsi, rbp 
    mov rdx, r13
    call set_pc
    xor rax, rax
    pop r13
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
