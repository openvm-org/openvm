.intel_syntax noprefix
.code64
.section .text
.global asm_run_internal

/*
rbx = vm_exec_state_ptr 
rbp = pre_compute_insns_ptr
r13 = cur_pc 
r14 = cur_instret
r12 is currently unused 

asm_run_internal:
    push rbp        ; push callee saved registers
    push rbx
    push r12
    push r13
    push r14        ; since we push 5 times, this already ensures the stack is 16 bytes aligned
    mov rbx, rdi    ; rbx = rdi = vm_exec_state_ptr 
    mov rbp, rsi    ; rbp = rsi = pre_compute_insns_ptr 
    mov r13, rdx    ; r13 = rdx = from_state_pc 
    mov r14, rcx    ; r14 = rcx = from_state_instret 

asm_execute:
    mov rdi, r14        ; rdi = cur_instret 
    mov rsi, r13        ; rsi = cur_pc 
    mov rdx, rbx        ; rdx = vm_exec_state_ptr 
    call should_suspend ; should_suspend(cur_instret, cur_pc, vm_exec_state_ptr)
    cmp rax, 1          ; if return value of should_suspend is 1 
    je asm_run_end      ; jump to asm_run_end
    mov rdi, rbx        ; rdi = vm_exec_state_ptr
    mov rsi, rbp        ; rsi = pre_compute_insns_ptr
    mov rdx, r13        ; rdx = cur_pc 
    mov rcx, r14        ; rcx = cur_instret
    call extern_handler ; extern_handler(vm_exec_state_ptr, pre_compute_insns_ptr, cur_pc, cur_instret)
    add r14, 1          ; cur_instret += 1
    cmp rax, 1          ; if return value of extern_handler is 1
    je asm_run_end      ; jump to asm_run_end 
    mov r13, rax        ; cur_pc = return value of extern_handler 
    jmp asm_execute     ; jump to asm_execute 

asm_run_end:
    mov rdi, rbx        ; rdi = vm_exec_state_ptr
    mov rsi, rbp        ; rsi = pre_compute_insns_ptr
    mov rdx, r13        ; rdx = cur_pc 
    mov rcx, r14        ; rcx = cur_instret
    call set_instret_and_pc ; set_instret_and_pc(vm_exec_state_ptr, pre_compute_insns_ptr, cur_pc, cur_instret)
    xor rax, rax        ; set return value to 0
    pop r14             ; pop callee saved registers 
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
    push r14        
    mov rbx, rdi    
    mov rbp, rsi    
    mov r13, rdx    
    mov r14, rcx    

asm_execute:
    mov rdi, r14    
    mov rsi, r13     
    mov rdx, rbx    
    call should_suspend 
    cmp rax, 1     
    je asm_run_end   
    mov rdi, rbx    
    mov rsi, rbp    
    mov rdx, r13
    mov rcx, r14
    call extern_handler
    add r14, 1
    cmp rax, 1
    je asm_run_end
    mov r13, rax
    jmp asm_execute

asm_run_end:
    mov rdi, rbx
    mov rsi, rbp 
    mov rdx, r13 
    mov rcx, r14
    call set_instret_and_pc
    xor rax, rax
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
