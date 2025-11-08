.intel_syntax noprefix
.code64
.section .text
.global asm_run

/*
rbx = aot_vm_exec_state_ptr 
rbp = pre_compute_insns_ptr
r13 = cur_pc 
r12 is currently unused 

Assembly code explanation

asm_run:
    push rbp     ; push callee saved register
    push rbx
    push r12
    push r13
    push r13     ; A dummy push to ensure the stack is 16 bytes aligned
    lea rbx, [rdi + 8]   ; rbx = extern_metered_handler_ptr
    mov rbx, [rbx]       ; rbx = extern_metered_handler
    pinsrq xmm15, rbx, 1 ; xmm15[64..128] = extern_metered_handler
    lea rbx, [rdi + 16]  ; rbx = should_suspend_ptr
    mov rbx, [rbx]       ; rbx = should_suspend
    pinsrq xmm14, rbx, 1 ; xmm14[64..128] = should_suspend
    mov rbx, rdi         ; rbx = rdi = aot_vm_exec_state_ptr 
    mov rbp, rsi         ; rbp = rsi = pre_compute_insns_ptr 
    mov r13, rdx         ; r13 = rdx = from_state_pc 

asm_execute:
    mov rdi, rbx         ; rdi = aot_vm_exec_state_ptr 
    pextrq rax, xmm15, 1 ; rax = extern_metered_handler(aot_vm_exec_state_ptr) TODO: remove this since it's constant at runtime.
    push rax             ; preserve extern_metered_handler
    pextrq rax, xmm14, 1 ; rax = should_suspend(aot_vm_exec_state_ptr)
    push rax             ; preserve should_suspend
    call rax             ; should_suspend(aot_vm_exec_state_ptr)
    pop rdi
    pinsrq xmm14, rdi, 1 ; restore should_suspend
    pop rdi
    pinsrq xmm15, rdi, 1 ; restore extern_metered_handler
    cmp rax, 1           ; if return value of should_suspend is 1 
    je asm_run_end       ; jump to asm_run_end
    mov rdi, rbx         ; rdi = aot_vm_exec_state_ptr
    mov rsi, rbp         ; rsi = pre_compute_insns_ptr
    mov rdx, r13         ; rdx = cur_pc 
    pextrq rax, xmm14, 1 ; rax = should_suspend(aot_vm_exec_state_ptr)
    push rax             ; preserve should_suspend
    pextrq rax, xmm15, 1 ; rax = extern_metered_handler(aot_vm_exec_state_ptr, pre_compute_insns_ptr, cur_pc)
    push rax             ; preserve extern_metered_handler
    call rax             ; rax(aot_vm_exec_state_ptr, pre_compute_insns_ptr, cur_pc)
    pop rdi
    pinsrq xmm15, rdi, 1 ; restore extern_metered_handler
    pop rdi
    pinsrq xmm14, rdi, 1 ; restore should_suspend
    cmp rax, 1           ; if return value of metered_extern_handler is 1 
    je asm_run_end       ; jump to asm_run_end 
    mov r13, rax         ; cur_pc = return value of metered_extern_handler
    jmp asm_execute      ; jump to asm_execute 

asm_run_end:
    mov rdi, rbx        ; rdi = aot_vm_exec_state
    mov rsi, r13        ; rsi = cur_pc
    lea rax, [rdi + 24] ; rax = set_pc_ptr
    mov rax, [rax]      ; rax = set_pc
    call rax            ;vm_exec_state.set_pc(cur_pc)
    xor rax, rax        ; set return value to 0
    pop r13
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
*/

asm_run:
    push rbp     
    push rbx
    push r12
    push r13  
    push r13     
    lea rbx, [rdi + 8]
    mov rbx, [rbx]
    pinsrq xmm15, rbx, 1
    lea rbx, [rdi + 16]
    mov rbx, [rbx]
    pinsrq xmm14, rbx, 1
    mov rbx, rdi  
    mov rbp, rsi  
    mov r13, rdx 

asm_execute: 
    mov rdi, rbx        
    pextrq rax, xmm15, 1
    push rax
    pextrq rax, xmm14, 1
    push rax
    call rax
    pop rdi
    pinsrq xmm14, rdi, 1
    pop rdi
    pinsrq xmm15, rdi, 1
    cmp rax, 1          
    je asm_run_end      
    mov rdi, rbx        
    mov rsi, rbp        
    mov rdx, r13       
    pextrq rax, xmm14, 1
    push rax
    pextrq rax, xmm15, 1
    push rax
    call rax
    pop rdi
    pinsrq xmm15, rdi, 1
    pop rdi
    pinsrq xmm14, rdi, 1
    cmp rax, 1          
    je asm_run_end      
    mov r13, rax        
    jmp asm_execute     

asm_run_end:
    mov rdi, rbx
    mov rsi, r13       
    lea rax, [rbx + 24]
    mov rax, [rax]     
    call rax           
    xor rax, rax        
    pop r13
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret
