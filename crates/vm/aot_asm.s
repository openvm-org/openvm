.intel_syntax noprefix
.code64
.section .text
.extern print_message
.extern initialize_vmstate
.extern write_to_vmstate
.global main

main:
    push rbp      
    mov rbp, rsp    
    and rsp, -16   
	
	call initialize_vmstate

	mov rbx, rax 

	mov rdi, rbx 
	call write_to_vmstate
    
    pop rbp
    xor eax, eax      
    ret