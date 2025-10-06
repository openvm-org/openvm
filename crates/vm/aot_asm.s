.intel_syntax noprefix
.code64
.section .text
.extern ADD_RV32
.global main

main:
    push rbp      
    mov rbp, rsp    
    and rsp, -16   
	
	mov rdi, 1
	mov rsi, 2
	mov rdx, 3
	mov rcx, 1

	call ADD_RV32

    pop rbp
    xor eax, eax      
    
	ret
