.text
main:
        li a0, 15
        li a1, 0
        li a2, 1
        jal x1, loop
loop:
        beq a0, zero, exit
        addi a0, a0, -1
        add a3, a1, a2
        add a1, zero, a2
        add a2, zero, a3
        jal x1, loop

exit:
        # Exit program
        li a7, 10
        ecall

