#![no_main]
#![no_std]

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

core::arch::global_asm! {
    "
.text
main:
        li a0, 15
        li a1, 0
        li a2, 1
        j loop
loop:
        beq a0, zero, exit
        addi a0, a0, -1
        add a3, a1, a2
        add a1, zero, a2
        add a2, zero, a3
        j loop

exit:
        # Exit program
        li a7, 10
        ecall
    "
}
