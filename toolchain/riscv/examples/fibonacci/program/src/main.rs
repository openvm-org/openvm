#![no_main]
#![no_std]

axvm::entry!(main);

pub fn main() {
    let n = 1 << 10;
    let mut a: u32 = 0;
    let mut b: u32 = 1;
    for _ in 1..n {
        let sum = a + b;
        a = b;
        b = sum;
    }
    if a == 0 {
        loop {}
    }
}

// #[panic_handler]
// fn panic(_info: &core::panic::PanicInfo) -> ! {
//     loop {}
// }
//
// core::arch::global_asm! {
//     "
// .text
// main:
//         li a0, 15
//         li a1, 0
//         li a2, 1
//         j loop
// loop:
//         beq a0, zero, exit
//         addi a0, a0, -1
//         add a3, a1, a2
//         add a1, zero, a2
//         add a2, zero, a3
//         j loop
//
// exit:
//         # Exit program
//         li a7, 10
//         ecall
//     "
// }
