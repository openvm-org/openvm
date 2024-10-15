#![no_main]
#![no_std]

axvm::entry!(main);

pub fn main() {
    let mut n = 1 << 20;
    let mut a: u32 = 0;
    let mut b: u32 = 1;
    let mut x = 0;
    loop {
        let sum = a + b;
        a = b;
        b = sum;
        n -= 1;
        x += 1;
        if n == x {
            unsafe {
                core::arch::asm!(
                    "ecall",
                    in("a7") a
                )
            }
        }
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
