#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use core::arch::asm;

openvm::entry!(main);

pub fn main() {
    unsafe {
        asm!("lui t0, 0x40000");
        asm!("lw x0, 0(t0)");
    }
}
