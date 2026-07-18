#![cfg_attr(
    all(not(feature = "std"), any(openvm_intrinsics, target_os = "openvm")),
    no_main
)]
#![cfg_attr(not(feature = "std"), no_std)]

use core::arch::asm;

openvm::entry!(main);

pub fn main() {
    unsafe {
        asm!(
            "j 2f",
            ".word 0x0020006f",
            "2:",
            options(nomem, nostack),
        );
    }
}
