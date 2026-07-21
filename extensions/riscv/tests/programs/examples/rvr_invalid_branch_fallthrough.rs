#![cfg_attr(
    all(not(feature = "std"), any(openvm_intrinsics, target_os = "openvm")),
    no_main
)]
#![cfg_attr(not(feature = "std"), no_std)]

use core::arch::asm;

openvm::entry!(main);

pub fn main() {
    unsafe {
        // bne x0, x0, +2: the invalid target is never taken.
        asm!(".word 0x00001163", options(nomem, nostack));
    }
}
