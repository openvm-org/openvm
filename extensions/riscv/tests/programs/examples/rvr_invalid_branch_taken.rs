#![cfg_attr(
    all(not(feature = "std"), any(openvm_intrinsics, target_os = "openvm")),
    no_main
)]
#![cfg_attr(not(feature = "std"), no_std)]

use core::arch::asm;

openvm::entry!(main);

pub fn main() {
    unsafe {
        // beq x0, x0, +2: taking the invalid target must trap.
        asm!(".word 0x00000163", options(nomem, nostack));
    }
}
