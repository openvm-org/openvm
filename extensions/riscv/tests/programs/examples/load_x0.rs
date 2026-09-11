#![cfg_attr(
    all(not(feature = "std"), any(openvm_intrinsics, target_os = "openvm")),
    no_main
)]
#![cfg_attr(not(feature = "std"), no_std)]

use core::arch::asm;

openvm::entry!(main);

pub fn main() {
    unsafe {
        // Wild address: 4 GiB, just past the guest region (0..0x1_0000_0000).
        // rd = x0 must not suppress the load's bounds check.
        asm!("li t0, 1", "slli t0, t0, 32", "lw x0, 0(t0)");
    }
}
