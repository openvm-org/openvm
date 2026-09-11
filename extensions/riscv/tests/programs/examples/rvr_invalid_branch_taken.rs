#![cfg_attr(
    all(not(feature = "std"), any(openvm_intrinsics, target_os = "openvm")),
    no_main
)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(any(openvm_intrinsics, target_os = "openvm"))]
use core::arch::asm;

openvm::entry!(main);

pub fn main() {
    #[cfg(any(openvm_intrinsics, target_os = "openvm"))]
    unsafe {
        // beq a0, a1, +2: equal values in distinct registers take the invalid target.
        asm!(
            ".word 0x00b50163",
            in("a0") 1u64,
            in("a1") 1u64,
            options(nomem, nostack),
        );
    }
}
