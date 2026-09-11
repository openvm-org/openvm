#![cfg_attr(
    all(not(feature = "std"), any(openvm_intrinsics, target_os = "openvm")),
    no_main
)]
#![cfg_attr(not(feature = "std"), no_std)]

use core::arch::asm;

openvm::entry!(main);

pub fn main() {
    let (left, right): (u64, u64);
    unsafe {
        asm!(
            "slli {left}, zero, 63",
            "srli {right}, zero, 63",
            left = lateout(reg) left,
            right = lateout(reg) right,
            options(nomem, nostack),
        );
    }
    assert_eq!((left, right), (0, 0));
}
