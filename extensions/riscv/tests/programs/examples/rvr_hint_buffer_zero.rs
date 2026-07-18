#![cfg_attr(
    all(not(feature = "std"), any(openvm_intrinsics, target_os = "openvm")),
    no_main
)]
#![cfg_attr(not(feature = "std"), no_std)]

use core::arch::asm;

openvm::entry!(main);

pub fn main() {
    let mut output = [0u64; 1];
    unsafe {
        asm!(
            ".insn i 0x0b, 1, {output}, {num_words}, 1",
            output = in(reg) output.as_mut_ptr(),
            num_words = in(reg) 0u64,
            options(nostack),
        );
    }
}
