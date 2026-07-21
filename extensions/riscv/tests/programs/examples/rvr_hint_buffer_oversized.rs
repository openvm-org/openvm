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
    {
        openvm_riscv_guest::hint_input();
        let mut output = [0u64; 1024];
        unsafe {
            asm!(
                ".insn i 0x0b, 1, {output}, {num_words}, 1",
                output = in(reg) output.as_mut_ptr(),
                num_words = in(reg) 1024u64,
                options(nostack),
            );
        }
    }
}
