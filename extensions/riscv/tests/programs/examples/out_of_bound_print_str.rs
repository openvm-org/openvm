#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use core::hint::black_box;

openvm::entry!(main);

pub fn main() {
    // The pointer is valid, but the full RV64 length exceeds guest memory.
    // This must reach the memory-range check rather than being narrowed to u32.
    let ptr = black_box(0x400usize) as *const u8;
    let len = black_box(1usize << 32);
    #[cfg(any(openvm_intrinsics, target_os = "openvm"))]
    openvm_riscv_guest::raw_print_str_from_bytes(ptr, len);
    #[cfg(not(any(openvm_intrinsics, target_os = "openvm")))]
    let _ = (ptr, len);
}
