#![cfg_attr(
    all(not(feature = "std"), any(openvm_intrinsics, target_os = "openvm")),
    no_main
)]
#![cfg_attr(not(feature = "std"), no_std)]

openvm::entry!(main);

pub fn main() {
    let address = core::hint::black_box(8u64);
    let value = core::hint::black_box(0x1122_3344_5566_7788u64);
    openvm_riscv_guest::reveal!(address, value, -8);
}
