#![cfg_attr(
    all(not(feature = "std"), any(openvm_intrinsics, target_os = "openvm")),
    no_main
)]
#![cfg_attr(not(feature = "std"), no_std)]

openvm::entry!(main);

pub fn main() {
    critical_section::with(|_| {});
}
