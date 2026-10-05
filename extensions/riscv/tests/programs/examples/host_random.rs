#![cfg_attr(
    all(not(feature = "std"), any(openvm_intrinsics, target_os = "openvm")),
    no_main
)]
#![cfg_attr(not(feature = "std"), no_std)]

openvm::entry!(main);

extern "C" {
    fn zkvm_random_u64() -> u64;
}

pub fn main() {
    openvm::io::reveal_u64(unsafe { zkvm_random_u64() }, 0);
}
