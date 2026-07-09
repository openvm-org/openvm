#![cfg_attr(
    all(not(feature = "std"), any(openvm_intrinsics, target_os = "openvm")),
    no_main
)]
#![cfg_attr(not(feature = "std"), no_std)]

use core::hint::black_box;

use openvm_keccak256::keccak256;

openvm::entry!(main);

pub fn main() {
    let input = [0x5au8; 200];
    black_box(keccak256(black_box(&input)));
}
