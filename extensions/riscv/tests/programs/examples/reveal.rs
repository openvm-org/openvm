#![cfg_attr(any(target_os = "none", target_os = "openvm"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use openvm::io::{reveal_bytes32, reveal_u64};

openvm::entry!(main);

pub fn main() {
    let mut bytes = [0u8; 32];
    for (i, byte) in bytes.iter_mut().enumerate() {
        *byte = i as u8;
    }
    reveal_bytes32(bytes);
    let x: u64 = core::hint::black_box(123);
    let y: u64 = core::hint::black_box(456);
    reveal_u64(x, 4);
    reveal_u64(y, 5);
}
