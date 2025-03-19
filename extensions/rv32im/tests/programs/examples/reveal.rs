#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use openvm::io::{reveal_bytes32, reveal_u32};

openvm::entry!(main);

pub fn main() {
    let mut bytes = [0u8; 32];
    for (i, byte) in bytes.iter_mut().enumerate() {
        *byte = i as u8;
    }
    reveal_bytes32(bytes);
    let x: u32 = core::hint::black_box(123);
    let y: u32 = core::hint::black_box(456);
    reveal_u32(x, 8);
    reveal_u32(y, 10);
}
