// use crate::custom_insn_i;

use alloc::vec::Vec;

use axvm_platform::{custom_insn_i, intrinsics::CUSTOM_0};

/// Store the next 4 bytes from the hint stream to [[rd] + imm]_2.
#[macro_export]
macro_rules! hint_store_u32 {
    ($x:ident, $imm:expr) => {
        custom_insn_i!(CUSTOM_0, 0b001, $x, "x0", $imm)
    };
}

/// Read the next 4 bytes from the hint stream.
pub fn read_u32() -> u32 {
    let mut x: u32;
    custom_insn_i!(CUSTOM_0, 0b001, x, "x0", 0);
    x
}

/// Read the next `len` bytes from the hint stream into a vector.
pub fn read_vec(len: usize) -> Vec<u8> {
    let mut vec = Vec::with_capacity(len);
    // we probably need to enforce len % 4 == 0 somewhere
    for _ in 0..len / 4 {
        vec.extend(read_u32().to_le_bytes());
    }
    vec
}

/// Read `size: u32` and then `size` bytes from the hint stream into a vector.
pub fn read_size_and_vec() -> Vec<u8> {
    read_vec(read_u32() as usize)
}

/// Reset the hint stream with the next hint.
pub fn hint_input() {
    custom_insn_i!(CUSTOM_0, 0b011, "x0", "x0", 0);
}
