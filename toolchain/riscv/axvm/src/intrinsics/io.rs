// use crate::custom_insn_i;

use alloc::vec::Vec;

use axvm_platform::{custom_insn_i, intrinsics::CUSTOM_0};

/// Store the next 4 bytes from the hint stream to [[rd] + imm]_2.
#[macro_export]
macro_rules! hint_store_u32 {
    ($rd:literal, $imm:expr) => {
        unsafe { custom_insn_i!(CUSTOM_0, 0b001, $rd, "x0", $imm) }
    };
}

/// Read the next byte from the hint stream.
pub fn read_byte() -> u8 {
    let mut x: u8;
    custom_insn_i!(CUSTOM_0, 0b001, x, "x0", 0);
    x
}

/// Read the next 4 bytes from the hint stream.
pub fn read_u32() -> u32 {
    let mut x = 0;
    for i in 0..4 {
        x |= (read_byte() as u32) << (i * 8);
    }
    x
}

/// Read the next `len` bytes from the hint stream into a vector.
pub fn read_vec(len: usize) -> Vec<u8> {
    let mut vec = Vec::with_capacity(len);
    for _ in 0..len {
        vec.push(read_byte());
    }
    vec
}

/// Read `size: u32` and then `size` bytes from the hint stream into a vector.
pub fn read_size_and_vec() -> Vec<u8> {
    read_vec(read_u32() as usize)
}
