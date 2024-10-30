// use crate::custom_insn_i;

use alloc::{alloc::Layout, vec::Vec};

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
    let layout = Layout::from_size_align(len, 4).expect("vec is too large");
    let ptr = unsafe { alloc::alloc::alloc(layout) };
    let mut vec = unsafe { Vec::from_raw_parts(ptr, 0, len) };
    let mut x: u32 = 0;
    // Note: if len % 4 != 0, this will discard some last bytes
    for i in 0..len {
        if i % 4 == 0 {
            x = read_u32();
        }
        unsafe {
            ptr.add(i).write_volatile((x & 255) as u8);
        }
        x >>= 8;
    }
    unsafe { vec.set_len(len) };
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
