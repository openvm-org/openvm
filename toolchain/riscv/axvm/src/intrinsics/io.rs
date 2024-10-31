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
#[allow(asm_sub_register)]
pub fn read_u32() -> u32 {
    let ptr = unsafe { alloc::alloc::alloc(Layout::from_size_align(4, 4).unwrap()) };
    let addr = ptr as u32;
    hint_store_u32!(addr, 0);
    let result: u32;
    unsafe {
        core::arch::asm!("lw {rd}, ({rs1})", rd = out(reg) result, rs1 = in(reg) addr);
    }
    result
}

/// Read the next `len` bytes from the hint stream into a vector.
fn read_vec_by_len(len: usize) -> Vec<u8> {
    // Note: this expect message doesn't matter until our panic handler actually cares about it
    let layout = Layout::from_size_align((len + 3) / 4 * 4, 4).expect("vec is too large");
    let ptr = unsafe { alloc::alloc::alloc(layout) };
    let mut x: u32 = 0;
    // Note: if len % 4 != 0, this will discard some last bytes
    for i in 0..len {
        if i % 4 == 0 {
            // TODO: it probably makes sense not to juggle the data between registers and memory here.
            // On the other hand, maybe it's not a big deal.
            x = read_u32();
        }
        unsafe {
            ptr.add(i).write_volatile((x & 255) as u8);
        }
        x >>= 8;
    }
    unsafe { Vec::from_raw_parts(ptr, len, len) }
}

/// Read `size: u32` and then `size` bytes from the hint stream into a vector.
pub fn read_vec() -> Vec<u8> {
    read_vec_by_len(read_u32() as usize)
}

/// Reset the hint stream with the next hint.
pub fn hint_input() {
    custom_insn_i!(CUSTOM_0, 0b011, "x0", "x0", 0);
}
