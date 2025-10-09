use crate::c_void;

use openvm_rv32im_circuit::{
    AluOp,
    AddOp,
    SubOp,
    XorOp,
    OrOp,
    AndOp,
};

use crate::{read_memory, write_memory};

// base, split_pre_compute_ptr, pc, instret
// instret only increase by 1

#[no_mangle]
pub extern "C" fn ADD_RV32(base: *mut c_void, a: u32, b: u32, c: u32, d : u32, e: u32) {
    println!("ADD_RV32 called with base: {:?}, a: {}, b: {}, c: {}, e: {} but do nothing", base, a, b, c, e);
    let b_val = u32::from_le_bytes(read_memory::<4>(base, 1, b));
    let c_val = u32::from_le_bytes(read_memory::<4>(base, e, c));
    let result = AddOp::compute(b_val, c_val);
    let data : [u8; 4] = result.to_le_bytes();
    write_memory::<4>(base, data, 1, a);
}