use crate::c_void;

use crate::{read_memory, write_memory, r32};

#[no_mangle]
pub extern "C" fn LOADB_RV32(base: *mut c_void, a: u32, b: u32, c: u32, e: u32, f: u32, g: u32) {
    println!("LOADB_RV32 called with base: {:?}, a: {}, b: {}, c: {}, e: {}, f: {}, g: {}", base, a, b, c, e, f, g);
    if f != 0 {
        let result = sign_extend()
    }
    let b_val = u32::from_le_bytes(read_memory::<4>(base, 1, b));
    let c_val = u32::from_le_bytes(read_memory::<4>(base, e, c));
    let result = AddOp::compute(b_val, c_val);
    let data : [u8; 4] = result.to_le_bytes();
    write_memory::<4>(base, data, 1, a);
}