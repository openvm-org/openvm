#[no_mangle]
pub extern "C" fn print_debug(reg_num: u64, reg_val: u64, pc_val: u64) {
    println!("currently at pc {}", pc_val);
    println!("register {} has value {}", reg_num, reg_val);
    println!("");
}


