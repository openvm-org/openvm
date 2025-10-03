#[no_mangle]
pub extern "C" fn print_message() {
    println!("Hello from Rust!");
}

#[no_mangle]
pub extern "C" fn write_to_vmstate(vm_state_addr: usize) {
    println!("vm_state_addr: 0x{:x}\n", vm_state_addr);

    let vm_state = *vm_state_addr;
    
}