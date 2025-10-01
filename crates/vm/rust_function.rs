#[no_mangle]
pub extern "C" fn print_message() {
    println!("Hello from Rust!");
}

#[no_mangle]
pub extern "C" fn print_register(r: u32) {
    println!("register value {}", r);
}
