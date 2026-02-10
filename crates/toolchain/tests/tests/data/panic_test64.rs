#![no_std]
#![no_main]
use core::panic::PanicInfo;

mod io;
use io::{print_str, print_u64, exit_qemu, EXIT_SUCCESS};

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    print_str("PANIC: ");
    if let Some(msg) = info.message().as_str() {
        print_str(msg);
    } else {
        print_str("(formatted message)");
    }
    if let Some(loc) = info.location() {
        print_str(" at line ");
        print_u64(loc.line() as u64);
    }
    print_str("\n");

    exit_qemu(EXIT_SUCCESS);
}

fn main() {
    print_str("Testing panic handler...\n");
    panic!("this is a test panic");
}
