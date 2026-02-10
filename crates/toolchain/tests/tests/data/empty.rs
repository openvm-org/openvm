#![no_std]
#![no_main]
use core::panic::PanicInfo;

mod io;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

fn main() {}
