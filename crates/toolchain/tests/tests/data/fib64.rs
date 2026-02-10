#![no_std]
#![no_main]
use core::panic::PanicInfo;

mod io;
use io::{print_str, print_u64, exit_qemu, EXIT_SUCCESS};

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

fn main() {
    let n = 10u64;
    let result = fibonacci(n);

    print_str("fib(");
    print_u64(n);
    print_str(") = ");
    print_u64(result);
    print_str("\n");

    exit_qemu(EXIT_SUCCESS);
}

fn fibonacci(n: u64) -> u64 {
    if n <= 1 {
        return n;
    }
    let mut a: u64 = 0;
    let mut b: u64 = 1;
    let mut i = 2;
    while i <= n {
        let tmp = a + b;
        a = b;
        b = tmp;
        i += 1;
    }
    b
}
