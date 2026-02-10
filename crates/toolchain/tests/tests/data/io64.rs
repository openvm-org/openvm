use core::arch::naked_asm;

const UART: *mut u8 = 0x1000_0000 as *mut u8;

pub fn uart_put(c: u8) {
    unsafe { core::ptr::write_volatile(UART, c) }
}

pub fn print_str(s: &str) {
    for b in s.bytes() {
        uart_put(b);
    }
}

pub fn print_u64(mut n: u64) {
    if n == 0 {
        uart_put(b'0');
        return;
    }
    let mut buf = [0u8; 20];
    let mut i = 0;
    while n > 0 {
        buf[i] = b'0' + (n % 10) as u8;
        n /= 10;
        i += 1;
    }
    while i > 0 {
        i -= 1;
        uart_put(buf[i]);
    }
}

pub fn exit_qemu(code: u32) -> ! {
    unsafe { core::ptr::write_volatile(0x10_0000 as *mut u32, code) }
    loop {}
}

#[allow(dead_code)]
pub const EXIT_SUCCESS: u32 = 0x5555;
#[allow(dead_code)]
pub const EXIT_FAILURE: u32 = 0x3333;

#[unsafe(no_mangle)]
#[unsafe(naked)]
#[unsafe(link_section = ".text.init")]
pub unsafe extern "C" fn _start() -> ! {
    naked_asm!(
        "la sp, _start",
        "li t0, 0x10000",
        "add sp, sp, t0",
        "call {main}",
        "1: j 1b",
        main = sym super::main,
    )
}
