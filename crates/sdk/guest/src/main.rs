#![cfg_attr(target_os = "zkvm", no_main)]
#![cfg_attr(target_os = "zkvm", no_std)]

openvm::entry!(main);

pub fn main() {
    let n = core::hint::black_box(1 << 3);
    let mut a: u32 = 0;
    let mut b: u32 = 1;
    for _ in 1..n {
        let sum = a + b;
        a = b;
        b = sum;
    }
    if a == 0 {
        panic!();
    }

    openvm::io::reveal_u32(a, 0);
    openvm::io::reveal_u32(b, 1);
}
