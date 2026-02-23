#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

openvm::entry!(main);

pub fn main() {
    let mut x: u32 = core::hint::black_box(837799);
    let mut count: u32 = 0;

    while count < 1000 && x != 1 {
        if x.is_multiple_of(2) {
            x /= 2;
        } else {
            x = 3 * x + 1;
        }
        count += 1;
    }

    // https://en.wikipedia.org/wiki/Collatz_conjecture#Empirical_data
    if count != 524 {
        openvm::process::panic();
    }
}
