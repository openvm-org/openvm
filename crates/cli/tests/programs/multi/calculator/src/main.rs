#![cfg_attr(any(target_os = "none", target_os = "openvm"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use calculator::count_primes;
openvm::entry!(main);

pub fn main() {
    let n = core::hint::black_box(100);
    let count = count_primes(n);
    if count == 0 {
        panic!();
    }
}
