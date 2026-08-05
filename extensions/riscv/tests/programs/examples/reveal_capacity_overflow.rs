#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use openvm::io::reveal_u64;

openvm::entry!(main);

pub fn main() {
    // The test config accepts four values; the fifth reveal must fail.
    for value in 0..5 {
        reveal_u64(value);
    }
}
