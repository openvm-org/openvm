#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use openvm::io::print;

#[openvm::entry]
pub fn main() {
    print("Hello, world!");
}
