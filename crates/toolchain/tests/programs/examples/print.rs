#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use axvm::io::print;

axvm::entry!(main);

pub fn main() {
    print("Hello, world!");
}
