#![no_main]
#![no_std]
#![recursion_limit = "10"]

use axvm::*;

axvm::entry!(main);

pub fn main() {
    hint_store_u32!("a0", 0);
}
