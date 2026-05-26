#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use core::hint::black_box;

use openvm_rv32im_guest::raw_print_str_from_bytes;

openvm::entry!(main);

pub fn main() {
    // Wild address: 4 bytes past the 512 MiB guest region (0..0x2000_0000).
    let wild = black_box(0x2000_0004u32) as *const u8;
    raw_print_str_from_bytes(wild, 4);
}
