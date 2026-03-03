#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use core::hint::black_box;

use openvm::io::reveal_u32;

openvm::entry!(main);

pub fn main() {
    // Wild address: 4 bytes past the 512 MiB guest region (0..0x2000_0000).
    let wild = black_box(0x2000_0004u32) as *const u32;

    let leaked = unsafe { black_box(*wild) };
    reveal_u32(leaked, 0);
}
