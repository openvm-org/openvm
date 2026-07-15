#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use core::hint::black_box;

use openvm::io::reveal_u64;

openvm::entry!(main);

pub fn main() {
    // Wild address: 8 bytes past the 512 MiB guest region (0..0x2000_0000).
    let wild = black_box(0x2000_0008u32) as *const u64;

    let leaked = unsafe { black_box(*wild) };
    reveal_u64(leaked, 0);
}
