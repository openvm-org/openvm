#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use openvm::io::reveal_u32;

openvm::entry!(main);

pub fn main() {
    // Write past the public_values buffer.
    // Default PV size is small.
    // reveal_u32 converts this index to a byte offset into public_values.
    reveal_u32(0xDEADBEEF, 9999);
}
