#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use openvm::io::reveal_u64;

openvm::entry!(main);

pub fn main() {
    // Write past the public_values buffer.
    // Default PV size is small.
    // reveal_u64 converts this index to a byte offset into public_values.
    reveal_u64(0xDEADBEEF, 9999);
}
