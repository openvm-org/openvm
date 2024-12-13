**Keccak Guest Library**

To use the Keccak Guest Library, the `openvm-keccak-guest` crate must be imported in your program. The library provides two functions:

- `keccak256`: Computes the Keccak-256 hash of the input data and returns an array of 32 bytes.
- `set_keccak256`: Sets the output to the Keccak-256 hash of the input data.

The key feature of these functions is that they utilize conditional compilation:
- If the `target_os` is `zkvm`, then the functions will use the native `KECCAK256_RV32` circuit.
- If the `target_os` is not `zkvm`, then the functions will use the regular Rust implementation of Keccak-256.

Example:
```rust
#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::vec::Vec;
use core::hint::black_box;

use openvm_keccak256_guest::keccak256;
use hex::FromHex;

openvm::entry!(main);

pub fn main() {
    let test_vectors = [
        ("", "C5D2460186F7233C927E7DB2DCC703C0E500B653CA82273B7BFAD8045D85A470"), // ShortMsgKAT_256 Len = 0
        ("CC", "EEAD6DBFC7340A56CAEDC044696A168870549A6A7F6F56961E84A54BD9970B8A"), // ShortMsgKAT_256 Len = 8
    ];
    for (input, expected_output) in test_vectors.iter() {
        let input = Vec::from_hex(input).unwrap();
        let expected_output = Vec::from_hex(expected_output).unwrap();
        let output = keccak256(&black_box(input));
        if output != *expected_output {
            panic!();
        }
    }
}
```

