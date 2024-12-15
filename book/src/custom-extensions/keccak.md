# OpenVM Keccak256

The OpenVm Keccak256 extension provides tools for using the Keccak-256 hash function. 
The functional part is provided by the `openvm-keccak-guest` crate, which is a guest library that can be used in any OpenVM program. 

## Functions for guest code

The OpenVM Keccak256 Guest extension provides two functions for using in your guest code:

- `keccak256(input: &[u8]) -> [u8; 32]`: Computes the Keccak-256 hash of the input data and returns it as an array of 32 bytes.
- `set_keccak256(input: &[u8], output: &mut [u8; 32])`: Sets the output to the Keccak-256 hash of the input data into the provided output buffer.

### Example:
```rust
#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::vec::Vec;
use openvm::io::read_vec;
use openvm_keccak256_guest::keccak256;

openvm::entry!(main);

pub fn main() {
    let input = Vec::read_vec();
    let expected_output = Vec::read_vec();
    let output = keccak256(&input);
    if output != *expected_output {
        panic!();
    }
}
```

To use the Keccak256 Guest extension, you need to add the following to your `openvm.toml` file:

```toml
[app_vm_config.keccak256]
```

See the another example [here](https://github.com/openvm-org/openvm/blob/main/crates/toolchain/tests/programs/examples/keccak.rs).

## Native Keccak256

Keccak guest extension also provides another way to use the native Keccak-256 implementation. It provides a function that is meant to be linked to other external libraries. The external libraries can use this function as a hook for the Keccak-256 native implementation. Enabled only when the target is `zkvm`.

- `native_keccak256(input: *const u8, len: usize, output: *mut u8)`: This function has `C` ABI. It takes in a pointer to the input, the length of the input, and a pointer to the output buffer.

In the external library, you can do the following:

```rust
extern "C" {
    fn native_keccak256(input: *const u8, len: usize, output: *mut u8);
}

fn keccak256(input: &[u8]) -> [u8; 32] {
    #[cfg(target_os = "zkvm")] {
    let mut output = [0u8; 32];
        unsafe {
            native_keccak256(input.as_ptr(), input.len(), output.as_mut_ptr() as *mut u8);
        }
        output
    }
    #[cfg(not(target_os = "zkvm"))] {
        // Regular Keccak-256 implementation
    }
}
```

