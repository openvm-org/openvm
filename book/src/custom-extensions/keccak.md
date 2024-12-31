# OpenVM Keccak256

The OpenVm Keccak256 extension provides tools for using the Keccak-256 hash function.
The functional part is provided by the `openvm-keccak-guest` crate, which is a guest library that can be used in any OpenVM program.

## Functions for guest code

The OpenVM Keccak256 Guest extension provides two functions for using in your guest code:

- `keccak256(input: &[u8]) -> [u8; 32]`: Computes the Keccak-256 hash of the input data and returns it as an array of 32 bytes.
- `set_keccak256(input: &[u8], output: &mut [u8; 32])`: Sets the output to the Keccak-256 hash of the input data into the provided output buffer.

See the full example [here](https://github.com/openvm-org/openvm/blob/main/crates/toolchain/tests/programs/examples/keccak.rs).

### Example:

```rust
use hex::FromHex;
use openvm_keccak256_guest::keccak256;

pub fn main() {
    let test_vectors = [
        ("", "C5D2460186F7233C927E7DB2DCC703C0E500B653CA82273B7BFAD8045D85A470"),
        ("CC", "EEAD6DBFC7340A56CAEDC044696A168870549A6A7F6F56961E84A54BD9970B8A"),
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

To be able to import the `keccak256` function, add the following to your `Cargo.toml` file:

```toml
openvm-keccak256-guest = { git = "https://github.com/openvm-org/openvm.git" }
hex = { version = "0.4.3", default-features = false, features = ["alloc"] }
```

## Native Keccak256

Keccak guest extension also provides another way to use the native Keccak-256 implementation. It provides a function that is meant to be linked to other external libraries. The external libraries can use this function as a hook for the Keccak-256 native implementation. Enabled only when the target is `zkvm`.

- `native_keccak256(input: *const u8, len: usize, output: *mut u8)`: This function has `C` ABI. It takes in a pointer to the input, the length of the input, and a pointer to the output buffer.

In the external library, you can do the following:

```rust
extern "C" {
    fn native_keccak256(input: *const u8, len: usize, output: *mut u8);
}

fn keccak256(input: &[u8]) -> [u8; 32] {
    #[cfg(target_os = "zkvm")]
    {
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

### Config parameters

For the guest program to build successfully add the following to your `.toml` file:

```toml
[app_vm_config.keccak]
```
