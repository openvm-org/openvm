# SHA-2

The OpenVM SHA-2 extension provides tools for using the SHA-256, SHA-512, and SHA-384 hash functions. Refer [here](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf) for more details on the SHA-2 family of hash functions.
The functional part is provided by the `openvm-sha2-guest` crate, which is a guest library that can be used in any OpenVM program.

## Functions for guest code

The OpenVM SHA-2 Guest extension provides three pairs of functions for using in your guest code:

- `sha256(input: &[u8]) -> [u8; 32]`: Computes the SHA-256 hash of the input data and returns it as an array of 32 bytes.
- `set_sha256(input: &[u8], output: &mut [u8; 32])`: Sets the provided output buffer to the SHA-256 hash of the input data.
- `sha512(input: &[u8]) -> [u8; 64]`: Computes the SHA-512 hash of the input data and returns it as an array of 64 bytes.
- `set_sha512(input: &[u8], output: &mut [u8; 64])`: Sets the provided output buffer to the SHA-512 hash of the input data.
- `sha384(input: &[u8]) -> [u8; 48]`: Computes the SHA-384 hash of the input data and returns it as an array of 48 bytes.
- `set_sha384(input: &[u8], output: &mut [u8; 64])`: Sets the first 48 bytes of the provided output buffer to the SHA-384 hash of the input data and sets the rest of the buffer to zero.

See the full example [here](https://github.com/openvm-org/openvm/blob/main/examples/sha2/src/main.rs).

### Example

```rust,no_run,noplayground
{{ #include ../../../examples/sha2/src/main.rs:imports }}
{{ #include ../../../examples/sha2/src/main.rs:main }}
```

To be able to import the `shaXXX` functions, add the following to your `Cargo.toml` file:

```toml
openvm-sha2-guest = { git = "https://github.com/openvm-org/openvm.git" }
hex = { version = "0.4.3", default-features = false, features = ["alloc"] }
```

## External Linking

The SHA-2 guest extension also provides another way to use the intrinsic SHA-2 implementations. It provides functions that are meant to be linked to other external libraries. The external libraries can use these functions as hooks for the SHA-2 intrinsics. This is enabled only when the target is `zkvm`.

- `zkvm_shaXXX_impl(input: *const u8, len: usize, output: *mut u8)`: where `XXX` is `256`, `512`, or `384`. These functions have `C` ABI. They take in a pointer to the input, the length of the input, and a pointer to the output buffer.

In the external library, you can do the following:

```rust
extern "C" {
    fn zkvm_sha256_impl(input: *const u8, len: usize, output: *mut u8);
}

fn sha256(input: &[u8]) -> [u8; 32] {
    #[cfg(target_os = "zkvm")]
    {
        let mut output = [0u8; 32];
        unsafe {
            zkvm_sha256_impl(input.as_ptr(), input.len(), output.as_mut_ptr() as *mut u8);
        }
        output
    }
    #[cfg(not(target_os = "zkvm"))] {
        // Regular SHA-256 implementation
    }
}
```

### Config parameters

For the guest program to build successfully add the following to your `.toml` file:

```toml
[app_vm_config.sha2]
```
