# SHA-256

The OpenVM SHA-256 extension provides tools for using the SHA-256 hash function. Refer [here](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf) for more details on SHA-256.
The functional part is provided by the `openvm-sha256-guest` crate, which is a guest library that can be used in any OpenVM program.

## Functions for guest code

The OpenVM SHA-256Guest extension provides two functions for using in your guest code:

- `sha256(input: &[u8]) -> [u8; 32]`: Computes the SHA-256 hash of the input data and returns it as an array of 32 bytes.
- `set_sha256(input: &[u8], output: &mut [u8; 32])`: Sets the output to the SHA-256 hash of the input data into the provided output buffer.

See the full example [here](https://github.com/openvm-org/openvm/blob/main/examples/sha256/src/main.rs).

### Example

```rust,no_run,noplayground
{{ #include ../../../examples/sha256/src/main.rs:imports }}
{{ #include ../../../examples/sha256/src/main.rs:main }}
```

To be able to import the `sha256` function, add the following to your `Cargo.toml` file:

```toml
openvm-sha256-guest = { git = "https://github.com/openvm-org/openvm.git" }
hex = { version = "0.4.3", default-features = false, features = ["alloc"] }
```

## External Linking

The SHA-256 guest extension also provides another way to use the intrinsic SHA-256 implementation. It provides a function that is meant to be linked to other external libraries. The external libraries can use this function as a hook for the SHA-256 intrinsic. This is enabled only when the target is `zkvm`.

- `zkvm_sha256_impl(input: *const u8, len: usize, output: *mut u8)`: This function has `C` ABI. It takes in a pointer to the input, the length of the input, and a pointer to the output buffer.

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
[app_vm_config.sha256]
```
