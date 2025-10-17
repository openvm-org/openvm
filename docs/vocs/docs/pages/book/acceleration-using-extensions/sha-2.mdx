# SHA-2

The SHA-2 extension guest provides functions that are meant to be linked to other external libraries. The external libraries can use these functions as a hook for SHA-2 intrinsics. This is enabled only when the target is `zkvm`. We support the SHA-256, SHA-512, and SHA-384 hash functions.

- `zkvm_shaXXX_impl(input: *const u8, len: usize, output: *mut u8)` where XXX is one of `256`, `512`, or `384`. These functions have `C` ABI. They take in a pointer to the input, the length of the input, and a pointer to the output buffer.

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
