# Keccak256

The Keccak256 extension guest provides a function that is meant to be linked to other external libraries. The external libraries can use this function as a hook for the keccak-256 intrinsic. This is enabled only when the target is `zkvm`.

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
