# SHA-2

The OpenVM SHA-2 guest library provides access to a set of accelerated SHA-2 family hash functions. Currently, it supports the following:

- SHA-256

## SHA-256

Refer [here](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf) for more details on SHA-256.

For SHA-256, the SHA2 guest library provides two functions for use in your guest code:

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
openvm-sha2 = { git = "https://github.com/openvm-org/openvm.git" }
hex = { version = "0.4.3" }
```

### Config parameters

For the guest program to build successfully add the following to your `.toml` file:

```toml
[app_vm_config.sha256]