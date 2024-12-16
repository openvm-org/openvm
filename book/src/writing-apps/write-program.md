# Writing a Program

## Writing a guest program

See the example [fibonacci program](https://github.com/openvm-org/openvm-example-fibonacci).

The guest program should be a `no_std` Rust crate. As long as it is `no_std`, you can import any other
`no_std` crates and write Rust as you normally would. Import the `openvm` library crate to use `openvm` intrinsic functions (for example `openvm::io::*`).

The guest program also needs `#![no_main]` because `no_std` does not have certain default handlers. These are provided by the `openvm::entry!` macro. You should still create a `main` function, and then add `openvm::entry!(main)` for the macro to set up the function to run as a normal `main` function. While the function can be named anything when `target_os = "zkvm"`, for compatibility with std you should still name the function `main`.

To support both `std` and `no_std` execution, the top of your guest program should have:

```rust
#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]
```

More examples of guest programs can be found in the [benchmarks/programs](https://github.com/openvm-org/openvm/tree/main/benchmarks/programs) directory.

### no-std

Although it's usually ok to use std (like in quickstart), not all std functionalities are supported (e.g., randomness). There might be unexpected runtime errors if one uses std, so it is recommended you develop no_std libraries if possible to reduce surprises.
Even without std, `assert!` and `panic!` can work as normal. To use `std` features, one should add the following to `Cargo.toml` feature sections:

```toml
[features]
std = ["openvm/std"]
```

## Handling I/O

The program can take input from stdin, with some functions provided by `openvm::io`.

`openvm::io::read` takes from stdin and deserializes it into a generic type `T`, so one should specify the type when calling it:

```rust
let n: u64 = read();
```

`openvm::io::read_vec` will just read a vector and return `Vec<u8>`.

`openvm::io::reveal` sends public values to the final proof (to be read by the smart contract).

For debugging purposes, `openvm::io::print` and `openvm::io::println` can be used normally, but `println!` will only work if `std` is enabled.

### Building and running

See the [overview](./overview.md) on how to build and run the program.
