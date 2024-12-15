# Writing a Program

## Writing a guest program

The guest program should be a `no_std` Rust crate. As long as it is `no_std`, you can import any other
`no_std` crates and write Rust as you normally would. Import the `openvm` library crate to use `openvm` intrinsic functions (for example `openvm::io::*`).

The guest program also needs `#![no_main]` because `no_std` does not have certain default handlers. These are provided by the `openvm::entry!` macro. You should still create a `main` function, and then add `openvm::entry!(main)` for the macro to set up the function to run as a normal `main` function. While the function can be named anything when `target_os = "zkvm"`, for compatibility with testing when `std` feature is enabled (see below), you should still name it `main`.

To support host machine execution, the top of your guest program should have:

```rust
#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]
```

Some examples of guest programs are in the [benchmarks/programs](https://github.com/openvm-org/openvm/tree/main/benchmarks/programs) directory.

### no-std

Although it's usually ok to use std (like in quickstart), not all std functionalities are supported (e.g., randomness). There might be unexpected runtime errors if one uses std, so it is recommended you develop no_std libraries if possible to reduce surprises.
Even without std, `assert!` and `panic!` can work as normal.

### reading input

`openvm::io::read_vec` and `openvm::io::read` will read from stdin. `read` takes the next vec and deserialize it into a generic type `T`, so one should specify the type when calling it:
```rust
let n: u64 = read();
```
`read_vec` reads the size of the vec first (`size: u32`) and then `size` bytes into a vector.
