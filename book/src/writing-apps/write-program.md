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

### Building and running

*TODO*: point to CLI installation instructions

First we need to build the program targetting OpenVM runtime, and that requires some configuration. Put the following in `app_config.toml`:
```toml
[app_fri_params]
log_blowup = 2
num_queries = 42
proof_of_work_bits = 16

[app_vm_config.io]
[app_vm_config.rv32i]
[app_vm_config.rv32m]
range_tuple_checker_sizes = [256, 2048]
```

And run the following command to build the program:
```
cargo openvm build --transpile --transpiler-config app_config.toml --transpile-to outputs/fibonacci.exe
```

Now we can keygen, prove and verify the program.

```bash
cargo openvm keygen --config app_config.toml --output outputs/pk --vk-output outputs/vk
cargo openvm prove app --app-pk outputs/pk --exe outputs/fibonacci.exe --input "0xA086010000000000" --output outputs/proof
cargo openvm verify app --app-vk outputs/vk --proof outputs/proof
```

## Handling I/O

`openvm::io` provides a few functions to read and write data.

`read` takes from stdin the next vec and deserialize it into a generic type `T`, so one should specify the type when calling it:
```rust
let n: u64 = read();
```

`read_vec` will just read a vector and return `Vec<u8>`.

`reveal` prints the value to stdout, but should be thought of as sending a public value to the final proof.

For debugging purposes, `print` and `println` can be used normally, but `println!` will only work if `std` is enabled.