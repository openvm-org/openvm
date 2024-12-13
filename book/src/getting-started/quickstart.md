# Quickstart

## Writing a guest program

The guest program should be a `no_std` Rust crate. As long as it is `no_std`, you can import any other
`no_std` crates and write Rust as you normally would. Import the `openvm` library crate to use `openvm` intrinsic functions (for example `openvm::io::*`).

The guest program also needs `#![no_main]` because `no_std` does not have certain default handlers. These are provided by the `openvm::entry!` macro. You should still create a `main` function, and then add `openvm::entry!(main)` for the macro to set up the function to run as a normal `main` function. While the function can be named anything when `target_os = "zkvm"`, for compatibility with testing when `std` feature is enabled (see below), you should still name it `main`.

To support host machine execution, the top of your guest program should have:

```rust
#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]
```

You can find some examples of guest programs in the `benchmarks/programs` directory.

### no-std

By default, the guest program is written in Rust with the `no-std` feature. This means that the program is not allowed to use any standard library features.
But one can also use std?

## Testing the program

### Running on the host machine

To test the program on the host machine, one can use the `std` feature: `cargo run --features std`. `openvm::io::read_vec` and `openvm::io::read` will read from stdin. So for example to run the [fibonacci program](https://github.com/openvm-org/openvm/tree/main/benchmarks/programs/fibonacci):

```bash
printf '\xA0\x86\x01\x00\x00\x00\x00\x00' | cargo run --features std
```

### Running with the OpenVM runtime

*TODO*: point to how to install SDK


```
cargo axiom build
```

-> ELF

Write another program that runs the ELF with openvm:

Build the vm config with the extensions needed.

```rust
let vm_config = SdkVmConfig::builder()
    .system(Default::default())
    .rv32i(Default::default())
    ...
```

Load the ELF

```rust
let sdk = Sdk;
let guest_opts = GuestOptions::default();
let mut pkg_dir = PathBuf::from("path_to_guest_program");
let program = sdk
    .build(guest_opts.clone(), &pkg_dir, &TargetFilter::default())
    .unwrap();
```

Run the program

```rust
let exe = sdk
    .transpile(program, vm_config.transpiler())
    .unwrap();
let input = // make your input data
```

How to run the program???