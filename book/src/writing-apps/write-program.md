# Writing a Program

## Writing a guest program

See the example [fibonacci program](https://github.com/openvm-org/openvm-example-fibonacci).

The guest program should be a `no_std` Rust crate. As long as it is `no_std`, you can import any other
`no_std` crates and write Rust as you normally would. Import the `openvm` library crate to use `openvm` intrinsic functions (for example `openvm::io::*`).

More examples of guest programs can be found in the [benchmarks/programs](https://github.com/openvm-org/openvm/tree/main/benchmarks/programs) directory.

## Handling I/O

The program can take input from stdin, with some functions provided by `openvm::io`.

`openvm::io::read` takes from stdin and deserializes it into a generic type `T`, so one should specify the type when calling it:

```rust
let n: u64 = read();
```

`openvm::io::read_vec` will just read a vector and return `Vec<u8>`.

`openvm::io::reveal_bytes32` sets the user public values in the final proof (to be read by the smart contract).

For debugging purposes, `openvm::io::print` and `openvm::io::println` can be used normally, but `println!` will only work if `std` is enabled.

> ⚠️ **WARNING**
>
> The maximum memory address for an OpenVM program is `2^29`. The majority of that (approximately 480-500 MB depending on transpilation) is available to the guest program, but large reads may exceed the maximum memory and thus fail.

## Rust `std` library support

OpenVM supports standard Rust written using the `std` library, with the following limitations that users should be aware of:

- Standard input (e.g., from console) is not supported. Use the `read` methods [above](#handling-io) instead.
- Standard output and standard error (e.g., `println!, eprintln!`) are supported and will _both_ print to the host standard output.
- System randomness calls are supported by default. **Important**: system randomness requests randomness from the host, and the provided randomness is unvalidated.
  Users must be aware of this and only use system randomness in settings where this meets their security requirements. In particular, system randomness should **not** be used for cryptographic purposes.
- Reading of environmental variables will always return `None`.
- Reading of `argc` and `argv` will always return `0`.

The above applies to the Rust `std` library. Users should also be aware that when writing a standard Rust program, usage of external crates that use foreign function interfaces (FFI) may not work as expected.

To use the standard library, you must enable the `"std"` feature in the `openvm` crate. This is **not** one of the default features.

**Note**: If you write a program that only imports `openvm` in `Cargo.toml` but does not import it anywhere in your crate, the Rust linker may optimize away the dependency, which will cause a compile error. To fix this, you may need to explicitly import the `openvm` crate in your code.

### When to use `std` vs `no_std`

Due to the limitations described above, our general recommendation is that developers should write OpenVM library crates as Rust `no_std` libraries when possible (see [below](#writing-no_std-rust)).
Binary crates can generally be written using the standard library, although for more control over the expected behavior, we provide [entrypoints](#no_std-binary-crates) for writing `no_std` binaries.

## Writing `no_std` Rust

OpenVM fully supports `no_std` Rust. We refer to the [Embedded Rust Book](https://docs.rust-embedded.org/book/intro/no-std.html) for a more detailed introduction to `no_std` Rust.

### `no_std` library crates

In a library crate, you should add the following to `lib.rs` to declare your crate as `no_std`:

```rust
// lib.rs
#![no_std]
```

If you want to feature gate the usage of the standard library, you can do so by adding a `"std"` feature to your `Cargo.toml`, where the feature must also enable
the `"std"` feature in the `openvm` crate:

```toml
[features]
std = ["openvm/std"]
```

To tell Rust to selectively enable the standard library, add the following to `lib.rs` (in place of the header above):

```rust
// lib.rs
#![cfg_attr(not(feature = "std"), no_std)]
```

### `no_std` binary crates

In addition to declaring a binary crate `no_std`, there is additional handling that must be done around the `main` function.
First, add the following header to `main.rs`:

```rust
// main.rs
#![no_std]
#![no_main]
```

This tells Rust there is no handler for the `main` function. OpenVM provides a separate entrypoint for the `main` function, with panic handler, via the `openvm::entry!` macro.
You should write a `main` function in the normal way, and add the following to `main.rs`:

```rust
openvm::entry!(main);

fn main() {
    // Your code here
}
```

If you want to feature gate the usage of the standard library, you can add

```toml
[features]
std = ["openvm/std"]
```

to `Cargo.toml` as discussed above. In this case, the `main.rs` header should be modified to:

```rust
// main.rs
#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]
```

and you still need the `openvm::entry!(main)` line.
This tells Rust to use the custom `main` handler when the environment is `no_std`, but to use the Rust `std` library and the standard `main` handler when the feature `"std"` is enabled.

## Building and running

See the [overview](./overview.md) on how to build and run the program.
