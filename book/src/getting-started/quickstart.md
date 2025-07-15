# Quickstart

In this section we will build and run a fibonacci program.

## Setup

First, create a new Rust project.

```bash
cargo openvm init fibonacci
```

This will generate an OpenVM-specific starter package. Notice `Cargo.toml` has the following dependency:

```toml
[dependencies]
openvm = { git = "https://github.com/openvm-org/openvm.git", tag = "v1.3.0", features = ["std"] }
```

Note that `std` is not enabled by default, so explicitly enabling it is required.

## The fibonacci program

The `read` function takes input from the stdin (it also works with OpenVM runtime).

```rust
// src/main.rs
use openvm::io::{read, reveal_u32};

fn main() {
    let n: u64 = read();
    let mut a: u64 = 0;
    let mut b: u64 = 1;
    for _ in 0..n {
        let c: u64 = a.wrapping_add(b);
        a = b;
        b = c;
    }
    reveal_u32(a as u32, 0);
    reveal_u32((a >> 32) as u32, 1);
}
```

## Build

To build the program, run:

```bash
cargo openvm build
```

This will output an OpenVM executable file to `./target/openvm/release/fibonacci.vmexe`.

## Keygen

Before generating any proofs, we will also need to generate the proving and verification keys.

```bash
cargo openvm keygen
```

This will output a serialized proving key to `./target/openvm/app.pk` and a verification key to `./target/openvm/app.vk`.

## Proof Generation

Now we are ready to generate a proof! Simply run:

```bash
cargo openvm prove app --input "0x010A00000000000000"
```

The `--input` field is passed to the program which receives it via the `io::read` function.
In our `main.rs` we called `read()` to get `n: u64`. The input here is `n = 10u64` _in little endian_. Note that this value must be padded to exactly 8 bytes (64 bits) and is prefixed with `0x01` to indicate that the input is composed of raw bytes.

The serialized proof will be output to `./fibonacci.app.proof`.

## Proof Verification

Finally, the proof can be verified.

```bash
cargo openvm verify app
```

The process should exit with no errors.

## Runtime Execution

If necessary, the executable can also be run _without_ proof generation. This can be useful for testing purposes.

```bash
cargo openvm run --input "0x010A00000000000000"
```
