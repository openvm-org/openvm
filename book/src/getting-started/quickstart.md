# Quickstart

In this section we will build and run a fibonacci program.

## Setup

First, create a new Rust project.

```bash
cargo init fibonacci
```

Since we are using some nightly features, we need to specify the Rust version. Run `rustup component add rust-src --toolchain nightly-2024-10-30` and create a `rust-toolchain.toml` file with the following content:

```toml
[toolchain]
channel = "nightly-2024-10-30"     # "1.82.0"
components = ["clippy", "rustfmt"]
```

In `Cargo.toml`, add the following dependency:

```toml
openvm = { git = "https://github.com/openvm-org/openvm.git", features = ["std"] }
```

Note that `std` is not enabled by default, so explicitly enabling it is required.

## The fibonacci program

The `read` function takes input from the stdin, and it also works with OpenVM runtime.
```rust
use openvm::io::read;

fn main() {
    let n: u64 = read();
    let mut a: u64 = 0;
    let mut b: u64 = 1;
    for _ in 0..n {
        let c: u64 = a.wrapping_add(b);
        a = b;
        b = c;
    }
    println!("{}", a);
}
```
