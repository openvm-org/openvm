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
[dependencies]
openvm = { git = "https://github.com/openvm-org/openvm.git", features = ["std"] }
```

Note that `std` is not enabled by default, so explicitly enabling it is required.

You will also need to create a configuration file for the vm to use. This will tell OpenVM how to setup zk-specific parameters and which chips to plug in and make available for use by the program. We'll call ours `openvm.toml`.

```toml
# openvm.toml
[app_fri_params]
log_blowup = 2
num_queries = 42
proof_of_work_bits = 16

[app_vm_config.io]
[app_vm_config.rv32i]
[app_vm_config.rv32m]
range_tuple_checker_sizes = [256, 2048]
```

## The fibonacci program

The `read` function takes input from the stdin (it also works with OpenVM runtime).

```rust
/// src/main.rs
use openvm::io::{read, reveal};

openvm::entry!(main);

fn main() {
    let n: u64 = read();
    let mut a: u64 = 0;
    let mut b: u64 = 1;
    for _ in 0..n {
        let c: u64 = a.wrapping_add(b);
        a = b;
        b = c;
    }
    reveal(a as u32, 0);
    reveal((a >> 32) as u32, 1);
}
```

## Build

To build the program, run:

```bash
cargo openvm build --transpile --transpiler-config openvm.toml --transpile-to outputs/fibonacci.vmexe
```

The argument passed to `--transpile-to` is the name of the executable file to be generated.

## Keygen

Before generating any proofs, we will also need to generate the proving and verification keys.

```bash
cargo openvm keygen --config openvm.toml --output outputs/pk --vk-output outputs/vk
```

## Proof Generation

Now we are ready to generate a proof! Simply run:

```bash
cargo openvm prove app --app-pk outputs/pk --exe outputs/fibonacci.vmexe --input "0x0A00000000000000" --output outputs/proof
```

The `--input` field is passed to the program which receives it via the `io::read` function. Note that this value must be padded to 8 bytes (32 bits) _in little-endian format_ since it represents a field element on the Baby Bear field.

## Proof Verification

Finally, the proof can be verified.

```bash
cargo openvm verify app --app-vk outputs/vk --proof outputs/proof
```

The process should exit quite quickly with no errors.

## Runtime Execution

If necessary, the executable can also be run _without_ proof generation. This can be useful for testing purposes.

```bash
cargo openvm run --exe outputs/fibonacci.vmexe --config openvm.toml --input "0x0A00000000000000"
```
