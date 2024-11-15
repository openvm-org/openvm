# Benchmarks

## How to Add a Benchmark

1. Add a new crate to the [programs](./programs) directory.
2. Add the [benchmark to CI](#adding-a-benchmark-to-ci).

This is called a "guest program" because it is intended to be run on the axVM architecture and
not on the machine doing the compilation (the "host machine"), although we will discuss shortly how you can still test it locally on the host machine.

### Writing the Guest Program

The guest program should be a `no_std` Rust crate. As long as it is `no_std`, you can import any other
`no_std` crates and write Rust as you normally would. Import the `axvm` library crate to use `axvm` intrinsic functions (for example `axvm::io::*, axvm::intrinsics::*`).

The guest program also needs `#![no_main]` because `no_std` does not have certain default handlers. These are provided by the `axvm::entry!` macro. You should still create a `main` function, and then add `axvm::entry!(main)` for the macro to set up the function to run as a normal `main` function. While the function can be named anything when `target_os = "zkvm"`, for compatibility with testing when `std` feature is enabled (see below), you should still name it `main`.

To support host machine execution, the top of your guest program should have:

```rust
#![cfg_attr(target_os = "zkvm", no_main)]
#![cfg_attr(not(feature = "std"), no_std)]
```

You can copy from [fibonacci](./programs/fibonacci) to get started.
The guest program crate should **not** be included in the main repository workspace. Instead the guest
`Cargo.toml` should have `[workspace]` at the top to keep it standalone. Your IDE will likely not
lint or use rust-analyzer on the crate while in the workspace, so the recommended setup is to open a separate IDE workspace from the directory of the guest program.

### Adding the Benchmark

Our proving benchmarks are written as standalone rust binaries. Add one by making a new file in [bin](./src/bin) by following the [fibonacci example](./bin/fibonacci.rs). We currently only run aggregation proofs when feature "aggregation" is on (off by default). Any general benchmarking utility functions can be added to the library in [`src`](./src). There are utility functions `build_bench_program` which compiles the guest program crate with target set to `axvm` and reads the output RISC-V ELF file.
This can then be fed into `bench_from_exe` which will generate a proof of the execution of the ELF (any any other `AxVmExe`) from a given `VmConfig`.

#### Providing Inputs

Inputs must be directly provided to the `bench_from_exe` function: the `input_stream: Vec<Vec<F>>` is a vector of vectors, where `input_stream[i]` will be what is provided to the guest program on the `i`-th call of `axvm::io::read_vec()`. Currently you must manually convert from `u8` to `F` using `AbstractField::from_canonical_u8`.

You can find an example of passing in a single `Vec<u8>` input in [base64_json](./src/bin/base64_json.rs).

#### Testing the Guest Program

You can test by directly running `cargo run --bin <bench_name>` which will run the program in the axVM runtime. For a more convenient dev experience, we created the `axvm` crate such that it will still build and run normally on the host machine. From the guest program root directory, you can run

```bash
cargo run --features std
```

To run the program on host (in normal rust runtime). This requires the std library, which is enabled by the `std` feature. To ensure that your guest program is still `no_std`, you should not make `std` the default feature.

The behavior of `axvm::io::read_vec` and `axvm::io::read` differs when run on axVM or the host machine. As mentioned above, when running on axVM, the inputs must be provided in the `bench_from_exe` function.
On the host machine, when you run `cargo run --features std`, each `read_vec` call will read bytes to end from stdin. For example here is how you would run the fibonacci guest program:

```bash
# from programs/fibonacci
printf '\xA0\x86\x01\x00\x00\x00\x00\x00' | cargo run --features std
```

(Alternatively, you can temporarily comment out the `read_vec` call and use `include_bytes!` or `include_str!` to directly include your input. Use `core::hint::black_box` to prevent the compiler from optimizing away the input.)

#### Local Builds

By default, if you run `cargo build` or `cargo run` from the guest program root directory, it will
build with target set to your **host** machine, while running `bench_from_exe` in the bench script will build with target set to `axvm`. If you want to directly build for `axvm` (more specifically a special RISC-V target), copy the `.cargo` folder from [here](./programs/revm_contract_deployment/.cargo) to the guest program root directory and uncomment the `.cargo/config.toml` file. (This config is equivalent to what the `build_bench_program` function does behind the scenes.) You can then `cargo build` or `cargo build --release` and it will output a RISC-V ELF file to `target/riscv32im-risc0-zkvm-elf/release/*`. You can install [cargo-binutils](https://github.com/rust-embedded/cargo-binutils) to be able to disassemble the ELF file:

```bash
rust-objdump -d target/riscv32im-risc0-zkvm-elf/release/axvm-fibonacci-program
```

## Adding a Benchmark to CI

To add the benchmark to CI, update the [.github/workflows/benchmark-config.json](../.github/workflows/benchmark-config.json) file and set it's configuration parameters. If you want the benchmark to run on every PR, follow the existing format.

TODO[stephenh]: Allow selectively run benchmarks via labels. Tracked in INT-2602.

The `benchmarks.yml` file reads this JSON and generates a matrix of inputs for the [.github/workflows/benchmark-call.yml](../.github/workflows/benchmark-call.yml) file, a reusable workflow for running the benchmark, collecting metrics, and storing and displaying results.

## Metric Labels

We use the `metrics` crate to collect metrics. Use `gauge!` for timers and `counter!` for numerical counters (e.g., cell count or opcode count). We distinguish different metrics using labels.
The most convenient way to add labels is to couple it with `tracing` spans: On a line like

```rust
info_span!("Fibonacci Program", group = "fibonacci_program").in_scope(|| {
```

The `"Fibonacci Program"` is the label for tracing logs, only for display purposes. The `group = "fibonacci_program"` adds a label `group -> "fibonacci_program"` to any metrics within the span.

Different labels can be added to provide more granularity on the metrics, but the `group` label should always be the top level label used to distinguish different proof workloads.

## Criterion Benchmarks

Most benchmarks are binaries that run once since proving benchmarks take longer. For smaller benchmarks, such as to benchmark VM runtime, we use Criterion. These are in the `benches` directory.

### Usage

```bash
cargo bench --bench fibonacci_execute
```

will run the normal criterion benchmark.

```bash
cargo bench --bench fibonacci_execute -- --profile-time=30
```

will generate a flamegraph report without running any criterion analysis.

Flamegraph reports can be found in `target/criterion/fibonacci/execute/profile/flamegraph.svg` of the repo root directory.
