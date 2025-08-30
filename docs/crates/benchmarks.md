# Benchmarks

Documentation for the `openvm-benchmarks-*` crates. By default, paths will be referenced from the [`benchmarks`](../../benchmarks) directory.

- Table of Contents
  - [Latest Benchmark Results](#latest-benchmark-results)
  - [How to Add a Benchmark](#how-to-add-a-benchmark)
  - [Running a Benchmark Locally](#running-a-benchmark-locally)
  - [Adding a Benchmark to CI](#adding-a-benchmark-to-ci)
  - [Profiling Execution](#profiling-execution)

## Latest Benchmark Results

Latest benchmark results can be found [here](https://github.com/openvm-org/openvm/blob/benchmark-results/index.md).
These are run via [github workflows](../../.github/workflows/benchmarks.yml) and should always be up to date with the latest `main` branch.

## How to Add a Benchmark

1. Add a new crate to the [guest](../../benchmarks/guest/) directory.
2. Add the [benchmark to CI](#adding-a-benchmark-to-ci).

This is called a "guest program" because it is intended to be run on the OpenVM architecture and
not on the machine doing the compilation (the "host machine"), although we will discuss shortly how you can still test it locally on the host machine.

### Writing the Guest Program

The guest program should be a `no_std` Rust crate. As long as it is `no_std`, you can import any other
`no_std` crates and write Rust as you normally would. Import the `openvm` library crate to use `openvm` intrinsic functions (for example `openvm::io::*`).

The guest program also needs `#![no_main]` because `no_std` does not have certain default handlers. These are provided by the `openvm::entry!` macro. You should still create a `main` function, and then add `openvm::entry!(main)` for the macro to set up the function to run as a normal `main` function. While the function can be named anything when `target_os = "zkvm"`, for compatibility with testing when `std` feature is enabled (see below), you should still name it `main`.

To support host machine execution, the top of your guest program should have:

```rust
#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]
```

You can copy from [fibonacci](../../benchmarks/guest/fibonacci) to get started.
The guest program crate should **not** be included in the main repository workspace. Instead the guest
`Cargo.toml` should have `[workspace]` at the top to keep it standalone. Your IDE will likely not
lint or use rust-analyzer on the crate while in the workspace, so the recommended setup is to open a separate IDE workspace from the directory of the guest program.

### Adding the Benchmark

Our proving benchmarks are written as standalone rust binaries. Add one by making a new file in [bin](../../benchmarks/prove/src/bin) by following the [fibonacci example](../../benchmarks/prove/src/bin/fibonacci.rs). We currently only run aggregation proofs when feature "aggregation" is on (off by default). Any general benchmarking utility functions can be added to the library in [`src`](../../benchmarks/utils/src). There are utility functions `build_bench_program` which compiles the guest program crate with target set to `openvm` and reads the output RISC-V ELF file.
This can then be fed into `bench_from_exe` which will generate a proof of the execution of the ELF (any other `VmExe`) from a given `VmConfig`.

#### Providing Inputs

Inputs must be directly provided to the `bench_from_exe` function: the `input_stream: Vec<Vec<F>>` is a vector of vectors, where `input_stream[i]` will be what is provided to the guest program on the `i`-th call of `openvm::io::read_vec()`. Currently you must manually convert from `u8` to `F` using `FieldAlgebra::from_canonical_u8`.

You can find an example of passing in a single `Vec<u8>` input in [base64_json](../../benchmarks/prove/src/bin/base64_json.rs).

#### Testing the Guest Program

You can test by directly running `cargo run --bin <bench_name>` which will run the program in the OpenVM runtime. For a more convenient dev experience, we created the `openvm` crate such that it will still build and run normally on the host machine. From the guest program root directory, you can run

```bash
cargo run --features std
```

To run the program on host (in normal rust runtime). This requires the std library, which is enabled by the `std` feature. To ensure that your guest program is still `no_std`, you should not make `std` the default feature.

The behavior of `openvm::io::read_vec` and `openvm::io::read` differs when run on OpenVM or the host machine. As mentioned above, when running on OpenVM, the inputs must be provided in the `bench_from_exe` function.
On the host machine, when you run `cargo run --features std`, each `read_vec` call will read bytes to end from stdin. For example here is how you would run the fibonacci guest program:

```bash
# from programs/fibonacci
printf '\xA0\x86\x01\x00\x00\x00\x00\x00' | cargo run --features std
```

(Alternatively, you can temporarily comment out the `read_vec` call and use `include_bytes!` or `include_str!` to directly include your input. Use `core::hint::black_box` to prevent the compiler from optimizing away the input.)

#### Local Builds

By default, if you run `cargo build` or `cargo run` from the guest program root directory, it will
build with target set to your **host** machine, while running `bench_from_exe` in the bench script will build with target set to `openvm`. If you want to directly build for `openvm` (more specifically a special RISC-V target), run `cargo openvm build` and it will output a RISC-V ELF file to `target/riscv32im-risc0-zkvm-elf/release/*`. You can install [cargo-binutils](https://github.com/rust-embedded/cargo-binutils) to be able to disassemble the ELF file:

```bash
rust-objdump -d target/riscv32im-risc0-zkvm-elf/release/openvm-fibonacci-program
```

## Running a Benchmark Locally

Running a benchmark locally is simple. Just run the following command:

```bash
OUTPUT_PATH="metrics.json" cargo run --release --bin <benchmark_name>
```

where `<benchmark_name>.rs` is one of the files in [`src/bin`](../../benchmarks/prove/src/bin).
The `OUTPUT_PATH` environment variable should be set to the file path where you want the collected metrics to be written to. If unset, then metrics are not printed to file.

To run a benchmark with the leaf aggregation, add `--features aggregation` to the above command.

### Markdown Output

To generate a markdown summary of the collected metrics, first install `openvm-prof`:

```bash
cd <repo_root>/crates/prof
cargo install --force --path .
```

Then run the command:

```bash
openvm-prof --json-paths $OUTPUT_PATH
```

This will generate a markdown file to the same path as $OUTPUT_PATH but with a `.md` extension. The `--json-paths` argument can take multiple files, comma separated.
There is also an optional `--prev-json-paths` argument to compare the metrics with a previous run.

### Circuit Flamegraphs

While traditional flamegraphs generated from instrumenting a proving binary run on the host machine are useful,
for more detailed profiling we generate special flamegraphs that visualize VM-specific metrics such as cycle counts and trace cell usage with stack traces.

The benchmark must be run with special configuration so that additional metrics are collected for profiling. Note that the additional metric collection will slow down the benchmark. To run a benchmark with the additional profiling, run the following command:

```bash
OUTPUT_PATH="metrics.json" GUEST_SYMBOLS_PATH="guest.syms" cargo run --release --bin <benchmark_name> --features perf-metrics -- --profiling
```

Add `--features aggregation,perf-metrics` to run with leaf aggregation. The `perf-metrics` feature tells the VM to run with additional metric collection. The `--profiling` CLI argument tells the script to build the guest program with `profile=profiling` so that the guest program is compiled without stripping debug symbols. When the `perf-metrics` feature is enabled, the `GUEST_SYMBOLS_PATH` environment variable must be set to the file path where function symbols of the guest program will be exported. Those symbols are then used to annotate the flamegraph with function names.

After the collected metrics are written to `$OUTPUT_PATH`, these flamegraphs can be generated if you have [inferno-flamegraph](https://crates.io/crates/inferno) installed. Install via

```bash
cargo install inferno
```

Then run

```bash
python <repo_root>/ci/scripts/metric_unify/flamegraph.py $OUTPUT_PATH --guest-symbols $GUEST_SYMBOLS_PATH
```

The flamegraphs will be written to `*.svg` files in `.bench_metrics/flamegraphs` with respect to the repo root.

## Running a Benchmark via Github Actions

Benchmarks are run on every pull request. By default, only the App VM is benchmarked. In order to run the leaf aggregation benchmark, add the `run-benchmark` label to your pull request. To run end-to-end benchmarks for EVM proofs, add the `run-benchmark-e2e` label. Pull request benchmarks do not run with additional profiling for flamegraphs.

You can also manually trigger benchmarks by using workflow dispatch from [this page](https://github.com/openvm-org/openvm/actions/workflows/benchmarks.yml). Here you will have the option to run with leaf aggregation, run end-to-end benchmarks, and run with additional profiling for flamegraphs. If flamegraphs are enabled, the workflow will generate flamegraphs and create a new markdown file in the [`benchmark-results` branch](https://github.com/openvm-org/openvm/tree/benchmark-results) displaying the flamegraphs. You can find a path to the markdown file in the workflow run details.

## Adding a Benchmark to CI

To add the benchmark to CI, update the [ci/benchmark-config.json](../../ci/benchmark-config.json) file and set it's configuration parameters. To make the benchmark run on every PR, follow the existing format with `e2e_bench = false`. To make the benchmark run only when label `run_benchmark_e2e` is present, set `e2e_bench = true` and specify values for `root_log_blowup` and `internal_log_blowup`.

The `benchmarks.yml` file reads this JSON and generates a matrix of inputs for the [.github/workflows/benchmark-call.yml](../../.github/workflows/benchmark-call.yml) file, a reusable workflow for running the benchmark, collecting metrics, and storing and displaying results.

## Execution Benchmarks

The crate [`openvm-benchmarks-execute`](../../benchmarks/execute) contains benchmarks for measuring the raw VM execution performance without proving. It includes a CLI tool that allows running various pre-defined benchmark programs to evaluate execution time. Note that this tool doesn't compile the guest ELF files and requires them to be precompiled before running the benchmarks.

### Using the CLI

The CLI provides several options for running execution benchmarks:

```bash
# Run all benchmark programs
cargo run --package openvm-benchmarks-execute

# List all available benchmark programs
cargo run --package openvm-benchmarks-execute -- --list

# Run specific benchmark programs
cargo run --package openvm-benchmarks-execute -- --programs fibonacci_recursive fibonacci_iterative

# Run all benchmark programs except specified ones
cargo run --package openvm-benchmarks-execute -- --skip keccak256 sha256
```

These benchmarks measure pure execution time without proving, making them useful for isolating performance bottlenecks in the VM runtime itself.

### Updating the ELFs

For execution benchmarks, the ELF files need to be compiled before running the benchmarks. The [`openvm-benchmarks-utils`](../../benchmarks/utils) crate provides a CLI tool to build all the benchmark ELFs:

```bash
# Build all benchmark ELFs
cargo run --package openvm-benchmarks-utils --bin build-elfs --features build-elfs

# Build specific benchmark ELFs
cargo run --package openvm-benchmarks-utils --bin build-elfs --features build-elfs -- fibonacci_recursive fibonacci_iterative

# Skip specific programs
cargo run --package openvm-benchmarks-utils --bin build-elfs --features build-elfs -- --skip keccak256 sha256

# Force rebuild even if ELFs already exist (overwrite)
cargo run --package openvm-benchmarks-utils --bin build-elfs --features build-elfs -- --force

# Set build profile (debug or release)
cargo run --package openvm-benchmarks-utils --bin build-elfs --features build-elfs -- --profile debug
```

## Profiling Execution

The following section discusses traditional profiling of the VM runtime execution, without ZK proving.

### Criterion Benchmarks

Most benchmarks are binaries that run once since proving benchmarks take longer. For smaller benchmarks, such as to benchmark VM runtime, we use Criterion. These are in the [`benches`](../../benchmarks/execute/benches) directory.

```bash
cargo bench --bench fibonacci_execute
cargo bench --bench regex_execute
```

will run the normal criterion benchmark.

We profile using executables without criterion in [`examples`](../../benchmarks/execute/examples). To prevent the ELF build time from being included in the benchmark, we pre-build the ELF using the CLI. Check that the included ELF file in `examples` is up to date before proceeding.

### Flamegraph

To generate flamegraphs, install `cargo-flamegraph` and run:

```bash
cargo flamegraph --example regex_execute --profile=profiling
```

will generate a flamegraph at `flamegraph.svg` without running any criterion analysis.
On MacOS, you will need to run the above command with `sudo`.

### Samply

To use [samply](https://github.com/mstange/samply), install it and then we must first build the executable.

```bash
cargo build --example regex_execute --profile=profiling
```

Then, run:

```bash
samply record ../target/profiling/examples/regex_execute
```

It will open an interactive UI in your browser (currently only Firefox and Chrome are supported).
See the samply github page for more information.
