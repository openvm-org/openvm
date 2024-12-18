# OpenVM Toolchain Integration Tests

This crate includes tests for OpenVM toolchain that involve starting from a Rust program, compiling to RISC-V target, transpiling RISC-V ELF to OpenVM executable, and then running the OpenVM executable.

## How to Add a Test

1. Add a new guest program file to [programs/examples](./programs/examples).

See [Writing the Guest Program](../../../benchmarks/README.md#writing-the-guest-program) for more detailed instructions.

The `programs` directory is a single crate to make it easier to add small test programs. The crate is **not** part of the main workspace.
Your IDE will likely not lint or use rust-analyzer on the crate while in the workspace, so you should open a separate IDE workspace from `programs` while writing your guest program.

2. Add a rust test to the current crate in [`src`](./src).

Follow the existing examples. There is a utility function `build_example_program` which will compile the guest program with target set to RISC-V and read the output RISC-V ELF file.
For runtime tests, create a `VmExecutor` with the `VmConfig` you want, and then call `execute`. The `execute` function take in `Into<VmExe<F>>`, which the decoded RISC-V ELF `Elf` struct implements (this `Into` is where transpilation to OpenVM executable happens).

See `basic_tests::test_read_vec_runtime` for an example of running execution with a single input `Vec<u8>` in the input stream. This is read via `read_vec()` in the guest program.

See `basic_tests::test_read_runtime` for an example of serializing a struct to pass as input to the guest program. This is read via `read()` in the guest program.

To keep tests fast, most tests here should only involve `execute` and not proof generation, since proof generation unit tests are covered in the `openvm-circuit` crate. However to run more comprehensive integration tests, you can run proving using the `air_test_with_min_segments` function.

3. Run all tests with `cargo nextest run`.
