# Overview of Basic Usage

## Writing a Program

The first step to using OpenVM is to write a Rust program that can be executed by an OpenVM virtual machine. Writing a program for OpenVM is very similar to writing a standard Rust program, with a few key differences necessary to support the OpenVM environment. For more detailed information about writing programs, see the [Writing Programs](./write-program.md) guide.

## Building and Transpiling a Program

At this point, you should have a guest program with a `Cargo.toml` file in the root of your project directory. What's next?

The first thing you will want to do is build and transpile your program using the following command:

```bash
cargo openvm build
```

By default this will build the project located in the current directory. To see if it runs correctly, you can try executing it with the following:

```bash
cargo openvm run --input <path_to_input | hex_string>
```

Note if your program doesn't require inputs, you can omit the `--input` flag.

For more information on both commands, see the [build](./build.md) docs.

### Inputs

The `--input` field needs to either be a hex string or a file path to a file that will be read as bytes. Note that if your hex string represents a single number, it should be written in little-endian format (as this is what the VM expects). To see how more complex inputs can be converted into a VM-readable format, see the **Using StdIn** section of the [SDK](../advanced-usage/sdk.md) doc.

## Generating a Proof

Given an app configuration TOML file, you first need to generate a proving and verifying key:

```bash
cargo openvm keygen
```

After generating the keys, you can generate a proof by running:

```bash
cargo openvm prove app --input <path_to_input | hex_string>
```

Again, if your program doesn't require inputs, you can omit the `--input` flag.

For more information on the `keygen` and `prove` commands, see the [prove](./prove.md) doc.

## Verifying a Proof

To verify a proof using the CLI, you need to provide the verifying key and the proof.

```bash
cargo openvm verify app
```

For more information on the `verify` command, see the [verify](./verify.md) doc.

## End-to-end EVM Proof Generation and Verification

The process above details the workflow necessary to build, prove, and verify a guest program at the application level. However, to generate the end-to-end EVM proof, you need to (a) setup the aggregation proving key and verifier contract and (b) generate/verify the proof at the EVM level.

To do (a), you need to run the following command. If you've run it previously on your machine, there is no need to do so again. This will write files necessary for EVM proving in `~/.openvm/`.

```bash
cargo openvm setup
```

> ⚠️ **WARNING**  
> This command requires very large amounts of computation and memory (~200 GB).

To do (b), you simply need to replace `app` in `cargo openvm prove` and `cargo openvm verify` as such:

```bash
cargo openvm prove evm --input <path_to_input | hex_string>
cargo openvm verify evm
```
