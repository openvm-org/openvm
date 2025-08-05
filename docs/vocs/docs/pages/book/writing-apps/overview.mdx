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

For more information see the [build](./build.md) and [run](./run.md) docs.

### Inputs

The `--input` field needs to either be a single hex string or a file path to a json file that contains the key `input` and an array of hex strings. Also note that if you need to provide multiple input streams, you have to use the file path option.
Each hex string (either in the file or as the direct input) is either:

- Hex string of bytes, which is prefixed with `0x01`
- Hex string of native field elements (represented as u32, little endian), prefixed with `0x02`

If you are providing input for a struct of type `T` that will be deserialized by the `openvm::io::read()` function, then the corresponding hex string should be prefixed by `0x01` followed by the serialization of `T` into bytes according to `openvm::serde::to_vec`. The serialization will serialize primitive types (e.g., `u8, u16, u32, u64`) into little-endian bytes. All serialized bytes are zero-padded to a multiple of `4` byte length. For more details on how to serialize complex types into a VM-readable format, see the **Using StdIn** section of the [SDK](../advanced-usage/sdk.md#using-stdin) doc.

## Generating a Proof

To generate a proof, you first need to generate a proving and verifying key:

```bash
cargo openvm keygen
```

If you are using custom VM extensions, this will depend on the `openvm.toml` file which encodes the VM extension configuration; see the [custom extensions](../custom-extensions/overview.md) docs for more information about `openvm.toml`. After generating the keys, you can generate a proof by running:

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
