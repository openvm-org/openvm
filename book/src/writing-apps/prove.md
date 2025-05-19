# Generating Proofs

By default, the `prove` command will both `build` and `keygen` before generating a proof. This looks like:

```bash
cargo openvm prove [app | stark | evm]
```

## Proof Generation

The `prove` CLI command, at its core, uses the options below. `prove` also gets access to all of the options that `run` has (see [Running a Program](../writing-apps/run.md) for more information).

```bash
cargo openvm prove [app | stark | evm]
    --app-pk <path_to_app_pk>
    --exe <path_to_transpiled_program>
    --input <path_to_input>
    --proof <path_to_proof_output>
```

If `--app-pk` and/or `--exe` are not provided, the command will call `keygen` and/or `build` respectively before generating a proof.

If your program doesn't require inputs, you can (and should) omit the `--input` flag.

If `--proof` is not provided then the command will write the proof to `./[app | stark | evm].proof` by default.


The `app` subcommand generates an application-level proof, the `stark` command generates an aggregated root-level proof, while the `evm` command generates an end-to-end EVM proof. For more information on aggregation, see [this specification](https://github.com/openvm-org/openvm/blob/bf8df90b13f4e80bb76dbb71f255a12154c84838/docs/specs/continuations.md).

> ⚠️ **WARNING**
> In order to run the `evm` subcommand, you must have previously called the costly `cargo openvm setup`, which requires very large amounts of computation and memory (~200 GB).

See [EVM Proof Format](./verify.md#evm-proof-json-format) for details on the output format for `cargo openvm prove evm`.


## Key Generation

You may want to generate keys separately - the `keygen` command allows you to do this.

```bash
cargo openvm keygen
    --config <path_to_app_config>
```

Similarly to `build`, `run`, and `prove`, options `--manifest-path`, `--target-dir`, and `--output-dir` are provided.

If `--config` is not specified, the command will search for `openvm.toml` in the manifest directory. If the file isn't found, a default configuration will be used.

The proving and verification key will be written to `${target_dir}/openvm/` (and `--output-dir` if specified).