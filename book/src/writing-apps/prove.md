# Generating Proofs

Generating a proof using the CLI is simple - first generate a key, then generate your proof. Using command defaults, this looks like:

```bash
cargo openvm keygen
cargo openvm prove [app | stark | evm]
```

## Key Generation

The `keygen` command generates both an application proving and verification key.

```bash
cargo openvm keygen
    --config <path_to_app_config>
```

Similarly to `build`, `run`, and `prove`, options `--manifest-path`, `--target-dir`, and `--output-dir` are provided.

If `--config` is not specified, the command will search for `openvm.toml` in the manifest directory. If the file isn't found, a default configuration will be used.

The proving and verification key will be written to `${target_dir}/openvm/` (and `--output-dir` if specified).

## Proof Generation

The `prove` CLI command, at its core, uses the options below. `prove` gets access to all of the options that `run` has (see [Running a Program](../writing-apps/run.md) for more information).

```bash
cargo openvm prove [app | stark | evm]
    --app-pk <path_to_app_pk>
    --exe <path_to_transpiled_program>
    --input <path_to_input>
    --proof <path_to_proof_output>
```

If `--app-pk` is not provided, the command will search for a proving key at `${target_dir}/openvm/app.pk`.

If `--exe` is not provided, the command will call `build` before generating a proof.

If your program doesn't require inputs, you can (and should) omit the `--input` flag.

If `--proof` is not provided then the command will write the proof to `./${bin_name}.[app | stark | evm].proof` by default, where `bin_name` is the file stem of the executable run.

The `app` subcommand generates an application-level proof, the `stark` command generates an aggregated root-level proof, while the `evm` command generates an end-to-end EVM proof. For more information on aggregation, see [this specification](https://github.com/openvm-org/openvm/blob/bf8df90b13f4e80bb76dbb71f255a12154c84838/docs/specs/continuations.md).

> ⚠️ **WARNING**
> In order to run the `evm` subcommand, you must have previously called the costly `cargo openvm setup`, which requires very large amounts of computation and memory (~200 GB).

See [EVM Proof Format](./verify.md#evm-proof-json-format) for details on the output format for `cargo openvm prove evm`.

## Commit Hashes

To see the commit hash for an executable, you may run:

```bash
cargo openvm commit
    --app-pk <path_to_app_pk>
    --exe <path_to_transpiled_program>
```

The `commit` command has all the auxiliary options that `prove` does, and outputs Bn254 commits for both your executable and VM. Commits are written to `${target_dir}/openvm/` (and `--output-dir` if specified).
