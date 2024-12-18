# Generating Proofs

Generating a proof using the CLI is simple - first generate a key, then generate your proof. Using command defaults, this looks like:

```bash
cargo openvm keygen
cargo openvm prove [app | evm]
```

## Key Generation

The `keygen` CLI command has the following optional arguments:

```bash
cargo openvm keygen
    --config <path_to_app_config>
    --output <path_to_app_pk>
    --vk_output <path_to_app_vk>
```

If `--config` is not provided, the command will search for `./openvm.toml` and use that as the application configuration if present. If it is not present, a default configuration will be used.

If `--output` and/or `--vk_output` are not provided, the keys will be written to default locations `./openvm/app.pk` and/or `./openvm/app.vk` respectively.

## Proof Generation

The `prove` CLI command has the following optional arguments:

```bash
cargo openvm prove [app | evm]
    --app_pk <path_to_app_pk>
    --exe <path_to_transpiled_program>
    --input <path_to_input>
    --output <path_to_output>
```

If your program doesn't require inputs, you can (and should) omit the `--input` flag.

If `--app_pk` and/or `--exe` are not provided, the command will search for these files in `./openvm/app.pk` and `./openvm/app.vmexe` respectively. Similarly, if `--output` is not provided then the command will write the proof to `./openvm/[app | evm].proof` by default.

The `app` subcommand is used to generate an application-level proof, while the `evm` command generates an end-to-end EVM proof.

> ⚠️ **WARNING**  
> In order to run the `evm` subcommand, you must have previously called the costly `cargo openvm setup`, which requires very large amounts of computation and memory (~200 GB).
