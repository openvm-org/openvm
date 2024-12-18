# Verifying Proofs

## Application Level

Verifying a proof at the application level requires both the proof and application verifying key.

```bash
cargo openvm verify app
    --app_vk <path_to_app_vk>
    --proof <path_to_proof>
```

If you omit `--app_vk` and/or `--proof`, the command will search for those files at `./openvm/app.vk` and `./openvm/app.proof` respectively.

Once again, if you omitted `--output` and `--vk_output` in the `keygen` and `prove` commands, you can omit `--app_vk` and `--proof` in the `verify` command.

## EVM Level
EVM level proof setup requires large amounts of computation and memory (~200GB). It is recommended to run this process on a server.

### Install Solc
Install  `solc` `0.8.19` using `svm`
```bash
# Install svm
cargo install --version 0.5.7 svm-rs
# Add the binary to your path
export PATH="$HOME/.cargo/bin:$PATH"

# Install solc 0.8.19
svm install 0.8.19
svm use 0.8.19
```

### Generating the Aggregation Proving Key and EVM Verifier Contract

The workflow for generating an end-to-end EVM proof requires first generating an aggregation proving key and EVM verifier contract. This can be done by running the following command:

```bash
cargo openvm setup
```
> ⚠️ **WARNING**  
> This command requires very large amounts of computation and memory (~200 GB).

Upon a successful run, the command will write `agg.pk` and `verifier.sol` to `~/.openvm/`, where `~` is the directory specified by environment variable `$HOME`. Every command that requires these files will look for them in this directory.

> ⚠️ **WARNING**  
> If the `$HOME` environment variable is not set, this command may fail.

Note that `cargo openvm setup` may attempt to download other files (i.e. KZG parameters) from an AWS S3 bucket into `~/.openvm/`.

This command can take ~20mins on a `m6a.16xlarge` instance due to the keygen time.

## Generating and Verifying an EVM Proof

To generate and verify an EVM proof, you need to run the following commands:

```bash
cargo openvm prove evm --input <path_to_input>
cargo openvm verify evm --proof <path_to_proof>
```

If `proof` is omitted, the `verify` command will search for the proof at `./openvm/evm.proof`.
