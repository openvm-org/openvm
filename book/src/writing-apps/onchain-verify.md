# Onchain Verification

## Generating the Aggregation Proving Key and EVM Verifier Contract

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

## Generating and Verifying an EVM Proof

To generate and verify an EVM proof, you need to run the following commands:

```bash
cargo openvm prove evm --input <path_to_input>
cargo openvm verify evm
```

These commands are very similar to their `app` subcommand counterparts. For more information on the `prove` and `verify` commands, see the [prove](./prove.md) and [verify](./verify.md) docs.
