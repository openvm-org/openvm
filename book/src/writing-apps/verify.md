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

Verifying a proof at the EVM level requires just the proof, as the command uses the verifier generated when `cargo openvm setup` was called.

```bash
cargo openvm verify evm --proof <path_to_proof>
```

If `proof` is omitted, the command will search for the proof at `./openvm/evm.proof`.

As with all other EVM-level commands, `cargo openvm setup` is a prerequisite for `verify`.
> ⚠️ **WARNING**  
> `cargo openvm setup` requires very large amounts of computation and memory (~200 GB).
