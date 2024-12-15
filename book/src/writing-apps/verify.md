# Verifying Proofs
## SDK
After generating a proof, you can verify it using the SDK. To do so, you need to your verifying key (which you can get from your `AppProvingKey`) and the output of your `generate_app_proof` call.
```rust
// 0. Get your AppProvingKey and proof
let app_pk = ...
let proof = ...

// 1. Verify your program
let result = sdk.verify_app_proof(app_pk.get_vk(), proof)?;
```
If the call returns `Ok(())`, then your proof is valid.

## CLI
To verify a proof using the CLI, you need to provide the verifying key and the proof.
```bash
cargo openvvm verify app --app_vk <path_to_app_vk> --proof <path_to_proof>
```
Once again, if you ommitted `--output` and `--vk_output` in the `keygen` and `prove` commands, you can omit `--app_vk` and `--proof` in the `verify` command.
```bash
cargo openvvm verify app
```