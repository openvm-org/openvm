# App Proof Generation and Verification

To generate an app proof using the SDK, follow the steps below.

## Building and Transpiling a Program

The SDK provides lower-level control over the building and transpiling process.

```rust,no_run,noplayground
{{ #include ../../../../crates/sdk/examples/sdk_app.rs:build }}
{{ #include ../../../../crates/sdk/examples/sdk_app.rs:read_elf}}

{{ #include ../../../../crates/sdk/examples/sdk_app.rs:transpilation }}
```

### Using `SdkVmConfig`

The `SdkVmConfig` struct allows you to specify the extensions and system configuration your VM will use. To customize your own configuration, you can use the `SdkVmConfig::builder()` method and set the extensions and system configuration you want.

```rust,no_run,noplayground
{{ #include ../../../../crates/sdk/examples/sdk_app.rs:vm_config }}
```

## Running a Program

To run your program and see the public value output, you can do the following:

```rust,no_run,noplayground
{{ #include ../../../../crates/sdk/examples/sdk_app.rs:execution }}
```

### Using `StdIn`

The `StdIn` struct allows you to format any serializable type into a VM-readable format by passing in a reference to your struct into `StdIn::write` as above. You also have the option to pass in a `&[u8]` into `StdIn::write_bytes`, or a `&[F]` into `StdIn::write_field` where `F` is the `openvm_stark_sdk::p3_baby_bear::BabyBear` field type.

> **Generating CLI Bytes**  
> To get the VM byte representation of a serializable struct `data` (i.e. for use in the CLI), you can print out the result of `openvm::serde::to_vec(data).unwrap()` in a Rust host program.

## Generating App Proofs

After building and transpiling a program, you can then generate a proof. To do so, you need to commit your `VmExe`, generate an `AppProvingKey`, format your input into `StdIn`, and then generate a proof.

```rust,no_run,noplayground
{{ #include ../../../../crates/sdk/examples/sdk_app.rs:proof_generation }}
```

## Verifying App Proofs

After generating a proof, you can verify it. To do so, you need your verifying key (which you can get from your `AppProvingKey`) and the output of your `generate_app_proof` call.

```rust,no_run,noplayground
{{ #include ../../../../crates/sdk/examples/sdk_app.rs:verification }}
```
