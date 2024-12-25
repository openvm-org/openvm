# Using the SDK

While the CLI provides a convenient way to build, prove, and verify programs, you may want more fine-grained control over the process. The OpenVM Rust SDK allows you to customize various aspects of the workflow programmatically.

For more information on the basic CLI flow, see [Overview of Basic Usage](../writing-apps/overview.md). Writing a guest program is the same as in the CLI.

## Imports and Setup

If you have a guest program and would like to try running the **host program** specified below, you can do so by adding the following imports and setup at the top of the file. You may need to modify the imports and/or the `SomeStruct` struct to match your program.

```rust,no_run,noplayground
{{ #include ../../../crates/sdk/examples/sdk.rs:dependencies }}
```

## Building and Transpiling a Program

The SDK provides lower-level control over the building and transpiling process.

```rust,no_run,noplayground
{{ #include ../../../crates/sdk/examples/sdk.rs:build }}
{{ #include ../../../crates/sdk/examples/sdk.rs:read_elf}}

{{ #include ../../../crates/sdk/examples/sdk.rs:transpilation }}
```

### Using `SdkVmConfig`

The `SdkVmConfig` struct allows you to specify the extensions and system configuration your VM will use. To customize your own configuration, you can use the `SdkVmConfig::builder()` method and set the extensions and system configuration you want.

```rust,no_run,noplayground
{{ #include ../../../crates/sdk/examples/sdk.rs:vm_config }}
```

## Running a Program

To run your program and see the public value output, you can do the following:

```rust,no_run,noplayground
{{ #include ../../../crates/sdk/examples/sdk.rs:execution }}
```

### Using `StdIn`

The `StdIn` struct allows you to format any serializable type into a VM-readable format by passing in a reference to your struct into `StdIn::write` as above. You also have the option to pass in a `&[u8]` into `StdIn::write_bytes`, or a `&[F]` into `StdIn::write_field` where `F` is the `openvm_stark_sdk::p3_baby_bear::BabyBear` field type.

> **Generating CLI Bytes**  
> To get the VM byte representation of a serializable struct `data` (i.e. for use in the CLI), you can print out the result of `openvm::serde::to_vec(data).unwrap()` in a Rust host program.

## Generating Proofs

After building and transpiling a program, you can then generate a proof. To do so, you need to commit your `VmExe`, generate an `AppProvingKey`, format your input into `StdIn`, and then generate a proof.

```rust,no_run,noplayground
{{ #include ../../../crates/sdk/examples/sdk.rs:proof_generation }}
```

## Verifying Proofs

After generating a proof, you can verify it. To do so, you need your verifying key (which you can get from your `AppProvingKey`) and the output of your `generate_app_proof` call.

```rust,no_run,noplayground
{{ #include ../../../crates/sdk/examples/sdk.rs:verification }}
```

## End-to-end EVM Proof Generation and Verification

Generating and verifying an EVM proof is an extension of the above process.

```rust,no_run,noplayground
{{ #include ../../../crates/sdk/examples/sdk.rs:evm_verification }}
```

> ⚠️ **WARNING**  
> Generating an EVM proof will require a substantial amount of computation and memory. If you have run `cargo openvm setup` and don't need a specialized aggregation configuration, consider deserializing the proving key from the file `~/.openvm/agg.pk` instead of generating it.

> ⚠️ **WARNING**  
> The aggregation proving key `agg_pk` above is large. Avoid cloning it if possible.

Note that `DEFAULT_PARAMS_DIR` is the directory where Halo2 parameters are stored by the `cargo openvm setup` CLI command. For more information on the setup process, see the `EVM Level` section of the [verify](../writing-apps/verify.md) doc.

> ⚠️ **WARNING**  
> `cargo openvm setup` requires very large amounts of computation and memory (~200 GB).
