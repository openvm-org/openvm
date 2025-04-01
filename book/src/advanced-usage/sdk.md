# Using the SDK

While the CLI provides a convenient way to build, prove, and verify programs, you may want more fine-grained control over the process. The OpenVM Rust SDK allows you to customize various aspects of the workflow programmatically.

For more information on the basic CLI flow, see [Overview of Basic Usage](../writing-apps/overview.md). Writing a guest program is the same as in the CLI.

## Imports and Setup

If you have a guest program and would like to try running the **host program** specified in the next section, you can do so by adding the following imports and setup at the top of the file. You may need to modify the imports and/or the `SomeStruct` struct to match your program.

```rust,no_run,noplayground
{{ #include ../../../crates/sdk/examples/sdk_app.rs:dependencies }}
```

## Building and Transpiling a Program

The SDK provides lower-level control over the building and transpiling process.

```rust,no_run,noplayground
{{ #include ../../../crates/sdk/examples/sdk_app.rs:build }}
{{ #include ../../../crates/sdk/examples/sdk_app.rs:read_elf}}

{{ #include ../../../crates/sdk/examples/sdk_app.rs:transpilation }}
```

### Using `SdkVmConfig`

The `SdkVmConfig` struct allows you to specify the extensions and system configuration your VM will use. To customize your own configuration, you can use the `SdkVmConfig::builder()` method and set the extensions and system configuration you want.

```rust,no_run,noplayground
{{ #include ../../../crates/sdk/examples/sdk_app.rs:vm_config }}
```

> ℹ️
> When using Rust to write the guest program, the VM system configuration should keep the default value `pointer_max_bits = 29` to match the hardcoded memory limit of the memory allocator. Otherwise, the guest program may fail due to out of bounds memory access in the VM.

## Running a Program

To run your program and see the public value output, you can do the following:

```rust,no_run,noplayground
{{ #include ../../../crates/sdk/examples/sdk_app.rs:execution }}
```

### Using `StdIn`

The `StdIn` struct allows you to format any serializable type into a VM-readable format by passing in a reference to your struct into `StdIn::write` as above. You also have the option to pass in a `&[u8]` into `StdIn::write_bytes`, or a `&[F]` into `StdIn::write_field` where `F` is the `openvm_stark_sdk::p3_baby_bear::BabyBear` field type.

> **Generating CLI Bytes**
> To get the VM byte representation of a serializable struct `data` (i.e. for use in the CLI), you can print out the result of `openvm::serde::to_vec(data).unwrap()` in a Rust host program.

## Generating and Verifying Proofs

There are two types of proofs that you can generate, with the sections below continuing from this point.

- [App Proof](#app-proof): Generates STARK proof(s) of the guest program
- [EVM Proof](#evm-proof): Generates a halo2 proof that can be posted on-chain

## App Proof

### Generating App Proofs

After building and transpiling a program, you can then generate a proof. To do so, you need to commit your `VmExe`, generate an `AppProvingKey`, format your input into `StdIn`, and then generate a proof.

```rust,no_run,noplayground
{{ #include ../../../crates/sdk/examples/sdk_app.rs:proof_generation }}
```

For large guest programs, the program will be proved in multiple continuation segments and the returned `proof: ContinuationVmProof` object consists of multiple STARK proofs, one for each segment.

### Verifying App Proofs

After generating a proof, you can verify it. To do so, you need your verifying key (which you can get from your `AppProvingKey`) and the output of your `generate_app_proof` call.

```rust,no_run,noplayground
{{ #include ../../../crates/sdk/examples/sdk_app.rs:verification }}
```

## EVM Proof

### Setup

To generate an EVM proof, you'll first need to ensure that you have followed the [CLI installation steps](../getting-started/install.md). get the appropriate KZG params by running the following command.

```bash
cargo openvm setup
```

> ⚠️ **WARNING**
>
> `cargo openvm setup` requires very large amounts of computation and memory (~200 GB).

<details>
<summary>Also note that there are additional dependencies for the EVM Proof flow. Click here to view.</summary>

```rust,no_run,noplayground
{{ #include ../../../crates/sdk/examples/sdk_app.rs:dependencies }}
```

</details>

### Keygen

Now, you'll need to generate the app proving key for the next step.

```rust,no_run,noplayground
{{ #include ../../../crates/sdk/examples/sdk_evm.rs:keygen }}
```

> ⚠️ **WARNING**
>
> If you have run `cargo openvm setup` and don't need a specialized aggregation configuration, consider deserializing the proving key from the file `~/.openvm/agg.pk` instead of generating it, to save computation.

### EVM Proof Generation and Verification

You can now run the aggregation keygen, proof, and verification functions for the EVM proof.

**Note**: you **do not** need to generate the app proof with the `generate_app_proof` function, as the EVM proof function will handle this automatically.

```rust,no_run,noplayground
{{ #include ../../../crates/sdk/examples/sdk_evm.rs:evm_verification }}
```

> ⚠️ **WARNING**
> The aggregation proving key `agg_pk` above is large. Avoid cloning it if possible.

Note that `DEFAULT_PARAMS_DIR` is the directory where Halo2 parameters are stored by the `cargo openvm setup` CLI command. For more information on the setup process, see the `EVM Level` section of the [verify](../writing-apps/verify.md) doc.
