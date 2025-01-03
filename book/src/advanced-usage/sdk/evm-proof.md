# EVM Proof Generation and Verification

Generating an end-to-end EVM proof that can be used on-chain contains many steps that are similar to previous App Proof section, but we will call some other SDK functions to generate and verify the final EVM proof.

## Setup

To generate an EVM proof, you'll first need to ensure that you have followed the [CLI installation steps](../../getting-started/install.md). get the appropraite KZG params by running the following command.

```bash
cargo openvm setup
```

> ⚠️ **WARNING**  
> Generating an EVM proof will require a substantial amount of computation and memory. If you have run `cargo openvm setup` and don't need a specialized aggregation configuration, consider deserializing the proving key from the file `~/.openvm/agg.pk` instead of generating it.

> ⚠️ **WARNING**  
> `cargo openvm setup` requires very large amounts of computation and memory (~200 GB).

## Building and Transpiling a Program

The SDK provides lower-level control over the building and transpiling process.

```rust,no_run,noplayground
{{ #include ../../../../crates/sdk/examples/sdk_evm.rs:build }}
{{ #include ../../../../crates/sdk/examples/sdk_evm.rs:read_elf}}

{{ #include ../../../../crates/sdk/examples/sdk_evm.rs:transpilation }}
```

### Using `SdkVmConfig`

The `SdkVmConfig` struct allows you to specify the extensions and system configuration your VM will use. To customize your own configuration, you can use the `SdkVmConfig::builder()` method and set the extensions and system configuration you want.

```rust,no_run,noplayground
{{ #include ../../../../crates/sdk/examples/sdk_evm.rs:vm_config }}
```

## Running a Program

To run your program and see the public value output, you can do the following:

```rust,no_run,noplayground
{{ #include ../../../../crates/sdk/examples/sdk_evm.rs:execution }}
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

## End-to-end EVM Proof Generation and Verification

Generating and verifying an EVM proof is an extension of the above process.

```rust,no_run,noplayground
{{ #include ../../../../crates/sdk/examples/sdk_evm.rs:evm_verification }}
```

> ⚠️ **WARNING**  
> The aggregation proving key `agg_pk` above is large. Avoid cloning it if possible.

Note that `DEFAULT_PARAMS_DIR` is the directory where Halo2 parameters are stored by the `cargo openvm setup` CLI command. For more information on the setup process, see the `EVM Level` section of the [verify](../../writing-apps/verify.md) doc.
