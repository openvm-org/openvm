# Using the SDK

While the CLI provides a convenient way to build, prove, and verify programs, you may want more fine-grained control over the process. The OpenVM Rust SDK allows you to customize various aspects of the workflow programmatically.

For more information on the basic CLI flow, see [Overview of Basic Usage](../writing-apps/overview.md). Writing a guest program is the same as in the CLI.

## Imports and Setup

If you have a guest program and would like to try running the **host program** specified below, you can do so by adding the following imports and setup at the top of the file. You may need to modify the imports and/or the `SomeStruct` struct to match your program.

```rust
use std::{fs, sync::Arc};
use eyre::Result;
use openvm::platform::memory::MEM_SIZE;
use openvm_build::{GuestOptions, TargetFilter};
use openvm_native_recursion::halo2::utils::CacheHalo2ParamsReader;
use openvm_sdk::{
    config::{AggConfig, AppConfig, SdkVmConfig},
    prover::AppProver,
    Sdk, StdIn,
};
use openvm_stark_sdk::config::FriParameters;
use openvm_transpiler::elf::Elf;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct SomeStruct {
    pub a: u64,
    pub b: u64,
}
```

## Building and Transpiling a Program

The SDK provides lower-level control over the building and transpiling process.

```rust
// 1. Build the VmConfig with the extensions needed.
let sdk = Sdk;
let vm_config = SdkVmConfig::builder()
    .system(Default::default())
    .rv32i(Default::default())
    .io(Default::default())
    .build();

// 2a. Build the ELF with guest options and a target filter.
let guest_opts = GuestOptions::default().with_features(vec!["parallel"]);
let target_filter = TargetFilter::default().with_kind("bin".to_string());
let elf = sdk.build(guest_opts, "your_path_project_root", &target_filter)?;
// 2b. Load the ELF from a file
let elf_bytes = fs::read("your_path_to_elf")?;
let elf = Elf::decode(&elf_bytes, MEM_SIZE as u32)?;

// 3. Transpile the ELF into a VmExe
let exe = sdk.transpile(elf, vm_config.transpiler())?;
```

### Using `SdkVmConfig`

The `SdkVmConfig` struct allows you to specify the extensions and system configuration your VM will use. To customize your own configuration, you can use the `SdkVmConfig::builder()` method and set the extensions and system configuration you want.

## Running a Program
To run your program and see the public value output, you can do the following:

```rust
// 4. Format your input into StdIn
let my_input = SomeStruct { a: 1, b: 2 }; // anything that can be serialized
let mut stdin = StdIn::default();
stdin.write(&my_input);

// 5. Run the program
let output = sdk.execute(exe.clone(), vm_config.clone(), stdin.clone())?;
println!("public values output: {:?}", output);
```

### Using `StdIn`

The `StdIn` struct allows you to format any serializable type into a VM-readable format by passing in a reference to your struct into `StdIn::write` as above. You also have the option to pass in a `&[u8]` into `StdIn::write_bytes`, or a `&[F]` into `StdIn::write_field` where `F` is the `openvm_stark_sdk::p3_baby_bear::BabyBear` field type.

> **Generating CLI Bytes**  
> To get the VM byte representation of a serializable struct `data` (i.e. for use in the CLI), you can print out the result of `openvm::serde::to_vec(data).unwrap()` in a Rust host program.

## Generating Proofs

After building and transpiling a program, you can then generate a proof. To do so, you need to commit your `VmExe`, generate an `AppProvingKey`, format your input into `StdIn`, and then generate a proof.

```rust
// 6. Set app configuration
let app_log_blowup = 2;
let app_fri_params = FriParameters::standard_with_100_bits_conjectured_security(app_log_blowup);
let app_config = AppConfig::new(app_fri_params, vm_config);

// 7. Commit the exe
let app_committed_exe = sdk.commit_app_exe(app_fri_params, exe)?;

// 8. Generate an AppProvingKey
let app_pk = Arc::new(sdk.app_keygen(app_config)?);

// 9a. Generate a proof
let proof = sdk.generate_app_proof(app_pk.clone(), app_committed_exe.clone(), stdin.clone())?;
// 9b. Generate a proof with an AppProver with custom fields
let app_prover = AppProver::new(app_pk.app_vm_pk.clone(), app_committed_exe.clone())
    .with_program_name("test_program");
let proof = app_prover.generate_app_proof(stdin.clone());
```

## Verifying Proofs
After generating a proof, you can verify it. To do so, you need your verifying key (which you can get from your `AppProvingKey`) and the output of your `generate_app_proof` call.

```rust
// 10. Verify your program
let app_vk = app_pk.get_vk();
sdk.verify_app_proof(&app_vk, &proof)?;
```

## End-to-end EVM Proof Generation and Verification

Generating and verifying an EVM proof is an extension of the above process.

```rust
// 11. Generate the aggregation proving key
const DEFAULT_PARAMS_DIR: &str = concat!(env!("HOME"), "/.openvm/params/");
let halo2_params_reader = CacheHalo2ParamsReader::new(DEFAULT_PARAMS_DIR);
let agg_config = AggConfig::default();
let agg_pk = sdk.agg_keygen(agg_config, &halo2_params_reader)?;

// 12. Generate the SNARK verifier contract
let verifier = sdk.generate_snark_verifier_contract(&halo2_params_reader, &agg_pk)?;

// 13. Generate an EVM proof
let proof = sdk.generate_evm_proof(
    &halo2_params_reader,
    app_pk,
    app_committed_exe,
    agg_pk,
    stdin,
)?;

// 14. Verify the EVM proof
let success = sdk.verify_evm_proof(&verifier, &proof);
assert!(success);
```

> ⚠️ **WARNING**  
> Generating an EVM proof will require a substantial amount of computation and memory. If you have run `cargo openvm setup` and don't need a specialized aggregation configuration, consider deserializing the proving key from the file `~/.openvm/agg.pk` instead of generating it.

> ⚠️ **WARNING**  
> The aggregation proving key `agg_pk` above is large. Avoid cloning it if possible.

Note that `DEFAULT_PARAMS_DIR` is the directory where Halo2 parameters are stored by the `cargo openvm setup` CLI command. For more information on the setup process, see the `EVM Level` section of the [verify](../writing-apps/verify.md) doc.

> ⚠️ **WARNING**  
> `cargo openvm setup` requires very large amounts of computation and memory (~200 GB).
