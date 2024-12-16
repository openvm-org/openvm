# Using the SDK

While the CLI provides a convenient way to build, prove, and verify programs, you may want more fine-grained control over the process. The OpenVM Rust SDK allows you to customize various aspects of the workflow programmatically.

For more information on the basic CLI flow, see [Overview of Basic Usage](./overview.md). Writing a guest program is the same as in the CLI.

## Building and Transpiling a Program

The SDK provides lower-level control over the building and transpiling process. The following should be done in a **host program**.

```rust
use openvm::transpiler::{openvm_platform::memory::MEM_SIZE, elf::Elf};
use openvm_circuit::arch::instructions::exe::OpenVmExe
use openvm_circuit::arch::VmExecutor;
use openvm_sdk::{config::SdkVmConfig, Sdk, StdIn};

let sdk = Sdk;

// 1. Build the VmConfig with the extensions needed.
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
let elf = Elf::decode("your_path_to_elf", MEM_SIZE as u32)?;

// 3. Transpile the ELF into a VmExe
let exe = sdk.transpile(elf, vm_config.transpiler())?;
```

## Running a Program
To run your program and see the public value output, you can do the following:

```rust
// 7. Format your input into StdIn
let my_input = SomeStruct; // anything that can be serialized
let mut stdin = StdIn::default();
stdin.write(&my_input);

// 8. Run the program
let output = sdk.execute(exe, vm_config, input)?;
```

## Generating Proofs

After building and transpiling a program, you can then generate a proof. To do so, you need to commit your `VmExe`, generate an `AppProvingKey`, format your input into `StdIn`, and then generate a proof.

```rust
// 4. Set app configuration
let app_log_blowup = 2;
let app_fri_params = FriParameters::standard_with_100_bits_conjectured_security(app_log_blowup);
let app_config = AppConfig::new(app_fri_params, vm_config);

// 5. Commit the exe
let app_committed_exe = sdk.commit_app_exe(app_fri_params, exe)?;

// 6. Generate an AppProvingKey
let app_pk = sdk.app_keygen(app_config)?;


// 8a. Generate a proof
let proof = sdk.generate_app_proof(app_pk, app_committed_exe, stdin)?;
// 8b. Generate a proof with an AppProver with custom fields
let mut app_prover =
    AppProver::new(app_pk.app_vm_pk.clone(), app_committed_exe)
        .with_program_name(program_name);
let proof = app_prover.generate_app_proof(stdin);
```

## Verifying Proofs
After generating a proof, you can verify it. To do so, you need your verifying key (which you can get from your `AppProvingKey`) and the output of your `generate_app_proof` call.

```rust
// 9. Verify your program
let app_vk = app_pk.get_vk();
sdk.verify_app_proof(&app_vk, &proof)?;
```

## End-to-end EVM Proof Generation and Verification

TODO