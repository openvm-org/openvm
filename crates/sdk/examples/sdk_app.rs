// ANCHOR: dependencies
use std::{fs, sync::Arc};

use eyre::Result;
use openvm::platform::memory::MEM_SIZE;
use openvm_build::GuestOptions;
use openvm_sdk::{
    config::{AppConfig, SdkVmConfig},
    prover::AppProver,
    Sdk, StdIn,
};
use openvm_stark_sdk::config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, FriParameters};
use openvm_transpiler::elf::Elf;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct SomeStruct {
    pub a: u64,
    pub b: u64,
}
// ANCHOR_END: dependencies

#[allow(dead_code, unused_variables)]
fn read_elf() -> Result<(), Box<dyn std::error::Error>> {
    // ANCHOR: read_elf
    // 2b. Load the ELF from a file
    let elf_bytes = fs::read("your_path_to_elf")?;
    let elf = Elf::decode(&elf_bytes, MEM_SIZE as u32)?;
    // ANCHOR_END: read_elf
    Ok(())
}

#[allow(unused_variables, unused_doc_comments)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ANCHOR: vm_config
    let vm_config = SdkVmConfig::builder()
        .system(Default::default())
        .rv32i(Default::default())
        .rv32m(Default::default())
        .io(Default::default())
        .build();
    // ANCHOR_END: vm_config

    /// to import example guest code in crate replace `target_path` for:
    /// ```
    /// use std::path::PathBuf;
    ///
    /// let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).to_path_buf();
    /// path.push("guest/fib");
    /// let target_path = path.to_str().unwrap();
    /// ```
    // ANCHOR: build
    // 1. Build the VmConfig with the extensions needed.
    let sdk = Sdk::new();

    // 2a. Build the ELF with guest options and a target filter.
    let guest_opts = GuestOptions::default();
    let target_path = "your_path_project_root";
    let elf = sdk.build(
        guest_opts,
        &vm_config,
        target_path,
        &Default::default(),
        None,
    )?;
    // ANCHOR_END: build

    // ANCHOR: transpilation
    // 3. Transpile the ELF into a VmExe
    let exe = sdk.transpile(elf, vm_config.transpiler())?;
    // ANCHOR_END: transpilation

    // ANCHOR: execution
    // 4. Format your input into StdIn
    let my_input = SomeStruct { a: 1, b: 2 }; // anything that can be serialized
    let mut stdin = StdIn::default();
    stdin.write(&my_input);

    // 5. Run the program
    let output = sdk.execute(exe.clone(), vm_config.clone(), stdin.clone())?;
    println!("public values output: {:?}", output);
    // ANCHOR_END: execution

    // ANCHOR: proof_generation
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
    let app_prover = AppProver::<_, BabyBearPoseidon2Engine>::new(
        app_pk.app_vm_pk.clone(),
        app_committed_exe.clone(),
    )
    .with_program_name("test_program");
    let proof = app_prover.generate_app_proof(stdin.clone());
    // ANCHOR_END: proof_generation

    // ANCHOR: verification
    // 10. Verify your program
    let app_vk = app_pk.get_app_vk();
    sdk.verify_app_proof(&app_vk, &proof)?;
    // ANCHOR_END: verification

    Ok(())
}
