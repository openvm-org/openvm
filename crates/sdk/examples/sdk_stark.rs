// ANCHOR: dependencies
use std::{fs, path::PathBuf, sync::Arc};

use eyre::Result;
use openvm::platform::memory::MEM_SIZE;
use openvm_build::GuestOptions;
use openvm_sdk::{
    commit::AppExecutionCommit,
    config::{AggStarkConfig, AppConfig, SdkVmConfig},
    fs::write_to_file_json,
    types::VmStarkProofBytes,
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
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).to_path_buf();
    path.push("guest/fib");
    let target_path = path.to_str().unwrap();
    let elf = sdk.build(
        guest_opts,
        &vm_config.clone(),
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
    // ANCHOR_END: execution

    // ANCHOR: keygen
    // 5. Set app configuration
    let app_log_blowup = 2;
    let app_fri_params = FriParameters::standard_with_100_bits_conjectured_security(app_log_blowup);
    let app_config = AppConfig::new(app_fri_params, vm_config.clone());

    // 7. Generate an AppProvingKey
    let app_pk = Arc::new(sdk.app_keygen(app_config)?);
    // ANCHOR_END: keygen

    // 6. Commit the exe
    let app_committed_exe = sdk.commit_app_exe(app_fri_params, exe)?;
    let y = AppExecutionCommit::compute(&vm_config, &app_committed_exe, &app_pk.leaf_committed_exe);

    // ANCHOR: stark_verification
    // 8. Generate the aggregation proving key
    let mut agg_stark_config = AggStarkConfig::default();
    agg_stark_config.internal_fri_params =
        FriParameters::standard_with_100_bits_conjectured_security(3);
    let agg_stark_pk = sdk.agg_stark_keygen(agg_stark_config)?;

    // 9. Generate an EVM proof
    let proof =
        sdk.generate_e2e_stark_proof(app_pk, app_committed_exe, agg_stark_pk.clone(), stdin)?;
    let stark_proof_bytes = VmStarkProofBytes::new(y, proof.clone())?;

    write_to_file_json(
        PathBuf::from(format!("proof2.stark.proof")),
        stark_proof_bytes,
    )?;

    // 10. Verify the E2E Stark proof
    let x = y.app_exe_commit.to_bn254();
    let z = y.app_vm_commit.to_bn254();
    sdk.verify_e2e_stark_proof(&agg_stark_pk, &proof, &x, &z)?;
    // ANCHOR_END: stark_verification

    Ok(())
}
