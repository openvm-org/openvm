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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Build the VmConfig with the extensions needed.
    let sdk = Sdk;
    let vm_config = SdkVmConfig::builder()
        .system(Default::default())
        .rv32i(Default::default())
        .io(Default::default())
        .build();

    // 2a. Build the ELF with guest options and a target filter.
    let guest_opts = GuestOptions::default();
    let target_filter = TargetFilter::default().with_kind("bin".to_string());
    let elf = sdk.build(guest_opts, "./example", &target_filter)?;
    println!("build complete");

    // 3. Transpile the ELF into a VmExe
    let exe = sdk.transpile(elf, vm_config.transpiler())?;
    println!("transpilation done");

    // 4. Format your input into StdIn
    let my_input = SomeStruct { a: 1, b: 2 }; // anything that can be serialized
    let mut stdin = StdIn::default();
    stdin.write(&my_input);

    // 5. Run the program
    let output = sdk.execute(exe.clone(), vm_config.clone(), stdin.clone())?;
    println!("public values output: {:?}", output);

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

    // 10. Verify your program
    let app_vk = app_pk.get_vk();
    sdk.verify_app_proof(&app_vk, &proof)?;

    // // 11. Generate the aggregation proving key
    // const DEFAULT_PARAMS_DIR: &str = concat!(env!("HOME"), "/.openvm/params/");
    // let halo2_params_reader = CacheHalo2ParamsReader::new(DEFAULT_PARAMS_DIR);
    // let agg_config = AggConfig::default();
    // let agg_pk = sdk.agg_keygen(agg_config, &halo2_params_reader)?;
    //
    // // 12. Generate the SNARK verifier contract
    // let verifier = sdk.generate_snark_verifier_contract(&halo2_params_reader, &agg_pk)?;
    //
    // // 13. Generate an EVM proof
    // let proof = sdk.generate_evm_proof(
    //     &halo2_params_reader,
    //     app_pk,
    //     app_committed_exe,
    //     agg_pk,
    //     stdin,
    // )?;
    //
    // // 14. Verify the EVM proof
    // let success = sdk.verify_evm_proof(&verifier, &proof);
    // assert!(success);

    Ok(())
}
