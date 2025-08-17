// [!region dependencies]
use std::{fs, sync::Arc};

use eyre::Result;
use openvm::platform::memory::MEM_SIZE;
use openvm_build::GuestOptions;
use openvm_native_recursion::halo2::utils::CacheHalo2ParamsReader;
use openvm_sdk::{
    config::{AggConfig, AppConfig, SdkVmConfig, SdkVmCpuBuilder},
    DefaultStaticVerifierPvHandler, Sdk, StdIn,
};
use openvm_stark_sdk::config::FriParameters;
use openvm_transpiler::elf::Elf;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct SomeStruct {
    pub a: u64,
    pub b: u64,
}
// [!endregion dependencies]

#[allow(dead_code, unused_variables)]
fn read_elf() -> Result<(), Box<dyn std::error::Error>> {
    // [!region read_elf]
    // 2b. Load the ELF from a file
    let elf_bytes = fs::read("your_path_to_elf")?;
    let elf = Elf::decode(&elf_bytes, MEM_SIZE as u32)?;
    // [!endregion read_elf]
    Ok(())
}

#[allow(unused_variables, unused_doc_comments)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // [!region vm_config]
    let vm_config = SdkVmConfig::builder()
        .system(Default::default())
        .rv32i(Default::default())
        .rv32m(Default::default())
        .io(Default::default())
        .build();
    // [!endregion vm_config]

    /// to import example guest code in crate replace `target_path` for:
    /// ```
    /// use std::path::PathBuf;
    ///
    /// let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).to_path_buf();
    /// path.push("guest/fib");
    /// let target_path = path.to_str().unwrap();
    /// ```
    // [!region build]
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
    // [!endregion build]

    // [!region transpilation]
    // 3. Transpile the ELF into a VmExe
    let exe = sdk.transpile(elf, vm_config.transpiler())?;
    // [!endregion transpilation]

    // [!region execution]
    // 4. Format your input into StdIn
    let my_input = SomeStruct { a: 1, b: 2 }; // anything that can be serialized
    let mut stdin = StdIn::default();
    stdin.write(&my_input);
    // [!endregion execution]

    // [!region keygen]
    // 5. Set app configuration
    let app_log_blowup = 2;
    let app_fri_params = FriParameters::standard_with_100_bits_conjectured_security(app_log_blowup);
    let app_config = AppConfig::new(app_fri_params, vm_config);

    // 6. Commit the exe
    let app_committed_exe = sdk.commit_app_exe(app_fri_params, exe)?;

    // 7. Generate an AppProvingKey
    let app_pk = Arc::new(sdk.app_keygen(app_config)?);
    // [!endregion keygen]

    // [!region evm_verification]
    // 8. Generate the aggregation proving key
    const DEFAULT_PARAMS_DIR: &str = concat!(env!("HOME"), "/.openvm/params/");
    let halo2_params_reader = CacheHalo2ParamsReader::new(DEFAULT_PARAMS_DIR);
    let agg_config = AggConfig::default();
    let agg_pk = sdk.agg_keygen(
        agg_config,
        &halo2_params_reader,
        &DefaultStaticVerifierPvHandler,
    )?;

    // 9. Generate the SNARK verifier smart contract
    let verifier = sdk.generate_halo2_verifier_solidity(&halo2_params_reader, &agg_pk)?;

    // 10. Generate an EVM proof
    let builder = SdkVmCpuBuilder;
    let proof = sdk.generate_evm_proof(
        &halo2_params_reader,
        builder,
        app_pk,
        app_committed_exe,
        agg_pk,
        stdin,
    )?;

    // 11. Verify the EVM proof
    sdk.verify_evm_halo2_proof(&verifier, proof)?;
    // [!endregion evm_verification]

    Ok(())
}
