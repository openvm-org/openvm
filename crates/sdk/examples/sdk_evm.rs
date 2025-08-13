// ANCHOR: dependencies
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
    let sdk = Sdk::riscv32();

    // 2a. Build the ELF with guest options and a target filter.
    let guest_opts = GuestOptions::default();
    let target_path = "your_path_project_root";
    let elf = sdk.build(guest_opts, target_path, &None, None)?;
    // ANCHOR_END: build

    // ANCHOR: transpilation
    // 3. Transpile the ELF into a VmExe
    let exe = sdk.transpile(elf)?;
    // ANCHOR_END: transpilation

    // ANCHOR: execution
    // 4. Format your input into StdIn
    let my_input = SomeStruct { a: 1, b: 2 }; // anything that can be serialized
    let mut stdin = StdIn::default();
    stdin.write(&my_input);
    // ANCHOR_END: execution

    // ANCHOR: keygen
    // 6. Commit the exe
    let app_committed_exe = sdk.commit_app_exe(app_fri_params, exe)?;
    // ANCHOR_END: keygen

    // ANCHOR: evm_verification
    // 9. Generate the SNARK verifier smart contract
    let verifier = sdk.generate_halo2_verifier_solidity()?;

    // 10. Generate an EVM proof
    // NOTE: this will do app_keygen, agg_keygen, halo2_keygen automatically if they have never been
    // called before. As a consequence, the first call to `prove_evm` will take longer if you do not
    // explicitly call `app_keygen`, `agg_keygen`, and `halo2_keygen` before calling `prove_evm`.
    let proof = sdk.prove_evm(app_committed_exe, stdin)?;

    // 11. Verify the EVM proof
    sdk.verify_evm_halo2_proof(&verifier, proof)?;
    // ANCHOR_END: evm_verification

    Ok(())
}
