// [!region dependencies]
use std::fs;

use openvm_build::GuestOptions;
use openvm_sdk::{config::AggregationSystemParams, Sdk, StdIn};
use openvm_stark_sdk::config::{app_params_with_100_bits_security, MAX_APP_LOG_STACKED_HEIGHT};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct SomeStruct {
    pub a: u64,
    pub b: u64,
}
// [!endregion dependencies]

#[allow(dead_code, unused_variables)]
fn read_elf() -> eyre::Result<()> {
    // [!region read_elf]
    // 2b. Load the ELF from a file
    let elf: Vec<u8> = fs::read("your_path_to_elf")?;
    // [!endregion read_elf]
    Ok(())
}

#[allow(unused_variables, unused_doc_comments)]
fn main() -> eyre::Result<()> {
    /// to import example guest code in crate replace `target_path` for:
    /// ```
    /// use std::path::PathBuf;
    ///
    /// let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).to_path_buf();
    /// path.push("guest/fib");
    /// let target_path = path.to_str().unwrap();
    /// ```
    // [!region build]
    // 1. Initialize the SDK with the RV32IM preset and default aggregation parameters.
    let app_params = app_params_with_100_bits_security(MAX_APP_LOG_STACKED_HEIGHT);
    let agg_params = AggregationSystemParams::default();
    let sdk = Sdk::riscv64(app_params, agg_params);

    // 2a. Build the ELF with guest options and a target filter.
    let guest_opts = GuestOptions::default();
    let target_path = "your_path_project_root";
    let elf = sdk.build(guest_opts, target_path, &None, None)?;
    // [!endregion build]

    // [!region transpilation]
    let exe = sdk.convert_to_exe(elf.clone())?;
    // [!endregion transpilation]

    // [!region execution]
    // 3. Format your input into StdIn
    let my_input = SomeStruct { a: 1, b: 2 }; // anything that can be serialized
    let mut stdin = StdIn::default();
    stdin.write(&my_input);

    // 4. Run the program
    let output = sdk.execute(exe.clone(), stdin.clone())?;
    println!("public values output: {output:?}");
    // [!endregion execution]

    // [!region proof_generation]
    // 5a. Generate a proof and verification baseline directly.
    let (proof, baseline) = sdk.prove(exe.clone(), stdin.clone(), &[])?;
    // 5b. Or build a StarkProver with custom fields and generate the baseline separately.
    let mut prover = sdk.prover(exe)?.with_program_name("test_program");
    let baseline = prover.generate_baseline();
    let (proof, _metadata) = prover.prove(stdin.clone(), &[])?;
    // [!endregion proof_generation]

    // [!region verification]
    // 6. Do this once to save the aggregation VK, independent of the proof.
    let (_agg_pk, agg_vk) = sdk.agg_keygen();
    // 7. Verify your program.
    Sdk::verify_proof(agg_vk, baseline, &proof)?;
    // [!endregion verification]

    Ok(())
}
