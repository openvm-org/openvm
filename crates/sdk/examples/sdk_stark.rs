// ANCHOR: dependencies
use std::fs;

use openvm_build::GuestOptions;
use openvm_sdk::{Sdk, StdIn};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct SomeStruct {
    pub a: u64,
    pub b: u64,
}
// ANCHOR_END: dependencies

#[allow(dead_code, unused_variables)]
fn read_elf() -> eyre::Result<()> {
    // ANCHOR: read_elf
    // 2b. Load the ELF from a file
    let elf: Vec<u8> = fs::read("your_path_to_elf")?;
    // ANCHOR_END: read_elf
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
    // ANCHOR: build
    // 1. Build the VmConfig with the extensions needed.
    let sdk = Sdk::riscv32();

    // 2a. Build the ELF with guest options and a target filter.
    let guest_opts = GuestOptions::default();
    let target_path = "your_path_project_root";
    let elf = sdk.build(guest_opts, target_path, &None, None)?;
    // ANCHOR_END: build

    // ANCHOR: execution
    // 3. Format your input into StdIn
    let my_input = SomeStruct { a: 1, b: 2 }; // anything that can be serialized
    let mut stdin = StdIn::default();
    stdin.write(&my_input);

    // 4. Run the program
    let output = sdk.execute(elf.clone(), stdin.clone())?;
    println!("public values output: {:?}", output);
    // ANCHOR_END: execution

    // ANCHOR: proof_generation
    // 5a. Generate a proof
    let (proof, app_commit) = sdk.prove(elf.clone(), stdin.clone())?;
    // 5b. Generate a proof with a StarkProver with custom fields
    let mut prover = sdk.prover(elf)?.with_program_name("test_program");
    let app_commit = prover.app_commit();
    let proof = prover.prove(stdin.clone())?;
    // ANCHOR_END: proof_generation

    // ANCHOR: verification
    // 6. Do this once to save the agg_vk, independent of the proof.
    let (_agg_pk, agg_vk) = sdk.agg_keygen()?;
    // 7. Verify your program
    Sdk::verify_proof(&agg_vk, app_commit, &proof)?;
    // ANCHOR_END: verification

    Ok(())
}
