// [!region dependencies]
use std::fs;

use eyre::Result;
use openvm_build::GuestOptions;
use openvm_sdk::{Sdk, StdIn};
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
    let elf: Vec<u8> = fs::read("your_path_to_elf")?;
    // [!endregion read_elf]
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
    // [!region build]
    // 1. Build the VmConfig with the extensions needed.
    let sdk = Sdk::riscv32();

    // 2a. Build the ELF with guest options and a target filter.
    let guest_opts = GuestOptions::default();
    let target_path = "your_path_project_root";
    let elf = sdk.build(guest_opts, target_path, &None, None)?;
    // [!endregion build]

    // [!region input]
    // 3. Format your input into StdIn
    let my_input = SomeStruct { a: 1, b: 2 }; // anything that can be serialized
    let mut stdin = StdIn::default();
    stdin.write(&my_input);
    // [!endregion input]

    // [!region evm_verification]
    // 5. Generate the SNARK verifier smart contract
    let verifier = sdk.generate_halo2_verifier_solidity()?;

    // 6. Generate an EVM proof
    // NOTE: this will do app_keygen, agg_keygen, halo2_keygen automatically if they have never been
    // called before. As a consequence, the first call to `prove_evm` will take longer if you do not
    // explicitly call `app_keygen`, `agg_keygen`, and `halo2_keygen` before calling `prove_evm`.
    let proof = sdk.prove_evm(elf, stdin)?;

    // 7. Verify the EVM proof
    Sdk::verify_evm_halo2_proof(&verifier, proof)?;
    // [!endregion evm_verification]

    Ok(())
}
