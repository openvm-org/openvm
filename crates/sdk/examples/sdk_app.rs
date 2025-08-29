// [!region dependencies]
use openvm_build::GuestOptions;
use openvm_sdk::{prover::verify_app_proof, Sdk, StdIn};
// [!endregion dependencies]

#[allow(unused_variables, unused_doc_comments)]
fn main() -> eyre::Result<()> {
    // [!region init]
    // 1. Initialize Sdk with the standard configuration.
    let sdk = Sdk::standard();
    // [!endregion init]

    // [!region build]
    // 2 Build the ELF with default guest options and target filter.
    let guest_opts = GuestOptions::default();
    let target_path = "your_path_project_root";
    let elf = sdk.build(guest_opts, target_path, &None, None)?;
    // [!endregion build]

    let stdin = StdIn::default();

    // [!region execution]
    // 3. Run the program with default inputs.
    let output = sdk.execute(elf.clone(), stdin.clone())?;
    println!("public values output: {:?}", output);
    // [!endregion execution]

    // [!region proof_generation]
    // 5. Generate an app proof.
    let mut prover = sdk.app_prover(elf)?.with_program_name("test_program");
    let proof = prover.prove(stdin)?;
    // [!endregion proof_generation]

    // [!region verification]
    // 6. Do this once to save the app_vk, independent of the proof.
    let (_app_pk, app_vk) = sdk.app_keygen();
    // 7. Verify your program.
    verify_app_proof(&app_vk, &proof)?;
    // [!endregion verification]

    Ok(())
}
