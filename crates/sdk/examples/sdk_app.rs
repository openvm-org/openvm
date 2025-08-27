use openvm_build::GuestOptions;
use openvm_sdk::{prover::verify_app_proof, Sdk, StdIn};

#[allow(unused_variables, unused_doc_comments)]
fn main() -> eyre::Result<()> {
    // 1. Build the VmConfig with the extensions needed.
    let sdk = Sdk::standard();

    // 2 Build the ELF with default guest options and target filter.
    let guest_opts = GuestOptions::default();
    let target_path = "your_path_project_root";
    let elf = sdk.build(guest_opts, target_path, &None, None)?;

    // 3. Run the program with default inputs.
    let output = sdk.execute(elf.clone(), StdIn::default())?;
    println!("public values output: {:?}", output);

    // 4. Generate an app proof.
    let mut prover = sdk.app_prover(elf)?.with_program_name("test_program");
    let app_commit = prover.app_commit();
    let proof = prover.prove(StdIn::default())?;

    // 5. Verify your program at the app level.
    verify_app_proof(&sdk.app_pk().get_app_vk(), &proof)?;

    Ok(())
}
