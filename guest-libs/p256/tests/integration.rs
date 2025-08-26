use eyre::Result;
use openvm_sdk::{Sdk, StdIn};
use openvm_toolchain_tests::{build_example_program_at_path, get_programs_dir};

#[test]
fn test_p256_with_standard_sdk() -> Result<()> {
    let sdk = Sdk::standard();
    let config = &sdk.app_config().app_vm_config;
    let elf =
        build_example_program_at_path(get_programs_dir!("tests/programs"), "various", config)?;

    let (proof, commit) = sdk.prove(elf, StdIn::default())?;
    Sdk::verify_proof(&sdk.agg_pk().get_agg_vk(), commit, &proof)?;
    Ok(())
}
