use std::str::FromStr;

use axvm_circuit::arch::{VmConfig, VmExecutor};
use eyre::Result;
use p3_baby_bear::BabyBear;

use crate::utils::build_example_program;

type F = BabyBear;

#[test]
fn test_moduli_setup_runtime() -> Result<()> {
    let elf = build_example_program("moduli_setup")?;
    let exe = axvm_circuit::arch::instructions::exe::AxVmExe::<F>::from(elf.clone());
    let executor = VmExecutor::<F>::new(
        VmConfig::rv32im().add_modular_support(
            exe.custom_op_config
                .intrinsics
                .field_arithmetic
                .primes
                .iter()
                .map(|s| num_bigint_dig::BigUint::from_str(s).unwrap())
                .collect(),
        ),
    );
    executor.execute(elf, vec![])?;
    assert!(!executor.config.supported_modulus.is_empty());
    Ok(())
}

#[test]
fn test_modular_runtime() -> Result<()> {
    let elf = build_example_program("little")?;
    let executor = VmExecutor::<F>::new(VmConfig::rv32im().add_canonical_modulus());
    executor.execute(elf, vec![])?;
    Ok(())
}

#[test]
fn test_complex_runtime() -> Result<()> {
    let elf = build_example_program("complex")?;
    // TODO[stephenh]: because of some issues w.r.t. moduli indexing, for
    // this test to pass we need to add BN254 twice. Once we separate
    // indexing for EcCurve and PairingCurve, remove the additional add.
    let executor = VmExecutor::<F>::new(
        VmConfig::rv32im()
            .add_canonical_modulus()
            .add_canonical_pairing_curves()
            .add_canonical_pairing_curves(),
    );
    executor.execute(elf, vec![])?;
    Ok(())
}

#[test]
fn test_ec_runtime() -> Result<()> {
    let elf = build_example_program("ec")?;
    let executor = VmExecutor::<F>::new(
        VmConfig::rv32im()
            .add_canonical_modulus()
            .add_canonical_ec_curves(),
    );
    executor.execute(elf, vec![])?;
    Ok(())
}
