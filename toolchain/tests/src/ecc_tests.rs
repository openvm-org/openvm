use std::str::FromStr;

use axvm_circuit::{
    arch::{new_vm, ExecutorName, VmConfig, VmExecutor},
    utils::new_air_test_with_min_segments,
};
use axvm_mod_circuit::{modular_chip::SECP256K1_COORD_PRIME, Rv32ModularConfig};
use eyre::Result;
use p3_baby_bear::BabyBear;

use crate::utils::build_example_program;

type F = BabyBear;

#[test]
fn test_moduli_setup_runtime() -> Result<()> {
    let elf = build_example_program("moduli_setup")?;
    let exe = axvm_circuit::arch::instructions::exe::AxVmExe::<F>::from(elf.clone());
    let moduli = exe
        .custom_op_config
        .intrinsics
        .field_arithmetic
        .primes
        .iter()
        .map(|s| num_bigint_dig::BigUint::from_str(s).unwrap())
        .collect();
    let config = Rv32ModularConfig::new(moduli);
    let executor = new_vm::VmExecutor::<F, _>::new(config);
    executor.execute(elf, vec![])?;
    assert!(!executor.config.modular.supported_modulus.is_empty());
    Ok(())
}

#[test]
fn test_modular_runtime() -> Result<()> {
    let elf = build_example_program("little")?;
    let config = Rv32ModularConfig::new(vec![SECP256K1_COORD_PRIME.clone()]);
    let executor = new_vm::VmExecutor::<F, _>::new(config);
    executor.execute(elf, vec![])?;
    Ok(())
}

/*
#[test]
fn test_complex_runtime() -> Result<()> {
    let elf = build_example_program("complex")?;
    let executor = VmExecutor::<F>::new(
        VmConfig::rv32im()
            .add_modular_support(vec![SECP256K1_COORD_PRIME.clone()])
            .add_complex_ext_support(vec![SECP256K1_COORD_PRIME.clone()]),
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

#[test]
fn test_ecdsa_runtime() -> Result<()> {
    let elf = build_example_program("ecdsa")?;
    let executor = VmExecutor::<F>::new(
        VmConfig::rv32im()
            .add_executor(ExecutorName::Keccak256Rv32)
            .add_canonical_modulus()
            .add_canonical_ec_curves(),
    );
    executor.execute(elf, vec![])?;
    Ok(())
}
*/
