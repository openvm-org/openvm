use std::str::FromStr;

use ax_circuit_derive::{Chip, ChipUsageGetter};
use axvm_circuit::{
    arch::{
        new_vm::VmExecutor, SystemConfig, SystemExecutor, SystemPeriphery, VmChipComplex,
        VmGenericConfig, VmInventoryError,
    },
    derive::{AnyEnum, InstructionExecutor, VmGenericConfig},
};
use axvm_ecc_circuit::{
    CurveConfig, Rv32WeierstrassConfig, WeierstrassExtension, WeierstrassExtensionExecutor,
    WeierstrassExtensionPeriphery, SECP256K1,
};
use axvm_keccak256_circuit::{Keccak256, Keccak256Executor, Keccak256Periphery};
use axvm_mod_circuit::{
    modular_chip::{SECP256K1_COORD_PRIME, SECP256K1_SCALAR_PRIME},
    ModularExtension, ModularExtensionExecutor, ModularExtensionPeriphery, Rv32ModularConfig,
    Rv32ModularWithFp2Config,
};
use axvm_rv32im_circuit::{
    Rv32I, Rv32IExecutor, Rv32IPeriphery, Rv32Io, Rv32IoExecutor, Rv32IoPeriphery, Rv32M,
    Rv32MExecutor, Rv32MPeriphery,
};
use derive_more::derive::From;
use eyre::Result;
use num_bigint_dig::BigUint;
use p3_baby_bear::BabyBear;
use p3_field::PrimeField32;

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
    let executor = VmExecutor::<F, _>::new(config);
    executor.execute(elf, vec![])?;
    assert!(!executor.config.modular.supported_modulus.is_empty());
    Ok(())
}

#[test]
fn test_modular_runtime() -> Result<()> {
    let elf = build_example_program("little")?;
    let config = Rv32ModularConfig::new(vec![SECP256K1_COORD_PRIME.clone()]);
    let executor = VmExecutor::<F, _>::new(config);
    executor.execute(elf, vec![])?;
    Ok(())
}

#[test]
fn test_complex_runtime() -> Result<()> {
    let elf = build_example_program("complex")?;
    let config = Rv32ModularWithFp2Config::new(vec![SECP256K1_COORD_PRIME.clone()]);
    let executor = VmExecutor::<F, _>::new(config);
    executor.execute(elf, vec![])?;
    Ok(())
}

#[test]
fn test_ec_runtime() -> Result<()> {
    let elf = build_example_program("ec")?;
    let config = Rv32WeierstrassConfig::new(vec![SECP256K1.clone()]);
    let executor = VmExecutor::<F, _>::new(config);
    executor.execute(elf, vec![])?;
    Ok(())
}

#[derive(Clone, Debug, VmGenericConfig)]
pub struct Rv32ModularKeccak256Config {
    #[system]
    pub system: SystemConfig,
    #[extension]
    pub base: Rv32I,
    #[extension]
    pub mul: Rv32M,
    #[extension]
    pub io: Rv32Io,
    #[extension]
    pub modular: ModularExtension,
    #[extension]
    pub keccak: Keccak256,
    #[extension]
    pub weierstrass: WeierstrassExtension,
}

impl Rv32ModularKeccak256Config {
    pub fn new(moduli: Vec<BigUint>, curves: Vec<CurveConfig>) -> Self {
        Self {
            system: SystemConfig::default().with_continuations(),
            base: Default::default(),
            mul: Default::default(),
            io: Default::default(),
            modular: ModularExtension::new(moduli),
            keccak: Default::default(),
            weierstrass: WeierstrassExtension::new(curves),
        }
    }
}

#[test]
fn test_ecdsa_runtime() -> Result<()> {
    let elf = build_example_program("ecdsa")?;
    let config = Rv32ModularKeccak256Config::new(
        vec![
            SECP256K1_COORD_PRIME.clone(),
            SECP256K1_SCALAR_PRIME.clone(),
        ],
        vec![SECP256K1.clone()],
    );
    let executor = VmExecutor::<F, _>::new(config);
    executor.execute(elf, vec![])?;
    Ok(())
}
