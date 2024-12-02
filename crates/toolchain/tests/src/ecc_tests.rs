use std::str::FromStr;

use ax_circuit_derive::{Chip, ChipUsageGetter};
use axvm_algebra_circuit::{
    ModularExtension, ModularExtensionExecutor, ModularExtensionPeriphery, Rv32ModularConfig,
    Rv32ModularWithFp2Config,
};
use axvm_algebra_transpiler::{Fp2TranspilerExtension, ModularTranspilerExtension};
use axvm_circuit::{
    arch::{
        instructions::exe::AxVmExe, SystemConfig, SystemExecutor, SystemPeriphery, VmChipComplex,
        VmConfig, VmExecutor, VmInventoryError,
    },
    derive::{AnyEnum, InstructionExecutor, VmConfig},
};
use axvm_ecc_circuit::{
    CurveConfig, Rv32WeierstrassConfig, WeierstrassExtension, WeierstrassExtensionExecutor,
    WeierstrassExtensionPeriphery, SECP256K1_CONFIG,
};
use axvm_ecc_transpiler::EccTranspilerExtension;
use axvm_keccak256_circuit::{Keccak256, Keccak256Executor, Keccak256Periphery};
use axvm_keccak256_transpiler::Keccak256TranspilerExtension;
use axvm_rv32im_circuit::{
    Rv32I, Rv32IExecutor, Rv32IPeriphery, Rv32Io, Rv32IoExecutor, Rv32IoPeriphery, Rv32M,
    Rv32MExecutor, Rv32MPeriphery,
};
use axvm_rv32im_transpiler::{
    Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
};
use axvm_transpiler::{transpiler::Transpiler, FromElf};
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
    let axvm_exe = AxVmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension)
            .with_extension(ModularTranspilerExtension),
    );

    let moduli = axvm_exe
        .custom_op_config
        .intrinsics
        .field_arithmetic
        .primes
        .iter()
        .map(|s| num_bigint_dig::BigUint::from_str(s).unwrap())
        .collect();
    let config = Rv32ModularConfig::new(moduli);
    let executor = VmExecutor::<F, _>::new(config);
    executor.execute(axvm_exe, vec![])?;
    assert!(!executor.config.modular.supported_modulus.is_empty());
    Ok(())
}

#[test]
fn test_modular_runtime() -> Result<()> {
    let elf = build_example_program("little")?;
    let axvm_exe = AxVmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension)
            .with_extension(ModularTranspilerExtension),
    );
    let config = Rv32ModularConfig::new(vec![SECP256K1_CONFIG.modulus.clone()]);
    let executor = VmExecutor::<F, _>::new(config);
    executor.execute(axvm_exe, vec![])?;
    Ok(())
}

#[test]
fn test_complex_runtime() -> Result<()> {
    let elf = build_example_program("complex")?;
    let axvm_exe = AxVmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension)
            .with_extension(Fp2TranspilerExtension)
            .with_extension(ModularTranspilerExtension),
    );
    let config = Rv32ModularWithFp2Config::new(vec![SECP256K1_CONFIG.modulus.clone()]);
    let executor = VmExecutor::<F, _>::new(config);
    executor.execute(axvm_exe, vec![])?;
    Ok(())
}

#[test]
fn test_ec_runtime() -> Result<()> {
    let elf = build_example_program("ec")?;
    let axvm_exe = AxVmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension)
            .with_extension(EccTranspilerExtension)
            .with_extension(ModularTranspilerExtension),
    );
    let config = Rv32WeierstrassConfig::new(vec![SECP256K1_CONFIG.clone()]);
    let executor = VmExecutor::<F, _>::new(config);
    executor.execute(axvm_exe, vec![])?;
    Ok(())
}

#[derive(Clone, Debug, VmConfig)]
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
    pub fn new(curves: Vec<CurveConfig>) -> Self {
        let primes: Vec<BigUint> = curves
            .iter()
            .flat_map(|c| [c.modulus.clone(), c.scalar.clone()])
            .collect();
        Self {
            system: SystemConfig::default().with_continuations(),
            base: Default::default(),
            mul: Default::default(),
            io: Default::default(),
            modular: ModularExtension::new(primes),
            keccak: Default::default(),
            weierstrass: WeierstrassExtension::new(curves),
        }
    }
}

#[test]
fn test_ecdsa_runtime() -> Result<()> {
    let elf = build_example_program("ecdsa")?;
    let config = Rv32ModularKeccak256Config::new(vec![SECP256K1_CONFIG.clone()]);
    let executor = VmExecutor::<F, _>::new(config);

    let exe = AxVmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension)
            .with_extension(Keccak256TranspilerExtension)
            .with_extension(EccTranspilerExtension)
            .with_extension(ModularTranspilerExtension),
    );
    executor.execute(exe, vec![])?;
    Ok(())
}
