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
        VmConfig, VmInventoryError,
    },
    derive::{AnyEnum, InstructionExecutor, VmConfig},
    utils::new_air_test_with_min_segments,
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
use p3_field::{AbstractField, PrimeField32};

use crate::utils::{build_example_program, build_example_program_with_features};

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

    let moduli = ["4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129015664037894272559787", "1000000000000000003", "2305843009213693951"]
        .map(|s| num_bigint_dig::BigUint::from_str(s).unwrap());
    let config = Rv32ModularConfig::new(moduli.to_vec());
    new_air_test_with_min_segments(config, axvm_exe, vec![], 1, false);
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
    new_air_test_with_min_segments(config, axvm_exe, vec![], 1, false);
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
    // Always run prove, as this caught a bug before.
    new_air_test_with_min_segments(config, axvm_exe, vec![], 1, true);
    Ok(())
}

#[test]
fn test_complex_two_moduli_runtime() -> Result<()> {
    let elf = build_example_program("complex-two-modulos")?;
    let axvm_exe = AxVmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension)
            .with_extension(Fp2TranspilerExtension)
            .with_extension(ModularTranspilerExtension),
    );
    let config = Rv32ModularWithFp2Config::new(vec![
        BigUint::from_str("998244353").unwrap(),
        BigUint::from_str("1000000007").unwrap(),
    ]);
    new_air_test_with_min_segments(config, axvm_exe, vec![], 1, false);
    Ok(())
}

#[test]
fn test_ec_runtime() -> Result<()> {
    let elf = build_example_program_with_features("ec", ["k256"])?;
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
    new_air_test_with_min_segments(config, axvm_exe, vec![], 1, false);
    Ok(())
}

#[test]
fn test_decompress() -> Result<()> {
    use axvm_ecc_guest::halo2curves::{group::Curve, secp256k1::Secp256k1Affine};

    let elf = build_example_program_with_features("decompress", ["k256"])?;
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

    let p = Secp256k1Affine::generator();
    let p = (p + p + p).to_affine();
    println!("decompressed: {:?}", p);
    let coords: Vec<_> = [p.x.to_bytes(), p.y.to_bytes()]
        .concat()
        .into_iter()
        .map(AbstractField::from_canonical_u8)
        .collect();
    new_air_test_with_min_segments(config, axvm_exe, vec![coords], 1, false);
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
    let elf = build_example_program_with_features("ecdsa", ["k256"])?;
    let config = Rv32ModularKeccak256Config::new(vec![SECP256K1_CONFIG.clone()]);

    let axvm_exe = AxVmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension)
            .with_extension(Keccak256TranspilerExtension)
            .with_extension(EccTranspilerExtension)
            .with_extension(ModularTranspilerExtension),
    );
    new_air_test_with_min_segments(config, axvm_exe, vec![], 1, true);
    Ok(())
}
