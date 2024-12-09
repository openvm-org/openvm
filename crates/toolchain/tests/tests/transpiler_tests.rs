use std::{
    fs::read,
    path::{Path, PathBuf},
};

use ax_circuit_derive::{Chip, ChipUsageGetter};
use ax_stark_backend::p3_field::PrimeField32;
use ax_stark_sdk::p3_baby_bear::BabyBear;
use axvm_algebra_circuit::{
    Fp2Extension, Fp2ExtensionExecutor, Fp2ExtensionPeriphery, ModularExtension,
    ModularExtensionExecutor, ModularExtensionPeriphery,
};
use axvm_algebra_transpiler::{Fp2TranspilerExtension, ModularTranspilerExtension};
use axvm_bigint_circuit::{Int256, Int256Executor, Int256Periphery};
use axvm_circuit::{
    arch::{
        SystemConfig, SystemExecutor, SystemPeriphery, VmChipComplex, VmConfig, VmExecutor,
        VmInventoryError,
    },
    derive::{AnyEnum, InstructionExecutor, VmConfig},
    utils::new_air_test_with_min_segments,
};
use axvm_ecc_guest::k256::{SECP256K1_MODULUS, SECP256K1_ORDER};
use axvm_instructions::exe::AxVmExe;
use axvm_platform::memory::MEM_SIZE;
use axvm_rv32im_circuit::{
    Rv32I, Rv32IExecutor, Rv32IPeriphery, Rv32ImConfig, Rv32Io, Rv32IoExecutor, Rv32IoPeriphery,
    Rv32M, Rv32MExecutor, Rv32MPeriphery,
};
use axvm_rv32im_transpiler::{
    Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
};
use axvm_transpiler::{elf::Elf, transpiler::Transpiler, FromElf};
use derive_more::derive::From;
use eyre::Result;
use num_bigint_dig::BigUint;
use test_case::test_case;

type F = BabyBear;

fn get_elf(elf_path: impl AsRef<Path>) -> Result<Elf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data = read(dir.join(elf_path))?;
    let elf = Elf::decode(&data, MEM_SIZE as u32)?;
    Ok(elf)
}

// An "eyeball test" only: prints the decoded ELF for eyeball inspection
#[test]
fn test_decode_elf() -> Result<()> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data = read(dir.join("tests/data/rv32im-empty-program-elf"))?;
    let elf = Elf::decode(&data, MEM_SIZE as u32)?;
    dbg!(elf);
    Ok(())
}

// To create ELF directly from .S file, `brew install riscv-gnu-toolchain` and run
// `riscv64-unknown-elf-gcc -march=rv32im -mabi=ilp32 -nostartfiles -e _start -Ttext 0 fib.S -o rv32im-fib-from-as`
// riscv64-unknown-elf-gcc supports rv32im if you set -march target
#[test_case("tests/data/rv32im-fib-from-as")]
#[test_case("tests/data/rv32im-intrin-from-as")]
fn test_generate_program(elf_path: &str) -> Result<()> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data = read(dir.join(elf_path))?;
    let elf = Elf::decode(&data, MEM_SIZE as u32)?;
    let program = Transpiler::<F>::default()
        .with_extension(Rv32ITranspilerExtension)
        .with_extension(Rv32MTranspilerExtension)
        .with_extension(Rv32IoTranspilerExtension)
        .with_extension(ModularTranspilerExtension)
        .transpile(&elf.instructions);
    for instruction in program {
        println!("{:?}", instruction);
    }
    Ok(())
}

#[test_case("tests/data/rv32im-exp-from-as")]
#[test_case("tests/data/rv32im-fib-from-as")]
fn test_rv32im_runtime(elf_path: &str) -> Result<()> {
    let elf = get_elf(elf_path)?;
    let exe = AxVmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension),
    );
    let config = Rv32ImConfig::default();
    let executor = VmExecutor::<F, _>::new(config);
    executor.execute(exe, vec![])?;
    Ok(())
}

#[derive(Clone, Debug, VmConfig)]
pub struct Rv32ModularFp2Int256Config {
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
    pub fp2: Fp2Extension,
    #[extension]
    pub int256: Int256,
}

impl Rv32ModularFp2Int256Config {
    pub fn new(modular_moduli: Vec<BigUint>, fp2_moduli: Vec<BigUint>) -> Self {
        Self {
            system: SystemConfig::default().with_continuations(),
            base: Default::default(),
            mul: Default::default(),
            io: Default::default(),
            modular: ModularExtension::new(modular_moduli),
            fp2: Fp2Extension::new(fp2_moduli),
            int256: Default::default(),
        }
    }
}

#[test_case("tests/data/rv32im-intrin-from-as")]
fn test_intrinsic_runtime(elf_path: &str) -> Result<()> {
    let config = Rv32ModularFp2Int256Config::new(
        vec![SECP256K1_MODULUS.clone(), SECP256K1_ORDER.clone()],
        vec![SECP256K1_MODULUS.clone()],
    );
    let elf = get_elf(elf_path)?;
    let axvm_exe = AxVmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension)
            .with_extension(ModularTranspilerExtension)
            .with_extension(Fp2TranspilerExtension),
    );
    let executor = VmExecutor::<F, _>::new(config);
    executor.execute(axvm_exe, vec![])?;
    Ok(())
}

#[test]
fn test_terminate_prove() -> Result<()> {
    let config = Rv32ImConfig::default();
    let elf = get_elf("tests/data/rv32im-terminate-from-as")?;
    let axvm_exe = AxVmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension)
            .with_extension(ModularTranspilerExtension),
    );
    new_air_test_with_min_segments(config, axvm_exe, vec![], 1, true);
    Ok(())
}
