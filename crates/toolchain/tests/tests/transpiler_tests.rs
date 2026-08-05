use std::{
    fs::read,
    path::{Path, PathBuf},
};

use eyre::Result;
use num_bigint::BigUint;
use openvm_algebra_circuit::*;
use openvm_algebra_transpiler::{Fp2TranspilerExtension, ModularTranspilerExtension};
use openvm_bigint_circuit::*;
use openvm_circuit::{
    arch::{InitFileGenerator, SystemConfig, VmExecutor},
    derive::VmConfig,
    system::SystemExecutor,
    utils::air_test,
};
use openvm_ecc_circuit::{SECP256K1_MODULUS, SECP256K1_ORDER};
use openvm_instructions::exe::VmExe;
use openvm_platform::memory::MEM_SIZE;
use openvm_riscv_circuit::{
    RiscvI, RiscvIExecutor, RiscvImBuilder, RiscvImConfig, RiscvIo, RiscvIoExecutor, RiscvM,
    RiscvMExecutor,
};
use openvm_riscv_transpiler::{
    RiscvITranspilerExtension, RiscvIoTranspilerExtension, RiscvMTranspilerExtension,
};
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use openvm_transpiler::{elf::Elf, transpiler::Transpiler, FromElf};
use serde::{Deserialize, Serialize};
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
    let elf = get_elf("tests/data/rv64im-empty-program-elf")?;
    dbg!(elf);
    Ok(())
}

// To create ELF directly from .S file, install riscv-gnu-toolchain and run
// `riscv64-unknown-elf-gcc -march=rv64im -mabi=lp64 -nostartfiles -nostdlib -e _start -Ttext 0
// fib.S -o rv64im-fib-from-as`
#[test_case("tests/data/rv64im-fib-from-as")]
#[test_case("tests/data/rv64im-intrin-from-as")]
fn test_generate_program(elf_path: &str) -> Result<()> {
    let elf = get_elf(elf_path)?;
    let program = Transpiler::<F>::default()
        .with_extension(RiscvITranspilerExtension)
        .with_extension(RiscvMTranspilerExtension)
        .with_extension(RiscvIoTranspilerExtension)
        .with_extension(ModularTranspilerExtension)
        .transpile(&elf.instructions)?;
    for instruction in program {
        println!("{instruction:?}");
    }
    Ok(())
}

#[test_case("tests/data/rv64im-exp-from-as")]
#[test_case("tests/data/rv64im-fib-from-as")]
fn test_riscv_im_runtime(elf_path: &str) -> Result<()> {
    let elf = get_elf(elf_path)?;
    let exe = VmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(RiscvITranspilerExtension)
            .with_extension(RiscvMTranspilerExtension)
            .with_extension(RiscvIoTranspilerExtension),
    )?;
    let config = RiscvImConfig::default();
    let executor = VmExecutor::new(config)?;
    let instance = executor.instance(&exe)?;
    instance.execute(vec![])?;
    Ok(())
}

#[derive(Clone, Debug, VmConfig, Serialize, Deserialize)]
pub struct ModularFp2Int256Config {
    #[config(executor = "SystemExecutor")]
    pub system: SystemConfig,
    #[extension]
    pub base: RiscvI,
    #[extension]
    pub mul: RiscvM,
    #[extension]
    pub io: RiscvIo,
    #[extension]
    pub modular: ModularExtension,
    #[extension]
    pub fp2: Fp2Extension,
    #[extension]
    pub int256: Int256,
}

impl ModularFp2Int256Config {
    pub fn new(modular_moduli: Vec<BigUint>, fp2_moduli: Vec<(String, BigUint)>) -> Self {
        Self {
            system: SystemConfig::default(),
            base: Default::default(),
            mul: Default::default(),
            io: Default::default(),
            modular: ModularExtension::new(modular_moduli),
            fp2: Fp2Extension::new(fp2_moduli),
            int256: Default::default(),
        }
    }
}

impl InitFileGenerator for ModularFp2Int256Config {
    fn generate_init_file_contents(&self) -> Option<String> {
        Some(format!(
            "{}\n{}\n",
            self.modular.generate_moduli_init(),
            self.fp2.generate_complex_init(&self.modular)
        ))
    }
}

#[test_case("tests/data/rv64im-intrin-from-as")]
fn test_intrinsic_runtime(elf_path: &str) -> Result<()> {
    let config = ModularFp2Int256Config::new(
        vec![SECP256K1_MODULUS.clone(), SECP256K1_ORDER.clone()],
        vec![("Secp256k1Coord".to_string(), SECP256K1_MODULUS.clone())],
    );
    let elf = get_elf(elf_path)?;
    let openvm_exe = VmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(RiscvITranspilerExtension)
            .with_extension(RiscvMTranspilerExtension)
            .with_extension(RiscvIoTranspilerExtension)
            .with_extension(ModularTranspilerExtension)
            .with_extension(Fp2TranspilerExtension),
    )?;
    let executor = VmExecutor::new(config)?;
    let instance = executor.instance(&openvm_exe)?;
    instance.execute(vec![])?;
    Ok(())
}

#[test]
fn test_terminate_prove() -> Result<()> {
    let config = RiscvImConfig::default();
    let elf = get_elf("tests/data/rv64im-terminate-from-as")?;
    let openvm_exe = VmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(RiscvITranspilerExtension)
            .with_extension(RiscvMTranspilerExtension)
            .with_extension(RiscvIoTranspilerExtension)
            .with_extension(ModularTranspilerExtension),
    )?;
    air_test(RiscvImBuilder, config, openvm_exe);
    Ok(())
}
