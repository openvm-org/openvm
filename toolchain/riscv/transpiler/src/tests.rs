use std::{
    fs::read,
    path::{Path, PathBuf},
};

use axvm_circuit::{
    arch::{new_vm, VmConfig, VmExecutor},
    intrinsics::modular::SECP256K1_COORD_PRIME,
    utils::new_air_test_with_min_segments,
};
use axvm_platform::memory::MEM_SIZE;
use axvm_rv32im_circuit::Rv32ImConfig;
use eyre::Result;
use p3_baby_bear::BabyBear;
use test_case::test_case;

use crate::{elf::Elf, transpiler::Transpiler, AxVmExe};

type F = BabyBear;

/// TODO: remove vm::VmExecutor and use new_vm::VmExecutor everywhere when all VmExtensions are implemented
fn get_elf(elf_path: impl AsRef<Path>) -> Result<AxVmExe<F>> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data = read(dir.join(elf_path))?;
    let elf = Elf::decode(&data, MEM_SIZE as u32)?;
    Ok(elf.into())
}

// An "eyeball test" only: prints the decoded ELF for eyeball inspection
#[test]
fn test_decode_elf() -> Result<()> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data = read(dir.join("data/rv32im-empty-program-elf"))?;
    let elf = Elf::decode(&data, MEM_SIZE as u32)?;
    dbg!(elf);
    Ok(())
}

// To create ELF directly from .S file, `brew install riscv-gnu-toolchain` and run
// `riscv64-unknown-elf-gcc -march=rv32im -mabi=ilp32 -nostartfiles -e _start -Ttext 0 fib.S -o rv32im-fib-from-as`
// riscv64-unknown-elf-gcc supports rv32im if you set -march target
#[test_case("data/rv32im-fib-from-as")]
#[test_case("data/rv32im-intrin-from-as")]
fn test_generate_program(elf_path: &str) -> Result<()> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data = read(dir.join(elf_path))?;
    let elf = Elf::decode(&data, MEM_SIZE as u32)?;
    let program = Transpiler::<BabyBear>::default_with_intrinsics().transpile(&elf.instructions);
    for instruction in program {
        println!("{:?}", instruction);
    }
    Ok(())
}

#[test_case("data/rv32im-exp-from-as")]
#[test_case("data/rv32im-fib-from-as")]
fn test_rv32im_runtime(elf_path: &str) -> Result<()> {
    let elf = get_elf(elf_path)?;
    let config = Rv32ImConfig::default();
    let executor = new_vm::VmExecutor::<F, _>::new(config);
    executor.execute(elf, vec![])?;
    Ok(())
}

#[test_case("data/rv32im-intrin-from-as")]
fn test_intrinsic_runtime(elf_path: &str) -> Result<()> {
    let config = VmConfig::rv32im()
        .add_canonical_modulus()
        .add_complex_ext_support(vec![SECP256K1_COORD_PRIME.clone()])
        .add_int256_alu()
        .add_int256_m();
    let elf = get_elf(elf_path)?;
    let executor = VmExecutor::<F>::new(config);
    executor.execute(elf, vec![])?;
    Ok(())
}

#[test]
fn test_terminate_prove() -> Result<()> {
    let config = Rv32ImConfig::default();
    let elf = get_elf("data/rv32im-terminate-from-as")?;
    new_air_test_with_min_segments(config, elf, vec![], 1);
    Ok(())
}
