use std::{
    fs::read,
    path::{Path, PathBuf},
};

use axvm_circuit::{
    arch::{VmConfig, VmExecutor},
    utils::air_test,
};
use axvm_platform::memory::MEM_SIZE;
use eyre::Result;
use p3_baby_bear::BabyBear;
use test_case::test_case;

use crate::{elf::Elf, rrs::transpile, AxVmExe};

type F = BabyBear;

fn setup_executor_from_elf(
    elf_path: impl AsRef<Path>,
    config: VmConfig,
) -> Result<(VmExecutor<F>, AxVmExe<F>)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data = read(dir.join(elf_path))?;
    let elf = Elf::decode(&data, MEM_SIZE as u32)?;
    let executor = VmExecutor::new(config);
    Ok((executor, elf.into()))
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
    let program = transpile::<BabyBear>(&elf.instructions);
    for instruction in program {
        println!("{:?}", instruction);
    }
    Ok(())
}

#[test_case("data/rv32im-exp-from-as")]
#[test_case("data/rv32im-fib-from-as")]
fn test_rv32im_runtime(elf_path: &str) -> Result<()> {
    let config = VmConfig::rv32im();
    let (executor, exe) = setup_executor_from_elf(elf_path, config)?;
    executor.execute(exe, vec![])?;
    Ok(())
}

#[test_case("data/rv32im-intrin-from-as")]
fn test_intrinsic_runtime(elf_path: &str) -> Result<()> {
    let config = VmConfig::rv32im()
        .add_canonical_modulus()
        .add_canonical_pairing_curves()
        .add_int256_alu()
        .add_int256_m();
    let (executor, exe) = setup_executor_from_elf(elf_path, config)?;
    executor.execute(exe, vec![])?;
    Ok(())
}

#[test]
fn test_terminate_prove() -> Result<()> {
    let config = VmConfig::rv32i();
    let (_, exe) = setup_executor_from_elf("data/rv32im-terminate-from-as", config.clone())?;
    air_test(config, exe);
    Ok(())
}
