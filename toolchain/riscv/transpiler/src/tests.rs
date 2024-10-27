use std::{fs::read, path::PathBuf};

use ax_sdk::config::setup_tracing;
use axvm_circuit::{
    arch::{VirtualMachine, VmConfig},
    sdk::air_test,
};
use axvm_instructions::program::Program;
use axvm_platform::memory::MEM_SIZE;
use color_eyre::eyre::Result;
use p3_baby_bear::BabyBear;
use test_case::test_case;

use crate::{elf::Elf, rrs::transpile, AxVmExe};

type F = BabyBear;

fn setup_vm_from_elf(elf_path: &str, config: VmConfig) -> Result<(VirtualMachine<F>, Program<F>)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data = read(dir.join(elf_path))?;
    let elf = Elf::decode(&data, MEM_SIZE as u32)?;
    dbg!(&elf.instructions);
    let exe = AxVmExe::<F>::from_elf(elf);
    let vm = VirtualMachine::new(config).with_initial_memory(exe.memory_image);
    Ok((vm, exe.program))
}

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

#[test_case("data/rv32im-fibonacci-program-elf-release")]
#[test_case("data/rv32im-exp-from-as")]
#[test_case("data/rv32im-fib-from-as")]
fn test_rv32im_runtime(elf_path: &str) -> Result<()> {
    setup_tracing();
    let config = VmConfig::rv32im();
    let (vm, program) = setup_vm_from_elf(elf_path, config)?;
    vm.execute(program)?;
    Ok(())
}

#[test_case("data/rv32im-fibonacci-program-elf-release")]
fn test_rv32i_prove(elf_path: &str) -> Result<()> {
    let config = VmConfig::rv32i();
    let (vm, program) = setup_vm_from_elf(elf_path, config)?;
    air_test(vm, program);
    Ok(())
}

#[test_case("data/rv32im-intrin-from-as")]
fn test_intrinsic_runtime(elf_path: &str) -> Result<()> {
    setup_tracing();
    let config = VmConfig::rv32im().add_canonical_modulus();
    let (vm, program) = setup_vm_from_elf(elf_path, config)?;
    vm.execute(program)?;
    Ok(())
}

#[test]
fn test_terminate_runtime() -> Result<()> {
    setup_tracing();
    let config = VmConfig::rv32i();
    let (vm, program) = setup_vm_from_elf("data/rv32im-terminate-from-as", config)?;
    air_test(vm, program);
    Ok(())
}
