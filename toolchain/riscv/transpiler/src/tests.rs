use std::{fs::read, path::PathBuf};

use ax_sdk::config::setup_tracing;
use axvm_platform::memory::MEM_SIZE;
use color_eyre::eyre::Result;
use p3_baby_bear::BabyBear;
use stark_vm::{
    sdk::air_test,
    system::vm::{config::VmConfig, VirtualMachine},
};
use test_case::test_case;

use crate::elf::AxVmExe;

type F = BabyBear;

#[test]
fn test_decode_elf() -> Result<()> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data = read(dir.join("data/rv32im-empty-program-elf"))?;
    let elf = AxVmExe::<F>::decode(&data, MEM_SIZE as u32)?;
    dbg!(elf);
    Ok(())
}

#[test_case("data/rv32im-fib-from-as")]
#[test_case("data/rv32im-intrin-from-as")]
fn test_generate_program(elf_path: &str) -> Result<()> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data = read(dir.join(elf_path))?;
    let exe = AxVmExe::<F>::decode(&data, MEM_SIZE as u32)?;
    for instruction in exe.program.instructions() {
        println!("{:?}", instruction);
    }
    Ok(())
}

#[test_case("data/rv32im-fibonacci-program-elf-release")]
#[test_case("data/rv32im-exp-from-as")]
#[test_case("data/rv32im-fib-from-as")]
fn test_rv32im_runtime(elf_path: &str) -> Result<()> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data = read(dir.join(elf_path))?;
    let exe = AxVmExe::<F>::decode(&data, MEM_SIZE as u32)?;
    setup_tracing();
    let config = VmConfig::rv32im();
    let vm = VirtualMachine::new(config).with_initial_memory(exe.memory_image);

    // TODO: use "execute_and_generate" when it's implemented

    vm.execute(exe.program)?;
    Ok(())
}

#[test_case("data/rv32im-fibonacci-program-elf-release")]
fn test_rv32i_prove(elf_path: &str) -> Result<()> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data = read(dir.join(elf_path))?;
    let exe = AxVmExe::<F>::decode(&data, MEM_SIZE as u32)?;
    let config = VmConfig::rv32i();
    let vm = VirtualMachine::new(config).with_initial_memory(exe.memory_image);

    air_test(vm, exe.program);
    Ok(())
}

#[test_case("data/rv32im-intrin-from-as")]
fn test_intrinsic_runtime(elf_path: &str) -> Result<()> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data = read(dir.join(elf_path))?;
    let exe = AxVmExe::<F>::decode(&data, MEM_SIZE as u32)?;
    setup_tracing();
    let config = VmConfig::rv32im().add_canonical_modulus();
    let vm = VirtualMachine::new(config).with_initial_memory(exe.memory_image);

    vm.execute(exe.program)?;
    Ok(())
}
