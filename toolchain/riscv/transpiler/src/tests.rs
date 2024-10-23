use std::{fs::read, path::PathBuf};

use ax_sdk::config::setup_tracing;
use axvm_platform::memory::MEM_SIZE;
use color_eyre::eyre::Result;
use p3_baby_bear::BabyBear;
use stark_vm::{
    sdk::air_test,
    system::{
        program::Program,
        vm::{config::VmConfig, VirtualMachine},
    },
};
use test_case::test_case;

use crate::{elf::Elf, rrs::transpile};

#[test]
fn test_decode_elf() -> Result<()> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data = read(dir.join("data/rv32im-empty-program-elf"))?;
    let elf = Elf::decode(&data, MEM_SIZE as u32)?;
    dbg!(elf);
    Ok(())
}

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
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data = read(dir.join(elf_path))?;
    let elf = Elf::decode(&data, MEM_SIZE as u32)?;
    let instructions = transpile::<BabyBear>(&elf.instructions);
    setup_tracing();
    let program = Program::from_instructions_and_step(&instructions, 4, elf.pc_start, elf.pc_base);
    let config = VmConfig::rv32im();
    let vm = VirtualMachine::new(config);

    // TODO: use "execute_and_generate" when it's implemented

    vm.execute(program)?;
    Ok(())
}

#[test_case("data/rv32im-fibonacci-program-elf-release")]
fn test_rv32i_prove(elf_path: &str) -> Result<()> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data = read(dir.join(elf_path))?;
    let elf = Elf::decode(&data, MEM_SIZE as u32)?;
    let instructions = transpile::<BabyBear>(&elf.instructions);
    let program = Program::from_instructions_and_step(&instructions, 4, elf.pc_start, elf.pc_base);
    let config = VmConfig::rv32i();
    let vm = VirtualMachine::new(config);

    air_test(vm, program);
    Ok(())
}

#[test_case("data/rv32im-intrin-from-as")]
fn test_intrinsic_runtime(elf_path: &str) -> Result<()> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data = read(dir.join(elf_path))?;
    let elf = Elf::decode(&data, MEM_SIZE as u32)?;
    let instructions = transpile::<BabyBear>(&elf.instructions);
    setup_tracing();
    let program = Program::from_instructions_and_step(&instructions, 4, elf.pc_start, elf.pc_base);
    let config = VmConfig::rv32im().add_canonical_modulus();
    let vm = VirtualMachine::new(config);

    vm.execute(program)?;
    Ok(())
}
