use std::{fs::read, path::PathBuf};

use ax_sdk::config::setup_tracing;
use axvm_platform::memory::MEM_SIZE;
use color_eyre::eyre::Result;
use p3_baby_bear::BabyBear;
use stark_vm::system::{
    program::Program,
    vm::{config::VmConfig, VirtualMachine},
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

#[test]
fn test_generate_program() -> Result<()> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data = read(dir.join("data/rv32im-fib-from-as"))?;
    let elf = Elf::decode(&data, MEM_SIZE as u32)?;
    let program = transpile::<BabyBear>(&elf.instructions);
    for instruction in program {
        println!("{:?}", instruction);
    }
    Ok(())
}

#[test]
fn test_tiny_asm_runtime() -> Result<()> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data = read(dir.join("data/rv32im-fib-from-as"))?;
    let elf = Elf::decode(&data, MEM_SIZE as u32)?;
    let instructions = transpile::<BabyBear>(&elf.instructions);
    for instruction in instructions.iter() {
        println!("{:?}", instruction);
    }
    let program = Program::from_instructions_and_step(&instructions, 4, elf.pc_start, elf.pc_base);
    let config = VmConfig::rv32();
    let vm = VirtualMachine::new(config, program, vec![]);

    // TODO: use "execute_and_generate" when it's implemented
    /*
    let perm = default_perm();
    let fri_params = FriParameters::standard_fast();

    let result = vm.execute_and_generate()?;

    for segment_result in result.segment_results {
        let engine = engine_from_perm(perm.clone(), segment_result.max_log_degree(), fri_params);
        engine
            .run_test_impl(&segment_result.air_infos)
            .expect("Verification failed");
    }
    */

    vm.execute()?;
    Ok(())
}

#[test_case("data/rv32im-fibonacci-program-elf-release")]
#[test_case("data/rv32im-exp-from-as")]
fn test_runtime(elf_path: &str) -> Result<()> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let data = read(dir.join(elf_path))?;
    let elf = Elf::decode(&data, MEM_SIZE as u32)?;
    let instructions = transpile::<BabyBear>(&elf.instructions);
    setup_tracing();
    let program = Program::from_instructions_and_step(&instructions, 4, elf.pc_start, elf.pc_base);
    let config = VmConfig::rv32();
    let vm = VirtualMachine::new(config, program, vec![]);

    // TODO: use "execute_and_generate" when it's implemented
    /*
    let perm = default_perm();
    let fri_params = FriParameters::standard_fast();

    let result = vm.execute_and_generate()?;

    for segment_result in result.segment_results {
        let engine = engine_from_perm(perm.clone(), segment_result.max_log_degree(), fri_params);
        engine
            .run_test_impl(&segment_result.air_infos)
            .expect("Verification failed");
    }
    */

    vm.execute()?;
    Ok(())
}
