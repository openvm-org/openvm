use std::{fs::read, path::PathBuf};

use axvm_platform::memory::MEM_SIZE;
use color_eyre::eyre::Result;
use p3_baby_bear::BabyBear;

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
    dbg!(elf.pc_start / 4 - elf.pc_base / 4, elf.instructions.len());
    let program = transpile::<BabyBear>(&elf.instructions);
    for instruction in program {
        println!("{:?}", instruction);
    }
    Ok(())
}
