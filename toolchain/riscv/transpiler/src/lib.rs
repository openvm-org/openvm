//! A transpiler from custom RISC-V ELFs to axVM machine code.

use axvm_instructions::{
    exe::AxVmExe,
    program::{Program, DEFAULT_PC_STEP},
};
pub use axvm_platform;
use elf::Elf;
use p3_field::PrimeField32;
use rrs::transpile;

use crate::util::elf_memory_image_to_axvm_memory_image;

pub mod elf;
pub mod rrs;
pub mod util;

#[cfg(test)]
mod tests;

impl<F: PrimeField32> From<Elf> for AxVmExe<F> {
    fn from(elf: Elf) -> Self {
        let program = Program::from_instructions_and_step(
            &transpile(&elf.instructions),
            DEFAULT_PC_STEP,
            elf.pc_base,
        );
        let init_memory = elf_memory_image_to_axvm_memory_image(elf.memory_image);
        Self {
            program,
            pc_start: elf.pc_start,
            init_memory,
        }
    }
}
