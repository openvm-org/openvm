//! A transpiler from custom RISC-V ELFs to axVM machine code.

use axvm_circuit::system::memory::Equipartition;
use axvm_instructions::program::{Program, DEFAULT_PC_STEP};
use elf::Elf;
use p3_field::PrimeField32;
use rrs::transpile;
use util::memory_image_to_equipartition;

pub mod elf;
pub mod rrs;
pub mod util;

#[cfg(test)]
mod tests;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct AxVmExe<F> {
    pub(crate) program: Program<F>,
    pub(crate) memory_image: Equipartition<F, 8>,
}

impl<F: PrimeField32> AxVmExe<F> {
    #[allow(dead_code)]
    pub fn from_elf(elf: Elf) -> Self {
        let program = Program::from_instructions_and_step(
            &transpile(&elf.instructions),
            DEFAULT_PC_STEP,
            elf.pc_start,
            elf.pc_base,
        );
        let memory_image = memory_image_to_equipartition(elf.memory_image);
        Self {
            program,
            memory_image,
        }
    }
}
