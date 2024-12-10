//! A transpiler from custom RISC-V ELFs to axVM executable binaries.

use ax_stark_backend::p3_field::PrimeField32;
use axvm_instructions::{
    exe::AxVmExe,
    program::{Program, DEFAULT_PC_STEP},
};
pub use axvm_platform;
use elf::Elf;
use transpiler::{Transpiler, TranspilerError};

use crate::util::elf_memory_image_to_axvm_memory_image;

pub mod elf;
pub mod transpiler;
pub mod util;

mod extension;
pub use extension::TranspilerExtension;

pub trait FromElf {
    type ElfContext;
    fn from_elf(elf: Elf, ctx: Self::ElfContext) -> Result<Self, TranspilerError>
    where
        Self: Sized;
}

impl<F: PrimeField32> FromElf for AxVmExe<F> {
    type ElfContext = Transpiler<F>;
    fn from_elf(elf: Elf, transpiler: Self::ElfContext) -> Result<Self, TranspilerError> {
        let instructions = transpiler.transpile(&elf.instructions)?;
        let program = Program::new_without_debug_infos(
            &instructions,
            DEFAULT_PC_STEP,
            elf.pc_base,
            elf.max_num_public_values,
        );
        let init_memory = elf_memory_image_to_axvm_memory_image(elf.memory_image);

        Ok(AxVmExe {
            program,
            pc_start: elf.pc_start,
            init_memory,
            fn_bounds: elf.fn_bounds,
        })
    }
}