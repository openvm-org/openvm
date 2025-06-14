//! A transpiler from custom RISC-V ELFs to OpenVM executable binaries.

use elf::Elf;
use openvm_instructions::{
    exe::VmExe,
    program::{Program, DEFAULT_PC_STEP},
};
pub use openvm_platform;
use openvm_stark_backend::p3_field::PrimeField32;
use transpiler::{Transpiler, TranspilerError};

use crate::util::elf_memory_image_to_openvm_memory_image;

pub mod elf;
pub mod transpiler;
pub mod util;

mod extension;
pub use extension::{TranspilerExtension, TranspilerOutput};

pub trait FromElf {
    type ElfContext;
    fn from_elf(elf: Elf, ctx: Self::ElfContext) -> Result<Self, TranspilerError>
    where
        Self: Sized;
}

impl<F: PrimeField32> FromElf for VmExe<F> {
    type ElfContext = Transpiler<F>;
    fn from_elf(elf: Elf, transpiler: Self::ElfContext) -> Result<Self, TranspilerError> {
        let instructions = transpiler.transpile(&elf.instructions)?;
        let program = Program::new_without_debug_infos_with_option(
            &instructions,
            DEFAULT_PC_STEP,
            elf.pc_base,
        );
        let init_memory = elf_memory_image_to_openvm_memory_image(elf.memory_image);

        Ok(VmExe {
            program,
            pc_start: elf.pc_start,
            init_memory,
            fn_bounds: elf.fn_bounds,
        })
    }
}
