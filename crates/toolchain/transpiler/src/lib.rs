//! A transpiler from custom RISC-V ELFs to axVM executable binaries.

use axvm_instructions::{
    config::{CustomOpConfig, FieldArithmeticOpConfig, IntrinsicsOpConfig},
    exe::AxVmExe,
    program::{Program, DEFAULT_PC_STEP},
};
pub use axvm_platform;
use elf::Elf;
use p3_field::PrimeField32;
use transpiler::Transpiler;

use crate::util::elf_memory_image_to_axvm_memory_image;

pub mod elf;
pub mod transpiler;
pub mod util;

mod extension;
pub use extension::TranspilerExtension;

pub trait FromElf {
    type ElfContext;
    fn from_elf(elf: Elf, ctx: Self::ElfContext) -> Self;
}

impl<F: PrimeField32> FromElf for AxVmExe<F> {
    type ElfContext = Transpiler<F>;
    fn from_elf(elf: Elf, transpiler: Self::ElfContext) -> Self {
        let program = Program::new_without_debug_infos(
            &transpiler.transpile(&elf.instructions),
            DEFAULT_PC_STEP,
            elf.pc_base,
            elf.max_num_public_values,
        );
        let init_memory = elf_memory_image_to_axvm_memory_image(elf.memory_image);

        AxVmExe {
            program,
            pc_start: elf.pc_start,
            init_memory,
            custom_op_config: CustomOpConfig {
                intrinsics: IntrinsicsOpConfig {
                    field_arithmetic: FieldArithmeticOpConfig {
                        primes: elf.supported_moduli,
                    },
                },
            },
            fn_bounds: elf.fn_bounds,
        }
    }
}
