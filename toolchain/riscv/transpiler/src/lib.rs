//! A transpiler from custom RISC-V ELFs to axVM machine code.

use axvm_instructions::{
    config::{CustomOpConfig, FieldArithmeticOpConfig, IntrinsicsOpConfig},
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
        let program = Program::new_without_debug_infos(
            &transpile(&elf.instructions),
            DEFAULT_PC_STEP,
            elf.pc_base,
            elf.max_num_public_values,
        );
        let init_memory = elf_memory_image_to_axvm_memory_image(elf.memory_image);

        #[cfg(not(feature = "function-span"))]
        let fn_bounds = Default::default();
        #[cfg(feature = "function-span")]
        let fn_bounds = elf.fn_bounds;

        Self {
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
            fn_bounds,
        }
    }
}
