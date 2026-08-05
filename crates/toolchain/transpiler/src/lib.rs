//! A transpiler from custom RISC-V ELFs to OpenVM executable binaries.

use elf::Elf;
use openvm_instructions::{exe::VmExe, program::Program};
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
        let transpilation = transpiler.transpile_with_pc_preservation(&elf.instructions)?;
        let instructions = transpilation.instructions;
        let program = Program::new_without_debug_infos_with_option(&instructions, elf.pc_base);
        let mut init_memory = elf_memory_image_to_openvm_memory_image(elf.memory_image);
        transpiler.modify_initial_memory(&mut init_memory)?;
        let cfg_hints = if transpilation.preserves_pcs {
            elf.cfg_hints
        } else {
            Default::default()
        };

        Ok(VmExe {
            program,
            pc_start: elf.pc_start,
            init_memory,
            fn_bounds: elf.fn_bounds,
            cfg_hints,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use openvm_instructions::exe::{CfgHints, FnBounds};
    use p3_baby_bear::BabyBear;

    use super::*;

    struct NonPositional;

    impl TranspilerExtension<BabyBear> for NonPositional {
        fn process_custom(&self, instruction_stream: &[u32]) -> Option<TranspilerOutput<BabyBear>> {
            (instruction_stream.len() >= 2).then(|| TranspilerOutput {
                instructions: vec![Some(crate::util::unimp()), Some(crate::util::unimp())],
                used_u32s: 2,
                preserves_pc_slots: false,
            })
        }
    }

    #[test]
    fn non_pc_preserving_transpilation_drops_elf_hints() {
        let elf = Elf::new(
            vec![0, 0],
            0,
            0,
            Default::default(),
            FnBounds::default(),
            CfgHints {
                basic_block_starts: BTreeSet::from([0, 4]),
                ..Default::default()
            },
        );

        let exe = VmExe::from_elf(elf, Transpiler::new().with_extension(NonPositional)).unwrap();

        assert_eq!(exe.program.instructions_and_debug_infos.len(), 2);
        assert_eq!(exe.cfg_hints, CfgHints::default());
    }
}
