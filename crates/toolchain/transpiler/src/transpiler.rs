use std::rc::Rc;

use axvm_instructions::instruction::Instruction;
use p3_field::PrimeField32;

use crate::TranspilerExtension;

/// Collection of [`TranspilerExtension`]s.
/// The transpiler can be configured to transpile any ELF in 32-bit chunks.
pub struct Transpiler<F> {
    processors: Vec<Rc<dyn TranspilerExtension<F>>>,
}

impl<F: PrimeField32> Default for Transpiler<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: PrimeField32> Transpiler<F> {
    pub fn new() -> Self {
        Self { processors: vec![] }
    }

    pub fn with_processor(self, proc: Rc<dyn TranspilerExtension<F>>) -> Self {
        let mut procs = self.processors;
        procs.push(proc);
        Self { processors: procs }
    }

    pub fn with_extension<T: TranspilerExtension<F> + 'static>(self, ext: T) -> Self {
        self.with_processor(Rc::new(ext))
    }

    /// Iterates over a sequence of 32-bit RISC-V instructions `instructions_u32`. The iterator
    /// applies every processor in the [`Transpiler`] to determine if one of them knows how to transpile
    /// the current instruction (and possibly a contiguous section of following instructions).
    /// If so, it advances the iterator by the amount specified by the processor.
    /// The transpiler will panic if two different processors claim to know how to transpile the same instruction
    /// to avoid ambiguity.
    pub fn transpile(&self, instructions_u32: &[u32]) -> Vec<Instruction<F>> {
        let mut instructions = Vec::new();
        let mut ptr = 0;
        while ptr < instructions_u32.len() {
            let mut options = self
                .processors
                .iter()
                .map(|proc| proc.process_custom(&instructions_u32[ptr..]))
                .filter(|opt| opt.is_some())
                .collect::<Vec<_>>();
            assert!(
                !options.is_empty(),
                "couldn't parse the next instruction: {:032b}",
                instructions_u32[ptr]
            );
            assert!(options.len() < 2, "ambiguous next instruction");
            let (instruction, advance) = options.pop().unwrap().unwrap();
            instructions.push(instruction);
            ptr += advance;
        }
        instructions
    }
}
