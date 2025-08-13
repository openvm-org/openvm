use std::rc::Rc;

use openvm_instructions::instruction::Instruction;
use openvm_stark_backend::p3_field::PrimeField32;
use thiserror::Error;

use crate::TranspilerExtension;

/// Collection of [`TranspilerExtension`]s.
/// The transpiler can be configured to transpile any ELF in 32-bit chunks.
#[derive(Clone)]
pub struct Transpiler<F> {
    processors: Vec<Rc<dyn TranspilerExtension<F>>>,
}

impl<F: PrimeField32> Default for Transpiler<F> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Error, Debug)]
pub enum TranspilerError {
    #[error("ambiguous next instruction")]
    AmbiguousNextInstruction,
    #[error("couldn't parse the next instruction: {0:032b}")]
    ParseError(u32),
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
    /// applies every processor in the [`Transpiler`] to determine if one of them knows how to
    /// transpile the current instruction (and possibly a contiguous section of following
    /// instructions). If so, it advances the iterator by the amount specified by the processor.
    /// The transpiler will panic if two different processors claim to know how to transpile the
    /// same instruction to avoid ambiguity.
    pub fn transpile(
        &self,
        instructions_u32: &[u32],
    ) -> Result<Vec<Option<Instruction<F>>>, TranspilerError> {
        let mut instructions = Vec::new();
        let mut ptr = 0;
        while ptr < instructions_u32.len() {
            let mut options = self
                .processors
                .iter()
                .map(|proc| proc.process_custom(&instructions_u32[ptr..]))
                .filter(|opt| opt.is_some())
                .collect::<Vec<_>>();
            if options.is_empty() {
                return Err(TranspilerError::ParseError(instructions_u32[ptr]));
            }
            if options.len() > 1 {
                return Err(TranspilerError::AmbiguousNextInstruction);
            }
            let transpiler_output = options.pop().unwrap().unwrap();
            instructions.extend(transpiler_output.instructions);
            ptr += transpiler_output.used_u32s;
        }
        Ok(instructions)
    }
}
