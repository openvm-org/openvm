use openvm_instructions::instruction::Instruction;

/// Trait to add custom RISC-V instruction transpilation to OpenVM instruction format.
/// RISC-V instructions always come in 32-bit chunks.
/// An important feature is that multiple 32-bit RISC-V instructions can be transpiled into a single
/// OpenVM instruction. See [process_custom](Self::process_custom) for details.
pub trait TranspilerExtension<F> {
    /// The `instruction_stream` provides a view of the remaining RISC-V instructions to be
    /// processed, presented as 32-bit chunks. The [process_custom](Self::process_custom) should
    /// determine if it knows how to transpile the next contiguous section of RISC-V
    /// instructions into an [`Instruction`]. It returns `None` if it cannot transpile.
    /// Otherwise it returns `TranspilerOutput { instructions, used_u32s }` to indicate that
    /// `instruction_stream[..used_u32s]` should be transpiled into `instructions`.
    fn process_custom(&self, instruction_stream: &[u32]) -> Option<TranspilerOutput<F>>;
}

pub struct TranspilerOutput<F> {
    pub instructions: Vec<Option<Instruction<F>>>,
    pub used_u32s: usize,
}

impl<F> TranspilerOutput<F> {
    pub fn one_to_one(instruction: Instruction<F>) -> Self {
        Self {
            instructions: vec![Some(instruction)],
            used_u32s: 1,
        }
    }

    pub fn many_to_one(instruction: Instruction<F>, used_u32s: usize) -> Self {
        Self {
            instructions: vec![Some(instruction)],
            used_u32s,
        }
    }

    pub fn gap(gap_length: usize, used_u32s: usize) -> Self {
        Self {
            instructions: (0..gap_length).map(|_| None).collect(),
            used_u32s,
        }
    }
}
