use eyre::Result;
use openvm_instructions::{exe::SparseMemoryImage, instruction::Instruction};

/// Trait to add custom RISC-V instruction transpilation to OpenVM instruction format.
/// RISC-V instructions always come in 32-bit chunks.
pub trait TranspilerExtension<F> {
    /// The `instruction_stream` provides a view of the remaining RISC-V instructions to be
    /// processed, presented as 32-bit chunks. The [process_custom](Self::process_custom) should
    /// determine if it knows how to transpile the next contiguous section of RISC-V
    /// instructions into an [`Instruction`]. It returns `None` if it cannot transpile.
    /// Otherwise it returns one output slot for each consumed input word. This positional mapping
    /// preserves ELF PCs in the transpiled program.
    fn process_custom(&self, instruction_stream: &[u32]) -> Option<TranspilerOutput<F>>;

    /// Each transpiler extension is given the opportunity to modify the initial memory state.
    /// By default, nothing is done.
    fn modify_initial_memory(&self, _init_memory: &mut SparseMemoryImage) -> Result<()> {
        Ok(())
    }
}

pub struct TranspilerOutput<F> {
    pub instructions: Vec<Option<Instruction<F>>>,
}

impl<F> TranspilerOutput<F> {
    pub fn one_to_one(instruction: Instruction<F>) -> Self {
        Self {
            instructions: vec![Some(instruction)],
        }
    }
}
