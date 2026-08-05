use eyre::Result;
use openvm_instructions::{exe::SparseMemoryImage, instruction::Instruction};

/// Trait to add custom RISC-V instruction transpilation to OpenVM instruction format.
/// RISC-V instructions always come in 32-bit chunks.
/// An important feature is that multiple 32-bit RISC-V instructions can be transpiled into a single
/// OpenVM instruction. See [process_custom](Self::process_custom) for details.
pub trait TranspilerExtension<F> {
    /// The `instruction_stream` provides a view of the remaining RISC-V instructions to be
    /// processed, presented as 32-bit chunks. The [process_custom](Self::process_custom) should
    /// determine if it knows how to transpile the next contiguous section of RISC-V
    /// instructions into an [`Instruction`]. It returns `None` if it cannot transpile.
    /// Otherwise it returns a [`TranspilerOutput`] describing the emitted instructions, consumed
    /// input, and whether output slots retain their corresponding input PCs.
    fn process_custom(&self, instruction_stream: &[u32]) -> Option<TranspilerOutput<F>>;

    /// Each transpiler extension is given the opportunity to modify the initial memory state.
    /// By default, nothing is done.
    fn modify_initial_memory(&self, _init_memory: &mut SparseMemoryImage) -> Result<()> {
        Ok(())
    }
}

pub struct TranspilerOutput<F> {
    pub instructions: Vec<Option<Instruction<F>>>,
    pub used_u32s: usize,
    /// Whether each output slot retains the PC and semantics of the corresponding input slot.
    pub preserves_pc_slots: bool,
}

impl<F> TranspilerOutput<F> {
    pub fn one_to_one(instruction: Instruction<F>) -> Self {
        Self {
            instructions: vec![Some(instruction)],
            used_u32s: 1,
            preserves_pc_slots: true,
        }
    }

    pub fn many_to_one(instruction: Instruction<F>, used_u32s: usize) -> Self {
        Self {
            instructions: vec![Some(instruction)],
            used_u32s,
            preserves_pc_slots: false,
        }
    }

    pub fn gap(gap_length: usize, used_u32s: usize) -> Self {
        Self {
            instructions: (0..gap_length).map(|_| None).collect(),
            used_u32s,
            preserves_pc_slots: gap_length == used_u32s,
        }
    }
}
