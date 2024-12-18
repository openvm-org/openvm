use openvm_instructions::instruction::Instruction;

/// Trait to add custom RISC-V instruction transpilation to OpenVM instruction format.
/// RISC-V instructions always come in 32-bit chunks.
/// An important feature is that multiple 32-bit RISC-V instructions can be transpiled into a single OpenVM instruction.
/// See `process_custom` for details.
pub trait TranspilerExtension<F> {
    /// The `instruction_stream` provides a view of the remaining RISC-V instructions to be processed,
    /// presented as 32-bit chunks. The [`CustomInstructionProcessor`] should determine if it knows how to transpile
    /// the next contiguous section of RISC-V instructions into an [`Instruction`].
    /// It returns `None` if it cannot transpile. Otherwise it returns `(instruction, how_many_u32s)` to indicate that
    /// `instruction_stream[..how_many_u32s]` should be transpiled into `instruction`.
    fn process_custom(&self, instruction_stream: &[u32]) -> Option<(Instruction<F>, usize)>;
}
