use axvm_instructions::instruction::Instruction;

pub trait CustomInstructionProcessor<F> {
    fn process_custom(&self, instruction_stream: &[u32]) -> Option<(Instruction<F>, usize)>; // (`Instruction`, how many u32's to advance the RISC-V instruction stream by)
}
