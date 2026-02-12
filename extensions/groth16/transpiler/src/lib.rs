use openvm_groth16_guest::GROTH16_VERIFY_OPCODE;
use openvm_transpiler::extension::TranspilerExtension;
use openvm_instructions::instruction::Instruction;

pub struct Groth16Transpiler;

impl TranspilerExtension for Groth16Transpiler {
    fn transpile(&self, instruction: &Instruction) -> Option<Instruction> {
        if instruction.opcode == GROTH16_VERIFY_OPCODE {
            return Some(instruction.clone());
        }
        None
    }
}
