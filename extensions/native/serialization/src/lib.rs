use openvm_instructions::{instruction::Instruction, VmOpcode};
use p3_field::PrimeField32;

pub const IMMEDIATE_ADDRESS_SPACE: usize = 0;
pub const RUST_REGISTER_ADDRESS_SPACE: usize = 1;
pub const KERNEL_ADDRESS_SPACE: usize = 5;
pub const LONG_FORM_INSTRUCTION_INDICATOR: u32 = (1 << 31) + 115115115;
pub const GAP_INDICATOR: u32 = (1 << 31) + 113113113;
pub const VARIABLE_REGISTER_INDICATOR: u32 = (1 << 31) + 116;

pub fn serialize_instructions<F: PrimeField32>(instructions: &[Instruction<F>]) -> Vec<u32> {
    let mut words = vec![];
    for instruction in instructions {
        words.push(LONG_FORM_INSTRUCTION_INDICATOR);
        let operands = instruction.operands();
        words.push(operands.len() as u32);
        words.push(instruction.opcode.as_usize() as u32);
        words.extend(operands.iter().map(F::as_canonical_u32))
    }
    words
}

pub fn deserialize_instructions<F: PrimeField32>(words: &[u32]) -> Vec<Instruction<F>> {
    let mut index = 0;
    let mut instructions = vec![];
    while index < words.len() {
        assert_eq!(words[index], LONG_FORM_INSTRUCTION_INDICATOR);
        let num_operands = words[index + 1] as usize;
        let opcode = VmOpcode::from_usize(words[index + 2] as usize);
        index += 3;
        let mut operands: Vec<usize> = words[index..index + num_operands]
            .iter()
            .map(|&x| x as usize)
            .collect();
        operands.resize(7, 0);
        let instruction =
            Instruction::from_usize::<7>(opcode, std::array::from_fn(|i| operands[i]));
        instructions.push(instruction);
        index += num_operands;
    }
    instructions
}
