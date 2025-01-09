use openvm_instructions::{instruction::Instruction, VmOpcode};
use p3_field::PrimeField32;

/*
 * The indicators use
 * - opcode = 0x0b (custom-0 as defined in RISC-V spec document)
 * - funct3 = 0b111
 *
 * `LONG_FORM_INSTRUCTION_INDICATOR` has funct7 = 0b0.
 * `GAP_INDICATOR` has funct7 = 0b1.
 *
 * `VARIABLE_REGISTER_INDICATOR` does not need to conform to RISC_V format,
 * because it occurs only within a block already prefixed with `LONG_FORM_INSTRUCTION_INDICATOR`.
 * Thus, we make its value larger than 2^31 to ensure that it is not equal to a possible field element.
 */
const OPCODE: u32 = 0x0b;
const FUNCT3: u32 = 0b111;
pub const LONG_FORM_INSTRUCTION_INDICATOR: u32 = (FUNCT3 << 12) + OPCODE;
pub const GAP_INDICATOR: u32 = (1 << 25) + (FUNCT3 << 12) + OPCODE;
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
        assert!(!operands.contains(&(VARIABLE_REGISTER_INDICATOR as usize)));
        operands.resize(7, 0);
        let instruction =
            Instruction::from_usize::<7>(opcode, std::array::from_fn(|i| operands[i]));
        instructions.push(instruction);
        index += num_operands;
    }
    instructions
}
