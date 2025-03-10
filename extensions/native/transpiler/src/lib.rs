use openvm_instructions::{instruction::Instruction, riscv::RV32_REGISTER_NUM_LIMBS, VmOpcode};
use openvm_transpiler::{TranspilerExtension, TranspilerOutput};
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

pub struct LongFormTranspilerExtension;

impl<F: PrimeField32> TranspilerExtension<F> for LongFormTranspilerExtension {
    fn process_custom(&self, instruction_stream: &[u32]) -> Option<TranspilerOutput<F>> {
        if instruction_stream[0] == LONG_FORM_INSTRUCTION_INDICATOR {
            let num_operands = instruction_stream[1] as usize;
            let opcode = VmOpcode::from_usize(instruction_stream[2] as usize);
            let mut operands = vec![];
            let mut j = 3;
            for _ in 0..num_operands {
                if instruction_stream[j] == VARIABLE_REGISTER_INDICATOR {
                    let register = (instruction_stream[j + 1] >> 7) & 0x1f;
                    let offset = instruction_stream[j + 1] >> 20;
                    let mut operand = (RV32_REGISTER_NUM_LIMBS as u32 * register) + offset;
                    if offset >= 1 << 12 {
                        operand -= 1 << 12;
                    }
                    operands.push(F::from_canonical_u32(operand));
                    j += 2;
                } else {
                    operands.push(F::from_canonical_u32(instruction_stream[j]));
                    j += 1;
                }
            }
            while operands.len() < 7 {
                operands.push(F::ZERO);
            }
            let instruction = Instruction {
                opcode,
                a: operands[0],
                b: operands[1],
                c: operands[2],
                d: operands[3],
                e: operands[4],
                f: operands[5],
                g: operands[6],
            };
            if operands.len() == 7 {
                Some(TranspilerOutput::many_to_one(instruction, j))
            } else {
                None
            }
        } else if instruction_stream[0] == GAP_INDICATOR {
            Some(TranspilerOutput::gap(instruction_stream[1] as usize, 2))
        } else {
            None
        }
    }
}

pub fn serialize_defined_instructions<F: PrimeField32>(
    instructions: &[Instruction<F>],
) -> Vec<u32> {
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

// panics if deserialization fails or results in gaps
pub fn deserialize_defined_instructions<F: PrimeField32>(words: &[u32]) -> Vec<Instruction<F>> {
    let mut index = 0;
    let mut instructions = vec![];
    while index < words.len() {
        let next = LongFormTranspilerExtension
            .process_custom(&words[index..])
            .unwrap();
        instructions.extend(next.instructions.into_iter().map(Option::unwrap));
        index += next.used_u32s;
    }
    instructions
}
