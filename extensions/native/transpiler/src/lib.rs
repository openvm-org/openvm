use openvm_instructions::{instruction::Instruction, riscv::RV32_REGISTER_NUM_LIMBS, VmOpcode};
use openvm_native_serialization::{
    GAP_INDICATOR, LONG_FORM_INSTRUCTION_INDICATOR, VARIABLE_REGISTER_INDICATOR,
};
use openvm_transpiler::{TranspilerExtension, TranspilerOutput};
use p3_field::PrimeField32;

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
            Some(TranspilerOutput::many_to_one(instruction, j))
        } else if instruction_stream[0] == GAP_INDICATOR {
            Some(TranspilerOutput::gap(instruction_stream[1] as usize, 2))
        } else {
            None
        }
    }
}
