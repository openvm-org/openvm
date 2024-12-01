use axvm_instructions::*;
use instruction::Instruction;
use p3_field::PrimeField32;
use rrs_lib::instruction_formats::IType;

use crate::util::{nop, unimp};

#[allow(unused)]
fn process_custom_instruction<F: PrimeField32>(instruction_u32: u32) -> Option<Instruction<F>> {
    let opcode = (instruction_u32 & 0x7f) as u8;
    let funct3 = ((instruction_u32 >> 12) & 0b111) as u8; // All our instructions are R-, I- or B-type

    let result = None;

    if result.is_some() {
        return result;
    }

    if opcode == 0b1110011 {
        let dec_insn = IType::new(instruction_u32);
        if dec_insn.funct3 == 0b001 {
            // CSRRW
            if dec_insn.rs1 == 0 && dec_insn.rd == 0 {
                // This resets the CSR counter to zero. Since we don't have any CSR registers, this is a nop.
                return Some(nop());
            }
        }
        eprintln!(
            "Transpiling system / CSR instruction: {:b} (opcode = {:07b}, funct3 = {:03b}) to unimp",
            instruction_u32, opcode, funct3
        );
        return Some(unimp());
    }

    None
}
