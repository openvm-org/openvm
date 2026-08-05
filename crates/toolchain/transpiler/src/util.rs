use std::collections::BTreeMap;

use openvm_decoder::instruction_formats::{BType, IType, ITypeShamt, JType, RType, SType, UType};
use openvm_instructions::{
    exe::SparseMemoryImage,
    instruction::{Instruction, InstructionOperand},
    program::DEFAULT_PC_STEP,
    riscv::{MEMORY_AS, REGISTER_NUM_LIMBS},
    LocalOpcode, SystemOpcode, VmOpcode,
};

fn i12_to_u24(imm: i32) -> u32 {
    (imm as u32) & 0xffffff
}

/// Create a new [`Instruction`] from an R-type instruction.
pub fn from_r_type(
    opcode: usize,
    e_as: usize,
    dec_insn: &RType,
    allow_rd_zero: bool,
) -> Instruction {
    // If `rd` is not allowed to be zero, we transpile to `NOP` to prevent a write
    // to `x0`. In the cases where `allow_rd_zero` is true, it is the responsibility of
    // the caller to guarantee that the resulting instruction does not write to `rd`.
    if !allow_rd_zero && dec_insn.rd == 0 {
        return nop();
    }
    Instruction::new(
        VmOpcode::from_usize(opcode),
        InstructionOperand::from_usize(REGISTER_NUM_LIMBS * dec_insn.rd),
        InstructionOperand::from_usize(REGISTER_NUM_LIMBS * dec_insn.rs1),
        InstructionOperand::from_usize(REGISTER_NUM_LIMBS * dec_insn.rs2),
        1u8,                                  // rd and rs1 are registers
        InstructionOperand::from_usize(e_as), // rs2 can be mem (eg modular arith)
        0u8,
        0u8,
    )
}

/// Create a new [`Instruction`] from an I-type instruction. Should only be used for ALU
/// instructions because `imm` is transpiled in a special way.
pub fn from_i_type(opcode: usize, dec_insn: &IType) -> Instruction {
    if dec_insn.rd == 0 {
        return nop();
    }
    Instruction::new(
        VmOpcode::from_usize(opcode),
        InstructionOperand::from_usize(REGISTER_NUM_LIMBS * dec_insn.rd),
        InstructionOperand::from_usize(REGISTER_NUM_LIMBS * dec_insn.rs1),
        InstructionOperand::from_u32(i12_to_u24(dec_insn.imm)),
        1u8, // rd and rs1 are registers
        0u8, // rs2 is an immediate
        0u8,
        0u8,
    )
}

/// Create a new [`Instruction`] from a load operation
pub fn from_load(opcode: usize, dec_insn: &IType) -> Instruction {
    Instruction::new(
        VmOpcode::from_usize(opcode),
        InstructionOperand::from_usize(REGISTER_NUM_LIMBS * dec_insn.rd),
        InstructionOperand::from_usize(REGISTER_NUM_LIMBS * dec_insn.rs1),
        InstructionOperand::from_u32((dec_insn.imm as u32) & 0xffff),
        1u8,              // rd is a register
        2u8,              // we load from memory
        dec_insn.rd != 0, // we may need to use this flag in the operation
        dec_insn.imm < 0, // flag for sign extension
    )
}

/// Create a new [`Instruction`] from an I-type instruction with a shamt.
/// It seems that shamt can only occur in SLLI, SRLI, SRAI.
pub fn from_i_type_shamt(opcode: usize, dec_insn: &ITypeShamt) -> Instruction {
    if dec_insn.rd == 0 {
        return nop();
    }
    Instruction::new(
        VmOpcode::from_usize(opcode),
        InstructionOperand::from_usize(REGISTER_NUM_LIMBS * dec_insn.rd),
        InstructionOperand::from_usize(REGISTER_NUM_LIMBS * dec_insn.rs1),
        InstructionOperand::from_u32(dec_insn.shamt),
        1u8, // rd and rs1 are registers
        0u8, // rs2 is an immediate
        0u8,
        0u8,
    )
}

/// Create a new [`Instruction`] from an S-type instruction.
pub fn from_s_type(opcode: usize, dec_insn: &SType) -> Instruction {
    Instruction::new(
        VmOpcode::from_usize(opcode),
        InstructionOperand::from_usize(REGISTER_NUM_LIMBS * dec_insn.rs2),
        InstructionOperand::from_usize(REGISTER_NUM_LIMBS * dec_insn.rs1),
        InstructionOperand::from_u32((dec_insn.imm as u32) & 0xffff),
        1u8,
        2u8,
        1u8,
        dec_insn.imm < 0,
    )
}

/// Create a new [`Instruction`] from a B-type instruction.
///
/// The branch offset must be `DEFAULT_PC_STEP`-aligned: the circuit represents pc values in
/// units of `DEFAULT_PC_STEP`, so a misaligned offset has no sound encoding. Without the C
/// extension, RISC-V branch targets are always 4-byte aligned.
pub fn from_b_type(opcode: usize, dec_insn: &BType) -> Instruction {
    assert_eq!(
        dec_insn.imm % DEFAULT_PC_STEP as i32,
        0,
        "branch offset must be a multiple of DEFAULT_PC_STEP"
    );
    Instruction::new(
        VmOpcode::from_usize(opcode),
        InstructionOperand::from_usize(REGISTER_NUM_LIMBS * dec_insn.rs1),
        InstructionOperand::from_usize(REGISTER_NUM_LIMBS * dec_insn.rs2),
        InstructionOperand::from_i32(dec_insn.imm),
        1u8, // rs1 is a register
        1u8, // rs2 is a register
        0u8,
        0u8,
    )
}

/// Create a new [`Instruction`] from a J-type instruction.
///
/// The jump offset must be `DEFAULT_PC_STEP`-aligned; see [`from_b_type`].
pub fn from_j_type(opcode: usize, dec_insn: &JType) -> Instruction {
    assert_eq!(
        dec_insn.imm % DEFAULT_PC_STEP as i32,
        0,
        "jump offset must be a multiple of DEFAULT_PC_STEP"
    );
    Instruction::new(
        VmOpcode::from_usize(opcode),
        InstructionOperand::from_usize(REGISTER_NUM_LIMBS * dec_insn.rd),
        0u8,
        InstructionOperand::from_i32(dec_insn.imm),
        1u8, // rd is a register
        0u8,
        dec_insn.rd != 0, // we may need to use this flag in the operation
        0u8,
    )
}

/// Create a new [`Instruction`] from a U-type instruction.
pub fn from_u_type(opcode: usize, dec_insn: &UType) -> Instruction {
    if dec_insn.rd == 0 {
        return nop();
    }
    Instruction::new(
        VmOpcode::from_usize(opcode),
        InstructionOperand::from_usize(REGISTER_NUM_LIMBS * dec_insn.rd),
        0u8,
        InstructionOperand::from_u32((dec_insn.imm as u32 >> 12) & 0xfffff),
        1u8, // rd is a register
        0u8,
        0u8,
        0u8,
    )
}

/// Create a new [`Instruction`] that exits with code 2. This is equivalent to program panic but
/// with a special exit code for debugging.
pub fn unimp() -> Instruction {
    Instruction {
        opcode: SystemOpcode::TERMINATE.global_opcode(),
        c: InstructionOperand::TWO,
        ..Default::default()
    }
}

pub fn nop() -> Instruction {
    Instruction {
        opcode: SystemOpcode::PHANTOM.global_opcode(),
        ..Default::default()
    }
}

/// Converts our memory image (u32 -> [u8; 4]) into Vm memory image ((as=2, address) -> byte)
pub fn elf_memory_image_to_openvm_memory_image(
    memory_image: BTreeMap<u32, u32>,
) -> SparseMemoryImage {
    let mut result = SparseMemoryImage::new();
    for (addr, word) in memory_image {
        for (i, byte) in word.to_le_bytes().into_iter().enumerate() {
            result.insert((MEMORY_AS, addr + i as u32), byte);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use openvm_decoder::instruction_formats::{BType, JType};

    use super::{from_b_type, from_j_type};

    #[test]
    fn branch_and_jal_immediates_stay_signed() {
        let branch = from_b_type(
            1,
            &BType {
                imm: -4096,
                rs2: 2,
                rs1: 1,
                funct3: 0,
            },
        );
        assert_eq!(branch.c.as_i32(), -4096);

        let jal = from_j_type(
            2,
            &JType {
                imm: -1_048_576,
                rd: 1,
            },
        );
        assert_eq!(jal.c.as_i32(), -1_048_576);
    }
}
