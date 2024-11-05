use std::collections::BTreeMap;

use axvm_instructions::{
    exe::MemoryImage, instruction::Instruction, riscv::RV32_REGISTER_NUM_LIMBS,
    utils::isize_to_field, SystemOpcode, UsizeOpcode,
};
use p3_field::PrimeField32;
use rrs_lib::instruction_formats::{BType, IType, ITypeShamt, JType, RType, SType, UType};

fn i12_to_u24(imm: i32) -> u32 {
    (imm as u32) & 0xffffff
}

/// Create a new [`Instruction`] from an R-type instruction.
pub fn from_r_type<F: PrimeField32>(
    opcode: usize,
    e_as: usize,
    dec_insn: &RType,
) -> Instruction<F> {
    if dec_insn.rd == 0 {
        return nop();
    }
    Instruction::new(
        opcode,
        F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rd),
        F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rs1),
        F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rs2),
        F::one(),                      // rd and rs1 are registers
        F::from_canonical_usize(e_as), // rs2 can be mem (eg modular arith)
        F::zero(),
        F::zero(),
    )
}

/// Create a new [`Instruction`] from an I-type instruction. Should only be used for ALU instructions because `imm` is transpiled in a special way.
pub fn from_i_type<F: PrimeField32>(opcode: usize, dec_insn: &IType) -> Instruction<F> {
    if dec_insn.rd == 0 {
        return nop();
    }
    Instruction::new(
        opcode,
        F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rd),
        F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rs1),
        F::from_canonical_u32(i12_to_u24(dec_insn.imm)),
        F::one(),  // rd and rs1 are registers
        F::zero(), // rs2 is an immediate
        F::zero(),
        F::zero(),
    )
}

/// Create a new [`Instruction`] from a load operation
pub fn from_load<F: PrimeField32>(opcode: usize, dec_insn: &IType) -> Instruction<F> {
    if dec_insn.rd == 0 {
        return nop();
    }
    Instruction::new(
        opcode,
        F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rd),
        F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rs1),
        F::from_canonical_u32((dec_insn.imm as u32) & 0xffff),
        F::one(), // rd is a register
        F::two(), // we load from memory
        F::zero(),
        F::zero(),
    )
}

/// Create a new [`Instruction`] from an I-type instruction with a shamt.
/// It seems that shamt can only occur in SLLI, SRLI, SRAI.
pub fn from_i_type_shamt<F: PrimeField32>(opcode: usize, dec_insn: &ITypeShamt) -> Instruction<F> {
    if dec_insn.rd == 0 {
        return nop();
    }
    Instruction::new(
        opcode,
        F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rd),
        F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rs1),
        F::from_canonical_u32(dec_insn.shamt),
        F::one(),  // rd and rs1 are registers
        F::zero(), // rs2 is an immediate
        F::zero(),
        F::zero(),
    )
}

/// Create a new [`Instruction`] from an S-type instruction.
pub fn from_s_type<F: PrimeField32>(opcode: usize, dec_insn: &SType) -> Instruction<F> {
    Instruction::new(
        opcode,
        F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rs2),
        F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rs1),
        F::from_canonical_u32((dec_insn.imm as u32) & 0xffff),
        F::one(),
        F::two(),
        F::zero(),
        F::zero(),
    )
}

// TODO: implement J and U, prove or disprove that the address spaces are currently correct

/// Create a new [`Instruction`] from a B-type instruction.
pub fn from_b_type<F: PrimeField32>(opcode: usize, dec_insn: &BType) -> Instruction<F> {
    Instruction::new(
        opcode,
        F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rs1),
        F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rs2),
        isize_to_field(dec_insn.imm as isize),
        F::one(), // rs1 is a register
        F::one(), // rs2 is a register
        F::zero(),
        F::zero(),
    )
}

/// Create a new [`Instruction`] from a J-type instruction.
pub fn from_j_type<F: PrimeField32>(opcode: usize, dec_insn: &JType) -> Instruction<F> {
    Instruction::new(
        opcode,
        F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rd),
        F::zero(),
        isize_to_field(dec_insn.imm as isize),
        F::one(), // rd is a register
        F::zero(),
        F::from_bool(dec_insn.rd != 0), // we may need to use this flag in the operation
        F::zero(),
    )
}

/// Create a new [`Instruction`] from a U-type instruction.
pub fn from_u_type<F: PrimeField32>(opcode: usize, dec_insn: &UType) -> Instruction<F> {
    if dec_insn.rd == 0 {
        return nop();
    }
    Instruction::new(
        opcode,
        F::from_canonical_usize(RV32_REGISTER_NUM_LIMBS * dec_insn.rd),
        F::zero(),
        F::from_canonical_u32((dec_insn.imm as u32 >> 12) & 0xfffff),
        F::one(), // rd is a register
        F::zero(),
        F::zero(),
        F::zero(),
    )
}

/// Create a new [`Instruction`] that exits with code 2. This is equivalent to program panic but with a special exit code for debugging.
pub fn unimp<F: PrimeField32>() -> Instruction<F> {
    Instruction {
        opcode: SystemOpcode::TERMINATE.with_default_offset(),
        c: F::two(),
        ..Default::default()
    }
}

pub fn nop<F: PrimeField32>() -> Instruction<F> {
    Instruction {
        opcode: SystemOpcode::PHANTOM.with_default_offset(),
        ..Default::default()
    }
}

pub fn terminate<F: PrimeField32>(code: u8) -> Instruction<F> {
    Instruction {
        opcode: SystemOpcode::TERMINATE.with_default_offset(),
        c: F::from_canonical_u8(code),
        ..Default::default()
    }
}

/// Converts our memory image (u32 -> [u8; 4]) into AxVm memory image ((as, address) -> word)
pub fn elf_memory_image_to_axvm_memory_image<F: PrimeField32>(
    memory_image: BTreeMap<u32, u32>,
) -> MemoryImage<F> {
    let mut result = MemoryImage::new();
    for (addr, word) in memory_image {
        for (i, byte) in word.to_le_bytes().into_iter().enumerate() {
            result.insert(
                (F::two(), F::from_canonical_u32(addr + i as u32)),
                F::from_canonical_u8(byte),
            );
        }
    }
    result
}
