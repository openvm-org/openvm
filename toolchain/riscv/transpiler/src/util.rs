use std::collections::BTreeMap;

use axvm_instructions::{NopOpcode, TerminateOpcode, UsizeOpcode};
use p3_field::PrimeField32;
use rrs_lib::instruction_formats::{BType, IType, ITypeShamt, JType, RType, SType, UType};
use stark_vm::{
    rv32im::adapters::RV32_REGISTER_NUM_LIMBS,
    system::{
        memory::Equipartition,
        program::{isize_to_field, Instruction},
    },
};

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
        String::new(),
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
        String::new(),
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
        String::new(),
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
        String::new(),
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
        String::new(),
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
        String::new(),
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
        String::new(),
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
        String::new(),
    )
}

/// Create a new [`Instruction`] that exits with code 1.
pub fn unimp<F: PrimeField32>() -> Instruction<F> {
    Instruction {
        opcode: TerminateOpcode::TERMINATE.with_default_offset(),
        c: F::one(),
        ..Default::default()
    }
}

pub fn nop<F: PrimeField32>() -> Instruction<F> {
    Instruction {
        opcode: NopOpcode::NOP.with_default_offset(),
        ..Default::default()
    }
}

pub fn terminate<F: PrimeField32>() -> Instruction<F> {
    Instruction {
        opcode: TerminateOpcode::TERMINATE.with_default_offset(),
        ..Default::default()
    }
}

/// Converts our memory image (u32 -> u32) into the Equipartition for Poseidon ((as, label) -> [F; 8])
pub fn memory_image_to_equipartition<F: PrimeField32>(
    memory_image: BTreeMap<u32, u32>,
) -> Equipartition<F, 8> {
    let mut result = Equipartition::new();
    for (addr, word) in memory_image {
        let key = (F::two(), (addr / 8) as usize);
        let shift = addr as usize % 8;
        result.entry(key).or_insert([F::zero(); 8])[shift..shift + 4]
            .copy_from_slice(&word.to_le_bytes().map(F::from_canonical_u8));
    }
    result
}
