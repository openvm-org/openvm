use openvm_instructions::{
    instruction::Instruction, program::MAX_ALLOWED_PC, riscv::RV64_REGISTER_NUM_LIMBS,
};
use openvm_stark_backend::p3_field::PrimeField32;

/// Decode register index from an OpenVM operand.
pub fn decode_reg<F: PrimeField32>(f: F) -> u8 {
    (f.as_canonical_u32() / RV64_REGISTER_NUM_LIMBS as u32) as u8
}

/// Decode the immediate from the (c, g) field pair used by JALR, LOAD, STORE,
/// and public-values stores, including REVEAL.
///
/// OpenVM stores the lower 16 bits of the sign-extended immediate in `c`, and
/// the sign bit in `g`. The full 32-bit value is reconstructed as:
///   imm = (c & 0xFFFF) + g * 0xFFFF0000
pub fn decode_imm_cg<F: PrimeField32>(insn: &Instruction<F>) -> u32 {
    let low16 = insn.c.as_canonical_u32() & 0xffff;
    let is_neg = insn.g.as_canonical_u32() != 0;
    low16.wrapping_add(if is_neg { 0xFFFF0000 } else { 0 })
}

/// Sign-extend a 32-bit value into an RV64 register value.
pub fn sext32(value: u32) -> u64 {
    value as i32 as i64 as u64
}

/// True if `pc` lies within the implemented PC address space (`<= MAX_ALLOWED_PC`).
#[inline]
pub fn is_pc_in_bounds(pc: u64) -> bool {
    pc <= u64::from(MAX_ALLOWED_PC)
}
