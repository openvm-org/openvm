use openvm_instructions::{instruction::Instruction, riscv::RV32_REGISTER_NUM_LIMBS};
use openvm_stark_backend::p3_field::PrimeField32;

/// Decode register index from an OpenVM operand.
pub fn decode_reg<F: PrimeField32>(f: F) -> u8 {
    (f.as_canonical_u32() / RV32_REGISTER_NUM_LIMBS as u32) as u8
}

/// Decode the immediate from the (c, g) field pair used by JALR, LOAD, STORE,
/// and REVEAL.
///
/// OpenVM stores the lower 16 bits of the sign-extended immediate in `c`, and
/// the sign bit in `g`. The full 32-bit value is reconstructed as:
///   imm = (c & 0xFFFF) + g * 0xFFFF0000
pub fn decode_imm_cg<F: PrimeField32>(insn: &Instruction<F>) -> u32 {
    let low16 = insn.c.as_canonical_u32() & 0xffff;
    let is_neg = insn.g.as_canonical_u32() != 0;
    low16.wrapping_add(if is_neg { 0xFFFF0000 } else { 0 })
}
