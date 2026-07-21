use openvm_instructions::{program::MAX_ALLOWED_PC, riscv::RV64_REGISTER_BYTES};

use crate::RvrInstruction;

/// Decode register index from an OpenVM operand.
pub fn decode_reg(value: u32) -> u8 {
    let value = u64::from(value);
    assert_eq!(
        value % RV64_REGISTER_BYTES,
        0,
        "register operand must be register-aligned"
    );
    let index = value / RV64_REGISTER_BYTES;
    assert!(index < 32, "register operand must name x0..x31");
    u8::try_from(index).unwrap()
}

/// Decode the immediate from the (c, g) field pair used by JALR, LOAD, STORE,
/// and public-values stores, including REVEAL.
///
/// OpenVM stores the lower 16 bits of the sign-extended immediate in `c`, and
/// the sign bit in `g`. The full 32-bit value is reconstructed as:
///   imm = (c & 0xFFFF) + g * 0xFFFF0000
pub fn decode_imm_cg(insn: &RvrInstruction) -> u32 {
    let low16 = insn.c & 0xffff;
    let is_neg = insn.g != 0;
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
