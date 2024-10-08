use rrs_lib::instruction_formats::{IType, RType};
use stark_vm::{arch::instructions::CoreOpcode, program::Instruction};

/// Create a new [`Instruction`] from an R-type instruction.
pub fn from_r_type(opcode: usize, dec_insn: &RType) -> Instruction<u32> {
    Instruction::new(
        opcode,
        dec_insn.rd as u32,
        dec_insn.rs1 as u32,
        dec_insn.rs2 as u32,
        0,
        0,
        0,
        0,
        String::new(),
    )
}

/// Create a new [`Instruction`] from an I-type instruction.
pub fn from_i_type(opcode: usize, dec_insn: &IType) -> Instruction<u32> {
    Instruction::new(
        opcode,
        dec_insn.rd as u32,
        dec_insn.rs1 as u32,
        dec_insn.imm as u32,
        0,
        1,
        0,
        0,
        String::new(),
    )
}

// /// Create a new [`Instruction`] from an I-type instruction with a shamt.
// #[must_use]
// pub const fn from_i_type_shamt(opcode: Opcode, dec_insn: &ITypeShamt) -> Self {
//     Self::new(
//         opcode,
//         dec_insn.rd as u32,
//         dec_insn.rs1 as u32,
//         dec_insn.shamt,
//         false,
//         true,
//     )
// }

// /// Create a new [`Instruction`] from an S-type instruction.
// #[must_use]
// pub const fn from_s_type(opcode: Opcode, dec_insn: &SType) -> Self {
//     Self::new(
//         opcode,
//         dec_insn.rs2 as u32,
//         dec_insn.rs1 as u32,
//         dec_insn.imm as u32,
//         false,
//         true,
//     )
// }

// /// Create a new [`Instruction`] from a B-type instruction.
// #[must_use]
// pub const fn from_b_type(opcode: Opcode, dec_insn: &BType) -> Self {
//     Self::new(
//         opcode,
//         dec_insn.rs1 as u32,
//         dec_insn.rs2 as u32,
//         dec_insn.imm as u32,
//         false,
//         true,
//     )
// }

/// Create a new [`Instruction`] that is not implemented.
pub fn unimp() -> Instruction<u32> {
    Instruction::new(
        CoreOpcode::FAIL as usize,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        String::new(),
    )
}

/// Returns if the [`Instruction`] is an R-type instruction.
#[inline]
pub const fn is_r_type(instruction: &Instruction<u32>) -> bool {
    instruction.e == 0 // TODO: 0 -> REGISTER_AS or smth?
}

/// Returns whether the [`Instruction`] is an I-type instruction.
#[inline]
pub const fn is_i_type(instruction: &Instruction<u32>) -> bool {
    instruction.e == 1 // TODO: 0 -> IMMEDIATE_AS or smth?
}

// /// Decode the [`Instruction`] in the R-type format.
// #[inline]
// pub fn r_type(instruction: &Instruction<u32>) -> (Register, Register, Register) {
//     (
//         Register::from_u32(instruction.op_a),
//         Register::from_u32(self.op_b),
//         Register::from_u32(self.op_c),
//     )
// }

// /// Decode the [`Instruction`] in the I-type format.
// #[inline]
// #[must_use]
// pub fn i_type(&self) -> (Register, Register, u32) {
//     (
//         Register::from_u32(self.op_a),
//         Register::from_u32(self.op_b),
//         self.op_c,
//     )
// }

// /// Decode the [`Instruction`] in the S-type format.
// #[inline]
// #[must_use]
// pub fn s_type(&self) -> (Register, Register, u32) {
//     (
//         Register::from_u32(self.op_a),
//         Register::from_u32(self.op_b),
//         self.op_c,
//     )
// }

// /// Decode the [`Instruction`] in the B-type format.
// #[inline]
// #[must_use]
// pub fn b_type(&self) -> (Register, Register, u32) {
//     (
//         Register::from_u32(self.op_a),
//         Register::from_u32(self.op_b),
//         self.op_c,
//     )
// }

// /// Decode the [`Instruction`] in the J-type format.
// #[inline]
// #[must_use]
// pub fn j_type(&self) -> (Register, u32) {
//     (Register::from_u32(self.op_a), self.op_b)
// }

// /// Decode the [`Instruction`] in the U-type format.
// #[inline]
// #[must_use]
// pub fn u_type(&self) -> (Register, u32) {
//     (Register::from_u32(self.op_a), self.op_b)
// }
