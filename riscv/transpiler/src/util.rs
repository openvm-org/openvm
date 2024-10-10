use p3_field::PrimeField32;
use rrs_lib::instruction_formats::{BType, IType, ITypeShamt, RType, SType};
use stark_vm::{
    arch::instructions::CoreOpcode,
    program::{isize_to_field, Instruction},
};

fn i12_to_u24(imm: i32) -> u32 {
    let imm_u = (imm as u32) & 0xfff;
    imm_u | ((imm_u & 0x80) * 0x17e)
}

/// Create a new [`Instruction`] from an R-type instruction.
pub fn from_r_type<F: PrimeField32>(opcode: usize, dec_insn: &RType) -> Instruction<F> {
    Instruction::new(
        opcode,
        F::from_canonical_u32(dec_insn.rd as u32),
        F::from_canonical_u32(dec_insn.rs1 as u32),
        F::from_canonical_u32(dec_insn.rs2 as u32),
        F::zero(),
        F::zero(),
        F::zero(),
        F::zero(),
        String::new(),
    )
}

/// Create a new [`Instruction`] from an I-type instruction.
pub fn from_i_type<F: PrimeField32>(opcode: usize, dec_insn: &IType) -> Instruction<F> {
    Instruction::new(
        opcode,
        F::from_canonical_u32(dec_insn.rd as u32),
        F::from_canonical_u32(dec_insn.rs1 as u32),
        F::from_canonical_u32(i12_to_u24(dec_insn.imm)),
        F::zero(),
        F::one(),
        F::zero(),
        F::zero(),
        String::new(),
    )
}

/// Create a new [`Instruction`] from an I-type instruction with a shamt.
/// It seems that shamt can only occur in SLLI, SRLI, SRAI.
pub fn from_i_type_shamt<F: PrimeField32>(opcode: usize, dec_insn: &ITypeShamt) -> Instruction<F> {
    Instruction::new(
        opcode,
        F::from_canonical_u32(dec_insn.rd as u32),
        F::from_canonical_u32(dec_insn.rs1 as u32),
        F::from_canonical_u32(i12_to_u24(dec_insn.shamt as i32)),
        F::zero(),
        F::one(),
        F::zero(),
        F::zero(),
        String::new(),
    )
}

/// Create a new [`Instruction`] from an S-type instruction.
pub fn from_s_type<F: PrimeField32>(opcode: usize, dec_insn: &SType) -> Instruction<F> {
    Instruction::new(
        opcode,
        F::from_canonical_u32(dec_insn.rs2 as u32),
        F::from_canonical_u32(dec_insn.rs1 as u32),
        F::from_canonical_u32(dec_insn.imm as u32),
        F::zero(),
        F::one(),
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
        F::from_canonical_u32(dec_insn.rs1 as u32),
        F::from_canonical_u32(dec_insn.rs2 as u32),
        isize_to_field(dec_insn.imm as isize / 2),
        F::one(),
        F::one(),
        F::zero(),
        F::zero(),
        String::new(),
    )
}

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
