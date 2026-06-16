use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_riscv_transpiler::ShiftOpcode;

use super::adapters::{
    Rv64BaseAluAdapterAir, Rv64BaseAluAdapterExecutor, Rv64BaseAluAdapterFiller, RV64_BYTE_BITS,
    RV64_REGISTER_NUM_LIMBS,
};

mod air;
mod core;
mod execution;
mod trace;
// The combined `ShiftCoreAir`/`ShiftExecutor`/`ShiftFiller`/`ShiftCoreCols`/`ShiftCoreRecord`
// are still used by the word-shift chip (`shift_w`) and the bigint `Shift256` chip.
pub use core::*;

pub use air::*;
pub use execution::*;
pub use trace::*;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

/// Per-opcode marker selecting which RISC-V shift a split chip implements.
pub trait ShiftOp: 'static + Send + Sync + Copy + Clone + std::fmt::Debug {
    const OPCODE: ShiftOpcode;
    /// Whether the shift moves bits towards more significant positions (`SLL`).
    const IS_LEFT: bool;

    /// Pure-execution shift over a little-endian RV64 register value.
    fn compute(rs1: [u8; RV64_REGISTER_NUM_LIMBS], rs2: u64) -> [u8; RV64_REGISTER_NUM_LIMBS];
}

#[derive(Clone, Copy, Debug)]
pub struct Sll;
#[derive(Clone, Copy, Debug)]
pub struct Srl;
#[derive(Clone, Copy, Debug)]
pub struct Sra;

impl ShiftOp for Sll {
    const OPCODE: ShiftOpcode = ShiftOpcode::SLL;
    const IS_LEFT: bool = true;
    #[inline(always)]
    fn compute(rs1: [u8; RV64_REGISTER_NUM_LIMBS], rs2: u64) -> [u8; RV64_REGISTER_NUM_LIMBS] {
        let rs1 = u64::from_le_bytes(rs1);
        // RV64: only the low 6 bits of rs2 are used for the shift amount.
        (rs1 << (rs2 & 0x3F)).to_le_bytes()
    }
}
impl ShiftOp for Srl {
    const OPCODE: ShiftOpcode = ShiftOpcode::SRL;
    const IS_LEFT: bool = false;
    #[inline(always)]
    fn compute(rs1: [u8; RV64_REGISTER_NUM_LIMBS], rs2: u64) -> [u8; RV64_REGISTER_NUM_LIMBS] {
        let rs1 = u64::from_le_bytes(rs1);
        (rs1 >> (rs2 & 0x3F)).to_le_bytes()
    }
}
impl ShiftOp for Sra {
    const OPCODE: ShiftOpcode = ShiftOpcode::SRA;
    const IS_LEFT: bool = false;
    #[inline(always)]
    fn compute(rs1: [u8; RV64_REGISTER_NUM_LIMBS], rs2: u64) -> [u8; RV64_REGISTER_NUM_LIMBS] {
        let rs1 = i64::from_le_bytes(rs1);
        (rs1 >> (rs2 & 0x3F)).to_le_bytes()
    }
}

pub type Rv64SllAir = VmAirWrapper<
    Rv64BaseAluAdapterAir,
    LogicalShiftCoreAir<Sll, RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>,
>;
pub type Rv64SrlAir = VmAirWrapper<
    Rv64BaseAluAdapterAir,
    LogicalShiftCoreAir<Srl, RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>,
>;
pub type Rv64SraAir =
    VmAirWrapper<Rv64BaseAluAdapterAir, SraCoreAir<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>>;

pub type Rv64SllExecutor = ShiftSplitExecutor<
    Rv64BaseAluAdapterExecutor<RV64_BYTE_BITS>,
    Sll,
    RV64_REGISTER_NUM_LIMBS,
    RV64_BYTE_BITS,
>;
pub type Rv64SrlExecutor = ShiftSplitExecutor<
    Rv64BaseAluAdapterExecutor<RV64_BYTE_BITS>,
    Srl,
    RV64_REGISTER_NUM_LIMBS,
    RV64_BYTE_BITS,
>;
pub type Rv64SraExecutor = ShiftSplitExecutor<
    Rv64BaseAluAdapterExecutor<RV64_BYTE_BITS>,
    Sra,
    RV64_REGISTER_NUM_LIMBS,
    RV64_BYTE_BITS,
>;

pub type Rv64SllChip<F> = VmChipWrapper<
    F,
    LogicalShiftFiller<
        Rv64BaseAluAdapterFiller<RV64_BYTE_BITS>,
        Sll,
        RV64_REGISTER_NUM_LIMBS,
        RV64_BYTE_BITS,
    >,
>;
pub type Rv64SrlChip<F> = VmChipWrapper<
    F,
    LogicalShiftFiller<
        Rv64BaseAluAdapterFiller<RV64_BYTE_BITS>,
        Srl,
        RV64_REGISTER_NUM_LIMBS,
        RV64_BYTE_BITS,
    >,
>;
pub type Rv64SraChip<F> = VmChipWrapper<
    F,
    ArithShiftFiller<
        Rv64BaseAluAdapterFiller<RV64_BYTE_BITS>,
        RV64_REGISTER_NUM_LIMBS,
        RV64_BYTE_BITS,
    >,
>;
