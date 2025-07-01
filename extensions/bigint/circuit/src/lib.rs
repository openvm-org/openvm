// use openvm_circuit::{
//     self,
//     arch::{MatrixRecordArena, NewVmChipWrapper, VmAirWrapper},
// };
// use openvm_rv32_adapters::{
//     Rv32HeapAdapterAir, Rv32HeapAdapterStep, Rv32HeapBranchAdapterAir, Rv32HeapBranchAdapterStep,
// };
// use openvm_rv32im_circuit::{
//     adapters::{INT256_NUM_LIMBS, RV32_CELL_BITS},
//     BaseAluCoreAir, BaseAluStep, BranchEqualCoreAir, BranchEqualStep, BranchLessThanCoreAir,
//     BranchLessThanStep, LessThanCoreAir, LessThanStep, MultiplicationCoreAir, MultiplicationStep,
//     ShiftCoreAir, ShiftStep,
// };

mod extension;
pub use extension::*;

mod base_alu;
mod branch_eq;
mod branch_lt;
pub(crate) mod common;
mod less_than;
mod mult;
mod shift;
#[cfg(test)]
mod tests;

pub use base_alu::*;
pub use branch_eq::*;
pub use branch_lt::*;
pub use less_than::*;
pub use mult::*;
pub use shift::*;

pub(crate) const INT256_NUM_LIMBS: usize = 32;
pub(crate) const RV32_CELL_BITS: usize = 8;
