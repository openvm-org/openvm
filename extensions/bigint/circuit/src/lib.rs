use openvm_circuit::{
    self,
    arch::{VmAirWrapper, VmChipWrapper},
};
use openvm_rv32_adapters::{
    Rv32HeapAdapterAir, Rv32HeapAdapterFiller, Rv32HeapAdapterStep, Rv32HeapBranchAdapterAir,
    Rv32HeapBranchAdapterFiller, Rv32HeapBranchAdapterStep,
};
use openvm_rv32im_circuit::{
    adapters::{INT256_NUM_LIMBS, RV32_CELL_BITS},
    BaseAluCoreAir, BaseAluFiller, BaseAluStep, BranchEqualCoreAir, BranchEqualFiller,
    BranchEqualStep, BranchLessThanCoreAir, BranchLessThanFiller, BranchLessThanStep,
    LessThanCoreAir, LessThanFiller, LessThanStep, MultiplicationCoreAir, MultiplicationFiller,
    MultiplicationStep, ShiftCoreAir, ShiftFiller, ShiftStep,
};

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

/// BaseAlu256
pub type Rv32BaseAlu256Air = VmAirWrapper<
    Rv32HeapAdapterAir<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
    BaseAluCoreAir<INT256_NUM_LIMBS, RV32_CELL_BITS>,
>;
pub type Rv32BaseAlu256Step = BaseAluStep<
    Rv32HeapAdapterStep<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
    INT256_NUM_LIMBS,
    RV32_CELL_BITS,
>;
pub type Rv32BaseAlu256Chip<F> = VmChipWrapper<
    F,
    BaseAluFiller<
        Rv32HeapAdapterFiller<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
        INT256_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;

/// LessThan256
pub type Rv32LessThan256Air = VmAirWrapper<
    Rv32HeapAdapterAir<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
    LessThanCoreAir<INT256_NUM_LIMBS, RV32_CELL_BITS>,
>;
pub type Rv32LessThan256Step = LessThanStep<
    Rv32HeapAdapterStep<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
    INT256_NUM_LIMBS,
    RV32_CELL_BITS,
>;
pub type Rv32LessThan256Chip<F> = VmChipWrapper<
    F,
    LessThanFiller<
        Rv32HeapAdapterFiller<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
        INT256_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;

/// Multiplication256
pub type Rv32Multiplication256Air = VmAirWrapper<
    Rv32HeapAdapterAir<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
    MultiplicationCoreAir<INT256_NUM_LIMBS, RV32_CELL_BITS>,
>;
pub type Rv32Multiplication256Step = MultiplicationStep<
    Rv32HeapAdapterStep<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
    INT256_NUM_LIMBS,
    RV32_CELL_BITS,
>;
pub type Rv32Multiplication256Chip<F> = VmChipWrapper<
    F,
    MultiplicationFiller<
        Rv32HeapAdapterFiller<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
        INT256_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;

/// Shift256
pub type Rv32Shift256Air = VmAirWrapper<
    Rv32HeapAdapterAir<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
    ShiftCoreAir<INT256_NUM_LIMBS, RV32_CELL_BITS>,
>;
pub type Rv32Shift256Step = ShiftStep<
    Rv32HeapAdapterStep<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
    INT256_NUM_LIMBS,
    RV32_CELL_BITS,
>;
pub type Rv32Shift256Chip<F> = VmChipWrapper<
    F,
    ShiftFiller<
        Rv32HeapAdapterFiller<2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
        INT256_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;

/// BranchEqual256
pub type Rv32BranchEqual256Air = VmAirWrapper<
    Rv32HeapBranchAdapterAir<2, INT256_NUM_LIMBS>,
    BranchEqualCoreAir<INT256_NUM_LIMBS>,
>;
pub type Rv32BranchEqual256Step =
    BranchEqualStep<Rv32HeapBranchAdapterStep<2, INT256_NUM_LIMBS>, INT256_NUM_LIMBS>;
pub type Rv32BranchEqual256Chip<F> = VmChipWrapper<
    F,
    BranchEqualFiller<Rv32HeapBranchAdapterFiller<2, INT256_NUM_LIMBS>, INT256_NUM_LIMBS>,
>;

/// BranchLessThan256
pub type Rv32BranchLessThan256Air = VmAirWrapper<
    Rv32HeapBranchAdapterAir<2, INT256_NUM_LIMBS>,
    BranchLessThanCoreAir<INT256_NUM_LIMBS, RV32_CELL_BITS>,
>;
pub type Rv32BranchLessThan256Step = BranchLessThanStep<
    Rv32HeapBranchAdapterStep<2, INT256_NUM_LIMBS>,
    INT256_NUM_LIMBS,
    RV32_CELL_BITS,
>;
pub type Rv32BranchLessThan256Chip<F> = VmChipWrapper<
    F,
    BranchLessThanFiller<
        Rv32HeapBranchAdapterFiller<2, INT256_NUM_LIMBS>,
        INT256_NUM_LIMBS,
        RV32_CELL_BITS,
    >,
>;
