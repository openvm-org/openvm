use openvm_circuit::{
    self,
    arch::{MatrixRecordArena, NewVmChipWrapper, VmAirWrapper},
};
use openvm_rv32_adapters::{
    Rv32HeapAdapterAir, Rv32HeapAdapterStep, Rv32HeapBranchAdapterAir, Rv32HeapBranchAdapterStep,
};
use openvm_rv32im_circuit::{
    adapters::{INT256_NUM_LIMBS, RV32_CELL_BITS},
    BaseAluCoreAir, BaseAluStep, BranchEqualCoreAir, BranchEqualStep, BranchLessThanCoreAir,
    BranchLessThanStep, LessThanCoreAir, LessThanStep, MultiplicationCoreAir, MultiplicationStep,
    ShiftCoreAir, ShiftStep,
};

mod extension;
pub use extension::*;

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
pub type Rv32BaseAlu256Chip<F> =
    NewVmChipWrapper<F, Rv32BaseAlu256Air, Rv32BaseAlu256Step, MatrixRecordArena<F>>;

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
pub type Rv32LessThan256Chip<F> =
    NewVmChipWrapper<F, Rv32LessThan256Air, Rv32LessThan256Step, MatrixRecordArena<F>>;

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
pub type Rv32Multiplication256Chip<F> =
    NewVmChipWrapper<F, Rv32Multiplication256Air, Rv32Multiplication256Step, MatrixRecordArena<F>>;

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
pub type Rv32Shift256Chip<F> =
    NewVmChipWrapper<F, Rv32Shift256Air, Rv32Shift256Step, MatrixRecordArena<F>>;

/// BranchEqual256
pub type Rv32BranchEqual256Air = VmAirWrapper<
    Rv32HeapBranchAdapterAir<2, INT256_NUM_LIMBS>,
    BranchEqualCoreAir<INT256_NUM_LIMBS>,
>;
pub type Rv32BranchEqual256Step =
    BranchEqualStep<Rv32HeapBranchAdapterStep<2, INT256_NUM_LIMBS>, INT256_NUM_LIMBS>;
pub type Rv32BranchEqual256Chip<F> =
    NewVmChipWrapper<F, Rv32BranchEqual256Air, Rv32BranchEqual256Step, MatrixRecordArena<F>>;

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
pub type Rv32BranchLessThan256Chip<F> =
    NewVmChipWrapper<F, Rv32BranchLessThan256Air, Rv32BranchLessThan256Step, MatrixRecordArena<F>>;
