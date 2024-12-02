use axvm_circuit::{self, arch::VmChipWrapper};
use axvm_rv32_adapters::{Rv32HeapAdapterChip, Rv32HeapBranchAdapterChip};
use axvm_rv32im_circuit::{
    adapters::{INT256_NUM_LIMBS, RV32_CELL_BITS},
    BaseAluCoreChip, BranchEqualCoreChip, BranchLessThanCoreChip, LessThanCoreChip,
    MultiplicationCoreChip, ShiftCoreChip,
};

mod extension;
pub use extension::*;

#[cfg(test)]
mod tests;

pub type Rv32BaseAlu256Chip<F> = VmChipWrapper<
    F,
    Rv32HeapAdapterChip<F, 2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
    BaseAluCoreChip<INT256_NUM_LIMBS, RV32_CELL_BITS>,
>;

pub type Rv32LessThan256Chip<F> = VmChipWrapper<
    F,
    Rv32HeapAdapterChip<F, 2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
    LessThanCoreChip<INT256_NUM_LIMBS, RV32_CELL_BITS>,
>;

pub type Rv32Multiplication256Chip<F> = VmChipWrapper<
    F,
    Rv32HeapAdapterChip<F, 2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
    MultiplicationCoreChip<INT256_NUM_LIMBS, RV32_CELL_BITS>,
>;

pub type Rv32Shift256Chip<F> = VmChipWrapper<
    F,
    Rv32HeapAdapterChip<F, 2, INT256_NUM_LIMBS, INT256_NUM_LIMBS>,
    ShiftCoreChip<INT256_NUM_LIMBS, RV32_CELL_BITS>,
>;

pub type Rv32BranchEqual256Chip<F> = VmChipWrapper<
    F,
    Rv32HeapBranchAdapterChip<F, 2, INT256_NUM_LIMBS>,
    BranchEqualCoreChip<INT256_NUM_LIMBS>,
>;

pub type Rv32BranchLessThan256Chip<F> = VmChipWrapper<
    F,
    Rv32HeapBranchAdapterChip<F, 2, INT256_NUM_LIMBS>,
    BranchLessThanCoreChip<INT256_NUM_LIMBS, RV32_CELL_BITS>,
>;
