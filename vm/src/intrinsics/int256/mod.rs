use crate::{
    arch::VmChipWrapper,
    rv32im::{
        adapters::{Rv32HeapAdapterChip, INT256_NUM_LIMBS, RV32_CELL_BITS},
        BaseAluCoreChip, LessThanCoreChip, MultiplicationCoreChip, ShiftCoreChip,
    },
};

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
