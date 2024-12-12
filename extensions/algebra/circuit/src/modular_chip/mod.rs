mod addsub;
pub use addsub::*;
mod is_eq;
pub use is_eq::*;
mod muldiv;
pub use muldiv::*;
use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_instructions::riscv::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use openvm_rv32_adapters::{
    Rv32IsEqualModAdapterChip, Rv32VecHeapAdapterAir, Rv32VecHeapAdapterChip,
};

#[cfg(test)]
mod tests;

/// Each prime field element will be represented as `NUM_LANES * LANE_SIZE` cells in memory.
/// The `LANE_SIZE` must be a power of 2 and determines the size of the batch memory read/writes.
pub type ModularAddSubAir<const NUM_LANES: usize, const LANE_SIZE: usize> = VmAirWrapper<
    Rv32VecHeapAdapterAir<2, NUM_LANES, NUM_LANES, LANE_SIZE, LANE_SIZE>,
    ModularAddSubCoreAir,
>;
/// See [ModularAddSubAir].
pub type ModularAddSubChip<F, const NUM_LANES: usize, const LANE_SIZE: usize> = VmChipWrapper<
    F,
    Rv32VecHeapAdapterChip<F, 2, NUM_LANES, NUM_LANES, LANE_SIZE, LANE_SIZE>,
    ModularAddSubCoreChip,
>;
/// Each prime field element will be represented as `NUM_LANES * LANE_SIZE` cells in memory.
/// The `LANE_SIZE` must be a power of 2 and determines the size of the batch memory read/writes.
pub type ModularMulDivAir<const NUM_LANES: usize, const LANE_SIZE: usize> = VmAirWrapper<
    Rv32VecHeapAdapterAir<2, NUM_LANES, NUM_LANES, LANE_SIZE, LANE_SIZE>,
    ModularMulDivCoreAir,
>;
/// See [ModularMulDivAir].
pub type ModularMulDivChip<F, const NUM_LANES: usize, const LANE_SIZE: usize> = VmChipWrapper<
    F,
    Rv32VecHeapAdapterChip<F, 2, NUM_LANES, NUM_LANES, LANE_SIZE, LANE_SIZE>,
    ModularMulDivCoreChip,
>;

// Must have TOTAL_LIMBS = NUM_LANES * LANE_SIZE
pub type ModularIsEqualChip<
    F,
    const NUM_LANES: usize,
    const LANE_SIZE: usize,
    const TOTAL_LIMBS: usize,
> = VmChipWrapper<
    F,
    Rv32IsEqualModAdapterChip<F, 2, NUM_LANES, LANE_SIZE, TOTAL_LIMBS>,
    ModularIsEqualCoreChip<TOTAL_LIMBS, RV32_REGISTER_NUM_LIMBS, RV32_CELL_BITS>,
>;
