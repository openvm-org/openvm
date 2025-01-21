mod addsub;
pub use addsub::*;
mod is_eq;
pub use is_eq::*;
mod muldiv;
pub use muldiv::*;
use openvm_circuit::arch::VmChipWrapper;
use openvm_instructions::riscv::{RV32_CELL_BITS, RV32_REGISTER_NUM_LIMBS};
use openvm_rv32_adapters::Rv32IsEqualModAdapterChip;

#[cfg(test)]
mod tests;

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
