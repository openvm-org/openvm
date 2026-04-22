mod add_ne;
mod curves;
mod double;
mod preflight;

pub use add_ne::*;
pub use curves::CurveType;
pub use double::*;

#[cfg(test)]
mod tests;

use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_mod_circuit_builder::{FieldExpressionCoreAir, FieldExpressionFiller};
use openvm_riscv_adapters::{Rv64VecHeapAdapterAir, Rv64VecHeapAdapterFiller};

pub type WeierstrassAir<const NUM_READS: usize, const BLOCKS: usize, const BLOCK_SIZE: usize> =
    VmAirWrapper<
        Rv64VecHeapAdapterAir<NUM_READS, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        FieldExpressionCoreAir,
    >;

pub type WeierstrassChip<F, const NUM_READS: usize, const BLOCKS: usize, const BLOCK_SIZE: usize> =
    VmChipWrapper<
        F,
        FieldExpressionFiller<
            Rv64VecHeapAdapterFiller<NUM_READS, BLOCKS, BLOCKS, BLOCK_SIZE, BLOCK_SIZE>,
        >,
    >;
