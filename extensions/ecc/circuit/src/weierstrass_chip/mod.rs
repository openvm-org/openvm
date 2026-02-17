mod add;
pub mod curves;
mod double;
mod preflight;

pub use add::*;
#[cfg(feature = "rvr")]
pub(crate) use curves::get_curve_type;
pub use curves::CurveType;
pub use double::*;

#[cfg(test)]
mod tests;


use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_mod_circuit_builder::{FieldExpressionCoreAir, FieldExpressionFiller};
use openvm_riscv_adapters::{Rv64VecHeapAdapterAir, Rv64VecHeapAdapterFiller};

pub type WeierstrassAir<const NUM_READS: usize, const BLOCKS: usize> =
    VmAirWrapper<Rv64VecHeapAdapterAir<NUM_READS, BLOCKS, BLOCKS>, FieldExpressionCoreAir>;

pub type WeierstrassChip<F, const NUM_READS: usize, const BLOCKS: usize> =
    VmChipWrapper<F, FieldExpressionFiller<Rv64VecHeapAdapterFiller<NUM_READS, BLOCKS, BLOCKS>>>;
