mod add_ne;
mod curves;
mod double;
mod mul;
mod trace;

pub use add_ne::*;
#[cfg(feature = "rvr")]
pub(crate) use curves::get_curve_type;
pub use curves::CurveType;
pub use double::*;
pub use mul::*;
pub(crate) use trace::{
    generate_add_ne_trace_from_postflight, generate_double_trace_from_postflight,
};
#[cfg(test)]
pub(crate) use trace::{
    generate_add_ne_trace_from_postflights, generate_double_trace_from_postflights,
};

#[cfg(test)]
mod tests;

use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_mod_circuit_builder::{FieldExpressionCoreAir, FieldExpressionFiller};
use openvm_riscv_adapters::{VecHeapAdapterAir, VecHeapAdapterFiller};

pub type WeierstrassAir<const NUM_READS: usize, const BLOCKS: usize> =
    VmAirWrapper<VecHeapAdapterAir<NUM_READS, BLOCKS, BLOCKS>, FieldExpressionCoreAir>;

pub type WeierstrassChip<F, const NUM_READS: usize, const BLOCKS: usize> =
    VmChipWrapper<F, FieldExpressionFiller<VecHeapAdapterFiller<NUM_READS, BLOCKS, BLOCKS>>>;
