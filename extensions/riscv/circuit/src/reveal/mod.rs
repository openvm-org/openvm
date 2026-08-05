use openvm_circuit::arch::{VmAirWrapper, VmChipWrapper};
use openvm_circuit_primitives::{
    bitwise_op_lookup::SharedBitwiseOperationLookupChip, var_range::SharedVariableRangeCheckerChip,
};
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::RevealOpcode;

use crate::{
    adapters::{DOUBLEWORD_ACCESS_WIDTH, BYTE_BITS},
    store::{core::StoreFiller, STORE_DOUBLEWORD_VALUE_CELLS},
};

mod adapter;
mod core;
mod execution;
pub(crate) mod trace;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(test)]
mod tests;

pub use core::*;

pub use adapter::*;
pub use execution::*;

#[derive(Clone)]
pub struct RevealFiller {
    pub(crate) inner:
        StoreFiller<RevealAdapterFiller, DOUBLEWORD_ACCESS_WIDTH, STORE_DOUBLEWORD_VALUE_CELLS>,
}

impl RevealFiller {
    pub fn new(
        pointer_max_bits: usize,
        range_checker: SharedVariableRangeCheckerChip,
        bitwise_lookup: SharedBitwiseOperationLookupChip<BYTE_BITS>,
    ) -> Self {
        Self {
            inner: StoreFiller::new_with_adapter(
                RevealAdapterFiller::new(pointer_max_bits, range_checker),
                RevealOpcode::CLASS_OFFSET,
                bitwise_lookup,
            ),
        }
    }
}

pub type RevealAir = VmAirWrapper<RevealAdapterAir, RevealCoreAir>;
pub type RevealChip<F> = VmChipWrapper<F, RevealFiller>;

const _: () = assert!(STORE_DOUBLEWORD_VALUE_CELLS == 4);
