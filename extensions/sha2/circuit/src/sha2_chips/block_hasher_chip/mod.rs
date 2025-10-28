mod air;
mod columns;
mod config;
mod trace;
mod utils;

pub use air::*;
pub use columns::*;
pub use config::*;
use openvm_circuit::arch::VmChipWrapper;
use openvm_circuit_primitives::{
    bitwise_op_lookup::SharedBitwiseOperationLookupChip, encoder::Encoder,
};
use openvm_instructions::riscv::RV32_CELL_BITS;
pub use trace::*;
pub use utils::*;

pub use super::config::*;

#[cfg(test)]
mod tests;

pub type Sha2BlockHasherChip<F> = VmChipWrapper<F, Sha2BlockHasherFiller>;

#[derive(derive_new::new)]
pub struct Sha2BlockHasherFiller {
    pub bitwise_lookup_chip: SharedBitwiseOperationLookupChip<RV32_CELL_BITS>,
    pub row_idx_encoder: Encoder,
    pub pointer_max_bits: usize,
}
