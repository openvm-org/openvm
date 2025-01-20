mod core;

pub use core::*;

use openvm_circuit::arch::VmChipWrapper;
use openvm_rv32im_transpiler::Rv32LoadStoreOpcode;

use super::adapters::{Rv32LoadStoreAdapterChip, RV32_REGISTER_NUM_LIMBS};

#[cfg(test)]
mod tests;

pub type Rv32LoadStoreChip<F> = VmChipWrapper<
    F,
    Rv32LoadStoreAdapterChip<F>,
    LoadStoreCoreChip<Rv32LoadStoreOpcode, RV32_REGISTER_NUM_LIMBS>,
>;
