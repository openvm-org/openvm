use std::convert::Into;

use ax_ecc_primitives::field_expression::FieldVariableConfig;

mod addsub;
pub use addsub::*;
mod muldiv;
pub use muldiv::*;

use crate::{
    arch::{VmAdapterInterface, VmAirWrapper, VmChipWrapper},
    rv32im::adapters::{
        Rv32VecHeapAdapterAir, Rv32VecHeapAdapterChip, Rv32VecHeapAdapterInterface,
    },
};

#[cfg(test)]
mod tests;

pub const FIELD_ELEMENT_BITS: usize = 30;
const LIMB_BITS: usize = 8;

#[derive(Clone)]
pub struct ModularConfig<const NUM_LIMBS: usize> {}
impl<const NUM_LIMBS: usize> FieldVariableConfig for ModularConfig<NUM_LIMBS> {
    fn canonical_limb_bits() -> usize {
        LIMB_BITS
    }
    fn max_limb_bits() -> usize {
        FIELD_ELEMENT_BITS - 1
    }
    fn num_limbs_per_field_element() -> usize {
        NUM_LIMBS
    }
}

pub type ModularAddSubV2Air<const NUM_LIMBS: usize, const LIMB_BITS: usize> = VmAirWrapper<
    Rv32VecHeapAdapterAir<1, 1, NUM_LIMBS, NUM_LIMBS>,
    ModularAddSubV2CoreAir<NUM_LIMBS, LIMB_BITS>,
>;
pub type ModularAddSubV2Chip<F, const NUM_LIMBS: usize, const LIMB_BITS: usize> = VmChipWrapper<
    F,
    Rv32VecHeapAdapterChip<F, 1, 1, NUM_LIMBS, NUM_LIMBS>,
    ModularAddSubV2CoreChip<NUM_LIMBS, LIMB_BITS>,
>;
pub type ModularMulDivV2Air<const NUM_LIMBS: usize, const LIMB_BITS: usize> = VmAirWrapper<
    Rv32VecHeapAdapterAir<1, 1, NUM_LIMBS, NUM_LIMBS>,
    ModularMulDivV2CoreAir<NUM_LIMBS, LIMB_BITS>,
>;
pub type ModularMulDivV2Chip<F, const NUM_LIMBS: usize, const LIMB_BITS: usize> = VmChipWrapper<
    F,
    Rv32VecHeapAdapterChip<F, 1, 1, NUM_LIMBS, NUM_LIMBS>,
    ModularMulDivV2CoreChip<NUM_LIMBS, LIMB_BITS>,
>;
