use ax_ecc_primitives::field_expression::FieldVariableConfig;

#[cfg(test)]
mod tests;

mod addsub;
pub use addsub::*;
mod muldiv;
pub use muldiv::*;

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
