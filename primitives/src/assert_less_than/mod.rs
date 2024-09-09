use std::sync::Arc;

use crate::var_range::{bus::VariableRangeCheckerBus, VariableRangeCheckerChip};

#[cfg(test)]
pub mod tests;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

pub use air::AssertLessThanAir;

/// This chip checks whether one number is less than another. The two numbers have a max number of bits,
/// given by max_bits. The chip assumes that the two numbers are within max_bits bits. The chip compares
/// the numbers by decomposing them into limbs of size decomp_bits, and interacts with a 
/// VariableRangeCheckerChip to range check the decompositions.
/// The number of auxilliary columns that this chip takes needs to be passed as a const generic.
/// This is because we want to have a static array storing the auxilliary columns:
///     AUX_LEN = (max_bits + decomp_bits - 1) / decomp_bits
#[derive(Clone, Debug)]
pub struct AssertLessThanChip<const AUX_LEN: usize> {
    pub air: AssertLessThanAir<AUX_LEN>,
    pub range_checker: Arc<VariableRangeCheckerChip>,
}

impl<const AUX_LEN: usize> AssertLessThanChip<AUX_LEN> {
    pub fn new(
        bus: VariableRangeCheckerBus,
        max_bits: usize,
        decomp_bits: usize,
        range_checker: Arc<VariableRangeCheckerChip>,
    ) -> Self {
        Self {
            air: AssertLessThanAir::new(bus, max_bits, decomp_bits),
            range_checker,
        }
    }
}
