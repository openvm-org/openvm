use std::sync::Arc;

use crate::var_range::VariableRangeCheckerChip;

#[cfg(test)]
pub mod tests;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

pub use air::IsLessThanAir;

/// This chip checks whether one number is less than another. The two numbers have a max number of bits,
/// given by limb_bits. The chip assumes that the two numbers are within limb_bits bits. The chip compares
/// the numbers by decomposing them into limbs of size decomp bits, and interacts with a VariableRangeCheckerChip
/// to range check the decompositions.
#[derive(Clone, Debug)]
pub struct IsLessThanChip {
    pub air: IsLessThanAir,
    pub range_checker: Arc<VariableRangeCheckerChip>,
}

impl IsLessThanChip {
    pub fn new(max_bits: usize, range_checker: Arc<VariableRangeCheckerChip>) -> Self {
        Self {
            air: IsLessThanAir::new(range_checker.bus(), max_bits),
            range_checker,
        }
    }
}
