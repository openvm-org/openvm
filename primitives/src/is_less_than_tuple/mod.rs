use std::sync::Arc;

use crate::var_range::VariableRangeCheckerChip;

#[cfg(test)]
pub mod tests;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

pub use air::IsLessThanTupleAir;

/// This chip computes whether one tuple is lexicographically less than another. Each element of the
/// tuple has its own max number of bits, given by the limb_bits array. The chip assumes that each limb
/// is within its given max limb_bits.
///
/// The IsLessThanTupleChip uses the IsLessThanChip as a subchip to check whether individual tuple elements
/// are less than each other.
#[derive(Clone, Debug)]
pub struct IsLessThanTupleChip {
    pub air: IsLessThanTupleAir,

    pub range_checker: Arc<VariableRangeCheckerChip>,
}

impl IsLessThanTupleChip {
    pub fn new(limb_bits: Vec<usize>, range_checker: Arc<VariableRangeCheckerChip>) -> Self {
        let air = IsLessThanTupleAir::new(range_checker.bus(), limb_bits);

        Self { air, range_checker }
    }
}
