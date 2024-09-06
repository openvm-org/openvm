use std::sync::Arc;

use crate::{range::bus::RangeCheckBus, range_gate::RangeCheckerGateChip};

#[cfg(test)]
pub mod tests;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

pub use air::AssertLessThanAir;

/// This chip checks whether one number is less than another. The two numbers have a max number of bits,
/// given by limb_bits. The chip assumes that the two numbers are within limb_bits bits. The chip compares
/// the numbers by decomposing them into limbs of size decomp bits, and interacts with a RangeCheckerGateChip
/// to range check the decompositions.
/// 
/// The number of auxilliary columns that this chip takes needs to be passed as a const generic.
/// This is because we want to have a static array storing the auxilliary columns
#[derive(Clone, Debug)]
pub struct AssertLessThanChip<const AUX_LEN: usize> {
    pub air: AssertLessThanAir<AUX_LEN>,
    pub range_checker: Arc<RangeCheckerGateChip>,
}

impl<const AUX_LEN: usize> AssertLessThanChip<AUX_LEN> {
    pub fn new(
        bus: RangeCheckBus,
        limb_bits: usize,
        decomp_bits: usize,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> Self {
        Self {
            air: AssertLessThanAir::new(bus, limb_bits, decomp_bits),
            range_checker,
        }
    }
}
