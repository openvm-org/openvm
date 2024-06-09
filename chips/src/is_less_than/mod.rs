use std::sync::Arc;

use crate::range_gate::RangeCheckerGateChip;
use getset::{CopyGetters, Getters};

#[cfg(test)]
pub mod tests;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[derive(Default, Clone, CopyGetters)]
#[getset(get_copy = "pub")]
pub struct IsLessThanAir {
    // The bus index for sends to range chip
    bus_index: usize,
    // The maximum range for the range checker
    range_max: u32,
    // The maximum number of bits for the numbers to compare
    limb_bits: usize,
    // The number of bits to decompose each number into, for less than checking
    decomp: usize,
    // num_limbs is the number of limbs we decompose each input into, not including the last shifted limb
    #[getset(get_copy)]
    num_limbs: usize,
}

impl IsLessThanAir {
    pub fn new(bus_index: usize, range_max: u32, limb_bits: usize, decomp: usize) -> Self {
        Self {
            bus_index,
            range_max,
            limb_bits,
            decomp,
            num_limbs: (limb_bits + decomp - 1) / decomp,
        }
    }
}

/**
 * This chip checks whether one number is less than another. The two numbers have a max number of bits,
 * given by limb_bits. The chip assumes that the two numbers are within limb_bits bits. The chip compares
 * the numbers by decomposing them into limbs of size decomp bits, and interacts with a RangeCheckerGateChip
 * to range check the decompositions.
 */
#[derive(Default, Getters)]
pub struct IsLessThanChip {
    pub air: IsLessThanAir,

    pub range_checker: Arc<RangeCheckerGateChip>,
}

impl IsLessThanChip {
    pub fn new(
        bus_index: usize,
        range_max: u32,
        limb_bits: usize,
        decomp: usize,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> Self {
        let air = IsLessThanAir {
            bus_index,
            range_max,
            limb_bits,
            decomp,
            num_limbs: (limb_bits + decomp - 1) / decomp,
        };

        Self { air, range_checker }
    }
}
