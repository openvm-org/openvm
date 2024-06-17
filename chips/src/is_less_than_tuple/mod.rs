use std::sync::Arc;

use getset::CopyGetters;

use crate::{is_less_than::IsLessThanAir, range_gate::RangeCheckerGateChip};

#[cfg(test)]
pub mod tests;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

#[derive(Default, CopyGetters)]
#[getset(get_copy = "pub")]
pub struct IsLessThanTupleAir {
    /// The bus index for sends to range chip
    bus_index: usize,
    /// The number of bits to decompose each number into, for less than checking
    decomp: usize,
    /// IsLessThanAirs for each tuple element
    #[getset(skip)]
    is_less_than_airs: Vec<IsLessThanAir>,
}

impl IsLessThanTupleAir {
    pub fn new(bus_index: usize, limb_bits: Vec<usize>, decomp: usize) -> Self {
        let is_less_than_airs = limb_bits
            .iter()
            .map(|&limb_bit| IsLessThanAir::new(bus_index, limb_bit, decomp))
            .collect::<Vec<_>>();

        Self {
            bus_index,
            decomp,
            is_less_than_airs,
        }
    }

    pub fn tuple_len(&self) -> usize {
        self.is_less_than_airs.len()
    }

    pub fn limb_bits(&self) -> Vec<usize> {
        self.is_less_than_airs
            .iter()
            .map(|air| air.limb_bits())
            .collect()
    }
}

/// This chip computes whether one tuple is lexicographically less than another. Each element of the
/// tuple has its own max number of bits, given by the limb_bits array. The chip assumes that each limb
/// is within its given max limb_bits.
///
/// The IsLessThanTupleChip uses the IsLessThanChip as a subchip to check whether individual tuple elements
/// are less than each other.
#[derive(Default)]
pub struct IsLessThanTupleChip {
    pub air: IsLessThanTupleAir,

    pub range_checker: Arc<RangeCheckerGateChip>,
}

impl IsLessThanTupleChip {
    pub fn new(
        bus_index: usize,
        limb_bits: Vec<usize>,
        decomp: usize,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> Self {
        let is_less_than_airs = limb_bits
            .iter()
            .map(|&limb_bit| IsLessThanAir::new(bus_index, limb_bit, decomp))
            .collect::<Vec<_>>();

        let air = IsLessThanTupleAir {
            bus_index,
            decomp,
            is_less_than_airs,
        };

        Self { air, range_checker }
    }
}
