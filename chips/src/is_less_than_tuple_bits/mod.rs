use getset::CopyGetters;

use crate::is_less_than_bits::IsLessThanBitsAir;

#[cfg(test)]
pub mod tests;

pub mod air;
pub mod columns;
pub mod trace;

#[derive(Default, CopyGetters)]
#[getset(get_copy = "pub")]
pub struct IsLessThanTupleBitsAir {
    // IsLessThanAirs for each tuple element
    #[getset(skip)]
    is_less_than_bits_airs: Vec<IsLessThanBitsAir>,
}

impl IsLessThanTupleBitsAir {
    pub fn new(limb_bits: Vec<usize>) -> Self {
        let is_less_than_bits_airs = limb_bits
            .iter()
            .map(|&limb_bit| IsLessThanBitsAir::new(limb_bit))
            .collect::<Vec<_>>();

        Self {
            is_less_than_bits_airs,
        }
    }

    pub fn tuple_len(&self) -> usize {
        self.is_less_than_bits_airs.len()
    }

    pub fn limb_bits(&self) -> Vec<usize> {
        self.is_less_than_bits_airs
            .iter()
            .map(|air| air.limb_bits())
            .collect()
    }
}

/**
 * This chip computes whether one tuple is lexicographically less than another. Each element of the
 * tuple has its own max number of bits, given by the limb_bits array. The chip assumes that each limb
 * is within its given max limb_bits.
 *
 * The IsLessThanTupleChip uses the IsLessThanChip as a subchip to check whether individual tuple elements
 * are less than each other.
 */
#[derive(Default)]
pub struct IsLessThanTupleBitsChip {
    pub air: IsLessThanTupleBitsAir,
}

impl IsLessThanTupleBitsChip {
    pub fn new(limb_bits: Vec<usize>) -> Self {
        let is_less_than_bits_airs = limb_bits
            .iter()
            .map(|&limb_bit| IsLessThanBitsAir::new(limb_bit))
            .collect::<Vec<_>>();

        let air = IsLessThanTupleBitsAir {
            is_less_than_bits_airs,
        };

        Self { air }
    }
}
