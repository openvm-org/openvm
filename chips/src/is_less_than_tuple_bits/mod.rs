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
