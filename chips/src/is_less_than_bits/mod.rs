use getset::{CopyGetters, Getters};

#[cfg(test)]
pub mod tests;

pub mod air;
pub mod columns;
pub mod trace;

#[derive(Default, Clone, CopyGetters)]
pub struct IsLessThanBitsAir {
    limb_bits: usize,
}

#[derive(Default, Getters)]
pub struct IsLessThanBitsChip {
    pub air: IsLessThanBitsAir,
}

impl IsLessThanBitsChip {
    pub fn new(limb_bits: usize) -> Self {
        let air = IsLessThanBitsAir { limb_bits };

        Self { air }
    }
}
