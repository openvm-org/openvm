use getset::CopyGetters;

#[cfg(test)]
pub mod tests;

pub mod air;
pub mod columns;
pub mod trace;

#[derive(Default, Clone, CopyGetters)]
#[getset(get_copy = "pub")]
pub struct IsLessThanBitsAir {
    limb_bits: usize,
}

impl IsLessThanBitsAir {
    pub fn new(limb_bits: usize) -> Self {
        Self { limb_bits }
    }
}
