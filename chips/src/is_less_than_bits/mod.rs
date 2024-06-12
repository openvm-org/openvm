use getset::CopyGetters;

#[cfg(test)]
pub mod tests;

pub mod air;
pub mod columns;
pub mod trace;

#[derive(Default, Clone, CopyGetters)]
pub struct IsLessThanBitsAir {
    limb_bits: usize,
}
