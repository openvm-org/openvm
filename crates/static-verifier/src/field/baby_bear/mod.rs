use openvm_stark_sdk::config::baby_bear_bn254_poseidon2::D_EF;

mod base;
mod extension;

pub use base::*;
pub use extension::*;

pub(crate) const BABY_BEAR_MODULUS_U64: u64 = 0x78000001; // BabyBear prime: 2013265921
pub(crate) const BABY_BEAR_EXT_DEGREE: usize = D_EF;
pub(crate) const BABY_BEAR_BITS: usize = BABYBEAR_MAX_BITS;

pub type BabyBearExtChip = BabyBearExt5Chip;
pub type BabyBearExtWire = BabyBearExt5Wire;

#[cfg(test)]
mod tests;
