mod base;
mod extension;

pub use base::*;
pub use extension::*;
use halo2_base::{halo2_proofs::halo2curves::bn256::Fr, AssignedValue};

pub(crate) const BABY_BEAR_MODULUS_U64: u64 = 0x78000001; // BabyBear prime: 2013265921
pub(crate) const BABY_BEAR_EXT_DEGREE: usize = 4;
pub(crate) const BABY_BEAR_BITS: usize = BABYBEAR_MAX_BITS;

pub type BabyBearExtChip = BabyBearExt4Chip;
pub type BabyBearExtWire<F = AssignedValue<Fr>> = BabyBearExt4Wire<F>;
pub type ReducedBabyBearExtWire<F = AssignedValue<Fr>> = ReducedBabyBearExt4Wire<F>;

#[cfg(test)]
mod tests;
