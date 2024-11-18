use axvm::moduli_setup;
use axvm_algebra::{DivUnsafe, IntMod};

use crate::field::Field;

mod fp2;
pub use fp2::*;

mod fp12;
pub use fp12::*;

mod miller;
pub use miller::*;

use super::{LineMulMType, MillerStep};

#[cfg(feature = "halo2curves")]
#[cfg(test)]
mod tests;

pub struct Bls12381Intrinsic;

moduli_setup! {
    Bls12381Fp = "0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab";
}

impl Field for Bls12381Fp {
    type SelfRef<'a> = &'a Self;

    fn zero() -> Self {
        <Self as IntMod>::ZERO
    }

    fn one() -> Self {
        <Self as IntMod>::ONE
    }

    fn invert(&self) -> Option<Self> {
        Some(<Bls12381Fp as IntMod>::ONE.div_unsafe(self))
    }
}

impl LineMulMType<Bls12381Fp, Bls12381Fp2, Bls12381Fp12> for Bls12381Intrinsic {}
