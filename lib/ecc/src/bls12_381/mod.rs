use axvm::moduli_setup;
use axvm_algebra::{Field, IntMod};

mod fp12;
mod fp2;
mod pairing;

pub use fp12::*;
pub use fp2::*;

use crate::pairing::PairingIntrinsics;

pub struct Bls12_381;

moduli_setup! {
    Fp = "0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab";
}

impl Field for Fp {
    type SelfRef<'a> = &'a Self;
    const ZERO: Self = <Self as IntMod>::ZERO;
    const ONE: Self = <Self as IntMod>::ONE;

    fn square_assign(&mut self) {
        IntMod::square_assign(self);
    }
}

impl PairingIntrinsics for Bls12_381 {
    type Fp = Fp;
    type Fp2 = Fp2;
    type Fp12 = Fp12;

    const PAIRING_IDX: usize = 1;
    const XI: Fp2 = Fp2::new(Fp::from_const_u8(1), Fp::from_const_u8(1));
}
