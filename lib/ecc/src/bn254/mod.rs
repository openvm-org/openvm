use axvm::moduli_setup;
use axvm_algebra::{Field, IntMod};

mod fp12;
mod fp2;
pub mod pairing;

pub use fp12::*;
pub use fp2::*;

use crate::pairing::PairingIntrinsics;

pub struct Bn254;

moduli_setup! {
    Bn254Fp = "21888242871839275222246405745257275088696311157297823662689037894645226208583";
}

pub type Fp = Bn254Fp;

impl Field for Fp {
    type SelfRef<'a> = &'a Self;
    const ZERO: Self = <Self as IntMod>::ZERO;
    const ONE: Self = <Self as IntMod>::ONE;

    fn square_assign(&mut self) {
        IntMod::square_assign(self);
    }
}

impl PairingIntrinsics for Bn254 {
    type Fp = Fp;
    type Fp2 = Fp2;
    type Fp12 = Fp12;

    const PAIRING_IDX: usize = 0;
    const XI: Fp2 = Fp2::new(Fp::from_const_u8(9), Fp::from_const_u8(1));
}
