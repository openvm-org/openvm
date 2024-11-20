use axvm::moduli_setup;
use axvm_algebra::{DivUnsafe, Field, IntMod};

mod fp12;
mod fp2;
mod fp6;
pub mod pairing;

pub use fp12::*;
pub use fp2::*;
pub use fp6::*;

use crate::pairing::PairingIntrinsics;

#[cfg(all(test, feature = "halo2curves", not(target_os = "zkvm")))]
mod tests;

pub struct Bn254;

moduli_setup! {
    Bn254Fp = "21888242871839275222246405745257275088696311157297823662689037894645226208583";
}

pub type Fp = Bn254Fp;

impl Field for Fp {
    type SelfRef<'a> = &'a Self;
    const ZERO: Self = <Self as IntMod>::ZERO;
    const ONE: Self = <Self as IntMod>::ONE;

    fn double_assign(&mut self) {
        IntMod::double_assign(self);
    }

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

// Inverse z = x⁻¹ (mod p)
pub(crate) fn fp_invert_assign(x: &mut Fp) {
    let res = <Fp as Field>::ONE.div_unsafe_refs_impl(x);
    *x = res;
}
