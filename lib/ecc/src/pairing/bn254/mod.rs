use axvm::moduli_setup;
use axvm_algebra::{DivUnsafe, IntMod};

use crate::field::Field;

mod fp2;
pub use fp2::*;

mod fp12;
pub use fp12::*;

mod miller;
pub use miller::*;

use super::{LineMulDType, MillerStep, MultiMillerLoop};

#[cfg(feature = "halo2curves")]
#[cfg(test)]
mod tests;

#[allow(non_snake_case)]
pub struct Bn254Intrinsic {
    pub FROBENIUS_COEFF_FQ6_C1: [Bn254Fp2; 3],
    pub XI_TO_Q_MINUS_1_OVER_2: Bn254Fp2,
}

moduli_setup! {
    Bn254Fp = "21888242871839275222246405745257275088696311157297823662689037894645226208583";
}

impl Field for Bn254Fp {
    type SelfRef<'a> = &'a Self;

    fn zero() -> Self {
        <Self as IntMod>::ZERO
    }

    fn one() -> Self {
        <Self as IntMod>::ONE
    }

    fn invert(&self) -> Option<Self> {
        Some(<Bn254Fp as IntMod>::ONE.div_unsafe(self))
    }
}

impl LineMulDType<Bn254Fp, Bn254Fp2, Bn254Fp12> for Bn254Intrinsic {}

impl Bn254Intrinsic {
    pub const FROBENIUS_COEFF_FQ6_C1: [Bn254Fp2; 3] = [
        Bn254Fp2 {
            c0: Bn254Fp([
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0,
            ]),
            c1: Bn254Fp([
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0,
            ]),
        },
        Bn254Fp2 {
            c0: Bn254Fp([
                61, 85, 111, 23, 87, 149, 227, 153, 12, 51, 195, 194, 16, 195, 140, 183, 67, 177,
                89, 245, 60, 236, 11, 76, 247, 17, 121, 79, 152, 71, 179, 47,
            ]),
            c1: Bn254Fp([
                162, 203, 15, 100, 28, 213, 101, 22, 206, 157, 124, 11, 29, 42, 174, 50, 148, 7,
                90, 215, 139, 204, 164, 75, 32, 174, 235, 97, 80, 229, 201, 22,
            ]),
        },
        Bn254Fp2 {
            c0: Bn254Fp([
                72, 253, 124, 96, 229, 68, 189, 228, 61, 110, 150, 187, 159, 6, 143, 194, 176, 204,
                172, 224, 231, 217, 109, 94, 41, 160, 49, 225, 114, 78, 100, 48,
            ]),
            c1: Bn254Fp([
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0,
            ]),
        },
    ];

    pub const XI_TO_Q_MINUS_1_OVER_2: Bn254Fp2 = Bn254Fp2 {
        c0: Bn254Fp([
            90, 19, 160, 113, 70, 1, 84, 220, 152, 89, 201, 169, 237, 224, 170, 219, 185, 249, 226,
            182, 152, 198, 94, 220, 220, 245, 154, 72, 5, 243, 60, 6,
        ]),
        c1: Bn254Fp([
            227, 176, 35, 38, 99, 127, 211, 130, 210, 91, 162, 143, 201, 125, 128, 33, 43, 111,
            121, 236, 167, 181, 4, 7, 154, 4, 65, 172, 188, 60, 192, 7,
        ]),
    };
}
