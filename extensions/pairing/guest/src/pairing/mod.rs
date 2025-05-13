mod final_exp;
mod line;
mod miller_loop;
mod miller_step;

pub use final_exp::*;
pub use line::*;
pub use miller_loop::*;
pub use miller_step::*;
use openvm_algebra_guest::{
    field::{ComplexConjugate, FieldExtension},
    ExpBytes, Field, IntMod,
};
use openvm_ecc_guest::AffinePoint;

use crate::PairingBaseFunct7;

pub trait PairingIntrinsics {
    type Fp: Field + IntMod;
    type Fp2: Field + FieldExtension<Self::Fp> + ComplexConjugate;
    type Fp12: FieldExtension<Self::Fp2> + ComplexConjugate;

    /// Index for custom intrinsic opcode determination.
    const PAIRING_IDX: usize;
    /// The sextic extension `Fp12` is `Fp2[X] / (X^6 - \xi)`, where `\xi` is a non-residue.
    const XI: Self::Fp2;
    /// Multiplication constants for the Frobenius map for coefficients in Fp2 c1..=c5 for powers
    /// 0..12 FROBENIUS_COEFFS\[i\]\[j\] = \xi^{(j + 1) * (p^i - 1)/6} when p = 1 (mod 6)
    const FROBENIUS_COEFFS: [[Self::Fp2; 5]; 12];

    const FP2_TWO: Self::Fp2;
    const FP2_THREE: Self::Fp2;
}

#[allow(non_snake_case)]
pub trait PairingCheck {
    type Fp: Field;
    type Fp2: Field + FieldExtension<Self::Fp> + ComplexConjugate;
    type Fp12: FieldExtension<Self::Fp2> + ComplexConjugate;

    /// Given points P[], Q[], computes the multi-Miller loop and then returns
    /// the final exponentiation hint from Novakovic-Eagon <https://eprint.iacr.org/2024/640.pdf>.
    ///
    /// Output is c (residue witness inverse) and u (cubic nonresidue power).
    fn pairing_check_hint(
        P: &[AffinePoint<Self::Fp>],
        Q: &[AffinePoint<Self::Fp2>],
    ) -> (Self::Fp12, Self::Fp12);

    fn pairing_check(
        P: &[AffinePoint<Self::Fp>],
        Q: &[AffinePoint<Self::Fp2>],
    ) -> Result<(), PairingCheckError>;
}

// Square and multiply implementation of final exponentiation. Used if the hint fails to prove
// the pairing check.
// `exp` should be big-endian.
pub fn exp_check_fallback<F: Field + ExpBytes>(f: &F, exp: &[u8]) -> Result<(), PairingCheckError>
where
    for<'a> &'a F: core::ops::Mul<&'a F, Output = F>,
{
    if f.exp_bytes(true, exp) == F::ONE {
        Ok(())
    } else {
        Err(PairingCheckError)
    }
}

pub const fn shifted_funct7<P: PairingIntrinsics>(funct7: PairingBaseFunct7) -> usize {
    P::PAIRING_IDX * (PairingBaseFunct7::PAIRING_MAX_KINDS as usize) + funct7 as usize
}

#[derive(Debug, Clone, PartialEq)]
pub struct PairingCheckError;

impl core::error::Error for PairingCheckError {}
impl core::fmt::Display for PairingCheckError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Pairing check failed")
    }
}

#[cfg(all(test, not(target_os = "zkvm")))]
mod tests {
    use num_bigint::BigUint;
    use openvm_algebra_moduli_macros::{moduli_declare, moduli_init};

    use super::*;

    moduli_declare! {
        F13 { modulus = "13" },
    }

    moduli_init! {
        "13",
    }

    #[test]
    fn test_pairing_check_fallback() {
        let a = F13::from_u8(2);
        let b = BigUint::from(12u32);
        let result = exp_check_fallback(&a, &b.to_bytes_be());
        assert_eq!(result, Ok(()));

        let b = BigUint::from(11u32);
        let result = exp_check_fallback(&a, &b.to_bytes_be());
        assert_eq!(result, Err(PairingCheckError));
    }
}
