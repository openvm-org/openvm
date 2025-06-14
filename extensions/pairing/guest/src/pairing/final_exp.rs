use openvm_algebra_guest::{field::FieldExtension, Field};
use openvm_ecc_guest::AffinePoint;

// Currently this is only used by VM runtime execution and not implemented for guest programs.
#[allow(non_snake_case)]
pub trait FinalExp {
    type Fp: Field;
    type Fp2: Field + FieldExtension<Self::Fp>;
    type Fp12: Field + FieldExtension<Self::Fp2>;

    /// Assert in circuit that the final exponentiation is equal to one. The actual final
    /// exponentiation is calculated out of circuit via final_exp_hint. Scalar coefficients
    /// to the curve points must equal to zero, which is checked in a debug_assert.
    fn assert_final_exp_is_one(
        f: &Self::Fp12,
        P: &[AffinePoint<Self::Fp>],
        Q: &[AffinePoint<Self::Fp2>],
    );

    /// Generates a hint for the final exponentiation to be calculated out of circuit
    /// Input is the result of the Miller loop
    /// Output is c (residue witness inverse) and u (cubic nonresidue power)
    fn final_exp_hint(f: &Self::Fp12) -> (Self::Fp12, Self::Fp12);
}
