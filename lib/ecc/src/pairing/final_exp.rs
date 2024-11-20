use crate::affine_point::AffinePoint;

#[allow(non_snake_case)]
pub trait FinalExp<Fp, Fp2, Fp12> {
    /// Assert in circuit that the final exponentiation is equal to one. The actual final
    /// exponentiaton is calculated out of circuit via final_exp_hint. Scalar coefficients
    /// to the curve points must equal to zero, which is checked in a debug_assert.
    fn assert_final_exp_is_one(f: &Fp12, P: &[AffinePoint<Fp>], Q: &[AffinePoint<Fp2>]);

    /// Generates a hint for the final exponentiation to be calculated out of circuit
    /// Input is the result of the Miller loop
    /// Output is c (residue witness inverse) and u (cubic nonresidue power)
    fn final_exp_hint(f: &Fp12) -> (Fp12, Fp12);
}
