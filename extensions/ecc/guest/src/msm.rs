use super::{weierstrass::ScalarMul, Group};

/// Multi-scalar multiplication, computed as one `EC_MUL` scalar product per base.
///
/// Each product routes through the curve's [`ScalarMul`] implementation, which discharges the
/// intrinsic's preconditions, so any scalar representation and identity points are handled. On a
/// curve with a cofactor, every base must lie in the prime-order subgroup.
pub fn msm<EcPoint, Scalar>(coeffs: &[Scalar], bases: &[EcPoint]) -> EcPoint
where
    EcPoint: Group + ScalarMul<Scalar>,
{
    assert_eq!(
        coeffs.len(),
        bases.len(),
        "msm requires matching scalar/base lengths"
    );

    let mut acc = <EcPoint as Group>::IDENTITY;
    for (coeff, base) in coeffs.iter().zip(bases.iter()) {
        acc += base.mul_scalar(coeff);
    }
    acc
}
