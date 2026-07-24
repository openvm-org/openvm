use std::sync::OnceLock;

use halo2curves_axiom::bn256::{Fq12, Fq2};
use mcl_rust::{CurveType, GT};
use openvm_ecc_guest::algebra::field::FieldExtension;

const FQ_BYTES: usize = 32;
const FP12_COEFFICIENTS: usize = 12;
const FP12_BYTES: usize = FQ_BYTES * FP12_COEFFICIENTS;

// OpenVM interleaves the two Fp6 coefficients. MCL stores each Fp6
// contiguously.
const MCL_TO_OPENVM_COEFFICIENT: [usize; FP12_COEFFICIENTS] =
    [0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    Initialization,
    Exponentiation,
}

// MCL stores the selected curve in process-global state. This static library
// uses MCL only for Ethereum's BN254 curve, so initialize and verify it once.
static MCL_INITIALIZED: OnceLock<Result<(), Error>> = OnceLock::new();

fn initialize() -> Result<(), Error> {
    *MCL_INITIALIZED.get_or_init(|| {
        (mcl_rust::init(CurveType::SNARK)
            && mcl_rust::get_fp_serialized_size() as usize == FQ_BYTES)
            .then_some(())
            .ok_or(Error::Initialization)
    })
}

pub fn pow_naf(value: &Fq12, digits: &[i8]) -> Result<Fq12, Error> {
    initialize()?;
    if digits.iter().any(|digit| !matches!(digit, -1..=1)) {
        return Err(Error::Exponentiation);
    }

    let mut base = GT::zero();
    if !base.deserialize(&to_mcl_bytes(value)) {
        return Err(Error::Exponentiation);
    }
    let mut result = GT::zero();
    result.set_int(1);
    let inverse = if digits.contains(&-1) {
        if base.is_zero() {
            return Err(Error::Exponentiation);
        }
        let mut inverse = GT::zero();
        // GT::inv uses the target-group shortcut. Division computes the
        // generic Fp12 inverse needed before final exponentiation.
        GT::div(&mut inverse, &result, &base);
        Some(inverse)
    } else {
        None
    };

    let mut square = GT::zero();
    for digit in digits.iter().rev() {
        GT::sqr(&mut square, &result);
        match digit {
            1 => square *= &base,
            -1 => square *= inverse.as_ref().unwrap(),
            _ => {}
        }
        std::mem::swap(&mut result, &mut square);
    }

    from_mcl_bytes(&result.serialize())
}

fn to_mcl_bytes(value: &Fq12) -> [u8; FP12_BYTES] {
    let openvm_bytes = value.to_bytes();
    let mut mcl_bytes = [0; FP12_BYTES];
    for (mcl_index, openvm_index) in MCL_TO_OPENVM_COEFFICIENT.into_iter().enumerate() {
        let mcl_offset = mcl_index * FQ_BYTES;
        let openvm_offset = openvm_index * FQ_BYTES;
        mcl_bytes[mcl_offset..mcl_offset + FQ_BYTES]
            .copy_from_slice(&openvm_bytes[openvm_offset..openvm_offset + FQ_BYTES]);
    }
    mcl_bytes
}

fn from_mcl_bytes(mcl_bytes: &[u8]) -> Result<Fq12, Error> {
    let mcl_bytes: &[u8; FP12_BYTES] = mcl_bytes.try_into().map_err(|_| Error::Exponentiation)?;
    let mut openvm_bytes = [0; FP12_BYTES];
    for (mcl_index, openvm_index) in MCL_TO_OPENVM_COEFFICIENT.into_iter().enumerate() {
        let mcl_offset = mcl_index * FQ_BYTES;
        let openvm_offset = openvm_index * FQ_BYTES;
        openvm_bytes[openvm_offset..openvm_offset + FQ_BYTES]
            .copy_from_slice(&mcl_bytes[mcl_offset..mcl_offset + FQ_BYTES]);
    }
    Ok(<Fq12 as FieldExtension<Fq2>>::from_bytes(&openvm_bytes))
}

#[cfg(test)]
mod tests {
    use halo2curves_axiom::{
        bn256::{Fq, Fq12, Fq2, Fr, G1Affine, G2Affine},
        ff::Field,
        group::Curve,
    };
    use openvm_ecc_guest::{
        algebra::{field::FieldExtension, ExpBytes},
        AffinePoint,
    };
    use openvm_pairing_guest::{
        halo2curves_shims::bn254::{
            final_exp_hint_naf_exponents, try_final_exp_hint_with_pow, Bn254, UNITY_ROOT_27,
        },
        pairing::{FinalExp, MultiMillerLoop},
    };
    use rand08::{rngs::StdRng, SeedableRng};

    use super::{pow_naf, Error};

    fn test_value() -> Fq12 {
        Fq12::from_coeffs(std::array::from_fn(|i| {
            Fq2::new(Fq::from((2 * i + 1) as u64), Fq::from((2 * i + 2) as u64))
        }))
    }

    #[test]
    fn matches_production_exponents() {
        let mut rng = StdRng::seed_from_u64(0x4d43_4c42_4e32_3534);
        let values = std::iter::once(test_value())
            .chain((0..16).map(|_| Fq12::random(&mut rng)))
            .collect::<Vec<_>>();
        for value in values {
            for digits in final_exp_hint_naf_exponents() {
                assert_eq!(pow_naf(&value, digits), Ok(value.exp_naf(true, digits)));
            }
        }
    }

    #[test]
    fn rejects_non_naf_digits() {
        assert_eq!(
            pow_naf(&test_value(), &[0, 1, 2, -1]),
            Err(Error::Exponentiation)
        );
    }

    #[test]
    fn rejects_zero_when_the_exponent_requires_an_inverse() {
        for digits in final_exp_hint_naf_exponents() {
            if digits.contains(&-1) {
                assert!(pow_naf(&Fq12::ZERO, digits).is_err());
            } else {
                assert_eq!(
                    pow_naf(&Fq12::ZERO, digits),
                    Ok(Fq12::ZERO.exp_naf(true, digits))
                );
            }
        }
        assert!(try_final_exp_hint_with_pow(&Fq12::ZERO, pow_naf).is_err());
    }

    #[test]
    fn matches_complete_pairing_hint() {
        let g1 = G1Affine::generator();
        let g2 = G2Affine::generator();
        let g1_scaled = (g1 * Fr::from(3)).to_affine();
        let g2_scaled = (g2 * Fr::from(5)).to_affine();
        let p = [
            AffinePoint::new(g1.x, g1.y),
            AffinePoint::new(g1_scaled.x, g1_scaled.y),
        ];
        let q = [
            AffinePoint::new(g2.x, g2.y),
            AffinePoint::new(g2_scaled.x, g2_scaled.y),
        ];
        let f = Bn254::multi_miller_loop(&p, &q);
        let expected = Bn254::final_exp_hint(&f);
        let actual = try_final_exp_hint_with_pow(&f, pow_naf).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn matches_all_root_selection_branches() {
        let unity_root = *UNITY_ROOT_27;
        let inputs = [
            Fq12::ONE,
            unity_root.invert().unwrap(),
            unity_root.square().invert().unwrap(),
        ];
        let expected_u = [Fq12::ONE, unity_root, unity_root.square()];
        for (input, expected_u) in inputs.into_iter().zip(expected_u) {
            let expected = Bn254::final_exp_hint(&input);
            assert_eq!(expected.1, expected_u);
            let actual = try_final_exp_hint_with_pow(&input, pow_naf).unwrap();
            assert_eq!(actual, expected);
        }
    }
}
