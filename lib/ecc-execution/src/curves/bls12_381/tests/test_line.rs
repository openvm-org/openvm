use axvm_ecc::{field::FieldExtension, pairing::LineMType, point::EcPoint};
use halo2curves_axiom::{
    bls12_381::{Fq, Fq12, Fq2, G1Affine},
    ff::Field,
};
use rand::{rngs::StdRng, SeedableRng};

use crate::curves::bls12_381::{
    mul_023_by_023, mul_by_023, mul_by_02345, tangent_line_023, Bls12_381,
};

#[test]
fn test_mul_023_by_023() {
    // Generate random curve points
    let mut rng = StdRng::seed_from_u64(8);
    let rnd_pt_0 = G1Affine::random(&mut rng);
    let rnd_pt_1 = G1Affine::random(&mut rng);
    let ec_point_0 = EcPoint::<Fq> {
        x: rnd_pt_0.x,
        y: rnd_pt_0.y,
    };
    let ec_point_1 = EcPoint::<Fq> {
        x: rnd_pt_1.x,
        y: rnd_pt_1.y,
    };

    // Get lines evaluated at rnd_pt_0 and rnd_pt_1
    let line_0 = tangent_line_023::<Fq, Fq2>(ec_point_0);
    let line_1 = tangent_line_023::<Fq, Fq2>(ec_point_1);

    // Multiply the two line functions & convert to Fq12 to compare
    let mul_023_by_023 = mul_023_by_023::<Fq, Fq2>(line_0, line_1, Bls12_381::xi());
    let mul_023_by_023 = Fq12::from_coeffs(&mul_023_by_023);

    // Compare with the result of multiplying two Fp12 elements
    let fp12_0 = Fq12::from_evaluated_line_m_type(line_0);
    let fp12_1 = Fq12::from_evaluated_line_m_type(line_1);
    let check_mul_fp12 = fp12_0 * fp12_1;
    assert_eq!(mul_023_by_023, check_mul_fp12);
}

#[test]
fn test_mul_by_023() {
    let mut rng = StdRng::seed_from_u64(8);
    let f = Fq12::random(&mut rng);
    let rnd_pt = G1Affine::random(&mut rng);
    let ec_point = EcPoint::<Fq> {
        x: rnd_pt.x,
        y: rnd_pt.y,
    };
    let line = tangent_line_023::<Fq, Fq2>(ec_point);
    let mul_by_023 = mul_by_023::<Fq, Fq2, Fq12>(f, line);

    let check_mul_fp12 = Fq12::from_evaluated_line_m_type(line) * f;
    assert_eq!(mul_by_023, check_mul_fp12);
}

#[test]
fn test_mul_by_02345() {
    let mut rng = StdRng::seed_from_u64(8);
    let f = Fq12::random(&mut rng);
    let x = [
        Fq2::random(&mut rng),
        Fq2::random(&mut rng),
        Fq2::random(&mut rng),
        Fq2::random(&mut rng),
        Fq2::random(&mut rng),
        Fq2::random(&mut rng),
    ];
    let mul_by_02345 = mul_by_02345::<Fq, Fq2, Fq12>(f, x);

    let x_f12 = Fq12::from_coeffs(&x);
    assert_eq!(mul_by_02345, f * x_f12);
}
