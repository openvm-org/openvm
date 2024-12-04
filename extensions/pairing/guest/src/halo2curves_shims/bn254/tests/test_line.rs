use axvm_ecc_guest::{algebra::field::FieldExtension, AffinePoint};
use halo2curves_axiom::{
    bn256::{Fq, Fq12, Fq2, G1Affine},
    ff::Field,
};
use rand::{rngs::StdRng, SeedableRng};

use crate::{
    halo2curves_shims::bn254::{tangent_line_013, Bn254},
    pairing::{FromLineDType, LineMulDType},
};

#[test]
fn test_mul_013_by_013() {
    // Generate random curve points
    let mut rng = StdRng::seed_from_u64(8);
    let rnd_pt_0 = G1Affine::random(&mut rng);
    let rnd_pt_1 = G1Affine::random(&mut rng);
    let ec_point_0 = AffinePoint::<Fq> {
        x: rnd_pt_0.x,
        y: rnd_pt_0.y,
    };
    let ec_point_1 = AffinePoint::<Fq> {
        x: rnd_pt_1.x,
        y: rnd_pt_1.y,
    };

    // Get lines evaluated at rnd_pt_0 and rnd_pt_1
    let line_0 = tangent_line_013::<Fq, Fq2>(ec_point_0);
    let line_1 = tangent_line_013::<Fq, Fq2>(ec_point_1);

    // Multiply the two line functions & convert to Fq12 to compare
    let mul_013_by_013 = Bn254::mul_013_by_013(&line_0, &line_1);
    let mul_013_by_013 = Fq12::from_coeffs([
        mul_013_by_013[0],
        mul_013_by_013[1],
        mul_013_by_013[2],
        mul_013_by_013[3],
        mul_013_by_013[4],
        Fq2::ZERO,
    ]);

    // Compare with the result of multiplying two Fp12 elements
    let fp12_0 = Fq12::from_evaluated_line_d_type(line_0);
    let fp12_1 = Fq12::from_evaluated_line_d_type(line_1);
    let check_mul_fp12 = fp12_0 * fp12_1;
    assert_eq!(mul_013_by_013, check_mul_fp12);
}

#[test]
fn test_mul_by_013() {
    let mut rng = StdRng::seed_from_u64(8);
    let f = Fq12::random(&mut rng);
    let rnd_pt = G1Affine::random(&mut rng);
    let ec_point = AffinePoint::<Fq> {
        x: rnd_pt.x,
        y: rnd_pt.y,
    };
    let line = tangent_line_013::<Fq, Fq2>(ec_point);
    let mul_by_013 = Bn254::mul_by_013(&f, &line);

    let check_mul_fp12 = Fq12::from_evaluated_line_d_type(line) * f;
    assert_eq!(mul_by_013, check_mul_fp12);
}

#[test]
fn test_mul_by_01234() {
    let mut rng = StdRng::seed_from_u64(8);
    let f = Fq12::random(&mut rng);
    let x = [
        Fq2::random(&mut rng),
        Fq2::random(&mut rng),
        Fq2::random(&mut rng),
        Fq2::random(&mut rng),
        Fq2::random(&mut rng),
    ];
    let mul_by_01234 = Bn254::mul_by_01234(&f, &x);

    let x_f12 = Fq12::from_coeffs([x[0], x[1], x[2], x[3], x[4], Fq2::ZERO]);
    assert_eq!(mul_by_01234, f * x_f12);
}
