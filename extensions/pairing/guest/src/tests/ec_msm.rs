use std::{ops::Mul, str::FromStr};

use ax_sdk::utils::create_seeded_rng;
use elliptic_curve::group::cofactor::CofactorCurveAffine;
use num_bigint::BigUint;
use rand::Rng;
use snark_verifier_sdk::snark_verifier::{
    halo2_base::{
        halo2_proofs::halo2curves::{
            msm::multiexp_serial,
            secp256k1::{Fp, Fq, Secp256k1Affine},
        },
        utils::ScalarField,
    },
    util::arithmetic::CurveAffine,
};

use crate::ec_msm::msm_axvm;

pub fn get_base() -> Secp256k1Affine {
    let base = (
        BigUint::from_str(
            "55066263022277343669578718895168534326250603453777594175500187360389116729240",
        )
        .unwrap(),
        BigUint::from_str(
            "32670510020758816978083085130507043184471273380659243275938904335757337482424",
        )
        .unwrap(),
    );
    Secp256k1Affine::from_xy(
        Fp::from_bytes_le(&base.0.to_bytes_le()),
        Fp::from_bytes_le(&base.1.to_bytes_le()),
    )
    .unwrap()
}

// pub fn base_line(g: Vec<Secp256k1Affine>, scalars: Vec<Fq>) -> Secp256k1Affine {
//     let mut res = Secp256k1Affine::identity();
//     for (g, s) in g.iter().zip(scalars.iter()) {
//         res = (res + g.mul(*s)).into();
//     }
//     res
// }

#[test]
pub fn msm_simple_test() {
    let base = get_base();
    let a = Fq::from(101);
    let b = Fq::from(102);
    let one = Fq::from(1);
    let base2 = base.mul(a).into();
    let g = vec![base, base2];
    let scalars = vec![b, one];
    let res = msm_axvm(g, scalars);
    assert_eq!(res, base.mul(a + b).into());
}

pub fn msm_rand_test(n: usize, disable_base_line: bool) {
    let base = get_base();
    let mut rng = create_seeded_rng();
    let base_muls = (0..n)
        .map(|_| Fq::from(rng.gen_range(0..100)))
        .collect::<Vec<_>>();
    let g = base_muls
        .iter()
        .map(|a| base.mul(*a).into())
        .collect::<Vec<_>>();
    let scalar_muls = (0..n)
        .map(|_| Fq::from(rng.gen_range(0..100)))
        .collect::<Vec<_>>();
    let time = std::time::Instant::now();
    let res = msm_axvm(g.clone(), scalar_muls.clone());
    println!("MSM Time Taken: {:?}", time.elapsed());
    let expected = if disable_base_line {
        let scalar = base_muls
            .iter()
            .zip(scalar_muls.iter())
            .map(|(a, b)| a.mul(b))
            .fold(Fq::from(0), |acc, x| acc + x);
        base.mul(scalar).into()
    } else {
        // let time = std::time::Instant::now();
        // let expected = base_line(g, scalar_muls);
        // println!("Base Line Time Taken: {:?}", time.elapsed());
        // expected
        let time = std::time::Instant::now();
        let mut acc = Secp256k1Affine::identity().into();
        multiexp_serial(&scalar_muls, &g, &mut acc);
        println!("Base Line Time Taken: {:?}", time.elapsed());
        acc.into()
    };
    assert_eq!(res, expected);
}

#[test]
pub fn msm_rand_test_10() {
    msm_rand_test(10, false);
}

#[test]
pub fn msm_rand_test_1000() {
    msm_rand_test(1000, false);
}

#[test]
pub fn msm_rand_test_10000() {
    msm_rand_test(10000, false);
}
