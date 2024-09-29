use std::str::FromStr;

use afs_compiler::{asm::AsmBuilder, util::execute_program};
use num_bigint_dig::BigUint;
use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use snark_verifier_sdk::snark_verifier::{
    halo2_base::{
        halo2_proofs::halo2curves::secp256k1::{self, Secp256k1Affine},
        utils::ScalarField,
    },
    util::arithmetic::CurveAffine,
};

use crate::{
    ec_fixed_scalar_multiply::{fixed_scalar_multiply_secp256k1, CachedPoints},
    ec_mul::scalar_multiply_secp256k1,
    types::ECPoint,
};

mod ecdsa;

const SECP256K1_COORD_BITS: usize = 256;

// Please note that these tests are for y^2 = x^3 - 7, which is easier. It has the same scalar field.

fn test_ec_mul(
    base: (BigUint, BigUint),
    scalar: BigUint,
    expected: (BigUint, BigUint),
    window_bits: usize,
) {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    let mut builder = AsmBuilder::<F, EF>::bigint_builder();

    let base = ECPoint {
        x: base.0,
        y: base.1,
    }
    .load_const(&mut builder, SECP256K1_COORD_BITS);
    let expected = ECPoint {
        x: expected.0,
        y: expected.1,
    }
    .load_const(&mut builder, SECP256K1_COORD_BITS);
    let s = builder.eval_biguint(scalar);

    let res = scalar_multiply_secp256k1(&mut builder, &base, s, window_bits);
    builder.assert_var_array_eq(&res.affine, &expected.affine);
    builder.halt();

    let program = builder.clone().compile_isa();
    execute_program(program, vec![]);
}

#[test]
fn test_compiler_ec_mul_simple() {
    test_ec_mul(
        (BigUint::from(2u64), BigUint::from(1u64)),
        BigUint::from(1u64),
        (BigUint::from(2u64), BigUint::from(1u64)),
        4,
    );
}

#[test]
fn test_compiler_ec_mul_double() {
    let secp256k1_coord = BigUint::from_str(
        "115792089237316195423570985008687907853269984665640564039457584007908834671663",
    )
    .unwrap();
    test_ec_mul(
        (BigUint::from(2u64), BigUint::from(1u64)),
        BigUint::from(2u64),
        (
            BigUint::from(32u64),
            secp256k1_coord - BigUint::from(181u64),
        ),
        4,
    );
}

#[test]
fn test_compiler_ec_mul_full_period() {
    let secp256k1_scalar = BigUint::from_str(
        "115792089237316195423570985008687907852837564279074904382605163141518161494337",
    )
    .unwrap();
    test_ec_mul(
        (
            BigUint::from_str(
                "55066263022277343669578718895168534326250603453777594175500187360389116729240",
            )
            .unwrap(),
            BigUint::from_str(
                "32670510020758816978083085130507043184471273380659243275938904335757337482424",
            )
            .unwrap(),
        ),
        secp256k1_scalar,
        (BigUint::from(0u64), BigUint::from(0u64)),
        4,
    );
}

#[test]
fn test_compiler_ec_mul_zero() {
    test_ec_mul(
        (BigUint::from(2u64), BigUint::from(1u64)),
        BigUint::from(0u64),
        (BigUint::from(0u64), BigUint::from(0u64)),
        4,
    );
}

fn test_fixed_ec_mul(
    base: (BigUint, BigUint),
    scalar: BigUint,
    expected: (BigUint, BigUint),
    window_bits: usize,
) {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    type Fp = secp256k1::Fp;
    let mut builder = AsmBuilder::<F, EF>::bigint_builder();
    let base = Secp256k1Affine::from_xy(
        Fp::from_bytes_le(&base.0.to_bytes_le()),
        Fp::from_bytes_le(&base.1.to_bytes_le()),
    )
    .unwrap();
    let expected = ECPoint {
        x: expected.0,
        y: expected.1,
    }
    .load_const(&mut builder, SECP256K1_COORD_BITS);

    let s = builder.eval_biguint(scalar);
    let cached_points = CachedPoints::new(&mut builder, base, window_bits, SECP256K1_COORD_BITS);
    let res = fixed_scalar_multiply_secp256k1(&mut builder, &cached_points, s);
    builder.assert_var_array_eq(&res.affine, &expected.affine);
    builder.halt();

    let program = builder.clone().compile_isa();
    execute_program(program, vec![]);
}

#[test]
fn test_compiler_fixed_ec_mul_simple() {
    test_fixed_ec_mul(
        (
            BigUint::from_str(
                "55066263022277343669578718895168534326250603453777594175500187360389116729240",
            )
            .unwrap(),
            BigUint::from_str(
                "32670510020758816978083085130507043184471273380659243275938904335757337482424",
            )
            .unwrap(),
        ),
        BigUint::from(1u64),
        (
            BigUint::from_str(
                "55066263022277343669578718895168534326250603453777594175500187360389116729240",
            )
            .unwrap(),
            BigUint::from_str(
                "32670510020758816978083085130507043184471273380659243275938904335757337482424",
            )
            .unwrap(),
        ),
        4,
    );
}

#[test]
fn test_compiler_fixed_ec_mul_double() {
    test_fixed_ec_mul(
        (
            BigUint::from_str(
                "55066263022277343669578718895168534326250603453777594175500187360389116729240",
            )
            .unwrap(),
            BigUint::from_str(
                "32670510020758816978083085130507043184471273380659243275938904335757337482424",
            )
            .unwrap(),
        ),
        BigUint::from(2u64),
        (
            BigUint::from_str(
                "89565891926547004231252920425935692360644145829622209833684329913297188986597",
            )
            .unwrap(),
            BigUint::from_str(
                "12158399299693830322967808612713398636155367887041628176798871954788371653930",
            )
            .unwrap(),
        ),
        4,
    );
}

#[test]
fn test_compiler_fixed_ec_mul_full_period() {
    let secp256k1_scalar = BigUint::from_str(
        "115792089237316195423570985008687907852837564279074904382605163141518161494337",
    )
    .unwrap();
    test_fixed_ec_mul(
        (
            BigUint::from_str(
                "55066263022277343669578718895168534326250603453777594175500187360389116729240",
            )
            .unwrap(),
            BigUint::from_str(
                "32670510020758816978083085130507043184471273380659243275938904335757337482424",
            )
            .unwrap(),
        ),
        secp256k1_scalar,
        (BigUint::from(0u64), BigUint::from(0u64)),
        4,
    );
}

#[test]
fn test_compiler_fixed_ec_mul_zero() {
    test_fixed_ec_mul(
        (
            BigUint::from_str(
                "55066263022277343669578718895168534326250603453777594175500187360389116729240",
            )
            .unwrap(),
            BigUint::from_str(
                "32670510020758816978083085130507043184471273380659243275938904335757337482424",
            )
            .unwrap(),
        ),
        BigUint::from(0u64),
        (BigUint::from(0u64), BigUint::from(0u64)),
        4,
    );
}
