//! Checks that the native ladder and the field expression agree.
//!
//! The executor writes the native result to memory while trace generation records the field
//! expression's, so the two must produce identical bytes rather than merely both being correct.

use halo2curves_axiom::ff::PrimeField;
use num_bigint::BigUint;
use num_traits::Zero;
use openvm_mod_circuit_builder::{ExprBuilderConfig, FieldExpressionProgram};

use super::{
    ec_mul_step_program, EC_MUL_SCALAR_BITS, FLAG_DBL, FLAG_DBL_ADD, FLAG_INF_STAY, FLAG_INF_TAKE,
    NUM_STEP_FLAGS, SCALAR_LIMBS,
};
use crate::weierstrass_chip::curves::ec_mul_impl;

const LIMB_BITS: usize = 8;
const RANGE_MAX_BITS: usize = 17;

fn to_biguint<F: PrimeField<Repr = [u8; 32]>>(f: &F) -> BigUint {
    BigUint::from_bytes_le(f.to_repr().as_ref())
}

/// Drives the field expression through a full ladder, as the trace filler does.
fn expr_ladder(
    program: &FieldExpressionProgram,
    px: &BigUint,
    py: &BigUint,
    scalar_le: &[u8; SCALAR_LIMBS],
) -> (BigUint, BigUint) {
    let mut rx = BigUint::zero();
    let mut ry = BigUint::zero();
    let mut is_inf = true;
    let outs = program.output_indices();

    for i in (0..EC_MUL_SCALAR_BITS).rev() {
        let bit = (scalar_le[i / 8] >> (i % 8)) & 1 == 1;
        let mut flags = [false; NUM_STEP_FLAGS];
        flags[match (is_inf, bit) {
            (false, false) => FLAG_DBL,
            (false, true) => FLAG_DBL_ADD,
            (true, false) => FLAG_INF_STAY,
            (true, true) => FLAG_INF_TAKE,
        }] = true;

        let vars = program.execute(&[rx, ry, px.clone(), py.clone()], &flags);
        rx = vars[outs[0]].clone();
        ry = vars[outs[1]].clone();
        is_inf = is_inf && !bit;
    }
    (rx, ry)
}

fn check_curve<F: PrimeField<Repr = [u8; 32]> + From<u64>, const NEG_A: u64>(
    name: &str,
    gx: F,
    gy: F,
    a: BigUint,
) {
    let modulus = BigUint::parse_bytes(F::MODULUS.trim_start_matches("0x").as_bytes(), 16).unwrap();
    let program = ec_mul_step_program(
        ExprBuilderConfig {
            modulus,
            num_limbs: SCALAR_LIMBS,
            limb_bits: LIMB_BITS,
        },
        RANGE_MAX_BITS,
        a,
    );

    // Covers the identity, the transition out of it, a long leading-zero run, and a full-width
    // scalar exercising all four cases.
    let scalars: [[u8; SCALAR_LIMBS]; 5] = [
        [0u8; SCALAR_LIMBS],
        {
            let mut s = [0u8; SCALAR_LIMBS];
            s[0] = 1;
            s
        },
        {
            let mut s = [0u8; SCALAR_LIMBS];
            s[0] = 3;
            s
        },
        {
            let mut s = [0u8; SCALAR_LIMBS];
            s[0] = 0xff;
            s[1] = 0xff;
            s[2] = 0xff;
            s
        },
        {
            let mut s = [0x5au8; SCALAR_LIMBS];
            // below any supported curve order
            s[SCALAR_LIMBS - 1] = 0x3c;
            s
        },
    ];

    for scalar in scalars {
        let (nx, ny) = ec_mul_impl::<F, NEG_A>(gx, gy, &scalar, EC_MUL_SCALAR_BITS);
        let (ex, ey) = expr_ladder(&program, &to_biguint(&gx), &to_biguint(&gy), &scalar);

        assert_eq!(
            (to_biguint(&nx), to_biguint(&ny)),
            (ex, ey),
            "{name}: native ec_mul_impl and the field expression disagree for scalar {scalar:02x?}"
        );
    }
}

#[test]
fn native_ladder_matches_field_expression() {
    use halo2curves_axiom::{secp256r1, secq256k1};

    // secp256k1 generator (a = 0). `secq256k1::Fq` is secp256k1's coordinate field.
    check_curve::<secq256k1::Fq, 0>(
        "secp256k1",
        secq256k1::Fq::from_repr(hex_literal::hex!(
            "9817f8165b81f259d928ce2ddbfc9b02070b87ce9562a055acbbdcf97e66be79"
        ))
        .unwrap(),
        secq256k1::Fq::from_repr(hex_literal::hex!(
            "b8d410fb8fd0479c195485a648b417fda808110efcfba45d65c4a32677da3a48"
        ))
        .unwrap(),
        BigUint::zero(),
    );

    // P-256 generator (a = p - 3).
    let p256_modulus = BigUint::parse_bytes(
        secp256r1::Fp::MODULUS.trim_start_matches("0x").as_bytes(),
        16,
    )
    .unwrap();
    check_curve::<secp256r1::Fp, 3>(
        "secp256r1",
        secp256r1::Fp::from_repr(hex_literal::hex!(
            "96c298d84539a1f4a033eb2d817d0377f240a463e5e6bcf847422ce1f2d1176b"
        ))
        .unwrap(),
        secp256r1::Fp::from_repr(hex_literal::hex!(
            "f551bf376840b6cbce5e316b5733ce2b169e0f7c4aebe78e9b7f1afee242e34f"
        ))
        .unwrap(),
        p256_modulus - BigUint::from(3u32),
    );
}
