//! Checks that the native ladder and the field expression agree.
//!
//! The executor writes the native result to memory while trace generation records the field
//! expression's, so the two must produce identical bytes rather than merely both being correct.

use halo2curves_axiom::ff::PrimeField;
use num_bigint::BigUint;
use num_traits::Zero;
use openvm_mod_circuit_builder::{ExprBuilderConfig, FieldExpressionProgram};

use super::{
    ec_mul_step_program, execution::sign_pattern_for_row, EC_MUL_COMPUTE_ROWS, EC_MUL_SCALAR_BITS,
    EC_MUL_SIGN_PATTERNS, SCALAR_LIMBS,
};
use crate::weierstrass_chip::curves::{ec_add_ne_impl, ec_double_impl, ec_mul_impl};

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
    // The most significant digit is `+1`, so the accumulator seeds itself from `P`.
    let mut rx = px.clone();
    let mut ry = py.clone();
    let outs = program.output_indices();

    for row in 0..EC_MUL_COMPUTE_ROWS {
        let mut flags = [false; EC_MUL_SIGN_PATTERNS];
        flags[sign_pattern_for_row(scalar_le, row)] = true;

        let vars = program.execute(&[px.clone(), py.clone(), rx, ry], &flags);
        rx = vars[outs[0]].clone();
        ry = vars[outs[1]].clone();
    }
    (rx, ry)
}

/// `k * P` by repeated addition: an independent reference sharing none of the digit recoding.
fn repeated_addition<F: halo2curves_axiom::ff::Field + From<u64>, const NEG_A: u64>(
    gx: F,
    gy: F,
    k: u32,
) -> (F, F) {
    if k == 1 {
        return (gx, gy);
    }
    // The first step is `P + P`, which the incomplete addition cannot express.
    let (mut x, mut y) = ec_double_impl::<F, NEG_A>(gx, gy);
    for _ in 2..k {
        (x, y) = ec_add_ne_impl::<F>(x, y, gx, gy);
    }
    (x, y)
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

    // Every scalar must be odd: `sum +-2^i` is odd, so an even one has no digit assignment. The
    // set covers the smallest values, where the accumulator is `+-P` and the ordering argument
    // matters, and a full-width scalar below every supported curve order.
    let scalars: [[u8; SCALAR_LIMBS]; 4] = [
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
            s[0] = 0x5b;
            // below any supported curve order
            s[SCALAR_LIMBS - 1] = 0x3c;
            s
        },
    ];

    // Anchor the recoding against a reference that shares none of it.
    for k in [1u32, 3, 5, 7, 9, 17] {
        let mut scalar = [0u8; SCALAR_LIMBS];
        scalar[..4].copy_from_slice(&k.to_le_bytes());
        let (nx, ny) = ec_mul_impl::<F, NEG_A>(gx, gy, &scalar, EC_MUL_SCALAR_BITS);
        let (rx, ry) = repeated_addition::<F, NEG_A>(gx, gy, k);
        assert_eq!(
            (to_biguint(&nx), to_biguint(&ny)),
            (to_biguint(&rx), to_biguint(&ry)),
            "{name}: ladder disagrees with repeated addition for k = {k}"
        );
    }

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

#[test]
fn supported_scalar_orders_are_accepted() {
    for hex in [
        // secp256k1 group order. Note its *coordinate* modulus is 3 mod 4, so passing the wrong
        // one of the two trips this check rather than silently building an unprovable chip.
        "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141",
        // secp256r1
        "ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551",
        // bn254 Fr
        "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001",
        // bls12-381 Fr
        "73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001",
    ] {
        let order = BigUint::parse_bytes(hex.as_bytes(), 16).unwrap();
        assert_eq!(&order % 4u32, BigUint::from(1u32), "{hex} is not 1 mod 4");
        super::assert_supported_scalar_order(&order);
    }
}

#[test]
#[should_panic(expected = "scalar order congruent to 1 mod 4")]
fn scalar_order_three_mod_four_is_rejected() {
    // 23 = 3 (mod 4): scalar 21 reaches prefix 11, where 2*11 = -1 makes the addend and the
    // doubled accumulator share an x-coordinate.
    super::assert_supported_scalar_order(&BigUint::from(23u32));
}

/// Pins the row layout the CUDA mirror restates by offsetof; a reordered field there would write
/// to the wrong column rather than fail to compile.
#[test]
fn ec_mul_column_widths_match_the_cuda_mirror() {
    use crate::{ECC_BLOCKS_32, ECC_BLOCKS_48, NUM_LIMBS_32, NUM_LIMBS_48};

    assert_eq!(super::ec_mul_header_width(), 135);
    assert_eq!(super::ec_mul_io_width::<NUM_LIMBS_32, ECC_BLOCKS_32>(), 153);
    assert_eq!(super::ec_mul_io_width::<NUM_LIMBS_48, ECC_BLOCKS_48>(), 185);
}
