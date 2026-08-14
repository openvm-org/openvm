//! Checks the `EC_MUL` ladder step expression against a reference scalar multiplication by
//! driving the expression itself through a full 256-step ladder.
//!
//! This covers the expression's compute path, which is what trace generation evaluates. It does not
//! cover the AIR constraints.

use num_bigint::BigUint;
use num_traits::{One, Zero};
use openvm_circuit_primitives::bigint::utils::{secp256k1_coord_prime, secp256r1_coord_prime};
use openvm_ecc_circuit::{
    ec_mul_step_program, EC_MUL_COMPUTE_ROWS, EC_MUL_SCALAR_BITS, EC_MUL_SIGN_PATTERNS,
    EC_MUL_STEPS_PER_ROW,
};
use openvm_mod_circuit_builder::ExprBuilderConfig;

const LIMB_BITS: usize = 8;
const RANGE_MAX_BITS: usize = 17;

// Reference affine EC arithmetic over BigUint, independent of the expression under test.

/// Affine point; `None` is the point at infinity.
type Pt = Option<(BigUint, BigUint)>;

fn inv(a: &BigUint, p: &BigUint) -> BigUint {
    // p is prime, so a^(p-2) mod p
    a.modpow(&(p - BigUint::from(2u32)), p)
}

fn ref_double(pt: &Pt, a: &BigUint, p: &BigUint) -> Pt {
    let (x, y) = pt.as_ref()?;
    if y.is_zero() {
        return None;
    }
    let num = (BigUint::from(3u32) * x * x + a) % p;
    let den = inv(&((BigUint::from(2u32) * y) % p), p);
    let lambda = (num * den) % p;
    let x3 = (&lambda * &lambda + p * 2u32 - BigUint::from(2u32) * x % p) % p;
    let y3 = (&lambda * ((x + p - &x3) % p) + p - y) % p;
    Some((x3, y3))
}

fn ref_add(p1: &Pt, p2: &Pt, a: &BigUint, p: &BigUint) -> Pt {
    let (x1, y1) = match p1 {
        None => return p2.clone(),
        Some(v) => v,
    };
    let (x2, y2) = match p2 {
        None => return p1.clone(),
        Some(v) => v,
    };
    if x1 == x2 {
        return if y1 == y2 { ref_double(p1, a, p) } else { None };
    }
    let num = (y2 + p - y1) % p;
    let den = inv(&((x2 + p - x1) % p), p);
    let lambda = (num * den) % p;
    let x3 = (&lambda * &lambda + p * 2u32 - x1 - x2) % p;
    let y3 = (&lambda * ((x1 + p - &x3) % p) + p - y1) % p;
    Some((x3, y3))
}

fn ref_mul(k: &BigUint, base: &Pt, a: &BigUint, p: &BigUint) -> Pt {
    let mut r: Pt = None;
    for i in (0..EC_MUL_SCALAR_BITS).rev() {
        r = ref_double(&r, a, p);
        if k.bit(i as u64) {
            r = ref_add(&r, base, a, p);
        }
    }
    r
}

/// Runs the ladder through `program.execute`, the same evaluation path used by trace generation.
///
/// The sign patterns are recomputed here rather than imported, so this restates the recoding
/// independently of the chip's own helper.
fn expr_mul(
    program: &openvm_mod_circuit_builder::FieldExpressionProgram,
    k: &BigUint,
    base: &(BigUint, BigUint),
) -> Pt {
    let (px, py) = base;
    // The most significant digit is `+1`, so the accumulator seeds itself from `P`.
    let mut rx = px.clone();
    let mut ry = py.clone();
    let outs = program.output_indices();

    for row in 0..EC_MUL_COMPUTE_ROWS {
        // Digit `i` is bit `i + 1` of the scalar: the ladder's value is `2B + 1`.
        let mut pattern = 0usize;
        for step in 0..EC_MUL_STEPS_PER_ROW {
            let i = EC_MUL_SCALAR_BITS - 1 - (row * EC_MUL_STEPS_PER_ROW + step);
            let bit = k.bit((i + 1) as u64);
            pattern |= (bit as usize) << (EC_MUL_STEPS_PER_ROW - 1 - step);
        }
        let mut flags = [false; EC_MUL_SIGN_PATTERNS];
        flags[pattern] = true;

        let vars = program.execute(&[px.clone(), py.clone(), rx.clone(), ry.clone()], &flags);
        rx = vars[outs[0]].clone();
        ry = vars[outs[1]].clone();
    }

    Some((rx, ry))
}

struct Curve {
    name: &'static str,
    p: BigUint,
    a: BigUint,
    gx: BigUint,
    gy: BigUint,
}

fn curves() -> Vec<Curve> {
    let k1_p = secp256k1_coord_prime();
    let r1_p = secp256r1_coord_prime();
    vec![
        Curve {
            name: "secp256k1",
            a: BigUint::zero(),
            gx: BigUint::parse_bytes(
                b"79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798",
                16,
            )
            .unwrap(),
            gy: BigUint::parse_bytes(
                b"483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8",
                16,
            )
            .unwrap(),
            p: k1_p,
        },
        Curve {
            name: "secp256r1",
            a: &r1_p - BigUint::from(3u32),
            gx: BigUint::parse_bytes(
                b"6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296",
                16,
            )
            .unwrap(),
            gy: BigUint::parse_bytes(
                b"4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5",
                16,
            )
            .unwrap(),
            p: r1_p,
        },
    ]
}

#[test]
fn step_expr_matches_reference_ladder() {
    for c in curves() {
        let config = ExprBuilderConfig {
            modulus: c.p.clone(),
            num_limbs: 32,
            limb_bits: LIMB_BITS,
        };
        let program = ec_mul_step_program(config, RANGE_MAX_BITS, c.a.clone());
        let g = (c.gx.clone(), c.gy.clone());
        let g_pt: Pt = Some(g.clone());

        // Only odd scalars: `sum +-2^i` is odd, so an even operand has no digit assignment and the
        // final row's `scalar = 2B + 1` check rejects it. That check is what makes an even operand
        // unprovable rather than silently answering `(k + 1) * P`, and it is not exercised here --
        // this test drives the expression alone.
        //
        // Covers one, small values, a long run of leading zeros, and a full-width scalar.
        let scalars = [
            BigUint::one(),
            BigUint::from(3u32),
            BigUint::from(5u32),
            BigUint::from(255u32),
            BigUint::from(0x1234_5679u32),
            BigUint::parse_bytes(
                b"0000000000000000000000000000000000000000000000000000000000FFFFFF",
                16,
            )
            .unwrap(),
            BigUint::parse_bytes(
                b"7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A1",
                16,
            )
            .unwrap(),
        ];

        for k in scalars {
            let expected = ref_mul(&k, &g_pt, &c.a, &c.p);
            let got = expr_mul(&program, &k, &g);
            assert_eq!(
                got, expected,
                "{}: ladder mismatch for scalar {}",
                c.name, k
            );
        }
    }
}
