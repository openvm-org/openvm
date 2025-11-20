use halo2curves_axiom::ff::PrimeField;
use num_bigint::BigUint;
use num_traits::Num;
use openvm_algebra_circuit::fields::{
    blocks_to_field_element, blocks_to_field_element_bls12_381_coordinate, field_element_to_blocks,
    field_element_to_blocks_bls12_381_coordinate, FieldType,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurveType {
    K256 = 0,
    P256 = 1,
    BN254 = 2,
    BLS12_381 = 3,
}

const P256_NEG_A: u64 = 3;

fn get_modulus_as_bigint<F: PrimeField>() -> BigUint {
    BigUint::from_str_radix(F::MODULUS.trim_start_matches("0x"), 16).unwrap()
}

pub(super) fn get_curve_type(modulus: &BigUint, a_coeff: &BigUint) -> Option<CurveType> {
    if modulus == &get_modulus_as_bigint::<halo2curves_axiom::secq256k1::Fq>()
        && a_coeff == &BigUint::ZERO
    {
        return Some(CurveType::K256);
    }

    let coeff_a = (-halo2curves_axiom::secp256r1::Fp::from(P256_NEG_A)).to_bytes();
    if modulus == &get_modulus_as_bigint::<halo2curves_axiom::secp256r1::Fp>()
        && a_coeff == &BigUint::from_bytes_le(&coeff_a)
    {
        return Some(CurveType::P256);
    }

    if modulus == &get_modulus_as_bigint::<halo2curves_axiom::bn256::Fq>()
        && a_coeff == &BigUint::ZERO
    {
        return Some(CurveType::BN254);
    }

    if modulus == &get_modulus_as_bigint::<halo2curves_axiom::bls12_381::Fq>()
        && a_coeff == &BigUint::ZERO
    {
        return Some(CurveType::BLS12_381);
    }

    None
}

#[inline(always)]
pub fn ec_add_ne<const FIELD_TYPE: u8, const BLOCKS: usize, const BLOCK_SIZE: usize>(
    input_data: [[[u8; BLOCK_SIZE]; BLOCKS]; 2],
) -> [[u8; BLOCK_SIZE]; BLOCKS] {
    match FIELD_TYPE {
        x if x == FieldType::K256Coordinate as u8 => {
            ec_add_ne_256bit::<halo2curves_axiom::secq256k1::Fq, BLOCKS, BLOCK_SIZE>(input_data)
        }
        x if x == FieldType::P256Coordinate as u8 => {
            ec_add_ne_256bit::<halo2curves_axiom::secp256r1::Fp, BLOCKS, BLOCK_SIZE>(input_data)
        }
        x if x == FieldType::BN254Coordinate as u8 => {
            ec_add_ne_256bit::<halo2curves_axiom::bn256::Fq, BLOCKS, BLOCK_SIZE>(input_data)
        }
        x if x == FieldType::BLS12_381Coordinate as u8 => {
            ec_add_ne_bls12_381::<BLOCKS, BLOCK_SIZE>(input_data)
        }
        _ => panic!("Unsupported field type: {FIELD_TYPE}"),
    }
}

/// Dispatch elliptic curve point doubling based on const generic curve type
#[inline(always)]
pub fn ec_double<const CURVE_TYPE: u8, const BLOCKS: usize, const BLOCK_SIZE: usize>(
    input_data: [[u8; BLOCK_SIZE]; BLOCKS],
) -> [[u8; BLOCK_SIZE]; BLOCKS] {
    match CURVE_TYPE {
        x if x == CurveType::K256 as u8 => {
            ec_double_256bit::<halo2curves_axiom::secq256k1::Fq, 0, BLOCKS, BLOCK_SIZE>(input_data)
        }
        x if x == CurveType::P256 as u8 => {
            ec_double_256bit::<halo2curves_axiom::secp256r1::Fp, P256_NEG_A, BLOCKS, BLOCK_SIZE>(
                input_data,
            )
        }
        x if x == CurveType::BN254 as u8 => {
            ec_double_256bit::<halo2curves_axiom::bn256::Fq, 0, BLOCKS, BLOCK_SIZE>(input_data)
        }
        x if x == CurveType::BLS12_381 as u8 => {
            ec_double_bls12_381::<BLOCKS, BLOCK_SIZE>(input_data)
        }
        _ => panic!("Unsupported curve type: {CURVE_TYPE}"),
    }
}

#[inline(always)]
pub fn ec_mul<const CURVE_TYPE: u8, const BLOCKS: usize, const BLOCK_SIZE: usize>(
    scalar_data: [u8; BLOCK_SIZE],
    point_data: [[u8; BLOCK_SIZE]; BLOCKS],
) -> [[u8; BLOCK_SIZE]; BLOCKS] {
    match CURVE_TYPE {
        x if x == CurveType::K256 as u8 => {
            // read scalar and point data
            let scalar = blocks_to_field_element::<halo2curves_axiom::secq256k1::Fp>(&scalar_data);
            let x1 = blocks_to_field_element::<halo2curves_axiom::secq256k1::Fq>(
                point_data[..BLOCKS / 2].as_flattened(),
            );
            let y1 = blocks_to_field_element::<halo2curves_axiom::secq256k1::Fq>(
                point_data[BLOCKS / 2..].as_flattened(),
            );

            // perform elliptic curve multiplication
            let (x3, y3) = ec_mul_impl::<
                halo2curves_axiom::secq256k1::Fq,
                halo2curves_axiom::secq256k1::Fp,
                0,
            >(scalar, x1, y1);

            // write output data to memory
            let mut output = [[0u8; BLOCK_SIZE]; BLOCKS];
            field_element_to_blocks::<halo2curves_axiom::secq256k1::Fq, BLOCK_SIZE>(
                &x3,
                &mut output[..BLOCKS / 2],
            );
            field_element_to_blocks::<halo2curves_axiom::secq256k1::Fq, BLOCK_SIZE>(
                &y3,
                &mut output[BLOCKS / 2..],
            );
            output
        }
        _ => panic!("Unsupported curve type: {}", CURVE_TYPE),
    }
}

#[inline(always)]
fn ec_add_ne_256bit<
    F: PrimeField<Repr = [u8; 32]>,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
>(
    input_data: [[[u8; BLOCK_SIZE]; BLOCKS]; 2],
) -> [[u8; BLOCK_SIZE]; BLOCKS] {
    let x1 = blocks_to_field_element::<F>(input_data[0][..BLOCKS / 2].as_flattened());
    let y1 = blocks_to_field_element::<F>(input_data[0][BLOCKS / 2..].as_flattened());
    let x2 = blocks_to_field_element::<F>(input_data[1][..BLOCKS / 2].as_flattened());
    let y2 = blocks_to_field_element::<F>(input_data[1][BLOCKS / 2..].as_flattened());

    let (x3, y3) = ec_add_ne_impl::<F>(x1, y1, x2, y2);

    let mut output = [[0u8; BLOCK_SIZE]; BLOCKS];
    field_element_to_blocks::<F, BLOCK_SIZE>(&x3, &mut output[..BLOCKS / 2]);
    field_element_to_blocks::<F, BLOCK_SIZE>(&y3, &mut output[BLOCKS / 2..]);
    output
}

#[inline(always)]
fn ec_double_256bit<
    F: PrimeField<Repr = [u8; 32]>,
    const NEG_A: u64,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
>(
    input_data: [[u8; BLOCK_SIZE]; BLOCKS],
) -> [[u8; BLOCK_SIZE]; BLOCKS] {
    let x1 = blocks_to_field_element::<F>(input_data[..BLOCKS / 2].as_flattened());
    let y1 = blocks_to_field_element::<F>(input_data[BLOCKS / 2..].as_flattened());

    let (x3, y3) = ec_double_impl::<F, NEG_A>(x1, y1);

    let mut output = [[0u8; BLOCK_SIZE]; BLOCKS];
    field_element_to_blocks::<F, BLOCK_SIZE>(&x3, &mut output[..BLOCKS / 2]);
    field_element_to_blocks::<F, BLOCK_SIZE>(&y3, &mut output[BLOCKS / 2..]);
    output
}

#[inline(always)]
fn ec_add_ne_bls12_381<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    input_data: [[[u8; BLOCK_SIZE]; BLOCKS]; 2],
) -> [[u8; BLOCK_SIZE]; BLOCKS] {
    // Extract coordinates
    let x1 =
        blocks_to_field_element_bls12_381_coordinate(input_data[0][..BLOCKS / 2].as_flattened());
    let y1 =
        blocks_to_field_element_bls12_381_coordinate(input_data[0][BLOCKS / 2..].as_flattened());
    let x2 =
        blocks_to_field_element_bls12_381_coordinate(input_data[1][..BLOCKS / 2].as_flattened());
    let y2 =
        blocks_to_field_element_bls12_381_coordinate(input_data[1][BLOCKS / 2..].as_flattened());

    let (x3, y3) = ec_add_ne_impl::<halo2curves_axiom::bls12_381::Fq>(x1, y1, x2, y2);

    // Final output
    let mut output = [[0u8; BLOCK_SIZE]; BLOCKS];
    field_element_to_blocks_bls12_381_coordinate(&x3, &mut output[..BLOCKS / 2]);
    field_element_to_blocks_bls12_381_coordinate(&y3, &mut output[BLOCKS / 2..]);
    output
}

#[inline(always)]
fn ec_double_bls12_381<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    input_data: [[u8; BLOCK_SIZE]; BLOCKS],
) -> [[u8; BLOCK_SIZE]; BLOCKS] {
    // Extract coordinates
    let x1 = blocks_to_field_element_bls12_381_coordinate(input_data[..BLOCKS / 2].as_flattened());
    let y1 = blocks_to_field_element_bls12_381_coordinate(input_data[BLOCKS / 2..].as_flattened());

    let (x3, y3) = ec_double_impl::<halo2curves_axiom::bls12_381::Fq, 0>(x1, y1);

    // Final output
    let mut output = [[0u8; BLOCK_SIZE]; BLOCKS];
    field_element_to_blocks_bls12_381_coordinate(&x3, &mut output[..BLOCKS / 2]);
    field_element_to_blocks_bls12_381_coordinate(&y3, &mut output[BLOCKS / 2..]);
    output
}

#[inline(always)]
pub fn ec_add_ne_impl<F: PrimeField>(x1: F, y1: F, x2: F, y2: F) -> (F, F) {
    // Calculate lambda = (y2 - y1) / (x2 - x1)
    let lambda = (y2 - y1) * (x2 - x1).invert().unwrap();

    // Calculate x3 = lambda^2 - x1 - x2
    let x3 = lambda.square() - x1 - x2;

    // Calculate y3 = lambda * (x1 - x3) - y1
    let y3 = lambda * (x1 - x3) - y1;

    (x3, y3)
}

#[inline(always)]
pub fn ec_double_impl<F: PrimeField, const NEG_A: u64>(x1: F, y1: F) -> (F, F) {
    // Calculate lambda based on curve coefficient 'a'
    let x1_squared = x1.square();
    let three_x1_squared = x1_squared + x1_squared.double();
    let two_y1 = y1.double();

    let lambda = if NEG_A == 0 {
        // For a = 0: lambda = (3 * x1^2) / (2 * y1)
        three_x1_squared * two_y1.invert().unwrap()
    } else {
        // lambda = (3 * x1^2 + a) / (2 * y1)
        (three_x1_squared - F::from(NEG_A)) * two_y1.invert().unwrap()
    };

    // Calculate x3 = lambda^2 - 2 * x1
    let x3 = lambda.square() - x1.double();

    // Calculate y3 = lambda * (x1 - x3) - y1
    let y3 = lambda * (x1 - x3) - y1;

    (x3, y3)
}

#[inline(always)]
fn ec_mul_impl<Fq: PrimeField, Fr: PrimeField, const NEG_A: u64>(
    scalar: Fr,
    x: Fq,
    y: Fq,
) -> (Fq, Fq) {
    // Convert scalar to bytes to process bit-by-bit
    let scalar_bytes = scalar.to_repr();

    // Initialize accumulator as point at infinity (represented by Z = 0)
    let mut acc_x = Fq::ZERO;
    let mut acc_y = Fq::ONE;
    let mut acc_z = Fq::ZERO;

    // Process scalar bits from most significant to least significant
    let mut found_one = false;
    for byte_idx in (0..32).rev() {
        for bit_idx in (0..8).rev() {
            let bit = (scalar_bytes.as_ref()[byte_idx] >> bit_idx) & 1;

            if !found_one {
                if bit == 1 {
                    // First bit set - initialize accumulator with the base point
                    acc_x = x;
                    acc_y = y;
                    acc_z = Fq::ONE;
                    found_one = true;
                }
                continue;
            }

            // Double: acc = 2 * acc (in Jacobian coordinates)
            (acc_x, acc_y, acc_z) = ec_double_jacobian::<Fq, NEG_A>(acc_x, acc_y, acc_z);

            // Add if bit is 1: acc = acc + point (mixed addition: Jacobian + Affine)
            if bit == 1 {
                (acc_x, acc_y, acc_z) = ec_add_mixed_jacobian::<Fq>(acc_x, acc_y, acc_z, x, y);
            }
        }
    }

    // Convert result from Jacobian to affine coordinates
    jacobian_to_affine(acc_x, acc_y, acc_z)
}

/// Point doubling in Jacobian coordinates
/// Input: (X, Y, Z) representing point (X/Z², Y/Z³)
/// Output: (X', Y', Z') representing 2P
#[inline(always)]
fn ec_double_jacobian<F: PrimeField, const NEG_A: u64>(x: F, y: F, z: F) -> (F, F, F) {
    // Handle point at infinity
    if z.is_zero().into() {
        return (F::ZERO, F::ONE, F::ZERO);
    }

    // Using the formula for curve y² = x³ + ax + b
    // For a = 0 (secp256k1, BN254, BLS12-381):
    //   S = 4*X*Y²
    //   M = 3*X²
    //   X' = M² - 2*S
    //   Y' = M*(S - X') - 8*Y⁴
    //   Z' = 2*Y*Z

    let yy = y.square();
    let s = x.double().double() * yy;
    let m = if NEG_A == 0 {
        x.square().double() + x.square() // 3*X²
    } else {
        // For P256: M = 3*X² + a*Z⁴
        let zz = z.square();
        let zzzz = zz.square();
        x.square().double() + x.square() - F::from(NEG_A) * zzzz
    };

    let x_new = m.square() - s.double();
    let yyyy = yy.square();
    let y_new = m * (s - x_new) - yyyy.double().double().double();
    let z_new = y.double() * z;

    (x_new, y_new, z_new)
}

/// Mixed addition: Jacobian + Affine -> Jacobian
/// Add affine point (x2, y2) to Jacobian point (X1, Y1, Z1)
#[inline(always)]
fn ec_add_mixed_jacobian<F: PrimeField>(x1: F, y1: F, z1: F, x2: F, y2: F) -> (F, F, F) {
    // Handle point at infinity cases
    if z1.is_zero().into() {
        return (x2, y2, F::ONE);
    }

    // Mixed addition formula:
    //   Z1Z1 = Z1²
    //   U2 = X2*Z1Z1
    //   S2 = Y2*Z1*Z1Z1
    //   H = U2 - X1
    //   HH = H²
    //   I = 4*HH
    //   J = H*I
    //   r = 2*(S2 - Y1)
    //   V = X1*I
    //   X3 = r² - J - 2*V
    //   Y3 = r*(V - X3) - 2*Y1*J
    //   Z3 = 2*Z1*H

    let z1z1 = z1.square();
    let u2 = x2 * z1z1;
    let s2 = y2 * z1 * z1z1;
    let h = u2 - x1;

    // Check if points are equal (need doubling instead)
    if h.is_zero().into() {
        if (s2 - y1).is_zero().into() {
            return ec_double_jacobian::<F, 0>(x1, y1, z1);
        } else {
            // Points are inverses, return point at infinity
            return (F::ZERO, F::ONE, F::ZERO);
        }
    }

    let hh = h.square();
    let i = hh.double().double();
    let j = h * i;
    let r = (s2 - y1).double();
    let v = x1 * i;

    let x3 = r.square() - j - v.double();
    let y3 = r * (v - x3) - (y1 * j).double();
    let z3 = z1.double() * h;

    (x3, y3, z3)
}

/// Convert Jacobian coordinates to affine coordinates
/// Input: (X, Y, Z) representing (X/Z², Y/Z³)
/// Output: (x, y) representing the affine point
#[inline(always)]
fn jacobian_to_affine<F: PrimeField>(x: F, y: F, z: F) -> (F, F) {
    // Handle point at infinity
    if z.is_zero().into() {
        // Return (0, 0) for point at infinity
        // Note: In practice, you might want to handle this differently
        return (F::ZERO, F::ZERO);
    }

    let z_inv = z.invert().unwrap();
    let z_inv_sq = z_inv.square();
    let x_affine = x * z_inv_sq;
    let y_affine = y * z_inv * z_inv_sq;

    (x_affine, y_affine)
}
