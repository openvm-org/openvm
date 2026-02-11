use halo2curves_axiom::ff::{Field, PrimeField};
use num_bigint::BigUint;
use num_traits::Num;
use openvm_algebra_circuit::fields::{
    blocks_to_field_element, blocks_to_field_element_bls12_381_coordinate, field_element_to_blocks,
    field_element_to_blocks_bls12_381_coordinate,
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
pub fn ec_add_proj_impl_a0<F: Field>(
    x1: F,
    y1: F,
    z1: F,
    x2: F,
    y2: F,
    z2: F,
    b3: F, // 3*b coefficient
) -> (F, F, F) {
    let t0 = x1 * x2;
    let t1 = y1 * y2;
    let t2 = z1 * z2;
    let t3 = x1 + y1;
    let t4 = x2 + y2;
    let t3 = t3 * t4;
    let t4 = t0 + t1;
    let t3 = t3 - t4;
    let t4 = y1 + z1;
    let x3 = y2 + z2;
    let t4 = t4 * x3;
    let x3 = t1 + t2;
    let t4 = t4 - x3;
    let x3 = x1 + z1;
    let y3 = x2 + z2;
    let x3 = x3 * y3;
    let y3 = t0 + t2;
    let y3 = x3 - y3;
    let x3 = t0.double() + t0;
    let t2 = b3 * t2;
    let z3 = t1 + t2;
    let t1 = t1 - t2;
    let y3 = b3 * y3;
    let x3_out = t4 * y3;
    let t2 = t3 * t1;
    let x3_out = t2 - x3_out;
    let y3 = y3 * x3;
    let t1 = t1 * z3;
    let y3_out = t1 + y3;
    let x3 = x3 * t3;
    let z3 = z3 * t4;
    let z3_out = z3 + x3;
    (x3_out, y3_out, z3_out)
}

#[inline(always)]
pub fn ec_double_proj_impl_a0<F: Field>(x1: F, y1: F, z1: F, b3: F) -> (F, F, F) {
    let t0 = y1.square();
    let z3 = t0.double().double().double();
    let t1 = y1 * z1;
    let t2 = z1.square();
    let t2 = b3 * t2;
    let x3 = t2 * z3;
    let y3 = t0 + t2;
    let z3 = t1 * z3;
    let t1 = t2.double();
    let t2 = t1 + t2;
    let t0 = t0 - t2;
    let y3 = t0 * y3;
    let y3 = x3 + y3;
    let t1 = x1 * y1;
    let x3 = t0 * t1;
    let x3 = x3.double();
    (x3, y3, z3)
}

#[inline(always)]
pub fn ec_add_proj_impl_general<F: Field>(
    x1: F,
    y1: F,
    z1: F,
    x2: F,
    y2: F,
    z2: F,
    a: F,
    b3: F,
) -> (F, F, F) {
    let t0 = x1 * x2;
    let t1 = y1 * y2;
    let t2 = z1 * z2;
    let t3 = x1 + y1;
    let t4 = x2 + y2;
    let t3 = t3 * t4;
    let t4 = t0 + t1;
    let t3 = t3 - t4;
    let t4 = x1 + z1;
    let t5 = x2 + z2;
    let t4 = t4 * t5;
    let t5 = t0 + t2;
    let t4 = t4 - t5;
    let t5 = y1 + z1;
    let x3 = y2 + z2;
    let t5 = t5 * x3;
    let x3 = t1 + t2;
    let t5 = t5 - x3;
    let z3 = a * t4;
    let x3 = b3 * t2;
    let z3 = x3 + z3;
    let x3 = t1 - z3;
    let z3 = t1 + z3;
    let y3 = x3 * z3;
    let t1 = t0.double() + t0;
    let t2 = a * t2;
    let t4 = b3 * t4;
    let t1 = t1 + t2;
    let t2 = t0 - t2;
    let t2 = a * t2;
    let t4 = t4 + t2;
    let t0 = t1 * t4;
    let y3 = y3 + t0;
    let t0 = t5 * t4;
    let x3 = t3 * x3;
    let x3 = x3 - t0;
    let t0 = t3 * t1;
    let z3 = t5 * z3;
    let z3 = z3 + t0;
    (x3, y3, z3)
}

#[inline(always)]
pub fn ec_double_proj_impl_general<F: Field>(x1: F, y1: F, z1: F, a: F, b3: F) -> (F, F, F) {
    let t0 = x1.square();
    let t1 = y1.square();
    let t2 = z1.square();
    let t3 = x1 * y1;
    let t3 = t3.double();
    let z3 = x1 * z1;
    let z3 = z3.double();
    let x3 = a * z3;
    let y3 = b3 * t2;
    let y3 = x3 + y3;
    let x3 = t1 - y3;
    let y3 = t1 + y3;
    let y3 = x3 * y3;
    let x3 = t3 * x3;
    let z3 = b3 * z3;
    let t2 = a * t2;
    let t3 = t0 - t2;
    let t3 = a * t3;
    let t3 = t3 + z3;
    let z3 = t0.double() + t0;
    let t0 = z3 + t2;
    let t0 = t0 * t3;
    let y3 = y3 + t0;
    let t2 = y1 * z1;
    let t2 = t2.double();
    let t0 = t2 * t3;
    let x3 = x3 - t0;
    let z3 = t2 * t1;
    let z3 = z3.double().double();
    (x3, y3, z3)
}

#[inline(always)]
pub fn ec_add_proj<const CURVE_TYPE: u8, const BLOCKS: usize, const BLOCK_SIZE: usize>(
    input_data: [[[u8; BLOCK_SIZE]; BLOCKS]; 2],
) -> [[u8; BLOCK_SIZE]; BLOCKS] {
    match CURVE_TYPE {
        x if x == CurveType::K256 as u8 => ec_add_proj_k256_bytes::<BLOCKS, BLOCK_SIZE>(input_data),
        x if x == CurveType::P256 as u8 => ec_add_proj_256bit_general::<
            halo2curves_axiom::secp256r1::Fp,
            P256_NEG_A,
            BLOCKS,
            BLOCK_SIZE,
        >(input_data),
        x if x == CurveType::BN254 as u8 => {
            ec_add_proj_256bit_a0::<halo2curves_axiom::bn256::Fq, BN254_B3, BLOCKS, BLOCK_SIZE>(
                input_data,
            )
        }
        x if x == CurveType::BLS12_381 as u8 => {
            ec_add_proj_bls12_381::<BLOCKS, BLOCK_SIZE>(input_data)
        }
        _ => panic!("Unsupported curve type: {CURVE_TYPE}"),
    }
}

/// Dispatch elliptic curve point doubling based on const generic curve type
#[inline(always)]
pub fn ec_double_proj<const CURVE_TYPE: u8, const BLOCKS: usize, const BLOCK_SIZE: usize>(
    input_data: [[u8; BLOCK_SIZE]; BLOCKS],
) -> [[u8; BLOCK_SIZE]; BLOCKS] {
    match CURVE_TYPE {
        x if x == CurveType::K256 as u8 => {
            ec_double_proj_k256_bytes::<BLOCKS, BLOCK_SIZE>(input_data)
        }
        x if x == CurveType::P256 as u8 => ec_double_proj_256bit_general::<
            halo2curves_axiom::secp256r1::Fp,
            P256_NEG_A,
            BLOCKS,
            BLOCK_SIZE,
        >(input_data),
        x if x == CurveType::BN254 as u8 => {
            ec_double_proj_256bit_a0::<halo2curves_axiom::bn256::Fq, BN254_B3, BLOCKS, BLOCK_SIZE>(
                input_data,
            )
        }
        x if x == CurveType::BLS12_381 as u8 => {
            ec_double_proj_bls12_381::<BLOCKS, BLOCK_SIZE>(input_data)
        }
        _ => panic!("Unsupported curve type: {CURVE_TYPE}"),
    }
}

const K256_B3: u64 = 21;

// k256's FieldElement tracks internal magnitude. Subtraction uses negate(1) which requires
// magnitude <= 1, so normalize_weak() is needed before subtracting values with magnitude > 1.
#[inline(always)]
fn ec_add_proj_k256(
    x1: k256::FieldElement,
    y1: k256::FieldElement,
    z1: k256::FieldElement,
    x2: k256::FieldElement,
    y2: k256::FieldElement,
    z2: k256::FieldElement,
    b3: k256::FieldElement,
) -> (k256::FieldElement, k256::FieldElement, k256::FieldElement) {
    let t0 = x1 * x2;
    let t1 = y1 * y2;
    let t2 = z1 * z2;
    let t3 = x1 + y1;
    let t4 = x2 + y2;
    let t3 = t3 * t4;
    let t4 = t0 + t1;
    let t3 = t3 - t4.normalize_weak();
    let t4 = y1 + z1;
    let x3 = y2 + z2;
    let t4 = t4 * x3;
    let x3 = t1 + t2;
    let t4 = t4 - x3.normalize_weak();
    let x3 = x1 + z1;
    let y3 = x2 + z2;
    let x3 = x3 * y3;
    let y3 = t0 + t2;
    let y3 = x3 - y3.normalize_weak();
    let x3 = t0.double() + t0;
    let t2 = b3 * t2;
    let z3 = t1 + t2;
    let t1 = t1 - t2;
    let y3 = b3 * y3;
    let x3_out = t4 * y3;
    let t2 = t3 * t1;
    let x3_out = t2 - x3_out;
    let y3 = y3 * x3;
    let t1 = t1 * z3;
    let y3_out = t1 + y3;
    let x3 = x3 * t3;
    let z3 = z3 * t4;
    let z3_out = z3 + x3;
    (x3_out, y3_out, z3_out)
}

// k256's FieldElement tracks internal magnitude. Subtraction uses negate(1) which requires
// magnitude <= 1, so normalize_weak() is needed before subtracting values with magnitude > 1.
#[inline(always)]
fn ec_double_proj_k256(
    x1: k256::FieldElement,
    y1: k256::FieldElement,
    z1: k256::FieldElement,
    b3: k256::FieldElement,
) -> (k256::FieldElement, k256::FieldElement, k256::FieldElement) {
    let t0 = y1.square();
    let z3 = t0.double();
    let z3 = z3.double();
    let z3 = z3.double();
    let t1 = y1 * z1;
    let t2 = z1.square();
    let t2 = b3 * t2;
    let x3 = t2 * z3;
    let y3 = t0 + t2;
    let z3 = t1 * z3;
    let t1 = t2.double();
    let t2 = t1 + t2;
    let t0 = t0 - t2.normalize_weak();
    let y3 = t0 * y3;
    let y3 = x3 + y3;
    let t1 = x1 * y1;
    let x3 = t0 * t1;
    let x3 = x3.double();
    (x3, y3, z3)
}

#[inline(always)]
fn bytes_le_to_k256_field(bytes: &[u8; 32]) -> k256::FieldElement {
    let mut be_bytes = *bytes;
    be_bytes.reverse();
    k256::FieldElement::from_bytes(&be_bytes.into()).unwrap()
}

#[inline(always)]
fn k256_field_to_bytes_le<const BLOCK_SIZE: usize>(
    field: &k256::FieldElement,
    output: &mut [[u8; BLOCK_SIZE]],
) {
    let be_bytes = field.to_bytes();
    let mut le_bytes = [0u8; 32];
    le_bytes.copy_from_slice(&be_bytes);
    le_bytes.reverse();
    for (i, byte) in le_bytes.iter().enumerate() {
        let block_idx = i / BLOCK_SIZE;
        let byte_idx = i % BLOCK_SIZE;
        if block_idx < output.len() {
            output[block_idx][byte_idx] = *byte;
        }
    }
}

#[inline(always)]
fn ec_add_proj_k256_bytes<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    input_data: [[[u8; BLOCK_SIZE]; BLOCKS]; 2],
) -> [[u8; BLOCK_SIZE]; BLOCKS] {
    let coord_blocks = BLOCKS / 3;

    // Extract P1 coordinates (X1, Y1, Z1)
    let x1_bytes: [u8; 32] = input_data[0][..coord_blocks]
        .as_flattened()
        .try_into()
        .unwrap();
    let y1_bytes: [u8; 32] = input_data[0][coord_blocks..2 * coord_blocks]
        .as_flattened()
        .try_into()
        .unwrap();
    let z1_bytes: [u8; 32] = input_data[0][2 * coord_blocks..]
        .as_flattened()
        .try_into()
        .unwrap();

    // Extract P2 coordinates (X2, Y2, Z2)
    let x2_bytes: [u8; 32] = input_data[1][..coord_blocks]
        .as_flattened()
        .try_into()
        .unwrap();
    let y2_bytes: [u8; 32] = input_data[1][coord_blocks..2 * coord_blocks]
        .as_flattened()
        .try_into()
        .unwrap();
    let z2_bytes: [u8; 32] = input_data[1][2 * coord_blocks..]
        .as_flattened()
        .try_into()
        .unwrap();

    // Convert to k256 field elements
    let x1 = bytes_le_to_k256_field(&x1_bytes);
    let y1 = bytes_le_to_k256_field(&y1_bytes);
    let z1 = bytes_le_to_k256_field(&z1_bytes);
    let x2 = bytes_le_to_k256_field(&x2_bytes);
    let y2 = bytes_le_to_k256_field(&y2_bytes);
    let z2 = bytes_le_to_k256_field(&z2_bytes);
    let b3 = k256::FieldElement::from(K256_B3);

    // Compute projective addition
    let (x3, y3, z3) = ec_add_proj_k256(x1, y1, z1, x2, y2, z2, b3);

    // Convert back to bytes
    let mut output = [[0u8; BLOCK_SIZE]; BLOCKS];
    k256_field_to_bytes_le(&x3, &mut output[..coord_blocks]);
    k256_field_to_bytes_le(&y3, &mut output[coord_blocks..2 * coord_blocks]);
    k256_field_to_bytes_le(&z3, &mut output[2 * coord_blocks..]);

    output
}

#[inline(always)]
fn ec_double_proj_k256_bytes<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    input_data: [[u8; BLOCK_SIZE]; BLOCKS],
) -> [[u8; BLOCK_SIZE]; BLOCKS] {
    let coord_blocks = BLOCKS / 3;

    // Extract coordinates (X1, Y1, Z1)
    let x1_bytes: [u8; 32] = input_data[..coord_blocks]
        .as_flattened()
        .try_into()
        .unwrap();
    let y1_bytes: [u8; 32] = input_data[coord_blocks..2 * coord_blocks]
        .as_flattened()
        .try_into()
        .unwrap();
    let z1_bytes: [u8; 32] = input_data[2 * coord_blocks..]
        .as_flattened()
        .try_into()
        .unwrap();

    // Convert to k256 field elements
    let x1 = bytes_le_to_k256_field(&x1_bytes);
    let y1 = bytes_le_to_k256_field(&y1_bytes);
    let z1 = bytes_le_to_k256_field(&z1_bytes);
    let b3 = k256::FieldElement::from(K256_B3);

    // Compute projective doubling
    let (x3, y3, z3) = ec_double_proj_k256(x1, y1, z1, b3);

    // Convert back to bytes
    let mut output = [[0u8; BLOCK_SIZE]; BLOCKS];
    k256_field_to_bytes_le(&x3, &mut output[..coord_blocks]);
    k256_field_to_bytes_le(&y3, &mut output[coord_blocks..2 * coord_blocks]);
    k256_field_to_bytes_le(&z3, &mut output[2 * coord_blocks..]);

    output
}

const BN254_B3: u64 = 9;

#[inline(always)]
fn ec_add_proj_256bit_a0<
    F: PrimeField<Repr = [u8; 32]>,
    const B3: u64,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
>(
    input_data: [[[u8; BLOCK_SIZE]; BLOCKS]; 2],
) -> [[u8; BLOCK_SIZE]; BLOCKS] {
    let coord_blocks = BLOCKS / 3;

    // Extract P1 coordinates
    let x1 = blocks_to_field_element::<F>(input_data[0][..coord_blocks].as_flattened());
    let y1 =
        blocks_to_field_element::<F>(input_data[0][coord_blocks..2 * coord_blocks].as_flattened());
    let z1 = blocks_to_field_element::<F>(input_data[0][2 * coord_blocks..].as_flattened());

    // Extract P2 coordinates
    let x2 = blocks_to_field_element::<F>(input_data[1][..coord_blocks].as_flattened());
    let y2 =
        blocks_to_field_element::<F>(input_data[1][coord_blocks..2 * coord_blocks].as_flattened());
    let z2 = blocks_to_field_element::<F>(input_data[1][2 * coord_blocks..].as_flattened());

    // Get 3b coefficient from const generic
    let b3 = F::from(B3);

    let (x3, y3, z3) = ec_add_proj_impl_a0(x1, y1, z1, x2, y2, z2, b3);

    let mut output = [[0u8; BLOCK_SIZE]; BLOCKS];
    field_element_to_blocks::<F, BLOCK_SIZE>(&x3, &mut output[..coord_blocks]);
    field_element_to_blocks::<F, BLOCK_SIZE>(&y3, &mut output[coord_blocks..2 * coord_blocks]);
    field_element_to_blocks::<F, BLOCK_SIZE>(&z3, &mut output[2 * coord_blocks..]);

    output
}

#[inline(always)]
fn ec_double_proj_256bit_a0<
    F: PrimeField<Repr = [u8; 32]>,
    const B3: u64,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
>(
    input_data: [[u8; BLOCK_SIZE]; BLOCKS],
) -> [[u8; BLOCK_SIZE]; BLOCKS] {
    let coord_blocks = BLOCKS / 3;

    let x1 = blocks_to_field_element::<F>(input_data[..coord_blocks].as_flattened());
    let y1 =
        blocks_to_field_element::<F>(input_data[coord_blocks..2 * coord_blocks].as_flattened());
    let z1 = blocks_to_field_element::<F>(input_data[2 * coord_blocks..].as_flattened());

    let b3 = F::from(B3);

    let (x3, y3, z3) = ec_double_proj_impl_a0(x1, y1, z1, b3);

    let mut output = [[0u8; BLOCK_SIZE]; BLOCKS];
    field_element_to_blocks::<F, BLOCK_SIZE>(&x3, &mut output[..coord_blocks]);
    field_element_to_blocks::<F, BLOCK_SIZE>(&y3, &mut output[coord_blocks..2 * coord_blocks]);
    field_element_to_blocks::<F, BLOCK_SIZE>(&z3, &mut output[2 * coord_blocks..]);

    output
}

#[inline(always)]
fn ec_add_proj_256bit_general<
    F: PrimeField<Repr = [u8; 32]>,
    const NEG_A: u64,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
>(
    input_data: [[[u8; BLOCK_SIZE]; BLOCKS]; 2],
) -> [[u8; BLOCK_SIZE]; BLOCKS] {
    let coord_blocks = BLOCKS / 3;

    // Extract P1 coordinates
    let x1 = blocks_to_field_element::<F>(input_data[0][..coord_blocks].as_flattened());
    let y1 =
        blocks_to_field_element::<F>(input_data[0][coord_blocks..2 * coord_blocks].as_flattened());
    let z1 = blocks_to_field_element::<F>(input_data[0][2 * coord_blocks..].as_flattened());

    // Extract P2 coordinates
    let x2 = blocks_to_field_element::<F>(input_data[1][..coord_blocks].as_flattened());
    let y2 =
        blocks_to_field_element::<F>(input_data[1][coord_blocks..2 * coord_blocks].as_flattened());
    let z2 = blocks_to_field_element::<F>(input_data[1][2 * coord_blocks..].as_flattened());

    // P256: a = -3, b = 0x5ac635...
    let a = -F::from(NEG_A);
    let b3 = get_p256_b3::<F>();

    let (x3, y3, z3) = ec_add_proj_impl_general(x1, y1, z1, x2, y2, z2, a, b3);

    let mut output = [[0u8; BLOCK_SIZE]; BLOCKS];
    field_element_to_blocks::<F, BLOCK_SIZE>(&x3, &mut output[..coord_blocks]);
    field_element_to_blocks::<F, BLOCK_SIZE>(&y3, &mut output[coord_blocks..2 * coord_blocks]);
    field_element_to_blocks::<F, BLOCK_SIZE>(&z3, &mut output[2 * coord_blocks..]);

    output
}

#[inline(always)]
fn ec_double_proj_256bit_general<
    F: PrimeField<Repr = [u8; 32]>,
    const NEG_A: u64,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
>(
    input_data: [[u8; BLOCK_SIZE]; BLOCKS],
) -> [[u8; BLOCK_SIZE]; BLOCKS] {
    let coord_blocks = BLOCKS / 3;

    let x1 = blocks_to_field_element::<F>(input_data[..coord_blocks].as_flattened());
    let y1 =
        blocks_to_field_element::<F>(input_data[coord_blocks..2 * coord_blocks].as_flattened());
    let z1 = blocks_to_field_element::<F>(input_data[2 * coord_blocks..].as_flattened());

    let a = -F::from(NEG_A);
    let b3 = get_p256_b3::<F>();

    let (x3, y3, z3) = ec_double_proj_impl_general(x1, y1, z1, a, b3);

    let mut output = [[0u8; BLOCK_SIZE]; BLOCKS];
    field_element_to_blocks::<F, BLOCK_SIZE>(&x3, &mut output[..coord_blocks]);
    field_element_to_blocks::<F, BLOCK_SIZE>(&y3, &mut output[coord_blocks..2 * coord_blocks]);
    field_element_to_blocks::<F, BLOCK_SIZE>(&z3, &mut output[2 * coord_blocks..]);

    output
}

#[inline(always)]
fn get_p256_b3<F: PrimeField<Repr = [u8; 32]>>() -> F {
    let b_bytes: [u8; 32] =
        hex_literal::hex!("5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b");
    let mut le_bytes = b_bytes;
    le_bytes.reverse();
    let b = F::from_repr(le_bytes.into()).unwrap();
    b + b + b
}

const BLS12_381_B3: u64 = 12;

#[inline(always)]
fn ec_add_proj_bls12_381<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    input_data: [[[u8; BLOCK_SIZE]; BLOCKS]; 2],
) -> [[u8; BLOCK_SIZE]; BLOCKS] {
    let coord_blocks = BLOCKS / 3;

    // Extract P1 coordinates
    let x1 =
        blocks_to_field_element_bls12_381_coordinate(input_data[0][..coord_blocks].as_flattened());
    let y1 = blocks_to_field_element_bls12_381_coordinate(
        input_data[0][coord_blocks..2 * coord_blocks].as_flattened(),
    );
    let z1 = blocks_to_field_element_bls12_381_coordinate(
        input_data[0][2 * coord_blocks..].as_flattened(),
    );

    // Extract P2 coordinates
    let x2 =
        blocks_to_field_element_bls12_381_coordinate(input_data[1][..coord_blocks].as_flattened());
    let y2 = blocks_to_field_element_bls12_381_coordinate(
        input_data[1][coord_blocks..2 * coord_blocks].as_flattened(),
    );
    let z2 = blocks_to_field_element_bls12_381_coordinate(
        input_data[1][2 * coord_blocks..].as_flattened(),
    );

    let b3 = blstrs::Fp::from(BLS12_381_B3);

    let (x3, y3, z3) = ec_add_proj_impl_a0(x1, y1, z1, x2, y2, z2, b3);

    let mut output = [[0u8; BLOCK_SIZE]; BLOCKS];
    field_element_to_blocks_bls12_381_coordinate(&x3, &mut output[..coord_blocks]);
    field_element_to_blocks_bls12_381_coordinate(&y3, &mut output[coord_blocks..2 * coord_blocks]);
    field_element_to_blocks_bls12_381_coordinate(&z3, &mut output[2 * coord_blocks..]);

    output
}

#[inline(always)]
fn ec_double_proj_bls12_381<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    input_data: [[u8; BLOCK_SIZE]; BLOCKS],
) -> [[u8; BLOCK_SIZE]; BLOCKS] {
    let coord_blocks = BLOCKS / 3;

    let x1 =
        blocks_to_field_element_bls12_381_coordinate(input_data[..coord_blocks].as_flattened());
    let y1 = blocks_to_field_element_bls12_381_coordinate(
        input_data[coord_blocks..2 * coord_blocks].as_flattened(),
    );
    let z1 =
        blocks_to_field_element_bls12_381_coordinate(input_data[2 * coord_blocks..].as_flattened());

    let b3 = blstrs::Fp::from(BLS12_381_B3);

    let (x3, y3, z3) = ec_double_proj_impl_a0(x1, y1, z1, b3);

    let mut output = [[0u8; BLOCK_SIZE]; BLOCKS];
    field_element_to_blocks_bls12_381_coordinate(&x3, &mut output[..coord_blocks]);
    field_element_to_blocks_bls12_381_coordinate(&y3, &mut output[coord_blocks..2 * coord_blocks]);
    field_element_to_blocks_bls12_381_coordinate(&z3, &mut output[2 * coord_blocks..]);

    output
}

#[cfg(test)]
mod tests {
    use halo2curves_axiom::ff::Field;

    use super::*;

    /// Test projective addition with secp256k1 using k256 crate
    #[test]
    fn test_secp256k1_projective_add() {
        use k256::FieldElement;

        // secp256k1 generator (big-endian)
        let gx = FieldElement::from_bytes(
            &hex_literal::hex!("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798")
                .into(),
        )
        .unwrap();
        let gy = FieldElement::from_bytes(
            &hex_literal::hex!("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8")
                .into(),
        )
        .unwrap();
        let one = FieldElement::ONE;
        let b3 = FieldElement::from(K256_B3);

        // Compute 2G using doubling (with normalize)
        let (x2g, y2g, z2g) = ec_double_proj_k256(gx, gy, one, b3);

        // Compute 3G = G + 2G (with normalize)
        let (x3g, y3g, z3g) = ec_add_proj_k256(gx, gy, one, x2g, y2g, z2g, b3);

        // Convert to affine: x = X/Z, y = Y/Z
        let z3g_inv = z3g.invert().unwrap();
        let x3g_affine = (x3g * z3g_inv).normalize();
        let y3g_affine = (y3g * z3g_inv).normalize();

        // Verify point is on curve: y² = x³ + 7
        let y_sq = y3g_affine.square().normalize();
        let x_cubed = (x3g_affine.square() * x3g_affine).normalize();
        let rhs = (x_cubed + FieldElement::from(7u64)).normalize();
        assert_eq!(y_sq, rhs, "3G should be on secp256k1 curve");
    }

    /// Test projective doubling with secp256k1 using k256 crate
    #[test]
    fn test_secp256k1_projective_double() {
        use k256::FieldElement;

        let gx = FieldElement::from_bytes(
            &hex_literal::hex!("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798")
                .into(),
        )
        .unwrap();
        let gy = FieldElement::from_bytes(
            &hex_literal::hex!("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8")
                .into(),
        )
        .unwrap();
        let one = FieldElement::ONE;
        let b3 = FieldElement::from(K256_B3);

        // Compute 2G (with normalize)
        let (x2g, y2g, z2g) = ec_double_proj_k256(gx, gy, one, b3);

        // Convert to affine
        let z2g_inv = z2g.invert().unwrap();
        let x2g_affine = (x2g * z2g_inv).normalize();
        let y2g_affine = (y2g * z2g_inv).normalize();

        // Verify 2G is on curve
        let y_sq = y2g_affine.square().normalize();
        let x_cubed = (x2g_affine.square() * x2g_affine).normalize();
        let rhs = (x_cubed + FieldElement::from(7u64)).normalize();
        assert_eq!(y_sq, rhs, "2G should be on secp256k1 curve");
    }

    /// Test k256 projective addition with two distinct points
    #[test]
    fn test_secp256k1_k256_add_two_points() {
        use k256::FieldElement;

        let gx = FieldElement::from_bytes(
            &hex_literal::hex!("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798")
                .into(),
        )
        .unwrap();
        let gy = FieldElement::from_bytes(
            &hex_literal::hex!("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8")
                .into(),
        )
        .unwrap();
        let one = FieldElement::ONE;
        let b3 = FieldElement::from(K256_B3);

        // Compute 2G using doubling
        let (x2g, y2g, z2g) = ec_double_proj_k256(gx, gy, one, b3);

        // Compute 3G = G + 2G
        let (x3g, y3g, z3g) = ec_add_proj_k256(gx, gy, one, x2g, y2g, z2g, b3);

        // Convert to affine
        let z3g_inv = z3g.invert().unwrap();
        let x3g_affine = (x3g * z3g_inv).normalize();
        let y3g_affine = (y3g * z3g_inv).normalize();

        // Verify 3G is on curve: y² = x³ + 7
        let y_sq = y3g_affine.square().normalize();
        let x_cubed = (x3g_affine.square() * x3g_affine).normalize();
        let rhs = (x_cubed + FieldElement::from(7u64)).normalize();
        assert_eq!(y_sq, rhs, "3G should be on secp256k1 curve");
    }

    /// Test BN254 projective operations
    #[test]
    fn test_bn254_projective_double() {
        use halo2curves_axiom::bn256::Fq;

        // BN254 generator
        let gx = Fq::ONE;
        let gy = Fq::from(2u64);
        let one = Fq::ONE;
        let b3 = Fq::from(BN254_B3);

        // Compute 2G
        let (x2g, y2g, z2g) = ec_double_proj_impl_a0(gx, gy, one, b3);

        // Convert to affine
        let z2g_inv = z2g.invert().unwrap();
        let x2g_affine = x2g * z2g_inv;
        let y2g_affine = y2g * z2g_inv;

        // Verify 2G is on curve: y² = x³ + 3
        let y_sq = y2g_affine.square();
        let x_cubed = x2g_affine.square() * x2g_affine;
        let rhs = x_cubed + Fq::from(3u64);
        assert_eq!(y_sq, rhs, "2G should be on BN254 curve");
    }

    /// Test P256 (general a) projective operations - verify point is on curve
    #[test]
    fn test_p256_projective_double() {
        use halo2curves_axiom::{
            group::Curve,
            secp256r1::{Fp, Secp256r1},
        };

        // Use the generator point from halo2curves
        let gen = Secp256r1::generator();
        let gen_affine = gen.to_affine();
        let x = gen_affine.x;
        let y = gen_affine.y;

        let a = -Fp::from(P256_NEG_A);
        let b3 = get_p256_b3::<Fp>();

        // P256 b coefficient
        let b_bytes: [u8; 32] =
            hex_literal::hex!("5ac635d8aa3a93e7b3ebbd55769886bc651d06b0cc53b0f63bce3c3e27d2604b");
        let mut le_bytes = b_bytes;
        le_bytes.reverse();
        let b = Fp::from_repr(le_bytes.into()).unwrap();

        // Compute 2G using projective doubling
        let (x2_proj, y2_proj, z2_proj) = ec_double_proj_impl_general(x, y, Fp::ONE, a, b3);
        let z2_inv = z2_proj.invert().unwrap();
        let x2_affine = x2_proj * z2_inv;
        let y2_affine = y2_proj * z2_inv;

        // Verify 2G is on curve: y² = x³ + ax + b
        let y_sq = y2_affine.square();
        let x_cubed = x2_affine.square() * x2_affine;
        let rhs = x_cubed + a * x2_affine + b;
        assert_eq!(y_sq, rhs, "2G should be on P256 curve");
    }
}
