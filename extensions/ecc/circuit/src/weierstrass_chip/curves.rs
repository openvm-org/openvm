use halo2curves_axiom::ff::PrimeField;
use num_bigint::BigUint;
use num_traits::Num;
use openvm_algebra_circuit::fields::{blocks_to_field_element, field_element_to_blocks, FieldType};
#[cfg(not(feature = "aot"))]
use openvm_algebra_circuit::fields::{
    blocks_to_field_element_bls12_381_coordinate, field_element_to_blocks_bls12_381_coordinate,
};

#[cfg(feature = "aot")]
use ff::Field;
#[cfg(feature = "aot")]
type Bls12CoordField = blstrs::Fp;
#[cfg(not(feature = "aot"))]
type Bls12CoordField = halo2curves_axiom::bls12_381::Fq;

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

#[cfg(feature = "aot")]
#[inline(always)]
fn bls12_coord_from_blocks(blocks: &[u8]) -> Bls12CoordField {
    debug_assert!(blocks.len() == 48);
    let mut bytes = [0u8; 48];
    bytes[..blocks.len()].copy_from_slice(&blocks[..blocks.len()]);

    Bls12CoordField::from_bytes_le(&bytes).unwrap_or_else(|| {
        let reduced_bytes = reduce_bls12_bytes(&bytes);
        Bls12CoordField::from_bytes_le(&reduced_bytes)
            .unwrap_or_else(|| panic!("reduced bytes must be within the field modulus"))
    })
}

#[cfg(feature = "aot")]
#[inline(always)]
fn bls12_coord_to_blocks<const BLOCK_SIZE: usize>(
    field_element: &Bls12CoordField,
    output: &mut [[u8; BLOCK_SIZE]],
) {
    debug_assert!(output.len() * BLOCK_SIZE == 48);
    let bytes = field_element.to_bytes_le();
    let mut byte_idx = 0;

    for block in output.iter_mut() {
        for byte in block.iter_mut() {
            *byte = if byte_idx < bytes.len() {
                bytes[byte_idx]
            } else {
                0
            };
            byte_idx += 1;
        }
    }
}

#[cfg(feature = "aot")]
#[inline(always)]
fn reduce_bls12_bytes(bytes: &[u8; 48]) -> [u8; 48] {
    let modulus = get_modulus_as_bigint::<halo2curves_axiom::bls12_381::Fq>();
    let value = BigUint::from_bytes_le(bytes);
    let reduced = value % modulus;

    let mut reduced_bytes = [0u8; 48];
    let reduced_le_bytes = reduced.to_bytes_le();
    reduced_bytes[..reduced_le_bytes.len()].copy_from_slice(&reduced_le_bytes);
    reduced_bytes
}

#[cfg(feature = "aot")]
#[inline(always)]
fn ec_add_ne_impl_bls(
    x1: Bls12CoordField,
    y1: Bls12CoordField,
    x2: Bls12CoordField,
    y2: Bls12CoordField,
) -> (Bls12CoordField, Bls12CoordField) {
    let lambda = (y2 - y1) * (x2 - x1).invert().unwrap();
    let x3 = lambda * lambda - x1 - x2;
    let y3 = lambda * (x1 - x3) - y1;

    (x3, y3)
}

#[cfg(feature = "aot")]
#[inline(always)]
fn ec_double_impl_bls(
    x1: Bls12CoordField,
    y1: Bls12CoordField,
) -> (Bls12CoordField, Bls12CoordField) {
    let x1_squared = x1 * x1;
    let three_x1_squared = x1_squared + x1_squared + x1_squared;
    let two_y1 = y1 + y1;
    let lambda = three_x1_squared * two_y1.invert().unwrap();

    let x3 = lambda * lambda - (x1 + x1);
    let y3 = lambda * (x1 - x3) - y1;

    (x3, y3)
}

#[cfg(not(feature = "aot"))]
#[inline(always)]
fn bls12_coord_from_blocks(blocks: &[u8]) -> Bls12CoordField {
    blocks_to_field_element_bls12_381_coordinate(blocks)
}

#[cfg(not(feature = "aot"))]
#[inline(always)]
fn bls12_coord_to_blocks<const BLOCK_SIZE: usize>(
    field_element: &Bls12CoordField,
    output: &mut [[u8; BLOCK_SIZE]],
) {
    field_element_to_blocks_bls12_381_coordinate(field_element, output)
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
    let x1 = bls12_coord_from_blocks(input_data[0][..BLOCKS / 2].as_flattened());
    let y1 = bls12_coord_from_blocks(input_data[0][BLOCKS / 2..].as_flattened());
    let x2 = bls12_coord_from_blocks(input_data[1][..BLOCKS / 2].as_flattened());
    let y2 = bls12_coord_from_blocks(input_data[1][BLOCKS / 2..].as_flattened());

    #[cfg(feature = "aot")]
    let (x3, y3) = ec_add_ne_impl_bls(x1, y1, x2, y2);
    #[cfg(not(feature = "aot"))]
    let (x3, y3) = ec_add_ne_impl::<Bls12CoordField>(x1, y1, x2, y2);

    // Final output
    let mut output = [[0u8; BLOCK_SIZE]; BLOCKS];
    bls12_coord_to_blocks(&x3, &mut output[..BLOCKS / 2]);
    bls12_coord_to_blocks(&y3, &mut output[BLOCKS / 2..]);
    output
}

#[inline(always)]
fn ec_double_bls12_381<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    input_data: [[u8; BLOCK_SIZE]; BLOCKS],
) -> [[u8; BLOCK_SIZE]; BLOCKS] {
    // Extract coordinates
    let x1 = bls12_coord_from_blocks(input_data[..BLOCKS / 2].as_flattened());
    let y1 = bls12_coord_from_blocks(input_data[BLOCKS / 2..].as_flattened());

    #[cfg(feature = "aot")]
    let (x3, y3) = ec_double_impl_bls(x1, y1);
    #[cfg(not(feature = "aot"))]
    let (x3, y3) = ec_double_impl::<Bls12CoordField, 0>(x1, y1);

    // Final output
    let mut output = [[0u8; BLOCK_SIZE]; BLOCKS];
    bls12_coord_to_blocks(&x3, &mut output[..BLOCKS / 2]);
    bls12_coord_to_blocks(&y3, &mut output[BLOCKS / 2..]);
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
