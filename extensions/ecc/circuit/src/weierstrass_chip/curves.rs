use halo2curves_axiom::ff::{Field, PrimeField};
use num_bigint::BigUint;
use num_traits::Num;
use once_cell::sync::Lazy;
use openvm_algebra_circuit::fields::{
    blocks_to_field_element, blocks_to_field_element_bls12_381_coordinate, field_element_to_blocks,
    field_element_to_blocks_bls12_381_coordinate, FieldType,
};
use openvm_circuit::arch::MEMORY_BLOCK_BYTES;

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

static K256_COORD_MODULUS: Lazy<BigUint> =
    Lazy::new(get_modulus_as_bigint::<halo2curves_axiom::secq256k1::Fq>);
static P256_COORD_MODULUS: Lazy<BigUint> =
    Lazy::new(get_modulus_as_bigint::<halo2curves_axiom::secp256r1::Fp>);
static BN254_COORD_MODULUS: Lazy<BigUint> =
    Lazy::new(get_modulus_as_bigint::<halo2curves_axiom::bn256::Fq>);
static BLS12_381_COORD_MODULUS: Lazy<BigUint> =
    Lazy::new(get_modulus_as_bigint::<halo2curves_axiom::bls12_381::Fq>);
static ZERO_A_COEFF: Lazy<BigUint> = Lazy::new(|| BigUint::ZERO);
static P256_A_COEFF: Lazy<BigUint> = Lazy::new(|| {
    BigUint::from_bytes_le(&(-halo2curves_axiom::secp256r1::Fp::from(P256_NEG_A)).to_bytes())
});

impl CurveType {
    pub fn coordinate_modulus(self) -> &'static BigUint {
        match self {
            Self::K256 => &K256_COORD_MODULUS,
            Self::P256 => &P256_COORD_MODULUS,
            Self::BN254 => &BN254_COORD_MODULUS,
            Self::BLS12_381 => &BLS12_381_COORD_MODULUS,
        }
    }

    pub fn a_coefficient(self) -> &'static BigUint {
        match self {
            Self::P256 => &P256_A_COEFF,
            Self::K256 | Self::BN254 | Self::BLS12_381 => &ZERO_A_COEFF,
        }
    }

    #[cfg(feature = "rvr")]
    pub(crate) fn from_struct_name(struct_name: &str) -> Option<Self> {
        match struct_name {
            "Secp256k1Point" => Some(Self::K256),
            "P256Point" => Some(Self::P256),
            "Bn254G1Affine" => Some(Self::BN254),
            "Bls12_381G1Affine" => Some(Self::BLS12_381),
            _ => None,
        }
    }
}

pub(crate) fn get_curve_type(modulus: &BigUint, a_coeff: &BigUint) -> Option<CurveType> {
    [
        CurveType::K256,
        CurveType::P256,
        CurveType::BN254,
        CurveType::BLS12_381,
    ]
    .into_iter()
    .find(|curve| modulus == curve.coordinate_modulus() && a_coeff == curve.a_coefficient())
}

#[inline(always)]
pub fn ec_add_ne<const FIELD_TYPE: u8, const BLOCKS: usize>(
    input_data: [[[u8; MEMORY_BLOCK_BYTES]; BLOCKS]; 2],
) -> [[u8; MEMORY_BLOCK_BYTES]; BLOCKS] {
    match FIELD_TYPE {
        x if x == FieldType::K256Coordinate as u8 => {
            ec_add_ne_256bit::<halo2curves_axiom::secq256k1::Fq, BLOCKS>(input_data)
        }
        x if x == FieldType::P256Coordinate as u8 => {
            ec_add_ne_256bit::<halo2curves_axiom::secp256r1::Fp, BLOCKS>(input_data)
        }
        x if x == FieldType::BN254Coordinate as u8 => {
            ec_add_ne_256bit::<halo2curves_axiom::bn256::Fq, BLOCKS>(input_data)
        }
        x if x == FieldType::BLS12_381Coordinate as u8 => ec_add_ne_bls12_381::<BLOCKS>(input_data),
        _ => panic!("Unsupported field type: {FIELD_TYPE}"),
    }
}

/// Dispatch elliptic curve point doubling based on const generic curve type
#[inline(always)]
pub fn ec_double<const CURVE_TYPE: u8, const BLOCKS: usize>(
    input_data: [[u8; MEMORY_BLOCK_BYTES]; BLOCKS],
) -> [[u8; MEMORY_BLOCK_BYTES]; BLOCKS] {
    match CURVE_TYPE {
        x if x == CurveType::K256 as u8 => {
            ec_double_256bit::<halo2curves_axiom::secq256k1::Fq, 0, BLOCKS>(input_data)
        }
        x if x == CurveType::P256 as u8 => {
            ec_double_256bit::<halo2curves_axiom::secp256r1::Fp, P256_NEG_A, BLOCKS>(input_data)
        }
        x if x == CurveType::BN254 as u8 => {
            ec_double_256bit::<halo2curves_axiom::bn256::Fq, 0, BLOCKS>(input_data)
        }
        x if x == CurveType::BLS12_381 as u8 => ec_double_bls12_381::<BLOCKS>(input_data),
        _ => panic!("Unsupported curve type: {CURVE_TYPE}"),
    }
}

#[inline(always)]
fn ec_add_ne_256bit<F: PrimeField<Repr = [u8; 32]>, const BLOCKS: usize>(
    input_data: [[[u8; MEMORY_BLOCK_BYTES]; BLOCKS]; 2],
) -> [[u8; MEMORY_BLOCK_BYTES]; BLOCKS] {
    let x1 = blocks_to_field_element::<F>(input_data[0][..BLOCKS / 2].as_flattened());
    let y1 = blocks_to_field_element::<F>(input_data[0][BLOCKS / 2..].as_flattened());
    let x2 = blocks_to_field_element::<F>(input_data[1][..BLOCKS / 2].as_flattened());
    let y2 = blocks_to_field_element::<F>(input_data[1][BLOCKS / 2..].as_flattened());

    let (x3, y3) = ec_add_ne_impl::<F>(x1, y1, x2, y2);

    let mut output = [[0u8; MEMORY_BLOCK_BYTES]; BLOCKS];
    field_element_to_blocks::<F>(&x3, &mut output[..BLOCKS / 2]);
    field_element_to_blocks::<F>(&y3, &mut output[BLOCKS / 2..]);
    output
}

#[inline(always)]
fn ec_double_256bit<F: PrimeField<Repr = [u8; 32]>, const NEG_A: u64, const BLOCKS: usize>(
    input_data: [[u8; MEMORY_BLOCK_BYTES]; BLOCKS],
) -> [[u8; MEMORY_BLOCK_BYTES]; BLOCKS] {
    let x1 = blocks_to_field_element::<F>(input_data[..BLOCKS / 2].as_flattened());
    let y1 = blocks_to_field_element::<F>(input_data[BLOCKS / 2..].as_flattened());

    let (x3, y3) = ec_double_impl::<F, NEG_A>(x1, y1);

    let mut output = [[0u8; MEMORY_BLOCK_BYTES]; BLOCKS];
    field_element_to_blocks::<F>(&x3, &mut output[..BLOCKS / 2]);
    field_element_to_blocks::<F>(&y3, &mut output[BLOCKS / 2..]);
    output
}

#[inline(always)]
fn ec_add_ne_bls12_381<const BLOCKS: usize>(
    input_data: [[[u8; MEMORY_BLOCK_BYTES]; BLOCKS]; 2],
) -> [[u8; MEMORY_BLOCK_BYTES]; BLOCKS] {
    // Extract coordinates
    let x1 =
        blocks_to_field_element_bls12_381_coordinate(input_data[0][..BLOCKS / 2].as_flattened());
    let y1 =
        blocks_to_field_element_bls12_381_coordinate(input_data[0][BLOCKS / 2..].as_flattened());
    let x2 =
        blocks_to_field_element_bls12_381_coordinate(input_data[1][..BLOCKS / 2].as_flattened());
    let y2 =
        blocks_to_field_element_bls12_381_coordinate(input_data[1][BLOCKS / 2..].as_flattened());

    let (x3, y3) = ec_add_ne_impl::<blstrs::Fp>(x1, y1, x2, y2);

    // Final output
    let mut output = [[0u8; MEMORY_BLOCK_BYTES]; BLOCKS];
    field_element_to_blocks_bls12_381_coordinate(&x3, &mut output[..BLOCKS / 2]);
    field_element_to_blocks_bls12_381_coordinate(&y3, &mut output[BLOCKS / 2..]);
    output
}

#[inline(always)]
fn ec_double_bls12_381<const BLOCKS: usize>(
    input_data: [[u8; MEMORY_BLOCK_BYTES]; BLOCKS],
) -> [[u8; MEMORY_BLOCK_BYTES]; BLOCKS] {
    // Extract coordinates
    let x1 = blocks_to_field_element_bls12_381_coordinate(input_data[..BLOCKS / 2].as_flattened());
    let y1 = blocks_to_field_element_bls12_381_coordinate(input_data[BLOCKS / 2..].as_flattened());

    let (x3, y3) = ec_double_impl::<blstrs::Fp, 0>(x1, y1);

    // Final output
    let mut output = [[0u8; MEMORY_BLOCK_BYTES]; BLOCKS];
    field_element_to_blocks_bls12_381_coordinate(&x3, &mut output[..BLOCKS / 2]);
    field_element_to_blocks_bls12_381_coordinate(&y3, &mut output[BLOCKS / 2..]);
    output
}

#[inline(always)]
pub fn ec_add_ne_impl<F: Field>(x1: F, y1: F, x2: F, y2: F) -> (F, F) {
    // Calculate lambda = (y2 - y1) / (x2 - x1)
    let lambda = (y2 - y1) * (x2 - x1).invert().unwrap();

    // Calculate x3 = lambda^2 - x1 - x2
    let x3 = lambda.square() - x1 - x2;

    // Calculate y3 = lambda * (x1 - x3) - y1
    let y3 = lambda * (x1 - x3) - y1;

    (x3, y3)
}

#[inline(always)]
pub fn ec_double_impl<F: Field + From<u64>, const NEG_A: u64>(x1: F, y1: F) -> (F, F) {
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
