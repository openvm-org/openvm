use halo2curves_axiom::ff::PrimeField;
use num_bigint::BigUint;
use num_traits::{FromPrimitive, Num};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CurveType {
    K256 = 0,
    P256 = 1,
    BN254 = 2,
    BLS12_381 = 3,
    Generic = 4,
}

const K256_A: i64 = 0;
const P256_A: i64 = -3;
const BN254_A: i64 = 0;
const BLS12_381_A: i64 = 0;

fn get_modulus_as_bigint<F: PrimeField>() -> BigUint {
    BigUint::from_str_radix(F::MODULUS.trim_start_matches("0x"), 16).unwrap()
}

pub fn get_curve_type_from_modulus(modulus: &BigUint) -> CurveType {
    if modulus == &get_modulus_as_bigint::<halo2curves_axiom::secq256k1::Fq>() {
        return CurveType::K256;
    }

    if modulus == &get_modulus_as_bigint::<halo2curves_axiom::secp256r1::Fq>() {
        return CurveType::P256;
    }

    if modulus == &get_modulus_as_bigint::<halo2curves_axiom::bn256::Fq>() {
        return CurveType::BN254;
    }

    if modulus == &get_modulus_as_bigint::<halo2curves_axiom::bls12_381::Fq>() {
        return CurveType::BLS12_381;
    }

    CurveType::Generic
}

pub fn get_curve_type(modulus: &BigUint, a_coeff: &BigUint) -> CurveType {
    if modulus == &get_modulus_as_bigint::<halo2curves_axiom::secq256k1::Fq>()
        && a_coeff == &BigUint::from_i64(K256_A).unwrap()
    {
        return CurveType::K256;
    }

    if modulus == &get_modulus_as_bigint::<halo2curves_axiom::secp256r1::Fq>()
        && a_coeff == &BigUint::from_i64(P256_A).unwrap()
    {
        return CurveType::P256;
    }

    if modulus == &get_modulus_as_bigint::<halo2curves_axiom::bn256::Fq>()
        && a_coeff == &BigUint::from_i64(BN254_A).unwrap()
    {
        return CurveType::BN254;
    }

    if modulus == &get_modulus_as_bigint::<halo2curves_axiom::bls12_381::Fq>()
        && a_coeff == &BigUint::from_i64(BLS12_381_A).unwrap()
    {
        return CurveType::BLS12_381;
    }

    CurveType::Generic
}

#[inline(always)]
pub fn ec_add_ne_impl<F: PrimeField, const A: i64>(x1: F, y1: F, x2: F, y2: F) -> (F, F) {
    // Calculate lambda = (y2 - y1) / (x2 - x1)
    let lambda = (y2 - y1) * (x2 - x1).invert().unwrap();

    // Calculate x3 = lambda^2 - x1 - x2
    let x3 = lambda.square() - x1 - x2;

    // Calculate y3 = lambda * (x1 - x3) - y1
    let y3 = lambda * (x1 - x3) - y1;

    (x3, y3)
}

#[inline(always)]
pub fn ec_double_impl<F: PrimeField, const A: i64>(x1: F, y1: F) -> (F, F) {
    // Calculate lambda based on curve coefficient 'a'
    let x1_squared = x1.square();
    let three_x1_squared = x1_squared + x1_squared.double();
    let two_y1 = y1.double();

    let lambda = if A == 0 {
        // For a = 0: lambda = (3 * x1^2) / (2 * y1)
        three_x1_squared * two_y1.invert().unwrap()
    } else {
        // For a = -3: lambda = (3 * x1^2 - 3) / (2 * y1)
        let three = F::from(3u64);
        (three_x1_squared - three) * two_y1.invert().unwrap()
    };

    // Calculate x3 = lambda^2 - 2 * x1
    let x3 = lambda.square() - x1.double();

    // Calculate y3 = lambda * (x1 - x3) - y1
    let y3 = lambda * (x1 - x3) - y1;

    (x3, y3)
}

#[inline(always)]
pub fn ec_add_ne<const CURVE: usize, const BLOCKS: usize, const BLOCK_SIZE: usize>(
    input_data: [[[u8; BLOCK_SIZE]; BLOCKS]; 2],
) -> [[u8; BLOCK_SIZE]; BLOCKS] {
    match CURVE {
        0 => {
            // K256
            let x1 = blocks_to_field_element::<halo2curves_axiom::secq256k1::Fq>(
                input_data[0][..BLOCKS / 2].as_flattened(),
            );
            let y1 = blocks_to_field_element::<halo2curves_axiom::secq256k1::Fq>(
                input_data[0][BLOCKS / 2..].as_flattened(),
            );
            let x2 = blocks_to_field_element::<halo2curves_axiom::secq256k1::Fq>(
                input_data[1][..BLOCKS / 2].as_flattened(),
            );
            let y2 = blocks_to_field_element::<halo2curves_axiom::secq256k1::Fq>(
                input_data[1][BLOCKS / 2..].as_flattened(),
            );

            let (x3, y3) =
                ec_add_ne_impl::<halo2curves_axiom::secq256k1::Fq, K256_A>(x1, y1, x2, y2);

            let mut output = [[0u8; BLOCK_SIZE]; BLOCKS];
            field_element_to_blocks::<halo2curves_axiom::secq256k1::Fq, BLOCKS, BLOCK_SIZE>(
                &x3,
                &mut output,
                0,
            );
            field_element_to_blocks::<halo2curves_axiom::secq256k1::Fq, BLOCKS, BLOCK_SIZE>(
                &y3,
                &mut output,
                BLOCKS / 2,
            );
            output
        }
        1 => {
            // P256
            let x1 = blocks_to_field_element::<halo2curves_axiom::secp256r1::Fq>(
                input_data[0][..BLOCKS / 2].as_flattened(),
            );
            let y1 = blocks_to_field_element::<halo2curves_axiom::secp256r1::Fq>(
                input_data[0][BLOCKS / 2..].as_flattened(),
            );
            let x2 = blocks_to_field_element::<halo2curves_axiom::secp256r1::Fq>(
                input_data[1][..BLOCKS / 2].as_flattened(),
            );
            let y2 = blocks_to_field_element::<halo2curves_axiom::secp256r1::Fq>(
                input_data[1][BLOCKS / 2..].as_flattened(),
            );

            let (x3, y3) =
                ec_add_ne_impl::<halo2curves_axiom::secp256r1::Fq, P256_A>(x1, y1, x2, y2);

            let mut output = [[0u8; BLOCK_SIZE]; BLOCKS];
            field_element_to_blocks::<halo2curves_axiom::secp256r1::Fq, BLOCKS, BLOCK_SIZE>(
                &x3,
                &mut output,
                0,
            );
            field_element_to_blocks::<halo2curves_axiom::secp256r1::Fq, BLOCKS, BLOCK_SIZE>(
                &y3,
                &mut output,
                BLOCKS / 2,
            );
            output
        }
        2 => {
            // BN254
            let x1 = blocks_to_field_element::<halo2curves_axiom::bn256::Fq>(
                input_data[0][..BLOCKS / 2].as_flattened(),
            );
            let y1 = blocks_to_field_element::<halo2curves_axiom::bn256::Fq>(
                input_data[0][BLOCKS / 2..].as_flattened(),
            );
            let x2 = blocks_to_field_element::<halo2curves_axiom::bn256::Fq>(
                input_data[1][..BLOCKS / 2].as_flattened(),
            );
            let y2 = blocks_to_field_element::<halo2curves_axiom::bn256::Fq>(
                input_data[1][BLOCKS / 2..].as_flattened(),
            );

            let (x3, y3) = ec_add_ne_impl::<halo2curves_axiom::bn256::Fq, BN254_A>(x1, y1, x2, y2);

            let mut output = [[0u8; BLOCK_SIZE]; BLOCKS];
            field_element_to_blocks::<halo2curves_axiom::bn256::Fq, BLOCKS, BLOCK_SIZE>(
                &x3,
                &mut output,
                0,
            );
            field_element_to_blocks::<halo2curves_axiom::bn256::Fq, BLOCKS, BLOCK_SIZE>(
                &y3,
                &mut output,
                BLOCKS / 2,
            );
            output
        }
        3 => {
            // Extract coordinates
            let x1 = blocks_to_field_element_bls12_381(input_data[0][..BLOCKS / 2].as_flattened());
            let y1 = blocks_to_field_element_bls12_381(input_data[0][BLOCKS / 2..].as_flattened());
            let x2 = blocks_to_field_element_bls12_381(input_data[1][..BLOCKS / 2].as_flattened());
            let y2 = blocks_to_field_element_bls12_381(input_data[1][BLOCKS / 2..].as_flattened());

            let (x3, y3) =
                ec_add_ne_impl::<halo2curves_axiom::bls12_381::Fq, BLS12_381_A>(x1, y1, x2, y2);

            // Final output
            let mut output = [[0u8; BLOCK_SIZE]; BLOCKS];
            field_element_to_blocks_bls12_381(&x3, &mut output, 0);
            field_element_to_blocks_bls12_381(&y3, &mut output, BLOCKS / 2);
            output
        }
        _ => panic!("Unsupported curve type: {}", CURVE),
    }
}

/// Dispatch elliptic curve point doubling based on const generic curve type
#[inline(always)]
pub fn ec_double<const CURVE: usize, const BLOCKS: usize, const BLOCK_SIZE: usize>(
    input_data: [[u8; BLOCK_SIZE]; BLOCKS],
) -> [[u8; BLOCK_SIZE]; BLOCKS] {
    match CURVE {
        0 => {
            // K256
            let x1 = blocks_to_field_element::<halo2curves_axiom::secq256k1::Fq>(
                input_data[..BLOCKS / 2].as_flattened(),
            );
            let y1 = blocks_to_field_element::<halo2curves_axiom::secq256k1::Fq>(
                input_data[BLOCKS / 2..].as_flattened(),
            );

            let (x3, y3) = ec_double_impl::<halo2curves_axiom::secq256k1::Fq, K256_A>(x1, y1);

            let mut output = [[0u8; BLOCK_SIZE]; BLOCKS];
            field_element_to_blocks::<halo2curves_axiom::secq256k1::Fq, BLOCKS, BLOCK_SIZE>(
                &x3,
                &mut output,
                0,
            );
            field_element_to_blocks::<halo2curves_axiom::secq256k1::Fq, BLOCKS, BLOCK_SIZE>(
                &y3,
                &mut output,
                BLOCKS / 2,
            );
            output
        }
        1 => {
            // P256
            let x1 = blocks_to_field_element::<halo2curves_axiom::secp256r1::Fq>(
                input_data[..BLOCKS / 2].as_flattened(),
            );
            let y1 = blocks_to_field_element::<halo2curves_axiom::secp256r1::Fq>(
                input_data[BLOCKS / 2..].as_flattened(),
            );

            let (x3, y3) = ec_double_impl::<halo2curves_axiom::secp256r1::Fq, P256_A>(x1, y1);

            let mut output = [[0u8; BLOCK_SIZE]; BLOCKS];
            field_element_to_blocks::<halo2curves_axiom::secp256r1::Fq, BLOCKS, BLOCK_SIZE>(
                &x3,
                &mut output,
                0,
            );
            field_element_to_blocks::<halo2curves_axiom::secp256r1::Fq, BLOCKS, BLOCK_SIZE>(
                &y3,
                &mut output,
                BLOCKS / 2,
            );
            output
        }
        2 => {
            // BN254
            let x1 = blocks_to_field_element::<halo2curves_axiom::bn256::Fq>(
                input_data[..BLOCKS / 2].as_flattened(),
            );
            let y1 = blocks_to_field_element::<halo2curves_axiom::bn256::Fq>(
                input_data[BLOCKS / 2..].as_flattened(),
            );

            let (x3, y3) = ec_double_impl::<halo2curves_axiom::bn256::Fq, BN254_A>(x1, y1);

            let mut output = [[0u8; BLOCK_SIZE]; BLOCKS];
            field_element_to_blocks::<halo2curves_axiom::bn256::Fq, BLOCKS, BLOCK_SIZE>(
                &x3,
                &mut output,
                0,
            );
            field_element_to_blocks::<halo2curves_axiom::bn256::Fq, BLOCKS, BLOCK_SIZE>(
                &y3,
                &mut output,
                BLOCKS / 2,
            );
            output
        }
        3 => {
            // Extract coordinates
            let x1 = blocks_to_field_element_bls12_381(input_data[..BLOCKS / 2].as_flattened());
            let y1 = blocks_to_field_element_bls12_381(input_data[BLOCKS / 2..].as_flattened());

            let (x3, y3) = ec_double_impl::<halo2curves_axiom::bls12_381::Fq, BLS12_381_A>(x1, y1);

            // Final output
            let mut output = [[0u8; BLOCK_SIZE]; BLOCKS];
            field_element_to_blocks_bls12_381(&x3, &mut output, 0);
            field_element_to_blocks_bls12_381(&y3, &mut output, BLOCKS / 2);
            output
        }
        _ => panic!("Unsupported curve type: {}", CURVE),
    }
}
#[inline(always)]
fn blocks_to_field_element<F: PrimeField<Repr = [u8; 32]>>(blocks: &[u8]) -> F {
    let mut bytes = [0u8; 32];
    let len = blocks.len().min(32);
    bytes[..len].copy_from_slice(&blocks[..len]);

    F::from_repr_vartime(bytes).unwrap()
}

#[inline(always)]
fn field_element_to_blocks<
    F: PrimeField<Repr = [u8; 32]>,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
>(
    field_element: &F,
    output: &mut [[u8; BLOCK_SIZE]; BLOCKS],
    start_block: usize,
) {
    let bytes = field_element.to_repr();
    let mut byte_idx = 0;

    for block in output.iter_mut().skip(start_block) {
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

#[inline(always)]
fn blocks_to_field_element_bls12_381(blocks: &[u8]) -> halo2curves_axiom::bls12_381::Fq {
    let mut bytes = [0u8; 48];
    let len = blocks.len().min(48);
    bytes[..len].copy_from_slice(&blocks[..len]);

    halo2curves_axiom::bls12_381::Fq::from_bytes(&bytes).unwrap()
}

#[inline(always)]
fn field_element_to_blocks_bls12_381<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    field_element: &halo2curves_axiom::bls12_381::Fq,
    output: &mut [[u8; BLOCK_SIZE]; BLOCKS],
    start_block: usize,
) {
    let bytes = field_element.to_bytes();
    let mut byte_idx = 0;

    for block in output.iter_mut().skip(start_block) {
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
