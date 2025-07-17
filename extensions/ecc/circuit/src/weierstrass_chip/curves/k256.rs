use crypto_bigint::Encoding;
use halo2curves_axiom::secq256k1::Fq;
use k256::{
    elliptic_curve::{FieldBytesEncoding, PrimeField},
    FieldElement, Secp256k1, U256,
};
use num_bigint::BigUint;
use num_traits::Num;

pub fn modulus() -> BigUint {
    BigUint::from_str_radix(Fq::MODULUS.trim_start_matches("0x"), 16).unwrap()
}

pub fn a() -> BigUint {
    BigUint::ZERO
}

#[inline(always)]
pub fn ec_add_ne_k256<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    input_data: [[[u8; BLOCK_SIZE]; BLOCKS]; 2],
) -> [[u8; BLOCK_SIZE]; BLOCKS] {
    // Extract coordinates
    let x1 = blocks_to_field_element(input_data[0][..BLOCKS / 2].as_flattened());
    let y1 = blocks_to_field_element(input_data[0][BLOCKS / 2..].as_flattened());
    let x2 = blocks_to_field_element(input_data[1][..BLOCKS / 2].as_flattened());
    let y2 = blocks_to_field_element(input_data[1][BLOCKS / 2..].as_flattened());

    // Calculate lambda = (y2 - y1) / (x2 - x1)
    let y2_minus_y1 = (y2 - y1).normalize();
    let x2_minus_x1 = (x2 - x1).normalize();
    let lambda = y2_minus_y1 * x2_minus_x1.invert().unwrap();

    // Calculate x3 = lambda^2 - x1 - x2
    let lambda_squared = lambda.square();
    let x3 = (lambda_squared - x1 - x2).normalize();

    // Calculate y3 = lambda * (x1 - x3) - y1
    let x1_minus_x3 = x1 - x3;
    let y3 = (lambda * x1_minus_x3 - y1).normalize();

    // Final output
    let mut output = [[0u8; BLOCK_SIZE]; BLOCKS];
    field_element_to_blocks(&x3, &mut output, 0);
    field_element_to_blocks(&y3, &mut output, BLOCKS / 2);
    output
}

#[inline(always)]
pub fn ec_double_k256<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    input_data: [[u8; BLOCK_SIZE]; BLOCKS],
) -> [[u8; BLOCK_SIZE]; BLOCKS] {
    // Extract coordinates
    let x1 = blocks_to_field_element(input_data[..BLOCKS / 2].as_flattened());
    let y1 = blocks_to_field_element(input_data[BLOCKS / 2..].as_flattened());

    // Calculate lambda = (3 * x1^2) / (2 * y1)
    let x1_squared = x1.square();
    let three_x1_squared = (x1_squared + x1_squared.double()).normalize(); // 3 * x1^2
    let two_y1 = y1.double().normalize(); // 2 * y1
    let lambda = three_x1_squared * two_y1.invert().unwrap();

    // Calculate x3 = lambda^2 - 2 * x1
    let lambda_squared = lambda.square();
    let two_x1 = x1.double();
    let x3 = (lambda_squared - two_x1).normalize();

    // Calculate y3 = lambda * (x1 - x3) - y1
    let x1_minus_x3 = x1 - x3;
    let y3 = (lambda * x1_minus_x3 - y1).normalize();

    // Final output
    let mut output = [[0u8; BLOCK_SIZE]; BLOCKS];
    field_element_to_blocks(&x3, &mut output, 0);
    field_element_to_blocks(&y3, &mut output, BLOCKS / 2);
    output
}

#[inline(always)]
fn blocks_to_field_element(blocks: &[u8]) -> FieldElement {
    let mut bytes = [0u8; 32];
    let len = blocks.len().min(32);
    bytes[..len].copy_from_slice(&blocks[..len]);

    let num = U256::from_le_bytes(bytes);
    FieldElement::from_repr_vartime(<U256 as FieldBytesEncoding<Secp256k1>>::encode_field_bytes(
        &num,
    ))
    .unwrap()
}

#[inline(always)]
fn field_element_to_blocks<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    field_element: &FieldElement,
    output: &mut [[u8; BLOCK_SIZE]; BLOCKS],
    start_block: usize,
) {
    let bytes = field_element.to_bytes();
    let mut byte_iter = bytes.iter().rev();
    for block in output.iter_mut().skip(start_block) {
        for byte in block.iter_mut() {
            *byte = byte_iter.next().copied().unwrap_or(0);
        }
    }
}
