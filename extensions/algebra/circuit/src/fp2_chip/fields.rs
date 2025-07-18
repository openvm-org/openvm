use halo2curves_axiom::ff::PrimeField;
use num_bigint::BigUint;
use num_traits::Num;

use halo2curves_axiom::bls12_381::Fq2 as Bls12_381Fq2;
use halo2curves_axiom::bn256::Fq2 as Bn254Fq2;

use crate::modular_chip::fields::{
    blocks_to_field_element, blocks_to_field_element_bls12_381, field_element_to_blocks,
    field_element_to_blocks_bls12_381,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldType {
    BN254 = 0,
    BLS12_381 = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    Add,
    Sub,
    Mul,
    Div,
}

fn get_modulus_as_bigint<F: PrimeField>() -> BigUint {
    BigUint::from_str_radix(F::MODULUS.trim_start_matches("0x"), 16).unwrap()
}

pub fn get_field_type_from_modulus(modulus: &BigUint) -> Option<FieldType> {
    if modulus == &get_modulus_as_bigint::<halo2curves_axiom::bn256::Fq>() {
        return Some(FieldType::BN254);
    }

    if modulus == &get_modulus_as_bigint::<halo2curves_axiom::bls12_381::Fq>() {
        return Some(FieldType::BLS12_381);
    }

    None
}

#[inline(always)]
pub fn fp2_operation<
    const FIELD: u8,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const OP: u8,
>(
    input_data: [[[u8; BLOCK_SIZE]; BLOCKS]; 2],
) -> [[u8; BLOCK_SIZE]; BLOCKS] {
    match FIELD {
        x if x == FieldType::BN254 as u8 => {
            fp2_operation_bn254::<BLOCKS, BLOCK_SIZE, OP>(input_data)
        }
        x if x == FieldType::BLS12_381 as u8 => {
            fp2_operation_bls12_381::<BLOCKS, BLOCK_SIZE, OP>(input_data)
        }
        _ => panic!("Unsupported field type for Fp2: {}", FIELD),
    }
}

// Generic BN254 Fp2 operation
#[inline(always)]
fn fp2_operation_bn254<const BLOCKS: usize, const BLOCK_SIZE: usize, const OP: u8>(
    input_data: [[[u8; BLOCK_SIZE]; BLOCKS]; 2],
) -> [[u8; BLOCK_SIZE]; BLOCKS] {
    let a = blocks_to_fp2_bn254(input_data[0].as_ref());
    let b = blocks_to_fp2_bn254(input_data[1].as_ref());
    let c = match OP {
        x if x == Operation::Add as u8 => a + b,
        x if x == Operation::Sub as u8 => a - b,
        x if x == Operation::Mul as u8 => a * b,
        x if x == Operation::Div as u8 => a * b.invert().unwrap(),
        _ => panic!("Unsupported operation: {}", OP),
    };

    let mut output = [[0u8; BLOCK_SIZE]; BLOCKS];
    fp2_to_blocks_bn254(&c, &mut output);
    output
}

// Generic BLS12-381 Fp2 operation
#[inline(always)]
fn fp2_operation_bls12_381<const BLOCKS: usize, const BLOCK_SIZE: usize, const OP: u8>(
    input_data: [[[u8; BLOCK_SIZE]; BLOCKS]; 2],
) -> [[u8; BLOCK_SIZE]; BLOCKS] {
    let a = blocks_to_fp2_bls12_381(input_data[0].as_ref());
    let b = blocks_to_fp2_bls12_381(input_data[1].as_ref());
    let c = match OP {
        x if x == Operation::Add as u8 => a + b,
        x if x == Operation::Sub as u8 => a - b,
        x if x == Operation::Mul as u8 => a * b,
        x if x == Operation::Div as u8 => a * b.invert().unwrap(),
        _ => panic!("Unsupported operation: {}", OP),
    };

    let mut output = [[0u8; BLOCK_SIZE]; BLOCKS];
    fp2_to_blocks_bls12_381(&c, &mut output);
    output
}

// Helper functions for Fp2
#[inline(always)]
fn blocks_to_fp2_bn254<const BLOCK_SIZE: usize>(blocks: &[[u8; BLOCK_SIZE]]) -> Bn254Fq2 {
    let mid = blocks.len() / 2;
    let c0 = blocks_to_field_element::<halo2curves_axiom::bn256::Fq>(blocks[..mid].as_flattened());
    let c1 = blocks_to_field_element::<halo2curves_axiom::bn256::Fq>(blocks[mid..].as_flattened());
    Bn254Fq2 { c0, c1 }
}

#[inline(always)]
fn fp2_to_blocks_bn254<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    fp2: &Bn254Fq2,
    output: &mut [[u8; BLOCK_SIZE]; BLOCKS],
) {
    field_element_to_blocks::<halo2curves_axiom::bn256::Fq, BLOCKS, BLOCK_SIZE>(&fp2.c0, output);
    let mid = BLOCKS / 2;
    let mut temp = [[0u8; BLOCK_SIZE]; BLOCKS];
    field_element_to_blocks::<halo2curves_axiom::bn256::Fq, BLOCKS, BLOCK_SIZE>(&fp2.c1, &mut temp);
    output[mid..].copy_from_slice(&temp[..mid]);
}

#[inline(always)]
fn blocks_to_fp2_bls12_381<const BLOCK_SIZE: usize>(blocks: &[[u8; BLOCK_SIZE]]) -> Bls12_381Fq2 {
    let mid = blocks.len() / 2;
    let c0 = blocks_to_field_element_bls12_381(blocks[..mid].as_flattened());
    let c1 = blocks_to_field_element_bls12_381(blocks[mid..].as_flattened());
    Bls12_381Fq2 { c0, c1 }
}

#[inline(always)]
fn fp2_to_blocks_bls12_381<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    fp2: &Bls12_381Fq2,
    output: &mut [[u8; BLOCK_SIZE]; BLOCKS],
) {
    field_element_to_blocks_bls12_381(&fp2.c0, output);
    let mid = BLOCKS / 2;
    let mut temp = [[0u8; BLOCK_SIZE]; BLOCKS];
    field_element_to_blocks_bls12_381(&fp2.c1, &mut temp);
    output[mid..].copy_from_slice(&temp[..mid]);
}
