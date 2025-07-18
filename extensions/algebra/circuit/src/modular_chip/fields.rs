use halo2curves_axiom::ff::PrimeField;
use num_bigint::BigUint;
use num_traits::Num;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldType {
    BN254 = 0,
    BLS12_381 = 1,
    K256 = 2,
    P256 = 3,
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

    if modulus == &get_modulus_as_bigint::<halo2curves_axiom::secq256k1::Fq>() {
        return Some(FieldType::K256);
    }

    if modulus == &get_modulus_as_bigint::<halo2curves_axiom::secp256r1::Fp>() {
        return Some(FieldType::P256);
    }

    None
}

#[inline(always)]
pub fn field_operation<
    const FIELD: u8,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const OP: u8,
>(
    input_data: [[[u8; BLOCK_SIZE]; BLOCKS]; 2],
) -> [[u8; BLOCK_SIZE]; BLOCKS] {
    match FIELD {
        x if x == FieldType::BN254 as u8 => {
            field_operation_256bit::<halo2curves_axiom::bn256::Fq, BLOCKS, BLOCK_SIZE, OP>(
                input_data,
            )
        }
        x if x == FieldType::K256 as u8 => {
            field_operation_256bit::<halo2curves_axiom::secq256k1::Fq, BLOCKS, BLOCK_SIZE, OP>(
                input_data,
            )
        }
        x if x == FieldType::P256 as u8 => {
            field_operation_256bit::<halo2curves_axiom::secp256r1::Fp, BLOCKS, BLOCK_SIZE, OP>(
                input_data,
            )
        }
        x if x == FieldType::BLS12_381 as u8 => {
            field_operation_bls12_381::<BLOCKS, BLOCK_SIZE, OP>(input_data)
        }
        _ => panic!("Unsupported field type: {}", FIELD),
    }
}

// Generic 256-bit field operation
#[inline(always)]
fn field_operation_256bit<
    F: PrimeField<Repr = [u8; 32]>,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const OP: u8,
>(
    input_data: [[[u8; BLOCK_SIZE]; BLOCKS]; 2],
) -> [[u8; BLOCK_SIZE]; BLOCKS] {
    let a = blocks_to_field_element::<F>(input_data[0].as_flattened());
    let b = blocks_to_field_element::<F>(input_data[1].as_flattened());
    let c = match OP {
        x if x == Operation::Add as u8 => a + b,
        x if x == Operation::Sub as u8 => a - b,
        x if x == Operation::Mul as u8 => a * b,
        x if x == Operation::Div as u8 => a * b.invert().unwrap(),
        _ => panic!("Unsupported operation: {}", OP),
    };

    let mut output = [[0u8; BLOCK_SIZE]; BLOCKS];
    field_element_to_blocks::<F, BLOCKS, BLOCK_SIZE>(&c, &mut output);
    output
}

// Generic BLS12-381 field operation
#[inline(always)]
fn field_operation_bls12_381<const BLOCKS: usize, const BLOCK_SIZE: usize, const OP: u8>(
    input_data: [[[u8; BLOCK_SIZE]; BLOCKS]; 2],
) -> [[u8; BLOCK_SIZE]; BLOCKS] {
    let a = blocks_to_field_element_bls12_381(input_data[0].as_flattened());
    let b = blocks_to_field_element_bls12_381(input_data[1].as_flattened());
    let c = match OP {
        x if x == Operation::Add as u8 => a + b,
        x if x == Operation::Sub as u8 => a - b,
        x if x == Operation::Mul as u8 => a * b,
        x if x == Operation::Div as u8 => a * b.invert().unwrap(),
        _ => panic!("Unsupported operation: {}", OP),
    };

    let mut output = [[0u8; BLOCK_SIZE]; BLOCKS];
    field_element_to_blocks_bls12_381(&c, &mut output);
    output
}

#[inline(always)]
pub fn blocks_to_field_element<F: PrimeField<Repr = [u8; 32]>>(blocks: &[u8]) -> F {
    let mut bytes = [0u8; 32];
    let len = blocks.len().min(32);
    bytes[..len].copy_from_slice(&blocks[..len]);

    F::from_repr_vartime(bytes).unwrap()
}

#[inline(always)]
pub fn field_element_to_blocks<
    F: PrimeField<Repr = [u8; 32]>,
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
>(
    field_element: &F,
    output: &mut [[u8; BLOCK_SIZE]; BLOCKS],
) {
    let bytes = field_element.to_repr();
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

#[inline(always)]
pub fn blocks_to_field_element_bls12_381(blocks: &[u8]) -> halo2curves_axiom::bls12_381::Fq {
    let mut bytes = [0u8; 48];
    let len = blocks.len().min(48);
    bytes[..len].copy_from_slice(&blocks[..len]);

    halo2curves_axiom::bls12_381::Fq::from_bytes(&bytes).unwrap()
}

#[inline(always)]
pub fn field_element_to_blocks_bls12_381<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    field_element: &halo2curves_axiom::bls12_381::Fq,
    output: &mut [[u8; BLOCK_SIZE]; BLOCKS],
) {
    let bytes = field_element.to_bytes();
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
