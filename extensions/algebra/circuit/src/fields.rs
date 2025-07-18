use halo2curves_axiom::ff::PrimeField;
use num_bigint::BigUint;
use num_traits::Num;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldType {
    K256 = 0,
    P256 = 1,
    BN254 = 2,
    BLS12_381 = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
}

fn get_modulus_as_bigint<F: PrimeField>() -> BigUint {
    BigUint::from_str_radix(F::MODULUS.trim_start_matches("0x"), 16).unwrap()
}

pub fn get_field_type_from_modulus(modulus: &BigUint) -> Option<FieldType> {
    if modulus == &get_modulus_as_bigint::<halo2curves_axiom::secq256k1::Fq>() {
        return Some(FieldType::K256);
    }

    if modulus == &get_modulus_as_bigint::<halo2curves_axiom::secp256r1::Fp>() {
        return Some(FieldType::P256);
    }

    if modulus == &get_modulus_as_bigint::<halo2curves_axiom::bn256::Fq>() {
        return Some(FieldType::BN254);
    }

    if modulus == &get_modulus_as_bigint::<halo2curves_axiom::bls12_381::Fq>() {
        return Some(FieldType::BLS12_381);
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
        x if x == FieldType::BN254 as u8 => {
            field_operation_256bit::<halo2curves_axiom::bn256::Fq, BLOCKS, BLOCK_SIZE, OP>(
                input_data,
            )
        }
        x if x == FieldType::BLS12_381 as u8 => {
            field_operation_bls12_381::<BLOCKS, BLOCK_SIZE, OP>(input_data)
        }
        _ => panic!("Unsupported field type: {}", FIELD),
    }
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

#[inline(always)]
fn blocks_to_fp2_bn254<const BLOCK_SIZE: usize>(
    blocks: &[[u8; BLOCK_SIZE]],
) -> halo2curves_axiom::bn256::Fq2 {
    let mid = blocks.len() / 2;
    let c0 = blocks_to_field_element::<halo2curves_axiom::bn256::Fq>(blocks[..mid].as_flattened());
    let c1 = blocks_to_field_element::<halo2curves_axiom::bn256::Fq>(blocks[mid..].as_flattened());
    halo2curves_axiom::bn256::Fq2 { c0, c1 }
}

#[inline(always)]
fn fp2_to_blocks_bn254<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    fp2: &halo2curves_axiom::bn256::Fq2,
    output: &mut [[u8; BLOCK_SIZE]; BLOCKS],
) {
    let mid = BLOCKS / 2;
    let mut temp_c0 = [[0u8; BLOCK_SIZE]; BLOCKS];
    let mut temp_c1 = [[0u8; BLOCK_SIZE]; BLOCKS];

    field_element_to_blocks::<halo2curves_axiom::bn256::Fq, BLOCKS, BLOCK_SIZE>(
        &fp2.c0,
        &mut temp_c0,
    );
    field_element_to_blocks::<halo2curves_axiom::bn256::Fq, BLOCKS, BLOCK_SIZE>(
        &fp2.c1,
        &mut temp_c1,
    );

    output[..mid].copy_from_slice(&temp_c0[..mid]);
    output[mid..].copy_from_slice(&temp_c1[..mid]);
}

#[inline(always)]
fn blocks_to_fp2_bls12_381<const BLOCK_SIZE: usize>(
    blocks: &[[u8; BLOCK_SIZE]],
) -> halo2curves_axiom::bls12_381::Fq2 {
    let mid = blocks.len() / 2;
    let c0 = blocks_to_field_element_bls12_381(blocks[..mid].as_flattened());
    let c1 = blocks_to_field_element_bls12_381(blocks[mid..].as_flattened());
    halo2curves_axiom::bls12_381::Fq2 { c0, c1 }
}

#[inline(always)]
fn fp2_to_blocks_bls12_381<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    fp2: &halo2curves_axiom::bls12_381::Fq2,
    output: &mut [[u8; BLOCK_SIZE]; BLOCKS],
) {
    let mid = BLOCKS / 2;
    let mut temp_c0 = [[0u8; BLOCK_SIZE]; BLOCKS];
    let mut temp_c1 = [[0u8; BLOCK_SIZE]; BLOCKS];

    field_element_to_blocks_bls12_381(&fp2.c0, &mut temp_c0);
    field_element_to_blocks_bls12_381(&fp2.c1, &mut temp_c1);

    output[..mid].copy_from_slice(&temp_c0[..mid]);
    output[mid..].copy_from_slice(&temp_c1[..mid]);
}
