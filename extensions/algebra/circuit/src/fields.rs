use halo2curves_axiom::ff::PrimeField;
use num_bigint::BigUint;
use num_traits::Num;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldType {
    K256Coordinate = 0,
    K256Scalar = 1,
    P256Coordinate = 2,
    P256Scalar = 3,
    BN254Coordinate = 4,
    BN254Scalar = 5,
    BLS12_381Coordinate = 6,
    BLS12_381Scalar = 7,
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

pub fn get_field_type(modulus: &BigUint) -> Option<FieldType> {
    if modulus == &get_modulus_as_bigint::<halo2curves_axiom::secq256k1::Fq>() {
        return Some(FieldType::K256Coordinate);
    }

    if modulus == &get_modulus_as_bigint::<halo2curves_axiom::secq256k1::Fp>() {
        return Some(FieldType::K256Scalar);
    }

    if modulus == &get_modulus_as_bigint::<halo2curves_axiom::secp256r1::Fp>() {
        return Some(FieldType::P256Coordinate);
    }

    if modulus == &get_modulus_as_bigint::<halo2curves_axiom::secp256r1::Fq>() {
        return Some(FieldType::P256Scalar);
    }

    if modulus == &get_modulus_as_bigint::<halo2curves_axiom::bn256::Fq>() {
        return Some(FieldType::BN254Coordinate);
    }

    if modulus == &get_modulus_as_bigint::<halo2curves_axiom::bn256::Fr>() {
        return Some(FieldType::BN254Scalar);
    }

    if modulus == &get_modulus_as_bigint::<halo2curves_axiom::bls12_381::Fq>() {
        return Some(FieldType::BLS12_381Coordinate);
    }

    if modulus == &get_modulus_as_bigint::<halo2curves_axiom::bls12_381::Fr>() {
        return Some(FieldType::BLS12_381Scalar);
    }

    None
}

pub fn get_fp2_field_type(modulus: &BigUint) -> Option<FieldType> {
    if modulus == &get_modulus_as_bigint::<halo2curves_axiom::bn256::Fq>() {
        return Some(FieldType::BN254Coordinate);
    }

    if modulus == &get_modulus_as_bigint::<halo2curves_axiom::bls12_381::Fq>() {
        return Some(FieldType::BLS12_381Coordinate);
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
        x if x == FieldType::K256Coordinate as u8 => {
            field_operation_256bit::<halo2curves_axiom::secq256k1::Fq, BLOCKS, BLOCK_SIZE, OP>(
                input_data,
            )
        }
        x if x == FieldType::K256Scalar as u8 => {
            field_operation_256bit::<halo2curves_axiom::secq256k1::Fp, BLOCKS, BLOCK_SIZE, OP>(
                input_data,
            )
        }
        x if x == FieldType::P256Coordinate as u8 => {
            field_operation_256bit::<halo2curves_axiom::secp256r1::Fp, BLOCKS, BLOCK_SIZE, OP>(
                input_data,
            )
        }
        x if x == FieldType::P256Scalar as u8 => {
            field_operation_256bit::<halo2curves_axiom::secp256r1::Fq, BLOCKS, BLOCK_SIZE, OP>(
                input_data,
            )
        }
        x if x == FieldType::BN254Coordinate as u8 => {
            field_operation_256bit::<halo2curves_axiom::bn256::Fq, BLOCKS, BLOCK_SIZE, OP>(
                input_data,
            )
        }
        x if x == FieldType::BN254Scalar as u8 => {
            field_operation_256bit::<halo2curves_axiom::bn256::Fr, BLOCKS, BLOCK_SIZE, OP>(
                input_data,
            )
        }
        x if x == FieldType::BLS12_381Coordinate as u8 => {
            field_operation_bls12_381_coordinate::<BLOCKS, BLOCK_SIZE, OP>(input_data)
        }
        x if x == FieldType::BLS12_381Scalar as u8 => {
            field_operation_256bit::<halo2curves_axiom::bls12_381::Fr, BLOCKS, BLOCK_SIZE, OP>(
                input_data,
            )
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
        x if x == FieldType::BN254Coordinate as u8 => {
            fp2_operation_bn254::<BLOCKS, BLOCK_SIZE, OP>(input_data)
        }
        x if x == FieldType::BLS12_381Coordinate as u8 => {
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
    field_element_to_blocks(&c, &mut output);
    output
}

#[inline(always)]
fn field_operation_bls12_381_coordinate<
    const BLOCKS: usize,
    const BLOCK_SIZE: usize,
    const OP: u8,
>(
    input_data: [[[u8; BLOCK_SIZE]; BLOCKS]; 2],
) -> [[u8; BLOCK_SIZE]; BLOCKS] {
    let a = blocks_to_field_element_bls12_381_coordinate(input_data[0].as_flattened());
    let b = blocks_to_field_element_bls12_381_coordinate(input_data[1].as_flattened());
    let c = match OP {
        x if x == Operation::Add as u8 => a + b,
        x if x == Operation::Sub as u8 => a - b,
        x if x == Operation::Mul as u8 => a * b,
        x if x == Operation::Div as u8 => a * b.invert().unwrap(),
        _ => panic!("Unsupported operation: {}", OP),
    };

    let mut output = [[0u8; BLOCK_SIZE]; BLOCKS];
    field_element_to_blocks_bls12_381_coordinate(&c, &mut output);
    output
}

#[inline(always)]
fn fp2_operation_bn254<const BLOCKS: usize, const BLOCK_SIZE: usize, const OP: u8>(
    input_data: [[[u8; BLOCK_SIZE]; BLOCKS]; 2],
) -> [[u8; BLOCK_SIZE]; BLOCKS] {
    let a = blocks_to_fp2_bn254::<BLOCKS, BLOCK_SIZE>(input_data[0].as_ref());
    let b = blocks_to_fp2_bn254::<BLOCKS, BLOCK_SIZE>(input_data[1].as_ref());
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
    let a = blocks_to_fp2_bls12_381::<BLOCKS, BLOCK_SIZE>(input_data[0].as_ref());
    let b = blocks_to_fp2_bls12_381::<BLOCKS, BLOCK_SIZE>(input_data[1].as_ref());
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
fn from_repr_with_reduction<F: PrimeField<Repr = [u8; 32]>>(bytes: [u8; 32]) -> F {
    F::from_repr_vartime(bytes).unwrap_or_else(|| {
        // Reduce modulo the field's modulus for non-canonical representations
        let modulus = get_modulus_as_bigint::<F>();
        let value = BigUint::from_bytes_le(&bytes);
        let reduced = value % modulus;

        let reduced_le_bytes = reduced.to_bytes_le();
        let mut reduced_bytes = [0u8; 32];
        reduced_bytes[..reduced_le_bytes.len()]
            .copy_from_slice(&reduced_le_bytes[..reduced_le_bytes.len()]);

        F::from_repr_vartime(reduced_bytes).unwrap()
    })
}

#[inline(always)]
fn from_repr_with_reduction_bls12_381_coordinate(
    bytes: [u8; 48],
) -> halo2curves_axiom::bls12_381::Fq {
    halo2curves_axiom::bls12_381::Fq::from_bytes(&bytes).unwrap_or_else(|| {
        // Reduce modulo the field's modulus for non-canonical representations
        let modulus = get_modulus_as_bigint::<halo2curves_axiom::bls12_381::Fq>();
        let value = BigUint::from_bytes_le(&bytes);
        let reduced = value % modulus;

        let reduced_le_bytes = reduced.to_bytes_le();
        let mut reduced_bytes = [0u8; 48];
        reduced_bytes[..reduced_le_bytes.len()]
            .copy_from_slice(&reduced_le_bytes[..reduced_le_bytes.len()]);

        halo2curves_axiom::bls12_381::Fq::from_bytes(&reduced_bytes).unwrap()
    })
}

#[inline(always)]
pub fn blocks_to_field_element<F: PrimeField<Repr = [u8; 32]>>(blocks: &[u8]) -> F {
    debug_assert!(blocks.len() == 32);
    let mut bytes = [0u8; 32];
    bytes[..blocks.len()].copy_from_slice(&blocks[..blocks.len()]);

    from_repr_with_reduction::<F>(bytes)
}

#[inline(always)]
pub fn field_element_to_blocks<F: PrimeField<Repr = [u8; 32]>, const BLOCK_SIZE: usize>(
    field_element: &F,
    output: &mut [[u8; BLOCK_SIZE]],
) {
    debug_assert!(output.len() * BLOCK_SIZE == 32);
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
pub fn blocks_to_field_element_bls12_381_coordinate(
    blocks: &[u8],
) -> halo2curves_axiom::bls12_381::Fq {
    debug_assert!(blocks.len() == 48);
    let mut bytes = [0u8; 48];
    bytes[..blocks.len()].copy_from_slice(&blocks[..blocks.len()]);

    from_repr_with_reduction_bls12_381_coordinate(bytes)
}

#[inline(always)]
pub fn field_element_to_blocks_bls12_381_coordinate<const BLOCK_SIZE: usize>(
    field_element: &halo2curves_axiom::bls12_381::Fq,
    output: &mut [[u8; BLOCK_SIZE]],
) {
    debug_assert!(output.len() * BLOCK_SIZE == 48);
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
fn blocks_to_fp2_bn254<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    blocks: &[[u8; BLOCK_SIZE]],
) -> halo2curves_axiom::bn256::Fq2 {
    let c0 = blocks_to_field_element::<halo2curves_axiom::bn256::Fq>(
        blocks[..BLOCKS / 2].as_flattened(),
    );
    let c1 = blocks_to_field_element::<halo2curves_axiom::bn256::Fq>(
        blocks[BLOCKS / 2..].as_flattened(),
    );
    halo2curves_axiom::bn256::Fq2::new(c0, c1)
}

#[inline(always)]
fn fp2_to_blocks_bn254<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    fp2: &halo2curves_axiom::bn256::Fq2,
    output: &mut [[u8; BLOCK_SIZE]; BLOCKS],
) {
    field_element_to_blocks::<halo2curves_axiom::bn256::Fq, BLOCK_SIZE>(
        &fp2.c0,
        &mut output[..BLOCKS / 2],
    );
    field_element_to_blocks::<halo2curves_axiom::bn256::Fq, BLOCK_SIZE>(
        &fp2.c1,
        &mut output[BLOCKS / 2..],
    );
}

#[inline(always)]
fn blocks_to_fp2_bls12_381<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    blocks: &[[u8; BLOCK_SIZE]],
) -> halo2curves_axiom::bls12_381::Fq2 {
    let c0 = blocks_to_field_element_bls12_381_coordinate(blocks[..BLOCKS / 2].as_flattened());
    let c1 = blocks_to_field_element_bls12_381_coordinate(blocks[BLOCKS / 2..].as_flattened());
    halo2curves_axiom::bls12_381::Fq2 { c0, c1 }
}

#[inline(always)]
fn fp2_to_blocks_bls12_381<const BLOCKS: usize, const BLOCK_SIZE: usize>(
    fp2: &halo2curves_axiom::bls12_381::Fq2,
    output: &mut [[u8; BLOCK_SIZE]; BLOCKS],
) {
    field_element_to_blocks_bls12_381_coordinate(&fp2.c0, &mut output[..BLOCKS / 2]);
    field_element_to_blocks_bls12_381_coordinate(&fp2.c1, &mut output[BLOCKS / 2..]);
}
