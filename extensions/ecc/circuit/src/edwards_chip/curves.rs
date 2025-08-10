use halo2curves_axiom::{ed25519::TwistedEdwardsCurveExt, ff::PrimeField};
use lazy_static::lazy_static;
use num_bigint::BigUint;
use openvm_algebra_circuit::fields::{blocks_to_field_element, field_element_to_blocks};

use crate::weierstrass_chip::curves::get_modulus_as_bigint;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TeCurveType {
    ED25519 = 0,
}

pub(super) fn get_te_curve_type(
    modulus: &BigUint,
    a_coeff: &BigUint,
    d_coeff: &BigUint,
) -> Option<TeCurveType> {
    if modulus == &ED25519_CURVE_PARAMS.modulus
        && a_coeff == &ED25519_CURVE_PARAMS.a
        && d_coeff == &ED25519_CURVE_PARAMS.d
    {
        return Some(TeCurveType::ED25519);
    }

    None
}

struct CurveParams {
    modulus: BigUint,
    a: BigUint,
    d: BigUint,
}

lazy_static! {
    static ref ED25519_CURVE_PARAMS: CurveParams = CurveParams {
        modulus: get_modulus_as_bigint::<halo2curves_axiom::ed25519::Fq>(),
        a: BigUint::from_bytes_le(
            &<halo2curves_axiom::ed25519::Ed25519 as halo2curves_axiom::ed25519::TwistedEdwardsCurveExt>::a().to_repr(),
        ),
        d: BigUint::from_bytes_le(
            &<halo2curves_axiom::ed25519::Ed25519 as halo2curves_axiom::ed25519::TwistedEdwardsCurveExt>::d().to_repr(),
        ),
    };
}

#[inline(always)]
pub fn te_add<const CURVE_TYPE: u8, const BLOCKS: usize, const BLOCK_SIZE: usize>(
    input_data: [[[u8; BLOCK_SIZE]; BLOCKS]; 2],
) -> [[u8; BLOCK_SIZE]; BLOCKS] {
    match CURVE_TYPE {
        x if x == TeCurveType::ED25519 as u8 => {
            te_add_256bit::<halo2curves_axiom::ed25519::Fq, BLOCKS, BLOCK_SIZE>(
                input_data,
                halo2curves_axiom::ed25519::Ed25519::a(),
                halo2curves_axiom::ed25519::Ed25519::d(),
            )
        }
        _ => panic!("Unsupported curve type: {}", CURVE_TYPE),
    }
}

#[inline(always)]
fn te_add_256bit<F: PrimeField<Repr = [u8; 32]>, const BLOCKS: usize, const BLOCK_SIZE: usize>(
    input_data: [[[u8; BLOCK_SIZE]; BLOCKS]; 2],
    a: F,
    d: F,
) -> [[u8; BLOCK_SIZE]; BLOCKS] {
    let x1 = blocks_to_field_element::<F>(input_data[0][..BLOCKS / 2].as_flattened());
    let y1 = blocks_to_field_element::<F>(input_data[0][BLOCKS / 2..].as_flattened());
    let x2 = blocks_to_field_element::<F>(input_data[1][..BLOCKS / 2].as_flattened());
    let y2 = blocks_to_field_element::<F>(input_data[1][BLOCKS / 2..].as_flattened());

    let (x3, y3) = te_add_impl::<F>(x1, y1, x2, y2, a, d);

    let mut output = [[0u8; BLOCK_SIZE]; BLOCKS];
    field_element_to_blocks::<F, BLOCK_SIZE>(&x3, &mut output[..BLOCKS / 2]);
    field_element_to_blocks::<F, BLOCK_SIZE>(&y3, &mut output[BLOCKS / 2..]);
    output
}

#[inline(always)]
pub fn te_add_impl<F: PrimeField>(x1: F, y1: F, x2: F, y2: F, a: F, d: F) -> (F, F) {
    println!("te_add_impl called, a: {:?}, d: {:?}", a, d);
    let dx1x2y1y2 = d * x1 * x2 * y1 * y2;
    let x3 = (x1 * y2 + x2 * y1) * (F::ONE + dx1x2y1y2).invert().unwrap();
    let y3 = (y1 * y2 - a * x1 * x2) * (F::ONE - dx1x2y1y2).invert().unwrap();

    (x3, y3)
}
