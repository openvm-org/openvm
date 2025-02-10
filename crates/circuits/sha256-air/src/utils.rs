use openvm_circuit_primitives::{
    encoder::Encoder,
    utils::{not, select},
};
use openvm_stark_backend::{p3_air::AirBuilder, p3_field::FieldAlgebra};
use rand::{rngs::StdRng, Rng};

use crate::{RotateRight, ShaConfig};

/// Convert a word into a list of 8-bit limbs in little endian
pub fn word_into_u8_limbs<C: ShaConfig>(num: impl Into<C::Word>) -> Vec<u32> {
    word_into_limbs::<C>(num.into(), C::WORD_U8S)
}

/// Convert a word into a list of 16-bit limbs in little endian
pub fn word_into_u16_limbs<C: ShaConfig>(num: impl Into<C::Word>) -> Vec<u32> {
    word_into_limbs::<C>(num.into(), C::WORD_U16S)
}

/// Convert a word into a list of 1-bit limbs in little endian
pub fn word_into_bits<C: ShaConfig>(num: impl Into<C::Word>) -> Vec<u32> {
    word_into_limbs::<C>(num.into(), C::WORD_BITS)
}

/// Convert a word into a list of limbs in little endian
pub fn word_into_limbs<C: ShaConfig>(num: C::Word, num_limbs: usize) -> Vec<u32> {
    let limb_bits = std::mem::size_of::<C::Word>() * 8 / num_limbs;
    (0..num_limbs)
        .map(|i| {
            let shifted = num >> (limb_bits * i);
            let mask: C::Word = ((1u32 << limb_bits) - 1).into();
            let masked = shifted & mask;
            masked.try_into().unwrap()
        })
        .collect()
}

/// Convert a u32 into a list of 1-bit limbs in little endian
pub fn u32_into_bits<C: ShaConfig>(num: u32) -> Vec<u32> {
    let limb_bits = 32 / C::WORD_BITS;
    (0..C::WORD_BITS)
        .map(|i| (num >> (limb_bits * i)) & ((1 << limb_bits) - 1))
        .collect()
}

// TODO: delete
/*
/// Convert a u32 into a list of limbs in little endian
pub fn u32_into_limbs<const NUM_LIMBS: usize>(num: u32) -> [u32; NUM_LIMBS] {
    let limb_bits = 32 / NUM_LIMBS;
    array::from_fn(|i| (num >> (limb_bits * i)) & ((1 << limb_bits) - 1))
}
*/

/// Convert a list of limbs in little endian into a Word
pub fn limbs_into_word<C: ShaConfig>(limbs: &[u32]) -> C::Word {
    let limb_bits = C::WORD_BITS / limbs.len();
    limbs.iter().rev().fold(C::Word::from(0), |acc, &limb| {
        (acc << limb_bits) | limb.into()
    })
}

/// Convert a list of limbs in little endian into a u32
pub fn limbs_into_u32(limbs: &[u32]) -> u32 {
    let limb_bits = 32 / limbs.len();
    limbs
        .iter()
        .rev()
        .fold(0, |acc, &limb| (acc << limb_bits) | limb)
}

/// Rotates `bits` right by `n` bits, assumes `bits` is in little-endian
#[inline]
pub(crate) fn rotr<F: FieldAlgebra + Clone>(bits: &[impl Into<F> + Clone], n: usize) -> Vec<F> {
    (0..bits.len())
        .map(|i| bits[(i + n) % bits.len()].clone().into())
        .collect()
}

/// Shifts `bits` right by `n` bits, assumes `bits` is in little-endian
#[inline]
pub(crate) fn shr<F: FieldAlgebra + Clone>(bits: &[impl Into<F> + Clone], n: usize) -> Vec<F> {
    (0..bits.len())
        .map(|i| {
            if i + n < bits.len() {
                bits[i + n].clone().into()
            } else {
                F::ZERO
            }
        })
        .collect()
}

/// Computes x ^ y ^ z, where x, y, z are assumed to be boolean
#[inline]
pub(crate) fn xor_bit<F: FieldAlgebra + Clone>(
    x: impl Into<F>,
    y: impl Into<F>,
    z: impl Into<F>,
) -> F {
    let (x, y, z) = (x.into(), y.into(), z.into());
    (x.clone() * y.clone() * z.clone())
        + (x.clone() * not::<F>(y.clone()) * not::<F>(z.clone()))
        + (not::<F>(x.clone()) * y.clone() * not::<F>(z.clone()))
        + (not::<F>(x) * not::<F>(y) * z)
}

/// Computes x ^ y ^ z, where x, y, z are [SHA256_WORD_BITS] bit numbers
#[inline]
pub(crate) fn xor<F: FieldAlgebra + Clone>(
    x: &[impl Into<F> + Clone],
    y: &[impl Into<F> + Clone],
    z: &[impl Into<F> + Clone],
) -> Vec<F> {
    (0..x.len())
        .map(|i| xor_bit(x[i].clone(), y[i].clone(), z[i].clone()))
        .collect()
}

/// Choose function from SHA256
#[inline]
pub fn ch<C: ShaConfig>(x: C::Word, y: C::Word, z: C::Word) -> C::Word {
    (x & y) ^ ((!x) & z)
}

/// Computes Ch(x,y,z), where x, y, z are [SHA256_WORD_BITS] bit numbers
#[inline]
pub(crate) fn ch_field<F: FieldAlgebra>(
    x: &[impl Into<F> + Clone],
    y: &[impl Into<F> + Clone],
    z: &[impl Into<F> + Clone],
) -> Vec<F> {
    (0..x.len())
        .map(|i| select(x[i].clone(), y[i].clone(), z[i].clone()))
        .collect()
}

/// Majority function from SHA256
pub fn maj<C: ShaConfig>(x: C::Word, y: C::Word, z: C::Word) -> C::Word {
    (x & y) ^ (x & z) ^ (y & z)
}

/// Computes Maj(x,y,z), where x, y, z are [SHA256_WORD_BITS] bit numbers
#[inline]
pub(crate) fn maj_field<F: FieldAlgebra + Clone>(
    x: &[impl Into<F> + Clone],
    y: &[impl Into<F> + Clone],
    z: &[impl Into<F> + Clone],
) -> Vec<F> {
    (0..x.len())
        .map(|i| {
            let (x, y, z) = (
                x[i].clone().into(),
                y[i].clone().into(),
                z[i].clone().into(),
            );
            x.clone() * y.clone() + x.clone() * z.clone() + y.clone() * z.clone()
                - F::TWO * x * y * z
        })
        .collect()
}

/// Big sigma_0 function from SHA256
pub fn big_sig0<C: ShaConfig>(x: C::Word) -> C::Word {
    x.rotate_right(2) ^ x.rotate_right(13) ^ x.rotate_right(22)
}

/// Computes BigSigma0(x), where x is a [SHA256_WORD_BITS] bit number in little-endian
#[inline]
pub(crate) fn big_sig0_field<F: FieldAlgebra + Clone>(x: &[impl Into<F> + Clone]) -> Vec<F> {
    xor(&rotr::<F>(x, 2), &rotr::<F>(x, 13), &rotr::<F>(x, 22))
}

/// Big sigma_1 function from SHA256
pub fn big_sig1<C: ShaConfig>(x: C::Word) -> C::Word {
    x.rotate_right(6) ^ x.rotate_right(11) ^ x.rotate_right(25)
}

/// Computes BigSigma1(x), where x is a [SHA256_WORD_BITS] bit number in little-endian
#[inline]
pub(crate) fn big_sig1_field<F: FieldAlgebra + Clone>(x: &[impl Into<F> + Clone]) -> Vec<F> {
    xor(&rotr::<F>(x, 6), &rotr::<F>(x, 11), &rotr::<F>(x, 25))
}

/// Small sigma_0 function from SHA256
pub fn small_sig0<C: ShaConfig>(x: C::Word) -> C::Word {
    x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3)
}

/// Computes SmallSigma0(x), where x is a [SHA256_WORD_BITS] bit number in little-endian
#[inline]
pub(crate) fn small_sig0_field<F: FieldAlgebra + Clone>(x: &[impl Into<F> + Clone]) -> Vec<F> {
    xor(&rotr::<F>(x, 7), &rotr::<F>(x, 18), &shr::<F>(x, 3))
}

/// Small sigma_1 function from SHA256
pub fn small_sig1<C: ShaConfig>(x: C::Word) -> C::Word {
    x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10)
}

/// Computes SmallSigma1(x), where x is a [SHA256_WORD_BITS] bit number in little-endian
#[inline]
pub(crate) fn small_sig1_field<F: FieldAlgebra + Clone>(x: &[impl Into<F> + Clone]) -> Vec<F> {
    xor(&rotr::<F>(x, 17), &rotr::<F>(x, 19), &shr::<F>(x, 10))
}

/// Generate a random message of a given length
pub fn get_random_message(rng: &mut StdRng, len: usize) -> Vec<u8> {
    let mut random_message: Vec<u8> = vec![0u8; len];
    rng.fill(&mut random_message[..]);
    random_message
}

/// Composes a list of limb values into a single field element
#[inline]
pub fn compose<F: FieldAlgebra>(a: &[impl Into<F> + Clone], limb_size: usize) -> F {
    a.iter().enumerate().fold(F::ZERO, |acc, (i, x)| {
        acc + x.clone().into() * F::from_canonical_usize(1 << (i * limb_size))
    })
}

/// Wrapper of `get_flag_pt` to get the flag pointer as an array
pub fn get_flag_pt_array(encoder: &Encoder, flag_idx: usize) -> Vec<u32> {
    encoder.get_flag_pt(flag_idx)
}

/// Constrain the addition of [SHA256_WORD_BITS] bit words in 16-bit limbs
/// It takes in the terms some in bits some in 16-bit limbs,
/// the expected sum in bits and the carries
pub fn constraint_word_addition<AB: AirBuilder, C: ShaConfig>(
    builder: &mut AB,
    terms_bits: &[&[impl Into<AB::Expr> + Clone]],
    terms_limb: &[&[impl Into<AB::Expr> + Clone]],
    expected_sum: &[impl Into<AB::Expr> + Clone],
    carries: &[impl Into<AB::Expr> + Clone],
) {
    debug_assert!(terms_bits.iter().all(|x| x.len() == C::WORD_BITS));
    debug_assert!(terms_limb.iter().all(|x| x.len() == C::WORD_U16S));
    assert_eq!(expected_sum.len(), C::WORD_BITS);
    assert_eq!(carries.len(), C::WORD_U16S);

    for i in 0..C::WORD_U16S {
        let mut limb_sum = if i == 0 {
            AB::Expr::ZERO
        } else {
            carries[i - 1].clone().into()
        };
        for term in terms_bits {
            limb_sum += compose::<AB::Expr>(&term[i * 16..(i + 1) * 16], 1);
        }
        for term in terms_limb {
            limb_sum += term[i].clone().into();
        }
        let expected_sum_limb = compose::<AB::Expr>(&expected_sum[i * 16..(i + 1) * 16], 1)
            + carries[i].clone().into() * AB::Expr::from_canonical_u32(1 << 16);
        builder.assert_eq(limb_sum, expected_sum_limb);
    }
}
