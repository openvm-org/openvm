use ndarray::ArrayViewMut;
pub use openvm_circuit_primitives::utils::compose;
use openvm_circuit_primitives::{
    encoder::Encoder,
    utils::{not, select},
};
use openvm_stark_backend::{
    p3_air::AirBuilder,
    p3_field::{FieldAlgebra, PrimeField32},
};
use rand::{rngs::StdRng, Rng};

use crate::{RotateRight, Sha2BlockHasherSubairConfig};

/// Convert a word into a list of 8-bit limbs in little endian
pub fn word_into_u8_limbs<C: Sha2BlockHasherSubairConfig>(num: impl Into<C::Word>) -> Vec<u32> {
    word_into_limbs::<C>(num.into(), C::WORD_U8S)
}

/// Convert a word into a list of 16-bit limbs in little endian
pub fn word_into_u16_limbs<C: Sha2BlockHasherSubairConfig>(num: impl Into<C::Word>) -> Vec<u32> {
    word_into_limbs::<C>(num.into(), C::WORD_U16S)
}

/// Convert a word into a list of 1-bit limbs in little endian
pub fn word_into_bits<C: Sha2BlockHasherSubairConfig>(num: impl Into<C::Word>) -> Vec<u32> {
    word_into_limbs::<C>(num.into(), C::WORD_BITS)
}

/// Convert a word into a list of limbs in little endian
pub fn word_into_limbs<C: Sha2BlockHasherSubairConfig>(num: C::Word, num_limbs: usize) -> Vec<u32> {
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
pub fn u32_into_bits<C: Sha2BlockHasherSubairConfig>(num: u32) -> Vec<u32> {
    let limb_bits = 32 / C::WORD_BITS;
    (0..C::WORD_BITS)
        .map(|i| (num >> (limb_bits * i)) & ((1 << limb_bits) - 1))
        .collect()
}

/// Convert a list of limbs in little endian into a Word
pub fn le_limbs_into_word<C: Sha2BlockHasherSubairConfig>(limbs: &[u32]) -> C::Word {
    let mut limbs = limbs.to_vec();
    limbs.reverse();
    be_limbs_into_word::<C>(&limbs)
}

/// Convert a list of limbs in big endian into a Word
pub fn be_limbs_into_word<C: Sha2BlockHasherSubairConfig>(limbs: &[u32]) -> C::Word {
    let limb_bits = C::WORD_BITS / limbs.len();
    limbs.iter().fold(C::Word::from(0), |acc, &limb| {
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

/// Computes x ^ y ^ z, where x, y, z are [C::WORD_BITS] bit numbers
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

/// Choose function from the SHA spec
#[inline]
pub fn ch<C: Sha2BlockHasherSubairConfig>(x: C::Word, y: C::Word, z: C::Word) -> C::Word {
    (x & y) ^ ((!x) & z)
}

/// Computes Ch(x,y,z), where x, y, z are [C::WORD_BITS] bit numbers
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

/// Majority function from the SHA spec
pub fn maj<C: Sha2BlockHasherSubairConfig>(x: C::Word, y: C::Word, z: C::Word) -> C::Word {
    (x & y) ^ (x & z) ^ (y & z)
}

/// Computes Maj(x,y,z), where x, y, z are [C::WORD_BITS] bit numbers
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

/// Big sigma_0 function from the SHA spec
pub fn big_sig0<C: Sha2BlockHasherSubairConfig>(x: C::Word) -> C::Word {
    if C::WORD_BITS == 32 {
        x.rotate_right(2) ^ x.rotate_right(13) ^ x.rotate_right(22)
    } else {
        x.rotate_right(28) ^ x.rotate_right(34) ^ x.rotate_right(39)
    }
}

/// Computes BigSigma0(x), where x is a [C::WORD_BITS] bit number in little-endian
#[inline]
pub(crate) fn big_sig0_field<F: FieldAlgebra + Clone, C: Sha2BlockHasherSubairConfig>(
    x: &[impl Into<F> + Clone],
) -> Vec<F> {
    if C::WORD_BITS == 32 {
        xor(&rotr::<F>(x, 2), &rotr::<F>(x, 13), &rotr::<F>(x, 22))
    } else {
        xor(&rotr::<F>(x, 28), &rotr::<F>(x, 34), &rotr::<F>(x, 39))
    }
}

/// Big sigma_1 function from the SHA spec
pub fn big_sig1<C: Sha2BlockHasherSubairConfig>(x: C::Word) -> C::Word {
    if C::WORD_BITS == 32 {
        x.rotate_right(6) ^ x.rotate_right(11) ^ x.rotate_right(25)
    } else {
        x.rotate_right(14) ^ x.rotate_right(18) ^ x.rotate_right(41)
    }
}

/// Computes BigSigma1(x), where x is a [C::WORD_BITS] bit number in little-endian
#[inline]
pub(crate) fn big_sig1_field<F: FieldAlgebra + Clone, C: Sha2BlockHasherSubairConfig>(
    x: &[impl Into<F> + Clone],
) -> Vec<F> {
    if C::WORD_BITS == 32 {
        xor(&rotr::<F>(x, 6), &rotr::<F>(x, 11), &rotr::<F>(x, 25))
    } else {
        xor(&rotr::<F>(x, 14), &rotr::<F>(x, 18), &rotr::<F>(x, 41))
    }
}

/// Small sigma_0 function from the SHA spec
pub fn small_sig0<C: Sha2BlockHasherSubairConfig>(x: C::Word) -> C::Word {
    if C::WORD_BITS == 32 {
        x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3)
    } else {
        x.rotate_right(1) ^ x.rotate_right(8) ^ (x >> 7)
    }
}

/// Computes SmallSigma0(x), where x is a [C::WORD_BITS] bit number in little-endian
#[inline]
pub(crate) fn small_sig0_field<F: FieldAlgebra + Clone, C: Sha2BlockHasherSubairConfig>(
    x: &[impl Into<F> + Clone],
) -> Vec<F> {
    if C::WORD_BITS == 32 {
        xor(&rotr::<F>(x, 7), &rotr::<F>(x, 18), &shr::<F>(x, 3))
    } else {
        xor(&rotr::<F>(x, 1), &rotr::<F>(x, 8), &shr::<F>(x, 7))
    }
}

/// Small sigma_1 function from the SHA spec
pub fn small_sig1<C: Sha2BlockHasherSubairConfig>(x: C::Word) -> C::Word {
    if C::WORD_BITS == 32 {
        x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10)
    } else {
        x.rotate_right(19) ^ x.rotate_right(61) ^ (x >> 6)
    }
}

/// Computes SmallSigma1(x), where x is a [C::WORD_BITS] bit number in little-endian
#[inline]
pub(crate) fn small_sig1_field<F: FieldAlgebra + Clone, C: Sha2BlockHasherSubairConfig>(
    x: &[impl Into<F> + Clone],
) -> Vec<F> {
    if C::WORD_BITS == 32 {
        xor(&rotr::<F>(x, 17), &rotr::<F>(x, 19), &shr::<F>(x, 10))
    } else {
        xor(&rotr::<F>(x, 19), &rotr::<F>(x, 61), &shr::<F>(x, 6))
    }
}

/// Generate a random message of a given length
pub fn get_random_message(rng: &mut StdRng, len: usize) -> Vec<u8> {
    let mut random_message: Vec<u8> = vec![0u8; len];
    rng.fill(&mut random_message[..]);
    random_message
}

/// Wrapper of `get_flag_pt` to get the flag pointer as an array
pub fn get_flag_pt_array(encoder: &Encoder, flag_idx: usize) -> Vec<u32> {
    encoder.get_flag_pt(flag_idx)
}

/// Constrain the addition of [C::WORD_BITS] bit words in 16-bit limbs
/// It takes in the terms some in bits some in 16-bit limbs,
/// the expected sum in bits and the carries
pub fn constraint_word_addition<AB: AirBuilder, C: Sha2BlockHasherSubairConfig>(
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

pub fn set_arrayview_from_u32_slice<F: PrimeField32, D: ndarray::Dimension>(
    arrayview: &mut ArrayViewMut<F, D>,
    data: impl IntoIterator<Item = u32>,
) {
    arrayview
        .iter_mut()
        .zip(data.into_iter().map(|x| F::from_canonical_u32(x)))
        .for_each(|(x, y)| *x = y);
}

pub fn set_arrayview_from_u8_slice<F: PrimeField32, D: ndarray::Dimension>(
    arrayview: &mut ArrayViewMut<F, D>,
    data: impl IntoIterator<Item = u8>,
) {
    arrayview
        .iter_mut()
        .zip(data.into_iter().map(|x| F::from_canonical_u8(x)))
        .for_each(|(x, y)| *x = y);
}
