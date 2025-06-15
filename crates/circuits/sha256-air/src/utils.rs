use std::array;

pub use openvm_circuit_primitives::utils::compose;
use openvm_circuit_primitives::{
    encoder::Encoder,
    utils::{not, select},
};
use openvm_stark_backend::{p3_air::AirBuilder, p3_field::FieldAlgebra};

use super::{Sha256DigestCols, Sha256RoundCols};

// ==== Do not change these constants! ====
/// Number of bits in a SHA256 word
pub const SHA256_WORD_BITS: usize = 32;
/// Number of 16-bit limbs in a SHA256 word
pub const SHA256_WORD_U16S: usize = SHA256_WORD_BITS / 16;
/// Number of 8-bit limbs in a SHA256 word
pub const SHA256_WORD_U8S: usize = SHA256_WORD_BITS / 8;
/// Number of words in a SHA256 block
pub const SHA256_BLOCK_WORDS: usize = 16;
/// Number of cells in a SHA256 block
pub const SHA256_BLOCK_U8S: usize = SHA256_BLOCK_WORDS * SHA256_WORD_U8S;
/// Number of bits in a SHA256 block
pub const SHA256_BLOCK_BITS: usize = SHA256_BLOCK_WORDS * SHA256_WORD_BITS;
/// Number of rows per block
pub const SHA256_ROWS_PER_BLOCK: usize = 17;
/// Number of rounds per row
pub const SHA256_ROUNDS_PER_ROW: usize = 4;
/// Number of words in a SHA256 hash
pub const SHA256_HASH_WORDS: usize = 8;
/// Number of vars needed to encode the row index with [Encoder]
pub const SHA256_ROW_VAR_CNT: usize = 5;
/// Width of the Sha256RoundCols
pub const SHA256_ROUND_WIDTH: usize = Sha256RoundCols::<u8>::width();
/// Width of the Sha256DigestCols
pub const SHA256_DIGEST_WIDTH: usize = Sha256DigestCols::<u8>::width();
/// Size of the buffer of the first 4 rows of a block (each row's size)
pub const SHA256_BUFFER_SIZE: usize = SHA256_ROUNDS_PER_ROW * SHA256_WORD_U16S * 2;
/// Width of the Sha256Cols
pub const SHA256_WIDTH: usize = if SHA256_ROUND_WIDTH > SHA256_DIGEST_WIDTH {
    SHA256_ROUND_WIDTH
} else {
    SHA256_DIGEST_WIDTH
};
/// We can notice that `carry_a`'s and `carry_e`'s are always the same on invalid rows
/// To optimize the trace generation of invalid rows, we have those values precomputed here
pub(crate) const SHA256_INVALID_CARRY_A: [[u32; SHA256_WORD_U16S]; SHA256_ROUNDS_PER_ROW] = [
    [1230919683, 1162494304],
    [266373122, 1282901987],
    [1519718403, 1008990871],
    [923381762, 330807052],
];
pub(crate) const SHA256_INVALID_CARRY_E: [[u32; SHA256_WORD_U16S]; SHA256_ROUNDS_PER_ROW] = [
    [204933122, 1994683449],
    [443873282, 1544639095],
    [719953922, 1888246508],
    [194580482, 1075725211],
];
/// SHA256 constant K's
pub const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// SHA256 initial hash values
pub const SHA256_H: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// Returns the number of blocks required to hash a message of length `len`
pub fn get_sha256_num_blocks(len: u32) -> u32 {
    // need to pad with one 1 bit, 64 bits for the message length and then pad until the length
    // is divisible by [SHA256_BLOCK_BITS]
    ((len << 3) as usize + 1 + 64).div_ceil(SHA256_BLOCK_BITS) as u32
}

/// Convert a u32 into a list of bits in little endian then convert each bit into a field element
pub fn u32_into_bits_field<F: FieldAlgebra + Clone>(num: u32) -> [F; SHA256_WORD_BITS] {
    array::from_fn(|i| F::from_bool((num >> i) & 1 == 1))
}

/// Convert a u32 into a an array of 2 16-bit limbs in little endian
pub fn u32_into_u16s(num: u32) -> [u32; 2] {
    [num & 0xffff, num >> 16]
}

/// Convert a list of limbs in little endian into a u32
pub fn limbs_into_u32<const NUM_LIMBS: usize>(limbs: [u32; NUM_LIMBS]) -> u32 {
    let limb_bits = 32 / NUM_LIMBS;
    limbs
        .iter()
        .rev()
        .fold(0, |acc, &limb| (acc << limb_bits) | limb)
}

/// Rotates `bits` right by `n` bits, assumes `bits` is in little-endian
#[inline]
pub(crate) fn rotr<F: FieldAlgebra + Clone>(
    bits: &[impl Into<F> + Clone; SHA256_WORD_BITS],
    n: usize,
) -> [F; SHA256_WORD_BITS] {
    array::from_fn(|i| bits[(i + n) % SHA256_WORD_BITS].clone().into())
}

/// Shifts `bits` right by `n` bits, assumes `bits` is in little-endian
#[inline]
pub(crate) fn shr<F: FieldAlgebra + Clone>(
    bits: &[impl Into<F> + Clone; SHA256_WORD_BITS],
    n: usize,
) -> [F; SHA256_WORD_BITS] {
    array::from_fn(|i| {
        if i + n < SHA256_WORD_BITS {
            bits[i + n].clone().into()
        } else {
            F::ZERO
        }
    })
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
    x: &[impl Into<F> + Clone; SHA256_WORD_BITS],
    y: &[impl Into<F> + Clone; SHA256_WORD_BITS],
    z: &[impl Into<F> + Clone; SHA256_WORD_BITS],
) -> [F; SHA256_WORD_BITS] {
    array::from_fn(|i| xor_bit(x[i].clone(), y[i].clone(), z[i].clone()))
}

/// Choose function from SHA256
#[inline]
pub fn ch(x: u32, y: u32, z: u32) -> u32 {
    (x & y) ^ ((!x) & z)
}

/// Computes Ch(x,y,z), where x, y, z are [SHA256_WORD_BITS] bit numbers
#[inline]
pub(crate) fn ch_field<F: FieldAlgebra>(
    x: &[impl Into<F> + Clone; SHA256_WORD_BITS],
    y: &[impl Into<F> + Clone; SHA256_WORD_BITS],
    z: &[impl Into<F> + Clone; SHA256_WORD_BITS],
) -> [F; SHA256_WORD_BITS] {
    array::from_fn(|i| select(x[i].clone(), y[i].clone(), z[i].clone()))
}

/// Majority function from SHA256
pub fn maj(x: u32, y: u32, z: u32) -> u32 {
    (x & y) ^ (x & z) ^ (y & z)
}

/// Computes Maj(x,y,z), where x, y, z are [SHA256_WORD_BITS] bit numbers
#[inline]
pub(crate) fn maj_field<F: FieldAlgebra + Clone>(
    x: &[impl Into<F> + Clone; SHA256_WORD_BITS],
    y: &[impl Into<F> + Clone; SHA256_WORD_BITS],
    z: &[impl Into<F> + Clone; SHA256_WORD_BITS],
) -> [F; SHA256_WORD_BITS] {
    array::from_fn(|i| {
        let (x, y, z) = (
            x[i].clone().into(),
            y[i].clone().into(),
            z[i].clone().into(),
        );
        x.clone() * y.clone() + x.clone() * z.clone() + y.clone() * z.clone() - F::TWO * x * y * z
    })
}

/// Big sigma_0 function from SHA256
pub fn big_sig0(x: u32) -> u32 {
    x.rotate_right(2) ^ x.rotate_right(13) ^ x.rotate_right(22)
}

/// Computes BigSigma0(x), where x is a [SHA256_WORD_BITS] bit number in little-endian
#[inline]
pub(crate) fn big_sig0_field<F: FieldAlgebra + Clone>(
    x: &[impl Into<F> + Clone; SHA256_WORD_BITS],
) -> [F; SHA256_WORD_BITS] {
    xor(&rotr::<F>(x, 2), &rotr::<F>(x, 13), &rotr::<F>(x, 22))
}

/// Big sigma_1 function from SHA256
pub fn big_sig1(x: u32) -> u32 {
    x.rotate_right(6) ^ x.rotate_right(11) ^ x.rotate_right(25)
}

/// Computes BigSigma1(x), where x is a [SHA256_WORD_BITS] bit number in little-endian
#[inline]
pub(crate) fn big_sig1_field<F: FieldAlgebra + Clone>(
    x: &[impl Into<F> + Clone; SHA256_WORD_BITS],
) -> [F; SHA256_WORD_BITS] {
    xor(&rotr::<F>(x, 6), &rotr::<F>(x, 11), &rotr::<F>(x, 25))
}

/// Small sigma_0 function from SHA256
pub fn small_sig0(x: u32) -> u32 {
    x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3)
}

/// Computes SmallSigma0(x), where x is a [SHA256_WORD_BITS] bit number in little-endian
#[inline]
pub(crate) fn small_sig0_field<F: FieldAlgebra + Clone>(
    x: &[impl Into<F> + Clone; SHA256_WORD_BITS],
) -> [F; SHA256_WORD_BITS] {
    xor(&rotr::<F>(x, 7), &rotr::<F>(x, 18), &shr::<F>(x, 3))
}

/// Small sigma_1 function from SHA256
pub fn small_sig1(x: u32) -> u32 {
    x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10)
}

/// Computes SmallSigma1(x), where x is a [SHA256_WORD_BITS] bit number in little-endian
#[inline]
pub(crate) fn small_sig1_field<F: FieldAlgebra + Clone>(
    x: &[impl Into<F> + Clone; SHA256_WORD_BITS],
) -> [F; SHA256_WORD_BITS] {
    xor(&rotr::<F>(x, 17), &rotr::<F>(x, 19), &shr::<F>(x, 10))
}

/// Wrapper of `get_flag_pt` to get the flag pointer as an array
pub fn get_flag_pt_array<const N: usize>(encoder: &Encoder, flag_idx: usize) -> [u32; N] {
    encoder.get_flag_pt(flag_idx).try_into().unwrap()
}

/// Constrain the addition of [SHA256_WORD_BITS] bit words in 16-bit limbs
/// It takes in the terms some in bits some in 16-bit limbs,
/// the expected sum in bits and the carries
pub fn constraint_word_addition<AB: AirBuilder>(
    builder: &mut AB,
    terms_bits: &[&[impl Into<AB::Expr> + Clone; SHA256_WORD_BITS]],
    terms_limb: &[&[impl Into<AB::Expr> + Clone; SHA256_WORD_U16S]],
    expected_sum: &[impl Into<AB::Expr> + Clone; SHA256_WORD_BITS],
    carries: &[impl Into<AB::Expr> + Clone; SHA256_WORD_U16S],
) {
    for i in 0..SHA256_WORD_U16S {
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
