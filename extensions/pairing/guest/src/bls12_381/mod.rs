use hex_literal::hex;
#[cfg(not(target_os = "zkvm"))]
use lazy_static::lazy_static;
#[cfg(not(target_os = "zkvm"))]
use num_bigint::BigUint;

#[cfg(all(test, feature = "halo2curves", not(target_os = "zkvm")))]
mod tests;

#[cfg(not(target_os = "zkvm"))]
lazy_static! {
    pub static ref BLS12_381_MODULUS: BigUint = BigUint::from_bytes_be(&hex!(
        "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab"
    ));
    pub static ref BLS12_381_ORDER: BigUint = BigUint::from_bytes_be(&hex!(
        "73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001"
    ));
}

pub const BLS12_381_XI_ISIZE: [isize; 2] = [1, 1];
pub const BLS12_381_NUM_LIMBS: usize = 48;
pub const BLS12_381_LIMB_BITS: usize = 8;
pub const BLS12_381_BLOCK_SIZE: usize = 16;

pub const BLS12_381_SEED_ABS: u64 = 0xd201000000010000;
// Encodes the Bls12_381 seed, x.
// x = sum_i BLS12_381_PSEUDO_BINARY_ENCODING[i] * 2^i
// where BLS12_381_PSEUDO_BINARY_ENCODING[i] is in {-1, 0, 1}
// Validated against BLS12_381_SEED_ABS by a test in tests.rs
pub const BLS12_381_PSEUDO_BINARY_ENCODING: [i8; 64] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1,
];

#[cfg(not(target_os = "zkvm"))]
// Used in WeierstrassExtension config
pub const BLS12_381_ECC_STRUCT_NAME: &str = "Bls12_381G1Affine";

#[cfg(not(target_os = "zkvm"))]
// Used in Fp2Extension config
pub const BLS12_381_COMPLEX_STRUCT_NAME: &str = "Bls12_381Fp2";
