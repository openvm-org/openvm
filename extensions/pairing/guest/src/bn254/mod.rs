#[cfg(not(target_os = "zkvm"))]
use hex_literal::hex;
#[cfg(not(target_os = "zkvm"))]
use lazy_static::lazy_static;
#[cfg(not(target_os = "zkvm"))]
use num_bigint::BigUint;

#[cfg(all(test, feature = "halo2curves", not(target_os = "zkvm")))]
pub mod tests;

#[cfg(not(target_os = "zkvm"))]
lazy_static! {
    pub static ref BN254_MODULUS: BigUint = BigUint::from_bytes_be(&hex!(
        "30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47"
    ));
    pub static ref BN254_ORDER: BigUint = BigUint::from_bytes_be(&hex!(
        "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001"
    ));
}

pub const BN254_XI_ISIZE: [isize; 2] = [9, 1];
pub const BN254_NUM_LIMBS: usize = 32;
pub const BN254_LIMB_BITS: usize = 8;
pub const BN254_BLOCK_SIZE: usize = 32;

pub const BN254_SEED: u64 = 0x44e992b44a6909f1;
// Encodes 6x+2 where x is the BN254 seed.
// 6*x+2 = sum_i BN254_PSEUDO_BINARY_ENCODING[i] * 2^i
// where BN254_PSEUDO_BINARY_ENCODING[i] is in {-1, 0, 1}
// Validated against BN254_SEED_ABS by a test in tests.rs
pub const BN254_PSEUDO_BINARY_ENCODING: [i8; 66] = [
    0, 0, 0, 1, 0, 1, 0, -1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0, -1, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0,
    -1, 0, 0, 1, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 1, 0, -1, 0, 0, 0, -1, 0, -1, 0,
    0, 0, 1, 0, -1, 0, 1,
];

#[cfg(not(target_os = "zkvm"))]
// Used in WeierstrassExtension config
pub const BN254_ECC_STRUCT_NAME: &str = "Bn254G1Affine";

#[cfg(not(target_os = "zkvm"))]
// Used in Fp2Extension config
pub const BN254_COMPLEX_STRUCT_NAME: &str = "Bn254Fp2";
