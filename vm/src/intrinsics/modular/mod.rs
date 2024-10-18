mod addsub;
pub use addsub::*;
mod muldiv;
use hex_literal::hex;
pub use muldiv::*;
use num_bigint_dig::BigUint;
use once_cell::sync::Lazy;

use crate::{
    arch::{VmAirWrapper, VmChipWrapper},
    rv32im::adapters::{Rv32VecHeapAdapterAir, Rv32VecHeapAdapterChip},
};

#[cfg(test)]
mod tests;

pub const FIELD_ELEMENT_BITS: usize = 30;

pub type ModularAddSubAir<const NUM_LIMBS: usize> =
    VmAirWrapper<Rv32VecHeapAdapterAir<1, 1, NUM_LIMBS, NUM_LIMBS>, ModularAddSubCoreAir>;
pub type ModularAddSubChip<F, const NUM_LIMBS: usize> =
    VmChipWrapper<F, Rv32VecHeapAdapterChip<F, 1, 1, NUM_LIMBS, NUM_LIMBS>, ModularAddSubCoreChip>;
pub type ModularMulDivAir<const NUM_LIMBS: usize> =
    VmAirWrapper<Rv32VecHeapAdapterAir<1, 1, NUM_LIMBS, NUM_LIMBS>, ModularMulDivCoreAir>;
pub type ModularMulDivChip<F, const NUM_LIMBS: usize> =
    VmChipWrapper<F, Rv32VecHeapAdapterChip<F, 1, 1, NUM_LIMBS, NUM_LIMBS>, ModularMulDivCoreChip>;

pub static SECP256K1_COORD_PRIME: Lazy<BigUint> = Lazy::new(|| {
    BigUint::from_bytes_be(&hex!(
        "FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F"
    ))
});

pub static SECP256K1_SCALAR_PRIME: Lazy<BigUint> = Lazy::new(|| {
    BigUint::from_bytes_be(&hex!(
        "FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D0364141"
    ))
});
