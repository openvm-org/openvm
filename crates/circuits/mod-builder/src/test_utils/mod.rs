use std::{array, cell::RefCell, rc::Rc, sync::Arc};

use ax_circuit_primitives::{
    bigint::utils::big_uint_to_limbs,
    var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip},
};
use ax_stark_backend::p3_field::PrimeField64;
use ax_stark_sdk::{p3_baby_bear::BabyBear, utils::create_seeded_rng};
use axvm_circuit::utils::generate_long_number;
use num_bigint_dig::BigUint;
use num_traits::{FromPrimitive, ToPrimitive, Zero};
use rand::{rngs::StdRng, RngCore};

use crate::{ExprBuilder, ExprBuilderConfig};

mod bls12381;
mod bn254;

pub use bls12381::*;
pub use bn254::*;

pub const LIMB_BITS: usize = 8;

pub fn setup(prime: &BigUint) -> (Arc<VariableRangeCheckerChip>, Rc<RefCell<ExprBuilder>>) {
    let range_bus = 1;
    let range_decomp = 17; // double needs 17, rests need 16.
    let range_checker = Arc::new(VariableRangeCheckerChip::new(VariableRangeCheckerBus::new(
        range_bus,
        range_decomp,
    )));
    let config = ExprBuilderConfig {
        modulus: prime.clone(),
        limb_bits: LIMB_BITS,
        num_limbs: 32,
    };
    let builder = ExprBuilder::new(config, range_checker.range_max_bits());
    (range_checker, Rc::new(RefCell::new(builder)))
}

pub fn generate_random_biguint(prime: &BigUint) -> BigUint {
    let mut rng = create_seeded_rng();
    let len = 32;
    let x = (0..len).map(|_| rng.next_u32()).collect();
    let x = BigUint::new(x);
    x % prime
}

pub fn evaluate_biguint(limbs: &[BabyBear], limb_bits: usize) -> BigUint {
    let mut res = BigUint::zero();
    let base = BigUint::from_u64(1 << limb_bits).unwrap();
    for limb in limbs.iter().rev() {
        res = res * base.clone() + BigUint::from_u64(limb.as_canonical_u64()).unwrap();
    }
    res
}

// little endian.
// Warning: This function only returns the last NUM_LIMBS*LIMB_BITS bits of
//          the input, while the input can have more than that.
pub fn biguint_to_limbs<const NUM_LIMBS: usize>(
    mut x: BigUint,
    limb_size: usize,
) -> [u32; NUM_LIMBS] {
    let mut result = [0; NUM_LIMBS];
    let base = BigUint::from_u32(1 << limb_size).unwrap();
    for r in result.iter_mut() {
        *r = (x.clone() % &base).to_u32().unwrap();
        x /= &base;
    }
    assert!(x.is_zero());
    result
}

pub fn generate_field_element<const NUM_LIMBS: usize, const LIMB_BITS: usize>(
    modulus: &BigUint,
    rng: &mut StdRng,
) -> [u32; NUM_LIMBS] {
    let x = generate_long_number::<NUM_LIMBS, LIMB_BITS>(rng);
    let bigint = BigUint::new(x.to_vec()) % modulus;
    let vec = big_uint_to_limbs(&bigint, LIMB_BITS);
    array::from_fn(|i| if i < vec.len() { vec[i] as u32 } else { 0 })
}
