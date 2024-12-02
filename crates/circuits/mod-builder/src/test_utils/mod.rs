use std::{cell::RefCell, rc::Rc, sync::Arc};

use ax_circuit_primitives::var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip};
use ax_stark_sdk::utils::create_seeded_rng;
use num_bigint_dig::BigUint;
use num_traits::{FromPrimitive, Zero};
use p3_baby_bear::BabyBear;
use p3_field::PrimeField64;
use rand::RngCore;

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
