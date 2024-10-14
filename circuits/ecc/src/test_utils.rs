use std::{cell::RefCell, rc::Rc, str::FromStr, sync::Arc};

use afs_primitives::{
    bigint::check_carry_mod_to_zero::CheckCarryModToZeroSubAir,
    var_range::{bus::VariableRangeCheckerBus, VariableRangeCheckerChip},
};
use ax_sdk::utils::create_seeded_rng;
use num_bigint_dig::BigUint;
use num_traits::{FromPrimitive, Zero};
use p3_baby_bear::BabyBear;
use p3_field::PrimeField64;
use rand::RngCore;

use super::field_expression::{ExprBuilder, FieldVariableConfig};

pub const LIMB_BITS: usize = 8;

pub fn setup(
    prime: &BigUint,
) -> (
    CheckCarryModToZeroSubAir,
    Arc<VariableRangeCheckerChip>,
    Rc<RefCell<ExprBuilder>>,
) {
    let field_element_bits = 30;
    let range_bus = 1;
    let range_decomp = 17; // double needs 17, rests need 16.
    let range_checker = Arc::new(VariableRangeCheckerChip::new(VariableRangeCheckerBus::new(
        range_bus,
        range_decomp,
    )));
    let subair = CheckCarryModToZeroSubAir::new(
        prime.clone(),
        LIMB_BITS,
        range_bus,
        range_decomp,
        field_element_bits,
    );
    let builder = ExprBuilder::new(prime.clone(), LIMB_BITS, 32, range_checker.range_max_bits());
    (subair, range_checker, Rc::new(RefCell::new(builder)))
}

pub fn generate_random_biguint(prime: &BigUint) -> BigUint {
    let mut rng = create_seeded_rng();
    let len = 32;
    let x = (0..len).map(|_| rng.next_u32()).collect();
    let x = BigUint::new(x);
    x % prime
}

#[derive(Clone)]
pub struct TestConfig;
impl FieldVariableConfig for TestConfig {
    fn canonical_limb_bits() -> usize {
        LIMB_BITS
    }

    fn max_limb_bits() -> usize {
        29
    }

    fn num_limbs_per_field_element() -> usize {
        32
    }
}

pub fn evaluate_biguint(limbs: &[BabyBear], limb_bits: usize) -> BigUint {
    let mut res = BigUint::zero();
    let base = BigUint::from_u64(1 << limb_bits).unwrap();
    for limb in limbs.iter().rev() {
        res = res * base.clone() + BigUint::from_u64(limb.as_canonical_u64()).unwrap();
    }
    res
}

pub fn bn254_prime() -> BigUint {
    BigUint::from_str(
        "21888242871839275222246405745257275088696311157297823662689037894645226208583",
    )
    .unwrap()
}
