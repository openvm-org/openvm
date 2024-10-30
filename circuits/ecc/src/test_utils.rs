use std::{cell::RefCell, rc::Rc, sync::Arc};

use ax_circuit_primitives::{
    bigint::check_carry_mod_to_zero::CheckCarryModToZeroSubAir,
    var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip},
};
use ax_stark_sdk::utils::{create_seeded_rng, create_seeded_rng_with_seed};
use halo2curves_axiom::{bls12_381, bn256};
use num_bigint_dig::BigUint;
use num_traits::{FromPrimitive, Zero};
use p3_baby_bear::BabyBear;
use p3_field::PrimeField64;
use rand::RngCore;

use super::field_expression::{ExprBuilder, ExprBuilderConfig};

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
    let config = ExprBuilderConfig {
        modulus: prime.clone(),
        limb_bits: LIMB_BITS,
        num_limbs: 32,
    };
    let builder = ExprBuilder::new(config, range_checker.range_max_bits());
    (subair, range_checker, Rc::new(RefCell::new(builder)))
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

pub fn bn254_fq_to_biguint(fq: &bn256::Fq) -> BigUint {
    let bytes = fq.to_bytes();
    BigUint::from_bytes_le(&bytes)
}

pub fn bn254_fq12_to_biguint_vec(x: &bn256::Fq12) -> Vec<BigUint> {
    vec![
        bn254_fq_to_biguint(&x.c0.c0.c0),
        bn254_fq_to_biguint(&x.c0.c0.c1),
        bn254_fq_to_biguint(&x.c0.c1.c0),
        bn254_fq_to_biguint(&x.c0.c1.c1),
        bn254_fq_to_biguint(&x.c0.c2.c0),
        bn254_fq_to_biguint(&x.c0.c2.c1),
        bn254_fq_to_biguint(&x.c1.c0.c0),
        bn254_fq_to_biguint(&x.c1.c0.c1),
        bn254_fq_to_biguint(&x.c1.c1.c0),
        bn254_fq_to_biguint(&x.c1.c1.c1),
        bn254_fq_to_biguint(&x.c1.c2.c0),
        bn254_fq_to_biguint(&x.c1.c2.c1),
    ]
}

pub fn bn254_fq2_to_biguint_vec(x: &bn256::Fq2) -> Vec<BigUint> {
    vec![bn254_fq_to_biguint(&x.c0), bn254_fq_to_biguint(&x.c1)]
}

pub fn bn254_fq12_random(seed: u64) -> Vec<BigUint> {
    use halo2curves_axiom::ff::Field;

    let seed = create_seeded_rng_with_seed(seed);
    let fq = bn256::Fq12::random(seed);
    bn254_fq12_to_biguint_vec(&fq)
}

pub fn bls12381_fq_to_biguint(fq: &bls12_381::Fq) -> BigUint {
    let bytes = fq.to_bytes();
    BigUint::from_bytes_le(&bytes)
}

pub fn bls12381_fq12_to_biguint_vec(x: &bls12_381::Fq12) -> Vec<BigUint> {
    vec![
        bls12381_fq_to_biguint(&x.c0.c0.c0),
        bls12381_fq_to_biguint(&x.c0.c0.c1),
        bls12381_fq_to_biguint(&x.c0.c1.c0),
        bls12381_fq_to_biguint(&x.c0.c1.c1),
        bls12381_fq_to_biguint(&x.c0.c2.c0),
        bls12381_fq_to_biguint(&x.c0.c2.c1),
        bls12381_fq_to_biguint(&x.c1.c0.c0),
        bls12381_fq_to_biguint(&x.c1.c0.c1),
        bls12381_fq_to_biguint(&x.c1.c1.c0),
        bls12381_fq_to_biguint(&x.c1.c1.c1),
        bls12381_fq_to_biguint(&x.c1.c2.c0),
        bls12381_fq_to_biguint(&x.c1.c2.c1),
    ]
}

pub fn bls12381_fq2_to_biguint_vec(x: &bls12_381::Fq2) -> Vec<BigUint> {
    vec![bls12381_fq_to_biguint(&x.c0), bls12381_fq_to_biguint(&x.c1)]
}

pub fn bls12381_fq12_random(seed: u64) -> Vec<BigUint> {
    use halo2curves_axiom::ff::Field;

    let seed = create_seeded_rng_with_seed(seed);
    let fq = bls12_381::Fq12::random(seed);
    bls12381_fq12_to_biguint_vec(&fq)
}
