use ax_ecc_execution::axvm_ecc_guest::algebra::field::FieldExtension;
use ax_stark_sdk::utils::create_seeded_rng_with_seed;
use halo2curves_axiom::{
    bls12_381::{Fq, Fq12, Fq2},
    ff::Field,
};
use num_bigint_dig::BigUint;

pub fn bls12381_fq_to_biguint(fq: Fq) -> BigUint {
    let bytes = fq.to_bytes();
    BigUint::from_bytes_le(&bytes)
}

pub fn bls12381_fq2_to_biguint_vec(x: Fq2) -> Vec<BigUint> {
    vec![bls12381_fq_to_biguint(x.c0), bls12381_fq_to_biguint(x.c1)]
}

pub fn bls12381_fq12_to_biguint_vec(x: Fq12) -> Vec<BigUint> {
    x.to_coeffs()
        .into_iter()
        .flat_map(bls12381_fq2_to_biguint_vec)
        .collect()
}

pub fn bls12381_fq12_random(seed: u64) -> Vec<BigUint> {
    let seed = create_seeded_rng_with_seed(seed);
    let fq = Fq12::random(seed);
    bls12381_fq12_to_biguint_vec(fq)
}
