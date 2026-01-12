use halo2curves_axiom::{
    bls12_381::{Fq, Fq12, Fq2},
    ff::Field,
};
use num_bigint::BigUint;
use openvm_pairing_guest::algebra::field::FieldExtension;
use rand08::{rngs::StdRng, SeedableRng};

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
    let mut rng = StdRng::seed_from_u64(seed);
    let fq = Fq12::random(&mut rng);
    bls12381_fq12_to_biguint_vec(fq)
}
