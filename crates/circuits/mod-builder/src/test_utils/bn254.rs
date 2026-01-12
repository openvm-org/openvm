use halo2curves_axiom::{
    bn256::{Fq, Fq12, Fq2},
    ff::Field,
};
use num_bigint::BigUint;
use openvm_pairing_guest::algebra::field::FieldExtension;
use rand08::{rngs::StdRng, SeedableRng};

pub fn bn254_fq_to_biguint(fq: Fq) -> BigUint {
    let bytes = fq.to_bytes();
    BigUint::from_bytes_le(&bytes)
}

pub fn bn254_fq2_to_biguint_vec(x: Fq2) -> Vec<BigUint> {
    vec![bn254_fq_to_biguint(x.c0), bn254_fq_to_biguint(x.c1)]
}

pub fn bn254_fq12_to_biguint_vec(x: Fq12) -> Vec<BigUint> {
    x.to_coeffs()
        .into_iter()
        .flat_map(bn254_fq2_to_biguint_vec)
        .collect()
}

pub fn bn254_fq2_random(seed: u64) -> Fq2 {
    let mut rng = StdRng::seed_from_u64(seed);
    Fq2::random(&mut rng)
}

pub fn bn254_fq12_random(seed: u64) -> Fq12 {
    let mut rng = StdRng::seed_from_u64(seed);
    Fq12::random(&mut rng)
}
