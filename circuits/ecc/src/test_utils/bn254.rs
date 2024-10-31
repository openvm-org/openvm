use ax_stark_sdk::utils::create_seeded_rng_with_seed;
use halo2curves_axiom::{
    bn256::{Fq, Fq12, Fq2},
    ff::Field,
};
use num_bigint_dig::BigUint;

pub fn bn254_fq_to_biguint(fq: Fq) -> BigUint {
    let bytes = fq.to_bytes();
    BigUint::from_bytes_le(&bytes)
}

pub fn bn254_fq2_to_biguint_vec(x: Fq2) -> Vec<BigUint> {
    vec![bn254_fq_to_biguint(x.c0), bn254_fq_to_biguint(x.c1)]
}

pub fn bn254_fq12_to_biguint_vec(x: Fq12) -> Vec<BigUint> {
    vec![
        bn254_fq_to_biguint(x.c0.c0.c0),
        bn254_fq_to_biguint(x.c0.c0.c1),
        bn254_fq_to_biguint(x.c0.c1.c0),
        bn254_fq_to_biguint(x.c0.c1.c1),
        bn254_fq_to_biguint(x.c0.c2.c0),
        bn254_fq_to_biguint(x.c0.c2.c1),
        bn254_fq_to_biguint(x.c1.c0.c0),
        bn254_fq_to_biguint(x.c1.c0.c1),
        bn254_fq_to_biguint(x.c1.c1.c0),
        bn254_fq_to_biguint(x.c1.c1.c1),
        bn254_fq_to_biguint(x.c1.c2.c0),
        bn254_fq_to_biguint(x.c1.c2.c1),
    ]
}

pub fn bn254_fq2_random(seed: u64) -> Fq2 {
    let seed = create_seeded_rng_with_seed(seed);
    Fq2::random(seed)
}

pub fn bn254_fq12_random(seed: u64) -> Fq12 {
    let seed = create_seeded_rng_with_seed(seed);
    Fq12::random(seed)
}
