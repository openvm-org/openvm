use crate::poseidon2::Poseidon2Air;
use afs_stark_backend::{prover::USE_DEBUG_BUILDER, verifier::VerificationError};
use afs_test_utils::{
    config::baby_bear_poseidon2::run_simple_test_no_pis, utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use rand::RngCore;

#[test]
fn test_poseidon2_trace() {
    let num_rows = 1 << 4;
    let num_ext_rounds = 8;
    let num_int_rounds = 13;
    let mut rng = create_seeded_rng();
    let external_constants: Vec<[BabyBear; 16]> = (0..num_ext_rounds)
        .map(|_| {
            let vec: Vec<BabyBear> = (0..16)
                .map(|_| BabyBear::from_canonical_u32(rng.next_u32() % (1 << 30)))
                .collect();
            vec.try_into().unwrap()
        })
        .collect();
    let internal_constants: Vec<BabyBear> = (0..num_int_rounds)
        .map(|_| BabyBear::from_canonical_u32(rng.next_u32() % (1 << 30)))
        .collect();
    let poseidon2_air = Poseidon2Air::<16, BabyBear>::new(external_constants, internal_constants);
    let states: Vec<[BabyBear; 16]> = (0..num_rows)
        .map(|_| {
            let vec: Vec<BabyBear> = (0..16)
                .map(|_| BabyBear::from_canonical_u32(rng.next_u32() % (1 << 30)))
                .collect();
            vec.try_into().unwrap()
        })
        .collect();

    let trace = poseidon2_air.generate_trace(states);
    println!("{:?}", trace);

    run_simple_test_no_pis(vec![&poseidon2_air], vec![trace]).expect("Verification failed");
}
