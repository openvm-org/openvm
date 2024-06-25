use crate::poseidon2::Poseidon2Air;
use afs_stark_backend::{prover::USE_DEBUG_BUILDER, verifier::VerificationError};
use afs_test_utils::{
    config::baby_bear_poseidon2::run_simple_test_no_pis, utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;

#[test]
fn test_poseidon2_trace() {
    let external_constants: Vec<[BabyBear; 16]> = (3..7)
        .map(|x| {
            let vec: Vec<BabyBear> = (1..=16)
                .map(|i| BabyBear::from_canonical_u32(x * i))
                .collect();
            vec.try_into().unwrap()
        })
        .collect();
    let internal_constants = (1..5).map(BabyBear::from_canonical_u32).collect();
    let poseidon2_air = Poseidon2Air::<16, BabyBear>::new(external_constants, internal_constants);
    let state: Vec<BabyBear> = (0..16).map(BabyBear::from_canonical_u32).collect();
    let state: Vec<[BabyBear; 16]> = vec![state.try_into().unwrap()];
    let trace = poseidon2_air.generate_trace(state);
    println!("{:?}", trace);

    run_simple_test_no_pis(vec![&poseidon2_air], vec![trace]).expect("Verification failed");
}
