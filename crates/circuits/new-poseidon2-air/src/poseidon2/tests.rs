use std::sync::Arc;

use openvm_stark_backend::p3_field::AbstractField;
use openvm_stark_sdk::{
    config::{
        baby_bear_poseidon2::{engine_from_perm, random_perm},
        fri_params::standard_fri_params_with_100_bits_conjectured_security,
        setup_tracing_with_log_level,
    },
    engine::StarkEngine,
    p3_baby_bear::BabyBear,
    utils::create_seeded_rng,
};
use rand::RngCore;
use tracing::Level;

use super::{Poseidon2Config, Poseidon2SubChip};

#[test]
fn test_poseidon2_default() {
    // config
    let num_rows = 1 << 1;

    // random constants, state generation
    let mut rng = create_seeded_rng();
    let states: Vec<[BabyBear; 16]> = (0..num_rows)
        .map(|_| {
            let vec: Vec<BabyBear> = (0..16)
                .map(|_| BabyBear::from_canonical_u32(rng.next_u32() % (1 << 30)))
                .collect();
            vec.try_into().unwrap()
        })
        .collect();

    let poseidon2_subchip = Arc::new(Poseidon2SubChip::<BabyBear, 0>::new(
        Poseidon2Config::default(),
    ));
    let poseidon2_trace = poseidon2_subchip.generate_trace(states.clone());

    // engine generation
    let perm = random_perm();
    let fri_params = standard_fri_params_with_100_bits_conjectured_security(3); // max constraint degree = 7 requires log blowup = 3
    let engine = engine_from_perm(perm, fri_params);

    setup_tracing_with_log_level(Level::DEBUG);

    // positive test
    engine
        .run_simple_test_impl(
            vec![poseidon2_subchip.air.clone()],
            vec![poseidon2_trace.clone()],
            vec![vec![]],
        )
        .expect("Verification failed");
}
