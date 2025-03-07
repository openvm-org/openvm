use std::{array::from_fn, sync::Arc};

use openvm_stark_backend::{
    interaction::LogUpSecurityParameters, p3_air::BaseAir, p3_field::FieldAlgebra,
    utils::disable_debug_builder, verifier::VerificationError,
};
use openvm_stark_sdk::{
    config::{
        baby_bear_poseidon2::{engine_from_perm, random_perm},
        fri_params::{standard_fri_params_with_100_bits_conjectured_security, SecurityParameters},
    },
    engine::StarkFriEngine,
    p3_baby_bear::BabyBear,
    utils::create_seeded_rng,
};
use p3_poseidon2::ExternalLayerConstants;
use rand::{rngs::StdRng, Rng, RngCore};

use super::{Poseidon2Config, Poseidon2Constants, Poseidon2SubChip};
use crate::BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS;

fn run_poseidon2_subchip_test(subchip: Arc<Poseidon2SubChip<BabyBear, 0>>, rng: &mut StdRng) {
    // random state and trace generation
    let num_rows = 1 << 4;
    let states: Vec<[BabyBear; 16]> = (0..num_rows)
        .map(|_| {
            let vec: Vec<BabyBear> = (0..16)
                .map(|_| BabyBear::from_canonical_u32(rng.next_u32() % (1 << 30)))
                .collect();
            vec.try_into().unwrap()
        })
        .collect();
    let mut poseidon2_trace = subchip.generate_trace(states.clone());

    // engine generation
    let perm = random_perm();
    let fri_params = standard_fri_params_with_100_bits_conjectured_security(3); // max constraint degree = 7 requires log blowup = 3
    let security_params = SecurityParameters {
        fri_params,
        log_up_params: LogUpSecurityParameters::default(),
    };
    let engine = engine_from_perm(perm, security_params);

    // positive test
    engine
        .run_simple_test_impl(
            vec![subchip.air.clone()],
            vec![poseidon2_trace.clone()],
            vec![vec![]],
        )
        .expect("Verification failed");

    // negative test
    disable_debug_builder();
    for _ in 0..10 {
        let rand_idx = rng.gen_range(0..subchip.air.width());
        let rand_inc = BabyBear::from_canonical_u32(rng.gen_range(1..=1 << 27));
        poseidon2_trace.row_mut((1 << 4) - 1)[rand_idx] += rand_inc;
        assert_eq!(
            engine
                .run_simple_test_impl(
                    vec![subchip.air.clone()],
                    vec![poseidon2_trace.clone()],
                    vec![vec![]],
                )
                .err(),
            Some(VerificationError::OodEvaluationMismatch),
            "Expected constraint to fail"
        );
        poseidon2_trace.row_mut((1 << 4) - 1)[rand_idx] -= rand_inc;
    }
}

#[test]
fn test_poseidon2_default() {
    let mut rng = create_seeded_rng();
    let poseidon2_config = Poseidon2Config::default();
    let poseidon2_subchip = Arc::new(Poseidon2SubChip::<BabyBear, 0>::new(
        poseidon2_config.constants,
    ));
    run_poseidon2_subchip_test(poseidon2_subchip, &mut rng);
}

#[test]
fn test_poseidon2_random_constants() {
    let mut rng = create_seeded_rng();
    let external_constants =
        ExternalLayerConstants::new_from_rng(2 * BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS, &mut rng);
    let beginning_full_round_constants_vec = external_constants.get_initial_constants();
    let beginning_full_round_constants = from_fn(|i| beginning_full_round_constants_vec[i]);
    let ending_full_round_constants_vec = external_constants.get_terminal_constants();
    let ending_full_round_constants = from_fn(|i| ending_full_round_constants_vec[i]);
    let partial_round_constants = from_fn(|_| BabyBear::from_wrapped_u32(rng.next_u32()));
    let constants = Poseidon2Constants {
        beginning_full_round_constants,
        partial_round_constants,
        ending_full_round_constants,
    };
    let poseidon2_subchip = Arc::new(Poseidon2SubChip::<BabyBear, 0>::new(constants));
    run_poseidon2_subchip_test(poseidon2_subchip, &mut rng);
}
