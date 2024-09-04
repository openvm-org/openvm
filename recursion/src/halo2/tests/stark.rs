use afs_compiler::ir::Witness;
use ax_sdk::config::{
    baby_bear_poseidon2_outer::BabyBearPoseidon2OuterConfig, fri_params::default_fri_params,
    setup_tracing,
};

use crate::{
    config::outer::new_from_outer_multi_vk,
    halo2::Halo2Prover,
    stark::outer::build_circuit_verify_operations,
    testing_utils::{outer::make_verification_data, StarkForTest},
    tests::{fibonacci_stark_for_test, interaction_stark_for_test},
    types::VerifierInput,
    witness::Witnessable,
};

#[test]
fn test_fibonacci() {
    setup_tracing();
    run_recursive_test(&fibonacci_stark_for_test::<BabyBearPoseidon2OuterConfig>(
        16,
    ))
}

#[test]
fn test_interactions() {
    // Please make sure kzg trusted params are downloaded before running the test.
    setup_tracing();
    run_recursive_test(&interaction_stark_for_test::<BabyBearPoseidon2OuterConfig>())
}

fn run_recursive_test(stark_for_test: &StarkForTest<BabyBearPoseidon2OuterConfig>) {
    let StarkForTest {
        any_raps,
        traces,
        pvs,
    } = stark_for_test;
    let any_raps: Vec<_> = any_raps.iter().map(|x| x.as_ref()).collect();

    let fri_params = default_fri_params();
    let vparams = make_verification_data(&any_raps, traces.clone(), pvs, fri_params);

    let advice = new_from_outer_multi_vk(&vparams.vk);
    let log_degree_per_air = vparams.proof.log_degrees();

    let input = VerifierInput {
        proof: vparams.proof,
        log_degree_per_air,
        public_values: pvs.clone(),
    };

    let mut witness = Witness::default();
    input.write(&mut witness);
    let operations = build_circuit_verify_operations(advice, &fri_params, &input);
    Halo2Prover::mock(20, operations, witness);
}
