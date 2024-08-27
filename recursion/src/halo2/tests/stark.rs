use afs_compiler::ir::Witness;
use afs_stark_backend::{prover::trace::TraceCommitmentBuilder, verifier::MultiTraceStarkVerifier};
use afs_test_utils::{
    config::{
        baby_bear_poseidon2_outer::{default_engine, BabyBearPoseidon2OuterConfig},
        setup_tracing,
    },
    engine::StarkEngine,
};
use p3_matrix::Matrix;
use p3_util::log2_strict_usize;

use crate::{
    config::outer::new_from_outer_multi_vk,
    halo2::Halo2Prover,
    stark::outer::build_circuit_verify_operations,
    tests::{fibonacci_stark_for_test, interaction_stark_for_test, StarkForTest},
    types::VerifierInput,
    witness::Witnessable,
};

#[test]
fn test_fibonacci() {
    setup_tracing();
    run_recursive_test(&fibonacci_stark_for_test::<BabyBearPoseidon2OuterConfig>())
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

    let num_pvs: Vec<usize> = pvs.iter().map(|pv| pv.len()).collect();

    let trace_heights: Vec<usize> = traces.iter().map(|t| t.height()).collect();
    let log_degree = log2_strict_usize(trace_heights.clone().into_iter().max().unwrap());

    let engine = default_engine(log_degree);

    let mut keygen_builder = engine.keygen_builder();
    for (&rap, &num_pv) in any_raps.iter().zip(num_pvs.iter()) {
        keygen_builder.add_air(rap, num_pv);
    }

    let pk = keygen_builder.generate_pk();
    let vk = pk.vk();

    let prover = engine.prover();
    let mut trace_builder = TraceCommitmentBuilder::new(prover.pcs());
    for trace in traces.clone() {
        trace_builder.load_trace(trace);
    }
    trace_builder.commit_current();

    let main_trace_data = trace_builder.view(&vk, any_raps.clone());

    let mut challenger = engine.new_challenger();
    let proof = prover.prove(&mut challenger, &pk, main_trace_data, &pvs);
    let log_degree_per_air = proof
        .degrees
        .iter()
        .map(|degree| log2_strict_usize(*degree))
        .collect();
    // Make sure proof verifies outside eDSL...
    let verifier = MultiTraceStarkVerifier::new(prover.config);
    verifier
        .verify(&mut engine.new_challenger(), &vk, &proof, &pvs)
        .expect("afs proof should verify");

    // Build verification program in eDSL.
    let advice = new_from_outer_multi_vk(&vk);
    let input = VerifierInput {
        proof,
        log_degree_per_air,
        public_values: pvs.clone(),
    };

    let mut witness = Witness::default();
    input.write(&mut witness);
    let operations = build_circuit_verify_operations(advice, &engine.fri_params, &input);
    Halo2Prover::mock(20, operations, witness);
}
