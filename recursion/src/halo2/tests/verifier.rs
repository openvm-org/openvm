/// Run with RANDOM_SRS=1 if you don't want to download the SRS.
use afs_stark_backend::{prover::trace::TraceCommitmentBuilder, verifier::MultiTraceStarkVerifier};
use ax_sdk::{
    config::{
        baby_bear_poseidon2_outer::{default_engine, BabyBearPoseidon2OuterConfig},
        setup_tracing,
    },
    engine::StarkEngine,
};
use p3_matrix::Matrix;
use p3_util::log2_strict_usize;
use snark_verifier_sdk::evm::evm_verify;

use crate::{
    config::outer::new_from_outer_multi_vk,
    halo2::verifier::{
        gen_wrapper_circuit_evm_proof, gen_wrapper_circuit_evm_verifier,
        generate_halo2_verifier_circuit,
    },
    tests::{fibonacci_stark_for_test, StarkForTest},
    types::VerifierInput,
};

#[cfg(not(debug_assertions))]
#[test]
fn fibonacci_evm_verifier_e2e() {
    setup_tracing();
    run_recursive_test(&fibonacci_stark_for_test::<BabyBearPoseidon2OuterConfig>())
}

// REVM is incompatible with our rust version. evm_verify will panic if it's running in debug mode.
#[cfg(not(debug_assertions))]
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

    let info_span = tracing::info_span!("keygen halo2 verifier circuit").entered();
    let stark_verifier_circuit =
        generate_halo2_verifier_circuit(21, advice, &engine.fri_params, &input);
    info_span.exit();

    let info_span = tracing::info_span!("prove halo2 verifier circuit").entered();
    let static_verifier_snark = stark_verifier_circuit.prove(input);
    info_span.exit();

    let info_span = tracing::info_span!("keygen halo2 wrapper circuit").entered();
    let keygen_circuit =
        stark_verifier_circuit.keygen_wrapper_circuit(23, static_verifier_snark.clone());
    info_span.exit();

    let info_span = tracing::info_span!("prove halo2 wrapper circuit").entered();
    let (wrapper_evm_proof, pvs) =
        gen_wrapper_circuit_evm_proof(&keygen_circuit, static_verifier_snark);
    info_span.exit();

    let info_span = tracing::info_span!("generate halo2 wrapper circuit evm verifier").entered();
    let evm_verifier = gen_wrapper_circuit_evm_verifier(&keygen_circuit);
    info_span.exit();

    evm_verify(evm_verifier, pvs, wrapper_evm_proof);
}
