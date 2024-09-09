use ax_sdk::config::{baby_bear_poseidon2_outer::BabyBearPoseidon2OuterConfig, FriParameters};
use snark_verifier_sdk::Snark;

use crate::{
    config::outer::new_from_outer_multi_vk,
    halo2::verifier::{generate_halo2_verifier_circuit, Halo2VerifierCircuit},
    testing_utils::{outer::make_verification_data, StarkForTest},
    types::VerifierInput,
};

pub fn run_static_verifier_test(
    stark_for_test: &StarkForTest<BabyBearPoseidon2OuterConfig>,
    fri_params: FriParameters,
) -> (Halo2VerifierCircuit, Snark) {
    let StarkForTest {
        any_raps,
        traces,
        pvs,
    } = stark_for_test;
    let any_raps: Vec<_> = any_raps.iter().map(|x| x.as_ref()).collect();
    let vparams = make_verification_data(&any_raps, traces.clone(), pvs, fri_params);

    // Build verification program in eDSL.
    let advice = new_from_outer_multi_vk(&vparams.vk);
    let log_degree_per_air = vparams.proof.log_degrees();
    let input = VerifierInput {
        proof: vparams.proof,
        log_degree_per_air,
        public_values: pvs.clone(),
    };

    let info_span = tracing::info_span!("keygen halo2 verifier circuit").entered();
    let stark_verifier_circuit = generate_halo2_verifier_circuit(21, advice, &fri_params, &input);
    info_span.exit();

    let info_span = tracing::info_span!("prove halo2 verifier circuit").entered();
    let static_verifier_snark = stark_verifier_circuit.prove(input);
    info_span.exit();
    (stark_verifier_circuit, static_verifier_snark)
}
