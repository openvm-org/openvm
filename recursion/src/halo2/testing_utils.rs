use ax_sdk::{
    config::{
        baby_bear_poseidon2_outer::{BabyBearPoseidon2OuterConfig, BabyBearPoseidon2OuterEngine},
        fri_params::standard_fri_params_with_100_bits_conjectured_security,
        FriParameters,
    },
    engine::{StarkForTest, StarkFriEngine},
};
use snark_verifier_sdk::Snark;

use crate::{
    config::outer::new_from_outer_multi_vk,
    halo2::{
        utils::sort_chips,
        verifier::{
            gen_wrapper_circuit_evm_proof, generate_halo2_verifier_circuit, Halo2VerifierCircuit,
        },
    },
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
    let (any_raps, traces, pvs) = sort_chips(any_raps, traces.clone(), pvs.clone());
    let info_span =
        tracing::info_span!("prove outer stark to verify", step = "outer_stark_prove").entered();
    let vparams = BabyBearPoseidon2OuterEngine::new(fri_params)
        .run_simple_test_impl(&any_raps, traces.clone(), &pvs)
        .unwrap();
    info_span.exit();

    // Build verification program in eDSL.
    let advice = new_from_outer_multi_vk(&vparams.data.vk);
    let log_degree_per_air = vparams.data.proof.log_degrees();
    let input = VerifierInput {
        proof: vparams.data.proof,
        log_degree_per_air,
    };

    let info_span = tracing::info_span!(
        "keygen halo2 verifier circuit",
        step = "static_verifier_keygen"
    )
    .entered();
    let stark_verifier_circuit =
        generate_halo2_verifier_circuit(21, advice, &vparams.fri_params, &input);
    info_span.exit();

    let info_span = tracing::info_span!(
        "prove halo2 verifier circuit",
        step = "static_verifier_prove"
    )
    .entered();
    let static_verifier_snark = stark_verifier_circuit.prove(input);
    info_span.exit();
    (stark_verifier_circuit, static_verifier_snark)
}

pub fn run_evm_verifier_e2e_test(
    stark_for_test: &StarkForTest<BabyBearPoseidon2OuterConfig>,
    fri_params: Option<FriParameters>,
) {
    let (stark_verifier_circuit, static_verifier_snark) = run_static_verifier_test(
        stark_for_test,
        fri_params.unwrap_or(standard_fri_params_with_100_bits_conjectured_security(3)),
    );

    let info_span = tracing::info_span!(
        "keygen halo2 wrapper circuit",
        step = "static_verifier_wrapper_keygen"
    )
    .entered();
    let keygen_circuit =
        stark_verifier_circuit.keygen_wrapper_circuit(23, static_verifier_snark.clone());
    info_span.exit();

    let info_span = tracing::info_span!(
        "prove halo2 wrapper circuit",
        step = "static_verifier_wrapper_prove"
    )
    .entered();
    #[cfg(debug_assertions)]
    let _ = gen_wrapper_circuit_evm_proof(&keygen_circuit, static_verifier_snark);
    #[cfg(not(debug_assertions))]
    let (wrapper_evm_proof, pvs) =
        gen_wrapper_circuit_evm_proof(&keygen_circuit, static_verifier_snark);
    info_span.exit();

    // REVM is incompatible with our rust version. evm_verify will panic if it's running in debug mode.
    #[cfg(not(debug_assertions))]
    {
        let info_span = tracing::info_span!(
            "generate halo2 wrapper circuit evm verifier",
            step = "evm_verifier_codegen"
        )
        .entered();
        let evm_verifier = super::verifier::gen_wrapper_circuit_evm_verifier(&keygen_circuit);
        info_span.exit();

        snark_verifier_sdk::evm::evm_verify(evm_verifier, pvs, wrapper_evm_proof);
    }
}
