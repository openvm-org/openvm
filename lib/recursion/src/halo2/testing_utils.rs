use ax_sdk::{
    config::{
        baby_bear_poseidon2_outer::{BabyBearPoseidon2OuterConfig, BabyBearPoseidon2OuterEngine},
        fri_params::standard_fri_params_with_100_bits_conjectured_security,
        FriParameters,
    },
    engine::{ProofInputForTest, StarkFriEngine},
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
};

pub fn run_static_verifier_test(
    test_proof_input: ProofInputForTest<BabyBearPoseidon2OuterConfig>,
    fri_params: FriParameters,
) -> (Halo2VerifierCircuit, Snark) {
    let test_proof_input = ProofInputForTest {
        per_air: sort_chips(test_proof_input.per_air),
    };
    let info_span =
        tracing::info_span!("prove outer stark to verify", step = "outer_stark_prove").entered();
    let engine = BabyBearPoseidon2OuterEngine::new(fri_params);
    let vparams = test_proof_input.run_test(&engine).unwrap();

    info_span.exit();

    // Build verification program in eDSL.
    let advice = new_from_outer_multi_vk(&vparams.data.vk);

    let info_span = tracing::info_span!(
        "keygen halo2 verifier circuit",
        step = "static_verifier_keygen"
    )
    .entered();
    let stark_verifier_circuit =
        generate_halo2_verifier_circuit(21, advice, &vparams.fri_params, &vparams.data.proof);
    info_span.exit();

    let info_span = tracing::info_span!(
        "prove halo2 verifier circuit",
        step = "static_verifier_prove"
    )
    .entered();
    let static_verifier_snark = stark_verifier_circuit.prove(vparams.data.proof);
    info_span.exit();
    (stark_verifier_circuit, static_verifier_snark)
}

pub fn run_evm_verifier_e2e_test(
    test_proof_input: ProofInputForTest<BabyBearPoseidon2OuterConfig>,
    fri_params: Option<FriParameters>,
) {
    let (stark_verifier_circuit, static_verifier_snark) = run_static_verifier_test(
        test_proof_input,
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
