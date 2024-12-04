use ax_stark_sdk::{
    config::{
        baby_bear_poseidon2_outer::{BabyBearPoseidon2OuterConfig, BabyBearPoseidon2OuterEngine},
        FriParameters,
    },
    engine::{ProofInputForTest, StarkFriEngine},
};
use axvm_native_compiler::prelude::Witness;
use snark_verifier_sdk::Snark;

use crate::{
    config::outer::new_from_outer_multi_vk,
    halo2::{
        utils::sort_chips,
        verifier::{generate_halo2_verifier_circuit, Halo2VerifierCircuit},
    },
    witness::Witnessable,
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
    let mut witness = Witness::default();
    vparams.data.proof.write(&mut witness);
    let static_verifier_snark = stark_verifier_circuit.prove(witness);
    info_span.exit();
    (stark_verifier_circuit, static_verifier_snark)
}
