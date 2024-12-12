use ax_stark_sdk::{
    config::{
        baby_bear_poseidon2_root::{BabyBearPoseidon2RootConfig, BabyBearPoseidon2RootEngine},
        FriParameters,
    },
    engine::{ProofInputForTest, StarkFriEngine},
};
use axvm_native_compiler::prelude::Witness;
use snark_verifier_sdk::Snark;

use crate::{
    config::outer::new_from_outer_multi_vk,
    halo2::{
        utils::{sort_chips, CacheHalo2ParamsReader, Halo2ParamsReader},
        verifier::{generate_halo2_verifier_proving_key, Halo2VerifierProvingKey},
    },
    witness::Witnessable,
};

pub fn run_static_verifier_test(
    test_proof_input: ProofInputForTest<BabyBearPoseidon2RootConfig>,
    fri_params: FriParameters,
) -> (Halo2VerifierProvingKey, Snark) {
    let k = 21;
    let halo2_params_reader = CacheHalo2ParamsReader::new_with_default_params_dir();
    let params = &halo2_params_reader.read_params(k);
    let test_proof_input = ProofInputForTest {
        per_air: sort_chips(test_proof_input.per_air),
    };
    let info_span =
        tracing::info_span!("prove outer stark to verify", step = "outer_stark_prove").entered();
    let engine = BabyBearPoseidon2RootEngine::new(fri_params);
    let vparams = test_proof_input.run_test(&engine).unwrap();

    info_span.exit();

    // Build verification program in eDSL.
    let advice = new_from_outer_multi_vk(&vparams.data.vk);

    let info_span = tracing::info_span!(
        "keygen halo2 verifier circuit",
        step = "static_verifier_keygen"
    )
    .entered();
    let stark_verifier_circuit = generate_halo2_verifier_proving_key(
        params,
        advice,
        &vparams.fri_params,
        &vparams.data.proof,
    );
    info_span.exit();

    let info_span = tracing::info_span!(
        "prove halo2 verifier circuit",
        step = "static_verifier_prove"
    )
    .entered();
    let mut witness = Witness::default();
    vparams.data.proof.write(&mut witness);
    let static_verifier_snark = stark_verifier_circuit.prove(params, witness);
    info_span.exit();
    (stark_verifier_circuit, static_verifier_snark)
}
