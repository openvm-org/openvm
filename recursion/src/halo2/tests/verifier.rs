/// Run with RANDOM_SRS=1 if you don't want to download the SRS.
use ax_sdk::config::{baby_bear_poseidon2_outer::BabyBearPoseidon2OuterConfig, setup_tracing};
use snark_verifier_sdk::evm::evm_verify;

use crate::{
    halo2::{
        testing_utils::run_static_verifier_test,
        verifier::{gen_wrapper_circuit_evm_proof, gen_wrapper_circuit_evm_verifier},
    },
    testing_utils::StarkForTest,
    tests::fibonacci_stark_for_test,
};

#[test]
fn fibonacci_evm_verifier_e2e() {
    setup_tracing();
    run_evm_verifier_e2e_test(&fibonacci_stark_for_test::<BabyBearPoseidon2OuterConfig>(
        16,
    ))
}

fn run_evm_verifier_e2e_test(stark_for_test: &StarkForTest<BabyBearPoseidon2OuterConfig>) {
    let (stark_verifier_circuit, static_verifier_snark) = run_static_verifier_test(stark_for_test);

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
