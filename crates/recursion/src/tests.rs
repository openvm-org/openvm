use openvm_stark_backend::engine::StarkEngine;
use openvm_stark_sdk::{
    config::{FriParameters, baby_bear_poseidon2::BabyBearPoseidon2Engine},
    engine::StarkFriEngine,
};
use stark_backend_v2::{
    poseidon2::sponge::DuplexSponge,
    test_utils::{FibFixture, test_system_params_small},
};

use crate::system::VerifierCircuit;

#[test]
fn test_recursion_circuit_single_fib() {
    let log_trace_degree = 3;

    let fib = FibFixture::build(test_system_params_small(), 0, 1, 1 << log_trace_degree);
    let proof = fib.prove();

    let circuit = VerifierCircuit::new();
    let engine = BabyBearPoseidon2Engine::new(FriParameters::standard_fast());
    let mut keygen_builder = engine.keygen_builder();
    for air in circuit.airs() {
        keygen_builder.add_air(air);
    }
    let pk = keygen_builder.generate_pk();
    let proof_inputs = circuit.generate_proof_inputs(&fib.vk, &proof);
    engine.debug(&circuit.airs(), &pk.per_air, &proof_inputs);
}

#[test]
fn test_preflight_single_fib() {
    let fib = FibFixture::build(test_system_params_small(), 0, 1, 1 << 5);

    let mut prover_sponge = DuplexSponge::default();
    let proof = fib.prove_with_sponge(&mut prover_sponge);

    let circuit = VerifierCircuit::new();
    let mut preflight = circuit.run_preflight(&fib.vk, &proof);

    let pro = prover_sponge.sample();
    let pre = preflight.transcript.sponge.sample();
    assert_eq!(pro, pre);
}
