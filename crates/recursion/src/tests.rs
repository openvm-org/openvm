use std::sync::Arc;

use itertools::Itertools;
use openvm_stark_backend::engine::StarkEngine;
use openvm_stark_sdk::{
    config::{FriParameters, baby_bear_poseidon2::BabyBearPoseidon2Engine},
    engine::StarkFriEngine,
};
use stark_backend_v2::{
    BabyBearPoseidon2CpuEngineV2,
    poseidon2::sponge::{DuplexSponge, DuplexSpongeRecorder, TranscriptHistory},
    test_utils::{
        CachedFixture11, DuplexSpongeValidator, FibFixture, InteractionsFixture11,
        PreprocessedFibFixture, TestFixture, test_engine_small, test_system_params_small,
    },
};

use crate::system::VerifierCircuit;

#[test]
fn test_recursion_circuit_single_fib() {
    let params = test_system_params_small();
    let log_trace_degree = 3;

    let engine = BabyBearPoseidon2CpuEngineV2::<DuplexSponge>::new(params);
    let fib = FibFixture::new(0, 1, 1 << log_trace_degree);
    let (vk, proof) = fib.keygen_and_prove(&engine);

    let sponge = DuplexSpongeRecorder::default();
    let circuit = VerifierCircuit::new(Arc::new(vk));
    let engine = BabyBearPoseidon2Engine::new(FriParameters::standard_fast());
    let mut keygen_builder = engine.keygen_builder();
    for air in circuit.airs() {
        keygen_builder.add_air(air);
    }
    let pk = keygen_builder.generate_pk();
    let proof_inputs = circuit.generate_proof_inputs(sponge, &proof);
    engine.debug(&circuit.airs(), &pk.per_air, &proof_inputs);
}

#[test]
fn test_recursion_circuit_interactions() {
    let params = test_system_params_small();
    let engine = BabyBearPoseidon2CpuEngineV2::<DuplexSponge>::new(params);
    let fx = InteractionsFixture11;
    let (vk, proof) = fx.keygen_and_prove(&engine);

    let circuit = VerifierCircuit::new(Arc::new(vk));
    let engine = BabyBearPoseidon2Engine::new(FriParameters::standard_fast());
    let mut keygen_builder = engine.keygen_builder();
    for air in circuit.airs() {
        keygen_builder.add_air(air);
    }
    let pk = keygen_builder.generate_pk();
    let sponge = DuplexSpongeRecorder::default();
    let proof_inputs = circuit.generate_proof_inputs(sponge, &proof);
    engine.debug(&circuit.airs(), &pk.per_air, &proof_inputs);
}

#[test]
fn test_preflight_single_fib_sponge() {
    let params = test_system_params_small();
    let engine = BabyBearPoseidon2CpuEngineV2::<DuplexSpongeRecorder>::new(params);
    let fib = FibFixture::new(0, 1, 1 << 5);
    let (pk, vk) = fib.keygen(&engine);

    let mut prover_sponge = DuplexSpongeRecorder::default();
    let proof = fib.prove_from_transcript(&engine, &pk, &mut prover_sponge);
    let prover_sponge_len = prover_sponge.len();

    let preflight_sponge = DuplexSpongeValidator::new(prover_sponge.into_log());
    let circuit = VerifierCircuit::<DuplexSpongeValidator>::new(Arc::new(vk));
    let preflight = circuit.run_preflight(preflight_sponge, &proof);
    assert_eq!(preflight.transcript.len(), prover_sponge_len);
}

#[test]
fn test_preflight_cached_trace() {
    let params = test_system_params_small();
    let engine = BabyBearPoseidon2CpuEngineV2::<DuplexSponge>::new(params);
    let fx = CachedFixture11::new(params);
    let (vk, proof) = fx.keygen_and_prove(&engine);

    let circuit = VerifierCircuit::new(Arc::new(vk));
    let engine = BabyBearPoseidon2Engine::new(FriParameters::standard_fast());
    let mut keygen_builder = engine.keygen_builder();
    for air in circuit.airs() {
        keygen_builder.add_air(air);
    }
    let pk = keygen_builder.generate_pk();
    let sponge = DuplexSpongeRecorder::default();
    let proof_inputs = circuit.generate_proof_inputs(sponge, &proof);
    engine.debug(&circuit.airs(), &pk.per_air, &proof_inputs);
}

#[test]
fn test_preflight_preprocessed_trace() {
    let engine = test_engine_small();
    let height = 1 << 5;
    let sels = (0..height).map(|i| i % 2 == 0).collect_vec();
    let fx = PreprocessedFibFixture::new(0, 1, sels);
    let (vk, proof) = fx.keygen_and_prove(&engine);

    let circuit = VerifierCircuit::new(Arc::new(vk));
    let engine = BabyBearPoseidon2Engine::new(FriParameters::standard_fast());
    let mut keygen_builder = engine.keygen_builder();
    for air in circuit.airs() {
        keygen_builder.add_air(air);
    }
    let pk = keygen_builder.generate_pk();
    let sponge = DuplexSpongeRecorder::default();
    let proof_inputs = circuit.generate_proof_inputs(sponge, &proof);
    engine.debug(&circuit.airs(), &pk.per_air, &proof_inputs);
}

#[test]
fn test_preflight_interactions() {
    let params = test_system_params_small();
    let engine = BabyBearPoseidon2CpuEngineV2::<DuplexSpongeRecorder>::new(params);
    let fx = InteractionsFixture11;
    let (pk, vk) = fx.keygen(&engine);

    let mut prover_sponge = DuplexSpongeRecorder::default();
    let proof = fx.prove_from_transcript(&engine, &pk, &mut prover_sponge);
    let prover_sponge_len = prover_sponge.len();

    let preflight_sponge = DuplexSpongeValidator::new(prover_sponge.into_log());
    let circuit = VerifierCircuit::new(Arc::new(vk));
    let preflight = circuit.run_preflight(preflight_sponge, &proof);
    assert_eq!(preflight.transcript.len(), prover_sponge_len);
}
