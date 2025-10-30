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

    let circuit = VerifierCircuit::<DuplexSpongeRecorder, 2>::new(Arc::new(vk));
    let engine = BabyBearPoseidon2Engine::new(
        FriParameters::standard_with_100_bits_conjectured_security(2),
    );
    let mut keygen_builder = engine.keygen_builder();
    for air in circuit.airs() {
        keygen_builder.add_air(air);
    }
    let pk = keygen_builder.generate_pk();
    let proof_inputs = circuit.generate_proof_inputs(&[proof]);
    engine.debug(&circuit.airs(), &pk.per_air, &proof_inputs);
}

#[test]
fn test_recursion_circuit_interactions() {
    let params = test_system_params_small();
    let engine = BabyBearPoseidon2CpuEngineV2::<DuplexSponge>::new(params);
    let fx = InteractionsFixture11;
    let (vk, proof) = fx.keygen_and_prove(&engine);

    let circuit = VerifierCircuit::<DuplexSpongeRecorder, 2>::new(Arc::new(vk));
    let engine = BabyBearPoseidon2Engine::new(
        FriParameters::standard_with_100_bits_conjectured_security(2),
    );
    let mut keygen_builder = engine.keygen_builder();
    for air in circuit.airs() {
        keygen_builder.add_air(air);
    }
    let pk = keygen_builder.generate_pk();
    let proof_inputs = circuit.generate_proof_inputs(&[proof]);
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
    let circuit = VerifierCircuit::<DuplexSpongeValidator, 2>::new(Arc::new(vk));
    let preflight = circuit.run_preflight(preflight_sponge, &proof);
    assert_eq!(preflight.transcript.len(), prover_sponge_len);
}

#[test]
fn test_preflight_cached_trace() {
    let params = test_system_params_small();
    let engine = BabyBearPoseidon2CpuEngineV2::<DuplexSponge>::new(params);
    let fx = CachedFixture11::new(params);
    let (vk, proof) = fx.keygen_and_prove(&engine);

    let circuit = VerifierCircuit::<DuplexSpongeRecorder, 2>::new(Arc::new(vk));
    let engine = BabyBearPoseidon2Engine::new(
        FriParameters::standard_with_100_bits_conjectured_security(2),
    );
    let mut keygen_builder = engine.keygen_builder();
    for air in circuit.airs() {
        keygen_builder.add_air(air);
    }
    let pk = keygen_builder.generate_pk();
    let proof_inputs = circuit.generate_proof_inputs(&[proof]);
    engine.debug(&circuit.airs(), &pk.per_air, &proof_inputs);
}

#[test]
fn test_preflight_preprocessed_trace() {
    let engine = test_engine_small();
    let height = 1 << 5;
    let sels = (0..height).map(|i| i % 2 == 0).collect_vec();
    let fx = PreprocessedFibFixture::new(0, 1, sels);
    let (vk, proof) = fx.keygen_and_prove(&engine);

    let circuit = VerifierCircuit::<DuplexSpongeRecorder, 2>::new(Arc::new(vk));
    let engine = BabyBearPoseidon2Engine::new(
        FriParameters::standard_with_100_bits_conjectured_security(2),
    );
    let mut keygen_builder = engine.keygen_builder();
    for air in circuit.airs() {
        keygen_builder.add_air(air);
    }
    let pk = keygen_builder.generate_pk();
    let proof_inputs = circuit.generate_proof_inputs(&[proof]);
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
    let circuit = VerifierCircuit::<_, 2>::new(Arc::new(vk));
    let preflight = circuit.run_preflight(preflight_sponge, &proof);
    assert_eq!(preflight.transcript.len(), prover_sponge_len);
}

///////////////////////////////////////////////////////////////////////////////
// Multi-proof tests
///////////////////////////////////////////////////////////////////////////////

#[test]
fn test_recursion_circuit_two_fib_proofs() {
    let params = test_system_params_small();
    let log_trace_degree = 3;

    let engine = BabyBearPoseidon2CpuEngineV2::<DuplexSponge>::new(params);

    let fib1 = FibFixture::new(0, 1, 1 << log_trace_degree);
    let fib2 = FibFixture::new(1, 1, 1 << log_trace_degree);

    let (vk, proof1) = fib1.keygen_and_prove(&engine);
    let (_, proof2) = fib2.keygen_and_prove(&engine);

    let circuit = VerifierCircuit::<DuplexSpongeRecorder, 2>::new(Arc::new(vk));
    let engine = BabyBearPoseidon2Engine::new(
        FriParameters::standard_with_100_bits_conjectured_security(2),
    );
    let mut keygen_builder = engine.keygen_builder();
    for air in circuit.airs() {
        keygen_builder.add_air(air);
    }
    let pk = keygen_builder.generate_pk();
    let proof_inputs = circuit.generate_proof_inputs(&[proof1, proof2]);
    engine.debug(&circuit.airs(), &pk.per_air, &proof_inputs);
}

#[test]
fn test_recursion_circuit_multiple_fib_proofs() {
    let params = test_system_params_small();
    let log_trace_degree = 3;

    let engine = BabyBearPoseidon2CpuEngineV2::<DuplexSponge>::new(params);

    let fib1 = FibFixture::new(0, 1, 1 << log_trace_degree);
    let fib2 = FibFixture::new(1, 1, 1 << log_trace_degree);
    let fib3 = FibFixture::new(2, 3, 1 << log_trace_degree);
    let fib4 = FibFixture::new(3, 5, 1 << log_trace_degree);
    let fib5 = FibFixture::new(5, 8, 1 << log_trace_degree);

    let (vk, proof1) = fib1.keygen_and_prove(&engine);
    let (_, proof2) = fib2.keygen_and_prove(&engine);
    let (_, proof3) = fib3.keygen_and_prove(&engine);
    let (_, proof4) = fib4.keygen_and_prove(&engine);
    let (_, proof5) = fib5.keygen_and_prove(&engine);

    let circuit = VerifierCircuit::<DuplexSpongeRecorder, 5>::new(Arc::new(vk));
    let engine = BabyBearPoseidon2Engine::new(
        FriParameters::standard_with_100_bits_conjectured_security(2),
    );
    let mut keygen_builder = engine.keygen_builder();
    for air in circuit.airs() {
        keygen_builder.add_air(air);
    }
    let pk = keygen_builder.generate_pk();
    let proof_inputs = circuit.generate_proof_inputs(&[proof1, proof2, proof3, proof4, proof5]);
    engine.debug(&circuit.airs(), &pk.per_air, &proof_inputs);
}

#[test]
fn test_recursion_circuit_two_preprocessed() {
    let params = test_system_params_small();
    let engine = BabyBearPoseidon2CpuEngineV2::<DuplexSponge>::new(params);

    let height = 1 << 4;
    let sels = (0..height).map(|i| i % 2 == 0).collect_vec();

    let preprocessed1 = PreprocessedFibFixture::new(0, 1, sels.clone());
    let preprocessed2 = PreprocessedFibFixture::new(1, 1, sels.clone());

    let (vk, proof1) = preprocessed1.keygen_and_prove(&engine);
    let (_, proof2) = preprocessed2.keygen_and_prove(&engine);

    let circuit = VerifierCircuit::<DuplexSpongeRecorder, 2>::new(Arc::new(vk));
    let engine = BabyBearPoseidon2Engine::new(
        FriParameters::standard_with_100_bits_conjectured_security(2),
    );
    let mut keygen_builder = engine.keygen_builder();
    for air in circuit.airs() {
        keygen_builder.add_air(air);
    }
    let pk = keygen_builder.generate_pk();
    let proof_inputs = circuit.generate_proof_inputs(&[proof1, proof2]);
    engine.debug(&circuit.airs(), &pk.per_air, &proof_inputs);
}

#[test]
fn test_recursion_circuit_multiple_preprocessed() {
    let params = test_system_params_small();
    let engine = BabyBearPoseidon2CpuEngineV2::<DuplexSponge>::new(params);

    let height = 1 << 4;
    let sels = (0..height).map(|i| i % 2 == 0).collect_vec();

    let preprocessed1 = PreprocessedFibFixture::new(0, 1, sels.clone());
    let preprocessed2 = PreprocessedFibFixture::new(1, 1, sels.clone());
    let preprocessed3 = PreprocessedFibFixture::new(2, 3, sels.clone());
    let preprocessed4 = PreprocessedFibFixture::new(3, 5, sels.clone());
    let preprocessed5 = PreprocessedFibFixture::new(5, 8, sels.clone());

    let (vk, proof1) = preprocessed1.keygen_and_prove(&engine);
    let (_, proof2) = preprocessed2.keygen_and_prove(&engine);
    let (_, proof3) = preprocessed3.keygen_and_prove(&engine);
    let (_, proof4) = preprocessed4.keygen_and_prove(&engine);
    let (_, proof5) = preprocessed5.keygen_and_prove(&engine);

    let circuit = VerifierCircuit::<DuplexSpongeRecorder, 5>::new(Arc::new(vk));
    let engine = BabyBearPoseidon2Engine::new(
        FriParameters::standard_with_100_bits_conjectured_security(2),
    );
    let mut keygen_builder = engine.keygen_builder();
    for air in circuit.airs() {
        keygen_builder.add_air(air);
    }
    let pk = keygen_builder.generate_pk();
    let proof_inputs = circuit.generate_proof_inputs(&[proof1, proof2, proof3, proof4, proof5]);
    engine.debug(&circuit.airs(), &pk.per_air, &proof_inputs);
}

#[test]
fn test_recursion_circuit_two_interactions() {
    let params = test_system_params_small();
    let engine = BabyBearPoseidon2CpuEngineV2::<DuplexSponge>::new(params);

    let fx = InteractionsFixture11;
    let (vk, proof) = fx.keygen_and_prove(&engine);

    let circuit = VerifierCircuit::<DuplexSpongeRecorder, 2>::new(Arc::new(vk));
    let engine = BabyBearPoseidon2Engine::new(
        FriParameters::standard_with_100_bits_conjectured_security(2),
    );
    let mut keygen_builder = engine.keygen_builder();
    for air in circuit.airs() {
        keygen_builder.add_air(air);
    }
    let pk = keygen_builder.generate_pk();
    let proof_inputs = circuit.generate_proof_inputs(&[proof.clone(), proof]);
    engine.debug(&circuit.airs(), &pk.per_air, &proof_inputs);
}

#[test]
fn test_recursion_circuit_multiple_interactions() {
    let params = test_system_params_small();
    let engine = BabyBearPoseidon2CpuEngineV2::<DuplexSponge>::new(params);

    // Generate multiple interaction proofs - they should use the same VK
    let fx = InteractionsFixture11;
    let (vk, proof) = fx.keygen_and_prove(&engine);

    let circuit = VerifierCircuit::<DuplexSpongeRecorder, 5>::new(Arc::new(vk));
    let engine = BabyBearPoseidon2Engine::new(
        FriParameters::standard_with_100_bits_conjectured_security(2),
    );
    let mut keygen_builder = engine.keygen_builder();
    for air in circuit.airs() {
        keygen_builder.add_air(air);
    }
    let pk = keygen_builder.generate_pk();
    let proof_inputs = circuit.generate_proof_inputs(&[
        proof.clone(),
        proof.clone(),
        proof.clone(),
        proof.clone(),
        proof,
    ]);
    engine.debug(&circuit.airs(), &pk.per_air, &proof_inputs);
}

#[test]
fn test_recursion_circuit_two_cached() {
    let params = test_system_params_small();
    let engine = BabyBearPoseidon2CpuEngineV2::<DuplexSponge>::new(params);

    let fx = CachedFixture11::new(params);
    let (vk, proof) = fx.keygen_and_prove(&engine);

    let circuit = VerifierCircuit::<DuplexSpongeRecorder, 5>::new(Arc::new(vk));
    let engine = BabyBearPoseidon2Engine::new(
        FriParameters::standard_with_100_bits_conjectured_security(2),
    );
    let mut keygen_builder = engine.keygen_builder();
    for air in circuit.airs() {
        keygen_builder.add_air(air);
    }
    let pk = keygen_builder.generate_pk();
    let proof_inputs = circuit.generate_proof_inputs(&[
        proof.clone(),
        proof.clone(),
        proof.clone(),
        proof.clone(),
        proof,
    ]);
    engine.debug(&circuit.airs(), &pk.per_air, &proof_inputs);
}

#[test]
fn test_recursion_circuit_multiple_cached() {
    let params = test_system_params_small();
    let engine = BabyBearPoseidon2CpuEngineV2::<DuplexSponge>::new(params);

    let fx = CachedFixture11::new(params);
    let (vk, proof) = fx.keygen_and_prove(&engine);

    let circuit = VerifierCircuit::<DuplexSpongeRecorder, 5>::new(Arc::new(vk));
    let engine = BabyBearPoseidon2Engine::new(
        FriParameters::standard_with_100_bits_conjectured_security(2),
    );
    let mut keygen_builder = engine.keygen_builder();
    for air in circuit.airs() {
        keygen_builder.add_air(air);
    }
    let pk = keygen_builder.generate_pk();
    let proof_inputs = circuit.generate_proof_inputs(&[
        proof.clone(),
        proof.clone(),
        proof.clone(),
        proof.clone(),
        proof,
    ]);
    engine.debug(&circuit.airs(), &pk.per_air, &proof_inputs);
}
