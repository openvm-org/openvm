use std::sync::Arc;

use itertools::Itertools;
use openvm_stark_backend::{
    AirRef,
    engine::StarkEngine,
    keygen::types::{MultiStarkProvingKey, StarkProvingKey},
    prover::{MatrixDimensions, types::AirProofRawInput},
};
use openvm_stark_sdk::{
    config::{
        FriParameters,
        baby_bear_poseidon2::{BabyBearPoseidon2Config, BabyBearPoseidon2Engine},
    },
    engine::StarkFriEngine,
};
use p3_field::FieldAlgebra;
use p3_matrix::dense::RowMajorMatrix;
use stark_backend_v2::{
    BabyBearPoseidon2CpuEngineV2, F,
    keygen::types::MultiStarkVerifyingKeyV2,
    poseidon2::sponge::{DuplexSponge, DuplexSpongeRecorder, TranscriptHistory},
    prover::{AirProvingContextV2, ColMajorMatrixView, CpuBackendV2, MatrixView},
    test_utils::{
        CachedFixture11, DuplexSpongeValidator, FibFixture, InteractionsFixture11,
        PreprocessedFibFixture, TestFixture, test_engine_small, test_system_params_small,
    },
};

use crate::system::VerifierSubCircuit;

// TODO[jpw]: switch to v2 types (currently using v1 for debugging)
fn verifier_circuit_keygen<const MAX_NUM_PROOFS: usize>(
    child_vk: &MultiStarkVerifyingKeyV2,
) -> (
    VerifierSubCircuit<MAX_NUM_PROOFS>,
    MultiStarkProvingKey<BabyBearPoseidon2Config>,
) {
    let circuit = VerifierSubCircuit::new(Arc::new(child_vk.clone()));
    let engine_v1 = BabyBearPoseidon2Engine::new(
        FriParameters::standard_with_100_bits_conjectured_security(2),
    );
    let mut keygen_builder = engine_v1.keygen_builder();
    for air in circuit.airs() {
        keygen_builder.add_air(air);
    }
    (circuit, keygen_builder.generate_pk())
}

fn debug(
    airs: &[AirRef<BabyBearPoseidon2Config>],
    pk: &[StarkProvingKey<BabyBearPoseidon2Config>],
    ctxs: &[AirProvingContextV2<CpuBackendV2>],
) {
    for (air_idx, air) in airs.iter().enumerate() {
        tracing::debug!(%air_idx, air_name = %air.name());
    }
    let transpose = |mat: ColMajorMatrixView<F>| {
        let mut values = F::zero_vec(mat.values.len());
        let width = mat.width();
        let height = mat.height();
        for r in 0..height {
            for c in 0..width {
                values[r * width + c] = *mat.get(r, c).unwrap();
            }
        }
        Arc::new(RowMajorMatrix::new(values, width))
    };
    let engine = BabyBearPoseidon2Engine::new(FriParameters::standard_fast());
    let inputs = ctxs
        .iter()
        .map(|ctx| AirProofRawInput {
            cached_mains: ctx
                .cached_mains
                .iter()
                .map(|(_, d)| transpose(d.layout.mat_view(0, d.matrix.as_view())))
                .collect_vec(),
            common_main: Some(transpose(ctx.common_main.as_view())),
            public_values: ctx.public_values.clone(),
        })
        .collect_vec();
    engine.debug(airs, pk, &inputs);
}

#[test]
fn test_recursion_circuit_single_fib() {
    let params = test_system_params_small();
    let log_trace_degree = 3;

    let engine = BabyBearPoseidon2CpuEngineV2::<DuplexSponge>::new(params);
    let fib = FibFixture::new(0, 1, 1 << log_trace_degree);
    let (vk, proof) = fib.keygen_and_prove(&engine);

    let (circuit, pk) = verifier_circuit_keygen::<2>(&vk);
    let vk_commit_data = circuit.commit_child_vk(&engine, &vk);
    let ctxs = circuit.generate_proving_ctxs::<DuplexSpongeRecorder>(&vk, vk_commit_data, &[proof]);
    debug(&circuit.airs(), &pk.per_air, &ctxs);
}

#[test]
fn test_recursion_circuit_interactions() {
    let params = test_system_params_small();
    let engine = BabyBearPoseidon2CpuEngineV2::<DuplexSponge>::new(params);
    let fx = InteractionsFixture11;
    let (vk, proof) = fx.keygen_and_prove(&engine);

    let (circuit, pk) = verifier_circuit_keygen::<2>(&vk);
    let vk_commit_data = circuit.commit_child_vk(&engine, &vk);
    let ctxs = circuit.generate_proving_ctxs::<DuplexSpongeRecorder>(&vk, vk_commit_data, &[proof]);
    debug(&circuit.airs(), &pk.per_air, &ctxs);
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
    let circuit = VerifierSubCircuit::<2>::new(Arc::new(vk.clone()));
    let preflight = circuit.run_preflight(preflight_sponge, &vk, &proof);
    assert_eq!(preflight.transcript.len(), prover_sponge_len);
}

#[test]
fn test_preflight_cached_trace() {
    let params = test_system_params_small();
    let engine = BabyBearPoseidon2CpuEngineV2::<DuplexSponge>::new(params);
    let fx = CachedFixture11::new(params);
    let (vk, proof) = fx.keygen_and_prove(&engine);

    let (circuit, pk) = verifier_circuit_keygen::<2>(&vk);
    let vk_commit_data = circuit.commit_child_vk(&engine, &vk);
    let ctxs = circuit.generate_proving_ctxs::<DuplexSpongeRecorder>(&vk, vk_commit_data, &[proof]);
    debug(&circuit.airs(), &pk.per_air, &ctxs);
}

#[test]
fn test_preflight_preprocessed_trace() {
    let engine = test_engine_small();
    let height = 1 << 5;
    let sels = (0..height).map(|i| i % 2 == 0).collect_vec();
    let fx = PreprocessedFibFixture::new(0, 1, sels);
    let (vk, proof) = fx.keygen_and_prove(&engine);

    let (circuit, pk) = verifier_circuit_keygen::<2>(&vk);
    let vk_commit_data = circuit.commit_child_vk(&engine, &vk);
    let ctxs = circuit.generate_proving_ctxs::<DuplexSpongeRecorder>(&vk, vk_commit_data, &[proof]);
    debug(&circuit.airs(), &pk.per_air, &ctxs);
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
    let circuit = VerifierSubCircuit::<2>::new(Arc::new(vk.clone()));
    let preflight = circuit.run_preflight(preflight_sponge, &vk, &proof);
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

    let (circuit, pk) = verifier_circuit_keygen::<2>(&vk);
    let vk_commit_data = circuit.commit_child_vk(&engine, &vk);
    let ctxs = circuit.generate_proving_ctxs::<DuplexSpongeRecorder>(
        &vk,
        vk_commit_data,
        &[proof1, proof2],
    );
    debug(&circuit.airs(), &pk.per_air, &ctxs);
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

    let (circuit, pk) = verifier_circuit_keygen::<5>(&vk);
    let vk_commit_data = circuit.commit_child_vk(&engine, &vk);
    let ctxs = circuit.generate_proving_ctxs::<DuplexSpongeRecorder>(
        &vk,
        vk_commit_data,
        &[proof1, proof2, proof3, proof4, proof5],
    );
    debug(&circuit.airs(), &pk.per_air, &ctxs);
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

    let (circuit, pk) = verifier_circuit_keygen::<2>(&vk);
    let vk_commit_data = circuit.commit_child_vk(&engine, &vk);
    let ctxs = circuit.generate_proving_ctxs::<DuplexSpongeRecorder>(
        &vk,
        vk_commit_data,
        &[proof1, proof2],
    );
    debug(&circuit.airs(), &pk.per_air, &ctxs);
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

    let (circuit, pk) = verifier_circuit_keygen::<5>(&vk);
    let vk_commit_data = circuit.commit_child_vk(&engine, &vk);
    let ctxs = circuit.generate_proving_ctxs::<DuplexSpongeRecorder>(
        &vk,
        vk_commit_data,
        &[proof1, proof2, proof3, proof4, proof5],
    );
    debug(&circuit.airs(), &pk.per_air, &ctxs);
}

#[test]
fn test_recursion_circuit_two_interactions() {
    let params = test_system_params_small();
    let engine = BabyBearPoseidon2CpuEngineV2::<DuplexSponge>::new(params);

    let fx = InteractionsFixture11;
    let (vk, proof) = fx.keygen_and_prove(&engine);

    let (circuit, pk) = verifier_circuit_keygen::<2>(&vk);
    let vk_commit_data = circuit.commit_child_vk(&engine, &vk);
    let ctxs = circuit.generate_proving_ctxs::<DuplexSpongeRecorder>(
        &vk,
        vk_commit_data,
        &[proof.clone(), proof],
    );
    debug(&circuit.airs(), &pk.per_air, &ctxs);
}

#[test]
fn test_recursion_circuit_multiple_interactions() {
    let params = test_system_params_small();
    let engine = BabyBearPoseidon2CpuEngineV2::<DuplexSponge>::new(params);

    // Generate multiple interaction proofs - they should use the same VK
    let fx = InteractionsFixture11;
    let (vk, proof) = fx.keygen_and_prove(&engine);

    let (circuit, pk) = verifier_circuit_keygen::<5>(&vk);
    let vk_commit_data = circuit.commit_child_vk(&engine, &vk);
    let ctxs = circuit.generate_proving_ctxs::<DuplexSpongeRecorder>(
        &vk,
        vk_commit_data,
        &[
            proof.clone(),
            proof.clone(),
            proof.clone(),
            proof.clone(),
            proof,
        ],
    );
    debug(&circuit.airs(), &pk.per_air, &ctxs);
}

#[test]
fn test_recursion_circuit_two_cached() {
    let params = test_system_params_small();
    let engine = BabyBearPoseidon2CpuEngineV2::<DuplexSponge>::new(params);

    let fx = CachedFixture11::new(params);
    let (vk, proof) = fx.keygen_and_prove(&engine);

    let (circuit, pk) = verifier_circuit_keygen::<5>(&vk);
    let vk_commit_data = circuit.commit_child_vk(&engine, &vk);
    let ctxs = circuit.generate_proving_ctxs::<DuplexSpongeRecorder>(
        &vk,
        vk_commit_data,
        &[
            proof.clone(),
            proof.clone(),
            proof.clone(),
            proof.clone(),
            proof,
        ],
    );
    debug(&circuit.airs(), &pk.per_air, &ctxs);
}

#[test]
fn test_recursion_circuit_multiple_cached() {
    let params = test_system_params_small();
    let engine = BabyBearPoseidon2CpuEngineV2::<DuplexSponge>::new(params);

    let fx = CachedFixture11::new(params);
    let (vk, proof) = fx.keygen_and_prove(&engine);

    let (circuit, pk) = verifier_circuit_keygen::<5>(&vk);
    let vk_commit_data = circuit.commit_child_vk(&engine, &vk);
    let ctxs = circuit.generate_proving_ctxs::<DuplexSpongeRecorder>(
        &vk,
        vk_commit_data,
        &[
            proof.clone(),
            proof.clone(),
            proof.clone(),
            proof.clone(),
            proof,
        ],
    );
    debug(&circuit.airs(), &pk.per_air, &ctxs);
}

///////////////////////////////////////////////////////////////////////////////
// CUDA TRACEGEN TESTS
///////////////////////////////////////////////////////////////////////////////
#[cfg(feature = "cuda")]
mod cuda {
    use cuda_backend_v2::BabyBearPoseidon2GpuEngineV2;
    use itertools::zip_eq;
    use openvm_cuda_common::copy::MemCopyD2H;
    use openvm_stark_sdk::config::setup_tracing_with_log_level;
    use stark_backend_v2::{keygen::types::SystemParams, prover::MatrixView};
    use tracing::Level;

    use super::*;

    /// `params` are system parameters of the parent.
    fn compare_cpu_tracegen_vs_gpu_tracegen<Fx: TestFixture>(
        fx: Fx,
        params: SystemParams,
        num_proofs: usize,
    ) {
        setup_tracing_with_log_level(Level::DEBUG);
        let cpu_engine = BabyBearPoseidon2CpuEngineV2::new(params);
        let gpu_engine = BabyBearPoseidon2GpuEngineV2::new(params);
        let (pk, vk) = fx.keygen(&cpu_engine);
        assert!(num_proofs <= 5);
        let proofs = (0..num_proofs)
            .map(|_| fx.prove(&cpu_engine, &pk))
            .collect_vec();
        let vk = Arc::new(vk);

        let circuit = VerifierSubCircuit::<5>::new(vk.clone());
        for (air_idx, air) in circuit.airs().iter().enumerate() {
            tracing::debug!(%air_idx, air_name = %air.name());
        }

        let vk_commit_data_cpu = circuit.commit_child_vk(&cpu_engine, &vk);
        let vk_commit_data_gpu = circuit.commit_child_vk_gpu(&gpu_engine, &vk);
        let cpu_ctx =
            circuit.generate_proving_ctxs::<DuplexSpongeRecorder>(&vk, vk_commit_data_cpu, &proofs);
        let gpu_ctx = circuit.generate_proving_ctxs_gpu::<DuplexSpongeRecorder>(
            &vk,
            vk_commit_data_gpu,
            &proofs,
        );
        let non_deterministic_air_idxs = [cpu_ctx.len() - 1]; // exp_bits is non-deterministic when multi-threaded

        for (i, (cpu, gpu)) in zip_eq(cpu_ctx, gpu_ctx).enumerate() {
            let cpu = cpu.common_main;
            let gpu = gpu.common_main;
            assert_eq!(gpu.width(), cpu.width(), "Width mismatch at AIR {i}");
            assert_eq!(gpu.height(), cpu.height(), "Height mismatch at AIR {i}");
            if non_deterministic_air_idxs.contains(&i) {
                continue;
            }
            let gpu = gpu.to_host().unwrap();
            for r in 0..cpu.height() {
                for c in 0..cpu.width() {
                    assert_eq!(
                        gpu[c * cpu.height() + r],
                        *cpu.get(r, c).unwrap(),
                        "Mismatch for AIR {i} at row {r} column {c}"
                    );
                }
            }
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_tracegen_single_fib() {
        let params = test_system_params_small();
        let fx = FibFixture::new(0, 1, 1 << 3);
        compare_cpu_tracegen_vs_gpu_tracegen(fx, params, 1);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_tracegen_cached() {
        let params = test_system_params_small();
        let fx = CachedFixture11::new(params);
        compare_cpu_tracegen_vs_gpu_tracegen(fx, params, 1);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_tracegen_preprocessed() {
        let params = test_system_params_small();
        let sels = (0..(1 << 5)).map(|i| i % 2 == 0).collect_vec();
        let fx = PreprocessedFibFixture::new(0, 1, sels);
        compare_cpu_tracegen_vs_gpu_tracegen(fx, params, 1);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_tracegen_multi_fib() {
        let params = test_system_params_small();
        let fx = FibFixture::new(0, 1, 1 << 3);
        compare_cpu_tracegen_vs_gpu_tracegen(fx, params, 4);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_tracegen_multi_cached() {
        let params = test_system_params_small();
        let fx = CachedFixture11::new(params);
        compare_cpu_tracegen_vs_gpu_tracegen(fx, params, 4);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_tracegen_multi_preprocessed() {
        let params = test_system_params_small();
        let sels = (0..(1 << 5)).map(|i| i % 2 == 0).collect_vec();
        let fx = PreprocessedFibFixture::new(0, 1, sels);
        compare_cpu_tracegen_vs_gpu_tracegen(fx, params, 4);
    }
}
