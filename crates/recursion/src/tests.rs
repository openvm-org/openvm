use std::{collections::BTreeSet, sync::Arc};

use itertools::Itertools;
use openvm_stark_backend::{
    keygen::types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
    prover::{AirProvingContext, CpuBackend, ProvingContext},
    test_utils::{
        default_test_params_small, test_system_params_small,
        test_system_params_small_with_poly_len, CachedFixture11, FibFixture, InteractionsFixture11,
        MixtureFixture, MixtureFixtureEnum, PreprocessedFibFixture, SelfInteractionFixture,
        TestFixture,
    },
    AirRef, StarkEngine, SystemParams, TranscriptHistory,
};
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::{
        default_duplex_sponge_recorder, default_duplex_sponge_validator, BabyBearPoseidon2Config,
        BabyBearPoseidon2CpuEngine, DuplexSponge, DuplexSpongeRecorder,
    },
    utils::setup_tracing_with_log_level,
};
use test_case::{test_case, test_matrix};
use tracing::Level;

use crate::system::{AggregationSubCircuit, CachedTraceCtx, VerifierSubCircuit, VerifierTraceGen};

pub const MAX_CONSTRAINT_DEGREE: usize = 4;

/// Creates test system params with all PoW bits set to zero.
fn test_system_params_zero_pow(l_skip: usize, n_stack: usize, k_whir: usize) -> SystemParams {
    let mut params = test_system_params_small(l_skip, n_stack, k_whir);
    params.whir.mu_pow_bits = 0;
    params.whir.folding_pow_bits = 0;
    params.whir.query_phase_pow_bits = 0;
    params
}

pub fn test_engine_small() -> BabyBearPoseidon2CpuEngine<DuplexSponge> {
    let mut params = test_system_params_small(2, 10, 3);
    params.max_constraint_degree = MAX_CONSTRAINT_DEGREE;
    BabyBearPoseidon2CpuEngine::new(params)
}

fn verifier_circuit_keygen<const MAX_NUM_PROOFS: usize>(
    engine: &BabyBearPoseidon2CpuEngine<DuplexSponge>,
    child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
) -> (
    VerifierSubCircuit<MAX_NUM_PROOFS>,
    MultiStarkProvingKey<BabyBearPoseidon2Config>,
) {
    let circuit = VerifierSubCircuit::new(Arc::new(child_vk.clone()));
    let (pk, _vk) = engine.keygen(&circuit.airs());
    (circuit, pk)
}

fn debug(
    engine: &BabyBearPoseidon2CpuEngine<DuplexSponge>,
    airs: &[AirRef<BabyBearPoseidon2Config>],
    ctxs: Vec<AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>>,
) {
    for (air_idx, air) in airs.iter().enumerate() {
        tracing::debug!(%air_idx, air_name = %air.name());
    }
    let ctx = ProvingContext::new(ctxs.into_iter().enumerate().collect());
    engine.debug(airs, &ctx);
}

fn run_test<const MAX_NUM_PROOFS: usize, Fx: TestFixture<BabyBearPoseidon2Config>>(
    fx: Fx,
    child_engine: &BabyBearPoseidon2CpuEngine<DuplexSponge>,
    parent_engine: &BabyBearPoseidon2CpuEngine<DuplexSponge>,
    num_proofs: usize,
) {
    assert!(num_proofs <= MAX_NUM_PROOFS);
    let (vk, proof) = fx.keygen_and_prove(child_engine);
    let (circuit, _pk) = verifier_circuit_keygen::<MAX_NUM_PROOFS>(parent_engine, &vk);
    let vk_commit_data = CachedTraceCtx::PcsData(circuit.commit_child_vk(parent_engine, &vk));
    let proofs: Vec<_> = (0..num_proofs).map(|_| proof.clone()).collect();
    let ctxs = circuit.generate_proving_ctxs_base(
        &vk,
        vk_commit_data,
        &proofs,
        default_duplex_sponge_recorder(),
    );
    debug(parent_engine, &circuit.airs(), ctxs);
}

#[test_matrix(
    [2,3],
    [8,5],
    [3],
    [0,1,3,5,8]
)]
fn test_recursion_circuit_single_fib(
    l_skip: usize,
    n_stack: usize,
    k_whir: usize,
    log_trace_degree: usize,
) {
    if log_trace_degree > l_skip + n_stack {
        return;
    }
    let params = test_system_params_small(l_skip, n_stack, k_whir);
    let child_engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(params);
    let parent_engine = test_engine_small();
    let fib = FibFixture::new(0, 1, 1 << log_trace_degree);
    run_test::<2, _>(fib, &child_engine, &parent_engine, 1);
}

#[test]
fn test_recursion_circuit_many_fib_airs() {
    let params = test_system_params_small(3, 5, 3);
    let child_engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(params);
    let parent_engine = test_engine_small();
    let fib = FibFixture::new_with_num_airs(0, 1, 1 << 8, 9);
    run_test::<2, _>(fib, &child_engine, &parent_engine, 1);
}

#[test]
fn test_recursion_circuit_many_fib_airs_some_missing() {
    let l_skip = 3;
    let n_stack = 5;
    let k_whir = 3;
    let log_trace_degree = 8;

    if log_trace_degree > l_skip + n_stack {
        return;
    }
    let params = test_system_params_small(l_skip, n_stack, k_whir);
    let child_engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(params);
    let empty_air_indices = vec![1, 3, 6];
    let fib = FibFixture::new_with_num_airs(0, 1, 1 << log_trace_degree, 9)
        .with_empty_air_indices(empty_air_indices.clone());
    let (vk, proof) = fib.keygen_and_prove(&child_engine);

    let empty_air_indices = empty_air_indices.into_iter().collect::<BTreeSet<_>>();
    for air_idx in 0..vk.inner.per_air.len() {
        if empty_air_indices.contains(&air_idx) {
            assert!(
                !vk.inner.per_air[air_idx].is_required,
                "air {air_idx} unexpectedly marked required"
            );
            assert!(proof.trace_vdata[air_idx].is_none());
            assert!(proof.public_values[air_idx].is_empty());
        } else {
            assert!(proof.trace_vdata[air_idx].is_some());
        }
    }

    let parent_engine = test_engine_small();
    let (circuit, _pk) = verifier_circuit_keygen::<2>(&parent_engine, &vk);
    let vk_commit_data = CachedTraceCtx::PcsData(circuit.commit_child_vk(&parent_engine, &vk));
    let ctxs = circuit.generate_proving_ctxs_base(
        &vk,
        vk_commit_data,
        &[proof],
        default_duplex_sponge_recorder(),
    );
    debug(&parent_engine, &circuit.airs(), ctxs);
}

#[test_case(2, 8, 3)]
#[test_case(5, 5, 4)]
#[test_case(5, 6, 4)]
#[test_case(6, 6, 5)]
fn test_recursion_circuit_interactions(l_skip: usize, n_stack: usize, k_whir: usize) {
    setup_tracing_with_log_level(Level::DEBUG);
    let params = test_system_params_small(l_skip, n_stack, k_whir);
    let child_engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(params);
    let parent_engine = test_engine_small();
    run_test::<2, _>(InteractionsFixture11, &child_engine, &parent_engine, 1);
}

#[test_case(2, 8, 3, 5)]
#[test_case(3, 5, 3, 5)]
#[test_case(3, 5, 3, 1)]
fn test_preflight_single_fib_sponge(
    l_skip: usize,
    n_stack: usize,
    k_whir: usize,
    log_trace_degree: usize,
) {
    let params = test_system_params_small(l_skip, n_stack, k_whir);
    let engine = BabyBearPoseidon2CpuEngine::<DuplexSpongeRecorder>::new(params);
    let fib = FibFixture::new(0, 1, 1 << log_trace_degree);
    let (pk, vk) = fib.keygen(&engine);

    let mut prover_sponge = default_duplex_sponge_recorder();
    let proof = fib.prove_from_transcript(&engine, &pk, &mut prover_sponge);
    let prover_sponge_len = prover_sponge.len();

    let preflight_sponge = default_duplex_sponge_validator(prover_sponge.into_log());
    let circuit = VerifierSubCircuit::<2>::new(Arc::new(vk.clone()));
    let preflight = circuit.run_preflight(preflight_sponge, &vk, &proof);
    assert_eq!(preflight.transcript.len(), prover_sponge_len);
}

#[test_case(2, 8, 3)]
#[test_case(5, 5, 4)]
#[test_case(5, 6, 4)]
fn test_preflight_cached_trace(l_skip: usize, n_stack: usize, k_whir: usize) {
    let params = test_system_params_small(l_skip, n_stack, k_whir);
    let child_engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(params.clone());
    let parent_engine = test_engine_small();
    let fx = CachedFixture11::new(BabyBearPoseidon2Config::default_from_params(params));
    run_test::<2, _>(fx, &child_engine, &parent_engine, 1);
}

#[test_matrix(
    [2,3,5,6],
    [8,5],
    [3,4],
    [5,7,8,11]
)]
fn test_preflight_preprocessed_trace(
    l_skip: usize,
    n_stack: usize,
    k_whir: usize,
    log_trace_height: usize,
) {
    if log_trace_height > l_skip + n_stack {
        return;
    }
    let params = test_system_params_small(l_skip, n_stack, k_whir);
    let child_engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(params);
    let parent_engine = test_engine_small();
    let height = 1 << log_trace_height;
    let sels = (0..height).map(|i| i % 2 == 0).collect_vec();
    let fx = PreprocessedFibFixture::new(0, 1, sels);
    run_test::<2, _>(fx, &child_engine, &parent_engine, 1);
}

#[test_matrix(
    [2, 3, 5],
    [8],
    [3],
    [3, 4, 5]
)]
fn test_preflight_preprocessed_and_cached_trace(
    l_skip: usize,
    n_stack: usize,
    k_whir: usize,
    log_trace_height: usize,
) {
    let params = test_system_params_small(l_skip, n_stack, k_whir);
    let child_engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(params.clone());
    let parent_engine = test_engine_small();
    let height = 1 << log_trace_height;
    let sels = (0..height).map(|i| i % 2 == 0).collect_vec();
    let fx = MixtureFixture::new(vec![
        MixtureFixtureEnum::PreprocessedFibFixture(PreprocessedFibFixture::new(0, 1, sels)),
        MixtureFixtureEnum::CachedFixture11(CachedFixture11::new(
            BabyBearPoseidon2Config::default_from_params(params),
        )),
    ]);
    run_test::<2, _>(fx, &child_engine, &parent_engine, 1);
}

#[test_case(2, 8, 3, 5)]
#[test_case(3, 5, 3, 5)]
#[test_case(3, 5, 3, 1)]
fn test_preflight_multi_interaction_trace(
    l_skip: usize,
    n_stack: usize,
    k_whir: usize,
    log_trace_height: usize,
) {
    let params = test_system_params_small(l_skip, n_stack, k_whir);
    let child_engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(params);
    let parent_engine = test_engine_small();
    let fx = SelfInteractionFixture {
        widths: vec![4, 7, 8, 8, 10, 100],
        log_height: log_trace_height,
        bus_index: 4,
    };
    run_test::<2, _>(fx, &child_engine, &parent_engine, 1);
}

#[test_case(2, 8, 3, 5)]
#[test_case(3, 5, 3, 5)]
#[test_case(3, 5, 3, 1)]
fn test_preflight_mixture_trace(
    l_skip: usize,
    n_stack: usize,
    k_whir: usize,
    log_trace_height: usize,
) {
    let params = test_system_params_small(l_skip, n_stack, k_whir);
    let child_engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(params.clone());
    let parent_engine = test_engine_small();
    let fx = MixtureFixture::standard(
        log_trace_height,
        BabyBearPoseidon2Config::default_from_params(params),
    );
    run_test::<2, _>(fx, &child_engine, &parent_engine, 1);
}

#[test]
fn test_preflight_preprocessed_and_cached_transcript() {
    let l_skip = 2;
    let n_stack = 8;
    let k_whir = 3;
    let log_trace_height = 4;
    let params = test_system_params_small(l_skip, n_stack, k_whir);
    let engine = BabyBearPoseidon2CpuEngine::<DuplexSpongeRecorder>::new(params.clone());
    let height = 1 << log_trace_height;
    let sels = (0..height).map(|i| i % 2 == 0).collect_vec();
    let fx = MixtureFixture::new(vec![
        MixtureFixtureEnum::PreprocessedFibFixture(PreprocessedFibFixture::new(0, 1, sels)),
        MixtureFixtureEnum::CachedFixture11(CachedFixture11::new(
            BabyBearPoseidon2Config::default_from_params(params),
        )),
    ]);
    let (pk, vk) = fx.keygen(&engine);

    let mut prover_sponge = default_duplex_sponge_recorder();
    let proof = fx.prove_from_transcript(&engine, &pk, &mut prover_sponge);
    let prover_sponge_len = prover_sponge.len();

    let preflight_sponge = default_duplex_sponge_validator(prover_sponge.into_log());
    let circuit = VerifierSubCircuit::<2>::new(Arc::new(vk.clone()));
    let preflight = circuit.run_preflight(preflight_sponge, &vk, &proof);
    assert_eq!(preflight.transcript.len(), prover_sponge_len);
}

#[test]
fn test_preflight_interactions() {
    let params = default_test_params_small();
    let engine = BabyBearPoseidon2CpuEngine::<DuplexSpongeRecorder>::new(params);
    let fx = InteractionsFixture11;
    let (pk, vk) = fx.keygen(&engine);

    let mut prover_sponge = default_duplex_sponge_recorder();
    let proof = fx.prove_from_transcript(&engine, &pk, &mut prover_sponge);
    let prover_sponge_len = prover_sponge.len();

    let preflight_sponge = default_duplex_sponge_validator(prover_sponge.into_log());
    let circuit = VerifierSubCircuit::<2>::new(Arc::new(vk.clone()));
    let preflight = circuit.run_preflight(preflight_sponge, &vk, &proof);
    assert_eq!(preflight.transcript.len(), prover_sponge_len);
}

///////////////////////////////////////////////////////////////////////////////
// Multi-proof tests
///////////////////////////////////////////////////////////////////////////////

#[test_case(10 ; "fib log_height maximum (i.e. equals n_stack + l_skip)")]
#[test_case(3 ; "when fib log_height greater than l_skip")]
#[test_case(2 ; "when fib log_height equals l_skip")]
#[test_case(1 ; "when fib log_height less than l_skip")]
#[test_case(0 ; "when fib log_height is zero")]
fn test_recursion_circuit_two_fib_proofs(log_trace_degree: usize) {
    let params = default_test_params_small();
    let child_engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(params);
    let parent_engine = test_engine_small();
    let fib = FibFixture::new(0, 1, 1 << log_trace_degree);
    run_test::<2, _>(fib, &child_engine, &parent_engine, 2);
}

#[test_case(10 ; "fib log_height maximum (i.e. equals n_stack + l_skip)")]
#[test_case(3 ; "when fib log_height greater than l_skip")]
#[test_case(2 ; "when fib log_height equals l_skip")]
#[test_case(1 ; "when fib log_height less than l_skip")]
#[test_case(0 ; "when fib log_height is zero")]
fn test_recursion_circuit_multiple_fib_proofs(log_trace_degree: usize) {
    let params = default_test_params_small();
    let child_engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(params);
    let parent_engine = test_engine_small();
    let fib = FibFixture::new(0, 1, 1 << log_trace_degree);
    run_test::<5, _>(fib, &child_engine, &parent_engine, 5);
}

#[test_matrix(
    [2,5,6],
    [8,5],
    [3,4],
    [4,8,11]
)]
fn test_recursion_circuit_two_preprocessed(
    l_skip: usize,
    n_stack: usize,
    k_whir: usize,
    log_trace_height: usize,
) {
    let log_final_poly_len = (n_stack + l_skip) % k_whir + k_whir;
    if log_trace_height > l_skip + n_stack || log_final_poly_len >= l_skip + n_stack {
        return;
    }
    let params =
        test_system_params_small_with_poly_len(l_skip, n_stack, k_whir, log_final_poly_len, 3);
    let child_engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(params);
    let parent_engine = test_engine_small();
    let height = 1 << log_trace_height;
    let sels = (0..height).map(|i| i % 2 == 0).collect_vec();
    let fx = PreprocessedFibFixture::new(0, 1, sels);
    run_test::<2, _>(fx, &child_engine, &parent_engine, 2);
}

#[test_case(2, 8, 3)]
#[test_case(5, 5, 4)]
#[test_case(6, 5, 4)]
fn test_recursion_circuit_multiple_preprocessed(l_skip: usize, n_stack: usize, k_whir: usize) {
    let params = test_system_params_small(l_skip, n_stack, k_whir);
    let child_engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(params);
    let parent_engine = test_engine_small();
    let height = 1 << 4;
    let sels = (0..height).map(|i| i % 2 == 0).collect_vec();
    let fx = PreprocessedFibFixture::new(0, 1, sels);
    run_test::<5, _>(fx, &child_engine, &parent_engine, 5);
}

#[test]
fn test_recursion_circuit_two_interactions() {
    let params = default_test_params_small();
    let child_engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(params);
    let parent_engine = test_engine_small();
    run_test::<2, _>(InteractionsFixture11, &child_engine, &parent_engine, 2);
}

#[test]
fn test_recursion_circuit_multiple_interactions() {
    let params = default_test_params_small();
    let child_engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(params);
    let parent_engine = test_engine_small();
    run_test::<5, _>(InteractionsFixture11, &child_engine, &parent_engine, 5);
}

#[test]
fn test_recursion_circuit_two_cached() {
    let params = default_test_params_small();
    let child_engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(params.clone());
    let parent_engine = test_engine_small();
    let fx = CachedFixture11::new(BabyBearPoseidon2Config::default_from_params(params));
    run_test::<5, _>(fx, &child_engine, &parent_engine, 2);
}

#[test]
fn test_recursion_circuit_multiple_cached() {
    let params = default_test_params_small();
    let child_engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(params.clone());
    let parent_engine = test_engine_small();
    let fx = CachedFixture11::new(BabyBearPoseidon2Config::default_from_params(params));
    run_test::<5, _>(fx, &child_engine, &parent_engine, 5);
}

///////////////////////////////////////////////////////////////////////////////
// NEGATIVE HYPERCUBE TESTS
///////////////////////////////////////////////////////////////////////////////
/// Negative hypercube refers to `n < 0`. These are still "positive" tests that are expected to
/// pass.
fn run_negative_hypercube_test<Fx: TestFixture<BabyBearPoseidon2Config>>(
    fx: Fx,
    mut params: SystemParams,
    num_proofs: usize,
) {
    params.l_skip += 3;
    let child_engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(params);
    let parent_engine = test_engine_small();
    run_test::<5, _>(fx, &child_engine, &parent_engine, num_proofs);
}

#[test_case(1)]
#[test_case(4)]
fn test_neg_hypercube_fib(num_proofs: usize) {
    let params = default_test_params_small();
    let fx = FibFixture::new(0, 1, 1 << 3);
    run_negative_hypercube_test(fx, params, num_proofs);
}

#[test_case(1)]
#[test_case(4)]
fn test_neg_hypercube_cached(num_proofs: usize) {
    let params = default_test_params_small();
    // NOTE: the CachedFixture's cached trace needs correct params, run_negative_hypercube_test will
    // +3 to params.l_skip
    let mut fx_params = params.clone();
    fx_params.l_skip += 3;
    let fx = CachedFixture11::new(BabyBearPoseidon2Config::default_from_params(
        fx_params.clone(),
    ));
    run_negative_hypercube_test(fx, params, num_proofs);
}

#[test_case(1)]
#[test_case(4)]
fn test_neg_hypercube_preprocessed(num_proofs: usize) {
    let params = default_test_params_small();
    let sels = (0..(1 << 4)).map(|i| i % 2 == 0).collect_vec();
    let fx = PreprocessedFibFixture::new(0, 1, sels);
    run_negative_hypercube_test(fx, params, num_proofs);
}

#[test_case(1)]
#[test_case(4)]
fn test_neg_hypercube_multi_interaction(num_proofs: usize) {
    let params = default_test_params_small();
    let fx = SelfInteractionFixture {
        widths: vec![4, 7, 8, 8, 10, 100],
        log_height: 3,
        bus_index: 4,
    };
    run_negative_hypercube_test(fx, params, num_proofs);
}

#[test_case(1)]
#[test_case(4)]
fn test_neg_hypercube_mixture(num_proofs: usize) {
    let params = default_test_params_small();
    // NOTE: the CachedFixture's cached trace needs correct params, run_negative_hypercube_test will
    // +3 to params.l_skip
    let mut fx_params = params.clone();
    fx_params.l_skip += 3;
    let fx = MixtureFixture::standard(3, BabyBearPoseidon2Config::default_from_params(fx_params));
    run_negative_hypercube_test(fx, params, num_proofs);
}

///////////////////////////////////////////////////////////////////////////////
// ZERO POW BITS TESTS
///////////////////////////////////////////////////////////////////////////////

#[test_case(2, 8, 3, 5)]
#[test_case(3, 5, 3, 5)]
fn test_recursion_circuit_zero_pow_bits(
    l_skip: usize,
    n_stack: usize,
    k_whir: usize,
    log_trace_degree: usize,
) {
    let child_params = test_system_params_zero_pow(l_skip, n_stack, k_whir);
    let child_engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(child_params);
    let parent_engine = test_engine_small();
    let fib = FibFixture::new(0, 1, 1 << log_trace_degree);
    run_test::<2, _>(fib, &child_engine, &parent_engine, 1);
}

#[test_case(5)]
fn test_recursion_circuit_zero_pow_bits_two_proofs(log_trace_degree: usize) {
    let child_params = test_system_params_zero_pow(2, 8, 3);
    let child_engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(child_params);
    let parent_engine = test_engine_small();
    let fib = FibFixture::new(0, 1, 1 << log_trace_degree);
    run_test::<2, _>(fib, &child_engine, &parent_engine, 2);
}

///////////////////////////////////////////////////////////////////////////////
// CUDA TRACEGEN TESTS
///////////////////////////////////////////////////////////////////////////////
#[cfg(feature = "cuda")]
mod cuda {
    use itertools::zip_eq;
    use openvm_cuda_backend::BabyBearPoseidon2GpuEngine;
    use openvm_cuda_common::copy::MemCopyD2H;
    use openvm_stark_backend::prover::{MatrixDimensions, MatrixView};
    #[cfg(feature = "touchemall")]
    use openvm_stark_sdk::config::baby_bear_poseidon2::F;
    use openvm_stark_sdk::utils::setup_tracing_with_log_level;
    use test_case::test_matrix;
    use tracing::Level;

    use super::*;

    /// `params` are system parameters of the parent.
    fn compare_cpu_tracegen_vs_gpu_tracegen<Fx: TestFixture<BabyBearPoseidon2Config>>(
        fx: Fx,
        params: SystemParams,
        num_proofs: usize,
    ) {
        setup_tracing_with_log_level(Level::INFO);
        let cpu_engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(params.clone());
        let gpu_engine = BabyBearPoseidon2GpuEngine::new(params);
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

        let vk_commit_data_cpu = CachedTraceCtx::PcsData(circuit.commit_child_vk(&cpu_engine, &vk));
        let vk_commit_data_gpu = CachedTraceCtx::PcsData(circuit.commit_child_vk(&gpu_engine, &vk));
        let cpu_ctx = circuit.generate_proving_ctxs_base(
            &vk,
            vk_commit_data_cpu,
            &proofs,
            default_duplex_sponge_recorder(),
        );
        let gpu_ctx = circuit.generate_proving_ctxs_base(
            &vk,
            vk_commit_data_gpu,
            &proofs,
            default_duplex_sponge_recorder(),
        );

        #[cfg(feature = "touchemall")]
        for (i, gpu) in gpu_ctx.iter().enumerate() {
            let gpu = &gpu.common_main;
            let name = circuit.airs()[i].name();

            let width = gpu.width();
            let height = gpu.height();

            let gpu = gpu.to_host().unwrap();

            for r in 0..height {
                for c in 0..width {
                    let val = gpu[c * height + r];
                    let val_32 = unsafe { *(&val as *const F as *const u32) };
                    assert!(
                        val_32 != 0xffffffff,
                        "potentially untouched value at ({r}, {c}) of a trace of size {height}x{width} for air {name}"
                    );
                }
            }
        }

        const POSEIDON2_AIR_ID: usize = 14;
        assert!(circuit.airs()[POSEIDON2_AIR_ID]
            .name()
            .starts_with("Poseidon2Air"));

        let non_deterministic_air_idxs = [
            POSEIDON2_AIR_ID,
            cpu_ctx.len() - 1, // exp_bits is non-deterministic when multi-threaded
        ];

        for (i, (cpu, gpu)) in zip_eq(cpu_ctx, gpu_ctx).enumerate() {
            let cpu = cpu.common_main;
            let gpu = gpu.common_main;
            assert_eq!(gpu.width(), cpu.width(), "Width mismatch at AIR {i}");
            assert_eq!(gpu.height(), cpu.height(), "Height mismatch at AIR {i}");

            let name = circuit.airs()[i].name();

            if non_deterministic_air_idxs.contains(&i) {
                continue;
            }
            let gpu = gpu.to_host().unwrap();

            for r in 0..cpu.height() {
                for c in 0..cpu.width() {
                    assert_eq!(
                        gpu[c * cpu.height() + r],
                        *cpu.get(r, c).unwrap(),
                        "Mismatch for {} at row {r} column {c}",
                        name
                    );
                }
            }
        }
    }

    #[test_matrix(
        [2,3],
        [8,5],
        [3],
        [1,3,5],
        [1,4]
    )]
    fn test_cuda_tracegen_single_fib(
        l_skip: usize,
        n_stack: usize,
        k_whir: usize,
        log_trace_degree: usize,
        num_proofs: usize,
    ) {
        let params = test_system_params_small(l_skip, n_stack, k_whir);
        let fx = FibFixture::new(0, 1, 1 << log_trace_degree);
        compare_cpu_tracegen_vs_gpu_tracegen(fx, params, num_proofs);
    }

    #[test_matrix(
        [2,3],
        [8],
        [3],
        [1,3,5],
        [1,4]
    )]
    fn test_cuda_tracegen_multiple_fib(
        l_skip: usize,
        n_stack: usize,
        k_whir: usize,
        log_trace_degree: usize,
        num_proofs: usize,
    ) {
        let params = test_system_params_small(l_skip, n_stack, k_whir);
        let empty_air_indices = vec![1, 3, 6];
        let fx = FibFixture::new_with_num_airs(0, 1, 1 << log_trace_degree, 9)
            .with_empty_air_indices(empty_air_indices);
        compare_cpu_tracegen_vs_gpu_tracegen(fx, params, num_proofs);
    }

    #[test_matrix(
        [1,2,5],
        [5,8],
        [3,4],
        [1,4]
    )]
    fn test_cuda_tracegen_cached(l_skip: usize, n_stack: usize, k_whir: usize, num_proofs: usize) {
        let params = test_system_params_small(l_skip, n_stack, k_whir);
        let fx = CachedFixture11::new(BabyBearPoseidon2Config::default_from_params(params.clone()));
        compare_cpu_tracegen_vs_gpu_tracegen(fx, params, num_proofs);
    }

    #[test_matrix(
        [1,2,5],
        [5,8],
        [3,4],
        [1,3,5],
        [1,4]
    )]
    fn test_cuda_tracegen_preprocessed(
        l_skip: usize,
        n_stack: usize,
        k_whir: usize,
        log_trace_degree: usize,
        num_proofs: usize,
    ) {
        let params = test_system_params_small(l_skip, n_stack, k_whir);
        let sels = (0..(1 << log_trace_degree))
            .map(|i| i % 2 == 0)
            .collect_vec();
        let fx = PreprocessedFibFixture::new(0, 1, sels);
        compare_cpu_tracegen_vs_gpu_tracegen(fx, params, num_proofs);
    }

    #[test_matrix(
        [1,2,5],
        [3,4],
        [1,3,5],
        [1,4]
    )]
    fn test_cuda_tracegen_multi_interaction(
        l_skip: usize,
        k_whir: usize,
        log_trace_degree: usize,
        num_proofs: usize,
    ) {
        let n_stack = 12 - l_skip;
        let params = test_system_params_small(l_skip, n_stack, k_whir);
        let fx = SelfInteractionFixture {
            widths: vec![4, 7, 8, 8, 10, 100],
            log_height: log_trace_degree,
            bus_index: 4,
        };
        compare_cpu_tracegen_vs_gpu_tracegen(fx, params, num_proofs);
    }

    #[test_matrix(
        [1,2,5],
        [3,4],
        [1,3,5],
        [1,4]
    )]
    fn test_cuda_tracegen_mixture(
        l_skip: usize,
        k_whir: usize,
        log_trace_degree: usize,
        num_proofs: usize,
    ) {
        let n_stack = 12 - l_skip;
        let params = test_system_params_small(l_skip, n_stack, k_whir);
        let fx = MixtureFixture::standard(
            log_trace_degree,
            BabyBearPoseidon2Config::default_from_params(params.clone()),
        );
        compare_cpu_tracegen_vs_gpu_tracegen(fx, params, num_proofs);
    }
}
