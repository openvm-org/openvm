use halo2_base::{
    gates::{
        circuit::{builder::BaseCircuitBuilder, CircuitBuilderStage},
        GateInstructions, RangeInstructions,
    },
    halo2_proofs::dev::MockProver,
};
use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2CpuEngine,
    openvm_stark_backend::{
        test_utils::{test_system_params_small, InteractionsFixture11, TestFixture},
        StarkEngine,
    },
};

use super::*;
use crate::{
    config::{STATIC_VERIFIER_LOOKUP_ADVICE_COLS_PHASE0, STATIC_VERIFIER_NUM_ADVICE_COLS_PHASE0},
    gadgets::baby_bear::BABY_BEAR_MODULUS_U64,
};

#[derive(Clone, Debug, PartialEq, Eq)]
struct BatchAndStackedStrictOwnership {
    pub batch_trace_id_to_air_id: Vec<usize>,
    pub batch_total_interactions: u64,
    pub batch_n_logup: usize,
    pub batch_n_max: usize,
    pub batch_degree: usize,
    pub batch_l_skip: usize,
    pub stacked_l_skip: usize,
}

fn derive_batch_and_stacked_strict_ownership(
    config: &NativeConfig,
    mvk: &MultiStarkVerifyingKey<NativeConfig>,
    proof: &Proof<NativeConfig>,
) -> Result<BatchAndStackedStrictOwnership, BatchAndStackedError> {
    let expected = derive_batch_and_stacked_intermediates(config, mvk, proof)?;
    Ok(BatchAndStackedStrictOwnership {
        batch_trace_id_to_air_id: expected.batch.trace_id_to_air_id,
        batch_total_interactions: expected.batch.total_interactions,
        batch_n_logup: expected.batch.n_logup,
        batch_n_max: expected.batch.n_max,
        batch_degree: expected.batch.batch_degree,
        batch_l_skip: expected.batch.l_skip,
        stacked_l_skip: expected.stacked_reduction.l_skip,
    })
}

fn constrain_batch_and_stacked_strict_metadata(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    raw: &RawBatchAndStackedWitnessState,
    assigned: &AssignedBatchAndStackedIntermediates,
    ownership: &BatchAndStackedStrictOwnership,
) {
    let gate = range.gate();
    assert_eq!(
        assigned.batch.trace_id_to_air_id.len(),
        ownership.batch_trace_id_to_air_id.len(),
        "strict ownership trace-id schedule length mismatch",
    );
    for (&assigned_air_id, &expected_air_id) in assigned
        .batch
        .trace_id_to_air_id
        .iter()
        .zip(ownership.batch_trace_id_to_air_id.iter())
    {
        gate.assert_is_const(ctx, &assigned_air_id, &Fr::from(expected_air_id as u64));
    }
    gate.assert_is_const(
        ctx,
        &assigned.batch.total_interactions,
        &Fr::from(ownership.batch_total_interactions),
    );
    gate.assert_is_const(
        ctx,
        &assigned.batch.n_logup,
        &Fr::from(ownership.batch_n_logup as u64),
    );
    gate.assert_is_const(
        ctx,
        &assigned.batch.n_max,
        &Fr::from(ownership.batch_n_max as u64),
    );
    gate.assert_is_const(
        ctx,
        &assigned.batch.batch_degree,
        &Fr::from(ownership.batch_degree as u64),
    );

    assert_eq!(raw.intermediates.batch.l_skip, ownership.batch_l_skip);
    assert_eq!(
        raw.intermediates.stacked_reduction.l_skip,
        ownership.stacked_l_skip
    );
}

/// Standalone batch-and-stacked derive+constrain wrapper is internal; external callers must use
/// transcript-owned stage composition (`stages::full_pipeline`) as the acceptance boundary.
fn derive_and_constrain_batch_and_stacked(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    config: &NativeConfig,
    mvk: &MultiStarkVerifyingKey<NativeConfig>,
    proof: &Proof<NativeConfig>,
) -> Result<AssignedBatchAndStackedIntermediates, BatchAndStackedError> {
    let raw = derive_raw_batch_and_stacked_witness_state(config, mvk, proof)?;
    let ownership = derive_batch_and_stacked_strict_ownership(config, mvk, proof)?;
    Ok(
        constrain_checked_batch_and_stacked_witness_state_strict(ctx, range, &raw, &ownership)
            .assigned,
    )
}

fn derive_raw_batch_and_stacked_witness_state(
    config: &NativeConfig,
    mvk: &MultiStarkVerifyingKey<NativeConfig>,
    proof: &Proof<NativeConfig>,
) -> Result<RawBatchAndStackedWitnessState, BatchAndStackedError> {
    Ok(RawBatchAndStackedWitnessState {
        intermediates: derive_batch_and_stacked_intermediates(config, mvk, proof)?,
    })
}

fn constrain_checked_batch_and_stacked_witness_state_strict(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    raw: &RawBatchAndStackedWitnessState,
    ownership: &BatchAndStackedStrictOwnership,
) -> CheckedBatchAndStackedWitnessState {
    let checked = constrain_checked_batch_and_stacked_witness_state_unchecked(ctx, range, raw);
    constrain_batch_and_stacked_strict_metadata(ctx, range, raw, &checked.assigned, ownership);
    checked
}

fn run_mock(expect_satisfied: bool, build: impl FnOnce(&mut BaseCircuitBuilder<Fr>)) {
    let mut builder = BaseCircuitBuilder::from_stage(CircuitBuilderStage::Mock)
        .use_k(22)
        .use_lookup_bits(8)
        .use_instance_columns(1);
    build(&mut builder);

    let params = builder.calculate_params(Some(32768));
    assert_eq!(
        params
            .num_advice_per_phase
            .first()
            .copied()
            .unwrap_or_default(),
        STATIC_VERIFIER_NUM_ADVICE_COLS_PHASE0
    );
    assert_eq!(
        params
            .num_lookup_advice_per_phase
            .first()
            .copied()
            .unwrap_or_default(),
        STATIC_VERIFIER_LOOKUP_ADVICE_COLS_PHASE0
    );

    let prover = MockProver::run(22, &builder, vec![vec![]])
        .expect("mock prover should initialize for batch-and-stacked circuit");
    if expect_satisfied {
        prover.assert_satisfied();
    } else {
        assert!(
            prover.verify().is_err(),
            "expected constraints to fail for tampered batch-and-stacked intermediates"
        );
    }
}

fn test_engine() -> BabyBearBn254Poseidon2CpuEngine {
    BabyBearBn254Poseidon2CpuEngine::new(test_system_params_small(2, 8, 3))
}

#[test]
fn batch_and_stacked_intermediates_match_native_for_interactions_fixture() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);

    run_mock(true, move |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let assigned =
            derive_and_constrain_batch_and_stacked(ctx, &range, engine.config(), &vk, &proof)
                .expect("batch-and-stacked derive+constrain should succeed");

        range
            .gate()
            .assert_is_const(ctx, &assigned.batch.logup_pow_witness_ok, &Fr::from(1u64));
    });
}

#[test]
fn batch_and_stacked_constraints_fail_when_batch_intermediate_is_tampered() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut actual = derive_batch_and_stacked_intermediates(engine.config(), &vk, &proof)
        .expect("native batch-and-stacked must pass");
    actual.batch.consistency_residual[0] =
        (actual.batch.consistency_residual[0] + 1) % BABY_BEAR_MODULUS_U64;

    run_mock(false, move |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let _assigned = constrain_batch_and_stacked_intermediates_unchecked(ctx, &range, &actual);
    });
}

#[test]
fn batch_and_stacked_constraints_fail_when_stacked_intermediate_is_tampered() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut actual = derive_batch_and_stacked_intermediates(engine.config(), &vk, &proof)
        .expect("native batch-and-stacked must pass");
    actual.stacked_reduction.final_residual[0] =
        (actual.stacked_reduction.final_residual[0] + 1) % BABY_BEAR_MODULUS_U64;

    run_mock(false, move |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let _assigned = constrain_batch_and_stacked_intermediates_unchecked(ctx, &range, &actual);
    });
}

#[test]
fn batch_and_stacked_strict_constraints_reject_forged_standalone_metadata() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut raw = derive_raw_batch_and_stacked_witness_state(engine.config(), &vk, &proof)
        .expect("raw batch-and-stacked witness derivation must pass");
    let ownership = derive_batch_and_stacked_strict_ownership(engine.config(), &vk, &proof)
        .expect("strict batch-and-stacked ownership derivation must pass");
    assert!(
        raw.intermediates.batch.total_interactions > 0,
        "fixture should produce non-zero interactions for metadata-forgery test",
    );
    raw.intermediates.batch.total_interactions =
        raw.intermediates.batch.total_interactions.saturating_sub(1);
    let raw_for_unchecked = raw.clone();

    run_mock(true, move |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let _unchecked = constrain_checked_batch_and_stacked_witness_state_unchecked(
            ctx,
            &range,
            &raw_for_unchecked,
        );
    });

    run_mock(false, move |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let _strict =
            constrain_checked_batch_and_stacked_witness_state_strict(ctx, &range, &raw, &ownership);
    });
}
