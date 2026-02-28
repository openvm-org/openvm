use halo2_base::{
    gates::GateInstructions,
    gates::RangeInstructions,
    gates::circuit::{CircuitBuilderStage, builder::BaseCircuitBuilder},
    halo2_proofs::dev::MockProver,
};
use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2CpuEngine,
    openvm_stark_backend::{
        StarkEngine,
        test_utils::{InteractionsFixture11, TestFixture, test_system_params_small},
    },
};

use crate::config::{
    STATIC_VERIFIER_LOOKUP_ADVICE_COLS_PHASE0, STATIC_VERIFIER_NUM_ADVICE_COLS_PHASE0,
};
use crate::gadgets::baby_bear::BABY_BEAR_MODULUS_U64;

use super::*;

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
        let assigned = derive_and_constrain_batch_and_stacked(ctx, &range, engine.config(), &vk, &proof)
            .expect("batch-and-stacked derive+constrain should succeed");

        range.gate().assert_is_const(
            ctx,
            &assigned.batch.logup_pow_witness_ok,
            &Fr::from(1u64),
        );
    });
}

#[test]
fn batch_and_stacked_constraints_fail_when_batch_intermediate_is_tampered() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut actual =
        derive_batch_and_stacked_intermediates(engine.config(), &vk, &proof).expect("native batch-and-stacked must pass");
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
    let mut actual =
        derive_batch_and_stacked_intermediates(engine.config(), &vk, &proof).expect("native batch-and-stacked must pass");
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
        let _unchecked =
            constrain_checked_batch_and_stacked_witness_state_unchecked(ctx, &range, &raw_for_unchecked);
    });

    run_mock(false, move |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let _strict = constrain_checked_batch_and_stacked_witness_state_strict(ctx, &range, &raw, &ownership);
    });
}
