use halo2_base::{
    gates::circuit::{CircuitBuilderStage, builder::BaseCircuitBuilder},
    halo2_proofs::dev::MockProver,
};
use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::{BabyBearBn254Poseidon2CpuEngine, EF as NativeEF},
    openvm_stark_backend::{
        StarkEngine,
        test_utils::{InteractionsFixture11, TestFixture, test_system_params_small},
        verifier::stacked_reduction::StackedReductionError,
    },
};

use crate::{
    config::{
        STATIC_VERIFIER_LOOKUP_ADVICE_COLS_PHASE0, STATIC_VERIFIER_NUM_ADVICE_COLS_PHASE0,
    },
    gadgets::baby_bear::BABY_BEAR_MODULUS_U64,
};

use super::*;

fn run_mock(expect_satisfied: bool, build: impl FnOnce(&mut BaseCircuitBuilder<Fr>)) {
    const MOCK_K: u32 = 22;
    let mut builder = BaseCircuitBuilder::from_stage(CircuitBuilderStage::Mock)
        .use_k(MOCK_K as usize)
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

    let prover = MockProver::run(MOCK_K, &builder, vec![vec![]])
        .expect("mock prover should initialize for stacked-reduction circuit");
    if expect_satisfied {
        prover.assert_satisfied();
    } else {
        assert!(
            prover.verify().is_err(),
            "expected constraints to fail for tampered stacked-reduction intermediates"
        );
    }
}

fn test_engine() -> BabyBearBn254Poseidon2CpuEngine {
    BabyBearBn254Poseidon2CpuEngine::new(test_system_params_small(2, 8, 3))
}

#[test]
fn stacked_intermediates_match_native_for_interactions_fixture() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);

    run_mock(true, move |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let _assigned = derive_and_constrain_stacked_reduction(
            ctx,
            &range,
            engine.config(),
            &vk,
            &proof,
        )
        .expect("stacked-reduction derive+constrain should succeed");
    });
}

#[test]
fn stacked_constraints_fail_on_tampered_intermediate_sum() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut actual = derive_stacked_reduction_intermediates(engine.config(), &vk, &proof)
        .expect("native stacked reduction must pass");

    actual.final_residual[0] = (actual.final_residual[0] + 1) % BABY_BEAR_MODULUS_U64;

    run_mock(false, move |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let _assigned = constrain_stacked_reduction_intermediates(ctx, &range, &actual);
    });
}

#[test]
fn stacked_constraints_reject_trailing_padded_q_coeff_width() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut actual = derive_stacked_reduction_intermediates(engine.config(), &vk, &proof)
        .expect("native stacked reduction must pass");
    actual.q_coeffs[0].push([0; BABY_BEAR_EXT_DEGREE]);
    actual.stacking_openings[0].push([0; BABY_BEAR_EXT_DEGREE]);

    run_mock(false, move |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let _assigned = constrain_stacked_reduction_intermediates(ctx, &range, &actual);
    });
}

#[test]
fn stacked_constraints_fail_on_coordinated_q_coeff_forgery() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut actual = derive_stacked_reduction_intermediates(engine.config(), &vk, &proof)
        .expect("native stacked reduction must pass");
    assert!(
        !actual.q_coeffs.is_empty() && !actual.q_coeffs[0].is_empty(),
        "fixture must include at least one q-coeff entry",
    );

    actual.q_coeffs[0][0][0] = (actual.q_coeffs[0][0][0] + 1) % BABY_BEAR_MODULUS_U64;
    let opening = coeffs_to_ext(actual.stacking_openings[0][0]);
    let delta = NativeEF::ONE * opening;
    let forged_final_sum = coeffs_to_ext(actual.final_sum) + delta;
    let forged_final_claim = coeffs_to_ext(actual.final_claim) + delta;
    actual.final_sum = ext_to_coeffs(forged_final_sum);
    actual.final_claim = ext_to_coeffs(forged_final_claim);
    actual.final_residual = [0; BABY_BEAR_EXT_DEGREE];

    run_mock(false, move |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let _assigned = constrain_stacked_reduction_intermediates(ctx, &range, &actual);
    });
}

#[test]
fn stacked_constraints_fail_on_coordinated_sumcheck_claim_chain_forgery() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut actual = derive_stacked_reduction_intermediates(engine.config(), &vk, &proof)
        .expect("native stacked reduction must pass");
    assert!(
        !actual.sumcheck_round_polys.is_empty(),
        "fixture must include stacked sumcheck rounds",
    );
    assert_eq!(
        actual.sumcheck_round_polys[0].len(),
        2,
        "stacked sumcheck rounds must expose [s(1), s(2)]",
    );

    actual.sumcheck_round_polys[0][0][0] =
        (actual.sumcheck_round_polys[0][0][0] + 1) % BABY_BEAR_MODULUS_U64;
    actual.final_claim = actual.final_sum;
    actual.final_residual = [0; BABY_BEAR_EXT_DEGREE];

    run_mock(false, move |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let _assigned = constrain_stacked_reduction_intermediates(ctx, &range, &actual);
    });
}

#[test]
fn stacked_rejects_s0_mismatch() {
    let engine = test_engine();
    let (vk, mut proof) = InteractionsFixture11.keygen_and_prove(&engine);

    proof.stacking_proof.univariate_round_coeffs[0] += NativeEF::ONE;

    let err = derive_stacked_reduction_intermediates(engine.config(), &vk, &proof)
        .expect_err("tampered univariate round should fail stacked s0 check");
    assert!(matches!(
        err,
        StackedReductionConstraintError::StackedReduction(StackedReductionError::S0Mismatch { .. })
    ));
}

#[test]
fn stacked_rejects_final_sum_mismatch() {
    let engine = test_engine();
    let (vk, mut proof) = InteractionsFixture11.keygen_and_prove(&engine);

    proof.stacking_proof.stacking_openings[0][0] += NativeEF::ONE;

    let err = derive_stacked_reduction_intermediates(engine.config(), &vk, &proof)
        .expect_err("tampered stacking opening should fail stacked final-sum check");
    assert!(matches!(
        err,
        StackedReductionConstraintError::StackedReduction(
            StackedReductionError::FinalSumMismatch { .. }
        )
    ));
}
