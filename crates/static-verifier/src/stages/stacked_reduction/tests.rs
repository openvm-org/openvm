use halo2_base::{
    gates::{
        circuit::{builder::BaseCircuitBuilder, CircuitBuilderStage},
        range::RangeChip,
    },
    halo2_proofs::dev::MockProver,
};
use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2CpuEngine,
    openvm_stark_backend::{
        p3_field::{BasedVectorSpace, PrimeCharacteristicRing, PrimeField64},
        test_utils::{test_system_params_small, InteractionsFixture11, TestFixture},
        verifier::stacked_reduction::StackedReductionError,
        StarkEngine,
    },
};

use super::*;
use crate::{
    config::{STATIC_VERIFIER_LOOKUP_ADVICE_COLS_PHASE0, STATIC_VERIFIER_NUM_ADVICE_COLS_PHASE0},
    field::baby_bear::{BabyBearChip, BabyBearExtChip, BABY_BEAR_MODULUS_U64},
    ChildEF, ChildF,
};

fn ext_to_coeffs(value: ChildEF) -> [u64; BABY_BEAR_EXT_DEGREE] {
    core::array::from_fn(|i| {
        <ChildEF as BasedVectorSpace<ChildF>>::as_basis_coefficients_slice(&value)[i]
            .as_canonical_u64()
    })
}

fn coeffs_to_ext(coeffs: [u64; BABY_BEAR_EXT_DEGREE]) -> ChildEF {
    ChildEF::from_basis_coefficients_fn(|i| ChildF::from_u64(coeffs[i]))
}

fn make_ext_chip(range: &RangeChip<Fr>) -> BabyBearExtChip<'_> {
    BabyBearExtChip::new(BabyBearChip::new(range))
}

/// Standalone stacked-reduction derive+constrain wrapper is internal; external callers must use
/// transcript-owned stage composition (`stages::full_pipeline`) as the acceptance boundary.
fn derive_and_constrain_stacked_reduction(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    config: &NativeConfig,
    mvk: &MultiStarkVerifyingKey<NativeConfig>,
    proof: &Proof<NativeConfig>,
) -> Result<AssignedStackedReductionIntermediates, StackedReductionConstraintError> {
    let raw = derive_raw_stacked_witness_state(config, mvk, proof)?;
    let ext_chip = make_ext_chip(range);
    Ok(constrain_checked_stacked_witness_state(ctx, &ext_chip, &raw).assigned)
}

fn derive_raw_stacked_witness_state(
    config: &NativeConfig,
    mvk: &MultiStarkVerifyingKey<NativeConfig>,
    proof: &Proof<NativeConfig>,
) -> Result<RawStackedWitnessState, StackedReductionConstraintError> {
    Ok(RawStackedWitnessState {
        intermediates: derive_stacked_reduction_intermediates(config, mvk, proof)?,
    })
}

fn constrain_checked_stacked_witness_state(
    ctx: &mut Context<Fr>,
    ext_chip: &BabyBearExtChip<'_>,
    raw: &RawStackedWitnessState,
) -> CheckedStackedWitnessState {
    let assigned = constrain_stacked_reduction_intermediates_with_shared_inputs(
        ctx,
        ext_chip,
        &raw.intermediates,
        None,
        None,
    );
    let derived = DerivedStackedState {
        s_0_residual: assigned.s_0_residual,
        final_residual: assigned.final_residual,
    };
    CheckedStackedWitnessState { assigned, derived }
}

fn constrain_stacked_reduction_for_test(
    ctx: &mut Context<Fr>,
    ext_chip: &BabyBearExtChip<'_>,
    actual: &StackedReductionIntermediates,
) -> AssignedStackedReductionIntermediates {
    constrain_stacked_reduction_intermediates_with_shared_inputs(ctx, ext_chip, actual, None, None)
}

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
        let _assigned =
            derive_and_constrain_stacked_reduction(ctx, &range, engine.config(), &vk, &proof)
                .expect("stacked-reduction derive+constrain should succeed");
    });
}

#[test]
fn stacked_constraints_fail_on_tampered_intermediate_sum() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut actual = derive_stacked_reduction_intermediates(engine.config(), &vk, &proof)
        .expect("native stacked reduction must pass");

    {
        let mut coeffs = ext_to_coeffs(actual.final_residual);
        coeffs[0] = (coeffs[0] + 1) % BABY_BEAR_MODULUS_U64;
        actual.final_residual = coeffs_to_ext(coeffs);
    }

    run_mock(false, move |builder| {
        let range = builder.range_chip();
        let ext_chip = make_ext_chip(&range);
        let ctx = builder.main(0);
        let _assigned = constrain_stacked_reduction_for_test(ctx, &ext_chip, &actual);
    });
}

#[test]
fn stacked_constraints_reject_trailing_padded_q_coeff_width() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut actual = derive_stacked_reduction_intermediates(engine.config(), &vk, &proof)
        .expect("native stacked reduction must pass");
    actual.q_coeffs[0].push(ChildEF::ZERO);
    actual.stacking_openings[0].push(ChildEF::ZERO);

    run_mock(false, move |builder| {
        let range = builder.range_chip();
        let ext_chip = make_ext_chip(&range);
        let ctx = builder.main(0);
        let _assigned = constrain_stacked_reduction_for_test(ctx, &ext_chip, &actual);
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

    {
        let mut coeffs = ext_to_coeffs(actual.q_coeffs[0][0]);
        coeffs[0] = (coeffs[0] + 1) % BABY_BEAR_MODULUS_U64;
        actual.q_coeffs[0][0] = coeffs_to_ext(coeffs);
    }
    let opening = actual.stacking_openings[0][0];
    let delta = ChildEF::ONE * opening;
    let forged_final_sum = actual.final_sum + delta;
    let forged_final_claim = actual.final_claim + delta;
    actual.final_sum = forged_final_sum;
    actual.final_claim = forged_final_claim;
    actual.final_residual = ChildEF::ZERO;

    run_mock(false, move |builder| {
        let range = builder.range_chip();
        let ext_chip = make_ext_chip(&range);
        let ctx = builder.main(0);
        let _assigned = constrain_stacked_reduction_for_test(ctx, &ext_chip, &actual);
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

    {
        let mut coeffs = ext_to_coeffs(actual.sumcheck_round_polys[0][0]);
        coeffs[0] = (coeffs[0] + 1) % BABY_BEAR_MODULUS_U64;
        actual.sumcheck_round_polys[0][0] = coeffs_to_ext(coeffs);
    }
    actual.final_claim = actual.final_sum;
    actual.final_residual = ChildEF::ZERO;

    run_mock(false, move |builder| {
        let range = builder.range_chip();
        let ext_chip = make_ext_chip(&range);
        let ctx = builder.main(0);
        let _assigned = constrain_stacked_reduction_for_test(ctx, &ext_chip, &actual);
    });
}

#[test]
fn stacked_rejects_s0_mismatch() {
    let engine = test_engine();
    let (vk, mut proof) = InteractionsFixture11.keygen_and_prove(&engine);

    proof.stacking_proof.univariate_round_coeffs[0] += ChildEF::ONE;

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

    proof.stacking_proof.stacking_openings[0][0] += ChildEF::ONE;

    let err = derive_stacked_reduction_intermediates(engine.config(), &vk, &proof)
        .expect_err("tampered stacking opening should fail stacked final-sum check");
    assert!(matches!(
        err,
        StackedReductionConstraintError::StackedReduction(
            StackedReductionError::FinalSumMismatch { .. }
        )
    ));
}
