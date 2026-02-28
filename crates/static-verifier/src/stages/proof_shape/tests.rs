use halo2_base::{
    gates::circuit::{CircuitBuilderStage, builder::BaseCircuitBuilder},
    halo2_proofs::dev::MockProver,
};
use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::{
        BabyBearBn254Poseidon2Config as NativeConfig, BabyBearBn254Poseidon2CpuEngine,
    },
    openvm_stark_backend::{
        StarkEngine,
        keygen::types::LinearConstraint,
        test_utils::{
            CachedFixture11, FibFixture, InteractionsFixture11, PreprocessedFibFixture,
            TestFixture, test_system_params_small,
        },
    },
};

use crate::config::{
    STATIC_VERIFIER_LOOKUP_ADVICE_COLS_PHASE0, STATIC_VERIFIER_NUM_ADVICE_COLS_PHASE0,
};
use crate::utils::usize_to_u64;

use super::*;

fn run_mock(expect_satisfied: bool, build: impl FnOnce(&mut BaseCircuitBuilder<Fr>)) {
    let mut builder = BaseCircuitBuilder::from_stage(CircuitBuilderStage::Mock)
        .use_k(15)
        .use_lookup_bits(8)
        .use_instance_columns(1);
    build(&mut builder);

    let params = builder.calculate_params(Some(256));
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

    let prover = MockProver::run(15, &builder, vec![vec![]])
        .expect("mock prover should initialize for proof-shape circuit");
    if expect_satisfied {
        prover.assert_satisfied();
    } else {
        assert!(
            prover.verify().is_err(),
            "expected constraints to fail for tampered proof-shape intermediates"
        );
    }
}

fn test_engine() -> BabyBearBn254Poseidon2CpuEngine {
    BabyBearBn254Poseidon2CpuEngine::new(test_system_params_small(2, 8, 3))
}

fn assert_fixture_matches_native<Fx>(engine: &BabyBearBn254Poseidon2CpuEngine, fixture: Fx)
where
    Fx: TestFixture<NativeConfig>,
{
    let (vk, proof) = fixture.keygen_and_prove(engine);
    let actual =
        derive_proof_shape_intermediates(engine.config(), &vk, &proof).expect("native proof-shape must pass");

    run_mock(true, move |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let assigned = derive_and_constrain_proof_shape(ctx, &range, engine.config(), &vk, &proof)
            .expect("proof-shape derive+constrain should succeed");

        range.gate().assert_is_const(
            ctx,
            &assigned.num_airs_present,
            &Fr::from(usize_to_u64(actual.num_airs_present)),
        );
    });
}

#[test]
fn proof_shape_intermediates_match_native_for_reference_fixtures() {
    let engine = test_engine();

    assert_fixture_matches_native(&engine, FibFixture::new(0, 1, 1 << 5));
    assert_fixture_matches_native(&engine, InteractionsFixture11);
    assert_fixture_matches_native(&engine, CachedFixture11::new(engine.config().clone()));

    let height = 1 << 5;
    let sels = (0..height).map(|i| i % 2 == 0).collect::<Vec<_>>();
    assert_fixture_matches_native(&engine, PreprocessedFibFixture::new(0, 1, sels));
}

#[test]
fn proof_shape_constraints_fail_when_trace_height_sum_is_tampered() {
    let engine = test_engine();
    let (vk, proof) = FibFixture::new(5, 8, 1 << 5).keygen_and_prove(&engine);
    let mut actual =
        derive_proof_shape_intermediates(engine.config(), &vk, &proof).expect("native proof-shape must pass");
    actual.trace_height_sums[0] = actual.trace_height_thresholds[0];

    run_mock(false, move |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let _assigned = constrain_proof_shape_intermediates(ctx, &range, &actual);
    });
}

#[test]
fn proof_shape_constraints_fail_when_required_air_presence_rule_is_tampered() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut actual =
        derive_proof_shape_intermediates(engine.config(), &vk, &proof).expect("native proof-shape must pass");
    actual.air_required_flags[0] = true;
    actual.air_presence_flags[0] = false;

    run_mock(false, move |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let _assigned = constrain_proof_shape_intermediates(ctx, &range, &actual);
    });
}

#[test]
fn proof_shape_constraints_ignore_proof_shape_checklist_mirror_tamper() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut actual =
        derive_proof_shape_intermediates(engine.config(), &vk, &proof).expect("native proof-shape must pass");
    actual.proof_shape_count_checks[0].0 += 1;

    run_mock(true, move |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let _assigned = constrain_proof_shape_intermediates(ctx, &range, &actual);
    });
}

#[test]
fn proof_shape_constraints_fail_when_trace_order_is_tampered() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut actual =
        derive_proof_shape_intermediates(engine.config(), &vk, &proof).expect("native proof-shape must pass");
    assert!(
        actual.trace_id_to_air_id.len() >= 2,
        "fixture should include at least two traces for ordering tamper test",
    );
    actual.trace_id_to_air_id.swap(0, 1);

    run_mock(false, move |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let _assigned = constrain_proof_shape_intermediates(ctx, &range, &actual);
    });
}

#[test]
fn proof_shape_rejects_trace_height_threshold_violations() {
    let engine = test_engine();
    let (mut vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);

    vk.inner.trace_height_constraints.push(LinearConstraint {
        coefficients: vec![1; vk.inner.per_air.len()],
        threshold: 0,
    });

    let err = derive_proof_shape_intermediates(engine.config(), &vk, &proof)
        .expect_err("trace-height guard should fail when threshold is forced to zero");
    assert!(matches!(
        err,
        ProofShapePreambleError::TraceHeightsTooLarge { .. }
    ));
}

#[test]
fn proof_shape_rules_reject_empty_trace_set() {
    let engine = test_engine();
    let (mut vk, mut proof) = InteractionsFixture11.keygen_and_prove(&engine);

    for air_vk in &mut vk.inner.per_air {
        air_vk.is_required = false;
    }
    for vdata in &mut proof.trace_vdata {
        *vdata = None;
    }
    for pvs in &mut proof.public_values {
        pvs.clear();
    }
    proof.gkr_proof.claims_per_layer.clear();
    proof.gkr_proof.sumcheck_polys.clear();

    let err = derive_proof_shape_rules(&vk.inner, &proof)
        .expect_err("empty per-trace set should be rejected explicitly");
    assert!(matches!(
        err,
        ProofShapeError::InvalidBatchConstraintProofShape(
            BatchProofShapeError::InvalidColumnOpeningsAirs { expected: 1, .. }
        )
    ));
}

#[test]
fn proof_shape_rules_reject_empty_stacked_layout_columns() {
    let engine = test_engine();
    let (mut vk, mut proof) = InteractionsFixture11.keygen_and_prove(&engine);

    let l_skip = vk.inner.params.l_skip;
    for (air_idx, air_vk) in vk.inner.per_air.iter_mut().enumerate() {
        air_vk.params.width.common_main = 0;
        air_vk.params.width.preprocessed = None;
        air_vk.params.width.cached_mains.clear();
        air_vk.params.num_public_values = 0;
        air_vk.symbolic_constraints.interactions.clear();
        if let Some(vdata) = &mut proof.trace_vdata[air_idx] {
            vdata.log_height = l_skip;
            vdata.cached_commitments.clear();
        }
    }
    for pvs in &mut proof.public_values {
        pvs.clear();
    }

    proof.gkr_proof.claims_per_layer.clear();
    proof.gkr_proof.sumcheck_polys.clear();
    proof.batch_constraint_proof.sumcheck_round_polys.clear();
    for per_air in &mut proof.batch_constraint_proof.column_openings {
        *per_air = vec![Vec::new()];
    }
    proof.stacking_proof.stacking_openings = vec![Vec::new()];

    let err = derive_proof_shape_rules(&vk.inner, &proof)
        .expect_err("empty stacked-layout column schedule should be rejected explicitly");
    assert!(matches!(
        err,
        ProofShapeError::InvalidStackingProofShape(
            StackingProofShapeError::InvalidStackOpeningsPerMatrix {
                commit_idx: 0,
                expected: 1,
                ..
            }
        )
    ));
}
