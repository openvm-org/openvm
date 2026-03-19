use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::{
        BabyBearBn254Poseidon2Config as RootConfig, BabyBearBn254Poseidon2CpuEngine,
    },
    openvm_stark_backend::{
        keygen::types::LinearConstraint,
        test_utils::{
            test_system_params_small, CachedFixture11, FibFixture, InteractionsFixture11,
            PreprocessedFibFixture, TestFixture,
        },
        StarkEngine,
    },
};

use super::*;

fn test_engine() -> BabyBearBn254Poseidon2CpuEngine {
    BabyBearBn254Poseidon2CpuEngine::new(test_system_params_small(2, 8, 3))
}

fn assert_fixture_derives_successfully<Fx>(engine: &BabyBearBn254Poseidon2CpuEngine, fixture: Fx)
where
    Fx: TestFixture<RootConfig>,
{
    let (vk, proof) = fixture.keygen_and_prove(engine);
    derive_proof_shape_intermediates(engine.config(), &vk, &proof)
        .expect("native proof-shape derivation must pass");
}

#[test]
fn proof_shape_intermediates_derive_for_reference_fixtures() {
    let engine = test_engine();

    assert_fixture_derives_successfully(&engine, FibFixture::new(0, 1, 1 << 5));
    assert_fixture_derives_successfully(&engine, InteractionsFixture11);
    assert_fixture_derives_successfully(&engine, CachedFixture11::new(engine.config().clone()));

    let height = 1 << 5;
    let sels = (0..height).map(|i| i % 2 == 0).collect::<Vec<_>>();
    assert_fixture_derives_successfully(&engine, PreprocessedFibFixture::new(0, 1, sels));
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
