
use halo2_base::{
    gates::circuit::{CircuitBuilderStage, builder::BaseCircuitBuilder},
    halo2_proofs::dev::MockProver,
};
use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::{
        BabyBearBn254Poseidon2CpuEngine, EF as NativeEF, F as NativeF,
    },
    openvm_stark_backend::{
        StarkEngine,
        p3_field::{PrimeCharacteristicRing, PrimeField64},
        test_utils::{FibFixture, InteractionsFixture11, TestFixture, test_system_params_small},
        verifier::batch_constraints::BatchConstraintError as NativeBatchConstraintError,
    },
};

use crate::{
    config::{STATIC_VERIFIER_LOOKUP_ADVICE_COLS_PHASE0, STATIC_VERIFIER_NUM_ADVICE_COLS_PHASE0},
    gadgets::baby_bear::BABY_BEAR_MODULUS_U64,
};

use super::*;

fn run_mock(expect_satisfied: bool, build: impl FnOnce(&mut BaseCircuitBuilder<Fr>)) {
    const BATCH_K: u32 = 22;
    let mut builder = BaseCircuitBuilder::from_stage(CircuitBuilderStage::Mock)
        .use_k(BATCH_K as usize)
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

    let prover = MockProver::run(BATCH_K, &builder, vec![vec![]])
        .expect("mock prover should initialize for batch-constraints circuit");
    if expect_satisfied {
        prover.assert_satisfied();
    } else {
        assert!(
            prover.verify().is_err(),
            "expected constraints to fail for tampered batch-constraints intermediates"
        );
    }
}

fn assert_rejected_without_host_panic(run: impl FnOnce()) {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(run));
    assert!(
        result.is_ok(),
        "expected tamper rejection via constraints without host panic",
    );
}

fn test_engine() -> BabyBearBn254Poseidon2CpuEngine {
    BabyBearBn254Poseidon2CpuEngine::new(test_system_params_small(2, 8, 3))
}

#[test]
fn batch_intermediates_match_native_for_interactions_fixture() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);

    run_mock(true, move |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let assigned = derive_and_constrain_batch(ctx, &range, engine.config(), &vk, &proof)
            .expect("batch derive+constrain should succeed");

        range
            .gate()
            .assert_is_const(ctx, &assigned.logup_pow_witness_ok, &Fr::from(1u64));
    });
}

#[test]
fn ext_from_base_const_rejects_constant_family_pranks() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let actual = derive_batch_intermediates(engine.config(), &vk, &proof)
        .expect("native batch derivation must pass");

    let subgroup_root = NativeF::two_adic_generator(actual.l_skip).as_canonical_u64();
    let bus_constant = actual
        .trace_interactions
        .iter()
        .flat_map(|interactions| interactions.iter())
        .map(|interaction| u64::from(interaction.bus_index) + 1)
        .find(|&value| value > 1)
        .unwrap_or(1);
    let normalization_constant = (0..actual.l_skip)
        .fold(NativeF::ONE, |acc, _| acc.halve())
        .as_canonical_u64();

    let constants = [
        ("one", 1u64),
        ("two", 2u64),
        ("subgroup_root", subgroup_root),
        ("bus_index_plus_one", bus_constant),
        ("normalization", normalization_constant),
    ];

    for (family, constant) in constants {
        run_mock(false, move |builder| {
            let range = builder.range_chip();
            let ctx = builder.main(0);
            let baby_bear = BabyBearArithmeticGadgets;
            let ext = ext_from_base_const(ctx, &range, &baby_bear, constant);
            ext.coeffs[0]
                .cell
                .debug_prank(ctx, Fr::from((constant + 1) % BABY_BEAR_MODULUS_U64));
            let _ = family;
        });
    }
}

#[test]
fn batch_derivation_omits_q0_claim_when_total_interactions_is_zero() {
    let engine = test_engine();
    let (vk, proof) = FibFixture::new(0, 1, 1 << 5).keygen_and_prove(&engine);
    let actual = derive_batch_intermediates(engine.config(), &vk, &proof)
        .expect("native batch must pass for zero-interaction fixture");
    assert_eq!(
        actual.total_interactions, 0,
        "fixture must hit zero-interaction GKR branch",
    );
    assert!(
        actual.gkr_q0_claim.is_none(),
        "zero-interaction branch should not assign q0 claim witness",
    );

    run_mock(true, move |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let _assigned = constrain_batch_intermediates_unchecked(ctx, &range, &actual);
    });
}

#[test]
fn batch_constraints_reject_q0_claim_witness_when_total_interactions_is_zero() {
    let engine = test_engine();
    let (vk, proof) = FibFixture::new(0, 1, 1 << 5).keygen_and_prove(&engine);
    let mut actual = derive_batch_intermediates(engine.config(), &vk, &proof)
        .expect("native batch must pass for zero-interaction fixture");
    assert_eq!(
        actual.total_interactions, 0,
        "fixture must hit zero-interaction GKR branch",
    );
    actual.gkr_q0_claim = Some([1, 0, 0, 0]);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_mock(true, move |builder| {
            let range = builder.range_chip();
            let ctx = builder.main(0);
            let _assigned = constrain_batch_intermediates_unchecked(ctx, &range, &actual);
        });
    }));
    assert!(
        result.is_err(),
        "zero-interaction branch should reject q0 claim witness assignment",
    );
}

#[test]
fn batch_derivation_keeps_backend_parity_on_zero_interaction_tampered_q0_claim() {
    let engine = test_engine();
    let (vk, mut proof) = FibFixture::new(0, 1, 1 << 5).keygen_and_prove(&engine);
    proof.gkr_proof.q0_claim += NativeEF::ONE;

    let actual = derive_batch_intermediates(engine.config(), &vk, &proof)
        .expect("backend-equivalent derivation must ignore q0_claim when interactions are zero");
    assert_eq!(
        actual.total_interactions, 0,
        "fixture must hit zero-interaction GKR branch",
    );
    assert!(
        actual.gkr_q0_claim.is_none(),
        "backend parity path should still omit q0 claim witness in zero branch",
    );

    run_mock(true, move |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let _assigned = constrain_batch_intermediates_unchecked(ctx, &range, &actual);
    });
}

#[test]
fn batch_constraints_fail_on_tampered_intermediate_claims() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut actual = derive_batch_intermediates(engine.config(), &vk, &proof)
        .expect("native batch must pass");

    actual.consistency_residual[0] = (actual.consistency_residual[0] + 1) % BABY_BEAR_MODULUS_U64;

    run_mock(false, move |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let _assigned = constrain_batch_intermediates_unchecked(ctx, &range, &actual);
    });
}

#[test]
fn batch_constraints_reject_trailing_padded_column_openings() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut actual = derive_batch_intermediates(engine.config(), &vk, &proof)
        .expect("native batch must pass");
    actual.column_openings[0][0].push([0; BABY_BEAR_EXT_DEGREE]);

    run_mock(false, move |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let _assigned = constrain_batch_intermediates_unchecked(ctx, &range, &actual);
    });
}

#[test]
fn batch_constraints_reject_numerator_denominator_cardinality_mismatch() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut actual = derive_batch_intermediates(engine.config(), &vk, &proof)
        .expect("native batch must pass");
    assert!(
        !actual.denominator_term_per_air.is_empty(),
        "fixture must include denominator terms",
    );
    actual.denominator_term_per_air.pop();

    assert_rejected_without_host_panic(|| {
        run_mock(false, move |builder| {
            let range = builder.range_chip();
            let ctx = builder.main(0);
            let _assigned = constrain_batch_intermediates_unchecked(ctx, &range, &actual);
        });
    });
}

#[test]
fn batch_constraints_reject_sumcheck_round_count_suffix() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut actual = derive_batch_intermediates(engine.config(), &vk, &proof)
        .expect("native batch must pass");
    actual
        .sumcheck_round_polys
        .push(vec![[0; BABY_BEAR_EXT_DEGREE]; actual.batch_degree]);

    assert_rejected_without_host_panic(|| {
        run_mock(false, move |builder| {
            let range = builder.range_chip();
            let ctx = builder.main(0);
            let _assigned = constrain_batch_intermediates_unchecked(ctx, &range, &actual);
        });
    });
}

#[test]
fn batch_constraints_reject_univariate_coeff_arity_suffix() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut actual = derive_batch_intermediates(engine.config(), &vk, &proof)
        .expect("native batch must pass");
    actual
        .univariate_round_coeffs
        .push([0; BABY_BEAR_EXT_DEGREE]);

    assert_rejected_without_host_panic(|| {
        run_mock(false, move |builder| {
            let range = builder.range_chip();
            let ctx = builder.main(0);
            let _assigned = constrain_batch_intermediates_unchecked(ctx, &range, &actual);
        });
    });
}

#[test]
fn batch_constraints_fail_on_tampered_pow_sample_bits() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut actual = derive_batch_intermediates(engine.config(), &vk, &proof)
        .expect("native batch must pass");

    actual.logup_pow_sampled_bits = 1;
    actual.logup_pow_witness_ok = true;

    run_mock(false, move |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let _assigned = constrain_batch_intermediates_unchecked(ctx, &range, &actual);
    });
}

#[test]
fn batch_constraints_ignore_tampered_pow_witness_mirror() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut actual = derive_batch_intermediates(engine.config(), &vk, &proof)
        .expect("native batch must pass");
    actual.logup_pow_witness = (actual.logup_pow_witness + 1) % BABY_BEAR_MODULUS_U64;

    run_mock(true, move |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let _assigned = constrain_batch_intermediates_unchecked(ctx, &range, &actual);
    });
}

#[test]
fn batch_constraints_fail_on_tampered_gkr_layer_claims() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut actual = derive_batch_intermediates(engine.config(), &vk, &proof)
        .expect("native batch must pass");
    assert!(
        !actual.gkr_claims_per_layer.is_empty(),
        "fixture should contain GKR layer claims when interactions are present",
    );
    actual.gkr_claims_per_layer[0][0][0] =
        (actual.gkr_claims_per_layer[0][0][0] + 1) % BABY_BEAR_MODULUS_U64;

    run_mock(false, move |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let _assigned = constrain_batch_intermediates_unchecked(ctx, &range, &actual);
    });
}

#[test]
fn batch_constraints_fail_on_tampered_gkr_sumcheck_shape() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut actual = derive_batch_intermediates(engine.config(), &vk, &proof)
        .expect("native batch must pass");
    let first_round = actual
        .gkr_sumcheck_polys
        .first_mut()
        .expect("fixture should include at least one GKR sumcheck round");
    assert!(
        !first_round.is_empty(),
        "fixture should include GKR round evaluations",
    );
    first_round.pop();

    run_mock(false, move |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let _assigned = constrain_batch_intermediates_unchecked(ctx, &range, &actual);
    });
}

#[test]
fn batch_constraints_fail_on_coordinated_consistency_rhs_forgery() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut actual = derive_batch_intermediates(engine.config(), &vk, &proof)
        .expect("native batch must pass");
    assert!(
        !actual.trace_interactions.is_empty() && !actual.trace_interactions[0].is_empty(),
        "fixture should include at least one interaction for consistency replay tamper",
    );

    actual.trace_interactions[0][0].bus_index =
        actual.trace_interactions[0][0].bus_index.wrapping_add(1);
    actual.consistency_rhs = actual.consistency_lhs;
    actual.consistency_residual = [0; BABY_BEAR_EXT_DEGREE];

    run_mock(false, move |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let _assigned = constrain_batch_intermediates_unchecked(ctx, &range, &actual);
    });
}

#[test]
fn batch_rejects_invalid_pow_witness() {
    let engine = test_engine();
    let (vk, mut proof) = InteractionsFixture11.keygen_and_prove(&engine);

    let old = proof.gkr_proof.logup_pow_witness.as_canonical_u64();
    let mut found_invalid = false;
    for delta in 1..=1024u64 {
        proof.gkr_proof.logup_pow_witness =
            NativeF::from_u64((old + delta) % BABY_BEAR_MODULUS_U64);
        let result = derive_batch_intermediates(engine.config(), &vk, &proof);
        if matches!(
            result,
            Err(BatchConstraintError::BatchConstraint(
                NativeBatchConstraintError::InvalidLogupPowWitness
            ))
        ) {
            found_invalid = true;
            break;
        }
    }
    assert!(
        found_invalid,
        "failed to find an invalid logup PoW witness candidate"
    );
}

#[test]
fn batch_rejects_gkr_numerator_mismatch() {
    let engine = test_engine();
    let (vk, mut proof) = InteractionsFixture11.keygen_and_prove(&engine);

    proof.batch_constraint_proof.numerator_term_per_air[0] += NativeEF::ONE;

    let err = derive_batch_intermediates(engine.config(), &vk, &proof)
        .expect_err("tampered numerator term should fail GKR numerator check");
    assert!(matches!(
        err,
        BatchConstraintError::BatchConstraint(NativeBatchConstraintError::GkrNumeratorMismatch { .. })
    ));
}

#[test]
fn batch_rejects_gkr_denominator_mismatch() {
    let engine = test_engine();
    let (vk, mut proof) = InteractionsFixture11.keygen_and_prove(&engine);

    proof.batch_constraint_proof.denominator_term_per_air[0] += NativeEF::ONE;

    let err = derive_batch_intermediates(engine.config(), &vk, &proof)
        .expect_err("tampered denominator term should fail GKR denominator check");
    assert!(matches!(
        err,
        BatchConstraintError::BatchConstraint(
            NativeBatchConstraintError::GkrDenominatorMismatch { .. }
        )
    ));
}

#[test]
fn batch_rejects_sum_claim_mismatch() {
    let engine = test_engine();
    let (vk, mut proof) = InteractionsFixture11.keygen_and_prove(&engine);

    proof.batch_constraint_proof.univariate_round_coeffs[0] += NativeEF::ONE;

    let err = derive_batch_intermediates(engine.config(), &vk, &proof)
        .expect_err("tampered univariate coefficients should fail sum-claim check");
    assert!(matches!(
        err,
        BatchConstraintError::BatchConstraint(NativeBatchConstraintError::SumClaimMismatch { .. })
    ));
}

#[test]
fn batch_rejects_inconsistent_claims() {
    let engine = test_engine();
    let (vk, mut proof) = InteractionsFixture11.keygen_and_prove(&engine);

    proof.batch_constraint_proof.column_openings[0][0][0] += NativeEF::ONE;

    let err = derive_batch_intermediates(engine.config(), &vk, &proof)
        .expect_err("tampered openings should fail consistency check");
    assert!(matches!(
        err,
        BatchConstraintError::BatchConstraint(NativeBatchConstraintError::InconsistentClaims)
    ));
}
