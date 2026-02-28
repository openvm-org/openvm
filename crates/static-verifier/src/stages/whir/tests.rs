
use halo2_base::{
    gates::circuit::{CircuitBuilderStage, builder::BaseCircuitBuilder},
    halo2_proofs::dev::MockProver,
};
use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::{BabyBearBn254Poseidon2CpuEngine, F as NativeF},
    openvm_stark_backend::{
        StarkEngine,
        p3_field::{PrimeCharacteristicRing, PrimeField64},
        test_utils::{InteractionsFixture11, TestFixture, test_system_params_small},
        verifier::whir::VerifyWhirError,
    },
};

use crate::{
    config::{STATIC_VERIFIER_LOOKUP_ADVICE_COLS_PHASE0, STATIC_VERIFIER_NUM_ADVICE_COLS_PHASE0},
    gadgets::baby_bear::BABY_BEAR_MODULUS_U64,
};

use super::*;

fn run_mock(expect_satisfied: bool, build: impl FnOnce(&mut BaseCircuitBuilder<Fr>)) {
    const MOCK_K: u32 = 22;
    const MOCK_MIN_ROWS: usize = 32768;
    let mut builder = BaseCircuitBuilder::from_stage(CircuitBuilderStage::Mock)
        .use_k(MOCK_K as usize)
        .use_lookup_bits(8)
        .use_instance_columns(1);
    build(&mut builder);

    let params = builder.calculate_params(Some(MOCK_MIN_ROWS));
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
        .expect("mock prover should initialize for whir circuit");
    if expect_satisfied {
        prover.assert_satisfied();
    } else {
        assert!(
            prover.verify().is_err(),
            "expected constraints to fail for tampered whir intermediates"
        );
    }
}

fn test_engine() -> BabyBearBn254Poseidon2CpuEngine {
    BabyBearBn254Poseidon2CpuEngine::new(test_system_params_small(2, 8, 3))
}

#[test]
fn whir_intermediates_match_native_for_interactions_fixture() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);

    run_mock(true, move |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let assigned = derive_and_constrain_whir(ctx, &range, engine.config(), &vk, &proof)
            .expect("whir derive+constrain should succeed");

        range
            .gate()
            .assert_is_const(ctx, &assigned.mu_pow_witness_ok, &Fr::from(1u64));
    });
}

#[test]
fn whir_strict_constraints_reject_forged_standalone_metadata() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut raw = derive_raw_whir_witness_state(engine.config(), &vk, &proof)
        .expect("raw whir witness derivation must pass");
    let ownership = derive_whir_strict_ownership(engine.config(), &vk, &proof)
        .expect("strict whir ownership derivation must pass");
    raw.intermediates.mu_pow_bits = raw.intermediates.mu_pow_bits.saturating_add(1);
    let raw_for_unchecked = raw.clone();

    run_mock(true, move |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let _unchecked =
            constrain_checked_whir_witness_state_unchecked(ctx, &range, &raw_for_unchecked);
    });

    run_mock(false, move |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let _strict = constrain_checked_whir_witness_state_strict(ctx, &range, &raw, &ownership);
    });
}

#[test]
fn whir_constraints_fail_on_tampered_intermediate_final_claim() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut actual = derive_whir_intermediates(engine.config(), &vk, &proof)
        .expect("native whir must pass");

    actual.final_residual[0] = (actual.final_residual[0] + 1) % BABY_BEAR_MODULUS_U64;

    run_mock(false, move |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let _assigned = constrain_whir_intermediates_unchecked(ctx, &range, &actual);
    });
}

#[test]
fn whir_constraints_fail_on_coordinated_final_claim_forgery() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut actual = derive_whir_intermediates(engine.config(), &vk, &proof)
        .expect("native whir must pass");

    actual.final_claim[0] = (actual.final_claim[0] + 1) % BABY_BEAR_MODULUS_U64;
    actual.final_acc[0] = (actual.final_acc[0] + 1) % BABY_BEAR_MODULUS_U64;
    // Keep residual mirror at zero so legacy residual-only checks would accept.
    actual.final_residual = [0; BABY_BEAR_EXT_DEGREE];

    run_mock(false, move |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let _assigned = constrain_whir_intermediates_unchecked(ctx, &range, &actual);
    });
}

#[test]
fn whir_constraints_fail_on_tampered_pow_sample_bits() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut actual = derive_whir_intermediates(engine.config(), &vk, &proof)
        .expect("native whir must pass");
    actual.mu_pow_sampled_bits = 1;
    actual.mu_pow_witness_ok = true;

    run_mock(false, move |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let _assigned = constrain_whir_intermediates_unchecked(ctx, &range, &actual);
    });
}

#[test]
fn whir_constraints_ignore_tampered_pow_witness_mirrors() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut actual = derive_whir_intermediates(engine.config(), &vk, &proof)
        .expect("native whir must pass");

    actual.mu_pow_witness = (actual.mu_pow_witness + 1) % BABY_BEAR_MODULUS_U64;
    if let Some(first) = actual.folding_pow_witnesses.first_mut() {
        *first = (*first + 1) % BABY_BEAR_MODULUS_U64;
    }
    if let Some(first) = actual.query_phase_pow_witnesses.first_mut() {
        *first = (*first + 1) % BABY_BEAR_MODULUS_U64;
    }

    run_mock(true, move |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let _assigned = constrain_whir_intermediates_unchecked(ctx, &range, &actual);
    });
}

#[test]
fn whir_constraints_fail_on_tampered_query_index() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut actual = derive_whir_intermediates(engine.config(), &vk, &proof)
        .expect("native whir must pass");
    actual.query_indices[0] += 1;

    run_mock(false, move |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let _assigned = constrain_whir_intermediates_unchecked(ctx, &range, &actual);
    });
}

#[test]
fn whir_constraints_fail_on_tampered_merkle_path() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut actual = derive_whir_intermediates(engine.config(), &vk, &proof)
        .expect("native whir must pass");
    actual.merkle_paths[0].siblings[0] += Fr::from(1u64);

    run_mock(false, move |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let _assigned = constrain_whir_intermediates_unchecked(ctx, &range, &actual);
    });
}

#[test]
fn whir_constraints_fail_on_merkle_depth_query_bit_mismatch() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut actual = derive_whir_intermediates(engine.config(), &vk, &proof)
        .expect("native whir must pass");
    let path = actual
        .merkle_paths
        .first_mut()
        .expect("fixture should include at least one Merkle path");
    path.siblings.push(Fr::from(0u64));

    run_mock(false, move |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let _assigned = constrain_whir_intermediates_unchecked(ctx, &range, &actual);
    });
}

#[test]
fn whir_constraints_fail_on_missing_coverage_tuple() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut actual = derive_whir_intermediates(engine.config(), &vk, &proof)
        .expect("native whir must pass");
    assert!(
        actual.merkle_paths.len() >= 2,
        "fixture should produce at least two Merkle paths"
    );
    actual.merkle_paths[1].query_position = actual.merkle_paths[0].query_position;

    run_mock(false, move |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let _assigned = constrain_whir_intermediates_unchecked(ctx, &range, &actual);
    });
}

#[test]
fn whir_constraints_fail_on_wrong_root_binding() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut actual = derive_whir_intermediates(engine.config(), &vk, &proof)
        .expect("native whir must pass");
    actual.initial_commitment_roots[0] += Fr::from(1u64);

    run_mock(false, move |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let _assigned = constrain_whir_intermediates_unchecked(ctx, &range, &actual);
    });
}

#[test]
fn whir_constraints_fail_on_tampered_final_poly_len() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut actual = derive_whir_intermediates(engine.config(), &vk, &proof)
        .expect("native whir must pass");
    actual.final_poly_len += 1;

    run_mock(false, move |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let _assigned = constrain_whir_intermediates_unchecked(ctx, &range, &actual);
    });
}

#[test]
fn whir_rejects_tampered_merkle_opening() {
    let engine = test_engine();
    let (vk, mut proof) = InteractionsFixture11.keygen_and_prove(&engine);

    proof.whir_proof.initial_round_opened_rows[0][0][0][0] += NativeF::ONE;

    let err = derive_whir_intermediates(engine.config(), &vk, &proof)
        .expect_err("tampered initial opened row should fail merkle verification");
    assert!(matches!(
        err,
        WhirError::Whir(VerifyWhirError::MerkleVerify)
    ));
}

#[test]
fn whir_rejects_invalid_mu_pow_witness() {
    let engine = test_engine();
    let (vk, mut proof) = InteractionsFixture11.keygen_and_prove(&engine);

    let old = proof.whir_proof.mu_pow_witness.as_canonical_u64();
    let mut found_invalid = false;
    for delta in 1..=4096u64 {
        proof.whir_proof.mu_pow_witness = NativeF::from_u64((old + delta) % BABY_BEAR_MODULUS_U64);
        let result = derive_whir_intermediates(engine.config(), &vk, &proof);
        if matches!(
            result,
            Err(WhirError::Whir(VerifyWhirError::MuPoWInvalid))
        ) {
            found_invalid = true;
            break;
        }
    }

    assert!(
        found_invalid,
        "failed to find an invalid mu PoW witness candidate"
    );
}

#[test]
fn whir_rejects_tampered_final_poly_constraint_input() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);

    let PreparedWhirInputs {
        mut transcript,
        commits,
        mut u_cube,
    } = prepare_whir_inputs(engine.config(), &vk, &proof)
        .expect("setup should succeed before direct whir constraint check");

    u_cube[0] += NativeEF::ONE;

    let err = derive_whir_intermediates_with_inputs(
        &mut transcript,
        engine.config(),
        &proof.whir_proof,
        &proof.stacking_proof.stacking_openings,
        &commits,
        &u_cube,
    )
    .expect_err("tampered final polynomial constraint input should fail");

    assert!(matches!(err, VerifyWhirError::FinalPolyConstraint));
}
