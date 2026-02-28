
use halo2_base::{
    gates::{
        GateInstructions, RangeInstructions,
        circuit::{CircuitBuilderStage, builder::BaseCircuitBuilder},
    },
    halo2_proofs::{
        dev::MockProver,
        halo2curves::bn256::{Bn256, G1Affine},
        plonk::{
            Circuit, ProvingKey, VerifyingKey, create_proof, keygen_pk, keygen_vk, verify_proof,
        },
        poly::commitment::ParamsProver,
        poly::kzg::{
            commitment::{KZGCommitmentScheme, ParamsKZG},
            multiopen::{ProverSHPLONK, VerifierSHPLONK},
            strategy::SingleStrategy,
        },
        transcript::{
            Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
        },
    },
    utils::fs::gen_srs,
};
use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::{
        BabyBearBn254Poseidon2Config as NativeConfig, BabyBearBn254Poseidon2CpuEngine,
        EF as NativeEF, F as NativeF,
    },
    openvm_stark_backend::{
        StarkEngine,
        p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing, PrimeField64, TwoAdicField},
        test_utils::{
            CachedFixture11, FibFixture, InteractionsFixture11, PreprocessedFibFixture,
            TestFixture, test_system_params_small,
        },
    },
};
use rand_chacha::{ChaCha20Rng, rand_core::SeedableRng};

use crate::{
    config::{STATIC_VERIFIER_LOOKUP_ADVICE_COLS_PHASE0, STATIC_VERIFIER_NUM_ADVICE_COLS_PHASE0},
    gadgets::baby_bear::BABY_BEAR_MODULUS_U64,
    stages::batch_constraints::{
        RecordedExtBaseConst, clear_recorded_ext_base_consts, take_recorded_ext_base_consts,
    },
};

use super::*;

const END_TO_END_K: u32 = 22;
const END_TO_END_LOOKUP_BITS: usize = 8;
const END_TO_END_MIN_ROWS: usize = 32768;

fn run_mock(
    expect_satisfied: bool,
    public_inputs: &[Fr],
    build: impl FnOnce(&mut BaseCircuitBuilder<Fr>),
) {
    let mut builder = BaseCircuitBuilder::from_stage(CircuitBuilderStage::Mock)
        .use_k(END_TO_END_K as usize)
        .use_lookup_bits(END_TO_END_LOOKUP_BITS)
        .use_instance_columns(1);
    build(&mut builder);

    let params = builder.calculate_params(Some(END_TO_END_MIN_ROWS));
    assert!(
        params
            .num_advice_per_phase
            .first()
            .copied()
            .unwrap_or_default()
            >= STATIC_VERIFIER_NUM_ADVICE_COLS_PHASE0
    );
    assert!(
        params
            .num_lookup_advice_per_phase
            .first()
            .copied()
            .unwrap_or_default()
            >= STATIC_VERIFIER_LOOKUP_ADVICE_COLS_PHASE0
    );

    let prover = MockProver::run(END_TO_END_K, &builder, vec![public_inputs.to_vec()])
        .expect("mock prover should initialize for pipeline end-to-end circuit");
    if expect_satisfied {
        prover.assert_satisfied();
    } else {
        assert!(
            prover.verify().is_err(),
            "expected constraints to fail for tampered pipeline intermediates"
        );
    }
}

fn run_mock_light(expect_satisfied: bool, build: impl FnOnce(&mut BaseCircuitBuilder<Fr>)) {
    let mut builder = BaseCircuitBuilder::from_stage(CircuitBuilderStage::Mock)
        .use_k(12)
        .use_lookup_bits(8)
        .use_instance_columns(1);
    build(&mut builder);
    let _ = builder.calculate_params(Some(512));

    let prover = MockProver::run(12, &builder, vec![vec![]])
        .expect("mock prover should initialize for lightweight pipeline helper checks");
    if expect_satisfied {
        prover.assert_satisfied();
    } else {
        assert!(
            prover.verify().is_err(),
            "expected lightweight helper constraints to fail",
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

fn gen_halo2_proof(
    params: &ParamsKZG<Bn256>,
    pk: &ProvingKey<G1Affine>,
    circuit: impl Circuit<Fr>,
    public_inputs: &[Fr],
) -> Vec<u8> {
    let rng = ChaCha20Rng::from_seed(Default::default());
    let instances: &[&[Fr]] = &[public_inputs];
    let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
    create_proof::<
        KZGCommitmentScheme<Bn256>,
        ProverSHPLONK<'_, Bn256>,
        Challenge255<_>,
        _,
        Blake2bWrite<Vec<u8>, G1Affine, _>,
        _,
    >(params, pk, &[circuit], &[instances], rng, &mut transcript)
    .expect("prover should not fail");
    transcript.finalize()
}

fn verify_halo2_proof(
    params: &ParamsKZG<Bn256>,
    vk: &VerifyingKey<G1Affine>,
    proof: &[u8],
    public_inputs: &[Fr],
) {
    let verifier_params = params.verifier_params();
    let strategy = SingleStrategy::new(params);
    let instances: &[&[Fr]] = &[public_inputs];
    let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(proof);
    verify_proof::<
        KZGCommitmentScheme<Bn256>,
        VerifierSHPLONK<'_, Bn256>,
        Challenge255<G1Affine>,
        Blake2bRead<&[u8], G1Affine, Challenge255<G1Affine>>,
        SingleStrategy<'_, Bn256>,
    >(verifier_params, vk, strategy, &[instances], &mut transcript)
    .expect("verifier should accept valid pipeline proof");
}

fn build_end_to_end_constraints_from_proof(
    builder: &mut BaseCircuitBuilder<Fr>,
    config: &NativeConfig,
    vk: &MultiStarkVerifyingKey<NativeConfig>,
    proof: &Proof<NativeConfig>,
) {
    let range = builder.range_chip();
    let public_input_cells = {
        let ctx = builder.main(0);
        let assigned = derive_and_constrain_pipeline(ctx, &range, config, vk, proof)
            .expect("pipeline derive+constrain should succeed");

        range
            .gate()
            .assert_is_const(ctx, &assigned.whir.mu_pow_witness_ok, &Fr::from(1u64));

        assigned.statement_public_inputs.to_vec()
    };
    builder.assigned_instances[0].extend(public_input_cells);
}

fn build_end_to_end_constraints_from_intermediates(
    builder: &mut BaseCircuitBuilder<Fr>,
    intermediates: &PipelineIntermediates,
    statement: &PipelineStatementWitness,
    schedule: &PipelineTranscriptSchedule,
) {
    let range = builder.range_chip();
    let public_input_cells = {
        let ctx = builder.main(0);
        let assigned = constrain_pipeline_intermediates(ctx, &range, intermediates, statement, schedule);

        range
            .gate()
            .assert_is_const(ctx, &assigned.whir.mu_pow_witness_ok, &Fr::from(1u64));

        assigned.statement_public_inputs.to_vec()
    };
    builder.assigned_instances[0].extend(public_input_cells);
}

fn assert_fixture_matches_native<Fx>(engine: &BabyBearBn254Poseidon2CpuEngine, fixture: Fx)
where
    Fx: TestFixture<NativeConfig>,
{
    let (vk, proof) = fixture.keygen_and_prove(engine);
    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);

    run_mock(true, &public_inputs, |builder| {
        build_end_to_end_constraints_from_proof(builder, engine.config(), &vk, &proof);
    });
}

fn recompute_trace_height_sums(raw: &mut RawPipelineWitnessState, threshold_failure_context: &str) {
    for (constraint_idx, coeffs) in raw
        .intermediates
        .proof_shape
        .trace_height_coefficients
        .iter()
        .enumerate()
    {
        let sum = raw
            .intermediates
            .proof_shape
            .trace_id_to_air_id
            .iter()
            .map(|&trace_air| {
                let log_height = raw.intermediates.proof_shape.air_log_heights[trace_air];
                let pow = 1u64 << usize::max(log_height, raw.intermediates.proof_shape.l_skip);
                pow.saturating_mul(coeffs[trace_air])
            })
            .sum::<u64>();
        assert!(
            sum < raw.intermediates.proof_shape.trace_height_thresholds[constraint_idx],
            "{threshold_failure_context}",
        );
        raw.intermediates.proof_shape.trace_height_sums[constraint_idx] = sum;
    }
}

fn native_ext_to_coeffs(value: NativeEF) -> [u64; BABY_BEAR_EXT_DEGREE] {
    core::array::from_fn(|i| {
        <NativeEF as BasedVectorSpace<NativeF>>::as_basis_coefficients_slice(&value)[i]
            .as_canonical_u64()
    })
}

fn prank_recorded_ext_constant(
    ctx: &mut Context<Fr>,
    records: &[RecordedExtBaseConst],
    family: &str,
    constant: u64,
) {
    let record = records
        .iter()
        .find(|record| record.constant == constant)
        .unwrap_or_else(|| panic!("missing recorded ext-base constant for {family}={constant}"));
    record
        .cell
        .debug_prank(ctx, Fr::from((constant + 1) % BABY_BEAR_MODULUS_U64));
}

fn add_delta_to_whir_stacking_opening(
    openings: &mut [Vec<[u64; BABY_BEAR_EXT_DEGREE]>],
    flat_idx: usize,
    delta: NativeEF,
) {
    let mut cursor = flat_idx;
    for commit_openings in openings {
        if cursor < commit_openings.len() {
            let value = stacked_coeffs_to_native_ext(commit_openings[cursor]);
            commit_openings[cursor] = native_ext_to_coeffs(value + delta);
            return;
        }
        cursor = cursor.saturating_sub(commit_openings.len());
    }
    panic!("stacking-opening flat index out of bounds");
}

fn tamper_stacked_batch_openings_claim_preserving(raw: &mut RawPipelineWitnessState) {
    let lambda = stacked_coeffs_to_native_ext(raw.intermediates.batch_and_stacked.stacked_reduction.lambda);
    let lambda_sqr = lambda * lambda;

    let need_rot_schedule = raw
        .intermediates
        .batch_and_stacked
        .stacked_reduction
        .batch_column_openings_need_rot
        .clone();
    let openings = &mut raw.intermediates.batch_and_stacked.stacked_reduction.batch_column_openings;

    #[derive(Clone, Copy)]
    struct TermLoc {
        trace_idx: usize,
        part_idx: usize,
        claim_idx: usize,
        weight: NativeEF,
    }

    let mut term_locs = Vec::new();
    let mut lambda_pow = NativeEF::ONE;
    let mut push_part_terms = |trace_idx: usize, part_idx: usize, need_rot: bool| {
        let len = openings[trace_idx][part_idx].len();
        if need_rot {
            assert_eq!(
                len % 2,
                0,
                "rotated claim vectors must be in (claim, rot) pairs",
            );
            for claim_idx in (0..len).step_by(2) {
                term_locs.push(TermLoc {
                    trace_idx,
                    part_idx,
                    claim_idx,
                    weight: lambda_pow,
                });
                lambda_pow *= lambda_sqr;
            }
        } else {
            for claim_idx in 0..len {
                term_locs.push(TermLoc {
                    trace_idx,
                    part_idx,
                    claim_idx,
                    weight: lambda_pow,
                });
                lambda_pow *= lambda_sqr;
            }
        }
    };

    for trace_idx in 0..openings.len() {
        push_part_terms(trace_idx, 0, need_rot_schedule[0][trace_idx]);
    }

    let mut commit_idx = 1usize;
    for trace_idx in 0..openings.len() {
        for part_idx in 1..openings[trace_idx].len() {
            let need_rot = *need_rot_schedule[commit_idx]
                .first()
                .expect("non-common commit need-rotation metadata must be singleton");
            push_part_terms(trace_idx, part_idx, need_rot);
            commit_idx += 1;
        }
    }
    assert_eq!(
        commit_idx,
        need_rot_schedule.len(),
        "all stacked commit-rotation rows must be consumed when deriving term schedule",
    );

    assert!(
        !term_locs.is_empty(),
        "stacked batch opening terms must be non-empty for tamper test",
    );

    let delta = NativeEF::ONE;
    if lambda == NativeEF::ZERO {
        let target = term_locs
            .iter()
            .copied()
            .find(|loc| loc.weight == NativeEF::ZERO)
            .unwrap_or(term_locs[0]);
        let value = stacked_coeffs_to_native_ext(
            openings[target.trace_idx][target.part_idx][target.claim_idx],
        );
        openings[target.trace_idx][target.part_idx][target.claim_idx] =
            native_ext_to_coeffs(value + delta);
        return;
    }

    assert!(
        term_locs.len() >= 2,
        "stacked batch opening schedule must expose at least two terms for cancellation tamper",
    );
    let first = term_locs[0];
    let second = term_locs[1];
    let cancel_delta = NativeEF::ZERO - (delta * first.weight * second.weight.inverse());

    let first_value =
        stacked_coeffs_to_native_ext(openings[first.trace_idx][first.part_idx][first.claim_idx]);
    openings[first.trace_idx][first.part_idx][first.claim_idx] =
        native_ext_to_coeffs(first_value + delta);

    let second_value =
        stacked_coeffs_to_native_ext(openings[second.trace_idx][second.part_idx][second.claim_idx]);
    openings[second.trace_idx][second.part_idx][second.claim_idx] =
        native_ext_to_coeffs(second_value + cancel_delta);
}

fn tamper_whir_stacking_openings_claim_preserving(raw: &mut RawPipelineWitnessState) {
    let mu = stacked_coeffs_to_native_ext(raw.intermediates.whir.mu_challenge);
    let openings = &mut raw.intermediates.whir.stacking_openings;
    let total_openings = openings.iter().map(Vec::len).sum::<usize>();
    assert!(
        total_openings >= 2,
        "WHIR stacking openings must include at least two values for decoupling tamper",
    );

    let delta = NativeEF::ONE;
    if mu == NativeEF::ZERO {
        add_delta_to_whir_stacking_opening(openings, 1, delta);
        return;
    }

    let cancel_delta = NativeEF::ZERO - (delta * mu.inverse());
    add_delta_to_whir_stacking_opening(openings, 0, delta);
    add_delta_to_whir_stacking_opening(openings, 1, cancel_delta);
}

fn tamper_stacked_claim_chain_payload_preserving_residual(raw: &mut RawPipelineWitnessState) {
    let stacked = &mut raw.intermediates.batch_and_stacked.stacked_reduction;
    assert!(
        !stacked.sumcheck_round_polys.is_empty(),
        "fixture must include stacked sumcheck rounds",
    );
    assert_eq!(
        stacked.sumcheck_round_polys[0].len(),
        2,
        "stacked sumcheck rounds must expose [s(1), s(2)]",
    );

    stacked.sumcheck_round_polys[0][0][0] =
        (stacked.sumcheck_round_polys[0][0][0] + 1) % BABY_BEAR_MODULUS_U64;
    stacked.final_claim = stacked.final_sum;
    stacked.final_residual = [0; BABY_BEAR_EXT_DEGREE];
}

#[test]
fn pipeline_end_to_end_matches_native_for_fib_fixture() {
    let engine = test_engine();
    assert_fixture_matches_native(&engine, FibFixture::new(0, 1, 1 << 5));
}

#[test]
fn pipeline_end_to_end_matches_native_for_interactions_fixture() {
    let engine = test_engine();
    assert_fixture_matches_native(&engine, InteractionsFixture11);
}

#[test]
fn pipeline_end_to_end_matches_native_for_cached_fixture() {
    let engine = test_engine();
    assert_fixture_matches_native(&engine, CachedFixture11::new(engine.config().clone()));
}

#[test]
fn pipeline_end_to_end_matches_native_for_preprocessed_fixture() {
    let engine = test_engine();
    let height = 1 << 5;
    let sels = (0..height).map(|i| i % 2 == 0).collect::<Vec<_>>();
    assert_fixture_matches_native(&engine, PreprocessedFibFixture::new(0, 1, sels));
}

#[test]
fn pipeline_constraints_fail_when_whir_claim_is_tampered() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");
    raw.intermediates.whir.final_claim[0] =
        (raw.intermediates.whir.final_claim[0] + 1) % BABY_BEAR_MODULUS_U64;
    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);

    run_mock(false, &public_inputs, |builder| {
        build_end_to_end_constraints_from_intermediates(
            builder,
            &raw.intermediates,
            &raw.statement,
            &raw.schedule,
        );
    });
}

#[test]
fn pipeline_constraints_fail_when_batch_challenge_is_tampered() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");
    raw.intermediates.batch_and_stacked.batch.r[0][0] =
        (raw.intermediates.batch_and_stacked.batch.r[0][0] + 1) % BABY_BEAR_MODULUS_U64;
    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);

    run_mock(false, &public_inputs, |builder| {
        build_end_to_end_constraints_from_intermediates(
            builder,
            &raw.intermediates,
            &raw.statement,
            &raw.schedule,
        );
    });
}

#[test]
fn pipeline_constraints_fail_when_ext_constant_families_are_pranked() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");

    let subgroup_root = NativeF::two_adic_generator(raw.schedule.l_skip).as_canonical_u64();
    let bus_constant = raw
        .schedule
        .batch_trace_interactions
        .iter()
        .flat_map(|interactions| interactions.iter())
        .map(|interaction| u64::from(interaction.bus_index) + 1)
        .find(|&value| value > 1)
        .unwrap_or(1);
    let normalization_family_constants = (1..=31usize)
        .map(|pow| {
            (0..pow)
                .fold(NativeF::ONE, |acc, _| acc.halve())
                .as_canonical_u64()
        })
        .collect::<Vec<_>>();
    let base_families = [
        ("one", 1u64),
        ("two", 2u64),
        ("subgroup_root", subgroup_root),
        ("bus_index_plus_one", bus_constant),
    ];

    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);
    run_mock(false, &public_inputs, move |builder| {
        let range = builder.range_chip();
        let public_input_cells = {
            let ctx = builder.main(0);
            clear_recorded_ext_base_consts();
            let assigned = derive_and_constrain_pipeline(ctx, &range, engine.config(), &vk, &proof)
                .expect("pipeline derive+constrain should succeed before ext-constant prank");
            let records = take_recorded_ext_base_consts();
            for (family, constant) in base_families {
                prank_recorded_ext_constant(ctx, &records, family, constant);
            }
            let normalization_constant = records
                .iter()
                .find(|record| normalization_family_constants.contains(&record.constant))
                .map(|record| record.constant)
                .unwrap_or(1);
            prank_recorded_ext_constant(ctx, &records, "normalization", normalization_constant);
            assigned.statement_public_inputs.to_vec()
        };
        builder.assigned_instances[0].extend(public_input_cells);
    });
}

#[test]
fn pipeline_constraints_fail_when_stacked_r_is_decoupled_from_batch_r() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");
    raw.intermediates.batch_and_stacked.stacked_reduction.r[0][0] =
        (raw.intermediates.batch_and_stacked.stacked_reduction.r[0][0] + 1) % BABY_BEAR_MODULUS_U64;
    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);

    run_mock(false, &public_inputs, |builder| {
        build_end_to_end_constraints_from_intermediates(
            builder,
            &raw.intermediates,
            &raw.statement,
            &raw.schedule,
        );
    });
}

#[test]
fn pipeline_constraints_fail_when_stacked_batch_openings_are_decoupled_from_batch_openings() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");
    tamper_stacked_batch_openings_claim_preserving(&mut raw);
    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);

    run_mock(false, &public_inputs, |builder| {
        build_end_to_end_constraints_from_intermediates(
            builder,
            &raw.intermediates,
            &raw.statement,
            &raw.schedule,
        );
    });
}

#[test]
fn pipeline_constraints_fail_when_batch_opening_family_width_is_padded() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");
    assert!(
        !raw.intermediates.batch_and_stacked.batch.column_openings.is_empty()
            && !raw.intermediates.batch_and_stacked.batch.column_openings[0].is_empty(),
        "fixture must include at least one batch opening family",
    );

    raw.intermediates.batch_and_stacked.batch.column_openings[0][0].push([0; BABY_BEAR_EXT_DEGREE]);
    raw.intermediates.batch_and_stacked.batch.column_openings[0][0].push([0; BABY_BEAR_EXT_DEGREE]);
    raw.intermediates.batch_and_stacked.stacked_reduction.batch_column_openings[0][0]
        .push([0; BABY_BEAR_EXT_DEGREE]);
    raw.intermediates.batch_and_stacked.stacked_reduction.batch_column_openings[0][0]
        .push([0; BABY_BEAR_EXT_DEGREE]);

    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);
    assert_rejected_without_host_panic(|| {
        run_mock(false, &public_inputs, |builder| {
            build_end_to_end_constraints_from_intermediates(
                builder,
                &raw.intermediates,
                &raw.statement,
                &raw.schedule,
            );
        });
    });
}

#[test]
fn pipeline_constraints_fail_when_stacked_opening_family_width_is_padded() {
    let engine = test_engine();
    let (vk, proof) = CachedFixture11::new(engine.config().clone()).keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");
    assert!(
        !raw.intermediates.batch_and_stacked.stacked_reduction.q_coeffs.is_empty()
            && !raw
                .intermediates
                .batch_and_stacked
                .stacked_reduction
                .stacking_openings
                .is_empty(),
        "fixture must include stacked opening families",
    );

    raw.intermediates.batch_and_stacked.stacked_reduction.q_coeffs[0].push([0; BABY_BEAR_EXT_DEGREE]);
    raw.intermediates.batch_and_stacked.stacked_reduction.stacking_openings[0].push([0; BABY_BEAR_EXT_DEGREE]);
    raw.intermediates.whir.stacking_openings[0].push([0; BABY_BEAR_EXT_DEGREE]);

    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);
    assert_rejected_without_host_panic(|| {
        run_mock(false, &public_inputs, |builder| {
            build_end_to_end_constraints_from_intermediates(
                builder,
                &raw.intermediates,
                &raw.statement,
                &raw.schedule,
            );
        });
    });
}

#[test]
fn pipeline_constraints_fail_when_whir_stacking_openings_are_decoupled_from_stacked_openings() {
    let engine = test_engine();
    let (vk, proof) = CachedFixture11::new(engine.config().clone()).keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");
    tamper_whir_stacking_openings_claim_preserving(&mut raw);
    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);

    run_mock(false, &public_inputs, |builder| {
        build_end_to_end_constraints_from_intermediates(
            builder,
            &raw.intermediates,
            &raw.statement,
            &raw.schedule,
        );
    });
}

#[test]
fn pipeline_constraints_fail_when_batch_ref_to_stacked_coupling_has_trailing_suffix() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");
    assert!(
        !raw.intermediates
            .batch_and_stacked
            .stacked_reduction
            .batch_column_openings
            .is_empty()
            && !raw.intermediates.batch_and_stacked.stacked_reduction.batch_column_openings[0].is_empty(),
        "fixture must include stacked batch-opening families",
    );
    raw.intermediates.batch_and_stacked.stacked_reduction.batch_column_openings[0][0]
        .push([0; BABY_BEAR_EXT_DEGREE]);
    raw.intermediates.batch_and_stacked.stacked_reduction.batch_column_openings[0][0]
        .push([0; BABY_BEAR_EXT_DEGREE]);

    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);
    assert_rejected_without_host_panic(|| {
        run_mock(false, &public_inputs, |builder| {
            build_end_to_end_constraints_from_intermediates(
                builder,
                &raw.intermediates,
                &raw.statement,
                &raw.schedule,
            );
        });
    });
}

#[test]
fn pipeline_constraints_fail_when_stacked_to_whir_coupling_has_trailing_suffix() {
    let engine = test_engine();
    let (vk, proof) = CachedFixture11::new(engine.config().clone()).keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");
    assert!(
        !raw.intermediates.whir.stacking_openings.is_empty(),
        "fixture must include WHIR stacking-opening families",
    );
    raw.intermediates.whir.stacking_openings[0].push([0; BABY_BEAR_EXT_DEGREE]);

    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);
    assert_rejected_without_host_panic(|| {
        run_mock(false, &public_inputs, |builder| {
            build_end_to_end_constraints_from_intermediates(
                builder,
                &raw.intermediates,
                &raw.statement,
                &raw.schedule,
            );
        });
    });
}

#[test]
fn pipeline_constraints_fail_on_coordinated_stacked_claim_chain_forgery() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");
    tamper_stacked_claim_chain_payload_preserving_residual(&mut raw);
    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);

    run_mock(false, &public_inputs, |builder| {
        build_end_to_end_constraints_from_intermediates(
            builder,
            &raw.intermediates,
            &raw.statement,
            &raw.schedule,
        );
    });
}

#[test]
fn pipeline_constraints_fail_when_proof_shape_log_height_is_forked_from_preamble() {
    let engine = test_engine();
    let (vk, proof) = FibFixture::new(0, 1, 1 << 5).keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");

    let air_idx = raw
        .schedule
        .air_has_preprocessed
        .iter()
        .enumerate()
        .find_map(|(idx, &has_preprocessed)| {
            (!has_preprocessed && raw.intermediates.proof_shape.air_presence_flags[idx]).then_some(idx)
        })
        .expect("fixture must include a present non-preprocessed AIR");
    let old_log_height = raw.intermediates.proof_shape.air_log_heights[air_idx];
    let mut new_log_height = old_log_height.saturating_add(1);
    if new_log_height > raw.intermediates.proof_shape.max_log_height_allowed {
        new_log_height = old_log_height.saturating_sub(1);
    }
    assert_ne!(
        new_log_height, old_log_height,
        "expected to find an alternate log-height value for tamper test",
    );
    raw.intermediates.proof_shape.air_log_heights[air_idx] = new_log_height;

    recompute_trace_height_sums(
        &mut raw,
        "tampered log-height must remain below threshold for preamble-ownership test",
    );

    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);
    run_mock(false, &public_inputs, |builder| {
        build_end_to_end_constraints_from_intermediates(
            builder,
            &raw.intermediates,
            &raw.statement,
            &raw.schedule,
        );
    });
}

#[test]
fn pipeline_constraints_fail_on_coordinated_preamble_stage_log_height_tamper() {
    let engine = test_engine();
    let (vk, proof) = FibFixture::new(0, 1, 1 << 5).keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");

    let air_idx = raw
        .schedule
        .air_has_preprocessed
        .iter()
        .enumerate()
        .find_map(|(idx, &has_preprocessed)| {
            (!has_preprocessed && raw.intermediates.proof_shape.air_presence_flags[idx]).then_some(idx)
        })
        .expect("fixture must include a present non-preprocessed AIR");
    let old_log_height = raw.intermediates.proof_shape.air_log_heights[air_idx];
    let mut new_log_height = old_log_height.saturating_add(1);
    if new_log_height > raw.intermediates.proof_shape.max_log_height_allowed {
        new_log_height = old_log_height.saturating_sub(1);
    }
    assert_ne!(
        new_log_height, old_log_height,
        "expected to find an alternate log-height value for tamper test",
    );
    raw.intermediates.proof_shape.air_log_heights[air_idx] = new_log_height;

    recompute_trace_height_sums(
        &mut raw,
        "tampered log-height must remain below threshold for preamble-ownership test",
    );

    let mut preamble_idx = 6usize;
    for prior_air in 0..air_idx {
        if !raw.schedule.air_is_required[prior_air] {
            preamble_idx += 1;
        }
        if raw.intermediates.proof_shape.air_presence_flags[prior_air] {
            preamble_idx += if raw.schedule.air_has_preprocessed[prior_air] {
                3
            } else {
                1
            };
            preamble_idx += 3 * raw.intermediates.proof_shape.air_cached_commitment_lens[prior_air];
        }
        preamble_idx += raw.intermediates.batch_and_stacked.batch.public_values[prior_air].len();
    }
    if !raw.schedule.air_is_required[air_idx] {
        preamble_idx += 1;
    }
    match &mut raw.intermediates.transcript_events[preamble_idx] {
        crate::gadgets::transcript::TranscriptEvent::Observe(value) => {
            *value = new_log_height as u64;
        }
        crate::gadgets::transcript::TranscriptEvent::Sample(_) => {
            panic!("expected preamble transcript event to be an observe");
        }
    }

    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);
    run_mock(false, &public_inputs, |builder| {
        build_end_to_end_constraints_from_intermediates(
            builder,
            &raw.intermediates,
            &raw.statement,
            &raw.schedule,
        );
    });
}

#[test]
fn pipeline_constraints_fail_when_transcript_sample_is_tampered() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");
    let sample_idx = raw
        .intermediates
        .transcript_events
        .iter()
        .position(|event| {
            matches!(
                event,
                crate::gadgets::transcript::TranscriptEvent::Sample(_)
            )
        })
        .expect("pipeline transcript event log should contain sampled challenges");
    if let crate::gadgets::transcript::TranscriptEvent::Sample(value) =
        &mut raw.intermediates.transcript_events[sample_idx]
    {
        *value = (*value + 1) % BABY_BEAR_MODULUS_U64;
    }
    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);

    run_mock(false, &public_inputs, |builder| {
        build_end_to_end_constraints_from_intermediates(
            builder,
            &raw.intermediates,
            &raw.statement,
            &raw.schedule,
        );
    });
}

#[test]
fn pipeline_constraints_fail_when_transcript_event_order_is_tampered() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");

    let sample_idx = raw
        .intermediates
        .transcript_events
        .iter()
        .position(|event| {
            matches!(
                event,
                crate::gadgets::transcript::TranscriptEvent::Sample(_)
            )
        })
        .expect("pipeline transcript event log should contain sampled challenges");
    assert!(
        sample_idx > 0,
        "sample event should not be first in the schedule"
    );
    raw.intermediates
        .transcript_events
        .swap(sample_idx - 1, sample_idx);

    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);
    run_mock(false, &public_inputs, |builder| {
        build_end_to_end_constraints_from_intermediates(
            builder,
            &raw.intermediates,
            &raw.statement,
            &raw.schedule,
        );
    });
}

#[test]
fn pipeline_constraints_fail_when_post_preamble_interleaving_is_reordered() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");

    let preamble_len = raw.schedule.raw_preamble_observe_count;
    let reorder_idx = (preamble_len..raw.intermediates.transcript_events.len().saturating_sub(1))
        .find(|&idx| {
            matches!(
                (
                    &raw.intermediates.transcript_events[idx],
                    &raw.intermediates.transcript_events[idx + 1]
                ),
                (
                    crate::gadgets::transcript::TranscriptEvent::Observe(_),
                    crate::gadgets::transcript::TranscriptEvent::Sample(_)
                ) | (
                    crate::gadgets::transcript::TranscriptEvent::Sample(_),
                    crate::gadgets::transcript::TranscriptEvent::Observe(_)
                )
            )
        })
        .expect("expected a post-preamble observe/sample boundary to reorder");
    raw.intermediates
        .transcript_events
        .swap(reorder_idx, reorder_idx + 1);

    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);
    run_mock(false, &public_inputs, |builder| {
        build_end_to_end_constraints_from_intermediates(
            builder,
            &raw.intermediates,
            &raw.statement,
            &raw.schedule,
        );
    });
}

#[test]
fn pipeline_constraints_fail_when_transcript_has_unconsumed_sample_event() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");
    raw.intermediates
        .transcript_events
        .push(crate::gadgets::transcript::TranscriptEvent::Sample(0));

    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);
    run_mock(false, &public_inputs, |builder| {
        build_end_to_end_constraints_from_intermediates(
            builder,
            &raw.intermediates,
            &raw.statement,
            &raw.schedule,
        );
    });
}

#[test]
fn pipeline_constraints_fail_when_schedule_metadata_is_tampered() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");

    raw.intermediates.batch_and_stacked.batch.logup_pow_bits =
        raw.intermediates.batch_and_stacked.batch.logup_pow_bits.saturating_add(1);

    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);
    run_mock(false, &public_inputs, |builder| {
        build_end_to_end_constraints_from_intermediates(
            builder,
            &raw.intermediates,
            &raw.statement,
            &raw.schedule,
        );
    });
}

#[test]
fn pipeline_constraints_fail_when_schedule_is_tampered() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");
    raw.schedule.logup_pow_bits = raw.schedule.logup_pow_bits.saturating_add(1);

    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);
    run_mock(false, &public_inputs, |builder| {
        build_end_to_end_constraints_from_intermediates(
            builder,
            &raw.intermediates,
            &raw.statement,
            &raw.schedule,
        );
    });
}

#[test]
fn pipeline_public_inputs_expose_only_statement_digests() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);
    assert_eq!(
        public_inputs.len(),
        2,
        "statement should bind only MVK pre-hash and common-main commitment",
    );
}

#[test]
fn pipeline_constraints_fail_when_batch_total_interactions_schedule_is_tampered() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");
    assert!(
        raw.schedule.batch_total_interactions > 0,
        "fixture should include non-zero interaction count for GKR scheduling",
    );
    raw.schedule.batch_total_interactions = raw.schedule.batch_total_interactions.saturating_sub(1);

    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);
    run_mock(false, &public_inputs, |builder| {
        build_end_to_end_constraints_from_intermediates(
            builder,
            &raw.intermediates,
            &raw.statement,
            &raw.schedule,
        );
    });
}

#[test]
fn pipeline_constraints_fail_when_proof_shape_required_flag_schedule_width_is_tampered() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");
    assert!(
        !raw.schedule.air_is_required.is_empty(),
        "fixture must include AIR requirement schedule entries",
    );
    raw.schedule.air_is_required.pop();

    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);
    assert_rejected_without_host_panic(|| {
        run_mock(false, &public_inputs, |builder| {
            build_end_to_end_constraints_from_intermediates(
                builder,
                &raw.intermediates,
                &raw.statement,
                &raw.schedule,
            );
        });
    });
}

#[test]
fn pipeline_constraints_fail_when_whir_challenge_schedule_cardinality_is_tampered() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");
    raw.schedule.query_counts_per_round.push(1);

    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);
    assert_rejected_without_host_panic(|| {
        run_mock(false, &public_inputs, |builder| {
            build_end_to_end_constraints_from_intermediates(
                builder,
                &raw.intermediates,
                &raw.statement,
                &raw.schedule,
            );
        });
    });
}

#[test]
fn pipeline_constraints_fail_when_batch_term_cardinality_is_tampered() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");
    assert!(
        !raw.intermediates
            .batch_and_stacked
            .batch
            .denominator_term_per_air
            .is_empty(),
        "fixture must include denominator terms",
    );
    raw.intermediates.batch_and_stacked.batch.denominator_term_per_air.pop();

    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);
    assert_rejected_without_host_panic(|| {
        run_mock(false, &public_inputs, |builder| {
            build_end_to_end_constraints_from_intermediates(
                builder,
                &raw.intermediates,
                &raw.statement,
                &raw.schedule,
            );
        });
    });
}

#[test]
fn pipeline_constraints_fail_when_batch_sumcheck_arity_is_tampered() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");
    raw.intermediates.batch_and_stacked.batch.sumcheck_round_polys.push(vec![
        [0; BABY_BEAR_EXT_DEGREE];
        raw.intermediates
            .batch_and_stacked
            .batch
            .batch_degree
    ]);

    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);
    assert_rejected_without_host_panic(|| {
        run_mock(false, &public_inputs, |builder| {
            build_end_to_end_constraints_from_intermediates(
                builder,
                &raw.intermediates,
                &raw.statement,
                &raw.schedule,
            );
        });
    });
}

#[test]
fn pipeline_constraints_fail_when_batch_univariate_arity_is_tampered() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");
    raw.intermediates
        .batch_and_stacked
        .batch
        .univariate_round_coeffs
        .push([0; BABY_BEAR_EXT_DEGREE]);

    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);
    assert_rejected_without_host_panic(|| {
        run_mock(false, &public_inputs, |builder| {
            build_end_to_end_constraints_from_intermediates(
                builder,
                &raw.intermediates,
                &raw.statement,
                &raw.schedule,
            );
        });
    });
}

#[test]
fn pipeline_constraints_fail_when_batch_ref_metadata_is_tampered() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");

    let trace_idx = raw
        .intermediates
        .batch_and_stacked
        .batch
        .n_per_trace
        .iter()
        .position(|&n| n > 0)
        .expect("fixture must include at least one positive lift trace");
    raw.intermediates.batch_and_stacked.batch.n_per_trace[trace_idx] -= 1;

    let selector_trace_idx = raw
        .intermediates
        .batch_and_stacked
        .batch
        .trace_interactions
        .iter()
        .position(|interactions| !interactions.is_empty())
        .expect("fixture must include interaction-selector metadata");
    raw.intermediates.batch_and_stacked.batch.trace_interactions[selector_trace_idx][0].bus_index =
        raw.intermediates.batch_and_stacked.batch.trace_interactions[selector_trace_idx][0]
            .bus_index
            .wrapping_add(1);

    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);
    run_mock(false, &public_inputs, |builder| {
        build_end_to_end_constraints_from_intermediates(
            builder,
            &raw.intermediates,
            &raw.statement,
            &raw.schedule,
        );
    });
}

#[test]
fn pipeline_constraints_fail_when_metadata_row_has_trailing_suffix_width() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");
    assert!(
        !raw.intermediates
            .batch_and_stacked
            .batch
            .column_opening_expected_widths
            .is_empty(),
        "fixture must include column-opening width metadata",
    );
    raw.intermediates.batch_and_stacked.batch.column_opening_expected_widths[0].push(0);

    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);
    assert_rejected_without_host_panic(|| {
        run_mock(false, &public_inputs, |builder| {
            build_end_to_end_constraints_from_intermediates(
                builder,
                &raw.intermediates,
                &raw.statement,
                &raw.schedule,
            );
        });
    });
}

#[test]
fn pipeline_constraints_fail_when_proof_shape_trace_height_coefficients_are_tampered() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");

    let (constraint_idx, coeff_idx) = raw
        .intermediates
        .proof_shape
        .trace_height_coefficients
        .iter()
        .enumerate()
        .find_map(|(constraint_idx, coeffs)| {
            coeffs
                .iter()
                .position(|&coeff| coeff > 0)
                .map(|coeff_idx| (constraint_idx, coeff_idx))
        })
        .expect("fixture must include at least one non-zero trace-height coefficient");
    raw.intermediates.proof_shape.trace_height_coefficients[constraint_idx][coeff_idx] -= 1;
    recompute_trace_height_sums(
        &mut raw,
        "coefficient tamper should remain below threshold for ownership test",
    );

    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);
    run_mock(false, &public_inputs, |builder| {
        build_end_to_end_constraints_from_intermediates(
            builder,
            &raw.intermediates,
            &raw.statement,
            &raw.schedule,
        );
    });
}

#[test]
fn pipeline_constraints_fail_when_symbolic_node_table_ownership_is_tampered() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");

    let trace_idx = raw
        .intermediates
        .batch_and_stacked
        .batch
        .trace_constraint_nodes
        .iter()
        .position(|nodes| !nodes.is_empty())
        .expect("fixture must include symbolic node metadata");
    let node = &mut raw.intermediates.batch_and_stacked.batch.trace_constraint_nodes[trace_idx][0];
    *node = match node {
        SymbolicExpressionNode::IsFirstRow => SymbolicExpressionNode::IsLastRow,
        _ => SymbolicExpressionNode::IsFirstRow,
    };

    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);
    run_mock(false, &public_inputs, |builder| {
        build_end_to_end_constraints_from_intermediates(
            builder,
            &raw.intermediates,
            &raw.statement,
            &raw.schedule,
        );
    });
}

#[test]
fn pipeline_constraints_ignore_proof_shape_proof_shape_checklist_mirror_only_tamper() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");

    assert!(
        !raw.intermediates.proof_shape.proof_shape_count_checks.is_empty(),
        "fixture must include proof-shape count checks",
    );
    let (actual_count, expected_count) = raw.intermediates.proof_shape.proof_shape_count_checks[0];
    raw.intermediates.proof_shape.proof_shape_count_checks[0] = (
        actual_count.saturating_add(1),
        expected_count.saturating_add(1),
    );

    assert!(
        !raw.intermediates
            .proof_shape
            .proof_shape_upper_bound_checks
            .is_empty(),
        "fixture must include proof-shape upper-bound checks",
    );
    let (actual_value, expected_max) = raw.intermediates.proof_shape.proof_shape_upper_bound_checks[0];
    let tampered_expected_max = expected_max
        .saturating_add(7)
        .max(actual_value.saturating_add(1));
    raw.intermediates.proof_shape.proof_shape_upper_bound_checks[0] = (actual_value, tampered_expected_max);

    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);
    run_mock(true, &public_inputs, |builder| {
        build_end_to_end_constraints_from_intermediates(
            builder,
            &raw.intermediates,
            &raw.statement,
            &raw.schedule,
        );
    });
}

#[test]
fn pipeline_constraints_fail_on_coordinated_stage_shape_and_checklist_tamper() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");

    assert!(
        !raw.intermediates.proof_shape.proof_shape_count_checks.is_empty(),
        "fixture must include proof-shape count checks",
    );
    let (actual_count, expected_count) = raw.intermediates.proof_shape.proof_shape_count_checks[0];
    raw.intermediates.proof_shape.proof_shape_count_checks[0] = (
        actual_count.saturating_add(1),
        expected_count.saturating_add(1),
    );

    let first_merkle_path = raw
        .intermediates
        .whir
        .merkle_paths
        .first_mut()
        .expect("fixture must include at least one WHIR Merkle path");
    first_merkle_path.siblings.push(Fr::from(0u64));

    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);
    run_mock(false, &public_inputs, |builder| {
        build_end_to_end_constraints_from_intermediates(
            builder,
            &raw.intermediates,
            &raw.statement,
            &raw.schedule,
        );
    });
}

#[test]
fn pipeline_constraints_fail_when_stacked_q_term_schedule_is_tampered() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");
    let first_term = raw
        .intermediates
        .batch_and_stacked
        .stacked_reduction
        .q_coeff_terms
        .first_mut()
        .expect("fixture must include stacked q-coefficient terms");
    first_term.need_rot = !first_term.need_rot;

    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);
    run_mock(false, &public_inputs, |builder| {
        build_end_to_end_constraints_from_intermediates(
            builder,
            &raw.intermediates,
            &raw.statement,
            &raw.schedule,
        );
    });
}

#[test]
fn pipeline_constraints_fail_when_whir_fold_domain_metadata_is_tampered() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");
    assert!(
        raw.intermediates.whir.k_whir > 1,
        "fixture must have k_whir > 1 for safe metadata decrement tamper",
    );
    raw.intermediates.whir.k_whir -= 1;
    raw.intermediates.whir.initial_log_rs_domain_size -= 1;

    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_mock(true, &public_inputs, |builder| {
            build_end_to_end_constraints_from_intermediates(
                builder,
                &raw.intermediates,
                &raw.statement,
                &raw.schedule,
            );
        });
    }));
    assert!(
        result.is_err(),
        "tampered WHIR fold/domain metadata should not be accepted",
    );
}

#[test]
fn pipeline_constraints_fail_when_proof_shape_shape_expectation_is_tampered() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");
    raw.intermediates.proof_shape.air_expected_public_value_lens[0] =
        raw.intermediates.proof_shape.air_expected_public_value_lens[0].saturating_add(1);

    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);
    run_mock(false, &public_inputs, |builder| {
        build_end_to_end_constraints_from_intermediates(
            builder,
            &raw.intermediates,
            &raw.statement,
            &raw.schedule,
        );
    });
}

#[test]
fn pipeline_constraints_fail_when_gkr_non_xi_challenge_is_tampered() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");
    assert!(
        !raw.intermediates.batch_and_stacked.batch.gkr_non_xi_samples.is_empty(),
        "fixture should produce non-xi GKR challenges",
    );
    raw.intermediates.batch_and_stacked.batch.gkr_non_xi_samples[0][0] =
        (raw.intermediates.batch_and_stacked.batch.gkr_non_xi_samples[0][0] + 1) % BABY_BEAR_MODULUS_U64;

    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);
    run_mock(false, &public_inputs, |builder| {
        build_end_to_end_constraints_from_intermediates(
            builder,
            &raw.intermediates,
            &raw.statement,
            &raw.schedule,
        );
    });
}

#[test]
fn pipeline_constraints_fail_when_non_preamble_observe_is_tampered() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");
    let preamble_len = raw.schedule.raw_preamble_observe_count;
    let observe_idx = raw
        .intermediates
        .transcript_events
        .iter()
        .enumerate()
        .skip(preamble_len)
        .find_map(|(idx, event)| match event {
            crate::gadgets::transcript::TranscriptEvent::Observe(_) => Some(idx),
            crate::gadgets::transcript::TranscriptEvent::Sample(_) => None,
        })
        .expect("expected a non-preamble observe event");
    if let crate::gadgets::transcript::TranscriptEvent::Observe(value) =
        &mut raw.intermediates.transcript_events[observe_idx]
    {
        *value = (*value + 1) % BABY_BEAR_MODULUS_U64;
    }

    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);
    run_mock(false, &public_inputs, |builder| {
        build_end_to_end_constraints_from_intermediates(
            builder,
            &raw.intermediates,
            &raw.statement,
            &raw.schedule,
        );
    });
}

#[test]
fn pipeline_constraints_fail_when_preamble_root_mapping_has_unconsumed_suffix() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");
    raw.intermediates
        .whir
        .initial_commitment_roots
        .push(Fr::from(0u64));

    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);
    assert_rejected_without_host_panic(|| {
        run_mock(false, &public_inputs, |builder| {
            build_end_to_end_constraints_from_intermediates(
                builder,
                &raw.intermediates,
                &raw.statement,
                &raw.schedule,
            );
        });
    });
}

#[test]
fn pipeline_constraints_fail_when_stage_payload_observe_stream_is_inconsistent() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");
    assert!(
        raw.intermediates.batch_and_stacked.batch.total_interactions > 0,
        "fixture should include non-empty GKR observe payloads",
    );
    let gkr_q0_claim = raw
        .intermediates
        .batch_and_stacked
        .batch
        .gkr_q0_claim
        .as_mut()
        .expect("non-zero interaction fixture should assign q0 claim witness");
    gkr_q0_claim[0] = (gkr_q0_claim[0] + 1) % BABY_BEAR_MODULUS_U64;

    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);
    run_mock(false, &public_inputs, |builder| {
        build_end_to_end_constraints_from_intermediates(
            builder,
            &raw.intermediates,
            &raw.statement,
            &raw.schedule,
        );
    });
}

#[test]
fn pipeline_constraints_fail_when_logup_pow_mirror_witness_is_tampered() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");
    raw.intermediates.batch_and_stacked.batch.logup_pow_witness =
        (raw.intermediates.batch_and_stacked.batch.logup_pow_witness + 1) % BABY_BEAR_MODULUS_U64;

    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);
    run_mock(false, &public_inputs, |builder| {
        build_end_to_end_constraints_from_intermediates(
            builder,
            &raw.intermediates,
            &raw.statement,
            &raw.schedule,
        );
    });
}

#[test]
fn pipeline_constraints_fail_when_mu_pow_mirror_witness_is_tampered() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");
    raw.intermediates.whir.mu_pow_witness =
        (raw.intermediates.whir.mu_pow_witness + 1) % BABY_BEAR_MODULUS_U64;

    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);
    run_mock(false, &public_inputs, |builder| {
        build_end_to_end_constraints_from_intermediates(
            builder,
            &raw.intermediates,
            &raw.statement,
            &raw.schedule,
        );
    });
}

#[test]
fn pipeline_constraints_fail_when_folding_pow_mirror_witness_is_tampered() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");
    assert!(
        !raw.intermediates.whir.folding_pow_witnesses.is_empty(),
        "fixture should include folding PoW witnesses",
    );
    raw.intermediates.whir.folding_pow_witnesses[0] =
        (raw.intermediates.whir.folding_pow_witnesses[0] + 1) % BABY_BEAR_MODULUS_U64;

    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);
    run_mock(false, &public_inputs, |builder| {
        build_end_to_end_constraints_from_intermediates(
            builder,
            &raw.intermediates,
            &raw.statement,
            &raw.schedule,
        );
    });
}

#[test]
fn pipeline_constraints_fail_when_query_phase_pow_mirror_witness_is_tampered() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");
    assert!(
        !raw.intermediates.whir.query_phase_pow_witnesses.is_empty(),
        "fixture should include query-phase PoW witnesses",
    );
    raw.intermediates.whir.query_phase_pow_witnesses[0] =
        (raw.intermediates.whir.query_phase_pow_witnesses[0] + 1) % BABY_BEAR_MODULUS_U64;

    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);
    run_mock(false, &public_inputs, |builder| {
        build_end_to_end_constraints_from_intermediates(
            builder,
            &raw.intermediates,
            &raw.statement,
            &raw.schedule,
        );
    });
}

#[test]
fn pipeline_constraints_fail_when_residual_only_mirrors_are_forged() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");

    raw.intermediates.whir.final_claim[0] =
        (raw.intermediates.whir.final_claim[0] + 1) % BABY_BEAR_MODULUS_U64;
    raw.intermediates.whir.final_acc[0] =
        (raw.intermediates.whir.final_acc[0] + 1) % BABY_BEAR_MODULUS_U64;
    raw.intermediates.whir.final_residual = [0; BABY_BEAR_EXT_DEGREE];

    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);
    run_mock(false, &public_inputs, |builder| {
        build_end_to_end_constraints_from_intermediates(
            builder,
            &raw.intermediates,
            &raw.statement,
            &raw.schedule,
        );
    });
}

#[test]
fn pipeline_constraints_fail_when_final_poly_expected_len_is_tampered() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut raw = derive_raw_pipeline_witness_state(engine.config(), &vk, &proof)
        .expect("native pipeline witness derivation must pass");
    raw.intermediates.whir.expected_final_poly_len = raw
        .intermediates
        .whir
        .expected_final_poly_len
        .saturating_add(1);

    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);
    run_mock(false, &public_inputs, |builder| {
        build_end_to_end_constraints_from_intermediates(
            builder,
            &raw.intermediates,
            &raw.statement,
            &raw.schedule,
        );
    });
}

#[test]
fn bind_sample_bits_consumes_sample_for_zero_bits_in_sample_bits_mode() {
    run_mock_light(true, |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let samples = [
            ctx.load_witness(Fr::from(7u64)),
            ctx.load_witness(Fr::from(11u64)),
        ];
        let mut cursor = SampleCursor::new(&samples);
        let zero_bits = ctx.load_witness(Fr::from(0u64));
        bind_sample_bits(ctx, &range, &mut cursor, 0, zero_bits, true);

        let consumed = ctx.load_witness(Fr::from(cursor.cursor as u64));
        range
            .gate()
            .assert_is_const(ctx, &consumed, &Fr::from(1u64));
    });
}

#[test]
fn bind_sample_bits_does_not_consume_sample_for_zero_bits_in_check_witness_mode() {
    run_mock_light(true, |builder| {
        let range = builder.range_chip();
        let ctx = builder.main(0);
        let samples = [
            ctx.load_witness(Fr::from(7u64)),
            ctx.load_witness(Fr::from(11u64)),
        ];
        let mut cursor = SampleCursor::new(&samples);
        let zero_bits = ctx.load_witness(Fr::from(0u64));
        bind_sample_bits(ctx, &range, &mut cursor, 0, zero_bits, false);

        let consumed = ctx.load_witness(Fr::from(cursor.cursor as u64));
        range
            .gate()
            .assert_is_const(ctx, &consumed, &Fr::from(0u64));
    });
}

#[test]
fn bind_sample_bits_rejects_bits_equal_31() {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_mock_light(true, |builder| {
            let range = builder.range_chip();
            let ctx = builder.main(0);
            let samples = [ctx.load_witness(Fr::from(7u64))];
            let mut cursor = SampleCursor::new(&samples);
            let target_bits = ctx.load_witness(Fr::from(0u64));
            bind_sample_bits(ctx, &range, &mut cursor, 31, target_bits, true);
        });
    }));
    assert!(
        result.is_err(),
        "bind_sample_bits(31) must be rejected to match backend bound semantics",
    );
}

#[test]
fn pipeline_constraints_fail_when_public_inputs_are_tampered() {
    let engine = test_engine();
    let (vk, proof) = InteractionsFixture11.keygen_and_prove(&engine);
    let mut tampered_public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);
    tampered_public_inputs[0] += Fr::from(1u64);

    run_mock(false, &tampered_public_inputs, |builder| {
        build_end_to_end_constraints_from_proof(builder, engine.config(), &vk, &proof);
    });
}

#[test]
fn pipeline_real_prover_keygen_prove_verify_roundtrip() {
    let engine = test_engine();
    let (vk, proof) = FibFixture::new(0, 1, 1 << 5).keygen_and_prove(&engine);
    let public_inputs = derive_pipeline_public_inputs(engine.config(), &vk, &proof);

    let mut keygen_builder = BaseCircuitBuilder::from_stage(CircuitBuilderStage::Keygen)
        .use_k(END_TO_END_K as usize)
        .use_lookup_bits(END_TO_END_LOOKUP_BITS)
        .use_instance_columns(1);
    build_end_to_end_constraints_from_proof(&mut keygen_builder, engine.config(), &vk, &proof);

    let config_params = keygen_builder.calculate_params(Some(END_TO_END_MIN_ROWS));
    assert!(
        config_params
            .num_advice_per_phase
            .first()
            .copied()
            .unwrap_or_default()
            >= STATIC_VERIFIER_NUM_ADVICE_COLS_PHASE0
    );
    assert!(
        config_params
            .num_lookup_advice_per_phase
            .first()
            .copied()
            .unwrap_or_default()
            >= STATIC_VERIFIER_LOOKUP_ADVICE_COLS_PHASE0
    );

    let srs = gen_srs(END_TO_END_K);
    let vk_halo2 =
        keygen_vk(&srs, &keygen_builder).expect("keygen_vk should succeed for pipeline circuit");
    let pk = keygen_pk(&srs, vk_halo2, &keygen_builder).expect("keygen_pk should succeed for pipeline");
    let break_points = keygen_builder.break_points();

    let mut prover_builder = BaseCircuitBuilder::prover(config_params, break_points);
    build_end_to_end_constraints_from_proof(&mut prover_builder, engine.config(), &vk, &proof);

    let halo2_proof = gen_halo2_proof(&srs, &pk, prover_builder, &public_inputs);
    verify_halo2_proof(&srs, pk.get_vk(), &halo2_proof, &public_inputs);
}
