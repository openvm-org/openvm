use core::iter::zip;

use halo2_base::{
    AssignedValue, Context,
    gates::{GateInstructions, RangeInstructions, range::RangeChip},
    utils::biguint_to_fe,
};
use num_bigint::BigUint;
use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::{
        BabyBearBn254Poseidon2Config as NativeConfig, Bn254Scalar, F as NativeF,
    },
    openvm_stark_backend::{
        StarkProtocolConfig, SystemParams,
        air_builders::symbolic::{SymbolicExpressionNode, symbolic_variable::Entry},
        calculate_n_logup,
        interaction::Interaction,
        keygen::types::MultiStarkVerifyingKey,
        p3_field::{PrimeField, PrimeField64},
        proof::Proof,
        prover::stacked_pcs::StackedLayout,
    },
};

use crate::{
    circuit::Fr,
    gadgets::{
        baby_bear::{
            BABY_BEAR_BITS, BABY_BEAR_EXT_DEGREE, BABY_BEAR_MODULUS_U64, BabyBearArithmeticGadgets,
            BabyBearExtVar,
        },
        transcript::{
            AssignedTranscriptEvent, LoggedTranscript, TranscriptEvent,
            constrain_transcript_events, split_assigned_bn254_to_babybear_limbs,
        },
    },
    stages::{
        batch_and_stacked::{
            AssignedBatchAndStackedIntermediates, BatchAndStackedError, BatchAndStackedIntermediates, RawBatchAndStackedWitnessState,
            constrain_checked_batch_and_stacked_witness_state_unchecked,
        },
        pipeline::{
            collect_trace_commitments, derive_need_rot_per_commit, derive_u_cube_from_prism,
            prepare_pipeline_inputs,
        },
        proof_shape::{
            AssignedProofShapeIntermediates, ProofShapeIntermediates, ProofShapeOwnershipSchedule, ProofShapePreambleError,
            RawProofShapeWitnessState, constrain_checked_proof_shape_witness_state_with_ownership,
            derive_proof_shape_intermediates, derive_proof_shape_ownership_schedule, derive_proof_shape_rules,
        },
        stacked_reduction::{
            StackedReductionConstraintError, QCoeffAccumulationTerm,
            coeffs_to_native_ext as stacked_coeffs_to_native_ext,
            derive_stacked_reduction_intermediates_with_inputs,
        },
        whir::{
            AssignedWhirIntermediates, WhirError, WhirIntermediates, RawWhirWitnessState,
            constrain_checked_whir_witness_state_unchecked,
            derive_whir_intermediates_with_inputs,
        },
    },
    utils::{assign_and_range_u64, usize_to_u64},
};

#[derive(Debug, PartialEq, Eq)]
pub enum PipelineError {
    ProofShape(ProofShapePreambleError),
    BatchAndStacked(BatchAndStackedError),
    Whir(WhirError),
}

impl From<ProofShapePreambleError> for PipelineError {
    fn from(value: ProofShapePreambleError) -> Self {
        Self::ProofShape(value)
    }
}

impl From<BatchAndStackedError> for PipelineError {
    fn from(value: BatchAndStackedError) -> Self {
        Self::BatchAndStacked(value)
    }
}

impl From<WhirError> for PipelineError {
    fn from(value: WhirError) -> Self {
        Self::Whir(value)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PipelineIntermediates {
    pub proof_shape: ProofShapeIntermediates,
    pub batch_and_stacked: BatchAndStackedIntermediates,
    pub whir: WhirIntermediates,
    pub transcript_events: Vec<TranscriptEvent>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PipelineStatementWitness {
    pub mvk_pre_hash: Fr,
    pub proof_common_main_commit: Fr,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PipelineTranscriptSchedule {
    pub raw_preamble_observe_count: usize,
    pub non_preamble_observe_count: usize,
    pub air_is_required: Vec<bool>,
    pub air_has_preprocessed: Vec<bool>,
    pub air_num_public_values: Vec<usize>,
    pub air_num_cached_mains: Vec<usize>,
    pub air_preprocessed_commit_roots: Vec<Option<Fr>>,
    pub batch_total_interactions: u64,
    pub has_gkr_observe_payload: bool,
    pub l_skip: usize,
    pub batch_n_per_trace: Vec<isize>,
    pub batch_n_logup: usize,
    pub batch_n_max: usize,
    pub batch_degree: usize,
    pub batch_univariate_coeffs_len: usize,
    pub batch_trace_has_preprocessed: Vec<bool>,
    pub batch_column_openings_need_rot: Vec<Vec<bool>>,
    pub batch_column_opening_expected_widths: Vec<Vec<usize>>,
    pub batch_trace_constraint_nodes: Vec<Vec<SymbolicExpressionNode<NativeF>>>,
    pub batch_trace_constraint_indices: Vec<Vec<usize>>,
    pub batch_trace_interactions: Vec<Vec<Interaction<usize>>>,
    pub stacked_q_coeff_terms: Vec<QCoeffAccumulationTerm>,
    pub stacked_matrix_expected_widths: Vec<usize>,
    pub proof_shape_ownership: ProofShapeOwnershipSchedule,
    pub proof_shape_max_log_height_allowed: usize,
    pub logup_pow_bits: usize,
    pub mu_pow_bits: usize,
    pub folding_pow_bits: usize,
    pub query_phase_pow_bits: usize,
    pub folding_counts_per_round: Vec<usize>,
    pub query_counts_per_round: Vec<usize>,
    pub query_index_bits: Vec<usize>,
    pub whir_k_whir: usize,
    pub whir_initial_log_rs_domain_size: usize,
    pub whir_expected_final_poly_len: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawPipelineWitnessState {
    pub statement: PipelineStatementWitness,
    pub schedule: PipelineTranscriptSchedule,
    pub intermediates: PipelineIntermediates,
}

#[derive(Clone, Debug)]
pub struct AssignedPipelineIntermediates {
    pub proof_shape: AssignedProofShapeIntermediates,
    pub batch_and_stacked: AssignedBatchAndStackedIntermediates,
    pub whir: AssignedWhirIntermediates,
    pub statement_public_inputs: [AssignedValue<Fr>; 2],
}

#[derive(Clone, Debug)]
pub struct CheckedPipelineWitnessState {
    pub assigned: AssignedPipelineIntermediates,
    pub derived: DerivedPipelineState,
}

#[derive(Clone, Debug)]
pub struct DerivedPipelineState {
    pub consumed_non_preamble_observes: usize,
}

const BN254_DIGEST_BABYBEAR_LIMBS: usize = 3;

fn constrain_usize_to_const(
    ctx: &mut Context<Fr>,
    gate: &impl GateInstructions<Fr>,
    actual: usize,
    expected: usize,
) {
    let actual_cell = ctx.load_witness(Fr::from(usize_to_u64(actual)));
    gate.assert_is_const(ctx, &actual_cell, &Fr::from(usize_to_u64(expected)));
}

fn constrain_usize_equal(
    ctx: &mut Context<Fr>,
    _gate: &impl GateInstructions<Fr>,
    lhs: usize,
    rhs: usize,
) {
    let lhs_cell = ctx.load_witness(Fr::from(usize_to_u64(lhs)));
    let rhs_cell = ctx.load_witness(Fr::from(usize_to_u64(rhs)));
    ctx.constrain_equal(&lhs_cell, &rhs_cell);
}

fn constrain_bool_to_const(
    ctx: &mut Context<Fr>,
    gate: &impl GateInstructions<Fr>,
    actual: bool,
    expected: bool,
) {
    let actual_cell = ctx.load_witness(Fr::from(actual as u64));
    gate.assert_bit(ctx, actual_cell);
    gate.assert_is_const(ctx, &actual_cell, &Fr::from(expected as u64));
}

fn derive_non_preamble_observes(events: &[TranscriptEvent], preamble_observes: usize) -> Vec<u64> {
    events
        .iter()
        .skip(preamble_observes)
        .filter_map(|event| match event {
            TranscriptEvent::Observe(value) => Some(*value),
            TranscriptEvent::Sample(_) => None,
        })
        .collect::<Vec<_>>()
}

pub fn derive_pipeline_intermediates(
    config: &NativeConfig,
    mvk: &MultiStarkVerifyingKey<NativeConfig>,
    proof: &Proof<NativeConfig>,
) -> Result<PipelineIntermediates, PipelineError> {
    let proof_shape = derive_proof_shape_intermediates(config, mvk, proof)?;

    let mut transcript = LoggedTranscript::new();
    let prepared = prepare_pipeline_inputs(&mut transcript, config, mvk, proof)
        .map_err(|err| PipelineError::BatchAndStacked(BatchAndStackedError::Batch(err)))?;

    let stacked_reduction = derive_stacked_reduction_intermediates_with_inputs(
        &mut transcript,
        &proof.stacking_proof,
        &prepared.layouts,
        &prepared.need_rot_per_commit,
        prepared.l_skip,
        mvk.inner.params.n_stack,
        &proof.batch_constraint_proof.column_openings,
        &prepared.r,
        &prepared.omega_skip_pows,
    )
    .map_err(|err| {
        PipelineError::BatchAndStacked(BatchAndStackedError::StackedReduction(
            StackedReductionConstraintError::StackedReduction(err),
        ))
    })?;

    let u_prism = stacked_reduction
        .u
        .iter()
        .copied()
        .map(stacked_coeffs_to_native_ext)
        .collect::<Vec<_>>();
    let u_cube = derive_u_cube_from_prism(&u_prism, prepared.l_skip)
        .map_err(|err| PipelineError::BatchAndStacked(BatchAndStackedError::Batch(err)))?;
    let commits = collect_trace_commitments(&mvk.inner, proof, &prepared.trace_id_to_air_id)
        .map_err(|err| PipelineError::BatchAndStacked(BatchAndStackedError::Batch(err)))?;

    let whir = derive_whir_intermediates_with_inputs(
        &mut transcript,
        config,
        &proof.whir_proof,
        &proof.stacking_proof.stacking_openings,
        &commits,
        &u_cube,
    )
    .map_err(|err| PipelineError::Whir(WhirError::Whir(err)))?;

    let batch_and_stacked = BatchAndStackedIntermediates {
        batch: prepared.batch,
        stacked_reduction,
    };

    let transcript_events = transcript.into_events();

    Ok(PipelineIntermediates {
        proof_shape,
        batch_and_stacked,
        whir,
        transcript_events,
    })
}

fn digest_scalar_to_fr(value: Bn254Scalar) -> Fr {
    biguint_to_fe(&value.as_canonical_biguint())
}

fn derive_pipeline_statement_witness(
    mvk: &MultiStarkVerifyingKey<NativeConfig>,
    proof: &Proof<NativeConfig>,
) -> PipelineStatementWitness {
    PipelineStatementWitness {
        mvk_pre_hash: digest_scalar_to_fr(mvk.pre_hash[0]),
        proof_common_main_commit: digest_scalar_to_fr(proof.common_main_commit[0]),
    }
}

fn derive_query_index_bits(params: &SystemParams) -> Vec<usize> {
    let mut log_rs_domain_size = params.l_skip + params.n_stack + params.log_blowup;
    let k_whir = params.k_whir();
    let mut query_index_bits = Vec::new();
    for round in &params.whir.rounds {
        let query_bits = log_rs_domain_size - k_whir;
        query_index_bits.extend(core::iter::repeat(query_bits).take(round.num_queries));
        log_rs_domain_size -= 1;
    }
    query_index_bits
}

fn derive_folding_counts_per_round(params: &SystemParams) -> Vec<usize> {
    let mut remaining = params.num_whir_sumcheck_rounds();
    let k_whir = params.k_whir();
    let mut per_round = Vec::with_capacity(params.num_whir_rounds());
    for _ in 0..params.num_whir_rounds() {
        let count = remaining.min(k_whir);
        per_round.push(count);
        remaining = remaining.saturating_sub(count);
    }
    per_round
}

fn derive_batch_n_per_trace(
    trace_id_to_air_id: &[usize],
    air_log_heights: &[usize],
    l_skip: usize,
) -> Vec<isize> {
    trace_id_to_air_id
        .iter()
        .map(|&air_id| air_log_heights[air_id] as isize - l_skip as isize)
        .collect::<Vec<_>>()
}

fn derive_batch_total_interactions(
    mvk: &MultiStarkVerifyingKey<NativeConfig>,
    trace_id_to_air_id: &[usize],
    n_per_trace: &[isize],
    l_skip: usize,
) -> u64 {
    zip(trace_id_to_air_id, n_per_trace)
        .map(|(&air_idx, &n)| {
            let n_lift = n.max(0) as usize;
            let num_interactions = mvk.inner.per_air[air_idx]
                .symbolic_constraints
                .interactions
                .len();
            (num_interactions as u64) << (l_skip + n_lift)
        })
        .sum::<u64>()
}

fn derive_batch_column_openings_need_rot(
    mvk: &MultiStarkVerifyingKey<NativeConfig>,
    trace_id_to_air_id: &[usize],
) -> Vec<Vec<bool>> {
    trace_id_to_air_id
        .iter()
        .map(|&air_id| {
            let need_rot = mvk.inner.per_air[air_id].params.need_rot;
            vec![need_rot; mvk.inner.per_air[air_id].num_parts()]
        })
        .collect::<Vec<_>>()
}

fn derive_batch_column_opening_expected_widths(
    mvk: &MultiStarkVerifyingKey<NativeConfig>,
    trace_id_to_air_id: &[usize],
) -> Vec<Vec<usize>> {
    trace_id_to_air_id
        .iter()
        .map(|&air_id| {
            let air = &mvk.inner.per_air[air_id];
            let openings_per_col = if air.params.need_rot { 2 } else { 1 };
            let mut widths = Vec::with_capacity(air.num_parts());
            widths.push(air.params.width.common_main * openings_per_col);
            if let Some(preprocessed) = air.params.width.preprocessed {
                widths.push(preprocessed * openings_per_col);
            }
            widths.extend(
                air.params
                    .width
                    .cached_mains
                    .iter()
                    .map(|&width| width * openings_per_col),
            );
            widths
        })
        .collect::<Vec<_>>()
}

fn derive_stacked_q_coeff_term_schedule(
    layouts: &[StackedLayout],
    need_rot_per_commit: &[Vec<bool>],
    l_skip: usize,
    n_stack: usize,
) -> Vec<QCoeffAccumulationTerm> {
    let mut lambda_idx = 0usize;
    let lambda_indices_per_layout: Vec<Vec<(usize, bool)>> = layouts
        .iter()
        .enumerate()
        .map(|(commit_idx, layout)| {
            let need_rot_for_commit = &need_rot_per_commit[commit_idx];
            layout
                .sorted_cols
                .iter()
                .map(|&(mat_idx, _col_idx, _slice)| {
                    lambda_idx += 1;
                    (lambda_idx - 1, need_rot_for_commit[mat_idx])
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let mut q_coeff_terms = Vec::new();
    for (commit_idx, layout) in layouts.iter().enumerate() {
        let lambda_indices = &lambda_indices_per_layout[commit_idx];
        for (col_idx, &(_, _, slice)) in layout.sorted_cols.iter().enumerate() {
            let (lambda_idx, need_rot) = lambda_indices[col_idx];
            let n = slice.log_height() as isize - l_skip as isize;
            let n_lift = n.max(0) as usize;
            let b_bits = (l_skip + n_lift..l_skip + n_stack)
                .map(|j| ((slice.row_idx >> j) & 1) == 1)
                .collect::<Vec<_>>();
            q_coeff_terms.push(QCoeffAccumulationTerm {
                commit_idx,
                target_col_idx: slice.col_idx,
                lambda_idx,
                need_rot,
                n,
                b_bits,
            });
        }
    }
    q_coeff_terms
}

fn derive_stacked_matrix_expected_widths(layouts: &[StackedLayout]) -> Vec<usize> {
    layouts
        .iter()
        .map(|layout| {
            layout
                .sorted_cols
                .last()
                .map(|(_, _, slice)| slice.col_idx + 1)
                .expect("stacked layout must contain at least one column")
        })
        .collect::<Vec<_>>()
}

fn derive_initial_log_rs_domain_size(params: &SystemParams) -> usize {
    params.l_skip + params.n_stack + params.log_blowup
}

fn derive_preamble_observe_count(
    proof_shape: &ProofShapeIntermediates,
    batch_and_stacked: &BatchAndStackedIntermediates,
    air_is_required: &[bool],
    air_has_preprocessed: &[bool],
) -> usize {
    let num_airs = air_is_required.len();
    assert_eq!(
        air_has_preprocessed.len(),
        num_airs,
        "preamble schedule protocol vectors must align",
    );
    assert_eq!(
        proof_shape.air_presence_flags.len(),
        num_airs,
        "proof-shape air-presence vector must align with protocol schedule",
    );
    assert_eq!(
        proof_shape.air_cached_commitment_lens.len(),
        num_airs,
        "proof-shape cached-commitment vector must align with protocol schedule",
    );
    assert_eq!(
        batch_and_stacked.batch.public_values.len(),
        num_airs,
        "batch public-value vector must align with protocol schedule",
    );

    let mut count = 6usize;
    for air_idx in 0..num_airs {
        if !air_is_required[air_idx] {
            count += 1;
        }
        if proof_shape.air_presence_flags[air_idx] {
            count += if air_has_preprocessed[air_idx] { 3 } else { 1 };
            count += 3 * proof_shape.air_cached_commitment_lens[air_idx];
        }
        count += batch_and_stacked.batch.public_values[air_idx].len();
    }
    count
}

fn derive_pipeline_transcript_schedule(
    config: &NativeConfig,
    mvk: &MultiStarkVerifyingKey<NativeConfig>,
    proof: &Proof<NativeConfig>,
    intermediates: &PipelineIntermediates,
) -> Result<PipelineTranscriptSchedule, PipelineError> {
    let mvk0 = &mvk.inner;
    let params = config.params();
    let l_skip = params.l_skip;
    let trace_id_to_air_id = &intermediates.proof_shape.trace_id_to_air_id;
    let query_counts_per_round = params
        .whir
        .rounds
        .iter()
        .map(|round| round.num_queries)
        .collect::<Vec<_>>();
    let batch_n_per_trace = derive_batch_n_per_trace(
        trace_id_to_air_id,
        &intermediates.proof_shape.air_log_heights,
        l_skip,
    );
    let batch_total_interactions =
        derive_batch_total_interactions(mvk, trace_id_to_air_id, &batch_n_per_trace, l_skip);
    let batch_n_logup = calculate_n_logup(l_skip, batch_total_interactions);
    let batch_n_max = batch_n_per_trace
        .iter()
        .copied()
        .max()
        .unwrap_or(0)
        .max(0) as usize;
    let batch_degree = mvk0.max_constraint_degree() + 1;
    let l_skip_width = if l_skip >= usize::BITS as usize {
        0usize
    } else {
        1usize << l_skip
    };
    let batch_univariate_coeffs_len = batch_degree
        .saturating_mul(l_skip_width.saturating_sub(1))
        .saturating_add(1);
    let batch_trace_has_preprocessed = trace_id_to_air_id
        .iter()
        .map(|&air_id| mvk0.per_air[air_id].preprocessed_data.is_some())
        .collect::<Vec<_>>();
    let batch_column_openings_need_rot =
        derive_batch_column_openings_need_rot(mvk, trace_id_to_air_id);
    let batch_column_opening_expected_widths =
        derive_batch_column_opening_expected_widths(mvk, trace_id_to_air_id);
    let batch_trace_constraint_nodes = trace_id_to_air_id
        .iter()
        .map(|&air_id| {
            mvk0.per_air[air_id]
                .symbolic_constraints
                .constraints
                .nodes
                .clone()
        })
        .collect::<Vec<_>>();
    let batch_trace_constraint_indices = trace_id_to_air_id
        .iter()
        .map(|&air_id| {
            mvk0.per_air[air_id]
                .symbolic_constraints
                .constraints
                .constraint_idx
                .clone()
        })
        .collect::<Vec<_>>();
    let batch_trace_interactions = trace_id_to_air_id
        .iter()
        .map(|&air_id| {
            mvk0.per_air[air_id]
                .symbolic_constraints
                .interactions
                .clone()
        })
        .collect::<Vec<_>>();
    let proof_shape_ownership = derive_proof_shape_ownership_schedule(mvk, proof)?;

    let layouts = derive_proof_shape_rules(mvk0, proof)
        .map_err(|err| PipelineError::ProofShape(ProofShapePreambleError::ProofShape(err)))?
        .layouts;
    let need_rot_per_commit = derive_need_rot_per_commit(mvk0, proof, trace_id_to_air_id)
        .map_err(|err| PipelineError::BatchAndStacked(BatchAndStackedError::Batch(err)))?;
    let stacked_q_coeff_terms = derive_stacked_q_coeff_term_schedule(
        &layouts,
        &need_rot_per_commit,
        l_skip,
        params.n_stack,
    );
    let stacked_matrix_expected_widths = derive_stacked_matrix_expected_widths(&layouts);
    let whir_k_whir = params.k_whir();
    let whir_initial_log_rs_domain_size = derive_initial_log_rs_domain_size(params);

    let air_is_required = mvk
        .inner
        .per_air
        .iter()
        .map(|air_vk| air_vk.is_required)
        .collect::<Vec<_>>();
    let air_has_preprocessed = mvk
        .inner
        .per_air
        .iter()
        .map(|air_vk| air_vk.preprocessed_data.is_some())
        .collect::<Vec<_>>();
    let air_num_public_values = mvk
        .inner
        .per_air
        .iter()
        .map(|air_vk| air_vk.params.num_public_values)
        .collect::<Vec<_>>();
    let air_num_cached_mains = mvk
        .inner
        .per_air
        .iter()
        .map(|air_vk| air_vk.num_cached_mains())
        .collect::<Vec<_>>();
    let air_preprocessed_commit_roots = mvk
        .inner
        .per_air
        .iter()
        .map(|air_vk| {
            air_vk
                .preprocessed_data
                .as_ref()
                .map(|preprocessed| digest_scalar_to_fr(preprocessed.commit[0]))
        })
        .collect::<Vec<_>>();
    let preamble_observe_count = derive_preamble_observe_count(
        &intermediates.proof_shape,
        &intermediates.batch_and_stacked,
        &air_is_required,
        &air_has_preprocessed,
    );
    Ok(PipelineTranscriptSchedule {
        raw_preamble_observe_count: preamble_observe_count,
        non_preamble_observe_count: derive_non_preamble_observes(
            &intermediates.transcript_events,
            preamble_observe_count,
        )
        .len(),
        air_is_required,
        air_has_preprocessed,
        air_num_public_values,
        air_num_cached_mains,
        air_preprocessed_commit_roots,
        batch_total_interactions,
        has_gkr_observe_payload: batch_total_interactions > 0,
        l_skip,
        batch_n_per_trace,
        batch_n_logup,
        batch_n_max,
        batch_degree,
        batch_univariate_coeffs_len,
        batch_trace_has_preprocessed,
        batch_column_openings_need_rot,
        batch_column_opening_expected_widths,
        batch_trace_constraint_nodes,
        batch_trace_constraint_indices,
        batch_trace_interactions,
        stacked_q_coeff_terms,
        stacked_matrix_expected_widths,
        proof_shape_ownership,
        proof_shape_max_log_height_allowed: l_skip + params.n_stack,
        logup_pow_bits: params.logup_pow_bits(),
        mu_pow_bits: params.whir.mu_pow_bits,
        folding_pow_bits: params.whir.folding_pow_bits,
        query_phase_pow_bits: params.whir.query_phase_pow_bits,
        folding_counts_per_round: derive_folding_counts_per_round(params),
        query_counts_per_round,
        query_index_bits: derive_query_index_bits(params),
        whir_k_whir,
        whir_initial_log_rs_domain_size,
        whir_expected_final_poly_len: 1usize << params.log_final_poly_len(),
    })
}

pub(crate) fn derive_raw_pipeline_witness_state(
    config: &NativeConfig,
    mvk: &MultiStarkVerifyingKey<NativeConfig>,
    proof: &Proof<NativeConfig>,
) -> Result<RawPipelineWitnessState, PipelineError> {
    // Strict assignment boundary: this is the only unchecked host-derivation entrypoint for the pipeline.
    // All downstream APIs consume this typed bundle and add explicit in-circuit checks.
    let intermediates = derive_pipeline_intermediates(config, mvk, proof)?;
    let schedule = derive_pipeline_transcript_schedule(config, mvk, proof, &intermediates)?;
    let statement = derive_pipeline_statement_witness(mvk, proof);
    Ok(RawPipelineWitnessState {
        statement,
        schedule,
        intermediates,
    })
}

pub fn derive_pipeline_public_inputs(
    _config: &NativeConfig,
    mvk: &MultiStarkVerifyingKey<NativeConfig>,
    proof: &Proof<NativeConfig>,
) -> Vec<Fr> {
    let statement = derive_pipeline_statement_witness(mvk, proof);
    vec![statement.mvk_pre_hash, statement.proof_common_main_commit]
}

struct EventCursor<'a> {
    events: &'a [AssignedTranscriptEvent],
    cursor: usize,
}

impl<'a> EventCursor<'a> {
    fn new(events: &'a [AssignedTranscriptEvent]) -> Self {
        Self { events, cursor: 0 }
    }

    fn consume_observe(
        &mut self,
        ctx: &mut Context<Fr>,
        gate: &impl GateInstructions<Fr>,
    ) -> AssignedValue<Fr> {
        if let Some(event) = self.events.get(self.cursor) {
            gate.assert_is_const(ctx, &event.is_sample, &Fr::from(0u64));
            self.cursor += 1;
            event.value
        } else {
            let missing_event = ctx.load_witness(Fr::from(0u64));
            gate.assert_is_const(ctx, &missing_event, &Fr::from(1u64));
            ctx.load_constant(Fr::from(0u64))
        }
    }

    fn consume_next_observe(
        &mut self,
        ctx: &mut Context<Fr>,
        gate: &impl GateInstructions<Fr>,
    ) -> AssignedValue<Fr> {
        while self.cursor < self.events.len() {
            let event = self
                .events
                .get(self.cursor)
                .expect("transcript event cursor exceeded available events");
            self.cursor += 1;
            if *event.is_sample.value() == Fr::from(0u64) {
                gate.assert_is_const(ctx, &event.is_sample, &Fr::from(0u64));
                return event.value;
            }
        }
        let missing_event = ctx.load_witness(Fr::from(0u64));
        gate.assert_is_const(ctx, &missing_event, &Fr::from(1u64));
        ctx.load_constant(Fr::from(0u64))
    }

    fn constrain_consumed_prefix(
        &self,
        ctx: &mut Context<Fr>,
        gate: &impl GateInstructions<Fr>,
        expected: usize,
    ) {
        let consumed = ctx.load_witness(Fr::from(usize_to_u64(self.cursor)));
        gate.assert_is_const(ctx, &consumed, &Fr::from(usize_to_u64(expected)));
    }
}

struct SampleCursor<'a> {
    samples: &'a [AssignedValue<Fr>],
    cursor: usize,
}

impl<'a> SampleCursor<'a> {
    fn new(samples: &'a [AssignedValue<Fr>]) -> Self {
        Self { samples, cursor: 0 }
    }

    fn consume(&mut self) -> Option<AssignedValue<Fr>> {
        let sample = self.samples.get(self.cursor).copied();
        if sample.is_some() {
            self.cursor += 1;
        }
        sample
    }
}

fn sampled_bits_from_sample(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    sample: AssignedValue<Fr>,
    bits: usize,
) -> AssignedValue<Fr> {
    assert!(
        bits < (u32::BITS as usize),
        "sample_bits requires bits < 32: {bits}",
    );
    assert!(
        (1u64 << bits) < BABY_BEAR_MODULUS_U64,
        "sample_bits requires (1 << bits) < modulus: bits={bits}",
    );
    if bits == 0 {
        return ctx.load_constant(Fr::from(0u64));
    }

    let (_, rem) = range.div_mod(ctx, sample, BigUint::from(1u64) << bits, BABY_BEAR_BITS);
    range.range_check(ctx, rem, bits);
    rem
}

fn bind_sample_bits(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    cursor: &mut SampleCursor<'_>,
    bits: usize,
    target_bits: AssignedValue<Fr>,
    consume_on_zero_bits: bool,
) {
    let gate = range.gate();
    assert!(
        bits < (u32::BITS as usize),
        "sample_bits requires bits < 32: {bits}",
    );
    assert!(
        (1u64 << bits) < BABY_BEAR_MODULUS_U64,
        "sample_bits requires (1 << bits) < modulus: bits={bits}",
    );
    if bits == 0 {
        if consume_on_zero_bits {
            if cursor.consume().is_none() {
                let missing_sample = ctx.load_witness(Fr::from(0u64));
                gate.assert_is_const(ctx, &missing_sample, &Fr::from(1u64));
            }
        }
        gate.assert_is_const(ctx, &target_bits, &Fr::from(0u64));
        return;
    }
    let sampled = cursor.consume().unwrap_or_else(|| {
        let missing_sample = ctx.load_witness(Fr::from(0u64));
        gate.assert_is_const(ctx, &missing_sample, &Fr::from(1u64));
        ctx.load_constant(Fr::from(0u64))
    });
    let sampled_bits = sampled_bits_from_sample(ctx, range, sampled, bits);
    ctx.constrain_equal(&sampled_bits, &target_bits);
}

fn bind_sample_ext(ctx: &mut Context<Fr>, cursor: &mut SampleCursor<'_>, target: &BabyBearExtVar) {
    for coeff in &target.coeffs {
        let sampled = cursor.consume().unwrap_or_else(|| {
            let zero = ctx.load_constant(Fr::from(0u64));
            let one = ctx.load_constant(Fr::from(1u64));
            ctx.constrain_equal(&zero, &one);
            zero
        });
        ctx.constrain_equal(&coeff.cell, &sampled);
    }
}

fn bind_u_cube_from_stacked(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    l_skip: usize,
    stacked_u: &[BabyBearExtVar],
    whir_u_cube: &[BabyBearExtVar],
) {
    let gate = range.gate();
    constrain_bool_to_const(ctx, gate, !stacked_u.is_empty(), true);
    let baby_bear = BabyBearArithmeticGadgets;
    let zero_ext = baby_bear.ext_zero(ctx, range);

    let mut derived_u_cube = Vec::with_capacity(l_skip + stacked_u.len().saturating_sub(1));
    let mut power = stacked_u
        .first()
        .cloned()
        .unwrap_or_else(|| zero_ext.clone());
    for _ in 0..l_skip {
        derived_u_cube.push(power.clone());
        power = baby_bear.ext_mul(ctx, range, &power, &power);
    }
    derived_u_cube.extend(stacked_u.iter().skip(1).cloned());

    constrain_usize_equal(ctx, gate, derived_u_cube.len(), whir_u_cube.len());
    for (idx, derived) in derived_u_cube.iter().enumerate() {
        let actual = whir_u_cube.get(idx).unwrap_or(&zero_ext);
        baby_bear.assert_ext_equal(ctx, derived, actual);
    }
}

fn push_ext_observe_cells(observes: &mut Vec<AssignedValue<Fr>>, value: &BabyBearExtVar) {
    for coeff in &value.coeffs {
        observes.push(coeff.cell);
    }
}

fn push_column_claim_observe_cells(
    observes: &mut Vec<AssignedValue<Fr>>,
    claims: &[BabyBearExtVar],
    need_rot: bool,
    zero: AssignedValue<Fr>,
) {
    if need_rot {
        for pair in claims.chunks_exact(2) {
            push_ext_observe_cells(observes, &pair[0]);
            push_ext_observe_cells(observes, &pair[1]);
        }
    } else {
        for claim in claims {
            push_ext_observe_cells(observes, claim);
            for _ in 0..BABY_BEAR_EXT_DEGREE {
                observes.push(zero);
            }
        }
    }
}

fn derive_stage_payload_observe_cells(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    actual: &PipelineIntermediates,
    batch_and_stacked: &AssignedBatchAndStackedIntermediates,
    whir: &AssignedWhirIntermediates,
    schedule: &PipelineTranscriptSchedule,
) -> Vec<AssignedValue<Fr>> {
    let mut observes = Vec::new();
    let zero = ctx.load_constant(Fr::from(0u64));
    let gate = range.gate();
    let baby_bear = BabyBearArithmeticGadgets;
    let zero_ext = baby_bear.ext_zero(ctx, range);

    let batch = &batch_and_stacked.batch;
    if schedule.logup_pow_bits > 0 {
        observes.push(assign_and_range_u64(
            ctx,
            range,
            actual.batch_and_stacked.batch.logup_pow_witness,
        ));
    }

    if schedule.has_gkr_observe_payload {
        push_ext_observe_cells(
            &mut observes,
            batch
                .gkr_q0_claim
                .as_ref()
                .expect("GKR observe payload requires q0-claim witness"),
        );
        if let Some(layer0) = batch.gkr_claims_per_layer.first() {
            for claim in layer0 {
                push_ext_observe_cells(&mut observes, claim);
            }
        }
        for (round_idx, layer_claims) in batch.gkr_claims_per_layer.iter().enumerate().skip(1) {
            let round_evals = batch.gkr_sumcheck_polys.get(round_idx - 1);
            for eval in round_evals.into_iter().flat_map(|round| round.iter()) {
                push_ext_observe_cells(&mut observes, eval);
            }
            for claim in layer_claims {
                push_ext_observe_cells(&mut observes, claim);
            }
        }
    }

    constrain_usize_equal(
        ctx,
        gate,
        batch.numerator_term_per_air.len(),
        batch.denominator_term_per_air.len(),
    );
    constrain_usize_equal(
        ctx,
        gate,
        batch.numerator_term_per_air.len(),
        schedule.batch_n_per_trace.len(),
    );
    for air_idx in 0..schedule.batch_n_per_trace.len() {
        let num = batch
            .numerator_term_per_air
            .get(air_idx)
            .unwrap_or(&zero_ext);
        let den = batch
            .denominator_term_per_air
            .get(air_idx)
            .unwrap_or(&zero_ext);
        push_ext_observe_cells(&mut observes, num);
        push_ext_observe_cells(&mut observes, den);
    }

    constrain_usize_equal(
        ctx,
        gate,
        batch.univariate_round_coeffs.len(),
        schedule.batch_univariate_coeffs_len,
    );
    for coeff_idx in 0..schedule.batch_univariate_coeffs_len {
        let coeff = batch
            .univariate_round_coeffs
            .get(coeff_idx)
            .unwrap_or(&zero_ext);
        push_ext_observe_cells(&mut observes, coeff);
    }

    constrain_usize_equal(
        ctx,
        gate,
        batch.sumcheck_round_polys.len(),
        schedule.batch_n_max,
    );
    for round_idx in 0..schedule.batch_n_max {
        let round_evals = batch.sumcheck_round_polys.get(round_idx);
        constrain_usize_to_const(
            ctx,
            gate,
            round_evals.map_or(0usize, |round| round.len()),
            schedule.batch_degree,
        );
        for eval_idx in 0..schedule.batch_degree {
            let eval = round_evals
                .and_then(|round| round.get(eval_idx))
                .unwrap_or(&zero_ext);
            push_ext_observe_cells(&mut observes, eval);
        }
    }

    for (trace_idx, per_air) in batch.column_openings.iter().enumerate() {
        let need_rot = batch
            .column_openings_need_rot
            .get(trace_idx)
            .and_then(|flags| flags.first())
            .copied()
            .unwrap_or(false);
        if let Some(common_main) = per_air.first() {
            push_column_claim_observe_cells(&mut observes, common_main, need_rot, zero);
        }
    }
    for (trace_idx, per_air) in batch.column_openings.iter().enumerate() {
        let need_rot = batch
            .column_openings_need_rot
            .get(trace_idx)
            .and_then(|flags| flags.first())
            .copied()
            .unwrap_or(false);
        for part in per_air.iter().skip(1) {
            push_column_claim_observe_cells(&mut observes, part, need_rot, zero);
        }
    }

    let stacked = &batch_and_stacked.stacked_reduction;
    for coeff in &stacked.univariate_round_coeffs {
        push_ext_observe_cells(&mut observes, coeff);
    }
    for round_evals in &stacked.sumcheck_round_polys {
        for eval in round_evals {
            push_ext_observe_cells(&mut observes, eval);
        }
    }
    for commit_openings in &stacked.stacking_openings {
        for opening in commit_openings {
            push_ext_observe_cells(&mut observes, opening);
        }
    }

    if schedule.mu_pow_bits > 0 {
        observes.push(assign_and_range_u64(ctx, range, actual.whir.mu_pow_witness));
    }
    let mut sumcheck_cursor = 0usize;
    let mut folding_pow_cursor = 0usize;
    let default_whir_round = vec![zero_ext.clone(), zero_ext.clone()];
    for round_idx in 0..schedule.query_counts_per_round.len() {
        let folding_count = schedule
            .folding_counts_per_round
            .get(round_idx)
            .copied()
            .unwrap_or(0);
        for _ in 0..folding_count {
            let round_poly = whir
                .whir_sumcheck_polys
                .get(sumcheck_cursor)
                .unwrap_or(&default_whir_round);
            let ev1 = round_poly.get(0).unwrap_or(&zero_ext);
            let ev2 = round_poly.get(1).unwrap_or(&zero_ext);
            push_ext_observe_cells(&mut observes, ev1);
            push_ext_observe_cells(&mut observes, ev2);
            if schedule.folding_pow_bits > 0 {
                let pow_witness = actual
                    .whir
                    .folding_pow_witnesses
                    .get(folding_pow_cursor)
                    .copied()
                    .unwrap_or(0);
                observes.push(assign_and_range_u64(ctx, range, pow_witness));
            }
            sumcheck_cursor += 1;
            folding_pow_cursor += 1;
        }

        let is_final_round = round_idx + 1 == schedule.query_counts_per_round.len();
        if is_final_round {
            for coeff in &whir.final_poly {
                push_ext_observe_cells(&mut observes, coeff);
            }
        } else {
            let codeword_root = whir
                .codeword_commitment_roots
                .get(round_idx)
                .copied()
                .unwrap_or(zero);
            let root_limbs = split_assigned_bn254_to_babybear_limbs(ctx, range, codeword_root);
            for limb in root_limbs {
                observes.push(limb.cell);
            }
            let ood = whir.ood_values.get(round_idx).unwrap_or(&zero_ext);
            push_ext_observe_cells(&mut observes, ood);
        }

        if schedule.query_phase_pow_bits > 0 {
            let pow_witness = actual
                .whir
                .query_phase_pow_witnesses
                .get(round_idx)
                .copied()
                .unwrap_or(0);
            observes.push(assign_and_range_u64(ctx, range, pow_witness));
        }
    }

    let sumcheck_consumed = ctx.load_witness(Fr::from(usize_to_u64(sumcheck_cursor)));
    gate.assert_is_const(
        ctx,
        &sumcheck_consumed,
        &Fr::from(usize_to_u64(whir.whir_sumcheck_polys.len())),
    );
    let folding_pow_consumed = ctx.load_witness(Fr::from(usize_to_u64(folding_pow_cursor)));
    gate.assert_is_const(
        ctx,
        &folding_pow_consumed,
        &Fr::from(usize_to_u64(actual.whir.folding_pow_witnesses.len())),
    );

    observes
}

fn push_event_kinds(target: &mut Vec<bool>, is_sample: bool, count: usize) {
    for _ in 0..count {
        target.push(is_sample);
    }
}

fn count_column_claim_observe_events(claim_count: usize, need_rot: bool) -> usize {
    if need_rot {
        claim_count * BABY_BEAR_EXT_DEGREE
    } else {
        claim_count * BABY_BEAR_EXT_DEGREE * 2
    }
}

fn derive_post_preamble_event_kind_schedule(
    actual: &PipelineIntermediates,
    schedule: &PipelineTranscriptSchedule,
) -> Vec<bool> {
    let mut kinds = Vec::new();
    let batch_ref = &actual.batch_and_stacked.batch;
    let stacked_ref = &actual.batch_and_stacked.stacked_reduction;
    let whir = &actual.whir;

    if schedule.logup_pow_bits > 0 {
        // check_witness(logup_pow_bits, witness): observe witness, then sample_bits.
        push_event_kinds(&mut kinds, false, 1);
        push_event_kinds(&mut kinds, true, 1);
    }

    // sample_ext(alpha_logup), sample_ext(beta_logup)
    push_event_kinds(&mut kinds, true, 2 * BABY_BEAR_EXT_DEGREE);

    if schedule.has_gkr_observe_payload {
        let gkr_rounds = schedule.l_skip + schedule.batch_n_logup;

        // verify_gkr round 0: observe q0 + layer claims, then sample mu.
        push_event_kinds(&mut kinds, false, BABY_BEAR_EXT_DEGREE);
        push_event_kinds(&mut kinds, false, 4 * BABY_BEAR_EXT_DEGREE);
        push_event_kinds(&mut kinds, true, BABY_BEAR_EXT_DEGREE);

        // verify_gkr rounds 1..R-1.
        for round_idx in 1..gkr_rounds {
            // sample lambda for this round.
            push_event_kinds(&mut kinds, true, BABY_BEAR_EXT_DEGREE);

            // verify_gkr_sumcheck interleaves per subround:
            // observe [s(1), s(2), s(3)], then sample one challenge.
            for _ in 0..round_idx {
                push_event_kinds(&mut kinds, false, 3 * BABY_BEAR_EXT_DEGREE);
                push_event_kinds(&mut kinds, true, BABY_BEAR_EXT_DEGREE);
            }

            // Observe layer claims and sample mu for this round.
            push_event_kinds(&mut kinds, false, 4 * BABY_BEAR_EXT_DEGREE);
            push_event_kinds(&mut kinds, true, BABY_BEAR_EXT_DEGREE);
        }
    }

    let gkr_xi_len = if schedule.has_gkr_observe_payload {
        schedule.l_skip + schedule.batch_n_logup
    } else {
        0
    };
    let n_global = core::cmp::max(schedule.batch_n_max, schedule.batch_n_logup);
    let expected_xi_len = schedule.l_skip + n_global;
    let extra_xi = expected_xi_len.saturating_sub(gkr_xi_len);
    // Extra xi samples that are not consumed inside verify_gkr.
    push_event_kinds(&mut kinds, true, extra_xi * BABY_BEAR_EXT_DEGREE);

    // sample_ext(lambda)
    push_event_kinds(&mut kinds, true, BABY_BEAR_EXT_DEGREE);

    // observe_ext(numerator_term), observe_ext(denominator_term) per AIR.
    for _ in 0..schedule.batch_n_per_trace.len() {
        push_event_kinds(&mut kinds, false, 2 * BABY_BEAR_EXT_DEGREE);
    }

    // sample_ext(mu)
    push_event_kinds(&mut kinds, true, BABY_BEAR_EXT_DEGREE);

    // observe_ext(univariate round coeffs)
    push_event_kinds(
        &mut kinds,
        false,
        schedule.batch_univariate_coeffs_len * BABY_BEAR_EXT_DEGREE,
    );

    // sample_ext(r_0)
    push_event_kinds(&mut kinds, true, BABY_BEAR_EXT_DEGREE);

    // For each sumcheck round: observe polynomial, then sample next r_i.
    for _ in 0..schedule.batch_n_max {
        push_event_kinds(
            &mut kinds,
            false,
            schedule.batch_degree * BABY_BEAR_EXT_DEGREE,
        );
        push_event_kinds(&mut kinds, true, BABY_BEAR_EXT_DEGREE);
    }

    // Observe column openings: common-main first, then preprocessed/cached.
    for (trace_idx, per_air) in batch_ref.column_openings.iter().enumerate() {
        let need_rot = batch_ref
            .column_openings_need_rot
            .get(trace_idx)
            .and_then(|flags| flags.first())
            .copied()
            .unwrap_or(false);
        if let Some(common_main) = per_air.first() {
            push_event_kinds(
                &mut kinds,
                false,
                count_column_claim_observe_events(common_main.len(), need_rot),
            );
        }
    }
    for (trace_idx, per_air) in batch_ref.column_openings.iter().enumerate() {
        let need_rot = batch_ref
            .column_openings_need_rot
            .get(trace_idx)
            .and_then(|flags| flags.first())
            .copied()
            .unwrap_or(false);
        for part in per_air.iter().skip(1) {
            push_event_kinds(
                &mut kinds,
                false,
                count_column_claim_observe_events(part.len(), need_rot),
            );
        }
    }

    // Stacked reduction:
    // sample_ext(lambda)
    push_event_kinds(&mut kinds, true, BABY_BEAR_EXT_DEGREE);
    // observe_ext(univariate round coeffs)
    push_event_kinds(
        &mut kinds,
        false,
        stacked_ref.univariate_round_coeffs.len() * BABY_BEAR_EXT_DEGREE,
    );
    // sample_ext(u_0)
    push_event_kinds(&mut kinds, true, BABY_BEAR_EXT_DEGREE);
    // For each sumcheck round: observe evals, then sample u_j.
    for round in &stacked_ref.sumcheck_round_polys {
        push_event_kinds(&mut kinds, false, round.len() * BABY_BEAR_EXT_DEGREE);
        push_event_kinds(&mut kinds, true, BABY_BEAR_EXT_DEGREE);
    }
    // Final stacking openings are observed in row-major order.
    for commit_openings in &stacked_ref.stacking_openings {
        push_event_kinds(
            &mut kinds,
            false,
            commit_openings.len() * BABY_BEAR_EXT_DEGREE,
        );
    }

    // WHIR:
    if schedule.mu_pow_bits > 0 {
        // check_witness(mu_pow_bits, witness)
        push_event_kinds(&mut kinds, false, 1);
        push_event_kinds(&mut kinds, true, 1);
    }
    // sample_ext(mu)
    push_event_kinds(&mut kinds, true, BABY_BEAR_EXT_DEGREE);

    for round_idx in 0..schedule.query_counts_per_round.len() {
        let folding_count = schedule
            .folding_counts_per_round
            .get(round_idx)
            .copied()
            .unwrap_or(0);
        for _ in 0..folding_count {
            // observe_ext(ev1), observe_ext(ev2)
            push_event_kinds(&mut kinds, false, 2 * BABY_BEAR_EXT_DEGREE);
            if schedule.folding_pow_bits > 0 {
                // check_witness(folding_pow_bits, witness)
                push_event_kinds(&mut kinds, false, 1);
                push_event_kinds(&mut kinds, true, 1);
            }
            // sample_ext(alpha)
            push_event_kinds(&mut kinds, true, BABY_BEAR_EXT_DEGREE);
        }

        let is_final_round = round_idx + 1 == schedule.query_counts_per_round.len();
        if is_final_round {
            // observe_ext(final_poly coeffs)
            push_event_kinds(
                &mut kinds,
                false,
                whir.final_poly.len() * BABY_BEAR_EXT_DEGREE,
            );
        } else {
            // observe_commit(codeword root), sample_ext(z0), observe_ext(y0)
            push_event_kinds(&mut kinds, false, BN254_DIGEST_BABYBEAR_LIMBS);
            push_event_kinds(&mut kinds, true, BABY_BEAR_EXT_DEGREE);
            push_event_kinds(&mut kinds, false, BABY_BEAR_EXT_DEGREE);
        }

        if schedule.query_phase_pow_bits > 0 {
            // check_witness(query_phase_pow_bits, witness)
            push_event_kinds(&mut kinds, false, 1);
            push_event_kinds(&mut kinds, true, 1);
        }

        // sample_bits(query_bits) per query.
        let query_count = schedule
            .query_counts_per_round
            .get(round_idx)
            .copied()
            .unwrap_or(0);
        push_event_kinds(&mut kinds, true, query_count);
        // sample_ext(gamma)
        push_event_kinds(&mut kinds, true, BABY_BEAR_EXT_DEGREE);
    }

    kinds
}

fn constrain_post_preamble_event_kinds(
    ctx: &mut Context<Fr>,
    gate: &impl GateInstructions<Fr>,
    replay_events: &[AssignedTranscriptEvent],
    preamble_observe_count: usize,
    actual: &PipelineIntermediates,
    schedule: &PipelineTranscriptSchedule,
) {
    let expected_post_preamble = derive_post_preamble_event_kind_schedule(actual, schedule);
    let post_preamble_start = core::cmp::min(preamble_observe_count, replay_events.len());
    let actual_post_preamble = &replay_events[post_preamble_start..];
    let actual_len = ctx.load_witness(Fr::from(usize_to_u64(actual_post_preamble.len())));
    gate.assert_is_const(
        ctx,
        &actual_len,
        &Fr::from(usize_to_u64(expected_post_preamble.len())),
    );

    for (event, is_sample_expected) in actual_post_preamble.iter().zip(expected_post_preamble) {
        gate.assert_is_const(ctx, &event.is_sample, &Fr::from(is_sample_expected as u64));
    }
}

fn bind_batch_to_stacked_inputs(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    actual: &PipelineIntermediates,
    batch_and_stacked: &AssignedBatchAndStackedIntermediates,
) {
    let gate = range.gate();
    let baby_bear = BabyBearArithmeticGadgets;

    constrain_usize_equal(ctx, gate, batch_and_stacked.batch.r.len(), batch_and_stacked.stacked_reduction.r.len());
    for (batch_r, stacked_r) in batch_and_stacked.batch.r.iter().zip(&batch_and_stacked.stacked_reduction.r) {
        baby_bear.assert_ext_equal(ctx, batch_r, stacked_r);
    }

    constrain_usize_equal(
        ctx,
        gate,
        batch_and_stacked.batch.column_openings.len(),
        batch_and_stacked.stacked_reduction.batch_column_openings.len(),
    );
    for (batch_trace, stacked_trace) in batch_and_stacked
        .batch
        .column_openings
        .iter()
        .zip(&batch_and_stacked.stacked_reduction.batch_column_openings)
    {
        constrain_usize_equal(ctx, gate, batch_trace.len(), stacked_trace.len());
        for (batch_part, stacked_part) in batch_trace.iter().zip(stacked_trace) {
            constrain_usize_equal(ctx, gate, batch_part.len(), stacked_part.len());
            for (batch_opening, stacked_opening) in batch_part.iter().zip(stacked_part) {
                baby_bear.assert_ext_equal(ctx, batch_opening, stacked_opening);
            }
        }
    }

    let batch_need_rot = &actual.batch_and_stacked.batch.column_openings_need_rot;
    let stacked_need_rot = &actual.batch_and_stacked.stacked_reduction.batch_column_openings_need_rot;
    let expected_stacked_rows = 1usize
        + batch_need_rot
            .iter()
            .map(|row| row.len().saturating_sub(1))
            .sum::<usize>();
    constrain_usize_equal(ctx, gate, stacked_need_rot.len(), expected_stacked_rows);
    let stacked_common_row_width = stacked_need_rot.first().map_or(0usize, Vec::len);
    constrain_usize_equal(ctx, gate, stacked_common_row_width, batch_need_rot.len());
    for trace_idx in 0..batch_need_rot.len() {
        let batch_common = batch_need_rot[trace_idx].first().copied().unwrap_or(false);
        let stacked_common = stacked_need_rot
            .first()
            .and_then(|row| row.get(trace_idx))
            .copied()
            .unwrap_or(false);
        let batch_cell = ctx.load_witness(Fr::from(batch_common as u64));
        let stacked_cell = ctx.load_witness(Fr::from(stacked_common as u64));
        gate.assert_bit(ctx, batch_cell);
        gate.assert_bit(ctx, stacked_cell);
        ctx.constrain_equal(&batch_cell, &stacked_cell);
    }

    let mut commit_idx = 1usize;
    for batch_row in batch_need_rot {
        for &batch_part in batch_row.iter().skip(1) {
            let stacked_row = stacked_need_rot.get(commit_idx);
            constrain_usize_to_const(ctx, gate, stacked_row.map_or(0usize, |row| row.len()), 1);
            let stacked_part = stacked_row
                .and_then(|row| row.first())
                .copied()
                .unwrap_or(false);
            let batch_cell = ctx.load_witness(Fr::from(batch_part as u64));
            let stacked_cell = ctx.load_witness(Fr::from(stacked_part as u64));
            gate.assert_bit(ctx, batch_cell);
            gate.assert_bit(ctx, stacked_cell);
            ctx.constrain_equal(&batch_cell, &stacked_cell);
            commit_idx += 1;
        }
    }
    constrain_usize_equal(ctx, gate, commit_idx, stacked_need_rot.len());
}

fn bind_stacked_openings_to_whir(
    ctx: &mut Context<Fr>,
    gate: &impl GateInstructions<Fr>,
    batch_and_stacked: &AssignedBatchAndStackedIntermediates,
    whir: &AssignedWhirIntermediates,
) {
    let baby_bear = BabyBearArithmeticGadgets;
    constrain_usize_equal(
        ctx,
        gate,
        batch_and_stacked.stacked_reduction.stacking_openings.len(),
        whir.stacking_openings.len(),
    );
    for (stacked_commit, whir_commit) in batch_and_stacked
        .stacked_reduction
        .stacking_openings
        .iter()
        .zip(&whir.stacking_openings)
    {
        constrain_usize_equal(ctx, gate, stacked_commit.len(), whir_commit.len());
        for (stacked_opening, whir_opening) in stacked_commit.iter().zip(whir_commit) {
            baby_bear.assert_ext_equal(ctx, stacked_opening, whir_opening);
        }
    }
}

fn map_initial_commitment_roots_by_air(
    ctx: &mut Context<Fr>,
    gate: &impl GateInstructions<Fr>,
    initial_commitment_roots: &[AssignedValue<Fr>],
    trace_id_to_air_id: &[usize],
    air_has_preprocessed: &[bool],
    air_cached_commitment_lens: &[usize],
) -> (Vec<Option<AssignedValue<Fr>>>, Vec<Vec<AssignedValue<Fr>>>) {
    let num_airs = air_has_preprocessed.len();
    constrain_usize_to_const(ctx, gate, air_cached_commitment_lens.len(), num_airs);

    let mut preprocessed_roots = vec![None; num_airs];
    let mut cached_roots = vec![Vec::new(); num_airs];
    let zero_root = ctx.load_constant(Fr::from(0u64));
    let mut root_cursor = 1usize;
    let mut invalid_air_index_count = 0usize;

    for &air_idx in trace_id_to_air_id {
        if air_idx >= num_airs {
            invalid_air_index_count += 1;
            continue;
        }
        if air_has_preprocessed[air_idx] {
            let root = initial_commitment_roots
                .get(root_cursor)
                .copied()
                .unwrap_or(zero_root);
            preprocessed_roots[air_idx] = Some(root);
            root_cursor += 1;
        }

        let cached_len = air_cached_commitment_lens[air_idx];
        let mut cached_for_air = Vec::with_capacity(cached_len);
        for offset in 0..cached_len {
            let root = initial_commitment_roots
                .get(root_cursor + offset)
                .copied()
                .unwrap_or(zero_root);
            cached_for_air.push(root);
        }
        cached_roots[air_idx] = cached_for_air;
        root_cursor += cached_len;
    }

    constrain_usize_to_const(ctx, gate, invalid_air_index_count, 0);
    constrain_usize_equal(ctx, gate, root_cursor, initial_commitment_roots.len());
    (preprocessed_roots, cached_roots)
}

fn derive_preamble_observe_cells_from_stage(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    actual: &PipelineIntermediates,
    proof_shape: &AssignedProofShapeIntermediates,
    batch_and_stacked: &AssignedBatchAndStackedIntermediates,
    whir: &AssignedWhirIntermediates,
    statement_public_inputs: [AssignedValue<Fr>; 2],
    schedule: &PipelineTranscriptSchedule,
) -> Vec<AssignedValue<Fr>> {
    let gate = range.gate();
    let num_airs = schedule.air_is_required.len();
    constrain_usize_to_const(ctx, gate, schedule.air_has_preprocessed.len(), num_airs);
    constrain_usize_to_const(
        ctx,
        gate,
        schedule.air_preprocessed_commit_roots.len(),
        num_airs,
    );
    constrain_usize_to_const(ctx, gate, proof_shape.air_presence_flags.len(), num_airs);
    constrain_usize_to_const(ctx, gate, proof_shape.air_log_heights.len(), num_airs);
    constrain_usize_to_const(
        ctx,
        gate,
        actual.proof_shape.air_cached_commitment_lens.len(),
        num_airs,
    );
    constrain_usize_to_const(ctx, gate, batch_and_stacked.batch.public_values.len(), num_airs);
    constrain_bool_to_const(ctx, gate, !whir.initial_commitment_roots.is_empty(), true);

    let mut observes = Vec::new();
    let pre_hash_limbs =
        split_assigned_bn254_to_babybear_limbs(ctx, range, statement_public_inputs[0]);
    let common_main_commit_limbs =
        split_assigned_bn254_to_babybear_limbs(ctx, range, statement_public_inputs[1]);
    for limb in pre_hash_limbs {
        observes.push(limb.cell);
    }
    for limb in common_main_commit_limbs {
        observes.push(limb.cell);
    }
    let first_initial_commitment_root = whir
        .initial_commitment_roots
        .first()
        .copied()
        .unwrap_or(statement_public_inputs[1]);
    ctx.constrain_equal(&first_initial_commitment_root, &statement_public_inputs[1]);

    let (preprocessed_roots_by_air, cached_roots_by_air) = map_initial_commitment_roots_by_air(
        ctx,
        gate,
        &whir.initial_commitment_roots,
        &actual.proof_shape.trace_id_to_air_id,
        &schedule.air_has_preprocessed,
        &actual.proof_shape.air_cached_commitment_lens,
    );

    for air_idx in 0..num_airs {
        if !schedule.air_is_required[air_idx] {
            observes.push(proof_shape.air_presence_flags[air_idx]);
        }

        if actual.proof_shape.air_presence_flags[air_idx] {
            if schedule.air_has_preprocessed[air_idx] {
                constrain_bool_to_const(
                    ctx,
                    gate,
                    preprocessed_roots_by_air[air_idx].is_some(),
                    true,
                );
                constrain_bool_to_const(
                    ctx,
                    gate,
                    schedule.air_preprocessed_commit_roots[air_idx].is_some(),
                    true,
                );
                let root = preprocessed_roots_by_air[air_idx]
                    .unwrap_or_else(|| ctx.load_constant(Fr::from(0u64)));
                let expected_root =
                    schedule.air_preprocessed_commit_roots[air_idx].unwrap_or(Fr::from(0u64));
                let expected_root_cell = ctx.load_constant(expected_root);
                ctx.constrain_equal(&root, &expected_root_cell);

                let root_limbs = split_assigned_bn254_to_babybear_limbs(ctx, range, root);
                for limb in root_limbs {
                    observes.push(limb.cell);
                }
            } else {
                observes.push(proof_shape.air_log_heights[air_idx]);
            }

            for &cached_root in &cached_roots_by_air[air_idx] {
                let cached_limbs = split_assigned_bn254_to_babybear_limbs(ctx, range, cached_root);
                for limb in cached_limbs {
                    observes.push(limb.cell);
                }
            }
        }

        for pv in &batch_and_stacked.batch.public_values[air_idx] {
            observes.push(pv.cell);
        }
    }

    observes
}

fn encode_symbolic_entry(entry: Entry) -> (u64, u64, u64) {
    match entry {
        Entry::Preprocessed { offset } => (0, offset as u64, 0),
        Entry::Main { part_index, offset } => (1, part_index as u64, offset as u64),
        Entry::Permutation { offset } => (2, offset as u64, 0),
        Entry::Public => (3, 0, 0),
        Entry::Challenge => (4, 0, 0),
        Entry::Exposed => (5, 0, 0),
    }
}

fn encode_symbolic_node(node: &SymbolicExpressionNode<NativeF>) -> [u64; 7] {
    match node {
        SymbolicExpressionNode::Variable(var) => {
            let (entry_tag, arg0, arg1) = encode_symbolic_entry(var.entry);
            [0, entry_tag, var.index as u64, arg0, arg1, 0, 0]
        }
        SymbolicExpressionNode::IsFirstRow => [1, 0, 0, 0, 0, 0, 0],
        SymbolicExpressionNode::IsLastRow => [2, 0, 0, 0, 0, 0, 0],
        SymbolicExpressionNode::IsTransition => [3, 0, 0, 0, 0, 0, 0],
        SymbolicExpressionNode::Constant(value) => [4, value.as_canonical_u64(), 0, 0, 0, 0, 0],
        SymbolicExpressionNode::Add {
            left_idx,
            right_idx,
            degree_multiple,
        } => [
            5,
            *left_idx as u64,
            *right_idx as u64,
            *degree_multiple as u64,
            0,
            0,
            0,
        ],
        SymbolicExpressionNode::Sub {
            left_idx,
            right_idx,
            degree_multiple,
        } => [
            6,
            *left_idx as u64,
            *right_idx as u64,
            *degree_multiple as u64,
            0,
            0,
            0,
        ],
        SymbolicExpressionNode::Neg {
            idx,
            degree_multiple,
        } => [7, *idx as u64, *degree_multiple as u64, 0, 0, 0, 0],
        SymbolicExpressionNode::Mul {
            left_idx,
            right_idx,
            degree_multiple,
        } => [
            8,
            *left_idx as u64,
            *right_idx as u64,
            *degree_multiple as u64,
            0,
            0,
            0,
        ],
    }
}

fn constrain_symbolic_node_ownership(
    ctx: &mut Context<Fr>,
    gate: &impl GateInstructions<Fr>,
    actual_nodes: &[SymbolicExpressionNode<NativeF>],
    owned_nodes: &[SymbolicExpressionNode<NativeF>],
) {
    constrain_usize_equal(ctx, gate, actual_nodes.len(), owned_nodes.len());
    for (actual_node, owned_node) in actual_nodes.iter().zip(owned_nodes.iter()) {
        let actual_encoding = encode_symbolic_node(actual_node);
        let owned_encoding = encode_symbolic_node(owned_node);
        for (&actual_word, &owned_word) in actual_encoding.iter().zip(owned_encoding.iter()) {
            let cell = ctx.load_witness(Fr::from(actual_word));
            gate.assert_is_const(ctx, &cell, &Fr::from(owned_word));
        }
    }
}

fn constrain_metadata_ownership(
    ctx: &mut Context<Fr>,
    gate: &impl GateInstructions<Fr>,
    actual: &PipelineIntermediates,
    proof_shape: &AssignedProofShapeIntermediates,
    batch_and_stacked: &AssignedBatchAndStackedIntermediates,
    schedule: &PipelineTranscriptSchedule,
) {
    let l_skip = schedule.l_skip as isize;

    constrain_usize_equal(
        ctx,
        gate,
        actual.batch_and_stacked.batch.n_per_trace.len(),
        schedule.batch_n_per_trace.len(),
    );
    constrain_usize_equal(
        ctx,
        gate,
        actual.batch_and_stacked.batch.n_per_trace.len(),
        actual.proof_shape.trace_id_to_air_id.len(),
    );
    let default_air_log_height = ctx.load_constant(Fr::from(0u64));
    for trace_idx in 0..core::cmp::min(
        actual.batch_and_stacked.batch.n_per_trace.len(),
        actual.proof_shape.trace_id_to_air_id.len(),
    ) {
        let actual_n = actual.batch_and_stacked.batch.n_per_trace[trace_idx];
        let expected_n = schedule
            .batch_n_per_trace
            .get(trace_idx)
            .copied()
            .unwrap_or(0);

        let air_idx = actual.proof_shape.trace_id_to_air_id[trace_idx];
        let shifted = actual_n + l_skip;
        let shifted_expected = expected_n + l_skip;
        constrain_bool_to_const(ctx, gate, shifted >= 0, true);
        constrain_bool_to_const(ctx, gate, shifted_expected >= 0, true);
        let shifted_cell = ctx.load_witness(Fr::from(shifted.max(0) as u64));
        gate.assert_is_const(
            ctx,
            &shifted_cell,
            &Fr::from(shifted_expected.max(0) as u64),
        );
        constrain_bool_to_const(ctx, gate, air_idx < proof_shape.air_log_heights.len(), true);
        let air_log_height = proof_shape
            .air_log_heights
            .get(air_idx)
            .copied()
            .unwrap_or(default_air_log_height);
        ctx.constrain_equal(&shifted_cell, &air_log_height);
    }

    let batch_l_skip = ctx.load_witness(Fr::from(actual.batch_and_stacked.batch.l_skip as u64));
    gate.assert_is_const(ctx, &batch_l_skip, &Fr::from(schedule.l_skip as u64));
    gate.assert_is_const(
        ctx,
        &batch_and_stacked.batch.n_logup,
        &Fr::from(schedule.batch_n_logup as u64),
    );
    gate.assert_is_const(
        ctx,
        &batch_and_stacked.batch.n_max,
        &Fr::from(schedule.batch_n_max as u64),
    );

    constrain_usize_equal(
        ctx,
        gate,
        actual.batch_and_stacked.batch.trace_has_preprocessed.len(),
        schedule.batch_trace_has_preprocessed.len(),
    );
    for (&actual_flag, &expected_flag) in actual
        .batch_and_stacked
        .batch
        .trace_has_preprocessed
        .iter()
        .zip(&schedule.batch_trace_has_preprocessed)
    {
        let flag = ctx.load_witness(Fr::from(actual_flag as u64));
        gate.assert_is_const(ctx, &flag, &Fr::from(expected_flag as u64));
    }

    constrain_usize_equal(
        ctx,
        gate,
        actual.batch_and_stacked.batch.column_openings_need_rot.len(),
        schedule.batch_column_openings_need_rot.len(),
    );
    for (actual_row, expected_row) in actual
        .batch_and_stacked
        .batch
        .column_openings_need_rot
        .iter()
        .zip(&schedule.batch_column_openings_need_rot)
    {
        constrain_usize_equal(ctx, gate, actual_row.len(), expected_row.len());
        for (&actual_flag, &expected_flag) in actual_row.iter().zip(expected_row) {
            let flag = ctx.load_witness(Fr::from(actual_flag as u64));
            gate.assert_is_const(ctx, &flag, &Fr::from(expected_flag as u64));
        }
    }
    constrain_usize_equal(
        ctx,
        gate,
        actual.batch_and_stacked.batch.column_opening_expected_widths.len(),
        schedule.batch_column_opening_expected_widths.len(),
    );
    for (actual_row, expected_row) in actual
        .batch_and_stacked
        .batch
        .column_opening_expected_widths
        .iter()
        .zip(&schedule.batch_column_opening_expected_widths)
    {
        constrain_usize_equal(ctx, gate, actual_row.len(), expected_row.len());
        for (&actual_width, &expected_width) in actual_row.iter().zip(expected_row) {
            let width = ctx.load_witness(Fr::from(actual_width as u64));
            gate.assert_is_const(ctx, &width, &Fr::from(expected_width as u64));
        }
    }

    constrain_usize_equal(
        ctx,
        gate,
        actual.batch_and_stacked.batch.trace_constraint_nodes.len(),
        schedule.batch_trace_constraint_nodes.len(),
    );
    for (actual_nodes, expected_nodes) in actual
        .batch_and_stacked
        .batch
        .trace_constraint_nodes
        .iter()
        .zip(&schedule.batch_trace_constraint_nodes)
    {
        constrain_symbolic_node_ownership(ctx, gate, actual_nodes, expected_nodes);
    }

    constrain_usize_equal(
        ctx,
        gate,
        actual.batch_and_stacked.batch.trace_constraint_indices.len(),
        schedule.batch_trace_constraint_indices.len(),
    );
    for (actual_indices, expected_indices) in actual
        .batch_and_stacked
        .batch
        .trace_constraint_indices
        .iter()
        .zip(&schedule.batch_trace_constraint_indices)
    {
        constrain_usize_equal(ctx, gate, actual_indices.len(), expected_indices.len());
        for (&actual_idx, &expected_idx) in actual_indices.iter().zip(expected_indices) {
            let idx = ctx.load_witness(Fr::from(actual_idx as u64));
            gate.assert_is_const(ctx, &idx, &Fr::from(expected_idx as u64));
        }
    }

    constrain_usize_equal(
        ctx,
        gate,
        actual.batch_and_stacked.batch.trace_interactions.len(),
        schedule.batch_trace_interactions.len(),
    );
    for (actual_interactions, expected_interactions) in actual
        .batch_and_stacked
        .batch
        .trace_interactions
        .iter()
        .zip(&schedule.batch_trace_interactions)
    {
        constrain_usize_equal(
            ctx,
            gate,
            actual_interactions.len(),
            expected_interactions.len(),
        );
        for (actual_interaction, expected_interaction) in
            actual_interactions.iter().zip(expected_interactions)
        {
            let bus_index = ctx.load_witness(Fr::from(actual_interaction.bus_index as u64));
            gate.assert_is_const(
                ctx,
                &bus_index,
                &Fr::from(expected_interaction.bus_index as u64),
            );

            let count = ctx.load_witness(Fr::from(actual_interaction.count as u64));
            gate.assert_is_const(ctx, &count, &Fr::from(expected_interaction.count as u64));

            constrain_usize_equal(
                ctx,
                gate,
                actual_interaction.message.len(),
                expected_interaction.message.len(),
            );
            for (&actual_msg_idx, &expected_msg_idx) in actual_interaction
                .message
                .iter()
                .zip(&expected_interaction.message)
            {
                let msg_idx = ctx.load_witness(Fr::from(actual_msg_idx as u64));
                gate.assert_is_const(ctx, &msg_idx, &Fr::from(expected_msg_idx as u64));
            }
        }
    }

    let stacked_l_skip = ctx.load_witness(Fr::from(actual.batch_and_stacked.stacked_reduction.l_skip as u64));
    gate.assert_is_const(ctx, &stacked_l_skip, &Fr::from(schedule.l_skip as u64));
    constrain_usize_equal(
        ctx,
        gate,
        actual.batch_and_stacked.stacked_reduction.q_coeff_terms.len(),
        schedule.stacked_q_coeff_terms.len(),
    );
    for (actual_term, expected_term) in actual
        .batch_and_stacked
        .stacked_reduction
        .q_coeff_terms
        .iter()
        .zip(&schedule.stacked_q_coeff_terms)
    {
        let commit_idx = ctx.load_witness(Fr::from(actual_term.commit_idx as u64));
        gate.assert_is_const(ctx, &commit_idx, &Fr::from(expected_term.commit_idx as u64));

        let target_col_idx = ctx.load_witness(Fr::from(actual_term.target_col_idx as u64));
        gate.assert_is_const(
            ctx,
            &target_col_idx,
            &Fr::from(expected_term.target_col_idx as u64),
        );

        let lambda_idx = ctx.load_witness(Fr::from(actual_term.lambda_idx as u64));
        gate.assert_is_const(ctx, &lambda_idx, &Fr::from(expected_term.lambda_idx as u64));

        let need_rot = ctx.load_witness(Fr::from(actual_term.need_rot as u64));
        gate.assert_is_const(ctx, &need_rot, &Fr::from(expected_term.need_rot as u64));

        let shifted_n = actual_term.n + l_skip;
        let shifted_expected_n = expected_term.n + l_skip;
        constrain_bool_to_const(ctx, gate, shifted_n >= 0, true);
        constrain_bool_to_const(ctx, gate, shifted_expected_n >= 0, true);
        let shifted_n_cell = ctx.load_witness(Fr::from(shifted_n.max(0) as u64));
        gate.assert_is_const(
            ctx,
            &shifted_n_cell,
            &Fr::from(shifted_expected_n.max(0) as u64),
        );
        constrain_usize_equal(
            ctx,
            gate,
            actual_term.b_bits.len(),
            expected_term.b_bits.len(),
        );
        for (&actual_bit, &expected_bit) in actual_term.b_bits.iter().zip(&expected_term.b_bits) {
            let bit = ctx.load_witness(Fr::from(actual_bit as u64));
            gate.assert_is_const(ctx, &bit, &Fr::from(expected_bit as u64));
        }
    }
    constrain_usize_equal(
        ctx,
        gate,
        actual
            .batch_and_stacked
            .stacked_reduction
            .stacking_matrix_expected_widths
            .len(),
        schedule.stacked_matrix_expected_widths.len(),
    );
    for (&actual_width, &expected_width) in actual
        .batch_and_stacked
        .stacked_reduction
        .stacking_matrix_expected_widths
        .iter()
        .zip(&schedule.stacked_matrix_expected_widths)
    {
        let width = ctx.load_witness(Fr::from(actual_width as u64));
        gate.assert_is_const(ctx, &width, &Fr::from(expected_width as u64));
    }

    let k_whir = ctx.load_witness(Fr::from(actual.whir.k_whir as u64));
    gate.assert_is_const(ctx, &k_whir, &Fr::from(schedule.whir_k_whir as u64));

    let initial_log_rs_domain_size =
        ctx.load_witness(Fr::from(actual.whir.initial_log_rs_domain_size as u64));
    gate.assert_is_const(
        ctx,
        &initial_log_rs_domain_size,
        &Fr::from(schedule.whir_initial_log_rs_domain_size as u64),
    );
}

// Unchecked/internal assignment path. External callers should use strict derive+constrain APIs.
pub(crate) fn constrain_pipeline_intermediates(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    actual: &PipelineIntermediates,
    statement: &PipelineStatementWitness,
    schedule: &PipelineTranscriptSchedule,
) -> AssignedPipelineIntermediates {
    // Statement/schedule inputs are constrained in-circuit: statement public inputs are replay-owned,
    // and schedule metadata is bound through a transcript-derived public commitment.
    let gate = range.gate();
    let baby_bear = BabyBearArithmeticGadgets;
    let zero_ext = baby_bear.ext_zero(ctx, range);
    let zero_cell = ctx.load_constant(Fr::from(0u64));

    constrain_usize_equal(
        ctx,
        gate,
        actual.whir.folding_counts_per_round.len(),
        schedule.folding_counts_per_round.len(),
    );
    constrain_usize_equal(
        ctx,
        gate,
        actual.whir.query_counts_per_round.len(),
        schedule.query_counts_per_round.len(),
    );
    constrain_usize_equal(
        ctx,
        gate,
        actual.whir.query_index_bits.len(),
        schedule.query_index_bits.len(),
    );
    constrain_usize_equal(
        ctx,
        gate,
        actual.proof_shape.trace_height_thresholds.len(),
        schedule.proof_shape_ownership.trace_height_thresholds.len(),
    );

    let proof_shape = constrain_checked_proof_shape_witness_state_with_ownership(
        ctx,
        range,
        &RawProofShapeWitnessState {
            intermediates: actual.proof_shape.clone(),
        },
        Some(&schedule.proof_shape_ownership),
    )
    .assigned;
    let mut batch_and_stacked_with_owned_nodes = actual.batch_and_stacked.clone();
    batch_and_stacked_with_owned_nodes.batch.trace_constraint_nodes =
        schedule.batch_trace_constraint_nodes.clone();
    let batch_and_stacked = constrain_checked_batch_and_stacked_witness_state_unchecked(
        ctx,
        range,
        &RawBatchAndStackedWitnessState {
            intermediates: batch_and_stacked_with_owned_nodes,
        },
    )
    .assigned;
    let whir = constrain_checked_whir_witness_state_unchecked(
        ctx,
        range,
        &RawWhirWitnessState {
            intermediates: actual.whir.clone(),
        },
    )
    .assigned;
    constrain_metadata_ownership(ctx, gate, actual, &proof_shape, &batch_and_stacked, schedule);
    bind_batch_to_stacked_inputs(ctx, range, actual, &batch_and_stacked);
    bind_stacked_openings_to_whir(ctx, gate, &batch_and_stacked, &whir);
    let transcript_replay = constrain_transcript_events(ctx, range, &actual.transcript_events);

    let statement_public_inputs = [
        ctx.load_witness(statement.mvk_pre_hash),
        ctx.load_witness(statement.proof_common_main_commit),
    ];
    let preamble_observe_cells = derive_preamble_observe_cells_from_stage(
        ctx,
        range,
        actual,
        &proof_shape,
        &batch_and_stacked,
        &whir,
        statement_public_inputs,
        schedule,
    );
    let preamble_observe_count =
        ctx.load_witness(Fr::from(usize_to_u64(preamble_observe_cells.len())));
    gate.assert_is_const(
        ctx,
        &preamble_observe_count,
        &Fr::from(usize_to_u64(schedule.raw_preamble_observe_count)),
    );
    let mut event_cursor = EventCursor::new(&transcript_replay.events);
    for preamble_cell in &preamble_observe_cells {
        let observed_cell = event_cursor.consume_observe(ctx, gate);
        ctx.constrain_equal(preamble_cell, &observed_cell);
    }
    event_cursor.constrain_consumed_prefix(ctx, gate, schedule.raw_preamble_observe_count);
    constrain_post_preamble_event_kinds(
        ctx,
        gate,
        &transcript_replay.events,
        schedule.raw_preamble_observe_count,
        actual,
        schedule,
    );

    let stage_payload_observe_cells =
        derive_stage_payload_observe_cells(ctx, range, actual, &batch_and_stacked, &whir, schedule);
    let mut non_preamble_observe_cells = Vec::with_capacity(stage_payload_observe_cells.len());
    for payload_cell in stage_payload_observe_cells {
        let observed_cell = event_cursor.consume_next_observe(ctx, gate);
        non_preamble_observe_cells.push(observed_cell);
        ctx.constrain_equal(&payload_cell, &observed_cell);
    }
    let non_preamble_observe_count =
        ctx.load_witness(Fr::from(usize_to_u64(non_preamble_observe_cells.len())));
    gate.assert_is_const(
        ctx,
        &non_preamble_observe_count,
        &Fr::from(usize_to_u64(schedule.non_preamble_observe_count)),
    );
    let total_observe_count =
        ctx.load_witness(Fr::from(usize_to_u64(transcript_replay.observes.len())));
    gate.assert_is_const(
        ctx,
        &total_observe_count,
        &Fr::from(usize_to_u64(
            preamble_observe_cells.len() + non_preamble_observe_cells.len(),
        )),
    );

    let proof_shape_l_skip = ctx.load_witness(Fr::from(actual.proof_shape.l_skip as u64));
    gate.assert_is_const(ctx, &proof_shape_l_skip, &Fr::from(schedule.l_skip as u64));
    gate.assert_is_const(
        ctx,
        &proof_shape.max_log_height_allowed,
        &Fr::from(schedule.proof_shape_max_log_height_allowed as u64),
    );
    constrain_usize_equal(
        ctx,
        gate,
        proof_shape.air_required_flags.len(),
        schedule.air_is_required.len(),
    );
    constrain_usize_equal(
        ctx,
        gate,
        proof_shape.air_expected_public_value_lens.len(),
        schedule.air_num_public_values.len(),
    );
    constrain_usize_equal(
        ctx,
        gate,
        proof_shape.air_expected_cached_commitment_lens.len(),
        schedule.air_num_cached_mains.len(),
    );
    constrain_usize_equal(
        ctx,
        gate,
        actual.proof_shape.air_public_value_lens.len(),
        schedule.air_is_required.len(),
    );
    constrain_usize_equal(
        ctx,
        gate,
        batch_and_stacked.batch.public_values.len(),
        schedule.air_is_required.len(),
    );
    for air_idx in 0..schedule.air_is_required.len() {
        let required_flag = proof_shape
            .air_required_flags
            .get(air_idx)
            .copied()
            .unwrap_or(zero_cell);
        gate.assert_is_const(
            ctx,
            &required_flag,
            &Fr::from(schedule.air_is_required[air_idx] as u64),
        );

        let has_vdata = actual
            .proof_shape
            .air_presence_flags
            .get(air_idx)
            .copied()
            .unwrap_or(false);
        let expected_public_values = if has_vdata {
            schedule.air_num_public_values[air_idx]
        } else {
            0
        };
        let expected_public_len = proof_shape
            .air_expected_public_value_lens
            .get(air_idx)
            .copied()
            .unwrap_or(zero_cell);
        gate.assert_is_const(
            ctx,
            &expected_public_len,
            &Fr::from(expected_public_values as u64),
        );
        let expected_cached_commitments = if has_vdata {
            schedule.air_num_cached_mains[air_idx]
        } else {
            0
        };
        let expected_cached_len = proof_shape
            .air_expected_cached_commitment_lens
            .get(air_idx)
            .copied()
            .unwrap_or(zero_cell);
        gate.assert_is_const(
            ctx,
            &expected_cached_len,
            &Fr::from(expected_cached_commitments as u64),
        );
        constrain_usize_equal(
            ctx,
            gate,
            batch_and_stacked.batch
                .public_values
                .get(air_idx)
                .map_or(0usize, |row| row.len()),
            actual
                .proof_shape
                .air_public_value_lens
                .get(air_idx)
                .copied()
                .unwrap_or(0),
        );
    }
    for (&actual_threshold, &schedule_threshold) in actual
        .proof_shape
        .trace_height_thresholds
        .iter()
        .zip(&schedule.proof_shape_ownership.trace_height_thresholds)
    {
        let threshold = ctx.load_witness(Fr::from(actual_threshold));
        gate.assert_is_const(ctx, &threshold, &Fr::from(schedule_threshold));
    }
    gate.assert_is_const(
        ctx,
        &batch_and_stacked.batch.logup_pow_bits,
        &Fr::from(schedule.logup_pow_bits as u64),
    );
    gate.assert_is_const(
        ctx,
        &batch_and_stacked.batch.total_interactions,
        &Fr::from(schedule.batch_total_interactions),
    );
    let expected_gkr_rounds = if schedule.has_gkr_observe_payload {
        schedule.l_skip + schedule.batch_n_logup
    } else {
        0
    };
    constrain_usize_to_const(
        ctx,
        gate,
        batch_and_stacked.batch.gkr_claims_per_layer.len(),
        expected_gkr_rounds,
    );
    constrain_usize_to_const(
        ctx,
        gate,
        batch_and_stacked.batch.gkr_sumcheck_polys.len(),
        expected_gkr_rounds.saturating_sub(1),
    );
    for claims in &batch_and_stacked.batch.gkr_claims_per_layer {
        constrain_usize_to_const(ctx, gate, claims.len(), 4);
    }
    for (round_idx, evals) in batch_and_stacked.batch.gkr_sumcheck_polys.iter().enumerate() {
        constrain_usize_to_const(ctx, gate, evals.len(), (round_idx + 1) * 3);
    }
    gate.assert_is_const(
        ctx,
        &batch_and_stacked.batch.batch_degree,
        &Fr::from(schedule.batch_degree as u64),
    );
    constrain_usize_equal(
        ctx,
        gate,
        batch_and_stacked.batch.numerator_term_per_air.len(),
        batch_and_stacked.batch.denominator_term_per_air.len(),
    );
    constrain_usize_equal(
        ctx,
        gate,
        batch_and_stacked.batch.numerator_term_per_air.len(),
        schedule.batch_n_per_trace.len(),
    );
    constrain_usize_equal(
        ctx,
        gate,
        batch_and_stacked.batch.univariate_round_coeffs.len(),
        schedule.batch_univariate_coeffs_len,
    );
    constrain_usize_equal(
        ctx,
        gate,
        batch_and_stacked.batch.sumcheck_round_polys.len(),
        schedule.batch_n_max,
    );
    for round in &batch_and_stacked.batch.sumcheck_round_polys {
        constrain_usize_to_const(ctx, gate, round.len(), schedule.batch_degree);
    }
    gate.assert_is_const(ctx, &whir.mu_pow_bits, &Fr::from(schedule.mu_pow_bits as u64));
    gate.assert_is_const(
        ctx,
        &whir.folding_pow_bits,
        &Fr::from(schedule.folding_pow_bits as u64),
    );
    gate.assert_is_const(
        ctx,
        &whir.query_phase_pow_bits,
        &Fr::from(schedule.query_phase_pow_bits as u64),
    );
    gate.assert_is_const(
        ctx,
        &whir.final_poly_len,
        &Fr::from(schedule.whir_expected_final_poly_len as u64),
    );
    let whir_expected_final_poly_len =
        ctx.load_witness(Fr::from(actual.whir.expected_final_poly_len as u64));
    gate.assert_is_const(
        ctx,
        &whir_expected_final_poly_len,
        &Fr::from(schedule.whir_expected_final_poly_len as u64),
    );

    for (&actual_count, &expected_count) in actual
        .whir
        .folding_counts_per_round
        .iter()
        .zip(&schedule.folding_counts_per_round)
    {
        let count = ctx.load_witness(Fr::from(actual_count as u64));
        gate.assert_is_const(ctx, &count, &Fr::from(expected_count as u64));
    }
    for (&actual_count, &expected_count) in actual
        .whir
        .query_counts_per_round
        .iter()
        .zip(&schedule.query_counts_per_round)
    {
        let count = ctx.load_witness(Fr::from(actual_count as u64));
        gate.assert_is_const(ctx, &count, &Fr::from(expected_count as u64));
    }
    for (&actual_bits, &expected_bits) in actual
        .whir
        .query_index_bits
        .iter()
        .zip(&schedule.query_index_bits)
    {
        let bits = ctx.load_witness(Fr::from(actual_bits as u64));
        gate.assert_is_const(ctx, &bits, &Fr::from(expected_bits as u64));
    }
    constrain_usize_equal(
        ctx,
        gate,
        whir.folding_alphas.len(),
        whir.folding_pow_sampled_bits.len(),
    );
    constrain_usize_equal(
        ctx,
        gate,
        whir.z0_challenges.len(),
        schedule.query_counts_per_round.len().saturating_sub(1),
    );
    constrain_usize_equal(
        ctx,
        gate,
        whir.gammas.len(),
        schedule.query_counts_per_round.len(),
    );
    constrain_usize_equal(
        ctx,
        gate,
        batch_and_stacked.batch.trace_id_to_air_id.len(),
        proof_shape.trace_id_to_air_id.len(),
    );
    for (batch_air, proof_shape_air) in batch_and_stacked
        .batch
        .trace_id_to_air_id
        .iter()
        .zip(&proof_shape.trace_id_to_air_id)
    {
        ctx.constrain_equal(batch_air, proof_shape_air);
    }

    let mut sample_cursor = SampleCursor::new(&transcript_replay.samples);

    bind_sample_bits(
        ctx,
        range,
        &mut sample_cursor,
        schedule.logup_pow_bits,
        batch_and_stacked.batch.logup_pow_sampled_bits,
        false,
    );
    bind_sample_ext(ctx, &mut sample_cursor, &batch_and_stacked.batch.alpha_logup);
    bind_sample_ext(ctx, &mut sample_cursor, &batch_and_stacked.batch.beta_logup);
    for gkr_challenge in &batch_and_stacked.batch.gkr_non_xi_samples {
        bind_sample_ext(ctx, &mut sample_cursor, gkr_challenge);
    }
    for gkr_xi_challenge in &batch_and_stacked.batch.gkr_xi_sample_order {
        bind_sample_ext(ctx, &mut sample_cursor, gkr_xi_challenge);
    }
    let gkr_xi_len_cell =
        assign_and_range_u64(ctx, range, usize_to_u64(batch_and_stacked.batch.gkr_xi_sample_order.len()));
    range.check_less_than_safe(
        ctx,
        gkr_xi_len_cell,
        usize_to_u64(batch_and_stacked.batch.xi.len()).saturating_add(1),
    );
    let gkr_xi_len = batch_and_stacked.batch.gkr_xi_sample_order.len();
    if gkr_xi_len > 0 {
        let xi_0 = batch_and_stacked.batch.xi.first().unwrap_or(&zero_ext);
        let gkr_last = batch_and_stacked
            .batch
            .gkr_xi_sample_order
            .get(gkr_xi_len - 1)
            .unwrap_or(&zero_ext);
        baby_bear.assert_ext_equal(ctx, xi_0, gkr_last);
        for idx in 1..gkr_xi_len {
            let xi_val = batch_and_stacked.batch.xi.get(idx).unwrap_or(&zero_ext);
            let gkr_prev = batch_and_stacked
                .batch
                .gkr_xi_sample_order
                .get(idx - 1)
                .unwrap_or(&zero_ext);
            baby_bear.assert_ext_equal(ctx, xi_val, gkr_prev);
        }
    }
    for xi_challenge in batch_and_stacked.batch.xi.iter().skip(gkr_xi_len) {
        bind_sample_ext(ctx, &mut sample_cursor, xi_challenge);
    }
    bind_sample_ext(ctx, &mut sample_cursor, &batch_and_stacked.batch.lambda);
    bind_sample_ext(ctx, &mut sample_cursor, &batch_and_stacked.batch.mu);

    for r_challenge in &batch_and_stacked.batch.r {
        bind_sample_ext(ctx, &mut sample_cursor, r_challenge);
    }

    bind_sample_ext(ctx, &mut sample_cursor, &batch_and_stacked.stacked_reduction.lambda);
    for u_challenge in &batch_and_stacked.stacked_reduction.u {
        bind_sample_ext(ctx, &mut sample_cursor, u_challenge);
    }

    bind_sample_bits(
        ctx,
        range,
        &mut sample_cursor,
        schedule.mu_pow_bits,
        whir.mu_pow_sampled_bits,
        false,
    );
    bind_sample_ext(ctx, &mut sample_cursor, &whir.mu_challenge);

    let mut folding_idx = 0usize;
    let mut query_idx = 0usize;
    let mut z0_idx = 0usize;
    let mut gamma_idx = 0usize;
    for round_idx in 0..schedule.query_counts_per_round.len() {
        let folding_count = schedule
            .folding_counts_per_round
            .get(round_idx)
            .copied()
            .unwrap_or(0);
        for _ in 0..folding_count {
            let folding_pow_bits = whir
                .folding_pow_sampled_bits
                .get(folding_idx)
                .copied()
                .unwrap_or(zero_cell);
            bind_sample_bits(
                ctx,
                range,
                &mut sample_cursor,
                schedule.folding_pow_bits,
                folding_pow_bits,
                false,
            );
            let folding_alpha = whir.folding_alphas.get(folding_idx).unwrap_or(&zero_ext);
            bind_sample_ext(ctx, &mut sample_cursor, folding_alpha);
            folding_idx += 1;
        }

        if round_idx + 1 < schedule.query_counts_per_round.len() {
            let z0 = whir.z0_challenges.get(z0_idx).unwrap_or(&zero_ext);
            bind_sample_ext(ctx, &mut sample_cursor, z0);
            z0_idx += 1;
        }

        let query_phase_pow_bits = whir
            .query_phase_pow_sampled_bits
            .get(round_idx)
            .copied()
            .unwrap_or(zero_cell);
        bind_sample_bits(
            ctx,
            range,
            &mut sample_cursor,
            schedule.query_phase_pow_bits,
            query_phase_pow_bits,
            false,
        );

        let query_count = schedule
            .query_counts_per_round
            .get(round_idx)
            .copied()
            .unwrap_or(0);
        for _ in 0..query_count {
            let query_index = whir
                .query_indices
                .get(query_idx)
                .copied()
                .unwrap_or(zero_cell);
            let query_bits = schedule
                .query_index_bits
                .get(query_idx)
                .copied()
                .unwrap_or(0);
            bind_sample_bits(
                ctx,
                range,
                &mut sample_cursor,
                query_bits,
                query_index,
                true,
            );
            query_idx += 1;
        }

        let gamma = whir.gammas.get(gamma_idx).unwrap_or(&zero_ext);
        bind_sample_ext(ctx, &mut sample_cursor, gamma);
        gamma_idx += 1;
    }

    gate.assert_is_const(ctx, &batch_and_stacked.batch.logup_pow_witness_ok, &Fr::from(1u64));
    gate.assert_is_const(ctx, &whir.mu_pow_witness_ok, &Fr::from(1u64));

    let folding_bound = ctx.load_witness(Fr::from(usize_to_u64(folding_idx)));
    gate.assert_is_const(
        ctx,
        &folding_bound,
        &Fr::from(usize_to_u64(whir.folding_pow_sampled_bits.len())),
    );
    let query_bound = ctx.load_witness(Fr::from(usize_to_u64(query_idx)));
    gate.assert_is_const(
        ctx,
        &query_bound,
        &Fr::from(usize_to_u64(whir.query_indices.len())),
    );
    let z0_bound = ctx.load_witness(Fr::from(usize_to_u64(z0_idx)));
    gate.assert_is_const(
        ctx,
        &z0_bound,
        &Fr::from(usize_to_u64(whir.z0_challenges.len())),
    );
    let gamma_bound = ctx.load_witness(Fr::from(usize_to_u64(gamma_idx)));
    gate.assert_is_const(ctx, &gamma_bound, &Fr::from(usize_to_u64(whir.gammas.len())));
    let consumed_samples = ctx.load_witness(Fr::from(usize_to_u64(sample_cursor.cursor)));
    gate.assert_is_const(
        ctx,
        &consumed_samples,
        &Fr::from(usize_to_u64(transcript_replay.samples.len())),
    );

    bind_u_cube_from_stacked(
        ctx,
        range,
        schedule.l_skip,
        &batch_and_stacked.stacked_reduction.u,
        &whir.u_cube,
    );

    AssignedPipelineIntermediates {
        proof_shape,
        batch_and_stacked,
        whir,
        statement_public_inputs,
    }
}

// Unchecked/internal assignment path. External callers should use strict derive+constrain APIs.
pub(crate) fn constrain_checked_pipeline_witness_state(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    raw: &RawPipelineWitnessState,
) -> CheckedPipelineWitnessState {
    let assigned = constrain_pipeline_intermediates(
        ctx,
        range,
        &raw.intermediates,
        &raw.statement,
        &raw.schedule,
    );
    let derived = DerivedPipelineState {
        consumed_non_preamble_observes: derive_non_preamble_observes(
            &raw.intermediates.transcript_events,
            raw.schedule.raw_preamble_observe_count,
        )
        .len(),
    };
    CheckedPipelineWitnessState { assigned, derived }
}

pub fn derive_and_constrain_pipeline(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    config: &NativeConfig,
    mvk: &MultiStarkVerifyingKey<NativeConfig>,
    proof: &Proof<NativeConfig>,
) -> Result<AssignedPipelineIntermediates, PipelineError> {
    let raw = derive_raw_pipeline_witness_state(config, mvk, proof)?;
    Ok(constrain_checked_pipeline_witness_state(ctx, range, &raw).assigned)
}

#[cfg(test)]
mod tests;
