use core::cmp::{Reverse, max};

use halo2_base::{
    AssignedValue, Context,
    gates::{GateInstructions, RangeInstructions, range::RangeChip},
};
use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2Config as NativeConfig,
    openvm_stark_backend::{
        StarkProtocolConfig, calculate_n_logup,
        keygen::types::{MultiStarkVerifyingKey, MultiStarkVerifyingKey0},
        proof::Proof,
        prover::stacked_pcs::StackedLayout,
        verifier::proof_shape::{
            BatchProofShapeError, GkrProofShapeError, ProofShapeError, ProofShapeVDataError,
            StackingProofShapeError, WhirProofShapeError,
        },
    },
};

use crate::{
    circuit::Fr,
    utils::{assign_and_range_u64, assign_and_range_usize, bits_for_u64, usize_to_u64},
};

#[derive(Debug, PartialEq, Eq)]
pub enum ProofShapePreambleError {
    SystemParamsMismatch,
    TraceHeightsTooLarge {
        constraint_idx: usize,
        sum: u64,
        threshold: u64,
    },
    ProofShape(ProofShapeError),
}

impl From<ProofShapeError> for ProofShapePreambleError {
    fn from(value: ProofShapeError) -> Self {
        Self::ProofShape(value)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProofShapeIntermediates {
    pub num_airs: usize,
    pub num_traces: usize,
    pub trace_id_to_air_id: Vec<usize>,
    pub trace_height_sums: Vec<u64>,
    pub trace_height_coefficients: Vec<Vec<u64>>,
    pub trace_height_thresholds: Vec<u64>,
    pub air_presence_flags: Vec<bool>,
    pub air_required_flags: Vec<bool>,
    pub air_public_value_lens: Vec<usize>,
    pub air_expected_public_value_lens: Vec<usize>,
    pub air_cached_commitment_lens: Vec<usize>,
    pub air_expected_cached_commitment_lens: Vec<usize>,
    pub air_log_heights: Vec<usize>,
    pub max_log_height_allowed: usize,
    pub l_skip: usize,
    pub num_airs_present: usize,
    pub proof_shape_count_checks: Vec<(usize, usize)>,
    pub proof_shape_upper_bound_checks: Vec<(usize, usize)>,
}

#[derive(Clone, Debug)]
pub struct AssignedProofShapeIntermediates {
    pub num_airs: AssignedValue<Fr>,
    pub num_traces: AssignedValue<Fr>,
    pub trace_id_to_air_id: Vec<AssignedValue<Fr>>,
    pub trace_height_sums: Vec<AssignedValue<Fr>>,
    pub air_presence_flags: Vec<AssignedValue<Fr>>,
    pub air_required_flags: Vec<AssignedValue<Fr>>,
    pub air_public_value_lens: Vec<AssignedValue<Fr>>,
    pub air_expected_public_value_lens: Vec<AssignedValue<Fr>>,
    pub air_cached_commitment_lens: Vec<AssignedValue<Fr>>,
    pub air_expected_cached_commitment_lens: Vec<AssignedValue<Fr>>,
    pub air_log_heights: Vec<AssignedValue<Fr>>,
    pub max_log_height_allowed: AssignedValue<Fr>,
    pub num_airs_present: AssignedValue<Fr>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawProofShapeWitnessState {
    pub intermediates: ProofShapeIntermediates,
}

#[derive(Clone, Debug)]
pub struct DerivedProofShapeState {
    pub trace_height_sums: Vec<AssignedValue<Fr>>,
}

#[derive(Clone, Debug)]
pub struct CheckedProofShapeWitnessState {
    pub assigned: AssignedProofShapeIntermediates,
    pub derived: DerivedProofShapeState,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProofShapeOwnershipSchedule {
    pub trace_height_coefficients: Vec<Vec<u64>>,
    pub trace_height_thresholds: Vec<u64>,
    pub proof_shape_count_checks: Vec<(usize, usize)>,
    pub proof_shape_upper_bound_checks: Vec<(usize, usize)>,
}

fn compute_trace_id_to_air_id(
    mvk0: &MultiStarkVerifyingKey0<NativeConfig>,
    proof: &Proof<NativeConfig>,
) -> Vec<usize> {
    let num_airs = mvk0.per_air.len();
    let mut trace_id_to_air_id: Vec<usize> = (0..num_airs).collect();
    trace_id_to_air_id.sort_by_key(|&air_id| {
        (
            proof.trace_vdata[air_id].is_none(),
            proof.trace_vdata[air_id]
                .as_ref()
                .map(|vdata| Reverse(vdata.log_height)),
            air_id,
        )
    });

    let num_traces = proof.trace_vdata.iter().flatten().count();
    trace_id_to_air_id.truncate(num_traces);
    trace_id_to_air_id
}

#[derive(Clone, Debug)]
pub(crate) struct ProofShapeRuleDerivation {
    pub layouts: Vec<StackedLayout>,
    pub air_required_flags: Vec<bool>,
    pub air_public_value_lens: Vec<usize>,
    pub air_expected_public_value_lens: Vec<usize>,
    pub air_cached_commitment_lens: Vec<usize>,
    pub air_expected_cached_commitment_lens: Vec<usize>,
    pub air_log_heights: Vec<usize>,
    pub max_log_height_allowed: usize,
    pub count_checks: Vec<(usize, usize)>,
    pub upper_bound_checks: Vec<(usize, usize)>,
}

pub(crate) fn derive_proof_shape_rules(
    mvk0: &MultiStarkVerifyingKey0<NativeConfig>,
    proof: &Proof<NativeConfig>,
) -> Result<ProofShapeRuleDerivation, ProofShapeError> {
    let num_airs = mvk0.per_air.len();
    let l_skip = mvk0.params.l_skip;
    let max_log_height_allowed = l_skip + mvk0.params.n_stack;

    let mut count_checks = Vec::new();
    let mut upper_bound_checks = Vec::new();

    count_checks.push((proof.trace_vdata.len(), num_airs));
    if proof.trace_vdata.len() != num_airs {
        return Err(ProofShapeError::InvalidVData(
            ProofShapeVDataError::InvalidVDataLength {
                len: proof.trace_vdata.len(),
                num_airs,
            },
        ));
    }
    count_checks.push((proof.public_values.len(), num_airs));
    if proof.public_values.len() != num_airs {
        return Err(ProofShapeError::InvalidVData(
            ProofShapeVDataError::InvalidPublicValuesLength {
                len: proof.public_values.len(),
                num_airs,
            },
        ));
    }

    let mut air_required_flags = Vec::with_capacity(num_airs);
    let mut air_public_value_lens = Vec::with_capacity(num_airs);
    let mut air_expected_public_value_lens = Vec::with_capacity(num_airs);
    let mut air_cached_commitment_lens = Vec::with_capacity(num_airs);
    let mut air_expected_cached_commitment_lens = Vec::with_capacity(num_airs);
    let mut air_log_heights = Vec::with_capacity(num_airs);

    for (air_idx, ((vk, vdata), pvs)) in mvk0
        .per_air
        .iter()
        .zip(&proof.trace_vdata)
        .zip(&proof.public_values)
        .enumerate()
    {
        let has_vdata = vdata.is_some();
        let required = vk.is_required;
        let actual_cached_commitments = vdata
            .as_ref()
            .map(|trace_vdata| trace_vdata.cached_commitments.len())
            .unwrap_or(0);
        let expected_cached_commitments = if has_vdata { vk.num_cached_mains() } else { 0 };
        let actual_public_values = pvs.len();
        let expected_public_values = if has_vdata {
            vk.params.num_public_values
        } else {
            0
        };
        let log_height = vdata
            .as_ref()
            .map(|trace_vdata| trace_vdata.log_height)
            .unwrap_or(0);

        air_required_flags.push(required);
        air_public_value_lens.push(actual_public_values);
        air_expected_public_value_lens.push(expected_public_values);
        air_cached_commitment_lens.push(actual_cached_commitments);
        air_expected_cached_commitment_lens.push(expected_cached_commitments);
        air_log_heights.push(log_height);
        upper_bound_checks.push((log_height, max_log_height_allowed));

        if !has_vdata {
            if required {
                return Err(ProofShapeError::InvalidVData(
                    ProofShapeVDataError::RequiredAirNoVData { air_idx },
                ));
            }
            if !pvs.is_empty() {
                return Err(ProofShapeError::InvalidVData(
                    ProofShapeVDataError::PublicValuesNoVData { air_idx },
                ));
            }
        } else if actual_cached_commitments != vk.num_cached_mains() {
            return Err(ProofShapeError::InvalidVData(
                ProofShapeVDataError::InvalidCachedCommitments {
                    air_idx,
                    expected: vk.num_cached_mains(),
                    actual: actual_cached_commitments,
                },
            ));
        } else if log_height > max_log_height_allowed {
            return Err(ProofShapeError::InvalidVData(
                ProofShapeVDataError::LogHeightOutOfBounds {
                    air_idx,
                    l_skip,
                    n_stack: mvk0.params.n_stack,
                    actual: log_height,
                },
            ));
        } else if vk.params.num_public_values != actual_public_values {
            return Err(ProofShapeError::InvalidVData(
                ProofShapeVDataError::InvalidPublicValues {
                    air_idx,
                    expected: vk.params.num_public_values,
                    actual: actual_public_values,
                },
            ));
        }
    }

    let mut per_trace = mvk0
        .per_air
        .iter()
        .zip(&proof.trace_vdata)
        .enumerate()
        .filter_map(|(air_idx, (vk, vdata))| vdata.as_ref().map(|vdata| (air_idx, vk, vdata)))
        .collect::<Vec<_>>();
    per_trace.sort_by_key(|(_, _, vdata)| Reverse(vdata.log_height));
    let num_airs_present = per_trace.len();

    let total_interactions = per_trace.iter().fold(0u64, |acc, (_, vk, vdata)| {
        acc + ((vk.num_interactions() as u64) << max(vdata.log_height, l_skip))
    });
    let n_logup = calculate_n_logup(l_skip, total_interactions);
    let num_gkr_rounds = if total_interactions == 0 {
        0
    } else {
        l_skip + n_logup
    };

    count_checks.push((proof.gkr_proof.claims_per_layer.len(), num_gkr_rounds));
    if proof.gkr_proof.claims_per_layer.len() != num_gkr_rounds {
        return Err(ProofShapeError::InvalidGkrProofShape(
            GkrProofShapeError::InvalidClaimsPerLayer {
                expected: num_gkr_rounds,
                actual: proof.gkr_proof.claims_per_layer.len(),
            },
        ));
    }
    let expected_gkr_sumcheck_polys = num_gkr_rounds.saturating_sub(1);
    count_checks.push((
        proof.gkr_proof.sumcheck_polys.len(),
        expected_gkr_sumcheck_polys,
    ));
    if proof.gkr_proof.sumcheck_polys.len() != expected_gkr_sumcheck_polys {
        return Err(ProofShapeError::InvalidGkrProofShape(
            GkrProofShapeError::InvalidSumcheckPolys {
                expected: expected_gkr_sumcheck_polys,
                actual: proof.gkr_proof.sumcheck_polys.len(),
            },
        ));
    }
    for (i, poly) in proof.gkr_proof.sumcheck_polys.iter().enumerate() {
        let expected = i + 1;
        count_checks.push((poly.len(), expected));
        if poly.len() != expected {
            return Err(ProofShapeError::InvalidGkrProofShape(
                GkrProofShapeError::InvalidSumcheckPolyEvals {
                    round: expected,
                    expected,
                    actual: poly.len(),
                },
            ));
        }
    }

    let batch_proof = &proof.batch_constraint_proof;
    if per_trace.is_empty() {
        return Err(ProofShapeError::InvalidBatchConstraintProofShape(
            BatchProofShapeError::InvalidColumnOpeningsAirs {
                expected: 1,
                actual: batch_proof.column_openings.len(),
            },
        ));
    }
    let n_max = per_trace[0].2.log_height.saturating_sub(l_skip);
    let s_0_deg = (mvk0.max_constraint_degree() + 1) * ((1usize << l_skip) - 1);
    let batch_degree = mvk0.max_constraint_degree() + 1;

    count_checks.push((batch_proof.numerator_term_per_air.len(), num_airs_present));
    if batch_proof.numerator_term_per_air.len() != num_airs_present {
        return Err(ProofShapeError::InvalidBatchConstraintProofShape(
            BatchProofShapeError::InvalidNumeratorTerms {
                expected: num_airs_present,
                actual: batch_proof.numerator_term_per_air.len(),
            },
        ));
    }
    count_checks.push((batch_proof.denominator_term_per_air.len(), num_airs_present));
    if batch_proof.denominator_term_per_air.len() != num_airs_present {
        return Err(ProofShapeError::InvalidBatchConstraintProofShape(
            BatchProofShapeError::InvalidDenominatorTerms {
                expected: num_airs_present,
                actual: batch_proof.denominator_term_per_air.len(),
            },
        ));
    }
    count_checks.push((batch_proof.univariate_round_coeffs.len(), s_0_deg + 1));
    if batch_proof.univariate_round_coeffs.len() != s_0_deg + 1 {
        return Err(ProofShapeError::InvalidBatchConstraintProofShape(
            BatchProofShapeError::InvalidUnivariateRoundCoeffs {
                expected: s_0_deg + 1,
                actual: batch_proof.univariate_round_coeffs.len(),
            },
        ));
    }
    count_checks.push((batch_proof.sumcheck_round_polys.len(), n_max));
    if batch_proof.sumcheck_round_polys.len() != n_max {
        return Err(ProofShapeError::InvalidBatchConstraintProofShape(
            BatchProofShapeError::InvalidSumcheckRoundPolys {
                expected: n_max,
                actual: batch_proof.sumcheck_round_polys.len(),
            },
        ));
    }
    count_checks.push((batch_proof.column_openings.len(), num_airs_present));
    if batch_proof.column_openings.len() != num_airs_present {
        return Err(ProofShapeError::InvalidBatchConstraintProofShape(
            BatchProofShapeError::InvalidColumnOpeningsAirs {
                expected: num_airs_present,
                actual: batch_proof.column_openings.len(),
            },
        ));
    }

    for (i, evals) in batch_proof.sumcheck_round_polys.iter().enumerate() {
        count_checks.push((evals.len(), batch_degree));
        if evals.len() != batch_degree {
            return Err(ProofShapeError::InvalidBatchConstraintProofShape(
                BatchProofShapeError::InvalidSumcheckRoundPolyEvals {
                    round: i,
                    expected: batch_degree,
                    actual: evals.len(),
                },
            ));
        }
    }

    for (part_openings, &(air_idx, vk, _)) in batch_proof.column_openings.iter().zip(&per_trace) {
        let need_rot = mvk0.per_air[air_idx].params.need_rot;
        let openings_per_col = if need_rot { 2 } else { 1 };
        count_checks.push((part_openings.len(), vk.num_parts()));
        if part_openings.len() != vk.num_parts() {
            return Err(ProofShapeError::InvalidBatchConstraintProofShape(
                BatchProofShapeError::InvalidColumnOpeningsPerAir {
                    air_idx,
                    expected: vk.num_parts(),
                    actual: part_openings.len(),
                },
            ));
        }

        let expected_main_openings = vk.params.width.common_main * openings_per_col;
        let actual_main_openings = part_openings[0].len();
        count_checks.push((actual_main_openings, expected_main_openings));
        if actual_main_openings != expected_main_openings {
            return Err(ProofShapeError::InvalidBatchConstraintProofShape(
                BatchProofShapeError::InvalidColumnOpeningsPerAirMain {
                    air_idx,
                    expected: vk.params.width.common_main,
                    actual: actual_main_openings,
                },
            ));
        }

        if let Some(preprocessed_width) = vk.params.width.preprocessed {
            let expected_preprocessed_openings = preprocessed_width * openings_per_col;
            let actual_preprocessed_openings = part_openings[1].len();
            count_checks.push((actual_preprocessed_openings, expected_preprocessed_openings));
            if actual_preprocessed_openings != expected_preprocessed_openings {
                return Err(ProofShapeError::InvalidBatchConstraintProofShape(
                    BatchProofShapeError::InvalidColumnOpeningsPerAirPreprocessed {
                        air_idx,
                        expected: preprocessed_width,
                        actual: actual_preprocessed_openings,
                    },
                ));
            }
        }

        let cached_openings = &part_openings[1 + (vk.preprocessed_data.is_some() as usize)..];
        for (cached_idx, (col_opening, &width)) in cached_openings
            .iter()
            .zip(&vk.params.width.cached_mains)
            .enumerate()
        {
            let expected_cached_openings = width * openings_per_col;
            count_checks.push((col_opening.len(), expected_cached_openings));
            if col_opening.len() != expected_cached_openings {
                return Err(ProofShapeError::InvalidBatchConstraintProofShape(
                    BatchProofShapeError::InvalidColumnOpeningsPerAirCached {
                        air_idx,
                        cached_idx,
                        expected: width,
                        actual: col_opening.len(),
                    },
                ));
            }
        }
    }

    let stacking_proof = &proof.stacking_proof;
    let stacking_s0_deg = 2 * ((1usize << l_skip) - 1);
    count_checks.push((
        stacking_proof.univariate_round_coeffs.len(),
        stacking_s0_deg + 1,
    ));
    if stacking_proof.univariate_round_coeffs.len() != stacking_s0_deg + 1 {
        return Err(ProofShapeError::InvalidStackingProofShape(
            StackingProofShapeError::InvalidUnivariateRoundCoeffs {
                expected: stacking_s0_deg + 1,
                actual: stacking_proof.univariate_round_coeffs.len(),
            },
        ));
    }
    count_checks.push((
        stacking_proof.sumcheck_round_polys.len(),
        mvk0.params.n_stack,
    ));
    if stacking_proof.sumcheck_round_polys.len() != mvk0.params.n_stack {
        return Err(ProofShapeError::InvalidStackingProofShape(
            StackingProofShapeError::InvalidSumcheckRoundPolys {
                expected: mvk0.params.n_stack,
                actual: stacking_proof.sumcheck_round_polys.len(),
            },
        ));
    }

    let common_main_layout = StackedLayout::new(
        l_skip,
        mvk0.params.n_stack + l_skip,
        per_trace
            .iter()
            .map(|(_, vk, vdata)| (vk.params.width.common_main, vdata.log_height))
            .collect::<Vec<_>>(),
    );
    let other_layouts = per_trace
        .iter()
        .flat_map(|(_, vk, vdata)| {
            vk.params
                .width
                .preprocessed
                .iter()
                .chain(&vk.params.width.cached_mains)
                .copied()
                .map(|width| (width, vdata.log_height))
                .collect::<Vec<_>>()
        })
        .map(|sorted| StackedLayout::new(l_skip, mvk0.params.n_stack + l_skip, vec![sorted]))
        .collect::<Vec<_>>();
    let layouts = core::iter::once(common_main_layout)
        .chain(other_layouts)
        .collect::<Vec<_>>();

    count_checks.push((stacking_proof.stacking_openings.len(), layouts.len()));
    if stacking_proof.stacking_openings.len() != layouts.len() {
        return Err(ProofShapeError::InvalidStackingProofShape(
            StackingProofShapeError::InvalidStackOpenings {
                expected: layouts.len(),
                actual: stacking_proof.stacking_openings.len(),
            },
        ));
    }
    for (commit_idx, (openings, layout)) in stacking_proof
        .stacking_openings
        .iter()
        .zip(&layouts)
        .enumerate()
    {
        let stacked_matrix_width = match layout.sorted_cols.last() {
            Some((_, _, s)) => s.col_idx + 1,
            None => {
                return Err(ProofShapeError::InvalidStackingProofShape(
                    StackingProofShapeError::InvalidStackOpeningsPerMatrix {
                        commit_idx,
                        expected: 1,
                        actual: openings.len(),
                    },
                ));
            }
        };
        count_checks.push((openings.len(), stacked_matrix_width));
        if openings.len() != stacked_matrix_width {
            return Err(ProofShapeError::InvalidStackingProofShape(
                StackingProofShapeError::InvalidStackOpeningsPerMatrix {
                    commit_idx,
                    expected: stacked_matrix_width,
                    actual: openings.len(),
                },
            ));
        }
    }

    let whir_proof = &proof.whir_proof;
    let log_stacked_height = mvk0.params.log_stacked_height();
    let num_whir_rounds = mvk0.params.num_whir_rounds();
    let num_whir_sumcheck_rounds = mvk0.params.num_whir_sumcheck_rounds();
    let k_whir = mvk0.params.k_whir();

    count_checks.push((
        whir_proof.whir_sumcheck_polys.len(),
        num_whir_sumcheck_rounds,
    ));
    if whir_proof.whir_sumcheck_polys.len() != num_whir_sumcheck_rounds {
        return Err(ProofShapeError::InvalidWhirProofShape(
            WhirProofShapeError::InvalidSumcheckPolys {
                expected: num_whir_sumcheck_rounds,
                actual: whir_proof.whir_sumcheck_polys.len(),
            },
        ));
    }
    count_checks.push((whir_proof.codeword_commits.len(), num_whir_rounds - 1));
    if whir_proof.codeword_commits.len() != num_whir_rounds - 1 {
        return Err(ProofShapeError::InvalidWhirProofShape(
            WhirProofShapeError::InvalidCodewordCommits {
                expected: num_whir_rounds - 1,
                actual: whir_proof.codeword_commits.len(),
            },
        ));
    }
    count_checks.push((whir_proof.ood_values.len(), num_whir_rounds - 1));
    if whir_proof.ood_values.len() != num_whir_rounds - 1 {
        return Err(ProofShapeError::InvalidWhirProofShape(
            WhirProofShapeError::InvalidOodValues {
                expected: num_whir_rounds - 1,
                actual: whir_proof.ood_values.len(),
            },
        ));
    }
    count_checks.push((
        whir_proof.folding_pow_witnesses.len(),
        num_whir_sumcheck_rounds,
    ));
    if whir_proof.folding_pow_witnesses.len() != num_whir_sumcheck_rounds {
        return Err(ProofShapeError::InvalidWhirProofShape(
            WhirProofShapeError::InvalidFoldingPowWitnesses {
                expected: num_whir_sumcheck_rounds,
                actual: whir_proof.folding_pow_witnesses.len(),
            },
        ));
    }
    count_checks.push((whir_proof.query_phase_pow_witnesses.len(), num_whir_rounds));
    if whir_proof.query_phase_pow_witnesses.len() != num_whir_rounds {
        return Err(ProofShapeError::InvalidWhirProofShape(
            WhirProofShapeError::InvalidQueryPhasePowWitnesses {
                expected: num_whir_rounds,
                actual: whir_proof.query_phase_pow_witnesses.len(),
            },
        ));
    }
    count_checks.push((whir_proof.initial_round_opened_rows.len(), layouts.len()));
    if whir_proof.initial_round_opened_rows.len() != layouts.len() {
        return Err(ProofShapeError::InvalidWhirProofShape(
            WhirProofShapeError::InvalidInitialRoundOpenedRows {
                expected: layouts.len(),
                actual: whir_proof.initial_round_opened_rows.len(),
            },
        ));
    }
    count_checks.push((whir_proof.initial_round_merkle_proofs.len(), layouts.len()));
    if whir_proof.initial_round_merkle_proofs.len() != layouts.len() {
        return Err(ProofShapeError::InvalidWhirProofShape(
            WhirProofShapeError::InvalidInitialRoundMerkleProofs {
                expected: layouts.len(),
                actual: whir_proof.initial_round_merkle_proofs.len(),
            },
        ));
    }
    count_checks.push((whir_proof.codeword_opened_values.len(), num_whir_rounds - 1));
    if whir_proof.codeword_opened_values.len() != num_whir_rounds - 1 {
        return Err(ProofShapeError::InvalidWhirProofShape(
            WhirProofShapeError::InvalidCodewordOpenedRows {
                expected: num_whir_rounds - 1,
                actual: whir_proof.codeword_opened_values.len(),
            },
        ));
    }
    count_checks.push((whir_proof.codeword_merkle_proofs.len(), num_whir_rounds - 1));
    if whir_proof.codeword_merkle_proofs.len() != num_whir_rounds - 1 {
        return Err(ProofShapeError::InvalidWhirProofShape(
            WhirProofShapeError::InvalidCodewordMerkleProofs {
                expected: num_whir_rounds - 1,
                actual: whir_proof.codeword_merkle_proofs.len(),
            },
        ));
    }
    let expected_final_poly_len = 1usize << mvk0.params.log_final_poly_len();
    count_checks.push((whir_proof.final_poly.len(), expected_final_poly_len));
    if whir_proof.final_poly.len() != expected_final_poly_len {
        return Err(ProofShapeError::InvalidWhirProofShape(
            WhirProofShapeError::InvalidFinalPolyLen {
                expected: expected_final_poly_len,
                actual: whir_proof.final_poly.len(),
            },
        ));
    }

    let initial_queries = mvk0.params.whir.rounds[0].num_queries;
    for (commit_idx, (opened_rows, merkle_proofs)) in whir_proof
        .initial_round_opened_rows
        .iter()
        .zip(&whir_proof.initial_round_merkle_proofs)
        .enumerate()
    {
        count_checks.push((opened_rows.len(), initial_queries));
        if opened_rows.len() != initial_queries {
            return Err(ProofShapeError::InvalidWhirProofShape(
                WhirProofShapeError::InvalidInitialRoundOpenedRowsQueries {
                    commit_idx,
                    expected: initial_queries,
                    actual: opened_rows.len(),
                },
            ));
        }
        count_checks.push((merkle_proofs.len(), initial_queries));
        if merkle_proofs.len() != initial_queries {
            return Err(ProofShapeError::InvalidWhirProofShape(
                WhirProofShapeError::InvalidInitialRoundMerkleProofsQueries {
                    commit_idx,
                    expected: initial_queries,
                    actual: merkle_proofs.len(),
                },
            ));
        }

        let width = stacking_proof.stacking_openings[commit_idx].len();
        for (opened_idx, rows) in opened_rows.iter().enumerate() {
            let expected_rows_len = 1usize << k_whir;
            count_checks.push((rows.len(), expected_rows_len));
            if rows.len() != expected_rows_len {
                return Err(ProofShapeError::InvalidWhirProofShape(
                    WhirProofShapeError::InvalidInitialRoundOpenedRowK {
                        opened_idx,
                        commit_idx,
                        expected: expected_rows_len,
                        actual: rows.len(),
                    },
                ));
            }
            for (row_idx, row) in rows.iter().enumerate() {
                count_checks.push((row.len(), width));
                if row.len() != width {
                    return Err(ProofShapeError::InvalidWhirProofShape(
                        WhirProofShapeError::InvalidInitialRoundOpenedRowWidth {
                            row_idx,
                            commit_idx,
                            expected: width,
                            actual: row.len(),
                        },
                    ));
                }
            }
        }

        let merkle_depth = (log_stacked_height + mvk0.params.log_blowup).saturating_sub(k_whir);
        for (opened_idx, proof) in merkle_proofs.iter().enumerate() {
            count_checks.push((proof.len(), merkle_depth));
            if proof.len() != merkle_depth {
                return Err(ProofShapeError::InvalidWhirProofShape(
                    WhirProofShapeError::InvalidInitialRoundMerkleProofDepth {
                        opened_idx,
                        commit_idx,
                        expected: merkle_depth,
                        actual: proof.len(),
                    },
                ));
            }
        }
    }

    for (round_minus_one, (opened_values_per_query, merkle_proofs)) in whir_proof
        .codeword_opened_values
        .iter()
        .zip(&whir_proof.codeword_merkle_proofs)
        .take(num_whir_rounds - 1)
        .enumerate()
    {
        let round = round_minus_one + 1;
        let num_queries = mvk0.params.whir.rounds[round].num_queries;
        count_checks.push((opened_values_per_query.len(), num_queries));
        if opened_values_per_query.len() != num_queries {
            return Err(ProofShapeError::InvalidWhirProofShape(
                WhirProofShapeError::InvalidCodewordOpenedRowsQueries {
                    round,
                    expected: num_queries,
                    actual: opened_values_per_query.len(),
                },
            ));
        }
        count_checks.push((merkle_proofs.len(), num_queries));
        if merkle_proofs.len() != num_queries {
            return Err(ProofShapeError::InvalidWhirProofShape(
                WhirProofShapeError::InvalidCodewordMerkleProofsQueries {
                    round,
                    expected: num_queries,
                    actual: merkle_proofs.len(),
                },
            ));
        }

        let expected_opened_values_len = 1usize << mvk0.params.k_whir();
        for (opened_idx, opened_values) in opened_values_per_query.iter().enumerate() {
            count_checks.push((opened_values.len(), expected_opened_values_len));
            if opened_values.len() != expected_opened_values_len {
                return Err(ProofShapeError::InvalidWhirProofShape(
                    WhirProofShapeError::InvalidCodewordOpenedValues {
                        round,
                        opened_idx,
                        expected: expected_opened_values_len,
                        actual: opened_values.len(),
                    },
                ));
            }
        }

        let expected_merkle_depth = log_stacked_height + mvk0.params.log_blowup - k_whir - round;
        for (opened_idx, proof) in merkle_proofs.iter().enumerate() {
            count_checks.push((proof.len(), expected_merkle_depth));
            if proof.len() != expected_merkle_depth {
                return Err(ProofShapeError::InvalidWhirProofShape(
                    WhirProofShapeError::InvalidCodewordMerkleProofDepth {
                        round,
                        opened_idx,
                        expected: expected_merkle_depth,
                        actual: proof.len(),
                    },
                ));
            }
        }
    }

    Ok(ProofShapeRuleDerivation {
        layouts,
        air_required_flags,
        air_public_value_lens,
        air_expected_public_value_lens,
        air_cached_commitment_lens,
        air_expected_cached_commitment_lens,
        air_log_heights,
        max_log_height_allowed,
        count_checks,
        upper_bound_checks,
    })
}

pub fn derive_proof_shape_intermediates(
    config: &NativeConfig,
    mvk: &MultiStarkVerifyingKey<NativeConfig>,
    proof: &Proof<NativeConfig>,
) -> Result<ProofShapeIntermediates, ProofShapePreambleError> {
    if config.params() != &mvk.inner.params {
        return Err(ProofShapePreambleError::SystemParamsMismatch);
    }

    let mvk0 = &mvk.inner;
    let per_air = &mvk0.per_air;
    let l_skip = mvk0.params.l_skip;

    let num_airs = per_air.len();
    let num_traces = proof.trace_vdata.iter().flatten().count();
    let trace_id_to_air_id = compute_trace_id_to_air_id(mvk0, proof);

    let mut trace_height_sums = Vec::with_capacity(mvk0.trace_height_constraints.len());
    let mut trace_height_coefficients = Vec::with_capacity(mvk0.trace_height_constraints.len());
    let mut trace_height_thresholds = Vec::with_capacity(mvk0.trace_height_constraints.len());
    for (constraint_idx, constraint) in mvk0.trace_height_constraints.iter().enumerate() {
        let sum = trace_id_to_air_id
            .iter()
            .map(|&air_id| {
                let log_height = proof.trace_vdata[air_id].as_ref().unwrap().log_height;
                (1 << log_height.max(l_skip)) as u64 * constraint.coefficients[air_id] as u64
            })
            .sum::<u64>();
        let threshold = constraint.threshold as u64;

        if sum >= threshold {
            return Err(ProofShapePreambleError::TraceHeightsTooLarge {
                constraint_idx,
                sum,
                threshold,
            });
        }
        trace_height_sums.push(sum);
        trace_height_coefficients.push(
            constraint
                .coefficients
                .iter()
                .map(|&coeff| coeff as u64)
                .collect::<Vec<_>>(),
        );
        trace_height_thresholds.push(threshold);
    }

    let air_presence_flags = proof.trace_vdata.iter().map(Option::is_some).collect();

    let shape_rules = derive_proof_shape_rules(&mvk.inner, proof)?;

    let mut per_trace = mvk0
        .per_air
        .iter()
        .zip(&proof.trace_vdata)
        .enumerate()
        .filter_map(|(air_idx, (vk, vdata))| vdata.as_ref().map(|vdata| (air_idx, vk, vdata)))
        .collect::<Vec<_>>();
    per_trace.sort_by_key(|(_, _, vdata)| Reverse(vdata.log_height));

    let num_airs_present = per_trace.len();

    Ok(ProofShapeIntermediates {
        num_airs,
        num_traces,
        trace_id_to_air_id,
        trace_height_sums,
        trace_height_coefficients,
        trace_height_thresholds,
        air_presence_flags,
        air_required_flags: shape_rules.air_required_flags,
        air_public_value_lens: shape_rules.air_public_value_lens,
        air_expected_public_value_lens: shape_rules.air_expected_public_value_lens,
        air_cached_commitment_lens: shape_rules.air_cached_commitment_lens,
        air_expected_cached_commitment_lens: shape_rules.air_expected_cached_commitment_lens,
        air_log_heights: shape_rules.air_log_heights,
        max_log_height_allowed: shape_rules.max_log_height_allowed,
        l_skip,
        num_airs_present,
        proof_shape_count_checks: shape_rules.count_checks,
        proof_shape_upper_bound_checks: shape_rules.upper_bound_checks,
    })
}

pub fn derive_proof_shape_ownership_schedule(
    mvk: &MultiStarkVerifyingKey<NativeConfig>,
    proof: &Proof<NativeConfig>,
) -> Result<ProofShapeOwnershipSchedule, ProofShapePreambleError> {
    let shape_rules = derive_proof_shape_rules(&mvk.inner, proof)?;
    Ok(ProofShapeOwnershipSchedule {
        trace_height_coefficients: mvk
            .inner
            .trace_height_constraints
            .iter()
            .map(|constraint| {
                constraint
                    .coefficients
                    .iter()
                    .map(|&coeff| coeff as u64)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
        trace_height_thresholds: mvk
            .inner
            .trace_height_constraints
            .iter()
            .map(|constraint| constraint.threshold as u64)
            .collect::<Vec<_>>(),
        proof_shape_count_checks: shape_rules.count_checks,
        proof_shape_upper_bound_checks: shape_rules.upper_bound_checks,
    })
}

fn assign_and_range_bool(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    value: bool,
) -> AssignedValue<Fr> {
    let bit = ctx.load_witness(Fr::from(value as u64));
    range.gate().assert_bit(ctx, bit);
    bit
}

pub(crate) fn constrain_proof_shape_intermediates(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    actual: &ProofShapeIntermediates,
) -> AssignedProofShapeIntermediates {
    constrain_proof_shape_intermediates_with_ownership(ctx, range, actual, None)
}

pub(crate) fn constrain_proof_shape_intermediates_with_ownership(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    actual: &ProofShapeIntermediates,
    ownership_schedule: Option<&ProofShapeOwnershipSchedule>,
) -> AssignedProofShapeIntermediates {
    let owned_trace_height_coefficients = ownership_schedule
        .map(|schedule| schedule.trace_height_coefficients.as_slice())
        .unwrap_or(actual.trace_height_coefficients.as_slice());
    let owned_trace_height_thresholds = ownership_schedule
        .map(|schedule| schedule.trace_height_thresholds.as_slice())
        .unwrap_or(actual.trace_height_thresholds.as_slice());

    assert!(
        actual.max_log_height_allowed < 64,
        "trace-height recomputation requires <=63-bit shift domain",
    );
    assert_eq!(
        actual.trace_height_sums.len(),
        actual.trace_height_thresholds.len()
    );
    assert_eq!(
        actual.trace_height_sums.len(),
        owned_trace_height_thresholds.len(),
        "trace-height sum vector must align with ownership schedule thresholds",
    );
    assert_eq!(
        actual.trace_height_coefficients.len(),
        actual.trace_height_thresholds.len()
    );
    assert_eq!(
        owned_trace_height_coefficients.len(),
        owned_trace_height_thresholds.len(),
        "trace-height coefficient schedule length must match ownership thresholds",
    );
    for coeffs in owned_trace_height_coefficients {
        assert_eq!(
            coeffs.len(),
            actual.num_airs,
            "trace-height coefficient vector must match num_airs",
        );
    }
    assert_eq!(
        actual.air_presence_flags.len(),
        actual.air_required_flags.len()
    );
    assert_eq!(
        actual.air_presence_flags.len(),
        actual.air_public_value_lens.len()
    );
    assert_eq!(
        actual.air_presence_flags.len(),
        actual.air_expected_public_value_lens.len()
    );
    assert_eq!(
        actual.air_presence_flags.len(),
        actual.air_cached_commitment_lens.len()
    );
    assert_eq!(
        actual.air_presence_flags.len(),
        actual.air_expected_cached_commitment_lens.len()
    );
    assert_eq!(
        actual.air_presence_flags.len(),
        actual.air_log_heights.len()
    );
    let gate = range.gate();

    if let Some(schedule) = ownership_schedule {
        assert_eq!(
            actual.trace_height_coefficients.len(),
            schedule.trace_height_coefficients.len(),
            "trace-height coefficient witness/schedule length mismatch",
        );
        for (actual_coeffs, owned_coeffs) in actual
            .trace_height_coefficients
            .iter()
            .zip(&schedule.trace_height_coefficients)
        {
            assert_eq!(
                actual_coeffs.len(),
                owned_coeffs.len(),
                "trace-height coefficient row witness/schedule width mismatch",
            );
            for (&actual_coeff, &owned_coeff) in actual_coeffs.iter().zip(owned_coeffs) {
                let coeff_cell = assign_and_range_u64(ctx, range, actual_coeff);
                gate.assert_is_const(ctx, &coeff_cell, &Fr::from(owned_coeff));
            }
        }
    }

    let num_airs = assign_and_range_usize(ctx, range, actual.num_airs);
    gate.assert_is_const(
        ctx,
        &num_airs,
        &Fr::from(usize_to_u64(actual.air_presence_flags.len())),
    );

    let num_traces = assign_and_range_usize(ctx, range, actual.num_traces);
    gate.assert_is_const(
        ctx,
        &num_traces,
        &Fr::from(usize_to_u64(actual.trace_id_to_air_id.len())),
    );

    let trace_id_to_air_id = actual
        .trace_id_to_air_id
        .iter()
        .map(|&actual_air_id| {
            let air_id = assign_and_range_usize(ctx, range, actual_air_id);
            range.check_less_than_safe(ctx, air_id, usize_to_u64(actual.num_airs));
            air_id
        })
        .collect::<Vec<_>>();
    for i in 0..trace_id_to_air_id.len() {
        for j in (i + 1)..trace_id_to_air_id.len() {
            let diff = gate.sub(ctx, trace_id_to_air_id[i], trace_id_to_air_id[j]);
            let equal = gate.is_zero(ctx, diff);
            gate.assert_is_const(ctx, &equal, &Fr::from(0u64));
        }
    }

    let air_presence_flags: Vec<AssignedValue<Fr>> = actual
        .air_presence_flags
        .iter()
        .map(|&actual_flag| {
            let flag = assign_and_range_bool(ctx, range, actual_flag);
            flag
        })
        .collect();

    let air_required_flags = actual
        .air_required_flags
        .iter()
        .map(|&actual_flag| assign_and_range_bool(ctx, range, actual_flag))
        .collect::<Vec<_>>();

    let air_public_value_lens = actual
        .air_public_value_lens
        .iter()
        .map(|&actual_len| assign_and_range_usize(ctx, range, actual_len))
        .collect::<Vec<_>>();
    let air_expected_public_value_lens = actual
        .air_expected_public_value_lens
        .iter()
        .map(|&actual_len| assign_and_range_usize(ctx, range, actual_len))
        .collect::<Vec<_>>();
    for (actual_len, expected_len) in air_public_value_lens
        .iter()
        .zip(air_expected_public_value_lens.iter())
    {
        ctx.constrain_equal(actual_len, expected_len);
    }

    let air_cached_commitment_lens = actual
        .air_cached_commitment_lens
        .iter()
        .map(|&actual_len| assign_and_range_usize(ctx, range, actual_len))
        .collect::<Vec<_>>();
    let air_expected_cached_commitment_lens = actual
        .air_expected_cached_commitment_lens
        .iter()
        .map(|&actual_len| assign_and_range_usize(ctx, range, actual_len))
        .collect::<Vec<_>>();
    for (actual_len, expected_len) in air_cached_commitment_lens
        .iter()
        .zip(air_expected_cached_commitment_lens.iter())
    {
        ctx.constrain_equal(actual_len, expected_len);
    }

    let max_log_height_allowed = assign_and_range_usize(ctx, range, actual.max_log_height_allowed);
    let air_log_heights = actual
        .air_log_heights
        .iter()
        .map(|&actual_log_height| {
            let log_height = assign_and_range_usize(ctx, range, actual_log_height);
            range.check_less_than_safe(
                ctx,
                log_height,
                usize_to_u64(actual.max_log_height_allowed).saturating_add(1),
            );
            log_height
        })
        .collect::<Vec<_>>();

    let air_index_consts = (0..actual.num_airs)
        .map(|air_idx| ctx.load_constant(Fr::from(usize_to_u64(air_idx))))
        .collect::<Vec<_>>();

    let mut trace_log_heights = Vec::with_capacity(trace_id_to_air_id.len());
    for &trace_air in &trace_id_to_air_id {
        let mut selected_presence = ctx.load_constant(Fr::from(0u64));
        let mut selected_log_height = ctx.load_constant(Fr::from(0u64));
        for (air_idx, &air_const) in air_index_consts.iter().enumerate() {
            let diff = gate.sub(ctx, trace_air, air_const);
            let is_match = gate.is_zero(ctx, diff);
            let presence_term = gate.mul(ctx, is_match, air_presence_flags[air_idx]);
            selected_presence = gate.add(ctx, selected_presence, presence_term);
            let log_height_term = gate.mul(ctx, is_match, air_log_heights[air_idx]);
            selected_log_height = gate.add(ctx, selected_log_height, log_height_term);
        }
        gate.assert_is_const(ctx, &selected_presence, &Fr::from(1u64));
        trace_log_heights.push(selected_log_height);
    }

    for (air_idx, &air_const) in air_index_consts.iter().enumerate() {
        let mut trace_count = ctx.load_constant(Fr::from(0u64));
        for &trace_air in &trace_id_to_air_id {
            let diff = gate.sub(ctx, trace_air, air_const);
            let is_match = gate.is_zero(ctx, diff);
            trace_count = gate.add(ctx, trace_count, is_match);
        }
        ctx.constrain_equal(&trace_count, &air_presence_flags[air_idx]);
    }

    for i in 0..trace_log_heights.len().saturating_sub(1) {
        let log_diff = gate.sub(ctx, trace_log_heights[i], trace_log_heights[i + 1]);
        range.range_check(
            ctx,
            log_diff,
            bits_for_u64(usize_to_u64(actual.max_log_height_allowed)),
        );

        let logs_equal = gate.is_zero(ctx, log_diff);
        let id_diff = gate.sub(ctx, trace_id_to_air_id[i + 1], trace_id_to_air_id[i]);
        let tied_id_diff = gate.mul(ctx, logs_equal, id_diff);
        range.range_check(
            ctx,
            tied_id_diff,
            bits_for_u64(usize_to_u64(actual.num_airs)),
        );
        let tied_id_diff_is_zero = gate.is_zero(ctx, tied_id_diff);
        let tie_break_violation = gate.mul(ctx, logs_equal, tied_id_diff_is_zero);
        gate.assert_is_const(ctx, &tie_break_violation, &Fr::from(0u64));
    }

    let trace_height_sums = owned_trace_height_thresholds
        .iter()
        .enumerate()
        .map(|(constraint_idx, &threshold)| {
            let mut sum = ctx.load_constant(Fr::from(0u64));
            let coeffs = &owned_trace_height_coefficients[constraint_idx];
            for (&trace_air, &trace_log_height) in trace_id_to_air_id.iter().zip(&trace_log_heights)
            {
                let mut coeff_for_trace = ctx.load_constant(Fr::from(0u64));
                for (air_idx, &air_const) in air_index_consts.iter().enumerate() {
                    let diff = gate.sub(ctx, trace_air, air_const);
                    let is_match = gate.is_zero(ctx, diff);
                    let coeff_const = ctx.load_constant(Fr::from(coeffs[air_idx]));
                    let weighted = gate.mul(ctx, is_match, coeff_const);
                    coeff_for_trace = gate.add(ctx, coeff_for_trace, weighted);
                }

                let mut trace_pow = ctx.load_constant(Fr::from(0u64));
                for log_height in 0..=actual.max_log_height_allowed {
                    let log_const = ctx.load_constant(Fr::from(usize_to_u64(log_height)));
                    let log_diff = gate.sub(ctx, trace_log_height, log_const);
                    let is_log = gate.is_zero(ctx, log_diff);
                    let pow = 1u64 << (max(log_height, actual.l_skip) as u32);
                    let pow_const = ctx.load_constant(Fr::from(pow));
                    let weighted = gate.mul(ctx, is_log, pow_const);
                    trace_pow = gate.add(ctx, trace_pow, weighted);
                }

                let contribution = gate.mul(ctx, coeff_for_trace, trace_pow);
                sum = gate.add(ctx, sum, contribution);
            }

            range.check_less_than_safe(ctx, sum, threshold);

            // Keep the host-derived sum as debug observability only.
            let debug_sum =
                assign_and_range_u64(ctx, range, actual.trace_height_sums[constraint_idx]);
            ctx.constrain_equal(&sum, &debug_sum);
            sum
        })
        .collect::<Vec<_>>();

    let one = ctx.load_constant(Fr::from(1u64));
    for (presence, required) in air_presence_flags.iter().zip(air_required_flags.iter()) {
        let one_minus_presence = gate.sub(ctx, one, *presence);
        let required_without_presence = gate.mul(ctx, *required, one_minus_presence);
        gate.assert_is_const(ctx, &required_without_presence, &Fr::from(0u64));
    }
    for (presence, log_height) in air_presence_flags.iter().zip(air_log_heights.iter()) {
        let one_minus_presence = gate.sub(ctx, one, *presence);
        let absent_log_height = gate.mul(ctx, *log_height, one_minus_presence);
        gate.assert_is_const(ctx, &absent_log_height, &Fr::from(0u64));
    }

    let num_airs_present = assign_and_range_usize(ctx, range, actual.num_airs_present);
    gate.assert_is_const(
        ctx,
        &num_airs_present,
        &Fr::from(usize_to_u64(actual.trace_id_to_air_id.len())),
    );
    ctx.constrain_equal(&num_airs_present, &num_traces);
    let presence_sum = gate.sum(ctx, air_presence_flags.iter().copied());
    ctx.constrain_equal(&num_airs_present, &presence_sum);

    AssignedProofShapeIntermediates {
        num_airs,
        num_traces,
        trace_id_to_air_id,
        trace_height_sums,
        air_presence_flags,
        air_required_flags,
        air_public_value_lens,
        air_expected_public_value_lens,
        air_cached_commitment_lens,
        air_expected_cached_commitment_lens,
        air_log_heights,
        max_log_height_allowed,
        num_airs_present,
    }
}

pub fn derive_and_constrain_proof_shape(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    config: &NativeConfig,
    mvk: &MultiStarkVerifyingKey<NativeConfig>,
    proof: &Proof<NativeConfig>,
) -> Result<AssignedProofShapeIntermediates, ProofShapePreambleError> {
    let raw = derive_raw_proof_shape_witness_state(config, mvk, proof)?;
    let ownership = derive_proof_shape_ownership_schedule(mvk, proof)?;
    Ok(
        constrain_checked_proof_shape_witness_state_with_ownership(ctx, range, &raw, Some(&ownership))
            .assigned,
    )
}

pub(crate) fn derive_raw_proof_shape_witness_state(
    config: &NativeConfig,
    mvk: &MultiStarkVerifyingKey<NativeConfig>,
    proof: &Proof<NativeConfig>,
) -> Result<RawProofShapeWitnessState, ProofShapePreambleError> {
    Ok(RawProofShapeWitnessState {
        intermediates: derive_proof_shape_intermediates(config, mvk, proof)?,
    })
}

pub(crate) fn constrain_checked_proof_shape_witness_state_with_ownership(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    raw: &RawProofShapeWitnessState,
    ownership_schedule: Option<&ProofShapeOwnershipSchedule>,
) -> CheckedProofShapeWitnessState {
    let assigned = constrain_proof_shape_intermediates_with_ownership(
        ctx,
        range,
        &raw.intermediates,
        ownership_schedule,
    );
    let derived = DerivedProofShapeState {
        trace_height_sums: assigned.trace_height_sums.clone(),
    };
    CheckedProofShapeWitnessState { assigned, derived }
}

#[cfg(test)]
mod tests;
