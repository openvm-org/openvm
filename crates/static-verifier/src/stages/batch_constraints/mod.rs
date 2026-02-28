use core::cmp::Reverse;
#[cfg(test)]
use std::cell::RefCell;
use std::{iter::zip, slice};

use halo2_base::{
    AssignedValue, Context,
    gates::{GateInstructions, RangeInstructions, range::RangeChip},
};
use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::{
        BabyBearBn254Poseidon2Config as NativeConfig, Digest as NativeDigest, EF as NativeEF,
        F as NativeF, default_transcript,
    },
    openvm_stark_backend::{
        FiatShamirTranscript, StarkProtocolConfig,
        air_builders::symbolic::{
            SymbolicConstraints, SymbolicExpressionNode,
            symbolic_expression::SymbolicEvaluator,
            symbolic_variable::{Entry, SymbolicVariable},
        },
        calculate_n_logup,
        interaction::Interaction,
        keygen::types::{MultiStarkVerifyingKey, MultiStarkVerifyingKey0},
        p3_field::{
            BasedVectorSpace, Field, PrimeCharacteristicRing, PrimeField64, TwoAdicField,
            batch_multiplicative_inverse,
        },
        poly_common::{UnivariatePoly, eval_eq_mle, eval_eq_sharp_uni, eval_eq_uni},
        proof::{BatchConstraintProof, GkrProof, Proof, column_openings_by_rot},
        verifier::{
            batch_constraints::BatchConstraintError as NativeBatchConstraintError,
            fractional_sumcheck_gkr::verify_gkr,
            proof_shape::ProofShapeError,
        },
    },
};

use crate::{
    circuit::Fr,
    gadgets::baby_bear::{BABY_BEAR_EXT_DEGREE, BabyBearArithmeticGadgets, BabyBearExtVar},
    stages::{proof_shape::derive_proof_shape_rules, shared_math},
    utils::{assign_and_range_u64, assign_and_range_usize, usize_to_u64},
};

#[derive(Debug, PartialEq, Eq)]
pub enum BatchConstraintError {
    SystemParamsMismatch,
    TraceHeightsTooLarge,
    MissingTraceVData {
        air_id: usize,
    },
    MissingPreprocessedView {
        air_id: usize,
    },
    UnsupportedSymbolicVariableEntry {
        air_id: usize,
        entry: &'static str,
    },
    SymbolicVariableOutOfBounds {
        air_id: usize,
        entry: &'static str,
        index: usize,
    },
    MissingStackedChallenges,
    ProofShape(ProofShapeError),
    BatchConstraint(NativeBatchConstraintError<NativeEF>),
}

impl From<ProofShapeError> for BatchConstraintError {
    fn from(value: ProofShapeError) -> Self {
        Self::ProofShape(value)
    }
}

impl From<NativeBatchConstraintError<NativeEF>> for BatchConstraintError {
    fn from(value: NativeBatchConstraintError<NativeEF>) -> Self {
        Self::BatchConstraint(value)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchIntermediates {
    pub trace_id_to_air_id: Vec<usize>,
    pub n_per_trace: Vec<isize>,
    pub l_skip: usize,
    pub total_interactions: u64,
    pub n_logup: usize,
    pub n_max: usize,
    pub batch_degree: usize,
    pub trace_has_preprocessed: Vec<bool>,
    pub trace_constraint_nodes: Vec<Vec<SymbolicExpressionNode<NativeF>>>,
    pub trace_constraint_indices: Vec<Vec<usize>>,
    pub trace_interactions: Vec<Vec<Interaction<usize>>>,
    pub public_values: Vec<Vec<u64>>,
    pub logup_pow_bits: usize,
    pub logup_pow_witness: u64,
    pub logup_pow_sampled_bits: u64,
    pub logup_pow_witness_ok: bool,
    pub gkr_q0_claim: Option<[u64; BABY_BEAR_EXT_DEGREE]>,
    pub gkr_claims_per_layer: Vec<[[u64; BABY_BEAR_EXT_DEGREE]; 4]>,
    pub gkr_sumcheck_polys: Vec<Vec<[u64; BABY_BEAR_EXT_DEGREE]>>,
    pub numerator_term_per_air: Vec<[u64; BABY_BEAR_EXT_DEGREE]>,
    pub denominator_term_per_air: Vec<[u64; BABY_BEAR_EXT_DEGREE]>,
    pub univariate_round_coeffs: Vec<[u64; BABY_BEAR_EXT_DEGREE]>,
    pub sumcheck_round_polys: Vec<Vec<[u64; BABY_BEAR_EXT_DEGREE]>>,
    pub column_openings: Vec<Vec<Vec<[u64; BABY_BEAR_EXT_DEGREE]>>>,
    pub column_openings_need_rot: Vec<Vec<bool>>,
    pub column_opening_expected_widths: Vec<Vec<usize>>,
    pub gkr_numerator_residual: [u64; BABY_BEAR_EXT_DEGREE],
    pub gkr_denominator_residual: [u64; BABY_BEAR_EXT_DEGREE],
    pub gkr_denominator_claim: [u64; BABY_BEAR_EXT_DEGREE],
    pub alpha_logup: [u64; BABY_BEAR_EXT_DEGREE],
    pub beta_logup: [u64; BABY_BEAR_EXT_DEGREE],
    pub gkr_non_xi_samples: Vec<[u64; BABY_BEAR_EXT_DEGREE]>,
    pub gkr_xi_sample_order: Vec<[u64; BABY_BEAR_EXT_DEGREE]>,
    pub xi: Vec<[u64; BABY_BEAR_EXT_DEGREE]>,
    pub lambda: [u64; BABY_BEAR_EXT_DEGREE],
    pub mu: [u64; BABY_BEAR_EXT_DEGREE],
    pub sum_claim: [u64; BABY_BEAR_EXT_DEGREE],
    pub sum_univ_domain_s_0: [u64; BABY_BEAR_EXT_DEGREE],
    pub consistency_lhs: [u64; BABY_BEAR_EXT_DEGREE],
    pub consistency_rhs: [u64; BABY_BEAR_EXT_DEGREE],
    pub consistency_residual: [u64; BABY_BEAR_EXT_DEGREE],
    pub r: Vec<[u64; BABY_BEAR_EXT_DEGREE]>,
}

#[derive(Clone, Debug)]
pub struct AssignedBatchIntermediates {
    pub trace_id_to_air_id: Vec<AssignedValue<Fr>>,
    pub public_values: Vec<Vec<crate::gadgets::baby_bear::BabyBearVar>>,
    pub total_interactions: AssignedValue<Fr>,
    pub n_logup: AssignedValue<Fr>,
    pub n_max: AssignedValue<Fr>,
    pub batch_degree: AssignedValue<Fr>,
    pub logup_pow_bits: AssignedValue<Fr>,
    pub logup_pow_sampled_bits: AssignedValue<Fr>,
    pub logup_pow_witness_ok: AssignedValue<Fr>,
    pub gkr_q0_claim: Option<BabyBearExtVar>,
    pub gkr_claims_per_layer: Vec<Vec<BabyBearExtVar>>,
    pub gkr_sumcheck_polys: Vec<Vec<BabyBearExtVar>>,
    pub numerator_term_per_air: Vec<BabyBearExtVar>,
    pub denominator_term_per_air: Vec<BabyBearExtVar>,
    pub univariate_round_coeffs: Vec<BabyBearExtVar>,
    pub sumcheck_round_polys: Vec<Vec<BabyBearExtVar>>,
    pub column_openings: Vec<Vec<Vec<BabyBearExtVar>>>,
    pub column_openings_need_rot: Vec<Vec<bool>>,
    pub gkr_numerator_residual: BabyBearExtVar,
    pub gkr_denominator_residual: BabyBearExtVar,
    pub gkr_denominator_claim: BabyBearExtVar,
    pub alpha_logup: BabyBearExtVar,
    pub beta_logup: BabyBearExtVar,
    pub gkr_non_xi_samples: Vec<BabyBearExtVar>,
    pub gkr_xi_sample_order: Vec<BabyBearExtVar>,
    pub xi: Vec<BabyBearExtVar>,
    pub lambda: BabyBearExtVar,
    pub mu: BabyBearExtVar,
    pub sum_claim: BabyBearExtVar,
    pub sum_univ_domain_s_0: BabyBearExtVar,
    pub consistency_lhs: BabyBearExtVar,
    pub consistency_rhs: BabyBearExtVar,
    pub consistency_residual: BabyBearExtVar,
    pub r: Vec<BabyBearExtVar>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RawBatchWitnessState {
    pub intermediates: BatchIntermediates,
}

#[derive(Clone, Debug)]
pub struct DerivedBatchState {
    pub sum_claim: BabyBearExtVar,
    pub sum_univ_domain_s_0: BabyBearExtVar,
    pub consistency_residual: BabyBearExtVar,
}

#[derive(Clone, Debug)]
pub struct CheckedBatchWitnessState {
    pub assigned: AssignedBatchIntermediates,
    pub derived: DerivedBatchState,
}

fn ext_to_coeffs(value: NativeEF) -> [u64; BABY_BEAR_EXT_DEGREE] {
    core::array::from_fn(|i| {
        <NativeEF as BasedVectorSpace<NativeF>>::as_basis_coefficients_slice(&value)[i]
            .as_canonical_u64()
    })
}

fn coeffs_to_ext(coeffs: [u64; BABY_BEAR_EXT_DEGREE]) -> NativeEF {
    NativeEF::from_basis_coefficients_fn(|i| NativeF::from_u64(coeffs[i]))
}

type ViewPair<'a, T> = &'a [(T, T)];

fn progression_exp_2<EF>(m: EF, l: usize) -> EF
where
    EF: PrimeCharacteristicRing + Copy,
{
    (0..l)
        .fold((m, EF::ONE), |(pow, sum), _| {
            (pow * pow, sum * (EF::ONE + pow))
        })
        .1
}

struct VerifierConstraintEvaluator<'a> {
    preprocessed: Option<ViewPair<'a, NativeEF>>,
    partitioned_main: &'a [ViewPair<'a, NativeEF>],
    is_first_row: NativeEF,
    is_last_row: NativeEF,
    public_values: &'a [NativeF],
}

impl<'a> VerifierConstraintEvaluator<'a> {
    fn new(
        preprocessed: Option<ViewPair<'a, NativeEF>>,
        partitioned_main: &'a [ViewPair<'a, NativeEF>],
        public_values: &'a [NativeF],
        rs: &'a [NativeEF],
        l_skip: usize,
    ) -> Self {
        let omega = NativeF::two_adic_generator(l_skip);
        let inv = NativeEF::from(NativeF::from_usize(1 << l_skip).inverse());
        let is_first_row = inv
            * progression_exp_2(rs[0], l_skip)
            * rs[1..]
                .iter()
                .fold(NativeEF::ONE, |acc, &x| acc * (NativeEF::ONE - x));
        let is_last_row = inv
            * progression_exp_2(rs[0] * omega, l_skip)
            * rs[1..].iter().fold(NativeEF::ONE, |acc, &x| acc * x);

        Self {
            preprocessed,
            partitioned_main,
            is_first_row,
            is_last_row,
            public_values,
        }
    }
}

fn symbolic_entry_tag(entry: &Entry) -> &'static str {
    match entry {
        Entry::Preprocessed { .. } => "preprocessed",
        Entry::Main { .. } => "main",
        Entry::Public => "public",
        _ => "unsupported",
    }
}

fn validate_symbolic_nodes(
    air_id: usize,
    nodes: &[SymbolicExpressionNode<NativeF>],
    preprocessed: Option<ViewPair<'_, NativeEF>>,
    partitioned_main: &[ViewPair<'_, NativeEF>],
    public_values: &[NativeF],
) -> Result<(), BatchConstraintError> {
    for node in nodes {
        let SymbolicExpressionNode::Variable(var) = node else {
            continue;
        };
        match var.entry {
            Entry::Preprocessed { .. } => {
                let vp = preprocessed
                    .ok_or(BatchConstraintError::MissingPreprocessedView { air_id })?;
                if var.index >= vp.len() {
                    return Err(BatchConstraintError::SymbolicVariableOutOfBounds {
                        air_id,
                        entry: "preprocessed",
                        index: var.index,
                    });
                }
            }
            Entry::Main { part_index, .. } => {
                let Some(vp) = partitioned_main.get(part_index) else {
                    return Err(BatchConstraintError::SymbolicVariableOutOfBounds {
                        air_id,
                        entry: "main-part",
                        index: part_index,
                    });
                };
                if var.index >= vp.len() {
                    return Err(BatchConstraintError::SymbolicVariableOutOfBounds {
                        air_id,
                        entry: "main",
                        index: var.index,
                    });
                }
            }
            Entry::Public => {
                if var.index >= public_values.len() {
                    return Err(BatchConstraintError::SymbolicVariableOutOfBounds {
                        air_id,
                        entry: "public",
                        index: var.index,
                    });
                }
            }
            _ => {
                return Err(BatchConstraintError::UnsupportedSymbolicVariableEntry {
                    air_id,
                    entry: symbolic_entry_tag(&var.entry),
                });
            }
        }
    }
    Ok(())
}

impl SymbolicEvaluator<NativeF, NativeEF> for VerifierConstraintEvaluator<'_> {
    fn eval_const(&self, c: NativeF) -> NativeEF {
        NativeEF::from(c)
    }

    fn eval_var(&self, symbolic_var: SymbolicVariable<NativeF>) -> NativeEF {
        let index = symbolic_var.index;
        match symbolic_var.entry {
            Entry::Preprocessed { offset } => self
                .preprocessed
                .and_then(|vp| vp.get(index).copied())
                .map(|value| if offset == 0 { value.0 } else { value.1 })
                .unwrap_or(NativeEF::ZERO),
            Entry::Main { part_index, offset } => self
                .partitioned_main
                .get(part_index)
                .and_then(|vp| vp.get(index).copied())
                .map(|value| if offset == 0 { value.0 } else { value.1 })
                .unwrap_or(NativeEF::ZERO),
            Entry::Public => self
                .public_values
                .get(index)
                .copied()
                .map(NativeEF::from)
                .unwrap_or(NativeEF::ZERO),
            _ => NativeEF::ZERO,
        }
    }

    fn eval_is_first_row(&self) -> NativeEF {
        self.is_first_row
    }

    fn eval_is_last_row(&self) -> NativeEF {
        self.is_last_row
    }

    fn eval_is_transition(&self) -> NativeEF {
        NativeEF::ONE - self.is_last_row
    }
}

pub(crate) fn compute_trace_id_to_air_id(
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

pub(crate) fn enforce_trace_height_constraints(
    mvk0: &MultiStarkVerifyingKey0<NativeConfig>,
    proof: &Proof<NativeConfig>,
    trace_id_to_air_id: &[usize],
) -> Result<(), BatchConstraintError> {
    let l_skip = mvk0.params.l_skip;
    for constraint in &mvk0.trace_height_constraints {
        let sum = trace_id_to_air_id
            .iter()
            .map(|&air_id| {
                let log_height = proof.trace_vdata[air_id]
                    .as_ref()
                    .ok_or(BatchConstraintError::MissingTraceVData { air_id })?
                    .log_height;
                Ok((1 << log_height.max(l_skip)) as u64 * constraint.coefficients[air_id] as u64)
            })
            .collect::<Result<Vec<_>, BatchConstraintError>>()?
            .into_iter()
            .sum::<u64>();
        if sum >= constraint.threshold as u64 {
            return Err(BatchConstraintError::TraceHeightsTooLarge);
        }
    }
    Ok(())
}

pub(crate) fn observe_preamble<TS: FiatShamirTranscript<NativeConfig>>(
    transcript: &mut TS,
    mvk0: &MultiStarkVerifyingKey0<NativeConfig>,
    mvk_pre_hash: NativeDigest,
    proof: &Proof<NativeConfig>,
) {
    transcript.observe_commit(mvk_pre_hash);
    transcript.observe_commit(proof.common_main_commit);

    for ((trace_vdata, avk), pvs) in proof
        .trace_vdata
        .iter()
        .zip(&mvk0.per_air)
        .zip(&proof.public_values)
    {
        let is_air_present = trace_vdata.is_some();

        if !avk.is_required {
            transcript.observe(NativeF::from_bool(is_air_present));
        }
        if let Some(trace_vdata) = trace_vdata {
            if let Some(pdata) = avk.preprocessed_data.as_ref() {
                transcript.observe_commit(pdata.commit);
            } else {
                transcript.observe(NativeF::from_usize(trace_vdata.log_height));
            }
            for commit in &trace_vdata.cached_commitments {
                transcript.observe_commit(*commit);
            }
        }
        for pv in pvs {
            transcript.observe(*pv);
        }
    }
}

fn check_witness_with_sample_bits<TS: FiatShamirTranscript<NativeConfig>>(
    transcript: &mut TS,
    bits: usize,
    witness: NativeF,
) -> (bool, u64) {
    if bits == 0 {
        return (true, 0);
    }
    transcript.observe(witness);
    let sampled_bits = transcript.sample_bits(bits) as u64;
    (sampled_bits == 0, sampled_bits)
}

#[derive(Clone)]
struct SampleLoggingTranscript<TS: FiatShamirTranscript<NativeConfig>> {
    inner: TS,
    sampled: Vec<u64>,
}

impl<TS: FiatShamirTranscript<NativeConfig>> SampleLoggingTranscript<TS> {
    fn new(inner: TS) -> Self {
        Self {
            inner,
            sampled: Vec::new(),
        }
    }

    fn into_sampled(self) -> Vec<u64> {
        self.sampled
    }
}

impl<TS: FiatShamirTranscript<NativeConfig>> FiatShamirTranscript<NativeConfig>
    for SampleLoggingTranscript<TS>
{
    fn observe(&mut self, value: NativeF) {
        self.inner.observe(value);
    }

    fn sample(&mut self) -> NativeF {
        let sampled = self.inner.sample();
        self.sampled.push(sampled.as_canonical_u64());
        sampled
    }

    fn observe_commit(&mut self, digest: NativeDigest) {
        self.inner.observe_commit(digest);
    }
}

pub(crate) fn derive_batch_intermediates_with_inputs(
    transcript: &mut impl FiatShamirTranscript<NativeConfig>,
    mvk0: &MultiStarkVerifyingKey0<NativeConfig>,
    public_values: &[Vec<NativeF>],
    gkr_proof: &GkrProof<NativeConfig>,
    batch_proof: &BatchConstraintProof<NativeConfig>,
    trace_id_to_air_id: &[usize],
    n_per_trace: &[isize],
    omega_skip_pows: &[NativeF],
) -> Result<BatchIntermediates, NativeBatchConstraintError<NativeEF>> {
    let l_skip = mvk0.params.l_skip;
    let BatchConstraintProof {
        numerator_term_per_air,
        denominator_term_per_air,
        univariate_round_coeffs,
        sumcheck_round_polys,
        column_openings,
    } = batch_proof;

    let logup_pow_bits = mvk0.params.logup.pow_bits;
    let (logup_pow_witness_ok, logup_pow_sampled_bits) =
        check_witness_with_sample_bits(transcript, logup_pow_bits, gkr_proof.logup_pow_witness);
    if !logup_pow_witness_ok {
        return Err(NativeBatchConstraintError::InvalidLogupPowWitness);
    }

    let alpha_logup = transcript.sample_ext();
    let beta_logup = transcript.sample_ext();

    let total_interactions = zip(trace_id_to_air_id, n_per_trace)
        .map(|(&air_idx, &n)| {
            let n_lift = n.max(0) as usize;
            let num_interactions = mvk0.per_air[air_idx]
                .symbolic_constraints
                .interactions
                .len();
            (num_interactions as u64) << (l_skip + n_lift)
        })
        .sum::<u64>();
    let n_logup = calculate_n_logup(l_skip, total_interactions);

    let mut xi = Vec::new();
    let mut gkr_non_xi_samples = Vec::new();
    let mut gkr_xi_sample_order = Vec::new();
    let mut p_xi_claim = NativeEF::ZERO;
    let mut q_xi_claim = alpha_logup;
    if total_interactions > 0 {
        let mut logging_transcript = SampleLoggingTranscript::new(transcript.clone());
        (p_xi_claim, q_xi_claim, xi) =
            verify_gkr::<NativeConfig, _>(gkr_proof, &mut logging_transcript, l_skip + n_logup)?;
        *transcript = logging_transcript.inner.clone();
        let sampled_ext = logging_transcript
            .into_sampled()
            .chunks_exact(BABY_BEAR_EXT_DEGREE)
            .map(|chunk| core::array::from_fn(|i| chunk[i]))
            .collect::<Vec<_>>();
        let gkr_xi_samples = xi.len();
        assert!(
            sampled_ext.len() >= gkr_xi_samples,
            "GKR sampled transcript trace must cover returned xi challenges",
        );
        gkr_non_xi_samples = sampled_ext[..sampled_ext.len() - gkr_xi_samples].to_vec();
        gkr_xi_sample_order = sampled_ext[sampled_ext.len() - gkr_xi_samples..].to_vec();
    }

    let n_max = n_per_trace.iter().copied().max().unwrap_or(0).max(0) as usize;
    let batch_degree = mvk0.params.max_constraint_degree + 1;
    let n_global = n_max.max(n_logup);
    while xi.len() != l_skip + n_global {
        xi.push(transcript.sample_ext());
    }

    let lambda = transcript.sample_ext();

    for (&sum_claim_p, &sum_claim_q) in zip(numerator_term_per_air, denominator_term_per_air) {
        p_xi_claim -= sum_claim_p;
        q_xi_claim -= sum_claim_q;
        transcript.observe_ext(sum_claim_p);
        transcript.observe_ext(sum_claim_q);
    }

    let gkr_numerator_residual = p_xi_claim;
    let gkr_denominator_residual = q_xi_claim - alpha_logup;
    if gkr_numerator_residual != NativeEF::ZERO {
        return Err(NativeBatchConstraintError::GkrNumeratorMismatch {
            claim: gkr_numerator_residual,
        });
    }
    if q_xi_claim != alpha_logup {
        return Err(NativeBatchConstraintError::GkrDenominatorMismatch { claim: q_xi_claim });
    }

    let mu = transcript.sample_ext();

    let mut sum_claim = NativeEF::ZERO;
    let mut cur_mu_pow = NativeEF::ONE;
    for (&sum_claim_p, &sum_claim_q) in zip(numerator_term_per_air, denominator_term_per_air) {
        sum_claim += sum_claim_p * cur_mu_pow;
        cur_mu_pow *= mu;
        sum_claim += sum_claim_q * cur_mu_pow;
        cur_mu_pow *= mu;
    }

    for &coeff in univariate_round_coeffs {
        transcript.observe_ext(coeff);
    }
    let r_0 = transcript.sample_ext();

    let sum_univ_domain_s_0 = UnivariatePoly::new(univariate_round_coeffs.clone())
        .coeffs()
        .iter()
        .step_by(1 << l_skip)
        .copied()
        .sum::<NativeEF>()
        * NativeEF::from_usize(1 << l_skip);
    if sum_claim != sum_univ_domain_s_0 {
        return Err(NativeBatchConstraintError::SumClaimMismatch {
            sum_claim,
            sum_univ_domain_s_0,
        });
    }

    let mut cur_sum = UnivariatePoly::new(univariate_round_coeffs.clone()).eval_at_point(r_0);
    let mut rs = vec![r_0];

    for batch_s_evals in sumcheck_round_polys.iter().take(n_max) {
        for &eval in batch_s_evals {
            transcript.observe_ext(eval);
        }

        let s_1 = batch_s_evals[0];
        let s_0 = cur_sum - s_1;

        let mut factorials = vec![NativeF::ONE; batch_degree + 1];
        for i in 1..=batch_degree {
            factorials[i] = factorials[i - 1] * NativeF::from_usize(i);
        }
        let invfact = batch_multiplicative_inverse(&factorials);

        let r = transcript.sample_ext();
        let mut pref_product = vec![NativeEF::ONE; batch_degree + 1];
        let mut suf_product = vec![NativeEF::ONE; batch_degree + 1];
        for i in 0..batch_degree {
            pref_product[i + 1] = pref_product[i] * (r - NativeEF::from_usize(i));
            suf_product[i + 1] = suf_product[i] * (NativeEF::from_usize(batch_degree - i) - r);
        }

        cur_sum = (0..=batch_degree)
            .map(|i| {
                let eval_i = if i == 0 { s_0 } else { batch_s_evals[i - 1] };
                eval_i
                    * pref_product[i]
                    * suf_product[batch_degree - i]
                    * invfact[i]
                    * invfact[batch_degree - i]
            })
            .sum::<NativeEF>();

        rs.push(r);
    }

    let mut stacked_idx = 0usize;
    let mut eq_3b_per_trace = Vec::with_capacity(n_per_trace.len());
    for (trace_idx, &n) in n_per_trace.iter().enumerate() {
        let air_idx = trace_id_to_air_id[trace_idx];
        let interactions = &mvk0.per_air[air_idx].symbolic_constraints.interactions;
        if interactions.is_empty() {
            eq_3b_per_trace.push(vec![]);
            continue;
        }

        let n_lift = n.max(0) as usize;
        let mut b_vec = vec![NativeF::ZERO; n_logup - n_lift];
        let mut eq_3b = Vec::with_capacity(interactions.len());
        for _ in 0..interactions.len() {
            let mut b_int = stacked_idx >> (l_skip + n_lift);
            for b in &mut b_vec {
                *b = NativeF::from_bool((b_int & 1) == 1);
                b_int >>= 1;
            }
            stacked_idx += 1 << (l_skip + n_lift);
            eq_3b.push(eval_eq_mle(&xi[l_skip + n_lift..l_skip + n_logup], &b_vec));
        }
        eq_3b_per_trace.push(eq_3b);
    }

    let mut eq_ns = vec![NativeEF::ONE; n_max + 1];
    let mut eq_sharp_ns = vec![NativeEF::ONE; n_max + 1];
    eq_ns[0] = eval_eq_uni(l_skip, xi[0], r_0);
    eq_sharp_ns[0] = eval_eq_sharp_uni(omega_skip_pows, &xi[..l_skip], r_0);
    for (i, r) in rs.iter().enumerate().skip(1) {
        let eq_mle = eval_eq_mle(&[xi[l_skip + i - 1]], slice::from_ref(r));
        eq_ns[i] = eq_ns[i - 1] * eq_mle;
        eq_sharp_ns[i] = eq_sharp_ns[i - 1] * eq_mle;
    }
    let mut r_rev_prod = rs[n_max];
    for i in (0..n_max).rev() {
        eq_ns[i] *= r_rev_prod;
        eq_sharp_ns[i] *= r_rev_prod;
        r_rev_prod *= rs[i];
    }

    let mut interactions_evals = Vec::new();
    let mut constraints_evals = Vec::new();
    let need_rot_per_trace = trace_id_to_air_id
        .iter()
        .map(|&air_idx| mvk0.per_air[air_idx].params.need_rot)
        .collect::<Vec<_>>();

    for (trace_idx, air_openings) in column_openings.iter().enumerate() {
        let need_rot = need_rot_per_trace[trace_idx];
        for (claim, claim_rot) in column_openings_by_rot(&air_openings[0], need_rot) {
            transcript.observe_ext(claim);
            transcript.observe_ext(claim_rot);
        }
    }

    for (trace_idx, air_openings) in column_openings.iter().enumerate() {
        let air_idx = trace_id_to_air_id[trace_idx];
        let vk = &mvk0.per_air[air_idx];
        let n = n_per_trace[trace_idx];
        let n_lift = n.max(0) as usize;
        let need_rot = need_rot_per_trace[trace_idx];

        for claims in air_openings.iter().skip(1) {
            for (claim, claim_rot) in column_openings_by_rot(claims, need_rot) {
                transcript.observe_ext(claim);
                transcript.observe_ext(claim_rot);
            }
        }

        let has_preprocessed = vk.preprocessed_data.is_some();
        let common_main = column_openings_by_rot(&air_openings[0], need_rot).collect::<Vec<_>>();
        let preprocessed = has_preprocessed
            .then(|| column_openings_by_rot(&air_openings[1], need_rot).collect::<Vec<_>>());
        let cached_idx = 1 + has_preprocessed as usize;
        let mut partitioned_main = air_openings[cached_idx..]
            .iter()
            .map(|opening| column_openings_by_rot(opening, need_rot).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        partitioned_main.push(common_main);
        let part_main_slices = partitioned_main
            .iter()
            .map(|x| x.as_slice())
            .collect::<Vec<_>>();

        let (l, rs_n, norm_factor) = if n.is_negative() {
            (
                l_skip.wrapping_add_signed(n),
                &[rs[0].exp_power_of_2((-n) as usize)] as &[_],
                NativeF::from_usize(1usize << n.unsigned_abs()).inverse(),
            )
        } else {
            (l_skip, &rs[..=n_lift], NativeF::ONE)
        };

        let evaluator = VerifierConstraintEvaluator::new(
            preprocessed.as_deref(),
            &part_main_slices,
            &public_values[air_idx],
            rs_n,
            l,
        );
        if validate_symbolic_nodes(
            air_idx,
            &vk.symbolic_constraints.constraints.nodes,
            preprocessed.as_deref(),
            &part_main_slices,
            &public_values[air_idx],
        )
        .is_err()
        {
            return Err(NativeBatchConstraintError::InconsistentClaims);
        }

        let nodes = evaluator.eval_nodes(&vk.symbolic_constraints.constraints.nodes);
        let expr = vk
            .symbolic_constraints
            .constraints
            .constraint_idx
            .iter()
            .zip(lambda.powers())
            .fold(NativeEF::ZERO, |acc, (&idx, lambda_pow)| {
                acc + nodes[idx] * lambda_pow
            });
        constraints_evals.push(eq_ns[n_lift] * expr);

        let symbolic_constraints = SymbolicConstraints::from(&vk.symbolic_constraints);
        let interactions = &symbolic_constraints.interactions;
        let mut cur_interactions_evals = Vec::with_capacity(interactions.len());
        for interaction in interactions {
            let num = evaluator.eval_expr(&interaction.count);
            let denom = interaction
                .message
                .iter()
                .map(|expr| evaluator.eval_expr(expr))
                .chain(core::iter::once(NativeEF::from_u16(
                    interaction.bus_index + 1,
                )))
                .zip(beta_logup.powers())
                .fold(NativeEF::ZERO, |acc, (x, y)| acc + x * y);
            cur_interactions_evals.push((num, denom));
        }

        let eq_3bs = &eq_3b_per_trace[trace_idx];
        let mut num = NativeEF::ZERO;
        let mut denom = NativeEF::ZERO;
        for (&eq_3b, (n_eval, d_eval)) in eq_3bs.iter().zip(cur_interactions_evals.iter()) {
            num += eq_3b * *n_eval;
            denom += eq_3b * *d_eval;
        }

        interactions_evals.push(num * norm_factor * eq_sharp_ns[n_lift]);
        interactions_evals.push(denom * eq_sharp_ns[n_lift]);
    }

    let evaluated_claim = interactions_evals
        .iter()
        .chain(constraints_evals.iter())
        .zip(mu.powers())
        .map(|(x, y)| *x * y)
        .sum::<NativeEF>();
    let consistency_residual = cur_sum - evaluated_claim;
    if cur_sum != evaluated_claim {
        return Err(NativeBatchConstraintError::InconsistentClaims);
    }

    Ok(BatchIntermediates {
        trace_id_to_air_id: trace_id_to_air_id.to_vec(),
        n_per_trace: n_per_trace.to_vec(),
        l_skip,
        total_interactions,
        n_logup,
        n_max,
        batch_degree,
        trace_has_preprocessed: trace_id_to_air_id
            .iter()
            .map(|&air_id| mvk0.per_air[air_id].preprocessed_data.is_some())
            .collect::<Vec<_>>(),
        trace_constraint_nodes: trace_id_to_air_id
            .iter()
            .map(|&air_id| {
                mvk0.per_air[air_id]
                    .symbolic_constraints
                    .constraints
                    .nodes
                    .clone()
            })
            .collect::<Vec<_>>(),
        trace_constraint_indices: trace_id_to_air_id
            .iter()
            .map(|&air_id| {
                mvk0.per_air[air_id]
                    .symbolic_constraints
                    .constraints
                    .constraint_idx
                    .clone()
            })
            .collect::<Vec<_>>(),
        trace_interactions: trace_id_to_air_id
            .iter()
            .map(|&air_id| {
                mvk0.per_air[air_id]
                    .symbolic_constraints
                    .interactions
                    .clone()
            })
            .collect::<Vec<_>>(),
        public_values: public_values
            .iter()
            .map(|values| {
                values
                    .iter()
                    .map(|value| value.as_canonical_u64())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
        logup_pow_bits,
        logup_pow_witness: gkr_proof.logup_pow_witness.as_canonical_u64(),
        logup_pow_sampled_bits,
        logup_pow_witness_ok,
        gkr_q0_claim: (total_interactions > 0).then(|| ext_to_coeffs(gkr_proof.q0_claim)),
        gkr_claims_per_layer: gkr_proof
            .claims_per_layer
            .iter()
            .map(|claims| {
                [
                    ext_to_coeffs(claims.p_xi_0),
                    ext_to_coeffs(claims.q_xi_0),
                    ext_to_coeffs(claims.p_xi_1),
                    ext_to_coeffs(claims.q_xi_1),
                ]
            })
            .collect::<Vec<_>>(),
        gkr_sumcheck_polys: gkr_proof
            .sumcheck_polys
            .iter()
            .map(|poly| {
                poly.iter()
                    .flat_map(|evals| evals.iter().copied().map(ext_to_coeffs))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
        numerator_term_per_air: numerator_term_per_air
            .iter()
            .copied()
            .map(ext_to_coeffs)
            .collect::<Vec<_>>(),
        denominator_term_per_air: denominator_term_per_air
            .iter()
            .copied()
            .map(ext_to_coeffs)
            .collect::<Vec<_>>(),
        univariate_round_coeffs: univariate_round_coeffs
            .iter()
            .copied()
            .map(ext_to_coeffs)
            .collect::<Vec<_>>(),
        sumcheck_round_polys: sumcheck_round_polys
            .iter()
            .map(|poly| poly.iter().copied().map(ext_to_coeffs).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        column_openings: column_openings
            .iter()
            .map(|per_air| {
                per_air
                    .iter()
                    .map(|part| part.iter().copied().map(ext_to_coeffs).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
        column_openings_need_rot: trace_id_to_air_id
            .iter()
            .map(|&air_id| {
                let need_rot = mvk0.per_air[air_id].params.need_rot;
                vec![need_rot; mvk0.per_air[air_id].num_parts()]
            })
            .collect::<Vec<_>>(),
        column_opening_expected_widths: trace_id_to_air_id
            .iter()
            .map(|&air_id| {
                let air = &mvk0.per_air[air_id];
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
            .collect::<Vec<_>>(),
        gkr_numerator_residual: ext_to_coeffs(gkr_numerator_residual),
        gkr_denominator_residual: ext_to_coeffs(gkr_denominator_residual),
        gkr_denominator_claim: ext_to_coeffs(q_xi_claim),
        alpha_logup: ext_to_coeffs(alpha_logup),
        beta_logup: ext_to_coeffs(beta_logup),
        gkr_non_xi_samples,
        gkr_xi_sample_order,
        xi: xi.iter().copied().map(ext_to_coeffs).collect(),
        lambda: ext_to_coeffs(lambda),
        mu: ext_to_coeffs(mu),
        sum_claim: ext_to_coeffs(sum_claim),
        sum_univ_domain_s_0: ext_to_coeffs(sum_univ_domain_s_0),
        consistency_lhs: ext_to_coeffs(cur_sum),
        consistency_rhs: ext_to_coeffs(evaluated_claim),
        consistency_residual: ext_to_coeffs(consistency_residual),
        r: rs.into_iter().map(ext_to_coeffs).collect(),
    })
}

pub fn derive_batch_intermediates(
    config: &NativeConfig,
    mvk: &MultiStarkVerifyingKey<NativeConfig>,
    proof: &Proof<NativeConfig>,
) -> Result<BatchIntermediates, BatchConstraintError> {
    if config.params() != &mvk.inner.params {
        return Err(BatchConstraintError::SystemParamsMismatch);
    }

    let mvk0 = &mvk.inner;
    let trace_id_to_air_id = compute_trace_id_to_air_id(mvk0, proof);
    enforce_trace_height_constraints(mvk0, proof, &trace_id_to_air_id)?;

    let mut transcript = default_transcript();
    observe_preamble(&mut transcript, mvk0, mvk.pre_hash, proof);
    derive_proof_shape_rules(mvk0, proof)?;

    let l_skip = mvk0.params.l_skip;
    let n_per_trace = trace_id_to_air_id
        .iter()
        .map(|&air_id| {
            Ok(proof.trace_vdata[air_id]
                .as_ref()
                .ok_or(BatchConstraintError::MissingTraceVData { air_id })?
                .log_height as isize
                - l_skip as isize)
        })
        .collect::<Result<Vec<_>, BatchConstraintError>>()?;

    let omega_skip = NativeF::two_adic_generator(l_skip);
    let omega_skip_pows: Vec<_> = omega_skip.powers().take(1 << l_skip).collect();

    derive_batch_intermediates_with_inputs(
        &mut transcript,
        mvk0,
        &proof.public_values,
        &proof.gkr_proof,
        &proof.batch_constraint_proof,
        &trace_id_to_air_id,
        &n_per_trace,
        &omega_skip_pows,
    )
    .map_err(Into::into)
}

fn assign_bool(ctx: &mut Context<Fr>, range: &RangeChip<Fr>, value: bool) -> AssignedValue<Fr> {
    let bit = ctx.load_witness(Fr::from(value as u64));
    range.gate().assert_bit(ctx, bit);
    bit
}

fn assign_ext(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    baby_bear: &BabyBearArithmeticGadgets,
    coeffs: [u64; BABY_BEAR_EXT_DEGREE],
) -> BabyBearExtVar {
    baby_bear.load_ext_witness(ctx, range, coeffs)
}

pub(crate) fn ext_mul_base_const(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    baby_bear: &BabyBearArithmeticGadgets,
    value: &BabyBearExtVar,
    constant: u64,
) -> BabyBearExtVar {
    let coeffs =
        core::array::from_fn(|idx| baby_bear.mul_const(ctx, range, &value.coeffs[idx], constant));
    BabyBearExtVar { coeffs }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct RecordedExtBaseConst {
    pub constant: u64,
    pub cell: AssignedValue<Fr>,
}

#[cfg(test)]
thread_local! {
    static RECORDED_EXT_BASE_CONSTS: RefCell<Vec<RecordedExtBaseConst>> = const { RefCell::new(Vec::new()) };
}

#[cfg(test)]
pub(crate) fn clear_recorded_ext_base_consts() {
    RECORDED_EXT_BASE_CONSTS.with(|records| records.borrow_mut().clear());
}

#[cfg(test)]
pub(crate) fn take_recorded_ext_base_consts() -> Vec<RecordedExtBaseConst> {
    RECORDED_EXT_BASE_CONSTS.with(|records| records.borrow_mut().drain(..).collect())
}

pub(crate) fn ext_from_base_const(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    baby_bear: &BabyBearArithmeticGadgets,
    constant: u64,
) -> BabyBearExtVar {
    let c0 = baby_bear.load_constant(ctx, range, constant);
    #[cfg(test)]
    RECORDED_EXT_BASE_CONSTS.with(|records| {
        records.borrow_mut().push(RecordedExtBaseConst {
            constant,
            cell: c0.cell,
        });
    });
    BabyBearExtVar {
        coeffs: core::array::from_fn(|idx| {
            if idx == 0 {
                c0.clone()
            } else {
                baby_bear.zero(ctx, range)
            }
        }),
    }
}

fn eval_ext_poly_horner(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    baby_bear: &BabyBearArithmeticGadgets,
    coeffs: &[BabyBearExtVar],
    point: &BabyBearExtVar,
) -> BabyBearExtVar {
    let mut acc = baby_bear.ext_zero(ctx, range);
    for coeff in coeffs.iter().rev() {
        acc = baby_bear.ext_mul(ctx, range, &acc, point);
        acc = baby_bear.ext_add(ctx, range, &acc, coeff);
    }
    acc
}

fn eval_lagrange_on_integer_grid(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    baby_bear: &BabyBearArithmeticGadgets,
    point: &BabyBearExtVar,
    evals: &[BabyBearExtVar],
) -> BabyBearExtVar {
    let n = evals.len().saturating_sub(1);
    let mut acc = baby_bear.ext_zero(ctx, range);
    for (i, eval_i) in evals.iter().enumerate() {
        let mut basis = ext_from_base_const(ctx, range, baby_bear, 1);
        let mut denom = NativeF::ONE;
        for j in 0..=n {
            if i == j {
                continue;
            }
            let x_j = ext_from_base_const(ctx, range, baby_bear, j as u64);
            let x_minus_j = baby_bear.ext_sub(ctx, range, point, &x_j);
            basis = baby_bear.ext_mul(ctx, range, &basis, &x_minus_j);

            let diff = if i >= j {
                NativeF::from_usize(i - j)
            } else {
                -NativeF::from_usize(j - i)
            };
            denom *= diff;
        }
        let denom_inv = denom.inverse().as_canonical_u64();
        let basis = ext_mul_base_const(ctx, range, baby_bear, &basis, denom_inv);
        let term = baby_bear.ext_mul(ctx, range, eval_i, &basis);
        acc = baby_bear.ext_add(ctx, range, &acc, &term);
    }
    acc
}

fn ext_from_base_var(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    baby_bear: &BabyBearArithmeticGadgets,
    value: &crate::gadgets::baby_bear::BabyBearVar,
) -> BabyBearExtVar {
    let zero = baby_bear.zero(ctx, range);
    BabyBearExtVar {
        coeffs: core::array::from_fn(|idx| {
            if idx == 0 {
                value.clone()
            } else {
                zero.clone()
            }
        }),
    }
}

fn ext_neg(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    baby_bear: &BabyBearArithmeticGadgets,
    value: &BabyBearExtVar,
) -> BabyBearExtVar {
    let zero = baby_bear.ext_zero(ctx, range);
    baby_bear.ext_sub(ctx, range, &zero, value)
}

pub(crate) fn ext_pow_power_of_two(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    baby_bear: &BabyBearArithmeticGadgets,
    value: &BabyBearExtVar,
    exp_power: usize,
) -> BabyBearExtVar {
    let mut acc = value.clone();
    for _ in 0..exp_power {
        acc = baby_bear.ext_mul(ctx, range, &acc, &acc);
    }
    acc
}

fn progression_exp_2_assigned(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    baby_bear: &BabyBearArithmeticGadgets,
    m: &BabyBearExtVar,
    l: usize,
) -> BabyBearExtVar {
    let mut pow = m.clone();
    let mut sum = ext_from_base_const(ctx, range, baby_bear, 1);
    let one = ext_from_base_const(ctx, range, baby_bear, 1);
    for _ in 0..l {
        let one_plus_pow = baby_bear.ext_add(ctx, range, &one, &pow);
        sum = baby_bear.ext_mul(ctx, range, &sum, &one_plus_pow);
        pow = baby_bear.ext_mul(ctx, range, &pow, &pow);
    }
    sum
}

pub(crate) fn eval_eq_mle_assigned(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    baby_bear: &BabyBearArithmeticGadgets,
    x: &[BabyBearExtVar],
    y: &[BabyBearExtVar],
) -> BabyBearExtVar {
    assert_eq!(x.len(), y.len(), "eq_mle vector length mismatch");
    let one = ext_from_base_const(ctx, range, baby_bear, 1);
    let mut acc = one.clone();
    for (x_i, y_i) in x.iter().zip(y.iter()) {
        let xy = baby_bear.ext_mul(ctx, range, x_i, y_i);
        let two_xy = ext_mul_base_const(ctx, range, baby_bear, &xy, 2);
        let one_minus_y = baby_bear.ext_sub(ctx, range, &one, y_i);
        let one_minus_y_minus_x = baby_bear.ext_sub(ctx, range, &one_minus_y, x_i);
        let factor = baby_bear.ext_add(ctx, range, &one_minus_y_minus_x, &two_xy);
        acc = baby_bear.ext_mul(ctx, range, &acc, &factor);
    }
    acc
}

pub(crate) fn eval_eq_mle_binary_assigned(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    baby_bear: &BabyBearArithmeticGadgets,
    x: &[BabyBearExtVar],
    y_bits: &[bool],
) -> BabyBearExtVar {
    assert_eq!(
        x.len(),
        y_bits.len(),
        "eq_mle binary vector length mismatch",
    );
    let one = ext_from_base_const(ctx, range, baby_bear, 1);
    let mut acc = one.clone();
    for (x_i, bit) in x.iter().zip(y_bits.iter().copied()) {
        let factor = if bit {
            x_i.clone()
        } else {
            baby_bear.ext_sub(ctx, range, &one, x_i)
        };
        acc = baby_bear.ext_mul(ctx, range, &acc, &factor);
    }
    acc
}

pub(crate) fn eval_eq_uni_assigned(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    baby_bear: &BabyBearArithmeticGadgets,
    l_skip: usize,
    x: &BabyBearExtVar,
    y: &BabyBearExtVar,
) -> BabyBearExtVar {
    let one = ext_from_base_const(ctx, range, baby_bear, 1);
    let mut res = one.clone();
    let mut x_pow = x.clone();
    let mut y_pow = y.clone();
    for _ in 0..l_skip {
        let x_plus_y = baby_bear.ext_add(ctx, range, &x_pow, &y_pow);
        let x_minus_one = baby_bear.ext_sub(ctx, range, &x_pow, &one);
        let y_minus_one = baby_bear.ext_sub(ctx, range, &y_pow, &one);
        let correction = baby_bear.ext_mul(ctx, range, &x_minus_one, &y_minus_one);
        let scaled_res = baby_bear.ext_mul(ctx, range, &x_plus_y, &res);
        res = baby_bear.ext_add(ctx, range, &scaled_res, &correction);
        x_pow = baby_bear.ext_mul(ctx, range, &x_pow, &x_pow);
        y_pow = baby_bear.ext_mul(ctx, range, &y_pow, &y_pow);
    }
    let half_pow_l = NativeF::ONE
        .halve()
        .exp_u64(l_skip as u64)
        .as_canonical_u64();
    ext_mul_base_const(ctx, range, baby_bear, &res, half_pow_l)
}

pub(crate) fn eval_eq_uni_at_one_assigned(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    baby_bear: &BabyBearArithmeticGadgets,
    l_skip: usize,
    x: &BabyBearExtVar,
) -> BabyBearExtVar {
    let one = ext_from_base_const(ctx, range, baby_bear, 1);
    let mut res = one.clone();
    let mut x_pow = x.clone();
    for _ in 0..l_skip {
        let x_plus_one = baby_bear.ext_add(ctx, range, &x_pow, &one);
        res = baby_bear.ext_mul(ctx, range, &res, &x_plus_one);
        x_pow = baby_bear.ext_mul(ctx, range, &x_pow, &x_pow);
    }
    let half_pow_l = NativeF::ONE
        .halve()
        .exp_u64(l_skip as u64)
        .as_canonical_u64();
    ext_mul_base_const(ctx, range, baby_bear, &res, half_pow_l)
}

fn eval_eq_sharp_uni_assigned(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    baby_bear: &BabyBearArithmeticGadgets,
    omega_skip_pows: &[NativeF],
    xi_1: &[BabyBearExtVar],
    z: &BabyBearExtVar,
) -> BabyBearExtVar {
    let one = ext_from_base_const(ctx, range, baby_bear, 1);
    let mut eq_xi_evals = vec![baby_bear.ext_zero(ctx, range); 1usize << xi_1.len()];
    eq_xi_evals[0] = one.clone();

    // Match `evals_eq_hypercube_serial` ordering from the native verifier:
    // mask bit `i` corresponds to `xi_1[i]`.
    for (i, xi) in xi_1.iter().enumerate() {
        let span = 1usize << i;
        let one_minus_xi = baby_bear.ext_sub(ctx, range, &one, xi);
        for idx in 0..span {
            let prev = eq_xi_evals[idx].clone();
            let lo = baby_bear.ext_mul(ctx, range, &prev, &one_minus_xi);
            let hi = baby_bear.ext_mul(ctx, range, &prev, xi);
            eq_xi_evals[idx] = lo;
            eq_xi_evals[span + idx] = hi;
        }
    }

    assert_eq!(
        eq_xi_evals.len(),
        omega_skip_pows.len(),
        "eq_sharp eval table width mismatch",
    );

    let mut res = baby_bear.ext_zero(ctx, range);
    let l_skip = xi_1.len();
    for (omega_pow, eq_xi_eval) in omega_skip_pows.iter().zip(eq_xi_evals.iter()) {
        let omega_ext = ext_from_base_const(ctx, range, baby_bear, omega_pow.as_canonical_u64());
        let eq_uni = eval_eq_uni_assigned(ctx, range, baby_bear, l_skip, z, &omega_ext);
        let term = baby_bear.ext_mul(ctx, range, &eq_uni, eq_xi_eval);
        res = baby_bear.ext_add(ctx, range, &res, &term);
    }
    res
}

pub(crate) fn eval_eq_prism_assigned(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    baby_bear: &BabyBearArithmeticGadgets,
    l_skip: usize,
    x: &[BabyBearExtVar],
    y: &[BabyBearExtVar],
) -> BabyBearExtVar {
    assert!(
        !x.is_empty() && !y.is_empty(),
        "eq_prism vectors must be non-empty",
    );
    let eq_uni = eval_eq_uni_assigned(ctx, range, baby_bear, l_skip, &x[0], &y[0]);
    let eq_mle = eval_eq_mle_assigned(ctx, range, baby_bear, &x[1..], &y[1..]);
    baby_bear.ext_mul(ctx, range, &eq_uni, &eq_mle)
}

fn eval_eq_rot_cube_assigned(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    baby_bear: &BabyBearArithmeticGadgets,
    x: &[BabyBearExtVar],
    y: &[BabyBearExtVar],
) -> (BabyBearExtVar, BabyBearExtVar) {
    assert_eq!(x.len(), y.len(), "eq_rot_cube vector length mismatch");
    let one = ext_from_base_const(ctx, range, baby_bear, 1);
    let mut rot = one.clone();
    let mut eq = one.clone();
    for i in (0..x.len()).rev() {
        let one_minus_y = baby_bear.ext_sub(ctx, range, &one, &y[i]);
        let one_minus_x = baby_bear.ext_sub(ctx, range, &one, &x[i]);
        let x_times = baby_bear.ext_mul(ctx, range, &x[i], &one_minus_y);
        let term1 = baby_bear.ext_mul(ctx, range, &x_times, &eq);
        let y_times = baby_bear.ext_mul(ctx, range, &one_minus_x, &y[i]);
        let term2 = baby_bear.ext_mul(ctx, range, &y_times, &rot);
        rot = baby_bear.ext_add(ctx, range, &term1, &term2);

        let xy = baby_bear.ext_mul(ctx, range, &x[i], &y[i]);
        let one_minus_xy = baby_bear.ext_mul(ctx, range, &one_minus_x, &one_minus_y);
        let eq_factor = baby_bear.ext_add(ctx, range, &xy, &one_minus_xy);
        eq = baby_bear.ext_mul(ctx, range, &eq, &eq_factor);
    }
    (eq, rot)
}

pub(crate) fn eval_rot_kernel_prism_assigned(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    baby_bear: &BabyBearArithmeticGadgets,
    l_skip: usize,
    x: &[BabyBearExtVar],
    y: &[BabyBearExtVar],
) -> BabyBearExtVar {
    assert!(
        !x.is_empty() && !y.is_empty(),
        "rot-kernel vectors must be non-empty",
    );
    let omega = NativeF::two_adic_generator(l_skip).as_canonical_u64();
    let y0_omega = ext_mul_base_const(ctx, range, baby_bear, &y[0], omega);
    let eq_uni_rot = eval_eq_uni_assigned(ctx, range, baby_bear, l_skip, &x[0], &y0_omega);
    let (eq_cube, rot_cube) = eval_eq_rot_cube_assigned(ctx, range, baby_bear, &x[1..], &y[1..]);
    let term_a = baby_bear.ext_mul(ctx, range, &eq_uni_rot, &eq_cube);

    let eq_uni_x_one = eval_eq_uni_at_one_assigned(ctx, range, baby_bear, l_skip, &x[0]);
    let eq_uni_y_one = eval_eq_uni_at_one_assigned(ctx, range, baby_bear, l_skip, &y0_omega);
    let rot_minus_eq = baby_bear.ext_sub(ctx, range, &rot_cube, &eq_cube);
    let eq_uni_product = baby_bear.ext_mul(ctx, range, &eq_uni_x_one, &eq_uni_y_one);
    let term_b = baby_bear.ext_mul(ctx, range, &eq_uni_product, &rot_minus_eq);
    baby_bear.ext_add(ctx, range, &term_a, &term_b)
}

fn interpolate_linear_at_01_assigned(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    baby_bear: &BabyBearArithmeticGadgets,
    eval0: &BabyBearExtVar,
    eval1: &BabyBearExtVar,
    x: &BabyBearExtVar,
) -> BabyBearExtVar {
    let delta = baby_bear.ext_sub(ctx, range, eval1, eval0);
    let scaled = baby_bear.ext_mul(ctx, range, &delta, x);
    baby_bear.ext_add(ctx, range, &scaled, eval0)
}

fn interpolate_cubic_at_0123_assigned(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    baby_bear: &BabyBearArithmeticGadgets,
    evals: [&BabyBearExtVar; 4],
    x: &BabyBearExtVar,
) -> BabyBearExtVar {
    let inv6 = NativeF::from_u64(6).inverse().as_canonical_u64();
    let s1 = baby_bear.ext_sub(ctx, range, evals[1], evals[0]);
    let s2 = baby_bear.ext_sub(ctx, range, evals[2], evals[0]);
    let s3 = baby_bear.ext_sub(ctx, range, evals[3], evals[0]);

    let s2_minus_s1 = baby_bear.ext_sub(ctx, range, &s2, &s1);
    let triple = ext_mul_base_const(ctx, range, baby_bear, &s2_minus_s1, 3);
    let d3 = baby_bear.ext_sub(ctx, range, &s3, &triple);

    let p = ext_mul_base_const(ctx, range, baby_bear, &d3, inv6);
    let s2_minus_d3 = baby_bear.ext_sub(ctx, range, &s2, &d3);
    let half = NativeF::ONE.halve().as_canonical_u64();
    let q_half = ext_mul_base_const(ctx, range, baby_bear, &s2_minus_d3, half);
    let q = baby_bear.ext_sub(ctx, range, &q_half, &s1);
    let p_plus_q = baby_bear.ext_add(ctx, range, &p, &q);
    let r = baby_bear.ext_sub(ctx, range, &s1, &p_plus_q);

    let p_mul_x = baby_bear.ext_mul(ctx, range, &p, x);
    let px_plus_q = baby_bear.ext_add(ctx, range, &p_mul_x, &q);
    let quad_mul_x = baby_bear.ext_mul(ctx, range, &px_plus_q, x);
    let quad = baby_bear.ext_add(ctx, range, &quad_mul_x, &r);
    let cubic = baby_bear.ext_mul(ctx, range, &quad, x);
    baby_bear.ext_add(ctx, range, &cubic, evals[0])
}

#[derive(Clone)]
struct AssignedViewPair {
    local: BabyBearExtVar,
    next: BabyBearExtVar,
}

struct AssignedConstraintEvaluator<'a> {
    preprocessed: Option<&'a [AssignedViewPair]>,
    partitioned_main: &'a [Vec<AssignedViewPair>],
    is_first_row: BabyBearExtVar,
    is_last_row: BabyBearExtVar,
    public_values: &'a [crate::gadgets::baby_bear::BabyBearVar],
}

impl AssignedConstraintEvaluator<'_> {
    fn reject_malformed_symbolic_var(
        &self,
        ctx: &mut Context<Fr>,
        range: &RangeChip<Fr>,
        baby_bear: &BabyBearArithmeticGadgets,
    ) -> BabyBearExtVar {
        let zero = baby_bear.ext_zero(ctx, range);
        let one = ext_from_base_const(ctx, range, baby_bear, 1);
        baby_bear.assert_ext_equal(ctx, &zero, &one);
        zero
    }

    fn eval_var(
        &self,
        ctx: &mut Context<Fr>,
        range: &RangeChip<Fr>,
        baby_bear: &BabyBearArithmeticGadgets,
        symbolic_var: SymbolicVariable<NativeF>,
    ) -> BabyBearExtVar {
        let index = symbolic_var.index;
        match symbolic_var.entry {
            Entry::Preprocessed { offset } => {
                let Some(vp) = self.preprocessed else {
                    return self.reject_malformed_symbolic_var(ctx, range, baby_bear);
                };
                let Some(value) = vp.get(index) else {
                    return self.reject_malformed_symbolic_var(ctx, range, baby_bear);
                };
                if offset == 0 {
                    value.local.clone()
                } else {
                    value.next.clone()
                }
            }
            Entry::Main { part_index, offset } => {
                let Some(vp) = self.partitioned_main.get(part_index) else {
                    return self.reject_malformed_symbolic_var(ctx, range, baby_bear);
                };
                let Some(value) = vp.get(index) else {
                    return self.reject_malformed_symbolic_var(ctx, range, baby_bear);
                };
                if offset == 0 {
                    value.local.clone()
                } else {
                    value.next.clone()
                }
            }
            Entry::Public => {
                let Some(value) = self.public_values.get(index) else {
                    return self.reject_malformed_symbolic_var(ctx, range, baby_bear);
                };
                ext_from_base_var(ctx, range, baby_bear, value)
            }
            _ => self.reject_malformed_symbolic_var(ctx, range, baby_bear),
        }
    }
}

fn eval_symbolic_nodes_assigned(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    baby_bear: &BabyBearArithmeticGadgets,
    evaluator: &AssignedConstraintEvaluator<'_>,
    nodes: &[SymbolicExpressionNode<NativeF>],
) -> Vec<BabyBearExtVar> {
    let one = ext_from_base_const(ctx, range, baby_bear, 1);
    let mut exprs: Vec<BabyBearExtVar> = Vec::with_capacity(nodes.len());
    for node in nodes {
        let expr = match node {
            SymbolicExpressionNode::Variable(var) => {
                evaluator.eval_var(ctx, range, baby_bear, *var)
            }
            SymbolicExpressionNode::Constant(c) => {
                ext_from_base_const(ctx, range, baby_bear, c.as_canonical_u64())
            }
            SymbolicExpressionNode::Add {
                left_idx,
                right_idx,
                ..
            } => baby_bear.ext_add(ctx, range, &exprs[*left_idx], &exprs[*right_idx]),
            SymbolicExpressionNode::Sub {
                left_idx,
                right_idx,
                ..
            } => baby_bear.ext_sub(ctx, range, &exprs[*left_idx], &exprs[*right_idx]),
            SymbolicExpressionNode::Neg { idx, .. } => ext_neg(ctx, range, baby_bear, &exprs[*idx]),
            SymbolicExpressionNode::Mul {
                left_idx,
                right_idx,
                ..
            } => baby_bear.ext_mul(ctx, range, &exprs[*left_idx], &exprs[*right_idx]),
            SymbolicExpressionNode::IsFirstRow => evaluator.is_first_row.clone(),
            SymbolicExpressionNode::IsLastRow => evaluator.is_last_row.clone(),
            SymbolicExpressionNode::IsTransition => {
                baby_bear.ext_sub(ctx, range, &one, &evaluator.is_last_row)
            }
        };
        exprs.push(expr);
    }
    exprs
}

fn column_openings_by_rot_assigned(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    baby_bear: &BabyBearArithmeticGadgets,
    openings: &[BabyBearExtVar],
    need_rot: bool,
) -> Vec<AssignedViewPair> {
    shared_math::column_openings_by_rot_assigned(ctx, range, baby_bear, openings, need_rot)
        .into_iter()
        .map(|(local, next)| AssignedViewPair { local, next })
        .collect::<Vec<_>>()
}

// Unchecked/internal assignment path. External callers should use strict derive+constrain APIs.
pub(crate) fn constrain_batch_intermediates_unchecked(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    actual: &BatchIntermediates,
) -> AssignedBatchIntermediates {
    assert!(!actual.r.is_empty(), "batch challenges must be non-empty");
    assert_eq!(
        actual.n_per_trace.len(),
        actual.trace_id_to_air_id.len(),
        "n_per_trace must align with trace_id_to_air_id",
    );
    assert_eq!(
        actual.trace_has_preprocessed.len(),
        actual.trace_id_to_air_id.len(),
        "trace preprocessed flags must align with trace count",
    );
    assert_eq!(
        actual.trace_constraint_nodes.len(),
        actual.trace_id_to_air_id.len(),
        "trace constraint node sets must align with trace count",
    );
    assert_eq!(
        actual.trace_constraint_indices.len(),
        actual.trace_id_to_air_id.len(),
        "trace constraint index sets must align with trace count",
    );
    assert_eq!(
        actual.trace_interactions.len(),
        actual.trace_id_to_air_id.len(),
        "trace interaction sets must align with trace count",
    );

    let gate = range.gate();
    let baby_bear = BabyBearArithmeticGadgets;

    let trace_id_to_air_id = actual
        .trace_id_to_air_id
        .iter()
        .map(|&actual_air_id| {
            let air_id = assign_and_range_usize(ctx, range, actual_air_id);
            air_id
        })
        .collect();

    let total_interactions = assign_and_range_u64(ctx, range, actual.total_interactions);

    let n_logup = assign_and_range_usize(ctx, range, actual.n_logup);

    let n_max = assign_and_range_usize(ctx, range, actual.n_max);
    gate.assert_is_const(
        ctx,
        &n_max,
        &Fr::from(usize_to_u64(actual.r.len().saturating_sub(1))),
    );
    let batch_degree = assign_and_range_usize(ctx, range, actual.batch_degree);

    let logup_pow_bits = assign_and_range_usize(ctx, range, actual.logup_pow_bits);
    let logup_pow_sampled_bits = assign_and_range_u64(ctx, range, actual.logup_pow_sampled_bits);
    if actual.logup_pow_bits > 0 {
        range.range_check(ctx, logup_pow_sampled_bits, actual.logup_pow_bits);
    } else {
        gate.assert_is_const(ctx, &logup_pow_sampled_bits, &Fr::from(0u64));
    }

    let logup_pow_witness_ok = assign_bool(ctx, range, actual.logup_pow_witness_ok);
    let logup_pow_is_zero = gate.is_zero(ctx, logup_pow_sampled_bits);
    ctx.constrain_equal(&logup_pow_witness_ok, &logup_pow_is_zero);
    gate.assert_is_const(ctx, &logup_pow_witness_ok, &Fr::from(1u64));
    let gkr_q0_claim = if actual.total_interactions == 0 {
        assert!(
            actual.gkr_q0_claim.is_none(),
            "zero-interaction branch must not include GKR q0 claim witness",
        );
        None
    } else {
        let q0_claim = actual
            .gkr_q0_claim
            .expect("non-zero interaction branch must include GKR q0 claim witness");
        Some(assign_ext(ctx, range, &baby_bear, q0_claim))
    };
    let gkr_claims_per_layer = actual
        .gkr_claims_per_layer
        .iter()
        .map(|claims| {
            claims
                .iter()
                .map(|&coeffs| assign_ext(ctx, range, &baby_bear, coeffs))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let gkr_sumcheck_polys = actual
        .gkr_sumcheck_polys
        .iter()
        .map(|poly| {
            poly.iter()
                .map(|&coeffs| assign_ext(ctx, range, &baby_bear, coeffs))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let numerator_term_per_air = actual
        .numerator_term_per_air
        .iter()
        .map(|&coeffs| assign_ext(ctx, range, &baby_bear, coeffs))
        .collect::<Vec<_>>();
    let denominator_term_per_air = actual
        .denominator_term_per_air
        .iter()
        .map(|&coeffs| assign_ext(ctx, range, &baby_bear, coeffs))
        .collect::<Vec<_>>();
    let univariate_round_coeffs = actual
        .univariate_round_coeffs
        .iter()
        .map(|&coeffs| assign_ext(ctx, range, &baby_bear, coeffs))
        .collect::<Vec<_>>();
    let sumcheck_round_polys = actual
        .sumcheck_round_polys
        .iter()
        .map(|poly| {
            poly.iter()
                .map(|&coeffs| assign_ext(ctx, range, &baby_bear, coeffs))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let trace_count = assign_and_range_usize(ctx, range, actual.trace_id_to_air_id.len());
    let numerator_len = assign_and_range_usize(ctx, range, numerator_term_per_air.len());
    let denominator_len = assign_and_range_usize(ctx, range, denominator_term_per_air.len());
    ctx.constrain_equal(&numerator_len, &denominator_len);
    ctx.constrain_equal(&numerator_len, &trace_count);

    let l_skip_width = if actual.l_skip >= usize::BITS as usize {
        0usize
    } else {
        1usize << actual.l_skip
    };
    let expected_univariate_len = actual
        .batch_degree
        .saturating_mul(l_skip_width.saturating_sub(1))
        .saturating_add(1);
    let univariate_len = assign_and_range_usize(ctx, range, univariate_round_coeffs.len());
    let expected_univariate_len_cell = assign_and_range_usize(ctx, range, expected_univariate_len);
    ctx.constrain_equal(&univariate_len, &expected_univariate_len_cell);

    let sumcheck_round_count = assign_and_range_usize(ctx, range, sumcheck_round_polys.len());
    ctx.constrain_equal(&sumcheck_round_count, &n_max);
    for round in &sumcheck_round_polys {
        let round_width = assign_and_range_usize(ctx, range, round.len());
        ctx.constrain_equal(&round_width, &batch_degree);
    }
    let column_openings = actual
        .column_openings
        .iter()
        .map(|per_air| {
            per_air
                .iter()
                .map(|part| {
                    part.iter()
                        .map(|&coeffs| assign_ext(ctx, range, &baby_bear, coeffs))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        actual.column_openings_need_rot.len(),
        actual.column_openings.len(),
        "column-opening rotation flags must align with opening vectors",
    );
    assert_eq!(
        actual.column_opening_expected_widths.len(),
        actual.column_openings.len(),
        "column-opening expected-width schedule must align with opening vectors",
    );
    for (trace_idx, (openings, expected_widths)) in actual
        .column_openings
        .iter()
        .zip(actual.column_opening_expected_widths.iter())
        .enumerate()
    {
        let part_count = assign_and_range_usize(ctx, range, openings.len());
        let expected_part_count = assign_and_range_usize(ctx, range, expected_widths.len());
        ctx.constrain_equal(&part_count, &expected_part_count);

        for (part_idx, (part_openings, &expected_width)) in
            openings.iter().zip(expected_widths.iter()).enumerate()
        {
            let opening_width = assign_and_range_usize(ctx, range, part_openings.len());
            let expected_width_cell = assign_and_range_usize(ctx, range, expected_width);
            ctx.constrain_equal(&opening_width, &expected_width_cell);

            let need_rot = *actual.column_openings_need_rot[trace_idx]
                .get(part_idx)
                .expect("rotation schedule must cover each opening part");
            if need_rot {
                let is_even = ctx.load_witness(Fr::from((part_openings.len() % 2 == 0) as u64));
                gate.assert_bit(ctx, is_even);
                gate.assert_is_const(ctx, &is_even, &Fr::from(1u64));
            }
        }
    }

    let gkr_numerator_residual = assign_ext(ctx, range, &baby_bear, actual.gkr_numerator_residual);

    let gkr_denominator_residual =
        assign_ext(ctx, range, &baby_bear, actual.gkr_denominator_residual);
    let gkr_denominator_claim = assign_ext(ctx, range, &baby_bear, actual.gkr_denominator_claim);
    let alpha_logup = assign_ext(ctx, range, &baby_bear, actual.alpha_logup);
    let beta_logup = assign_ext(ctx, range, &baby_bear, actual.beta_logup);
    let gkr_non_xi_samples = actual
        .gkr_non_xi_samples
        .iter()
        .map(|&value| assign_ext(ctx, range, &baby_bear, value))
        .collect::<Vec<_>>();
    let gkr_xi_sample_order = actual
        .gkr_xi_sample_order
        .iter()
        .map(|&value| assign_ext(ctx, range, &baby_bear, value))
        .collect::<Vec<_>>();
    let xi = actual
        .xi
        .iter()
        .map(|&value| assign_ext(ctx, range, &baby_bear, value))
        .collect::<Vec<_>>();
    let lambda = assign_ext(ctx, range, &baby_bear, actual.lambda);
    let mu = assign_ext(ctx, range, &baby_bear, actual.mu);
    let r = actual
        .r
        .iter()
        .map(|&actual_r| assign_ext(ctx, range, &baby_bear, actual_r))
        .collect::<Vec<_>>();
    let assigned_public_values = actual
        .public_values
        .iter()
        .map(|values| {
            values
                .iter()
                .map(|&value| baby_bear.load_witness(ctx, range, value))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let zero = baby_bear.ext_zero(ctx, range);
    let one = ext_from_base_const(ctx, range, &baby_bear, 1);
    let mut gkr_sample_stream =
        Vec::with_capacity(gkr_non_xi_samples.len() + gkr_xi_sample_order.len());
    gkr_sample_stream.extend(gkr_non_xi_samples.iter().cloned());
    gkr_sample_stream.extend(gkr_xi_sample_order.iter().cloned());
    let mut gkr_sample_cursor = 0usize;
    let total_gkr_rounds = actual.l_skip + actual.n_logup;

    let gkr_layer_claim_count = assign_and_range_usize(ctx, range, gkr_claims_per_layer.len());
    let gkr_sumcheck_count = assign_and_range_usize(ctx, range, gkr_sumcheck_polys.len());
    if actual.total_interactions == 0 {
        gate.assert_is_const(ctx, &gkr_layer_claim_count, &Fr::from(0u64));
        gate.assert_is_const(ctx, &gkr_sumcheck_count, &Fr::from(0u64));
    } else {
        gate.assert_is_const(
            ctx,
            &gkr_layer_claim_count,
            &Fr::from(usize_to_u64(total_gkr_rounds)),
        );
        gate.assert_is_const(
            ctx,
            &gkr_sumcheck_count,
            &Fr::from(usize_to_u64(total_gkr_rounds.saturating_sub(1))),
        );
    }

    let (mut gkr_p_xi_claim, mut gkr_q_xi_claim, gkr_xi_claims) = if actual.total_interactions == 0
    {
        (zero.clone(), alpha_logup.clone(), Vec::new())
    } else {
        let layer0 = gkr_claims_per_layer
            .first()
            .cloned()
            .unwrap_or_else(|| vec![zero.clone(), zero.clone(), zero.clone(), zero.clone()]);
        let layer0_len = assign_and_range_usize(ctx, range, layer0.len());
        gate.assert_is_const(ctx, &layer0_len, &Fr::from(4u64));

        let p0_q1 = baby_bear.ext_mul(ctx, range, &layer0[0], &layer0[3]);
        let p1_q0 = baby_bear.ext_mul(ctx, range, &layer0[2], &layer0[1]);
        let p_cross = baby_bear.ext_add(ctx, range, &p0_q1, &p1_q0);
        let q_cross = baby_bear.ext_mul(ctx, range, &layer0[1], &layer0[3]);
        baby_bear.assert_ext_equal(ctx, &p_cross, &zero);
        baby_bear.assert_ext_equal(
            ctx,
            &q_cross,
            gkr_q0_claim
                .as_ref()
                .expect("non-zero interaction branch must include GKR q0 claim witness"),
        );

        let mu0 = gkr_sample_stream
            .get(gkr_sample_cursor)
            .cloned()
            .unwrap_or_else(|| zero.clone());
        gkr_sample_cursor += 1;
        let mut numer_claim =
            interpolate_linear_at_01_assigned(ctx, range, &baby_bear, &layer0[0], &layer0[2], &mu0);
        let mut denom_claim =
            interpolate_linear_at_01_assigned(ctx, range, &baby_bear, &layer0[1], &layer0[3], &mu0);
        let mut gkr_r = vec![mu0];

        for round in 1..total_gkr_rounds {
            let lambda_round = gkr_sample_stream
                .get(gkr_sample_cursor)
                .cloned()
                .unwrap_or_else(|| zero.clone());
            gkr_sample_cursor += 1;

            let lambda_denom = baby_bear.ext_mul(ctx, range, &lambda_round, &denom_claim);
            let mut claim = baby_bear.ext_add(ctx, range, &numer_claim, &lambda_denom);
            let round_polys = gkr_sumcheck_polys
                .get(round - 1)
                .cloned()
                .unwrap_or_default();
            let round_poly_len = assign_and_range_usize(ctx, range, round_polys.len());
            gate.assert_is_const(ctx, &round_poly_len, &Fr::from(usize_to_u64(round * 3)));

            let mut gkr_r_prime = Vec::with_capacity(round);
            let mut eq = one.clone();
            for subround in 0..round {
                let ev1 = round_polys
                    .get(subround * 3)
                    .cloned()
                    .unwrap_or_else(|| zero.clone());
                let ev2 = round_polys
                    .get(subround * 3 + 1)
                    .cloned()
                    .unwrap_or_else(|| zero.clone());
                let ev3 = round_polys
                    .get(subround * 3 + 2)
                    .cloned()
                    .unwrap_or_else(|| zero.clone());
                let ri = gkr_sample_stream
                    .get(gkr_sample_cursor)
                    .cloned()
                    .unwrap_or_else(|| zero.clone());
                gkr_sample_cursor += 1;
                gkr_r_prime.push(ri.clone());

                let ev0 = baby_bear.ext_sub(ctx, range, &claim, &ev1);
                claim = interpolate_cubic_at_0123_assigned(
                    ctx,
                    range,
                    &baby_bear,
                    [&ev0, &ev1, &ev2, &ev3],
                    &ri,
                );

                let xi_prev = &gkr_r[subround];
                let xi_ri = baby_bear.ext_mul(ctx, range, xi_prev, &ri);
                let one_minus_xi = baby_bear.ext_sub(ctx, range, &one, xi_prev);
                let one_minus_ri = baby_bear.ext_sub(ctx, range, &one, &ri);
                let one_minus_term = baby_bear.ext_mul(ctx, range, &one_minus_xi, &one_minus_ri);
                let eq_factor = baby_bear.ext_add(ctx, range, &xi_ri, &one_minus_term);
                eq = baby_bear.ext_mul(ctx, range, &eq, &eq_factor);
            }

            let layer_claims = gkr_claims_per_layer
                .get(round)
                .cloned()
                .unwrap_or_else(|| vec![zero.clone(), zero.clone(), zero.clone(), zero.clone()]);
            let layer_claim_count = assign_and_range_usize(ctx, range, layer_claims.len());
            gate.assert_is_const(ctx, &layer_claim_count, &Fr::from(4u64));
            let p0_q1 = baby_bear.ext_mul(ctx, range, &layer_claims[0], &layer_claims[3]);
            let p1_q0 = baby_bear.ext_mul(ctx, range, &layer_claims[2], &layer_claims[1]);
            let p_cross = baby_bear.ext_add(ctx, range, &p0_q1, &p1_q0);
            let q_cross = baby_bear.ext_mul(ctx, range, &layer_claims[1], &layer_claims[3]);
            let lambda_q_cross = baby_bear.ext_mul(ctx, range, &lambda_round, &q_cross);
            let claim_sum = baby_bear.ext_add(ctx, range, &p_cross, &lambda_q_cross);
            let expected_claim = baby_bear.ext_mul(ctx, range, &claim_sum, &eq);
            baby_bear.assert_ext_equal(ctx, &expected_claim, &claim);

            let mu_round = gkr_sample_stream
                .get(gkr_sample_cursor)
                .cloned()
                .unwrap_or_else(|| zero.clone());
            gkr_sample_cursor += 1;
            numer_claim = interpolate_linear_at_01_assigned(
                ctx,
                range,
                &baby_bear,
                &layer_claims[0],
                &layer_claims[2],
                &mu_round,
            );
            denom_claim = interpolate_linear_at_01_assigned(
                ctx,
                range,
                &baby_bear,
                &layer_claims[1],
                &layer_claims[3],
                &mu_round,
            );
            gkr_r = core::iter::once(mu_round)
                .chain(gkr_r_prime.into_iter())
                .collect();
        }

        (numer_claim, denom_claim, gkr_r)
    };

    let gkr_sample_cursor_cell = assign_and_range_usize(ctx, range, gkr_sample_cursor);
    gate.assert_is_const(
        ctx,
        &gkr_sample_cursor_cell,
        &Fr::from(usize_to_u64(gkr_sample_stream.len())),
    );
    let gkr_xi_claim_count = assign_and_range_usize(ctx, range, gkr_xi_claims.len());
    range.check_less_than_safe(
        ctx,
        gkr_xi_claim_count,
        usize_to_u64(xi.len()).saturating_add(1),
    );
    for (expected_xi, assigned_xi) in gkr_xi_claims.iter().zip(xi.iter()) {
        baby_bear.assert_ext_equal(ctx, expected_xi, assigned_xi);
    }

    for (num_term, den_term) in numerator_term_per_air
        .iter()
        .zip(denominator_term_per_air.iter())
    {
        gkr_p_xi_claim = baby_bear.ext_sub(ctx, range, &gkr_p_xi_claim, num_term);
        gkr_q_xi_claim = baby_bear.ext_sub(ctx, range, &gkr_q_xi_claim, den_term);
    }
    let derived_gkr_numerator_residual = gkr_p_xi_claim;
    let derived_gkr_denominator_claim = gkr_q_xi_claim.clone();
    let derived_gkr_denominator_residual =
        baby_bear.ext_sub(ctx, range, &derived_gkr_denominator_claim, &alpha_logup);
    baby_bear.assert_ext_equal(
        ctx,
        &derived_gkr_numerator_residual,
        &gkr_numerator_residual,
    );
    baby_bear.assert_ext_equal(ctx, &derived_gkr_denominator_claim, &gkr_denominator_claim);
    baby_bear.assert_ext_equal(
        ctx,
        &derived_gkr_denominator_residual,
        &gkr_denominator_residual,
    );
    baby_bear.assert_ext_equal(ctx, &gkr_numerator_residual, &zero);
    baby_bear.assert_ext_equal(ctx, &gkr_denominator_residual, &zero);

    let sum_claim = assign_ext(ctx, range, &baby_bear, actual.sum_claim);
    let mut derived_sum_claim = baby_bear.ext_zero(ctx, range);
    let mut cur_mu_pow = ext_from_base_const(ctx, range, &baby_bear, 1);
    for (num_term, den_term) in numerator_term_per_air
        .iter()
        .zip(denominator_term_per_air.iter())
    {
        let num_weighted = baby_bear.ext_mul(ctx, range, num_term, &cur_mu_pow);
        derived_sum_claim = baby_bear.ext_add(ctx, range, &derived_sum_claim, &num_weighted);
        cur_mu_pow = baby_bear.ext_mul(ctx, range, &cur_mu_pow, &mu);

        let den_weighted = baby_bear.ext_mul(ctx, range, den_term, &cur_mu_pow);
        derived_sum_claim = baby_bear.ext_add(ctx, range, &derived_sum_claim, &den_weighted);
        cur_mu_pow = baby_bear.ext_mul(ctx, range, &cur_mu_pow, &mu);
    }
    baby_bear.assert_ext_equal(ctx, &sum_claim, &derived_sum_claim);

    let sum_univ_domain_s_0 = assign_ext(ctx, range, &baby_bear, actual.sum_univ_domain_s_0);
    let stride = 1usize << actual.l_skip;
    let mut derived_univ_sum = baby_bear.ext_zero(ctx, range);
    for coeff in univariate_round_coeffs.iter().step_by(stride) {
        derived_univ_sum = baby_bear.ext_add(ctx, range, &derived_univ_sum, coeff);
    }
    let derived_univ_sum =
        ext_mul_base_const(ctx, range, &baby_bear, &derived_univ_sum, stride as u64);
    baby_bear.assert_ext_equal(ctx, &sum_univ_domain_s_0, &derived_univ_sum);
    baby_bear.assert_ext_equal(ctx, &sum_claim, &sum_univ_domain_s_0);

    let consistency_lhs = assign_ext(ctx, range, &baby_bear, actual.consistency_lhs);
    let mut derived_consistency_lhs =
        eval_ext_poly_horner(ctx, range, &baby_bear, &univariate_round_coeffs, &r[0]);
    for (round_idx, round_evals) in sumcheck_round_polys.iter().enumerate() {
        let s_1 = round_evals.first().cloned().unwrap_or_else(|| zero.clone());
        let s_0 = baby_bear.ext_sub(ctx, range, &derived_consistency_lhs, &s_1);
        let mut interpolation_evals = Vec::with_capacity(round_evals.len() + 1);
        interpolation_evals.push(s_0);
        interpolation_evals.extend(round_evals.iter().cloned());
        let next_r = r
            .get(round_idx + 1)
            .cloned()
            .unwrap_or_else(|| zero.clone());
        derived_consistency_lhs =
            eval_lagrange_on_integer_grid(ctx, range, &baby_bear, &next_r, &interpolation_evals);
    }
    baby_bear.assert_ext_equal(ctx, &consistency_lhs, &derived_consistency_lhs);

    let required_xi_len =
        assign_and_range_usize(ctx, range, actual.l_skip.saturating_add(actual.n_logup));
    range.check_less_than_safe(
        ctx,
        required_xi_len,
        usize_to_u64(xi.len()).saturating_add(1),
    );
    let omega_skip = NativeF::two_adic_generator(actual.l_skip);
    let omega_skip_pows = omega_skip.powers().take(1usize << actual.l_skip).collect();

    let mut eq_3b_per_trace = Vec::with_capacity(actual.n_per_trace.len());
    let mut stacked_idx = 0usize;
    for (trace_idx, &n) in actual.n_per_trace.iter().enumerate() {
        let n_lift = n.max(0) as usize;
        let interactions = &actual.trace_interactions[trace_idx];
        if interactions.is_empty() {
            eq_3b_per_trace.push(Vec::new());
            continue;
        }

        let mut eq_3b = Vec::with_capacity(interactions.len());
        for _ in 0..interactions.len() {
            let mut b_int = stacked_idx >> (actual.l_skip + n_lift);
            let mut b_vec = Vec::with_capacity(actual.n_logup.saturating_sub(n_lift));
            for _ in 0..actual.n_logup.saturating_sub(n_lift) {
                b_vec.push((b_int & 1) == 1);
                b_int >>= 1;
            }
            stacked_idx += 1 << (actual.l_skip + n_lift);
            let eq = eval_eq_mle_binary_assigned(
                ctx,
                range,
                &baby_bear,
                &xi[actual.l_skip + n_lift..actual.l_skip + actual.n_logup],
                &b_vec,
            );
            eq_3b.push(eq);
        }
        eq_3b_per_trace.push(eq_3b);
    }

    let mut eq_ns = vec![one.clone(); actual.n_max + 1];
    let mut eq_sharp_ns = vec![one.clone(); actual.n_max + 1];
    eq_ns[0] = eval_eq_uni_assigned(ctx, range, &baby_bear, actual.l_skip, &xi[0], &r[0]);
    eq_sharp_ns[0] = eval_eq_sharp_uni_assigned(
        ctx,
        range,
        &baby_bear,
        &omega_skip_pows,
        &xi[..actual.l_skip],
        &r[0],
    );
    for (i, r_i) in r.iter().enumerate().skip(1).take(actual.n_max) {
        let eq_mle = eval_eq_mle_assigned(
            ctx,
            range,
            &baby_bear,
            &[xi[actual.l_skip + i - 1].clone()],
            core::slice::from_ref(r_i),
        );
        eq_ns[i] = baby_bear.ext_mul(ctx, range, &eq_ns[i - 1], &eq_mle);
        eq_sharp_ns[i] = baby_bear.ext_mul(ctx, range, &eq_sharp_ns[i - 1], &eq_mle);
    }
    if actual.n_max > 0 {
        let mut r_rev_prod = r[actual.n_max].clone();
        for i in (0..actual.n_max).rev() {
            eq_ns[i] = baby_bear.ext_mul(ctx, range, &eq_ns[i], &r_rev_prod);
            eq_sharp_ns[i] = baby_bear.ext_mul(ctx, range, &eq_sharp_ns[i], &r_rev_prod);
            r_rev_prod = baby_bear.ext_mul(ctx, range, &r_rev_prod, &r[i]);
        }
    }

    let mut interactions_evals = Vec::new();
    let mut constraints_evals = Vec::new();
    for (trace_idx, air_openings) in column_openings.iter().enumerate() {
        let air_idx = actual.trace_id_to_air_id[trace_idx];
        let n = actual.n_per_trace[trace_idx];
        let n_lift = n.max(0) as usize;
        assert!(
            n >= -(actual.l_skip as isize),
            "trace lift exponent must be >= -l_skip",
        );

        let need_rot_flags = actual
            .column_openings_need_rot
            .get(trace_idx)
            .expect("per-trace rotation flags must exist");
        assert_eq!(
            need_rot_flags.len(),
            air_openings.len(),
            "per-trace rotation flags must align with opening parts",
        );

        let common_main = column_openings_by_rot_assigned(
            ctx,
            range,
            &baby_bear,
            &air_openings[0],
            need_rot_flags[0],
        );
        let has_preprocessed = actual.trace_has_preprocessed[trace_idx];
        let preprocessed = has_preprocessed.then(|| {
            column_openings_by_rot_assigned(
                ctx,
                range,
                &baby_bear,
                &air_openings[1],
                need_rot_flags[1],
            )
        });
        let cached_idx = 1 + has_preprocessed as usize;
        let mut partitioned_main = air_openings[cached_idx..]
            .iter()
            .enumerate()
            .map(|(part_offset, opening)| {
                column_openings_by_rot_assigned(
                    ctx,
                    range,
                    &baby_bear,
                    opening,
                    need_rot_flags[cached_idx + part_offset],
                )
            })
            .collect::<Vec<_>>();
        partitioned_main.push(common_main);

        let (l, rs_n, norm_factor) = if n.is_negative() {
            (
                actual.l_skip.wrapping_add_signed(n),
                vec![ext_pow_power_of_two(
                    ctx,
                    range,
                    &baby_bear,
                    &r[0],
                    n.unsigned_abs(),
                )],
                NativeF::from_usize(1usize << n.unsigned_abs())
                    .inverse()
                    .as_canonical_u64(),
            )
        } else {
            (actual.l_skip, r[..=n_lift].to_vec(), 1u64)
        };

        let inv_l = NativeF::from_usize(1usize << l)
            .inverse()
            .as_canonical_u64();
        let mut is_first_row = progression_exp_2_assigned(ctx, range, &baby_bear, &rs_n[0], l);
        is_first_row = ext_mul_base_const(ctx, range, &baby_bear, &is_first_row, inv_l);
        for x in rs_n.iter().skip(1) {
            let one_minus_x = baby_bear.ext_sub(ctx, range, &one, x);
            is_first_row = baby_bear.ext_mul(ctx, range, &is_first_row, &one_minus_x);
        }

        let omega = NativeF::two_adic_generator(l).as_canonical_u64();
        let rs0_omega = ext_mul_base_const(ctx, range, &baby_bear, &rs_n[0], omega);
        let mut is_last_row = progression_exp_2_assigned(ctx, range, &baby_bear, &rs0_omega, l);
        is_last_row = ext_mul_base_const(ctx, range, &baby_bear, &is_last_row, inv_l);
        for x in rs_n.iter().skip(1) {
            is_last_row = baby_bear.ext_mul(ctx, range, &is_last_row, x);
        }

        let evaluator = AssignedConstraintEvaluator {
            preprocessed: preprocessed.as_deref(),
            partitioned_main: &partitioned_main,
            is_first_row,
            is_last_row,
            public_values: assigned_public_values
                .get(air_idx)
                .map(|values| values.as_slice())
                .unwrap_or(&[]),
        };

        let node_values = eval_symbolic_nodes_assigned(
            ctx,
            range,
            &baby_bear,
            &evaluator,
            &actual.trace_constraint_nodes[trace_idx],
        );

        let mut expr = baby_bear.ext_zero(ctx, range);
        let mut lambda_pow = one.clone();
        for &constraint_idx in &actual.trace_constraint_indices[trace_idx] {
            let term = baby_bear.ext_mul(ctx, range, &node_values[constraint_idx], &lambda_pow);
            expr = baby_bear.ext_add(ctx, range, &expr, &term);
            lambda_pow = baby_bear.ext_mul(ctx, range, &lambda_pow, &lambda);
        }
        constraints_evals.push(baby_bear.ext_mul(ctx, range, &eq_ns[n_lift], &expr));

        let interactions = &actual.trace_interactions[trace_idx];
        let eq_3bs = &eq_3b_per_trace[trace_idx];
        assert_eq!(
            interactions.len(),
            eq_3bs.len(),
            "trace interaction count must match computed eq_3b weights",
        );
        let mut num = baby_bear.ext_zero(ctx, range);
        let mut denom = baby_bear.ext_zero(ctx, range);
        for (eq_3b, interaction) in eq_3bs.iter().zip(interactions.iter()) {
            let count_eval = node_values[interaction.count].clone();
            let mut denom_eval = baby_bear.ext_zero(ctx, range);
            let mut beta_pow = one.clone();
            for &msg_idx in &interaction.message {
                let term = baby_bear.ext_mul(ctx, range, &node_values[msg_idx], &beta_pow);
                denom_eval = baby_bear.ext_add(ctx, range, &denom_eval, &term);
                beta_pow = baby_bear.ext_mul(ctx, range, &beta_pow, &beta_logup);
            }
            let bus_const =
                ext_from_base_const(ctx, range, &baby_bear, u64::from(interaction.bus_index) + 1);
            let bus_term = baby_bear.ext_mul(ctx, range, &bus_const, &beta_pow);
            denom_eval = baby_bear.ext_add(ctx, range, &denom_eval, &bus_term);

            let eq_times_count = baby_bear.ext_mul(ctx, range, eq_3b, &count_eval);
            num = baby_bear.ext_add(ctx, range, &num, &eq_times_count);
            let eq_times_denom = baby_bear.ext_mul(ctx, range, eq_3b, &denom_eval);
            denom = baby_bear.ext_add(ctx, range, &denom, &eq_times_denom);
        }

        let norm = ext_from_base_const(ctx, range, &baby_bear, norm_factor);
        let num_norm = baby_bear.ext_mul(ctx, range, &num, &norm);
        let num_scaled = baby_bear.ext_mul(ctx, range, &num_norm, &eq_sharp_ns[n_lift]);
        let denom_scaled = baby_bear.ext_mul(ctx, range, &denom, &eq_sharp_ns[n_lift]);
        interactions_evals.push(num_scaled);
        interactions_evals.push(denom_scaled);
    }

    let consistency_rhs = assign_ext(ctx, range, &baby_bear, actual.consistency_rhs);
    let mut derived_consistency_rhs = baby_bear.ext_zero(ctx, range);
    let mut cur_mu_pow = one.clone();
    for term in interactions_evals.iter().chain(constraints_evals.iter()) {
        let weighted_term = baby_bear.ext_mul(ctx, range, term, &cur_mu_pow);
        derived_consistency_rhs =
            baby_bear.ext_add(ctx, range, &derived_consistency_rhs, &weighted_term);
        cur_mu_pow = baby_bear.ext_mul(ctx, range, &cur_mu_pow, &mu);
    }
    baby_bear.assert_ext_equal(ctx, &consistency_rhs, &derived_consistency_rhs);

    let consistency_residual = assign_ext(ctx, range, &baby_bear, actual.consistency_residual);
    let derived_consistency_residual =
        baby_bear.ext_sub(ctx, range, &consistency_lhs, &consistency_rhs);
    baby_bear.assert_ext_equal(ctx, &derived_consistency_residual, &consistency_residual);
    baby_bear.assert_ext_equal(ctx, &consistency_residual, &zero);

    AssignedBatchIntermediates {
        trace_id_to_air_id,
        public_values: assigned_public_values,
        total_interactions,
        n_logup,
        n_max,
        batch_degree,
        logup_pow_bits,
        logup_pow_sampled_bits,
        logup_pow_witness_ok,
        gkr_q0_claim,
        gkr_claims_per_layer,
        gkr_sumcheck_polys,
        numerator_term_per_air,
        denominator_term_per_air,
        univariate_round_coeffs,
        sumcheck_round_polys,
        column_openings,
        column_openings_need_rot: actual.column_openings_need_rot.clone(),
        gkr_numerator_residual,
        gkr_denominator_residual,
        gkr_denominator_claim,
        alpha_logup,
        beta_logup,
        gkr_non_xi_samples,
        gkr_xi_sample_order,
        xi,
        lambda,
        mu,
        sum_claim,
        sum_univ_domain_s_0,
        consistency_lhs,
        consistency_rhs,
        consistency_residual,
        r,
    }
}

pub(crate) fn derive_and_constrain_batch(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    config: &NativeConfig,
    mvk: &MultiStarkVerifyingKey<NativeConfig>,
    proof: &Proof<NativeConfig>,
) -> Result<AssignedBatchIntermediates, BatchConstraintError> {
    let raw = derive_raw_batch_witness_state(config, mvk, proof)?;
    Ok(constrain_checked_batch_witness_state(ctx, range, &raw).assigned)
}

pub(crate) fn derive_raw_batch_witness_state(
    config: &NativeConfig,
    mvk: &MultiStarkVerifyingKey<NativeConfig>,
    proof: &Proof<NativeConfig>,
) -> Result<RawBatchWitnessState, BatchConstraintError> {
    Ok(RawBatchWitnessState {
        intermediates: derive_batch_intermediates(config, mvk, proof)?,
    })
}

pub(crate) fn constrain_checked_batch_witness_state(
    ctx: &mut Context<Fr>,
    range: &RangeChip<Fr>,
    raw: &RawBatchWitnessState,
) -> CheckedBatchWitnessState {
    let assigned = constrain_batch_intermediates_unchecked(ctx, range, &raw.intermediates);
    let derived = DerivedBatchState {
        sum_claim: assigned.sum_claim.clone(),
        sum_univ_domain_s_0: assigned.sum_univ_domain_s_0.clone(),
        consistency_residual: assigned.consistency_residual.clone(),
    };
    CheckedBatchWitnessState { assigned, derived }
}

pub fn coeffs_to_native_ext(coeffs: [u64; BABY_BEAR_EXT_DEGREE]) -> NativeEF {
    coeffs_to_ext(coeffs)
}

#[cfg(test)]
mod tests;
