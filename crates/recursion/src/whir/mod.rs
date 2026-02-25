use core::{cmp, iter::zip, ops::Range};
use std::sync::Arc;

use itertools::{izip, Itertools};
use openvm_circuit_primitives::encoder::Encoder;
use openvm_stark_backend::{
    keygen::types::MultiStarkVerifyingKey,
    p3_maybe_rayon::prelude::*,
    poly_common::{eval_mle_evals_at_point, interpolate_quadratic_at_012, Squarable},
    proof::{Proof, WhirProof},
    prover::{AirProvingContext, CpuBackend},
    AirRef, FiatShamirTranscript, SystemParams, TranscriptHistory,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, CHUNK, EF, F};
use p3_field::{Field, PrimeCharacteristicRing, PrimeField32, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use strum::{EnumCount, EnumDiscriminants};

use crate::{
    primitives::exp_bits_len::ExpBitsLenTraceGenerator,
    system::{
        AirModule, BusIndexManager, BusInventory, GlobalCtxCpu, Preflight, TraceGenModule,
        WhirPreflight,
    },
    tracegen::{ModuleChip, RowMajorChip, StandardTracegenCtx},
    utils::{pow_observe_sample, FlattenedLayout, FlattenedVec},
    whir::{
        bus::{
            FinalPolyFoldingBus, FinalPolyMleEvalBus, FinalPolyQueryEvalBus, VerifyQueriesBus,
            VerifyQueryBus, WhirAlphaBus, WhirEqAlphaUBus, WhirFinalPolyBus, WhirFoldingBus,
            WhirGammaBus, WhirQueryBus, WhirSumcheckBus,
        },
        final_poly_mle_eval::FinalPolyMleEvalAir,
        final_poly_query_eval::FinalPolyQueryEvalAir,
        folding::{FoldRecord, WhirFoldingAir},
        initial_opened_values::InitialOpenedValuesAir,
        non_initial_opened_values::NonInitialOpenedValuesAir,
        query::WhirQueryAir,
        sumcheck::SumcheckAir,
        whir_round::WhirRoundAir,
    },
};

mod bus;
mod final_poly_mle_eval;
mod final_poly_query_eval;
pub mod folding;
mod initial_opened_values;
mod non_initial_opened_values;
mod query;
mod sumcheck;
mod whir_round;

pub(crate) fn num_queries_per_round(params: &SystemParams) -> Vec<usize> {
    params
        .whir
        .rounds
        .iter()
        .map(|round| round.num_queries)
        .collect()
}

#[inline]
fn eval_final_poly_at_u(final_poly: &[EF], u_tail: &[EF]) -> EF {
    let mut evals = final_poly.to_vec();
    eval_mle_evals_at_point(&mut evals, u_tail)
}

pub(in crate::whir) type PerProofIdx = (usize, usize);
pub(in crate::whir) type QueryIdx = (usize, usize, usize);

#[derive(Clone, Debug)]
pub(in crate::whir) struct PerProofLayout {
    num_proofs: usize,
    items_per_proof: usize,
}

impl PerProofLayout {
    pub(in crate::whir) fn new(num_proofs: usize, items_per_proof: usize) -> Self {
        Self {
            num_proofs,
            items_per_proof,
        }
    }

    #[inline]
    pub(in crate::whir) fn items_per_proof(&self) -> usize {
        self.items_per_proof
    }
}

impl FlattenedLayout for PerProofLayout {
    type Index = PerProofIdx;

    #[inline]
    fn len(&self) -> usize {
        self.num_proofs * self.items_per_proof
    }

    #[inline]
    fn offset(&self, idx: Self::Index) -> usize {
        let (proof_idx, item_idx) = idx;
        debug_assert!(
            proof_idx < self.num_proofs,
            "proof index out of bounds: {proof_idx} >= {}",
            self.num_proofs
        );
        debug_assert!(
            item_idx < self.items_per_proof,
            "index out of bounds: {item_idx} >= {}",
            self.items_per_proof
        );
        proof_idx * self.items_per_proof + item_idx
    }
}

#[derive(Clone, Debug)]
pub(in crate::whir) struct VariablePerProofLayout {
    proof_offsets: Vec<usize>,
}

impl VariablePerProofLayout {
    pub(in crate::whir) fn new(per_proof_items: impl IntoIterator<Item = usize>) -> Self {
        let mut proof_offsets = vec![0];
        let mut total = 0usize;
        for items in per_proof_items {
            total += items;
            proof_offsets.push(total);
        }
        Self { proof_offsets }
    }
}

impl FlattenedLayout for VariablePerProofLayout {
    type Index = PerProofIdx;

    #[inline]
    fn len(&self) -> usize {
        *self.proof_offsets.last().unwrap_or(&0)
    }

    #[inline]
    fn offset(&self, idx: Self::Index) -> usize {
        let (proof_idx, item_idx) = idx;
        debug_assert!(
            proof_idx + 1 < self.proof_offsets.len(),
            "proof index out of bounds: {proof_idx} >= {}",
            self.proof_offsets.len().saturating_sub(1)
        );
        let proof_start = self.proof_offsets[proof_idx];
        let proof_end = self.proof_offsets[proof_idx + 1];
        debug_assert!(
            item_idx < proof_end - proof_start,
            "index out of bounds: {} >= {} for proof {}",
            item_idx,
            proof_end - proof_start,
            proof_idx
        );
        proof_start + item_idx
    }
}

#[derive(Clone, Debug)]
pub(in crate::whir) struct WhirQueryLayout {
    num_proofs: usize,
    query_offsets: Vec<usize>,
}

impl WhirQueryLayout {
    pub(in crate::whir) fn new(num_proofs: usize, num_queries_per_round: &[usize]) -> Self {
        let mut query_offsets = Vec::with_capacity(num_queries_per_round.len() + 1);
        query_offsets.push(0);
        for &num_queries in num_queries_per_round {
            query_offsets.push(query_offsets.last().copied().unwrap() + num_queries);
        }
        Self {
            num_proofs,
            query_offsets,
        }
    }

    #[inline]
    pub(in crate::whir) fn num_rounds(&self) -> usize {
        debug_assert!(!self.query_offsets.is_empty());
        self.query_offsets.len() - 1
    }

    #[inline]
    pub(in crate::whir) fn round_query_range(&self, whir_round: usize) -> Range<usize> {
        debug_assert!(
            whir_round + 1 < self.query_offsets.len(),
            "WHIR round out of bounds: {whir_round} >= {}",
            self.query_offsets.len().saturating_sub(1)
        );
        self.query_offsets[whir_round]..self.query_offsets[whir_round + 1]
    }

    #[inline]
    pub(in crate::whir) fn round_num_queries(&self, whir_round: usize) -> usize {
        self.round_query_range(whir_round).len()
    }

    #[inline]
    pub(in crate::whir) fn iter_round_query_ranges(
        &self,
    ) -> impl Iterator<Item = (usize, Range<usize>)> + '_ {
        (0..self.num_rounds())
            .map(move |whir_round| (whir_round, self.round_query_range(whir_round)))
    }

    #[inline]
    pub(in crate::whir) fn round_and_query_idx(&self, proof_query_idx: usize) -> (usize, usize) {
        debug_assert!(
            proof_query_idx < self.queries_per_proof(),
            "proof query index out of bounds: {proof_query_idx} >= {}",
            self.queries_per_proof()
        );
        let whir_round =
            self.query_offsets[1..].partition_point(|&offset| offset <= proof_query_idx);
        let query_idx = proof_query_idx - self.query_offsets[whir_round];
        (whir_round, query_idx)
    }

    #[inline]
    pub(in crate::whir) fn queries_per_proof(&self) -> usize {
        *self.query_offsets.last().unwrap_or(&0)
    }

    /// Returns the raw query offset array (length = num_rounds + 1).
    #[cfg(feature = "cuda")]
    #[inline]
    pub(in crate::whir) fn query_offsets(&self) -> &[usize] {
        &self.query_offsets
    }
}

impl FlattenedLayout for WhirQueryLayout {
    type Index = QueryIdx;

    #[inline]
    fn len(&self) -> usize {
        self.num_proofs * self.queries_per_proof()
    }

    #[inline]
    fn offset(&self, idx: Self::Index) -> usize {
        let (proof_idx, whir_round, query_idx) = idx;
        debug_assert!(
            proof_idx < self.num_proofs,
            "proof index out of bounds: {proof_idx} >= {}",
            self.num_proofs
        );
        debug_assert!(
            whir_round + 1 < self.query_offsets.len(),
            "WHIR round out of bounds: {whir_round} >= {}",
            self.query_offsets.len().saturating_sub(1)
        );
        let proof_start = proof_idx * self.queries_per_proof();
        let round_start = proof_start + self.query_offsets[whir_round];
        let round_end = proof_start + self.query_offsets[whir_round + 1];
        debug_assert!(
            query_idx < round_end - round_start,
            "query index out of bounds: {} >= {} for proof {}, round {}",
            query_idx,
            round_end - round_start,
            proof_idx,
            whir_round
        );
        round_start + query_idx
    }
}

pub(in crate::whir) type CodewordAccsIdx = (usize, usize, usize, usize, usize);

/// Layout for the flattened `codeword_value_accs` array in `InitialOpenedValues`.
/// Data is ordered as `[proof][query][coset][commit][chunk]`.
#[derive(Clone, Debug)]
pub(in crate::whir) struct CodewordAccsLayout {
    num_queries: usize,
    num_cosets: usize,
    rows_per_proof_offsets: Vec<usize>,
    commits_per_proof_offsets: Vec<usize>,
    stacking_chunks_offsets: Vec<usize>,
    stacking_widths_offsets: Vec<usize>,
}

impl CodewordAccsLayout {
    pub(in crate::whir) fn new(
        num_queries: usize,
        num_cosets: usize,
        per_proof_commit_widths: &[Vec<usize>],
    ) -> Self {
        let num_proofs = per_proof_commit_widths.len();
        let mut rows_per_proof_offsets = Vec::with_capacity(num_proofs + 1);
        let mut commits_per_proof_offsets = Vec::with_capacity(num_proofs + 1);
        let total_commits: usize = per_proof_commit_widths.iter().map(Vec::len).sum();
        let mut stacking_chunks_offsets = Vec::with_capacity(total_commits + 1);
        let mut stacking_widths_offsets = Vec::with_capacity(total_commits + 1);

        rows_per_proof_offsets.push(0);
        commits_per_proof_offsets.push(0);
        stacking_chunks_offsets.push(0);
        stacking_widths_offsets.push(0);

        for per_commit_widths in per_proof_commit_widths {
            let mut total_chunks_for_proof = 0usize;
            for &width in per_commit_widths {
                let chunks = width.div_ceil(CHUNK);
                total_chunks_for_proof += chunks;
                stacking_chunks_offsets.push(*stacking_chunks_offsets.last().unwrap() + chunks);
                stacking_widths_offsets.push(*stacking_widths_offsets.last().unwrap() + width);
            }
            rows_per_proof_offsets.push(
                *rows_per_proof_offsets.last().unwrap()
                    + num_queries * num_cosets * total_chunks_for_proof,
            );
            commits_per_proof_offsets
                .push(*commits_per_proof_offsets.last().unwrap() + per_commit_widths.len());
        }
        Self {
            num_queries,
            num_cosets,
            rows_per_proof_offsets,
            commits_per_proof_offsets,
            stacking_chunks_offsets,
            stacking_widths_offsets,
        }
    }

    #[inline]
    pub(in crate::whir) fn num_proofs(&self) -> usize {
        self.rows_per_proof_offsets.len().saturating_sub(1)
    }

    #[inline]
    pub(in crate::whir) fn total_chunks(&self, proof_idx: usize) -> usize {
        let commit_start = self.commits_per_proof_offsets[proof_idx];
        let commit_end = self.commits_per_proof_offsets[proof_idx + 1];
        self.stacking_chunks_offsets[commit_end] - self.stacking_chunks_offsets[commit_start]
    }

    #[cfg(feature = "cuda")]
    #[inline]
    pub(in crate::whir) fn rows_per_proof_offsets(&self) -> &[usize] {
        &self.rows_per_proof_offsets
    }

    #[cfg(feature = "cuda")]
    #[inline]
    pub(in crate::whir) fn commits_per_proof_offsets(&self) -> &[usize] {
        &self.commits_per_proof_offsets
    }

    #[cfg(feature = "cuda")]
    #[inline]
    pub(in crate::whir) fn stacking_chunks_offsets(&self) -> &[usize] {
        &self.stacking_chunks_offsets
    }

    #[cfg(feature = "cuda")]
    #[inline]
    pub(in crate::whir) fn stacking_widths_offsets(&self) -> &[usize] {
        &self.stacking_widths_offsets
    }

    #[inline]
    pub(in crate::whir) fn total_width(&self, proof_idx: usize) -> usize {
        let commit_start = self.commits_per_proof_offsets[proof_idx];
        let commit_end = self.commits_per_proof_offsets[proof_idx + 1];
        self.stacking_widths_offsets[commit_end] - self.stacking_widths_offsets[commit_start]
    }

    #[inline]
    pub(in crate::whir) fn num_commits(&self, proof_idx: usize) -> usize {
        self.commits_per_proof_offsets[proof_idx + 1] - self.commits_per_proof_offsets[proof_idx]
    }

    /// Decompose a flat row index into `(proof, query, coset, commit, chunk)`.
    #[inline]
    pub(in crate::whir) fn decompose(&self, row_idx: usize) -> CodewordAccsIdx {
        let proof_idx = self.rows_per_proof_offsets[1..].partition_point(|&x| x <= row_idx);
        let record_idx = row_idx - self.rows_per_proof_offsets[proof_idx];
        let commit_start = self.commits_per_proof_offsets[proof_idx];
        let commit_end = self.commits_per_proof_offsets[proof_idx + 1];

        let chunks_before_proof = self.stacking_chunks_offsets[commit_start];
        let chunks_after_proof = self.stacking_chunks_offsets[commit_end];
        let total_chunks_for_proof = chunks_after_proof - chunks_before_proof;

        let query_idx = record_idx / (self.num_cosets * total_chunks_for_proof);
        let coset_idx = (record_idx / total_chunks_for_proof) % self.num_cosets;
        let local_chunk_idx = record_idx % total_chunks_for_proof;
        let absolute_chunk_idx = chunks_before_proof + local_chunk_idx;
        let commit_idx = self.stacking_chunks_offsets[commit_start + 1..=commit_end]
            .partition_point(|&x| x <= absolute_chunk_idx);
        let chunk_idx =
            absolute_chunk_idx - self.stacking_chunks_offsets[commit_start + commit_idx];

        (proof_idx, query_idx, coset_idx, commit_idx, chunk_idx)
    }

    /// Number of chunks in the given commit.
    #[inline]
    pub(in crate::whir) fn commit_num_chunks(&self, proof_idx: usize, commit_idx: usize) -> usize {
        let global_idx = self.commits_per_proof_offsets[proof_idx] + commit_idx;
        self.stacking_chunks_offsets[global_idx + 1] - self.stacking_chunks_offsets[global_idx]
    }

    /// Width (number of opened values) in the given commit.
    #[inline]
    pub(in crate::whir) fn commit_width(&self, proof_idx: usize, commit_idx: usize) -> usize {
        let global_idx = self.commits_per_proof_offsets[proof_idx] + commit_idx;
        self.stacking_widths_offsets[global_idx + 1] - self.stacking_widths_offsets[global_idx]
    }

    /// Cumulative width offset for the given commit (for `mu_pows` indexing).
    #[inline]
    pub(in crate::whir) fn commit_width_offset(
        &self,
        proof_idx: usize,
        commit_idx: usize,
    ) -> usize {
        let commit_start = self.commits_per_proof_offsets[proof_idx];
        self.stacking_widths_offsets[commit_start + commit_idx]
            - self.stacking_widths_offsets[commit_start]
    }

    /// Length of the chunk at `(commit_idx, chunk_idx)`.
    /// The last chunk of a commit may be shorter than `CHUNK`.
    #[inline]
    pub(in crate::whir) fn chunk_len(
        &self,
        proof_idx: usize,
        commit_idx: usize,
        chunk_idx: usize,
    ) -> usize {
        (self.commit_width(proof_idx, commit_idx) - chunk_idx * CHUNK).min(CHUNK)
    }
}

impl FlattenedLayout for CodewordAccsLayout {
    type Index = CodewordAccsIdx;

    #[inline]
    fn len(&self) -> usize {
        *self.rows_per_proof_offsets.last().unwrap_or(&0)
    }

    #[inline]
    fn offset(&self, (proof, query, coset, commit, chunk): Self::Index) -> usize {
        debug_assert!(proof < self.num_proofs());
        debug_assert!(query < self.num_queries);
        debug_assert!(coset < self.num_cosets);
        debug_assert!(commit < self.num_commits(proof));
        debug_assert!(chunk < self.commit_num_chunks(proof, commit));

        let commit_start = self.commits_per_proof_offsets[proof];
        let chunks_before_proof = self.stacking_chunks_offsets[commit_start];
        let total_chunks_for_proof = self.total_chunks(proof);
        let local_commit_chunk_offset =
            self.stacking_chunks_offsets[commit_start + commit] - chunks_before_proof;

        self.rows_per_proof_offsets[proof]
            + query * self.num_cosets * total_chunks_for_proof
            + coset * total_chunks_for_proof
            + local_commit_chunk_offset
            + chunk
    }
}

#[cfg(feature = "cuda")]
mod cuda_abi;

pub struct WhirModule {
    params: SystemParams,
    bus_inventory: BusInventory,

    // "execution" buses
    sumcheck_bus: WhirSumcheckBus,
    verify_queries_bus: VerifyQueriesBus,
    verify_query_bus: VerifyQueryBus,
    folding_bus: WhirFoldingBus,
    final_poly_mle_eval_bus: FinalPolyMleEvalBus,
    final_poly_query_eval_bus: FinalPolyQueryEvalBus,

    // data buses
    alpha_bus: WhirAlphaBus,
    gamma_bus: WhirGammaBus,
    query_bus: WhirQueryBus,
    eq_alpha_u_bus: WhirEqAlphaUBus,
    final_poly_bus: WhirFinalPolyBus,
    final_poly_folding_bus: FinalPolyFoldingBus,
}

impl WhirModule {
    pub fn new(
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        b: &mut BusIndexManager,
        bus_inventory: BusInventory,
    ) -> Self {
        let sumcheck_bus = WhirSumcheckBus::new(b.new_bus_idx());
        let alpha_bus = WhirAlphaBus::new(b.new_bus_idx());
        let gamma_bus = WhirGammaBus::new(b.new_bus_idx());
        let query_bus = WhirQueryBus::new(b.new_bus_idx());
        let verify_queries_bus = VerifyQueriesBus::new(b.new_bus_idx());
        let verify_query_bus = VerifyQueryBus::new(b.new_bus_idx());
        let eq_alpha_u_bus = WhirEqAlphaUBus::new(b.new_bus_idx());
        let folding_bus = WhirFoldingBus::new(b.new_bus_idx());
        let final_poly_mle_eval_bus = FinalPolyMleEvalBus::new(b.new_bus_idx());
        let final_poly_query_eval_bus = FinalPolyQueryEvalBus::new(b.new_bus_idx());
        let final_poly_bus = WhirFinalPolyBus::new(b.new_bus_idx());
        let final_poly_folding_bus = FinalPolyFoldingBus::new(b.new_bus_idx());
        Self {
            params: child_vk.inner.params.clone(),
            bus_inventory,
            sumcheck_bus,
            verify_queries_bus,
            verify_query_bus,
            final_poly_mle_eval_bus,
            final_poly_query_eval_bus,
            folding_bus,
            alpha_bus,
            gamma_bus,
            query_bus,
            eq_alpha_u_bus,
            final_poly_bus,
            final_poly_folding_bus,
        }
    }
}

impl WhirModule {
    #[tracing::instrument(level = "trace", skip_all)]
    pub fn run_preflight<TS: FiatShamirTranscript<BabyBearPoseidon2Config> + TranscriptHistory>(
        &self,
        proof: &Proof<BabyBearPoseidon2Config>,
        preflight: &mut Preflight,
        ts: &mut TS,
    ) {
        let WhirProof {
            mu_pow_witness: _, // Handled in stacking module preflight
            whir_sumcheck_polys,
            codeword_commits,
            ood_values,
            initial_round_opened_rows: _,
            initial_round_merkle_proofs: _,
            codeword_opened_values: _,
            codeword_merkle_proofs: _,
            folding_pow_witnesses,
            query_phase_pow_witnesses,
            final_poly,
        } = &proof.whir_proof;

        let k_whir = self.params.k_whir();
        let num_queries_per_round = num_queries_per_round(&self.params);
        let query_layout = WhirQueryLayout::new(1, &num_queries_per_round);
        let num_whir_rounds = self.params.num_whir_rounds();
        let total_queries = query_layout.queries_per_proof();
        let mut gammas = Vec::with_capacity(num_whir_rounds);
        let mut z0s = Vec::with_capacity(num_whir_rounds - 1);
        let mut alphas = Vec::with_capacity(num_whir_rounds * k_whir);
        let mut folding_pow_samples = Vec::with_capacity(num_whir_rounds * k_whir);
        let mut query_pow_samples = Vec::with_capacity(num_whir_rounds);
        let mut queries = Vec::with_capacity(total_queries);
        let mut whir_round_tidx_per_round = Vec::with_capacity(num_whir_rounds);
        let mut query_tidx_per_round = Vec::with_capacity(num_whir_rounds);

        debug_assert_eq!(whir_sumcheck_polys.len(), num_whir_rounds * k_whir);
        debug_assert_eq!(folding_pow_witnesses.len(), num_whir_rounds * k_whir);
        debug_assert_eq!(query_phase_pow_witnesses.len(), num_whir_rounds);
        debug_assert_eq!(ood_values.len(), num_whir_rounds - 1);
        debug_assert_eq!(codeword_commits.len(), num_whir_rounds - 1);

        for i in 0..num_whir_rounds {
            whir_round_tidx_per_round.push(ts.len());

            let num_round_queries = num_queries_per_round[i];
            for j in 0..k_whir {
                let evals = &whir_sumcheck_polys[i * k_whir + j];
                let &[ev1, ev2] = evals;
                ts.observe_ext(ev1);
                ts.observe_ext(ev2);

                folding_pow_samples.push(pow_observe_sample(
                    ts,
                    self.params.whir.folding_pow_bits,
                    folding_pow_witnesses[i * k_whir + j],
                ));
                alphas.push(ts.sample_ext());
            }

            if i != num_whir_rounds - 1 {
                ts.observe_commit(codeword_commits[i]);
                z0s.push(ts.sample_ext());
                ts.observe_ext(ood_values[i]);
            } else {
                for coeff in final_poly {
                    ts.observe_ext(*coeff);
                }
            };

            query_pow_samples.push(pow_observe_sample(
                ts,
                self.params.whir.query_phase_pow_bits,
                query_phase_pow_witnesses[i],
            ));
            query_tidx_per_round.push(ts.len());

            for _ in 0..num_round_queries {
                queries.push(ts.sample());
            }
            gammas.push(ts.sample_ext());
        }
        preflight.whir = WhirPreflight {
            whir_round_tidx_per_round,
            query_tidx_per_round,
            alphas,
            z0s,
            gammas,
            folding_pow_samples,
            query_pow_samples,
            queries,
        };
    }
}

pub(crate) struct WhirBlobCpu {
    // Flattened per-proof WHIR-derived data.
    whir_round_tidx_per_round: FlattenedVec<usize, PerProofLayout>,
    query_tidx_per_round: FlattenedVec<usize, PerProofLayout>,
    initial_claim_per_round: FlattenedVec<EF, PerProofLayout>,
    post_sumcheck_claims: FlattenedVec<EF, PerProofLayout>,
    pre_query_claims: FlattenedVec<EF, PerProofLayout>,
    eq_partials: FlattenedVec<EF, PerProofLayout>,
    final_poly_at_u: Vec<EF>,
    zi_roots: FlattenedVec<F, WhirQueryLayout>,
    zis: FlattenedVec<F, WhirQueryLayout>,
    yis: FlattenedVec<EF, WhirQueryLayout>,
    fold_records: FlattenedVec<FoldRecord, PerProofLayout>,

    /// Initial opened values data with layout encoding (proof, query, coset, commit, chunk).
    codeword_value_accs: FlattenedVec<EF, CodewordAccsLayout>,
    /// Flattened as `[proof][mu_power_idx]` with per-proof variable widths.
    mu_pows: FlattenedVec<EF, VariablePerProofLayout>,
}

struct WhirBlobBuilder {
    whir_round_tidx_per_round: Vec<usize>,
    query_tidx_per_round: Vec<usize>,
    initial_claim_per_round: Vec<EF>,
    post_sumcheck_claims: Vec<EF>,
    pre_query_claims: Vec<EF>,
    eq_partials: Vec<EF>,
    final_poly_at_u: Vec<EF>,
    zi_roots: Vec<F>,
    zis: Vec<F>,
    yis: Vec<EF>,
    fold_records: Vec<FoldRecord>,
    codeword_value_accs: Vec<EF>,
    mu_pows: Vec<EF>,
}

struct WhirBlobLayouts {
    tidx_layout: PerProofLayout,
    initial_claim_layout: PerProofLayout,
    pre_query_claim_layout: PerProofLayout,
    sumcheck_layout: PerProofLayout,
    query_layout: WhirQueryLayout,
    fold_layout: PerProofLayout,
    accs_layout: CodewordAccsLayout,
    mu_pows_layout: VariablePerProofLayout,
}

impl WhirBlobBuilder {
    fn with_capacities(
        num_proofs: usize,
        num_whir_rounds: usize,
        sumcheck_rows_per_proof: usize,
        queries_per_proof: usize,
        fold_records_per_proof: usize,
        codeword_value_accs_len: usize,
        mu_pows_len: usize,
    ) -> Self {
        Self {
            whir_round_tidx_per_round: Vec::with_capacity(num_proofs * num_whir_rounds),
            query_tidx_per_round: Vec::with_capacity(num_proofs * num_whir_rounds),
            initial_claim_per_round: Vec::with_capacity(num_proofs * (num_whir_rounds + 1)),
            post_sumcheck_claims: Vec::with_capacity(num_proofs * sumcheck_rows_per_proof),
            pre_query_claims: Vec::with_capacity(num_proofs * num_whir_rounds),
            eq_partials: Vec::with_capacity(num_proofs * sumcheck_rows_per_proof),
            final_poly_at_u: Vec::with_capacity(num_proofs),
            zi_roots: Vec::with_capacity(num_proofs * queries_per_proof),
            zis: Vec::with_capacity(num_proofs * queries_per_proof),
            yis: Vec::with_capacity(num_proofs * queries_per_proof),
            fold_records: Vec::with_capacity(num_proofs * fold_records_per_proof),
            codeword_value_accs: Vec::with_capacity(codeword_value_accs_len),
            mu_pows: Vec::with_capacity(mu_pows_len),
        }
    }

    fn into_blob(self, layouts: WhirBlobLayouts) -> WhirBlobCpu {
        let WhirBlobLayouts {
            tidx_layout,
            initial_claim_layout,
            pre_query_claim_layout,
            sumcheck_layout,
            query_layout,
            fold_layout,
            accs_layout,
            mu_pows_layout,
        } = layouts;
        WhirBlobCpu {
            whir_round_tidx_per_round: FlattenedVec::from_parts(
                tidx_layout.clone(),
                self.whir_round_tidx_per_round,
            ),
            query_tidx_per_round: FlattenedVec::from_parts(tidx_layout, self.query_tidx_per_round),
            initial_claim_per_round: FlattenedVec::from_parts(
                initial_claim_layout,
                self.initial_claim_per_round,
            ),
            post_sumcheck_claims: FlattenedVec::from_parts(
                sumcheck_layout.clone(),
                self.post_sumcheck_claims,
            ),
            pre_query_claims: FlattenedVec::from_parts(
                pre_query_claim_layout,
                self.pre_query_claims,
            ),
            eq_partials: FlattenedVec::from_parts(sumcheck_layout, self.eq_partials),
            final_poly_at_u: self.final_poly_at_u,
            zi_roots: FlattenedVec::from_parts(query_layout.clone(), self.zi_roots),
            zis: FlattenedVec::from_parts(query_layout.clone(), self.zis),
            yis: FlattenedVec::from_parts(query_layout, self.yis),
            fold_records: FlattenedVec::from_parts(fold_layout, self.fold_records),
            codeword_value_accs: FlattenedVec::from_parts(accs_layout, self.codeword_value_accs),
            mu_pows: FlattenedVec::from_parts(mu_pows_layout, self.mu_pows),
        }
    }
}

impl AirModule for WhirModule {
    fn num_airs(&self) -> usize {
        WhirModuleChipDiscriminants::COUNT
    }

    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        let params = &self.params;
        let initial_log_domain_size = params.n_stack + params.l_skip + params.log_blowup;

        let num_rounds = params.num_whir_rounds();
        let num_queries_per_round = num_queries_per_round(params);

        // Encoder requires at least 2 flags to work correctly
        let whir_round_encoder = Encoder::new(num_rounds.max(2), 2, false);

        let whir_round_air: AirRef<BabyBearPoseidon2Config> = Arc::new(WhirRoundAir {
            whir_module_bus: self.bus_inventory.whir_module_bus,
            commitments_bus: self.bus_inventory.commitments_bus,
            transcript_bus: self.bus_inventory.transcript_bus,
            exp_bits_len_bus: self.bus_inventory.exp_bits_len_bus,
            sumcheck_bus: self.sumcheck_bus,
            verify_queries_bus: self.verify_queries_bus,
            final_poly_mle_eval_bus: self.final_poly_mle_eval_bus,
            final_poly_query_eval_bus: self.final_poly_query_eval_bus,
            query_bus: self.query_bus,
            gamma_bus: self.gamma_bus,
            k: params.k_whir(),
            num_rounds,
            final_poly_len: 1 << params.log_final_poly_len(),
            pow_bits: params.whir.query_phase_pow_bits,
            folding_pow_bits: params.whir.folding_pow_bits,
            generator: F::GENERATOR,
            whir_round_encoder,
            num_queries_per_round: num_queries_per_round.clone(),
        });
        let whir_sumcheck_air = SumcheckAir {
            sumcheck_bus: self.sumcheck_bus,
            whir_opening_point_bus: self.bus_inventory.whir_opening_point_bus,
            transcript_bus: self.bus_inventory.transcript_bus,
            exp_bits_len_bus: self.bus_inventory.exp_bits_len_bus,
            alpha_bus: self.alpha_bus,
            eq_alpha_u_bus: self.eq_alpha_u_bus,
            k: params.k_whir(),
            folding_pow_bits: params.whir.folding_pow_bits,
            generator: F::GENERATOR,
        };
        let initial_round_opened_values_air = InitialOpenedValuesAir {
            stacking_indices_bus: self.bus_inventory.stacking_indices_bus,
            whir_mu_bus: self.bus_inventory.whir_mu_bus,
            verify_query_bus: self.verify_query_bus,
            folding_bus: self.folding_bus,
            poseidon_permute_bus: self.bus_inventory.poseidon2_permute_bus,
            merkle_verify_bus: self.bus_inventory.merkle_verify_bus,
            k: params.k_whir(),
            initial_log_domain_size,
        };
        let non_initial_round_opened_values_air = NonInitialOpenedValuesAir {
            verify_query_bus: self.verify_query_bus,
            folding_bus: self.folding_bus,
            poseidon2_compress_bus: self.bus_inventory.poseidon2_compress_bus,
            merkle_verify_bus: self.bus_inventory.merkle_verify_bus,
            k: params.k_whir(),
            initial_log_domain_size,
        };
        let query_air = WhirQueryAir {
            transcript_bus: self.bus_inventory.transcript_bus,
            exp_bits_len_bus: self.bus_inventory.exp_bits_len_bus,
            query_bus: self.query_bus,
            verify_queries_bus: self.verify_queries_bus,
            verify_query_bus: self.verify_query_bus,
            k: params.k_whir(),
            initial_log_domain_size,
        };
        let folding_air = WhirFoldingAir {
            alpha_bus: self.alpha_bus,
            folding_bus: self.folding_bus,
            k: params.k_whir(),
        };
        let final_poly_mle_eval_air = FinalPolyMleEvalAir {
            whir_opening_point_bus: self.bus_inventory.whir_opening_point_bus,
            whir_opening_point_lookup_bus: self.bus_inventory.whir_opening_point_lookup_bus,
            transcript_bus: self.bus_inventory.transcript_bus,
            final_poly_mle_eval_bus: self.final_poly_mle_eval_bus,
            eq_alpha_u_bus: self.eq_alpha_u_bus,
            final_poly_bus: self.final_poly_bus,
            folding_bus: self.final_poly_folding_bus,
            num_vars: params.log_final_poly_len(),
            num_sumcheck_rounds: params.num_whir_sumcheck_rounds(),
            num_whir_rounds: params.num_whir_rounds(),
            total_whir_queries: params
                .whir
                .rounds
                .iter()
                .map(|cfg| cfg.num_queries + 1)
                .sum(),
        };
        let final_poly_query_eval_air = FinalPolyQueryEvalAir {
            query_bus: self.query_bus,
            alpha_bus: self.alpha_bus,
            gamma_bus: self.gamma_bus,
            final_poly_bus: self.final_poly_bus,
            final_poly_query_eval_bus: self.final_poly_query_eval_bus,
            num_whir_rounds: params.num_whir_rounds(),
            k_whir: params.k_whir(),
            log_final_poly_len: params.log_final_poly_len(),
        };
        vec![
            whir_round_air,
            Arc::new(whir_sumcheck_air),
            Arc::new(query_air),
            Arc::new(initial_round_opened_values_air),
            Arc::new(non_initial_round_opened_values_air),
            Arc::new(folding_air),
            Arc::new(final_poly_mle_eval_air),
            Arc::new(final_poly_query_eval_air),
        ]
    }
}

impl WhirModule {
    fn append_derived_whir_data_for_proof(
        proof: &Proof<BabyBearPoseidon2Config>,
        preflight: &Preflight,
        local_mu_pows: &[EF],
        params: &SystemParams,
        query_layout: &WhirQueryLayout,
        blob: &mut WhirBlobBuilder,
    ) {
        let k_whir = params.k_whir();
        let num_whir_rounds = params.num_whir_rounds();
        let l_skip = params.l_skip;
        let initial_log_rs_domain_size = params.l_skip + params.n_stack + params.log_blowup;

        let mut sumcheck_poly_iter = proof.whir_proof.whir_sumcheck_polys.iter();
        let mut claim = proof
            .stacking_proof
            .stacking_openings
            .iter()
            .flatten()
            .zip(local_mu_pows.iter())
            .fold(EF::ZERO, |acc, (&opening, &mu_pow)| acc + mu_pow * opening);

        let u = preflight.stacking.sumcheck_rnd[0]
            .exp_powers_of_2()
            .take(l_skip)
            .chain(preflight.stacking.sumcheck_rnd[1..].iter().copied())
            .collect_vec();

        let mut eq_partial = EF::ONE;
        for (i, query_range) in query_layout.iter_round_query_ranges() {
            let round_queries = &preflight.whir.queries[query_range];
            let log_rs_domain_size = initial_log_rs_domain_size - i;

            blob.initial_claim_per_round.push(claim);

            for j in 0..k_whir {
                let evals = sumcheck_poly_iter.next().unwrap();
                let &[ev1, ev2] = evals;
                let ev0 = claim - ev1;
                let alpha = preflight.whir.alphas[i * k_whir + j];
                let uj = u[i * k_whir + j];
                // Möbius eq kernel: mobius_eq_1(u, alpha) = (1 - 2*u)*(1 - alpha) + u*alpha
                //                              = 1 - alpha - 2*u + 3*u*alpha
                eq_partial *= EF::ONE - alpha - uj.double() + EF::from_u8(3) * uj * alpha;
                blob.eq_partials.push(eq_partial);

                claim = interpolate_quadratic_at_012(&[ev0, ev1, ev2], alpha);
                blob.post_sumcheck_claims.push(claim);
            }

            let gamma = preflight.whir.gammas[i];
            if let Some(&y0) = proof.whir_proof.ood_values.get(i) {
                claim += gamma * y0;
            }
            let mut gamma_pows = gamma.powers().skip(2);

            blob.pre_query_claims.push(claim);

            let omega = F::two_adic_generator(log_rs_domain_size);
            let round_alphas = &preflight.whir.alphas[i * k_whir..(i + 1) * k_whir];
            for (query_idx, &sample) in round_queries.iter().enumerate() {
                let index = sample.as_canonical_u32() & ((1 << (log_rs_domain_size - k_whir)) - 1);
                let zi_root = omega.exp_u64(index as u64);
                let zi = zi_root.exp_power_of_2(k_whir);
                let record_start = blob.fold_records.len();
                let yi = if i == 0 {
                    let mut codeword_vals = vec![EF::ZERO; 1 << k_whir];
                    let mut mu_pow_iter = local_mu_pows.iter();
                    for opened_rows_per_query in proof.whir_proof.initial_round_opened_rows.iter() {
                        let opened_rows = &opened_rows_per_query[query_idx];
                        let width = opened_rows[0].len();
                        for c in 0..width {
                            let mu_pow = mu_pow_iter.next().unwrap();
                            for j in 0..(1 << k_whir) {
                                codeword_vals[j] += *mu_pow * opened_rows[j][c];
                            }
                        }
                    }
                    binary_k_fold(
                        codeword_vals,
                        round_alphas,
                        zi_root,
                        i,
                        query_idx,
                        &mut blob.fold_records,
                    )
                } else {
                    let opened_values =
                        proof.whir_proof.codeword_opened_values[i - 1][query_idx].clone();
                    binary_k_fold(
                        opened_values,
                        round_alphas,
                        zi_root,
                        i,
                        query_idx,
                        &mut blob.fold_records,
                    )
                };
                for rec in &mut blob.fold_records[record_start..] {
                    rec.set_final_values(zi, yi);
                }
                blob.zi_roots.push(zi_root);
                blob.zis.push(zi);
                blob.yis.push(yi);

                claim += gamma_pows.next().unwrap() * yi;
            }
            let _ = gamma_pows.next().unwrap();
        }

        // Push one for the final claim.
        blob.initial_claim_per_round.push(claim);
        debug_assert!(sumcheck_poly_iter.next().is_none());

        // Evaluate the MLE of the table `final_poly` (interpreted as hypercube evaluations)
        // at `u[t..]`. This matches the eval-to-coeff RS encoding semantics.
        let t = k_whir * num_whir_rounds;
        blob.final_poly_at_u
            .push(eval_final_poly_at_u(&proof.whir_proof.final_poly, &u[t..]));
    }

    fn enqueue_pow_requests_for_proof(
        exp_bits_len_gen: &ExpBitsLenTraceGenerator,
        preflight: &Preflight,
        params: &SystemParams,
        query_layout: &WhirQueryLayout,
    ) {
        let mu_pow_bits = params.whir.mu_pow_bits;
        let folding_pow_bits = params.whir.folding_pow_bits;
        let query_phase_pow_bits = params.whir.query_phase_pow_bits;
        let k_whir = params.k_whir();
        let initial_log_rs_domain_size = params.l_skip + params.n_stack + params.log_blowup;

        // μ PoW lookup (from stacking module)
        if mu_pow_bits > 0 {
            exp_bits_len_gen.add_requests(std::iter::once((
                F::GENERATOR,
                preflight.stacking.mu_pow_sample,
                mu_pow_bits,
            )));
        }

        if folding_pow_bits > 0 {
            exp_bits_len_gen.add_requests(
                preflight
                    .whir
                    .folding_pow_samples
                    .iter()
                    .map(|pow_sample| (F::GENERATOR, *pow_sample, folding_pow_bits)),
            );
        }
        if query_phase_pow_bits > 0 {
            exp_bits_len_gen.add_requests(
                preflight
                    .whir
                    .query_pow_samples
                    .iter()
                    .map(|pow_sample| (F::GENERATOR, *pow_sample, query_phase_pow_bits)),
            );
        }

        for (i, query_range) in query_layout.iter_round_query_ranges() {
            let round_queries = &preflight.whir.queries[query_range];
            let log_rs_domain_size = initial_log_rs_domain_size - i;
            let omega = F::two_adic_generator(log_rs_domain_size);
            exp_bits_len_gen.add_requests(
                round_queries
                    .iter()
                    .copied()
                    .map(|sample| (omega, sample, log_rs_domain_size - k_whir)),
            );
        }
    }

    fn append_initial_opened_values_accs_for_proof(
        blob: &mut WhirBlobBuilder,
        accs_layout: &CodewordAccsLayout,
        proof_idx: usize,
        proof: &Proof<BabyBearPoseidon2Config>,
        local_mu_pows: &[EF],
        num_initial_queries: usize,
        k_whir: usize,
    ) {
        debug_assert_eq!(
            proof.whir_proof.initial_round_opened_rows.len(),
            accs_layout.num_commits(proof_idx)
        );
        #[cfg(debug_assertions)]
        for (i, openings_per_commit) in proof
            .whir_proof
            .initial_round_opened_rows
            .iter()
            .enumerate()
        {
            debug_assert_eq!(
                openings_per_commit[0][0].len(),
                accs_layout.commit_width(proof_idx, i)
            );
        }

        for query_idx in 0..num_initial_queries {
            let mut codeword_vals = EF::zero_vec(1 << k_whir);
            for (coset_idx, codeword_val) in codeword_vals.iter_mut().enumerate() {
                let mut base = 0;
                for opened_rows_per_query in proof.whir_proof.initial_round_opened_rows.iter() {
                    let opened_rows = &opened_rows_per_query[query_idx];
                    let width = opened_rows[0].len();
                    let num_chunks = width.div_ceil(CHUNK);

                    for chunk_idx in 0..num_chunks {
                        let chunk_start = chunk_idx * CHUNK;
                        let chunk_len = cmp::min(CHUNK, width - chunk_start);

                        let opened_chunk =
                            &opened_rows[coset_idx][chunk_start..chunk_start + chunk_len];

                        blob.codeword_value_accs.push(*codeword_val);

                        for (offset, &val) in opened_chunk.iter().enumerate() {
                            *codeword_val += local_mu_pows[base + chunk_start + offset] * val;
                        }
                    }
                    base += width;
                }
            }
        }
    }

    #[tracing::instrument(skip_all)]
    fn generate_blob(
        &self,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        proofs: &[&Proof<BabyBearPoseidon2Config>],
        preflights: &[&Preflight],
        exp_bits_len_gen: &ExpBitsLenTraceGenerator,
    ) -> WhirBlobCpu {
        let params = &child_vk.inner.params;
        let k_whir = params.k_whir();
        let num_queries_per_round = num_queries_per_round(params);
        let num_initial_queries = *num_queries_per_round.first().unwrap_or(&0);
        let num_whir_rounds = params.num_whir_rounds();
        let query_layout = WhirQueryLayout::new(proofs.len(), &num_queries_per_round);
        let queries_per_proof = query_layout.queries_per_proof();
        let sumcheck_rows_per_proof = params.num_whir_sumcheck_rounds();
        let fold_records_per_proof = queries_per_proof * ((1 << k_whir) - 1);
        let tidx_layout = PerProofLayout::new(proofs.len(), num_whir_rounds);

        let per_proof_commit_widths: Vec<Vec<usize>> = proofs
            .iter()
            .map(|proof| {
                proof
                    .whir_proof
                    .initial_round_opened_rows
                    .iter()
                    .map(|openings| openings[0][0].len())
                    .collect()
            })
            .collect();
        let accs_layout =
            CodewordAccsLayout::new(num_initial_queries, 1 << k_whir, &per_proof_commit_widths);
        let mu_pows_layout =
            VariablePerProofLayout::new((0..proofs.len()).map(|i| accs_layout.total_width(i)));

        let mut blob = WhirBlobBuilder::with_capacities(
            proofs.len(),
            num_whir_rounds,
            sumcheck_rows_per_proof,
            queries_per_proof,
            fold_records_per_proof,
            accs_layout.len(),
            mu_pows_layout.len(),
        );

        for (proof_idx, (proof, preflight)) in zip(proofs, preflights).enumerate() {
            let mu = preflight.stacking.stacking_batching_challenge;
            let local_mu_pows = mu
                .powers()
                .take(accs_layout.total_width(proof_idx))
                .collect_vec();

            Self::append_derived_whir_data_for_proof(
                proof,
                preflight,
                &local_mu_pows,
                params,
                &query_layout,
                &mut blob,
            );

            blob.whir_round_tidx_per_round
                .extend_from_slice(&preflight.whir.whir_round_tidx_per_round);
            blob.query_tidx_per_round
                .extend_from_slice(&preflight.whir.query_tidx_per_round);

            Self::enqueue_pow_requests_for_proof(
                exp_bits_len_gen,
                preflight,
                params,
                &query_layout,
            );

            Self::append_initial_opened_values_accs_for_proof(
                &mut blob,
                &accs_layout,
                proof_idx,
                proof,
                &local_mu_pows,
                num_initial_queries,
                k_whir,
            );

            blob.mu_pows.extend(local_mu_pows);
        }
        let initial_claim_layout = PerProofLayout::new(proofs.len(), num_whir_rounds + 1);
        let pre_query_claim_layout = PerProofLayout::new(proofs.len(), num_whir_rounds);
        let sumcheck_layout = PerProofLayout::new(proofs.len(), sumcheck_rows_per_proof);
        let fold_layout = PerProofLayout::new(proofs.len(), fold_records_per_proof);
        let layouts = WhirBlobLayouts {
            tidx_layout,
            initial_claim_layout,
            pre_query_claim_layout,
            sumcheck_layout,
            query_layout,
            fold_layout,
            accs_layout,
            mu_pows_layout,
        };

        blob.into_blob(layouts)
    }
}

impl TraceGenModule<GlobalCtxCpu, CpuBackend<BabyBearPoseidon2Config>> for WhirModule {
    type ModuleSpecificCtx<'a> = ExpBitsLenTraceGenerator;

    #[tracing::instrument(skip_all)]
    fn generate_proving_ctxs(
        &self,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        preflights: &[Preflight],
        exp_bits_len_gen: &ExpBitsLenTraceGenerator,
        required_heights: Option<&[usize]>,
    ) -> Option<Vec<AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>>> {
        let proofs = proofs.iter().collect_vec();
        let preflights = preflights.iter().collect_vec();
        let blob = self.generate_blob(child_vk, &proofs, &preflights, exp_bits_len_gen);
        let ctx = (
            StandardTracegenCtx {
                vk: child_vk,
                proofs: &proofs,
                preflights: &preflights,
            },
            &blob,
        );

        let chips = [
            WhirModuleChip::WhirRound,
            WhirModuleChip::Sumcheck,
            WhirModuleChip::Query,
            WhirModuleChip::InitialOpenedValues,
            WhirModuleChip::NonInitialOpenedValues,
            WhirModuleChip::Folding,
            WhirModuleChip::FinalPolyMleEval,
            WhirModuleChip::FinalPolyQueryEval,
        ];
        let span = tracing::Span::current();
        chips
            .par_iter()
            .map(|chip| {
                let _guard = span.enter();
                chip.generate_proving_ctx(
                    &ctx,
                    required_heights.map(|heights| heights[chip.index()]),
                )
            })
            .collect::<Vec<_>>()
            .into_iter()
            .collect()
    }
}

fn binary_k_fold(
    mut values: Vec<EF>,
    alphas: &[EF],
    base_coset_shift: F,
    whir_round: usize,
    query_idx: usize,
    records: &mut Vec<FoldRecord>,
) -> EF {
    let n = values.len();
    let k = alphas.len();
    debug_assert_eq!(n, 1 << k);

    let omega_k = F::two_adic_generator(k);
    let omega_k_inv = omega_k.inverse();

    let tw = omega_k.powers().take(1 << (k - 1)).collect_vec();
    let inv_tw = omega_k_inv.powers().take(1 << (k - 1)).collect_vec();

    for (j, (&alpha, coset_shift, coset_shift_inv)) in izip!(
        alphas.iter(),
        base_coset_shift.exp_powers_of_2(),
        base_coset_shift.inverse().exp_powers_of_2()
    )
    .enumerate()
    {
        let m = n >> (j + 1);
        let (lo, hi) = values.split_at_mut(m);

        for i in 0..m {
            let eval_point = tw[i << j] * coset_shift;
            let eval_point_inv = inv_tw[i << j] * coset_shift_inv;
            let new_val = lo[i] + (alpha - eval_point) * (lo[i] - hi[i]) * eval_point_inv.halve();
            records.push(FoldRecord::new(
                whir_round,
                query_idx,
                tw[i << j],
                coset_shift,
                m,
                i,
                j + 1,
                lo[i],
                hi[i],
                new_val,
                alpha,
            ));
            lo[i] = new_val;
        }
    }
    values[0]
}

#[derive(Clone, Copy, strum_macros::Display, EnumDiscriminants)]
#[strum_discriminants(derive(strum_macros::EnumCount))]
#[strum_discriminants(repr(usize))]
enum WhirModuleChip {
    WhirRound,
    Sumcheck,
    Query,
    InitialOpenedValues,
    NonInitialOpenedValues,
    Folding,
    FinalPolyMleEval,
    FinalPolyQueryEval,
}

impl WhirModuleChip {
    fn index(&self) -> usize {
        WhirModuleChipDiscriminants::from(self) as usize
    }
}

impl RowMajorChip<F> for WhirModuleChip {
    type Ctx<'a> = (StandardTracegenCtx<'a>, &'a WhirBlobCpu);

    #[tracing::instrument(
        name = "wrapper.generate_trace",
        level = "trace",
        skip_all,
        fields(air = %self)
    )]
    fn generate_trace(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>> {
        use WhirModuleChip::*;
        match self {
            WhirRound => whir_round::WhirRoundTraceGenerator.generate_trace(ctx, required_height),
            Sumcheck => sumcheck::WhirSumcheckTraceGenerator.generate_trace(ctx, required_height),
            Query => query::WhirQueryTraceGenerator.generate_trace(ctx, required_height),
            InitialOpenedValues => {
                let initial_opened_values_ctx = initial_opened_values::InitialOpenedValuesCtx {
                    vk: ctx.0.vk,
                    proofs: ctx.0.proofs,
                    preflights: ctx.0.preflights,
                    blob: ctx.1,
                };
                initial_opened_values::InitialOpenedValuesTraceGenerator
                    .generate_trace(&initial_opened_values_ctx, required_height)
            }
            NonInitialOpenedValues => {
                non_initial_opened_values::NonInitialOpenedValuesTraceGenerator
                    .generate_trace(ctx, required_height)
            }
            Folding => folding::FoldingTraceGenerator.generate_trace(ctx, required_height),
            FinalPolyMleEval => final_poly_mle_eval::FinalPolyMleEvalTraceGenerator
                .generate_trace(ctx, required_height),
            FinalPolyQueryEval => {
                let records = final_poly_query_eval::build_final_poly_query_eval_records(
                    &ctx.0.vk.inner.params,
                    ctx.0.proofs,
                    ctx.0.preflights,
                    &ctx.1.zis,
                );
                let final_poly_query_eval_ctx = final_poly_query_eval::FinalPolyQueryEvalCtx {
                    vk: ctx.0.vk,
                    preflights: ctx.0.preflights,
                    records: records.as_slice(),
                };
                final_poly_query_eval::FinalPolyQueryEvalTraceGenerator
                    .generate_trace(&final_poly_query_eval_ctx, required_height)
            }
        }
    }
}

#[cfg(feature = "cuda")]
mod cuda_tracegen {
    use std::cmp;

    use openvm_cuda_backend::GpuBackend;
    use openvm_cuda_common::d_buffer::DeviceBuffer;
    use openvm_poseidon2_air::POSEIDON2_WIDTH;
    use openvm_stark_backend::p3_maybe_rayon::prelude::*;
    use openvm_stark_sdk::config::baby_bear_poseidon2::CHUNK;

    use super::*;
    use crate::{
        cuda::{
            preflight::PreflightGpu, proof::ProofGpu, to_device_or_nullptr, vk::VerifyingKeyGpu,
            GlobalCtxGpu,
        },
        tracegen::cuda::{generate_gpu_proving_ctx, StandardTracegenGpuCtx},
        whir::cuda_abi::PoseidonStatePair,
    };

    pub(in crate::whir) struct WhirBlobGpu {
        pub zis: DeviceBuffer<F>,
        pub zi_roots: DeviceBuffer<F>,
        pub yis: DeviceBuffer<EF>,
        pub raw_queries: DeviceBuffer<F>,
        pub codeword_value_accs: DeviceBuffer<EF>,
        pub poseidon_states: DeviceBuffer<PoseidonStatePair>,
        pub rows_per_proof_offsets: DeviceBuffer<usize>,
        pub commits_per_proof_offsets: DeviceBuffer<usize>,
        pub stacking_chunks_offsets: DeviceBuffer<usize>,
        pub stacking_widths_offsets: DeviceBuffer<usize>,
        pub mus: DeviceBuffer<EF>,
        pub mu_pows: DeviceBuffer<EF>,
        pub codeword_opened_values: DeviceBuffer<EF>,
        pub codeword_states: DeviceBuffer<F>,
        pub folding_records: DeviceBuffer<FoldRecord>,
    }

    fn build_initial_poseidon_state_pairs(
        proofs: &[&Proof<BabyBearPoseidon2Config>],
        preflights: &[&Preflight],
    ) -> Vec<PoseidonStatePair> {
        let expected_pairs: usize = preflights
            .iter()
            .flat_map(|preflight| {
                preflight
                    .initial_row_states
                    .iter()
                    .flat_map(|commit| commit.iter().flat_map(|query| query.iter()))
            })
            .map(|coset_states| coset_states.len())
            .sum();

        let mut pairs = Vec::with_capacity(expected_pairs);
        for (proof, preflight) in proofs.iter().zip(preflights) {
            if preflight.initial_row_states.is_empty() {
                continue;
            }
            let num_commits = preflight.initial_row_states.len();
            let num_queries = preflight.initial_row_states[0].len();
            let num_cosets = preflight.initial_row_states[0][0].len();

            for query_idx in 0..num_queries {
                for coset_idx in 0..num_cosets {
                    for commit_idx in 0..num_commits {
                        let chunk_states =
                            &preflight.initial_row_states[commit_idx][query_idx][coset_idx];
                        let opened_row = &proof.whir_proof.initial_round_opened_rows[commit_idx]
                            [query_idx][coset_idx];
                        for (chunk_idx, &post_state) in chunk_states.iter().enumerate() {
                            let mut pre_state = if chunk_idx > 0 {
                                chunk_states[chunk_idx - 1]
                            } else {
                                [F::ZERO; POSEIDON2_WIDTH]
                            };
                            let chunk_start = chunk_idx * CHUNK;
                            let chunk_len = cmp::min(CHUNK, opened_row.len() - chunk_start);
                            pre_state[..chunk_len]
                                .copy_from_slice(&opened_row[chunk_start..chunk_start + chunk_len]);
                            pairs.push(PoseidonStatePair {
                                pre_state,
                                post_state,
                            });
                        }
                    }
                }
            }
        }
        pairs
    }

    impl WhirBlobGpu {
        fn new(
            proofs: &[&Proof<BabyBearPoseidon2Config>],
            preflights: &[&Preflight],
            blob: &WhirBlobCpu,
        ) -> Self {
            let mus = to_device_or_nullptr(
                &preflights
                    .iter()
                    .map(|preflight| preflight.stacking.stacking_batching_challenge)
                    .collect_vec(),
            )
            .unwrap();
            let zis = to_device_or_nullptr(blob.zis.as_slice()).unwrap();
            let zi_roots = to_device_or_nullptr(blob.zi_roots.as_slice()).unwrap();
            let yis = to_device_or_nullptr(blob.yis.as_slice()).unwrap();
            let raw_queries = to_device_or_nullptr(
                &preflights
                    .iter()
                    .flat_map(|preflight| preflight.whir.queries.iter().copied())
                    .collect_vec(),
            )
            .unwrap();
            let accs_layout = blob.codeword_value_accs.layout();
            let rows_per_proof_offsets =
                to_device_or_nullptr(accs_layout.rows_per_proof_offsets()).unwrap();
            let commits_per_proof_offsets =
                to_device_or_nullptr(accs_layout.commits_per_proof_offsets()).unwrap();
            let stacking_chunks_offsets =
                to_device_or_nullptr(accs_layout.stacking_chunks_offsets()).unwrap();
            let stacking_widths_offsets =
                to_device_or_nullptr(accs_layout.stacking_widths_offsets()).unwrap();
            let mu_pows = to_device_or_nullptr(blob.mu_pows.as_slice()).unwrap();
            let codeword_value_accs =
                to_device_or_nullptr(blob.codeword_value_accs.as_slice()).unwrap();

            // Build poseidon state pairs in kernel order: [query][coset][commit][chunk]
            let poseidon_states_host = build_initial_poseidon_state_pairs(proofs, preflights);
            let poseidon_states = to_device_or_nullptr(&poseidon_states_host).unwrap();

            let folding_records = to_device_or_nullptr(blob.fold_records.as_slice()).unwrap();

            let codeword_opened_values_cap: usize = proofs
                .iter()
                .map(|p| {
                    p.whir_proof
                        .codeword_opened_values
                        .iter()
                        .map(|r| r.iter().map(|q| q.len()).sum::<usize>())
                        .sum::<usize>()
                })
                .sum();
            let mut codeword_opened_values_host = Vec::with_capacity(codeword_opened_values_cap);
            for proof in proofs.iter() {
                for round in proof.whir_proof.codeword_opened_values.iter() {
                    for query in round.iter() {
                        codeword_opened_values_host.extend_from_slice(query);
                    }
                }
            }
            let codeword_opened_values =
                to_device_or_nullptr(&codeword_opened_values_host).unwrap();

            // Must be in same order as codeword_opened_values
            let mut codeword_states_host =
                Vec::with_capacity(codeword_opened_values_cap * POSEIDON2_WIDTH);
            for preflight in preflights.iter() {
                for round in preflight.codeword_states.iter() {
                    for query in round.iter() {
                        for state in query.iter() {
                            codeword_states_host.extend_from_slice(state);
                        }
                    }
                }
            }
            let codeword_states = to_device_or_nullptr(&codeword_states_host).unwrap();

            WhirBlobGpu {
                zis,
                zi_roots,
                yis,
                raw_queries,
                codeword_value_accs,
                poseidon_states,
                rows_per_proof_offsets,
                commits_per_proof_offsets,
                stacking_chunks_offsets,
                stacking_widths_offsets,
                mus,
                mu_pows,
                folding_records,
                codeword_opened_values,
                codeword_states,
            }
        }
    }

    impl ModuleChip<GpuBackend> for WhirModuleChip {
        type Ctx<'a> = (StandardTracegenGpuCtx<'a>, &'a WhirBlobGpu, &'a WhirBlobCpu);

        fn generate_proving_ctx(
            &self,
            ctx: &Self::Ctx<'_>,
            required_height: Option<usize>,
        ) -> Option<AirProvingContext<GpuBackend>> {
            match self {
                WhirModuleChip::InitialOpenedValues => {
                    initial_opened_values::cuda::InitialOpenedValuesGpuTraceGenerator
                        .generate_proving_ctx(
                            &initial_opened_values::cuda::InitialOpenedValuesGpuCtx {
                                num_proofs: ctx.0.proofs.len(),
                                blob: ctx.1,
                                params: &ctx.0.vk.system_params,
                            },
                            required_height,
                        )
                }
                WhirModuleChip::NonInitialOpenedValues => {
                    non_initial_opened_values::cuda::NonInitialOpenedValuesGpuTraceGenerator
                        .generate_proving_ctx(
                            &non_initial_opened_values::cuda::NonInitialOpenedValuesGpuCtx {
                                blob: ctx.1,
                                params: &ctx.0.vk.system_params,
                            },
                            required_height,
                        )
                }
                WhirModuleChip::FinalPolyQueryEval => {
                    let proofs_cpu = ctx.0.proofs.iter().map(|proof| &proof.cpu).collect_vec();
                    let preflights_cpu = ctx
                        .0
                        .preflights
                        .iter()
                        .map(|preflight| &preflight.cpu)
                        .collect_vec();
                    let records = final_poly_query_eval::build_final_poly_query_eval_records(
                        &ctx.0.vk.system_params,
                        &proofs_cpu,
                        &preflights_cpu,
                        &ctx.2.zis,
                    );
                    final_poly_query_eval::cuda::FinalPolyQueryEvalGpuTraceGenerator
                        .generate_proving_ctx(
                            &final_poly_query_eval::cuda::FinalPolyQueryEvalGpuCtx {
                                records: records.as_slice(),
                                params: &ctx.0.vk.system_params,
                                preflights: ctx.0.preflights,
                            },
                            required_height,
                        )
                }
                WhirModuleChip::Folding => folding::cuda::FoldingGpuTraceGenerator
                    .generate_proving_ctx(
                        &folding::cuda::FoldingGpuCtx {
                            blob: ctx.1,
                            params: &ctx.0.vk.system_params,
                            num_proofs: ctx.0.proofs.len(),
                        },
                        required_height,
                    ),
                _ => {
                    let proofs_cpu = ctx.0.proofs.iter().map(|p| &p.cpu).collect_vec();
                    let preflights_cpu = ctx.0.preflights.iter().map(|p| &p.cpu).collect_vec();
                    let cpu_ctx = (
                        StandardTracegenCtx {
                            vk: &ctx.0.vk.cpu,
                            proofs: &proofs_cpu,
                            preflights: &preflights_cpu,
                        },
                        ctx.2,
                    );
                    generate_gpu_proving_ctx(self, &cpu_ctx, required_height)
                }
            }
        }
    }

    impl TraceGenModule<GlobalCtxGpu, GpuBackend> for WhirModule {
        type ModuleSpecificCtx<'a> = ExpBitsLenTraceGenerator;

        #[tracing::instrument(skip_all)]
        fn generate_proving_ctxs(
            &self,
            child_vk: &VerifyingKeyGpu,
            proofs: &[ProofGpu],
            preflights: &[PreflightGpu],
            exp_bits_len_gen: &ExpBitsLenTraceGenerator,
            required_heights: Option<&[usize]>,
        ) -> Option<Vec<AirProvingContext<GpuBackend>>> {
            let proofs_cpu = proofs.iter().map(|proof| &proof.cpu).collect_vec();
            let preflights_cpu = preflights
                .iter()
                .map(|preflight| &preflight.cpu)
                .collect_vec();

            let blob = self.generate_blob(
                &child_vk.cpu,
                &proofs_cpu,
                &preflights_cpu,
                exp_bits_len_gen,
            );
            let blob_gpu = WhirBlobGpu::new(&proofs_cpu, &preflights_cpu, &blob);
            let ctx = (
                StandardTracegenGpuCtx {
                    vk: child_vk,
                    proofs,
                    preflights,
                },
                &blob_gpu,
                &blob,
            );

            let gpu_chips = [
                WhirModuleChip::InitialOpenedValues,
                WhirModuleChip::NonInitialOpenedValues,
                WhirModuleChip::Folding,
                WhirModuleChip::FinalPolyQueryEval,
            ];
            let cpu_chips = [
                WhirModuleChip::WhirRound,
                WhirModuleChip::Sumcheck,
                WhirModuleChip::Query,
                WhirModuleChip::FinalPolyMleEval,
            ];

            // Launch all CUDA tracegen kernels serially first (default stream).
            let indexed_gpu_ctxs = gpu_chips
                .iter()
                .map(|chip| {
                    (
                        chip.index(),
                        chip.generate_proving_ctx(
                            &ctx,
                            required_heights.map(|heights| heights[chip.index()]),
                        ),
                    )
                })
                .collect::<Vec<_>>();

            // Then run CPU tracegen for the remaining AIRs in parallel.
            let span = tracing::Span::current();
            let indexed_cpu_ctxs = cpu_chips
                .par_iter()
                .map(|chip| {
                    let _guard = span.enter();
                    (
                        chip.index(),
                        chip.generate_proving_ctx(
                            &ctx,
                            required_heights.map(|heights| heights[chip.index()]),
                        ),
                    )
                })
                .collect::<Vec<_>>();

            indexed_gpu_ctxs
                .into_iter()
                .chain(indexed_cpu_ctxs)
                .sorted_by(|a, b| a.0.cmp(&b.0))
                .map(|(_idx, ctx)| ctx)
                .collect()
        }
    }
}
