use core::borrow::BorrowMut;

use openvm_stark_backend::{keygen::types::MultiStarkVerifyingKey, proof::Proof, SystemParams};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, EF, F};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;

use crate::{
    system::Preflight,
    tracegen::RowMajorChip,
    utils::FlattenedVec,
    whir::{
        final_poly_query_eval::air::FinalPolyQueryEvalCols, num_queries_per_round, WhirQueryLayout,
    },
};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FinalPolyQueryEvalRecord {
    pub alpha: EF,
    pub query_pow: EF,
    pub gamma_eq_acc: EF,
    pub horner_acc: EF,
    pub final_poly_coeff: EF,
    pub final_value_acc: EF,
    pub gamma_pow: EF,
}

pub(in crate::whir) fn build_final_poly_query_eval_records(
    params: &SystemParams,
    proofs: &[&Proof<BabyBearPoseidon2Config>],
    preflights: &[&Preflight],
    zis: &FlattenedVec<F, WhirQueryLayout>,
) -> Vec<FinalPolyQueryEvalRecord> {
    debug_assert_eq!(proofs.len(), preflights.len());
    let k_whir = params.k_whir();
    let query_layout = zis.layout();
    let final_poly_len = 1usize << params.log_final_poly_len();
    let num_whir_rounds = params.num_whir_rounds();

    let mut rows_per_proof = 0usize;
    for whir_round in 0..num_whir_rounds {
        let eq_phase_len = k_whir * (num_whir_rounds - (whir_round + 1));
        let query_count = query_layout.round_num_queries(whir_round) + 1;
        rows_per_proof += query_count * (eq_phase_len + final_poly_len);
    }
    let mut records = Vec::with_capacity(rows_per_proof * proofs.len());

    for (proof_idx, proof) in proofs.iter().enumerate() {
        let preflight = preflights[proof_idx];

        let final_poly_coeffs = &proof.whir_proof.final_poly;
        debug_assert_eq!(final_poly_coeffs.len(), final_poly_len);

        let gammas = &preflight.whir.gammas;
        let z0s = &preflight.whir.z0s;
        let alphas = &preflight.whir.alphas;

        let mut final_value_acc = EF::ZERO;
        for whir_round in 0..num_whir_rounds {
            let eq_phase_len = k_whir * (num_whir_rounds - (whir_round + 1));
            let gamma = gammas[whir_round];
            let mut gamma_pow = gamma;

            let query_count = query_layout.round_num_queries(whir_round) + 1;
            for query_idx in 0..query_count {
                let mut gamma_eq_acc = gamma_pow;
                let mut horner_acc = EF::ZERO;
                let mut query_pow = if query_idx == 0 {
                    if whir_round < num_whir_rounds - 1 {
                        z0s[whir_round]
                    } else {
                        EF::ZERO
                    }
                } else {
                    EF::from(zis[(proof_idx, whir_round, query_idx - 1)])
                };

                for eval_idx in 0..eq_phase_len {
                    let alpha = alphas[(whir_round + 1) * k_whir + eval_idx];
                    records.push(FinalPolyQueryEvalRecord {
                        alpha,
                        query_pow,
                        gamma_eq_acc,
                        horner_acc,
                        final_poly_coeff: EF::ZERO,
                        final_value_acc,
                        gamma_pow,
                    });
                    gamma_eq_acc *= EF::ONE - query_pow - alpha + query_pow * alpha.double();
                    query_pow *= query_pow;
                }

                for &final_poly_coeff in final_poly_coeffs.iter().rev() {
                    records.push(FinalPolyQueryEvalRecord {
                        alpha: EF::ZERO,
                        query_pow,
                        gamma_eq_acc,
                        horner_acc,
                        final_poly_coeff,
                        final_value_acc,
                        gamma_pow,
                    });
                    horner_acc = horner_acc * query_pow + final_poly_coeff;
                }

                if query_idx != 0 || whir_round != num_whir_rounds - 1 {
                    final_value_acc += gamma_eq_acc * horner_acc;
                }

                gamma_pow *= gamma;
            }
        }
    }

    records
}

#[inline]
pub(in crate::whir) fn compute_round_offsets(
    num_whir_rounds: usize,
    k_whir: usize,
    final_poly_len: usize,
    num_queries_per_round: &[usize],
) -> Vec<usize> {
    let mut offsets = Vec::with_capacity(num_whir_rounds + 1);
    offsets.push(0);
    let mut rows_acc = 0;
    for (whir_round, &num_queries) in num_queries_per_round.iter().enumerate() {
        let eq_phase_len = k_whir * (num_whir_rounds - (whir_round + 1));
        let query_count = num_queries + 1;
        let rows_per_round = query_count * (eq_phase_len + final_poly_len);
        rows_acc += rows_per_round;
        offsets.push(rows_acc);
    }
    offsets
}

#[inline]
fn compute_indices_from_row_idx(
    row_idx: usize,
    rows_per_proof: usize,
    round_offsets: &[usize],
    num_whir_rounds: usize,
    k_whir: usize,
    final_poly_len: usize,
) -> (usize, usize, usize, usize, usize, usize) {
    debug_assert!(rows_per_proof > 0);
    debug_assert!(!round_offsets.is_empty());
    debug_assert_eq!(*round_offsets.last().unwrap(), rows_per_proof);

    let proof_idx = row_idx / rows_per_proof;
    let row_in_proof = row_idx % rows_per_proof;

    let round_idx = round_offsets
        .partition_point(|&start| start <= row_in_proof)
        .saturating_sub(1);
    debug_assert!(round_idx + 1 < round_offsets.len());

    let round_start = round_offsets[round_idx];
    let row_in_round = row_in_proof - round_start;

    let eq_phase_len_round = k_whir * (num_whir_rounds - (round_idx + 1));
    let rows_per_query = eq_phase_len_round + final_poly_len;

    let query_idx = row_in_round / rows_per_query;
    let row_in_query = row_in_round % rows_per_query;

    let (phase_idx, eval_idx) = if row_in_query < eq_phase_len_round {
        (0, row_in_query)
    } else {
        (1, row_in_query - eq_phase_len_round)
    };

    (
        proof_idx,
        round_idx,
        query_idx,
        phase_idx,
        eval_idx,
        eq_phase_len_round,
    )
}

pub(crate) struct FinalPolyQueryEvalCtx<'a> {
    pub vk: &'a MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
    pub records: &'a [FinalPolyQueryEvalRecord],
    pub preflights: &'a [&'a Preflight],
}

pub(crate) struct FinalPolyQueryEvalTraceGenerator;

impl RowMajorChip<F> for FinalPolyQueryEvalTraceGenerator {
    type Ctx<'a> = FinalPolyQueryEvalCtx<'a>;

    #[tracing::instrument(level = "trace", skip_all)]
    fn generate_trace(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>> {
        let mvk = ctx.vk;
        let records = ctx.records;
        let preflights = ctx.preflights;

        let params = &mvk.inner.params;
        let k_whir = params.k_whir();
        let num_queries_per_round = num_queries_per_round(params);
        let final_poly_len = 1usize << params.log_final_poly_len();
        let num_whir_rounds = params.num_whir_rounds();

        let round_offsets = compute_round_offsets(
            num_whir_rounds,
            k_whir,
            final_poly_len,
            &num_queries_per_round,
        );
        let rows_per_proof = *round_offsets
            .last()
            .expect("round offsets vector must include sentinel");
        debug_assert!(rows_per_proof > 0);
        let total_valid_rows = records.len();
        debug_assert_eq!(preflights.len(), total_valid_rows / rows_per_proof);
        let height = if let Some(h) = required_height {
            if h < total_valid_rows {
                return None;
            }
            h
        } else {
            total_valid_rows.next_power_of_two()
        };
        let width = FinalPolyQueryEvalCols::<F>::width();
        let mut trace = F::zero_vec(width * height);
        trace
            .par_chunks_exact_mut(width)
            .zip(records.par_iter())
            .enumerate()
            .for_each(|(row_idx, (row, record))| {
                let (proof_idx, whir_round, query_idx, phase_idx, eval_idx, num_alphas) =
                    compute_indices_from_row_idx(
                        row_idx,
                        rows_per_proof,
                        &round_offsets,
                        num_whir_rounds,
                        k_whir,
                        final_poly_len,
                    );

                let is_first_in_phase = eval_idx == 0;
                let is_first_in_query = is_first_in_phase && (phase_idx == 0 || num_alphas == 0);
                let is_first_in_round = is_first_in_query && query_idx == 0;
                let is_first_in_proof = is_first_in_round && whir_round == 0;

                let num_in_domain_queries = num_queries_per_round[whir_round];
                let is_same_phase = if phase_idx == 0 {
                    eval_idx + 1 < num_alphas
                } else {
                    eval_idx + 1 < final_poly_len
                };
                let is_same_query = is_same_phase || phase_idx == 0;
                let is_same_round = is_same_query || query_idx < num_in_domain_queries;
                let is_same_proof = is_same_round || whir_round + 1 < num_whir_rounds;

                let cols: &mut FinalPolyQueryEvalCols<F> = row.borrow_mut();
                cols.is_enabled = F::ONE;
                cols.proof_idx = F::from_usize(proof_idx);
                cols.whir_round = F::from_usize(whir_round);
                cols.query_idx = F::from_usize(query_idx);
                cols.phase_idx = F::from_usize(phase_idx);
                cols.eval_idx = F::from_usize(eval_idx);
                cols.is_first_in_proof = F::from_bool(is_first_in_proof);
                cols.is_first_in_round = F::from_bool(is_first_in_round);
                cols.is_first_in_query = F::from_bool(is_first_in_query);
                cols.is_first_in_phase = F::from_bool(is_first_in_phase);
                cols.is_query_zero = F::from_bool(query_idx == 0);
                cols.is_last_round = F::from_bool(whir_round + 1 == num_whir_rounds);

                let is_q0_last = query_idx == 0 && (whir_round + 1 == num_whir_rounds);
                let do_carry = (!is_same_query) && is_same_proof && !is_q0_last;
                cols.do_carry = F::from_bool(do_carry);

                cols.gamma_eq_acc
                    .copy_from_slice(record.gamma_eq_acc.as_basis_coefficients_slice());

                cols.alpha
                    .copy_from_slice(record.alpha.as_basis_coefficients_slice());
                let gamma = preflights[proof_idx].whir.gammas[whir_round];
                cols.gamma
                    .copy_from_slice(gamma.as_basis_coefficients_slice());
                cols.gamma_pow
                    .copy_from_slice(record.gamma_pow.as_basis_coefficients_slice());

                cols.query_pow
                    .copy_from_slice(record.query_pow.as_basis_coefficients_slice());
                cols.final_poly_coeff
                    .copy_from_slice(record.final_poly_coeff.as_basis_coefficients_slice());
                cols.final_value_acc
                    .copy_from_slice(record.final_value_acc.as_basis_coefficients_slice());
                cols.horner_acc
                    .copy_from_slice(record.horner_acc.as_basis_coefficients_slice());
            });

        Some(RowMajorMatrix::new(trace, width))
    }
}
