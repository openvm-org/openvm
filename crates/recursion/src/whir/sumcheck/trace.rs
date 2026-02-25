use core::borrow::BorrowMut;

use itertools::Itertools;
use openvm_stark_backend::poly_common::Squarable;
use openvm_stark_sdk::config::baby_bear_poseidon2::{D_EF, F};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;

use crate::{
    tracegen::{RowMajorChip, StandardTracegenCtx},
    utils::pow_tidx_count,
    whir::{num_queries_per_round, sumcheck::air::SumcheckCols, WhirBlobCpu},
};

pub(crate) struct WhirSumcheckTraceGenerator;

impl RowMajorChip<F> for WhirSumcheckTraceGenerator {
    type Ctx<'a> = (StandardTracegenCtx<'a>, &'a WhirBlobCpu);

    #[tracing::instrument(level = "trace", skip_all)]
    fn generate_trace(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>> {
        let proofs = ctx.0.proofs;
        let preflights = ctx.0.preflights;
        let blob = ctx.1;
        let tidx_per_round = &blob.whir_round_tidx_per_round;
        let initial_claim_per_round = &blob.initial_claim_per_round;
        let post_sumcheck_claims = &blob.post_sumcheck_claims;
        let eq_partials = &blob.eq_partials;
        debug_assert_eq!(proofs.len(), preflights.len());

        let params = &ctx.0.vk.inner.params;
        let k_whir = params.k_whir();
        let num_whir_rounds = params.num_whir_rounds();

        let whir_opening_point_per_proof = preflights
            .iter()
            .map(|preflight| {
                let sumcheck_rnd = &preflight.stacking.sumcheck_rnd;
                sumcheck_rnd[0]
                    .exp_powers_of_2()
                    .take(params.l_skip)
                    .chain(sumcheck_rnd.iter().skip(1).copied())
                    .collect_vec()
            })
            .collect_vec();

        let num_queries_per_round = num_queries_per_round(params);
        let mut alpha_lookup_counts = vec![0usize; params.num_whir_sumcheck_rounds()];
        let mut base = 0usize;
        for r in 0..num_whir_rounds {
            let q = num_queries_per_round[r];
            for j in 0..k_whir {
                alpha_lookup_counts[r * k_whir + j] = base + q * (1 << (k_whir - 1 - j));
            }
            base += q + 1; // OOD query (or final poly) + in-domain queries
        }

        let rows_per_proof = params.num_whir_sumcheck_rounds();
        let total_valid_rows = rows_per_proof * proofs.len();

        let width = SumcheckCols::<F>::width();
        let height = if let Some(h) = required_height {
            if h < total_valid_rows {
                return None;
            }
            h
        } else {
            total_valid_rows.next_power_of_two()
        };
        let mut trace = F::zero_vec(width * height);

        trace
            .par_chunks_exact_mut(width)
            .take(total_valid_rows)
            .enumerate()
            .for_each(|(row_idx, row)| {
                let proof_idx = row_idx / rows_per_proof;
                let i = row_idx % rows_per_proof;

                let whir_round = i / k_whir;
                let j = i % k_whir;

                let proof = &proofs[proof_idx];
                let whir = &preflights[proof_idx].whir;

                let num_rounds = params.num_whir_rounds();
                debug_assert_eq!(whir.alphas.len(), num_rounds * k_whir);

                let is_first_in_group = j == 0;
                let last_group_row_idx = (whir_round + 1) * k_whir - 1;
                let folding_pow_offset = pow_tidx_count(params.whir.folding_pow_bits);
                let tidx =
                    tidx_per_round[(proof_idx, whir_round)] + (3 * D_EF + folding_pow_offset) * j;

                let cols: &mut SumcheckCols<F> = row.borrow_mut();
                cols.is_enabled = F::ONE;
                cols.proof_idx = F::from_usize(proof_idx);
                cols.is_first_in_proof = F::from_bool(i == 0);
                cols.is_first_in_round = F::from_bool(is_first_in_group);
                cols.whir_round = F::from_usize(whir_round);
                cols.subidx = F::from_usize(j);
                cols.tidx = F::from_usize(tidx);
                let sumcheck_polys = &proof.whir_proof.whir_sumcheck_polys[i];
                cols.ev1
                    .copy_from_slice(sumcheck_polys[0].as_basis_coefficients_slice());
                cols.ev2
                    .copy_from_slice(sumcheck_polys[1].as_basis_coefficients_slice());
                cols.folding_pow_witness = proof.whir_proof.folding_pow_witnesses[i];
                cols.folding_pow_sample = whir.folding_pow_samples[i];
                cols.eq_partial
                    .copy_from_slice(eq_partials[(proof_idx, i)].as_basis_coefficients_slice());
                cols.alpha
                    .copy_from_slice(whir.alphas[i].as_basis_coefficients_slice());
                cols.u.copy_from_slice(
                    whir_opening_point_per_proof[proof_idx][i].as_basis_coefficients_slice(),
                );
                cols.pre_claim.copy_from_slice(
                    if is_first_in_group {
                        initial_claim_per_round[(proof_idx, whir_round)]
                    } else {
                        post_sumcheck_claims[(proof_idx, i - 1)]
                    }
                    .as_basis_coefficients_slice(),
                );
                cols.post_group_claim.copy_from_slice(
                    post_sumcheck_claims[(proof_idx, last_group_row_idx)]
                        .as_basis_coefficients_slice(),
                );
                cols.alpha_lookup_count =
                    F::from_usize(alpha_lookup_counts[whir_round * k_whir + j]);
            });

        Some(RowMajorMatrix::new(trace, width))
    }
}
