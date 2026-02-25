use core::borrow::BorrowMut;

use openvm_stark_sdk::config::baby_bear_poseidon2::F;
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;

use crate::{
    tracegen::{RowMajorChip, StandardTracegenCtx},
    whir::{query::air::WhirQueryCols, WhirBlobCpu},
};

pub(crate) struct WhirQueryTraceGenerator;

impl RowMajorChip<F> for WhirQueryTraceGenerator {
    type Ctx<'a> = (StandardTracegenCtx<'a>, &'a WhirBlobCpu);

    #[tracing::instrument(level = "trace", skip_all)]
    fn generate_trace(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>> {
        let preflights = ctx.0.preflights;
        let blob = ctx.1;
        let query_tidx_per_round = &blob.query_tidx_per_round;
        let pre_query_claims = &blob.pre_query_claims;
        let initial_claim_per_round = &blob.initial_claim_per_round;
        let zi_roots = &blob.zi_roots;
        let zis = &blob.zis;
        let yis = &blob.yis;

        let params = &ctx.0.vk.inner.params;
        let m = params.n_stack + params.l_skip + params.log_blowup;
        let num_whir_rounds = params.num_whir_rounds();

        let query_layout = zi_roots.layout();
        debug_assert_eq!(query_layout.num_rounds(), num_whir_rounds);
        let num_rows_per_proof = query_layout.queries_per_proof();

        let num_valid_rows: usize = num_rows_per_proof * preflights.len();

        let height = if let Some(h) = required_height {
            if h < num_valid_rows {
                return None;
            }
            h
        } else {
            num_valid_rows.next_power_of_two()
        };
        let width = WhirQueryCols::<usize>::width();
        let mut trace = F::zero_vec(width * height);

        trace
            .par_chunks_exact_mut(width)
            .take(num_valid_rows)
            .enumerate()
            .for_each(|(row_idx, row)| {
                let proof_idx = row_idx / num_rows_per_proof;
                let i = row_idx % num_rows_per_proof;

                let (whir_round, query_idx) = query_layout.round_and_query_idx(i);

                let preflight = &preflights[proof_idx];
                let query_range = query_layout.round_query_range(whir_round);
                let query_offset = query_range.start;
                let num_queries = query_range.len();

                let cols: &mut WhirQueryCols<F> = row.borrow_mut();
                cols.is_enabled = F::ONE;
                cols.proof_idx = F::from_usize(proof_idx);
                cols.is_first_in_proof = F::from_bool(whir_round == 0 && query_idx == 0);
                cols.is_first_in_round = F::from_bool(query_idx == 0);
                cols.tidx =
                    F::from_usize(query_tidx_per_round[(proof_idx, whir_round)] + query_idx);
                cols.sample = preflight.whir.queries[query_offset + query_idx];
                cols.whir_round = F::from_usize(whir_round);
                cols.query_idx = F::from_usize(query_idx);
                cols.num_queries = F::from_usize(num_queries);
                cols.omega = F::two_adic_generator(m - whir_round);
                let query = (proof_idx, whir_round, query_idx);
                cols.zi = zis[query];
                cols.zi_root = zi_roots[query];
                cols.yi
                    .copy_from_slice(yis[query].as_basis_coefficients_slice());
                let gamma = preflight.whir.gammas[whir_round];
                cols.gamma
                    .copy_from_slice(gamma.as_basis_coefficients_slice());
                let gamma_pow = gamma.exp_u64(query_idx as u64 + 2);
                cols.gamma_pow
                    .copy_from_slice(gamma_pow.as_basis_coefficients_slice());
                let mut pre_claim = pre_query_claims[(proof_idx, whir_round)];
                for (query_idx, gamma_pow) in gamma.powers().skip(2).take(query_idx).enumerate() {
                    pre_claim += gamma_pow * yis[(proof_idx, whir_round, query_idx)];
                }
                if query_idx == num_queries - 1 {
                    debug_assert_eq!(
                        pre_claim + gamma_pow * yis[query],
                        initial_claim_per_round[(proof_idx, whir_round + 1)]
                    );
                }
                cols.pre_claim
                    .copy_from_slice(pre_claim.as_basis_coefficients_slice());
                cols.post_claim.copy_from_slice(
                    initial_claim_per_round[(proof_idx, whir_round + 1)]
                        .as_basis_coefficients_slice(),
                );
            });

        Some(RowMajorMatrix::new(trace, width))
    }
}
