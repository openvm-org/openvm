use core::borrow::BorrowMut;

use openvm_stark_sdk::config::baby_bear_poseidon2::{CHUNK, F};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;

use crate::{
    tracegen::{RowMajorChip, StandardTracegenCtx},
    whir::{non_initial_opened_values::air::NonInitialOpenedValuesCols, WhirBlobCpu},
};

pub(crate) struct NonInitialOpenedValuesTraceGenerator;

impl RowMajorChip<F> for NonInitialOpenedValuesTraceGenerator {
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
        let zi_roots = &blob.zi_roots;
        let zis = &blob.zis;
        let yis = &blob.yis;
        debug_assert_eq!(proofs.len(), preflights.len());

        let params = &ctx.0.vk.inner.params;

        let num_rounds = params.num_whir_rounds();
        let query_layout = zi_roots.layout();
        let k_whir = params.k_whir();
        let omega_k = F::two_adic_generator(k_whir);
        let rows_per_query = 1 << k_whir;

        let mut round_row_offsets = Vec::with_capacity(num_rounds);
        round_row_offsets.push(0usize);
        for whir_round in 1..num_rounds {
            let num_queries = query_layout.round_num_queries(whir_round);
            let rows_this_round = num_queries * rows_per_query;
            round_row_offsets.push(round_row_offsets.last().unwrap() + rows_this_round);
        }
        let num_rows_per_proof = *round_row_offsets.last().unwrap();

        let num_valid_rows = num_rows_per_proof * preflights.len();
        let height = if let Some(h) = required_height {
            if h < num_valid_rows {
                return None;
            }
            h
        } else {
            num_valid_rows.next_power_of_two()
        };
        let width = NonInitialOpenedValuesCols::<F>::width();

        let mut trace = vec![F::ZERO; height * width];

        trace
            .par_chunks_exact_mut(width)
            .take(num_valid_rows)
            .enumerate()
            .for_each(|(row_idx, row)| {
                let proof_idx = row_idx / num_rows_per_proof;
                let i = row_idx % num_rows_per_proof;

                let proof = &proofs[proof_idx];
                let preflight = &preflights[proof_idx];

                let cols: &mut NonInitialOpenedValuesCols<F> = row.borrow_mut();
                cols.is_enabled = F::ONE;
                cols.proof_idx = F::from_usize(proof_idx);

                let round_minus_1 = round_row_offsets[1..].partition_point(|&offset| offset <= i);
                let whir_round = round_minus_1 + 1;
                let row_in_round = i - round_row_offsets[round_minus_1];

                let coset_idx = row_in_round % rows_per_query;
                let query_idx = row_in_round / rows_per_query;

                let is_first_in_proof = i == 0;
                let is_first_in_query = coset_idx == 0;
                let is_first_in_round = is_first_in_query && query_idx == 0;

                cols.whir_round = F::from_usize(whir_round);
                cols.query_idx = F::from_usize(query_idx);
                cols.coset_idx = F::from_usize(coset_idx);
                cols.twiddle = omega_k.exp_u64(coset_idx as u64);
                cols.is_first_in_proof = F::from_bool(is_first_in_proof);
                cols.is_first_in_round = F::from_bool(is_first_in_round);
                cols.is_first_in_query = F::from_bool(is_first_in_query);
                let query_offset = query_layout.round_query_range(whir_round).start;
                let query = (proof_idx, whir_round, query_idx);
                cols.zi_root = zi_roots[query];
                cols.value_hash.copy_from_slice(
                    &preflight.codeword_states[whir_round - 1][query_idx][coset_idx][..CHUNK],
                );
                let value =
                    &proof.whir_proof.codeword_opened_values[whir_round - 1][query_idx][coset_idx];
                cols.value
                    .copy_from_slice(value.as_basis_coefficients_slice());
                cols.merkle_idx_bit_src = preflight.whir.queries[query_offset + query_idx];
                cols.zi = zis[query];
                cols.yi
                    .copy_from_slice(yis[query].as_basis_coefficients_slice());
            });

        Some(RowMajorMatrix::new(trace, width))
    }
}
