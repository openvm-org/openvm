use core::borrow::BorrowMut;

use openvm_circuit_primitives::encoder::Encoder;
use openvm_stark_sdk::config::baby_bear_poseidon2::F;
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;

use crate::{
    tracegen::{RowMajorChip, StandardTracegenCtx},
    whir::{num_queries_per_round, whir_round::air::WhirRoundCols, WhirBlobCpu},
};

pub(crate) struct WhirRoundTraceGenerator;

impl RowMajorChip<F> for WhirRoundTraceGenerator {
    type Ctx<'a> = (StandardTracegenCtx<'a>, &'a WhirBlobCpu);

    #[tracing::instrument(level = "trace", skip_all)]
    fn generate_trace(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>> {
        let params = &ctx.0.vk.inner.params;
        let num_rounds = params.num_whir_rounds();
        // Encoder requires at least 2 flags to work correctly
        let encoder = Encoder::new(num_rounds.max(2), 2, false);
        match encoder.width() {
            1 => generate_trace_impl::<1>(ctx, &encoder, required_height),
            2 => generate_trace_impl::<2>(ctx, &encoder, required_height),
            3 => generate_trace_impl::<3>(ctx, &encoder, required_height),
            w => panic!("unsupported encoder width: {w}"),
        }
    }
}

fn generate_trace_impl<const ENC_WIDTH: usize>(
    ctx: &(StandardTracegenCtx<'_>, &WhirBlobCpu),
    whir_round_encoder: &Encoder,
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    let proofs = ctx.0.proofs;
    let preflights = ctx.0.preflights;
    let blob = ctx.1;
    let tidx_per_round = &blob.whir_round_tidx_per_round;
    let initial_claim_per_round = &blob.initial_claim_per_round;
    let post_sumcheck_claims = &blob.post_sumcheck_claims;
    let eq_partials = &blob.eq_partials;
    let final_poly_at_u = &blob.final_poly_at_u;
    debug_assert_eq!(proofs.len(), preflights.len());
    let sumcheck_rows_per_proof = post_sumcheck_claims.layout().items_per_proof();

    let params = &ctx.0.vk.inner.params;
    let k_whir = params.k_whir();
    let num_whir_rounds = params.num_whir_rounds();
    let initial_log_domain_size = params.n_stack + params.l_skip + params.log_blowup;
    let num_queries_per_round = num_queries_per_round(params);

    let rows_per_proof = num_whir_rounds;
    let total_valid_rows = rows_per_proof * proofs.len();

    let height = if let Some(h) = required_height {
        if h < total_valid_rows {
            return None;
        }
        h
    } else {
        total_valid_rows.next_power_of_two()
    };
    let width = WhirRoundCols::<F, ENC_WIDTH>::width();
    let mut trace = F::zero_vec(width * height);

    trace
        .par_chunks_exact_mut(width)
        .take(total_valid_rows)
        .enumerate()
        .for_each(|(row_idx, row)| {
            let proof_idx = row_idx / rows_per_proof;
            let i = row_idx % rows_per_proof;

            let proof = &proofs[proof_idx];
            let preflight = &preflights[proof_idx];
            let whir = &preflight.whir;
            let whir_proof = &proof.whir_proof;

            let final_poly_eval =
                final_poly_at_u[proof_idx] * eq_partials[(proof_idx, sumcheck_rows_per_proof - 1)];

            let cols: &mut WhirRoundCols<F, ENC_WIDTH> = row.borrow_mut();
            cols.is_enabled = F::ONE;
            cols.proof_idx = F::from_usize(proof_idx);
            cols.whir_round = F::from_usize(i);
            cols.is_first_in_proof = F::from_bool(i == 0);
            cols.tidx = F::from_usize(tidx_per_round[(proof_idx, i)]);
            cols.num_queries = F::from_usize(num_queries_per_round[i]);
            cols.omega = F::two_adic_generator(initial_log_domain_size - i);
            cols.claim.copy_from_slice(
                initial_claim_per_round[(proof_idx, i)].as_basis_coefficients_slice(),
            );
            cols.final_poly_mle_eval
                .copy_from_slice(final_poly_eval.as_basis_coefficients_slice());

            cols.next_claim.copy_from_slice(
                initial_claim_per_round[(proof_idx, i + 1)].as_basis_coefficients_slice(),
            );
            let sumcheck_idx = if k_whir == 0 {
                sumcheck_rows_per_proof - 1
            } else {
                (i + 1) * k_whir - 1
            };
            cols.post_sumcheck_claim.copy_from_slice(
                post_sumcheck_claims[(proof_idx, sumcheck_idx)].as_basis_coefficients_slice(),
            );
            cols.gamma
                .copy_from_slice(whir.gammas[i].as_basis_coefficients_slice());
            cols.query_pow_witness = whir_proof.query_phase_pow_witnesses[i];
            cols.query_pow_sample = whir.query_pow_samples[i];

            if i < rows_per_proof - 1 {
                cols.commit = whir_proof.codeword_commits[i];
                cols.z0
                    .copy_from_slice(whir.z0s[i].as_basis_coefficients_slice());
                cols.y0
                    .copy_from_slice(whir_proof.ood_values[i].as_basis_coefficients_slice());
            }

            let enc_pt = whir_round_encoder.get_flag_pt(i);
            for (j, &val) in enc_pt.iter().enumerate() {
                cols.whir_round_enc[j] = F::from_u32(val);
            }
        });

    Some(RowMajorMatrix::new(trace, width))
}
