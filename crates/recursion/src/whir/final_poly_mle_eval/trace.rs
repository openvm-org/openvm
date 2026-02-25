use core::borrow::BorrowMut;

use openvm_stark_backend::poly_common::Squarable;
use openvm_stark_sdk::config::baby_bear_poseidon2::{D_EF, EF, F};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;

use crate::{
    tracegen::{RowMajorChip, StandardTracegenCtx},
    utils::pow_tidx_count,
    whir::{final_poly_mle_eval::air::FinalyPolyMleEvalCols, WhirBlobCpu},
};

pub(crate) struct FinalPolyMleEvalTraceGenerator;

impl RowMajorChip<F> for FinalPolyMleEvalTraceGenerator {
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
        let whir_round_tidx_per_round = &blob.whir_round_tidx_per_round;
        let final_poly_at_u = &blob.final_poly_at_u;
        let eq_partials = &blob.eq_partials;
        let sumcheck_rows_per_proof = eq_partials.layout().items_per_proof();
        debug_assert_eq!(proofs.len(), preflights.len());

        let params = &ctx.0.vk.inner.params;
        let num_vars = params.log_final_poly_len();
        let num_whir_rounds = params.num_whir_rounds();

        let rows_per_proof = (1 << (num_vars + 1)) - 1;
        let total_valid_rows = rows_per_proof * proofs.len();
        let height = if let Some(h) = required_height {
            if h < total_valid_rows {
                return None;
            }
            h
        } else {
            total_valid_rows.next_power_of_two()
        };
        let width = FinalyPolyMleEvalCols::<F>::width();

        let mut trace = vec![F::ZERO; height * width];

        let folding_pow_offset = pow_tidx_count(params.whir.folding_pow_bits);
        let tidx_base_offset = params.k_whir() * (D_EF * 3 + folding_pow_offset);
        let final_round = num_whir_rounds
            .checked_sub(1)
            .expect("WHIR must have at least one round");
        let mut global_row = 0usize;

        for (proof_idx, proof) in proofs.iter().enumerate() {
            let preflight = &preflights[proof_idx];

            let num_sumcheck_rounds = params.n_stack + params.l_skip - num_vars;
            let (stacking_first, stacking_rest) =
                preflight.stacking.sumcheck_rnd.split_first().unwrap();
            let u_all = stacking_first
                .exp_powers_of_2()
                .take(params.l_skip)
                .chain(stacking_rest.iter().copied())
                .collect::<Vec<_>>();
            let eval_points = &u_all[num_sumcheck_rounds..num_sumcheck_rounds + num_vars];

            let result = final_poly_at_u[proof_idx];
            let eq_alpha_u = eq_partials[(proof_idx, sumcheck_rows_per_proof - 1)];

            let mut buf = proof.whir_proof.final_poly.clone();
            let tidx = whir_round_tidx_per_round[(proof_idx, final_round)] + tidx_base_offset;

            let mut row_in_proof = 0usize;

            for layer in 0..=num_vars {
                let len = 1 << (num_vars - layer);
                let point = if layer > 0 {
                    eval_points[num_vars - layer]
                } else {
                    EF::ZERO
                };

                let (left_slice, right_slice) = buf.split_at_mut(len);
                for node_idx in 0..len {
                    let row_idx = global_row + row_in_proof;
                    let row = &mut trace[row_idx * width..(row_idx + 1) * width];
                    let cols: &mut FinalyPolyMleEvalCols<F> = row.borrow_mut();

                    let (left, right, value) = if layer == 0 {
                        let coeff = left_slice[node_idx];
                        (coeff, EF::ZERO, coeff)
                    } else {
                        let l = left_slice[node_idx];
                        let r = right_slice[node_idx];
                        // Evaluation-form MLE folding: value = l + (r - l) * point
                        let value = l + (r - l) * point;
                        left_slice[node_idx] = value;
                        (l, r, value)
                    };

                    cols.is_enabled = F::ONE;
                    cols.proof_idx = F::from_usize(proof_idx);
                    cols.is_root = F::from_bool(layer == num_vars);
                    cols.layer = F::from_usize(layer);
                    cols.layer_inv = cols.layer.try_inverse().unwrap_or_default();
                    cols.node_idx = F::from_usize(node_idx);
                    cols.node_idx_inv = cols.node_idx.try_inverse().unwrap_or_default();
                    cols.tidx_final_poly_start = F::from_usize(tidx);
                    cols.point
                        .copy_from_slice(point.as_basis_coefficients_slice());
                    cols.left_value
                        .copy_from_slice(left.as_basis_coefficients_slice());
                    cols.right_value
                        .copy_from_slice(right.as_basis_coefficients_slice());
                    cols.value
                        .copy_from_slice(value.as_basis_coefficients_slice());
                    cols.result
                        .copy_from_slice(result.as_basis_coefficients_slice());
                    cols.eq_alpha_u
                        .copy_from_slice(eq_alpha_u.as_basis_coefficients_slice());
                    cols.num_nodes_in_layer = F::from_usize(1 << (num_vars - layer));
                    cols.is_nonleaf_and_first_in_layer = F::from_bool(layer != 0 && node_idx == 0);

                    row_in_proof += 1;
                }
            }

            debug_assert_eq!(row_in_proof, rows_per_proof);
            global_row += row_in_proof;
        }

        Some(RowMajorMatrix::new(trace, width))
    }
}
