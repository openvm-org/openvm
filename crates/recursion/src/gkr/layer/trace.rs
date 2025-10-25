use core::borrow::BorrowMut;

use p3_field::{FieldAlgebra, FieldExtensionAlgebra};
use p3_matrix::dense::RowMajorMatrix;
use stark_backend_v2::{D_EF, EF, F, poly_common::interpolate_linear_at_01, proof::Proof};

use super::GkrLayerCols;
use crate::system::Preflight;

pub fn generate_trace(proof: &Proof, preflight: &Preflight) -> RowMajorMatrix<F> {
    let width = GkrLayerCols::<F>::width();

    let n_logup = preflight.proof_shape.n_logup;
    let l_skip = preflight.proof_shape.l_skip;
    let num_layers = if n_logup != 0 { n_logup + l_skip } else { 0 };

    let mut trace = vec![F::ZERO; num_layers.next_power_of_two() * width];

    // Skip grinding nonce observe and grinding challenge sampling
    // Skip alpha_logup, beta_logup sampling
    let mut tidx = preflight.proof_shape.post_tidx + 2 + 2 * D_EF + D_EF;
    let gkr_proof = &proof.gkr_proof;
    let (mut numer_claim, mut denom_claim) = (EF::ZERO, EF::ZERO);

    for (layer_idx, row_data) in trace.chunks_mut(width).take(num_layers).enumerate() {
        let cols: &mut GkrLayerCols<F> = row_data.borrow_mut();

        // Constant for all rows
        cols.is_enabled = F::ONE;
        cols.is_first_layer = F::from_bool(layer_idx == 0);

        cols.layer_idx = F::from_canonical_usize(layer_idx);
        cols.tidx = F::from_canonical_usize(tidx);

        // Sample lambda
        let lambda = if layer_idx == 0 {
            EF::ZERO
        } else {
            let lambda_slice = &preflight.transcript[tidx..tidx + D_EF];
            cols.lambda = lambda_slice.try_into().unwrap();
            tidx += D_EF;
            EF::from_base_slice(lambda_slice)
        };

        // Skip sumcheck rounds for this layer
        tidx += layer_idx * 4 * D_EF;

        // Observe layer claims: numer0, denom0, numer1, denom1
        let layer_claims = &gkr_proof.claims_per_layer[layer_idx];
        cols.p_xi_0 = layer_claims.p_xi_0.as_base_slice().try_into().unwrap();
        cols.q_xi_0 = layer_claims.q_xi_0.as_base_slice().try_into().unwrap();
        cols.p_xi_1 = layer_claims.p_xi_1.as_base_slice().try_into().unwrap();
        cols.q_xi_1 = layer_claims.q_xi_1.as_base_slice().try_into().unwrap();
        tidx += 4 * D_EF;

        // Sample mu
        let mu_slice = &preflight.transcript[tidx..tidx + D_EF];
        let mu_ef = EF::from_base_slice(mu_slice);
        cols.mu = mu_slice.try_into().unwrap();
        tidx += D_EF;

        if layer_idx == 0 {
            cols.sumcheck_claim_in = gkr_proof.q0_claim.as_base_slice().try_into().unwrap();
        } else {
            let claim = numer_claim + lambda * denom_claim;
            cols.sumcheck_claim_in = claim.as_base_slice().try_into().unwrap();

            // Get new_claim and eq_at_r_prime from preflight (for layers 1..num_layers)
            let (_, eq_at_r_prime) = preflight.gkr.layer_sumcheck_output[layer_idx - 1];
            cols.eq_at_r_prime = eq_at_r_prime.as_base_slice().try_into().unwrap();
        }

        numer_claim = interpolate_linear_at_01(&[layer_claims.p_xi_0, layer_claims.p_xi_1], mu_ef);
        denom_claim = interpolate_linear_at_01(&[layer_claims.q_xi_0, layer_claims.q_xi_1], mu_ef);

        cols.numer_claim = numer_claim.as_base_slice().try_into().unwrap();
        cols.denom_claim = denom_claim.as_base_slice().try_into().unwrap();
    }

    // Fill padding rows
    for row_data in trace.chunks_mut(width).skip(num_layers) {
        let cols: &mut GkrLayerCols<F> = row_data.borrow_mut();
        cols.proof_idx = F::ONE;
    }

    RowMajorMatrix::new(trace, width)
}
