use core::borrow::BorrowMut;

use p3_field::{FieldAlgebra, FieldExtensionAlgebra};
use p3_matrix::dense::RowMajorMatrix;
use stark_backend_v2::{D_EF, EF, F, poseidon2::sponge::FiatShamirTranscript, proof::Proof};

use super::GkrLayerCols;
use crate::system::Preflight;

pub fn generate_trace<TS: FiatShamirTranscript>(
    proof: &Proof,
    preflight: &Preflight<TS>,
) -> RowMajorMatrix<F> {
    let width = GkrLayerCols::<F>::width();

    let n_logup = preflight.proof_shape.n_logup;
    let l_skip = preflight.proof_shape.l_skip;
    let num_layers = if n_logup != 0 { n_logup + l_skip } else { 0 };

    let mut trace = vec![F::ZERO; num_layers.next_power_of_two() * width];

    // Skip grinding nonce observe and grinding challenge sampling
    // Skip alpha_logup, beta_logup sampling
    let mut tidx = preflight.proof_shape.post_tidx + 2 + 2 * D_EF;
    let gkr_proof = &proof.gkr_proof;

    for (layer_idx, row_data) in trace.chunks_mut(width).take(num_layers).enumerate() {
        let cols: &mut GkrLayerCols<F> = row_data.borrow_mut();

        // Constant for all rows
        cols.is_real = F::ONE;
        cols.num_layers = F::from_canonical_usize(num_layers);
        cols.tidx_beg = F::from_canonical_usize(tidx);

        cols.layer = F::from_canonical_usize(layer_idx);
        cols.is_first_layer = F::from_bool(layer_idx == 0);
        cols.is_final_layer = F::from_bool(layer_idx == num_layers - 1);

        if layer_idx == 0 {
            // Observe q0_claim
            cols.q0_claim = gkr_proof.q0_claim.as_base_slice().try_into().unwrap();
        } else {
            // Sample lambda
            cols.lambda = preflight.transcript.data[tidx..tidx + D_EF]
                .try_into()
                .unwrap();
        }
        tidx += D_EF;

        // Skip sumcheck rounds for this layer
        tidx += layer_idx * 4 * D_EF;
        cols.tidx_after_sumcheck = F::from_canonical_usize(tidx);

        // Observe layer claims: numer0, denom0, numer1, denom1
        let layer_claims = &gkr_proof.claims_per_layer[layer_idx];
        cols.numer0 = layer_claims.p_xi_0.as_base_slice().try_into().unwrap();
        cols.denom0 = layer_claims.q_xi_0.as_base_slice().try_into().unwrap();
        cols.numer1 = layer_claims.p_xi_1.as_base_slice().try_into().unwrap();
        cols.denom1 = layer_claims.q_xi_1.as_base_slice().try_into().unwrap();
        tidx += 4 * D_EF;

        // Sample mu
        let mu_slice = &preflight.transcript.data[tidx..tidx + D_EF];
        cols.mu = mu_slice.try_into().unwrap();
        tidx += D_EF;

        // Compute derived values
        let p_xi_0 = layer_claims.p_xi_0;
        let p_xi_1 = layer_claims.p_xi_1;
        let q_xi_0 = layer_claims.q_xi_0;
        let q_xi_1 = layer_claims.q_xi_1;

        // Extract mu as EF
        let mu = EF::from_base_slice(&cols.mu);

        // numer_claim = p_xi_0 + (p_xi_1 - p_xi_0) * mu
        let numer_claim = p_xi_0 + (p_xi_1 - p_xi_0) * mu;
        cols.numer_claim = numer_claim.as_base_slice().try_into().unwrap();

        // denom_claim = q_xi_0 + (q_xi_1 - q_xi_0) * mu
        let denom_claim = q_xi_0 + (q_xi_1 - q_xi_0) * mu;
        cols.denom_claim = denom_claim.as_base_slice().try_into().unwrap();

        if layer_idx == 0 {
            cols.claim = cols.q0_claim;
        } else {
            // Get precomputed claim from preflight (computed from previous layer's numer_claim +
            // lambda * denom_claim)
            let claim = preflight.gkr.layer_claim[layer_idx - 1];
            cols.claim = claim.as_base_slice().try_into().unwrap();

            // Get new_claim and eq_at_r_prime from preflight (for layers 1..num_layers)
            let (new_claim, eq_at_r_prime) = preflight.gkr.layer_sumcheck_output[layer_idx - 1];
            cols.new_claim = new_claim.as_base_slice().try_into().unwrap();
            cols.eq_at_r_prime = eq_at_r_prime.as_base_slice().try_into().unwrap();

            // Compute expected_claim = (p_cross_term + lambda * q_cross_term) * eq_at_r_prime
            let lambda = EF::from_base_slice(&cols.lambda);
            let p_cross_term = p_xi_0 * q_xi_1 + p_xi_1 * q_xi_0;
            let q_cross_term = q_xi_0 * q_xi_1;
            let expected_claim = (p_cross_term + lambda * q_cross_term) * eq_at_r_prime;
            cols.expected_claim = expected_claim.as_base_slice().try_into().unwrap();
        }

        // p_cross_term = p_xi_0 * q_xi_1 + p_xi_1 * q_xi_0
        let p_cross_term = p_xi_0 * q_xi_1 + p_xi_1 * q_xi_0;
        cols.p_cross_term = p_cross_term.as_base_slice().try_into().unwrap();

        // q_cross_term = q_xi_0 * q_xi_1
        let q_cross_term = q_xi_0 * q_xi_1;
        cols.q_cross_term = q_cross_term.as_base_slice().try_into().unwrap();
    }

    RowMajorMatrix::new(trace, width)
}
