use core::borrow::BorrowMut;

use p3_field::{FieldAlgebra, FieldExtensionAlgebra};
use p3_matrix::dense::RowMajorMatrix;
use stark_backend_v2::{
    D_EF, EF, F, keygen::types::MultiStarkVerifyingKeyV2, poseidon2::sponge::FiatShamirTranscript,
    proof::Proof,
};

use super::GkrLayerSumcheckCols;
use crate::system::Preflight;

pub fn generate_trace<TS: FiatShamirTranscript>(
    _vk: &MultiStarkVerifyingKeyV2,
    proof: &Proof,
    preflight: &Preflight<TS>,
) -> RowMajorMatrix<F> {
    let width = GkrLayerSumcheckCols::<F>::width();

    let gkr_proof = &proof.gkr_proof;

    let n_logup = preflight.proof_shape.n_logup;
    let l_skip = preflight.proof_shape.l_skip;

    let num_layers = if n_logup != 0 { n_logup + l_skip } else { 0 };

    let tidx_start = preflight.proof_shape.post_tidx;

    let total_rounds = if num_layers != 0 {
        num_layers * (num_layers - 1) / 2
    } else {
        0
    };
    let mut trace = vec![F::ZERO; total_rounds.next_power_of_two() * width];

    // Skip grinding nonce observe and grinding challenge sampling
    // Skip alpha_logup, beta_logup sampling
    let mut tidx = tidx_start + 2 + 2 * D_EF;
    let mut gkr_r: Vec<EF> = Vec::new();
    let mut row_idx = 0;

    for layer_idx in 0..num_layers {
        // Increment tidx in GkrLayer
        tidx += D_EF;

        let mut gkr_r_prime = Vec::with_capacity(layer_idx);
        for (sumcheck_round, prev_challenge) in gkr_r.iter().enumerate() {
            let offset = row_idx * width;
            let row_data = &mut trace[offset..offset + width];
            let cols: &mut GkrLayerSumcheckCols<F> = row_data.borrow_mut();

            cols.is_real = F::ONE;
            cols.layer = F::from_canonical_usize(layer_idx);
            cols.is_final_layer = F::from_bool(layer_idx == num_layers - 1);
            cols.sumcheck_round = F::from_canonical_usize(sumcheck_round);
            cols.is_first_round = F::from_bool(sumcheck_round == 0);
            cols.is_final_round = F::from_bool(sumcheck_round == layer_idx - 1);
            cols.tidx_beg = F::from_canonical_usize(tidx);

            // Get the evaluations s(1), s(2), s(3) from the proof
            let poly = &gkr_proof.sumcheck_polys[layer_idx - 1][sumcheck_round];
            cols.ev1 = poly[0].as_base_slice().try_into().unwrap();
            cols.ev2 = poly[1].as_base_slice().try_into().unwrap();
            cols.ev3 = poly[2].as_base_slice().try_into().unwrap();
            // Observe the evaluations
            tidx += 3 * D_EF;

            // Sample challenge ri
            let challenge = EF::from_base_slice(&preflight.transcript.data[tidx..tidx + D_EF]);
            cols.challenge = challenge.as_base_slice().try_into().unwrap();
            tidx += D_EF;

            cols.prev_challenge = prev_challenge.as_base_slice().try_into().unwrap();

            // Get precomputed claim_in, claim_out, eq_in, eq_out from preflight
            let round_data = &preflight.gkr.sumcheck_round_data[row_idx];
            debug_assert_eq!(round_data.0, layer_idx, "layer_idx mismatch");
            debug_assert_eq!(round_data.1, sumcheck_round, "sumcheck_round mismatch");
            cols.claim_in = round_data.2.as_base_slice().try_into().unwrap();
            cols.claim_out = round_data.3.as_base_slice().try_into().unwrap();
            cols.eq_in = round_data.4.as_base_slice().try_into().unwrap();
            cols.eq_out = round_data.5.as_base_slice().try_into().unwrap();

            gkr_r_prime.push(challenge);
            row_idx += 1;
        }
        // Observe layer claims
        tidx += 4 * D_EF;

        // Sample mu
        let mu = EF::from_base_slice(&preflight.transcript.data[tidx..tidx + D_EF]);
        tidx += D_EF;

        gkr_r = std::iter::once(mu).chain(gkr_r_prime).collect();
    }

    RowMajorMatrix::new(trace, width)
}
