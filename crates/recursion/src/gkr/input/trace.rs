use core::borrow::BorrowMut;

use p3_field::{FieldAlgebra, FieldExtensionAlgebra};
use p3_matrix::dense::RowMajorMatrix;
use stark_backend_v2::{
    F,
    poseidon2::sponge::FiatShamirTranscript,
    proof::{GkrLayerClaims, Proof},
};

use super::GkrInputCols;
use crate::system::Preflight;

pub fn generate_trace<TS: FiatShamirTranscript>(
    proof: &Proof,
    preflight: &Preflight<TS>,
) -> RowMajorMatrix<F> {
    let width = GkrInputCols::<F>::width();

    let gkr_proof = &proof.gkr_proof;

    let n_max = preflight.proof_shape.n_max;
    let n_logup = preflight.proof_shape.n_logup;
    let l_skip = preflight.proof_shape.l_skip;
    let n_global = preflight.proof_shape.n_global;

    let num_layers = if n_logup != 0 { n_logup + l_skip } else { 0 };

    let tidx_beg = preflight.proof_shape.post_tidx;

    let logup_pow_witness = gkr_proof.logup_pow_witness;
    let logup_pow_sample = preflight.transcript.data[tidx_beg + 1];

    let num_rows: usize = 1;
    let mut trace = vec![F::ZERO; num_rows.next_power_of_two() * width];
    let cols: &mut GkrInputCols<F> = trace[0..width].borrow_mut();

    // Constant for all rows
    cols.is_real = F::ONE;

    cols.num_layers = F::from_canonical_usize(num_layers);

    cols.tidx_beg = F::from_canonical_usize(tidx_beg);
    cols.tidx_after_gkr_layers = F::from_canonical_usize(preflight.gkr.post_layer_tidx);
    cols.tidx_end = F::from_canonical_usize(preflight.gkr.post_tidx);

    cols.n_logup = F::from_canonical_usize(n_logup);
    cols.n_max = F::from_canonical_usize(n_max);
    cols.n_global = F::from_canonical_usize(n_global);

    cols.is_n_logup_zero = F::from_bool(n_logup == 0);
    cols.is_n_logup_equal_to_n_global = F::from_bool(n_logup == n_global);

    cols.logup_pow_witness = logup_pow_witness;
    cols.logup_pow_sample = logup_pow_sample;

    cols.input_layer_claim = if let Some(last_layer_claims) = gkr_proof.claims_per_layer.last() {
        let &GkrLayerClaims {
            p_xi_0,
            p_xi_1,
            q_xi_0,
            q_xi_1,
        } = last_layer_claims;
        let rho = preflight.gkr.xi[0].1;
        let input_layer_p_claim = p_xi_0 + rho * (p_xi_1 - p_xi_0);
        let input_layer_q_claim = q_xi_0 + rho * (q_xi_1 - q_xi_0);
        [
            input_layer_p_claim.as_base_slice().try_into().unwrap(),
            input_layer_q_claim.as_base_slice().try_into().unwrap(),
        ]
    } else {
        [[F::ZERO; 4], [F::ZERO; 4]]
    };

    RowMajorMatrix::new(trace, width)
}
