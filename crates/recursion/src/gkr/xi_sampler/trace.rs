use core::borrow::BorrowMut;

use p3_field::{FieldAlgebra, FieldExtensionAlgebra};
use p3_matrix::dense::RowMajorMatrix;
use stark_backend_v2::{
    D_EF, F, keygen::types::MultiStarkVerifyingKeyV2, poseidon2::sponge::FiatShamirTranscript,
    proof::Proof,
};

use super::GkrXiSamplerCols;
use crate::system::Preflight;

pub fn generate_trace<TS: FiatShamirTranscript>(
    _vk: &MultiStarkVerifyingKeyV2,
    _proof: &Proof,
    preflight: &Preflight<TS>,
) -> RowMajorMatrix<F> {
    let width = GkrXiSamplerCols::<F>::width();

    let n_max = preflight.proof_shape.n_max;
    let n_logup = preflight.proof_shape.n_logup;
    let l_skip = preflight.proof_shape.l_skip;

    let num_layers = if n_logup != 0 { n_logup + l_skip } else { 0 };
    let num_challenges = n_max + l_skip;

    let num_rows = if n_logup != 0 {
        n_max.saturating_sub(n_logup)
    } else {
        n_max + l_skip
    };

    let mut trace = vec![F::ZERO; num_rows.next_power_of_two() * width];

    let mut tidx = preflight.gkr.post_layer_tidx;
    for (i, row_data) in trace.chunks_mut(width).take(num_rows).enumerate() {
        let cols: &mut GkrXiSamplerCols<F> = row_data.borrow_mut();

        let xi_index = i + num_layers;
        let xi = preflight.gkr.xi[xi_index].1;

        cols.is_real = F::ONE;
        cols.is_first_row = F::from_bool(i == 0);
        cols.is_final_row = F::from_bool(i == num_rows - 1);

        cols.num_challenges = F::from_canonical_usize(num_challenges);
        cols.tidx = F::from_canonical_usize(tidx);

        cols.xi_index = F::from_canonical_usize(xi_index);
        cols.challenge = xi.as_base_slice().try_into().unwrap();

        tidx += D_EF;
    }

    RowMajorMatrix::new(trace, width)
}
