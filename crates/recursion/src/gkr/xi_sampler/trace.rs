use core::borrow::BorrowMut;

use p3_field::{FieldAlgebra, FieldExtensionAlgebra};
use p3_matrix::dense::RowMajorMatrix;
use stark_backend_v2::{D_EF, F, proof::Proof};

use super::GkrXiSamplerCols;
use crate::system::Preflight;

pub fn generate_trace(_proof: &Proof, preflight: &Preflight) -> RowMajorMatrix<F> {
    let width = GkrXiSamplerCols::<F>::width();

    let n_max = preflight.proof_shape.n_max;
    let n_logup = preflight.proof_shape.n_logup;
    let l_skip = preflight.proof_shape.l_skip;

    let num_layers = if n_logup != 0 { n_logup + l_skip } else { 0 };

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

        cols.is_enabled = F::ONE;
        // TODO(ayush): fix this
        cols.proof_idx = F::ZERO;
        cols.is_first_challenge = F::from_bool(i == 0);

        cols.tidx = F::from_canonical_usize(tidx);

        cols.challenge_idx = F::from_canonical_usize(xi_index);
        cols.challenge = xi.as_base_slice().try_into().unwrap();

        tidx += D_EF;
    }

    // Fill padding rows
    for row_data in trace.chunks_mut(width).skip(num_rows) {
        let cols: &mut GkrXiSamplerCols<F> = row_data.borrow_mut();
        cols.proof_idx = F::ONE;
    }

    RowMajorMatrix::new(trace, width)
}
