use core::borrow::BorrowMut;
use std::cmp::max;

use openvm_circuit_primitives::{
    TraceSubRowGenerator, is_equal::IsEqSubAir, is_zero::IsZeroSubAir,
};
use p3_field::{FieldAlgebra, FieldExtensionAlgebra};
use p3_matrix::dense::RowMajorMatrix;
use stark_backend_v2::{
    F,
    proof::{GkrLayerClaims, Proof},
};

use super::GkrInputCols;
use crate::system::Preflight;

pub fn generate_trace(proof: &Proof, preflight: &Preflight) -> RowMajorMatrix<F> {
    let width = GkrInputCols::<F>::width();

    let gkr_proof = &proof.gkr_proof;

    let n_max = preflight.proof_shape.n_max;
    let n_logup = preflight.proof_shape.n_logup;
    let n_global = max(n_max, n_logup);

    let tidx = preflight.proof_shape.post_tidx;

    let logup_pow_witness = gkr_proof.logup_pow_witness;
    let logup_pow_sample = preflight.transcript[tidx + 1];

    let num_rows: usize = 1;
    let mut trace = vec![F::ZERO; num_rows.next_power_of_two() * width];
    let cols: &mut GkrInputCols<F> = trace[0..width].borrow_mut();

    // Constant for all rows
    cols.is_enabled = F::ONE;
    // TODO(ayush): fix this
    cols.proof_idx = F::ZERO;

    cols.tidx = F::from_canonical_usize(tidx);

    cols.n_logup = F::from_canonical_usize(n_logup);
    cols.n_max = F::from_canonical_usize(n_max);
    cols.n_global = F::from_canonical_usize(n_global);

    IsZeroSubAir.generate_subrow(
        F::from_canonical_usize(n_logup),
        (&mut cols.is_n_logup_zero_aux.inv, &mut cols.is_n_logup_zero),
    );
    IsEqSubAir.generate_subrow(
        (
            F::from_canonical_usize(n_logup),
            F::from_canonical_usize(n_global),
        ),
        (
            &mut cols.is_n_logup_equal_to_n_global_aux.inv,
            &mut cols.is_n_logup_equal_to_n_global,
        ),
    );

    cols.logup_pow_witness = logup_pow_witness;
    cols.logup_pow_sample = logup_pow_sample;

    cols.q0_claim = gkr_proof.q0_claim.as_base_slice().try_into().unwrap();
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
        [
            [F::ZERO; 4],
            core::array::from_fn(|i| {
                preflight.transcript.values()[preflight.proof_shape.post_tidx + 2 + i]
            }),
        ]
    };

    RowMajorMatrix::new(trace, width)
}
