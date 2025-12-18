use core::borrow::BorrowMut;

use openvm_circuit_primitives::{TraceSubRowGenerator, is_zero::IsZeroSubAir};
use openvm_stark_backend::p3_maybe_rayon::prelude::*;
use p3_field::{FieldAlgebra, FieldExtensionAlgebra};
use p3_matrix::dense::RowMajorMatrix;
use stark_backend_v2::{EF, F};

use super::GkrInputCols;

#[derive(Debug, Clone, Default)]
pub struct GkrInputRecord {
    pub tidx: usize,
    pub n_logup: usize,
    pub n_max: usize,
    pub logup_pow_witness: F,
    pub logup_pow_sample: F,
    pub alpha_logup: EF,
    pub input_layer_claim: [EF; 2],
}

#[tracing::instrument(level = "trace", skip_all)]
pub fn generate_trace(gkr_input_records: &[GkrInputRecord], q0_claims: &[EF]) -> RowMajorMatrix<F> {
    debug_assert_eq!(gkr_input_records.len(), q0_claims.len());

    let width = GkrInputCols::<F>::width();

    if gkr_input_records.is_empty() {
        let trace = vec![F::ZERO; width];
        return RowMajorMatrix::new(trace, width);
    }

    // Each record generates exactly 1 row
    let total_rows = gkr_input_records.len();
    let padded_rows = total_rows.next_power_of_two();
    let mut trace = vec![F::ZERO; padded_rows * width];

    let (data_slice, _) = trace.split_at_mut(total_rows * width);

    // Process each proof row
    data_slice
        .par_chunks_mut(width)
        .zip(gkr_input_records.par_iter().zip(q0_claims.par_iter()))
        .enumerate()
        .for_each(|(proof_idx, (row_data, (record, q0_claim)))| {
            let cols: &mut GkrInputCols<F> = row_data.borrow_mut();

            cols.is_enabled = F::ONE;
            cols.proof_idx = F::from_canonical_usize(proof_idx);

            cols.tidx = F::from_canonical_usize(record.tidx);

            cols.n_logup = F::from_canonical_usize(record.n_logup);
            cols.n_max = F::from_canonical_usize(record.n_max);
            cols.is_n_max_greater_than_n_logup = F::from_bool(record.n_max > record.n_logup);

            IsZeroSubAir.generate_subrow(
                cols.n_logup,
                (&mut cols.is_n_logup_zero_aux.inv, &mut cols.is_n_logup_zero),
            );

            cols.logup_pow_witness = record.logup_pow_witness;
            cols.logup_pow_sample = record.logup_pow_sample;

            cols.q0_claim = q0_claim.as_base_slice().try_into().unwrap();
            cols.alpha_logup = record.alpha_logup.as_base_slice().try_into().unwrap();
            cols.input_layer_claim = [
                record.input_layer_claim[0]
                    .as_base_slice()
                    .try_into()
                    .unwrap(),
                record.input_layer_claim[1]
                    .as_base_slice()
                    .try_into()
                    .unwrap(),
            ];
        });

    RowMajorMatrix::new(trace, width)
}
