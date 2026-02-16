use core::borrow::BorrowMut;

use openvm_circuit_primitives::{is_zero::IsZeroSubAir, TraceSubRowGenerator};
use openvm_stark_backend::p3_maybe_rayon::prelude::*;
use openvm_stark_sdk::config::baby_bear_poseidon2::{EF, F};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;

use super::GkrInputCols;
use crate::tracegen::RowMajorChip;

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

pub struct GkrInputTraceGenerator;

impl RowMajorChip<F> for GkrInputTraceGenerator {
    // (gkr_input_records, q0_claims)
    type Ctx<'a> = (&'a [GkrInputRecord], &'a [EF]);

    #[tracing::instrument(level = "trace", skip_all)]
    fn generate_trace(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>> {
        let (gkr_input_records, q0_claims) = ctx;
        debug_assert_eq!(gkr_input_records.len(), q0_claims.len());

        let width = GkrInputCols::<F>::width();

        // Each record generates exactly 1 row
        let num_valid_rows = gkr_input_records.len();
        let height = if let Some(height) = required_height {
            if height < num_valid_rows {
                return None;
            }
            height
        } else {
            num_valid_rows.next_power_of_two()
        };
        let mut trace = vec![F::ZERO; height * width];

        let (data_slice, _) = trace.split_at_mut(num_valid_rows * width);

        // Process each proof row
        data_slice
            .par_chunks_mut(width)
            .zip(gkr_input_records.par_iter().zip(q0_claims.par_iter()))
            .enumerate()
            .for_each(|(proof_idx, (row_data, (record, q0_claim)))| {
                let cols: &mut GkrInputCols<F> = row_data.borrow_mut();

                cols.is_enabled = F::ONE;
                cols.proof_idx = F::from_usize(proof_idx);

                cols.tidx = F::from_usize(record.tidx);

                cols.n_logup = F::from_usize(record.n_logup);
                cols.n_max = F::from_usize(record.n_max);
                cols.is_n_max_greater_than_n_logup = F::from_bool(record.n_max > record.n_logup);

                IsZeroSubAir.generate_subrow(
                    cols.n_logup,
                    (&mut cols.is_n_logup_zero_aux.inv, &mut cols.is_n_logup_zero),
                );

                cols.logup_pow_witness = record.logup_pow_witness;
                cols.logup_pow_sample = record.logup_pow_sample;

                cols.q0_claim = q0_claim.as_basis_coefficients_slice().try_into().unwrap();
                cols.alpha_logup = record
                    .alpha_logup
                    .as_basis_coefficients_slice()
                    .try_into()
                    .unwrap();
                cols.input_layer_claim = [
                    record.input_layer_claim[0]
                        .as_basis_coefficients_slice()
                        .try_into()
                        .unwrap(),
                    record.input_layer_claim[1]
                        .as_basis_coefficients_slice()
                        .try_into()
                        .unwrap(),
                ];
            });

        Some(RowMajorMatrix::new(trace, width))
    }
}
