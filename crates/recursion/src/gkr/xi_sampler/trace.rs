use core::borrow::BorrowMut;

use openvm_stark_backend::p3_maybe_rayon::prelude::*;
use openvm_stark_sdk::config::baby_bear_poseidon2::{D_EF, EF, F};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;

use super::GkrXiSamplerCols;
use crate::tracegen::RowMajorChip;

#[derive(Debug, Clone, Default)]
pub struct GkrXiSamplerRecord {
    pub tidx: usize,
    pub idx: usize,
    pub xis: Vec<EF>,
}

pub struct GkrXiSamplerTraceGenerator;

impl RowMajorChip<F> for GkrXiSamplerTraceGenerator {
    // xi_sampler_records
    type Ctx<'a> = &'a [GkrXiSamplerRecord];

    #[tracing::instrument(level = "trace", skip_all)]
    fn generate_trace(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>> {
        let xi_sampler_records = ctx;
        let width = GkrXiSamplerCols::<F>::width();

        // Calculate rows per proof (minimum 1 row per proof)
        let rows_per_proof: Vec<usize> = xi_sampler_records
            .iter()
            .map(|record| record.xis.len().max(1))
            .collect();

        // Calculate total rows
        let num_valid_rows: usize = rows_per_proof.iter().sum();
        let height = if let Some(height) = required_height {
            if height < num_valid_rows {
                return None;
            }
            height
        } else {
            num_valid_rows.next_power_of_two()
        };

        let mut trace = vec![F::ZERO; height * width];

        // Split trace into chunks for each proof
        let (data_slice, _) = trace.split_at_mut(num_valid_rows * width);
        let mut trace_slices: Vec<&mut [F]> = Vec::with_capacity(rows_per_proof.len());
        let mut remaining = data_slice;

        for &num_rows in &rows_per_proof {
            let chunk_size = num_rows * width;
            let (chunk, rest) = remaining.split_at_mut(chunk_size);
            trace_slices.push(chunk);
            remaining = rest;
        }

        // Process each proof
        trace_slices
            .par_iter_mut()
            .zip(xi_sampler_records.par_iter())
            .enumerate()
            .for_each(|(proof_idx, (proof_trace, xi_sampler_record))| {
                if xi_sampler_record.xis.is_empty() {
                    debug_assert_eq!(proof_trace.len(), width);
                    let row_data = &mut proof_trace[..width];
                    let cols: &mut GkrXiSamplerCols<F> = row_data.borrow_mut();
                    cols.is_enabled = F::ONE;
                    cols.proof_idx = F::from_usize(proof_idx);
                    cols.is_first_challenge = F::ONE;
                    cols.is_dummy = F::ONE;
                    return;
                }

                let challenge_indices: Vec<usize> = (0..xi_sampler_record.xis.len())
                    .map(|i| xi_sampler_record.idx + i)
                    .collect();
                let tidxs: Vec<usize> = (0..xi_sampler_record.xis.len())
                    .map(|i| xi_sampler_record.tidx + i * D_EF)
                    .collect();

                proof_trace
                    .par_chunks_mut(width)
                    .zip(
                        xi_sampler_record
                            .xis
                            .par_iter()
                            .zip(challenge_indices.par_iter())
                            .zip(tidxs.par_iter()),
                    )
                    .enumerate()
                    .for_each(|(row_idx, (row_data, ((xi, idx), tidx)))| {
                        let cols: &mut GkrXiSamplerCols<F> = row_data.borrow_mut();
                        cols.proof_idx = F::from_usize(proof_idx);

                        cols.is_enabled = F::ONE;
                        cols.is_first_challenge = F::from_bool(row_idx == 0);
                        cols.tidx = F::from_usize(*tidx);
                        cols.idx = F::from_usize(*idx);
                        cols.xi = xi.as_basis_coefficients_slice().try_into().unwrap();
                    });
            });

        Some(RowMajorMatrix::new(trace, width))
    }
}
