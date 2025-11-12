use core::borrow::BorrowMut;

use openvm_stark_backend::p3_maybe_rayon::prelude::*;
use p3_field::{FieldAlgebra, FieldExtensionAlgebra};
use p3_matrix::dense::RowMajorMatrix;
use stark_backend_v2::{D_EF, EF, F};

use super::GkrXiSamplerCols;

#[derive(Debug, Clone, Default)]
pub struct GkrXiSamplerRecord {
    pub tidx: usize,
    pub idx: usize,
    pub xis: Vec<EF>,
}

#[tracing::instrument(name = "generate_trace(GkrXiSamplerAir)", skip_all)]
pub fn generate_trace(xi_sampler_records: &[GkrXiSamplerRecord]) -> RowMajorMatrix<F> {
    let width = GkrXiSamplerCols::<F>::width();

    if xi_sampler_records.is_empty() {
        let trace = vec![F::ZERO; width];
        return RowMajorMatrix::new(trace, width);
    }

    // Calculate rows per proof (minimum 1 row per proof)
    let rows_per_proof: Vec<usize> = xi_sampler_records
        .iter()
        .map(|record| record.xis.len().max(1))
        .collect();

    // Calculate total rows
    let total_rows: usize = rows_per_proof.iter().sum();
    let padded_rows = total_rows.next_power_of_two();

    let mut trace = vec![F::ZERO; padded_rows * width];

    // Split trace into chunks for each proof
    let (data_slice, _) = trace.split_at_mut(total_rows * width);
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
                cols.proof_idx = F::from_canonical_usize(proof_idx);
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
                    cols.proof_idx = F::from_canonical_usize(proof_idx);

                    cols.is_enabled = F::ONE;
                    cols.is_first_challenge = F::from_bool(row_idx == 0);
                    cols.tidx = F::from_canonical_usize(*tidx);
                    cols.idx = F::from_canonical_usize(*idx);
                    cols.xi = xi.as_base_slice().try_into().unwrap();
                });
        });

    RowMajorMatrix::new(trace, width)
}
