use core::borrow::BorrowMut;

use openvm_stark_backend::p3_maybe_rayon::prelude::*;
use p3_field::{FieldAlgebra, FieldExtensionAlgebra};
use p3_matrix::dense::RowMajorMatrix;
use stark_backend_v2::{D_EF, EF, F};

use super::{GkrLayerCols, air::reduce_to_single_evaluation};

/// Minimal record for parallel gkr layer trace generation
#[derive(Debug, Clone, Default)]
pub struct GkrLayerRecord {
    pub tidx: usize,
    pub layer_claims: Vec<[EF; 4]>,
    pub lambdas: Vec<EF>,
    pub eq_at_r_primes: Vec<EF>,
}

impl GkrLayerRecord {
    #[inline]
    fn layer_count(&self) -> usize {
        self.layer_claims.len()
    }

    #[inline]
    fn lambda_at(&self, layer_idx: usize) -> EF {
        layer_idx
            .checked_sub(1)
            .and_then(|idx| self.lambdas.get(idx))
            .copied()
            .unwrap_or(EF::ZERO)
    }

    #[inline]
    fn eq_at(&self, layer_idx: usize) -> EF {
        layer_idx
            .checked_sub(1)
            .and_then(|idx| self.eq_at_r_primes.get(idx))
            .copied()
            .unwrap_or(EF::ZERO)
    }

    #[inline]
    fn layer_tidx(&self, layer_idx: usize) -> usize {
        if layer_idx == 0 {
            self.tidx
        } else {
            let j = layer_idx;
            self.tidx + D_EF * (2 * j * j + 4 * j - 1)
        }
    }
}

#[tracing::instrument(name = "generate_trace(GkrLayerAir)", skip_all)]
pub fn generate_trace(
    gkr_layer_records: &[GkrLayerRecord],
    mus: &[Vec<EF>],
    q0_claims: &[EF],
) -> RowMajorMatrix<F> {
    debug_assert_eq!(gkr_layer_records.len(), mus.len());
    debug_assert_eq!(gkr_layer_records.len(), q0_claims.len());

    let width = GkrLayerCols::<F>::width();

    if gkr_layer_records.is_empty() {
        let trace = vec![F::ZERO; width];
        return RowMajorMatrix::new(trace, width);
    }

    // Calculate rows per proof (each record has layer_claims.len() rows)
    let rows_per_proof: Vec<usize> = gkr_layer_records
        .iter()
        .map(|record| record.layer_claims.len().max(1))
        .collect();

    // Calculate total rows
    let total_rows: usize = rows_per_proof.iter().sum();
    let padded_rows = total_rows.next_power_of_two();
    let mut trace = vec![F::ZERO; padded_rows * width];

    // Split trace into chunks for each proof and process in parallel
    let (data_slice, _) = trace.split_at_mut(total_rows * width);
    let mut trace_slices: Vec<&mut [F]> = Vec::with_capacity(rows_per_proof.len());
    let mut remaining = data_slice;

    for &num_rows in &rows_per_proof {
        let chunk_size = num_rows * width;
        let (chunk, rest) = remaining.split_at_mut(chunk_size);
        trace_slices.push(chunk);
        remaining = rest;
    }

    // Process each proof in parallel
    trace_slices
        .par_iter_mut()
        .zip(
            gkr_layer_records
                .par_iter()
                .zip(mus.par_iter())
                .zip(q0_claims.par_iter()),
        )
        .enumerate()
        .for_each(
            |(proof_idx, (proof_trace, ((record, mus_for_proof), q0_claim)))| {
                let mus_for_proof = mus_for_proof.as_slice();
                let q0_claim = *q0_claim;

                if record.layer_claims.is_empty() {
                    debug_assert_eq!(proof_trace.len(), width);
                    let row_data = &mut proof_trace[..width];
                    let cols: &mut GkrLayerCols<F> = row_data.borrow_mut();
                    cols.is_enabled = F::ONE;
                    cols.proof_idx = F::from_canonical_usize(proof_idx);
                    cols.is_first = F::ONE;
                    cols.is_dummy = F::ONE;
                    cols.sumcheck_claim_in = [F::ONE, F::ZERO, F::ZERO, F::ZERO];
                    cols.q_xi_0 = [F::ONE, F::ZERO, F::ZERO, F::ZERO];
                    cols.q_xi_1 = [F::ONE, F::ZERO, F::ZERO, F::ZERO];
                    cols.denom_claim = [F::ONE, F::ZERO, F::ZERO, F::ZERO];
                    return;
                }

                let layer_count = record.layer_count();
                let mut prev_layer_eval: Option<(EF, EF)> = None;

                proof_trace
                    .chunks_mut(width)
                    .take(layer_count)
                    .enumerate()
                    .for_each(|(layer_idx, row_data)| {
                        let cols: &mut GkrLayerCols<F> = row_data.borrow_mut();
                        cols.proof_idx = F::from_canonical_usize(proof_idx);
                        cols.is_enabled = F::ONE;
                        cols.is_first = F::from_bool(layer_idx == 0);
                        cols.layer_idx = F::from_canonical_usize(layer_idx);
                        cols.tidx = F::from_canonical_usize(record.layer_tidx(layer_idx));

                        let lambda = record.lambda_at(layer_idx);
                        let eq_at_r_prime = record.eq_at(layer_idx);

                        cols.lambda = lambda.as_base_slice().try_into().unwrap();
                        cols.eq_at_r_prime = eq_at_r_prime.as_base_slice().try_into().unwrap();

                        let claims = &record.layer_claims[layer_idx];
                        let mu = mus_for_proof[layer_idx];

                        cols.p_xi_0 = claims[0].as_base_slice().try_into().unwrap();
                        cols.q_xi_0 = claims[1].as_base_slice().try_into().unwrap();
                        cols.p_xi_1 = claims[2].as_base_slice().try_into().unwrap();
                        cols.q_xi_1 = claims[3].as_base_slice().try_into().unwrap();

                        cols.mu = mu.as_base_slice().try_into().unwrap();

                        let sumcheck_claim_in = prev_layer_eval
                            .map(|(numer_prev, denom_prev)| numer_prev + lambda * denom_prev)
                            .unwrap_or(q0_claim);
                        cols.sumcheck_claim_in =
                            sumcheck_claim_in.as_base_slice().try_into().unwrap();

                        let (numer_base, denom_base): ([F; D_EF], [F; D_EF]) =
                            reduce_to_single_evaluation::<F, F>(
                                claims[0].as_base_slice().try_into().unwrap(),
                                claims[2].as_base_slice().try_into().unwrap(),
                                claims[1].as_base_slice().try_into().unwrap(),
                                claims[3].as_base_slice().try_into().unwrap(),
                                mu.as_base_slice().try_into().unwrap(),
                            );
                        cols.numer_claim = numer_base;
                        cols.denom_claim = denom_base;

                        let numer = claims[0] * (EF::ONE - mu) + claims[2] * mu;
                        let denom = claims[1] * (EF::ONE - mu) + claims[3] * mu;
                        prev_layer_eval = Some((numer, denom));
                    });
            },
        );

    RowMajorMatrix::new(trace, width)
}
