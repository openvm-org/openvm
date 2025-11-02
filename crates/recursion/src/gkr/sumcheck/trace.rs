use core::borrow::BorrowMut;

use openvm_stark_backend::p3_maybe_rayon::prelude::*;
use p3_field::{FieldAlgebra, FieldExtensionAlgebra};
use p3_matrix::dense::RowMajorMatrix;
use stark_backend_v2::{D_EF, EF, F, poly_common::interpolate_cubic_at_0123};

use super::GkrLayerSumcheckCols;

#[derive(Default, Debug, Clone)]
pub struct GkrSumcheckRecord {
    pub tidx: usize,
    pub evals: Vec<[EF; 3]>,
    pub ris: Vec<EF>,
    pub claims: Vec<EF>,
}

impl GkrSumcheckRecord {
    #[inline]
    pub fn num_layers(&self) -> usize {
        self.claims.len()
    }

    #[inline]
    pub fn total_rounds(&self) -> usize {
        let layers = self.num_layers();
        layers * (layers + 1) / 2
    }

    #[inline]
    fn layer_start_index(layer_idx: usize) -> usize {
        layer_idx * (layer_idx + 1) / 2
    }

    #[inline]
    fn layer_rounds(layer_idx: usize) -> usize {
        layer_idx + 1
    }

    #[inline]
    fn derive_tidx(&self, layer_idx: usize, round_in_layer: usize) -> usize {
        let rounds_before_layer = Self::layer_start_index(layer_idx);
        self.tidx + 4 * D_EF * (rounds_before_layer + round_in_layer) + 6 * D_EF * layer_idx
    }

    #[inline]
    fn prev_challenge(layer_idx: usize, round_in_layer: usize, mus: &[EF], ris: &[EF]) -> EF {
        if round_in_layer == 0 {
            mus[layer_idx]
        } else {
            let prev_layer = layer_idx
                .checked_sub(1)
                .expect("round_in_layer > 0 only occurs for non-root layers");
            let offset = Self::layer_start_index(prev_layer) + (round_in_layer - 1);
            ris[offset]
        }
    }
}

pub fn generate_trace(
    gkr_sumcheck_records: &[GkrSumcheckRecord],
    mus: &[Vec<EF>],
) -> RowMajorMatrix<F> {
    debug_assert_eq!(gkr_sumcheck_records.len(), mus.len());

    let width = GkrLayerSumcheckCols::<F>::width();

    if gkr_sumcheck_records.is_empty() {
        let trace = vec![F::ZERO; width];
        return RowMajorMatrix::new(trace, width);
    }

    // Calculate rows per proof
    let rows_per_proof: Vec<usize> = gkr_sumcheck_records
        .iter()
        .map(|record| record.total_rounds().max(1))
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

    // Process each proof in parallel
    trace_slices
        .par_iter_mut()
        .zip(gkr_sumcheck_records.par_iter().zip(mus.par_iter()))
        .enumerate()
        .for_each(|(proof_idx, (proof_trace, (record, mus_for_proof)))| {
            let mus_for_proof = mus_for_proof.as_slice();
            let total_rounds = record.total_rounds();
            let num_layers = record.num_layers();

            debug_assert_eq!(record.ris.len(), total_rounds);
            debug_assert_eq!(record.evals.len(), total_rounds);
            debug_assert!(mus_for_proof.len() >= num_layers);

            if total_rounds == 0 {
                debug_assert_eq!(proof_trace.len(), width);
                let row_data = &mut proof_trace[..width];
                let cols: &mut GkrLayerSumcheckCols<F> = row_data.borrow_mut();
                cols.is_enabled = F::ONE;
                cols.tidx = F::from_canonical_usize(D_EF);
                cols.proof_idx = F::from_canonical_usize(proof_idx);
                cols.layer_idx = F::ONE;
                cols.is_first_round = F::ONE;
                cols.is_layer_start = F::ONE;
                cols.is_last_layer = F::ONE;
                cols.is_dummy = F::ONE;
                cols.eq_in = [F::ONE, F::ZERO, F::ZERO, F::ZERO];
                cols.eq_out = [F::ONE, F::ZERO, F::ZERO, F::ZERO];
                cols.claim_in = [F::ONE, F::ZERO, F::ZERO, F::ZERO];
                cols.claim_out = [F::ONE, F::ZERO, F::ZERO, F::ZERO];
                return;
            }

            let mut global_round_idx = 0usize;
            let mut row_iter = proof_trace.chunks_mut(width);

            for layer_idx in 0..num_layers {
                let layer_rounds = GkrSumcheckRecord::layer_rounds(layer_idx);
                let layer_idx_value = layer_idx + 1;
                let is_last_layer = layer_idx == num_layers.saturating_sub(1);

                let mut claim = record.claims[layer_idx];
                let mut eq = EF::ONE;

                for round_in_layer in 0..layer_rounds {
                    let challenge = record.ris[global_round_idx];
                    let evals = record.evals[global_round_idx];
                    let prev_challenge = GkrSumcheckRecord::prev_challenge(
                        layer_idx,
                        round_in_layer,
                        mus_for_proof,
                        &record.ris,
                    );

                    let prev_challenge_base: [F; D_EF] =
                        prev_challenge.as_base_slice().try_into().unwrap();
                    let challenge_base: [F; D_EF] = challenge.as_base_slice().try_into().unwrap();

                    let eval1_base: [F; D_EF] = evals[0].as_base_slice().try_into().unwrap();
                    let eval2_base: [F; D_EF] = evals[1].as_base_slice().try_into().unwrap();
                    let eval3_base: [F; D_EF] = evals[2].as_base_slice().try_into().unwrap();

                    let claim_in_base: [F; D_EF] = claim.as_base_slice().try_into().unwrap();
                    let eq_in_base: [F; D_EF] = eq.as_base_slice().try_into().unwrap();

                    let ev0 = claim - evals[0];
                    let evals_full = [ev0, evals[0], evals[1], evals[2]];
                    let claim_out = interpolate_cubic_at_0123(&evals_full, challenge);
                    let eq_factor = prev_challenge * challenge
                        + (EF::ONE - prev_challenge) * (EF::ONE - challenge);
                    let eq_out = eq * eq_factor;

                    let claim_out_base: [F; D_EF] = claim_out.as_base_slice().try_into().unwrap();
                    let eq_out_base: [F; D_EF] = eq_out.as_base_slice().try_into().unwrap();

                    let cols: &mut GkrLayerSumcheckCols<F> = row_iter.next().unwrap().borrow_mut();
                    cols.is_enabled = F::ONE;
                    cols.proof_idx = F::from_canonical_usize(proof_idx);

                    cols.layer_idx = F::from_canonical_usize(layer_idx_value);
                    cols.is_last_layer = F::from_bool(is_last_layer);

                    cols.round = F::from_canonical_usize(round_in_layer);
                    cols.is_first_round = F::from_bool(round_in_layer == 0);
                    cols.is_layer_start = F::from_bool(layer_idx_value == 1 && round_in_layer == 0);

                    cols.nested_for_loop_aux_cols.is_transition[0] =
                        F::from_bool(global_round_idx + 1 != total_rounds);

                    let tidx = record.derive_tidx(layer_idx, round_in_layer);
                    cols.tidx = F::from_canonical_usize(tidx);

                    cols.ev1 = eval1_base;
                    cols.ev2 = eval2_base;
                    cols.ev3 = eval3_base;

                    cols.prev_challenge = prev_challenge_base;
                    cols.challenge = challenge_base;

                    cols.claim_in = claim_in_base;
                    cols.claim_out = claim_out_base;

                    cols.eq_in = eq_in_base;
                    cols.eq_out = eq_out_base;

                    claim = claim_out;
                    eq = eq_out;
                    global_round_idx += 1;
                }
            }

            debug_assert_eq!(global_round_idx, total_rounds);
        });

    RowMajorMatrix::new(trace, width)
}
