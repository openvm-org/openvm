use core::{borrow::BorrowMut, cmp::min};

use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_stark_backend::{keygen::types::MultiStarkVerifyingKey, proof::Proof};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, CHUNK, D_EF, F};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;

use crate::{
    system::Preflight,
    tracegen::RowMajorChip,
    whir::{initial_opened_values::air::InitialOpenedValuesCols, WhirBlobCpu},
};

pub(crate) struct InitialOpenedValuesTraceGenerator;

pub(crate) struct InitialOpenedValuesCtx<'a> {
    pub vk: &'a MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
    pub proofs: &'a [&'a Proof<BabyBearPoseidon2Config>],
    pub preflights: &'a [&'a Preflight],
    pub blob: &'a WhirBlobCpu,
}

impl RowMajorChip<F> for InitialOpenedValuesTraceGenerator {
    type Ctx<'a> = InitialOpenedValuesCtx<'a>;

    #[tracing::instrument(level = "trace", skip_all)]
    fn generate_trace(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>> {
        let params = &ctx.vk.inner.params;
        let proofs = ctx.proofs;
        let preflights = ctx.preflights;
        let blob = ctx.blob;
        let zi_roots = &blob.zi_roots;
        let zis = &blob.zis;
        let yis = &blob.yis;
        let accs_layout = blob.codeword_value_accs.layout();
        let codeword_value_accs = blob.codeword_value_accs.as_slice();
        debug_assert_eq!(proofs.len(), preflights.len());

        let k_whir = params.k_whir();

        let omega_k = F::two_adic_generator(k_whir);
        let num_valid_rows = codeword_value_accs.len();
        let height = if let Some(h) = required_height {
            if h < num_valid_rows {
                return None;
            }
            h
        } else {
            num_valid_rows.next_power_of_two()
        };
        let width = InitialOpenedValuesCols::<F>::width();
        let mut trace = vec![F::ZERO; height * width];
        let mu_pows = &blob.mu_pows;
        trace
            .par_chunks_exact_mut(width)
            .take(num_valid_rows)
            .zip(codeword_value_accs)
            .enumerate()
            .for_each(|(row_idx, (row, &codeword_value_acc))| {
                let (proof_idx, query_idx, coset_idx, commit_idx, chunk_idx) =
                    accs_layout.decompose(row_idx);
                let preflight = &preflights[proof_idx];
                let chunk_len = accs_layout.chunk_len(commit_idx, chunk_idx);

                let mu = preflight.stacking.stacking_batching_challenge;

                let cols: &mut InitialOpenedValuesCols<F> = row.borrow_mut();

                let is_first_in_commit = chunk_idx == 0;
                let is_first_in_coset = is_first_in_commit && commit_idx == 0;
                let is_first_in_query = is_first_in_coset && coset_idx == 0;
                let is_first_in_proof = is_first_in_query && query_idx == 0;

                cols.proof_idx = F::from_usize(proof_idx);
                cols.is_first_in_proof = F::from_bool(is_first_in_proof);
                cols.is_first_in_query = F::from_bool(is_first_in_query);
                cols.is_first_in_coset = F::from_bool(is_first_in_coset);
                cols.is_first_in_commit = F::from_bool(is_first_in_commit);
                cols.query_idx = F::from_usize(query_idx);
                cols.commit_idx = F::from_usize(commit_idx);
                cols.col_chunk_idx = F::from_usize(chunk_idx);
                cols.coset_idx = F::from_usize(coset_idx);
                for flag in cols.flags.iter_mut().take(chunk_len) {
                    *flag = F::ONE;
                }
                cols.twiddle = omega_k.exp_u64(coset_idx as u64);
                cols.codeword_value_acc
                    .copy_from_slice(codeword_value_acc.as_basis_coefficients_slice());
                let query = (proof_idx, 0, query_idx);
                cols.zi_root = zi_roots[query];
                cols.zi = zis[query];
                cols.yi
                    .copy_from_slice(yis[query].as_basis_coefficients_slice());
                cols.mu.copy_from_slice(mu.as_basis_coefficients_slice());
                cols.merkle_idx_bit_src = preflight.whir.queries[query_idx];

                let exponent_base = accs_layout.commit_width_offset(commit_idx);
                let chunk_base = exponent_base + chunk_idx * CHUNK;

                // Fill mu_pows_even_clamped[k] = mu^(min(b + 2k, opened_row_len - 1)),
                // where b = exponent_base + chunk_idx*CHUNK.
                for k in 0..CHUNK / 2 {
                    let offset = 2 * k;
                    let exponent = chunk_base + min(offset, chunk_len - 1);
                    let mu_pow = mu_pows[(proof_idx, exponent)];
                    cols.mu_pows_even_clamped[k]
                        .copy_from_slice(mu_pow.as_basis_coefficients_slice());
                }
                let exponent = chunk_base + chunk_len - 1;
                let mu_pow = mu_pows[(proof_idx, exponent)];
                cols.mu_pow_last_clamped
                    .copy_from_slice(mu_pow.as_basis_coefficients_slice());

                let states = &preflight.initial_row_states[commit_idx][query_idx][coset_idx];
                let opened_row = &proofs[proof_idx].whir_proof.initial_round_opened_rows
                    [commit_idx][query_idx][coset_idx];
                let chunk_start = chunk_idx * CHUNK;

                cols.pre_state = if chunk_idx > 0 {
                    states[chunk_idx - 1]
                } else {
                    [F::ZERO; POSEIDON2_WIDTH]
                };
                cols.pre_state[..chunk_len]
                    .copy_from_slice(&opened_row[chunk_start..chunk_start + chunk_len]);
                cols.post_state = states[chunk_idx];

                let mut next_acc = cols.codeword_value_acc;
                for i in 0..chunk_len {
                    let exponent = chunk_base + i;
                    let mu_pow_coeffs: &[F] =
                        mu_pows[(proof_idx, exponent)].as_basis_coefficients_slice();
                    let scalar = cols.pre_state[i];
                    for j in 0..D_EF {
                        next_acc[j] += mu_pow_coeffs[j] * scalar;
                    }
                }
                cols.codeword_value_next_acc = next_acc;
            });

        Some(RowMajorMatrix::new(trace, width))
    }
}
