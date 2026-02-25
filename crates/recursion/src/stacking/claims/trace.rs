use std::borrow::BorrowMut;

use itertools::{izip, Itertools};
use openvm_stark_sdk::config::baby_bear_poseidon2::{D_EF, EF, F};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;

use crate::{
    stacking::{
        claims::air::StackingClaimsCols,
        utils::{compute_coefficients, get_stacked_slice_data},
    },
    tracegen::{RowMajorChip, StandardTracegenCtx},
};

pub struct StackingClaimsTraceGenerator;

impl RowMajorChip<F> for StackingClaimsTraceGenerator {
    type Ctx<'a> = StandardTracegenCtx<'a>;

    #[tracing::instrument(level = "trace", skip_all)]
    fn generate_trace(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>> {
        let vk = ctx.vk;
        let proofs = ctx.proofs;
        let preflights = ctx.preflights;
        debug_assert_eq!(proofs.len(), preflights.len());

        let w_stack = vk.inner.params.w_stack;
        let width = StackingClaimsCols::<usize>::width();
        // Each proof gets exactly w_stack rows (valid + padding).
        let minimum_height = proofs.len() * w_stack;
        let height = if let Some(height) = required_height {
            if height < minimum_height {
                return None;
            }
            height
        } else {
            minimum_height.next_power_of_two()
        };

        let mut trace = vec![F::ZERO; height * width];
        let mut chunks = trace.chunks_mut(width);

        for (proof_idx, (proof, preflight)) in proofs.iter().zip(preflights).enumerate() {
            let claims = proof
                .stacking_proof
                .stacking_openings
                .iter()
                .enumerate()
                .flat_map(|(commit_idx, openings)| {
                    openings
                        .iter()
                        .enumerate()
                        .map(move |(stacked_col_idx, opening)| {
                            (commit_idx, stacked_col_idx, opening)
                        })
                })
                .collect_vec();
            let stacked_slices =
                get_stacked_slice_data(vk, &preflight.proof_shape.sorted_trace_vdata);

            let coeffs = compute_coefficients(
                proof,
                &stacked_slices,
                &preflight.stacking.sumcheck_rnd,
                &preflight.batch_constraint.sumcheck_rnd,
                &preflight.stacking.lambda,
                vk.inner.params.l_skip,
                vk.inner.params.n_stack,
            )
            .0
            .into_iter()
            .flatten()
            .collect_vec();

            let num_valid = claims.len();
            debug_assert!(
                num_valid <= w_stack,
                "proof {proof_idx} has {num_valid} stacking claims but w_stack = {w_stack}"
            );
            let proof_idx_value = F::from_usize(proof_idx);

            let initial_tidx = preflight.stacking.intermediate_tidx[2];

            let mu = preflight.stacking.stacking_batching_challenge;
            let mu_pows = mu.powers().take(num_valid).collect_vec();

            // μ PoW witness and sample from preflight
            let mu_pow_witness = preflight.stacking.mu_pow_witness;
            let mu_pow_sample = preflight.stacking.mu_pow_sample;

            let mut final_s_eval = EF::ZERO;
            let mut whir_claim = EF::ZERO;

            for (idx, (&(commit_idx, stacked_col_idx, &claim), coeff)) in
                izip!(&claims, coeffs).enumerate()
            {
                let chunk = chunks.next().unwrap();
                let cols: &mut StackingClaimsCols<F> = chunk.borrow_mut();

                cols.proof_idx = proof_idx_value;
                cols.is_valid = F::ONE;
                cols.is_first = F::from_bool(idx == 0);
                cols.is_last = F::from_bool(num_valid == w_stack && idx + 1 == num_valid);

                cols.commit_idx = F::from_usize(commit_idx);
                cols.stacked_col_idx = F::from_usize(stacked_col_idx);
                cols.global_col_idx = F::from_usize(idx);

                cols.tidx = F::from_usize(initial_tidx + (D_EF * idx));
                cols.mu.copy_from_slice(mu.as_basis_coefficients_slice());
                cols.mu_pow
                    .copy_from_slice(mu_pows[idx].as_basis_coefficients_slice());

                cols.stacking_claim
                    .copy_from_slice(claim.as_basis_coefficients_slice());

                // μ PoW columns (only used on last row, but set for all rows for simplicity)
                cols.mu_pow_witness = mu_pow_witness;
                cols.mu_pow_sample = mu_pow_sample;

                cols.claim_coefficient
                    .copy_from_slice(coeff.as_basis_coefficients_slice());
                final_s_eval += claim * coeff;
                cols.final_s_eval
                    .copy_from_slice(final_s_eval.as_basis_coefficients_slice());

                whir_claim += mu_pows[idx] * claim;
                cols.whir_claim
                    .copy_from_slice(whir_claim.as_basis_coefficients_slice());
            }

            // Padding rows (fill up to w_stack)
            for idx in num_valid..w_stack {
                let chunk = chunks.next().unwrap();
                let cols: &mut StackingClaimsCols<F> = chunk.borrow_mut();

                cols.proof_idx = proof_idx_value;
                cols.is_padding = F::ONE;
                cols.is_last = F::from_bool(idx + 1 == w_stack);
                cols.global_col_idx = F::from_usize(idx);
            }
        }

        let padding_proof_idx = F::from_usize(proofs.len());
        let mut chunks = chunks.peekable();

        while let Some(chunk) = chunks.next() {
            let cols: &mut StackingClaimsCols<F> = chunk.borrow_mut();
            cols.proof_idx = padding_proof_idx;
            if chunks.peek().is_none() {
                cols.is_last = F::ONE;
            }
        }

        Some(RowMajorMatrix::new(trace, width))
    }
}

#[cfg(feature = "cuda")]
pub(crate) mod cuda {
    use openvm_cuda_backend::{base::DeviceMatrix, GpuBackend};
    use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
    use openvm_stark_backend::prover::AirProvingContext;

    use super::*;
    use crate::{
        stacking::{
            cuda_abi::{
                stacking_claims_tracegen, stacking_claims_tracegen_temp_bytes,
                ClaimsRecordsPerProof, StackingClaim,
            },
            cuda_tracegen::StackingBlob,
        },
        tracegen::{cuda::StandardTracegenGpuCtx, ModuleChip},
    };

    pub struct StackingClaimsTraceGeneratorGpu;

    impl ModuleChip<GpuBackend> for StackingClaimsTraceGeneratorGpu {
        type Ctx<'a> = (StandardTracegenGpuCtx<'a>, &'a StackingBlob);

        fn generate_proving_ctx(
            &self,
            ctx: &Self::Ctx<'_>,
            required_height: Option<usize>,
        ) -> Option<openvm_stark_backend::prover::AirProvingContext<GpuBackend>> {
            let proofs_gpu = ctx.0.proofs;
            let preflights_gpu = ctx.0.preflights;
            let blob = ctx.1;
            let w_stack = ctx.0.vk.system_params.w_stack;

            let mut row_bounds = Vec::with_capacity(proofs_gpu.len());
            let claims = proofs_gpu
                .iter()
                .enumerate()
                .map(|(proof_idx, proof)| {
                    let claims = proof
                        .cpu
                        .stacking_proof
                        .stacking_openings
                        .iter()
                        .enumerate()
                        .flat_map(|(commit_idx, openings)| {
                            openings
                                .iter()
                                .enumerate()
                                .map(move |(stacked_col_idx, opening)| StackingClaim {
                                    commit_idx: commit_idx as u32,
                                    stacked_col_idx: stacked_col_idx as u32,
                                    claim: *opening,
                                })
                        })
                        .collect_vec();

                    let num_valid = claims.len();
                    assert!(
                        num_valid <= w_stack,
                        "proof {proof_idx} has {num_valid} stacking claims but w_stack = {w_stack}"
                    );
                    row_bounds.push(((proof_idx + 1) * w_stack) as u32);

                    claims.to_device().unwrap()
                })
                .collect_vec();

            let mu_pows = preflights_gpu
                .iter()
                .enumerate()
                .map(|(proof_idx, preflight)| {
                    let mu = preflight.cpu.stacking.stacking_batching_challenge;
                    mu.powers()
                        .take(claims[proof_idx].len())
                        .collect_vec()
                        .to_device()
                        .unwrap()
                })
                .collect_vec();

            let minimum_height = proofs_gpu.len() * w_stack;
            let height = if let Some(height) = required_height {
                if height < minimum_height {
                    return None;
                }
                height
            } else {
                minimum_height.next_power_of_two()
            };
            let width = StackingClaimsCols::<usize>::width();
            let d_trace = DeviceMatrix::with_capacity(height, width);

            let d_claims = claims.iter().map(|buf| buf.as_ptr()).collect_vec();
            let d_coeffs = blob.coeffs.iter().map(|buf| buf.as_ptr()).collect_vec();
            let d_mu_pows = mu_pows.iter().map(|buf| buf.as_ptr()).collect_vec();
            let d_records = preflights_gpu
                .iter()
                .enumerate()
                .map(|(proof_idx, preflight)| ClaimsRecordsPerProof {
                    initial_tidx: preflight.cpu.stacking.intermediate_tidx[2] as u32,
                    num_valid: claims[proof_idx].len() as u32,
                    mu: preflight.cpu.stacking.stacking_batching_challenge,
                    mu_pow_witness: preflight.cpu.stacking.mu_pow_witness,
                    mu_pow_sample: preflight.cpu.stacking.mu_pow_sample,
                })
                .collect_vec()
                .to_device()
                .unwrap();

            unsafe {
                let temp_bytes =
                    stacking_claims_tracegen_temp_bytes(d_trace.buffer(), height).unwrap();
                let d_temp_buffer = DeviceBuffer::<u8>::with_capacity(temp_bytes);
                stacking_claims_tracegen(
                    d_trace.buffer(),
                    height,
                    width,
                    &row_bounds,
                    d_claims,
                    d_coeffs,
                    d_mu_pows,
                    &d_records,
                    proofs_gpu.len() as u32,
                    &d_temp_buffer,
                    temp_bytes,
                )
                .unwrap();
            }

            Some(AirProvingContext::simple_no_pis(d_trace))
        }
    }
}
