use std::{borrow::BorrowMut, iter::zip};

use itertools::{izip, Itertools};
use openvm_stark_sdk::config::baby_bear_poseidon2::{D_EF, EF, F};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;

use crate::{
    stacking::{
        opening::air::OpeningClaimsCols,
        utils::{
            compute_coefficients, get_stacked_slice_data, sorted_column_claims, ColumnOpeningPair,
        },
    },
    tracegen::{RowMajorChip, StandardTracegenCtx},
};

pub struct OpeningClaimsTraceGenerator;

impl RowMajorChip<F> for OpeningClaimsTraceGenerator {
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

        let width = OpeningClaimsCols::<usize>::width();
        let column_claims = zip(proofs.iter(), preflights.iter())
            .map(|(proof, preflight)| {
                sorted_column_claims(vk, proof, &preflight.proof_shape.sorted_trace_vdata)
            })
            .collect_vec();
        let minimum_height: usize = column_claims.iter().map(|c| c.len()).sum();
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

        for (proof_idx, (proof, preflight, claims)) in
            izip!(proofs, preflights, column_claims).enumerate()
        {
            let stacked_slices =
                get_stacked_slice_data(vk, &preflight.proof_shape.sorted_trace_vdata);

            let (_, per_slice) = compute_coefficients(
                proof,
                &stacked_slices,
                &preflight.stacking.sumcheck_rnd,
                &preflight.batch_constraint.sumcheck_rnd,
                &preflight.stacking.lambda,
                vk.inner.params.l_skip,
                vk.inner.params.n_stack,
            );

            let num_rows = claims.len();
            let proof_idx_value = F::from_usize(proof_idx);

            let mut lambda_pows = preflight.stacking.lambda.square().powers().take(num_rows);
            let mut stacking_claim_coefficient = EF::ZERO;
            let mut s_0 = EF::ZERO;

            let last_main_idx = claims
                .iter()
                .enumerate()
                .skip(1)
                .find_map(|(i, claim)| {
                    if claim.part_idx != 0 {
                        Some(i - 1)
                    } else {
                        None
                    }
                })
                .unwrap_or(num_rows - 1);

            for (row_idx, (claim, slice, (eq_in, k_rot_in, eq_bits))) in
                izip!(claims, stacked_slices, per_slice,).enumerate()
            {
                let chunk = chunks.next().unwrap();
                let ColumnOpeningPair {
                    sort_idx,
                    part_idx,
                    col_idx,
                    col_claim,
                    rot_claim,
                } = claim;
                let cols: &mut OpeningClaimsCols<F> = chunk.borrow_mut();
                cols.proof_idx = proof_idx_value;
                cols.is_valid = F::ONE;
                cols.is_first = F::from_bool(row_idx == 0);
                cols.is_last = F::from_bool(row_idx + 1 == num_rows);

                cols.sort_idx = F::from_usize(sort_idx);
                cols.part_idx = F::from_usize(part_idx);
                cols.col_idx = F::from_usize(col_idx);
                cols.col_claim
                    .copy_from_slice(col_claim.as_basis_coefficients_slice());
                cols.rot_claim
                    .copy_from_slice(rot_claim.as_basis_coefficients_slice());
                cols.need_rot = F::from_bool(slice.need_rot);

                cols.is_main = F::from_bool(part_idx == 0);
                cols.is_transition_main =
                    F::from_bool(row_idx + 1 != num_rows && row_idx != last_main_idx);

                let n_lift = slice.n.max(0) as usize;
                cols.hypercube_dim = if slice.n.is_positive() {
                    F::from_usize(n_lift)
                } else {
                    -F::from_usize(slice.n.unsigned_abs())
                };
                cols.log_lifted_height = F::from_usize(n_lift + vk.inner.params.l_skip);
                cols.lifted_height = F::from_usize(1 << (n_lift + vk.inner.params.l_skip));
                cols.lifted_height_inv = cols.lifted_height.inverse();

                cols.tidx = F::from_usize(
                    preflight.batch_constraint.tidx_before_column_openings + 2 * row_idx * D_EF,
                );
                cols.lambda
                    .copy_from_slice(preflight.stacking.lambda.as_basis_coefficients_slice());

                let lambda_pow = lambda_pows.next().unwrap();
                cols.lambda_pow
                    .copy_from_slice(lambda_pow.as_basis_coefficients_slice());

                cols.commit_idx = F::from_usize(slice.commit_idx);
                cols.stacked_col_idx = F::from_usize(slice.col_idx);
                cols.row_idx = F::from_usize(slice.row_idx);
                cols.is_last_for_claim = F::from_bool(slice.is_last_for_claim);

                cols.eq_in
                    .copy_from_slice(eq_in.as_basis_coefficients_slice());
                cols.k_rot_in
                    .copy_from_slice(k_rot_in.as_basis_coefficients_slice());
                if slice.need_rot {
                    cols.k_rot_in_when_needed
                        .copy_from_slice(k_rot_in.as_basis_coefficients_slice());
                }
                cols.eq_bits
                    .copy_from_slice(eq_bits.as_basis_coefficients_slice());

                let lambda_pow_eq_bits = lambda_pow * eq_bits;
                cols.lambda_pow_eq_bits
                    .copy_from_slice(lambda_pow_eq_bits.as_basis_coefficients_slice());

                let k_rot_term = if slice.need_rot { k_rot_in } else { EF::ZERO };
                stacking_claim_coefficient +=
                    lambda_pow_eq_bits * (eq_in + preflight.stacking.lambda * k_rot_term);
                cols.stacking_claim_coefficient
                    .copy_from_slice(stacking_claim_coefficient.as_basis_coefficients_slice());
                if slice.is_last_for_claim {
                    stacking_claim_coefficient = EF::ZERO;
                }

                s_0 += lambda_pow * (col_claim + preflight.stacking.lambda * rot_claim);
                cols.s_0.copy_from_slice(s_0.as_basis_coefficients_slice());
            }
        }

        let padding_proof_idx = F::from_usize(proofs.len());
        let mut chunks = chunks.peekable();

        while let Some(chunk) = chunks.next() {
            let cols: &mut OpeningClaimsCols<F> = chunk.borrow_mut();
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
    use itertools::Itertools;
    use openvm_cuda_backend::{base::DeviceMatrix, GpuBackend};
    use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
    use openvm_stark_backend::prover::AirProvingContext;

    use super::*;
    use crate::{
        stacking::{
            cuda_abi::{
                opening_claims_tracegen, opening_claims_tracegen_temp_bytes, ColumnOpeningClaims,
                OpeningRecordsPerProof,
            },
            cuda_tracegen::StackingBlob,
        },
        tracegen::{cuda::StandardTracegenGpuCtx, ModuleChip},
    };

    pub struct OpeningClaimsTraceGeneratorGpu;

    impl ModuleChip<GpuBackend> for OpeningClaimsTraceGeneratorGpu {
        type Ctx<'a> = (StandardTracegenGpuCtx<'a>, &'a StackingBlob);

        fn generate_proving_ctx(
            &self,
            ctx: &Self::Ctx<'_>,
            required_height: Option<usize>,
        ) -> Option<AirProvingContext<GpuBackend>> {
            let child_vk = ctx.0.vk;
            let proofs_gpu = ctx.0.proofs;
            let preflights_gpu = ctx.0.preflights;
            let blob = ctx.1;

            let mut num_valid_rows = 0;
            let row_bounds = blob
                .slice_data
                .iter()
                .map(|buf| {
                    num_valid_rows += buf.len();
                    num_valid_rows as u32
                })
                .collect_vec();
            let mut last_main_idx_per_proof = Vec::with_capacity(proofs_gpu.len());
            let claims = proofs_gpu
                .iter()
                .zip_eq(preflights_gpu.iter())
                .map(|(proof, preflight)| {
                    let claims = sorted_column_claims(
                        &child_vk.cpu,
                        &proof.cpu,
                        &preflight.cpu.proof_shape.sorted_trace_vdata,
                    )
                    .into_iter()
                    .map(|claim| ColumnOpeningClaims {
                        sort_idx: claim.sort_idx as u32,
                        part_idx: claim.part_idx as u32,
                        col_idx: claim.col_idx as u32,
                        col_claim: claim.col_claim,
                        rot_claim: claim.rot_claim,
                    })
                    .collect_vec();
                    let last_main_idx = claims
                        .iter()
                        .enumerate()
                        .skip(1)
                        .find_map(|(i, claim)| {
                            if claim.part_idx != 0 {
                                Some(i - 1)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(claims.len() - 1);
                    last_main_idx_per_proof.push(last_main_idx);
                    claims.to_device().unwrap()
                })
                .collect_vec();
            let lambda_pows = preflights_gpu
                .iter()
                .enumerate()
                .map(|(proof_idx, preflight)| {
                    preflight
                        .cpu
                        .stacking
                        .lambda
                        .square()
                        .powers()
                        .take(claims[proof_idx].len())
                        .collect_vec()
                        .to_device()
                        .unwrap()
                })
                .collect_vec();

            let height = if let Some(height) = required_height {
                if height < num_valid_rows {
                    return None;
                }
                height
            } else {
                num_valid_rows.next_power_of_two()
            };
            let width = OpeningClaimsCols::<usize>::width();
            let d_trace = DeviceMatrix::with_capacity(height, width);
            let d_keys_buffer = DeviceBuffer::<F>::with_capacity(height);

            let d_claims = claims.iter().map(|buf| buf.as_ptr()).collect_vec();
            let d_slice_data = blob.slice_data.iter().map(|buf| buf.as_ptr()).collect_vec();
            let d_precomps = blob.precomps.iter().map(|buf| buf.as_ptr()).collect_vec();
            let d_lambda_pows = lambda_pows.iter().map(|buf| buf.as_ptr()).collect_vec();
            let d_records = preflights_gpu
                .iter()
                .zip(last_main_idx_per_proof)
                .map(|(preflight, last_main_idx)| OpeningRecordsPerProof {
                    tidx_before_column_openings: preflight
                        .cpu
                        .batch_constraint
                        .tidx_before_column_openings
                        as u32,
                    last_main_idx: last_main_idx as u32,
                    lambda: preflight.cpu.stacking.lambda,
                })
                .collect_vec()
                .to_device()
                .unwrap();

            unsafe {
                let temp_bytes =
                    opening_claims_tracegen_temp_bytes(d_trace.buffer(), height, &d_keys_buffer)
                        .unwrap();
                let d_temp_buffer = DeviceBuffer::<u8>::with_capacity(temp_bytes);
                opening_claims_tracegen(
                    d_trace.buffer(),
                    height,
                    width,
                    &row_bounds,
                    d_claims,
                    d_slice_data,
                    d_precomps,
                    d_lambda_pows,
                    &d_records,
                    proofs_gpu.len() as u32,
                    child_vk.system_params.l_skip as u32,
                    &d_keys_buffer,
                    &d_temp_buffer,
                    temp_bytes,
                )
                .unwrap();
            }

            Some(AirProvingContext::simple_no_pis(d_trace))
        }
    }
}
