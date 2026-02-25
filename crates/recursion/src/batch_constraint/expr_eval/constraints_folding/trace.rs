use std::borrow::BorrowMut;

use itertools::Itertools;
use openvm_stark_backend::keygen::types::MultiStarkVerifyingKey0;
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, D_EF, EF, F};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;

use crate::{
    batch_constraint::expr_eval::constraints_folding::air::ConstraintsFoldingCols,
    system::Preflight,
    tracegen::RowMajorChip,
    utils::{MultiProofVecVec, MultiVecWithBounds},
};

#[derive(Copy, Clone)]
#[repr(C)]
pub(crate) struct ConstraintsFoldingRecord {
    sort_idx: usize,
    air_idx: usize,
    constraint_idx: usize,
    node_idx: usize,
    is_first_in_air: bool,
    value: EF,
}

pub(crate) struct ConstraintsFoldingBlob {
    pub(crate) records: MultiProofVecVec<ConstraintsFoldingRecord>,
    // (n, value), n is before lift, can be negative
    pub(crate) folded_claims: MultiProofVecVec<(isize, EF)>,
}

impl ConstraintsFoldingBlob {
    pub fn new(
        vk: &MultiStarkVerifyingKey0<BabyBearPoseidon2Config>,
        expr_evals: &MultiVecWithBounds<EF, 2>,
        preflights: &[&Preflight],
    ) -> Self {
        let constraints = vk
            .per_air
            .iter()
            .map(|vk| vk.symbolic_constraints.constraints.constraint_idx.clone())
            .collect_vec();

        let mut records = MultiProofVecVec::new();
        let mut folded = MultiProofVecVec::new();
        for (pidx, preflight) in preflights.iter().enumerate() {
            let lambda_tidx = preflight.batch_constraint.lambda_tidx;
            let lambda = EF::from_basis_coefficients_slice(
                &preflight.transcript.values()[lambda_tidx..lambda_tidx + D_EF],
            )
            .unwrap();

            let vdata = &preflight.proof_shape.sorted_trace_vdata;
            for (sort_idx, (air_idx, v)) in vdata.iter().enumerate() {
                let constrs = &constraints[*air_idx];
                records.push(ConstraintsFoldingRecord {
                    // dummy to avoid handling case with no constraints
                    sort_idx,
                    air_idx: *air_idx,
                    constraint_idx: 0,
                    node_idx: 0,
                    is_first_in_air: true,
                    value: EF::ZERO,
                });
                let mut folded_claim = EF::ZERO;
                let mut lambda_pow = EF::ONE;
                for (constraint_idx, &constr) in constrs.iter().enumerate() {
                    let value = expr_evals[[pidx, *air_idx]][constr];
                    folded_claim += lambda_pow * value;
                    lambda_pow *= lambda;
                    records.push(ConstraintsFoldingRecord {
                        sort_idx,
                        air_idx: *air_idx,
                        constraint_idx: constraint_idx + 1,
                        node_idx: constr,
                        is_first_in_air: false,
                        value,
                    });
                }
                let n_lift = v.log_height.saturating_sub(vk.params.l_skip);
                let n = v.log_height as isize - vk.params.l_skip as isize;
                folded.push((
                    n,
                    folded_claim * preflight.batch_constraint.eq_ns_frontloaded[n_lift],
                ));
            }
            records.end_proof();
            folded.end_proof();
        }
        Self {
            records,
            folded_claims: folded,
        }
    }
}

pub struct ConstraintsFoldingTraceGenerator;

impl RowMajorChip<F> for ConstraintsFoldingTraceGenerator {
    type Ctx<'a> = (&'a ConstraintsFoldingBlob, &'a [&'a Preflight]);

    #[tracing::instrument(level = "trace", skip_all)]
    fn generate_trace(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>> {
        let (blob, preflights) = ctx;
        let width = ConstraintsFoldingCols::<F>::width();

        let total_height = blob.records.len();
        debug_assert!(total_height > 0);
        let padding_height = if let Some(height) = required_height {
            if height < total_height {
                return None;
            }
            height
        } else {
            total_height.next_power_of_two()
        };
        let mut trace = vec![F::ZERO; padding_height * width];

        let mut cur_height = 0;
        for (pidx, preflight) in preflights.iter().enumerate() {
            let lambda_tidx = preflight.batch_constraint.lambda_tidx;
            let lambda_slice = &preflight.transcript.values()[lambda_tidx..lambda_tidx + D_EF];
            let records = &blob.records[pidx];

            trace[cur_height * width..(cur_height + records.len()) * width]
                .par_chunks_exact_mut(width)
                .zip(records.par_iter())
                .for_each(|(chunk, record)| {
                    let cols: &mut ConstraintsFoldingCols<_> = chunk.borrow_mut();
                    let n_lift = preflight.proof_shape.sorted_trace_vdata[record.sort_idx]
                        .1
                        .log_height
                        .saturating_sub(preflight.proof_shape.l_skip);

                    cols.is_valid = F::ONE;
                    cols.proof_idx = F::from_usize(pidx);
                    cols.air_idx = F::from_usize(record.air_idx);
                    cols.sort_idx = F::from_usize(record.sort_idx);
                    cols.constraint_idx = F::from_usize(record.constraint_idx);
                    cols.n_lift = F::from_usize(n_lift);
                    cols.lambda_tidx = F::from_usize(lambda_tidx);
                    cols.lambda.copy_from_slice(lambda_slice);
                    cols.value
                        .copy_from_slice(record.value.as_basis_coefficients_slice());
                    cols.eq_n.copy_from_slice(
                        preflight.batch_constraint.eq_ns_frontloaded[n_lift]
                            .as_basis_coefficients_slice(),
                    );
                    cols.is_first_in_air = F::from_bool(record.is_first_in_air);
                });

            // Setting `cur_sum`
            let mut cur_sum = EF::ZERO;
            let lambda = EF::from_basis_coefficients_slice(lambda_slice).unwrap();
            trace[cur_height * width..(cur_height + records.len()) * width]
                .chunks_exact_mut(width)
                .rev()
                .for_each(|chunk| {
                    let cols: &mut ConstraintsFoldingCols<_> = chunk.borrow_mut();
                    cur_sum =
                        cur_sum * lambda + EF::from_basis_coefficients_slice(&cols.value).unwrap();
                    cols.cur_sum
                        .copy_from_slice(cur_sum.as_basis_coefficients_slice());
                    if cols.is_first_in_air == F::ONE {
                        cur_sum = EF::ZERO;
                    }
                });

            {
                let cols: &mut ConstraintsFoldingCols<_> =
                    trace[cur_height * width..(cur_height + 1) * width].borrow_mut();
                cols.is_first = F::ONE;
            }
            cur_height += records.len();
        }
        Some(RowMajorMatrix::new(trace, width))
    }
}

#[cfg(feature = "cuda")]
pub(in crate::batch_constraint) mod cuda {
    use openvm_circuit_primitives::cuda_abi::UInt2;
    use openvm_cuda_backend::{base::DeviceMatrix, GpuBackend};
    use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
    use openvm_stark_backend::prover::AirProvingContext;

    use super::*;
    use crate::{
        batch_constraint::cuda_abi::{
            constraints_folding_tracegen, constraints_folding_tracegen_temp_bytes, AffineFpExt,
            FpExtWithTidx,
        },
        cuda::{preflight::PreflightGpu, vk::VerifyingKeyGpu},
        tracegen::ModuleChip,
    };

    pub struct ConstraintsFoldingBlobGpu {
        // Per proof, per AIR, per constraint
        pub values: Vec<Vec<Vec<EF>>>,
        // Per proof
        pub constraints_folding_per_proof: Vec<FpExtWithTidx>,
        // For compatibility with CPU tracegen
        pub folded_claims: MultiProofVecVec<(isize, EF)>,
    }

    impl ConstraintsFoldingBlobGpu {
        pub fn new(
            vk: &VerifyingKeyGpu,
            expr_evals: &MultiVecWithBounds<EF, 2>,
            preflights: &[PreflightGpu],
        ) -> Self {
            let constraints = vk
                .cpu
                .inner
                .per_air
                .iter()
                .map(|vk| vk.symbolic_constraints.constraints.constraint_idx.clone())
                .collect_vec();

            let mut values = Vec::with_capacity(preflights.len());
            let mut constraints_folding_per_proof = Vec::with_capacity(preflights.len());
            let mut folded_claims = MultiProofVecVec::new();

            for (pidx, preflight) in preflights.iter().enumerate() {
                let lambda_tidx = preflight.cpu.batch_constraint.lambda_tidx;
                let lambda = EF::from_basis_coefficients_slice(
                    &preflight.cpu.transcript.values()[lambda_tidx..lambda_tidx + D_EF],
                )
                .unwrap();

                let vdata = &preflight.cpu.proof_shape.sorted_trace_vdata;
                let mut proof_values = Vec::with_capacity(vdata.len());

                for (air_idx, v) in vdata.iter() {
                    let mut folded_claim = EF::ZERO;
                    let mut lambda_pow = EF::ONE;

                    let air_values = std::iter::once(EF::ZERO)
                        .chain(constraints[*air_idx].iter().map(|&constr| {
                            let value = expr_evals[[pidx, *air_idx]][constr];
                            folded_claim += lambda_pow * value;
                            lambda_pow *= lambda;
                            value
                        }))
                        .collect_vec();
                    proof_values.push(air_values);

                    let n_lift = v.log_height.saturating_sub(vk.system_params.l_skip);
                    let n = v.log_height as isize - vk.system_params.l_skip as isize;
                    folded_claims.push((
                        n,
                        folded_claim * preflight.cpu.batch_constraint.eq_ns_frontloaded[n_lift],
                    ));
                }

                values.push(proof_values);
                constraints_folding_per_proof.push(FpExtWithTidx {
                    value: lambda,
                    tidx: lambda_tidx as u32,
                });
                folded_claims.end_proof();
            }

            Self {
                values,
                constraints_folding_per_proof,
                folded_claims,
            }
        }
    }

    impl ModuleChip<GpuBackend> for ConstraintsFoldingTraceGenerator {
        type Ctx<'a> = (
            &'a VerifyingKeyGpu,
            &'a [PreflightGpu],
            &'a ConstraintsFoldingBlobGpu,
        );

        #[tracing::instrument(level = "trace", skip_all)]
        fn generate_proving_ctx(
            &self,
            ctx: &Self::Ctx<'_>,
            required_height: Option<usize>,
        ) -> Option<AirProvingContext<GpuBackend>> {
            let (child_vk, preflights_gpu, blob) = ctx;

            let mut num_valid_rows = 0u32;
            let mut row_bounds = Vec::with_capacity(preflights_gpu.len());
            let mut constraint_bounds = Vec::with_capacity(preflights_gpu.len());
            let mut proof_and_sort_idxs = vec![];

            let flat_values = blob
                .values
                .iter()
                .enumerate()
                .flat_map(|(proof_idx, proof_values)| {
                    let mut num_constraints_in_proof = 0;
                    let mut proof_constraint_bounds = Vec::with_capacity(proof_values.len());
                    for (sort_idx, air_values) in proof_values.iter().enumerate() {
                        let num_constraints = air_values.len();
                        num_constraints_in_proof += num_constraints as u32;
                        proof_constraint_bounds.push(num_constraints_in_proof);
                        proof_and_sort_idxs.extend(std::iter::repeat_n(
                            UInt2 {
                                x: proof_idx as u32,
                                y: sort_idx as u32,
                            },
                            num_constraints,
                        ));
                    }
                    num_valid_rows += num_constraints_in_proof;
                    row_bounds.push(num_valid_rows);
                    constraint_bounds.push(proof_constraint_bounds.to_device().unwrap());
                    proof_values.iter().flatten().copied()
                })
                .collect_vec();
            let eq_ns = preflights_gpu
                .iter()
                .map(|preflight| {
                    preflight
                        .cpu
                        .batch_constraint
                        .eq_ns_frontloaded
                        .to_device()
                        .unwrap()
                })
                .collect_vec();

            let height = if let Some(height) = required_height {
                if height < num_valid_rows as usize {
                    return None;
                }
                height
            } else {
                (num_valid_rows as usize).next_power_of_two()
            };
            let width = ConstraintsFoldingCols::<F>::width();
            let d_trace = DeviceMatrix::<F>::with_capacity(height, width);

            let d_proof_and_sort_idxs = proof_and_sort_idxs.to_device().unwrap();
            let d_values = flat_values.to_device().unwrap();
            let d_cur_sum_evals = DeviceBuffer::<AffineFpExt>::with_capacity(d_values.len());

            let d_constraint_bounds = constraint_bounds.iter().map(|b| b.as_ptr()).collect_vec();
            let d_sorted_trace_heights = preflights_gpu
                .iter()
                .map(|preflight| preflight.proof_shape.sorted_trace_heights.as_ptr())
                .collect_vec();
            let d_eq_ns = eq_ns.iter().map(|b| b.as_ptr()).collect_vec();

            let d_per_proof = blob.constraints_folding_per_proof.to_device().unwrap();

            unsafe {
                let temp_bytes = constraints_folding_tracegen_temp_bytes(
                    &d_proof_and_sort_idxs,
                    &d_cur_sum_evals,
                    num_valid_rows,
                )
                .unwrap();
                let d_temp_buffer = DeviceBuffer::<u8>::with_capacity(temp_bytes);
                constraints_folding_tracegen(
                    d_trace.buffer(),
                    height,
                    width,
                    &d_proof_and_sort_idxs,
                    &d_cur_sum_evals,
                    &d_values,
                    &row_bounds,
                    d_constraint_bounds,
                    d_sorted_trace_heights,
                    d_eq_ns,
                    &d_per_proof,
                    preflights_gpu.len() as u32,
                    child_vk.per_air.len() as u32,
                    num_valid_rows,
                    child_vk.system_params.l_skip as u32,
                    &d_temp_buffer,
                    temp_bytes,
                )
                .unwrap();
            }

            Some(AirProvingContext::simple_no_pis(d_trace))
        }
    }
}
