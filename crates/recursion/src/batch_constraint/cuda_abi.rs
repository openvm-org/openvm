#![allow(clippy::missing_safety_doc)]

use cuda_backend_v2::{EF, F};
use openvm_cuda_backend::base::DeviceMatrix;
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer, error::CudaError};
use stark_backend_v2::keygen::types::MultiStarkVerifyingKeyV2;

use crate::{
    batch_constraint::{
        cuda_utils::*,
        eq_airs::{Eq3bBlob, Eq3bColumns, StackedIdxRecord},
        expr_eval::SingleMainSymbolicExpressionColumns,
    },
    cuda::{preflight::PreflightGpu, proof::ProofGpu, to_device_or_nullptr},
    utils::MultiVecWithBounds,
};

extern "C" {
    fn _sym_expr_common_tracegen(
        d_trace: *mut F,
        height: usize,
        l_skip: usize,
        d_log_heights: *const usize,
        d_sort_idx_by_air_idx: *const usize,
        num_airs: usize,
        num_proofs: usize,
        max_num_proofs: usize,
        d_expr_evals: *const EF,
        d_ee_bounds_0: *const usize,
        d_ee_bounds_1: *const usize,
        d_constraint_nodes: *const FlatSymbolicConstraintNode,
        d_constraint_nodes_bounds: *const usize,
        d_interactions: *const FlatInteraction,
        d_interactions_bounds: *const usize,
        d_interaction_messages: *const usize,
        d_unused_variables: *const FlatSymbolicVariable,
        d_unused_variables_bounds: *const usize,
        d_record_bounds: *const u32,
        d_air_ids_per_record: *const u32,
        num_records_per_proof: usize,
        d_sumcheck_rnds: *const EF,
        d_sumcheck_bounds: *const usize,
    ) -> i32;

    fn _eq_3b_tracegen(
        d_trace: *mut F,
        num_valid_rows: usize,
        height: usize,
        num_proofs: usize,
        l_skip: usize,
        records: *const StackedIdxRecord,
        record_bounds: *const usize,
        rows_per_proof_bounds: *const usize,
        n_logups: *const usize,
        xis: *const EF,
        xi_bounds: *const usize,
    ) -> i32;
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn sym_expr_common_tracegen(
    d_trace: &DeviceBuffer<F>,
    height: usize,
    l_skip: usize,
    d_log_heights: &DeviceBuffer<usize>,
    d_sort_idx_by_air_idx: &DeviceBuffer<usize>,
    num_airs: usize,
    num_proofs: usize,
    max_num_proofs: usize,
    d_expr_evals: &DeviceBuffer<EF>,
    d_ee_bounds_0: &DeviceBuffer<usize>,
    d_ee_bounds_1: &DeviceBuffer<usize>,
    d_constraint_nodes: &DeviceBuffer<FlatSymbolicConstraintNode>,
    d_constraint_nodes_bounds: &DeviceBuffer<usize>,
    d_interactions: &DeviceBuffer<FlatInteraction>,
    d_interactions_bounds: &DeviceBuffer<usize>,
    d_interaction_messages: &DeviceBuffer<usize>,
    d_unused_variables: &DeviceBuffer<FlatSymbolicVariable>,
    d_unused_variables_bounds: &DeviceBuffer<usize>,
    d_record_bounds: &DeviceBuffer<u32>,
    d_air_ids_per_record: &DeviceBuffer<u32>,
    num_records_per_proof: usize,
    d_sumcheck_rnds: &DeviceBuffer<EF>,
    d_sumcheck_bounds: &DeviceBuffer<usize>,
) -> Result<(), CudaError> {
    CudaError::from_result(_sym_expr_common_tracegen(
        d_trace.as_mut_ptr(),
        height,
        l_skip,
        d_log_heights.as_ptr(),
        d_sort_idx_by_air_idx.as_ptr(),
        num_airs,
        num_proofs,
        max_num_proofs,
        d_expr_evals.as_ptr(),
        d_ee_bounds_0.as_ptr(),
        d_ee_bounds_1.as_ptr(),
        d_constraint_nodes.as_ptr(),
        d_constraint_nodes_bounds.as_ptr(),
        d_interactions.as_ptr(),
        d_interactions_bounds.as_ptr(),
        d_interaction_messages.as_ptr(),
        d_unused_variables.as_ptr(),
        d_unused_variables_bounds.as_ptr(),
        d_record_bounds.as_ptr(),
        d_air_ids_per_record.as_ptr(),
        num_records_per_proof,
        d_sumcheck_rnds.as_ptr(),
        d_sumcheck_bounds.as_ptr(),
    ))
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn eq_3b_tracegen(
    d_trace: &DeviceBuffer<F>,
    num_valid_rows: usize,
    height: usize,
    num_proofs: usize,
    l_skip: usize,
    d_records: &DeviceBuffer<StackedIdxRecord>,
    d_record_bounds: &DeviceBuffer<usize>,
    d_rows_per_proof_bounds: &DeviceBuffer<usize>,
    d_n_logups: &DeviceBuffer<usize>,
    d_xis: &DeviceBuffer<EF>,
    d_xi_bounds: &DeviceBuffer<usize>,
) -> Result<(), CudaError> {
    CudaError::from_result(_eq_3b_tracegen(
        d_trace.as_mut_ptr(),
        num_valid_rows,
        height,
        num_proofs,
        l_skip,
        d_records.as_ptr(),
        d_record_bounds.as_ptr(),
        d_rows_per_proof_bounds.as_ptr(),
        d_n_logups.as_ptr(),
        d_xis.as_ptr(),
        d_xi_bounds.as_ptr(),
    ))
}

pub fn generate_sym_expr_trace(
    child_vk: &MultiStarkVerifyingKeyV2,
    proofs: &[ProofGpu],
    preflights: &[PreflightGpu],
    max_num_proofs: usize,
    expr_evals: &MultiVecWithBounds<EF, 2>,
) -> DeviceMatrix<F> {
    debug_assert_eq!(proofs.len(), preflights.len());

    let num_airs = child_vk.inner.per_air.len();

    let mut constraint_nodes = MultiVecWithBounds::<_, 1>::new();

    let mut interactions = MultiVecWithBounds::<_, 1>::new();

    let mut interaction_messages = Vec::new();

    let mut unused_variables = MultiVecWithBounds::<_, 1>::new();

    let mut record_bounds = Vec::with_capacity(num_airs + 1);
    record_bounds.push(0);

    let mut total_rows = 0;

    for vk in &child_vk.inner.per_air {
        let constraints = &vk.symbolic_constraints.constraints;
        for node in &constraints.nodes {
            constraint_nodes.push(flatten_constraint_node(vk, node));
        }
        constraint_nodes.close_level(0);

        for interaction in &vk.symbolic_constraints.interactions {
            let message_start = interaction_messages.len();
            interaction_messages.extend(&interaction.message);
            interactions.push(FlatInteraction {
                count: interaction.count as u32,
                message_start: message_start as u32,
                message_len: interaction.message.len() as u32,
                bus_index: u32::from(interaction.bus_index),
                count_weight: interaction.count_weight,
            });
        }
        interactions.close_level(0);

        for unused in &vk.unused_variables {
            unused_variables.push(flatten_unused_symbolic_variable(unused));
        }
        unused_variables.close_level(0);

        let interaction_message_rows: usize = vk
            .symbolic_constraints
            .interactions
            .iter()
            .map(|interaction| interaction.message.len())
            .sum();
        let rows_for_air = constraints.nodes.len()
            + vk.symbolic_constraints.interactions.len()
            + interaction_message_rows
            + vk.unused_variables.len();
        total_rows += rows_for_air;
        record_bounds.push(total_rows as u32);
    }

    let mut air_ids_per_record = vec![0; total_rows];
    for i in 0..(record_bounds.len() - 1) {
        air_ids_per_record[(record_bounds[i] as usize)..(record_bounds[i + 1] as usize)]
            .fill(i as u32);
    }

    let height = total_rows.max(1).next_power_of_two();
    let width = SingleMainSymbolicExpressionColumns::<F>::width() * max_num_proofs;
    let trace = DeviceMatrix::with_capacity(height, width);

    let d_log_heights = proofs
        .iter()
        .flat_map(|proof| {
            proof
                .cpu
                .trace_vdata
                .iter()
                .map(|v| v.as_ref().map_or(0, |td| td.log_height))
        })
        .collect::<Vec<_>>()
        .to_device()
        .unwrap();

    let mut sort_idx_by_air_idx = vec![0usize; num_airs * proofs.len()];
    for (chunk, preflight) in sort_idx_by_air_idx
        .chunks_exact_mut(num_airs)
        .zip(preflights.iter())
    {
        for (sort_idx, (air_idx, _)) in preflight
            .cpu
            .proof_shape
            .sorted_trace_vdata
            .iter()
            .enumerate()
        {
            chunk[*air_idx] = sort_idx;
        }
    }
    let d_sort_idx_by_air_idx = sort_idx_by_air_idx.to_device().unwrap();

    let d_expr_evals = expr_evals.data.to_device().unwrap();
    let d_ee_bounds_0 = expr_evals.bounds[0].to_device().unwrap();
    let d_ee_bounds_1 = expr_evals.bounds[1].to_device().unwrap();

    let d_constraint_nodes = constraint_nodes.data.to_device().unwrap();
    let d_constraint_nodes_bounds = constraint_nodes.bounds[0].to_device().unwrap();
    let d_interactions = to_device_or_nullptr(&interactions.data).unwrap();
    let d_interactions_bounds = interactions.bounds[0].to_device().unwrap();
    let d_interaction_messages = to_device_or_nullptr(&interaction_messages).unwrap();
    let d_unused_variables = to_device_or_nullptr(&unused_variables.data).unwrap();
    let d_unused_variables_bounds = unused_variables.bounds[0].to_device().unwrap();
    let d_record_bounds = record_bounds.to_device().unwrap();
    let d_air_ids_per_record = air_ids_per_record.to_device().unwrap();

    let mut sumcheck_data = Vec::new();
    let mut sumcheck_bounds = Vec::with_capacity(preflights.len() + 1);
    sumcheck_bounds.push(0);
    for preflight in preflights {
        sumcheck_data.extend_from_slice(&preflight.cpu.batch_constraint.sumcheck_rnd);
        sumcheck_bounds.push(sumcheck_data.len());
    }
    let d_sumcheck_rnds = if sumcheck_data.is_empty() {
        DeviceBuffer::new()
    } else {
        sumcheck_data.to_device().unwrap()
    };
    let d_sumcheck_bounds = sumcheck_bounds.to_device().unwrap();

    unsafe {
        sym_expr_common_tracegen(
            trace.buffer(),
            height,
            child_vk.inner.params.l_skip,
            &d_log_heights,
            &d_sort_idx_by_air_idx,
            num_airs,
            proofs.len(),
            max_num_proofs,
            &d_expr_evals,
            &d_ee_bounds_0,
            &d_ee_bounds_1,
            &d_constraint_nodes,
            &d_constraint_nodes_bounds,
            &d_interactions,
            &d_interactions_bounds,
            &d_interaction_messages,
            &d_unused_variables,
            &d_unused_variables_bounds,
            &d_record_bounds,
            &d_air_ids_per_record,
            total_rows,
            &d_sumcheck_rnds,
            &d_sumcheck_bounds,
        )
        .unwrap();
    }
    trace
}

pub fn generate_eq_3b_trace(
    child_vk: &MultiStarkVerifyingKeyV2,
    blob: &Eq3bBlob,
    preflights: &[PreflightGpu],
) -> DeviceMatrix<F> {
    debug_assert_eq!(blob.all_stacked_ids.num_proofs(), preflights.len());

    let num_proofs = preflights.len();
    let width = Eq3bColumns::<F>::width();
    let l_skip = child_vk.inner.params.l_skip;

    let mut device_records = MultiVecWithBounds::<_, 1>::with_capacity(blob.all_stacked_ids.len());

    let mut rows_per_proof_bounds = Vec::with_capacity(num_proofs + 1);
    rows_per_proof_bounds.push(0);

    let mut n_logups = Vec::with_capacity(num_proofs);
    let mut all_xi = MultiVecWithBounds::<_, 1>::new();

    let mut num_valid_rows = 0usize;

    for (pidx, preflight) in preflights.iter().enumerate() {
        let records = &blob.all_stacked_ids[pidx];

        device_records.extend(records.iter().cloned());
        device_records.close_level(0);

        let n_logup = preflight.cpu.proof_shape.n_logup;
        n_logups.push(n_logup);

        let one_height = n_logup + 1;
        let rows_for_proof = records.len() * one_height;
        num_valid_rows += rows_for_proof;
        rows_per_proof_bounds.push(num_valid_rows);

        all_xi.extend(preflight.cpu.batch_constraint.xi.iter().cloned());
        all_xi.close_level(0);
    }

    let height = num_valid_rows.max(1).next_power_of_two();
    let trace = DeviceMatrix::with_capacity(height, width);

    let d_records = to_device_or_nullptr(&device_records.data).unwrap();
    let d_record_bounds = device_records.bounds[0].to_device().unwrap();
    let d_rows_per_proof_bounds = rows_per_proof_bounds.to_device().unwrap();
    let d_n_logups = n_logups.to_device().unwrap();
    let d_xis = to_device_or_nullptr(&all_xi.data).unwrap();
    let d_xi_bounds = all_xi.bounds[0].to_device().unwrap();

    unsafe {
        eq_3b_tracegen(
            trace.buffer(),
            num_valid_rows,
            height,
            num_proofs,
            l_skip,
            &d_records,
            &d_record_bounds,
            &d_rows_per_proof_bounds,
            &d_n_logups,
            &d_xis,
            &d_xi_bounds,
        )
        .unwrap();
    }

    trace
}
