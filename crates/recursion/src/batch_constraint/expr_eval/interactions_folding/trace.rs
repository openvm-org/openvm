use std::borrow::BorrowMut;

use itertools::Itertools;
use openvm_stark_backend::keygen::types::{MultiStarkVerifyingKey, MultiStarkVerifyingKey0};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, D_EF, EF, F};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;

use crate::{
    batch_constraint::{
        eq_airs::Eq3bBlob, expr_eval::interactions_folding::air::InteractionsFoldingCols,
        BatchConstraintBlobCpu,
    },
    system::Preflight,
    tracegen::RowMajorChip,
    utils::{pow_tidx_count, MultiProofVecVec, MultiVecWithBounds},
};

#[derive(Copy, Clone)]
#[repr(C)]
pub(in crate::batch_constraint) struct InteractionsFoldingRecord {
    value: EF,
    air_idx: usize,
    sort_idx: usize,
    interaction_idx: usize,
    node_idx: usize,
    idx_in_message: usize,
    has_interactions: bool,
    is_first_in_air: bool,
    is_last_in_air: bool,
    is_mult: bool,
    is_bus_index: bool,
}

pub(crate) struct InteractionsFoldingBlob {
    pub(in crate::batch_constraint) records: MultiProofVecVec<InteractionsFoldingRecord>,
    // (n, value), n is before lift, can be negative
    pub(in crate::batch_constraint) folded_claims: MultiProofVecVec<(isize, EF)>,
}

impl InteractionsFoldingBlob {
    pub fn new(
        vk: &MultiStarkVerifyingKey0<BabyBearPoseidon2Config>,
        expr_evals: &MultiVecWithBounds<EF, 2>,
        eq_3b_blob: &Eq3bBlob,
        preflights: &[&Preflight],
    ) -> Self {
        let l_skip = vk.params.l_skip;
        let interactions = vk
            .per_air
            .iter()
            .map(|vk| vk.symbolic_constraints.interactions.clone())
            .collect_vec();

        let logup_pow_offset = pow_tidx_count(vk.params.logup.pow_bits);
        let mut records = MultiProofVecVec::new();
        let mut folded = MultiProofVecVec::new();
        for (pidx, preflight) in preflights.iter().enumerate() {
            let beta_tidx = preflight.proof_shape.post_tidx + logup_pow_offset + D_EF;
            let beta = EF::from_basis_coefficients_slice(
                &preflight.transcript.values()[beta_tidx..beta_tidx + D_EF],
            )
            .unwrap();

            let eq_3bs = &eq_3b_blob.all_stacked_ids[pidx];
            let mut cur_eq3b_idx = 0;

            let vdata = &preflight.proof_shape.sorted_trace_vdata;
            for (sort_idx, (air_idx, vdata)) in vdata.iter().enumerate() {
                let n = vdata.log_height as isize - l_skip as isize;
                let inters = &interactions[*air_idx];
                let mut num_sum = EF::ZERO;
                let mut denom_sum = EF::ZERO;
                if inters.is_empty() {
                    records.push(InteractionsFoldingRecord {
                        value: EF::ZERO,
                        air_idx: *air_idx,
                        sort_idx,
                        interaction_idx: 0,
                        node_idx: 0,
                        idx_in_message: 0,
                        has_interactions: false,
                        is_first_in_air: true,
                        is_last_in_air: true,
                        is_mult: false,
                        is_bus_index: false,
                    });
                    cur_eq3b_idx += 1;
                } else {
                    // `cur_interactions_evals` in rust verifier are the list of evaluated
                    // node_claims After multiplying with eq_3b and sum together we get the
                    // `num` and `denom` in rust verifier.
                    for (interaction_idx, inter) in inters.iter().enumerate() {
                        let eq_3b = eq_3bs[cur_eq3b_idx].eq_mle(
                            &preflight.batch_constraint.xi,
                            vk.params.l_skip,
                            preflight.proof_shape.n_logup,
                        );
                        cur_eq3b_idx += 1;
                        records.push(InteractionsFoldingRecord {
                            value: expr_evals[[pidx, *air_idx]][inter.count],
                            air_idx: *air_idx,
                            sort_idx,
                            interaction_idx,
                            node_idx: inter.count,
                            idx_in_message: 0,
                            has_interactions: true,
                            is_first_in_air: interaction_idx == 0,
                            is_last_in_air: false,
                            is_mult: true, /* for each interaction, only the first record with
                                            * is_mult = true */
                            is_bus_index: false,
                        });
                        num_sum += expr_evals[[pidx, *air_idx]][inter.count] * eq_3b;

                        let mut beta_pow = EF::ONE;
                        let mut cur_sum = EF::ZERO;
                        for (j, &node_idx) in inter.message.iter().enumerate() {
                            let value = expr_evals[[pidx, *air_idx]][node_idx];
                            cur_sum += beta_pow * value;
                            beta_pow *= beta;
                            records.push(InteractionsFoldingRecord {
                                value,
                                air_idx: *air_idx,
                                sort_idx,
                                interaction_idx,
                                node_idx,
                                idx_in_message: j,
                                has_interactions: true,
                                is_first_in_air: false,
                                is_last_in_air: false,
                                is_mult: false,
                                is_bus_index: false,
                            });
                        }

                        cur_sum += beta_pow * EF::from_u16(inter.bus_index + 1);
                        records.push(InteractionsFoldingRecord {
                            value: EF::from_u16(inter.bus_index + 1),
                            air_idx: *air_idx,
                            sort_idx,
                            interaction_idx,
                            node_idx: inter.bus_index as usize + 1,
                            idx_in_message: inter.message.len(),
                            has_interactions: true,
                            is_first_in_air: false,
                            is_last_in_air: interaction_idx + 1 == inters.len(),
                            is_mult: false,
                            is_bus_index: true,
                        });
                        denom_sum += cur_sum * eq_3b;
                    }
                }
                // Finally, this should be `interactions_evals`, minus norm_factor and eq_sharp_ns.
                folded.push((n, num_sum));
                folded.push((n, denom_sum));
            }
            folded.end_proof();
            records.end_proof();
        }
        Self {
            records,
            folded_claims: folded,
        }
    }
}

pub struct InteractionsFoldingTraceGenerator;

impl RowMajorChip<F> for InteractionsFoldingTraceGenerator {
    type Ctx<'a> = (
        &'a MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        &'a BatchConstraintBlobCpu,
        &'a [&'a Preflight],
    );

    #[tracing::instrument(level = "trace", skip_all)]
    fn generate_trace(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>> {
        let (vk, blob, preflights) = ctx;
        let eq_3b_blob = &blob.common_blob.eq_3b_blob;
        let if_blob = blob.if_blob.as_ref().unwrap();

        let width = InteractionsFoldingCols::<F>::width();

        let total_height = if_blob.records.len();
        let padding_height = if let Some(height) = required_height {
            if height < total_height {
                return None;
            }
            height
        } else {
            total_height.next_power_of_two()
        };
        let mut trace = vec![F::ZERO; padding_height * width];

        let logup_pow_offset = pow_tidx_count(vk.inner.params.logup.pow_bits);
        let mut cur_height = 0;
        for (pidx, preflight) in preflights.iter().enumerate() {
            let beta_tidx = preflight.proof_shape.post_tidx + logup_pow_offset + D_EF;
            let beta_slice = &preflight.transcript.values()[beta_tidx..beta_tidx + D_EF];
            let records = &if_blob.records[pidx];
            let eq_3bs = &eq_3b_blob.all_stacked_ids[pidx];

            let mut is_first_in_message_indices = vec![];
            let mut cur_eq3b_idx = -1i32;
            let mut was_first_interaction_in_message = false;
            trace[cur_height * width..(cur_height + records.len()) * width]
                .chunks_exact_mut(width)
                .enumerate()
                .for_each(|(i, chunk)| {
                    let cols: &mut InteractionsFoldingCols<_> = chunk.borrow_mut();
                    let record = &records[i];
                    cols.is_valid = F::ONE;
                    cols.proof_idx = F::from_usize(pidx);
                    cols.beta_tidx = F::from_usize(beta_tidx);
                    cols.air_idx = F::from_usize(record.air_idx);
                    cols.sort_idx = F::from_usize(record.sort_idx);
                    cols.interaction_idx = F::from_usize(record.interaction_idx);
                    cols.node_idx = F::from_usize(record.node_idx);
                    cols.has_interactions = F::from_bool(record.has_interactions);
                    cols.is_first_in_air = F::from_bool(record.is_first_in_air);
                    cols.is_first_in_message =
                        F::from_bool(record.is_mult || !record.has_interactions);
                    cols.is_second_in_message = F::from_bool(was_first_interaction_in_message);
                    was_first_interaction_in_message = record.is_mult;
                    cols.is_bus_index = F::from_bool(record.is_bus_index);
                    cols.idx_in_message = F::from_usize(record.idx_in_message);
                    cols.value
                        .copy_from_slice(record.value.as_basis_coefficients_slice());
                    cols.beta.copy_from_slice(beta_slice);

                    if !record.has_interactions || record.is_mult {
                        cur_eq3b_idx += 1;
                    }
                    if record.has_interactions {
                        cols.eq_3b.copy_from_slice(
                            eq_3bs[cur_eq3b_idx as usize]
                                .eq_mle(
                                    &preflight.batch_constraint.xi,
                                    vk.inner.params.l_skip,
                                    preflight.proof_shape.n_logup,
                                )
                                .as_basis_coefficients_slice(),
                        );
                    }

                    if cols.is_first_in_message == F::ONE && record.has_interactions {
                        is_first_in_message_indices.push(i);
                    }
                });

            // Setting `cur_sum` and final acc
            let mut cur_sum = EF::ZERO;
            let beta = EF::from_basis_coefficients_slice(beta_slice).unwrap();
            let mut cur_acc_num = EF::ZERO;
            let mut cur_acc_denom = EF::ZERO;
            trace[cur_height * width..(cur_height + records.len()) * width]
                .chunks_exact_mut(width)
                .enumerate()
                .rev()
                .for_each(|(i, chunk)| {
                    let cols: &mut InteractionsFoldingCols<_> = chunk.borrow_mut();
                    // Handling cur_sum
                    if cols.is_first_in_message == F::ONE {
                        cols.cur_sum.copy_from_slice(&cols.value);
                        cur_sum = EF::ZERO;
                    } else {
                        cur_sum = cur_sum * beta
                            + EF::from_basis_coefficients_slice(&cols.value).unwrap();
                        cols.cur_sum
                            .copy_from_slice(cur_sum.as_basis_coefficients_slice());
                    }

                    // Adding to the final acc
                    if cols.is_first_in_message == F::ONE {
                        // Case 1: first in message, only accumulate the num
                        cur_acc_num += EF::from_basis_coefficients_slice(&cols.cur_sum).unwrap()
                            * EF::from_basis_coefficients_slice(&cols.eq_3b).unwrap();
                        if cols.has_interactions == F::ZERO {
                            debug_assert_eq!(cols.cur_sum, [F::ZERO; D_EF]);
                        }
                    } else if is_first_in_message_indices.contains(&(i - 1)) {
                        // Case 2: second in message, accumulate the denom
                        cur_acc_denom += EF::from_basis_coefficients_slice(&cols.cur_sum).unwrap()
                            * EF::from_basis_coefficients_slice(&cols.eq_3b).unwrap();
                    }
                    cols.final_acc_num
                        .copy_from_slice(cur_acc_num.as_basis_coefficients_slice());
                    cols.final_acc_denom
                        .copy_from_slice(cur_acc_denom.as_basis_coefficients_slice());

                    // Reset per AIR
                    if cols.is_first_in_air == F::ONE {
                        cur_acc_num = EF::ZERO;
                        cur_acc_denom = EF::ZERO;
                    }
                });

            // Setting is_first and is_last for this proof
            {
                let cols: &mut InteractionsFoldingCols<_> =
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
            interactions_folding_tracegen, interactions_folding_tracegen_temp_bytes, AffineFpExt,
            FpExtWithTidx, InteractionRecord,
        },
        cuda::{preflight::PreflightGpu, vk::VerifyingKeyGpu},
        tracegen::ModuleChip,
        utils::interaction_length,
    };

    pub struct InteractionsFoldingBlobGpu {
        // Per proof, per AIR, per interaction, per index
        pub values: Vec<Vec<Vec<Vec<EF>>>>,
        // Per valid row
        pub node_idxs: Vec<u32>,
        // Per proof, per interaction
        pub interaction_records: Vec<Vec<InteractionRecord>>,
        // Per proof
        pub interactions_folding_per_proof: Vec<FpExtWithTidx>,
        // For compatibility with CPU tracegen
        pub folded_claims: MultiProofVecVec<(isize, EF)>,
    }

    impl InteractionsFoldingBlobGpu {
        pub fn new(
            vk: &VerifyingKeyGpu,
            expr_evals: &MultiVecWithBounds<EF, 2>,
            eq_3b_blob: &Eq3bBlob,
            preflights: &[PreflightGpu],
        ) -> Self {
            let l_skip = vk.system_params.l_skip;
            let interactions = vk
                .cpu
                .inner
                .per_air
                .iter()
                .map(|vk| vk.symbolic_constraints.interactions.clone())
                .collect_vec();

            let mut global_current_row = 0;

            let mut values = Vec::with_capacity(preflights.len());
            let mut node_idxs = vec![];
            let mut interaction_records = Vec::with_capacity(preflights.len());
            let mut interactions_folding_per_proof = Vec::with_capacity(preflights.len());
            let mut folded_claims = MultiProofVecVec::new();
            let logup_pow_offset = pow_tidx_count(vk.cpu.inner.params.logup.pow_bits);

            for (pidx, preflight) in preflights.iter().enumerate() {
                let beta_tidx = preflight.proof_shape.post_tidx + logup_pow_offset + D_EF;
                let beta = EF::from_basis_coefficients_slice(
                    &preflight.cpu.transcript.values()[beta_tidx..beta_tidx + D_EF],
                )
                .unwrap();

                let eq_3bs = &eq_3b_blob.all_stacked_ids[pidx];
                let mut cur_eq3b_idx = 0;

                let vdata = &preflight.cpu.proof_shape.sorted_trace_vdata;
                let mut proof_values = Vec::with_capacity(vdata.len());
                let mut proof_interaction_records = vec![];

                for (air_idx, vdata) in vdata {
                    let n = vdata.log_height as isize - l_skip as isize;
                    let inters = &interactions[*air_idx];

                    let mut num_sum = EF::ZERO;
                    let mut denom_sum = EF::ZERO;
                    let mut air_values = Vec::with_capacity(inters.len());

                    if inters.is_empty() {
                        // Note differs from what is written in CPU blob generation, but matches
                        // tracegen
                        air_values.push(vec![EF::ZERO]);
                        node_idxs.push(0);
                        proof_interaction_records.push(InteractionRecord {
                            interaction_num_rows: 1,
                            global_start_row: global_current_row,
                            stacked_idx: 0,
                        });
                        global_current_row += 1;
                        cur_eq3b_idx += 1;
                    } else {
                        for inter in inters {
                            let stacked_idx_record = eq_3bs[cur_eq3b_idx];
                            let eq_3b = stacked_idx_record.eq_mle(
                                &preflight.cpu.batch_constraint.xi,
                                l_skip,
                                preflight.proof_shape.n_logup,
                            );
                            cur_eq3b_idx += 1;
                            num_sum += expr_evals[[pidx, *air_idx]][inter.count] * eq_3b;

                            let interaction_num_rows = interaction_length(inter);
                            proof_interaction_records.push(InteractionRecord {
                                interaction_num_rows: interaction_num_rows as u32,
                                global_start_row: global_current_row,
                                stacked_idx: stacked_idx_record.stacked_idx,
                            });
                            global_current_row += interaction_num_rows as u32;

                            let mut interaction_values = Vec::with_capacity(interaction_num_rows);
                            interaction_values.push(expr_evals[[pidx, *air_idx]][inter.count]);
                            node_idxs.push(inter.count as u32);

                            let mut beta_pow = EF::ONE;
                            let mut cur_sum = EF::ZERO;
                            for &node_idx in &inter.message {
                                let value = expr_evals[[pidx, *air_idx]][node_idx];
                                cur_sum += beta_pow * value;
                                beta_pow *= beta;
                                interaction_values.push(value);
                                node_idxs.push(node_idx as u32);
                            }

                            let bus_value = EF::from_u16(inter.bus_index + 1);
                            cur_sum += beta_pow * bus_value;
                            interaction_values.push(bus_value);
                            node_idxs.push(inter.bus_index as u32 + 1);
                            denom_sum += cur_sum * eq_3b;

                            air_values.push(interaction_values);
                        }
                    }

                    proof_values.push(air_values);
                    folded_claims.push((n, num_sum));
                    folded_claims.push((n, denom_sum));
                }

                values.push(proof_values);
                interaction_records.push(proof_interaction_records);
                interactions_folding_per_proof.push(FpExtWithTidx {
                    value: beta,
                    tidx: beta_tidx as u32,
                });
                folded_claims.end_proof();
            }

            Self {
                values,
                node_idxs,
                interaction_records,
                interactions_folding_per_proof,
                folded_claims,
            }
        }
    }

    impl ModuleChip<GpuBackend> for InteractionsFoldingTraceGenerator {
        type Ctx<'a> = (
            &'a VerifyingKeyGpu,
            &'a [PreflightGpu],
            &'a InteractionsFoldingBlobGpu,
        );

        #[tracing::instrument(name = "generate_trace", level = "trace", skip_all)]
        fn generate_proving_ctx(
            &self,
            ctx: &Self::Ctx<'_>,
            required_height: Option<usize>,
        ) -> Option<AirProvingContext<GpuBackend>> {
            let (child_vk, preflights_gpu, blob) = ctx;

            let num_airs = preflights_gpu
                .iter()
                .map(|preflight| preflight.cpu.proof_shape.sorted_trace_vdata.len() as u32)
                .collect_vec();
            let n_logups = preflights_gpu
                .iter()
                .map(|p| p.proof_shape.n_logup as u32)
                .collect_vec();
            let mut num_valid_rows = 0u32;

            let mut row_bounds = Vec::with_capacity(preflights_gpu.len());
            let mut air_interaction_bounds = Vec::with_capacity(preflights_gpu.len());
            let mut interaction_row_bounds = Vec::with_capacity(preflights_gpu.len());

            let expected_num_valid_rows = blob.node_idxs.len();
            let mut idx_keys = Vec::with_capacity(expected_num_valid_rows);
            let mut flat_values = Vec::with_capacity(expected_num_valid_rows);

            for (proof_idx, proof_values) in blob.values.iter().enumerate() {
                let mut proof_num_rows = 0;
                let mut proof_num_interactions = 0;
                let mut proof_air_interaction_bounds =
                    Vec::with_capacity(num_airs[proof_idx] as usize);
                let mut proof_interaction_row_bounds = vec![];

                for air_values in proof_values {
                    for interaction_values in air_values {
                        let global_interaction_idx = proof_num_interactions;
                        proof_num_interactions += 1;
                        for v in interaction_values {
                            flat_values.push(*v);
                            idx_keys.push(UInt2 {
                                x: proof_idx as u32,
                                y: global_interaction_idx,
                            });
                        }
                        proof_num_rows += interaction_values.len() as u32;
                        proof_interaction_row_bounds.push(proof_num_rows);
                    }
                    proof_air_interaction_bounds.push(proof_num_interactions);
                }

                num_valid_rows += proof_num_rows;
                row_bounds.push(num_valid_rows);
                air_interaction_bounds.push(proof_air_interaction_bounds.to_device().unwrap());
                interaction_row_bounds.push(proof_interaction_row_bounds.to_device().unwrap());
            }

            assert_eq!(num_valid_rows as usize, expected_num_valid_rows);

            let records = blob
                .interaction_records
                .iter()
                .map(|records| records.to_device().unwrap())
                .collect_vec();
            let xis = preflights_gpu
                .iter()
                .map(|preflight| preflight.cpu.batch_constraint.xi.to_device().unwrap())
                .collect_vec();

            let height = if let Some(height) = required_height {
                if height < num_valid_rows as usize {
                    return None;
                }
                height
            } else {
                (num_valid_rows as usize).next_power_of_two()
            };
            let width = InteractionsFoldingCols::<F>::width();
            let d_trace = DeviceMatrix::<F>::with_capacity(height, width);

            let d_idx_keys = idx_keys.to_device().unwrap();
            let d_values = flat_values.to_device().unwrap();
            let d_node_idxs = blob.node_idxs.to_device().unwrap();
            let d_cur_sum_evals = DeviceBuffer::<AffineFpExt>::with_capacity(d_values.len());

            let d_air_interaction_bounds = air_interaction_bounds
                .iter()
                .map(|b| b.as_ptr())
                .collect_vec();
            let d_interaction_row_bounds = interaction_row_bounds
                .iter()
                .map(|b| b.as_ptr())
                .collect_vec();
            let d_sorted_trace_vdata = preflights_gpu
                .iter()
                .map(|preflight| preflight.proof_shape.sorted_trace_heights.as_ptr())
                .collect_vec();
            let d_records = records.iter().map(|b| b.as_ptr()).collect_vec();
            let d_xis = xis.iter().map(|b| b.as_ptr()).collect_vec();

            let d_per_proof = blob.interactions_folding_per_proof.to_device().unwrap();

            unsafe {
                let temp_bytes = interactions_folding_tracegen_temp_bytes(
                    d_trace.buffer(),
                    height,
                    &d_idx_keys,
                    &d_cur_sum_evals,
                    num_valid_rows,
                )
                .unwrap();
                let d_temp_buffer = DeviceBuffer::<u8>::with_capacity(temp_bytes);
                interactions_folding_tracegen(
                    d_trace.buffer(),
                    height,
                    width,
                    &d_idx_keys,
                    &d_cur_sum_evals,
                    &d_values,
                    &d_node_idxs,
                    &row_bounds,
                    d_air_interaction_bounds,
                    d_interaction_row_bounds,
                    d_sorted_trace_vdata,
                    d_records,
                    d_xis,
                    &d_per_proof,
                    &num_airs,
                    &n_logups,
                    preflights_gpu.len() as u32,
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
