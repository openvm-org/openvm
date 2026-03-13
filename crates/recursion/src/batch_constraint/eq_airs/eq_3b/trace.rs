use std::borrow::BorrowMut;

use openvm_stark_backend::keygen::types::{MultiStarkVerifyingKey, MultiStarkVerifyingKey0};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, EF, F};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;

use crate::{
    batch_constraint::eq_airs::eq_3b::air::Eq3bColumns, system::Preflight, tracegen::RowMajorChip,
    utils::MultiProofVecVec,
};

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub(crate) struct StackedIdxRecord {
    pub sort_idx: u32,
    pub interaction_idx: u32,
    pub stacked_idx: u32,
    pub n_lift: u32,
    pub is_last_in_air: bool,
    pub no_interactions: bool,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub(crate) struct RecordIdx {
    pub record_idx: u32,
    pub local_n: u32,
}

impl StackedIdxRecord {
    pub fn from_usizes(
        sort_idx: usize,
        interaction_idx: usize,
        stacked_idx: usize,
        n_lift: usize,
        is_last_in_air: bool,
        no_interactions: bool,
    ) -> Self {
        Self {
            sort_idx: sort_idx as u32,
            interaction_idx: interaction_idx as u32,
            stacked_idx: stacked_idx as u32,
            n_lift: n_lift as u32,
            is_last_in_air,
            no_interactions,
        }
    }

    pub fn eq_mle(&self, xi: &[EF], l_skip: usize, n_logup: usize) -> EF {
        xi[l_skip + (self.n_lift as usize)..l_skip + n_logup]
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                if self.stacked_idx & (1 << (l_skip + (self.n_lift as usize) + i)) > 0 {
                    x
                } else {
                    EF::ONE - x
                }
            })
            .fold(EF::ONE, |acc, x| acc * x)
    }
}

pub struct Eq3bBlob {
    pub(crate) all_stacked_ids: MultiProofVecVec<StackedIdxRecord>,
    pub(crate) record_idxs: MultiProofVecVec<RecordIdx>,
}

impl Eq3bBlob {
    fn new() -> Self {
        Self {
            all_stacked_ids: MultiProofVecVec::new(),
            record_idxs: MultiProofVecVec::new(),
        }
    }
}

pub(crate) fn generate_eq_3b_blob(
    vk: &MultiStarkVerifyingKey0<BabyBearPoseidon2Config>,
    preflights: &[&Preflight],
) -> Eq3bBlob {
    let l_skip = vk.params.l_skip;
    let mut blob = Eq3bBlob::new();
    for preflight in preflights.iter() {
        let mut row_idx = 0;
        let mut record_idx = 0;
        let n_logup = preflight.proof_shape.n_logup;
        for (sort_idx, (air_idx, vdata)) in
            preflight.proof_shape.sorted_trace_vdata.iter().enumerate()
        {
            let n_lift = vdata.log_height.saturating_sub(l_skip);
            let num_interactions = vk.per_air[*air_idx].num_interactions();
            if num_interactions == 0 {
                blob.record_idxs.push(RecordIdx {
                    record_idx,
                    local_n: 0,
                });
                blob.all_stacked_ids.push(StackedIdxRecord::from_usizes(
                    sort_idx, 0, row_idx, n_lift, true, true,
                ));
                record_idx += 1;
            } else {
                for i in 0..num_interactions {
                    blob.record_idxs
                        .extend((0..=n_logup).map(|local_n| RecordIdx {
                            record_idx,
                            local_n: local_n as u32,
                        }));
                    blob.all_stacked_ids.push(StackedIdxRecord::from_usizes(
                        sort_idx,
                        i,
                        row_idx,
                        n_lift,
                        i + 1 == num_interactions,
                        false,
                    ));
                    row_idx += 1 << (l_skip + n_lift);
                    record_idx += 1;
                }
            }
        }
        debug_assert!(row_idx <= 1 << (l_skip + n_logup));
        blob.all_stacked_ids.end_proof();
        blob.record_idxs.end_proof();
    }
    blob
}

pub struct Eq3bTraceGenerator;

impl RowMajorChip<F> for Eq3bTraceGenerator {
    type Ctx<'a> = (
        &'a MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        &'a Eq3bBlob,
        &'a [&'a Preflight],
    );

    #[tracing::instrument(level = "trace", skip_all)]
    fn generate_trace(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>> {
        let (vk, blob, preflights) = ctx;
        let width = Eq3bColumns::<F>::width();
        let l_skip = vk.inner.params.l_skip;

        let total_valid = preflights
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let num_dummy_records = blob.all_stacked_ids[i]
                    .iter()
                    .filter(|record| record.no_interactions)
                    .count();
                (p.proof_shape.n_logup + 1) * (blob.all_stacked_ids[i].len() - num_dummy_records)
                    + num_dummy_records
            })
            .sum();
        let padding_height = if let Some(height) = required_height {
            if height < total_valid {
                return None;
            }
            height
        } else {
            total_valid.next_power_of_two()
        };

        let mut trace = vec![F::ZERO; padding_height * width];

        let mut cur_height = 0;
        for (pidx, preflight) in preflights.iter().enumerate() {
            let xi = &preflight.batch_constraint.xi;
            let n_logup = preflight.proof_shape.n_logup;

            let stacked_ids = &blob.all_stacked_ids[pidx];
            let one_height = n_logup + 1;

            for (j, record) in stacked_ids.iter().enumerate() {
                let this_height = if record.no_interactions {
                    1
                } else {
                    one_height
                };
                let shifted_idx = record.stacked_idx >> l_skip;
                let mut cur_eq = EF::ONE;
                trace[cur_height * width..(cur_height + this_height) * width]
                    .chunks_exact_mut(width)
                    .enumerate()
                    .for_each(|(n, chunk)| {
                        let cols: &mut Eq3bColumns<_> = chunk.borrow_mut();
                        cols.is_valid = F::ONE;
                        cols.is_first = F::from_bool(j == 0 && n == 0);
                        cols.proof_idx = F::from_usize(pidx);

                        cols.sort_idx = F::from_u32(record.sort_idx);
                        cols.interaction_idx = F::from_u32(record.interaction_idx);
                        cols.n_lift = F::from_u32(record.n_lift);
                        cols.n_logup = F::from_usize(n_logup);
                        cols.two_to_the_n_lift = F::from_usize(1 << record.n_lift);
                        cols.n = F::from_usize(n);
                        cols.n_at_least_n_lift = F::from_bool(n >= record.n_lift as usize);
                        cols.has_no_interactions = F::from_bool(record.no_interactions);
                        cols.hypercube_volume = F::from_usize(1 << n);
                        cols.is_first_in_air = F::from_bool(
                            record.interaction_idx == 0 && n == 0 || record.no_interactions,
                        );
                        cols.is_first_in_interaction =
                            F::from_bool(n == 0 || record.no_interactions);
                        cols.idx = F::from_u32(shifted_idx & ((1 << n) - 1));
                        cols.running_idx = F::from_u32(shifted_idx);
                        let nth_bit = (shifted_idx & (1 << n)) > 0;
                        cols.nth_bit = F::from_bool(nth_bit);
                        let xi = if (record.n_lift as usize..n_logup).contains(&n) {
                            xi[l_skip + n]
                        } else if nth_bit {
                            EF::ONE
                        } else {
                            EF::ZERO
                        };
                        cols.xi.copy_from_slice(xi.as_basis_coefficients_slice());
                        cols.eq
                            .copy_from_slice(cur_eq.as_basis_coefficients_slice());
                        cur_eq *= if nth_bit { xi } else { EF::ONE - xi };
                    });
                cur_height += this_height;
            }
        }

        Some(RowMajorMatrix::new(trace, width))
    }
}

#[cfg(feature = "cuda")]
pub(in crate::batch_constraint) mod cuda {
    use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
    use openvm_cuda_common::copy::MemCopyH2D;
    use openvm_stark_backend::prover::AirProvingContext;

    use super::*;
    use crate::{
        batch_constraint::cuda_abi::eq_3b_tracegen,
        cuda::{preflight::PreflightGpu, to_device_or_nullptr},
        tracegen::ModuleChip,
        utils::MultiVecWithBounds,
    };

    impl ModuleChip<GpuBackend> for Eq3bTraceGenerator {
        type Ctx<'a> = (
            &'a MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
            &'a Eq3bBlob,
            &'a [PreflightGpu],
        );

        #[tracing::instrument(name = "generate_trace", level = "trace", skip_all)]
        fn generate_proving_ctx(
            &self,
            ctx: &Self::Ctx<'_>,
            required_height: Option<usize>,
        ) -> Option<AirProvingContext<GpuBackend>> {
            let (child_vk, blob, preflights) = ctx;
            debug_assert_eq!(blob.all_stacked_ids.num_proofs(), preflights.len());

            let num_proofs = preflights.len();
            let width = Eq3bColumns::<F>::width();
            let l_skip = child_vk.inner.params.l_skip;

            let mut device_records =
                MultiVecWithBounds::<_, 1>::with_capacity(blob.all_stacked_ids.len());
            let mut record_idxs = MultiVecWithBounds::<_, 1>::with_capacity(blob.record_idxs.len());

            let mut rows_per_proof_bounds = Vec::with_capacity(num_proofs + 1);
            rows_per_proof_bounds.push(0);

            let mut n_logups = Vec::with_capacity(num_proofs);
            let mut all_xi = MultiVecWithBounds::<_, 1>::new();

            let mut num_valid_rows = 0usize;

            for (pidx, preflight) in preflights.iter().enumerate() {
                let records = &blob.all_stacked_ids[pidx];

                device_records.extend(records.iter().cloned());
                device_records.close_level(0);
                record_idxs.extend(blob.record_idxs[pidx].iter().cloned());
                record_idxs.close_level(0);

                let n_logup = preflight.cpu.proof_shape.n_logup;
                n_logups.push(n_logup);

                let one_height = n_logup + 1;
                let rows_for_proof = records.iter().fold(0, |acc, record| {
                    acc + if record.no_interactions {
                        1
                    } else {
                        one_height
                    }
                });
                num_valid_rows += rows_for_proof;
                rows_per_proof_bounds.push(num_valid_rows);

                all_xi.extend(preflight.cpu.batch_constraint.xi.iter().cloned());
                all_xi.close_level(0);
            }

            let height = if let Some(height) = required_height {
                if height < num_valid_rows {
                    return None;
                }
                height
            } else {
                num_valid_rows.max(1).next_power_of_two()
            };
            let d_trace = DeviceMatrix::with_capacity(height, width);

            let d_records = to_device_or_nullptr(&device_records.data).unwrap();
            let d_record_bounds = device_records.bounds[0].to_device().unwrap();
            let d_record_idxs = to_device_or_nullptr(&record_idxs.data).unwrap();
            let d_record_idxs_bounds = record_idxs.bounds[0].to_device().unwrap();
            let d_rows_per_proof_bounds = rows_per_proof_bounds.to_device().unwrap();
            let d_n_logups = n_logups.to_device().unwrap();
            let d_xis = to_device_or_nullptr(&all_xi.data).unwrap();
            let d_xi_bounds = all_xi.bounds[0].to_device().unwrap();

            unsafe {
                eq_3b_tracegen(
                    d_trace.buffer(),
                    num_valid_rows,
                    height,
                    num_proofs,
                    l_skip,
                    &d_records,
                    &d_record_bounds,
                    &d_record_idxs,
                    &d_record_idxs_bounds,
                    &d_rows_per_proof_bounds,
                    &d_n_logups,
                    &d_xis,
                    &d_xi_bounds,
                )
                .unwrap();
            }

            Some(AirProvingContext::simple_no_pis(d_trace))
        }
    }
}
