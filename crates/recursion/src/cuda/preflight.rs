use std::iter::once;

use itertools::Itertools;
use openvm_cuda_backend::prelude::EF;
use openvm_cuda_common::d_buffer::DeviceBuffer;
use openvm_stark_backend::{keygen::types::MultiStarkVerifyingKey, proof::Proof};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, Digest};

use crate::{
    cuda::{
        to_device_or_nullptr,
        types::{TraceHeight, TraceMetadata},
    },
    proof_shape::proof_shape::compute_air_shape_lookup_counts,
    system::Preflight,
};

/*
 * Tracegen information (i.e. records) on a GPU device. Each field should
 * be computable as soon as the verifier circuit runs preflight.
 */
#[derive(Debug)]
pub struct PreflightGpu {
    // TODO[TEMP]: cpu preflight for hybrid usage; remove this when no longer needed
    // If you need something from `cpu` for actual cuda tracegen, move it to a direct field of
    // PreflightGpu. Host and/or device types allowed.
    pub cpu: Preflight,
    pub transcript: TranscriptLog,
    pub proof_shape: ProofShapePreflightGpu,
    pub gkr: GkrPreflightGpu,
    pub batch_constraint: BatchConstraintPreflightGpu,
    pub stacking: StackingPreflightGpu,
    pub whir: WhirPreflightGpu,
}

#[derive(Debug)]
pub struct TranscriptLog {
    _dummy: usize,
}

#[derive(Debug)]
pub struct ProofShapePreflightGpu {
    pub sorted_trace_heights: DeviceBuffer<TraceHeight>,
    pub sorted_trace_metadata: DeviceBuffer<TraceMetadata>,
    pub sorted_cached_commits: DeviceBuffer<Digest>,

    pub per_row_tidx: DeviceBuffer<usize>,
    pub pvs_tidx: DeviceBuffer<usize>,
    pub post_tidx: usize,

    pub num_present: usize,
    pub n_max: usize,
    pub n_logup: usize,
    pub final_cidx: usize,
    pub final_total_interactions: usize,
    pub main_commit: Digest,
}

#[derive(Debug)]
pub struct GkrPreflightGpu {
    _dummy: usize,
}

#[derive(Debug)]
pub struct BatchConstraintPreflightGpu {
    pub sumcheck_rnd: DeviceBuffer<EF>,
}

#[derive(Debug)]
pub struct StackingPreflightGpu {
    pub sumcheck_rnd: DeviceBuffer<EF>,
}

#[derive(Debug)]
pub struct WhirPreflightGpu {
    _dummy: usize,
}

impl PreflightGpu {
    pub fn new(
        vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        proof: &Proof<BabyBearPoseidon2Config>,
        preflight: &Preflight,
    ) -> Self {
        PreflightGpu {
            cpu: preflight.clone(),
            transcript: Self::transcript(preflight),
            proof_shape: Self::proof_shape(vk, proof, preflight),
            gkr: Self::gkr(preflight),
            batch_constraint: Self::batch_constraint(preflight),
            stacking: Self::stacking(preflight),
            whir: Self::whir(preflight),
        }
    }

    fn transcript(_preflight: &Preflight) -> TranscriptLog {
        TranscriptLog { _dummy: 0 }
    }

    fn proof_shape(
        vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        proof: &Proof<BabyBearPoseidon2Config>,
        preflight: &Preflight,
    ) -> ProofShapePreflightGpu {
        let mut sorted_cached_commits: Vec<Digest> = vec![];
        let mut cidx = 1;
        let mut total_interactions = 0;
        let l_skip = vk.inner.params.l_skip;

        let bc_air_shape_lookups = compute_air_shape_lookup_counts(vk);

        let (sorted_trace_heights, sorted_trace_metadata): (Vec<_>, Vec<_>) = preflight
            .proof_shape
            .sorted_trace_vdata
            .iter()
            .map(|(air_idx, vdata)| {
                let height = TraceHeight {
                    air_idx: *air_idx,
                    log_height: vdata.log_height.try_into().unwrap(),
                };
                let metadata = TraceMetadata {
                    cached_idx: sorted_cached_commits.len(),
                    starting_cidx: cidx,
                    total_interactions,
                    num_air_id_lookups: bc_air_shape_lookups[*air_idx],
                };
                cidx += vdata.cached_commitments.len()
                    + vk.inner.per_air[*air_idx].preprocessed_data.is_some() as usize;
                total_interactions += (1 << vdata.log_height.max(l_skip))
                    * vk.inner.per_air[*air_idx].num_interactions();
                sorted_cached_commits.extend_from_slice(&vdata.cached_commitments);
                (height, metadata)
            })
            .chain(
                proof
                    .trace_vdata
                    .iter()
                    .enumerate()
                    .filter_map(|(air_idx, vdata)| {
                        if vdata.is_none() {
                            Some((
                                TraceHeight {
                                    air_idx,
                                    ..Default::default()
                                },
                                TraceMetadata::default(),
                            ))
                        } else {
                            None
                        }
                    }),
            )
            .unzip();
        let per_row_tidx = preflight
            .proof_shape
            .starting_tidx
            .iter()
            .copied()
            .chain(once(preflight.proof_shape.post_tidx))
            .collect_vec();

        let mut pvs_tidx_by_air_id = vec![0usize; vk.inner.per_air.len()];
        for ((air_idx, pvs), &starting_tidx) in proof
            .public_values
            .iter()
            .enumerate()
            .filter(|(_, per_air)| !per_air.is_empty())
            .zip(&preflight.proof_shape.pvs_tidx)
        {
            debug_assert!(!pvs.is_empty());
            pvs_tidx_by_air_id[air_idx] = starting_tidx;
        }

        ProofShapePreflightGpu {
            sorted_trace_heights: to_device_or_nullptr(&sorted_trace_heights).unwrap(),
            sorted_trace_metadata: to_device_or_nullptr(&sorted_trace_metadata).unwrap(),
            sorted_cached_commits: to_device_or_nullptr(&sorted_cached_commits).unwrap(),
            per_row_tidx: to_device_or_nullptr(&per_row_tidx).unwrap(),
            pvs_tidx: to_device_or_nullptr(&pvs_tidx_by_air_id).unwrap(),
            post_tidx: preflight.proof_shape.post_tidx,
            num_present: preflight.proof_shape.sorted_trace_vdata.len(),
            n_max: preflight.proof_shape.n_max,
            n_logup: preflight.proof_shape.n_logup,
            final_cidx: cidx,
            final_total_interactions: total_interactions,
            main_commit: proof.common_main_commit,
        }
    }

    fn gkr(_preflight: &Preflight) -> GkrPreflightGpu {
        GkrPreflightGpu { _dummy: 0 }
    }

    fn batch_constraint(preflight: &Preflight) -> BatchConstraintPreflightGpu {
        BatchConstraintPreflightGpu {
            sumcheck_rnd: to_device_or_nullptr(&preflight.batch_constraint.sumcheck_rnd).unwrap(),
        }
    }

    fn stacking(preflight: &Preflight) -> StackingPreflightGpu {
        StackingPreflightGpu {
            sumcheck_rnd: to_device_or_nullptr(&preflight.stacking.sumcheck_rnd).unwrap(),
        }
    }

    fn whir(_preflight: &Preflight) -> WhirPreflightGpu {
        WhirPreflightGpu { _dummy: 0 }
    }
}
