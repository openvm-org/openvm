use std::iter::once;

use itertools::Itertools;
use openvm_cuda_common::d_buffer::DeviceBuffer;
use stark_backend_v2::{Digest, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};

use crate::{
    cuda::{to_device_or_nullptr, types::TraceMetadata},
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
    pub sorted_trace_vdata: DeviceBuffer<TraceMetadata>,
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
    _dummy: usize,
}

#[derive(Debug)]
pub struct StackingPreflightGpu {
    _dummy: usize,
}

#[derive(Debug)]
pub struct WhirPreflightGpu {
    _dummy: usize,
}

impl PreflightGpu {
    pub fn new(vk: &MultiStarkVerifyingKeyV2, proof: &Proof, preflight: &Preflight) -> Self {
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
        vk: &MultiStarkVerifyingKeyV2,
        proof: &Proof,
        preflight: &Preflight,
    ) -> ProofShapePreflightGpu {
        let mut sorted_cached_commits: Vec<Digest> = vec![];
        let mut cidx = 1;
        let mut total_interactions = 0;
        let l_skip = vk.inner.params.l_skip;

        let bc_air_shape_lookups = compute_air_shape_lookup_counts(
            vk,
            &preflight.proof_shape.sorted_trace_vdata,
            l_skip,
            preflight.proof_shape.n_max,
        );

        let sorted_trace_vdata = preflight
            .proof_shape
            .sorted_trace_vdata
            .iter()
            .map(|(air_idx, vdata)| {
                let metadata = TraceMetadata {
                    air_idx: *air_idx,
                    log_height: vdata.log_height.try_into().unwrap(),
                    cached_idx: sorted_cached_commits.len(),
                    starting_cidx: cidx,
                    total_interactions,
                    num_air_id_lookups: bc_air_shape_lookups[vdata.log_height],
                };
                cidx += vdata.cached_commitments.len()
                    + vk.inner.per_air[*air_idx].preprocessed_data.is_some() as usize;
                total_interactions += (1 << vdata.log_height.max(l_skip))
                    * vk.inner.per_air[*air_idx].num_interactions();
                sorted_cached_commits.extend_from_slice(&vdata.cached_commitments);
                metadata
            })
            .chain(
                proof
                    .trace_vdata
                    .iter()
                    .enumerate()
                    .filter_map(|(air_idx, vdata)| {
                        if vdata.is_none() {
                            Some(TraceMetadata {
                                air_idx,
                                ..Default::default()
                            })
                        } else {
                            None
                        }
                    }),
            )
            .collect_vec();
        let per_row_tidx = preflight
            .proof_shape
            .starting_tidx
            .iter()
            .copied()
            .chain(once(preflight.proof_shape.post_tidx))
            .collect_vec();

        ProofShapePreflightGpu {
            sorted_trace_vdata: to_device_or_nullptr(&sorted_trace_vdata).unwrap(),
            sorted_cached_commits: to_device_or_nullptr(&sorted_cached_commits).unwrap(),
            per_row_tidx: to_device_or_nullptr(&per_row_tidx).unwrap(),
            pvs_tidx: to_device_or_nullptr(&preflight.proof_shape.pvs_tidx).unwrap(),
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

    fn batch_constraint(_preflight: &Preflight) -> BatchConstraintPreflightGpu {
        BatchConstraintPreflightGpu { _dummy: 0 }
    }

    fn stacking(_preflight: &Preflight) -> StackingPreflightGpu {
        StackingPreflightGpu { _dummy: 0 }
    }

    fn whir(_preflight: &Preflight) -> WhirPreflightGpu {
        WhirPreflightGpu { _dummy: 0 }
    }
}
