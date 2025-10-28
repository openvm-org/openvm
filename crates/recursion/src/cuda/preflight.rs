use itertools::Itertools;
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
use stark_backend_v2::Digest;

use crate::{cuda::types::TraceMetadata, system::Preflight};

#[derive(Debug)]
pub struct GpuPreflight {
    pub transcript: TranscriptLog,
    pub proof_shape: ProofShapePreflight,
    pub gkr: GkrPreflight,
    pub batch_constraint: BatchConstraintPreflight,
    pub stacking: StackingPreflight,
    pub whir: WhirPreflight,
}

#[derive(Debug)]
pub struct TranscriptLog {
    _dummy: usize,
}

#[derive(Debug)]
pub struct ProofShapePreflight {
    pub sorted_trace_vdata: DeviceBuffer<(usize, TraceMetadata)>,
    pub sorted_cached_commits: DeviceBuffer<Digest>,
    pub pvs_tidx: DeviceBuffer<usize>,
    pub post_tidx: usize,
    pub n_max: usize,
    pub n_logup: usize,
    pub l_skip: usize,
}

#[derive(Debug)]
pub struct GkrPreflight {
    _dummy: usize,
}

#[derive(Debug)]
pub struct BatchConstraintPreflight {
    _dummy: usize,
}

#[derive(Debug)]
pub struct StackingPreflight {
    _dummy: usize,
}

#[derive(Debug)]
pub struct WhirPreflight {
    _dummy: usize,
}

impl GpuPreflight {
    pub fn new(preflight: &Preflight) -> Self {
        GpuPreflight {
            transcript: Self::transcript(preflight),
            proof_shape: Self::proof_shape(preflight),
            gkr: Self::gkr(preflight),
            batch_constraint: Self::batch_constraint(preflight),
            stacking: Self::stacking(preflight),
            whir: Self::whir(preflight),
        }
    }

    fn transcript(_preflight: &Preflight) -> TranscriptLog {
        TranscriptLog { _dummy: 0 }
    }

    fn proof_shape(preflight: &Preflight) -> ProofShapePreflight {
        // TODO[stephen]: we should derive this using the GPU vdata in GpuProof
        let mut sorted_cached_commits: Vec<Digest> = vec![];
        let sorted_trace_vdata = preflight
            .proof_shape
            .sorted_trace_vdata
            .iter()
            .map(|(air_idx, vdata)| {
                let ret = TraceMetadata {
                    hypercube_dim: vdata.hypercube_dim,
                    is_present: true,
                    num_cached: vdata.cached_commitments.len(),
                    cached_idx: sorted_cached_commits.len(),
                };
                sorted_cached_commits.extend_from_slice(&vdata.cached_commitments);
                (*air_idx, ret)
            })
            .collect_vec();
        ProofShapePreflight {
            sorted_trace_vdata: sorted_trace_vdata.to_device().unwrap(),
            sorted_cached_commits: sorted_cached_commits.to_device().unwrap(),
            pvs_tidx: preflight.proof_shape.pvs_tidx.to_device().unwrap(),
            post_tidx: preflight.proof_shape.post_tidx,
            n_max: preflight.proof_shape.n_max,
            n_logup: preflight.proof_shape.n_logup,
            l_skip: preflight.proof_shape.l_skip,
        }
    }

    fn gkr(_preflight: &Preflight) -> GkrPreflight {
        GkrPreflight { _dummy: 0 }
    }

    fn batch_constraint(_preflight: &Preflight) -> BatchConstraintPreflight {
        BatchConstraintPreflight { _dummy: 0 }
    }

    fn stacking(_preflight: &Preflight) -> StackingPreflight {
        StackingPreflight { _dummy: 0 }
    }

    fn whir(_preflight: &Preflight) -> WhirPreflight {
        WhirPreflight { _dummy: 0 }
    }
}
