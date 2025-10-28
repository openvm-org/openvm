use itertools::Itertools;
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
use stark_backend_v2::{Digest, proof::Proof};

use crate::cuda::types::{PublicValueData, TraceMetadata};

#[derive(Debug)]
pub struct GpuProof {
    pub common_main_commit: Digest,
    pub cached_commits: DeviceBuffer<Digest>,
    pub trace_vdata: DeviceBuffer<TraceMetadata>,
    pub public_values: DeviceBuffer<PublicValueData>,

    pub gkr_proof: GkrProof,
    pub batch_constraint_proof: BatchConstraintProof,
    pub stacking_proof: StackingProof,
    pub whir_proof: WhirProof,
}

#[derive(Debug)]
pub struct GkrProof {
    _dummy: usize,
}

#[derive(Debug)]
pub struct BatchConstraintProof {
    _dummy: usize,
}

#[derive(Debug)]
pub struct StackingProof {
    _dummy: usize,
}

#[derive(Debug)]
pub struct WhirProof {
    _dummy: usize,
}

impl GpuProof {
    pub fn new(proof: &Proof) -> Self {
        let (cached_commits, trace_vdata) = Self::trace_vdata_and_cached_commits(proof);
        GpuProof {
            common_main_commit: proof.common_main_commit,
            cached_commits,
            trace_vdata,
            public_values: Self::public_values(proof),
            gkr_proof: Self::gkr_proof(proof),
            batch_constraint_proof: Self::batch_constraint_proof(proof),
            stacking_proof: Self::stacking_proof(proof),
            whir_proof: Self::whir_proof(proof),
        }
    }

    fn trace_vdata_and_cached_commits(
        proof: &Proof,
    ) -> (DeviceBuffer<Digest>, DeviceBuffer<TraceMetadata>) {
        let mut cached_commits: Vec<Digest> = vec![];
        let trace_vdata = proof
            .trace_vdata
            .iter()
            .map(|vdata| {
                if let Some(vdata) = vdata {
                    let ret = TraceMetadata {
                        hypercube_dim: vdata.hypercube_dim,
                        is_present: true,
                        num_cached: vdata.cached_commitments.len(),
                        cached_idx: cached_commits.len(),
                    };
                    cached_commits.extend_from_slice(&vdata.cached_commitments);
                    ret
                } else {
                    TraceMetadata {
                        hypercube_dim: 0,
                        is_present: false,
                        num_cached: 0,
                        cached_idx: cached_commits.len(),
                    }
                }
            })
            .collect_vec()
            .to_device()
            .unwrap();
        (cached_commits.to_device().unwrap(), trace_vdata)
    }

    fn public_values(proof: &Proof) -> DeviceBuffer<PublicValueData> {
        proof
            .public_values
            .iter()
            .enumerate()
            .flat_map(|(air_idx, pvs)| {
                pvs.iter()
                    .enumerate()
                    .map(move |(pv_idx, &value)| PublicValueData {
                        air_idx,
                        pv_idx,
                        value,
                    })
            })
            .collect_vec()
            .to_device()
            .unwrap()
    }

    fn gkr_proof(_proof: &Proof) -> GkrProof {
        GkrProof { _dummy: 0 }
    }

    fn batch_constraint_proof(_proof: &Proof) -> BatchConstraintProof {
        BatchConstraintProof { _dummy: 0 }
    }

    fn stacking_proof(_proof: &Proof) -> StackingProof {
        StackingProof { _dummy: 0 }
    }

    fn whir_proof(_proof: &Proof) -> WhirProof {
        WhirProof { _dummy: 0 }
    }
}
