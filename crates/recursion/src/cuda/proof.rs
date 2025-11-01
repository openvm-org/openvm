use itertools::Itertools;
use openvm_cuda_common::d_buffer::DeviceBuffer;
use stark_backend_v2::{keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};

use crate::cuda::{to_device_or_nullptr, types::PublicValueData};

/*
 * Tracegen information (i.e. records) on a GPU device. Each field should
 * be computable as soon as the verifier circuit has access to the child
 * proof and verifying key.
 */
#[derive(Debug)]
pub struct ProofGpu {
    // TODO[TEMP]: cpu proof for hybrid usage; remove this when no longer needed
    pub cpu: Proof,
    pub proof_shape: ProofShapeProofGpu,
    pub gkr: GkrProofGpu,
    pub batch_constraint: BatchConstraintProofGpu,
    pub stacking: StackingProofGpu,
    pub whir: WhirProofGpu,
}

#[derive(Debug)]
pub struct ProofShapeProofGpu {
    pub public_values: DeviceBuffer<PublicValueData>,
}

#[derive(Debug)]
pub struct GkrProofGpu {
    _dummy: usize,
}

#[derive(Debug)]
pub struct BatchConstraintProofGpu {
    _dummy: usize,
}

#[derive(Debug)]
pub struct StackingProofGpu {
    _dummy: usize,
}

#[derive(Debug)]
pub struct WhirProofGpu {
    _dummy: usize,
}

impl ProofGpu {
    pub fn new(vk: &MultiStarkVerifyingKeyV2, proof: &Proof) -> Self {
        ProofGpu {
            cpu: proof.clone(),
            proof_shape: Self::proof_shape(vk, proof),
            gkr: Self::gkr(proof),
            batch_constraint: Self::batch_constraint(proof),
            stacking: Self::stacking(proof),
            whir: Self::whir(proof),
        }
    }

    fn proof_shape(_vk: &MultiStarkVerifyingKeyV2, proof: &Proof) -> ProofShapeProofGpu {
        let public_values = proof
            .public_values
            .iter()
            .enumerate()
            .flat_map(|(air_idx, pvs)| {
                let air_num_pvs = pvs.len();
                pvs.iter()
                    .enumerate()
                    .map(move |(pv_idx, &value)| PublicValueData {
                        air_idx,
                        air_num_pvs,
                        pv_idx,
                        value,
                    })
            })
            .collect_vec();
        ProofShapeProofGpu {
            public_values: to_device_or_nullptr(&public_values).unwrap(),
        }
    }

    fn gkr(_proof: &Proof) -> GkrProofGpu {
        GkrProofGpu { _dummy: 0 }
    }

    fn batch_constraint(_proof: &Proof) -> BatchConstraintProofGpu {
        BatchConstraintProofGpu { _dummy: 0 }
    }

    fn stacking(_proof: &Proof) -> StackingProofGpu {
        StackingProofGpu { _dummy: 0 }
    }

    fn whir(_proof: &Proof) -> WhirProofGpu {
        WhirProofGpu { _dummy: 0 }
    }
}
