use itertools::Itertools;
use openvm_cuda_common::d_buffer::DeviceBuffer;
use openvm_stark_backend::{keygen::types::MultiStarkVerifyingKey, proof::Proof};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;

use crate::cuda::{to_device_or_nullptr, types::PublicValueData};

/*
 * Tracegen information (i.e. records) on a GPU device. Each field should
 * be computable as soon as the verifier circuit has access to the child
 * proof and verifying key.
 */
#[derive(Debug)]
pub struct ProofGpu {
    // TODO[TEMP]: cpu proof for hybrid usage; remove this when no longer needed
    // If you need something from `cpu` for actual cuda tracegen, move it to a direct field of
    // ProofGpu. Host and/or device types allowed.
    pub cpu: Proof<BabyBearPoseidon2Config>,
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
    pub fn new(
        vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        proof: &Proof<BabyBearPoseidon2Config>,
    ) -> Self {
        ProofGpu {
            cpu: proof.clone(),
            proof_shape: Self::proof_shape(vk, proof),
            gkr: Self::gkr(proof),
            batch_constraint: Self::batch_constraint(proof),
            stacking: Self::stacking(proof),
            whir: Self::whir(proof),
        }
    }

    fn proof_shape(
        _vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        proof: &Proof<BabyBearPoseidon2Config>,
    ) -> ProofShapeProofGpu {
        let num_airs = proof.public_values.len();
        let public_values = proof
            .public_values
            .iter()
            .enumerate()
            .flat_map(move |(air_idx, pvs)| {
                let air_num_pvs = pvs.len();
                let total_airs = num_airs;
                pvs.iter()
                    .enumerate()
                    .map(move |(pv_idx, &value)| PublicValueData {
                        air_idx,
                        air_num_pvs,
                        num_airs: total_airs,
                        pv_idx,
                        value,
                    })
            })
            .collect_vec();
        ProofShapeProofGpu {
            public_values: to_device_or_nullptr(&public_values).unwrap(),
        }
    }

    fn gkr(_proof: &Proof<BabyBearPoseidon2Config>) -> GkrProofGpu {
        GkrProofGpu { _dummy: 0 }
    }

    fn batch_constraint(_proof: &Proof<BabyBearPoseidon2Config>) -> BatchConstraintProofGpu {
        BatchConstraintProofGpu { _dummy: 0 }
    }

    fn stacking(_proof: &Proof<BabyBearPoseidon2Config>) -> StackingProofGpu {
        StackingProofGpu { _dummy: 0 }
    }

    fn whir(_proof: &Proof<BabyBearPoseidon2Config>) -> WhirProofGpu {
        WhirProofGpu { _dummy: 0 }
    }
}
