use itertools::Itertools;
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
use openvm_stark_backend::{keygen::types::MultiStarkVerifyingKey, SystemParams};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, Digest};

use crate::cuda::types::AirData;

/*
 * Tracegen information (i.e. records) on a GPU device. Each field should
 * be computable as soon as the verifier circuit has access to the child
 * proof's verifying key. Only one of these will be generated per verifier
 * circuit, i.e. regardless of the number of proofs.
 */
pub struct VerifyingKeyGpu {
    // TODO[TEMP]: cpu vk for hybrid usage; remove this when no longer needed
    // If you need something from `cpu` for actual cuda tracegen, move it to a direct field of
    // VerifyingKeyGpu. Host and/or device types allowed.
    pub cpu: MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
    pub per_air: DeviceBuffer<AirData>,
    pub system_params: SystemParams,
    pub pre_hash: Digest,
}

impl VerifyingKeyGpu {
    pub fn new(vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>) -> Self {
        let per_air = vk
            .inner
            .per_air
            .iter()
            .map(|vk| AirData {
                num_cached: vk.num_cached_mains(),
                num_interactions_per_row: vk.num_interactions(),
                total_width: vk.params.width.total_width(0),
                has_preprocessed: vk.preprocessed_data.is_some(),
                need_rot: vk.params.need_rot,
            })
            .collect_vec()
            .to_device()
            .unwrap();
        Self {
            cpu: vk.clone(),
            per_air,
            system_params: vk.inner.params.clone(),
            pre_hash: vk.pre_hash,
        }
    }
}
