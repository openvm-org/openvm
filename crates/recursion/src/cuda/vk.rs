use itertools::Itertools;
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
use stark_backend_v2::{Digest, SystemParams, keygen::types::MultiStarkVerifyingKeyV2};

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
    pub cpu: MultiStarkVerifyingKeyV2,
    pub per_air: DeviceBuffer<AirData>,
    pub system_params: SystemParams,
    pub pre_hash: Digest,
}

impl VerifyingKeyGpu {
    pub fn new(vk: &MultiStarkVerifyingKeyV2) -> Self {
        let per_air = vk
            .inner
            .per_air
            .iter()
            .map(|vk| AirData {
                num_cached: vk.num_cached_mains(),
                num_interactions_per_row: vk.num_interactions(),
                has_preprocessed: vk.preprocessed_data.is_some(),
            })
            .collect_vec()
            .to_device()
            .unwrap();
        Self {
            cpu: vk.clone(),
            per_air,
            system_params: vk.inner.params,
            pre_hash: vk.pre_hash,
        }
    }
}
