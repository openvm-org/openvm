use itertools::Itertools;
use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
use stark_backend_v2::keygen::types::MultiStarkVerifyingKey0V2;

use crate::cuda::types::AirData;

pub struct GpuVerifyingKey {
    pub per_air: DeviceBuffer<AirData>,
}

impl GpuVerifyingKey {
    pub fn new(mvk: &MultiStarkVerifyingKey0V2) -> Self {
        let per_air = mvk
            .per_air
            .iter()
            .map(|vk| AirData {
                num_interactions: vk.num_interactions(),
                has_preprocessed: vk.preprocessed_data.is_some(),
            })
            .collect_vec()
            .to_device()
            .unwrap();
        Self { per_air }
    }
}
