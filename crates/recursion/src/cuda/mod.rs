use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer, error::MemCopyError};

use crate::{
    cuda::{preflight::PreflightGpu, proof::ProofGpu, vk::VerifyingKeyGpu},
    system::GlobalTraceGenCtx,
};

pub mod abi;
pub mod preflight;
pub mod proof;
pub mod types;
pub mod vk;

pub struct GlobalCtxGpu;

impl GlobalTraceGenCtx for GlobalCtxGpu {
    type ChildVerifyingKey = VerifyingKeyGpu;
    type MultiProof = [ProofGpu];
    type PreflightRecords = [PreflightGpu];
}

pub fn to_device_or_nullptr<T>(h2d: &[T]) -> Result<DeviceBuffer<T>, MemCopyError> {
    if h2d.is_empty() {
        Ok(DeviceBuffer::new())
    } else {
        h2d.to_device()
    }
}
