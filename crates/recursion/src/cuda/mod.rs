use openvm_cuda_common::{
    common::get_device,
    copy::MemCopyH2D,
    d_buffer::DeviceBuffer,
    error::MemCopyError,
    stream::{CudaStream, DeviceContext, StreamGuard},
};

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

pub fn temp_device_ctx() -> DeviceContext {
    DeviceContext {
        device_id: get_device().unwrap() as u32,
        stream: StreamGuard::new(CudaStream::new_non_blocking().unwrap()),
    }
}

pub fn to_device_or_nullptr_on<T>(
    h2d: &[T],
    ctx: &DeviceContext,
) -> Result<DeviceBuffer<T>, MemCopyError> {
    if h2d.is_empty() {
        Ok(DeviceBuffer::new())
    } else {
        h2d.to_device_on(ctx)
    }
}
