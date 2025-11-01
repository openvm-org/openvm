use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer, error::MemCopyError};

pub mod preflight;
pub mod proof;
pub mod types;
pub mod vk;

fn to_device_or_nullptr<T>(h2d: &[T]) -> Result<DeviceBuffer<T>, MemCopyError> {
    if h2d.is_empty() {
        Ok(DeviceBuffer::new())
    } else {
        h2d.to_device()
    }
}
