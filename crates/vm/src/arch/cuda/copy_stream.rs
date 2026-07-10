//! Dedicated host-to-device copy stream for record uploads.
//!
//! Record uploads and tracegen kernels have no cross-chip dependencies, but on
//! a single stream they serialize: ~36 GB of records per reth block costs
//! ~0.7 s of DMA that the stream cannot overlap with its own kernels. PCIe
//! copy engines run concurrently with SM compute, so routing the uploads
//! through a dedicated stream turns copy+kernel time from a sum into a max.
//!
//! Ordering with the VPMM allocator: allocations are stream-tagged, and a
//! region freed on a stream may be reallocated to the same stream without
//! synchronization (stream order makes that safe). A copy-stream write into a
//! kernel-stream allocation therefore needs a two-event fence:
//!
//! 1. an event recorded on the kernel stream after allocation, waited by the copy stream — orders
//!    the write after any prior kernel-stream work that used the reallocated region;
//! 2. an event recorded on the copy stream after the copy, waited by the kernel stream — orders
//!    every later kernel-stream read (and any future reallocation of this region) after the write.
//!
//! Host-side lifetime is unchanged: pinned sources return through the arena
//! cleaner, which synchronizes the device (all streams) before reuse, and
//! pageable sources are consumed by the staging pipeline before the copy call
//! returns.

use std::sync::OnceLock;

use openvm_cuda_common::{
    copy::cuda_memcpy_on,
    d_buffer::DeviceBuffer,
    error::MemCopyError,
    stream::{CudaEvent, GpuDeviceCtx},
};

fn copy_ctx() -> Option<&'static GpuDeviceCtx> {
    static CTX: OnceLock<Option<GpuDeviceCtx>> = OnceLock::new();
    CTX.get_or_init(|| match GpuDeviceCtx::for_current_device() {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            tracing::debug!("no copy stream ({e:?}); record uploads use the kernel stream");
            None
        }
    })
    .as_ref()
}

/// Uploads `records` to a device buffer allocated on `kernel_ctx`, running the
/// copy on the shared copy stream so it overlaps kernels already enqueued on
/// `kernel_ctx`. All subsequent work on `kernel_ctx` is ordered after the
/// copy. Falls back to a plain same-stream copy if the copy stream is
/// unavailable.
pub fn records_to_device(
    records: &[u8],
    kernel_ctx: &GpuDeviceCtx,
) -> Result<DeviceBuffer<u8>, MemCopyError> {
    let dst = DeviceBuffer::<u8>::with_capacity_on(records.len(), kernel_ctx);
    let copy = match copy_ctx() {
        Some(c) => c,
        None => {
            // SAFETY: `dst` holds `records.len()` bytes; same-stream copy.
            unsafe {
                cuda_memcpy_on::<false, true>(
                    dst.as_mut_raw_ptr(),
                    records.as_ptr() as *const std::ffi::c_void,
                    records.len(),
                    kernel_ctx,
                )?;
            }
            return Ok(dst);
        }
    };
    let alloc_done = CudaEvent::new().map_err(MemCopyError::from)?;
    alloc_done
        .record_on(&kernel_ctx.stream)
        .map_err(MemCopyError::from)?;
    copy.stream.wait(&alloc_done).map_err(MemCopyError::from)?;
    // SAFETY: `dst` holds `records.len()` bytes; the copy stream is ordered
    // after prior kernel-stream users of the region (alloc_done) and before
    // later ones (copy_done).
    unsafe {
        cuda_memcpy_on::<false, true>(
            dst.as_mut_raw_ptr(),
            records.as_ptr() as *const std::ffi::c_void,
            records.len(),
            copy,
        )?;
    }
    let copy_done = CudaEvent::new().map_err(MemCopyError::from)?;
    copy_done
        .record_on(&copy.stream)
        .map_err(MemCopyError::from)?;
    kernel_ctx
        .stream
        .wait(&copy_done)
        .map_err(MemCopyError::from)?;
    Ok(dst)
}
