use std::{any::Any, ffi::c_void};

use openvm_cuda_common::{copy::MemCopyH2D, d_buffer::DeviceBuffer, stream::GpuDeviceCtx};

/// CUDA-event timer for the env-gated G2 production-gate stage breakdown.
pub struct CudaStageTimer {
    start: *mut c_void,
    stop: *mut c_void,
    stream: *mut c_void,
}

unsafe extern "C" {
    fn cudaEventCreate(event: *mut *mut c_void) -> i32;
    fn cudaEventDestroy(event: *mut c_void) -> i32;
    fn cudaEventRecord(event: *mut c_void, stream: *mut c_void) -> i32;
    fn cudaEventSynchronize(event: *mut c_void) -> i32;
    fn cudaEventElapsedTime(ms: *mut f32, start: *mut c_void, stop: *mut c_void) -> i32;
}

impl CudaStageTimer {
    pub fn start(device_ctx: &GpuDeviceCtx) -> Option<Self> {
        if std::env::var("OPENVM_RVR_G2_GPU_PROFILE").as_deref() != Ok("1") {
            return None;
        }
        let mut start = std::ptr::null_mut();
        let mut stop = std::ptr::null_mut();
        unsafe {
            assert_eq!(
                cudaEventCreate(&mut start),
                0,
                "CUDA profile start-event create"
            );
            assert_eq!(
                cudaEventCreate(&mut stop),
                0,
                "CUDA profile stop-event create"
            );
            assert_eq!(
                cudaEventRecord(start, device_ctx.stream.as_raw().cast()),
                0,
                "CUDA profile start-event record"
            );
        }
        Some(Self {
            start,
            stop,
            stream: device_ctx.stream.as_raw().cast(),
        })
    }

    /// Start on a generic prover-device context when it is the CUDA backend.
    /// Generic VM orchestration cannot name the backend's associated context
    /// type in its impl bounds, so the profiling-only path resolves it here.
    pub fn start_from_device_ctx<T: Any>(device_ctx: &T) -> Option<Self> {
        (device_ctx as &dyn Any)
            .downcast_ref::<GpuDeviceCtx>()
            .and_then(Self::start)
    }

    pub fn finish(self, stage: &str, segment_id: u32, bytes: usize) -> f32 {
        let mut elapsed_ms = 0.0f32;
        unsafe {
            assert_eq!(
                cudaEventRecord(self.stop, self.stream),
                0,
                "CUDA profile stop-event record"
            );
            assert_eq!(
                cudaEventSynchronize(self.stop),
                0,
                "CUDA profile stop-event synchronize"
            );
            assert_eq!(
                cudaEventElapsedTime(&mut elapsed_ms, self.start, self.stop),
                0,
                "CUDA profile elapsed-time query"
            );
        }
        eprintln!(
            "OPENVM_RVR_G2_GPU_STAGE segment={segment_id} stage={stage} ms={elapsed_ms:.6} bytes={bytes}"
        );
        elapsed_ms
    }
}

/// Upload one opaque-final arena and attribute its bytes to the owning G2
/// segment. With profiling disabled this is the ordinary asynchronous H2D.
pub fn opaque_h2d<T>(
    records: &[T],
    segment_id: Option<u32>,
    device_ctx: &GpuDeviceCtx,
) -> DeviceBuffer<T> {
    let timer = segment_id.and_then(|_| CudaStageTimer::start(device_ctx));
    let device = records.to_device_on(device_ctx).unwrap();
    if let (Some(timer), Some(segment_id)) = (timer, segment_id) {
        timer.finish("opaque_h2d", segment_id, std::mem::size_of_val(records));
    }
    device
}

impl Drop for CudaStageTimer {
    fn drop(&mut self) {
        unsafe {
            let _ = cudaEventDestroy(self.stop);
            let _ = cudaEventDestroy(self.start);
        }
    }
}
