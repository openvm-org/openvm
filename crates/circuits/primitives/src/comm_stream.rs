//! Copy/compute overlap for host-to-device uploads during trace generation.
//!
//! Chips generate traces one after another, each uploading its record arena and
//! then launching a tracegen kernel. With both on one stream, the copy engine
//! and the SMs take turns: chip B's upload cannot start until chip A's kernel
//! finishes. This module routes uploads through a dedicated per-device *comm*
//! stream so that uploads pipeline ahead of compute:
//!
//! ```text
//! comm:  [copy A][copy B][copy C]
//!              ↘        ↘        ↘   (copy-complete events)
//! main:    [kernel A]  [kernel B]  [kernel C]
//! ```
//!
//! [`MemCopyH2DOverlapped::to_device_overlapped_on`] allocates the destination
//! and enqueues the copy on the comm stream, then makes the caller's main
//! stream wait device-side (`cudaStreamWaitEvent`) on a copy-complete event, so
//! everything subsequently enqueued on the main stream — in particular the
//! kernel consuming the buffer — is correctly ordered. The host never blocks.
//!
//! Freeing needs the reverse ordering: the memory pool recycles a freed region
//! in *allocating-stream* order, and the comm stream knows nothing about the
//! main-stream kernel still reading the buffer. Enqueuing a comm-stream wait at
//! drop time would re-serialize later uploads behind that kernel, so instead
//! [`CommDeviceBuffer`]'s `Drop` records a release event on the main stream and
//! hands the buffer to a reaper thread, which frees it back to the pool only
//! once the event — and therefore the consuming kernel — has completed.
//!
//! Host-side source lifetime: pinned-pool record arenas are safe because the
//! pool's cleaner synchronizes the device before recycling a returned buffer;
//! plain (pageable) host memory is safe because `cudaMemcpyAsync` does not
//! return until a pageable source has been staged.

use std::{
    collections::HashMap,
    mem::ManuallyDrop,
    ops::Deref,
    sync::{mpsc, Mutex, OnceLock},
};

use openvm_cuda_common::{
    copy::MemCopyH2D,
    d_buffer::DeviceBuffer,
    error::MemCopyError,
    stream::{CudaEvent, GpuDeviceCtx, StreamGuard},
};

/// One comm (copy) stream per device, created lazily on first upload.
fn comm_ctxs() -> &'static Mutex<HashMap<u32, GpuDeviceCtx>> {
    static CTXS: OnceLock<Mutex<HashMap<u32, GpuDeviceCtx>>> = OnceLock::new();
    CTXS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn comm_ctx_for(main_ctx: &GpuDeviceCtx) -> Result<GpuDeviceCtx, MemCopyError> {
    let mut ctxs = comm_ctxs().lock().unwrap();
    if let Some(ctx) = ctxs.get(&main_ctx.device_id) {
        return Ok(ctx.clone());
    }
    // `for_device` sets the calling thread's current device, which is a no-op
    // here: the caller is already issuing work on `main_ctx`'s device.
    let ctx = GpuDeviceCtx::for_device(main_ctx.device_id)?;
    ctxs.insert(main_ctx.device_id, ctx.clone());
    Ok(ctx)
}

/// A buffer and the main-stream event that must complete before it may be
/// freed back to the pool.
type ReaperItem = (DeviceBuffer<u8>, CudaEvent);

/// Reaper thread: frees comm-stream buffers once their release event (recorded
/// on the consuming main stream at drop time) completes, off the critical path.
fn reaper() -> &'static Mutex<mpsc::Sender<ReaperItem>> {
    static TX: OnceLock<Mutex<mpsc::Sender<ReaperItem>>> = OnceLock::new();
    TX.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<ReaperItem>();
        std::thread::Builder::new()
            .name("comm-stream-reaper".into())
            .spawn(move || {
                while let Ok((buffer, release)) = rx.recv() {
                    if let Err(e) = release.synchronize() {
                        tracing::error!(
                            "comm-stream reaper: release event sync failed ({e:?}); \
                             leaking buffer of {} bytes",
                            buffer.len()
                        );
                        std::mem::forget(buffer);
                    }
                    // buffer drops here: the consuming kernel has completed, so
                    // the pool may recycle the region on any stream.
                }
            })
            .expect("failed to spawn comm-stream-reaper thread");
        Mutex::new(tx)
    })
}

/// Synchronous fallback used when event machinery fails: wait for all pending
/// main-stream work (including the buffer's consumer), then free inline.
fn sync_free(main_stream: &StreamGuard, buffer: DeviceBuffer<u8>) {
    if let Err(e) = main_stream.synchronize() {
        tracing::error!(
            "comm-stream: main stream sync failed during free ({e:?}); \
             leaking buffer of {} bytes",
            buffer.len()
        );
        std::mem::forget(buffer);
    }
}

/// Device buffer uploaded via the comm stream, safe to use on the main stream
/// it was created against (accesses are ordered by a device-side event wait).
///
/// Dropping does not free immediately: the free is deferred until every
/// main-stream operation enqueued before the drop — in particular the kernel
/// consuming the buffer — has completed. Drop it only after the consuming work
/// has been enqueued on the main stream.
pub struct CommDeviceBuffer<T> {
    buffer: ManuallyDrop<DeviceBuffer<T>>,
    main_stream: StreamGuard,
}

impl<T> Deref for CommDeviceBuffer<T> {
    type Target = DeviceBuffer<T>;

    fn deref(&self) -> &DeviceBuffer<T> {
        &self.buffer
    }
}

impl<T> Drop for CommDeviceBuffer<T> {
    fn drop(&mut self) {
        // SAFETY: drop runs exactly once and the buffer is never used again.
        let buffer = unsafe { ManuallyDrop::take(&mut self.buffer) }.as_buffer::<u8>();
        let release = CudaEvent::new().and_then(|event| {
            event.record_on(&self.main_stream)?;
            Ok(event)
        });
        match release {
            Ok(event) => {
                // The reaper thread lives for the process lifetime; send fails
                // only if it died, in which case free synchronously.
                if let Err(mpsc::SendError((buffer, _))) =
                    reaper().lock().unwrap().send((buffer, event))
                {
                    sync_free(&self.main_stream, buffer);
                }
            }
            Err(e) => {
                tracing::warn!(
                    "comm-stream: release event failed ({e:?}); falling back to stream sync"
                );
                sync_free(&self.main_stream, buffer);
            }
        }
    }
}

pub trait MemCopyH2DOverlapped<T> {
    /// Uploads to the device via the per-device comm stream, overlapping the
    /// copy with compute already enqueued on `main_ctx`'s stream. The returned
    /// buffer is safe to use in work subsequently enqueued on that stream.
    fn to_device_overlapped_on(
        &self,
        main_ctx: &GpuDeviceCtx,
    ) -> Result<CommDeviceBuffer<T>, MemCopyError>;
}

impl<T> MemCopyH2DOverlapped<T> for [T] {
    fn to_device_overlapped_on(
        &self,
        main_ctx: &GpuDeviceCtx,
    ) -> Result<CommDeviceBuffer<T>, MemCopyError> {
        let comm_ctx = comm_ctx_for(main_ctx)?;
        // Allocating on the comm stream keeps the pool's stream-ordered reuse
        // sound for the copy below: comm-freed regions (returned by the reaper
        // only once fully idle) are handed out in comm-stream order, and
        // regions freed on other streams are gated on their release events.
        let mut buffer = DeviceBuffer::with_capacity_on(self.len(), &comm_ctx);
        self.copy_to_on(&mut buffer, &comm_ctx)?;
        let copied = CudaEvent::new()?;
        copied.record_on(&comm_ctx.stream)?;
        main_ctx.stream.wait(&copied)?;
        Ok(CommDeviceBuffer {
            buffer: ManuallyDrop::new(buffer),
            main_stream: main_ctx.stream.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use openvm_cuda_common::copy::MemCopyD2H;

    use super::*;

    /// Round-trips buffers through the comm stream: the D2H readback is
    /// enqueued on the main stream, so it validates the copy-complete event
    /// ordering; each drop routes the buffer through the reaper.
    #[test]
    fn overlapped_upload_roundtrip() {
        let main_ctx = GpuDeviceCtx::for_current_device().unwrap();
        for i in 0..8u32 {
            let host: Vec<u32> = (i..(1 << 16) + i).collect();
            let d_buf = host.to_device_overlapped_on(&main_ctx).unwrap();
            let back = d_buf.to_host_on(&main_ctx).unwrap();
            assert_eq!(host, back);
        }
    }
}
