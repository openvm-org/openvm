//! Page-locked (pinned) buffer pool for
//! [`DenseRecordArena`](crate::arch::record_arena::DenseRecordArena).
//!
//! Record arenas are allocated fresh for every chip in every segment and their
//! contents are copied host-to-device at trace generation time. Copies from
//! pageable memory run at a fraction of PCIe bandwidth, but page-locking a
//! buffer is itself expensive (~1-2 GB/s), and arenas are provisioned at full
//! trace size while typically only partially written, so neither registration
//! nor re-zeroing may sit on the preflight critical path. Dropped buffers are
//! therefore handed to a background cleaner thread which registers them once
//! (`cudaHostRegister`), zeroes the prefix the previous owner wrote, and only
//! then returns them to the pool. [`take`] hands out ready (registered,
//! all-zero) buffers on a pool hit; on a miss it falls back to a fresh
//! pageable allocation — exactly the pre-pool behavior — so the worst case (no
//! CUDA device, cleaner not yet caught up) matches the status quo. Capacities
//! are rounded up to the next power of two so recurring per-chip arenas of
//! varying heights share pool entries.
//!
//! Lifetime hazard: `cudaMemcpyAsync` from *pageable* memory returns only
//! after the source has been staged, so the pre-pool code could free an arena
//! right after enqueueing its copy. From *pinned* memory the call returns
//! immediately with the DMA still in flight, so a returned buffer must not be
//! zeroed or reused until previously enqueued work has drained. The cleaner
//! therefore calls `cudaDeviceSynchronize` (batched over every buffer waiting
//! in its queue) before touching buffer contents.

use std::{
    collections::BTreeMap,
    ffi::c_void,
    sync::{mpsc, Mutex, OnceLock},
};

use crate::arch::pending_return::PendingReturn;

/// A dropped arena buffer together with its dirty-prefix length, quarantined
/// until the cleaner has synchronized the CUDA device.
type ReturnedBuffer = PendingReturn<(Vec<u8>, usize)>;

/// Page-locks `len` bytes at `ptr` in a single `cudaHostRegister` call.
/// NOTE: registration must be one call per buffer: `cudaMemcpyAsync` rejects
/// (cudaErrorInvalidValue) source ranges that span multiple distinct
/// page-locked registrations, so chunked registration corrupts nothing but
/// breaks every copy crossing a chunk boundary.
pub(crate) fn register_region(ptr: *mut u8, len: usize) -> bool {
    // SAFETY: [ptr, ptr+len) is a live allocation owned by the caller.
    let rc = unsafe { cudaHostRegister(ptr as *mut c_void, len, 0) };
    if rc != 0 {
        tracing::debug!("cudaHostRegister failed with {rc}; record arena buffer stays pageable");
        return false;
    }
    true
}

extern "C" {
    fn cudaHostRegister(ptr: *mut c_void, size: usize, flags: u32) -> i32;
    fn cudaHostUnregister(ptr: *mut c_void) -> i32;
    fn cudaDeviceSynchronize() -> i32;
}

/// Reverses a successful [`register_region`]. The caller must ensure no copy
/// from the region is still in flight.
pub(crate) fn unregister_region(ptr: *mut u8) {
    // SAFETY: mirrors a successful registration of the same base pointer.
    unsafe { cudaHostUnregister(ptr as *mut c_void) };
}

/// Registered, all-zero buffers ready for reuse, keyed by allocation size.
fn pool() -> &'static Mutex<BTreeMap<usize, Vec<Vec<u8>>>> {
    static POOL: OnceLock<Mutex<BTreeMap<usize, Vec<Vec<u8>>>>> = OnceLock::new();
    POOL.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// Base pointers of buffers whose `cudaHostRegister` succeeded.
fn registered() -> &'static Mutex<std::collections::HashSet<usize>> {
    static REGISTERED: OnceLock<Mutex<std::collections::HashSet<usize>>> = OnceLock::new();
    REGISTERED.get_or_init(|| Mutex::new(std::collections::HashSet::new()))
}

/// Cleaner thread: registers (first cycle) and re-zeroes buffers off the
/// critical path, then makes them available to [`take`].
fn cleaner() -> &'static Mutex<mpsc::Sender<ReturnedBuffer>> {
    static TX: OnceLock<Mutex<mpsc::Sender<ReturnedBuffer>>> = OnceLock::new();
    TX.get_or_init(|| {
        let (tx, rx) = mpsc::channel::<ReturnedBuffer>();
        std::thread::Builder::new()
            .name("record-arena-pinner".into())
            .spawn(move || {
                let mut batch_idx = 0usize;
                while let Ok(first) = rx.recv() {
                    // Coalesce the per-segment burst of arena drops behind
                    // one device sync; repeated device-wide syncs stall
                    // concurrent kernel launches. Buffers are not needed
                    // again before the next segment's preflight, so a
                    // short idle window costs nothing.
                    let mut batch = vec![first];
                    while batch.len() < 64 {
                        match rx.recv_timeout(std::time::Duration::from_millis(100)) {
                            Ok(next) => batch.push(next),
                            Err(_) => break,
                        }
                    }
                    // The H2D copies reading these buffers were enqueued
                    // before the owning arenas dropped; wait for them (and
                    // anything else in flight) before touching contents.
                    // Unique label per batch: the timing metric derived from
                    // this span is a gauge, so identical label sets overwrite.
                    let _span =
                        tracing::info_span!("arena_cleaner_batch", batch = batch_idx.to_string())
                            .entered();
                    batch_idx += 1;
                    let rc = unsafe { cudaDeviceSynchronize() };
                    if rc != 0 {
                        // No usable CUDA context (teardown or no device):
                        // the buffers cannot be proven idle. Dropping the
                        // wrappers quarantines (leaks) their allocations
                        // instead of risking a free while CUDA still owns a
                        // registration or asynchronous read.
                        tracing::debug!(
                            "cudaDeviceSynchronize failed with {rc}; \
                             quarantining {} record arena buffers",
                            batch.len()
                        );
                        continue;
                    }
                    for returned in batch {
                        let (mut buffer, dirty_len) = returned.release();
                        if buffer.is_empty() || !buffer.len().is_power_of_two() {
                            continue; // synchronized but not pool-shaped
                        }
                        let ptr = buffer.as_mut_ptr();
                        let is_new = !registered().lock().unwrap().contains(&(ptr as usize));
                        if is_new {
                            if !register_region(ptr, buffer.len()) {
                                // Out of pinnable memory: drop the buffer,
                                // never pool it.
                                continue;
                            }
                            registered().lock().unwrap().insert(ptr as usize);
                        }
                        // Restore the fresh-arena invariant (all zero). Bytes
                        // past the dirty prefix were never written or were
                        // cleared on an earlier cycle.
                        let dirty_len = dirty_len.min(buffer.len());
                        buffer[..dirty_len].fill(0);
                        pool()
                            .lock()
                            .unwrap()
                            .entry(buffer.len())
                            .or_default()
                            .push(buffer);
                    }
                }
            })
            .expect("failed to spawn record-arena pinner thread");
        Mutex::new(tx)
    })
}

/// Returns a ready buffer and whether its freshly allocated pages still need
/// to be faulted in by a latency-sensitive caller. Pool hits are registered,
/// resident, and already zeroed by the cleaner.
pub(crate) fn take_with_prefault_status(min_size: usize) -> (Vec<u8>, bool) {
    let size = min_size.next_power_of_two();
    if let Some(buffer) = pool()
        .lock()
        .unwrap()
        .get_mut(&size)
        .and_then(|bufs| bufs.pop())
    {
        debug_assert_eq!(buffer.len(), size);
        return (buffer, false);
    }
    // Pool miss: pageable memory, zeroed lazily by the kernel, exactly as
    // without the pool. The buffer becomes pinned when first given back.
    (vec![0u8; size], true)
}

pub(crate) fn take(min_size: usize) -> Vec<u8> {
    take_with_prefault_status(min_size).0
}

/// `dirty_len` is an upper bound on the prefix of `buffer` that may have
/// been written since it left [`take`]; the rest must still be zero.
pub(crate) fn give_back(buffer: Vec<u8>, dirty_len: usize) {
    if buffer.is_empty() {
        return;
    }
    // If the send races process teardown, `PendingReturn` deliberately leaks
    // the backing: without the cleaner's sync, freeing it is not safe.
    let returned = PendingReturn::new((buffer, dirty_len));
    let _ = cleaner().lock().unwrap().send(returned);
}

/// Unregisters and frees all pooled buffers (test hygiene; optional).
#[allow(dead_code)]
pub(crate) fn clear() {
    let mut pool = pool().lock().unwrap();
    let mut reg = registered().lock().unwrap();
    for (_, bufs) in pool.iter_mut() {
        for mut buf in bufs.drain(..) {
            reg.remove(&(buf.as_ptr() as usize));
            unsafe { cudaHostUnregister(buf.as_mut_ptr() as *mut c_void) };
        }
    }
    pool.clear();
}
