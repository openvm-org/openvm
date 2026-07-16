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
    collections::{BTreeMap, HashSet},
    ffi::c_void,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc, Mutex, MutexGuard, OnceLock,
    },
    thread::JoinHandle,
};

use crate::arch::pending_return::{
    quarantine as quarantine_pending, run_pending_return_worker, shutdown_pending_return_worker,
    PendingReturn, PendingReturnMessage,
};

/// A dropped arena buffer together with its dirty-prefix length, quarantined
/// until the cleaner has synchronized the CUDA device.
type ReturnedBuffer = PendingReturn<(Vec<u8>, usize)>;
type CleanerMessage = PendingReturnMessage<(Vec<u8>, usize)>;

#[derive(Default)]
struct PoolStats {
    hits: AtomicU64,
    misses: AtomicU64,
    #[cfg(feature = "rvr")]
    populate_calls: AtomicU64,
    #[cfg(feature = "rvr")]
    populate_bytes: AtomicU64,
    returns_enqueued: AtomicU64,
    returns_synchronized: AtomicU64,
    returns_pooled: AtomicU64,
    pending: AtomicU64,
    pending_peak: AtomicU64,
    quarantined: AtomicU64,
    sync_failures: AtomicU64,
    registration_failures: AtomicU64,
    zeroed_bytes: AtomicU64,
    zero_time_us: AtomicU64,
}

#[derive(Default)]
struct PoolState {
    stats: PoolStats,
    ready: Mutex<BTreeMap<usize, Vec<Vec<u8>>>>,
    registered: Mutex<HashSet<usize>>,
}

/// All state a cleaner thread may touch has process lifetime. In particular,
/// none of it participates in Rust or CUDA runtime destruction ordering.
fn state() -> &'static PoolState {
    static STATE: OnceLock<&'static PoolState> = OnceLock::new();
    STATE.get_or_init(|| Box::leak(Box::new(PoolState::default())))
}

fn stats() -> &'static PoolStats {
    &state().stats
}

/// Cheap cumulative counters used to correlate per-segment preflight latency with pinned-pool
/// availability. Snapshotting also samples the ready queues; callers do that only when the
/// diagnostic environment flag is enabled.
#[derive(Clone, Copy, Debug, Default)]
#[cfg(feature = "rvr")]
pub(crate) struct PoolStatsSnapshot {
    pub hits: u64,
    pub misses: u64,
    pub populate_calls: u64,
    pub populate_bytes: u64,
    pub returns_enqueued: u64,
    pub returns_synchronized: u64,
    pub returns_pooled: u64,
    pub pending: u64,
    pub pending_peak: u64,
    pub quarantined: u64,
    pub sync_failures: u64,
    pub registration_failures: u64,
    pub zeroed_bytes: u64,
    pub zero_time_us: u64,
    pub ready_buffers: u64,
    pub ready_bytes: u64,
}

#[cfg(feature = "rvr")]
impl PoolStatsSnapshot {
    pub(crate) fn capture() -> Self {
        let stats = stats();
        let ready = pool().lock().unwrap();
        let ready_buffers = ready.values().map(|buffers| buffers.len() as u64).sum();
        let ready_bytes = ready
            .iter()
            .map(|(&size, buffers)| size as u64 * buffers.len() as u64)
            .sum();
        Self {
            hits: stats.hits.load(Ordering::Relaxed),
            misses: stats.misses.load(Ordering::Relaxed),
            populate_calls: stats.populate_calls.load(Ordering::Relaxed),
            populate_bytes: stats.populate_bytes.load(Ordering::Relaxed),
            returns_enqueued: stats.returns_enqueued.load(Ordering::Relaxed),
            returns_synchronized: stats.returns_synchronized.load(Ordering::Relaxed),
            returns_pooled: stats.returns_pooled.load(Ordering::Relaxed),
            pending: stats.pending.load(Ordering::Relaxed),
            pending_peak: stats.pending_peak.load(Ordering::Relaxed),
            quarantined: stats.quarantined.load(Ordering::Relaxed),
            sync_failures: stats.sync_failures.load(Ordering::Relaxed),
            registration_failures: stats.registration_failures.load(Ordering::Relaxed),
            zeroed_bytes: stats.zeroed_bytes.load(Ordering::Relaxed),
            zero_time_us: stats.zero_time_us.load(Ordering::Relaxed),
            ready_buffers,
            ready_bytes,
        }
    }
}

#[cfg(feature = "rvr")]
pub(crate) fn stats_enabled() -> bool {
    std::env::var("OPENVM_RVR_CUDA_POOL_STATS").as_deref() == Ok("1")
}

#[cfg(feature = "rvr")]
pub(crate) fn emit_segment_stats(segment: usize, before: PoolStatsSnapshot) {
    if !stats_enabled() {
        return;
    }
    let after = PoolStatsSnapshot::capture();
    eprintln!(
        "OPENVM_RVR_CUDA_POOL_STATS segment={segment} hits={} misses={} populate_calls={} \
         populate_bytes={} returns_enqueued={} returns_synchronized={} returns_pooled={} \
         pending={} pending_peak={} ready_buffers={} ready_bytes={} quarantined_total={} \
         sync_failures_total={} registration_failures_total={} zeroed_bytes={} zero_time_us={}",
        after.hits.saturating_sub(before.hits),
        after.misses.saturating_sub(before.misses),
        after.populate_calls.saturating_sub(before.populate_calls),
        after.populate_bytes.saturating_sub(before.populate_bytes),
        after
            .returns_enqueued
            .saturating_sub(before.returns_enqueued),
        after
            .returns_synchronized
            .saturating_sub(before.returns_synchronized),
        after.returns_pooled.saturating_sub(before.returns_pooled),
        after.pending,
        after.pending_peak,
        after.ready_buffers,
        after.ready_bytes,
        after.quarantined,
        after.sync_failures,
        after.registration_failures,
        after.zeroed_bytes.saturating_sub(before.zeroed_bytes),
        after.zero_time_us.saturating_sub(before.zero_time_us),
    );
}

/// Page-locks `len` bytes at `ptr` in a single `cudaHostRegister` call.
/// NOTE: registration must be one call per buffer: `cudaMemcpyAsync` rejects
/// (cudaErrorInvalidValue) source ranges that span multiple distinct
/// page-locked registrations, so chunked registration corrupts nothing but
/// breaks every copy crossing a chunk boundary.
pub(crate) fn register_region(ptr: *mut u8, len: usize) -> bool {
    let _lifecycle = lock_unpoisoned(&LIFECYCLE_GATE);
    if SHUTTING_DOWN.load(Ordering::Acquire) {
        return false;
    }
    register_region_inner(ptr, len)
}

fn register_region_inner(ptr: *mut u8, len: usize) -> bool {
    // SAFETY: [ptr, ptr+len) is a live allocation owned by the caller.
    let rc = unsafe { cudaHostRegister(ptr as *mut c_void, len, 0) };
    if rc != 0 {
        tracing::debug!("cudaHostRegister failed with {rc}; record arena buffer stays pageable");
        return false;
    }
    true
}

extern "C" {
    fn cudaGetDevice(device: *mut i32) -> i32;
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaHostRegister(ptr: *mut c_void, size: usize, flags: u32) -> i32;
    fn cudaHostUnregister(ptr: *mut c_void) -> i32;
    fn cudaDeviceSynchronize() -> i32;
}

/// Reverses a successful [`register_region`]. The caller must ensure no copy
/// from the region is still in flight. Returns false without touching CUDA
/// after pool shutdown has begun, so the caller can quarantine the allocation.
pub(crate) fn unregister_region(ptr: *mut u8) -> bool {
    let _lifecycle = lock_unpoisoned(&LIFECYCLE_GATE);
    if SHUTTING_DOWN.load(Ordering::Acquire) {
        return false;
    }
    unregister_region_inner(ptr)
}

fn unregister_region_inner(ptr: *mut u8) -> bool {
    // SAFETY: mirrors a successful registration of the same base pointer.
    unsafe { cudaHostUnregister(ptr as *mut c_void) == 0 }
}

/// Registered, all-zero buffers ready for reuse, keyed by allocation size.
fn pool() -> &'static Mutex<BTreeMap<usize, Vec<Vec<u8>>>> {
    &state().ready
}

/// Base pointers of buffers whose `cudaHostRegister` succeeded.
fn registered() -> &'static Mutex<HashSet<usize>> {
    &state().registered
}

struct CleanerRuntime {
    device: i32,
    sender: Mutex<Option<mpsc::Sender<CleanerMessage>>>,
    worker: Mutex<Option<JoinHandle<()>>>,
}

static SHUTTING_DOWN: AtomicBool = AtomicBool::new(false);
static LIFECYCLE_GATE: Mutex<()> = Mutex::new(());
static CLEANER_WORK_GATE: Mutex<()> = Mutex::new(());
static CLEANER_INIT: Mutex<()> = Mutex::new(());
static CLEANER: OnceLock<&'static CleanerRuntime> = OnceLock::new();

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(feature = "rvr")]
pub(crate) fn is_shutting_down() -> bool {
    SHUTTING_DOWN.load(Ordering::Acquire)
}

/// Keep an allocation alive until process exit without touching CUDA or pool
/// state. This is the only safe fallback once unregistering is unavailable.
pub(crate) fn quarantine<T>(value: T) {
    quarantine_pending(value);
}

extern "C" fn shutdown_at_exit() {
    shutdown_cleaner();
}

/// Enter the no-touch state before CUDA's own process-exit handlers run. The
/// lifecycle gate stops producers, while the work gate lets an already active
/// batch finish while CUDA is still valid. Queued returns are never released.
fn shutdown_cleaner() {
    let Some(runtime) = CLEANER.get().copied() else {
        return;
    };
    shutdown_pending_return_worker(
        &SHUTTING_DOWN,
        &LIFECYCLE_GATE,
        &CLEANER_WORK_GATE,
        &runtime.sender,
        &runtime.worker,
    );
}

fn process_returned_batch(batch: Vec<ReturnedBuffer>, batch_idx: usize) {
    // The H2D copies reading these buffers were enqueued before the owning
    // arenas dropped; wait for them (and anything else in flight) before
    // touching contents. Unique label per batch: the timing metric derived
    // from this span is a gauge, so identical label sets overwrite.
    let _span = tracing::info_span!("arena_cleaner_batch", batch = batch_idx.to_string()).entered();
    let rc = unsafe { cudaDeviceSynchronize() };
    if rc != 0 {
        // No usable CUDA context: the buffers cannot be proven idle. Dropping
        // the wrappers quarantines their allocations.
        tracing::debug!(
            "cudaDeviceSynchronize failed with {rc}; quarantining {} record arena buffers",
            batch.len()
        );
        stats().sync_failures.fetch_add(1, Ordering::Relaxed);
        stats()
            .quarantined
            .fetch_add(batch.len() as u64, Ordering::Relaxed);
        stats()
            .pending
            .fetch_sub(batch.len() as u64, Ordering::Relaxed);
        return;
    }

    let mut zeroed_bytes = 0u64;
    let mut zero_time_us = 0u64;
    for returned in batch {
        let (mut buffer, dirty_len) = returned.release();
        stats().pending.fetch_sub(1, Ordering::Relaxed);
        stats().returns_synchronized.fetch_add(1, Ordering::Relaxed);
        if buffer.is_empty() || !buffer.len().is_power_of_two() {
            continue; // synchronized but not pool-shaped
        }
        let ptr = buffer.as_mut_ptr();
        let is_new = !registered().lock().unwrap().contains(&(ptr as usize));
        if is_new {
            if !register_region_inner(ptr, buffer.len()) {
                // Out of pinnable memory: drop the buffer, never pool it.
                stats()
                    .registration_failures
                    .fetch_add(1, Ordering::Relaxed);
                continue;
            }
            registered().lock().unwrap().insert(ptr as usize);
        }
        // Restore the fresh-arena invariant (all zero). Bytes past the dirty
        // prefix were never written or were cleared on an earlier cycle.
        let dirty_len = dirty_len.min(buffer.len());
        let zero_started = std::time::Instant::now();
        buffer[..dirty_len].fill(0);
        zero_time_us += zero_started.elapsed().as_micros() as u64;
        zeroed_bytes += dirty_len as u64;
        pool()
            .lock()
            .unwrap()
            .entry(buffer.len())
            .or_default()
            .push(buffer);
        stats().returns_pooled.fetch_add(1, Ordering::Relaxed);
    }
    stats()
        .zeroed_bytes
        .fetch_add(zeroed_bytes, Ordering::Relaxed);
    stats()
        .zero_time_us
        .fetch_add(zero_time_us, Ordering::Relaxed);
}

fn run_cleaner(rx: mpsc::Receiver<CleanerMessage>) {
    run_pending_return_worker(
        rx,
        &SHUTTING_DOWN,
        &CLEANER_WORK_GATE,
        std::time::Duration::from_millis(100),
        64,
        process_returned_batch,
    );
}

/// Cleaner thread: registers (first cycle) and re-zeroes buffers off the
/// critical path, then makes them available to [`take`].
fn cleaner() -> Option<&'static CleanerRuntime> {
    if SHUTTING_DOWN.load(Ordering::Acquire) {
        return None;
    }
    if let Some(runtime) = CLEANER.get().copied() {
        return Some(runtime);
    }

    let _init = lock_unpoisoned(&CLEANER_INIT);
    if SHUTTING_DOWN.load(Ordering::Acquire) {
        return None;
    }
    if let Some(runtime) = CLEANER.get().copied() {
        return Some(runtime);
    }

    // `cudaSetDevice` explicitly initializes the runtime and primary context.
    // Do that before registering our atexit hook: handlers run in reverse
    // registration order, so the cleaner stops before CUDA tears down.
    let mut device = 0;
    let mut init_rc = unsafe { cudaGetDevice(&mut device) };
    if init_rc == 0 {
        init_rc = unsafe { cudaSetDevice(device) };
    }
    if init_rc != 0 {
        tracing::debug!(
            "CUDA runtime initialization failed with {init_rc}; record arena returns stay quarantined"
        );
        return None;
    }

    let (sender, receiver) = mpsc::channel();
    let runtime = Box::leak(Box::new(CleanerRuntime {
        device,
        sender: Mutex::new(Some(sender)),
        worker: Mutex::new(None),
    }));
    assert!(CLEANER.set(runtime).is_ok());

    // CUDA installed its exit handlers above. atexit's reverse order therefore
    // stops and joins this worker before CUDA runtime destruction begins.
    if unsafe { libc::atexit(shutdown_at_exit) } != 0 {
        SHUTTING_DOWN.store(true, Ordering::Release);
        lock_unpoisoned(&runtime.sender).take();
        return None;
    }

    let cleaner_device = runtime.device;
    let worker = std::thread::Builder::new()
        .name("record-arena-pinner".into())
        .spawn(move || {
            {
                let _work = lock_unpoisoned(&CLEANER_WORK_GATE);
                if SHUTTING_DOWN.load(Ordering::Acquire)
                    || unsafe { cudaSetDevice(cleaner_device) } != 0
                {
                    return;
                }
            }
            run_cleaner(receiver);
        })
        .expect("failed to spawn record-arena pinner thread");
    *lock_unpoisoned(&runtime.worker) = Some(worker);
    Some(runtime)
}

/// Returns a ready buffer and whether its freshly allocated pages still need
/// to be faulted in by a latency-sensitive caller. Pool hits are registered,
/// resident, and already zeroed by the cleaner.
pub(crate) fn take_with_prefault_status(min_size: usize) -> (Vec<u8>, bool) {
    let size = min_size.next_power_of_two();
    let _lifecycle = lock_unpoisoned(&LIFECYCLE_GATE);
    if SHUTTING_DOWN.load(Ordering::Acquire) {
        return (vec![0u8; size], true);
    }
    if let Some(buffer) = pool()
        .lock()
        .unwrap()
        .get_mut(&size)
        .and_then(|bufs| bufs.pop())
    {
        debug_assert_eq!(buffer.len(), size);
        stats().hits.fetch_add(1, Ordering::Relaxed);
        return (buffer, false);
    }
    // Pool miss: pageable memory, zeroed lazily by the kernel, exactly as
    // without the pool. The buffer becomes pinned when first given back.
    stats().misses.fetch_add(1, Ordering::Relaxed);
    (vec![0u8; size], true)
}

pub(crate) fn take(min_size: usize) -> Vec<u8> {
    take_with_prefault_status(min_size).0
}

/// Make a fresh lazy-zero allocation resident with a batched kernel population request. This is
/// used only for arena-native pool misses: generated C immediately streams across the backing, so
/// leaving the pages lazy would put one minor fault per 4 KiB back on the preflight critical path.
/// Recycled pool hits are already resident and skip this function.
#[cfg(feature = "rvr")]
pub(crate) fn populate_write(buffer: &mut [u8]) {
    if buffer.is_empty() {
        return;
    }
    let _lifecycle = lock_unpoisoned(&LIFECYCLE_GATE);
    if SHUTTING_DOWN.load(Ordering::Acquire) {
        return;
    }
    stats().populate_calls.fetch_add(1, Ordering::Relaxed);
    stats()
        .populate_bytes
        .fetch_add(buffer.len() as u64, Ordering::Relaxed);

    #[cfg(target_os = "linux")]
    {
        const PAGE_BYTES: usize = 4096;
        let allocation_start = buffer.as_mut_ptr() as usize;
        let allocation_end = allocation_start + buffer.len();
        let interior_start = allocation_start.next_multiple_of(PAGE_BYTES);
        let interior_end = allocation_end & !(PAGE_BYTES - 1);
        if interior_start < interior_end {
            // SAFETY: the range is page-aligned and wholly contained in the live allocation.
            let rc = unsafe {
                libc::madvise(
                    interior_start as *mut libc::c_void,
                    interior_end - interior_start,
                    libc::MADV_POPULATE_WRITE,
                )
            };
            if rc == 0 {
                // The unaligned boundary pages are outside the advised interior. Touching their
                // first/last bytes preserves the all-zero invariant while making them resident.
                unsafe { std::ptr::write_volatile(buffer.as_mut_ptr(), 0) };
                if buffer.len() > 1 {
                    unsafe {
                        std::ptr::write_volatile(buffer.as_mut_ptr().add(buffer.len() - 1), 0)
                    };
                }
                return;
            }
        }
    }

    // Portable fallback and fallback for kernels without MADV_POPULATE_WRITE.
    for page in (0..buffer.len()).step_by(4096) {
        unsafe { std::ptr::write_volatile(buffer.as_mut_ptr().add(page), 0) };
    }
    if buffer.len() > 1 {
        unsafe { std::ptr::write_volatile(buffer.as_mut_ptr().add(buffer.len() - 1), 0) };
    }
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
    let _lifecycle = lock_unpoisoned(&LIFECYCLE_GATE);
    if SHUTTING_DOWN.load(Ordering::Acquire) {
        return;
    }
    let Some(cleaner) = cleaner() else {
        return;
    };
    let sender = lock_unpoisoned(&cleaner.sender);
    let Some(sender) = sender.as_ref() else {
        return;
    };
    let pending = stats().pending.fetch_add(1, Ordering::Relaxed) + 1;
    stats().pending_peak.fetch_max(pending, Ordering::Relaxed);
    match sender.send(CleanerMessage::Return(returned)) {
        Ok(()) => {
            stats().returns_enqueued.fetch_add(1, Ordering::Relaxed);
        }
        Err(error) => {
            stats().pending.fetch_sub(1, Ordering::Relaxed);
            stats().quarantined.fetch_add(1, Ordering::Relaxed);
            // Dropping the PendingReturn from SendError intentionally leaks the allocation.
            drop(error);
        }
    }
}

/// Unregisters and frees all pooled buffers (test hygiene; optional).
#[allow(dead_code)]
pub(crate) fn clear() {
    let _lifecycle = lock_unpoisoned(&LIFECYCLE_GATE);
    if SHUTTING_DOWN.load(Ordering::Acquire) {
        return;
    }
    let mut pool = pool().lock().unwrap();
    let mut reg = registered().lock().unwrap();
    for (_, bufs) in pool.iter_mut() {
        for mut buf in bufs.drain(..) {
            reg.remove(&(buf.as_ptr() as usize));
            if !unregister_region_inner(buf.as_mut_ptr()) {
                drop(PendingReturn::new(buf));
            }
        }
    }
    pool.clear();
}
