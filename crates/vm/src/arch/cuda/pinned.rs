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
//! therefore records the current CUDA device on return and calls
//! `cudaDeviceSynchronize` on that device (batched by device) before touching
//! buffer contents.

use std::{
    cell::Cell,
    collections::{BTreeMap, HashMap, VecDeque},
    ffi::c_void,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc, Condvar, Mutex, MutexGuard, OnceLock,
    },
    thread::JoinHandle,
};

use crate::arch::pending_return::{
    quarantine as quarantine_pending, run_pending_return_worker, shutdown_pending_return_worker,
    PendingReturn, PendingReturnMessage,
};

const MAX_REGISTERED_BYTES: usize = 2 * 1024 * 1024 * 1024;
const MAX_READY_SIZE_CLASSES: usize = 16;
const MAX_READY_BUFFERS_PER_CLASS: usize = 4;
const CLEANER_QUEUE_CAPACITY: usize = 128;
const MAX_PENDING_BYTES: usize = 2 * 1024 * 1024 * 1024;
/// Upper bound on how many returns one cleaner batch coalesces before it
/// fences and recycles them. A batch pays a single full-device
/// `cudaDeviceSynchronize` per originating device, so this must comfortably
/// exceed a segment's return burst (empirically ~37, peaking near the
/// [`CLEANER_QUEUE_CAPACITY`]) — otherwise a large burst splits across batches
/// and each split pays another redundant device fence. It is only a safety cap
/// against an unbounded batch under a sustained producer; the in-flight count
/// is already bounded by [`CLEANER_QUEUE_CAPACITY`] plus blocked producers.
const CLEANER_BATCH_LIMIT: usize = 1024;

/// Device association is kept outside the inner quarantine wrapper so the
/// cleaner can batch fences without releasing a backing before it is idle.
struct ReturnedBuffer {
    device: i32,
    pending: PendingBuffer,
}

type PendingBuffer = PendingReturn<(Vec<u8>, usize, PendingBytePermit)>;
type PendingReturnedBuffer = PendingReturn<ReturnedBuffer>;
type CleanerMessage = PendingReturnMessage<ReturnedBuffer>;
type ReturnsByDevice = BTreeMap<i32, Vec<PendingBuffer>>;

struct ReadyBuffer {
    returned_at: u64,
    buffer: Vec<u8>,
}

#[derive(Default)]
struct ReadyPool {
    buffers: BTreeMap<usize, VecDeque<ReadyBuffer>>,
    bytes: usize,
    next_return: u64,
}

impl ReadyPool {
    // Only consumed by the rvr-gated `PoolStatsSnapshot::capture`; gate the
    // method to match so a `cuda`-without-`rvr` build does not see it as dead.
    #[cfg(feature = "rvr")]
    fn buffer_count(&self) -> usize {
        self.buffers.values().map(VecDeque::len).sum()
    }

    fn take(&mut self, size: usize) -> Option<Vec<u8>> {
        let (buffer, remove_class) = {
            let class = self.buffers.get_mut(&size)?;
            let buffer = class.pop_back()?.buffer;
            (buffer, class.is_empty())
        };
        if remove_class {
            self.buffers.remove(&size);
        }
        self.bytes -= size;
        Some(buffer)
    }

    fn insert(&mut self, buffer: Vec<u8>) -> Vec<Vec<u8>> {
        self.insert_with_limits(
            buffer,
            MAX_REGISTERED_BYTES,
            MAX_READY_SIZE_CLASSES,
            MAX_READY_BUFFERS_PER_CLASS,
        )
    }

    fn insert_with_limits(
        &mut self,
        buffer: Vec<u8>,
        max_bytes: usize,
        max_size_classes: usize,
        max_buffers_per_class: usize,
    ) -> Vec<Vec<u8>> {
        let size = buffer.len();
        let returned_at = self.next_return;
        self.next_return = self.next_return.saturating_add(1);
        self.buffers
            .entry(size)
            .or_default()
            .push_back(ReadyBuffer {
                returned_at,
                buffer,
            });
        self.bytes = self.bytes.saturating_add(size);

        let mut evicted = Vec::new();
        while self
            .buffers
            .get(&size)
            .is_some_and(|class| class.len() > max_buffers_per_class)
        {
            let oldest = self
                .buffers
                .get_mut(&size)
                .and_then(VecDeque::pop_front)
                .expect("overfull ready-buffer class was empty");
            self.bytes -= size;
            evicted.push(oldest.buffer);
        }
        if self.buffers.get(&size).is_some_and(VecDeque::is_empty) {
            self.buffers.remove(&size);
        }

        while self.buffers.len() > max_size_classes {
            let oldest_class = self
                .oldest_class()
                .expect("overfull ready pool had no size class");
            self.evict_class(oldest_class, &mut evicted);
        }
        while self.bytes > max_bytes {
            let Some(oldest_class) = self.oldest_class() else {
                break;
            };
            self.evict_oldest_from_class(oldest_class, &mut evicted);
        }
        evicted
    }

    fn evict_oldest_bytes(&mut self, bytes: usize) -> Vec<Vec<u8>> {
        let target = self.bytes.saturating_sub(bytes);
        let mut evicted = Vec::new();
        while self.bytes > target {
            let Some(oldest_class) = self.oldest_class() else {
                break;
            };
            self.evict_oldest_from_class(oldest_class, &mut evicted);
        }
        evicted
    }

    fn oldest_class(&self) -> Option<usize> {
        self.buffers
            .iter()
            .filter_map(|(&size, class)| class.front().map(|buffer| (buffer.returned_at, size)))
            .min()
            .map(|(_, size)| size)
    }

    fn evict_oldest_from_class(&mut self, size: usize, evicted: &mut Vec<Vec<u8>>) {
        let (oldest, remove_class) = {
            let class = self
                .buffers
                .get_mut(&size)
                .expect("selected ready-buffer class disappeared");
            let oldest = class
                .pop_front()
                .expect("selected ready-buffer class was empty");
            (oldest, class.is_empty())
        };
        if remove_class {
            self.buffers.remove(&size);
        }
        self.bytes -= size;
        evicted.push(oldest.buffer);
    }

    fn evict_class(&mut self, size: usize, evicted: &mut Vec<Vec<u8>>) {
        let class = self
            .buffers
            .remove(&size)
            .expect("selected ready-buffer class disappeared");
        self.bytes -= size * class.len();
        evicted.extend(class.into_iter().map(|buffer| buffer.buffer));
    }

    fn drain(&mut self) -> Vec<Vec<u8>> {
        let buffers = std::mem::take(&mut self.buffers)
            .into_values()
            .flatten()
            .map(|buffer| buffer.buffer)
            .collect();
        self.bytes = 0;
        buffers
    }
}

#[derive(Default)]
struct RegisteredRegions {
    sizes: HashMap<usize, usize>,
    bytes: usize,
}

#[derive(Default)]
struct PendingByteBudget {
    bytes: Mutex<usize>,
    available: Condvar,
}

struct PendingBytePermit {
    bytes: usize,
}

impl PendingByteBudget {
    fn wake_all(&self) {
        // Take the predicate mutex so a waiter cannot miss the notification
        // between observing SHUTTING_DOWN=false and entering Condvar::wait.
        let _pending = lock_unpoisoned(&self.bytes);
        self.available.notify_all();
    }
}

impl PendingBytePermit {
    fn acquire(bytes: usize) -> Option<Self> {
        let budget = &state().pending_bytes;
        let mut pending = lock_unpoisoned(&budget.bytes);
        loop {
            if SHUTTING_DOWN.load(Ordering::Acquire) {
                return None;
            }
            if pending_bytes_fit(*pending, bytes) {
                *pending = pending.saturating_add(bytes);
                return Some(Self { bytes });
            }
            pending = budget
                .available
                .wait(pending)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
    }
}

impl Drop for PendingBytePermit {
    fn drop(&mut self) {
        let budget = &state().pending_bytes;
        let mut pending = lock_unpoisoned(&budget.bytes);
        *pending = pending
            .checked_sub(self.bytes)
            .expect("pinned cleaner pending-byte accounting underflow");
        budget.available.notify_all();
    }
}

fn pending_bytes_fit(current: usize, requested: usize) -> bool {
    if requested > MAX_PENDING_BYTES {
        // A single arena can legitimately exceed the pool budget. Let it
        // make progress only after all ordinary returns have drained, then
        // hold every other producer until that oversized return is safe.
        current == 0
    } else {
        current <= MAX_PENDING_BYTES - requested
    }
}

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
    ready: Mutex<ReadyPool>,
    registered: Mutex<RegisteredRegions>,
    pending_bytes: PendingByteBudget,
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
        let ready_buffers = ready.buffer_count() as u64;
        let ready_bytes = ready.bytes as u64;
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

/// Register an allocation that will subsequently be owned by this pool. Unlike
/// [`register_region`], this records the base pointer so the cleaner reuses the registration and
/// [`clear`] can unregister it exactly once.
#[cfg(feature = "rvr")]
pub(crate) fn register_pool_region(buffer: &mut [u8]) -> bool {
    if buffer.is_empty() {
        return true;
    }
    let _lifecycle = lock_unpoisoned(&LIFECYCLE_GATE);
    if SHUTTING_DOWN.load(Ordering::Acquire) {
        return false;
    }
    if !register_owned_region(buffer.as_mut_ptr(), buffer.len()) {
        stats()
            .registration_failures
            .fetch_add(1, Ordering::Relaxed);
        return false;
    }
    true
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
fn pool() -> &'static Mutex<ReadyPool> {
    &state().ready
}

/// Base pointers and lengths of pool-owned buffers whose `cudaHostRegister` succeeded.
fn registered() -> &'static Mutex<RegisteredRegions> {
    &state().registered
}

/// Serialize registration admission so eager prewarm and the cleaner cannot
/// both pass the byte-limit check.
static REGISTRATION_GATE: Mutex<()> = Mutex::new(());

fn register_owned_region(ptr: *mut u8, len: usize) -> bool {
    let _registration = lock_unpoisoned(&REGISTRATION_GATE);
    if registered()
        .lock()
        .unwrap()
        .sizes
        .contains_key(&(ptr as usize))
    {
        return true;
    }
    if len > MAX_REGISTERED_BYTES {
        return false;
    }

    let bytes_to_evict = {
        let registered = registered().lock().unwrap();
        registered
            .bytes
            .saturating_add(len)
            .saturating_sub(MAX_REGISTERED_BYTES)
    };
    if bytes_to_evict != 0 {
        let evicted = pool().lock().unwrap().evict_oldest_bytes(bytes_to_evict);
        retire_idle_buffers(evicted);
    }

    let mut registered = registered().lock().unwrap();
    if registered.bytes.saturating_add(len) > MAX_REGISTERED_BYTES {
        return false;
    }
    if !register_region_inner(ptr, len) {
        return false;
    }
    registered.sizes.insert(ptr as usize, len);
    registered.bytes += len;
    true
}

/// Unregister and free buffers already removed from the ready map. Every
/// caller must first establish that no CUDA work can still reference them.
fn retire_idle_buffers(buffers: Vec<Vec<u8>>) {
    for mut buffer in buffers {
        let ptr = buffer.as_ptr() as usize;
        let mut registered = registered().lock().unwrap();
        if let Some(&size) = registered.sizes.get(&ptr) {
            if unregister_region_inner(buffer.as_mut_ptr()) {
                registered.sizes.remove(&ptr);
                registered.bytes -= size;
            } else {
                // Keep failed unregistrations charged against the global cap.
                drop(registered);
                quarantine_pending(buffer);
            }
        }
    }
}

/// Transfer a registered, unused startup reserve into the bounded ready pool.
/// No CUDA work has ever referenced this backing, so it needs no device fence.
#[cfg(feature = "rvr")]
pub(crate) fn recycle_idle_pool_region(buffer: Vec<u8>) {
    if buffer.is_empty() {
        return;
    }
    let _lifecycle = lock_unpoisoned(&LIFECYCLE_GATE);
    if SHUTTING_DOWN.load(Ordering::Acquire) {
        quarantine_pending(buffer);
        return;
    }
    let _registration = lock_unpoisoned(&REGISTRATION_GATE);
    let is_registered = registered()
        .lock()
        .unwrap()
        .sizes
        .contains_key(&(buffer.as_ptr() as usize));
    if !is_registered || !buffer.len().is_power_of_two() {
        retire_idle_buffers(vec![buffer]);
        return;
    }
    let evicted = pool().lock().unwrap().insert(buffer);
    retire_idle_buffers(evicted);
}

struct CleanerRuntime {
    sender: Mutex<Option<mpsc::SyncSender<CleanerMessage>>>,
    worker: Mutex<Option<JoinHandle<()>>>,
}

static SHUTTING_DOWN: AtomicBool = AtomicBool::new(false);
static LIFECYCLE_GATE: Mutex<()> = Mutex::new(());
static CLEANER_WORK_GATE: Mutex<()> = Mutex::new(());
static CLEANER_INIT: Mutex<()> = Mutex::new(());
static CLEANER: OnceLock<&'static CleanerRuntime> = OnceLock::new();

thread_local! {
    /// Startup pool priming runs on the proving thread before any asynchronous H2D can reference
    /// a newly allocated arena. Register fresh misses immediately in that narrow scope so the
    /// first real segment receives pinned buffers instead of racing the background cleaner.
    static EAGER_REGISTRATION: Cell<bool> = const { Cell::new(false) };
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Run a startup allocation pass with fresh pool misses registered synchronously. The scope is
/// thread-local: ordinary segment execution retains the non-blocking pageable-miss fallback.
#[cfg(feature = "rvr")]
pub(crate) fn with_eager_registration<T>(f: impl FnOnce() -> T) -> T {
    struct Reset(bool);

    impl Drop for Reset {
        fn drop(&mut self) {
            EAGER_REGISTRATION.with(|enabled| enabled.set(self.0));
        }
    }

    let previous = EAGER_REGISTRATION.with(|enabled| enabled.replace(true));
    let _reset = Reset(previous);
    f()
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
        || state().pending_bytes.wake_all(),
    );
}

fn group_returned_by_device(batch: Vec<PendingReturnedBuffer>) -> ReturnsByDevice {
    let mut by_device = BTreeMap::<_, Vec<_>>::new();
    for returned in batch {
        let ReturnedBuffer { device, pending } = returned.release();
        by_device.entry(device).or_default().push(pending);
    }
    by_device
}

fn process_returned_batch(batch: Vec<PendingReturnedBuffer>, batch_idx: usize) {
    // The H2D copies reading these buffers were enqueued before the owning
    // arenas dropped. Fence the originating device of each return before
    // touching contents; grouping retains one synchronization per device,
    // rather than one per buffer. Unique label per batch: the timing metric
    // derived from this span is a gauge, so identical label sets overwrite.
    let _span = tracing::info_span!("arena_cleaner_batch", batch = batch_idx.to_string()).entered();
    let mut zeroed_bytes = 0u64;
    let mut zero_time_us = 0u64;
    let mut current_device = {
        let mut device = 0;
        (unsafe { cudaGetDevice(&mut device) } == 0).then_some(device)
    };
    for (device, returned_buffers) in group_returned_by_device(batch) {
        let device_rc = if current_device == Some(device) {
            0
        } else {
            unsafe { cudaSetDevice(device) }
        };
        if device_rc == 0 {
            current_device = Some(device);
        }
        let sync_rc = if device_rc == 0 {
            unsafe { cudaDeviceSynchronize() }
        } else {
            device_rc
        };
        if sync_rc != 0 {
            // The buffers cannot be proven idle. Dropping the inner wrappers
            // quarantines their allocations without touching their contents.
            tracing::debug!(
                "CUDA device {device} synchronization failed with {sync_rc}; quarantining {} \
                 record arena buffers",
                returned_buffers.len()
            );
            stats().sync_failures.fetch_add(1, Ordering::Relaxed);
            stats()
                .quarantined
                .fetch_add(returned_buffers.len() as u64, Ordering::Relaxed);
            stats()
                .pending
                .fetch_sub(returned_buffers.len() as u64, Ordering::Release);
            continue;
        }

        for returned in returned_buffers {
            let (mut buffer, dirty_len, _pending_bytes) = returned.release();
            stats().returns_synchronized.fetch_add(1, Ordering::Relaxed);
            (|| {
                if buffer.is_empty() || !buffer.len().is_power_of_two() {
                    return; // synchronized but not pool-shaped
                }
                if !register_owned_region(buffer.as_mut_ptr(), buffer.len()) {
                    // Registration cap or CUDA failure: drop the synchronized
                    // pageable buffer instead of retaining it.
                    stats()
                        .registration_failures
                        .fetch_add(1, Ordering::Relaxed);
                    return;
                }
                // Restore the fresh-arena invariant (all zero). Bytes past the dirty
                // prefix were never written or were cleared on an earlier cycle.
                let dirty_len = dirty_len.min(buffer.len());
                let zero_started = std::time::Instant::now();
                buffer[..dirty_len].fill(0);
                zero_time_us += zero_started.elapsed().as_micros() as u64;
                zeroed_bytes += dirty_len as u64;
                let evicted = pool().lock().unwrap().insert(buffer);
                retire_idle_buffers(evicted);
                stats().returns_pooled.fetch_add(1, Ordering::Relaxed);
            })();
            // Publish completion only after a reusable buffer has reached the ready map (or after
            // a failed/non-pool-shaped return has been disposed of). Startup drain relies on this.
            stats().pending.fetch_sub(1, Ordering::Release);
        }
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
        CLEANER_BATCH_LIMIT,
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

    let (sender, receiver) = mpsc::sync_channel(CLEANER_QUEUE_CAPACITY);
    let runtime = Box::leak(Box::new(CleanerRuntime {
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

    let worker = std::thread::Builder::new()
        .name("record-arena-pinner".into())
        .spawn(move || run_cleaner(receiver))
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
    if let Some(buffer) = pool().lock().unwrap().take(size) {
        debug_assert_eq!(buffer.len(), size);
        stats().hits.fetch_add(1, Ordering::Relaxed);
        return (buffer, false);
    }
    // Pool miss: pageable memory, zeroed lazily by the kernel, exactly as
    // without the pool. The buffer becomes pinned when first given back.
    stats().misses.fetch_add(1, Ordering::Relaxed);
    let mut buffer = vec![0u8; size];
    let eager = EAGER_REGISTRATION.with(Cell::get);
    let registered_eagerly = eager
        && if register_owned_region(buffer.as_mut_ptr(), buffer.len()) {
            true
        } else {
            stats()
                .registration_failures
                .fetch_add(1, Ordering::Relaxed);
            false
        };
    if registered_eagerly {
        (buffer, false)
    } else {
        (buffer, true)
    }
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
    let buffer_len = buffer.len();
    // If the send races process teardown, `PendingReturn` deliberately leaks
    // the backing: without the cleaner's sync, freeing it is not safe.
    let pending = PendingReturn::new((buffer, dirty_len));
    let device = {
        let _lifecycle = lock_unpoisoned(&LIFECYCLE_GATE);
        if SHUTTING_DOWN.load(Ordering::Acquire) {
            return;
        }
        let mut device = 0;
        let device_rc = unsafe { cudaGetDevice(&mut device) };
        if device_rc != 0 {
            tracing::debug!(
                "cudaGetDevice failed with {device_rc}; returned record arena stays quarantined"
            );
            stats().quarantined.fetch_add(1, Ordering::Relaxed);
            return;
        }
        if cleaner().is_none() {
            stats().quarantined.fetch_add(1, Ordering::Relaxed);
            return;
        }
        device
    };

    // Capacity admission must not hold the lifecycle gate: shutdown sets the
    // flag and wakes this wait so teardown can always make progress.
    let Some(pending_bytes) = PendingBytePermit::acquire(buffer_len) else {
        stats().quarantined.fetch_add(1, Ordering::Relaxed);
        return;
    };
    let (buffer, dirty_len) = pending.release();
    let pending = PendingReturn::new((buffer, dirty_len, pending_bytes));

    let sender = {
        let _lifecycle = lock_unpoisoned(&LIFECYCLE_GATE);
        if SHUTTING_DOWN.load(Ordering::Acquire) {
            return;
        }
        let cleaner = CLEANER
            .get()
            .copied()
            .expect("initialized pinned cleaner disappeared");
        let sender = lock_unpoisoned(&cleaner.sender);
        let Some(sender) = sender.as_ref() else {
            stats().quarantined.fetch_add(1, Ordering::Relaxed);
            return;
        };
        sender.clone()
    };
    // Bound queued storage by bytes as well as item count. The permit remains
    // inside the quarantine wrapper, so a return whose DMA cannot be proven
    // complete continues to consume budget instead of admitting an unbounded
    // succession of unsafe allocations.
    let returned = PendingReturn::new(ReturnedBuffer { device, pending });
    let pending = stats().pending.fetch_add(1, Ordering::Relaxed) + 1;
    stats().pending_peak.fetch_max(pending, Ordering::Relaxed);
    match sender.send(CleanerMessage::Return(returned)) {
        Ok(()) => {
            stats().returns_enqueued.fetch_add(1, Ordering::Relaxed);
        }
        Err(error) => {
            stats().pending.fetch_sub(1, Ordering::Relaxed);
            stats().quarantined.fetch_add(1, Ordering::Relaxed);
            // Dropping the nested PendingReturn from SendError intentionally leaks the allocation.
            drop(error);
        }
    }
}

/// Wait until every buffer queued ahead of this call's cleaner barrier has completed CUDA
/// synchronization, zeroing, and insertion into the ready map. Returns queued by concurrent
/// proving after the barrier do not extend the wait. Used only by startup pool priming after it has
/// dropped all temporary arenas; ordinary proving never waits for the cleaner.
#[cfg(feature = "rvr")]
pub(crate) fn drain_returns(timeout: std::time::Duration) -> bool {
    let (acknowledge, acknowledged) = mpsc::channel();
    {
        let _lifecycle = lock_unpoisoned(&LIFECYCLE_GATE);
        if SHUTTING_DOWN.load(Ordering::Acquire) {
            return false;
        }
        let Some(cleaner) = CLEANER.get().copied() else {
            return stats().pending.load(Ordering::Acquire) == 0;
        };
        let sender = lock_unpoisoned(&cleaner.sender);
        let Some(sender) = sender.as_ref() else {
            return false;
        };
        if sender.send(CleanerMessage::Barrier(acknowledge)).is_err() {
            return false;
        }
    }
    acknowledged.recv_timeout(timeout).is_ok()
}

/// Unregisters and frees all pooled buffers (test hygiene; optional).
#[allow(dead_code)]
pub(crate) fn clear() {
    let _lifecycle = lock_unpoisoned(&LIFECYCLE_GATE);
    if SHUTTING_DOWN.load(Ordering::Acquire) {
        return;
    }
    let _work = lock_unpoisoned(&CLEANER_WORK_GATE);
    let _registration = lock_unpoisoned(&REGISTRATION_GATE);
    let buffers = pool().lock().unwrap().drain();
    retire_idle_buffers(buffers);
}

#[cfg(test)]
mod tests {
    use super::{
        group_returned_by_device, pending_bytes_fit, PendingBytePermit, PendingReturn, ReadyPool,
        ReturnedBuffer, MAX_PENDING_BYTES,
    };

    #[test]
    fn ready_pool_evicts_oldest_buffers_at_each_limit() {
        let mut pool = ReadyPool::default();

        let oldest = vec![0u8; 4];
        let oldest_ptr = oldest.as_ptr();
        assert!(pool.insert_with_limits(oldest, 32, 2, 1).is_empty());

        let newer_same_class = vec![0u8; 4];
        let newer_same_class_ptr = newer_same_class.as_ptr();
        let evicted = pool.insert_with_limits(newer_same_class, 32, 2, 1);
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0].as_ptr(), oldest_ptr);

        let second_class = vec![0u8; 8];
        assert!(pool.insert_with_limits(second_class, 32, 2, 1).is_empty());
        let third_class = vec![0u8; 16];
        let evicted = pool.insert_with_limits(third_class, 32, 2, 1);
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0].as_ptr(), newer_same_class_ptr);
        assert_eq!(pool.buffers.len(), 2);
        assert_eq!(pool.bytes, 24);

        let over_byte_limit = vec![0u8; 16];
        let evicted = pool.insert_with_limits(over_byte_limit, 20, 2, 2);
        assert_eq!(evicted.iter().map(Vec::len).sum::<usize>(), 24);
        assert!(pool.bytes <= 20);
    }

    #[test]
    fn returned_buffers_keep_their_originating_device_until_fenced() {
        let returned = |device, marker| {
            PendingReturn::new(ReturnedBuffer {
                device,
                pending: PendingReturn::new((vec![marker], 1, PendingBytePermit { bytes: 0 })),
            })
        };
        let mut grouped =
            group_returned_by_device(vec![returned(1, 10), returned(0, 20), returned(1, 30)]);

        assert_eq!(grouped.keys().copied().collect::<Vec<_>>(), vec![0, 1]);
        let device_zero = grouped.remove(&0).unwrap();
        assert_eq!(device_zero.len(), 1);
        assert_eq!(device_zero.into_iter().next().unwrap().release().0, [20]);
        let device_one = grouped.remove(&1).unwrap();
        assert_eq!(
            device_one
                .into_iter()
                .map(|returned| returned.release().0[0])
                .collect::<Vec<_>>(),
            vec![10, 30]
        );
    }

    #[test]
    fn pending_byte_budget_admits_one_oversized_return_exclusively() {
        assert!(pending_bytes_fit(0, MAX_PENDING_BYTES));
        assert!(!pending_bytes_fit(1, MAX_PENDING_BYTES));
        assert!(!pending_bytes_fit(MAX_PENDING_BYTES, 1));
        assert!(pending_bytes_fit(0, MAX_PENDING_BYTES + 1));
        assert!(!pending_bytes_fit(1, MAX_PENDING_BYTES + 1));
    }
}
