//! Cross-segment buffer pool for rvr preflight execution.
//!
//! Every `execute_rvr_preflight` call needs hundreds of MB of scratch: the
//! per-chip inline record buffers, the program/memory logs, the touched-block
//! list, and the per-address-space timestamp shadows. Allocating them fresh
//! per segment makes the proving loop pay the first-touch fault + kernel
//! page-zeroing volume of the whole working set on every segment (a measured
//! dominant term of large-segment preflight wall time). This pool keeps the
//! allocations alive across calls: capacities are still re-derived per
//! segment (grow-only reuse), and the loud capacity / exact-consumption
//! invariants are unchanged.
//!
//! Reuse safety:
//! - The uninit-class buffers (logs, touched, record bytes) are write-only from the generated C and
//!   prefix-read by the host against C-advanced cursors, so stale bytes from a previous segment are
//!   never read — the same contract a fresh uninitialized allocation relies on. Debug runs poison
//!   recycled spares so any violation reads deterministic garbage instead of silently plausible
//!   previous-segment bytes.
//! - The shadows must be all-zero at segment start (0 = untouched this segment). They re-enter the
//!   pool only through [`RvrPreflightBufferPool::recycle_shadows`] after an exact O(touched) scrub,
//!   and debug runs re-verify all-zero on reuse.
//! - The counter arrays must likewise read as all-zero at segment start. Generated C records each
//!   counter's first 0→nonzero transition in a write-only touched-index side buffer; consumers
//!   scrub exactly that prefix before returning both buffers to the pool. Untouched counters are
//!   never written and remain zero, so reuse avoids both a full scan and a full-buffer memset.
//!
//! `OPENVM_RVR_PREFLIGHT_POOL=0|false|off` disables pooling (fresh
//! allocations per call, recycles drop). `OPENVM_RVR_ARENA_NATIVE_POOL`
//! independently gates arena-native backing reuse for controlled A/B runs.
//! `OPENVM_RVR_PREFLIGHT_THP` gates the `MADV_HUGEPAGE` advice on the large
//! buffers the same way. All default on.

use std::{
    any::Any,
    collections::{HashMap, HashSet},
    mem::{ManuallyDrop, MaybeUninit},
    sync::{Arc, LazyLock, Mutex, MutexGuard},
};

use openvm_instructions::riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS};
use openvm_stark_backend::p3_field::Field;

use super::{
    compile::env_flag_is_off,
    preflight::{
        DeltaMemoryLogEntry, DeviceProgramEntry, MemoryLogEntry, PreflightRawLogs, ProgramLogEntry,
        ProgramRunEntry, RvrInlineChipRecords, TouchedBlock,
    },
    preflight_normalizer::WORD_BYTES,
};
use crate::system::memory::merkle::public_values::PUBLIC_VALUES_AS;

/// Reusable per-segment preflight scratch buffers (see module docs).
///
/// One pool per preflight instance; the proving loop's cached executor and
/// the routed [`super::preflight::RvrPreflightInstance`] each own one, so the
/// buffers persist exactly as long as the compiled library they serve.
#[derive(Clone)]
pub struct RvrPreflightBufferPool {
    enabled: bool,
    arena_native_enabled: bool,
    // Boxed to keep the pool (and the instance/route types embedding it)
    // pointer-small; the pool is constructed once per compiled library.
    inner: Arc<Mutex<PoolInner>>,
}

#[derive(Default)]
struct PoolInner {
    program_log: Option<Vec<MaybeUninit<ProgramLogEntry>>>,
    program_runs: Option<Vec<MaybeUninit<ProgramRunEntry>>>,
    device_program_references: Option<Vec<MaybeUninit<DeviceProgramEntry>>>,
    memory_log: Option<Vec<MaybeUninit<MemoryLogEntry>>>,
    delta_memory_log: Option<Vec<MaybeUninit<DeltaMemoryLogEntry>>>,
    touched: Option<Vec<MaybeUninit<TouchedBlock>>>,
    /// Kept all-zero between segments (scrub-on-recycle).
    shadow_register: Option<Vec<u32>>,
    shadow_memory: Option<Vec<u32>>,
    shadow_public_values: Option<Vec<u32>>,
    /// All-zero between segments; only indices in the paired touched list are scrubbed on recycle.
    chip_counts: Option<Vec<u32>>,
    chip_counts_touched: Option<Vec<MaybeUninit<u32>>>,
    /// Program-defined instruction count is invariant for this executor. Computing it scans the
    /// whole (potentially sparse) program, so cache it with the pooled frequency table.
    exec_frequencies_len: Option<usize>,
    /// Returned after system trace generation consumes the program frequencies. Direct callers
    /// that retain multiple outputs simply allocate until those outputs are consumed.
    exec_frequencies: Option<Vec<u32>>,
    exec_frequencies_touched: Option<Vec<MaybeUninit<u32>>>,
    /// Huge-page-aligned virtual regions of the main-memory timestamp shadow that were explicitly
    /// collapsed after a prior segment touched them. Keyed by virtual address because the pooled
    /// shadow allocation is stable for the executor lifetime.
    shadow_memory_hugepages: HashSet<usize>,
    /// Failed collapse attempts by virtual region. A small retry budget tolerates transient kernel
    /// failures without issuing an unsupported hint on every later segment.
    shadow_memory_hugepage_failures: HashMap<usize, u8>,
    /// Nonempty initial samples used to discover and promote the recurring shadow working set.
    shadow_locality_training_samples: usize,
    /// Spare inline-record byte buffers, one per migrated chip in steady
    /// state (best-fit take, since per-air capacities differ).
    record_bufs: Vec<Vec<MaybeUninit<u8>>>,
    /// Stage-2 chronological backing. Kept separate because generated C
    /// requires a 32-byte-aligned interior pointer and CUDA builds source it
    /// from the page-locked arena pool.
    #[cfg(not(feature = "cuda"))]
    delta_backing: Option<Vec<u8>>,
    /// Direct-final compact backings keyed by AIR. These leave the pool while
    /// tracegen owns the DenseRecordArena and return from its Drop impl.
    wire_backings: HashMap<usize, Vec<u8>>,
    /// Arena-native matrix backings keyed by generated-C geometry and retained capacity. The field
    /// type is builder-specific, so keep it type-erased inside the likewise builder-local pool and
    /// downcast at the staging boundary.
    arena_native_matrix_backings: HashMap<ArenaNativeBackingKey, Box<dyn Any + Send>>,
    /// Arena-native dense backings on CPU. CUDA backings round-trip through the page-locked pool,
    /// whose cleaner also waits for outstanding device copies before making a backing reusable.
    arena_native_dense_backings: HashMap<ArenaNativeBackingKey, Vec<u8>>,
}

/// Identity of an arena-native generated-C target. Capacity participates in best-fit selection;
/// the descriptor still exposes the current segment's exact bound to C. The stride captures the
/// backing flavor's AIR geometry without depending on the generated-C metadata types here.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct ArenaNativeBackingKey {
    pub air: usize,
    pub stride_bytes: usize,
    pub capacity_bytes: usize,
}

impl ArenaNativeBackingKey {
    pub(crate) fn new(air: usize, stride_bytes: usize, capacity_bytes: usize) -> Self {
        Self {
            air,
            stride_bytes,
            capacity_bytes,
        }
    }
}

/// Steady-state spare bound for `record_bufs`; buffers round-trip 1:1 per
/// inline air (RV64IM has ~17 migrated AIRs), so this only trims
/// pathological accumulation, never the steady-state working set.
const MAX_RECORD_SPARES: usize = 32;

impl Default for RvrPreflightBufferPool {
    fn default() -> Self {
        Self::from_env()
    }
}

impl RvrPreflightBufferPool {
    pub fn from_env() -> Self {
        let enabled = !env_flag_is_off("OPENVM_RVR_PREFLIGHT_POOL");
        Self {
            enabled,
            arena_native_enabled: enabled && !env_flag_is_off("OPENVM_RVR_ARENA_NATIVE_POOL"),
            inner: Arc::new(Mutex::new(PoolInner::default())),
        }
    }

    fn lock(&self) -> MutexGuard<'_, PoolInner> {
        self.inner
            .lock()
            .expect("rvr preflight buffer pool poisoned")
    }

    pub(crate) fn take_program_log(&self, cap: usize) -> Vec<MaybeUninit<ProgramLogEntry>> {
        if !self.enabled {
            return vec_uninit(cap);
        }
        take_uninit_slot(&mut self.lock().program_log, cap)
    }

    pub(crate) fn take_program_runs(&self, cap: usize) -> Vec<MaybeUninit<ProgramRunEntry>> {
        if !self.enabled {
            return vec_uninit(cap);
        }
        take_uninit_slot(&mut self.lock().program_runs, cap)
    }

    pub(crate) fn take_device_program_references(
        &self,
        cap: usize,
    ) -> Vec<MaybeUninit<DeviceProgramEntry>> {
        if !self.enabled {
            return vec_uninit(cap);
        }
        take_uninit_slot(&mut self.lock().device_program_references, cap)
    }

    pub(crate) fn take_memory_log(&self, cap: usize) -> Vec<MaybeUninit<MemoryLogEntry>> {
        if !self.enabled {
            return vec_uninit(cap);
        }
        take_uninit_slot(&mut self.lock().memory_log, cap)
    }

    pub(crate) fn take_delta_memory_log(
        &self,
        cap: usize,
    ) -> Vec<MaybeUninit<DeltaMemoryLogEntry>> {
        if !self.enabled {
            return vec_uninit(cap);
        }
        take_uninit_slot(&mut self.lock().delta_memory_log, cap)
    }

    pub(crate) fn take_touched(&self, cap: usize) -> Vec<MaybeUninit<TouchedBlock>> {
        if !self.enabled {
            return vec_uninit(cap);
        }
        take_uninit_slot(&mut self.lock().touched, cap)
    }

    /// Take the two zero-valued counter tables plus write-only first-touch index buffers.
    /// Recycled tables are already zero because [`Self::recycle_chip_counts`] and
    /// [`Self::recycle_exec_frequencies`] scrub exactly the indices written in their segment.
    #[allow(clippy::type_complexity)]
    pub(crate) fn take_counters(
        &self,
        chip_counts_len: usize,
        compute_exec_frequencies_len: impl FnOnce() -> usize,
    ) -> (
        Vec<u32>,
        Vec<MaybeUninit<u32>>,
        Vec<u32>,
        Vec<MaybeUninit<u32>>,
    ) {
        if !self.enabled {
            let exec_frequencies_len = compute_exec_frequencies_len();
            return (
                vec![0; chip_counts_len],
                vec_uninit(chip_counts_len),
                vec![0; exec_frequencies_len],
                vec_uninit(exec_frequencies_len),
            );
        }
        let mut inner = self.lock();
        let exec_frequencies_len = match inner.exec_frequencies_len {
            Some(len) => len,
            None => {
                let len = compute_exec_frequencies_len();
                inner.exec_frequencies_len = Some(len);
                len
            }
        };
        (
            take_zero_counter_slot(&mut inner.chip_counts, chip_counts_len),
            take_uninit_slot(&mut inner.chip_counts_touched, chip_counts_len),
            take_zero_counter_slot(&mut inner.exec_frequencies, exec_frequencies_len),
            take_uninit_slot(&mut inner.exec_frequencies_touched, exec_frequencies_len),
        )
    }

    pub(crate) fn recycle_chip_counts(&self, values: Vec<u32>, touched: Vec<u32>) {
        if !self.enabled {
            return;
        }
        let mut inner = self.lock();
        let PoolInner {
            chip_counts,
            chip_counts_touched,
            ..
        } = &mut *inner;
        recycle_counter_slot(
            chip_counts,
            chip_counts_touched,
            values,
            touched,
            "chip_counts",
        );
    }

    pub(crate) fn recycle_exec_frequencies(&self, values: Vec<u32>, touched: Vec<u32>) {
        if !self.enabled {
            return;
        }
        let mut inner = self.lock();
        let PoolInner {
            exec_frequencies,
            exec_frequencies_touched,
            ..
        } = &mut *inner;
        recycle_counter_slot(
            exec_frequencies,
            exec_frequencies_touched,
            values,
            touched,
            "exec_frequencies",
        );
    }

    /// All three timestamp shadows, all-zero, sized to the given block counts
    /// (which are config-constant, so reuse is an exact-size match).
    pub(crate) fn take_shadows(
        &self,
        register_blocks: usize,
        memory_blocks: usize,
        public_values_blocks: usize,
    ) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
        if !self.enabled {
            return (
                vec_zeroed_advised(register_blocks),
                vec_zeroed_advised(memory_blocks),
                vec_zeroed_advised(public_values_blocks),
            );
        }
        let mut inner = self.lock();
        (
            take_shadow_slot(&mut inner.shadow_register, register_blocks),
            take_shadow_slot(&mut inner.shadow_memory, memory_blocks),
            take_shadow_slot(&mut inner.shadow_public_values, public_values_blocks),
        )
    }

    /// Best-fit spare with `capacity >= cap`, else a fresh allocation.
    pub(crate) fn take_record_buf(&self, cap: usize) -> Vec<MaybeUninit<u8>> {
        if !self.enabled {
            return vec_uninit(cap);
        }
        let mut inner = self.lock();
        let best = inner
            .record_bufs
            .iter()
            .enumerate()
            .filter(|(_, spare)| spare.capacity() >= cap)
            .min_by_key(|(_, spare)| spare.capacity())
            .map(|(idx, _)| idx);
        match best {
            Some(idx) => {
                let mut spare = inner.record_bufs.swap_remove(idx);
                // SAFETY: `MaybeUninit<u8>` requires no initialization and
                // `cap <= spare.capacity()` was just checked.
                unsafe { spare.set_len(cap) };
                spare
            }
            None => vec_uninit(cap),
        }
    }

    /// Take a backing with enough alignment slack for a `len`-byte delta
    /// stream. CUDA uses the #2990 page-locked pool; CPU keeps one grow-only
    /// allocation in this per-executor pool.
    /// Returns the delta backing and whether a fresh lazy allocation needs a
    /// one-time prefault. Recycled CPU and CUDA-pinned buffers are already
    /// resident; walking every page again would put scratch-pool maintenance
    /// back on each segment's preflight critical path.
    pub(crate) fn take_delta_backing(&self, len: usize) -> (Vec<u8>, bool) {
        let min_len = len.checked_add(31).expect("delta backing size overflow");
        if !self.enabled {
            return (vec![0u8; min_len], true);
        }
        #[cfg(feature = "cuda")]
        {
            crate::arch::cuda::pinned::take_with_prefault_status(min_len)
        }
        #[cfg(not(feature = "cuda"))]
        {
            match self.lock().delta_backing.take() {
                Some(backing) if backing.len() >= min_len => (backing, false),
                _ => (vec![0u8; min_len], true),
            }
        }
    }

    pub(crate) fn recycle_delta_backing(&self, backing: Vec<u8>, dirty_len: usize) {
        if !self.enabled {
            return;
        }
        #[cfg(feature = "cuda")]
        {
            crate::arch::cuda::pinned::give_back(backing, dirty_len);
        }
        #[cfg(not(feature = "cuda"))]
        {
            let mut inner = self.lock();
            match inner.delta_backing.as_ref() {
                Some(existing) if existing.len() >= backing.len() => {}
                _ => inner.delta_backing = Some(backing),
            }
            let _ = dirty_len;
        }
    }

    /// Return the shadows after the caller scrubbed them back to all-zero
    /// (see `scrub_shadows`); never call with a dirty shadow.
    pub(crate) fn recycle_shadows(&self, register: Vec<u32>, memory: Vec<u32>, pv: Vec<u32>) {
        if !self.enabled {
            return;
        }
        let mut inner = self.lock();
        inner.shadow_register = Some(register);
        inner.shadow_memory = Some(memory);
        inner.shadow_public_values = Some(pv);
    }

    /// Promote only the main-memory shadow regions observed in this segment to resident THPs.
    /// The generated DSO keeps its unchanged flat `shadow[block_idx]` ABI, while the following
    /// segment gets a page/block-local translation footprint. Promotion happens after the shadow
    /// is scrubbed, preserving the all-zero segment-start invariant and exact serial chronology.
    pub(crate) fn prepare_shadow_locality(
        &self,
        shadow_memory: &mut [u32],
        touched: &[TouchedBlock],
    ) {
        if !self.enabled || !*SHADOW_LOCALITY {
            return;
        }
        #[cfg(target_os = "linux")]
        {
            let allocation_start = shadow_memory.as_mut_ptr() as usize;
            let allocation_end = allocation_start + std::mem::size_of_val(shadow_memory);
            let interior_start = allocation_start.next_multiple_of(HUGE_PAGE_BYTES);
            let interior_end = allocation_end & !(HUGE_PAGE_BYTES - 1);
            if interior_end <= interior_start {
                return;
            }

            let mut regions = touched
                .iter()
                .filter(|block| block.addr_space == RV64_MEMORY_AS)
                .filter_map(|block| {
                    let block_idx = block.block_addr as usize / WORD_BYTES;
                    let byte_offset = block_idx.checked_mul(size_of::<u32>())?;
                    let slot = allocation_start.checked_add(byte_offset)?;
                    let region = slot & !(HUGE_PAGE_BYTES - 1);
                    (region >= interior_start && region < interior_end).then_some(region)
                })
                .collect::<Vec<_>>();
            regions.sort_unstable();
            regions.dedup();
            if regions.is_empty() {
                return;
            }
            {
                let mut inner = self.lock();
                let training =
                    inner.shadow_locality_training_samples < SHADOW_LOCALITY_TRAINING_SAMPLES;
                if training {
                    inner.shadow_locality_training_samples += 1;
                }
                regions.retain(|region| {
                    let failures = inner
                        .shadow_memory_hugepage_failures
                        .get(region)
                        .copied()
                        .unwrap_or(0);
                    !inner.shadow_memory_hugepages.contains(region)
                        && failures < SHADOW_LOCALITY_MAX_ATTEMPTS
                        && (training || failures != 0)
                });
            }

            let mut promoted = Vec::with_capacity(regions.len());
            let mut failed = Vec::new();
            for region in regions {
                // SAFETY: every selected 2 MiB range is page-aligned and lies wholly within the
                // live shadow allocation. MADV_COLLAPSE preserves contents and mapping validity.
                let rc = unsafe {
                    libc::madvise(
                        region as *mut libc::c_void,
                        HUGE_PAGE_BYTES,
                        libc::MADV_COLLAPSE,
                    )
                };
                if rc == 0 {
                    promoted.push(region);
                } else {
                    failed.push(region);
                }
            }
            if !promoted.is_empty() || !failed.is_empty() {
                let mut inner = self.lock();
                for region in promoted {
                    inner.shadow_memory_hugepage_failures.remove(&region);
                    inner.shadow_memory_hugepages.insert(region);
                }
                for region in failed {
                    *inner
                        .shadow_memory_hugepage_failures
                        .entry(region)
                        .or_default() += 1;
                }
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (shadow_memory, touched);
        }
    }

    pub(crate) fn recycle_touched(&self, touched: Vec<TouchedBlock>) {
        if !self.enabled {
            return;
        }
        recycle_uninit_slot(&mut self.lock().touched, into_uninit_spare(touched));
    }

    /// Retry-path recycle of the still-uninit log/touched buffers (the pool
    /// drops any spare smaller than the next, grown take).
    pub(crate) fn recycle_raw_uninit(
        &self,
        program_log: Vec<MaybeUninit<ProgramLogEntry>>,
        program_runs: Vec<MaybeUninit<ProgramRunEntry>>,
        device_program_references: Vec<MaybeUninit<DeviceProgramEntry>>,
        memory_log: Vec<MaybeUninit<MemoryLogEntry>>,
        delta_memory_log: Vec<MaybeUninit<DeltaMemoryLogEntry>>,
        touched: Vec<MaybeUninit<TouchedBlock>>,
    ) {
        if !self.enabled {
            return;
        }
        let mut inner = self.lock();
        recycle_uninit_slot(&mut inner.program_log, program_log);
        recycle_uninit_slot(&mut inner.program_runs, program_runs);
        recycle_uninit_slot(
            &mut inner.device_program_references,
            device_program_references,
        );
        recycle_uninit_slot(&mut inner.memory_log, memory_log);
        recycle_uninit_slot(&mut inner.delta_memory_log, delta_memory_log);
        recycle_uninit_slot(&mut inner.touched, touched);
    }

    pub(crate) fn recycle_record_buf(&self, buf: Vec<MaybeUninit<u8>>) {
        if !self.enabled {
            return;
        }
        push_record_spare(&mut self.lock().record_bufs, buf);
    }

    /// Take the smallest resident matrix backing that fits this shape. Recycled values are
    /// all-zero, matching
    /// `MatrixRecordArena::with_capacity`; this is required because generated arena-native code
    /// intentionally leaves some suppressed-write fields and padding untouched.
    pub(crate) fn take_arena_native_matrix_backing<F>(
        &self,
        key: ArenaNativeBackingKey,
    ) -> Option<Vec<F>>
    where
        F: Field + Send + 'static,
    {
        if !self.arena_native_enabled {
            return None;
        }
        let mut inner = self.lock();
        let pooled_key = inner
            .arena_native_matrix_backings
            .keys()
            .filter(|candidate| {
                candidate.air == key.air
                    && candidate.stride_bytes == key.stride_bytes
                    && candidate.capacity_bytes >= key.capacity_bytes
            })
            .min_by_key(|candidate| candidate.capacity_bytes)
            .copied()?;
        let erased = inner
            .arena_native_matrix_backings
            .remove(&pooled_key)
            .expect("selected arena-native matrix backing disappeared");
        drop(inner);
        let backing = erased.downcast::<Vec<F>>().unwrap_or_else(|_| {
            panic!(
                "arena-native matrix backing type mismatch for air {}",
                key.air
            )
        });
        let backing = *backing;
        assert_eq!(
            backing.len() * size_of::<F>(),
            pooled_key.capacity_bytes,
            "arena-native matrix backing capacity mismatch for air {}",
            key.air
        );
        Some(backing)
    }

    /// Scrub and retain a matrix backing after its consumer releases the arena. Clearing the full
    /// allocation (used prefix plus padding) makes reuse byte-identical to a fresh zeroed arena.
    pub(crate) fn recycle_arena_native_matrix_backing<F>(
        &self,
        key: ArenaNativeBackingKey,
        mut backing: Vec<F>,
    ) where
        F: Field + Send + 'static,
    {
        if !self.arena_native_enabled {
            return;
        }
        assert_eq!(
            backing.len() * size_of::<F>(),
            key.capacity_bytes,
            "arena-native matrix backing capacity mismatch for air {}",
            key.air
        );
        backing.fill(F::ZERO);
        let mut inner = self.lock();
        if inner.arena_native_matrix_backings.keys().any(|existing| {
            existing.air == key.air
                && existing.stride_bytes == key.stride_bytes
                && existing.capacity_bytes >= key.capacity_bytes
        }) {
            return;
        }
        inner.arena_native_matrix_backings.retain(|existing, _| {
            existing.air != key.air
                || existing.stride_bytes != key.stride_bytes
                || existing.capacity_bytes > key.capacity_bytes
        });
        inner
            .arena_native_matrix_backings
            .insert(key, Box::new(backing));
    }

    /// Take the smallest resident dense backing that fits this shape. CPU uses this executor-local
    /// map; CUDA uses the
    /// existing page-locked size-class pool so its cleaner can enforce the asynchronous H2D
    /// lifetime before a backing is handed out again.
    pub(crate) fn take_arena_native_dense_backing(
        &self,
        key: ArenaNativeBackingKey,
    ) -> Option<Vec<u8>> {
        if !self.arena_native_enabled {
            return None;
        }
        #[cfg(feature = "cuda")]
        {
            Some(crate::arch::cuda::pinned::take(key.capacity_bytes + 32))
        }
        #[cfg(not(feature = "cuda"))]
        {
            let mut inner = self.lock();
            let pooled_key = inner
                .arena_native_dense_backings
                .keys()
                .filter(|candidate| {
                    candidate.air == key.air
                        && candidate.stride_bytes == key.stride_bytes
                        && candidate.capacity_bytes >= key.capacity_bytes
                })
                .min_by_key(|candidate| candidate.capacity_bytes)
                .copied()?;
            let backing = inner
                .arena_native_dense_backings
                .remove(&pooled_key)
                .expect("selected arena-native dense backing disappeared");
            assert!(
                backing.len() >= key.capacity_bytes + 32,
                "arena-native dense backing capacity mismatch for air {}",
                key.air
            );
            Some(backing)
        }
    }

    /// Return a dense backing only after its arena consumer is done. CPU clears the entire backing
    /// synchronously. CUDA delegates the same scrub to the pinned cleaner after recording the
    /// outstanding copy lifetime.
    pub(crate) fn recycle_arena_native_dense_backing(
        &self,
        key: ArenaNativeBackingKey,
        backing: Vec<u8>,
        dirty_len: usize,
    ) {
        if !self.arena_native_enabled {
            return;
        }
        assert!(
            backing.len() >= key.capacity_bytes + 32,
            "arena-native dense backing capacity mismatch for air {}",
            key.air
        );
        #[cfg(feature = "cuda")]
        {
            crate::arch::cuda::pinned::give_back(backing, dirty_len);
        }
        #[cfg(not(feature = "cuda"))]
        {
            let mut backing = backing;
            backing.fill(0);
            let mut inner = self.lock();
            if inner.arena_native_dense_backings.keys().any(|existing| {
                existing.air == key.air
                    && existing.stride_bytes == key.stride_bytes
                    && existing.capacity_bytes >= key.capacity_bytes
            }) {
                return;
            }
            inner.arena_native_dense_backings.retain(|existing, _| {
                existing.air != key.air
                    || existing.stride_bytes != key.stride_bytes
                    || existing.capacity_bytes > key.capacity_bytes
            });
            inner.arena_native_dense_backings.insert(key, backing);
            let _ = dirty_len;
        }
    }

    /// Allocate and fault in a direct-final wire backing before the segment's
    /// preflight clock starts. Repeated calls are grow-only no-ops.
    pub fn prepare_wire_backing(&self, air: usize, len: usize) {
        if !self.enabled {
            return;
        }
        let mut inner = self.lock();
        if inner
            .wire_backings
            .get(&air)
            .is_some_and(|backing| backing.len() >= len + 32)
        {
            return;
        }
        #[cfg(feature = "cuda")]
        let mut backing = crate::arch::cuda::pinned::take(len + 32);
        #[cfg(not(feature = "cuda"))]
        let mut backing = vec![0u8; len + 32];
        advise_hugepages(backing.as_ptr(), backing.len());
        // `vec![0]` may be a lazy calloc mapping. A volatile byte per page
        // makes the fault cost unambiguously one-time and outside preflight.
        for page in (0..backing.len()).step_by(4096) {
            unsafe { std::ptr::write_volatile(backing.as_mut_ptr().add(page), 0) };
        }
        if let Some(last) = backing.last_mut() {
            unsafe { std::ptr::write_volatile(last, 0) };
        }
        inner.wire_backings.insert(air, backing);
    }

    pub(crate) fn take_wire_backing(&self, air: usize, len: usize) -> Option<Vec<u8>> {
        if !self.enabled {
            return None;
        }
        let backing = self.lock().wire_backings.remove(&air)?;
        assert!(
            backing.len() >= len + 32,
            "prepared wire backing for air {air} is too small"
        );
        Some(backing)
    }

    pub(crate) fn recycle_wire_backing(&self, air: usize, backing: Vec<u8>, _dirty_len: usize) {
        if !self.enabled {
            return;
        }
        #[cfg(feature = "cuda")]
        {
            // The pinned pool's cleaner waits for outstanding H2D work,
            // clears the dirty backing off the preflight path, and returns a
            // registered size-class buffer for the next prepared segment.
            crate::arch::cuda::pinned::give_back(backing, _dirty_len);
            let _ = air;
        }
        #[cfg(not(feature = "cuda"))]
        {
            let mut inner = self.lock();
            match inner.wire_backings.get(&air) {
                Some(existing) if existing.len() >= backing.len() => {}
                _ => {
                    inner.wire_backings.insert(air, backing);
                }
            }
        }
    }

    /// Return a consumed segment output's escaping buffers (the raw logs and
    /// the inline compact record bytes) to the pool. Callers invoke this once
    /// the output's payload has been moved out / the records assembled; the
    /// arenas hold expanded copies, so the compact bytes are dead here.
    pub fn recycle_segment_buffers(
        &self,
        raw_logs: PreflightRawLogs,
        inline_records: Vec<RvrInlineChipRecords>,
        delta_records: Option<super::preflight::RvrDeltaRecords>,
    ) {
        if !self.enabled {
            return;
        }
        let PreflightRawLogs {
            program_log,
            program_runs,
            device_program_references,
            memory_log,
            delta_memory_log,
            chip_counts,
            chip_counts_touched,
            touched,
            device_aux_patches: _,
            device_aux_references: _,
            device_aux_arena_references: _,
        } = raw_logs;
        let mut inner = self.lock();
        let PoolInner {
            chip_counts: pooled_chip_counts,
            chip_counts_touched: pooled_chip_counts_touched,
            ..
        } = &mut *inner;
        recycle_counter_slot(
            pooled_chip_counts,
            pooled_chip_counts_touched,
            chip_counts,
            chip_counts_touched,
            "chip_counts",
        );
        recycle_uninit_slot(&mut inner.program_log, into_uninit_spare(program_log));
        recycle_uninit_slot(&mut inner.program_runs, into_uninit_spare(program_runs));
        recycle_uninit_slot(
            &mut inner.device_program_references,
            into_uninit_spare(device_program_references),
        );
        recycle_uninit_slot(&mut inner.memory_log, into_uninit_spare(memory_log));
        recycle_uninit_slot(
            &mut inner.delta_memory_log,
            into_uninit_spare(delta_memory_log),
        );
        recycle_uninit_slot(&mut inner.touched, into_uninit_spare(touched));
        for chip in inline_records {
            push_record_spare(&mut inner.record_bufs, into_uninit_spare(chip.bytes));
        }
        drop(inner);
        drop(delta_records);
    }
}

/// Zero every shadow slot recorded in `touched`, restoring the all-zero
/// segment-start state for pool reuse. Exact by the tracer contract: the C
/// `preflight_touch` is the single shadow-write site, appends one
/// `TouchedBlock` per 0→nonzero transition, and timestamps are never 0 — so
/// {nonzero shadow slots} = {touched entries}. The address-space dispatch and
/// `block_addr / WORD_BYTES` indexing mirror `PreflightShadowsView`.
pub(crate) fn scrub_shadows(
    shadow_register: &mut [u32],
    shadow_memory: &mut [u32],
    shadow_public_values: &mut [u32],
    touched: &[TouchedBlock],
) {
    for tb in touched {
        let block_idx = tb.block_addr as usize / WORD_BYTES;
        let shadow: &mut [u32] = if tb.addr_space == RV64_REGISTER_AS {
            &mut *shadow_register
        } else if tb.addr_space == PUBLIC_VALUES_AS {
            &mut *shadow_public_values
        } else {
            &mut *shadow_memory
        };
        debug_assert!(
            block_idx < shadow.len(),
            "touched block ({}, {:#x}) outside its shadow",
            tb.addr_space,
            tb.block_addr
        );
        if let Some(slot) = shadow.get_mut(block_idx) {
            *slot = 0;
        }
    }
}

/// Debug pool-poisoning checks (verify-zero on shadow reuse, poison recycled
/// uninit spares). Both are O(buffer capacity) per segment, so the standing
/// `OPENVM_SKIP_DEBUG=1` fast-run knob disables them like other debug-only
/// checking.
static POOL_DEBUG_CHECKS: LazyLock<bool> = LazyLock::new(|| {
    cfg!(debug_assertions) && std::env::var("OPENVM_SKIP_DEBUG").as_deref() != Ok("1")
});

const POOL_POISON_BYTE: u8 = 0xA5;

fn take_uninit_slot<T>(slot: &mut Option<Vec<MaybeUninit<T>>>, cap: usize) -> Vec<MaybeUninit<T>> {
    match slot.take() {
        Some(mut spare) if spare.capacity() >= cap => {
            // SAFETY: `MaybeUninit<T>` requires no initialization and the
            // capacity bound was just checked.
            unsafe { spare.set_len(cap) };
            spare
        }
        // Too small (or empty slot): allocate fresh, dropping any spare —
        // capacities are grow-only per pool, so the bigger buffer round-trips
        // back and wins the slot.
        _ => vec_uninit(cap),
    }
}

fn recycle_uninit_slot<T>(slot: &mut Option<Vec<MaybeUninit<T>>>, mut buf: Vec<MaybeUninit<T>>) {
    if matches!(slot, Some(existing) if existing.capacity() >= buf.capacity()) {
        return;
    }
    // SAFETY: `MaybeUninit<T>` requires no initialization; restore the full
    // allocation length so future takes shrink from the top.
    unsafe { buf.set_len(buf.capacity()) };
    poison_spare(&mut buf);
    *slot = Some(buf);
}

fn take_shadow_slot(slot: &mut Option<Vec<u32>>, len: usize) -> Vec<u32> {
    match slot.take() {
        Some(spare) if spare.len() == len => {
            if *POOL_DEBUG_CHECKS {
                assert!(
                    spare.iter().all(|&ts| ts == 0),
                    "pooled timestamp shadow not scrubbed to zero — pool poisoning"
                );
            }
            spare
        }
        _ => vec_zeroed_advised(len),
    }
}

fn take_zero_counter_slot(slot: &mut Option<Vec<u32>>, len: usize) -> Vec<u32> {
    match slot.take() {
        Some(mut spare) if spare.capacity() >= len => {
            // SAFETY: every element in the allocation was initialized when it was first created;
            // recycling only changes initialized u32 values back to zero.
            unsafe { spare.set_len(len) };
            if *POOL_DEBUG_CHECKS {
                assert!(
                    spare.iter().all(|&value| value == 0),
                    "pooled counter table not scrubbed to zero"
                );
            }
            spare
        }
        _ => vec_zeroed_advised(len),
    }
}

fn recycle_counter_slot(
    values_slot: &mut Option<Vec<u32>>,
    touched_slot: &mut Option<Vec<MaybeUninit<u32>>>,
    mut values: Vec<u32>,
    touched: Vec<u32>,
    label: &str,
) {
    let values_len = values.len();
    for &index in &touched {
        let slot = values
            .get_mut(index as usize)
            .unwrap_or_else(|| panic!("{label} touched index {index} outside length {values_len}"));
        *slot = 0;
    }
    if *POOL_DEBUG_CHECKS {
        assert!(
            values.iter().all(|&value| value == 0),
            "{label} first-touch list did not cover every nonzero counter"
        );
    }
    if !matches!(values_slot, Some(existing) if existing.capacity() >= values.capacity()) {
        *values_slot = Some(values);
    }
    recycle_uninit_slot(touched_slot, into_uninit_spare(touched));
}

fn push_record_spare(spares: &mut Vec<Vec<MaybeUninit<u8>>>, mut buf: Vec<MaybeUninit<u8>>) {
    // SAFETY: `MaybeUninit<u8>` requires no initialization.
    unsafe { buf.set_len(buf.capacity()) };
    poison_spare(&mut buf);
    spares.push(buf);
    if spares.len() > MAX_RECORD_SPARES {
        let smallest = spares
            .iter()
            .enumerate()
            .min_by_key(|(_, spare)| spare.capacity())
            .map(|(idx, _)| idx)
            .expect("spares is non-empty");
        spares.swap_remove(smallest);
    }
}

fn poison_spare<T>(buf: &mut Vec<MaybeUninit<T>>) {
    if !*POOL_DEBUG_CHECKS {
        return;
    }
    // SAFETY: writing the poison pattern over the full owned allocation;
    // `MaybeUninit<T>` permits any byte content.
    unsafe {
        std::ptr::write_bytes(
            buf.as_mut_ptr().cast::<u8>(),
            POOL_POISON_BYTE,
            buf.len() * size_of::<T>(),
        );
    }
}

/// Reclaims an escaped, initialized buffer as an uninit spare, restoring the
/// full allocation length. `T: Copy` bounds out destructors, so dropping the
/// initialized prefix as `MaybeUninit` is a no-op.
fn into_uninit_spare<T: Copy>(v: Vec<T>) -> Vec<MaybeUninit<T>> {
    let mut v = ManuallyDrop::new(v);
    let (ptr, cap) = (v.as_mut_ptr(), v.capacity());
    // SAFETY: same allocation and layout (`MaybeUninit<T>` is layout-identical
    // to `T`); `len = cap` marks every element possibly-uninit, which is
    // vacuously sound.
    unsafe { Vec::from_raw_parts(ptr.cast::<MaybeUninit<T>>(), cap, cap) }
}

/// Allocates an uninitialized buffer for an external (C) writer, advising
/// transparent huge pages for large capacities: the record/log streams fault
/// in hundreds of MB of fresh pages per segment, and 2 MB mappings cut the
/// first-touch fault count (a measured term of the large-segment preflight
/// cost — though with `defrag=madvise` the kernel may still fall back to 4 KB
/// pages under fragmentation, which is what makes pooled reuse the reliable
/// fix).
pub(crate) fn vec_uninit<T>(cap: usize) -> Vec<MaybeUninit<T>> {
    let mut buffer: Vec<MaybeUninit<T>> = Vec::with_capacity(cap);
    // SAFETY: `MaybeUninit<T>` requires no initialization.
    unsafe { buffer.set_len(cap) };
    advise_hugepages(buffer.as_ptr().cast(), cap * size_of::<T>());
    buffer
}

/// Zero-initialized allocation with the same huge-page advice as
/// [`vec_uninit`] (the advice lands before first touch, so it still steers
/// fault-time THP).
fn vec_zeroed_advised(len: usize) -> Vec<u32> {
    let buffer = vec![0u32; len];
    advise_hugepages(buffer.as_ptr().cast(), len * size_of::<u32>());
    buffer
}

const HUGE_PAGE_BYTES: usize = 2 * 1024 * 1024;
const SHADOW_LOCALITY_TRAINING_SAMPLES: usize = 5;
const SHADOW_LOCALITY_MAX_ATTEMPTS: u8 = 2;

static THP_ADVICE: LazyLock<bool> = LazyLock::new(|| !env_flag_is_off("OPENVM_RVR_PREFLIGHT_THP"));

/// Target-2 kill switch. Default-on after validation; `0|false|off` restores ordinary sparse
/// 4 KiB-backed timestamp shadows for same-process-shape A/B measurements.
static SHADOW_LOCALITY: LazyLock<bool> =
    LazyLock::new(|| !env_flag_is_off("OPENVM_RVR_SHADOW_LOCALITY"));

/// Best-effort `MADV_HUGEPAGE` over the page-aligned interior of a buffer.
/// Purely a performance hint; failures are ignored.
fn advise_hugepages(ptr: *const u8, len: usize) {
    #[cfg(target_os = "linux")]
    {
        if len < HUGE_PAGE_BYTES || !*THP_ADVICE {
            return;
        }
        let page = 4096usize;
        let start = (ptr as usize).next_multiple_of(page);
        let end = (ptr as usize + len) & !(page - 1);
        if end > start {
            // SAFETY: the advised range lies within the owned allocation;
            // MADV_HUGEPAGE does not alter memory contents or validity.
            unsafe {
                libc::madvise(start as *mut libc::c_void, end - start, libc::MADV_HUGEPAGE);
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (ptr, len);
    }
}

#[cfg(all(test, not(feature = "cuda")))]
mod tests {
    use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
    use openvm_stark_sdk::p3_baby_bear::BabyBear;

    use super::*;

    #[test]
    fn recycled_delta_backing_skips_prefault() {
        let pool = RvrPreflightBufferPool {
            enabled: true,
            arena_native_enabled: true,
            inner: Arc::new(Mutex::new(PoolInner::default())),
        };
        let (first, first_needs_prefault) = pool.take_delta_backing(4096);
        assert!(first_needs_prefault);
        let first_ptr = first.as_ptr();
        pool.recycle_delta_backing(first, 0);

        let (recycled, recycled_needs_prefault) = pool.take_delta_backing(2048);
        assert!(!recycled_needs_prefault);
        assert_eq!(recycled.as_ptr(), first_ptr);
    }

    #[test]
    fn empty_shadow_locality_sample_does_not_consume_training_budget() {
        let pool = RvrPreflightBufferPool {
            enabled: true,
            arena_native_enabled: true,
            inner: Arc::new(Mutex::new(PoolInner::default())),
        };
        let mut shadow = Vec::new();
        pool.prepare_shadow_locality(&mut shadow, &[]);
        assert_eq!(pool.lock().shadow_locality_training_samples, 0);
    }

    #[test]
    fn arena_native_matrix_backing_is_reused_and_fully_zeroed() {
        let pool = RvrPreflightBufferPool {
            enabled: true,
            arena_native_enabled: true,
            inner: Arc::new(Mutex::new(PoolInner::default())),
        };
        let key = ArenaNativeBackingKey::new(7, 16, 4096);
        let mut first = crate::arch::MatrixRecordArena::<BabyBear>::with_recycled_rvr_capacity(
            256,
            4,
            key,
            pool.clone(),
        );
        let first_ptr = first.trace_buffer.as_ptr();
        first.trace_buffer.fill(BabyBear::ONE);
        drop(first);

        let smaller_key = ArenaNativeBackingKey::new(7, 16, 2048);
        let recycled = crate::arch::MatrixRecordArena::<BabyBear>::with_recycled_rvr_capacity(
            128,
            4,
            smaller_key,
            pool,
        );
        assert_eq!(recycled.trace_buffer.as_ptr(), first_ptr);
        assert!(recycled
            .trace_buffer
            .iter()
            .all(|&value| value == BabyBear::ZERO));
    }

    #[test]
    fn arena_native_dense_backing_is_reused_and_fully_zeroed() {
        let pool = RvrPreflightBufferPool {
            enabled: true,
            arena_native_enabled: true,
            inner: Arc::new(Mutex::new(PoolInner::default())),
        };
        let key = ArenaNativeBackingKey::new(11, 64, 4096);
        let mut first = crate::arch::DenseRecordArena::with_recycled_rvr_arena_native_capacity(
            4096,
            key,
            pool.clone(),
        );
        let first_ptr = first.records_buffer.get_ref().as_ptr();
        first.records_buffer.get_mut().fill(0xa5);
        first.records_buffer.set_position(4096);
        drop(first);

        let smaller_key = ArenaNativeBackingKey::new(11, 64, 2048);
        let recycled = crate::arch::DenseRecordArena::with_recycled_rvr_arena_native_capacity(
            2048,
            smaller_key,
            pool,
        );
        assert_eq!(recycled.records_buffer.get_ref().as_ptr(), first_ptr);
        assert!(recycled
            .records_buffer
            .get_ref()
            .iter()
            .all(|&byte| byte == 0));
    }
}
