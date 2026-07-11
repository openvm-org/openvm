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
//!
//! `OPENVM_RVR_PREFLIGHT_POOL=0|false|off` disables pooling (fresh
//! allocations per call, recycles drop). `OPENVM_RVR_PREFLIGHT_THP` gates the
//! `MADV_HUGEPAGE` advice on the large buffers the same way. Both default on.

use std::{
    mem::{ManuallyDrop, MaybeUninit},
    sync::{LazyLock, Mutex, MutexGuard},
};

use openvm_instructions::riscv::RV64_REGISTER_AS;

use super::{
    compile::env_flag_is_off,
    preflight::{
        MemoryLogEntry, PreflightRawLogs, ProgramLogEntry, RvrInlineChipRecords, TouchedBlock,
    },
    preflight_normalizer::WORD_BYTES,
};
use crate::system::memory::merkle::public_values::PUBLIC_VALUES_AS;

/// Reusable per-segment preflight scratch buffers (see module docs).
///
/// One pool per preflight instance; the proving loop's cached executor and
/// the routed [`super::preflight::RvrPreflightInstance`] each own one, so the
/// buffers persist exactly as long as the compiled library they serve.
pub struct RvrPreflightBufferPool {
    enabled: bool,
    // Boxed to keep the pool (and the instance/route types embedding it)
    // pointer-small; the pool is constructed once per compiled library.
    inner: Box<Mutex<PoolInner>>,
}

#[derive(Default)]
struct PoolInner {
    program_log: Option<Vec<MaybeUninit<ProgramLogEntry>>>,
    memory_log: Option<Vec<MaybeUninit<MemoryLogEntry>>>,
    touched: Option<Vec<MaybeUninit<TouchedBlock>>>,
    /// Kept all-zero between segments (scrub-on-recycle).
    shadow_register: Option<Vec<u32>>,
    shadow_memory: Option<Vec<u32>>,
    shadow_public_values: Option<Vec<u32>>,
    /// Spare inline-record byte buffers, one per migrated chip in steady
    /// state (best-fit take, since per-air capacities differ).
    record_bufs: Vec<Vec<MaybeUninit<u8>>>,
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
        Self {
            enabled: !env_flag_is_off("OPENVM_RVR_PREFLIGHT_POOL"),
            inner: Box::new(Mutex::new(PoolInner::default())),
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

    pub(crate) fn take_memory_log(&self, cap: usize) -> Vec<MaybeUninit<MemoryLogEntry>> {
        if !self.enabled {
            return vec_uninit(cap);
        }
        take_uninit_slot(&mut self.lock().memory_log, cap)
    }

    pub(crate) fn take_touched(&self, cap: usize) -> Vec<MaybeUninit<TouchedBlock>> {
        if !self.enabled {
            return vec_uninit(cap);
        }
        take_uninit_slot(&mut self.lock().touched, cap)
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
        memory_log: Vec<MaybeUninit<MemoryLogEntry>>,
        touched: Vec<MaybeUninit<TouchedBlock>>,
    ) {
        if !self.enabled {
            return;
        }
        let mut inner = self.lock();
        recycle_uninit_slot(&mut inner.program_log, program_log);
        recycle_uninit_slot(&mut inner.memory_log, memory_log);
        recycle_uninit_slot(&mut inner.touched, touched);
    }

    pub(crate) fn recycle_record_buf(&self, buf: Vec<MaybeUninit<u8>>) {
        if !self.enabled {
            return;
        }
        push_record_spare(&mut self.lock().record_bufs, buf);
    }

    /// Return a consumed segment output's escaping buffers (the raw logs and
    /// the inline compact record bytes) to the pool. Callers invoke this once
    /// the output's payload has been moved out / the records assembled; the
    /// arenas hold expanded copies, so the compact bytes are dead here.
    pub fn recycle_segment_buffers(
        &self,
        raw_logs: PreflightRawLogs,
        inline_records: Vec<RvrInlineChipRecords>,
    ) {
        if !self.enabled {
            return;
        }
        let PreflightRawLogs {
            program_log,
            memory_log,
            chip_counts: _,
        } = raw_logs;
        let mut inner = self.lock();
        recycle_uninit_slot(&mut inner.program_log, into_uninit_spare(program_log));
        recycle_uninit_slot(&mut inner.memory_log, into_uninit_spare(memory_log));
        for chip in inline_records {
            push_record_spare(&mut inner.record_bufs, into_uninit_spare(chip.bytes));
        }
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

static THP_ADVICE: LazyLock<bool> = LazyLock::new(|| !env_flag_is_off("OPENVM_RVR_PREFLIGHT_THP"));

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
