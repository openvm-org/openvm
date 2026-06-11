//! Per-chip metered execution: page tracking and segmentation
//! matching OpenVM's `MeteredCtx`.

use std::{
    ffi::c_void,
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::Arc,
};

use openvm_instructions::{
    exe::VmExe,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
    DEFERRAL_AS,
};
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm::{DEFERRAL_PAGE_BUF_CAP, MEM_PAGE_BUF_CAP, PV_PAGE_BUF_CAP};
use rvr_openvm_lift::ExtensionRegistry;

use super::{
    bridge::map_rvr_execute_error,
    execute_metered, execute_metered_segment_boundary,
    state::{TracerPayload, TracerPtr},
    RvrCompiled,
};
#[cfg(feature = "metrics")]
use crate::arch::execution_metrics::{ExecutionMetric, ExecutionMetricTimer};
use crate::{
    arch::{
        execution_mode::{
            metered::{
                ctx::{MeteredCtxParts, DEFAULT_PAGE_BITS},
                memory_ctx::MemoryCtx,
                segment_ctx::{Segment, SegmentationCtx},
            },
            MeteredCtx,
        },
        ExecutionError, Streams, SystemConfig, VmState, BOUNDARY_AIR_ID, MERKLE_AIR_ID,
        U16_CELL_SIZE_BITS,
    },
    system::memory::{
        merkle::public_values::PUBLIC_VALUES_AS, online::GuestMemory, DIGEST_WIDTH_BITS,
    },
};

pub struct RunToCompletion;

pub struct SegmentBoundary;

pub type RvrMeteredInstance<'a, F> = RvrMeteredInstanceWith<'a, F, RunToCompletion>;

pub type RvrMeteredSegmentInstance<'a, F> = RvrMeteredInstanceWith<'a, F, SegmentBoundary>;

pub struct RvrMeteredInstanceWith<'a, F: PrimeField32, S> {
    pub(crate) system_config: &'a SystemConfig,
    pub(crate) exe: Arc<VmExe<F>>,
    pub(crate) extensions: ExtensionRegistry<F>,
    pub(crate) compiled: RvrCompiled,
    pub(crate) _mode: PhantomData<S>,
}

static_assertions::assert_impl_all!(RvrMeteredInstance<'static, p3_baby_bear::BabyBear>: Send, Sync);
static_assertions::assert_impl_all!(
    RvrMeteredSegmentInstance<'static, p3_baby_bear::BabyBear>: Send,
    Sync
);

pub struct RvrMeteredResult {
    pub seg_state: SegmentationState,
    pub suspended: bool,
    pub exit_code: Option<u32>,
}

// ── C-compatible tracer struct ───────────────────────────────────────────────

/// C-compatible metered tracer data.
///
/// Layout must exactly match the C `Tracer` struct in `openvm_tracer_metered.h`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MeteredTracerData {
    pub trace_heights: *mut u32,
    pub mem_page_buf: *mut u32,
    pub pv_page_buf: *mut u32,
    pub deferral_page_buf: *mut u32,
    /// Periodic-check callback. Always initialized; generated C calls it
    /// unconditionally to keep the hot metered path branch-free.
    pub on_check: unsafe extern "C" fn(*mut MeteredTracerData) -> u8,
    pub seg_state: *mut c_void,
    pub mem_page_buf_len: u32,
    pub pv_page_buf_len: u32,
    pub deferral_page_buf_len: u32,
    pub check_counter: u32,
    /// Dedup cache for AS_MEMORY pages. `u32::MAX` = none. Reset on flush.
    pub last_mem_page: u32,
}

/// Sentinel indicating no last-seen page (matches `NO_LAST_PAGE` in C).
pub const NO_LAST_PAGE: u32 = u32::MAX;

impl Default for MeteredTracerData {
    fn default() -> Self {
        Self {
            trace_heights: std::ptr::null_mut(),
            mem_page_buf: std::ptr::null_mut(),
            pv_page_buf: std::ptr::null_mut(),
            deferral_page_buf: std::ptr::null_mut(),
            on_check: metered_periodic_check,
            seg_state: std::ptr::null_mut(),
            mem_page_buf_len: 0,
            pv_page_buf_len: 0,
            deferral_page_buf_len: 0,
            check_counter: 0,
            last_mem_page: NO_LAST_PAGE,
        }
    }
}

impl TracerPayload for MeteredTracerData {
    const KIND: u32 = 11;
}

pub type MeteredTracer = TracerPtr<MeteredTracerData>;

// ── Segmentation runtime ─────────────────────────────────────────────────────

// TODO: generalize non-memory page buffers to a config-driven set of N
// buffers (one per additional address space) instead of hardcoding
// pv + deferral. The memory AS buffer stays separate as the hot path.
pub struct SegmentationState {
    pub segmentation_ctx: SegmentationCtx,
    trace_heights: Vec<u32>,
    is_trace_height_constant: Vec<bool>,
    memory_ctx: MemoryCtx<DEFAULT_PAGE_BITS>,
    suspend_on_segment: bool,
    /// Per-address-space page buffers. Each entry = 1 u32 page id.
    mem_page_buf: Vec<u32>,
    pv_page_buf: Vec<u32>,
    deferral_page_buf: Vec<u32>,
    address_height: u32,
    addr_space_height: u32,
    byte_space_leaf_bits: u32,
}

impl SegmentationState {
    pub fn new(ctx: MeteredCtx, system_config: &SystemConfig) -> Self {
        let ctx = ctx.into_parts();
        let segmentation_ctx = ctx.segmentation_ctx;
        let trace_heights = ctx.trace_heights;
        let is_trace_height_constant = ctx.is_trace_height_constant;
        let memory_ctx = ctx.memory_ctx;
        let suspend_on_segment = ctx.suspend_on_segment;

        let mem_config = &system_config.memory_config;
        let memory_dimensions = mem_config.memory_dimensions();
        let address_height = memory_dimensions.address_height as u32;
        let addr_space_height = mem_config.addr_space_height as u32;
        let byte_space_leaf_bits = (U16_CELL_SIZE_BITS + DIGEST_WIDTH_BITS) as u32;

        Self {
            segmentation_ctx,
            trace_heights,
            is_trace_height_constant,
            memory_ctx,
            suspend_on_segment,
            mem_page_buf: vec![0u32; MEM_PAGE_BUF_CAP],
            pv_page_buf: vec![0u32; PV_PAGE_BUF_CAP],
            deferral_page_buf: vec![0u32; DEFERRAL_PAGE_BUF_CAP],
            address_height,
            addr_space_height,
            byte_space_leaf_bits,
        }
    }

    pub fn into_metered_ctx(self) -> MeteredCtx {
        MeteredCtx::from_parts(MeteredCtxParts {
            trace_heights: self.trace_heights,
            is_trace_height_constant: self.is_trace_height_constant,
            memory_ctx: self.memory_ctx,
            segmentation_ctx: self.segmentation_ctx,
            suspend_on_segment: self.suspend_on_segment,
        })
    }

    pub(crate) fn suspend_on_segment(&self) -> bool {
        self.suspend_on_segment
    }

    /// Get mutable pointer to trace_heights for the C tracer.
    pub fn trace_heights_ptr(&mut self) -> *mut u32 {
        self.trace_heights.as_mut_ptr()
    }

    /// Get mutable pointer to the AS_MEMORY page buffer for the C tracer.
    pub fn mem_page_buf_ptr(&mut self) -> *mut u32 {
        self.mem_page_buf.as_mut_ptr()
    }

    /// Get mutable pointer to the AS_PUBLIC_VALUES page buffer for the C tracer.
    pub fn pv_page_buf_ptr(&mut self) -> *mut u32 {
        self.pv_page_buf.as_mut_ptr()
    }

    /// Get mutable pointer to the AS_DEFERRAL page buffer for the C tracer.
    pub fn deferral_page_buf_ptr(&mut self) -> *mut u32 {
        self.deferral_page_buf.as_mut_ptr()
    }

    /// Add initial register merkle height contributions (matches OpenVM's
    /// `add_register_merkle_heights` + `update_boundary_merkle_heights`).
    ///
    /// OpenVM records pages for the entire register space
    /// (AS=1, ptr=0, size=32*8=256) at init and after each segment boundary.
    fn add_register_merkle_heights(&mut self) {
        // RV64_REGISTER_AS=1, RV64_NUM_REGISTERS=32, RV64_REGISTER_NUM_LIMBS=8
        const REG_SIZE: u32 = 32 * 8; // 256 bytes

        let leaf_ptrs = 1u32 << self.byte_space_leaf_bits;
        let num_blocks = (REG_SIZE + leaf_ptrs - 1) >> self.byte_space_leaf_bits;
        let start_leaf_id = 0u32; // ptr=0
                                  // label_to_index: ((addr_space - 1) << address_height) + leaf_id
        let start_block_id =
            ((RV64_REGISTER_AS as u64 - 1) << self.address_height) + start_leaf_id as u64;
        let end_block_id = start_block_id + num_blocks as u64;
        let start_page_id = start_block_id >> DEFAULT_PAGE_BITS;
        let end_page_id = ((end_block_id - 1) >> DEFAULT_PAGE_BITS) + 1;

        for page_id in start_page_id..end_page_id {
            if self.memory_ctx.page_indices.insert(page_id as usize) {
                self.memory_ctx.addr_space_access_count[RV64_REGISTER_AS as usize] += 1;
            }
        }
    }

    /// Flush all page buffers: convert local pages to global ids, deduplicate
    /// via the BitSet, and update `addr_space_access_count` for each new page.
    fn flush_page_buffer(&mut self, mem_len: u32, pv_len: u32, deferral_len: u32) {
        let num_as = self.memory_ctx.addr_space_access_count.len();
        let page_shift = self.address_height as usize - DEFAULT_PAGE_BITS;
        for &(buf_len, addr_space) in &[
            (mem_len, RV64_MEMORY_AS),
            (pv_len, PUBLIC_VALUES_AS),
            (deferral_len, DEFERRAL_AS),
        ] {
            let as_idx = addr_space as usize;
            if as_idx >= num_as {
                continue;
            }
            let as_offset = (as_idx - 1) << page_shift;
            let buf = match addr_space {
                RV64_MEMORY_AS => &self.mem_page_buf,
                PUBLIC_VALUES_AS => &self.pv_page_buf,
                _ => &self.deferral_page_buf,
            };
            for &local_page in &buf[..buf_len as usize] {
                if self
                    .memory_ctx
                    .page_indices
                    .insert(as_offset + local_page as usize)
                {
                    self.memory_ctx.addr_space_access_count[as_idx] += 1;
                }
            }
        }
    }

    /// Apply boundary and merkle height updates from accumulated page accesses.
    fn apply_height_updates(&mut self) {
        let page_access_count: u32 = self.memory_ctx.addr_space_access_count.iter().sum();
        let leaves = page_access_count << DEFAULT_PAGE_BITS;

        let trace_heights = &mut self.trace_heights;
        let poseidon2_idx = trace_heights.len() - 2;

        let merkle_height = self.addr_space_height as usize + self.address_height as usize;
        let nodes_per_page =
            (((1usize << DEFAULT_PAGE_BITS) - 1) + (merkle_height - DEFAULT_PAGE_BITS)) as u32;

        // Boundary chip: 2 rows per leaf (init + final)
        trace_heights[BOUNDARY_AIR_ID] += leaves * 2;
        // Merkle tree + Poseidon2
        trace_heights[MERKLE_AIR_ID] += nodes_per_page * page_access_count * 2;
        trace_heights[poseidon2_idx] += leaves * 2 + nodes_per_page * page_access_count * 2;

        self.memory_ctx.addr_space_access_count.fill(0);
    }

    fn initialize_segment_memory(&mut self, mem_len: u32, pv_len: u32, deferral_len: u32) {
        let poseidon2_idx = self.trace_heights.len() - 2;
        self.trace_heights[BOUNDARY_AIR_ID] = 0;
        self.trace_heights[MERKLE_AIR_ID] = 0;
        self.trace_heights[poseidon2_idx] = 0;

        self.memory_ctx.page_indices.clear();
        self.memory_ctx.addr_space_access_count.fill(0);
        self.memory_ctx.page_indices_since_checkpoint_len = 0;

        // RVR can only suspend at block boundaries, so segmentation uses the
        // last safe checkpoint. The pages accumulated since that checkpoint
        // belong to the next segment and must seed its memory trace heights
        // after the previous segment's heights are subtracted.
        self.flush_page_buffer(mem_len, pv_len, deferral_len);
        self.apply_height_updates();

        self.add_register_merkle_heights();
        self.apply_height_updates();
    }

    /// Called on each periodic check (approximately every `segment_check_insns` instructions).
    /// Invoked from the C tracer's `trace_block` callback when the block-level
    /// countdown crosses zero. Returns true if a segment boundary was created.
    pub fn on_periodic_check(
        &mut self,
        mem_len: u32,
        pv_len: u32,
        deferral_len: u32,
        remaining_counter: u32,
    ) -> bool {
        let seg_check_insns = self.segmentation_ctx.segment_check_insns;
        let insns_since_last_check = seg_check_insns - remaining_counter as u64;
        let instret = self.segmentation_ctx.instret + insns_since_last_check;
        self.segmentation_ctx.instret = instret;
        self.segmentation_ctx.instrets_until_check = seg_check_insns;

        self.flush_page_buffer(mem_len, pv_len, deferral_len);
        self.apply_height_updates();

        let did_segment = self.segmentation_ctx.check_and_segment(
            instret,
            &mut self.trace_heights,
            &self.is_trace_height_constant,
        );

        if did_segment {
            self.segmentation_ctx
                .initialize_segment(&mut self.trace_heights, &self.is_trace_height_constant);
            self.initialize_segment_memory(mem_len, pv_len, deferral_len);

            self.segmentation_ctx.warn_if_exceeds_limits(
                instret,
                &self.trace_heights,
                &self.is_trace_height_constant,
            );
        }

        self.segmentation_ctx
            .update_checkpoint(instret, &self.trace_heights);

        did_segment
    }

    /// Called when execution terminates. Creates the final segment.
    /// `remaining_counter` is the tracer's `check_counter` value at termination,
    /// representing unaccounted instructions since the last periodic check.
    pub fn on_termination(
        &mut self,
        mem_len: u32,
        pv_len: u32,
        deferral_len: u32,
        remaining_counter: u32,
    ) {
        self.flush_page_buffer(mem_len, pv_len, deferral_len);
        self.apply_height_updates();

        self.segmentation_ctx.instrets_until_check = remaining_counter as u64;
        self.segmentation_ctx
            .create_final_segment(&self.trace_heights);
    }
}

// ── Inline callback from C tracer ─────────────────────────────────────────────

/// Callback invoked from the C tracer's `trace_block` when the
/// segmentation counter is about to underflow. Called BEFORE the
/// decrement, so `check_counter` still holds the remaining count.
///
/// # Safety
/// `t` must point to a valid `MeteredTracerData` whose `seg_state` pointer
/// references a live `SegmentationState`.
pub unsafe extern "C" fn metered_periodic_check(t: *mut MeteredTracerData) -> u8 {
    let tracer = &mut *t;
    let seg_state = &mut *(tracer.seg_state as *mut SegmentationState);
    let mem_len = tracer.mem_page_buf_len;
    let pv_len = tracer.pv_page_buf_len;
    let deferral_len = tracer.deferral_page_buf_len;
    tracer.mem_page_buf_len = 0;
    tracer.pv_page_buf_len = 0;
    tracer.deferral_page_buf_len = 0;
    // Reset dedup cache — required for correctness across segment boundaries
    // that clear the global BitSet.
    tracer.last_mem_page = NO_LAST_PAGE;

    let did_segment =
        seg_state.on_periodic_check(mem_len, pv_len, deferral_len, tracer.check_counter);

    let segment_check_insns = seg_state.segmentation_ctx.segment_check_insns as u32;
    debug_assert_eq!(
        u64::from(segment_check_insns),
        seg_state.segmentation_ctx.segment_check_insns
    );

    // We are at the start of a block that would cross the old countdown.
    // `remaining_counter` was used to record this block start as the metering
    // boundary, so the next interval starts here with a full countdown.
    tracer.check_counter = segment_check_insns;
    did_segment as u8
}

impl<F: PrimeField32, S> RvrMeteredInstanceWith<'_, F, S> {
    pub fn create_initial_vm_state(
        &self,
        inputs: impl Into<Streams<F>>,
    ) -> VmState<F, GuestMemory> {
        VmState::initial(
            self.system_config,
            &self.exe.init_memory,
            self.exe.pc_start,
            inputs,
        )
    }

    /// Persist the compiled shared library into `dir`. Returns the path to
    /// the copied artifact. The user must re-supply `exe`, `executor_idx_to_air_idx`,
    /// and any mode-specific data when loading.
    pub fn save(&self, dir: &Path) -> Result<PathBuf, super::CompileError> {
        let dest_lib = self.compiled.lib_file_name_with_suffix("metered")?;
        self.compiled.save_artifact(&dir.join(dest_lib))
    }
}

impl<F: PrimeField32> RvrMeteredInstance<'_, F> {
    pub fn execute_metered(
        &self,
        inputs: impl Into<Streams<F>>,
        ctx: MeteredCtx,
    ) -> Result<(Vec<Segment>, VmState<F, GuestMemory>), ExecutionError> {
        let vm_state = self.create_initial_vm_state(inputs);
        self.execute_metered_from_state(vm_state, ctx)
    }

    pub fn execute_metered_from_state(
        &self,
        mut vm_state: VmState<F, GuestMemory>,
        ctx: MeteredCtx,
    ) -> Result<(Vec<Segment>, VmState<F, GuestMemory>), ExecutionError> {
        let seg_state = SegmentationState::new(ctx, self.system_config);

        #[cfg(feature = "metrics")]
        let metrics = ExecutionMetricTimer::start(ExecutionMetric::Metered);
        let result_seg_state = tracing::info_span!("execute_metered")
            .in_scope(|| {
                execute_metered(&self.compiled, &self.extensions, &mut vm_state, seg_state)
            })
            .map_err(map_rvr_execute_error)?;
        let result_seg_ctx = result_seg_state.segmentation_ctx;
        #[cfg(feature = "metrics")]
        {
            let insns = result_seg_ctx.instret;
            metrics.record(insns);
        }

        Ok((result_seg_ctx.segments, vm_state))
    }
}

impl<F: PrimeField32> RvrMeteredSegmentInstance<'_, F> {
    /// Executes until termination or the next segment-boundary suspension.
    pub fn execute_metered_until_segment_boundary(
        &self,
        inputs: impl Into<Streams<F>>,
        ctx: MeteredCtx,
    ) -> Result<(RvrMeteredResult, VmState<F, GuestMemory>), ExecutionError> {
        let vm_state = self.create_initial_vm_state(inputs);
        self.execute_metered_from_state_until_segment_boundary(vm_state, ctx)
    }

    pub fn execute_metered_from_state_until_segment_boundary(
        &self,
        mut vm_state: VmState<F, GuestMemory>,
        ctx: MeteredCtx,
    ) -> Result<(RvrMeteredResult, VmState<F, GuestMemory>), ExecutionError> {
        #[cfg(feature = "metrics")]
        let metrics = ExecutionMetricTimer::start(ExecutionMetric::Metered);
        #[cfg(feature = "metrics")]
        let start_instret = ctx.segmentation_ctx.instret;
        let seg_state = SegmentationState::new(ctx, self.system_config);

        let result = tracing::info_span!("execute_metered").in_scope(|| {
            execute_metered_segment_boundary(
                &self.compiled,
                &self.extensions,
                &mut vm_state,
                seg_state,
            )
        });
        let result = result.map_err(map_rvr_execute_error)?;
        #[cfg(feature = "metrics")]
        {
            let insns = result.seg_state.segmentation_ctx.instret - start_instret;
            metrics.record(insns);
        }
        Ok((result, vm_state))
    }
}

#[cfg(all(test, feature = "rvr"))]
mod tests {
    use openvm_stark_backend::StarkEngine;

    use super::*;
    use crate::{
        arch::{execution_mode::metered::ctx::DEFAULT_PAGE_BITS, BOUNDARY_AIR_ID, MERKLE_AIR_ID},
        utils::{test_cpu_engine, test_system_config},
    };

    fn make_segmentation_state() -> SegmentationState {
        let system_config = test_system_config();
        let num_airs = 6;
        let mut air_names = (0..num_airs)
            .map(|idx| format!("Air {idx}"))
            .collect::<Vec<_>>();
        air_names[BOUNDARY_AIR_ID] = "Memory Boundary".to_string();
        air_names[MERKLE_AIR_ID] = "Memory Merkle".to_string();

        let ctx = MeteredCtx::<DEFAULT_PAGE_BITS>::new(
            vec![None; num_airs],
            air_names,
            vec![1; num_airs],
            vec![0; num_airs],
            vec![false; num_airs],
            &system_config,
            test_cpu_engine().proving_memory_config(),
        );
        SegmentationState::new(ctx, &system_config)
    }

    #[test]
    fn test_initialize_segment_memory_replays_checkpoint_pages() {
        let mut with_interval_buffer = make_segmentation_state();
        with_interval_buffer.mem_page_buf[0] = 7;
        with_interval_buffer.pv_page_buf[0] = 3;
        with_interval_buffer.deferral_page_buf[0] = 2;
        with_interval_buffer.initialize_segment_memory(1, 1, 1);

        let mut clean = make_segmentation_state();
        clean.initialize_segment_memory(0, 0, 0);

        assert!(
            with_interval_buffer.trace_heights[BOUNDARY_AIR_ID]
                > clean.trace_heights[BOUNDARY_AIR_ID]
        );
        assert!(
            with_interval_buffer.trace_heights[MERKLE_AIR_ID] > clean.trace_heights[MERKLE_AIR_ID]
        );
        let poseidon2_idx = clean.trace_heights.len() - 2;
        assert!(
            with_interval_buffer.trace_heights[poseidon2_idx] > clean.trace_heights[poseidon2_idx]
        );
    }

    #[test]
    fn test_periodic_check_records_block_boundary_instret() {
        let mut seg_state = make_segmentation_state();
        seg_state.segmentation_ctx.segment_check_insns = 1000;
        seg_state.segmentation_ctx.instrets_until_check = 1000;

        assert!(!seg_state.on_periodic_check(0, 0, 0, 250));

        assert_eq!(seg_state.segmentation_ctx.instret, 750);
        assert_eq!(seg_state.segmentation_ctx.instrets_until_check, 1000);
    }

    #[test]
    fn test_periodic_callback_starts_next_interval_at_block_boundary() {
        let mut seg_state = make_segmentation_state();
        seg_state.segmentation_ctx.segment_check_insns = 1000;
        seg_state.segmentation_ctx.instrets_until_check = 1000;
        let mut tracer = MeteredTracerData {
            trace_heights: seg_state.trace_heights_ptr(),
            mem_page_buf: seg_state.mem_page_buf_ptr(),
            pv_page_buf: seg_state.pv_page_buf_ptr(),
            deferral_page_buf: seg_state.deferral_page_buf_ptr(),
            on_check: metered_periodic_check,
            seg_state: &mut seg_state as *mut SegmentationState as *mut c_void,
            mem_page_buf_len: 0,
            pv_page_buf_len: 0,
            deferral_page_buf_len: 0,
            check_counter: 250,
            last_mem_page: NO_LAST_PAGE,
        };

        let did_segment = unsafe { metered_periodic_check(&mut tracer) };

        assert_eq!(did_segment, 0);
        assert_eq!(tracer.check_counter, 1000);
        assert_eq!(seg_state.segmentation_ctx.instret, 750);
    }

    #[test]
    fn test_periodic_callback_starts_next_interval_when_suspending() {
        let mut seg_state = make_segmentation_state();
        seg_state.segmentation_ctx.segment_check_insns = 1000;
        seg_state.segmentation_ctx.instrets_until_check = 1000;
        seg_state.segmentation_ctx.limits.max_trace_height = 1;
        *seg_state.trace_heights.last_mut().unwrap() = 2;
        let mut tracer = MeteredTracerData {
            trace_heights: seg_state.trace_heights_ptr(),
            mem_page_buf: seg_state.mem_page_buf_ptr(),
            pv_page_buf: seg_state.pv_page_buf_ptr(),
            deferral_page_buf: seg_state.deferral_page_buf_ptr(),
            on_check: metered_periodic_check,
            seg_state: &mut seg_state as *mut SegmentationState as *mut c_void,
            mem_page_buf_len: 0,
            pv_page_buf_len: 0,
            deferral_page_buf_len: 0,
            check_counter: 250,
            last_mem_page: NO_LAST_PAGE,
        };

        let did_segment = unsafe { metered_periodic_check(&mut tracer) };

        assert_eq!(did_segment, 1);
        assert_eq!(tracer.check_counter, 1000);
        assert_eq!(seg_state.segmentation_ctx.instret, 750);
    }
}
