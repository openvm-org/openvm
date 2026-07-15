//! Per-chip metered execution: page tracking and segmentation
//! matching OpenVM's `MeteredCtx`.

use std::{
    borrow::Cow,
    ffi::c_void,
    path::{Path, PathBuf},
};

use openvm_instructions::{riscv::RV64_MEMORY_AS, DEFERRAL_AS, MEMORY_PAGE_BITS, PUBLIC_VALUES_AS};
use rvr_openvm::{DEFERRAL_PAGE_BUF_CAP, MEM_PAGE_BUF_CAP, PV_PAGE_BUF_CAP};
use rvr_openvm_lift::RvrRuntimeExtension;

use super::{
    bridge::map_rvr_execute_error,
    execute_metered, execute_metered_segment_boundary,
    state::{TracerPayload, TracerPtr},
    RvrCompiled, RvrInitialImage,
};
#[cfg(feature = "metrics")]
use crate::arch::execution_metrics::{ExecutionMetric, ExecutionMetricTimer};
use crate::{
    arch::{
        execution_mode::{
            metered::{
                memory_ctx::{MemoryCtx, PageTouch},
                segment_ctx::Segment,
            },
            MeteredCtx,
        },
        ExecutionError, Streams, SystemConfig, VmState, ADDR_SPACE_OFFSET,
    },
    system::memory::online::GuestMemory,
};

struct RvrMeteredInstanceInner<'a> {
    system_config: Cow<'a, SystemConfig>,
    initial_image: RvrInitialImage,
    runtime_hooks: Vec<Box<dyn RvrRuntimeExtension>>,
    compiled: RvrCompiled,
}

pub struct RvrMeteredInstance<'a> {
    inner: RvrMeteredInstanceInner<'a>,
}

pub struct RvrMeteredSegmentInstance<'a> {
    inner: RvrMeteredInstanceInner<'a>,
}

static_assertions::assert_impl_all!(RvrMeteredInstance<'static>: Send, Sync);
static_assertions::assert_impl_all!(RvrMeteredSegmentInstance<'static>: Send, Sync);

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
    pub mem_page_buf: *mut PageTouch,
    pub pv_page_buf: *mut PageTouch,
    pub deferral_page_buf: *mut PageTouch,
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
    /// OpenVM metered execution context - holds trace heights, memory tracking,
    /// and segmentation logic.
    pub ctx: MeteredCtx,
    /// Per-address-space page buffers. Each entry = local page id + leaf mask.
    mem_page_buf: Vec<PageTouch>,
    pv_page_buf: Vec<PageTouch>,
    deferral_page_buf: Vec<PageTouch>,
    address_height: u32,
}

impl SegmentationState {
    pub fn new(ctx: MeteredCtx, system_config: &SystemConfig) -> Self {
        let memory_dimensions = system_config.memory_config.memory_dimensions();
        let address_height = memory_dimensions.address_height as u32;
        Self {
            ctx,
            mem_page_buf: vec![PageTouch::default(); MEM_PAGE_BUF_CAP],
            pv_page_buf: vec![PageTouch::default(); PV_PAGE_BUF_CAP],
            deferral_page_buf: vec![PageTouch::default(); DEFERRAL_PAGE_BUF_CAP],
            address_height,
        }
    }

    pub fn into_metered_ctx(self) -> MeteredCtx {
        self.ctx
    }

    pub(crate) fn suspend_on_segment(&self) -> bool {
        self.ctx.config.suspend_on_segment
    }

    /// Get mutable pointer to trace_heights for the C tracer.
    pub fn trace_heights_ptr(&mut self) -> *mut u32 {
        self.ctx.trace_heights.as_mut_ptr()
    }

    /// Get mutable pointer to the AS_MEMORY page buffer for the C tracer.
    pub fn mem_page_buf_ptr(&mut self) -> *mut PageTouch {
        self.mem_page_buf.as_mut_ptr()
    }

    /// Get mutable pointer to the AS_PUBLIC_VALUES page buffer for the C tracer.
    pub fn pv_page_buf_ptr(&mut self) -> *mut PageTouch {
        self.pv_page_buf.as_mut_ptr()
    }

    /// Get mutable pointer to the AS_DEFERRAL page buffer for the C tracer.
    pub fn deferral_page_buf_ptr(&mut self) -> *mut PageTouch {
        self.deferral_page_buf.as_mut_ptr()
    }

    #[inline(always)]
    fn apply_addr_space_buffer(
        memory_ctx: &mut MemoryCtx,
        address_height: u32,
        addr_space: u32,
        buffer: &[PageTouch],
        len: u32,
    ) {
        if len == 0 {
            return;
        }
        // C buffers use page ids local to one address space. `MemoryCtx`
        // deduplicates against one global memory tree, so convert to global
        // page ids before applying the leaf masks.
        let page_shift = address_height as usize - MEMORY_PAGE_BITS;
        let page_offset = ((addr_space as usize - ADDR_SPACE_OFFSET as usize) << page_shift) as u32;
        memory_ctx.apply_page_touches_with_offset(page_offset, &buffer[..len as usize]);
    }

    /// Apply all page buffers: convert local pages to global ids and update
    /// memory metering state.
    #[inline(always)]
    fn apply_page_buffers(&mut self, mem_len: u32, pv_len: u32, deferral_len: u32) {
        Self::apply_addr_space_buffer(
            &mut self.ctx.memory_ctx,
            self.address_height,
            RV64_MEMORY_AS,
            &self.mem_page_buf,
            mem_len,
        );
        Self::apply_addr_space_buffer(
            &mut self.ctx.memory_ctx,
            self.address_height,
            PUBLIC_VALUES_AS,
            &self.pv_page_buf,
            pv_len,
        );
        Self::apply_addr_space_buffer(
            &mut self.ctx.memory_ctx,
            self.address_height,
            DEFERRAL_AS,
            &self.deferral_page_buf,
            deferral_len,
        );
    }

    fn initialize_segment_memory(&mut self, mem_len: u32, pv_len: u32, deferral_len: u32) {
        // RVR can only suspend at block boundaries, so segmentation uses the
        // last safe checkpoint. The pages accumulated since that checkpoint
        // belong to the next segment and must seed its memory trace heights
        // after the previous segment's heights are subtracted.
        self.ctx
            .memory_ctx
            .reset_segment_without_replay(&mut self.ctx.trace_heights);
        self.apply_page_buffers(mem_len, pv_len, deferral_len);
        self.ctx
            .memory_ctx
            .apply_height_updates(&mut self.ctx.trace_heights);

        self.ctx.memory_ctx.add_register_merkle_heights();
        self.ctx
            .memory_ctx
            .apply_height_updates(&mut self.ctx.trace_heights);
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
        let seg_check_insns = self.ctx.segmentation_ctx.segment_check_insns();
        let insns_since_last_check = seg_check_insns - remaining_counter as u64;
        let instret = self.ctx.segmentation_ctx.instret + insns_since_last_check;
        self.ctx.segmentation_ctx.instret = instret;
        self.ctx.segmentation_ctx.instrets_until_check = seg_check_insns;

        self.apply_page_buffers(mem_len, pv_len, deferral_len);
        self.ctx
            .memory_ctx
            .apply_height_updates(&mut self.ctx.trace_heights);

        let did_segment = self.ctx.segmentation_ctx.check_and_segment(
            instret,
            &mut self.ctx.trace_heights,
            &self.ctx.config.is_trace_height_constant,
        );

        if did_segment {
            self.ctx.segmentation_ctx.initialize_segment(
                &mut self.ctx.trace_heights,
                &self.ctx.config.is_trace_height_constant,
            );
            self.initialize_segment_memory(mem_len, pv_len, deferral_len);

            self.ctx.segmentation_ctx.warn_if_exceeds_limits(
                instret,
                &self.ctx.trace_heights,
                &self.ctx.config.is_trace_height_constant,
            );
        }

        self.ctx
            .segmentation_ctx
            .update_checkpoint(instret, &self.ctx.trace_heights);
        self.ctx.memory_ctx.update_checkpoint();

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
        self.apply_page_buffers(mem_len, pv_len, deferral_len);
        self.ctx
            .memory_ctx
            .apply_height_updates(&mut self.ctx.trace_heights);

        self.ctx.segmentation_ctx.instrets_until_check = remaining_counter as u64;
        self.ctx
            .segmentation_ctx
            .create_final_segment(&self.ctx.trace_heights);
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
    // The cleared buffer no longer contains the entry cached by last_mem_page.
    tracer.last_mem_page = NO_LAST_PAGE;

    let did_segment =
        seg_state.on_periodic_check(mem_len, pv_len, deferral_len, tracer.check_counter);

    let segment_check_insns = seg_state.ctx.segmentation_ctx.segment_check_insns() as u32;
    debug_assert_eq!(
        u64::from(segment_check_insns),
        seg_state.ctx.segmentation_ctx.segment_check_insns()
    );

    // We are at the start of a block that would cross the old countdown.
    // `remaining_counter` was used to record this block start as the metering
    // boundary, so the next interval starts here with a full countdown.
    tracer.check_counter = segment_check_insns;
    did_segment as u8
}

impl RvrMeteredInstanceInner<'_> {
    fn create_initial_vm_state(&self, inputs: impl Into<Streams>) -> VmState<GuestMemory> {
        self.initial_image
            .create_vm_state(&self.system_config, inputs)
    }

    /// Persist the compiled shared library into `dir`. Returns the path to
    /// the copied artifact. The user must re-supply `exe`, `executor_idx_to_air_idx`,
    /// and any mode-specific data when loading.
    fn save(&self, dir: &Path) -> Result<PathBuf, super::CompileError> {
        let dest_lib = self.compiled.lib_file_name_with_suffix("metered")?;
        self.compiled.save_artifact(&dir.join(dest_lib))
    }

    /// Persist generated C sources for inspection.
    fn save_generated_sources(&self, dir: &Path) -> Result<(), super::CompileError> {
        self.compiled.save_generated_sources(dir)
    }
}

impl RvrMeteredInstance<'_> {
    pub(crate) fn new(
        system_config: &SystemConfig,
        initial_image: RvrInitialImage,
        runtime_hooks: Vec<Box<dyn RvrRuntimeExtension>>,
        compiled: RvrCompiled,
    ) -> RvrMeteredInstance<'_> {
        RvrMeteredInstance {
            inner: RvrMeteredInstanceInner {
                system_config: Cow::Borrowed(system_config),
                initial_image,
                runtime_hooks,
                compiled,
            },
        }
    }

    pub fn create_initial_vm_state(&self, inputs: impl Into<Streams>) -> VmState<GuestMemory> {
        self.inner.create_initial_vm_state(inputs)
    }

    /// Persist the compiled shared library into `dir`. Returns the path to
    /// the copied artifact. The user must re-supply `exe`, `executor_idx_to_air_idx`,
    /// and any mode-specific data when loading.
    pub fn save(&self, dir: &Path) -> Result<PathBuf, super::CompileError> {
        self.inner.save(dir)
    }

    /// Persist generated C sources for inspection.
    pub fn save_generated_sources(&self, dir: &Path) -> Result<(), super::CompileError> {
        self.inner.save_generated_sources(dir)
    }
    /// Convert the instance into an owned form suitable for caching on a VM
    /// prover instance. The compiled library and executable are already
    /// owned; only the small system configuration needs to be cloned.
    pub(crate) fn into_owned(self) -> RvrMeteredInstance<'static> {
        RvrMeteredInstance {
            inner: RvrMeteredInstanceInner {
                system_config: Cow::Owned(self.inner.system_config.into_owned()),
                initial_image: self.inner.initial_image,
                runtime_hooks: self.inner.runtime_hooks,
                compiled: self.inner.compiled,
            },
        }
    }

    pub fn execute_metered(
        &self,
        inputs: impl Into<Streams>,
        ctx: MeteredCtx,
    ) -> Result<(Vec<Segment>, VmState<GuestMemory>), ExecutionError> {
        let vm_state = self.create_initial_vm_state(inputs);
        self.execute_metered_from_state(vm_state, ctx)
    }

    pub fn execute_metered_from_state(
        &self,
        mut vm_state: VmState<GuestMemory>,
        ctx: MeteredCtx,
    ) -> Result<(Vec<Segment>, VmState<GuestMemory>), ExecutionError> {
        let seg_state = SegmentationState::new(ctx, &self.inner.system_config);

        #[cfg(feature = "metrics")]
        let metrics = ExecutionMetricTimer::start(ExecutionMetric::Metered);
        let result_seg_state = tracing::info_span!("execute_metered")
            .in_scope(|| {
                execute_metered(
                    &self.inner.compiled,
                    &self.inner.runtime_hooks,
                    &mut vm_state,
                    seg_state,
                )
            })
            .map_err(map_rvr_execute_error)?;
        let result_seg_ctx = result_seg_state.ctx.segmentation_ctx;
        #[cfg(feature = "metrics")]
        {
            let insns = result_seg_ctx.instret;
            metrics.record(insns);
        }

        Ok((result_seg_ctx.segments, vm_state))
    }
}

impl RvrMeteredSegmentInstance<'_> {
    pub(crate) fn new(
        system_config: &SystemConfig,
        initial_image: RvrInitialImage,
        runtime_hooks: Vec<Box<dyn RvrRuntimeExtension>>,
        compiled: RvrCompiled,
    ) -> RvrMeteredSegmentInstance<'_> {
        RvrMeteredSegmentInstance {
            inner: RvrMeteredInstanceInner {
                system_config: Cow::Borrowed(system_config),
                initial_image,
                runtime_hooks,
                compiled,
            },
        }
    }

    pub fn create_initial_vm_state(&self, inputs: impl Into<Streams>) -> VmState<GuestMemory> {
        self.inner.create_initial_vm_state(inputs)
    }

    /// Persist the compiled shared library into `dir`. Returns the path to
    /// the copied artifact. The user must re-supply `exe`, `executor_idx_to_air_idx`,
    /// and any mode-specific data when loading.
    pub fn save(&self, dir: &Path) -> Result<PathBuf, super::CompileError> {
        self.inner.save(dir)
    }

    /// Persist generated C sources for inspection.
    pub fn save_generated_sources(&self, dir: &Path) -> Result<(), super::CompileError> {
        self.inner.save_generated_sources(dir)
    }

    /// Executes until termination or the next segment-boundary suspension.
    pub fn execute_metered_until_segment_boundary(
        &self,
        inputs: impl Into<Streams>,
        ctx: MeteredCtx,
    ) -> Result<(RvrMeteredResult, VmState<GuestMemory>), ExecutionError> {
        let vm_state = self.create_initial_vm_state(inputs);
        self.execute_metered_from_state_until_segment_boundary(vm_state, ctx)
    }

    pub fn execute_metered_from_state_until_segment_boundary(
        &self,
        mut vm_state: VmState<GuestMemory>,
        ctx: MeteredCtx,
    ) -> Result<(RvrMeteredResult, VmState<GuestMemory>), ExecutionError> {
        #[cfg(feature = "metrics")]
        let metrics = ExecutionMetricTimer::start(ExecutionMetric::Metered);
        #[cfg(feature = "metrics")]
        let start_instret = ctx.segmentation_ctx.instret;
        let seg_state = SegmentationState::new(ctx, &self.inner.system_config);

        let result = tracing::info_span!("execute_metered").in_scope(|| {
            execute_metered_segment_boundary(
                &self.inner.compiled,
                &self.inner.runtime_hooks,
                &mut vm_state,
                seg_state,
            )
        });
        let result = result.map_err(map_rvr_execute_error)?;
        #[cfg(feature = "metrics")]
        {
            let insns = result.seg_state.ctx.segmentation_ctx.instret - start_instret;
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
        arch::{
            execution_mode::metered::{
                ctx::MeteredCtxInputs,
                segment_ctx::{SegmentationLimits, DEFAULT_MAX_MEMORY},
            },
            BOUNDARY_AIR_ID, MERKLE_AIR_ID,
        },
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
        air_names[num_airs - 2] = "Poseidon2".to_string();
        let constant_trace_heights = vec![None; num_airs];
        let widths = vec![1; num_airs];
        let interactions = vec![0; num_airs];
        let need_rot = vec![false; num_airs];

        let ctx = MeteredCtx::new(
            MeteredCtxInputs {
                constant_trace_heights: &constant_trace_heights,
                air_names: &air_names,
                widths: &widths,
                interactions: &interactions,
                need_rot: &need_rot,
                segmentation_limits: SegmentationLimits {
                    max_trace_height_bits: 11,
                    max_memory: DEFAULT_MAX_MEMORY,
                    max_interactions: u32::MAX,
                },
            },
            &system_config,
            test_cpu_engine().proving_memory_config(),
        );
        SegmentationState::new(ctx, &system_config)
    }

    #[test]
    fn test_initialize_segment_memory_replays_page_buffers() {
        let mut with_interval_buffer = make_segmentation_state();
        with_interval_buffer.mem_page_buf[0] = PageTouch {
            page_id: 7,
            leaf_mask: 1,
        };
        with_interval_buffer.pv_page_buf[0] = PageTouch {
            page_id: 3,
            leaf_mask: 1,
        };
        with_interval_buffer.deferral_page_buf[0] = PageTouch {
            page_id: 2,
            leaf_mask: 1,
        };
        with_interval_buffer.initialize_segment_memory(1, 1, 1);

        let mut clean = make_segmentation_state();
        clean.initialize_segment_memory(0, 0, 0);

        assert!(
            with_interval_buffer.ctx.trace_heights[BOUNDARY_AIR_ID]
                > clean.ctx.trace_heights[BOUNDARY_AIR_ID]
        );
        assert!(
            with_interval_buffer.ctx.trace_heights[MERKLE_AIR_ID]
                > clean.ctx.trace_heights[MERKLE_AIR_ID]
        );
        let poseidon2_idx = clean.ctx.trace_heights.len() - 2;
        assert!(
            with_interval_buffer.ctx.trace_heights[poseidon2_idx]
                > clean.ctx.trace_heights[poseidon2_idx]
        );
    }

    #[test]
    fn test_periodic_checks_share_default_poseidon_rows() {
        let mut seg_state = make_segmentation_state();
        seg_state.mem_page_buf[0] = PageTouch {
            page_id: 0,
            leaf_mask: 1,
        };
        assert!(!seg_state.on_periodic_check(1, 0, 0, 0));

        let poseidon2_idx = seg_state.ctx.trace_heights.len() - 2;
        let poseidon_before = seg_state.ctx.trace_heights[poseidon2_idx];
        seg_state.mem_page_buf[0] = PageTouch {
            page_id: 1,
            leaf_mask: 1,
        };
        assert!(!seg_state.on_periodic_check(1, 0, 0, 0));

        assert_eq!(
            seg_state.ctx.trace_heights[poseidon2_idx] - poseidon_before,
            1 + MEMORY_PAGE_BITS as u32
        );
    }

    #[test]
    fn test_memory_page_buffer_applies_cross_leaf_mask() {
        let mut buffered = make_segmentation_state();
        buffered.mem_page_buf[0] = PageTouch {
            page_id: 0,
            leaf_mask: 0b11,
        };
        buffered.on_termination(1, 0, 0, 0);

        let mut explicit = make_segmentation_state();
        explicit
            .ctx
            .memory_ctx
            .update_boundary_merkle_heights(RV64_MEMORY_AS, 0, 17);
        explicit
            .ctx
            .memory_ctx
            .apply_height_updates(&mut explicit.ctx.trace_heights);
        explicit.ctx.segmentation_ctx.instrets_until_check = 0;
        explicit
            .ctx
            .segmentation_ctx
            .create_final_segment(&explicit.ctx.trace_heights);

        assert_eq!(buffered.ctx.trace_heights, explicit.ctx.trace_heights);
        assert_eq!(
            buffered.ctx.segmentation_ctx.segments.len(),
            explicit.ctx.segmentation_ctx.segments.len()
        );
        assert_eq!(
            buffered.ctx.segmentation_ctx.segments[0].trace_heights,
            explicit.ctx.segmentation_ctx.segments[0].trace_heights
        );
    }

    #[test]
    fn test_periodic_check_records_block_boundary_instret() {
        let mut seg_state = make_segmentation_state();
        seg_state.ctx.segmentation_ctx.instrets_until_check = 1000;

        assert!(!seg_state.on_periodic_check(0, 0, 0, 250));

        assert_eq!(seg_state.ctx.segmentation_ctx.instret, 750);
        assert_eq!(seg_state.ctx.segmentation_ctx.instrets_until_check, 1000);
    }

    #[test]
    fn test_periodic_callback_starts_next_interval_at_block_boundary() {
        let mut seg_state = make_segmentation_state();
        seg_state.ctx.segmentation_ctx.instrets_until_check = 1000;
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
        assert_eq!(seg_state.ctx.segmentation_ctx.instret, 750);
    }

    #[test]
    fn test_periodic_callback_starts_next_interval_when_suspending() {
        let mut seg_state = make_segmentation_state();
        seg_state.ctx.segmentation_ctx.instrets_until_check = 1000;
        *seg_state.ctx.trace_heights.last_mut().unwrap() = 4096;
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
        assert_eq!(seg_state.ctx.segmentation_ctx.instret, 750);
    }
}
