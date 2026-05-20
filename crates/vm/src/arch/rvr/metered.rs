//! Per-chip metered execution: page tracking and segmentation
//! matching OpenVM's `MeteredCtx`.

use std::{ffi::c_void, sync::Arc};

use openvm_instructions::{
    exe::VmExe, riscv::RV32_MEMORY_AS, LocalOpcode, SystemOpcode, VmOpcode, DEFERRAL_AS,
};
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm::{DEFERRAL_PAGE_BUF_CAP, MEM_PAGE_BUF_CAP, PV_PAGE_BUF_CAP};
use rvr_openvm_lift::ExtensionRegistry;

use super::{
    bridge::map_rvr_execute_error,
    execute_metered,
    state::{TracerPayload, TracerPtr},
    RvrCompiled,
};
use crate::{
    arch::{
        execution_mode::{
            metered::{
                ctx::DEFAULT_PAGE_BITS,
                memory_ctx::BitSet,
                segment_ctx::{Segment, SegmentationCtx},
            },
            MeteredCtx,
        },
        ExecutionError, ExecutorInventory, Streams, SystemConfig, VmState, BOUNDARY_AIR_ID,
        MERKLE_AIR_ID,
    },
    system::memory::{
        merkle::public_values::PUBLIC_VALUES_AS, online::GuestMemory, CHUNK as MERKLE_CHUNK,
    },
};

pub struct RvrMeteredInstance<F: PrimeField32> {
    pub(crate) system_config: SystemConfig,
    pub(crate) exe: Arc<VmExe<F>>,
    pub(crate) extensions: ExtensionRegistry<F>,
    pub(crate) compiled: RvrCompiled,
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
    pub on_check: Option<unsafe extern "C" fn(*mut MeteredTracerData)>,
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
            on_check: None,
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

// ── Chip mapping ─────────────────────────────────────────────────────────────

pub fn build_pc_to_chip<F, E>(
    exe: &VmExe<F>,
    inventory: &ExecutorInventory<E>,
    executor_idx_to_air_idx: &[usize],
) -> Vec<u32>
where
    F: PrimeField32,
{
    let terminate_opcode = SystemOpcode::TERMINATE.global_opcode();
    exe.program
        .instructions_and_debug_infos
        .iter()
        .map(|slot| {
            if let Some((inst, _)) = slot {
                let opcode: VmOpcode = inst.opcode;
                if opcode == terminate_opcode {
                    u32::MAX
                } else if let Some(&executor_idx) = inventory.instruction_lookup.get(&opcode) {
                    executor_idx_to_air_idx[executor_idx as usize] as u32
                } else {
                    u32::MAX
                }
            } else {
                u32::MAX
            }
        })
        .collect()
}

// ── Segmentation runtime ─────────────────────────────────────────────────────

// TODO: generalize non-memory page buffers to a config-driven set of N
// buffers (one per additional address space) instead of hardcoding
// pv + deferral. The memory AS buffer stays separate as the hot path.
pub struct SegmentationState {
    pub segmentation_ctx: SegmentationCtx,
    trace_heights: Vec<u32>,
    is_trace_height_constant: Vec<bool>,
    /// Per-address-space page buffers. Each entry = 1 u32 page id.
    mem_page_buf: Vec<u32>,
    pv_page_buf: Vec<u32>,
    deferral_page_buf: Vec<u32>,
    // Page tracking
    page_indices: BitSet,
    addr_space_access_count: Vec<u32>,
    address_height: u32,
    addr_space_height: u32,
    chunk_bits: u32,
}

impl SegmentationState {
    pub fn new(ctx: MeteredCtx, system_config: &SystemConfig) -> Self {
        let segmentation_ctx = ctx.segmentation_ctx;
        let trace_heights = ctx.trace_heights;
        let is_trace_height_constant = ctx.is_trace_height_constant;

        let mem_config = &system_config.memory_config;
        let memory_dimensions = mem_config.memory_dimensions();
        let address_height = memory_dimensions.address_height as u32;
        let addr_space_height = mem_config.addr_space_height as u32;
        let chunk_bits = MERKLE_CHUNK.ilog2();
        let num_addr_spaces = mem_config.addr_spaces.len();

        let overall_height = addr_space_height as usize + address_height as usize;
        let bitset_size = 1usize << overall_height.saturating_sub(DEFAULT_PAGE_BITS);

        Self {
            segmentation_ctx,
            trace_heights,
            is_trace_height_constant,
            mem_page_buf: vec![0u32; MEM_PAGE_BUF_CAP],
            pv_page_buf: vec![0u32; PV_PAGE_BUF_CAP],
            deferral_page_buf: vec![0u32; DEFERRAL_PAGE_BUF_CAP],
            page_indices: BitSet::new(bitset_size),
            addr_space_access_count: vec![0u32; num_addr_spaces],
            address_height,
            addr_space_height,
            chunk_bits,
        }
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
    /// (AS=1, ptr=0, size=32*4=128) at init and after each segment boundary.
    fn add_register_merkle_heights(&mut self) {
        // RV32_REGISTER_AS=1, RV32_NUM_REGISTERS=32, RV32_REGISTER_NUM_LIMBS=4
        const REG_AS: u32 = 1;
        const REG_SIZE: u32 = 32 * 4; // 128 bytes

        let chunk = 1u32 << self.chunk_bits;
        let num_blocks = (REG_SIZE + chunk - 1) >> self.chunk_bits;
        let start_chunk_id = 0u32; // ptr=0
                                   // label_to_index: ((addr_space - 1) << address_height) + chunk_id
        let start_block_id = ((REG_AS as u64 - 1) << self.address_height) + start_chunk_id as u64;
        let end_block_id = start_block_id + num_blocks as u64;
        let start_page_id = start_block_id >> DEFAULT_PAGE_BITS;
        let end_page_id = ((end_block_id - 1) >> DEFAULT_PAGE_BITS) + 1;

        for page_id in start_page_id..end_page_id {
            if self.page_indices.insert(page_id as usize) {
                self.addr_space_access_count[REG_AS as usize] += 1;
            }
        }
    }

    /// Flush all page buffers: convert local pages to global ids, deduplicate
    /// via the BitSet, and update `addr_space_access_count` for each new page.
    fn flush_page_buffer(&mut self, mem_len: u32, pv_len: u32, deferral_len: u32) {
        let num_as = self.addr_space_access_count.len();
        let page_shift = self.address_height as usize - DEFAULT_PAGE_BITS;
        for &(buf_len, addr_space) in &[
            (mem_len, RV32_MEMORY_AS),
            (pv_len, PUBLIC_VALUES_AS),
            (deferral_len, DEFERRAL_AS),
        ] {
            let as_idx = addr_space as usize;
            if as_idx >= num_as {
                continue;
            }
            let as_offset = (as_idx - 1) << page_shift;
            let buf = match addr_space {
                RV32_MEMORY_AS => &self.mem_page_buf,
                PUBLIC_VALUES_AS => &self.pv_page_buf,
                _ => &self.deferral_page_buf,
            };
            for &local_page in &buf[..buf_len as usize] {
                if self.page_indices.insert(as_offset + local_page as usize) {
                    self.addr_space_access_count[as_idx] += 1;
                }
            }
        }
    }

    /// Apply boundary and merkle height updates from accumulated page accesses.
    fn apply_height_updates(&mut self) {
        let page_access_count: u32 = self.addr_space_access_count.iter().sum();
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

        // Reset counts
        self.addr_space_access_count.fill(0);
    }

    fn initialize_segment_memory(&mut self, mem_len: u32, pv_len: u32, deferral_len: u32) {
        let poseidon2_idx = self.trace_heights.len() - 2;
        self.trace_heights[BOUNDARY_AIR_ID] = 0;
        self.trace_heights[MERKLE_AIR_ID] = 0;
        self.trace_heights[poseidon2_idx] = 0;

        self.page_indices.clear();
        self.addr_space_access_count.fill(0);

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
        self.segmentation_ctx.instret += seg_check_insns;
        self.segmentation_ctx.instrets_until_check = seg_check_insns;

        self.flush_page_buffer(mem_len, pv_len, deferral_len);
        self.apply_height_updates();

        let instret = self.segmentation_ctx.instret - remaining_counter as u64;
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
pub unsafe extern "C" fn metered_periodic_check(t: *mut MeteredTracerData) {
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

    seg_state.on_periodic_check(mem_len, pv_len, deferral_len, tracer.check_counter);

    // Reset the countdown for the next interval.
    tracer.check_counter += seg_state.segmentation_ctx.segment_check_insns as u32;
}

impl<F: PrimeField32> RvrMeteredInstance<F> {
    pub fn execute_metered(
        &self,
        inputs: impl Into<Streams<F>>,
        ctx: MeteredCtx,
    ) -> Result<(Vec<Segment>, VmState<F, GuestMemory>), ExecutionError> {
        let vm_state = VmState::initial(
            &self.system_config,
            &self.exe.init_memory,
            self.exe.pc_start,
            inputs,
        );
        self.execute_metered_from_state(vm_state, ctx)
    }

    pub fn execute_metered_from_state(
        &self,
        mut vm_state: VmState<F, GuestMemory>,
        ctx: MeteredCtx,
    ) -> Result<(Vec<Segment>, VmState<F, GuestMemory>), ExecutionError> {
        let seg_state = SegmentationState::new(ctx, &self.system_config);

        #[cfg(feature = "metrics")]
        let start = std::time::Instant::now();
        let result_seg_state = tracing::info_span!("execute_metered")
            .in_scope(|| {
                execute_metered(&self.compiled, &self.extensions, &mut vm_state, seg_state)
            })
            .map_err(map_rvr_execute_error)?;
        let result_seg_ctx = result_seg_state.segmentation_ctx;
        #[cfg(feature = "metrics")]
        {
            let elapsed = start.elapsed();
            let insns = result_seg_ctx.instret;
            tracing::info!("instructions_executed={insns}");
            metrics::counter!("execute_metered_insns").absolute(insns);
            metrics::gauge!("execute_metered_insn_mi/s")
                .set(insns as f64 / elapsed.as_micros() as f64);
        }

        Ok((result_seg_ctx.segments, vm_state))
    }
}
