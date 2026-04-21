//! Per-chip metered execution: page tracking and segmentation
//! matching OpenVM's `MeteredCtx`.

use openvm_instructions::{
    exe::VmExe, riscv::RV32_MEMORY_AS, LocalOpcode, SystemOpcode, VmOpcode, DEFERRAL_AS,
};
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm::{DEFERRAL_PAGE_BUF_CAP, MEM_PAGE_BUF_CAP, PV_PAGE_BUF_CAP};
use rvr_state::TracerState;

use crate::{
    arch::{
        execution_mode::metered::{
            ctx::DEFAULT_PAGE_BITS,
            segment_ctx::{SegmentationConfig, DEFAULT_SEGMENT_CHECK_INSNS},
        },
        ExecutorInventory, SystemConfig,
    },
    system::memory::{merkle::public_values::PUBLIC_VALUES_AS, CHUNK as MERKLE_CHUNK},
};

/// Constant overhead for interaction memory (matches OpenVM's
/// DEFAULT_INTERACTION_CONSTANT_OVERHEAD).
const INTERACTION_CONSTANT_OVERHEAD: usize = 2 << 20; // 2 MiB

const NO_CHIP: u32 = u32::MAX;

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
    pub seg_state: *mut std::ffi::c_void,
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

/// Pointer wrapper stored in RvState's tracer field. Matches C `Tracer*` (8 bytes).
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct MeteredTracer(pub *mut MeteredTracerData);

impl Default for MeteredTracer {
    fn default() -> Self {
        Self(std::ptr::null_mut())
    }
}

impl TracerState for MeteredTracer {
    const KIND: u32 = 11;
}

impl std::ops::Deref for MeteredTracer {
    type Target = MeteredTracerData;
    fn deref(&self) -> &MeteredTracerData {
        unsafe { &*self.0 }
    }
}

impl std::ops::DerefMut for MeteredTracer {
    fn deref_mut(&mut self) -> &mut MeteredTracerData {
        unsafe { &mut *self.0 }
    }
}

// ── Configuration ────────────────────────────────────────────────────────────

/// Configuration for per-chip metered execution.
#[derive(Clone)]
pub struct MeteredConfig {
    /// pc_index -> chip_idx (u32::MAX = no chip).
    pub pc_to_chip: Vec<u32>,
    pub pc_base: u32,
    /// Per-AIR widths.
    pub widths: Vec<usize>,
    /// Per-AIR interaction counts (for max_interactions check).
    pub interactions: Vec<usize>,
    /// AIR index for boundary chip.
    pub boundary_idx: usize,
    /// AIR index for merkle tree chip.
    pub merkle_tree_idx: usize,
    /// Initial trace_heights (constants filled in, variables at 0).
    pub initial_trace_heights: Vec<u32>,
    /// Which heights are constant (never change across segments).
    pub is_constant: Vec<bool>,
    /// Segmentation config.
    pub segmentation_config: SegmentationConfig,
    /// Instructions between segmentation checks.
    pub segment_check_insns: u64,
    /// Memory dimensions for page computation.
    pub address_height: u32,
    pub addr_space_height: u32,
    pub chunk_bits: u32,
    /// Number of address spaces configured.
    pub num_addr_spaces: usize,
    /// Chip index for HINT_BUFFER/HINT_STOREW (for IO corrections).
    pub hint_store_chip_idx: Option<u32>,
}

impl MeteredConfig {
    /// Extract chip mapping for compilation.
    pub fn chip_mapping(&self) -> super::compile::ChipMapping {
        super::compile::ChipMapping {
            pc_to_chip: self.pc_to_chip.clone(),
            hint_store_chip_idx: self.hint_store_chip_idx,
            chip_widths: None,
        }
    }
}

/// Build a [`MeteredConfig`] from OpenVM configuration.
#[allow(clippy::too_many_arguments)]
pub fn build_metered_config<F, E>(
    exe: &VmExe<F>,
    inventory: &ExecutorInventory<E>,
    executor_idx_to_air_idx: &[usize],
    widths: &[usize],
    interactions: &[usize],
    constant_trace_heights: &[Option<usize>],
    system_config: &SystemConfig,
    hint_buffer_opcode: Option<VmOpcode>,
) -> MeteredConfig
where
    F: PrimeField32,
{
    let program = &exe.program;
    let pc_base = program.pc_base;
    let terminate_opcode = SystemOpcode::TERMINATE.global_opcode();

    let hint_store_chip_idx = hint_buffer_opcode.and_then(|opcode| {
        inventory
            .instruction_lookup
            .get(&opcode)
            .map(|&executor_idx| executor_idx_to_air_idx[executor_idx as usize] as u32)
    });

    let pc_to_chip: Vec<u32> = program
        .instructions_and_debug_infos
        .iter()
        .map(|slot| {
            if let Some((inst, _)) = slot {
                let opcode: VmOpcode = inst.opcode;
                if opcode == terminate_opcode {
                    NO_CHIP
                } else if let Some(&executor_idx) = inventory.instruction_lookup.get(&opcode) {
                    executor_idx_to_air_idx[executor_idx as usize] as u32
                } else {
                    NO_CHIP
                }
            } else {
                NO_CHIP
            }
        })
        .collect();

    let boundary_idx = system_config.memory_boundary_air_id();
    let merkle_tree_idx = system_config.memory_merkle_air_id();

    let initial_trace_heights: Vec<u32> = constant_trace_heights
        .iter()
        .map(|opt| opt.map(|h| h as u32).unwrap_or(0))
        .collect();
    let is_constant: Vec<bool> = constant_trace_heights
        .iter()
        .map(|opt| opt.is_some())
        .collect();

    let segmentation_config = system_config.segmentation_config.clone();

    let mem_config = &system_config.memory_config;
    let chunk_bits = MERKLE_CHUNK.ilog2();
    let memory_dimensions = mem_config.memory_dimensions();
    let address_height = memory_dimensions.address_height as u32;
    // addr_spaces has length (1 << addr_space_height) + 1 (index 0 is unused)
    let num_addr_spaces = mem_config.addr_spaces.len();

    let addr_space_height = mem_config.addr_space_height as u32;

    MeteredConfig {
        pc_to_chip,
        pc_base,
        widths: widths.to_vec(),
        interactions: interactions.to_vec(),
        boundary_idx,
        merkle_tree_idx,
        initial_trace_heights,
        is_constant,
        segmentation_config,
        segment_check_insns: DEFAULT_SEGMENT_CHECK_INSNS,
        address_height,
        addr_space_height,
        chunk_bits,
        num_addr_spaces,
        hint_store_chip_idx,
    }
}

// ── Page tracking ────────────────────────────────────────────────────────────

/// Efficient bitset for tracking unique pages.
pub struct BitSet {
    words: Vec<u64>,
}

impl BitSet {
    pub fn new(num_bits: usize) -> Self {
        Self {
            words: vec![0u64; num_bits.div_ceil(64)],
        }
    }

    /// Insert a bit, returning true if it was NOT previously set (newly inserted).
    pub fn insert(&mut self, index: usize) -> bool {
        let word_idx = index >> 6;
        let bit_idx = index & 63;
        let mask = 1u64 << bit_idx;
        let word = &mut self.words[word_idx];
        let was_set = (*word & mask) != 0;
        *word |= mask;
        !was_set
    }

    pub fn clear(&mut self) {
        self.words.fill(0);
    }
}

// ── Segmentation runtime ─────────────────────────────────────────────────────

/// A completed execution segment.
#[derive(Clone, Debug)]
pub struct RvrSegment {
    pub instret_start: u64,
    pub num_insns: u64,
    pub trace_heights: Vec<u32>,
}

/// Result of per-chip metered execution.
pub struct RvrMeteredResult {
    pub segments: Vec<RvrSegment>,
    pub instret: u64,
}

/// Runtime state for metered segmentation.
// TODO: generalize non-memory page buffers to a config-driven set of N
// buffers (one per additional address space) instead of hardcoding
// pv + deferral. The memory AS buffer stays separate as the hot path.
pub struct SegmentationState {
    config: MeteredConfig,
    // Owned data arrays that C tracer points into
    trace_heights: Vec<u32>,
    /// Per-address-space page buffers. Each entry = 1 u32 page id.
    mem_page_buf: Vec<u32>,
    pv_page_buf: Vec<u32>,
    deferral_page_buf: Vec<u32>,
    // Page tracking
    page_indices: BitSet,
    addr_space_access_count: Vec<u32>,
    // Checkpoint for segmentation rollback
    checkpoint_trace_heights: Vec<u32>,
    checkpoint_instret: u64,
    // Segments
    segments: Vec<RvrSegment>,
    instret: u64,
}

impl SegmentationState {
    pub fn new(config: MeteredConfig) -> Self {
        let trace_heights = config.initial_trace_heights.clone();
        let checkpoint_trace_heights = trace_heights.clone();

        // Compute total number of pages for BitSet sizing.
        // BitSet covers pages for all address spaces in the merkle tree.
        let overall_height = config.addr_space_height as usize + config.address_height as usize;
        let bitset_size = 1usize << overall_height.saturating_sub(DEFAULT_PAGE_BITS);
        let page_indices = BitSet::new(bitset_size);
        let addr_space_access_count = vec![0u32; config.num_addr_spaces];

        let mem_page_buf = vec![0u32; MEM_PAGE_BUF_CAP];
        let pv_page_buf = vec![0u32; PV_PAGE_BUF_CAP];
        let deferral_page_buf = vec![0u32; DEFERRAL_PAGE_BUF_CAP];

        let mut state = Self {
            config,
            trace_heights,
            mem_page_buf,
            pv_page_buf,
            deferral_page_buf,
            page_indices,
            addr_space_access_count,
            checkpoint_trace_heights,
            checkpoint_instret: 0,
            segments: Vec::new(),
            instret: 0,
        };

        // Match OpenVM: add initial register merkle height contributions
        state.add_register_merkle_heights();
        state.apply_height_updates();

        // Update checkpoint to include initial heights
        state
            .checkpoint_trace_heights
            .copy_from_slice(&state.trace_heights);

        state
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

    /// Get the config reference.
    pub fn config(&self) -> &MeteredConfig {
        &self.config
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

        let chunk_bits = self.config.chunk_bits;
        let chunk = 1u32 << chunk_bits;
        let num_blocks = (REG_SIZE + chunk - 1) >> chunk_bits;
        let start_chunk_id = 0u32; // ptr=0
                                   // label_to_index: ((addr_space - 1) << address_height) + chunk_id
        let start_block_id =
            ((REG_AS as u64 - 1) << self.config.address_height) + start_chunk_id as u64;
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
        let page_shift = self.config.address_height as usize - DEFAULT_PAGE_BITS;
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

        // Boundary chip: 2 rows per leaf (init + final)
        self.trace_heights[self.config.boundary_idx] += leaves * 2;

        // Merkle tree + Poseidon2
        let merkle_idx = self.config.merkle_tree_idx;
        let poseidon2_idx = self.trace_heights.len() - 2;
        self.trace_heights[poseidon2_idx] += leaves * 2;

        let merkle_height =
            self.config.addr_space_height as usize + self.config.address_height as usize;
        let nodes =
            (((1usize << DEFAULT_PAGE_BITS) - 1) + (merkle_height - DEFAULT_PAGE_BITS)) as u32;
        self.trace_heights[poseidon2_idx] += nodes * page_access_count * 2;
        self.trace_heights[merkle_idx] += nodes * page_access_count * 2;

        // Reset counts
        self.addr_space_access_count.fill(0);
    }

    /// Check segmentation limits. Returns true if we should segment.
    ///
    /// Matches OpenVM's `SegmentationCtx::should_segment` with the new memory-based
    /// calculation from v2.0.0-rc.1.
    fn should_segment(&self) -> bool {
        let instret_start = self
            .segments
            .last()
            .map_or(0, |s| s.instret_start + s.num_insns);
        let num_insns = self.instret - instret_start;
        if num_insns == 0 {
            return false;
        }

        let config = &self.config.segmentation_config;
        let main_weight = config.main_cell_weight;
        let main_secondary_weight = config.main_cell_secondary_weight;
        let interaction_weight = config.interaction_cell_weight;
        let base_field_size = config.base_field_size;

        let mut main_cnt = 0usize;
        let mut interaction_cnt = 0usize;
        for i in 0..self.trace_heights.len() {
            let h = self.trace_heights[i];
            let padded = h.next_power_of_two();
            if !self.config.is_constant[i] && padded > config.limits.max_trace_height {
                return true;
            }
            let padded = padded as usize;
            main_cnt += padded * self.config.widths[i];
            interaction_cnt += padded * self.config.interactions[i];
        }

        let main_memory = main_cnt * main_weight * base_field_size;
        let main_secondary_memory =
            ((main_cnt * base_field_size) as f64 * main_secondary_weight).ceil() as usize;
        let interaction_memory = (((interaction_cnt + 1).next_power_of_two() * base_field_size)
            as f64
            * interaction_weight)
            .ceil() as usize
            + INTERACTION_CONSTANT_OVERHEAD;
        let total_memory = main_memory + std::cmp::max(main_secondary_memory, interaction_memory);

        if total_memory > config.limits.max_memory {
            return true;
        }

        let total_interactions: usize = self
            .trace_heights
            .iter()
            .zip(self.config.interactions.iter())
            .map(|(&h, &i)| (h + 1) as usize * i)
            .sum();
        if total_interactions > config.limits.max_interactions {
            return true;
        }

        false
    }

    /// Create a segment from checkpoint heights.
    fn create_segment(&mut self) {
        let instret_start = self
            .segments
            .last()
            .map_or(0, |s| s.instret_start + s.num_insns);
        let num_insns = self.checkpoint_instret - instret_start;
        self.segments.push(RvrSegment {
            instret_start,
            num_insns,
            trace_heights: self.checkpoint_trace_heights.clone(),
        });
    }

    /// Initialize state for a new segment after segmentation.
    ///
    /// Matches OpenVM's `segment_ctx.initialize_segment` + `memory_ctx.initialize_segment`.
    /// The page buffer contents from the batch that triggered segmentation are
    /// still intact, so we replay them via `flush_page_buffer`.
    fn initialize_new_segment(&mut self, mem_len: u32, pv_len: u32, deferral_len: u32) {
        // Step 1: Subtract checkpoint heights from current to get the delta
        // (matches OpenVM's segment_ctx.reset_trace_heights).
        for i in 0..self.trace_heights.len() {
            if !self.config.is_constant[i] {
                self.trace_heights[i] =
                    self.trace_heights[i].wrapping_sub(self.checkpoint_trace_heights[i]);
            }
        }

        // Step 2: Reset memory-specific heights to 0 (will be recomputed from page replay).
        self.trace_heights[self.config.boundary_idx] = 0;
        self.trace_heights[self.config.merkle_tree_idx] = 0;
        let poseidon2_idx = self.trace_heights.len() - 2;
        self.trace_heights[poseidon2_idx] = 0;

        // Step 3: Clear page tracking and replay current batch pages.
        self.page_indices.clear();
        self.addr_space_access_count.fill(0);
        self.flush_page_buffer(mem_len, pv_len, deferral_len);

        // Step 4: Apply height updates from replayed pages.
        self.apply_height_updates();

        // Step 5: Add register merkle heights and apply.
        self.add_register_merkle_heights();
        self.apply_height_updates();
    }

    /// Validate that a freshly initialized segment is within limits.
    ///
    /// This mirrors OpenVM's post-initialization sanity check and surfaces
    /// cases where the new segment already exceeds segmentation limits.
    fn validate_initialized_segment_state(&self) {
        if !self.should_segment() {
            return;
        }

        let trace_heights_str = self
            .trace_heights
            .iter()
            .enumerate()
            .filter(|(_, &height)| height > 0)
            .map(|(idx, &height)| format!("  [{idx}] = {height}"))
            .collect::<Vec<_>>()
            .join("\n");

        tracing::warn!(
            "Segment initialized with heights that exceed limits\n\
             instret={}\n\
             trace_heights=[\n{}\n]",
            self.instret,
            trace_heights_str
        );
    }

    /// Called on each periodic check (approximately every `segment_check_insns` instructions).
    /// Invoked from the C tracer's `trace_block` callback when the block-level
    /// countdown crosses zero. Returns true if a segment boundary was created.
    pub fn on_periodic_check(&mut self, mem_len: u32, pv_len: u32, deferral_len: u32) -> bool {
        self.instret += self.config().segment_check_insns;

        // Flush all page buffers
        self.flush_page_buffer(mem_len, pv_len, deferral_len);

        // Apply boundary height updates
        self.apply_height_updates();

        // Check segmentation
        let did_segment = if self.should_segment() {
            self.create_segment();
            self.initialize_new_segment(mem_len, pv_len, deferral_len);
            self.validate_initialized_segment_state();
            true
        } else {
            false
        };

        // Update checkpoint
        self.checkpoint_trace_heights
            .copy_from_slice(&self.trace_heights);
        self.checkpoint_instret = self.instret;

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
        // Compute exact instret: accumulate instructions elapsed since last check.
        let elapsed = self.config().segment_check_insns - remaining_counter as u64;
        self.instret += elapsed;

        // Flush and apply remaining pages
        self.flush_page_buffer(mem_len, pv_len, deferral_len);
        self.apply_height_updates();

        // Create final segment
        let instret_start = self
            .segments
            .last()
            .map_or(0, |s| s.instret_start + s.num_insns);
        let num_insns = self.instret - instret_start;
        self.segments.push(RvrSegment {
            instret_start,
            num_insns,
            trace_heights: self.trace_heights.clone(),
        });
    }

    /// Consume and return the result.
    pub fn into_result(self) -> RvrMeteredResult {
        RvrMeteredResult {
            instret: self.instret,
            segments: self.segments,
        }
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

    // Reset the countdown for the next interval.
    tracer.check_counter += seg_state.config().segment_check_insns as u32;

    seg_state.on_periodic_check(mem_len, pv_len, deferral_len);
}
