use std::collections::{BTreeMap, BTreeSet};

use super::{
    execution_control::{
        E1ExecutionControl, ExecutionControl, MeteredExecutionControl, TracegenExecutionControl,
    },
    ExecutionError, GenerationError, SystemConfig, VmChipComplex, VmComplexTraceHeights, VmConfig,
};
#[cfg(feature = "bench-metrics")]
use crate::metrics::VmMetrics;
use crate::{
    arch::{instructions::*, InstructionExecutor},
    system::{
        connector::DEFAULT_SUSPEND_EXIT_CODE,
        memory::{dimensions::MemoryDimensions, online::GuestMemory, CHUNK, CHUNK_BITS},
    },
};
use backtrace::Backtrace;
use openvm_instructions::{
    exe::FnBounds,
    instruction::{DebugInfo, Instruction},
};
use openvm_stark_backend::{
    config::{Domain, StarkGenericConfig},
    keygen::types::LinearConstraint,
    p3_commit::PolynomialSpace,
    p3_field::PrimeField32,
    p3_util::log2_strict_usize,
    prover::types::{CommittedTraceData, ProofInput},
    utils::metrics_span,
    Chip,
};
use riscv::RV32_IMM_AS;

const BOUNDARY_CHIP_IDX: usize = 2;

pub struct VmSegmentExecutor<F, VC, Ctrl>
where
    F: PrimeField32,
    VC: VmConfig<F>,
    Ctrl: ExecutionControl<F, VC>,
{
    pub chip_complex: VmChipComplex<F, VC::Executor, VC::Periphery>,
    /// Execution control for determining segmentation and stopping conditions
    pub ctrl: Ctrl,

    pub trace_height_constraints: Vec<LinearConstraint>,

    /// Air names for debug purposes only.
    pub(crate) air_names: Vec<String>,
    /// Metrics collected for this execution segment alone.
    #[cfg(feature = "bench-metrics")]
    pub metrics: VmMetrics,
}

impl<F, VC, Ctrl> VmSegmentExecutor<F, VC, Ctrl>
where
    F: PrimeField32,
    VC: VmConfig<F>,
    Ctrl: ExecutionControl<F, VC>,
{
    /// Creates a new execution segment from a program and initial state, using parent VM config
    pub fn new(
        chip_complex: VmChipComplex<F, VC::Executor, VC::Periphery>,
        trace_height_constraints: Vec<LinearConstraint>,
        #[allow(unused_variables)] fn_bounds: FnBounds,
        ctrl: Ctrl,
    ) -> Self {
        let air_names = chip_complex.air_names();

        Self {
            chip_complex,
            ctrl,
            air_names,
            trace_height_constraints,
            #[cfg(feature = "bench-metrics")]
            metrics: VmMetrics {
                fn_bounds,
                ..Default::default()
            },
        }
    }

    pub fn system_config(&self) -> &SystemConfig {
        self.chip_complex.config()
    }

    pub fn set_override_trace_heights(&mut self, overridden_heights: VmComplexTraceHeights) {
        self.chip_complex
            .set_override_system_trace_heights(overridden_heights.system);
        self.chip_complex
            .set_override_inventory_trace_heights(overridden_heights.inventory);
    }

    /// Stopping is triggered by should_stop() or if VM is terminated
    pub fn execute_from_pc(
        &mut self,
        pc: u32,
        memory: Option<GuestMemory>,
        ctx: Ctrl::Ctx,
    ) -> Result<ExecutionSegmentState<Ctrl::Ctx>, ExecutionError> {
        let mut prev_backtrace: Option<Backtrace> = None;

        let mut state = ExecutionSegmentState::new(pc, memory, ctx, 0, false);

        // Call the pre-execution hook
        self.ctrl
            .on_segment_start(&mut state, &mut self.chip_complex);

        loop {
            // Fetch, decode and execute single instruction
            let terminated_exit_code = self.execute_instruction(&mut state, &mut prev_backtrace)?;

            if let Some(exit_code) = terminated_exit_code {
                state.exit_code = exit_code;
                state.is_terminated = true;
                self.ctrl
                    .on_terminate(&mut state, &mut self.chip_complex, exit_code);
                break;
            }
            if self.should_suspend(&mut state) {
                state.exit_code = DEFAULT_SUSPEND_EXIT_CODE;
                self.ctrl.on_segment_end(&mut state, &mut self.chip_complex);
                break;
            }
        }

        Ok(state)
    }

    /// Executes a single instruction and updates VM state
    // TODO(ayush): clean this up, separate to smaller functions
    fn execute_instruction(
        &mut self,
        state: &mut ExecutionSegmentState<Ctrl::Ctx>,
        prev_backtrace: &mut Option<Backtrace>,
    ) -> Result<Option<u32>, ExecutionError> {
        let pc = state.pc;
        let timestamp = self.chip_complex.memory_controller().timestamp();

        // Process an instruction and update VM state
        let (instruction, debug_info) = self.chip_complex.base.program_chip.get_instruction(pc)?;

        tracing::trace!("pc: {pc:#x} | time: {timestamp} | {:?}", instruction);

        let &Instruction { opcode, c, .. } = instruction;

        // Handle termination instruction
        if opcode == SystemOpcode::TERMINATE.global_opcode() {
            return Ok(Some(c.as_canonical_u32()));
        }

        // Extract debug info components
        #[allow(unused_variables)]
        let (dsl_instr, trace) = debug_info.as_ref().map_or(
            (None, None),
            |DebugInfo {
                 dsl_instruction,
                 trace,
             }| (Some(dsl_instruction.clone()), trace.as_ref()),
        );

        // Handle phantom instructions
        if opcode == SystemOpcode::PHANTOM.global_opcode() {
            let discriminant = c.as_canonical_u32() as u16;
            if let Some(phantom) = SysPhantom::from_repr(discriminant) {
                tracing::trace!("pc: {pc:#x} | system phantom: {phantom:?}");

                if phantom == SysPhantom::DebugPanic {
                    if let Some(mut backtrace) = prev_backtrace.take() {
                        backtrace.resolve();
                        eprintln!("openvm program failure; backtrace:\n{:?}", backtrace);
                    } else {
                        eprintln!("openvm program failure; no backtrace");
                    }
                    return Err(ExecutionError::Fail { pc });
                }

                #[cfg(feature = "bench-metrics")]
                {
                    let dsl_str = dsl_instr.clone().unwrap_or_else(|| "Default".to_string());
                    match phantom {
                        SysPhantom::CtStart => self.metrics.cycle_tracker.start(dsl_str),
                        SysPhantom::CtEnd => self.metrics.cycle_tracker.end(dsl_str),
                        _ => {}
                    }
                }
            }
        }

        // TODO(ayush): move to vm state?
        *prev_backtrace = trace.cloned();

        // Execute the instruction using the control implementation
        // TODO(AG): maybe avoid cloning the instruction?
        self.ctrl
            .execute_instruction(state, &instruction.clone(), &mut self.chip_complex)?;

        // Update metrics if enabled
        #[cfg(feature = "bench-metrics")]
        {
            self.update_instruction_metrics(pc, opcode, dsl_instr);
        }

        Ok(None)
    }

    /// Returns bool of whether to switch to next segment or not.
    fn should_suspend(&mut self, state: &mut ExecutionSegmentState<Ctrl::Ctx>) -> bool {
        if !self.system_config().continuation_enabled {
            return false;
        }

        // Check with the execution control policy
        self.ctrl.should_suspend(state, &self.chip_complex)
    }

    // TODO(ayush): this is not relevant for e1/e2 execution
    /// Generate ProofInput to prove the segment. Should be called after ::execute
    pub fn generate_proof_input<SC: StarkGenericConfig>(
        #[allow(unused_mut)] mut self,
        cached_program: Option<CommittedTraceData<SC>>,
    ) -> Result<ProofInput<SC>, GenerationError>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
        VC::Executor: Chip<SC>,
        VC::Periphery: Chip<SC>,
    {
        metrics_span("trace_gen_time_ms", || {
            self.chip_complex.generate_proof_input(
                cached_program,
                &self.trace_height_constraints,
                #[cfg(feature = "bench-metrics")]
                &mut self.metrics,
            )
        })
    }

    #[cfg(feature = "bench-metrics")]
    #[allow(unused_variables)]
    pub fn update_instruction_metrics(
        &mut self,
        pc: u32,
        opcode: VmOpcode,
        dsl_instr: Option<String>,
    ) {
        self.metrics.cycle_count += 1;

        if self.system_config().profiling {
            let executor = self.chip_complex.inventory.get_executor(opcode).unwrap();
            let opcode_name = executor.get_opcode_name(opcode.as_usize());
            self.metrics.update_trace_cells(
                &self.air_names,
                self.chip_complex.current_trace_cells(),
                opcode_name,
                dsl_instr,
            );

            #[cfg(feature = "function-span")]
            self.metrics.update_current_fn(pc);
        }
    }
}

// TODO(ayush): add clk cycle count
// E1 execution
pub type E1Ctx = ();
pub type E1VmSegmentExecutor<F, VC> = VmSegmentExecutor<F, VC, E1ExecutionControl>;

// TODO(ayush): can segmentation also be triggered by timestamp overflow? should that be tracked?
// E2 (metered) execution
#[derive(Debug)]
pub struct MeteredCtx {
    // Trace heights for each chip
    pub trace_heights: Vec<usize>,

    continuations_enabled: bool,
    num_access_adapters: usize,
    as_byte_alignment_bits: Vec<usize>,
    pub memory_dimensions: MemoryDimensions,

    // Map from (addr_space, addr) -> (size, offset)
    pub last_memory_access: BTreeMap<(u8, u32), (u8, u8)>,
    // Indices of leaf nodes in the memory merkle tree
    pub leaf_indices: Vec<u64>,
}

impl MeteredCtx {
    pub fn new(
        num_traces: usize,
        continuations_enabled: bool,
        num_access_adapters: usize,
        as_byte_alignment_bits: Vec<usize>,
        memory_dimensions: MemoryDimensions,
    ) -> Self {
        Self {
            trace_heights: vec![0; num_traces],
            continuations_enabled,
            num_access_adapters,
            as_byte_alignment_bits,
            memory_dimensions,
            last_memory_access: BTreeMap::new(),
            leaf_indices: Vec::new(),
        }
    }
}

pub type MeteredVmSegmentExecutor<'a, F, VC> =
    VmSegmentExecutor<F, VC, MeteredExecutionControl<'a>>;

// E3 (tracegen) execution
pub type TracegenCtx = ();
pub type TracegenVmSegmentExecutor<F, VC> = VmSegmentExecutor<F, VC, TracegenExecutionControl>;

#[derive(derive_new::new)]
pub struct ExecutionSegmentState<Ctx> {
    pub pc: u32,
    pub memory: Option<GuestMemory>,
    pub ctx: Ctx,
    // TODO(ayush): do we need both exit_code and is_terminated?
    pub exit_code: u32,
    pub is_terminated: bool,
}

// TODO(ayush): better name
pub trait E1E2ExecutionCtx {
    fn on_memory_operation(&mut self, address_space: u32, ptr: u32, size: usize);
}

impl E1E2ExecutionCtx for E1Ctx {
    fn on_memory_operation(&mut self, _address_space: u32, _ptr: u32, _size: usize) {}
}

impl MeteredCtx {
    fn update_boundary_merkle_heights(&mut self, address_space: u32, ptr: u32, size: usize) {
        let num_blocks = (size + CHUNK - 1) >> CHUNK_BITS;
        for i in 0..num_blocks {
            let addr = ptr.wrapping_add((i * CHUNK) as u32);
            let block_id = addr >> CHUNK_BITS;
            let leaf_id = self
                .memory_dimensions
                .label_to_index((address_space, block_id));

            if let Err(insert_idx) = self.leaf_indices.binary_search(&leaf_id) {
                self.leaf_indices.insert(insert_idx, leaf_id);
                self.trace_heights[BOUNDARY_CHIP_IDX] += 1;

                if self.continuations_enabled {
                    let pred_id = insert_idx.checked_sub(1).map(|idx| self.leaf_indices[idx]);
                    let succ_id = (insert_idx < self.leaf_indices.len() - 1)
                        .then(|| self.leaf_indices[insert_idx + 1]);
                    let height_change = calculate_merkle_node_updates(
                        pred_id,
                        succ_id,
                        leaf_id,
                        self.memory_dimensions.overall_height(),
                    );
                    self.trace_heights[BOUNDARY_CHIP_IDX + 1] += height_change * 2;
                }
            }
        }
    }

    #[allow(clippy::type_complexity)]
    fn calculate_splits_and_merges(
        &self,
        address_space: u32,
        ptr: u32,
        size: usize,
    ) -> (Vec<(u32, usize)>, Vec<(u32, usize)>) {
        // Skip adapters if this is a repeated access to the same location with same size
        let last_access = self.last_memory_access.get(&(address_space as u8, ptr));
        if matches!(last_access, Some(&(last_access_size, 0)) if size == last_access_size as usize)
        {
            return (vec![], vec![]);
        }

        // Go to the start of block
        let mut ptr_start = ptr;
        if let Some(&(_, last_access_offset)) = last_access {
            ptr_start = ptr.wrapping_sub(last_access_offset as u32);
        }

        let align_bits = self.as_byte_alignment_bits[address_space as usize];
        let align = 1 << align_bits;

        // Split intersecting blocks to align bytes
        let mut curr_block = ptr_start >> align_bits;
        let end_block = curr_block + (size as u32 >> align_bits);
        let mut splits = vec![];
        while curr_block < end_block {
            let curr_block_size = if let Some(&(last_access_size, _)) = self
                .last_memory_access
                .get(&(address_space as u8, curr_block.wrapping_mul(align as u32)))
            {
                last_access_size as usize
            } else {
                // Initial memory access only happens at CHUNK boundary
                let chunk_ratio = 1 << (CHUNK_BITS - align_bits);
                let chunk_offset = curr_block & (chunk_ratio - 1);
                curr_block -= chunk_offset;
                CHUNK
            };

            // TODO(ayush): why are we splitting in the case when we only
            //              read at mutually exclusive CHUNK boundaries?
            if curr_block_size > align {
                let curr_ptr = curr_block.wrapping_mul(align as u32);
                splits.push((curr_ptr, curr_block_size));
            }

            curr_block += (curr_block_size >> align_bits) as u32;
        }
        // Merge added blocks from align to size bytes
        let merges = vec![(ptr, size)];

        (splits, merges)
    }

    #[allow(clippy::type_complexity)]
    fn apply_adapter_updates(
        &mut self,
        addr_space: u32,
        ptr: u32,
        size: usize,
        trace_heights: &mut [usize],
        memory_updates: &mut Option<Vec<((u8, u32), Option<(u8, u8)>)>>,
    ) {
        let (splits, merges) = self.calculate_splits_and_merges(addr_space, ptr, size);
        for (curr_ptr, curr_size) in splits {
            apply_single_adapter_heights_update(trace_heights, curr_size);
            let updates = add_memory_access_split_with_return(
                &mut self.last_memory_access,
                (addr_space, curr_ptr),
                curr_size,
                self.as_byte_alignment_bits[addr_space as usize],
            );
            if let Some(memory_updates) = memory_updates {
                memory_updates.extend(&updates);
            }
        }
        for (curr_ptr, curr_size) in merges {
            apply_single_adapter_heights_update(trace_heights, curr_size);
            let updates = add_memory_access_merge_with_return(
                &mut self.last_memory_access,
                (addr_space, curr_ptr),
                curr_size,
                self.as_byte_alignment_bits[addr_space as usize],
            );
            if let Some(memory_updates) = memory_updates {
                memory_updates.extend(updates);
            }
        }
    }

    fn update_adapter_heights(&mut self, addr_space: u32, ptr: u32, size: usize) {
        let adapter_offset = if self.continuations_enabled {
            BOUNDARY_CHIP_IDX + 2
        } else {
            BOUNDARY_CHIP_IDX + 1
        };

        let mut trace_height_updates = vec![0; self.num_access_adapters];
        self.apply_adapter_updates(addr_space, ptr, size, &mut trace_height_updates, &mut None);
        for (i, height) in trace_height_updates.iter().enumerate() {
            self.trace_heights[adapter_offset + i] += height;
        }
    }

    pub fn finalize_access_adapter_heights(&mut self) {
        let indices_to_process: Vec<_> = self
            .leaf_indices
            .iter()
            .map(|&idx| {
                let (addr_space, block_id) = self.memory_dimensions.index_to_label(idx);
                (addr_space, block_id)
            })
            .collect();
        for (addr_space, block_id) in indices_to_process {
            self.update_adapter_heights(addr_space, block_id * CHUNK as u32, CHUNK);
        }
    }

    pub fn trace_heights_if_finalized(&mut self) -> Vec<usize> {
        let indices_to_process: Vec<_> = self
            .leaf_indices
            .iter()
            .map(|&idx| {
                let (addr_space, block_id) = self.memory_dimensions.index_to_label(idx);
                (addr_space, block_id)
            })
            .collect();

        let mut trace_height_updates = vec![];
        let mut memory_updates = Some(vec![]);
        for (addr_space, block_id) in indices_to_process {
            let ptr = block_id * CHUNK as u32;
            self.apply_adapter_updates(
                addr_space,
                ptr,
                CHUNK,
                &mut trace_height_updates,
                &mut memory_updates,
            );
        }

        // Restore original memory state
        for (key, old_value) in memory_updates.unwrap().into_iter().rev() {
            match old_value {
                Some(value) => {
                    self.last_memory_access.insert(key, value);
                }
                None => {
                    self.last_memory_access.remove(&key);
                }
            }
        }

        let adapter_offset = if self.continuations_enabled {
            BOUNDARY_CHIP_IDX + 2
        } else {
            BOUNDARY_CHIP_IDX + 1
        };
        self.trace_heights
            .iter()
            .enumerate()
            .map(|(i, &height)| {
                if i >= adapter_offset && i < adapter_offset + trace_height_updates.len() {
                    height + trace_height_updates[i - adapter_offset]
                } else {
                    height
                }
            })
            .collect()
    }
}

impl E1E2ExecutionCtx for MeteredCtx {
    fn on_memory_operation(&mut self, address_space: u32, ptr: u32, size: usize) {
        debug_assert!(
            address_space != RV32_IMM_AS,
            "address space must not be immediate"
        );
        debug_assert!(size.is_power_of_two(), "size must be a power of 2");

        // Handle access adapter updates
        self.update_adapter_heights(address_space, ptr, size);

        // Handle merkle tree updates
        // TODO(ayush): see if this can be approximated by total number of reads/writes for AS != register
        self.update_boundary_merkle_heights(address_space, ptr, size);

        // TODO(ayush): Poseidon2PeripheryChip
    }
}

fn apply_single_adapter_heights_update(trace_heights: &mut [usize], size: usize) {
    let size_bits = log2_strict_usize(size);
    for adapter_bits in (3..=size_bits).rev() {
        trace_heights[adapter_bits - 1] += 1 << (size_bits - adapter_bits);
    }
}

#[allow(clippy::type_complexity)]
fn add_memory_access(
    memory_access_map: &mut BTreeMap<(u8, u32), (u8, u8)>,
    (address_space, ptr): (u32, u32),
    size: usize,
    align_bits: usize,
    is_split: bool,
) -> Vec<((u8, u32), Option<(u8, u8)>)> {
    let align = 1 << align_bits;
    debug_assert_eq!(
        size & (align - 1),
        0,
        "Size must be a multiple of alignment"
    );

    let num_chunks = size >> align_bits;
    let mut old_values = Vec::with_capacity(num_chunks);

    for i in 0..num_chunks {
        let curr_ptr = ptr.wrapping_add(i as u32 * align as u32);
        let key = (address_space as u8, curr_ptr);

        let value = if is_split {
            (align as u8, 0)
        } else {
            (size as u8, (i * align) as u8)
        };

        let old_value = memory_access_map.insert(key, value);
        old_values.push((key, old_value));
    }

    old_values
}

#[allow(clippy::type_complexity)]
fn add_memory_access_split_with_return(
    memory_access_map: &mut BTreeMap<(u8, u32), (u8, u8)>,
    (address_space, ptr): (u32, u32),
    size: usize,
    align_bits: usize,
) -> Vec<((u8, u32), Option<(u8, u8)>)> {
    add_memory_access(
        memory_access_map,
        (address_space, ptr),
        size,
        align_bits,
        true,
    )
}

#[allow(clippy::type_complexity)]
fn add_memory_access_merge_with_return(
    memory_access_map: &mut BTreeMap<(u8, u32), (u8, u8)>,
    (address_space, ptr): (u32, u32),
    size: usize,
    align_bits: usize,
) -> Vec<((u8, u32), Option<(u8, u8)>)> {
    add_memory_access(
        memory_access_map,
        (address_space, ptr),
        size,
        align_bits,
        false,
    )
}

fn calculate_merkle_node_updates(
    pred_id: Option<u64>,
    succ_id: Option<u64>,
    leaf_id: u64,
    height: usize,
) -> usize {
    // First node requires height many updates
    if pred_id.is_none() && succ_id.is_none() {
        return height;
    }

    // Calculate the difference in divergence
    let mut diff = 0;

    // Add new divergences between pred and leaf_index
    if let Some(p) = pred_id {
        let new_divergence = (p ^ leaf_id).ilog2() as usize;
        diff += new_divergence;
    }

    // Add new divergences between leaf_index and succ
    if let Some(s) = succ_id {
        let new_divergence = (leaf_id ^ s).ilog2() as usize;
        diff += new_divergence;
    }

    // Remove old divergence between pred and succ if both existed
    if let (Some(p), Some(s)) = (pred_id, succ_id) {
        let old_divergence = (p ^ s).ilog2() as usize;
        diff -= old_divergence;
    }

    diff
}
