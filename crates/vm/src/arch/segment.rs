use std::collections::BTreeSet;

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
        memory::{dimensions::MemoryDimensions, CHUNK},
    },
};

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
    pub fn execute_from_pc_with_ctx(
        &mut self,
        pc: u32,
        memory: Option<Mem>,
        ctx: Ctrl::Ctx,
    ) -> Result<ExecutionSegmentState<Mem, Ctrl::Ctx>, ExecutionError> {
        let mut prev_backtrace: Option<Backtrace> = None;

        let mut state = ExecutionSegmentState::new_with_pc_and_ctx(pc, ctx);

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
            if self.should_suspend(&state) {
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
        state: &mut ExecutionSegmentState<Mem, Ctrl::Ctx>,
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
    fn should_suspend(&mut self, state: &ExecutionSegmentState<Ctrl::Ctx>) -> bool {
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

// E1 execution
pub type E1Ctx = ();
pub type E1VmSegmentExecutor<F, VC> = VmSegmentExecutor<F, VC, E1ExecutionControl>;

// E2 (metered) execution
#[derive(Debug)]
pub struct MeteredCtx {
    continuations_enabled: bool,
    memory_dimensions: MemoryDimensions,
    // addr_space_alignment_bytes: Vec<usize>,

    // Trace heights for each chip
    pub trace_heights: Vec<usize>,
    // Accesses of size [1, 2, 4, 8, 16, 32]
    // TODO(ayush): no magic number
    pub memory_ops: [usize; 6],
    // Indices of leaf nodes in the memory merkle tree
    pub leaf_indices: BTreeSet<u64>,
}

impl MeteredCtx {
    pub fn new(
        num_traces: usize,
        continuations_enabled: bool,
        memory_dimensions: MemoryDimensions,
    ) -> Self {
        Self {
            continuations_enabled,
            memory_dimensions,
            trace_heights: vec![0; num_traces],
            memory_ops: [0; 6],
            leaf_indices: BTreeSet::new(),
        }
    }
}

pub type MeteredVmSegmentExecutor<'a, F, VC> =
    VmSegmentExecutor<F, VC, MeteredExecutionControl<'a>>;

// E3 (tracegen) execution
pub type TracegenCtx = ();
pub type TracegenVmSegmentExecutor<F, VC> = VmSegmentExecutor<F, VC, TracegenExecutionControl>;

pub struct ExecutionSegmentState<Mem, Ctx> {
    pub memory: Option<Mem>,
    pub pc: u32,
    // TODO(ayush): do we need both exit_code and is_terminated?
    pub exit_code: u32,
    pub is_terminated: bool,
    pub ctx: Ctx,
}

impl<Ctx> ExecutionSegmentState<Ctx> {
    pub fn new_with_pc_and_ctx(pc: u32, ctx: Ctx) -> Self {
        Self {
            memory: None,
            pc,
            ctx,
            exit_code: 0,
            is_terminated: false,
        }
    }
}

// TODO(ayush): better name
pub trait E1E2ExecutionCtx {
    fn on_memory_operation(&mut self, address_space: u32, ptr: u32, size: usize);
}

impl E1E2ExecutionCtx for E1Ctx {
    fn on_memory_operation(&mut self, _address_space: u32, _ptr: u32, _size: usize) {}
}

impl E1E2ExecutionCtx for MeteredCtx {
    fn on_memory_operation(&mut self, address_space: u32, ptr: u32, size: usize) {
        debug_assert!(address_space != RV32_IMM_AS);

        self.memory_ops[log2_strict_usize(size)] += 1;
        // TODO(ayush): access adapter heights based on this

        // Calculate unique chunks and inner nodes in Merkle tree
        let mt_height = self.memory_dimensions.overall_height();

        // TODO(ayush): see if this can be approximated by total number of reads/writes for AS != register
        let num_chunks = size.div_ceil(CHUNK);
        for i in 0..num_chunks {
            let addr = ptr.wrapping_add((i * CHUNK) as u32);
            let block_id = addr / CHUNK as u32;
            let leaf_index = self
                .memory_dimensions
                .label_to_index((address_space, block_id));

            // TODO(ayush): see if insertion and finding pred/succ can be done in single binary search pass
            if self.leaf_indices.insert(leaf_index) {
                // | Boundary | Merkle | Access Adapters |
                // TODO(ayush): no magic numbers
                let mut offset = 2;

                // Boundary chip
                self.trace_heights[offset] += 1;

                if self.continuations_enabled {
                    // Merkle chip
                    offset += 1;

                    let height_change =
                        calculate_merkle_height_changes(leaf_index, &self.leaf_indices, mt_height);
                    self.trace_heights[offset] += height_change * 2;
                }

                // 8-byte access adapter
                let log2_chunk = log2_strict_usize(CHUNK);
                self.trace_heights[offset + log2_chunk] += 2;
            }
        }
    }
}

/// Updates Merkle tree heights based on a new leaf index
fn calculate_merkle_height_changes(
    leaf_index: u64,
    leaf_indices: &BTreeSet<u64>,
    height: usize,
) -> usize {
    if leaf_indices.len() == 1 {
        return height;
    }

    // Find predecessor and successor nodes
    let pred = leaf_indices.range(..leaf_index).next_back().copied();
    let succ = leaf_indices.range(leaf_index + 1..).next().copied();

    let mut diff = 0;

    // Add new divergences between pred and leaf_index
    if let Some(p) = pred {
        let new_divergence = (p ^ leaf_index).ilog2() as usize;
        diff += new_divergence;
    }

    // Add new divergences between leaf_index and succ
    if let Some(s) = succ {
        let new_divergence = (leaf_index ^ s).ilog2() as usize;
        diff += new_divergence;
    }

    // Remove old divergence between pred and succ if both existed
    if let (Some(p), Some(s)) = (pred, succ) {
        let old_divergence = (p ^ s).ilog2() as usize;
        diff -= old_divergence;
    }

    diff
}
