use std::fmt::Debug;

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
    prover::types::{CommittedTraceData, ProofInput},
    Chip,
};
use rand::rngs::StdRng;
use tracing::instrument;

#[cfg(feature = "bench-metrics")]
use super::InstructionExecutor;
use super::{
    execution_control::ExecutionControl, ExecutionError, GenerationError, Streams, SystemConfig,
    VmChipComplex, VmComplexTraceHeights, VmConfig,
};
#[cfg(feature = "bench-metrics")]
use crate::metrics::VmMetrics;
use crate::{
    arch::{execution_mode::E1ExecutionCtx, instructions::*},
    system::memory::online::GuestMemory,
};

pub struct VmSegmentState<F, Ctx> {
    pub instret: u64,
    pub pc: u32,
    pub memory: GuestMemory,
    pub streams: Streams<F>,
    pub rng: StdRng,
    pub exit_code: Result<Option<u32>, ExecutionError>,
    pub ctx: Ctx,
}

impl<F, Ctx> VmSegmentState<F, Ctx> {
    pub fn new(
        instret: u64,
        pc: u32,
        memory: Option<GuestMemory>,
        streams: Streams<F>,
        rng: StdRng,
        ctx: Ctx,
    ) -> Self {
        Self {
            instret,
            pc,
            memory: if let Some(mem) = memory {
                mem
            } else {
                GuestMemory::new(Default::default())
            },
            streams,
            rng,
            ctx,
            exit_code: Ok(None),
        }
    }
    /// Runtime read operation for a block of memory
    #[inline(always)]
    pub fn vm_read<T: Copy + Debug, const BLOCK_SIZE: usize>(
        &mut self,
        addr_space: u32,
        ptr: u32,
    ) -> [T; BLOCK_SIZE]
    where
        Ctx: E1ExecutionCtx,
    {
        self.ctx
            .on_memory_operation(addr_space, ptr, BLOCK_SIZE as u32);
        self.host_read(addr_space, ptr)
    }

    /// Runtime write operation for a block of memory
    #[inline(always)]
    pub fn vm_write<T: Copy + Debug, const BLOCK_SIZE: usize>(
        &mut self,
        addr_space: u32,
        ptr: u32,
        data: &[T; BLOCK_SIZE],
    ) where
        Ctx: E1ExecutionCtx,
    {
        self.ctx
            .on_memory_operation(addr_space, ptr, BLOCK_SIZE as u32);
        self.host_write(addr_space, ptr, data)
    }

    #[inline(always)]
    pub fn vm_read_slice<T: Copy + Debug>(&mut self, addr_space: u32, ptr: u32, len: usize) -> &[T]
    where
        Ctx: E1ExecutionCtx,
    {
        self.ctx.on_memory_operation(addr_space, ptr, len as u32);
        self.host_read_slice(addr_space, ptr, len)
    }

    #[inline(always)]
    pub fn host_read<T: Copy + Debug, const BLOCK_SIZE: usize>(
        &self,
        addr_space: u32,
        ptr: u32,
    ) -> [T; BLOCK_SIZE]
    where
        Ctx: E1ExecutionCtx,
    {
        unsafe { self.memory.read(addr_space, ptr) }
    }
    #[inline(always)]
    pub fn host_write<T: Copy + Debug, const BLOCK_SIZE: usize>(
        &mut self,
        addr_space: u32,
        ptr: u32,
        data: &[T; BLOCK_SIZE],
    ) where
        Ctx: E1ExecutionCtx,
    {
        unsafe { self.memory.write(addr_space, ptr, *data) }
    }
    #[inline(always)]
    pub fn host_read_slice<T: Copy + Debug>(&self, addr_space: u32, ptr: u32, len: usize) -> &[T]
    where
        Ctx: E1ExecutionCtx,
    {
        unsafe { self.memory.get_slice(addr_space, ptr, len) }
    }
}

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
    #[cfg(feature = "bench-metrics")]
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
        #[cfg(feature = "bench-metrics")]
        let air_names = chip_complex.air_names();

        Self {
            chip_complex,
            ctrl,
            #[cfg(feature = "bench-metrics")]
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
    pub fn execute_from_state(
        &mut self,
        state: &mut VmSegmentState<F, Ctrl::Ctx>,
    ) -> Result<(), ExecutionError> {
        let mut prev_backtrace: Option<Backtrace> = None;

        // Call the pre-execution hook
        self.ctrl.on_start(state, &mut self.chip_complex);

        loop {
            if let Ok(Some(exit_code)) = state.exit_code {
                self.ctrl
                    .on_terminate(state, &mut self.chip_complex, exit_code);
                break;
            }
            if self.should_suspend(state) {
                self.ctrl.on_suspend(state, &mut self.chip_complex);
                break;
            }

            // Fetch, decode and execute single instruction
            self.execute_instruction(state, &mut prev_backtrace)?;
            state.instret += 1;
        }
        Ok(())
    }

    /// Executes a single instruction and updates VM state
    fn execute_instruction(
        &mut self,
        state: &mut VmSegmentState<F, Ctrl::Ctx>,
        prev_backtrace: &mut Option<Backtrace>,
    ) -> Result<(), ExecutionError> {
        let pc = state.pc;
        let timestamp = self.chip_complex.memory_controller().timestamp();

        // Process an instruction and update VM state
        let (instruction, debug_info) = self.chip_complex.base.program_chip.get_instruction(pc)?;

        tracing::trace!("pc: {pc:#x} | time: {timestamp} | {:?}", instruction);

        let &Instruction { opcode, c, .. } = instruction;

        // Handle termination instruction
        if opcode == SystemOpcode::TERMINATE.global_opcode() {
            state.exit_code = Ok(Some(c.as_canonical_u32()));
            return Ok(());
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

        Ok(())
    }

    /// Returns bool of whether to switch to next segment or not.
    fn should_suspend(&mut self, state: &mut VmSegmentState<F, Ctrl::Ctx>) -> bool {
        if !self.system_config().continuation_enabled {
            return false;
        }

        // Check with the execution control policy
        self.ctrl.should_suspend(state, &self.chip_complex)
    }

    /// Generate ProofInput to prove the segment. Should be called after ::execute
    #[instrument(name = "trace_gen", skip_all)]
    pub fn generate_proof_input<SC: StarkGenericConfig>(
        #[allow(unused_mut)] mut self,
        cached_program: Option<CommittedTraceData<SC>>,
    ) -> Result<ProofInput<SC>, GenerationError>
    where
        Domain<SC>: PolynomialSpace<Val = F>,
        VC::Executor: Chip<SC>,
        VC::Periphery: Chip<SC>,
    {
        self.chip_complex.generate_proof_input(
            cached_program,
            &self.trace_height_constraints,
            #[cfg(feature = "bench-metrics")]
            &mut self.metrics,
        )
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

/// Macro for executing with a compile-time span name for better tracing performance
#[macro_export]
macro_rules! execute_spanned {
    ($name:literal, $executor:expr, $state:expr) => {{
        #[cfg(feature = "bench-metrics")]
        let start = std::time::Instant::now();
        #[cfg(feature = "bench-metrics")]
        let start_instret = $state.instret;

        let result = tracing::info_span!($name).in_scope(|| $executor.execute_from_state($state));

        #[cfg(feature = "bench-metrics")]
        {
            let elapsed = start.elapsed();
            let insns = $state.instret - start_instret;
            metrics::counter!("insns").absolute(insns);
            metrics::gauge!(concat!($name, "_insn_mi/s"))
                .set(insns as f64 / elapsed.as_micros() as f64);
        }
        result
    }};
}
