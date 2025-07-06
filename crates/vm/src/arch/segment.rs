use backtrace::Backtrace;
#[cfg(feature = "bench-metrics")]
use openvm_instructions::exe::FnBounds;
use openvm_stark_backend::p3_field::PrimeField32;
use rand::rngs::StdRng;

use super::{execution_control::ExecutionControl, ExecutionError, Streams};
#[cfg(feature = "bench-metrics")]
use crate::metrics::VmMetrics;
use crate::{
    arch::{instructions::*, VmStateMut},
    system::program::ProgramHandler,
};

pub struct VmSegmentState<F, MEM, CTX> {
    pub instret: u64,
    pub pc: u32,
    pub memory: MEM,
    pub streams: Streams<F>,
    pub rng: StdRng,
    pub exit_code: Option<u32>,
    pub ctx: CTX,
}

impl<F, MEM, CTX> VmSegmentState<F, MEM, CTX> {
    pub fn new(
        instret: u64,
        pc: u32,
        memory: MEM,
        streams: Streams<F>,
        rng: StdRng,
        ctx: CTX,
    ) -> Self {
        Self {
            instret,
            pc,
            memory,
            streams,
            rng,
            exit_code: None,
            ctx,
        }
    }

    pub fn state_mut(&mut self) -> VmStateMut<F, MEM, CTX> {
        VmStateMut {
            pc: &mut self.pc,
            memory: &mut self.memory,
            streams: &mut self.streams,
            rng: &mut self.rng,
            ctx: &mut self.ctx,
        }
    }
}

// TODO[jpw]: rename. this will essentially be just interpreted instance for preflight(E3)
pub struct VmSegmentExecutor<F, E, Ctrl> {
    handler: ProgramHandler<F, E>,
    /// Execution control for determining segmentation and stopping conditions
    pub ctrl: Ctrl,

    // Air names for debug purposes only.
    // #[cfg(feature = "bench-metrics")]
    // pub(crate) air_names: Vec<String>,
    /// Metrics collected for this execution segment alone.
    #[cfg(feature = "bench-metrics")]
    pub metrics: VmMetrics,
}

impl<F, E, Ctrl> VmSegmentExecutor<F, E, Ctrl>
where
    F: PrimeField32,
    Ctrl: ExecutionControl<F, E>,
{
    /// Creates a new execution segment from a program and initial state, using parent VM config
    pub fn new(handler: ProgramHandler<F, E>, ctrl: Ctrl) -> Self {
        Self {
            handler,
            ctrl,
            // #[cfg(feature = "bench-metrics")]
            // air_names,
            #[cfg(feature = "bench-metrics")]
            metrics: VmMetrics::default(),
        }
    }

    #[cfg(feature = "bench-metrics")]
    pub fn set_fn_bounds(&mut self, fn_bounds: FnBounds) {
        self.metrics.fn_bounds = fn_bounds;
    }

    /// Stopping is triggered by should_stop() or if VM is terminated
    pub fn execute_from_state(
        &mut self,
        state: &mut VmSegmentState<F, Ctrl::Memory, Ctrl::Ctx>,
    ) -> Result<(), ExecutionError> {
        let mut prev_backtrace: Option<Backtrace> = None;

        loop {
            if let Some(exit_code) = state.exit_code {
                self.ctrl.on_terminate(state, exit_code);
                break;
            }
            if self.ctrl.should_suspend(state) {
                self.ctrl.on_suspend(state);
                break;
            }

            // Fetch, decode and execute single instruction
            self.execute_instruction(state, &mut prev_backtrace)?;
            state.instret += 1;
        }
        Ok(())
    }

    /// Executes a single instruction and updates VM state
    // TODO(ayush): clean this up, separate to smaller functions
    #[inline(always)]
    fn execute_instruction(
        &mut self,
        state: &mut VmSegmentState<F, Ctrl::Memory, Ctrl::Ctx>,
        prev_backtrace: &mut Option<Backtrace>,
    ) -> Result<(), ExecutionError> {
        let pc = state.pc;
        let (executor, pc_entry) = self.handler.get_executor(pc)?;
        tracing::trace!("pc: {pc:#x} | {:?}", pc_entry.insn);

        let opcode = pc_entry.insn.opcode;
        let c = pc_entry.insn.c;
        // Handle termination instruction
        if opcode.as_usize() == SystemOpcode::CLASS_OFFSET + SystemOpcode::TERMINATE as usize {
            state.exit_code = Some(c.as_canonical_u32());
            return Ok(());
        }

        // // Extract debug info components
        // #[allow(unused_variables)]
        // let (dsl_instr, trace) = debug_info.as_ref().map_or(
        //     (None, None),
        //     |DebugInfo {
        //          dsl_instruction,
        //          trace,
        //      }| (Some(dsl_instruction.clone()), trace.as_ref()),
        // );

        // Handle phantom instructions
        // TODO[jpw]: this is an extra handler for phantom instructions that should only be enabled
        // when cfg(debug_assertions) or fn-bound feature is on
        // if opcode == SystemOpcode::PHANTOM.global_opcode() {
        //     let discriminant = c.as_canonical_u32() as u16;
        //     if let Some(phantom) = SysPhantom::from_repr(discriminant) {
        //         tracing::trace!("pc: {pc:#x} | system phantom: {phantom:?}");

        //         if phantom == SysPhantom::DebugPanic {
        //             if let Some(mut backtrace) = prev_backtrace.take() {
        //                 backtrace.resolve();
        //                 eprintln!("openvm program failure; backtrace:\n{:?}", backtrace);
        //             } else {
        //                 eprintln!("openvm program failure; no backtrace");
        //             }
        //             return Err(ExecutionError::Fail { pc });
        //         }

        //         #[cfg(feature = "bench-metrics")]
        //         {
        //             let dsl_str = dsl_instr.clone().unwrap_or_else(|| "Default".to_string());
        //             match phantom {
        //                 SysPhantom::CtStart => self.metrics.cycle_tracker.start(dsl_str),
        //                 SysPhantom::CtEnd => self.metrics.cycle_tracker.end(dsl_str),
        //                 _ => {}
        //             }
        //         }
        //     }
        // }

        // // TODO(ayush): move to vm state?
        // // TODO(jpw): move metrics to state as well?
        // *prev_backtrace = trace.cloned();

        // Execute the instruction using the control implementation
        self.ctrl.execute_instruction(state, executor, pc_entry)?;

        // Update metrics if enabled
        // #[cfg(feature = "bench-metrics")]
        // {
        //     self.update_instruction_metrics(pc, opcode, dsl_instr);
        // }

        Ok(())
    }

    // TODO[jpw]: figure out metrics later
    // #[cfg(feature = "bench-metrics")]
    // #[allow(unused_variables)]
    // pub fn update_instruction_metrics(
    //     &mut self,
    //     pc: u32,
    //     opcode: VmOpcode,
    //     dsl_instr: Option<String>,
    // ) {
    //     self.metrics.cycle_count += 1;

    //     if self.system_config().profiling {
    //         use crate::arch::InstructionExecutor;

    //         let executor = self.chip_complex.inventory.get_executor(opcode).unwrap();
    //         let opcode_name = executor.get_opcode_name(opcode.as_usize());
    //         self.metrics.update_trace_cells(
    //             &self.air_names,
    //             self.chip_complex.current_trace_cells(),
    //             opcode_name,
    //             dsl_instr,
    //         );

    //         #[cfg(feature = "function-span")]
    //         self.metrics.update_current_fn(pc);
    //     }
    // }
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
