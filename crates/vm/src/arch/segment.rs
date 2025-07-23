use std::fmt::Debug;

use backtrace::Backtrace;
use openvm_stark_backend::p3_field::PrimeField32;
use rand::rngs::StdRng;

use super::{ExecutionError, Streams};
use crate::{
    arch::{
        execution_mode::{
            tracegen::{TracegenCtx, TracegenExecutionControl},
            E1ExecutionCtx,
        },
        instructions::*,
        InstructionExecutor, VmStateMut,
    },
    system::{
        memory::online::{GuestMemory, TracingMemory},
        program::ProgramHandler,
    },
};

pub struct VmSegmentState<F, MEM, CTX> {
    pub instret: u64,
    pub pc: u32,
    pub memory: MEM,
    pub streams: Streams<F>,
    pub rng: StdRng,
    pub exit_code: Result<Option<u32>, ExecutionError>,
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
            ctx,
            exit_code: Ok(None),
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

impl<F, CTX> VmSegmentState<F, GuestMemory, CTX>
where
    CTX: E1ExecutionCtx,
{
    /// Runtime read operation for a block of memory
    #[inline(always)]
    pub fn vm_read<T: Copy + Debug, const BLOCK_SIZE: usize>(
        &mut self,
        addr_space: u32,
        ptr: u32,
    ) -> [T; BLOCK_SIZE] {
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
    ) {
        self.ctx
            .on_memory_operation(addr_space, ptr, BLOCK_SIZE as u32);
        self.host_write(addr_space, ptr, data)
    }

    #[inline(always)]
    pub fn vm_read_slice<T: Copy + Debug>(
        &mut self,
        addr_space: u32,
        ptr: u32,
        len: usize,
    ) -> &[T] {
        self.ctx.on_memory_operation(addr_space, ptr, len as u32);
        self.host_read_slice(addr_space, ptr, len)
    }

    #[inline(always)]
    pub fn host_read<T: Copy + Debug, const BLOCK_SIZE: usize>(
        &self,
        addr_space: u32,
        ptr: u32,
    ) -> [T; BLOCK_SIZE] {
        unsafe { self.memory.read(addr_space, ptr) }
    }

    #[inline(always)]
    pub fn host_write<T: Copy + Debug, const BLOCK_SIZE: usize>(
        &mut self,
        addr_space: u32,
        ptr: u32,
        data: &[T; BLOCK_SIZE],
    ) {
        unsafe { self.memory.write(addr_space, ptr, *data) }
    }

    #[inline(always)]
    pub fn host_read_slice<T: Copy + Debug>(&self, addr_space: u32, ptr: u32, len: usize) -> &[T] {
        unsafe { self.memory.get_slice(addr_space, ptr, len) }
    }
}

// TODO[jpw]: rename. this will essentially be just interpreted instance for preflight(E3)
pub struct VmSegmentExecutor<F, E> {
    pub handler: ProgramHandler<F, E>,
    /// Execution control for determining segmentation and stopping conditions
    pub ctrl: TracegenExecutionControl<F>,
}

impl<F, E> VmSegmentExecutor<F, E>
where
    F: PrimeField32,
{
    /// Creates a new execution segment from a program and initial state, using parent VM config
    pub fn new(handler: ProgramHandler<F, E>, ctrl: TracegenExecutionControl<F>) -> Self {
        Self { handler, ctrl }
    }

    /// Stopping is triggered by should_stop() or if VM is terminated
    pub fn execute_from_state<RA>(
        &mut self,
        state: &mut VmSegmentState<F, TracingMemory, TracegenCtx<RA>>,
    ) -> Result<(), ExecutionError>
    where
        E: InstructionExecutor<F, RA>,
    {
        let mut prev_backtrace: Option<Backtrace> = None;

        loop {
            if let Ok(Some(exit_code)) = state.exit_code {
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
    #[inline(always)]
    fn execute_instruction<RA>(
        &mut self,
        state: &mut VmSegmentState<F, TracingMemory, TracegenCtx<RA>>,
        _prev_backtrace: &mut Option<Backtrace>,
    ) -> Result<(), ExecutionError>
    where
        E: InstructionExecutor<F, RA>,
    {
        let pc = state.pc;
        let (executor, pc_entry) = self.handler.get_executor(pc)?;
        tracing::trace!("pc: {pc:#x} | {:?}", pc_entry.insn);

        let opcode = pc_entry.insn.opcode;
        let c = pc_entry.insn.c;
        // Handle termination instruction
        if opcode.as_usize() == SystemOpcode::CLASS_OFFSET + SystemOpcode::TERMINATE as usize {
            state.exit_code = Ok(Some(c.as_canonical_u32()));
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
