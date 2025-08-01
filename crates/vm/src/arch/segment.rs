use std::{
    fmt::Debug,
    ops::{Deref, DerefMut},
};

use openvm_stark_backend::p3_field::PrimeField32;

use super::{ExecutionError, VmState};
use crate::{
    arch::{
        execution_mode::{
            tracegen::{TracegenCtx, TracegenExecutionControl},
            E1ExecutionCtx,
        },
        instructions::*,
        Arena, InstructionExecutor,
    },
    system::{
        memory::online::{GuestMemory, TracingMemory},
        program::ProgramHandler,
    },
};

/// Represents the full execution state of a VM during segment execution.
pub struct VmSegmentState<F, MEM, CTX> {
    /// Core VM state
    pub vm_state: VmState<F, MEM>,
    /// Execution-specific fields
    pub exit_code: Result<Option<u32>, ExecutionError>,
    pub ctx: CTX,
}

impl<F, MEM, CTX> VmSegmentState<F, MEM, CTX> {
    pub fn new(vm_state: VmState<F, MEM>, ctx: CTX) -> Self {
        Self {
            vm_state,
            ctx,
            exit_code: Ok(None),
        }
    }
}

impl<F, MEM, CTX> Deref for VmSegmentState<F, MEM, CTX> {
    type Target = VmState<F, MEM>;

    fn deref(&self) -> &Self::Target {
        &self.vm_state
    }
}

impl<F, MEM, CTX> DerefMut for VmSegmentState<F, MEM, CTX> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.vm_state
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
        RA: Arena,
        E: InstructionExecutor<F, RA>,
    {
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
            self.execute_instruction(state)?;
            state.instret += 1;
        }
        Ok(())
    }

    /// Executes a single instruction and updates VM state
    #[inline(always)]
    fn execute_instruction<RA>(
        &mut self,
        state: &mut VmSegmentState<F, TracingMemory, TracegenCtx<RA>>,
    ) -> Result<(), ExecutionError>
    where
        RA: Arena,
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

        // Execute the instruction using the control implementation
        self.ctrl.execute_instruction(state, executor, pc_entry)?;

        #[cfg(feature = "metrics")]
        {
            crate::metrics::update_instruction_metrics(state, executor, pc_entry);
        }

        Ok(())
    }
}

/// Macro for executing with a compile-time span name for better tracing performance
#[macro_export]
macro_rules! execute_spanned {
    ($name:literal, $executor:expr, $state:expr) => {{
        #[cfg(feature = "metrics")]
        let start = std::time::Instant::now();
        #[cfg(feature = "metrics")]
        let start_instret = $state.instret;

        let result = tracing::info_span!($name).in_scope(|| $executor.execute_from_state($state));

        #[cfg(feature = "metrics")]
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
