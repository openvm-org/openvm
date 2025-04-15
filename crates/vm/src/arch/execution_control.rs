use openvm_instructions::instruction::Instruction;
use openvm_stark_backend::p3_field::PrimeField32;

use super::{segment_new::VmExecutionState, ExecutionError, TracegenCtx};
use crate::{
    arch::{ExecutionState, InstructionExecutor},
    system::memory::{online::GuestMemory, AddressMap, MemoryImage, PAGE_SIZE},
};

/// Check segment every 100 instructions.
const SEGMENT_CHECK_INTERVAL: usize = 100;

/// Trait for execution control, determining segmentation and stopping conditions
pub trait ExecutionControl {
    /// Guest memory type
    type Mem: GuestMemory;
    /// Host context
    type Ctx;

    /// Determines if execution should stop
    fn should_stop(&mut self, state: &VmExecutionState<Self::Mem, Self::Ctx>) -> bool;

    /// Called before segment execution begins
    fn on_segment_start(&mut self, vm_state: &VmExecutionState<Self::Mem, Self::Ctx>);

    /// Called after segment execution completes
    fn on_segment_end(&mut self, vm_state: &VmExecutionState<Self::Mem, Self::Ctx>);

    /// Execute a single instruction
    // TODO: change instruction to Instruction<u32> / PInstruction
    fn execute_instruction<F>(
        &mut self,
        vm_state: &mut VmExecutionState<Self::Mem, Self::Ctx>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError>
    where
        F: PrimeField32;
}

/// Implementation of the ExecutionControl trait using the old segmentation strategy
pub struct TracegenExecutionControl {
    pub final_memory: Option<MemoryImage>,
    pub since_last_segment_check: usize,
    air_names: Vec<String>,
}

impl ExecutionControl for TracegenExecutionControl {
    type Ctx = TracegenCtx;
    type Mem = AddressMap<PAGE_SIZE>;

    fn should_stop(&mut self, _state: &VmExecutionState<Self::Mem, Self::Ctx>) -> bool {
        // Avoid checking segment too often.
        if self.since_last_segment_check != SEGMENT_CHECK_INTERVAL {
            self.since_last_segment_check += 1;
            return false;
        }
        self.since_last_segment_check = 0;
        self.chip_complex
            .borrow()
            .config()
            .segmentation_strategy
            .should_segment(
                &self.air_names,
                &self
                    .chip_complex
                    .borrow_mut()
                    .dynamic_trace_heights()
                    .collect::<Vec<_>>(),
                &self.chip_complex.borrow().current_trace_cells(),
            )
    }

    fn on_segment_start(&mut self, vm_state: &VmExecutionState<Self::Mem, Self::Ctx>) {
        let timestamp = vm_state.ctx.timestamp;
        self.chip_complex
            .borrow_mut()
            .connector_chip_mut()
            .begin(ExecutionState::new(vm_state.pc, timestamp));
    }

    fn on_segment_end(&mut self, vm_state: &VmExecutionState<Self::Mem, Self::Ctx>) {
        let timestamp = vm_state.ctx.timestamp;
        // End the current segment with connector chip
        self.chip_complex
            .borrow_mut()
            .connector_chip_mut()
            .end(ExecutionState::new(vm_state.pc, timestamp), None);
        self.final_memory = Some(
            self.chip_complex
                .borrow()
                .base
                .memory_controller
                .memory_image()
                .clone(),
        );
    }

    /// Execute a single instruction
    fn execute_instruction<F>(
        &mut self,
        vm_state: &mut VmExecutionState<Self::Mem, Self::Ctx>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError>
    where
        F: PrimeField32,
    {
        let timestamp = vm_state.ctx.timestamp;
        let &Instruction { opcode, .. } = instruction;

        if let Some(executor) = self
            .chip_complex
            .borrow_mut()
            .inventory
            .get_mut_executor(&opcode)
        {
            let memory_controller = &mut self.chip_complex.borrow_mut().base.memory_controller;
            let new_state = executor.execute(
                memory_controller,
                instruction,
                ExecutionState::new(vm_state.pc, timestamp),
            )?;
            vm_state.pc = new_state.pc;
        } else {
            return Err(ExecutionError::DisabledOperation {
                pc: vm_state.pc,
                opcode,
            });
        };

        Ok(())
    }
}
