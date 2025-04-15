use std::marker::PhantomData;

use openvm_instructions::instruction::Instruction;
use openvm_stark_backend::p3_field::PrimeField32;

use super::{
    segment_new::VmExecutionState,
    segmentation_strategy::{DefaultSegmentationStrategy, SegmentationStrategy},
    ExecutionError, VmChipComplex, VmConfig,
};
use crate::{
    arch::{ExecutionState, InstructionExecutor},
    system::memory::{online::GuestMemory, MemoryImage},
};

/// Check segment every 100 instructions.
const SEGMENT_CHECK_INTERVAL: usize = 100;

/// Trait for execution control, determining segmentation and stopping conditions
pub trait ExecutionControl<Mem, Ctx, F>
where
    Mem: GuestMemory,
    F: PrimeField32,
{
    /// Determines if execution should stop
    fn should_stop(&mut self, state: &VmExecutionState<Mem, Ctx>) -> bool;

    /// Called before segment execution begins
    fn on_segment_start(&mut self, vm_state: &VmExecutionState<Mem, Ctx>);

    /// Called after segment execution completes
    fn on_segment_end(&mut self, vm_state: &VmExecutionState<Mem, Ctx>);

    /// Execute a single instruction
    // TODO: change instruction to Instruction<u32> / PInstruction
    fn execute_instruction(
        &mut self,
        vm_state: &mut VmExecutionState<Mem, Ctx>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError>;
}

/// Implementation of the ExecutionControl trait using the old segmentation strategy
pub struct TracegenExecutionControl<F, VC, Mem, Ctx>
where
    F: PrimeField32,
    VC: VmConfig<F>,
    Mem: GuestMemory,
{
    pub chip_complex: VmChipComplex<F, VC::Executor, VC::Periphery>,
    pub air_names: Vec<String>,
    pub segmentation_strategy: DefaultSegmentationStrategy,
    pub final_memory: Option<MemoryImage>,
    pub since_last_segment_check: usize,
    phantom: PhantomData<(Mem, Ctx)>,
}

impl<F, VC, Mem, Ctx> ExecutionControl<Mem, Ctx, F> for TracegenExecutionControl<F, VC, Mem, Ctx>
where
    F: PrimeField32,
    VC: VmConfig<F>,
    Mem: GuestMemory,
{
    fn should_stop(&mut self, _state: &VmExecutionState<Mem, Ctx>) -> bool {
        // Avoid checking segment too often.
        if self.since_last_segment_check != SEGMENT_CHECK_INTERVAL {
            self.since_last_segment_check += 1;
            return false;
        }
        self.since_last_segment_check = 0;
        let segmentation_strategy = &self.segmentation_strategy;
        segmentation_strategy.should_segment(
            &self.air_names,
            &self
                .chip_complex
                .dynamic_trace_heights()
                .collect::<Vec<_>>(),
            &self.chip_complex.current_trace_cells(),
        )
    }

    fn on_segment_start(&mut self, vm_state: &VmExecutionState<Mem, Ctx>) {
        self.chip_complex
            .connector_chip_mut()
            .begin(ExecutionState::new(vm_state.pc, vm_state.timestamp));
    }

    fn on_segment_end(&mut self, vm_state: &VmExecutionState<Mem, Ctx>) {
        // End the current segment with connector chip
        self.chip_complex
            .connector_chip_mut()
            .end(ExecutionState::new(vm_state.pc, vm_state.timestamp), None);
        self.final_memory = Some(
            self.chip_complex
                .base
                .memory_controller
                .memory_image()
                .clone(),
        );
    }

    /// Execute a single instruction
    fn execute_instruction(
        &mut self,
        vm_state: &mut VmExecutionState<Mem, Ctx>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let &Instruction { opcode, .. } = instruction;

        if let Some(executor) = self.chip_complex.inventory.get_mut_executor(&opcode) {
            let memory_controller = &mut self.chip_complex.base.memory_controller;
            let new_state = executor.execute(
                memory_controller,
                instruction,
                ExecutionState::new(vm_state.pc, vm_state.timestamp),
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
