use openvm_instructions::instruction::Instruction;
use openvm_stark_backend::p3_field::PrimeField32;

use super::{segment::VmExecutionState, ExecutionError, TracegenCtx, VmChipComplex, VmConfig};
use crate::{
    arch::{ExecutionState, InstructionExecutor},
    system::memory::{online::GuestMemory, AddressMap, MemoryImage, PAGE_SIZE},
};

/// Check segment every 100 instructions.
const SEGMENT_CHECK_INTERVAL: usize = 100;

/// Trait for execution control, determining segmentation and stopping conditions
pub trait ExecutionControl<F, VC>
where
    F: PrimeField32,
    VC: VmConfig<F>,
{
    /// Guest memory type
    type Mem: GuestMemory;
    /// Host context
    type Ctx;

    fn new(chip_complex: &VmChipComplex<F, VC::Executor, VC::Periphery>) -> Self;

    /// Determines if execution should stop
    fn should_stop(
        &mut self,
        state: &VmExecutionState<Self::Mem, Self::Ctx>,
        chip_complex: &VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) -> bool;

    /// Called before segment execution begins
    fn on_segment_start(
        &mut self,
        vm_state: &VmExecutionState<Self::Mem, Self::Ctx>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    );

    /// Called after segment execution completes
    fn on_segment_end(
        &mut self,
        vm_state: &VmExecutionState<Self::Mem, Self::Ctx>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    );

    /// Execute a single instruction
    // TODO(ayush): change instruction to Instruction<u32> / PInstruction
    fn execute_instruction(
        &mut self,
        vm_state: &mut VmExecutionState<Self::Mem, Self::Ctx>,
        // instruction: &Instruction<F>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
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

impl TracegenExecutionControl {
    pub fn new(air_names: Vec<String>) -> Self {
        Self {
            final_memory: None,
            since_last_segment_check: 0,
            air_names,
        }
    }
}

impl<F, VC> ExecutionControl<F, VC> for TracegenExecutionControl
where
    F: PrimeField32,
    VC: VmConfig<F>,
{
    type Ctx = TracegenCtx;
    type Mem = AddressMap<PAGE_SIZE>;

    fn new(chip_complex: &VmChipComplex<F, VC::Executor, VC::Periphery>) -> Self {
        Self {
            final_memory: None,
            since_last_segment_check: 0,
            air_names: chip_complex.air_names(),
        }
    }

    fn should_stop(
        &mut self,
        _state: &VmExecutionState<Self::Mem, Self::Ctx>,
        chip_complex: &VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) -> bool {
        // Avoid checking segment too often.
        if self.since_last_segment_check != SEGMENT_CHECK_INTERVAL {
            self.since_last_segment_check += 1;
            return false;
        }
        self.since_last_segment_check = 0;
        chip_complex.config().segmentation_strategy.should_segment(
            &self.air_names,
            &chip_complex.dynamic_trace_heights().collect::<Vec<_>>(),
            &chip_complex.current_trace_cells(),
        )
    }

    fn on_segment_start(
        &mut self,
        vm_state: &VmExecutionState<Self::Mem, Self::Ctx>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) {
        let timestamp = chip_complex.memory_controller().timestamp();
        chip_complex
            .connector_chip_mut()
            .begin(ExecutionState::new(vm_state.pc, timestamp));
    }

    fn on_segment_end(
        &mut self,
        vm_state: &VmExecutionState<Self::Mem, Self::Ctx>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) {
        let timestamp = chip_complex.memory_controller().timestamp();
        // End the current segment with connector chip
        chip_complex
            .connector_chip_mut()
            .end(ExecutionState::new(vm_state.pc, timestamp), None);
        self.final_memory = Some(chip_complex.base.memory_controller.memory_image().clone());
    }

    /// Execute a single instruction
    fn execute_instruction(
        &mut self,
        vm_state: &mut VmExecutionState<Self::Mem, Self::Ctx>,
        // instruction: &Instruction<F>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) -> Result<(), ExecutionError>
    where
        F: PrimeField32,
    {
        let timestamp = chip_complex.memory_controller().timestamp();
        let (instruction, _) = chip_complex
            .base
            .program_chip
            .get_instruction(vm_state.pc)?;

        let &Instruction { opcode, .. } = instruction;

        if let Some(executor) = chip_complex.inventory.get_mut_executor(&opcode) {
            let memory_controller = &mut chip_complex.base.memory_controller;
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
