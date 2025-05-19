use openvm_instructions::instruction::Instruction;
use openvm_stark_backend::p3_field::PrimeField32;

use super::{
    ExecutionError, ExecutionSegmentState, TracegenCtx, VmChipComplex, VmConfig, VmStateMut,
};
use crate::{
    arch::{ExecutionState, InsExecutorE1, InstructionExecutor},
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
    /// Host context
    type Ctx;

    fn new(chip_complex: &VmChipComplex<F, VC::Executor, VC::Periphery>) -> Self;

    /// Determines if execution should stop
    // TODO(ayush): rename to should_suspend
    fn should_stop(&mut self, chip_complex: &VmChipComplex<F, VC::Executor, VC::Periphery>)
        -> bool;

    /// Called before segment execution begins
    fn on_segment_start(
        &mut self,
        pc: u32,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    );

    // TODO(ayush): maybe combine with on_terminate
    /// Called after segment execution completes
    fn on_segment_end(
        &mut self,
        pc: u32,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    );

    /// Called after program termination
    fn on_terminate(
        &mut self,
        pc: u32,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
        exit_code: u32,
    );

    /// Execute a single instruction
    // TODO(ayush): change instruction to Instruction<u32> / PInstruction
    fn execute_instruction(
        &mut self,
        vm_state: &mut ExecutionSegmentState,
        instruction: &Instruction<F>,
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

    fn new(chip_complex: &VmChipComplex<F, VC::Executor, VC::Periphery>) -> Self {
        Self {
            final_memory: None,
            since_last_segment_check: 0,
            air_names: chip_complex.air_names(),
        }
    }

    fn should_stop(
        &mut self,
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
        pc: u32,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) {
        let timestamp = chip_complex.memory_controller().timestamp();
        chip_complex
            .connector_chip_mut()
            .begin(ExecutionState::new(pc, timestamp));
    }

    fn on_segment_end(
        &mut self,
        pc: u32,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) {
        let timestamp = chip_complex.memory_controller().timestamp();
        // End the current segment with connector chip
        chip_complex
            .connector_chip_mut()
            .end(ExecutionState::new(pc, timestamp), None);
        self.final_memory = Some(chip_complex.base.memory_controller.memory_image().clone());
    }

    fn on_terminate(
        &mut self,
        pc: u32,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
        exit_code: u32,
    ) {
        let timestamp = chip_complex.memory_controller().timestamp();
        chip_complex
            .connector_chip_mut()
            .end(ExecutionState::new(pc, timestamp), Some(exit_code));
        self.final_memory = Some(chip_complex.base.memory_controller.memory_image().clone());
    }

    /// Execute a single instruction
    fn execute_instruction(
        &mut self,
        state: &mut ExecutionSegmentState,
        instruction: &Instruction<F>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) -> Result<(), ExecutionError>
    where
        F: PrimeField32,
    {
        let timestamp = chip_complex.memory_controller().timestamp();

        let &Instruction { opcode, .. } = instruction;

        if let Some(executor) = chip_complex.inventory.get_mut_executor(&opcode) {
            let memory_controller = &mut chip_complex.base.memory_controller;
            let new_state = executor.execute(
                memory_controller,
                instruction,
                ExecutionState::new(state.pc, timestamp),
            )?;
            state.pc = new_state.pc;
        } else {
            return Err(ExecutionError::DisabledOperation {
                pc: state.pc,
                opcode,
            });
        };

        Ok(())
    }
}

/// Implementation of the ExecutionControl trait using the old segmentation strategy
pub struct E1ExecutionControl {
    pub final_memory: Option<MemoryImage>,
}

impl<F, VC> ExecutionControl<F, VC> for E1ExecutionControl
where
    F: PrimeField32,
    VC: VmConfig<F>,
    VC::Executor: InsExecutorE1<F>,
{
    type Ctx = ();

    fn new(_chip_complex: &VmChipComplex<F, VC::Executor, VC::Periphery>) -> Self {
        Self { final_memory: None }
    }

    fn should_stop(
        &mut self,
        _chip_complex: &VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) -> bool {
        false
    }

    fn on_segment_start(
        &mut self,
        _pc: u32,
        _chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) {
    }

    fn on_segment_end(
        &mut self,
        _pc: u32,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) {
        self.final_memory = Some(chip_complex.base.memory_controller.memory_image().clone());
    }

    fn on_terminate(
        &mut self,
        _pc: u32,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
        _exit_code: u32,
    ) {
        self.final_memory = Some(chip_complex.base.memory_controller.memory_image().clone());
    }

    /// Execute a single instruction
    fn execute_instruction(
        &mut self,
        state: &mut ExecutionSegmentState,
        instruction: &Instruction<F>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) -> Result<(), ExecutionError>
    where
        F: PrimeField32,
    {
        let &Instruction { opcode, .. } = instruction;

        if let Some(executor) = chip_complex.inventory.get_mut_executor(&opcode) {
            let vm_state = VmStateMut {
                pc: &mut state.pc,
                memory: state.memory.as_mut().unwrap(),
                ctx: &mut (),
            };
            executor.execute_e1(vm_state, instruction)?;
        } else {
            return Err(ExecutionError::DisabledOperation {
                pc: state.pc,
                opcode,
            });
        };

        Ok(())
    }
}
