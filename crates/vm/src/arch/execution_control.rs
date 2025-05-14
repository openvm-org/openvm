use openvm_instructions::instruction::Instruction;
use openvm_stark_backend::p3_field::PrimeField32;

use super::{
    E1Ctx, ExecutionError, ExecutionSegmentState, MeteredCtx, TracegenCtx, VmChipComplex, VmConfig,
    VmStateMut,
};
use crate::{
    arch::{ExecutionState, InsExecutorE1, InstructionExecutor},
    system::memory::{online::GuestMemory, AddressMap, MemoryImage, PAGE_SIZE},
};

// Metered execution thresholds
// TODO(ayush): fix these values
const MAX_TRACE_HEIGHT: usize = (1 << 22) - 100;
const MAX_TRACE_CELLS: usize = MAX_TRACE_HEIGHT * 120;
const MAX_INTERACTIONS: usize = MAX_TRACE_HEIGHT * 120;

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

    /// Determines if execution should suspend
    fn should_suspend(
        &mut self,
        state: &ExecutionSegmentState<Self::Ctx>,
        chip_complex: &VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) -> bool;

    /// Called before segment execution begins
    fn on_segment_start(
        &mut self,
        state: &ExecutionSegmentState<Self::Ctx>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    );

    // TODO(ayush): maybe combine with on_terminate
    /// Called after segment execution completes
    fn on_segment_end(
        &mut self,
        state: &ExecutionSegmentState<Self::Ctx>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    );

    /// Called after program termination
    fn on_terminate(
        &mut self,
        state: &ExecutionSegmentState<Self::Ctx>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
        exit_code: u32,
    );

    /// Execute a single instruction
    // TODO(ayush): change instruction to Instruction<u32> / PInstruction
    fn execute_instruction<Mem: GuestMemory>(
        &mut self,
        state: &mut ExecutionSegmentState<Mem, Self::Ctx>,
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
    type Mem = AddressMap<PAGE_SIZE>;

    fn should_suspend(
        &mut self,
        _state: &ExecutionSegmentState<Self::Ctx>,
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
        state: &ExecutionSegmentState<Self::Ctx>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) {
        let timestamp = chip_complex.memory_controller().timestamp();
        chip_complex
            .connector_chip_mut()
            .begin(ExecutionState::new(state.pc, timestamp));
    }

    fn on_segment_end(
        &mut self,
        state: &ExecutionSegmentState<Self::Ctx>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) {
        let timestamp = chip_complex.memory_controller().timestamp();
        // End the current segment with connector chip
        chip_complex
            .connector_chip_mut()
            .end(ExecutionState::new(state.pc, timestamp), None);
        self.final_memory = Some(chip_complex.base.memory_controller.memory_image().clone());
    }

    fn on_terminate(
        &mut self,
        state: &ExecutionSegmentState<Self::Ctx>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
        exit_code: u32,
    ) {
        for (name, height) in self
            .air_names
            .iter()
            .zip(chip_complex.current_trace_heights())
        {
            println!("{:<10} \t|\t{}", height, name);
        }
        // dbg!(chip_complex.current_trace_cells());

        let timestamp = chip_complex.memory_controller().timestamp();
        chip_complex
            .connector_chip_mut()
            .end(ExecutionState::new(state.pc, timestamp), Some(exit_code));
        self.final_memory = Some(chip_complex.base.memory_controller.memory_image().clone());
    }

    /// Execute a single instruction
    fn execute_instruction<Mem: GuestMemory>(
        &mut self,
        state: &mut ExecutionSegmentState<Mem>,
        instruction: &Instruction<F>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) -> Result<(), ExecutionError>
    where
        F: PrimeField32,
        Self::Ctx: Default,
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
#[derive(Default)]
pub struct E1ExecutionControl {
    pub final_memory: Option<MemoryImage>,
}

impl<F, VC> ExecutionControl<F, VC> for E1ExecutionControl
where
    F: PrimeField32,
    VC: VmConfig<F>,
    VC::Executor: InsExecutorE1<F>,
{
    type Ctx = E1Ctx;
    type Mem = AddressMap<PAGE_SIZE>;

    fn should_suspend(
        &mut self,
        _state: &ExecutionSegmentState<Self::Ctx>,
        _chip_complex: &VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) -> bool {
        false
    }

    fn on_segment_start(
        &mut self,
        _state: &ExecutionSegmentState<Self::Ctx>,
        _chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) {
    }

    fn on_segment_end(
        &mut self,
        _state: &ExecutionSegmentState<Self::Ctx>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) {
        self.final_memory = Some(chip_complex.base.memory_controller.memory_image().clone());
    }

    fn on_terminate(
        &mut self,
        _state: &ExecutionSegmentState<Self::Ctx>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
        _exit_code: u32,
    ) {
        self.final_memory = Some(chip_complex.base.memory_controller.memory_image().clone());
    }

    /// Execute a single instruction
    fn execute_instruction<Mem: GuestMemory>(
        &mut self,
        state: &mut ExecutionSegmentState<Mem>,
        instruction: &Instruction<F>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) -> Result<(), ExecutionError>
    where
        F: PrimeField32,
        Self::Ctx: Default,
    {
        let &Instruction { opcode, .. } = instruction;

        if let Some(executor) = chip_complex.inventory.get_mut_executor(&opcode) {
            let vm_state = VmStateMut {
                pc: &mut state.pc,
                memory: state.memory.as_mut().unwrap(),
                ctx: &mut state.ctx,
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

pub struct MeteredExecutionControl<'a> {
    pub num_interactions: &'a [usize],
    pub since_last_segment_check: usize,
    pub final_memory: Option<MemoryImage>,
    // TODO(ayush): remove
    air_names: Vec<String>,
}

impl<'a> MeteredExecutionControl<'a> {
    pub fn new(num_interactions: &'a [usize], air_names: Vec<String>) -> Self {
        Self {
            num_interactions,
            since_last_segment_check: 0,
            final_memory: None,
            air_names,
        }
    }
}

impl<F, VC> ExecutionControl<F, VC> for MeteredExecutionControl<'_>
where
    F: PrimeField32,
    VC: VmConfig<F>,
    VC::Executor: InsExecutorE1<F>,
{
    type Ctx = MeteredCtx;
    type Mem = AddressMap<PAGE_SIZE>;

    fn should_suspend(
        &mut self,
        state: &ExecutionSegmentState<Self::Ctx>,
        _chip_complex: &VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) -> bool {
        // Avoid checking segment too often.
        if self.since_last_segment_check != SEGMENT_CHECK_INTERVAL {
            self.since_last_segment_check += 1;
            return false;
        }
        self.since_last_segment_check = 0;

        if state.ctx.total_trace_cells > MAX_TRACE_CELLS {
            return true;
        }
        if state.ctx.total_interactions > MAX_INTERACTIONS {
            return true;
        }
        for &height in state.ctx.trace_heights.iter() {
            if height > MAX_TRACE_HEIGHT {
                return true;
            }
        }

        false
    }

    fn on_segment_start(
        &mut self,
        _state: &ExecutionSegmentState<Self::Ctx>,
        _chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) {
    }

    fn on_segment_end(
        &mut self,
        state: &ExecutionSegmentState<Self::Ctx>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) {
        self.final_memory = Some(chip_complex.base.memory_controller.memory_image().clone());
    }

    fn on_terminate(
        &mut self,
        state: &ExecutionSegmentState<Self::Ctx>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
        _exit_code: u32,
    ) {
        for (name, height) in self.air_names.iter().zip(state.ctx.trace_heights.iter()) {
            println!("{:<10} \t|\t{}", height, name);
        }
        self.final_memory = Some(chip_complex.base.memory_controller.memory_image().clone());
    }

    /// Execute a single instruction
    fn execute_instruction(
        &mut self,
        state: &mut ExecutionSegmentState<Self::Ctx>,
        // instruction: &Instruction<F>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) -> Result<(), ExecutionError>
    where
        F: PrimeField32,
        Self::Ctx: Default,
    {
        let (instruction, _) = chip_complex
            .base
            .program_chip
            .get_instruction(state.pc)?
            .clone();

        let Instruction { opcode, .. } = instruction;

        // Program | Connector | Public Values | Memory ... | Executors (except Public Values) | Range Checker
        // TODO(ayush): no magic number, cache
        let mut offset = 2 + chip_complex.memory_controller().num_airs();
        if chip_complex.config().has_public_values_chip() {
            offset += 1;
        }

        if let Some((executor, i)) = chip_complex.inventory.get_mut_executor_with_index(&opcode) {
            let memory_controller = &mut chip_complex.base.memory_controller;
            let vm_state = VmStateMut {
                pc: &mut state.pc,
                memory: &mut memory_controller.memory.data,
                ctx: &mut state.ctx,
            };
            let index = offset + i;
            executor.execute_e2(vm_state, &instruction, index, self.num_interactions[index])?;
        } else {
            return Err(ExecutionError::DisabledOperation {
                pc: state.pc,
                opcode,
            });
        };

        Ok(())
    }
}
