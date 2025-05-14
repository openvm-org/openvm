use std::collections::BTreeSet;

use openvm_instructions::instruction::Instruction;
use openvm_stark_backend::{
    p3_field::PrimeField32, p3_matrix::Matrix, p3_util::log2_strict_usize, ChipUsageGetter,
};

use super::{
    ChipId, E1Ctx, ExecutionError, ExecutionSegmentState, MeteredCtx, TracegenCtx, VmChipComplex,
    VmConfig, VmStateMut,
};
use crate::{
    arch::{ExecutionState, InsExecutorE1, InstructionExecutor},
    system::memory::{
        adapter::GenericAccessAdapterChip, interface::MemoryInterface, MemoryImage, CHUNK,
    },
};

// Metered execution thresholds
// TODO(ayush): fix these values
const MAX_TRACE_HEIGHT: usize = usize::MAX - 1;
const MAX_TRACE_CELLS: usize = usize::MAX - 1;
const MAX_INTERACTIONS: usize = usize::MAX - 1;

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

    /// Determines if execution should suspend
    fn should_suspend(
        &mut self,
        state: &ExecutionSegmentState<Self::Ctx>,
        chip_complex: &VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) -> bool;

    /// Called before segment execution begins
    fn on_segment_start(
        &mut self,
        state: &mut ExecutionSegmentState<Self::Ctx>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    );

    // TODO(ayush): maybe combine with on_terminate
    /// Called after segment execution completes
    fn on_segment_end(
        &mut self,
        state: &mut ExecutionSegmentState<Self::Ctx>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    );

    /// Called after program termination
    fn on_terminate(
        &mut self,
        state: &mut ExecutionSegmentState<Self::Ctx>,
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
        state: &mut ExecutionSegmentState<Self::Ctx>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) {
        let timestamp = chip_complex.memory_controller().timestamp();
        chip_complex
            .connector_chip_mut()
            .begin(ExecutionState::new(state.pc, timestamp));
    }

    fn on_segment_end(
        &mut self,
        state: &mut ExecutionSegmentState<Self::Ctx>,
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
        state: &mut ExecutionSegmentState<Self::Ctx>,
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

        // TODO(ayush): remove
        chip_complex.finalize_memory();

        let timestamp = chip_complex.memory_controller().timestamp();
        chip_complex
            .connector_chip_mut()
            .end(ExecutionState::new(state.pc, timestamp), Some(exit_code));
        self.final_memory = Some(chip_complex.base.memory_controller.memory_image().clone());
    }

    /// Execute a single instruction
    fn execute_instruction<Mem: GuestMemory>(
        &mut self,
        state: &mut ExecutionSegmentState<Mem, Self::Ctx>,
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

    fn should_suspend(
        &mut self,
        _state: &ExecutionSegmentState<Self::Ctx>,
        _chip_complex: &VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) -> bool {
        false
    }

    fn on_segment_start(
        &mut self,
        _state: &mut ExecutionSegmentState<Self::Ctx>,
        _chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) {
    }

    fn on_segment_end(
        &mut self,
        _state: &mut ExecutionSegmentState<Self::Ctx>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) {
        self.final_memory = Some(chip_complex.base.memory_controller.memory_image().clone());
    }

    fn on_terminate(
        &mut self,
        _state: &mut ExecutionSegmentState<Self::Ctx>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
        _exit_code: u32,
    ) {
        self.final_memory = Some(chip_complex.base.memory_controller.memory_image().clone());
    }

    /// Execute a single instruction
    fn execute_instruction<Mem: GuestMemory>(
        &mut self,
        state: &mut ExecutionSegmentState<Mem, Self::Ctx>,
        instruction: &Instruction<F>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) -> Result<(), ExecutionError>
    where
        F: PrimeField32,
    {
        let &Instruction { opcode, .. } = instruction;

        if let Some(executor) = chip_complex.inventory.get_mut_executor(&opcode) {
            let mut vm_state = VmStateMut {
                pc: &mut state.pc,
                memory: state.memory.as_mut().unwrap(),
                ctx: &mut state.ctx,
            };
            executor.execute_e1(&mut vm_state, instruction)?;
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
    pub widths: &'a [usize],
    pub interactions: &'a [usize],
    pub since_last_segment_check: usize,
    pub final_memory: Option<MemoryImage>,
    // TODO(ayush): remove
    air_names: Vec<String>,
}

impl<'a> MeteredExecutionControl<'a> {
    pub fn new(widths: &'a [usize], interactions: &'a [usize], air_names: Vec<String>) -> Self {
        Self {
            widths,
            interactions,
            since_last_segment_check: 0,
            final_memory: None,
            air_names,
        }
    }

    /// Calculate the total cells used based on trace heights and widths
    // TODO(ayush): account for preprocessed and permutation columns
    fn calculate_total_cells(&self, trace_heights: &[usize]) -> usize {
        trace_heights
            .iter()
            .zip(self.widths)
            .map(|(&height, &width)| height.next_power_of_two() * width)
            .sum()
    }

    /// Calculate the total interactions based on trace heights and interaction counts
    fn calculate_total_interactions(&self, trace_heights: &[usize]) -> usize {
        trace_heights
            .iter()
            .zip(self.interactions)
            // TODO(ayush): should this have next_power_of_two
            .map(|(&height, &interactions)| height.next_power_of_two() * interactions)
            .sum()
    }
}

impl<F, VC> ExecutionControl<F, VC> for MeteredExecutionControl<'_>
where
    F: PrimeField32,
    VC: VmConfig<F>,
    VC::Executor: InsExecutorE1<F>,
{
    type Ctx = MeteredCtx;

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

        if state
            .ctx
            .trace_heights
            .iter()
            .any(|&height| height.next_power_of_two() > MAX_TRACE_HEIGHT)
        {
            return true;
        }

        let total_cells = self.calculate_total_cells(&state.ctx.trace_heights);
        if total_cells > MAX_TRACE_CELLS {
            return true;
        }

        let total_interactions = self.calculate_total_interactions(&state.ctx.trace_heights);
        if total_interactions > MAX_INTERACTIONS {
            return true;
        }

        false
    }

    fn on_segment_start(
        &mut self,
        state: &mut ExecutionSegmentState<Self::Ctx>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) {
        // Program | Connector | Public Values | Memory ... | Executors (except Public Values) | Range Checker

        state.ctx.trace_heights[0] = chip_complex.program_chip().true_program_length;
        state.ctx.trace_heights[1] = 2; // Connector chip

        let mut offset = 2;
        offset += chip_complex.memory_controller().num_airs();

        // Add heights for periphery chips with constant heights
        for (i, chip_id) in chip_complex
            .inventory
            .insertion_order
            .iter()
            .rev()
            .enumerate()
        {
            if let &ChipId::Periphery(id) = chip_id {
                let chip_index = offset + i;
                if let Some(constant_height) =
                    chip_complex.inventory.periphery[id].constant_trace_height()
                {
                    state.ctx.trace_heights[chip_index] = constant_height;
                }
            }
        }

        if let (Some(range_checker_height), Some(last_height)) = (
            chip_complex.range_checker_chip().constant_trace_height(),
            state.ctx.trace_heights.last_mut(),
        ) {
            *last_height = range_checker_height;
        }
    }

    fn on_segment_end(
        &mut self,
        state: &mut ExecutionSegmentState<Self::Ctx>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) {
        self.final_memory = Some(chip_complex.base.memory_controller.memory_image().clone());
    }

    fn on_terminate(
        &mut self,
        state: &mut ExecutionSegmentState<Self::Ctx>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
        _exit_code: u32,
    ) {
        for ((name, height), width) in self
            .air_names
            .iter()
            .zip(state.ctx.trace_heights.iter())
            .zip(self.widths.iter())
        {
            println!("{:<10} \t|\t{:<5} \t|\t{}", height, width, name);
        }

        self.final_memory = Some(chip_complex.base.memory_controller.memory_image().clone());
    }

    /// Execute a single instruction
    fn execute_instruction(
        &mut self,
        state: &mut ExecutionSegmentState<Self::Ctx>,
        instruction: &Instruction<F>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) -> Result<(), ExecutionError>
    where
        F: PrimeField32,
    {
        let &Instruction { opcode, .. } = instruction;

        // Program | Connector | Public Values | Memory ... | Executors (except Public Values) | Range Checker
        // TODO(ayush): no magic number, cache
        let mut offset = 2;
        offset += chip_complex.memory_controller().num_airs();
        if chip_complex.config().has_public_values_chip() {
            offset += 1;
        }

        if let Some((executor, i)) = chip_complex.inventory.get_mut_executor_with_index(&opcode) {
            let memory_controller = &mut chip_complex.base.memory_controller;
            let mut vm_state = VmStateMut {
                pc: &mut state.pc,
                memory: &mut memory_controller.memory.data,
                ctx: &mut state.ctx,
            };
            let index = offset + i;
            executor.execute_e2(&mut vm_state, instruction, index)?;
        } else {
            return Err(ExecutionError::DisabledOperation {
                pc: state.pc,
                opcode,
            });
        };

        Ok(())
    }
}
