pub mod ctx;
pub mod memory_ctx;
pub mod segment_ctx;

pub use ctx::MeteredCtx;
use openvm_instructions::instruction::Instruction;
use openvm_stark_backend::{p3_field::PrimeField32, ChipUsageGetter};
pub use segment_ctx::Segment;

use crate::arch::{
    execution_control::ExecutionControl, ChipId, ExecutionError, InsExecutorE1, VmChipComplex,
    VmConfig, VmSegmentState, VmStateMut, CONNECTOR_AIR_ID, PROGRAM_AIR_ID, PUBLIC_VALUES_AIR_ID,
};

#[derive(Default)]
pub struct MeteredExecutionControl;

impl<F, VC> ExecutionControl<F, VC> for MeteredExecutionControl
where
    F: PrimeField32,
    VC: VmConfig<F>,
    VC::Executor: InsExecutorE1<F>,
{
    type Ctx = MeteredCtx;

    fn initialize_context(&self) -> Self::Ctx {
        todo!()
    }

    fn should_suspend(
        &self,
        _state: &mut VmSegmentState<F, Self::Ctx>,
        _chip_complex: &VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) -> bool {
        false
    }

    fn on_start(
        &self,
        state: &mut VmSegmentState<F, Self::Ctx>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) {
        // Program | Connector | Public Values | Memory ... | Executors (except Public Values) |
        // Range Checker
        assert_eq!(
            state.ctx.trace_heights[PROGRAM_AIR_ID],
            chip_complex
                .program_chip()
                .true_program_length
                .next_power_of_two() as u32
        );
        assert!(state.ctx.is_trace_height_constant[PROGRAM_AIR_ID]);
        assert_eq!(state.ctx.trace_heights[CONNECTOR_AIR_ID], 2);
        assert!(state.ctx.is_trace_height_constant[CONNECTOR_AIR_ID]);

        let offset = if chip_complex.config().has_public_values_chip() {
            PUBLIC_VALUES_AIR_ID + 1 + chip_complex.memory_controller().num_airs()
        } else {
            PUBLIC_VALUES_AIR_ID + chip_complex.memory_controller().num_airs()
        };
        // Periphery chips with constant heights
        for (i, chip_id) in chip_complex
            .inventory
            .insertion_order
            .iter()
            .rev()
            .enumerate()
        {
            if let &ChipId::Periphery(id) = chip_id {
                if let Some(constant_height) =
                    chip_complex.inventory.periphery[id].constant_trace_height()
                {
                    assert_eq!(state.ctx.trace_heights[offset + i], constant_height as u32);
                    assert!(state.ctx.is_trace_height_constant[offset + i]);
                }
            }
        }

        // Range checker chip
        if let (Some(range_checker_height), Some(last_height), Some(last_is_height_constant)) = (
            chip_complex.range_checker_chip().constant_trace_height(),
            state.ctx.trace_heights.last(),
            state.ctx.is_trace_height_constant.last(),
        ) {
            assert_eq!(*last_height, range_checker_height as u32);
            assert!(*last_is_height_constant);
        }
    }

    fn on_suspend_or_terminate(
        &self,
        state: &mut VmSegmentState<F, Self::Ctx>,
        _chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
        _exit_code: Option<u32>,
    ) {
        state
            .ctx
            .segmentation_ctx
            .add_final_segment(state.instret, &state.ctx.trace_heights);
    }

    /// Execute a single instruction
    fn execute_instruction(
        &self,
        state: &mut VmSegmentState<F, Self::Ctx>,
        instruction: &Instruction<F>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) -> Result<(), ExecutionError>
    where
        F: PrimeField32,
    {
        // Check if segmentation needs to happen
        state.ctx.check_and_segment(state.instret);

        let offset = if chip_complex.config().has_public_values_chip() {
            PUBLIC_VALUES_AIR_ID + 1 + chip_complex.memory_controller().num_airs()
        } else {
            PUBLIC_VALUES_AIR_ID + chip_complex.memory_controller().num_airs()
        };
        let &Instruction { opcode, .. } = instruction;
        if let Some((executor, i)) = chip_complex.inventory.get_mut_executor_with_index(&opcode) {
            let mut vm_state = VmStateMut {
                pc: &mut state.pc,
                memory: state.memory.as_mut().unwrap(),
                streams: &mut state.streams,
                rng: &mut state.rng,
                ctx: &mut state.ctx,
            };
            executor.execute_metered(&mut vm_state, instruction, offset + i)?;
        } else {
            return Err(ExecutionError::DisabledOperation {
                pc: state.pc,
                opcode,
            });
        };

        Ok(())
    }
}
