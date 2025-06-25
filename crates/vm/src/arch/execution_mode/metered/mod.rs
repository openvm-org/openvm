pub mod ctx;
pub mod memory_ctx;
pub mod segment_ctx;

pub use ctx::MeteredCtx;
use openvm_instructions::instruction::Instruction;
use openvm_stark_backend::p3_field::PrimeField32;
pub use segment_ctx::Segment;

use crate::arch::{
    execution_control::ExecutionControl, ExecutionError, InsExecutorE1, VmChipComplex, VmConfig,
    VmSegmentState, VmStateMut, PUBLIC_VALUES_AIR_ID,
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
        _state: &mut VmSegmentState<F, Self::Ctx>,
        _chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) {
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
