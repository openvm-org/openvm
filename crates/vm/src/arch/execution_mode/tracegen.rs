use openvm_instructions::instruction::Instruction;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::{
    arch::{
        execution_control::ExecutionControl, BaseRecordArena, ExecutionError, ExecutionState,
        InsExecutor, MatrixRecordArena, VmChipComplex, VmConfig, VmSegmentState, VmStateMut,
        PUBLIC_VALUES_AIR_ID,
    },
    system::memory::INITIAL_TIMESTAMP,
};

#[derive(Debug)]
pub struct TracegenCtx<RA> {
    pub arenas: Vec<RA>,
}

impl<RA> TracegenCtx<RA>
where
    RA: BaseRecordArena,
{
    pub fn new(count: usize) -> Self {
        let arenas = (0..count)
            .map(|_| RA::with_capacity(RA::Capacity::default()))
            .collect();
        Self { arenas }
    }

    pub fn new_with_capacity(capacities: &[RA::Capacity]) -> Self {
        let arenas = capacities
            .iter()
            .map(|&capacity| RA::with_capacity(capacity))
            .collect();

        Self { arenas }
    }
}

#[derive(Default, derive_new::new)]
pub struct TracegenExecutionControl {
    pub instret_end: Option<u64>,
}

impl<F, VC> ExecutionControl<F, VC> for TracegenExecutionControl
where
    F: PrimeField32,
    VC: VmConfig<F>,
    VC::Executor: InsExecutor<F, MatrixRecordArena<F>>,
{
    // TODO(ayush): make generic
    type Ctx = TracegenCtx<MatrixRecordArena<F>>;

    fn initialize_context(&self) -> Self::Ctx {
        unimplemented!()
    }

    fn should_suspend(
        &self,
        state: &mut VmSegmentState<F, Self::Ctx>,
        _chip_complex: &VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) -> bool {
        self.instret_end
            .is_some_and(|instret_end| state.instret >= instret_end)
    }

    fn on_start(
        &self,
        state: &mut VmSegmentState<F, Self::Ctx>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) {
        chip_complex
            .connector_chip_mut()
            .begin(ExecutionState::new(state.pc, INITIAL_TIMESTAMP + 1));
    }

    fn on_suspend_or_terminate(
        &self,
        state: &mut VmSegmentState<F, Self::Ctx>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
        exit_code: Option<u32>,
    ) {
        let timestamp = chip_complex.memory_controller().timestamp();
        chip_complex
            .connector_chip_mut()
            .end(ExecutionState::new(state.pc, timestamp), exit_code);
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
        let offset = if chip_complex.config().has_public_values_chip() {
            PUBLIC_VALUES_AIR_ID + 1 + chip_complex.memory_controller().num_airs()
        } else {
            PUBLIC_VALUES_AIR_ID + chip_complex.memory_controller().num_airs()
        };

        let &Instruction { opcode, .. } = instruction;
        if let Some((executor, i)) = chip_complex.inventory.get_mut_executor_with_index(&opcode) {
            let vm_state = VmStateMut {
                pc: &mut state.pc,
                memory: &mut chip_complex.base.memory_controller.memory,
                streams: &mut state.streams,
                rng: &mut state.rng,
                ctx: &mut state.ctx,
            };
            executor.execute_tracegen(vm_state, instruction, offset + i)?;
        } else {
            return Err(ExecutionError::DisabledOperation {
                pc: state.pc,
                opcode,
            });
        };

        Ok(())
    }
}
