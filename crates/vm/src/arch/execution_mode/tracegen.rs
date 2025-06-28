use openvm_circuit_primitives::utils::next_power_of_two_or_zero;
use openvm_instructions::instruction::Instruction;
use openvm_stark_backend::{p3_field::PrimeField32, ChipUsageGetter};

use crate::{
    arch::{
        execution_control::ExecutionControl, ChipId, ExecutionError, ExecutionState,
        InstructionExecutor, MatrixRecordArena, RowMajorMatrixArena, VmChipComplex, VmConfig,
        VmSegmentState, VmStateMut, PUBLIC_VALUES_AIR_ID,
    },
    system::memory::INITIAL_TIMESTAMP,
};

#[derive(Default)]
pub struct TracegenCtx<F> {
    pub arenas: Vec<MatrixRecordArena<F>>,
    pub instret_end: Option<u64>,
}

impl<F> TracegenCtx<F>
where
    F: PrimeField32,
{
    /// `capacities` is list of `(height, width)` dimensions for each matrix arena.
    pub fn new_with_capacity(capacities: &[(usize, usize)], instret_end: Option<u64>) -> Self {
        let arenas = capacities
            .iter()
            .map(|&(height, width)| {
                MatrixRecordArena::with_capacity(next_power_of_two_or_zero(height), width)
            })
            .collect();

        Self {
            arenas,
            instret_end,
        }
    }
}

#[derive(Default)]
pub struct TracegenExecutionControl;

impl<F, VC> ExecutionControl<F, VC> for TracegenExecutionControl
where
    F: PrimeField32,
    VC: VmConfig<F>,
{
    type Ctx = TracegenCtx<F>;

    fn initialize_context(&self) -> Self::Ctx {
        unreachable!()
    }

    fn should_suspend(
        &self,
        state: &mut VmSegmentState<F, Self::Ctx>,
        _chip_complex: &VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) -> bool {
        state
            .ctx
            .instret_end
            .is_some_and(|instret_end| state.instret >= instret_end)
    }

    fn on_start(
        &self,
        state: &mut VmSegmentState<F, Self::Ctx>,
        chip_complex: &mut VmChipComplex<F, VC::Executor, VC::Periphery>,
    ) {
        let offset = if chip_complex.config().has_public_values_chip() {
            PUBLIC_VALUES_AIR_ID + 1 + chip_complex.memory_controller().num_airs()
        } else {
            PUBLIC_VALUES_AIR_ID + chip_complex.memory_controller().num_airs()
        };
        let mut executor_id_to_air_id = vec![0; chip_complex.inventory.executors().len()];
        for (insertion_id, chip_id) in chip_complex
            .inventory
            .insertion_order
            .iter()
            .rev()
            .enumerate()
        {
            match chip_id {
                ChipId::Executor(exec_id) => {
                    executor_id_to_air_id[*exec_id] = insertion_id + offset
                }
                _ => {}
            }
        }
        chip_complex.inventory.executor_id_to_air_id = executor_id_to_air_id;

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
        let &Instruction { opcode, .. } = instruction;

        if let Some((executor, air_id)) =
            chip_complex.inventory.get_mut_executor_with_air_id(&opcode)
        {
            let memory = &mut chip_complex.base.memory_controller.memory;
            let arena = &mut state.ctx.arenas[air_id];
            println!(
                "executor: {}, arena size={}",
                executor.air_name(),
                arena.trace_buffer.len()
            );
            let state_mut = VmStateMut {
                pc: &mut state.pc,
                memory,
                streams: &mut state.streams,
                rng: &mut state.rng,
                ctx: arena,
            };
            executor.execute(state_mut, instruction)?;
        } else {
            return Err(ExecutionError::DisabledOperation {
                pc: state.pc,
                opcode,
            });
        };

        Ok(())
    }
}
