use std::{borrow::Borrow, iter, sync::Arc};

use afs_derive::AlignedBorrow;
use afs_stark_backend::{
    config::{StarkGenericConfig, Val},
    interaction::InteractionBuilder,
    prover::types::AirProofInput,
    rap::{get_air_name, AnyRap, BaseAirWithPublicValues, PartitionedBaseAir},
    Chip, ChipUsageGetter,
};
use axvm_instructions::{instruction::Instruction, NopOpcode, UsizeOpcode};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    arch::{ExecutionBridge, ExecutionBus, ExecutionState, InstructionExecutor, PcIncOrSet},
    system::{
        memory::MemoryControllerRef,
        program::{ExecutionError, ProgramBus},
        DEFAULT_PC_STEP,
    },
};

#[cfg(test)]
mod tests;

#[derive(Clone, Debug)]
pub struct NopAir {
    pub execution_bridge: ExecutionBridge,
    pub nop_opcode: usize,
}

#[derive(AlignedBorrow)]
pub struct NopCols<T> {
    pub pc: T,
    pub timestamp: T,
    pub is_valid: T,
}

impl<F: Field> BaseAir<F> for NopAir {
    fn width(&self) -> usize {
        NopCols::<F>::width()
    }
}
impl<F: Field> PartitionedBaseAir<F> for NopAir {}
impl<F: Field> BaseAirWithPublicValues<F> for NopAir {}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for NopAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let &NopCols {
            pc,
            timestamp,
            is_valid,
        } = (*local).borrow();

        self.execution_bridge
            .execute_and_increment_or_set_pc(
                AB::Expr::from_canonical_usize(self.nop_opcode),
                iter::empty::<AB::Expr>(),
                ExecutionState::<AB::Expr>::new(pc, timestamp),
                AB::Expr::one(),
                PcIncOrSet::Inc(AB::Expr::from_canonical_u32(DEFAULT_PC_STEP)),
            )
            .eval(builder, is_valid);
    }
}

pub struct NopChip<F: Field> {
    pub air: NopAir,
    pub rows: Vec<NopCols<F>>,
    pub nop_opcode: usize,
    memory: MemoryControllerRef<F>,
}

impl<F: Field> NopChip<F> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_controller: MemoryControllerRef<F>,
        offset: usize,
    ) -> Self {
        Self {
            air: NopAir {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                nop_opcode: offset + NopOpcode::NOP.as_usize(),
            },
            rows: vec![],
            memory: memory_controller,
            nop_opcode: offset,
        }
    }
}

impl<F: PrimeField32> InstructionExecutor<F> for NopChip<F> {
    fn execute(
        &mut self,
        instruction: Instruction<F>,
        from_state: ExecutionState<u32>,
    ) -> Result<ExecutionState<u32>, ExecutionError> {
        let Instruction { opcode, .. } = instruction;
        assert_eq!(opcode, self.nop_opcode);
        self.rows.push(NopCols {
            pc: F::from_canonical_u32(from_state.pc),
            timestamp: F::from_canonical_u32(from_state.timestamp),
            is_valid: F::one(),
        });
        self.memory.borrow_mut().increment_timestamp();
        Ok(ExecutionState::new(
            from_state.pc + DEFAULT_PC_STEP,
            from_state.timestamp + 1,
        ))
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        let local_opcode_index = NopOpcode::from_usize(opcode - self.nop_opcode);
        format!("{local_opcode_index:?}")
    }
}

impl<F: PrimeField32> ChipUsageGetter for NopChip<F> {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }
    fn current_trace_height(&self) -> usize {
        self.rows.len()
    }
    fn trace_width(&self) -> usize {
        NopCols::<F>::width()
    }
    fn current_trace_cells(&self) -> usize {
        self.trace_width() * self.current_trace_height()
    }
}

impl<SC: StarkGenericConfig> Chip<SC> for NopChip<Val<SC>>
where
    Val<SC>: PrimeField32,
{
    fn air(&self) -> Arc<dyn AnyRap<SC>> {
        Arc::new(self.air.clone())
    }

    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        let curr_height = self.rows.len();
        let correct_height = self.rows.len().next_power_of_two();
        let width = NopCols::<Val<SC>>::width();

        let trace = RowMajorMatrix::new(
            self.rows
                .iter()
                .flat_map(|row| vec![row.pc, row.timestamp, row.is_valid])
                .chain(iter::repeat(Val::<SC>::zero()).take((correct_height - curr_height) * width))
                .collect::<Vec<_>>(),
            width,
        );
        AirProofInput::simple(self.air(), trace, vec![])
    }
}
