use std::{borrow::Borrow, iter, sync::Arc};

use afs_derive::AlignedBorrow;
use afs_stark_backend::{
    config::{StarkGenericConfig, Val},
    interaction::InteractionBuilder,
    prover::types::AirProofInput,
    rap::{get_air_name, AnyRap, BaseAirWithPublicValues, PartitionedBaseAir},
    Chip, ChipUsageGetter,
};
use axvm_instructions::{Rv32NopOpcode, UsizeOpcode};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    arch::{ExecutionBridge, ExecutionBus, ExecutionState, InstructionExecutor},
    system::program::{bridge::ProgramBus, ExecutionError, Instruction},
};

#[cfg(test)]
mod tests;

#[derive(Clone, Debug)]
pub struct Rv32TerminateNopAir {
    pub execution_bridge: ExecutionBridge,
    pub nop_opcode: usize,
}

#[derive(AlignedBorrow)]
pub struct Rv32TerminateNopCols<T> {
    pub pc: T,
    pub timestamp: T,
    pub is_valid: T,
}

impl<F: Field> BaseAir<F> for Rv32TerminateNopAir {
    fn width(&self) -> usize {
        Rv32TerminateNopCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for Rv32TerminateNopAir {
    fn num_public_values(&self) -> usize {
        0
    }
}

impl<F: Field> PartitionedBaseAir<F> for Rv32TerminateNopAir {}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for Rv32TerminateNopAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let Rv32TerminateNopCols {
            pc,
            timestamp,
            is_valid,
        } = (*local).borrow();

        self.execution_bridge
            .execute(
                AB::Expr::from_canonical_usize(self.nop_opcode),
                iter::empty::<AB::Expr>(),
                ExecutionState::<AB::Expr>::new((*pc).into(), (*timestamp).into()),
                ExecutionState::<AB::Expr>::new(
                    (*pc).into() + AB::Expr::from_canonical_usize(4) * (*is_valid),
                    (*timestamp).into(),
                ),
            )
            .eval(builder, (*is_valid).into());
    }
}

pub struct Rv32TerminateNopChip<F> {
    pub air: Rv32TerminateNopAir,
    pub rows: Vec<Rv32TerminateNopCols<F>>,
    pub nop_opcode: usize,
}

impl<F> Rv32TerminateNopChip<F> {
    pub fn new(execution_bus: ExecutionBus, program_bus: ProgramBus, offset: usize) -> Self {
        Self {
            air: Rv32TerminateNopAir {
                execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
                nop_opcode: offset + Rv32NopOpcode::NOP.as_usize(),
            },
            rows: vec![],
            nop_opcode: offset,
        }
    }
}

impl<F: PrimeField32> InstructionExecutor<F> for Rv32TerminateNopChip<F> {
    fn execute(
        &mut self,
        instruction: Instruction<F>,
        from_state: ExecutionState<u32>,
    ) -> Result<ExecutionState<u32>, ExecutionError> {
        let Instruction { opcode, .. } = instruction;
        let local_opcode_index = Rv32NopOpcode::from_usize(opcode - self.nop_opcode);
        self.rows.push(Rv32TerminateNopCols {
            pc: F::from_canonical_u32(from_state.pc),
            timestamp: F::from_canonical_u32(from_state.timestamp),
            is_valid: F::from_bool(local_opcode_index == Rv32NopOpcode::NOP),
        });
        Ok(ExecutionState::new(from_state.pc + 4, from_state.timestamp))
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        let local_opcode_index = Rv32NopOpcode::from_usize(opcode - self.nop_opcode);
        format!("{local_opcode_index:?}")
    }
}

impl<F: PrimeField32> ChipUsageGetter for Rv32TerminateNopChip<F> {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }
    fn current_trace_height(&self) -> usize {
        self.rows.len()
    }
    fn trace_width(&self) -> usize {
        Rv32TerminateNopCols::<F>::width()
    }
    fn current_trace_cells(&self) -> usize {
        self.trace_width() * self.current_trace_height()
    }
}

impl<SC: StarkGenericConfig> Chip<SC> for Rv32TerminateNopChip<Val<SC>>
where
    Val<SC>: PrimeField32,
{
    fn air(&self) -> Arc<dyn AnyRap<SC>> {
        Arc::new(self.air.clone())
    }

    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        let curr_height = self.rows.len();
        let correct_height = self.rows.len().next_power_of_two();
        let width = Rv32TerminateNopCols::<Val<SC>>::width();

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
