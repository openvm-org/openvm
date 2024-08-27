use std::iter;

use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use crate::arch::columns::{ExecutionState, InstructionCols};

#[derive(Clone, Copy, Debug)]
pub struct ExecutionBus(pub usize);

impl ExecutionBus {
    pub fn execute_increment_pc<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        multiplicity: impl Into<AB::Expr>,
        prev_state: ExecutionState<AB::Expr>,
        timestamp_change: impl Into<AB::Expr>,
        instruction: InstructionCols<AB::Expr>,
    ) {
        let next_state = ExecutionState {
            pc: prev_state.pc.clone() + AB::F::one(),
            timestamp: prev_state.timestamp.clone() + timestamp_change.into(),
        };
        self.execute(builder, multiplicity, prev_state, next_state, instruction);
    }
    pub fn execute<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        multiplicity: impl Into<AB::Expr>,
        prev_state: ExecutionState<impl Into<AB::Expr>>,
        next_state: ExecutionState<impl Into<AB::Expr>>,
        instruction: InstructionCols<AB::Expr>,
    ) {
        let fields = iter::empty()
            .chain(prev_state.flatten().map(Into::into))
            .chain(next_state.flatten().map(Into::into))
            .chain(instruction.flatten());
        builder.push_receive(self.0, fields, multiplicity);
    }
    /*pub fn initial_final<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        prev_state: ExecutionState<AB::Expr>,
        next_state: ExecutionState<AB::Expr>,
    ) {
    }*/
}
