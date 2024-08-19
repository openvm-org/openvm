use p3_field::AbstractField;

use afs_stark_backend::interaction::InteractionBuilder;

use crate::arch::columns::{ExecutionState, InstructionCols};

#[derive(Clone, Copy, Debug)]
pub struct ExecutionBus(pub usize);

impl ExecutionBus {
    pub fn execute_increment_pc<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        prev_state: ExecutionState<AB::Expr>,
        timestamp_change: impl Into<AB::Expr>,
        instruction: InstructionCols<AB::Expr>,
    ) {
        let next_state = ExecutionState {
            pc: prev_state.pc.clone() + AB::F::one(),
            timestamp: prev_state.timestamp.clone() + timestamp_change.into(),
        };
        self.execute(builder, prev_state, next_state, instruction);
    }
    pub fn execute<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        prev_state: ExecutionState<AB::Expr>,
        next_state: ExecutionState<AB::Expr>,
        instruction: InstructionCols<AB::Expr>,
    ) {
        self.execute_with_multiplicity(
            builder,
            AB::Expr::one(),
            prev_state,
            next_state,
            instruction,
        );
    }
    pub fn execute_with_multiplicity<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        multiplicity: impl Into<AB::Expr>,
        prev_state: ExecutionState<AB::Expr>,
        next_state: ExecutionState<AB::Expr>,
        instruction: InstructionCols<AB::Expr>,
    ) {
        let mut fields = vec![];
        fields.extend(prev_state.flatten());
        fields.extend(next_state.flatten());
        fields.extend(instruction.flatten());
        builder.push_receive(self.0, fields, multiplicity);
    }
    pub fn initial_final<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        prev_state: ExecutionState<AB::Expr>,
        next_state: ExecutionState<AB::Expr>,
    ) {
    }
}
