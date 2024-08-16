use p3_field::AbstractField;

use afs_stark_backend::interaction::InteractionBuilder;

use crate::arch::columns::{ExecutionState, InstructionCols};

#[derive(Clone, Copy, Debug)]
pub struct ExecutionBus(pub usize);

impl ExecutionBus {
    pub fn interact_execute<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        prev_state: ExecutionState<AB::Expr>,
        next_state: ExecutionState<AB::Expr>,
        instruction: InstructionCols<AB::Expr>,
    ) {
        self.interact_execute_with_multiplicity(
            builder,
            AB::Expr::one(),
            prev_state,
            next_state,
            instruction,
        );
    }
    pub fn interact_execute_with_multiplicity<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        multiplicity: AB::Expr,
        prev_state: ExecutionState<AB::Expr>,
        next_state: ExecutionState<AB::Expr>,
        instruction: InstructionCols<AB::Expr>,
    ) {
    }
    pub fn interact_initial_final<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        prev_state: ExecutionState<AB::Expr>,
        next_state: ExecutionState<AB::Expr>,
    ) {
    }
}
