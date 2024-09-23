use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use crate::arch::columns::ExecutionState;

#[derive(Clone, Copy, Debug)]
pub struct ExecutionBus(pub usize);

impl ExecutionBus {
    pub fn execute_and_increment_pc<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        multiplicity: impl Into<AB::Expr>,
        prev_state: ExecutionState<AB::Expr>,
        timestamp_change: impl Into<AB::Expr>,
    ) {
        let next_state = ExecutionState {
            pc: prev_state.pc.clone() + AB::F::one(),
            timestamp: prev_state.timestamp.clone() + timestamp_change.into(),
        };
        self.execute(builder, multiplicity, prev_state, next_state);
    }
    pub fn execute<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        multiplicity: impl Into<AB::Expr>,
        prev_state: ExecutionState<impl Into<AB::Expr>>,
        next_state: ExecutionState<impl Into<AB::Expr>>,
    ) {
        let multiplicity = multiplicity.into();
        builder.push_receive(
            self.0,
            [prev_state.pc.into(), prev_state.timestamp.into()],
            multiplicity.clone(),
        );
        builder.push_send(
            self.0,
            [next_state.pc.into(), next_state.timestamp.into()],
            multiplicity,
        );
    }
}
