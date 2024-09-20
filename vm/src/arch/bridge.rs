use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use crate::{
    arch::{bus::ExecutionBus, columns::ExecutionState},
    program::bridge::ProgramBus,
};

#[derive(Copy, Clone, Debug)]
pub struct ExecutionBridge {
    execution_bus: ExecutionBus,
    program_bus: ProgramBus,
}

impl ExecutionBridge {
    pub fn new(execution_bus: ExecutionBus, program_bus: ProgramBus) -> Self {
        Self {
            execution_bus,
            program_bus,
        }
    }

    pub fn execute_increment_pc<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        opcode: impl Into<AB::Expr>,
        instruction: impl IntoIterator<Item = impl Into<AB::Expr>>,
        from_state: ExecutionState<impl Into<AB::Expr> + Clone>,
        timestamp_change: impl Into<AB::Expr>,
        multiplicity: impl Into<AB::Expr> + Clone,
    ) {
        let to_state = ExecutionState {
            pc: from_state.pc.clone().into() + AB::Expr::one(),
            timestamp: from_state.timestamp.clone().into() + timestamp_change.into(),
        };
        self.execute(
            builder,
            opcode,
            instruction,
            from_state,
            to_state,
            multiplicity,
        );
    }

    pub fn execute<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        opcode: impl Into<AB::Expr>,
        instruction: impl IntoIterator<Item = impl Into<AB::Expr>>,
        from_state: ExecutionState<impl Into<AB::Expr> + Clone>,
        to_state: ExecutionState<impl Into<AB::Expr>>,
        multiplicity: impl Into<AB::Expr> + Clone,
    ) {
        // Interaction with program
        self.program_bus.send_instruction(
            builder,
            from_state.pc.clone().into(),
            opcode,
            instruction,
            multiplicity.clone().into(),
        );

        self.execution_bus
            .execute(builder, multiplicity, from_state, to_state);
    }
}
