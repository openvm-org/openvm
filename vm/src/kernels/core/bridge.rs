use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use super::{columns::CoreIoCols, CoreAir};
use crate::{arch::ExecutionState, kernels::core::columns::CoreAuxCols, system::DEFAULT_PC_STEP};

impl CoreAir {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        io: &CoreIoCols<AB::Var>,
        aux: &CoreAuxCols<AB::Var>,
    ) {
        self.execution_bridge
            .execute(
                io.opcode + AB::Expr::from_canonical_usize(self.offset),
                [io.a, io.b, io.c, io.d, io.e],
                ExecutionState::new(io.pc, io.timestamp),
                ExecutionState::<AB::Expr>::new(
                    io.pc + AB::Expr::from_canonical_u32(DEFAULT_PC_STEP),
                    io.timestamp + AB::Expr::one(),
                ),
            )
            .eval(builder, aux.is_valid);
    }
}
