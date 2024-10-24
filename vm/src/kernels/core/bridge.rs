use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use super::{columns::CoreIoCols, timestamp_delta, CoreAir};
use crate::{arch::ExecutionState, kernels::core::columns::CoreAuxCols};

impl CoreAir {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        io: &CoreIoCols<AB::Var>,
        aux: &CoreAuxCols<AB::Var>,
    ) {
        let &CoreAuxCols {
            is_valid,
            ref operation_flags,
        } = aux;

        let next_pc = io.pc + AB::Expr::from_canonical_usize(1);

        self.execution_bridge
            .execute(
                io.opcode + AB::Expr::from_canonical_usize(self.offset),
                [io.a, io.b, io.c, io.d, io.e],
                ExecutionState::new(io.pc, io.timestamp),
                ExecutionState::<AB::Expr>::new(
                    next_pc,
                    io.timestamp
                        + operation_flags
                            .iter()
                            .map(|(op, flag)| {
                                AB::Expr::from_canonical_u32(timestamp_delta(*op)) * (*flag).into()
                            })
                            .fold(AB::Expr::zero(), |x, y| x + y),
                ),
            )
            .eval(builder, is_valid);
    }
}
