use std::collections::BTreeMap;

use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use super::{columns::CoreIoCols, timestamp_delta, CoreAir};
use crate::arch::{instructions::CoreOpcode, ExecutionState};

impl CoreAir {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        io: CoreIoCols<AB::Var>,
        next_pc: AB::Var,
        operation_flags: &BTreeMap<CoreOpcode, AB::Var>,
    ) {
        self.execution_bridge
            .execute(
                io.opcode + AB::Expr::from_canonical_usize(self.offset),
                [io.a, io.b, io.c, io.d, io.e, io.f, io.g],
                ExecutionState::new(io.pc, io.timestamp),
                ExecutionState::<AB::Expr>::new(
                    next_pc.into(),
                    io.timestamp
                        + operation_flags
                            .iter()
                            .map(|(op, flag)| {
                                AB::Expr::from_canonical_u32(timestamp_delta(*op)) * (*flag).into()
                            })
                            .fold(AB::Expr::zero(), |x, y| x + y),
                ),
            )
            .eval(
                builder,
                AB::Expr::one() - operation_flags[&CoreOpcode::DUMMY],
            );
    }
}
