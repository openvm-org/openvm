use std::collections::BTreeMap;

use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use super::{columns::CoreIoCols, timestamp_delta, CoreAir, READ_INSTRUCTION_BUS};
use crate::arch::{columns::ExecutionState, instructions::Opcode};

impl CoreAir {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        io: CoreIoCols<AB::Var>,
        next_pc: AB::Var,
        operation_flags: &BTreeMap<Opcode, AB::Var>,
    ) {
        // Interaction with program
        builder.push_send(
            READ_INSTRUCTION_BUS,
            [
                io.pc, io.opcode, io.op_a, io.op_b, io.op_c, io.d, io.e, io.op_f, io.op_g,
            ],
            AB::Expr::one() - operation_flags[&Opcode::NOP],
        );

        self.execution_bus.execute(
            builder,
            AB::Expr::one() - operation_flags[&Opcode::NOP],
            ExecutionState::new(io.pc, io.timestamp),
            ExecutionState::<AB::Expr>::new(
                next_pc.into(),
                io.timestamp
                    + operation_flags
                        .iter()
                        .map(|(op, flag)| {
                            AB::Expr::from_canonical_usize(timestamp_delta(*op)) * (*flag).into()
                        })
                        .fold(AB::Expr::zero(), |x, y| x + y),
            ),
        );
    }
}
