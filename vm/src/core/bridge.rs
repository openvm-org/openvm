use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use super::{columns::CoreIoCols, timestamp_delta, CoreAir};
use crate::arch::{
    columns::ExecutionState,
    instructions::{Opcode, OpcodeEncoderWithBuilder, CORE_INSTRUCTIONS},
};

impl CoreAir {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        io: CoreIoCols<AB::Var>,
        next_pc: AB::Var,
        encoder: &OpcodeEncoderWithBuilder<AB, 4, 6>,
    ) {
        self.execution_bridge
            .execute(
                io.opcode,
                [io.op_a, io.op_b, io.op_c, io.d, io.e, io.op_f, io.op_g],
                ExecutionState::new(io.pc, io.timestamp),
                ExecutionState::<AB::Expr>::new(
                    next_pc.into(),
                    io.timestamp
                        + CORE_INSTRUCTIONS
                            .iter()
                            .map(|&op| {
                                AB::Expr::from_canonical_usize(timestamp_delta(op))
                                    * encoder.expression_for(op)
                            })
                            .fold(AB::Expr::zero(), |x, y| x + y),
                ),
            )
            .eval(
                builder,
                AB::Expr::one() - encoder.expression_for(Opcode::NOP),
            );
    }
}
