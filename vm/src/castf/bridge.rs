use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use super::{
    air::{CastFAir, FINAL_LIMB_SIZE, LIMB_SIZE},
    columns::{CastFAuxCols, CastFIoCols},
};
use crate::{arch::columns::InstructionCols, memory::MemoryAddress};

impl CastFAir {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        io: &CastFIoCols<AB::Var>,
        aux: &CastFAuxCols<AB::Var>,
        expected_opcode: AB::Expr,
    ) {
        let mut timestamp_delta = AB::Expr::zero();

        let timestamp: AB::Var = io.from_state.timestamp;

        self.memory_bridge
            .read(
                MemoryAddress::new(io.d, io.op_a),
                io.x,
                timestamp + timestamp_delta.clone(),
                &aux.read_x_aux_cols,
            )
            .eval(builder, aux.is_valid);
        timestamp_delta += AB::Expr::one();

        self.memory_bridge
            .write(
                MemoryAddress::new(io.e, io.op_b),
                [io.y],
                timestamp + timestamp_delta.clone(),
                &aux.write_y_aux_cols,
            )
            .eval(builder, aux.is_valid);
        timestamp_delta += AB::Expr::one();

        self.execution_bus.execute_increment_pc(
            builder,
            aux.is_valid,
            io.from_state.map(Into::into),
            timestamp_delta,
            InstructionCols::<AB::Expr>::new(
                expected_opcode,
                [
                    io.op_a.into(),
                    io.op_b.into(),
                    AB::Expr::zero(),
                    io.d.into(),
                    io.e.into(),
                ],
            ),
        );

        for i in 0..4 {
            self.bus
                .range_check(
                    io.x[i],
                    match i {
                        0..=2 => LIMB_SIZE,
                        3 => FINAL_LIMB_SIZE,
                        _ => unreachable!(),
                    },
                )
                .eval(builder, aux.is_valid);
        }
    }
}
