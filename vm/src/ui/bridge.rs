use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use super::{
    air::UiAir,
    columns::{UiAuxCols, UiIoCols},
};
use crate::memory::MemoryAddress;

impl UiAir {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        io: &UiIoCols<AB::Var>,
        aux: &UiAuxCols<AB::Var>,
        expected_opcode: AB::Expr,
    ) {
        let timestamp = io.from_state.timestamp;
        let mut timestamp_delta = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::Expr::from_canonical_usize(timestamp_delta - 1)
        };

        self.memory_bridge
            .write(
                MemoryAddress::new(AB::Expr::one(), io.op_a),
                [
                    AB::Expr::zero(),
                    aux.imm_lo_hex * AB::Expr::from_canonical_u32(16),
                    io.x_cols[0].into(),
                    io.x_cols[1].into(),
                ],
                timestamp_pp(),
                &aux.write_x_aux_cols,
            )
            .eval(builder, aux.is_valid);

        self.execution_bridge
            .execute_and_increment_pc(
                expected_opcode + AB::Expr::from_canonical_usize(self.offset),
                [io.op_a, io.op_b],
                io.from_state,
                AB::F::from_canonical_usize(timestamp_delta),
            )
            .eval(builder, aux.is_valid);

        self.bus
            .range_check(io.x_cols[0], 8)
            .eval(builder, aux.is_valid);
        self.bus
            .range_check(io.x_cols[1], 8)
            .eval(builder, aux.is_valid);
        self.bus
            .range_check(aux.imm_lo_hex, 4)
            .eval(builder, aux.is_valid);
    }
}
