use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use super::{
    air::FieldExtensionArithmeticAir,
    columns::{FieldExtensionArithmeticCols, FieldExtensionArithmeticIoCols},
};
use crate::{
    arch::columns::ExecutionState, field_extension::columns::FieldExtensionArithmeticAuxCols,
    memory::MemoryAddress,
};

impl FieldExtensionArithmeticAir {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: FieldExtensionArithmeticCols<AB::Var>,
        expected_opcode: AB::Expr,
    ) {
        let FieldExtensionArithmeticCols { io, aux } = local;

        let FieldExtensionArithmeticIoCols {
            pc,
            timestamp,
            op_a,
            op_b,
            op_c,
            d,
            e,
            x,
            y,
            z,
            ..
        } = io;

        let FieldExtensionArithmeticAuxCols {
            read_x_aux_cols,
            read_y_aux_cols,
            write_aux_cols,
            is_valid,
            ..
        } = aux;

        // Interaction with program
        self.program_bus.send_instruction(
            builder,
            [
                pc.into(),
                expected_opcode.clone(),
                op_a.into(),
                op_b.into(),
                op_c.into(),
                d.into(),
                e.into(),
                AB::Expr::zero(),
                AB::Expr::zero(),
            ]
            .into_iter(),
            is_valid,
        );

        let mut timestamp_delta = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_canonical_usize(timestamp_delta - 1)
        };

        // Reads for x
        self.memory_bridge
            .read(
                MemoryAddress::new(d, op_b),
                x,
                timestamp_pp(),
                &read_x_aux_cols,
            )
            .eval(builder, is_valid);

        // Reads for y
        self.memory_bridge
            .read(
                MemoryAddress::new(e, op_c),
                y,
                timestamp_pp(),
                &read_y_aux_cols,
            )
            .eval(builder, is_valid);

        // Writes for z
        self.memory_bridge
            .write(
                MemoryAddress::new(d, op_a),
                z,
                timestamp_pp(),
                &write_aux_cols,
            )
            .eval(builder, is_valid);

        self.execution_bus.execute_increment_pc(
            builder,
            aux.is_valid,
            ExecutionState::new(pc, timestamp),
            AB::F::from_canonical_usize(timestamp_delta),
        );
    }
}
