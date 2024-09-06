use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use super::{
    air::FieldExtensionArithmeticAir,
    chip::EXT_DEG,
    columns::{FieldExtensionArithmeticCols, FieldExtensionArithmeticIoCols},
};
use crate::{
    arch::columns::{ExecutionState, InstructionCols},
    field_extension::columns::FieldExtensionArithmeticAuxCols,
    memory::{offline_checker::bridge::MemoryBridge, MemoryAddress},
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

        let memory_bridge = MemoryBridge::new(self.mem_oc);

        // TODO[zach]: Change to 1 after proper batch access support.
        // The amount that timestamp increases after each access.
        let delta_per_access = AB::F::from_canonical_usize(EXT_DEG);
        let mut timestamp_delta = AB::Expr::zero();

        // Reads for x
        memory_bridge
            .read(MemoryAddress::new(d, op_b), x, timestamp, read_x_aux_cols)
            .eval(builder, is_valid);

        timestamp_delta += delta_per_access.into();

        // Reads for y
        memory_bridge
            .read(
                MemoryAddress::new(e, op_c),
                y,
                timestamp + timestamp_delta.clone(),
                read_y_aux_cols,
            )
            .eval(builder, is_valid);

        timestamp_delta += delta_per_access.into();

        // Writes for z
        memory_bridge
            .write(
                MemoryAddress::new(d, op_a),
                z,
                timestamp + timestamp_delta.clone(),
                write_aux_cols,
            )
            .eval(builder, is_valid);

        timestamp_delta += delta_per_access.into();

        self.execution_bus.execute_increment_pc(
            builder,
            aux.is_valid,
            ExecutionState::new(pc, timestamp),
            timestamp_delta,
            InstructionCols::new(expected_opcode, [op_a, op_b, op_c, d, e]),
        );
    }
}
