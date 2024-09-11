use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use super::{columns::FieldArithmeticIoCols, FieldArithmeticAir};
use crate::{
    arch::columns::InstructionCols, field_arithmetic::columns::FieldArithmeticAuxCols,
    memory::offline_checker::MemoryBridge,
};

/// Receives all IO columns from another chip.
impl FieldArithmeticAir {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        io: FieldArithmeticIoCols<AB::Var>,
        aux: FieldArithmeticAuxCols<AB::Var>,
        expected_opcode: AB::Expr,
    ) {
        let FieldArithmeticIoCols {
            from_state,
            z: result,
            x: operand1,
            y: operand2,
        } = io;
        let is_valid = aux.is_valid;

        self.execution_bus.execute_increment_pc(
            builder,
            is_valid,
            io.from_state.map(Into::into),
            AB::F::from_canonical_usize(Self::TIMESTAMP_DELTA),
            InstructionCols::new(
                expected_opcode,
                [
                    result.address,
                    operand1.address,
                    operand2.address,
                    result.address_space,
                    operand1.address_space,
                    operand2.address_space,
                ],
            ),
        );

        let memory_bridge = MemoryBridge::new(self.mem_oc);
        let mut timestamp: AB::Expr = from_state.timestamp.into();
        memory_bridge
            .read_or_immediate(
                operand1.memory_address(),
                operand1.value,
                timestamp.clone(),
                aux.read_x_aux_cols,
            )
            .eval(builder, is_valid);
        timestamp += is_valid.into();

        memory_bridge
            .read_or_immediate(
                operand2.memory_address(),
                operand2.value,
                timestamp.clone(),
                aux.read_y_aux_cols,
            )
            .eval(builder, is_valid);
        timestamp += is_valid.into();

        memory_bridge
            .write(
                result.memory_address(),
                [result.value],
                timestamp,
                aux.write_z_aux_cols,
            )
            .eval(builder, is_valid);
    }
}
