use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use super::{columns::FieldArithmeticIoCols, FieldArithmeticAir};
use crate::kernels::field_arithmetic::columns::FieldArithmeticAuxCols;

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

        self.execution_bridge
            .execute_and_increment_pc(
                expected_opcode + AB::Expr::from_canonical_usize(self.offset),
                [
                    result.address,
                    operand1.address,
                    operand2.address,
                    result.address_space,
                    operand1.address_space,
                    operand2.address_space,
                ],
                from_state,
                AB::F::from_canonical_u32(Self::TIMESTAMP_DELTA),
            )
            .eval(builder, is_valid);

        let mut timestamp: AB::Expr = from_state.timestamp.into();
        self.memory_bridge
            .read_or_immediate(
                operand1.memory_address(),
                operand1.value,
                timestamp.clone(),
                &aux.read_x_aux_cols,
            )
            .eval(builder, is_valid);
        timestamp += is_valid.into();

        self.memory_bridge
            .read_or_immediate(
                operand2.memory_address(),
                operand2.value,
                timestamp.clone(),
                &aux.read_y_aux_cols,
            )
            .eval(builder, is_valid);
        timestamp += is_valid.into();

        self.memory_bridge
            .write(
                result.memory_address(),
                [result.value],
                timestamp,
                &aux.write_z_aux_cols,
            )
            .eval(builder, is_valid);
    }
}
