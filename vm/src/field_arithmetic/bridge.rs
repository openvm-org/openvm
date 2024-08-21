use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use super::{columns::FieldArithmeticIoCols, FieldArithmeticAir};
use crate::{arch::columns::InstructionCols, cpu::MEMORY_BUS};

impl FieldArithmeticAir {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        io: FieldArithmeticIoCols<AB::Var>,
    ) {
        self.execution_bus.execute_increment_pc(
            builder,
            io.rcv_count,
            io.prev_state.map(Into::into),
            AB::F::from_canonical_usize(3),
            InstructionCols::new(
                io.opcode,
                io.z_address,
                io.x_address,
                io.y_address,
                io.xz_as,
                io.y_as,
            ),
        );

        let start_timestamp = io.prev_state.timestamp;
        let mut memory_interaction = |timestamp_increment: usize,
                                      is_write: bool,
                                      address_space: AB::Var,
                                      address: AB::Var,
                                      value: AB::Var| {
            builder.push_send(
                MEMORY_BUS,
                [
                    start_timestamp + AB::F::from_canonical_usize(timestamp_increment),
                    AB::Expr::from_bool(is_write),
                    address_space.into(),
                    address.into(),
                    value.into(),
                ],
                io.rcv_count,
            );
        };
        memory_interaction(0, false, io.xz_as, io.x_address, io.x);
        memory_interaction(1, false, io.y_as, io.y_address, io.y);
        memory_interaction(2, true, io.xz_as, io.z_address, io.z);
    }
}
