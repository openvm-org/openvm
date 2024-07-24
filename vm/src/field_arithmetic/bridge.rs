use afs_stark_backend::interaction::InteractionBuilder;

use super::{columns::FieldArithmeticIoCols, FieldArithmeticAir};

/// Receives all IO columns from another chip on bus 2 (FieldArithmeticAir::BUS_INDEX).
impl FieldArithmeticAir {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        io: FieldArithmeticIoCols<AB::Var>,
    ) {
        builder.push_receive(Self::BUS_INDEX, [io.opcode, io.x, io.y, io.z], io.rcv_count);
    }
}
