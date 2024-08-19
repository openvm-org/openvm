use afs_stark_backend::interaction::InteractionBuilder;

use super::{columns::FieldArithmeticIoCols, FieldArithmeticAir};
use crate::cpu::ARITHMETIC_BUS;

/// Receives all IO columns from another chip.
impl FieldArithmeticAir {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        io: FieldArithmeticIoCols<AB::Var>,
    ) {
        builder.push_receive(ARITHMETIC_BUS, [io.opcode, io.x, io.y, io.z], io.rcv_count);
    }
}
