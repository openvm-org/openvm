use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use super::{
    air::UiAir,
    columns::{UiAuxCols, UiIoCols},
};
use crate::{arch::columns::InstructionCols, memory::MemoryAddress};

impl UiAir {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        io: &UiIoCols<AB::Var>,
        aux: &UiAuxCols<AB::Var>,
        expected_opcode: AB::Expr,
    ) {
    }
}
