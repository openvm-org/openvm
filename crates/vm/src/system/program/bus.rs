use std::iter;

use ax_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

#[derive(Debug, Clone, Copy)]
pub struct ProgramBus(pub usize);

impl ProgramBus {
    pub fn send_instruction<AB: InteractionBuilder, E: Into<AB::Expr>>(
        &self,
        builder: &mut AB,
        pc: impl Into<AB::Expr>,
        opcode: impl Into<AB::Expr>,
        operands: impl IntoIterator<Item = E>,
        multiplicity: impl Into<AB::Expr>,
    ) {
        builder.push_send(
            self.0,
            [pc.into(), opcode.into()].into_iter().chain(
                operands
                    .into_iter()
                    .map(Into::into)
                    .chain(iter::repeat(AB::Expr::ZERO))
                    .take(7),
            ),
            multiplicity,
        );
    }
}
