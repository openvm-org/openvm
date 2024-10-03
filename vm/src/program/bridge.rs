use std::iter;

use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::PairBuilder;
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::ProgramAir;

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
                    .chain(iter::repeat(AB::Expr::zero()))
                    .take(7),
            ),
            multiplicity,
        );
    }
}

impl<F: Field> ProgramAir<F> {
    pub fn eval_interactions<AB: PairBuilder<F = F> + InteractionBuilder>(&self, builder: &mut AB) {
        let main = builder.main();
        let execution_frequency = main.row_slice(0)[0];
        let preprocessed = &builder.preprocessed();
        let prep_local: &[AB::Var] = &preprocessed.row_slice(0);

        builder.push_receive(self.bus.0, prep_local.iter().cloned(), execution_frequency);
    }
}
