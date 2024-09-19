use std::iter;

use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::PairBuilder;
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::ProgramAir;

#[derive(Debug, Clone, Copy)]
pub struct ProgramBus(pub usize);

impl ProgramBus {
    pub fn send_instruction<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        instruction: impl Iterator<Item = AB::Expr>,
        multiplicity: impl Into<AB::Expr>,
    ) {
        builder.push_send(
            self.0,
            instruction.chain(iter::repeat(AB::Expr::zero())).take(9),
            multiplicity,
        );
    }

    pub fn receive_instruction<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        instruction: impl Iterator<Item = AB::Expr>,
        multiplicity: impl Into<AB::Expr>,
    ) {
        builder.push_receive(
            self.0,
            instruction.chain(iter::repeat(AB::Expr::zero())).take(9),
            multiplicity,
        );
    }
}

impl<F: Field> ProgramAir<F> {
    pub fn eval_interactions<AB: PairBuilder<F = F> + InteractionBuilder>(&self, builder: &mut AB) {
        let main = builder.main();
        let execution_frequency = main.row_slice(0)[0];
        let preprocessed = &builder.preprocessed();
        let prep_local = preprocessed.row_slice(0);
        let fields = prep_local.iter().map(|&x| x.into());

        self.bus
            .receive_instruction(builder, fields, execution_frequency);
    }
}
