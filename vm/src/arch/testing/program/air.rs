use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use super::ProgramTester;
use crate::program::bridge::ProgramBus;

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct ProgramDummyAir {
    pub bus: ProgramBus,
}

impl<F: Field> BaseAir<F> for ProgramDummyAir {
    fn width(&self) -> usize {
        ProgramTester::<F>::width()
    }
}

impl<AB: InteractionBuilder> Air<AB> for ProgramDummyAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local = local.iter().map(|x| (*x).into()).collect::<Vec<AB::Expr>>();
        self.bus.receive_instruction(
            builder,
            local[..local.len() - 1].iter().cloned(),
            local[local.len() - 1].clone(),
        );
    }
}
