use afs_stark_backend::{interaction::InteractionBuilder, rap::BaseAirWithPublicValues};
use p3_air::{Air, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use super::ProgramTester;
use crate::program::bridge::ProgramBus;

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct ProgramDummyAir {
    pub bus: ProgramBus,
}

impl<F: Field> BaseAirWithPublicValues<F> for ProgramDummyAir {}
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
        builder.push_receive(
            self.bus.0,
            local[..local.len() - 1].iter().cloned(),
            local[local.len() - 1].clone(),
        );
    }
}
