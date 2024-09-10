use std::borrow::Borrow;

use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use super::{columns::ModularArithmeticCols, ModularArithmeticAir};

impl<F: Field> BaseAir<F> for ModularArithmeticAir {
    fn width(&self) -> usize {
        todo!()
    }
}

impl<AB: InteractionBuilder> Air<AB> for ModularArithmeticAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();
        let ModularArithmeticCols { io, aux } =
            ModularArithmeticCols::<AB::Var>::from_iterator(local.iter().copied(), self);

        // FIXME: rn assume op is add

        // self.eval_interactions(builder, io, aux);
    }
}
