use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use crate::memory::expand::columns::ExpandCols;

pub struct ExpandAir<const CHUNK: usize> {}

impl<const CHUNK: usize, F: Field> BaseAir<F> for ExpandAir<CHUNK> {
    fn width(&self) -> usize {
        ExpandCols::<CHUNK, F>::get_width()
    }
}

impl<const CHUNK: usize, AB: InteractionBuilder> Air<AB> for ExpandAir<CHUNK> {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local = ExpandCols::<CHUNK, AB::Var>::from_slice(&local);

        // `direction` should be -1, 0, 1
        builder.assert_eq(
            local.direction,
            local.direction * local.direction * local.direction,
        );

        builder.assert_bool(local.left_is_final);
        builder.assert_bool(local.right_is_final);

        self.eval_interactions(builder, local);
    }
}
