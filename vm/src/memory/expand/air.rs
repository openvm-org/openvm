use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
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

        // `expand_direction` should be -1, 0, 1
        builder.assert_eq(
            local.expand_direction,
            local.expand_direction * local.expand_direction * local.expand_direction,
        );

        builder.assert_bool(local.left_direction_different);
        builder.assert_bool(local.right_direction_different);

        // if `expand_direction` != -1, then `*_direction_different` should be 0
        builder
            .when_ne(local.expand_direction, AB::F::neg_one())
            .assert_zero(local.left_direction_different);
        builder
            .when_ne(local.expand_direction, AB::F::neg_one())
            .assert_zero(local.right_direction_different);

        self.eval_interactions(builder, local);
    }
}
