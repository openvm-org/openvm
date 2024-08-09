use std::borrow::Borrow;

use afs_primitives::{
    is_less_than::columns::IsLessThanCols,
    sub_chip::{AirConfig, SubAir},
};
use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use super::{columns::IsLessThanVmCols, IsLessThanVmAir};

impl AirConfig for IsLessThanVmAir {
    type Cols<T> = IsLessThanVmCols<T>;
}

impl<F: Field> BaseAir<F> for IsLessThanVmAir {
    fn width(&self) -> usize {
        IsLessThanCols::<F>::width(&self.inner) + 1
    }
}

impl<AB: InteractionBuilder> Air<AB> for IsLessThanVmAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &[<AB>::Var] = (*local).borrow();
        let cols = IsLessThanVmCols::<AB::Var>::from_slice(local);

        SubAir::eval(
            &self.inner,
            builder,
            cols.internal.io.clone(),
            cols.internal.aux.clone(),
        );

        self.eval_interactions(builder, cols);
    }
}
