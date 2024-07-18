use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use crate::sub_chip::{AirConfig, SubAir};

use super::{
    columns::{DummyHashAuxCols, DummyHashCols, DummyHashIoCols},
    DummyHashAir,
};

impl<F: Field> BaseAir<F> for DummyHashAir {
    fn width(&self) -> usize {
        2 * self.hash_width + self.rate + 1
    }
}

impl AirConfig for DummyHashAir {
    type Cols<T> = DummyHashCols<T>;
}

impl<AB: AirBuilder> Air<AB> for DummyHashAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let dummy_hash_cols: &DummyHashCols<_> =
            &DummyHashCols::from_slice(local.as_ref(), self.hash_width, self.rate);

        SubAir::<AB>::eval(
            self,
            builder,
            dummy_hash_cols.io.clone(),
            dummy_hash_cols.aux,
        );
    }
}

impl<AB: AirBuilder> SubAir<AB> for DummyHashAir {
    type IoView = DummyHashIoCols<AB::Var>;
    type AuxView = DummyHashAuxCols;

    fn eval(&self, builder: &mut AB, io: Self::IoView, _aux: Self::AuxView) {
        for i in 0..self.rate {
            builder.assert_eq(io.curr_state[i] + io.to_absorb[i], io.new_state[i]);
        }
        for i in self.rate..self.hash_width {
            builder.assert_eq(io.curr_state[i], io.new_state[i]);
        }
    }
}
