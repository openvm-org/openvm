use std::borrow::Borrow;

use afs_stark_backend::interaction::Chip;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use crate::sub_chip::{AirConfig, SubAir};

use super::{
    columns::{DummyHashAuxCols, DummyHashCols, DummyHashIOCols},
    DummyHashChip,
};

impl<F: Field, const N: usize, const R: usize> BaseAir<F> for DummyHashChip<N, R> {
    fn width(&self) -> usize {
        2 * N + R
    }
}

impl<const N: usize, const R: usize> AirConfig for DummyHashChip<N, R> {
    type Cols<T> = DummyHashCols<T, N, R>;
}

// No interactions
impl<F: Field, const N: usize, const R: usize> Chip<F> for DummyHashChip<N, R> {}

impl<AB: AirBuilder, const N: usize, const R: usize> Air<AB> for DummyHashChip<N, R> {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let dummy_hash_cols: &DummyHashCols<_, N, R> = &DummyHashCols::from_slice(local.as_ref());

        SubAir::<AB>::eval(
            self,
            builder,
            dummy_hash_cols.io.clone(),
            dummy_hash_cols.aux,
        );
    }
}

impl<AB: AirBuilder, const N: usize, const R: usize> SubAir<AB> for DummyHashChip<N, R> {
    type IoView = DummyHashIOCols<AB::Var, N, R>;
    type AuxView = DummyHashAuxCols;

    fn eval(&self, builder: &mut AB, io: Self::IoView, _aux: Self::AuxView) {
        for i in 0..R {
            builder.assert_eq(io.curr_state[i] + io.to_absorb[i], io.new_state[i]);
        }
        for i in R..N {
            builder.assert_eq(io.curr_state[i], io.new_state[i]);
        }
    }
}
