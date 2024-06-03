// use std::borrow::Borrow;

use afs_stark_backend::interaction::Chip;
use p3_air::{Air, AirBuilder, BaseAir};
// use p3_field::AbstractField;
use p3_field::Field;
// use p3_matrix::Matrix;

use crate::sub_chip::{AirConfig, SubAir};

use super::{
    columns::{DummyHashAuxCols, DummyHashCols, DummyHashIOCols},
    DummyHashChip,
};

impl<F: Field, const N: usize, const R: usize> BaseAir<F> for DummyHashChip<N, R> {
    fn width(&self) -> usize {
        N + R
    }
}

impl<const N: usize, const R: usize> AirConfig for DummyHashChip<N, R> {
    type Cols<T> = DummyHashCols<T, N, R>;
}

// No interactions
impl<F: Field, const N: usize, const R: usize> Chip<F> for DummyHashChip<N, R> {}

impl<AB: AirBuilder, const N: usize, const R: usize> Air<AB> for DummyHashChip<N, R> {
    fn eval(&self, _builder: &mut AB) {}
}

impl<AB: AirBuilder, const N: usize, const R: usize> SubAir<AB> for DummyHashChip<N, R> {
    type IoView = DummyHashIOCols<AB::Var, N, R>;
    type AuxView = DummyHashAuxCols;

    fn eval(&self, _builder: &mut AB, _io: Self::IoView, _aux: Self::AuxView) {}
}
