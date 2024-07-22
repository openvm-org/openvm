use super::columns::BitDecomposeCols;
use crate::{
    sub_chip::{AirConfig, SubAir},
    utils::Word32,
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;
use std::borrow::Borrow;

pub struct BitDecomposeAir<const N: usize> {
    // TOOD: do we need this?
    bus_index: usize,
}

impl<const N: usize> BitDecomposeAir<N> {
    pub fn new(bus_index: usize) -> Self {
        Self { bus_index }
    }
}

impl<const N: usize> AirConfig for BitDecomposeAir<N> {
    type Cols<T> = BitDecomposeCols<N, T>;
}

impl<F: Field, const N: usize> BaseAir<F> for BitDecomposeAir<N> {
    fn width(&self) -> usize {
        N + 1
    }
}

impl<AB: AirBuilder, const N: usize> Air<AB> for BitDecomposeAir<N> {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();
        let cols = BitDecomposeCols::<N, AB::Var>::from_slice(local);

        SubAir::eval(self, builder, cols, ());
    }
}

impl<const N: usize, AB: AirBuilder> SubAir<AB> for BitDecomposeAir<N> {
    type IoView = BitDecomposeCols<N, AB::Var>;
    type AuxView = ();

    fn eval(&self, builder: &mut AB, io: Self::IoView, _aux: Self::AuxView) {
        let mut from_bits = AB::Expr::zero();
        for (i, &bit) in io.x_bits.iter().enumerate() {
            from_bits += bit * AB::Expr::from_canonical_u32(1 << i);
        }
        // How to make this a function of Word32?
        let from_x = io.x.0[0] * AB::Expr::from_canonical_u32(1 << 16) + io.x.0[1];
        builder.assert_eq(from_bits, from_x);
    }
}
