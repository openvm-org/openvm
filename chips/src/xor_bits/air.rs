use std::{borrow::Borrow, iter::zip};

use itertools::Itertools;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use crate::sub_chip::{AirConfig, SubAir};

use super::{
    columns::{XorBitCols, XorCols, XorIOCols},
    XorBitsAir,
};

impl<const N: usize> AirConfig for XorBitsAir<N> {
    type Cols<T> = XorCols<N, T>;
}

impl<F: Field, const N: usize> BaseAir<F> for XorBitsAir<N> {
    fn width(&self) -> usize {
        XorCols::<N, F>::get_width()
    }
}

impl<AB: AirBuilder, const N: usize> Air<AB> for XorBitsAir<N> {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();

        let xor_cols = XorCols::<N, AB::Var>::from_slice(local);

        SubAir::eval(self, builder, xor_cols.io, xor_cols.bits);
    }
}

/// Imposes AIR constraints within each row of the trace
/// Constrains x, y, z to be equal to their bit representation in x_bits, y_bits, z_bits.
/// For each x_bit[i], y_bit[i], and z_bit[i], constraints x_bit[i] + y_bit[i] - 2 * x_bit[i] * y_bit[i] == z_bit[i],
/// which is equivalent to ensuring that x_bit[i] ^ y_bit[i] == z_bit[i].
/// Overall, this ensures that x^y == z.
impl<const N: usize, AB: AirBuilder> SubAir<AB> for XorBitsAir<N> {
    type IoView = XorIOCols<AB::Var>;
    type AuxView = XorBitCols<AB::Var>;

    fn eval(&self, builder: &mut AB, io: Self::IoView, bits: Self::AuxView) {
        for (x, bit_decomp) in zip([io.x, io.y, io.z], [&bits.x, &bits.y, &bits.z]) {
            let mut from_bits = AB::Expr::zero();
            for (i, &bit) in bit_decomp.iter().enumerate() {
                from_bits += bit * AB::Expr::from_canonical_u32(1 << i);
            }
            builder.assert_eq(from_bits, x);
        }

        for ((x, y), z) in bits.x.into_iter().zip_eq(bits.y).zip_eq(bits.z) {
            builder.assert_eq(x + y - AB::Expr::two() * x * y, z);
        }
    }
}
