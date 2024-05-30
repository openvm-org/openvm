use std::borrow::Borrow;

use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use crate::sub_chip::{AirConfig, SubAir};

use super::{columns::XorCols, XorBitsChip};

impl<F: Field, const N: usize> BaseAir<F> for XorBitsChip<N> {
    fn width(&self) -> usize {
        XorCols::<N, F>::get_width()
    }
}

impl<AB: AirBuilderWithPublicValues, const N: usize> Air<AB> for XorBitsChip<N>
where
    AB: AirBuilder,
    AB::Var: Clone,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();

        let xor_cols = XorCols::<N, AB::Var>::from_slice(local);

        SubAir::eval(self, builder, xor_cols);
    }
}

impl<const N: usize> AirConfig for XorBitsChip<N> {
    type Cols<T> = XorCols<N, T>;
}

/// Imposes AIR constraints within each row of the trace
/// Constrains x, y, z to be equal to their bit representation in x_bits, y_bits, z_bits.
/// For each x_bit[i], y_bit[i], and z_bit[i], constraints x_bit[i] + y_bit[i] - 2 * x_bit[i] * y_bit[i] == z_bit[i],
/// which is equivalent to ensuring that x_bit[i] ^ y_bit[i] == z_bit[i].
/// Overall, this ensures that x^y == z.
impl<const N: usize, AB: AirBuilder> SubAir<AB> for XorBitsChip<N> {
    fn eval(&self, builder: &mut AB, xor_cols: Self::Cols<AB::Var>) {
        let mut x_from_bits: AB::Expr = AB::Expr::zero();
        for i in 0..N {
            x_from_bits += xor_cols.x_bits[i] * AB::Expr::from_canonical_u64(1 << i);
        }
        builder.assert_eq(x_from_bits, xor_cols.io.x);

        let mut y_from_bits: AB::Expr = AB::Expr::zero();
        for i in 0..N {
            y_from_bits += xor_cols.y_bits[i] * AB::Expr::from_canonical_u64(1 << i);
        }
        builder.assert_eq(y_from_bits, xor_cols.io.y);

        let mut z_from_bits: AB::Expr = AB::Expr::zero();
        for i in 0..N {
            z_from_bits += xor_cols.z_bits[i] * AB::Expr::from_canonical_u64(1 << i);
        }
        builder.assert_eq(z_from_bits, xor_cols.io.z);

        for i in 0..N {
            builder.assert_eq(
                xor_cols.x_bits[i] + xor_cols.y_bits[i]
                    - AB::Expr::two() * xor_cols.x_bits[i] * xor_cols.y_bits[i],
                xor_cols.z_bits[i],
            );
        }
    }
}
