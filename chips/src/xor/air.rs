use std::borrow::Borrow;
use std::fmt::Debug;

use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::columns::XorCols;
use super::XorChip;

impl<F: Field, const N: usize> BaseAir<F> for XorChip<N> {
    fn width(&self) -> usize {
        self.get_width()
    }
}

impl<AB: AirBuilderWithPublicValues, const N: usize> Air<AB> for XorChip<N>
where
    AB: AirBuilder,
    AB::Var: Clone + Debug,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let _pis = builder.public_values();

        let (local, _next) = (main.row_slice(0), main.row_slice(1));
        let local: &[AB::Var] = (*local).borrow();

        let xor_cols = XorCols::<N, AB::Var>::from_slice(local);

        let mut x_from_bits: AB::Expr = AB::Expr::zero();
        for i in 0..N {
            x_from_bits += xor_cols.x_bits[i] * AB::Expr::from_canonical_u64(1 << i);
        }
        builder.assert_eq(x_from_bits, xor_cols.helper.x);

        let mut y_from_bits: AB::Expr = AB::Expr::zero();
        for i in 0..N {
            y_from_bits += xor_cols.y_bits[i] * AB::Expr::from_canonical_u64(1 << i);
        }
        builder.assert_eq(y_from_bits, xor_cols.helper.y);

        let mut z_from_bits: AB::Expr = AB::Expr::zero();
        for i in 0..N {
            z_from_bits += xor_cols.z_bits[i] * AB::Expr::from_canonical_u64(1 << i);
        }
        builder.assert_eq(z_from_bits, xor_cols.helper.z);

        for i in 0..N {
            builder.assert_eq(
                xor_cols.x_bits[i] + xor_cols.y_bits[i]
                    - AB::Expr::two() * xor_cols.x_bits[i] * xor_cols.y_bits[i],
                xor_cols.z_bits[i],
            );
        }
    }
}
