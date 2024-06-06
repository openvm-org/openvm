use std::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::columns::XorLimbsCols;
use super::XorLimbsAir;

impl<F: Field, const N: usize, const M: usize> BaseAir<F> for XorLimbsAir<N, M> {
    fn width(&self) -> usize {
        XorLimbsCols::<N, M, F>::get_width()
    }
}

impl<AB: AirBuilder, const N: usize, const M: usize> Air<AB> for XorLimbsAir<N, M> {
    fn eval(&self, builder: &mut AB) {
        let num_limbs = (N + M - 1) / M;

        let main = builder.main();

        let (local, _next) = (main.row_slice(0), main.row_slice(1));
        let local: &[AB::Var] = (*local).borrow();

        let xor_cols = XorLimbsCols::<N, M, AB::Var>::from_slice(local);

        let mut x_from_limbs: AB::Expr = AB::Expr::zero();
        for i in 0..num_limbs {
            x_from_limbs += xor_cols.x_limbs[i] * AB::Expr::from_canonical_u64(1 << (i * M));
        }
        builder.assert_eq(x_from_limbs, xor_cols.x);

        let mut y_from_limbs: AB::Expr = AB::Expr::zero();
        for i in 0..num_limbs {
            y_from_limbs += xor_cols.y_limbs[i] * AB::Expr::from_canonical_u64(1 << (i * M));
        }
        builder.assert_eq(y_from_limbs, xor_cols.y);

        let mut z_from_limbs: AB::Expr = AB::Expr::zero();
        for i in 0..num_limbs {
            z_from_limbs += xor_cols.z_limbs[i] * AB::Expr::from_canonical_u64(1 << (i * M));
        }
        builder.assert_eq(z_from_limbs, xor_cols.z);
    }
}
