use std::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use crate::field_extension::{BETA, EXTENSION_DEGREE};

use super::{columns::FieldExtensionArithmeticCols, FieldExtensionArithmeticAir};
use afs_chips::sub_chip::AirConfig;

impl AirConfig for FieldExtensionArithmeticAir {
    type Cols<T> = FieldExtensionArithmeticCols<T>;
}

impl<F: Field> BaseAir<F> for FieldExtensionArithmeticAir {
    fn width(&self) -> usize {
        FieldExtensionArithmeticCols::<F>::NUM_COLS
    }
}

impl<AB: AirBuilder> Air<AB> for FieldExtensionArithmeticAir
where
    AB::Var: std::fmt::Debug,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();
        let local_cols = FieldExtensionArithmeticCols::<AB::Var>::from_slice(local);

        let FieldExtensionArithmeticCols { io, aux } = local_cols;

        builder.assert_bool(aux.opcode_lo);
        builder.assert_bool(aux.opcode_hi);

        builder.assert_eq(
            io.opcode,
            aux.opcode_lo
                + aux.opcode_hi * AB::Expr::two()
                + AB::F::from_canonical_u8(FieldExtensionArithmeticAir::BASE_OP),
        );

        builder.assert_eq(
            aux.is_mul,
            aux.opcode_hi * (AB::Expr::one() - aux.opcode_lo),
        );

        builder.assert_eq(aux.is_inv, aux.opcode_hi * aux.opcode_lo);

        let add_sub_coeff = AB::Expr::one() - AB::Expr::two() * aux.opcode_lo;

        for i in 0..EXTENSION_DEGREE {
            builder.assert_eq(
                io.x[i] + add_sub_coeff.clone() * io.y[i],
                aux.sum_or_diff[i],
            );
        }

        // constrain multiplication
        builder.assert_eq(
            io.x[0] * io.y[0]
                + AB::Expr::from_canonical_usize(BETA)
                    * (io.x[1] * io.y[3] + io.x[2] * io.y[2] + io.x[3] * io.y[1]),
            aux.product[0],
        );
        builder.assert_eq(
            io.x[0] * io.y[1]
                + io.x[1] * io.y[0]
                + AB::Expr::from_canonical_usize(BETA) * (io.x[2] * io.y[3] + io.x[3] * io.y[2]),
            aux.product[1],
        );
        builder.assert_eq(
            io.x[0] * io.y[2]
                + io.x[1] * io.y[1]
                + io.x[2] * io.y[0]
                + AB::Expr::from_canonical_usize(BETA) * (io.x[3] * io.y[3]),
            aux.product[2],
        );
        builder.assert_eq(
            io.x[0] * io.y[3] + io.x[1] * io.y[2] + io.x[2] * io.y[1] + io.x[3] * io.y[0],
            aux.product[3],
        );

        // Let x be the vector we are taking the inverse of ([x[0], x[1], x[2], x[3]]), and define
        // x' = [x[0], -x[1], x[2], -x[3]]. We want to compute 1 / x = x' / (x * x'). Let the
        // denominator x * x' = y. By construction, y will have the degree 1 and degree 3 coefficients
        // equal to 0. Let the degree 0 coefficient be b0 and the degree 2 coefficient be b2. Now,
        // define y' as y but with the b2 negated. Note that y * y' = b0^2 - 11 * b2^2, which is an
        // element of the original field, which we can call c. We can invert c as usual and find that
        // 1 / x = x' / (x * x') = x' * y' / c = x' * y' * c^(-1). We multiply out as usual to obtain
        // the answer.
        let mut b0 = io.x[0] * io.x[0]
            - AB::Expr::from_canonical_usize(BETA)
                * (AB::Expr::two() * io.x[1] * io.x[3] - io.x[2] * io.x[2]);
        let mut b2 = AB::Expr::two() * io.x[0] * io.x[2]
            - io.x[1] * io.x[1]
            - AB::Expr::from_canonical_usize(BETA) * io.x[3] * io.x[3];

        let c = b0.clone() * b0.clone()
            - AB::Expr::from_canonical_usize(BETA) * b2.clone() * b2.clone();
        builder.assert_one(c * aux.inv_c);

        b0 *= aux.inv_c.into();
        b2 *= aux.inv_c.into();

        builder.assert_eq(
            io.x[0] * b0.clone() - AB::Expr::from_canonical_usize(BETA) * io.x[2] * b2.clone(),
            aux.inv[0],
        );
        builder.assert_eq(
            AB::Expr::from_canonical_usize(BETA) * io.x[3] * b2.clone() - io.x[1] * b0.clone(),
            aux.inv[1],
        );
        builder.assert_eq(io.x[2] * b0.clone() - io.x[0] * b2.clone(), aux.inv[2]);
        builder.assert_eq(io.x[1] * b2.clone() - io.x[3] * b0.clone(), aux.inv[3]);

        // constrain that the overall output is correct
        for i in 0..EXTENSION_DEGREE {
            builder.assert_eq(
                io.z[i],
                aux.product[i] * aux.is_mul
                    + aux.sum_or_diff[i] * (AB::Expr::one() - aux.opcode_hi)
                    + aux.inv[i] * aux.is_inv,
            );
        }
    }
}
