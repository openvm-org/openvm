use std::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use crate::field_extension::BETA;

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

impl<AB: AirBuilder> Air<AB> for FieldExtensionArithmeticAir {
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

        let add_sub_coeff = AB::Expr::one() - AB::Expr::two() * aux.opcode_lo;

        for i in 0..4 {
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

        // constrain that the overall output is correct
        for i in 0..4 {
            builder.assert_eq(
                io.z[i],
                aux.is_mul * aux.product[i]
                    + aux.sum_or_diff[i] * (AB::Expr::one() - aux.opcode_hi),
            );
        }
    }
}
