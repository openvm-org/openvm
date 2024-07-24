use std::borrow::Borrow;

use afs_stark_backend::interaction::InteractionBuilder;
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
        FieldExtensionArithmeticCols::<F>::get_width()
    }
}

impl<AB: InteractionBuilder> Air<AB> for FieldExtensionArithmeticAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let beta_f = AB::Expr::from_canonical_usize(BETA);

        let local = main.row_slice(0);
        let local_cols: &FieldExtensionArithmeticCols<AB::Var> = (*local).borrow();

        let FieldExtensionArithmeticCols { io, aux } = local_cols;

        builder.assert_bool(aux.is_valid);
        // valid_y_read is 1 iff is_valid and not is_inv
        // the previous constraint along with this one imply valid_y_read is boolean
        builder.assert_eq(
            aux.valid_y_read,
            aux.is_valid * (AB::Expr::one() - aux.is_inv),
        );

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
                + beta_f.clone() * (io.x[1] * io.y[3] + io.x[2] * io.y[2] + io.x[3] * io.y[1]),
            aux.product[0],
        );
        builder.assert_eq(
            io.x[0] * io.y[1]
                + io.x[1] * io.y[0]
                + beta_f.clone() * (io.x[2] * io.y[3] + io.x[3] * io.y[2]),
            aux.product[1],
        );
        builder.assert_eq(
            io.x[0] * io.y[2]
                + io.x[1] * io.y[1]
                + io.x[2] * io.y[0]
                + beta_f.clone() * (io.x[3] * io.y[3]),
            aux.product[2],
        );
        builder.assert_eq(
            io.x[0] * io.y[3] + io.x[1] * io.y[2] + io.x[2] * io.y[1] + io.x[3] * io.y[0],
            aux.product[3],
        );

        // constrain inverse using multiplication: x * x^(-1) = 1
        // ignores when not inv compute (will fail if x = 0 and try to compute inv)
        builder.when(aux.is_inv).assert_one(
            io.x[0] * aux.inv[0]
                + beta_f.clone()
                    * (io.x[1] * aux.inv[3] + io.x[2] * aux.inv[2] + io.x[3] * aux.inv[1]),
        );
        builder.assert_zero(
            io.x[0] * aux.inv[1]
                + io.x[1] * aux.inv[0]
                + beta_f.clone() * (io.x[2] * aux.inv[3] + io.x[3] * aux.inv[2]),
        );
        builder.assert_zero(
            io.x[0] * aux.inv[2]
                + io.x[1] * aux.inv[1]
                + io.x[2] * aux.inv[0]
                + beta_f.clone() * (io.x[3] * aux.inv[3]),
        );
        builder.assert_zero(
            io.x[0] * aux.inv[3]
                + io.x[1] * aux.inv[2]
                + io.x[2] * aux.inv[1]
                + io.x[3] * aux.inv[0],
        );

        // constrain that the overall output is correct
        for i in 0..EXTENSION_DEGREE {
            builder.assert_eq(
                io.z[i],
                aux.product[i] * aux.is_mul
                    + aux.sum_or_diff[i] * (AB::Expr::one() - aux.opcode_hi)
                    + aux.inv[i] * aux.is_inv,
            );
        }

        self.eval_interactions(builder, local_cols);
    }
}
