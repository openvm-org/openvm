use std::borrow::Borrow;

use afs_primitives::{
    sub_chip::AirConfig,
    utils::{and, not},
};
use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::{columns::FieldExtensionArithmeticCols, FieldExtensionArithmeticAir};
use crate::{
    cpu::OpCode::{BBE4INV, BBE4MUL, FE4ADD, FE4SUB},
    field_extension::{BETA, EXTENSION_DEGREE},
};

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

        builder.assert_bool(aux.is_add);
        builder.assert_bool(aux.is_sub);
        builder.assert_bool(aux.is_mul);
        builder.assert_bool(aux.is_inv);

        let mut indicator_sum = AB::Expr::zero();
        indicator_sum += aux.is_add.into();
        indicator_sum += aux.is_sub.into();
        indicator_sum += aux.is_mul.into();
        indicator_sum += aux.is_inv.into();
        builder.assert_one(indicator_sum);

        builder.assert_eq(
            io.opcode,
            aux.is_add * AB::F::from_canonical_u32(FE4ADD as u32)
                + aux.is_sub * AB::F::from_canonical_u32(FE4SUB as u32)
                + aux.is_mul * AB::F::from_canonical_u32(BBE4MUL as u32)
                + aux.is_inv * AB::F::from_canonical_u32(BBE4INV as u32),
        );

        builder.assert_bool(aux.is_valid);
        // valid_y_read is 1 iff is_valid and not is_inv
        // the previous constraint along with this one imply valid_y_read is boolean
        builder.assert_eq(
            aux.valid_y_read,
            and(aux.is_valid.into(), not(aux.is_inv.into())),
        );

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
                aux.is_add * (io.x[i] + io.y[i])
                    + aux.is_sub * (io.x[i] - io.y[i])
                    + aux.is_mul * aux.product[i]
                    + aux.is_inv * aux.inv[i],
            );
        }

        self.eval_interactions(builder, local_cols);
    }
}
