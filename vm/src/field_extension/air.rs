use std::borrow::Borrow;

use afs_primitives::{
    sub_chip::AirConfig,
    utils::{and, not},
};
use afs_stark_backend::interaction::InteractionBuilder;
use itertools::izip;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::{columns::FieldExtensionArithmeticCols, FieldExtensionArithmeticAir};
use crate::{
    cpu::OpCode::{BBE4INV, BBE4MUL, FE4ADD, FE4SUB},
    field_extension::BETA,
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

        let local = main.row_slice(0);
        let local_cols: &FieldExtensionArithmeticCols<AB::Var> = (*local).borrow();

        let FieldExtensionArithmeticCols { io, aux } = local_cols;

        let flags = [aux.is_add, aux.is_sub, aux.is_mul, aux.is_inv];
        let opcodes = [FE4ADD, FE4SUB, BBE4MUL, BBE4INV];

        let mut flag_sum = AB::Expr::zero();
        let mut expected_opcode = AB::Expr::zero();
        for (flag, opcode) in izip!(flags, opcodes) {
            builder.assert_bool(flag);

            flag_sum += flag.into();
            expected_opcode += flag * AB::F::from_canonical_u32(opcode as u32);
        }
        builder.assert_one(flag_sum);
        builder.assert_eq(io.opcode, expected_opcode);

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

        // constrain inverse using multiplication: x * x^(-1) = 1
        // ignores when not inv compute (will fail if x = 0 and try to compute inv)
        let x_times_x_inv = multiply::<AB>(io.x, aux.inv);
        for (i, prod_i) in x_times_x_inv.into_iter().enumerate() {
            if i == 0 {
                builder.when(aux.is_inv).assert_one(prod_i);
            } else {
                builder.assert_zero(prod_i);
            }
        }

        let expected_product = multiply::<AB>(io.x, io.y);

        // constrain that the overall output is correct
        for (x_i, y_i, z_i, product_i, inv_i) in izip!(io.x, io.y, io.z, expected_product, aux.inv)
        {
            builder.assert_eq(
                z_i,
                aux.is_add * (x_i + y_i)
                    + aux.is_sub * (x_i - y_i)
                    + aux.is_mul * product_i
                    + aux.is_inv * inv_i,
            );
        }

        self.eval_interactions(builder, local_cols);
    }
}

fn multiply<AB: AirBuilder>(x: [AB::Var; 4], y: [AB::Var; 4]) -> [AB::Expr; 4] {
    let beta_f = AB::F::from_canonical_usize(BETA);
    let [x0, x1, x2, x3] = x;
    let [y0, y1, y2, y3] = y;
    [
        x0 * y0 + (x1 * y3 + x2 * y2 + x3 * y1) * beta_f,
        x0 * y1 + x1 * y0 + (x2 * y3 + x3 * y2) * beta_f,
        x0 * y2 + x1 * y1 + x2 * y0 + (x3 * y3) * beta_f,
        x0 * y3 + x1 * y2 + x2 * y1 + x3 * y0,
    ]
}
