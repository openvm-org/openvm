use std::borrow::Borrow;

use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::{columns::FieldArithmeticCols, FieldArithmeticAir};
use afs_chips::sub_chip::AirConfig;

impl AirConfig for FieldArithmeticAir {
    type Cols<T> = FieldArithmeticCols<T>;
}

impl<F: Field> BaseAir<F> for FieldArithmeticAir {
    fn width(&self) -> usize {
        FieldArithmeticCols::<F>::NUM_COLS
    }
}

impl<AB: InteractionBuilder> Air<AB> for FieldArithmeticAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let au_cols: &FieldArithmeticCols<_> = (*local).borrow();

        let FieldArithmeticCols { io, aux } = au_cols;

        builder.assert_bool(aux.opcode_lo);
        builder.assert_bool(aux.opcode_hi);

        builder.assert_eq(
            io.opcode,
            aux.opcode_lo
                + aux.opcode_hi * AB::Expr::two()
                + AB::F::from_canonical_u8(FieldArithmeticAir::BASE_OP),
        );

        builder.assert_eq(
            aux.is_mul,
            aux.opcode_hi * (AB::Expr::one() - aux.opcode_lo),
        );
        builder.assert_eq(aux.is_div, aux.opcode_hi * aux.opcode_lo);

        builder.assert_eq(aux.product, io.x * io.y);
        builder.assert_eq(aux.quotient * io.y, io.x * aux.is_div);
        builder.assert_eq(
            au_cols.aux.sum_or_diff,
            io.x + io.y - AB::Expr::two() * aux.opcode_lo * io.y,
        );

        builder.assert_eq(
            io.z,
            aux.is_mul * aux.product
                + aux.is_div * aux.quotient
                + aux.sum_or_diff * (AB::Expr::one() - aux.opcode_hi),
        );

        builder.assert_eq(aux.divisor_inv * io.y, aux.is_div);

        self.eval_interactions(builder, *io);
    }
}
