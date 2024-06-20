use std::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir};
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

impl<AB: AirBuilder> Air<AB> for FieldArithmeticAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let au_cols: &FieldArithmeticCols<_> = (*local).borrow();

        let FieldArithmeticCols { io, aux } = au_cols;

        builder.assert_bool(aux.opcode_0bit);
        builder.assert_bool(aux.opcode_1bit);

        builder.assert_eq(
            io.opcode,
            aux.opcode_0bit
                + aux.opcode_1bit * AB::Expr::two()
                + AB::F::from_canonical_u8(FieldArithmeticAir::BASE_OP),
        );

        builder.assert_eq(
            aux.is_mul,
            aux.opcode_1bit * (AB::Expr::one() - aux.opcode_0bit),
        );
        builder.assert_eq(aux.is_div, aux.opcode_1bit * aux.opcode_0bit);

        builder.assert_eq(aux.mul_result, io.x * io.y);
        builder.assert_eq(aux.div_result * io.y, io.x);
        builder.assert_eq(
            au_cols.aux.lin_term,
            io.x + io.y - AB::Expr::two() * aux.opcode_0bit * io.y,
        );

        builder.assert_eq(
            io.z,
            aux.is_mul * aux.mul_result
                + aux.is_div * aux.div_result
                + aux.lin_term * (AB::Expr::one() - aux.opcode_1bit),
        );
    }
}
