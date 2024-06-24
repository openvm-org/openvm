use std::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use crate::cpu::OpCode;

use super::{columns::FieldExtensionAddSubCols, FieldExtensionAddSubAir};
use afs_chips::sub_chip::AirConfig;

impl AirConfig for FieldExtensionAddSubAir {
    type Cols<T> = FieldExtensionAddSubCols<T>;
}

impl<F: Field> BaseAir<F> for FieldExtensionAddSubAir {
    fn width(&self) -> usize {
        FieldExtensionAddSubCols::<F>::NUM_COLS
    }
}

impl<AB: AirBuilder> Air<AB> for FieldExtensionAddSubAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();
        let local_cols = FieldExtensionAddSubCols::<AB::Var>::from_slice(local);

        // if opcode is FEADD then this is 1, otherwise (FESUB) it is -1
        let op_ind = (AB::Expr::from_canonical_u8(OpCode::FEADD as u8) * AB::Expr::two()
            + AB::Expr::one())
            - AB::Expr::two() * local_cols.io.opcode;

        for i in 0..4 {
            builder.assert_eq(
                local_cols.io.x[i] + op_ind.clone() * local_cols.io.y[i],
                local_cols.io.z[i],
            );
        }
    }
}
