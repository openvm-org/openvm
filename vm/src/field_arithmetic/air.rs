use std::borrow::Borrow;

use afs_primitives::sub_chip::AirConfig;
use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::{columns::FieldArithmeticCols, FieldArithmeticAir};
use crate::cpu::OpCode::{FADD, FDIV, FMUL, FSUB};

impl AirConfig for FieldArithmeticAir {
    type Cols<T> = FieldArithmeticCols<T>;
}

impl<F: Field> BaseAir<F> for FieldArithmeticAir {
    fn width(&self) -> usize {
        FieldArithmeticCols::<F>::get_width()
    }
}

impl<AB: InteractionBuilder> Air<AB> for FieldArithmeticAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &FieldArithmeticCols<_> = (*local).borrow();

        let FieldArithmeticCols { io, aux } = local;

        let mut indicator_sum = AB::Expr::zero();
        indicator_sum += aux.is_sub.into();
        indicator_sum += aux.is_mul.into();
        indicator_sum += aux.is_div.into();
        builder.assert_bool(indicator_sum.clone());

        let is_add = AB::Expr::one() - indicator_sum;

        builder.assert_eq(
            io.opcode,
            is_add.clone() * AB::F::from_canonical_u32(FADD as u32)
                + aux.is_sub * AB::F::from_canonical_u32(FSUB as u32)
                + aux.is_mul * AB::F::from_canonical_u32(FMUL as u32)
                + aux.is_div * AB::F::from_canonical_u32(FDIV as u32),
        );

        builder.when(is_add).assert_eq(io.z, io.x + io.y);
        builder.when(aux.is_sub).assert_eq(io.z, io.x - io.y);
        builder.when(aux.is_mul).assert_eq(io.z, io.x * io.y); // deg 3

        builder.assert_eq(io.y * aux.divisor_inv, aux.is_div);
        builder.assert_eq(io.z * aux.is_div, io.x * aux.divisor_inv);

        self.eval_interactions(builder, *io);
    }
}
