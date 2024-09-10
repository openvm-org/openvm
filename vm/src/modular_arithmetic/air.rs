use std::borrow::Borrow;

use afs_primitives::{
    bigint::modular_arithmetic::ModularArithmeticCols as PrimitiveArithmeticCols, sub_chip::SubAir,
};
use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::{columns::ModularArithmeticCols, ModularArithmeticAir};
use crate::arch::instructions::Opcode;

impl<F: Field> BaseAir<F> for ModularArithmeticAir {
    fn width(&self) -> usize {
        todo!()
    }
}

impl<AB: InteractionBuilder> Air<AB> for ModularArithmeticAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();
        let ModularArithmeticCols { io, aux } =
            ModularArithmeticCols::<AB::Var>::from_iterator(local.iter().copied(), self);

        let cols = PrimitiveArithmeticCols {
            x: io.x.data.clone(),
            y: io.y.data.clone(),
            r: io.z.data.clone(),
            q: aux.q.clone(),
            carries: aux.carries.clone(),
        };
        SubAir::eval(&self.air, builder, cols, ());

        let expected_opcode = AB::Expr::from_canonical_u8(Opcode::SECP256K1_COORD_ADD as u8);
        self.eval_interactions(builder, io, aux, expected_opcode);
    }
}
