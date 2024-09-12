use std::borrow::Borrow;

use afs_primitives::bigint::modular_arithmetic::ModularArithmeticCols as PrimitiveArithmeticCols;
use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use super::{columns::ModularArithmeticCols, ModularArithmeticAirVariant, ModularArithmeticVmAir};

impl<F: Field> BaseAir<F> for ModularArithmeticVmAir<ModularArithmeticAirVariant> {
    fn width(&self) -> usize {
        ModularArithmeticCols::<F>::width(self)
    }
}

impl<AB: InteractionBuilder> Air<AB> for ModularArithmeticVmAir<ModularArithmeticAirVariant> {
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
            is_valid: aux.is_valid,
        };
        self.air.eval(builder, cols, ());

        self.eval_interactions(builder, io, aux);
    }
}
