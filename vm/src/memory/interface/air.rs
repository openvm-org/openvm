use p3_air::{Air, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use afs_stark_backend::interaction::InteractionBuilder;

use crate::memory::interface::columns::MemoryInterfaceCols;

pub struct MemoryInterfaceAir<const CHUNK: usize> {}

impl<const CHUNK: usize, F: Field> BaseAir<F> for MemoryInterfaceAir<CHUNK> {
    fn width(&self) -> usize {
        MemoryInterfaceCols::<CHUNK, F>::get_width()
    }
}

impl<const CHUNK: usize, AB: InteractionBuilder> Air<AB> for MemoryInterfaceAir<CHUNK> {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local = MemoryInterfaceCols::<CHUNK, AB::Var>::from_slice(&local);

        // `direction` should be -1, 0, 1
        builder.assert_eq(
            local.expand_direction,
            local.expand_direction * local.expand_direction * local.expand_direction,
        );

        for i in 0..CHUNK {
            builder.assert_bool(local.auxes[i]);
        }

        self.eval_interactions(builder, local);
    }
}
