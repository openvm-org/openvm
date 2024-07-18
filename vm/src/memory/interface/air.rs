use std::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use crate::memory::interface::columns::MemoryInterfaceCols;

pub struct MemoryInterfaceAir<const CHUNK: usize> {}

impl<const CHUNK: usize, F: Field> BaseAir<F> for MemoryInterfaceAir<CHUNK> {
    fn width(&self) -> usize {
        MemoryInterfaceCols::<CHUNK, F>::get_width()
    }
}

impl<const CHUNK: usize, AB: AirBuilder> Air<AB> for MemoryInterfaceAir<CHUNK> {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();
        let local_cols = MemoryInterfaceCols::<CHUNK, AB::Var>::from_slice(local);

        // `direction` should be -1, 0, 1
        builder.assert_eq(
            local_cols.direction,
            local_cols.direction * local_cols.direction * local_cols.direction,
        );
        // -1 -> 0, 1 -> 1
        let direction_bool = (local_cols.direction + AB::F::one()) * AB::F::two().inverse();

        for i in 0..CHUNK {
            builder.assert_bool(local_cols.auxes[i]);
            builder.assert_eq(
                local_cols.temp_multiplicity[i],
                AB::Expr::one()
                    - (direction_bool.clone() * (AB::Expr::one() - local_cols.auxes[i])),
            );
            builder.assert_eq(
                local_cols.temp_is_final[i],
                (AB::Expr::one() - direction_bool.clone()) * local_cols.auxes[i],
            );
        }
    }
}
