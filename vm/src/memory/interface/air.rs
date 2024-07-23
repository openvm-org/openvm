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

        for i in 0..CHUNK {
            builder.assert_bool(local_cols.auxes[i]);
            // when `direction` is 1, multiplicity of interaction should be `!auxes[i]`
            // otherwise, multiplicity should be `direction`
            builder.assert_eq(
                local_cols.temp_multiplicity[i],
                local_cols.direction
                    * (AB::Expr::one()
                        - ((local_cols.direction + AB::F::one())
                            * AB::F::two().inverse()
                            * (AB::Expr::one() - local_cols.auxes[i]))),
            );
            // when `direction` is -1, is_final field of interaction should be `auxes[i]`
            // otherwise, is_final field should be 1
            builder.assert_eq(
                local_cols.temp_is_final[i],
                (AB::Expr::one() - local_cols.direction)
                    * AB::F::two().inverse()
                    * local_cols.auxes[i],
            );
        }
    }
}
