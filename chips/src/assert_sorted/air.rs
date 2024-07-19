use std::borrow::Borrow;

use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use crate::is_less_than_tuple::columns::{IsLessThanTupleCols, IsLessThanTupleIoCols};
use crate::is_less_than_tuple::IsLessThanTupleAir;

use super::columns::AssertSortedCols;

#[derive(Clone, Debug)]
pub struct AssertSortedAir {
    pub is_less_than_tuple_air: IsLessThanTupleAir,
}

impl AssertSortedAir {
    pub fn new(bus_index: usize, limb_bits: Vec<usize>, decomp: usize) -> Self {
        // We do not enable interactions for IsLessThanTupleAir because that AIR assumes
        // that `x, y` are on the same row. We will separately enable interactions for this Air.
        Self {
            is_less_than_tuple_air: IsLessThanTupleAir::new(bus_index, limb_bits, decomp),
        }
    }
}

impl<F: Field> BaseAir<F> for AssertSortedAir {
    fn width(&self) -> usize {
        AssertSortedCols::<F>::get_width(
            self.is_less_than_tuple_air.limb_bits().clone(),
            self.is_less_than_tuple_air.decomp,
            self.is_less_than_tuple_air.tuple_len(),
        )
    }
}

impl<AB: InteractionBuilder> Air<AB> for AssertSortedAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        // get the current row and the next row
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &[AB::Var] = (*local).borrow();
        let next: &[AB::Var] = (*next).borrow();

        let local_cols = AssertSortedCols::from_slice(
            local,
            self.is_less_than_tuple_air.limb_bits().clone(),
            self.is_less_than_tuple_air.decomp,
            self.is_less_than_tuple_air.tuple_len(),
        );

        let next_cols = AssertSortedCols::from_slice(
            next,
            self.is_less_than_tuple_air.limb_bits().clone(),
            self.is_less_than_tuple_air.decomp,
            self.is_less_than_tuple_air.tuple_len(),
        );

        // constrain that the current key is less than the next
        builder
            .when_transition()
            .assert_one(local_cols.less_than_next_key);

        let is_less_than_tuple_cols = IsLessThanTupleCols {
            io: IsLessThanTupleIoCols {
                x: local_cols.key,
                y: next_cols.key,
                tuple_less_than: local_cols.less_than_next_key,
            },
            aux: local_cols.is_less_than_tuple_aux,
        };

        // The `eval_interactions` below performs range checks on lower_decomp on every row, even
        // though in this AIR the lower_decomp is not used on the last row.
        // This simply means the trace generation must fill in the last row with numbers in range (e.g., with zeros)
        self.is_less_than_tuple_air
            .eval_interactions(builder, &is_less_than_tuple_cols.aux.less_than_aux);

        // constrain the indicator that we used to check whether the current key < next key is correct
        self.is_less_than_tuple_air.eval_without_interactions(
            &mut builder.when_transition(),
            is_less_than_tuple_cols.io,
            is_less_than_tuple_cols.aux,
        );
    }
}
