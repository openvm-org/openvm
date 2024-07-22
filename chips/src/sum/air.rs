use std::borrow::Borrow;

use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, AirBuilder};
use p3_field::AbstractField;
use p3_matrix::Matrix;

use crate::is_less_than::{columns::IsLessThanIoCols, IsLessThanAir};

use super::columns::SumCols;

/// The `SumAir` defines constraints for a trace matrix that accumulates the sums of
/// values grouped by keys from key-value pair inputs.
///
/// Each state in valid trace matrix is a 4-tuple `(key, value, partial_sum, is_final)`.
/// - `key`: Defines the group.
/// - `value`: The value associated with the key for that row.
/// - `partial_sum`: The cumulative sum of values for the key up to the current row.
/// - `is_final`: Indicates whether this row is the last for the current group (`1` if true, otherwise `0`).
///
/// The data for `key` and `value` is sourced from the input bus. For each unique key, a `(key, sum)` pair
/// is sent to the output bus, where `sum` is the total sum of all values associated with that key.
#[derive(Copy, Clone, Debug)]
pub struct SumAir {
    pub input_bus: usize,
    pub output_bus: usize,

    pub is_lt_air: IsLessThanAir,
}

/// The `SumAir` implements the following constraints:
/// - `is_final` is boolean (global).
/// - `is_final` is true for the last row. (This is necessary for the case when all keys are the same.)
/// - Group initialization (global): `local.is_final => next.partial_sum = next.value`.
/// - Group initialization (transition): `local.is_final => next.key < next.value`.
/// - Group transition (transition): `!local.is_final => local.partial_sum = local.partial_sum + next.value`.
/// - Group transition (transition): !local.is_final => local.key = next.key.
impl<AB: InteractionBuilder> Air<AB> for SumAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &[AB::Var] = (*local).borrow();
        let next: &[AB::Var] = (*next).borrow();

        let local = SumCols::from_slice(local, self.is_lt_air.limb_bits, self.is_lt_air.decomp);
        let next = SumCols::from_slice(next, self.is_lt_air.limb_bits, self.is_lt_air.decomp);

        builder.assert_bool(local.is_final);
        builder.when_last_row().assert_one(local.is_final);

        // next starts a new group. combined with the above constraint, this also applies for the first row
        builder
            .when(local.is_final)
            .assert_eq(next.partial_sum, next.value);

        let mut when_transition = builder.when_transition();

        let mut when_not_final = when_transition.when_ne(local.is_final, AB::Expr::one());
        when_not_final.assert_eq(local.key, next.key);
        when_not_final.assert_eq(next.partial_sum, local.partial_sum + next.value);

        let is_lt_io_cols = IsLessThanIoCols {
            x: local.key,
            y: next.key,
            less_than: local.is_final,
        };
        self.is_lt_air
            .eval_when_transition(builder, is_lt_io_cols, local.is_lt_aux_cols);

        self.eval_interactions(
            builder,
            local.key,
            local.value,
            local.partial_sum,
            local.is_final,
        );
    }
}
