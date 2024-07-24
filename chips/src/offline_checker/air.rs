use std::iter;

use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field, PrimeField64};
use p3_matrix::Matrix;

use super::{
    columns::OfflineCheckerCols, OfflineChecker, OfflineCheckerChip, OfflineCheckerOperation,
};
use crate::{
    is_equal_vec::columns::IsEqualVecCols,
    is_less_than_tuple::columns::IsLessThanTupleIoCols,
    sub_chip::{AirConfig, SubAir},
    utils::{implies, or},
};

impl AirConfig for OfflineChecker {
    type Cols<T> = OfflineCheckerCols<T>;
}

impl<F: PrimeField64, Operation: OfflineCheckerOperation<F>> AirConfig
    for OfflineCheckerChip<F, Operation>
{
    type Cols<T> = OfflineCheckerCols<T>;
}

impl<F: Field> BaseAir<F> for OfflineChecker {
    fn width(&self) -> usize {
        self.air_width()
    }
}

impl<AB: InteractionBuilder> Air<AB> for OfflineChecker {
    /// This constrains extra rows to be at the bottom and the following on non-extra rows:
    /// same_addr_space, same_pointer, same_data, lt_bit is correct (see definition in columns.rs)
    /// A read must be preceded by a write with the same address space, pointer, and data
    fn eval(&self, builder: &mut AB) {
        let main = &builder.main();

        let local_cols = OfflineCheckerCols::from_slice(&main.row_slice(0), self);
        let next_cols = OfflineCheckerCols::from_slice(&main.row_slice(1), self);

        SubAir::eval(self, builder, (local_cols, next_cols), ());
    }
}

impl<AB: InteractionBuilder> SubAir<AB> for OfflineChecker {
    type IoView = (OfflineCheckerCols<AB::Var>, OfflineCheckerCols<AB::Var>);
    type AuxView = ();

    fn eval(&self, builder: &mut AB, io: Self::IoView, _: Self::AuxView) {
        let (local_cols, next_cols) = io;

        self.eval_interactions(builder, &local_cols);

        // Making sure bits are bools
        builder.assert_bool(local_cols.same_idx);
        builder.assert_bool(local_cols.is_valid);

        // Making sure first row starts with same_idx, same_data being false
        builder.when_first_row().assert_zero(local_cols.same_idx);

        // Making sure same_idx is correct across rows
        let is_equal_idx_cols = IsEqualVecCols::new(
            local_cols.idx.clone(),
            next_cols.idx.clone(),
            next_cols.same_idx,
            next_cols.is_equal_idx_aux.prods.clone(),
            next_cols.is_equal_idx_aux.invs,
        );

        SubAir::eval(
            &self.is_equal_idx_air,
            &mut builder.when_transition(),
            is_equal_idx_cols.io,
            is_equal_idx_cols.aux,
        );

        // Ensuring all rows are sorted by (idx, clk)
        let lt_io_cols = IsLessThanTupleIoCols::<AB::Var> {
            x: local_cols
                .idx
                .into_iter()
                .chain(iter::once(local_cols.clk))
                .collect(),
            y: next_cols
                .idx
                .into_iter()
                .chain(iter::once(next_cols.clk))
                .collect(),
            tuple_less_than: next_cols.lt_bit,
        };

        self.lt_tuple_air
            .eval_when_transition(builder, lt_io_cols, next_cols.lt_aux);

        // Ensuring lt_bit is on
        builder.when_transition().assert_one(or::<AB>(
            AB::Expr::one() - next_cols.is_valid.into(),
            next_cols.lt_bit.into(),
        ));

        // Making sure is_extra rows are at the bottom
        builder.when_transition().assert_one(implies::<AB>(
            next_cols.is_valid.into(),
            local_cols.is_valid.into(),
        ));
    }
}
