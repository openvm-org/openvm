use std::borrow::Borrow;
use std::iter;

use afs_stark_backend::air_builders::PartitionedAirBuilder;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field, PrimeField64};
use p3_matrix::Matrix;

use super::{
    columns::GeneralOfflineCheckerCols, GeneralOfflineChecker, GeneralOfflineCheckerChip,
    GeneralOfflineCheckerOperation,
};
use crate::{
    is_equal_vec::{columns::IsEqualVecCols, IsEqualVecAir},
    is_less_than_tuple::{columns::IsLessThanTupleIOCols, IsLessThanTupleAir},
    sub_chip::{AirConfig, SubAir},
};

impl AirConfig for GeneralOfflineChecker {
    type Cols<T> = GeneralOfflineCheckerCols<T>;
}

impl<F: PrimeField64, Operation: GeneralOfflineCheckerOperation<F>> AirConfig
    for GeneralOfflineCheckerChip<F, Operation>
{
    type Cols<T> = GeneralOfflineCheckerCols<T>;
}

impl<F: Field> BaseAir<F> for GeneralOfflineChecker {
    fn width(&self) -> usize {
        self.air_width()
    }
}

impl<AB: PartitionedAirBuilder> Air<AB> for GeneralOfflineChecker
where
    AB::M: Clone,
{
    /// This constrains extra rows to be at the bottom and the following on non-extra rows:
    /// same_addr_space, same_pointer, same_data, lt_bit is correct (see definition in columns.rs)
    /// A read must be preceded by a write with the same address space, pointer, and data
    fn eval(&self, builder: &mut AB) {
        let main = &builder.partitioned_main()[0].clone();

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &[AB::Var] = (*local).borrow();
        let next: &[AB::Var] = (*next).borrow();

        let local_cols = GeneralOfflineCheckerCols::from_slice(local, self);
        let next_cols = GeneralOfflineCheckerCols::from_slice(next, self);

        SubAir::eval(self, builder, (local_cols, next_cols), ());
    }
}

impl<AB: PartitionedAirBuilder> SubAir<AB> for GeneralOfflineChecker
where
    AB::M: Clone,
{
    type IoView = (
        GeneralOfflineCheckerCols<AB::Var>,
        GeneralOfflineCheckerCols<AB::Var>,
    );
    type AuxView = ();

    fn eval(&self, builder: &mut AB, io: Self::IoView, _: Self::AuxView) {
        let (local_cols, next_cols) = io;

        // Some helpers
        let not = |a: AB::Expr| AB::Expr::one() - a;
        let and = |a: AB::Expr, b: AB::Expr| a * b;
        let or = |a: AB::Expr, b: AB::Expr| a.clone() + b.clone() - a * b;
        let implies = |a: AB::Expr, b: AB::Expr| not(and(a, not(b)));

        // Making sure bits are bools
        builder.assert_bool(local_cols.same_idx);
        builder.assert_bool(local_cols.same_data);
        builder.assert_bool(local_cols.is_valid);

        // Constrain that same_idx_and_data is same_idx * same_data
        builder.assert_eq(
            local_cols.same_idx_and_data,
            local_cols.same_idx * local_cols.same_data,
        );

        // Making sure first row starts with same_idx, same_data being false
        builder.when_first_row().assert_zero(local_cols.same_idx);
        builder.when_first_row().assert_zero(local_cols.same_data);

        // Making sure same_idx is correct across rows
        let is_equal_idx_cols = IsEqualVecCols::new(
            local_cols.idx.clone(),
            next_cols.idx.clone(),
            next_cols.same_idx,
            next_cols.is_equal_idx_aux.prods.clone(),
            next_cols.is_equal_idx_aux.invs,
        );

        let is_equal_idx_air = IsEqualVecAir::new(self.idx_len);
        SubAir::eval(
            &is_equal_idx_air,
            &mut builder.when_transition(),
            is_equal_idx_cols.io,
            is_equal_idx_cols.aux,
        );

        // Making sure same_data is correct across rows
        let is_equal_data = IsEqualVecCols::new(
            local_cols.data.to_vec(),
            next_cols.data.to_vec(),
            next_cols.same_data,
            next_cols.is_equal_data_aux.prods.clone(),
            next_cols.is_equal_data_aux.invs,
        );
        let is_equal_data_air = IsEqualVecAir::new(self.data_len);

        SubAir::eval(
            &is_equal_data_air,
            &mut builder.when_transition(),
            is_equal_data.io,
            is_equal_data.aux,
        );

        // Ensuring all rows are sorted by (idx, clk)
        let lt_io_cols = IsLessThanTupleIOCols::<AB::Var> {
            x: local_cols
                .idx
                .iter()
                .copied()
                .chain(iter::once(local_cols.clk))
                .collect(),
            y: next_cols
                .idx
                .iter()
                .copied()
                .chain(iter::once(next_cols.clk))
                .collect(),
            tuple_less_than: next_cols.lt_bit,
        };

        let lt_chip =
            IsLessThanTupleAir::new(self.range_bus, self.idx_clk_limb_bits.clone(), self.decomp);

        SubAir::eval(
            &lt_chip,
            &mut builder.when_transition(),
            lt_io_cols,
            next_cols.lt_aux,
        );

        // Ensuring lt_bit is on
        builder.when_transition().assert_one(or(
            AB::Expr::one() - next_cols.is_valid.into(),
            next_cols.lt_bit.into(),
        ));

        // Making sure is_extra rows are at the bottom
        builder.when_transition().assert_one(implies(
            next_cols.is_valid.into(),
            local_cols.is_valid.into(),
        ));
    }
}
