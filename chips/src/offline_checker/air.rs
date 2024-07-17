use std::borrow::Borrow;
use std::iter;

use afs_stark_backend::air_builders::PartitionedAirBuilder;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::{columns::OfflineCheckerCols, OfflineChecker};
use crate::{
    is_equal_vec::{columns::IsEqualVecCols, IsEqualVecAir},
    is_less_than_tuple::{columns::IsLessThanTupleIOCols, IsLessThanTupleAir},
    sub_chip::{AirConfig, SubAir},
};

impl<const WORD_SIZE: usize> AirConfig for OfflineChecker<WORD_SIZE> {
    type Cols<T> = OfflineCheckerCols<T>;
}

impl<const WORD_SIZE: usize, F: Field> BaseAir<F> for OfflineChecker<WORD_SIZE> {
    fn width(&self) -> usize {
        self.air_width()
    }
}

impl<const WORD_SIZE: usize, AB: PartitionedAirBuilder> Air<AB> for OfflineChecker<WORD_SIZE>
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

        let local_cols = OfflineCheckerCols::from_slice(local, self);
        let next_cols = OfflineCheckerCols::from_slice(next, self);

        // Some helpers
        let not = |a: AB::Expr| AB::Expr::one() - a;
        let and = |a: AB::Expr, b: AB::Expr| a * b;
        let or = |a: AB::Expr, b: AB::Expr| a.clone() + b.clone() - a * b;
        let implies = |a: AB::Expr, b: AB::Expr| not(and(a, not(b)));

        // Making sure bits are bools
        builder.assert_bool(local_cols.op_type);
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

        // Make sure that same_idx comes from is_equal_idx_aux (last element of prods indicates whether equal)
        builder.assert_eq(
            next_cols.same_idx,
            next_cols.is_equal_idx_aux.prods[self.idx_len - 1],
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

        // Making sure that every read uses the same data as the last operation if it is not the first
        // operation in the block
        // NOTE: constraint degree is 4
        builder.assert_one(or(
            AB::Expr::one() - local_cols.is_valid.into(),
            or(
                local_cols.op_type.into(),
                // if same_idx = 0 and read, then same_data can be anything
                // if same_idx = 1 and read, then same_data must be 1
                AB::Expr::one() - local_cols.same_idx.into() + local_cols.same_idx_and_data,
            ),
        ));

        // Making sure is_extra rows are at the bottom
        builder.when_transition().assert_one(implies(
            next_cols.is_valid.into(),
            local_cols.is_valid.into(),
        ));
    }
}
