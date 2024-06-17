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

impl AirConfig for OfflineChecker {
    type Cols<T> = OfflineCheckerCols<T>;
}

impl<F: Field> BaseAir<F> for OfflineChecker {
    fn width(&self) -> usize {
        self.air_width()
    }
}

impl<AB: PartitionedAirBuilder> Air<AB> for OfflineChecker
where
    AB::M: Clone,
{
    /// This constrains extra rows to be at the bottom and the following on non-extra rows:
    /// Every row is tagged with exactly one of is_initial, is_internal, is_final
    /// is_initial rows must be writes, is_final rows must be reads, and is_internal rows can be either
    /// same_idx, same_data, lt_bit is correct (see definition in columns.rs)
    /// An internal read is preceded by a write (initial or internal) with the same index and data
    /// Every key block ends in an is_final row preceded by an is_internal row
    fn eval(&self, builder: &mut AB) {
        let main = &builder.partitioned_main()[0].clone();

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &[AB::Var] = (*local).borrow();
        let next: &[AB::Var] = (*next).borrow();

        let local_cols = OfflineCheckerCols::from_slice(local, self);
        let next_cols = OfflineCheckerCols::from_slice(next, self);

        // Some helpers
        let and = |a: AB::Expr, b: AB::Expr| a * b;
        let or = |a: AB::Expr, b: AB::Expr| a.clone() + b.clone() - a * b;
        let implies = |a: AB::Expr, b: AB::Expr| or(AB::Expr::one() - a, b);

        // Making sure bits are bools
        builder.assert_bool(local_cols.is_initial);
        builder.assert_bool(local_cols.is_final);
        builder.assert_bool(local_cols.is_internal);
        builder.assert_bool(local_cols.op_type);
        builder.assert_bool(local_cols.same_idx);
        builder.assert_bool(local_cols.same_data);
        builder.assert_bool(local_cols.is_extra);

        // Ensuring is_final_x3 is correct
        builder.assert_eq(
            local_cols.is_final_x3,
            local_cols.is_final * AB::Expr::from_canonical_u8(3),
        );

        // Making sure first row starts with same_idx, same_data being false
        builder.when_first_row().assert_zero(local_cols.same_idx);
        builder.when_first_row().assert_zero(local_cols.same_data);

        // Making sure same_idx is correct across rows
        let is_equal_idx = IsEqualVecCols::new(
            local_cols.page_row[1..self.idx_len + 1].to_vec(),
            next_cols.page_row[1..self.idx_len + 1].to_vec(),
            next_cols.is_equal_idx_aux.prods,
            next_cols.is_equal_idx_aux.invs,
        );

        let is_equal_idx_air = IsEqualVecAir::new(self.idx_len);

        SubAir::eval(
            &is_equal_idx_air,
            &mut builder.when_transition(),
            is_equal_idx.io,
            is_equal_idx.aux,
        );

        // Making sure same_data is correct across rows
        let is_equal_data = IsEqualVecCols::new(
            local_cols.page_row[self.idx_len + 1..].to_vec(),
            next_cols.page_row[self.idx_len + 1..].to_vec(),
            next_cols.is_equal_data_aux.prods,
            next_cols.is_equal_data_aux.invs,
        );
        let is_equal_data_air = IsEqualVecAir::new(self.data_len);

        SubAir::eval(
            &is_equal_data_air,
            &mut builder.when_transition(),
            is_equal_data.io,
            is_equal_data.aux,
        );

        // Ensuring all rows are sorted by (key, clk)
        let lt_io_cols = IsLessThanTupleIOCols::<AB::Var> {
            x: local_cols.page_row[1..self.idx_len + 1]
                .iter()
                .copied()
                .chain(iter::once(local_cols.clk))
                .collect(),
            y: next_cols.page_row[1..self.idx_len + 1]
                .iter()
                .copied()
                .chain(iter::once(next_cols.clk))
                .collect(),
            tuple_less_than: next_cols.lt_bit,
        };

        let lt_chip = IsLessThanTupleAir::new(
            self.range_bus_index,
            self.idx_clk_limb_bits.clone(),
            self.idx_decomp,
        );

        SubAir::eval(
            &lt_chip,
            &mut builder.when_transition(),
            lt_io_cols,
            next_cols.lt_aux,
        );

        // Ensuring lt_bit is on
        builder
            .when_transition()
            .assert_one(or(next_cols.is_extra.into(), next_cols.lt_bit.into()));

        // Making sure every idx block starts with a write
        // not same_idx => write
        // NOTE: constraint degree is 3
        builder.assert_one(or(
            local_cols.is_extra.into(),
            or(local_cols.same_idx.into(), local_cols.op_type.into()),
        ));

        // Making sure every idx block ends with a is_final
        // NOTE: constraint degree is 3
        builder.when_transition().assert_one(or(
            local_cols.is_extra.into(),
            or(next_cols.same_idx.into(), local_cols.is_final.into()),
        ));
        // NOTE: constraint degree is 3
        builder.when_transition().assert_one(implies(
            and(
                AB::Expr::one() - local_cols.is_extra.into(),
                next_cols.is_extra.into(),
            ),
            local_cols.is_final.into(),
        ));
        builder.when_last_row().assert_one(implies(
            AB::Expr::one() - local_cols.is_extra,
            local_cols.is_final.into(),
        ));

        // Making sure that is_initial rows only appear at the start of blocks
        // is_initial => not same_idx
        builder.assert_one(implies(
            local_cols.is_initial.into(),
            AB::Expr::one() - local_cols.same_idx,
        ));

        // Making sure that every read uses the same data as the last operation
        // read => same_data
        // NOTE: constraint degree is 3
        builder.assert_one(or(
            local_cols.is_extra.into(),
            or(local_cols.op_type.into(), local_cols.same_data.into()),
        ));

        // is_final => read
        // NOTE: constraint degree is 3
        builder.assert_one(or(
            local_cols.is_extra.into(),
            implies(
                local_cols.is_final.into(),
                AB::Expr::one() - local_cols.op_type.into(),
            ),
        ));

        // is_internal => not is_initial
        builder.assert_one(implies(
            local_cols.is_internal.into(),
            AB::Expr::one() - local_cols.is_initial,
        ));

        // is_internal => not is_final
        builder.assert_one(implies(
            local_cols.is_internal.into(),
            AB::Expr::one() - local_cols.is_final,
        ));

        // next.is_final => local.is_internal
        builder.when_transition().assert_one(implies(
            next_cols.is_final.into(),
            local_cols.is_internal.into(),
        ));

        // Ensuring at least one of is_initial, is_internal, is_final is on
        builder.assert_zero(
            (AB::Expr::one() - local_cols.is_extra)
                * (local_cols.is_initial + local_cols.is_internal + local_cols.is_final
                    - AB::Expr::one()),
        );

        // Making sure is_extra rows are at the bottom
        builder.when_transition().assert_one(implies(
            AB::Expr::one() - next_cols.is_extra,
            AB::Expr::one() - local_cols.is_extra,
        ));

        // Note that the following is implied:
        // - for every row: (is_initial => write) because is_initial => not same_idx => write
        // - for every row: (is_initial => not is_final) because is_final => read and is_initial => not same_idx => write
        // - for every row: exactly one of is_initial, is_internal, is_final is on because we know at least one of them
        //   is on, and we know each of them implies the other two are off
        // - for every row: read => same_idx because not same_idx => write
        // - there is at most 1 is_initial per index block because every row is sent at most once from the inital page chip
        // - there is exactly 1 is_final per index block because every row is received at most once from the final page chip
        //   and we make sure that is_final is the last row in the block
    }
}
