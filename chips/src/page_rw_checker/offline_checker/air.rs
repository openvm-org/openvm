use std::borrow::Borrow;
use std::iter;

use afs_stark_backend::air_builders::PartitionedAirBuilder;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::{columns::OfflineCheckerCols, OfflineChecker};
use crate::{
    is_equal_vec::columns::IsEqualVecCols,
    is_less_than_tuple::columns::IsLessThanTupleIOCols,
    sub_chip::{AirConfig, SubAir},
    utils::{and, implies, or},
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
    /// Every row is tagged with exactly one of is_initial, is_internal, is_final_write, is_final_delete
    /// is_initial rows must be writes, is_final rows must be reads, and is_internal rows can be either
    /// same_idx, lt_bit is correct (see definition in columns.rs)
    /// An internal read is preceded by a write (initial or internal) with the same index and data
    /// Every key block ends in an is_final_write or is_final_delete row preceded by an is_internal row
    fn eval(&self, builder: &mut AB) {
        let main = &builder.partitioned_main()[0].clone();

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &[AB::Var] = (*local).borrow();
        let next: &[AB::Var] = (*next).borrow();

        let local_cols = OfflineCheckerCols::from_slice(local, self);
        let next_cols = OfflineCheckerCols::from_slice(next, self);

        // Some helpers
        let and = and::<AB>;
        let or = or::<AB>;
        let implies = implies::<AB>;

        // Making sure bits are bools
        builder.assert_bool(local_cols.is_initial);
        builder.assert_bool(local_cols.is_final_write);
        builder.assert_bool(local_cols.is_final_delete);
        builder.assert_bool(local_cols.is_internal);
        builder.assert_bool(local_cols.is_read);
        builder.assert_bool(local_cols.is_write);
        builder.assert_bool(local_cols.is_delete);
        builder.assert_bool(local_cols.same_idx);
        builder.assert_bool(local_cols.is_extra);

        // Making sure op_type is one of 0, 1, 2 (R, W, D)
        builder.assert_zero(
            local_cols.op_type
                * (local_cols.op_type - AB::Expr::one())
                * (local_cols.op_type - AB::Expr::two()),
        );

        // Ensuring that op_type is decomposed into is_read, is_write, is_delete correctly
        builder.assert_eq(
            local_cols.op_type,
            local_cols.is_write + local_cols.is_delete * AB::Expr::from_canonical_u8(2),
        );

        // Ensuring the sum of is_initial, is_internal, is_final_write, is_final_delete is 1
        // This ensures exactly one of them is on because they're all bool
        builder.assert_zero(
            (AB::Expr::one() - local_cols.is_extra)
                * (local_cols.is_initial
                    + local_cols.is_internal
                    + local_cols.is_final_write
                    + local_cols.is_final_delete
                    - AB::Expr::one()),
        );

        // Ensuring is_final_write_x3 is correct
        builder.assert_eq(
            local_cols.is_final_write_x3,
            local_cols.is_final_write * AB::Expr::from_canonical_u8(3),
        );

        // Making sure first row starts with same_idx being false
        builder.when_first_row().assert_zero(local_cols.same_idx);

        // Making sure same_idx is correct across rows
        let is_equal_idx_cols = IsEqualVecCols::new(
            local_cols.idx.to_vec(),
            next_cols.idx.to_vec(),
            next_cols.is_equal_idx_aux.prods,
            next_cols.is_equal_idx_aux.invs,
        );

        SubAir::eval(
            &self.is_equal_idx_air,
            &mut builder.when_transition(),
            is_equal_idx_cols.io,
            is_equal_idx_cols.aux,
        );

        // Ensuring all rows are sorted by (key, clk)
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

        SubAir::eval(
            &self.lt_idx_clk_air,
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
            or(local_cols.same_idx.into(), local_cols.is_write.into()),
        ));

        // Making sure every idx block ends with a is_final_write or is_final_delete (in the three constraints below)
        // First, when local and next are not extra
        // NOTE: constraint degree is 3
        builder.when_transition().assert_one(or(
            next_cols.is_extra.into(),
            or(
                next_cols.same_idx.into(),
                local_cols.is_final_write.into() + local_cols.is_final_delete.into(),
            ),
        ));
        // NOTE: constraint degree is 3
        // Second, when local is not extra but next is extra
        builder.when_transition().assert_one(implies(
            and(
                AB::Expr::one() - local_cols.is_extra.into(),
                next_cols.is_extra.into(),
            ),
            local_cols.is_final_write.into() + local_cols.is_final_delete.into(),
        ));
        // Third, when it's the last row
        builder.when_last_row().assert_one(implies(
            AB::Expr::one() - local_cols.is_extra,
            local_cols.is_final_write.into() + local_cols.is_final_delete.into(),
        ));

        // Making sure that is_initial rows only appear at the start of blocks
        // is_initial => not same_idx
        builder.assert_one(implies(
            local_cols.is_initial.into(),
            AB::Expr::one() - local_cols.same_idx,
        ));

        // Making sure that every read uses the same data as the last operation
        // We do this by looping over the data part of next row and ensuring that
        // every entry matches the one in local in case next is_read (and not is_extra)
        // read => same_data (data in next matches data in local)
        for i in 0..self.data_len {
            // NOTE: constraint degree is 3
            builder.when_transition().assert_zero(
                (next_cols.is_read * (AB::Expr::one() - next_cols.is_extra))
                    * (local_cols.data[i] - next_cols.data[i]),
            );
        }

        // is_final => read
        // NOTE: constraint degree is 3
        builder.assert_one(or(
            local_cols.is_extra.into(),
            implies(local_cols.is_final_write.into(), local_cols.is_read.into()),
        ));

        // is_internal => not is_initial
        builder.assert_one(implies(
            local_cols.is_internal.into(),
            AB::Expr::one() - local_cols.is_initial,
        ));

        // is_internal => not is_final
        builder.assert_one(implies(
            local_cols.is_internal.into(),
            AB::Expr::one()
                - (local_cols.is_final_write.into() + local_cols.is_final_delete.into()),
        ));

        // next is_final_write or next is_final_delete => local is_internal
        builder.when_transition().assert_one(implies(
            next_cols.is_final_write.into() + next_cols.is_final_delete.into(),
            local_cols.is_internal.into(),
        ));

        // Ensuring that next read => not local delete
        // NOTE: constraint degree is 3
        builder.when_transition().assert_one(or(
            next_cols.is_extra.into(),
            implies(
                next_cols.is_read.into(),
                AB::Expr::one() - local_cols.is_delete,
            ),
        ));

        // Ensuring local is_final_delete => next not same_idx
        // NOTE: constraint degree is 3
        builder.when_transition().assert_one(or(
            next_cols.is_extra.into(),
            implies(
                local_cols.is_final_delete.into(),
                AB::Expr::one() - next_cols.same_idx,
            ),
        ));

        // Ensuring that next is_final_delete => local is_delete
        // NOTE: constraint degree is 3
        builder.when_transition().assert_one(or(
            next_cols.is_extra.into(),
            implies(
                next_cols.is_final_delete.into(),
                local_cols.is_delete.into(),
            ),
        ));

        // Making sure is_extra rows are at the bottom
        builder.when_transition().assert_one(implies(
            AB::Expr::one() - next_cols.is_extra,
            AB::Expr::one() - local_cols.is_extra,
        ));

        // Note that the following is implied:
        // - for every row: (is_initial => write) because is_initial => not same_idx => write
        // - for every row: (is_initial => not is_final_write) because is_final_write => read and is_initial => not same_idx => write
        // - for every row: exactly one of is_initial, is_internal, is_final_write, is_final_delete is on because we know their sum if 1
        //   and that they're bool
        // - for every row: read => same_idx because not same_idx => write
        // - there is at most 1 is_initial per index block because every row is sent at most once from the inital page chip
        // - there is exactly 1 is_final_write or is_final_delete per index block because we enforce the row below is_final_write
        //   or is_final_delete to have a different idx
    }
}
