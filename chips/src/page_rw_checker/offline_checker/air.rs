use std::borrow::Borrow;

use afs_stark_backend::air_builders::PartitionedAirBuilder;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::{columns::PageOfflineCheckerCols, PageOfflineChecker};
use crate::{
    sub_chip::SubAir,
    utils::{and, implies, or},
};

impl<F: Field> BaseAir<F> for PageOfflineChecker {
    fn width(&self) -> usize {
        self.air_width()
    }
}

impl<AB: PartitionedAirBuilder> Air<AB> for PageOfflineChecker
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

        let local_cols = PageOfflineCheckerCols::from_slice(local, self);
        let next_cols = PageOfflineCheckerCols::from_slice(next, self);

        let local_offline_checker_cols = local_cols.offline_checker_cols;
        let next_offline_checker_cols = next_cols.offline_checker_cols;
        SubAir::eval(
            &self.offline_checker,
            builder,
            (
                local_offline_checker_cols.clone(),
                next_offline_checker_cols.clone(),
            ),
            (),
        );

        // Making sure bits are bools
        builder.assert_bool(local_cols.is_initial);
        builder.assert_bool(local_cols.is_final_write);
        builder.assert_bool(local_cols.is_final_delete);
        builder.assert_bool(local_cols.is_internal);
        builder.assert_bool(local_cols.is_read);
        builder.assert_bool(local_cols.is_write);
        builder.assert_bool(local_cols.is_delete);

        // Making sure op_type is one of 0, 1, 2 (R, W, D)
        builder.assert_zero(
            local_offline_checker_cols.op_type
                * (local_offline_checker_cols.op_type - AB::Expr::one())
                * (local_offline_checker_cols.op_type - AB::Expr::two()),
        );

        // Ensuring that op_type is decomposed into is_read, is_write, is_delete correctly
        builder.assert_eq(
            local_offline_checker_cols.op_type,
            local_cols.is_write + local_cols.is_delete * AB::Expr::from_canonical_u8(2),
        );

        // Making sure that every read uses the same data as the last operation
        // We do this by looping over the data part of next row and ensuring that
        // every entry matches the one in local in case next is_read (and not is_extra)
        // read => same_data (data in next matches data in local)
        for i in 0..self.offline_checker.data_len {
            // NOTE: constraint degree is 3
            builder.when_transition().assert_zero(
                (next_cols.is_read * next_offline_checker_cols.is_valid)
                    * (local_offline_checker_cols.data[i] - next_offline_checker_cols.data[i]),
            );
        }

        // Ensuring the sum of is_initial, is_internal, is_final_write, is_final_delete is 1
        // This ensures exactly one of them is on because they're all bool
        builder.assert_zero(
            local_offline_checker_cols.is_valid
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

        // Making sure every idx block starts with a write
        // not same_idx => write
        // NOTE: constraint degree is 3
        builder.assert_one(or::<AB>(
            AB::Expr::one() - local_offline_checker_cols.is_valid.into(),
            or::<AB>(
                local_offline_checker_cols.same_idx.into(),
                local_cols.is_write.into(),
            ),
        ));

        // Making sure every idx block ends with a is_final_write or is_final_delete (in the three constraints below)
        // First, when local and next are not extra
        // NOTE: constraint degree is 3
        builder.when_transition().assert_one(or::<AB>(
            AB::Expr::one() - next_offline_checker_cols.is_valid.into(),
            or::<AB>(
                next_offline_checker_cols.same_idx.into(),
                local_cols.is_final_write.into() + local_cols.is_final_delete.into(),
            ),
        ));
        // NOTE: constraint degree is 3
        // Second, when local is not extra but next is extra
        builder.when_transition().assert_one(implies::<AB>(
            and::<AB>(
                local_offline_checker_cols.is_valid.into(),
                AB::Expr::one() - next_offline_checker_cols.is_valid.into(),
            ),
            local_cols.is_final_write.into() + local_cols.is_final_delete.into(),
        ));
        // Third, when it's the last row
        builder.when_last_row().assert_one(implies::<AB>(
            local_offline_checker_cols.is_valid.into(),
            local_cols.is_final_write.into() + local_cols.is_final_delete.into(),
        ));

        // Making sure that is_initial rows only appear at the start of blocks
        // is_initial => not same_idx
        builder.assert_one(implies::<AB>(
            local_cols.is_initial.into(),
            AB::Expr::one() - local_offline_checker_cols.same_idx,
        ));

        let local_data = &local_offline_checker_cols.data;
        let next_data = &next_offline_checker_cols.data;

        // Making sure that every read uses the same data as the last operation
        // We do this by looping over the data part of next row and ensuring that
        // every entry matches the one in local in case next is_read (and not is_extra)
        // read => same_data (data in next matches data in local)
        for i in 0..self.offline_checker.data_len {
            // NOTE: constraint degree is 3
            builder.when_transition().assert_zero(
                (next_cols.is_read * next_offline_checker_cols.is_valid.into())
                    * (local_data[i] - next_data[i]),
            );
        }

        // is_final => read
        // NOTE: constraint degree is 3
        builder.assert_one(or::<AB>(
            AB::Expr::one() - local_offline_checker_cols.is_valid.into(),
            implies::<AB>(local_cols.is_final_write.into(), local_cols.is_read.into()),
        ));

        // is_internal => not is_initial
        builder.assert_one(implies::<AB>(
            local_cols.is_internal.into(),
            AB::Expr::one() - local_cols.is_initial,
        ));

        // is_internal => not is_final
        builder.assert_one(implies::<AB>(
            local_cols.is_internal.into(),
            AB::Expr::one()
                - (local_cols.is_final_write.into() + local_cols.is_final_delete.into()),
        ));

        // next is_final_write or next is_final_delete => local is_internal
        builder.when_transition().assert_one(implies::<AB>(
            next_cols.is_final_write.into() + next_cols.is_final_delete.into(),
            local_cols.is_internal.into(),
        ));

        // Ensuring that next read => not local delete
        // NOTE: constraint degree is 3
        builder.when_transition().assert_one(or::<AB>(
            AB::Expr::one() - next_offline_checker_cols.is_valid.into(),
            implies::<AB>(
                next_cols.is_read.into(),
                AB::Expr::one() - local_cols.is_delete,
            ),
        ));

        // Ensuring local is_final_delete => next not same_idx
        // NOTE: constraint degree is 3
        builder.when_transition().assert_one(or::<AB>(
            AB::Expr::one() - next_offline_checker_cols.is_valid.into(),
            implies::<AB>(
                local_cols.is_final_delete.into(),
                AB::Expr::one() - next_offline_checker_cols.same_idx,
            ),
        ));

        // Ensuring that next is_final_delete => local is_delete
        // NOTE: constraint degree is 3
        builder.when_transition().assert_one(or::<AB>(
            AB::Expr::one() - next_offline_checker_cols.is_valid.into(),
            implies::<AB>(
                next_cols.is_final_delete.into(),
                local_cols.is_delete.into(),
            ),
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
