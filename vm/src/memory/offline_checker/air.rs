use afs_chips::{
    offline_checker::columns::GeneralOfflineCheckerCols,
    sub_chip::{AirConfig, SubAir},
};
use afs_stark_backend::air_builders::PartitionedAirBuilder;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;
use std::borrow::Borrow;

use super::OfflineChecker;

impl AirConfig for OfflineChecker {
    type Cols<T> = GeneralOfflineCheckerCols<T>;
}

impl<F: Field> BaseAir<F> for OfflineChecker {
    fn width(&self) -> usize {
        self.general_offline_checker.air_width()
    }
}

impl<AB: PartitionedAirBuilder> Air<AB> for OfflineChecker
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

        let general_offline_checker = self.general_offline_checker.clone();

        let local_cols = GeneralOfflineCheckerCols::from_slice(local, &general_offline_checker);
        let next_cols = GeneralOfflineCheckerCols::from_slice(next, &general_offline_checker);

        let or = |a: AB::Expr, b: AB::Expr| a.clone() + b.clone() - a * b;

        SubAir::eval(
            &general_offline_checker,
            builder,
            (local_cols.clone(), next_cols.clone()),
            (),
        );

        builder.assert_bool(local_cols.op_type);

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
    }
}
