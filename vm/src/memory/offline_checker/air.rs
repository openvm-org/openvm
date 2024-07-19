use afs_chips::{
    offline_checker::columns::OfflineCheckerCols,
    sub_chip::{AirConfig, SubAir},
    utils::or,
};
use afs_stark_backend::air_builders::PartitionedAirBuilder;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;
use std::borrow::Borrow;

use super::MemoryOfflineChecker;

impl AirConfig for MemoryOfflineChecker {
    type Cols<T> = OfflineCheckerCols<T>;
}

impl<F: Field> BaseAir<F> for MemoryOfflineChecker {
    fn width(&self) -> usize {
        self.offline_checker.air_width()
    }
}

impl<AB: PartitionedAirBuilder> Air<AB> for MemoryOfflineChecker
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

        let offline_checker = self.offline_checker.clone();

        let local_cols = OfflineCheckerCols::from_slice(local, &offline_checker);
        let next_cols = OfflineCheckerCols::from_slice(next, &offline_checker);

        SubAir::eval(
            &offline_checker,
            builder,
            (local_cols.clone(), next_cols.clone()),
            (),
        );

        builder.assert_bool(local_cols.op_type);

        // Making sure that every read uses the same data as the last operation if it is not the first
        // operation in the block
        // NOTE: constraint degree is 4
        builder.assert_one(or::<AB>(
            AB::Expr::one() - local_cols.is_valid.into(),
            or::<AB>(
                local_cols.op_type.into(),
                // if same_idx = 0 and read, then same_data can be anything
                // if same_idx = 1 and read, then same_data must be 1
                AB::Expr::one() - local_cols.same_idx.into() + local_cols.same_idx_and_data,
            ),
        ));
    }
}
