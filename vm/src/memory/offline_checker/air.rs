use afs_chips::{
    is_equal_vec::{columns::IsEqualVecCols, IsEqualVecAir},
    sub_chip::{AirConfig, SubAir},
    utils::or,
};

use afs_stark_backend::air_builders::PartitionedAirBuilder;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field, PrimeField32};
use p3_matrix::Matrix;
use std::borrow::Borrow;

use super::{columns::MemoryOfflineCheckerCols, MemoryChip, MemoryOfflineChecker};

impl<const WORD_SIZE: usize, F: PrimeField32> AirConfig for MemoryChip<WORD_SIZE, F> {
    type Cols<T> = MemoryOfflineCheckerCols<T>;
}

impl<F: Field> BaseAir<F> for MemoryOfflineChecker {
    fn width(&self) -> usize {
        self.air_width()
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
        let main = &builder.main();

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &[AB::Var] = (*local).borrow();
        let next: &[AB::Var] = (*next).borrow();

        let offline_checker = self.offline_checker.clone();

        let local_cols = MemoryOfflineCheckerCols::from_slice(local, self);
        let next_cols = MemoryOfflineCheckerCols::from_slice(next, self);

        let local_offline_checker_cols = local_cols.offline_checker_cols;
        let next_offline_checker_cols = next_cols.offline_checker_cols;

        SubAir::eval(
            &offline_checker,
            builder,
            (
                local_offline_checker_cols.clone(),
                next_offline_checker_cols.clone(),
            ),
            (),
        );

        builder.assert_bool(local_offline_checker_cols.op_type);
        builder.assert_bool(local_cols.same_data);

        // Constrain that same_idx_and_data is same_idx * same_data
        builder.assert_eq(
            local_cols.same_idx_and_data,
            local_offline_checker_cols.same_idx * local_cols.same_data,
        );

        builder.when_first_row().assert_zero(local_cols.same_data);

        // Making sure same_data is correct across rows
        let is_equal_data = IsEqualVecCols::new(
            local_offline_checker_cols.data.to_vec(),
            next_offline_checker_cols.data.to_vec(),
            next_cols.same_data,
            next_cols.is_equal_data_aux.prods.clone(),
            next_cols.is_equal_data_aux.invs,
        );
        let is_equal_data_air = IsEqualVecAir::new(self.offline_checker.data_len);

        SubAir::eval(
            &is_equal_data_air,
            &mut builder.when_transition(),
            is_equal_data.io,
            is_equal_data.aux,
        );

        // Making sure that every read uses the same data as the last operation if it is not the first
        // operation in the block
        // NOTE: constraint degree is 4
        builder.assert_one(or::<AB>(
            AB::Expr::one() - local_offline_checker_cols.is_valid.into(),
            or::<AB>(
                local_offline_checker_cols.op_type.into(),
                // if same_idx = 0 and read, then same_data can be anything
                // if same_idx = 1 and read, then same_data must be 1
                AB::Expr::one() - local_offline_checker_cols.same_idx.into()
                    + local_cols.same_idx_and_data,
            ),
        ));
    }
}
