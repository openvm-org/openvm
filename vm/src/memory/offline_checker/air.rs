use afs_primitives::{
    is_less_than::{
        columns::{IsLessThanCols, IsLessThanIoCols},
        IsLessThanAir,
    },
    offline_checker::columns::OfflineCheckerCols,
    sub_chip::{AirConfig, SubAir},
    utils::or,
};
use afs_stark_backend::{air_builders::PartitionedAirBuilder, interaction::InteractionBuilder};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field, PrimeField32};
use p3_matrix::Matrix;

use super::{columns::MemoryOfflineCheckerCols, MemoryChip, MemoryOfflineChecker};
use crate::memory::manager::eval_memory_interactions;

pub struct NewMemoryOfflineChecker<const WORD_SIZE: usize> {
    pub clk_lt_air: IsLessThanAir,
}

impl<const WORD_SIZE: usize> AirConfig for NewMemoryOfflineChecker<WORD_SIZE> {
    type Cols<T> = MemoryOfflineCheckerCols<WORD_SIZE, T>;
}

impl<const WORD_SIZE: usize, F: Field> BaseAir<F> for NewMemoryOfflineChecker<WORD_SIZE> {
    fn width(&self) -> usize {
        MemoryOfflineCheckerCols::<WORD_SIZE, usize>::width(self)
    }
}

impl<const WORD_SIZE: usize, AB: InteractionBuilder> Air<AB>
    for NewMemoryOfflineChecker<WORD_SIZE>
{
    fn eval(&self, builder: &mut AB) {
        let main = &builder.main();
        let local = MemoryOfflineCheckerCols::<WORD_SIZE, AB::Var>::from_slice(&main.row_slice(0));

        // Ensuring clk_lt is correct
        let clk_lt_cols = IsLessThanCols::<AB::Var>::new(
            IsLessThanIoCols::new(
                local.op_cols.clk_read,
                local.op_cols.clk_write,
                local.clk_lt,
            ),
            local.clk_lt_aux,
        );

        SubAir::eval(&self.clk_lt_air, builder, clk_lt_cols.io, clk_lt_cols.aux);

        // Ensuring clk_read is less than clk_write
        // TODO[osama]: I think this should be <=
        builder.assert_one(or::<AB>(local.clk_lt.into(), local.is_extra.into()));

        eval_memory_interactions(
            builder,
            local.op_cols,
            AB::Expr::one() - local.is_extra.into(),
        );
    }
}

impl<const WORD_SIZE: usize, F: PrimeField32> AirConfig for MemoryChip<WORD_SIZE, F> {
    type Cols<T> = OfflineCheckerCols<T>;
}

impl<F: Field> BaseAir<F> for MemoryOfflineChecker {
    fn width(&self) -> usize {
        self.air_width()
    }
}

impl<AB: PartitionedAirBuilder + InteractionBuilder> Air<AB> for MemoryOfflineChecker {
    /// This constrains extra rows to be at the bottom and the following on non-extra rows:
    /// same_addr_space, same_pointer, same_data, lt_bit is correct (see definition in columns.rs)
    /// A read must be preceded by a write with the same address space, pointer, and data
    fn eval(&self, builder: &mut AB) {
        let main = &builder.main();

        let local_cols = OfflineCheckerCols::from_slice(&main.row_slice(0), &self.offline_checker);
        let next_cols = OfflineCheckerCols::from_slice(&main.row_slice(1), &self.offline_checker);

        builder.assert_bool(local_cols.op_type);

        // loop over data_len
        // is_valid * (1 - op_type) * same_idx * (x[i] - y[i])
        for i in 0..self.offline_checker.data_len {
            // NOTE: constraint degree is 4
            builder.when_transition().assert_zero(
                next_cols.is_valid.into()
                    * (AB::Expr::one() - next_cols.op_type.into())
                    * next_cols.same_idx.into()
                    * (local_cols.data[i] - next_cols.data[i]),
            );
        }

        SubAir::eval(&self.offline_checker, builder, (local_cols, next_cols), ());
    }
}
