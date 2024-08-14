use afs_primitives::{
    is_less_than::{
        columns::{IsLessThanCols, IsLessThanIoCols},
        IsLessThanAir,
    },
    is_zero::{columns::IsZeroCols, IsZeroAir},
    offline_checker::columns::OfflineCheckerCols,
    sub_chip::{AirConfig, SubAir},
    utils::{and, implies},
};
use afs_stark_backend::{air_builders::PartitionedAirBuilder, interaction::InteractionBuilder};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field, PrimeField32};
use p3_matrix::Matrix;

use super::{
    columns::{MemoryOfflineCheckerAuxCols, MemoryOfflineCheckerCols},
    MemoryChip, MemoryOfflineChecker,
};
use crate::{cpu::RANGE_CHECKER_BUS, memory::manager::operation::MemoryOperation};

pub struct NewMemoryOfflineChecker<const WORD_SIZE: usize> {
    pub clk_lt_air: IsLessThanAir,
    pub is_zero_air: IsZeroAir,
}

impl<const WORD_SIZE: usize> NewMemoryOfflineChecker<WORD_SIZE> {
    pub fn new(clk_max_bits: usize, decomp: usize) -> Self {
        Self {
            clk_lt_air: IsLessThanAir::new(RANGE_CHECKER_BUS, clk_max_bits, decomp),
            is_zero_air: IsZeroAir,
        }
    }

    fn assert_compose<AB: AirBuilder>(
        &self,
        builder: &mut AB,
        word: [AB::Var; WORD_SIZE],
        field_elem: AB::Expr,
    ) {
        builder.assert_eq(word[0], field_elem);
        for &cell in word.iter().take(WORD_SIZE).skip(1) {
            builder.assert_zero(cell);
        }
    }
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
        let main = builder.main();
        let local = main.row_slice(0);
        let local = MemoryOfflineCheckerCols::<WORD_SIZE, AB::Var>::from_slice(&local);

        SubAir::eval(self, builder, local.io, local.aux);
    }
}

impl<const WORD_SIZE: usize, AB: InteractionBuilder> SubAir<AB>
    for NewMemoryOfflineChecker<WORD_SIZE>
{
    type IoView = MemoryOperation<WORD_SIZE, AB::Var>;
    type AuxView = MemoryOfflineCheckerAuxCols<WORD_SIZE, AB::Var>;

    fn eval(
        &self,
        builder: &mut AB,
        op: MemoryOperation<WORD_SIZE, AB::Var>,
        aux: MemoryOfflineCheckerAuxCols<WORD_SIZE, AB::Var>,
    ) {
        builder.assert_bool(op.op_type);
        builder.assert_bool(op.enabled);

        // Ensuring is_immediate is correct
        let addr_space_is_zero_cols =
            IsZeroCols::new(op.addr_space, aux.is_immediate, aux.is_zero_aux);

        SubAir::eval(
            &self.is_zero_air,
            builder,
            addr_space_is_zero_cols.io,
            addr_space_is_zero_cols.inv,
        );

        self.assert_compose(
            &mut builder.when(aux.is_immediate),
            op.cell.data,
            op.pointer.into(),
        );

        // is_immediate => read
        builder.assert_one(implies::<AB>(
            aux.is_immediate.into(),
            AB::Expr::one() - op.op_type.into(),
        ));

        // Ensuring clk_lt is correct
        let clk_lt_cols = IsLessThanCols::<AB::Var>::new(
            IsLessThanIoCols::new(aux.old_cell.clk, op.cell.clk, aux.clk_lt),
            aux.clk_lt_aux,
        );

        SubAir::eval(&self.clk_lt_air, builder, clk_lt_cols.io, clk_lt_cols.aux);

        // TODO[osama]: make it so that we don't have to call .into() explicitly
        // TODO[osama]: this should be reduced to degree 2
        builder.assert_one(implies::<AB>(
            and::<AB>(op.enabled.into(), AB::Expr::one() - aux.is_immediate.into()),
            aux.clk_lt.into(),
        ));

        // Ensuring that if op_type is Read, data_read is the same as data_write
        for i in 0..WORD_SIZE {
            builder.assert_zero(
                (AB::Expr::one() - op.op_type.into()) * (op.cell.data[i] - aux.old_cell.data[i]),
            );
        }

        Self::eval_memory_interactions(
            builder,
            op.addr_space,
            op.pointer,
            aux.old_cell,
            op.cell,
            op.enabled.into() - aux.is_immediate.into(),
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
