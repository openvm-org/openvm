use afs_primitives::{
    is_less_than::{columns::IsLessThanIoCols, IsLessThanAir},
    is_zero::{
        columns::{IsZeroCols, IsZeroIoCols},
        IsZeroAir,
    },
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
        word: [AB::Expr; WORD_SIZE],
        field_elem: AB::Expr,
    ) {
        builder.assert_eq(word[0].clone(), field_elem);
        for cell in word.iter().take(WORD_SIZE).skip(1).cloned() {
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

        self.subair_eval(builder, local.io.into_expr::<AB>(), local.aux);
    }
}

impl<const WORD_SIZE: usize> NewMemoryOfflineChecker<WORD_SIZE> {
    pub fn subair_eval<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        op: MemoryOperation<WORD_SIZE, AB::Expr>,
        aux: MemoryOfflineCheckerAuxCols<WORD_SIZE, AB::Var>,
    ) {
        builder.assert_bool(op.op_type.clone());
        builder.assert_bool(op.enabled.clone());

        // Ensuring is_immediate is correct
        let addr_space_is_zero_cols = IsZeroCols::<AB::Expr>::new(
            IsZeroIoCols::<AB::Expr>::new(op.addr_space.clone(), aux.is_immediate.into()),
            aux.is_zero_aux.into(),
        );

        self.is_zero_air.subair_eval(
            builder,
            addr_space_is_zero_cols.io,
            addr_space_is_zero_cols.inv,
        );

        self.assert_compose(
            &mut builder.when(aux.is_immediate),
            op.cell.data.clone(),
            op.pointer.clone(),
        );

        // is_immediate => read
        builder.assert_one(implies::<AB>(
            aux.is_immediate.into(),
            AB::Expr::one() - op.op_type.clone(),
        ));

        let clk_lt_io_cols = IsLessThanIoCols::<AB::Expr>::new(
            aux.old_cell.clk.into(),
            op.cell.clk.clone(),
            aux.clk_lt.into(),
        );

        self.clk_lt_air
            .subair_eval(builder, clk_lt_io_cols, aux.clk_lt_aux);

        // TODO[osama]: this should be reduced to degree 2
        builder.assert_one(implies::<AB>(
            and::<AB>(
                op.enabled.clone(),
                AB::Expr::one() - aux.is_immediate.into(),
            ),
            aux.clk_lt.into(),
        ));

        // Ensuring that if op_type is Read, data_read is the same as data_write
        for i in 0..WORD_SIZE {
            builder.assert_zero(
                (AB::Expr::one() - op.op_type.clone())
                    * (op.cell.data[i].clone() - aux.old_cell.data[i]),
            );
        }

        Self::eval_memory_interactions(
            builder,
            op.addr_space,
            op.pointer,
            aux.old_cell.into_expr::<AB>(),
            op.cell,
            op.enabled - aux.is_immediate.into(),
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
