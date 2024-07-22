use std::borrow::Borrow;
use std::iter;

use afs_stark_backend::air_builders::PartitionedAirBuilder;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use crate::cpu::RANGE_CHECKER_BUS;

use super::{columns::OfflineCheckerCols, OfflineChecker};
use afs_chips::{
    is_equal::{columns::IsEqualCols, IsEqualAir},
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
        builder.assert_bool(local_cols.same_addr_space);
        builder.assert_bool(local_cols.same_pointer);
        builder.assert_bool(local_cols.same_addr);
        builder.assert_bool(local_cols.same_data);
        builder.assert_bool(local_cols.is_valid);

        // Making sure first row starts with same_addr_space, same_pointer, same_data being false
        builder
            .when_first_row()
            .assert_zero(local_cols.same_addr_space);
        builder
            .when_first_row()
            .assert_zero(local_cols.same_pointer);
        builder.when_first_row().assert_zero(local_cols.same_data);

        // Making sure same_addr_space is correct across rows
        let is_equal_addr_space = IsEqualCols::new(
            local_cols.mem_row[0],
            next_cols.mem_row[0],
            next_cols.same_addr_space,
            next_cols.is_equal_addr_space_aux.inv,
        );

        let is_equal_addr_space_air = IsEqualAir {};
        SubAir::eval(
            &is_equal_addr_space_air,
            &mut builder.when_transition(),
            is_equal_addr_space.io,
            is_equal_addr_space.aux,
        );

        // Making sure same_pointer is correct across rows
        let is_equal_pointer = IsEqualCols::new(
            local_cols.mem_row[1],
            next_cols.mem_row[1],
            next_cols.same_pointer,
            next_cols.is_equal_pointer_aux.inv,
        );

        let is_equal_pointer_air = IsEqualAir {};
        SubAir::eval(
            &is_equal_pointer_air,
            &mut builder.when_transition(),
            is_equal_pointer.io,
            is_equal_pointer.aux,
        );

        // Making sure same_data is correct across rows
        let is_equal_data = IsEqualVecCols::new(
            local_cols.mem_row[2..].to_vec(),
            next_cols.mem_row[2..].to_vec(),
            next_cols.same_data,
            next_cols.is_equal_data_aux.prods,
            next_cols.is_equal_data_aux.invs,
        );
        let is_equal_data_air = IsEqualVecAir::new(WORD_SIZE);

        SubAir::eval(
            &is_equal_data_air,
            &mut builder.when_transition(),
            is_equal_data.io,
            is_equal_data.aux,
        );

        // Ensuring all rows are sorted by (addr_space, addr, clk)
        let lt_io_cols = IsLessThanTupleIOCols::<AB::Var> {
            x: local_cols.mem_row[0..2]
                .iter()
                .copied()
                .chain(iter::once(local_cols.clk))
                .collect(),
            y: next_cols.mem_row[0..2]
                .iter()
                .copied()
                .chain(iter::once(next_cols.clk))
                .collect(),
            tuple_less_than: next_cols.lt_bit,
        };

        let lt_chip = IsLessThanTupleAir::new(
            RANGE_CHECKER_BUS,
            self.addr_clk_limb_bits.clone(),
            self.decomp,
        );

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

        // Constraining that same_addr is correct
        builder.when_transition().assert_eq(
            local_cols.same_addr,
            and(
                local_cols.same_addr_space.into(),
                local_cols.same_pointer.into(),
            ),
        );

        // Making sure that every read uses the same data as the last operation if it is not the first
        // operation in the block
        // read => same_data
        // NOTE: constraint degree is 3
        builder.assert_one(or(
            AB::Expr::one() - local_cols.is_valid.into(),
            or(
                local_cols.op_type.into(),
                // if b is 2 (when same_addr = 0 and same_data = 1), then this or will give 0 or 2,
                // and the outer or will be 0 or 2, failing the constraint
                // in our trace generation, we set same_data = 0 when same_addr = 0
                AB::Expr::one() - local_cols.same_addr.into() + local_cols.same_data.into(),
            ),
        ));

        // Making sure is_extra rows are at the bottom
        builder.when_transition().assert_one(implies(
            next_cols.is_valid.into(),
            local_cols.is_valid.into(),
        ));

        // Note that the following is implied:
        // - for every row: read => same_addr because not same_addr => write
    }
}
