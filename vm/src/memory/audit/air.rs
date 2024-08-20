use afs_primitives::{
    is_less_than_tuple::{
        columns::{IsLessThanTupleCols, IsLessThanTupleIoCols},
        IsLessThanTupleAir,
    },
    utils::{implies, or},
};
use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use super::columns::AuditCols;
use crate::cpu::RANGE_CHECKER_BUS;

#[derive(Clone)]
pub struct MemoryAuditAir<const WORD_SIZE: usize> {
    pub addr_lt_air: IsLessThanTupleAir,
}

impl<const WORD_SIZE: usize> MemoryAuditAir<WORD_SIZE> {
    pub fn new(addr_space_max_bits: usize, pointer_max_bits: usize, decomp: usize) -> Self {
        Self {
            addr_lt_air: IsLessThanTupleAir::new(
                RANGE_CHECKER_BUS,
                vec![addr_space_max_bits, pointer_max_bits],
                decomp,
            ),
        }
    }

    pub fn air_width(&self) -> usize {
        AuditCols::<WORD_SIZE, usize>::width(self)
    }
}

impl<const WORD_SIZE: usize, F: Field> BaseAir<F> for MemoryAuditAir<WORD_SIZE> {
    fn width(&self) -> usize {
        self.air_width()
    }
}

impl<const WORD_SIZE: usize, AB: InteractionBuilder> Air<AB> for MemoryAuditAir<WORD_SIZE> {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let [local, next] = [0, 1].map(|i| {
            let row = main.row_slice(i);
            AuditCols::<WORD_SIZE, AB::Var>::from_slice(&row, self)
        });

        // Ensuring all is_extra rows are at the bottom
        builder
            .when_transition()
            .assert_one(implies(local.is_extra.into(), next.is_extra.into()));

        // Ensuring addr_lt is correct
        let lt_cols = IsLessThanTupleCols::new(
            IsLessThanTupleIoCols::new(
                vec![local.addr_space, local.pointer],
                vec![next.addr_space, next.pointer],
                next.addr_lt,
            ),
            next.addr_lt_aux.clone(),
        );

        self.addr_lt_air
            .eval_when_transition(builder, lt_cols.io, lt_cols.aux);

        // Ensuring that all addresses are sorted
        builder
            .when_transition()
            .assert_one(or(next.is_extra.into(), next.addr_lt.into()));

        self.eval_interactions(builder, local);
    }
}
