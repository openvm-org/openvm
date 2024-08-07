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

pub struct AuditAir<const WORD_SIZE: usize> {
    pub addr_lt_air: IsLessThanTupleAir,
}

impl<const WORD_SIZE: usize> AuditAir<WORD_SIZE> {
    pub fn new(address_space_limb_bits: usize, address_limb_bits: usize, decomp: usize) -> Self {
        Self {
            addr_lt_air: IsLessThanTupleAir::new(
                RANGE_CHECKER_BUS,
                vec![address_space_limb_bits, address_limb_bits],
                decomp,
            ),
        }
    }

    pub fn air_width(&self) -> usize {
        AuditCols::<WORD_SIZE, usize>::width(self)
    }
}

impl<const WORD_SIZE: usize, F: Field> BaseAir<F> for AuditAir<WORD_SIZE> {
    fn width(&self) -> usize {
        self.air_width()
    }
}

impl<const WORD_SIZE: usize, AB: InteractionBuilder> Air<AB> for AuditAir<WORD_SIZE> {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let [local, next] = [0, 1].map(|i| {
            let row = main.row_slice(i);
            AuditCols::<WORD_SIZE, AB::Var>::from_slice(&row, self)
        });

        // Ensuring all is_extra rows are at the bottom
        builder
            .when_transition()
            .assert_one(implies::<AB>(local.is_extra.into(), next.is_extra.into()));

        // Ensuring addr_lt is correct
        let lt_cols = IsLessThanTupleCols::new(
            IsLessThanTupleIoCols::new(
                vec![local.op_cols.address_space, local.op_cols.address],
                vec![next.op_cols.address_space, next.op_cols.address],
                next.addr_lt,
            ),
            next.addr_lt_aux.clone(),
        );

        self.addr_lt_air
            .eval_when_transition(builder, lt_cols.io, lt_cols.aux);

        // Ensuring that all addresses are sorted
        builder
            .when_transition()
            .assert_one(or::<AB>(next.is_extra.into(), next.addr_lt.into()));

        self.eval_interactions(builder, local);
    }
}
