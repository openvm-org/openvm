use afs_primitives::{
    is_less_than_tuple::{
        columns::{IsLessThanTupleCols, IsLessThanTupleIoCols},
        IsLessThanTupleAir,
    },
    utils::{implies, or},
    var_range::bus::VariableRangeCheckerBus,
};
use afs_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use super::columns::AuditCols;
use crate::{core::RANGE_CHECKER_BUS, memory::offline_checker::MemoryBus};

#[derive(Clone, Debug)]
pub struct MemoryAuditAir {
    pub memory_bus: MemoryBus,
    pub addr_lt_air: IsLessThanTupleAir,
    pub for_testing: bool,
}

impl MemoryAuditAir {
    // TODO[jpw]: pass in range bus
    pub fn new(
        memory_bus: MemoryBus,
        addr_space_max_bits: usize,
        pointer_max_bits: usize,
        decomp: usize,
        for_testing: bool,
    ) -> Self {
        Self {
            memory_bus,
            addr_lt_air: IsLessThanTupleAir::new(
                VariableRangeCheckerBus::new(RANGE_CHECKER_BUS, decomp),
                vec![addr_space_max_bits, pointer_max_bits],
            ),
            for_testing,
        }
    }

    pub fn air_width(&self) -> usize {
        AuditCols::<usize>::width(self)
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for MemoryAuditAir {}
impl<F: Field> PartitionedBaseAir<F> for MemoryAuditAir {}
impl<F: Field> BaseAir<F> for MemoryAuditAir {
    fn width(&self) -> usize {
        self.air_width()
    }
}

impl<AB: InteractionBuilder> Air<AB> for MemoryAuditAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let [local, next] = [0, 1].map(|i| {
            let row = main.row_slice(i);
            AuditCols::<AB::Var>::from_slice(&row, self)
        });

        // TODO[jpw]: ideally make this work for testing too
        if !self.for_testing {
            builder.assert_bool(local.is_extra);

            // Ensuring all is_extra rows are at the bottom
            builder
                .when_transition()
                .assert_one(implies(local.is_extra, next.is_extra));

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
                .assert_one(or(next.is_extra, next.addr_lt));
        }

        self.eval_interactions(builder, local);
    }
}
