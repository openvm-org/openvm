use std::borrow::Borrow;

use afs_primitives::var_range::bus::VariableRangeCheckerBus;
use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::columns::UiCols;
use crate::{
    arch::{bus::ExecutionBus, instructions::Opcode},
    memory::offline_checker::MemoryBridge,
};

#[derive(Copy, Clone, Debug)]
pub struct UiAir {
    pub(super) execution_bus: ExecutionBus,
    pub(super) memory_bridge: MemoryBridge,

    pub bus: VariableRangeCheckerBus,
}

impl<F: Field> BaseAir<F> for UiAir {
    fn width(&self) -> usize {
        UiCols::<F>::width()
    }
}

impl<AB: InteractionBuilder + AirBuilder> Air<AB> for UiAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local_cols: &UiCols<AB::Var> = (*local).borrow();
        builder.assert_bool(local_cols.aux.is_valid);
        builder.when(local_cols.aux.is_valid).assert_eq(
            local_cols.io.x[2],
            local_cols.aux.imm_lo_hex * AB::Expr::from_canonical_u32(16),
        );
        builder.when(local_cols.aux.is_valid).assert_eq(
            local_cols.io.op_b,
            local_cols.io.x[0] * AB::Expr::from_canonical_u32(1 << 12)
                + local_cols.io.x[1] * AB::Expr::from_canonical_u32(1 << 4)
                + local_cols.aux.imm_lo_hex,
        );
        builder
            .when(local_cols.aux.is_valid)
            .assert_zero(local_cols.io.x[3]);

        let expected_opcode = AB::Expr::from_canonical_u32(Opcode::LUI as u32);

        self.eval_interactions(builder, &local_cols.io, &local_cols.aux, expected_opcode);
    }
}
