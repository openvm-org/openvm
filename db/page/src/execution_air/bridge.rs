use std::iter;

use afs_stark_backend::interaction::InteractionBuilder;

use super::{columns::ExecutionCols, ExecutionAir};

impl ExecutionAir {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: ExecutionCols<AB::Var>,
    ) {
        let fields = iter::once(local.clk)
            .chain(iter::once(local.op_type))
            .chain(local.idx)
            .chain(local.data);
        builder.push_send(self.bus_index, fields, local.mult);
    }
}
